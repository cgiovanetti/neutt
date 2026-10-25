"""The hybrid decoupling solver: a fixed-step exponential-midpoint head, then adaptive Tsit5.

In the stiff 10 → 2.2 MeV head f sits on the quasi-static attractor A/B of the per-bin affine
collision RHS df/dln a = A − B·f, and an explicit solve is stability-limited.  The exponential
midpoint (expMid) step is exact for frozen (A, B), positivity-preserving and has no stability
ceiling, so a ``lax.scan`` of fixed steps clears the head; Tsit5 on ``decoupling.make_rhs`` then
covers the annihilation tail.

The handoff is qualified: at ``lna_sw`` the state must be finite, in [0, 1] and on the attractor
(max|f − A/B| ≤ ``qual_tol``).  Otherwise the tail silently restarts from ln a = 0, i.e. runs the
full adaptive solve; both paths share one compiled diffeqsolve (traced t0/y0), so a failed handoff
costs time, not accuracy.  N_eff agrees with a tight-tolerance adaptive solve to well within 2e-6.

Two fixed-step solvers for stiff dark sectors use the same exp_step machinery:
``make_chi_exact_solver`` (the strongly coupled χ ↔ ν sector: a coupled backward-Euler Newton for
(f_ν, f_χ) per sub-step, then the SM relaxation) and ``make_dark_decay_implicit_solver`` (an
overabundant decaying boson: an exp_step head for X and ν, then an SM-only Tsit5 tail).
"""

from __future__ import annotations

import numpy as np
import diffrax
import jax
import jax.numpy as jnp

from . import compiling, thermo
from . import dark as dark_mod
from .constants import M_ELECTRON, PI
from .decoupling import (make_rhs, make_split_rates, nanrobust_rms_norm,
                         neff_from_spectrum, _plasma_eos,
                         rho_nu_from_spectrum, chi_xspecies_sumrule,
                         make_decay_tail_norm, high_eps_energy_frac)
from .grid import make_eps_grid
from .interp import FD_ENVELOPE_C
from .transport import assemble as A
from .transport import chi_brute
from .transport.nunu import build_nunu_kernel
from .transport.nunubar import build_nunubar_kernel

QSS_TINY = 1.0e-100  # floor on B in the attractor A/B


def phi1(z):
    """φ₁(z) = (e^z − 1)/z, with the series 1 + z/2 for ``|z| ≤ 1e-8``."""
    return jnp.where(jnp.abs(z) > 1e-8,
                     jnp.expm1(z) / jnp.where(z == 0.0, 1.0, z),
                     1.0 + 0.5 * z)


def exp_step(f, a, b, h):
    """One exponential-Euler update f′ = f + h·φ₁(−bh)·(a − b·f).

    Equals f·e^{−bh} + (a/b)(1 − e^{−bh}), a convex combination of f and the attractor a/b
    (positivity-preserving, unconditionally stable); the φ₁ form stays exact as b → 0.
    """
    return f + h * phi1(-b * h) * (a - b * f)


def head_edges(lna_sw, n_head, power=1.0):
    """Head step edges on [0, lna_sw], shape (n_head+1,), as static numpy.

    ``power = 1`` is uniform; ``power > 1`` concentrates steps toward the switch, where the head
    error accrues.  power ≠ 1 has erratic NaN / off-attractor failure pockets: the qualification
    catches them, but at the cost of a full solve, so check any power ≠ 1 over many parameter draws.
    """
    i = np.linspace(0.0, 1.0, n_head + 1)
    return lna_sw * (1.0 - (1.0 - i) ** power)


def make_hybrid_solver(grid=None, *, lna_sw=1.5, n_head=1600, head_power=1.0,
                       n_save_head=128, n_save_tail=128,
                       use_sched=True, use_renorm=True, renorm_pdep=True, qed_nlo=True,
                       nuratiotol=None, tcm_cutoff=0.015, phie=0.0, eta0=None, q_baryon=0.5,
                       xi=(0.0, 0.0, 0.0), lag_tail_fix=True,
                       max_steps=50_000, pcoeff=0.4, icoeff=0.3, dcoeff=0.0, throw=True):
    """Build the kernels, noise floor and initial state once; return a jitted ``solve``.

    The physics options (use_sched … q_baryon, lag_tail_fix) are those of make_rhs /
    make_split_rates, identical in head and tail; max_steps and the PID coefficients set the tail's step-size controller.
    ``throw`` (default True) raises on a failed tail solve; False returns the partial result with
    ``tail_ok`` False instead.

    Parameters
    ----------
    lna_sw : float
        Head → tail handoff in ln a.  Keep ≲ 1.5: a later handoff leaves the state off the attractor
        and stalls the tail.
    head_power : float
        Step grading for head_edges (1 = uniform; see its caveat).
    xi : array-like, shape (3,)
        Build-time per-flavour ν degeneracy: sets the FD initial occupation and the noise floor.
        ξ ≠ 0 shifts the attractor, so watch ``qualified``.

    Returns
    -------
    (solve, grid)
        ``solve(T_init=10.0, lna_max=7.5, rtol=1e-6, atol=1e-30, dtmax=0.08, delta_neff_init=0.0,
        qual_tol=1e-4, me=None, xi=None)`` returns the keys of LINX's transport_adapter solver
        (lna, T_pl, T_cm, ratio, f_traj, f_final, ratio_final, neff_final) plus diagnostics:
        ``qualified`` (False: the fallback ran; the result is correct either way), ``qss_dist``
        (max|f − A/B| at the handoff), the head_* sanity checks and tail_steps/tail_rejected/
        tail_ok.  ``me`` [MeV] and ``xi`` may be passed per call as traced values (no recompile
        across values); None means the bare mass and the build-time ``xi``.
    """
    if grid is None:
        grid = make_eps_grid()
    if nuratiotol is None:
        nuratiotol = A.NURATIOTOL

    # ---- built once ----
    nunu_kernel = build_nunu_kernel(grid)
    nunubar_kernel = build_nunubar_kernel(grid)
    prec_trans = A.compute_prec_trans(grid, nunu_kernel, nunubar_kernel, xi=xi,
                                      use_sched=use_sched, lag_tail_fix=lag_tail_fix)
    occ0 = A.equilibrium_occ(grid, xi=xi)

    edges = head_edges(lna_sw, n_head, head_power)           # (n_head+1,) static
    starts = jnp.asarray(edges[:-1])
    hs = jnp.asarray(np.diff(edges))
    # saved head nodes: n_save_head step edges, excluding lna_sw (the tail's first node)
    sel = np.linspace(0, n_head, n_save_head, endpoint=False).astype(int)
    lna_head_nodes = jnp.asarray(edges[sel])

    def _interp_cols(x, xp, fp):
        """jnp.interp along the leading axis of fp (n, ...) at x."""
        flat = fp.reshape(fp.shape[0], -1)
        out = jax.vmap(lambda col: jnp.interp(x, xp, col), in_axes=1, out_axes=1)(flat)
        return out.reshape((x.shape[0],) + fp.shape[1:])

    @jax.jit
    def solve(T_init=10.0, lna_max=7.5, rtol=1e-6, atol=1e-30, dtmax=0.08,
              delta_neff_init=0.0, qual_tol=1e-4, me=None, xi=None):
        compiling()
        # qual_tol = 1e-4: ~2× the attractor distance at lna_sw = 1.5; rejects handoffs past ~1.7.
        # A passed me reaches the kernels, EoS and QED; it shifts the attractor mildly.
        if me is None:
            me = M_ELECTRON
        T_init = jnp.asarray(T_init, dtype=jnp.float64)
        # a passed xi recomputes the noise floor and initial state inside the jit
        if xi is None:
            prec_use, occ_use = prec_trans, occ0
        else:
            _xi = jnp.asarray(xi, dtype=jnp.float64)
            prec_use = A.compute_prec_trans(grid, nunu_kernel, nunubar_kernel, xi=_xi,
                                            use_sched=use_sched, lag_tail_fix=lag_tail_fix)
            occ_use = A.equilibrium_occ(grid, xi=_xi)
        split = make_split_rates(grid, prec_use, nunu_kernel, nunubar_kernel, T_init,
                                 nuratiotol, use_renorm, use_sched=use_sched,
                                 renorm_pdep=renorm_pdep, qed_nlo=qed_nlo, tcm_cutoff=tcm_cutoff,
                                 phie=phie, eta0=eta0, q_baryon=q_baryon,
                                 delta_neff_init=delta_neff_init, me=me, lag_tail_fix=lag_tail_fix)

        # ---- expMid head: fixed graded steps over [0, lna_sw] ----
        def body(carry, x):
            T_pl, f = carry
            lna, h = x
            a1, b1, dT1, _ = split(lna, T_pl, f)
            f_half = exp_step(f, a1, b1, 0.5 * h)
            T_half = T_pl + 0.5 * h * dT1
            a2, b2, dT2, _ = split(lna + 0.5 * h, T_half, f_half)
            f_new = exp_step(f, a2, b2, h)
            T_new = T_pl + h * dT2
            return (T_new, f_new), (T_new, f_new)

        (T_sw, f_sw), (T_traj, f_traj_head) = jax.lax.scan(body, (T_init, occ_use),
                                                           (starts, hs))
        T_at_edges = jnp.concatenate([T_init[None], T_traj])           # (n_head+1,)
        f_at_edges = jnp.concatenate([occ_use[None], f_traj_head])     # (n_head+1,3,2,N)

        # ---- handoff qualification: on the attractor, finite, f ∈ [0, 1] ----
        a_sw, b_sw, _, _ = split(jnp.asarray(lna_sw), T_sw, f_sw)
        attractor = a_sw / jnp.maximum(b_sw, QSS_TINY)
        qss_dist = jnp.max(jnp.where(b_sw > 0.0, jnp.abs(f_sw - attractor), 0.0))
        T_ok = jnp.isfinite(T_sw) & (T_sw > 0.0)
        f_finite = jnp.all(jnp.isfinite(f_sw))
        n_neg = jnp.sum(f_sw < 0.0)          # round-off negatives seed NaNs downstream
        n_super = jnp.sum(f_sw > 1.0)
        sane = T_ok & f_finite & (n_neg == 0) & (n_super == 0)
        qualified = sane & (qss_dist <= qual_tol)

        # ---- tail: Tsit5 from the handoff, or from ln a = 0 if unqualified (traced t0/y0) ----
        t0 = jnp.where(qualified, lna_sw, 0.0)
        y0 = {"T_pl": jnp.where(qualified, T_sw, T_init),
              "f": jnp.where(qualified, f_sw, occ_use)}
        dt0 = jnp.where(qualified, 1e-3, 1e-4)
        ts_tail = jnp.linspace(t0, lna_max, n_save_tail)
        rhs = make_rhs(grid, prec_use, nunu_kernel, nunubar_kernel, T_init,
                       nuratiotol, use_renorm, use_sched=use_sched,
                       renorm_pdep=renorm_pdep, qed_nlo=qed_nlo, tcm_cutoff=tcm_cutoff,
                       phie=phie, eta0=eta0, q_baryon=q_baryon, delta_neff_init=delta_neff_init, me=me,
                       lag_tail_fix=lag_tail_fix)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs), diffrax.Tsit5(), t0=t0, t1=lna_max, dt0=dt0, y0=y0,
            saveat=diffrax.SaveAt(ts=ts_tail),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol, dtmax=dtmax,
                                                      pcoeff=pcoeff, icoeff=icoeff,
                                                      dcoeff=dcoeff,
                                                      norm=nanrobust_rms_norm),
            max_steps=max_steps, throw=throw)

        # ---- trajectory: head + tail nodes, or the fallback solve resampled on uniform nodes ----
        n_tot = n_save_head + n_save_tail
        nodes_q = jnp.concatenate([lna_head_nodes, ts_tail])
        nodes_fb = jnp.linspace(0.0, lna_max, n_tot)
        lna = jnp.where(qualified, nodes_q, nodes_fb)
        T_q = jnp.concatenate([T_at_edges[sel], sol.ys["T_pl"]])
        T_fb = jnp.interp(nodes_fb, ts_tail, sol.ys["T_pl"])
        T_pl = jnp.where(qualified, T_q, T_fb)
        f_q = jnp.concatenate([f_at_edges[sel], sol.ys["f"]])
        f_fb = _interp_cols(nodes_fb, ts_tail, sol.ys["f"])
        f_traj = jnp.where(qualified, f_q, f_fb)

        T_cm = T_init * jnp.exp(-lna)
        T_fin = sol.ys["T_pl"][-1]
        f_fin = sol.ys["f"][-1]
        T_cm_fin = T_init * jnp.exp(-lna_max)
        return {"lna": lna, "T_pl": T_pl, "T_cm": T_cm, "ratio": T_pl / T_cm,
                "f_traj": f_traj, "f_final": f_fin,
                "ratio_final": T_fin / T_cm_fin,
                "neff_final": neff_from_spectrum(grid, f_fin, T_cm_fin, T_fin),
                "qualified": qualified, "qss_dist": qss_dist,
                "head_T_ok": T_ok, "head_f_finite": f_finite,
                "head_n_neg": n_neg, "head_n_super": n_super,
                "tail_steps": sol.stats["num_accepted_steps"],
                "tail_rejected": sol.stats["num_rejected_steps"],
                "tail_ok": sol.result == diffrax.RESULTS.successful}

    return solve, grid


def make_chi_exact_solver(grid=None, *, chi_sector, T_init=7.0, lna_max=5.9, h=0.005,
                          k_newton=6, k_sub=8, x_switch=0.5, x_freeze=4.0,
                          trust=4.0, lam=0.0, fdcap_logC=0.5, deepfix="freeze",
                          cons_tol=0.3, bt_max=200.0, n_save=128, use_sched=False,
                          nuratiotol=None, tcm_cutoff=0.015, me=M_ELECTRON, chi_nodes=None,
                          f_chi0=None, cap_evolved=False, fd_decay=False, monotone_read=False,
                          iso_evolved=False, robust=True, rcond=1e-6, analytic_phi=False,
                          solve_first=False):
    """EXACT shape-resolved coupled χ→ν solver — the BBN-grade strong-S2 deliverable.

    The strong-coupling (m_Z′≈91 GeV, K≈100·G_F², Γ/H≫1) χ↔ν solver. It evolves (f_ν, f_χ) under the
    TRUE coupled kernel net (net_ν AND net_χ from ONE ``s2_brute_rates`` call), so ν↔χ number AND
    energy are conserved to the kernel quadrature floor and the energy-deposition SHAPE is exact. (An
    energy-only delivery — depositing the implicit χ energy loss over a fixed forward annihilation
    shape — is energy-exact but ν-number short, ``num_rel~0.55`` ≈ a 1–2% Yp/D systematic at the BBN
    floor; the true-net solve closes that.) Pinned corner m_χ=1, K=100·G_F² → N_eff≈3.99, relic≈0.60
    at the coarse-h headline; h+node refinement pulls the converged value to ≈3.86.

    Default ``robust=True`` (a rank-revealing lstsq Newton, NO amplitude bound) reproduces that
    headline at matched coarse h with no clip; it is the principled successor to the old amplitude cap.
    The old ``cap_evolved=True, fdcap_logC=0.5`` amplitude clip stays opt-in to reproduce the
    historical pin.

    Per OUTER step (dlna=h) it runs ``k_sub`` inner sub-steps (dt_sub=h/k_sub), each:
      1. CROSS deposit over dt_sub — the coupled backward-Euler step for (f_ν, f_χ) under the TRUE net,
         via a log-occupation-preconditioned Newton with a backtracking line search (+ optional
         Levenberg–Marquardt ``lam``, trust-region ``trust``). ``lax.cond``-gated on ``deliver`` so
         out-of-window sub-steps skip the dense (3·2·N + Nx)-dim jacfwd.
      2. SM exp_step relaxation of f_ν + T_pl over dt_sub (a separate ν↔e± channel, SM rates at the
         bumped sub-state) so the χ deposit thermalises gradually (the 1/k_sub deposit keeps f_ν<1, the
         SM relaxation spreads the super-thermal bump between deposits).

    Three ingredients make the deep tail (x>1.6) stable and N_eff clean (each necessary):
      * expm1 detailed-balance ``_Ffac`` (in :mod:`neutt.transport.chi_brute`, default on) kills the
        catastrophic forward−reverse float cancellation at Γ≫H (the relic instability + most of the
        spurious N_eff tail).
      * Suppression of the spurious super-thermal high-ε ν tail the lepton kernels amplify (the νχ→νχ
        leg up-scatters ν off-grid to f/FD~1e4 at ε~25) — either ``robust=True`` (default) or the
        opt-in amplitude cap; see the per-arg notes below.
      * Energy-gated freeze backstop (``deepfix='freeze'``): after the coupled Newton, if the step
        violates ν↔χ NUMBER or ENERGY conservation beyond ``cons_tol`` (or fails to converge / goes
        non-finite), reject it (hold f_ν, f_χ ⇒ 0 deposit ⇒ trivially conserved). ``deepfix='none'``
        disables it (UNSTABLE at x>1.6 — expm1 alone does not obviate it).

    Key knobs:
      robust : bool (default True)
          Rank-revealing (truncated-SVD lstsq) Newton: truncates the spurious super-thermal growing
          MODE at the linear-solve level instead of clipping the spectrum. No amplitude cap, yet at
          matched coarse h reproduces the cap reference AND stabilises the heavy m=3 corner the cap
          could not. ⚠ COST: lstsq (SVD) is ~33× slower than LU on GPU → ~2–3× the cap path;
          ``solve_first=True`` recovers most of it (LU first, lstsq only on near-singular steps) at a
          ~1% accuracy loss (default off).
      cap_evolved, fdcap_logC : bool, float
          Opt-in amplitude clip on the EVOLVED ν spectrum (f ≤ e^logC·FD_ν each step) — the historical
          certified fix. logC=0.5 (C=1.65) sits just above the physical injection peak (f/FD≈1.4 at
          ε≲5) and below the spurious onset (f/FD~2.6 at ε~14): cap-insensitive over logC 0.5→1.0 and
          bit-reproducible. logC≲0.3 over-clips the peak; logC≳2 leaves residual tail. cap_evolved=False
          = reads-only cap (fragile, not recommended).
      fd_decay, monotone_read, iso_evolved : bool
          Parameter-free kernel-read source fixes for the lepton-kernel amplification. ``iso_evolved``
          (isotonic cumulative-min projection) removes the spike with no cap parameter and is
          m=3-K-monotone, but UNDER-DEPLETES the m=1 deep-eq (Γ≫H) regime (relic ~0.80 vs ~0.60), the
          depletion ``robust`` resolves. fd_decay/monotone_read are SM-bit-identical (benefit QKE/majoron).

    The SM ν↔e± + T_pl integration (exp_step, isolated-stiffness B_T) and the x-regime (pin f_χ=eq
    for x<x_switch, coupled in [x_switch, x_freeze], frozen relic above) are the same IMEX structure as
    the dark-decay head. ``h`` is the fixed ln a step (the IMEX is L-stable). The kernel quadrature
    nodes default to 8/8/8/4 (``chi_nodes``); ``num_rel``/``rho_rel`` are node-convergeable.

    Headline corner runs ~77 min on an A100 at h=0.005 (the expm1 ``_Ffac`` is ~3× the bare kernel).

    Returns
    -------
    (callable, EpsGrid)
        ``solve()`` (no args) → dict with the ``n_save``-downsampled per-flavor ν trajectory
        ``f_traj`` (n_save, 3, 2, N) + ``T_pl``/``T_cm``/``ratio``/``lna`` (so it feeds the LINX
        transport adapter directly), the final state ``f_final``/``f_chi_final``, ``neff_final``,
        ``relic_N_over_Nrel``, ``neff_chi_final``, and the conservation telemetry
        ``num_rel_max``/``rho_rel_max`` (→0 ⇒ conserved), ``cross_res_max``, ``n_badres``,
        ``nsat_total``, ``solve_ok``.
    """
    if deepfix not in ("freeze", "none"):
        raise ValueError(f"deepfix={deepfix!r} not supported; use 'freeze' (validated) or 'none'. "
                         "The shared-scalar 'gate' was rejected (suppresses the reliable freeze-out "
                         "transition).")
    if grid is None:
        grid = make_eps_grid()
    if nuratiotol is None:
        nuratiotol = A.NURATIOTOL
    if chi_nodes is None:
        chi_nodes = {"n_p2": 8, "n_c2": 8, "n_e3": 8, "n_phi": 4}
    chi_nodes = dict(chi_nodes)
    chi_nodes.setdefault("exact_e3", True)
    chi_nodes["fd_decay"] = fd_decay   # parameter-free off-grid FD-decay closure (SOURCE fix)
    chi_nodes["monotone_read"] = monotone_read   # on-grid 2-node bracket (lepton-kernel SOURCE fix; opt-in)
    chi_nodes["analytic_phi"] = analytic_phi   # EXACT azimuthal integral: drops the n_phi axis (≈4×
    # fewer kernel evals) and is the n_phi→∞ limit (removes the GL-4 ~2e-4 error). Opt-in: ON shifts
    # the headline by that error, so the default uses the GL-4 default.
    range_limit = True

    cg = chi_sector.grid
    tr = chi_sector.tratio
    Kf = chi_sector.Kflav
    gdof = chi_sector.spin_dof
    m_chi = chi_sector.m_chi
    N = grid.abscissas.shape[0]
    Nx = cg.abscissas.shape[0]
    Nocc = 3 * 2 * N
    Iz = jnp.eye(Nocc + Nx)
    OCC_FLOOR = A.OCC_FLOOR
    LOGF_MIN = -300.0
    we2_chi = cg.weights * cg.abscissas ** 2
    we2_nu = grid.weights * grid.abscissas ** 2          # ν number weights (cons gate)
    we3_nu = grid.weights * grid.abscissas ** 3          # ν energy weights (cons gate)
    N_rel = jnp.sum(we2_chi * dark_mod.dark_fd_equil(cg.abscissas, 0.0))
    # FD-envelope cap on what the χ kernel READS (lepton-kernel-boundary amplification fix).
    fd_nu = 1.0 / (jnp.exp(grid.abscissas) + 1.0)
    nu_cap = jnp.exp(fdcap_logC) * fd_nu[None, None, :]

    def _cap_nu(f):
        fc = jnp.clip(f, OCC_FLOOR, 1.0)
        if iso_evolved:
            # ISOTONIC (parameter-free, opt-in) spike removal: enforce f non-increasing in ε. Removes
            # the spurious SM-amplified Nyquist spike; no-op on the monotone genuine injection. NOTE:
            # under-depletes the m=1 deep-eq (Γ≫H) regime (relic ~0.80 vs cap_evolved ~0.60).
            return jax.lax.cummin(fc, axis=2)
        return jnp.minimum(fc, nu_cap)

    nunu_kernel = build_nunu_kernel(grid)
    nunubar_kernel = build_nunubar_kernel(grid)
    prec_trans = A.compute_prec_trans(grid, nunu_kernel, nunubar_kernel, use_sched=use_sched,
                                      range_limit=range_limit)
    occ_eq = A.equilibrium_occ(grid)

    if f_chi0 is None:
        f_chi0 = dark_mod.dark_fd_equil(cg.abscissas, m_chi / T_init * tr)
    f_chi0 = jnp.asarray(f_chi0, dtype=jnp.float64)

    nsteps = int(round(lna_max / h))
    starts = jnp.asarray(np.arange(nsteps) * h)
    sel = np.unique(np.linspace(0, nsteps - 1, n_save).astype(int))
    T_init_j = jnp.asarray(T_init, dtype=jnp.float64)
    dt_sub = h / k_sub

    def _sm_split(f_state, T_cm, T_pl, active):
        a_s, b_s, drho = A.assemble_split_rhs(grid, f_state, T_cm, T_pl, 0.0, prec_trans,
                                              nunu_kernel, nunubar_kernel, nuratiotol, m_e_eff=me,
                                              use_sched=use_sched, range_limit=range_limit,
                                              fd_cap_logC=fdcap_logC, fd_decay=fd_decay,
                                              monotone_read=monotone_read)
        return (jnp.where(active, a_s, 0.0), jnp.where(active, b_s, 0.0),
                jnp.where(active, drho, 0.0))

    @jax.jit
    def solve():
        compiling()
        def body(carry, lna):
            T_pl, f, f_chi = carry

            def subcycle(scarry, k):
                f_k, fchi_k, Tpl_k, acc_dnu, max_res, nsat_acc = scarry
                lna_k = lna + (k + 1) * dt_sub
                T_cm = T_init_j * jnp.exp(-lna_k)
                meps = m_chi / T_cm
                tcm5 = T_cm ** 5
                active = T_cm >= tcm_cutoff
                x = meps * tr
                below = (x >= x_switch) & (x <= x_freeze)
                pinned = x < x_switch
                deliver = below & active

                rho_pl, P_pl, C = _plasma_eos(Tpl_k, 0.0, True, True, True, me=me)
                rho_nu = rho_nu_from_spectrum(grid, f_k, T_cm)
                f_eq_now = dark_mod.dark_fd_equil(cg.abscissas, meps)
                _, rho_chi, _ = dark_mod.boson_moments(cg, fchi_k, meps * tr, T_cm / tr, gdof)
                H_hub = thermo.hubble(rho_pl + rho_nu + rho_chi)
                fclip = _cap_nu(f_k)                 # FD-capped ν the χ kernel reads

                # ======== 1. CROSS deposit over dt_sub (TRUE coupled net, ungated in window) ========
                def do_cross(operand):
                    f_o, fc_o = operand

                    def cross_src(z):
                        fn = z[:Nocc].reshape(3, 2, N)
                        fc = z[Nocc:]
                        fnc = _cap_nu(fn)
                        net_nu, _, net_chi, _ = chi_brute.s2_brute_rates(grid, fnc, fc, cg, meps,
                                                                         tr, Kf, **chi_nodes)
                        sn = jnp.where(deliver, tcm5 * net_nu / H_hub, 0.0)
                        sc = tcm5 * net_chi / H_hub
                        return jnp.concatenate([sn.ravel(), sc])

                    z_old = jnp.concatenate([f_o.ravel(), fc_o])

                    def Gu(u):
                        return jnp.exp(u) - z_old - dt_sub * cross_src(jnp.exp(u))

                    u0 = jnp.clip(jnp.log(jnp.clip(z_old, jnp.exp(LOGF_MIN), 1.0)), LOGF_MIN, 0.0)

                    alphas = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625])

                    def _linesearch(u, du):
                        du = jnp.clip(du, -trust, trust)

                        def try_a(a):
                            return jnp.linalg.norm(Gu(jnp.clip(u + a * du, LOGF_MIN, 0.0)))

                        res_a = jax.vmap(try_a)(alphas)
                        ib = jnp.argmin(res_a)
                        return jnp.clip(u + alphas[ib] * du, LOGF_MIN, 0.0), res_a[ib]

                    def nstep(u, _):
                        Fu = Gu(u)
                        nrm = jnp.linalg.norm(Fu)
                        J = jax.jacfwd(Gu)(u)
                        if robust:
                            # RANK-REVEALING (truncated-SVD lstsq) Newton — the no-amplitude-bound
                            # alternative to cap_evolved. On a super-thermal f_ν the coupled log-occ
                            # Jacobian is near-rank-deficient (cond~1e17): a positive-feedback growing
                            # mode (the lepton-kernel super-thermal amplification cap_evolved clips) sits
                            # at the backward-Euler stability boundary, so solve(J,−Fu) returns garbage,
                            # the Newton floors, the freeze gate rejects, and χ UNDER-depletes. lstsq
                            # truncates singular values < rcond·σ_max (the spurious direction) and takes
                            # the minimum-norm step in the well-conditioned subspace. rcond is a numerical
                            # conditioning floor, NOT physics: flat over [1e-8, 1e-5], default 1e-6. ⚠ Do
                            # NOT lower toward the float64 floor: rcond=1e-10 detonates (stops truncating
                            # the spurious mode). The freeze gate still vetoes non-conserving steps.
                            # ★ solve_first (opt-in FAST path): lstsq (SVD) is ~33× costlier than LU
                            # solve on GPU, and on a well-conditioned step solve≡lstsq. So try LU first,
                            # fall back to lstsq only when its line search fails to reduce the residual
                            # (lax.cond ⇒ SVD runs only on the rare problem steps). ⚠ NOT bit-faithful:
                            # in the gray zone where solve makes partial progress it keeps the suboptimal
                            # solve step → ~1% accuracy loss. A speed↔accuracy trade; default off.
                            if solve_first:
                                u_s, res_s = _linesearch(u, jnp.linalg.solve(J + lam * Iz, -Fu))

                                def _fallback(_):
                                    return _linesearch(u, jnp.linalg.lstsq(J, -Fu, rcond=rcond)[0])

                                u_best, res_best = jax.lax.cond(
                                    res_s < nrm, lambda _: (u_s, res_s), _fallback, operand=None)
                                return jnp.where(res_best < nrm, u_best, u), None
                            du = jnp.linalg.lstsq(J, -Fu, rcond=rcond)[0]
                        else:
                            du = jnp.linalg.solve(J + lam * Iz, -Fu)
                        u_try, res_b = _linesearch(u, du)
                        return jnp.where(res_b < nrm, u_try, u), None

                    uN, _ = jax.lax.scan(nstep, u0, None, length=k_newton)
                    zc = jnp.exp(uN)
                    f_cr = zc[:Nocc].reshape(3, 2, N)
                    fchi_cr = zc[Nocc:]
                    cres = jnp.max(jnp.abs(Gu(uN)))
                    if deepfix == "freeze":
                        # CONSERVATION-GATED freeze: accept the step only if it converged AND conserves
                        # ν↔χ NUMBER and ENERGY to ~the kernel floor; else freeze (hold f_o, fc_o ⇒ 0
                        # deposit ⇒ trivially conserved). ENERGY is essential (a number-only gate lets
                        # energy-creating steps through → N_eff blow-up).
                        dnu = jnp.sum(f_cr - f_o, axis=(0, 1))
                        dN_nu = jnp.sum(we2_nu * dnu)
                        dN_chi = (gdof / tr ** 3) * jnp.sum(we2_chi * (fchi_cr - fc_o))
                        numrel_step = (jnp.abs(dN_nu + dN_chi)
                                       / jnp.maximum(jnp.abs(dN_nu) + jnp.abs(dN_chi), 1e-300))
                        dE_nu = jnp.sum(we3_nu * dnu)
                        Echi_b = jnp.sqrt((cg.abscissas / tr) ** 2 + meps ** 2)
                        dE_chi = (gdof / tr ** 3) * jnp.sum(we2_chi * Echi_b * (fchi_cr - fc_o))
                        rhorel_step = (jnp.abs(dE_nu + dE_chi)
                                       / jnp.maximum(jnp.abs(dE_nu) + jnp.abs(dE_chi), 1e-300))
                        bad = ((numrel_step > cons_tol) | (rhorel_step > cons_tol) | (cres > 1e-3)
                               | jnp.logical_not(jnp.all(jnp.isfinite(fchi_cr)))
                               | jnp.logical_not(jnp.all(jnp.isfinite(f_cr))))
                        f_cr = jnp.where(bad, f_o, f_cr)
                        fchi_cr = jnp.where(bad, fc_o, fchi_cr)
                        cres = jnp.where(bad, 0.0, cres)
                    return f_cr, fchi_cr, cres

                def skip_cross(operand):
                    f_o, fc_o = operand
                    return f_o, fc_o, jnp.array(0.0)

                f_cross, f_chi_impl, cross_res = jax.lax.cond(deliver, do_cross, skip_cross,
                                                              (f_k, fchi_k))

                # ---- regime: in-window coupled; pin below; freeze above ----
                f_chi_new = jnp.where(below, f_chi_impl, jnp.where(pinned, f_eq_now, fchi_k))
                f_cross = jnp.where(below, f_cross, f_k)
                if iso_evolved:
                    # ISOTONIC the post-deposit ν BEFORE the SM kernels read it (else they amplify the
                    # Nyquist seed → T_pl crash). No-op on the monotone genuine deposit.
                    f_cross = jax.lax.cummin(jnp.clip(f_cross, OCC_FLOOR, 1.0), axis=2)

                # ======== 2. SM exp_step relaxation of f_ν + T_pl over dt_sub ====
                a_s, b_s, drho = _sm_split(f_cross, T_cm, Tpl_k, active)
                dT0 = (-3.0 * (rho_pl + P_pl) - drho / H_hub) / C
                _dE = 1.0e-3
                _, _, dr1 = _sm_split(f_cross, T_cm, Tpl_k * (1.0 + _dE), active)
                B_T = jnp.clip((dr1 - drho) / (Tpl_k * _dE) / (H_hub * C), 0.0, bt_max)
                A_T = dT0 + B_T * Tpl_k
                f_pre = exp_step(f_cross, a_s / H_hub, b_s / H_hub, dt_sub)
                f_next = jnp.clip(f_pre, OCC_FLOOR, 1.0)
                if iso_evolved:
                    f_next = jax.lax.cummin(f_next, axis=2)   # OPT-IN parameter-free spike removal
                elif cap_evolved:
                    f_next = jnp.minimum(f_next, nu_cap)
                Tpl_next = jnp.clip(exp_step(Tpl_k, A_T, B_T, dt_sub), 0.3 * T_cm, 2.5 * T_cm)

                acc_dnu = acc_dnu + jnp.where(deliver, f_cross - f_k, 0.0)
                max_res = jnp.maximum(max_res, jnp.where(deliver, cross_res, 0.0))
                nsat_acc = nsat_acc + jnp.sum(f_pre > 1.0) + jnp.sum(f_pre < 0.0)
                return (f_next, f_chi_new, Tpl_next, acc_dnu, max_res, nsat_acc), None

            (f_new, f_chi_new, Tpl_new, acc_dnu, max_res, nsat_acc), _ = jax.lax.scan(
                subcycle, (f, f_chi, T_pl, jnp.zeros_like(f), 0.0, 0.0), jnp.arange(k_sub))

            # ---- outer-step conservation telemetry (cross deposit summed over sub-cycle) ----
            lna1 = lna + h
            T_cm_o = T_init_j * jnp.exp(-lna1)
            meps_o = m_chi / T_cm_o
            x_o = meps_o * tr
            deliver_o = (x_o >= x_switch) & (x_o <= x_freeze) & (T_cm_o >= tcm_cutoff)
            dnu_cross = acc_dnu / h
            dfchi = (f_chi_new - f_chi) / h
            sr = chi_xspecies_sumrule(grid, chi_sector, dnu_cross, dfchi, meps_o)
            rho_rel = jnp.where(deliver_o, sr["rho_rel"], 0.0)
            num_rel = jnp.where(deliver_o, sr["num_rel"], 0.0)
            relic_now = jnp.sum(we2_chi * f_chi_new) / N_rel
            tele = jnp.array([max_res, rho_rel, nsat_acc.astype(jnp.float64), num_rel,
                              deliver_o.astype(jnp.float64), relic_now, Tpl_new])
            return (Tpl_new, f_new, f_chi_new), (tele, f_new)

        (T_f, f_f, fc_f), (tele, f_traj_all) = jax.lax.scan(
            body, (T_init_j, occ_eq, f_chi0), starts)

        cres = tele[:, 0]
        act = tele[:, 4] > 0
        nr = tele[:, 3]
        rr = tele[:, 1]
        T_cm_fin = T_init_j * jnp.exp(-lna_max)
        _, rho_chi_fin, _ = dark_mod.boson_moments(cg, fc_f, (m_chi / T_cm_fin) * tr,
                                                   T_cm_fin / tr, gdof)
        rho_1nu_std = (7.0 / 8.0) * (PI**2 / 15.0) * ((4.0 / 11.0) ** (1.0 / 3.0) * T_f) ** 4
        relic = jnp.sum(we2_chi * fc_f) / N_rel

        # ---- n_save-downsampled trajectory (per-flavor ν for the LINX transport adapter) ----
        lna_nodes = starts[sel] + h
        T_pl_nodes = tele[sel, 6]
        T_cm_nodes = T_init_j * jnp.exp(-lna_nodes)
        f_traj_nodes = f_traj_all[sel]                  # (n_save, 3, 2, N)

        nsat_total = jnp.sum(tele[:, 2])
        n_badres = jnp.sum(cres > 1e-3)
        solve_ok = (jnp.all(jnp.isfinite(f_f)) & jnp.all(jnp.isfinite(fc_f)) & jnp.isfinite(T_f)
                    & (T_f > 0.0) & (n_badres == 0))
        return {"lna": lna_nodes, "T_pl": T_pl_nodes, "T_cm": T_cm_nodes,
                "ratio": T_pl_nodes / T_cm_nodes, "f_traj": f_traj_nodes,
                "f_final": f_f, "ratio_final": T_f / T_cm_fin,
                "neff_final": neff_from_spectrum(grid, f_f, T_cm_fin, T_f),
                "f_chi_final": fc_f, "rho_chi_final": rho_chi_fin,
                "neff_chi_final": rho_chi_fin / rho_1nu_std,
                "relic_N_over_Nrel": relic,
                "rho_rel_max": jnp.max(rr),
                "rho_rel_med": jnp.nanmedian(jnp.where(act, rr, jnp.nan)),
                "num_rel_max": jnp.max(nr),
                "num_rel_med": jnp.nanmedian(jnp.where(act, nr, jnp.nan)),
                "cross_res_max": jnp.max(cres), "n_badres": n_badres,
                "nsat_total": nsat_total, "solve_ok": solve_ok,
                "relic_traj": tele[:, 5], "lna_all": starts + h}

    return solve, grid


def make_dark_decay_implicit_solver(dark, grid=None, *, lna_sw=3.0, n_head=600, k_sub=8,
                                    delivery="kernel", monotone_x=True, x_cap=None, e_cap_frac=0.1,
                                    dark_range_limit=False, dark_offgrid_mode="floor",
                                    dark_forward_only_tcm=0.0,
                                    head_power=1.0, n_save_head=96, n_save_tail=96,
                                    use_sched=False, use_renorm=True, renorm_pdep=True,
                                    nuratiotol=None, tcm_cutoff=0.015, phie=0.0,
                                    eta0=None, q_baryon=0.5,
                                    xi=(0.0, 0.0, 0.0), tail_eps_split=10.0, tail_feature_weight=0.05,
                                    sm_tail_closure="srcfix", sm_fd_cap_logC=None,
                                    max_steps=50_000, pcoeff=0.4, icoeff=0.3, dcoeff=0.0):
    """IMEX implicit-X + sub-cycled-ν solver for the stiff overabundant-majoron decay.

    The strong-χ energy-delivery sub-cycle pattern ported to the linear νν↔J decay. Two earlier heads
    were removed (last present in 430dfb3): a plain per-bin expMid head (no Pauli f ≤ 1: the over-deposited
    ν feature drives B_X < 0 and the majoron never depletes) and a Pauli-clipped sub-cycled head that
    operator-split the stiff X↔ν exchange (X decayed on the start-of-step cold ν → B_X over-estimated →
    X over-depleted ~4× and the energy lost → ρ_ν ~18% low).

    The fix has three ingredients:
      1. X depleted once per outer step h by the exact linear-affine ``exp_step(f_x,a_x,b_x,h)`` (the
         decay X eq is linear in f_x → exp_step is the exact, L-stable solution — no Newton).
         ``monotone_x`` clamps f_x ≤ its start value (the overabundant majoron only depletes; kills the
         b_X<0 inverse-decay growth). A_X,B_X read the start-of-step ν, so the operator-split error is
         O(h), controlled by raising ``n_head``.
      2. Energy-matched delivery: ν gains exactly the energy X lost over the step (ε³ comoving measure),
         spread over the forward decay ν-gain spectrum ``a_nu_d`` (energy-normalized). ``e_cap_frac``
         caps |E_dep| at frac·ρ_ν/step (a safety; watch ``cap_bind_frac``).
      3. Sub-cycle the SM ν+T_pl relaxation k_sub times, carrying the frozen χ→ν source as a constant
         affine gain. The SM loss b (incl. ν-ν) is recomputed at the bumped f each sub-step so the
         feature thermalises on its physical timescale and stays ≤1 via the L-stable exp_step.

    If the fast decay over-depletes per step, lower h (raise n_head) until X and ρ_ν match the explicit
    reference.

    ``dark_offgrid_mode`` (default ``"floor"``): how the dark kernels read the ν spectrum above ε_max
    (the inverse-decay integral samples ν up to E_X/T_cm ≫ ε_max). ``"floor"`` zeros reads above ε_max
    — correct and cap-free, since the decay feature peaks at ε=(m/2)/T_cm < ε_max and the off-grid ν
    energy is FD-suppressed. ``"linear"`` is the (also cap-free) log-linear downward extrapolation.
    ``"default"`` order-5-extrapolates and OVERSHOOTS (a spurious energy-creating inverse-decay pump),
    so it must be paired with ``dark_range_limit=True`` and is no longer the default.
    ``dark_range_limit`` (default ``False``): window-max cap on the off-grid ν reads — only needed by
    the overshooting ``"default"`` mode (inert under ``"floor"``/``"linear"``).

    ``dark_forward_only_tcm`` (default 0.0 = off): below this T_cm [MeV] the dark kernels drop the
    inverse decay νν̄→X (forward X→νν̄ only). The inverse decay is physically ∝e^{−m/T}→0 at T≲0.5 MeV,
    so gating it off lets the head run clean to full X depletion (capturing the matter-boosted residual
    that capping the head early would leave uncaptured). Set ~0.6–0.8 so the switch sits at the off-grid break.

    The energy ledger is in the output: ``E_nu_decay`` (comoving ν energy the affine decay source
    injects) vs ``E_x_lost`` (X energy released); ``decay_residual_frac`` is their fractional mismatch
    (≈0 ⇒ energy-conserving). ``exlost_traj``/``enudec_traj`` give the cumulative trajectory.

    Returns
    -------
    (callable, EpsGrid)
        ``solve(T_init=4.0, lna_max=7.0, f_x0=None, rtol=1e-6, atol=1e-30, dtmax=0.08,
        delta_neff_init=0.0, me=None, xi=None)`` → dict with lna/T_pl/T_cm/ratio/f_traj/f_final/
        ratio_final/neff_final (the stitched head + tail trajectory), the handoff state ``f_sw``/
        ``T_sw``/``ratio_sw``, the X diagnostics ``fx_head``, ``f_x_handoff``, ``rho_x_handoff``/
        ``neff_x_handoff`` (the frozen residual — energy-honest iff ≪ ρ_1ν) and ``f_x_number_ratio``,
        the clip diagnostics ``head_max_f``/``head_n_clip``/``E_clip_leak``/``clip_leak_frac``/
        ``cap_bind_frac``, the energy and number ledgers above, ``eps_tail_energy_frac``/
        ``eps_top_energy_frac``, and the solve health ``head_T_ok``/``head_f_finite``/``head_n_neg``/
        ``tail_steps``/``tail_rejected``/``tail_ok``.
    """
    if grid is None:
        grid = make_eps_grid()
    if nuratiotol is None:
        nuratiotol = A.NURATIOTOL
    OCC_FLOOR = A.OCC_FLOOR
    if sm_fd_cap_logC is None:
        sm_fd_cap_logC = float(np.log(FD_ENVELOPE_C))
    # ---- SM-leg tail closure (the lepton kernels inside the dark run) ----
    # The dark decay injects a genuinely super-FD feature into f_ν that the SM lepton kernels then
    # READ off-grid. "srcfix" (DEFAULT — the more physical, parameter-free choice) uses fd_decay
    # (off-grid read anchored to the LIVE grid-edge f/FD ratio — preserves a genuine super-FD edge,
    # only bounds the polynomial overshoot) + monotone_read, with NO FD-cap. This is the QKE
    # tail_closure pattern applied to the dark run's SM legs; it does NOT touch the dark decay kernels
    # (which read off-grid via dark_offgrid_mode). "cap" is the legacy bound f ≤ C·FD (sm_fd_cap_logC,
    # C=10) — but an absolute C·FD cap CAN clip the injected super-thermal feature (the SCRUB_CAPS §10
    # concern). Validated equivalent: canonical majoron (m_J=5, g=3.5e-10, 5×BE) gives ΔN_eff=+2.43216
    # (cap) vs +2.43215 (srcfix) — identical to 1e-5, energy+number conserving, C=10 inert
    # (cap_bind_frac=0) — so srcfix is the default with no deliverable change.
    if sm_tail_closure == "srcfix":
        sm_fd_decay, sm_monotone_read = True, True
        sm_fd_cap_logC = float("inf")
    elif sm_tail_closure == "cap":
        sm_fd_decay, sm_monotone_read = False, False
    else:
        raise ValueError("sm_tail_closure must be 'srcfix' or 'cap'; got %r" % (sm_tail_closure,))

    nunu_kernel = build_nunu_kernel(grid)
    nunubar_kernel = build_nunubar_kernel(grid)
    prec_trans = A.compute_prec_trans(grid, nunu_kernel, nunubar_kernel, xi=xi,
                                      use_sched=use_sched)
    occ0 = A.equilibrium_occ(grid, xi=xi)

    edges = head_edges(lna_sw, n_head, head_power)
    starts = jnp.asarray(edges[:-1])
    hs = jnp.asarray(np.diff(edges))
    sel = np.linspace(0, n_head, n_save_head, endpoint=False).astype(int)
    lna_head_nodes = jnp.asarray(edges[sel])
    tail_norm = make_decay_tail_norm(grid, tail_eps_split, tail_feature_weight)

    @jax.jit
    def solve(T_init=4.0, lna_max=7.0, f_x0=None, rtol=1e-6, atol=1e-30, dtmax=0.08,
              delta_neff_init=0.0, me=None, xi=None):
        compiling()
        if me is None:
            me = M_ELECTRON
        T_init = jnp.asarray(T_init, dtype=jnp.float64)
        # Per-call degeneracy (see make_hybrid_solver): xi=None reuses the build-time bake
        # bit-for-bit; a traced xi (3,) recomputes prec_trans/occ0 inside the jit (no recompile).
        if xi is None:
            prec_use, occ_use = prec_trans, occ0
        else:
            _xi = jnp.asarray(xi, dtype=jnp.float64)
            prec_use = A.compute_prec_trans(grid, nunu_kernel, nunubar_kernel, xi=_xi,
                                            use_sched=use_sched)
            occ_use = A.equilibrium_occ(grid, xi=_xi)
        if f_x0 is None:
            f_x0 = dark_mod.dark_be_equil(dark.grid.abscissas, dark.m_x / T_init * dark.tratio)
        f_x0 = jnp.asarray(f_x0, dtype=jnp.float64)
        fx0_number = jnp.sum(dark.grid.weights * dark.grid.abscissas**2 * f_x0)
        x_cap_v = jnp.max(f_x0) if x_cap is None else jnp.asarray(x_cap, dtype=jnp.float64)

        # dark+SM affine for the OUTER step (X depletion + decay shape); SM-only for the sub-cycle.
        split_dark = make_split_rates(grid, prec_use, nunu_kernel, nunubar_kernel, T_init,
                                      nuratiotol, use_renorm, use_sched=use_sched,
                                      renorm_pdep=renorm_pdep, tcm_cutoff=tcm_cutoff, phie=phie,
                                      eta0=eta0, q_baryon=q_baryon,
                                      delta_neff_init=delta_neff_init, me=me, fd_cap_logC=sm_fd_cap_logC,
                                      fd_decay=sm_fd_decay, monotone_read=sm_monotone_read,
                                      dark=dark, separate_dark=True,
                                      dark_range_limit=dark_range_limit,
                                      dark_offgrid_mode=dark_offgrid_mode,
                                      dark_forward_only_tcm=dark_forward_only_tcm)
        split_sm = make_split_rates(grid, prec_use, nunu_kernel, nunubar_kernel, T_init,
                                    nuratiotol, use_renorm, use_sched=use_sched,
                                    renorm_pdep=renorm_pdep, tcm_cutoff=tcm_cutoff, phie=phie,
                                    eta0=eta0, q_baryon=q_baryon,
                                    delta_neff_init=delta_neff_init, me=me, fd_cap_logC=sm_fd_cap_logC,
                                    fd_decay=sm_fd_decay, monotone_read=sm_monotone_read)

        we3 = grid.weights * grid.abscissas**3
        we2 = grid.weights * grid.abscissas**2          # ν number weight (ε³-blind sum-rule)
        dxw2 = dark.grid.weights * dark.grid.abscissas**2
        dxq = dark.grid.abscissas

        def x_energy(f_x, T_cm):
            meps = dark.m_x / T_cm
            e_bin = jnp.sqrt((dxq / dark.tratio) ** 2 + meps ** 2)
            return (dark.spin_dof / dark.tratio**3) * jnp.sum(dxw2 * e_bin * f_x)

        def x_number(f_x):
            # X number moment in the comoving T_cm³/2π² units (n_X·a³ = x_number·T_init³/2π²), the
            # ε³-blind partner of x_energy for the number sum rule (1 X → 2 leptons).
            return (dark.spin_dof / dark.tratio**3) * jnp.sum(dxw2 * f_x)

        zero = jnp.zeros((), dtype=jnp.float64)

        def outer_kernel(carry, x):
            # delivery='kernel' (default): sub-cycle X and ν together with the live decay affine — the
            # ν source is the actual kernel a_nu_d − b_nu_d·f (not an energy-renormalized surrogate), so
            # the head solves the same equations as the explicit Tsit5 (the linear decay is clean). X by
            # exp_step (exact, linear), monotone-clamped (kills the b_X<0 inverse-decay runaway). Energy
            # conserved by the kernel's own detailed balance (X loss == ν gain per the affine).
            T_pl, f, f_x, e_xlost, e_deliv, e_clip, e_nudec, n_nudec = carry
            lna0, h = x
            dt_sub = h / k_sub
            T_cm0 = T_init * jnp.exp(-lna0)
            fx0_loc = f_x

            def sub(sc, k):
                f_k, T_k, fx_k, ecl, end, nnd = sc
                lk = lna0 + (k + 1) * dt_sub
                a_sm, b_sm, a_nu_d, b_nu_d, a_x, b_x, dT_k, _H, _rx = split_dark(lk, T_k, f_k, fx_k)
                fx_raw = jnp.clip(exp_step(fx_k, a_x, b_x, dt_sub), 0.0, x_cap_v)
                fx_next = jnp.minimum(fx_raw, fx_k) if monotone_x else fx_raw
                f_raw = exp_step(f_k, a_sm + a_nu_d, b_sm + b_nu_d, dt_sub)
                f_next = jnp.clip(f_raw, OCC_FLOOR, 1.0)
                ecl = ecl + jnp.sum(we3 * jnp.sum(f_raw - f_next, axis=(0, 1)))
                # energy ledger (not the E_delivered==E_chi tautology): the comoving ν energy the decay
                # actually lands this sub-step, via a decay-alone L-stable exp_step clipped to the Pauli
                # bound (the forward-Euler form diverges to ~1e17 once the off-grid rates blow up; exp_step
                # saturates instead). residual vs E_x_lost = the discrete detailed-balance / off-grid
                # energy violation, in T_init⁴/2π² comoving units.
                f_dec = jnp.clip(exp_step(f_k, a_nu_d, b_nu_d, dt_sub), OCC_FLOOR, 1.0)
                ddec = jnp.sum(f_dec - f_k, axis=(0, 1))             # (N,) cp+fl-summed Δf from decay
                end = end + jnp.sum(we3 * ddec)
                # number sum rule (ε³-blind): ν number injected = Σ we2·Δf_decay. vs 2×(X number lost):
                # nonzero ⇒ the off-grid pump creates particles; ~0 with a large energy residual ⇒ pure
                # ε³ high-tail amplification (no spurious particles).
                nnd = nnd + jnp.sum(we2 * ddec)
                T_next = T_k + dt_sub * dT_k
                return (f_next, T_next, fx_next, ecl, end, nnd), None

            (f_new, T_new, fx_new, ecl_step, end_step, nnd_step), _ = jax.lax.scan(
                sub, (f, T_pl, f_x, zero, zero, zero), jnp.arange(k_sub))
            E_chi = x_energy(fx0_loc, T_cm0) - x_energy(fx_new, T_cm0)
            e_xlost = e_xlost + E_chi
            e_deliv = e_deliv + E_chi                                  # kernel: delivery == kernel gain
            e_clip = e_clip + ecl_step
            e_nudec = e_nudec + end_step
            n_nudec = n_nudec + nnd_step
            head_maxf = jnp.max(f_new)
            return (T_new, f_new, fx_new, e_xlost, e_deliv, e_clip, e_nudec, n_nudec), \
                (T_new, f_new, fx_new, head_maxf, e_xlost, e_nudec)

        def outer_energy(carry, x):
            # delivery='energy': deplete X once per outer step, deliver −E_chi over the energy-normalized
            # forward shape, sub-cycle SM-only ν relaxation. Converges (flat in h) to the WRONG ν spectrum
            # (the renormalized surrogate ≠ the real kernel) — kept for contrast/diagnostics.
            T_pl, f, f_x, e_xlost, e_deliv, e_clip, e_nudec, n_nudec = carry
            lna0, h = x
            dt_sub = h / k_sub
            T_cm0 = T_init * jnp.exp(-lna0)
            a_sm, b_sm, a_nu_d, _b_nu_d, a_x, b_x, _dT0, H0, _rx = split_dark(lna0, T_pl, f, f_x)
            fx_step = jnp.clip(exp_step(f_x, a_x, b_x, h), 0.0, x_cap_v)
            f_x_new = jnp.minimum(fx_step, f_x) if monotone_x else fx_step
            E_chi = x_energy(f_x, T_cm0) - x_energy(f_x_new, T_cm0)
            nu_E = jnp.sum(we3 * jnp.sum(f, axis=(0, 1)))
            shape = jnp.maximum(a_nu_d, 0.0)
            shape_E = jnp.sum(we3 * jnp.sum(shape, axis=(0, 1)))
            E_dep = jnp.clip(E_chi, -e_cap_frac * nu_E, e_cap_frac * nu_E)
            okE = shape_E > 1e-300
            inj_step = jnp.where(okE, E_dep * shape / jnp.where(okE, shape_E, 1.0), 0.0)
            S_rate = inj_step * (H0 / h)

            def sub(sc, k):
                f_k, T_k, ecl = sc
                lk = lna0 + (k + 1) * dt_sub
                a_k, b_k, dT_k, H_k = split_sm(lk, T_k, f_k)
                f_raw = exp_step(f_k, a_k + S_rate / H_k, b_k, dt_sub)
                f_next = jnp.clip(f_raw, OCC_FLOOR, 1.0)
                ecl = ecl + jnp.sum(we3 * jnp.sum(f_raw - f_next, axis=(0, 1)))
                T_next = T_k + dt_sub * dT_k
                return (f_next, T_next, ecl), None

            (f_new, T_new, ecl_step), _ = jax.lax.scan(sub, (f, T_pl, zero), jnp.arange(k_sub))
            e_xlost = e_xlost + E_chi
            e_deliv = e_deliv + E_dep
            e_clip = e_clip + ecl_step
            e_nudec = e_nudec + E_dep                                  # energy path: delivered == E_dep
            n_nudec = n_nudec + jnp.sum(we2 * jnp.sum(inj_step, axis=(0, 1)))  # number delivered (shape)
            head_maxf = jnp.max(f_new)
            return (T_new, f_new, f_x_new, e_xlost, e_deliv, e_clip, e_nudec, n_nudec), \
                (T_new, f_new, f_x_new, head_maxf, e_xlost, e_nudec)

        outer = outer_kernel if delivery == "kernel" else outer_energy

        (T_sw, f_sw, fx_sw, E_xlost, E_deliv, E_clip, E_nudec, N_nudec), \
            (T_traj, f_traj_head, fx_traj_head, maxf_traj, exlost_traj, enudec_traj) = jax.lax.scan(
                outer, (T_init, occ_use, f_x0, zero, zero, zero, zero, zero), (starts, hs))
        # number sum rule: X number lost is ε³-blind / T_cm-independent ⇒ just the moment drop (no per-
        # step accumulation needed); 2 leptons per X decay ⇒ compare N_nudec to 2·N_x_lost.
        N_x_lost = x_number(f_x0) - x_number(fx_sw)
        T_at_edges = jnp.concatenate([T_init[None], T_traj])
        f_at_edges = jnp.concatenate([occ_use[None], f_traj_head])
        fx_at_edges = jnp.concatenate([f_x0[None], fx_traj_head])

        # ---- head sanity + hand-off ----
        T_ok = jnp.isfinite(T_sw) & (T_sw > 0.0)
        f_finite = jnp.all(jnp.isfinite(f_sw)) & jnp.all(jnp.isfinite(fx_sw))
        n_neg = jnp.sum(f_sw < 0.0)
        head_max_f = jnp.max(maxf_traj)
        head_n_clip = jnp.sum(f_sw >= 1.0 - 1e-12)
        fx_number_ratio = jnp.sum(dark.grid.weights * dark.grid.abscissas**2 * fx_sw) / fx0_number
        T_cm_sw = T_init * jnp.exp(-jnp.asarray(lna_sw))
        meps_sw = dark.m_x / T_cm_sw
        _, rho_x_sw, _ = dark_mod.boson_moments(dark.grid, fx_sw, meps_sw * dark.tratio,
                                                T_cm_sw / dark.tratio, dark.spin_dof)
        rho_1nu_sw = (7.0 / 8.0) * (PI**2 / 15.0) * ((4.0 / 11.0) ** (1.0 / 3.0) * T_sw) ** 4

        # ---- SM-only Tsit5 tail from lna_sw (f_x dropped; depleted residual frozen in f) ----
        ts_tail = jnp.linspace(jnp.asarray(lna_sw), lna_max, n_save_tail)
        sm_rhs = make_rhs(grid, prec_use, nunu_kernel, nunubar_kernel, T_init,
                          nuratiotol, use_renorm, use_sched=use_sched,
                          renorm_pdep=renorm_pdep, tcm_cutoff=tcm_cutoff,
                          phie=phie, eta0=eta0, q_baryon=q_baryon, delta_neff_init=delta_neff_init, me=me)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(sm_rhs), diffrax.Tsit5(), t0=jnp.asarray(lna_sw), t1=lna_max,
            dt0=1e-3, y0={"T_pl": T_sw, "f": f_sw},
            saveat=diffrax.SaveAt(ts=ts_tail), adjoint=diffrax.DirectAdjoint(),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol, dtmax=dtmax,
                                                      pcoeff=pcoeff, icoeff=icoeff,
                                                      dcoeff=dcoeff, norm=tail_norm),
            max_steps=max_steps, throw=False)

        lna = jnp.concatenate([lna_head_nodes, ts_tail])
        T_pl = jnp.concatenate([T_at_edges[sel], sol.ys["T_pl"]])
        f_traj = jnp.concatenate([f_at_edges[sel], sol.ys["f"]])
        T_cm = T_init * jnp.exp(-lna)
        T_fin = sol.ys["T_pl"][-1]
        f_fin = sol.ys["f"][-1]
        T_cm_fin = T_init * jnp.exp(-lna_max)
        tail_frac, top_frac = high_eps_energy_frac(grid, f_fin, tail_eps_split)
        return {"lna": lna, "T_pl": T_pl, "T_cm": T_cm, "ratio": T_pl / T_cm,
                "f_traj": f_traj, "f_final": f_fin,
                "ratio_final": T_fin / T_cm_fin,
                "neff_final": neff_from_spectrum(grid, f_fin, T_cm_fin, T_fin),
                "f_sw": f_sw, "T_sw": T_sw, "ratio_sw": T_sw / T_cm_sw,
                "lna_head_nodes": lna_head_nodes, "T_head": T_at_edges[sel],
                "fx_head": fx_at_edges[sel], "f_x_handoff": fx_sw, "rho_x_handoff": rho_x_sw,
                "neff_x_handoff": rho_x_sw / rho_1nu_sw,
                "head_max_f": head_max_f, "head_n_clip": head_n_clip,
                "f_x_number_ratio": fx_number_ratio,
                "E_x_lost": E_xlost, "E_delivered": E_deliv, "E_clip_leak": E_clip,
                "clip_leak_frac": E_clip / jnp.maximum(jnp.abs(E_xlost), 1e-300),
                "cap_bind_frac": (E_xlost - E_deliv) / jnp.maximum(jnp.abs(E_xlost), 1e-300),
                # energy ledger: E_nu_decay (comoving ν energy the affine decay source actually injects)
                # vs E_x_lost (X energy released). residual>0 ⇒ energy created (the off-grid inverse-decay
                # pump). Not the E_delivered==E_x_lost tautology above.
                "E_nu_decay": E_nudec, "decay_residual": E_nudec - E_xlost,
                "decay_residual_frac": (E_nudec - E_xlost) / jnp.maximum(jnp.abs(E_xlost), 1e-300),
                "exlost_traj": exlost_traj[sel], "enudec_traj": enudec_traj[sel],
                # number sum rule (ε³-blind): n_ν injected vs 2×(X number lost). residual≈0 + large
                # energy residual ⇒ pure ε³ high-tail amplification (no spurious particles); both
                # nonzero ⇒ the off-grid inverse-decay pump creates real particles.
                "N_nu_decay": N_nudec, "N_x_lost": N_x_lost,
                "number_residual_frac": (N_nudec - 2.0 * N_x_lost)
                / jnp.maximum(jnp.abs(2.0 * N_x_lost), 1e-300),
                "eps_tail_energy_frac": tail_frac, "eps_top_energy_frac": top_frac,
                "head_T_ok": T_ok, "head_f_finite": f_finite, "head_n_neg": n_neg,
                "tail_steps": sol.stats["num_accepted_steps"],
                "tail_rejected": sol.stats["num_rejected_steps"],
                "tail_ok": sol.result == diffrax.RESULTS.successful}

    return solve, grid
