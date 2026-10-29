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
"""

from __future__ import annotations

import numpy as np
import diffrax
import jax
import jax.numpy as jnp

from . import compiling
from .constants import M_ELECTRON
from .decoupling import make_rhs, make_split_rates, nanrobust_rms_norm, neff_from_spectrum
from .thermo import Thermo
from .grid import make_eps_grid
from .transport import assemble as A
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


def make_hybrid_solver(grid=None, *, thermo=None, lna_sw=1.5, n_head=1600, head_power=1.0,
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
    if thermo is None:
        thermo = Thermo()
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
                                 delta_neff_init=delta_neff_init, me=me, lag_tail_fix=lag_tail_fix,
                                 thermo=thermo)

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
                       lag_tail_fix=lag_tail_fix, thermo=thermo)
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
