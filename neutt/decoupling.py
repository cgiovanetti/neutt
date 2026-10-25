"""The coupled decoupling system: plasma temperature T_pl and neutrino spectrum f[3, 2, N] in ln a.

State ``{'T_pl': scalar [MeV], 'f': (3, 2, N)}``; T_cm = T_init·e^{−ln a} (comoving momenta are
frozen).  The temperature equation and the collision RHS are BURST's:

    dT_pl/dln a = [−3(ρ_pl + P_pl) − ρ̇_ν/H] / (dρ_pl/dT),     df/dln a = C[f]/H,

with H = H(ρ_pl + ρ_ν + ρ_DR).  ``make_rhs`` gives the vector field for an adaptive solve,
``make_split_rates`` the same physics in affine form A − B·f for the exponential-midpoint step, and
``make_decoupling_solver`` the reference adaptive solve.
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx

from . import compiling, thermo
from .constants import M_ELECTRON, PI
from .grid import make_eps_grid
from .transport import assemble as A
from .transport.nunu import build_nunu_kernel
from .transport.nunubar import build_nunubar_kernel

_8_7 = 8.0 / 7.0
_11_4_43 = (11.0 / 4.0) ** (4.0 / 3.0)
# delta_neff_init uses LINX's Delt_Neff_init convention: ρ_DR = Δ·(4/11)^{4/3}·(7/8)(π²/15)T_cm⁴, one
# fermionic dof per unit at the transport start.  (BURST's ``rhors`` has no (4/11)^{4/3}.)
_DR_DNEFF_FACTOR = (4.0 / 11.0) ** (4.0 / 3.0)


def plasma_history_ok(T_pl_hist, T_init, ratio_final):
    """True while a fixed-step solve's plasma history is physical: finite, T_pl never above T_init,
    and the final T_pl/T_cm in [0.99, 1.45] (physically 1 … (11/4)^{1/3}).  A step too coarse for the
    T_pl update breaks it by orders of magnitude."""
    ok = jnp.all(jnp.isfinite(T_pl_hist)) & jnp.all(T_pl_hist <= T_init * (1.0 + 1e-3))
    return ok & (ratio_final >= 0.99) & (ratio_final <= 1.45)


def nanrobust_rms_norm(x):
    """diffrax's RMS error norm with NaN mapped to +inf.

    A NaN error estimate slips the PID controller's guards and sets dt = NaN (a permanent rejection
    cascade); +inf takes the controller's designed reject-and-shrink path instead.  Identical on
    every finite step.
    """
    r = optx.rms_norm(x)
    return jnp.where(jnp.isnan(r), jnp.asarray(jnp.inf, r.dtype), r)


def rho_nu_from_spectrum(grid, occ, T_cm):
    """ρ_ν [MeV⁴] = (T_cm⁴/2π²) Σ_{flavour, cp} Σ_i w_i ε_i³ f (negative tail excursions clipped)."""
    eps = grid.abscissas
    w = grid.weights
    occ = jnp.maximum(occ, 0.0)
    return T_cm**4 / (2.0 * PI**2) * jnp.sum(w * eps**3 * jnp.sum(occ, axis=(0, 1)))


def neff_from_spectrum(grid, occ, T_cm, T_pl):
    """N_eff = (8/7)(11/4)^{4/3} ρ_ν/ρ_γ (valid once e± annihilation is complete)."""
    return _8_7 * _11_4_43 * rho_nu_from_spectrum(grid, occ, T_cm) / thermo.rho_gamma(T_pl)


def _plasma_eos(T_pl, phie, use_renorm, renorm_pdep, phie_static_zero, me=M_ELECTRON,
                qed_nlo=True):
    """(ρ_pl, P_pl, C = dρ_pl/dT) of the e±/γ plasma, plus the finite-T QED δρ/δP/δC if use_renorm."""
    rho_pl = thermo.rho_plasma(T_pl, phie, me)
    P_pl = thermo.pressure_plasma(T_pl, phie, me)
    C = (thermo.drho_plasma_dT(T_pl, me) if phie_static_zero
         else thermo.drho_plasma_dT_phi(T_pl, phie, me))
    if use_renorm:
        drho_q, dP_q, dC_q = thermo.qed_plasma_deltas(T_pl, phie, pdep=renorm_pdep, m_e=me,
                                                       nlo=qed_nlo)
        rho_pl = rho_pl + drho_q
        P_pl = P_pl + dP_q
        C = C + dC_q
    return rho_pl, P_pl, C


def _phie_fn(phie, eta0, q_baryon, me):
    """φ_e(T_pl, T_cm): the fixed float ``phie``, or charge neutrality for ``phie="auto"``."""
    if isinstance(phie, str) and phie == "auto":
        _eta0 = thermo.ETA0_DEFAULT if eta0 is None else eta0
        return lambda T_pl, T_cm: thermo.solve_phie(T_pl, T_cm / T_pl, _eta0, q_baryon, m_e=me)
    return lambda T_pl, T_cm: phie


def _background(grid, occ, T_pl, T_cm, phie, drhonudt, use_renorm, renorm_pdep, phie_static_zero,
                me, qed_nlo, delta_neff_init):
    """H and dT_pl/dln a from the plasma EoS, ρ_ν, the inert dark radiation and ρ̇_ν."""
    rho_pl, P_pl, C = _plasma_eos(T_pl, phie, use_renorm, renorm_pdep, phie_static_zero,
                                  me=me, qed_nlo=qed_nlo)
    rho_nu = rho_nu_from_spectrum(grid, occ, T_cm)
    rho_dr = delta_neff_init * _DR_DNEFF_FACTOR * (7.0 / 8.0) * (PI**2 / 15.0) * T_cm**4
    H = thermo.hubble(rho_pl + rho_nu + rho_dr)
    dT_pl = (-3.0 * (rho_pl + P_pl) - drhonudt / H) / C
    return H, dT_pl


def make_rhs(grid, prec_trans, nunu_kernel, nunubar_kernel, T_cm_init,
             nuratiotol=A.NURATIOTOL, use_renorm=True, use_sched=False, renorm_pdep=True,
             qed_nlo=True, tcm_cutoff=0.015, phie=0.0, eta0=None, q_baryon=0.5,
             delta_neff_init=0.0, me=M_ELECTRON, lag_tail_fix=True):
    """The vector field ``rhs(lna, state, args) -> {'T_pl': dT_pl, 'f': df}`` per d(ln a).

    Parameters
    ----------
    grid, prec_trans, nunu_kernel, nunubar_kernel
        The ε-grid, the equilibrium noise floor (``assemble.compute_prec_trans``) and the two
        grid-only ν–ν kernels.
    T_cm_init : float
        T_cm at ln a = 0 [MeV].
    nuratiotol : float
        BURST's precision-gate threshold.
    use_renorm, renorm_pdep, qed_nlo : bool
        Finite-T QED in the plasma EoS: the O(α) δρ/δP (BURST), its momentum-dependent part, and the
        NLO O(e³) term.  All on by default; ``use_renorm=False`` gives the decoupling-only N_eff.
    use_sched : bool or str
        Lepton-kernel implementation: False (reference), True (scheduled, per-bin regime select) or
        "union" (scheduled, regime-masked).
    tcm_cutoff : float or None
        Rates are zero below this T_cm [MeV] (BURST's 0.015); None disables.
    phie, eta0, q_baryon
        Electron degeneracy μ_e/T_pl: a fixed float, or ``"auto"`` for charge neutrality
        n_{e⁻} − n_{e⁺} = q_baryon·n_b at the present baryon-to-photon ratio ``eta0``.
    delta_neff_init : float
        Inert dark radiation in H (LINX's Delt_Neff_init convention).
    me : float
        Electron mass [MeV]; reaches the lepton kernels, the free-gas EoS and the QED terms (may be
        traced).
    lag_tail_fix : bool
        Use the Gauss–Laguerre tail rule on the wide-regime nuer1/nuer2/nuepma bins, where BURST's
        Legendre segment spans a near-infinite interval (False = bit-faithful to BURST).
    """
    phie_at = _phie_fn(phie, eta0, q_baryon, me)
    phie_static_zero = isinstance(phie, (int, float)) and phie == 0.0

    def rhs(lna, state, args):
        T_pl = state["T_pl"]
        occ = state["f"]
        T_cm = T_cm_init * jnp.exp(-lna)
        phie_use = phie_at(T_pl, T_cm)
        dfdt, drhonudt = A.assemble_rhs(grid, occ, T_cm, T_pl, phie_use, prec_trans,
                                        nunu_kernel, nunubar_kernel, nuratiotol,
                                        m_e_eff=me, use_sched=use_sched, lag_tail_fix=lag_tail_fix)
        if tcm_cutoff is not None:
            active = T_cm >= tcm_cutoff
            dfdt = jnp.where(active, dfdt, 0.0)
            drhonudt = jnp.where(active, drhonudt, 0.0)
        H, dT_pl = _background(grid, occ, T_pl, T_cm, phie_use, drhonudt, use_renorm, renorm_pdep,
                               phie_static_zero, me, qed_nlo, delta_neff_init)
        return {"T_pl": dT_pl, "f": dfdt / H}

    return rhs


def make_split_rates(grid, prec_trans, nunu_kernel, nunubar_kernel, T_cm_init,
                     nuratiotol=A.NURATIOTOL, use_renorm=True, use_sched=False,
                     renorm_pdep=True, qed_nlo=True, tcm_cutoff=0.015, phie=0.0,
                     eta0=None, q_baryon=0.5, delta_neff_init=0.0,
                     me=M_ELECTRON, fd_cap_logC=float("inf"), lag_tail_fix=True,
                     fd_decay=False, monotone_read=False):
    """:func:`make_rhs` in affine form, for the exponential-midpoint step.

    Returns ``split(lna, T_pl, occ) -> (A, B, dT_pl, H)`` with df/dln a = A − B·f (A, B per d(ln a),
    gated, (3, 2, N)); A − B·f equals make_rhs's ``f`` to round-off.  Same arguments as make_rhs, plus
    the off-grid tail closures of the lepton kernels (``fd_cap_logC``, ``fd_decay``,
    ``monotone_read``; see ``interp``), which the QKE solver uses.
    """
    phie_at = _phie_fn(phie, eta0, q_baryon, me)
    phie_static_zero = isinstance(phie, (int, float)) and phie == 0.0

    def split(lna, T_pl, occ):
        T_cm = T_cm_init * jnp.exp(-lna)
        phie_use = phie_at(T_pl, T_cm)
        a_s, b_s, drhonudt = A.assemble_split_rhs(grid, occ, T_cm, T_pl, phie_use, prec_trans,
                                                  nunu_kernel, nunubar_kernel, nuratiotol,
                                                  m_e_eff=me, use_sched=use_sched,
                                                  fd_cap_logC=fd_cap_logC,
                                                  fd_decay=fd_decay, monotone_read=monotone_read,
                                                  lag_tail_fix=lag_tail_fix)
        if tcm_cutoff is not None:
            active = T_cm >= tcm_cutoff
            a_s = jnp.where(active, a_s, 0.0)
            b_s = jnp.where(active, b_s, 0.0)
            drhonudt = jnp.where(active, drhonudt, 0.0)
        H, dT_pl = _background(grid, occ, T_pl, T_cm, phie_use, drhonudt, use_renorm, renorm_pdep,
                               phie_static_zero, me, qed_nlo, delta_neff_init)
        return a_s / H, b_s / H, dT_pl, H

    return split


def make_decoupling_solver(grid=None, *, n_save=64, max_steps=50_000,
                           pcoeff=0.4, icoeff=0.3, dcoeff=0.0, solver=None, implicit=False,
                           nuratiotol=A.NURATIOTOL, use_renorm=True, use_sched=True,
                           renorm_pdep=True, qed_nlo=True, tcm_cutoff=0.015, phie=0.0, eta0=None,
                           q_baryon=0.5, xi=(0.0, 0.0, 0.0), sumrules=False, throw=True,
                           lag_tail_fix=True):
    """The reference transport solve: one adaptive diffrax solve over the whole window.

    The kernels, the noise floor and the equilibrium initial condition are built here, once; the
    returned ``solve`` is jitted and closes over them.

    Parameters (build-time)
    -----------------------
    grid : EpsGrid, optional
        The ε-grid (default ``make_eps_grid()``).
    n_save : int
        Saved states, uniform in ln a.
    max_steps, pcoeff, icoeff, dcoeff
        diffrax step budget and PID gains.  PI(0.4, 0.3) damps the oscillation a pure-I controller
        shows at the explicit stability limit; it changes the step adaptation, not the tolerance.
    solver, implicit
        A diffrax solver, or None for Tsit5 (``implicit=False``) / Kvaerno5 (``implicit=True``).
    use_sched : bool or str
        Lepton-kernel implementation (default True, the scheduled kernels; False is the
        bit-faithful reference).  The noise floor uses the same kernels.
    xi : (3,) array-like
        ν degeneracies (ξ_e, ξ_μ, ξ_τ): the degenerate FD initial spectrum (ν: +ξ, ν̄: −ξ) and the
        matching noise floor.  ``solve(xi=...)`` overrides it per call.
    sumrules : bool
        Also return BURST's sumrule1–5 conservation monitors at the saved states.
    throw : bool
        Raise on a failed diffrax solve.
    nuratiotol, use_renorm, renorm_pdep, qed_nlo, tcm_cutoff, phie, eta0, q_baryon, lag_tail_fix
        As in :func:`make_rhs`.

    Returns
    -------
    (solve, grid)
        ``solve(T_init=10.0, lna_max=7.0, rtol=1e-6, atol=1e-30, dtmax=None, delta_neff_init=0.0,
        me=None, xi=None)`` → dict with ``lna``, ``T_pl``, ``T_cm``, ``ratio`` and ``f_traj``
        (n_save, 3, 2, N) at the saved states, ``f_final``, ``ratio_final``, ``neff_final``, the
        diffrax solution ``sol``, and ``sumrules`` if requested.

        Per-call arguments: ``T_init`` the start T_pl = T_cm [MeV] (neutrinos tightly coupled);
        ``lna_max`` the end of the window (≈7 completes e± annihilation); ``rtol``/``atol`` the
        tolerances — atol = 1e-30 is BURST's pure-relative norm, and raising it leaves the stiff ν̄
        tail bins uncontrolled; ``dtmax`` the step cap (None: 0.08 with the scheduled kernels, 0.15
        otherwise); ``delta_neff_init`` the inert dark radiation; ``me`` the electron mass (None is
        M_ELECTRON as a compile-time constant); ``xi`` a per-call degeneracy (None keeps the baked
        floor and initial spectrum).
    """
    if grid is None:
        grid = make_eps_grid()
    dtmax_default = 0.08 if use_sched else 0.15
    if solver is None:
        solver = diffrax.Kvaerno5() if implicit else diffrax.Tsit5()

    nunu_kernel = build_nunu_kernel(grid)
    nunubar_kernel = build_nunubar_kernel(grid)
    prec_trans = A.compute_prec_trans(grid, nunu_kernel, nunubar_kernel, xi=xi,
                                      use_sched=use_sched, lag_tail_fix=lag_tail_fix)
    occ0 = A.equilibrium_occ(grid, xi=xi)

    @jax.jit
    def solve(T_init=10.0, lna_max=7.0, rtol=1e-6, atol=1e-30, dtmax=None, delta_neff_init=0.0,
              me=None, xi=None):
        compiling()
        if me is None:
            me = M_ELECTRON
        if dtmax is None:
            dtmax = dtmax_default
        T_init = jnp.asarray(T_init, dtype=jnp.float64)
        if xi is None:
            prec_use, occ_use = prec_trans, occ0
        else:
            _xi = jnp.asarray(xi, dtype=jnp.float64)
            prec_use = A.compute_prec_trans(grid, nunu_kernel, nunubar_kernel, xi=_xi,
                                            use_sched=use_sched, lag_tail_fix=lag_tail_fix)
            occ_use = A.equilibrium_occ(grid, xi=_xi)

        rhs = make_rhs(grid, prec_use, nunu_kernel, nunubar_kernel, T_init, nuratiotol, use_renorm,
                       use_sched=use_sched, renorm_pdep=renorm_pdep, qed_nlo=qed_nlo,
                       tcm_cutoff=tcm_cutoff, phie=phie, eta0=eta0, q_baryon=q_baryon,
                       delta_neff_init=delta_neff_init, me=me, lag_tail_fix=lag_tail_fix)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs), solver, t0=0.0, t1=lna_max, dt0=1e-4,
            y0={"T_pl": T_init, "f": occ_use},
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, lna_max, n_save)),
            stepsize_controller=diffrax.PIDController(atol=atol, rtol=rtol, dtmax=dtmax,
                                                      pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff,
                                                      norm=nanrobust_rms_norm),
            max_steps=max_steps, throw=throw)

        lna = sol.ts
        T_pl = sol.ys["T_pl"]
        f = sol.ys["f"]
        T_cm = T_init * jnp.exp(-lna)
        out = {
            "lna": lna,
            "T_pl": T_pl,
            "T_cm": T_cm,
            "ratio": T_pl / T_cm,
            "f_final": f[-1],
            "f_traj": f,
            "neff_final": neff_from_spectrum(grid, f[-1], T_cm[-1], T_pl[-1]),
            "ratio_final": T_pl[-1] / T_cm[-1],
            "sol": sol,
        }
        if sumrules:
            phie_at = _phie_fn(phie, eta0, q_baryon, me)

            def _sr_one(tc, tp, occ):
                return A.compute_sumrules(grid, occ, tc, tp, phie_at(tp, tc), prec_use,
                                          nunu_kernel, nunubar_kernel, nuratiotol, m_e_eff=me,
                                          use_sched=use_sched, lag_tail_fix=lag_tail_fix)

            out["sumrules"] = jax.vmap(_sr_one)(T_cm, T_pl, f)
        return out

    return solve, grid
