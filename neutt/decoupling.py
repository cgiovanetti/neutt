"""The coupled decoupling system: plasma temperature T_pl and neutrino spectrum f[3, 2, N] in ln a.

State ``{'T_pl': scalar [MeV], 'f': (3, 2, N)}``; T_cm = T_init·e^{−ln a} (comoving momenta are
frozen).  The temperature equation and the collision RHS are BURST's:

    dT_pl/dln a = [−3(ρ_pl + P_pl) − ρ̇_ν/H] / (dρ_pl/dT),     df/dln a = C[f]/H,

with H = H(ρ_pl + ρ_ν + ρ_DR).  ``make_rhs`` gives the vector field for an adaptive solve,
``make_split_rates`` the same physics in affine form A − B·f for the exponential-midpoint step, and
``make_decoupling_solver`` the reference adaptive solve.

The dark sectors (``neutt.dark``) extend all three: a boson X ↔ νν̄ decay sector (``dark``, state
key ``'f_x'``) and a Dirac χ ↔ ν contact sector (``chi_sector``, state key ``'f_chi'``), whose
energy densities enter H.  ``make_chi_split_rates`` is the χ sector's affine form,
``two_leg_decay_solve`` / ``decay_to_sm_solve`` the multi-leg solves for a decay tail.  With both
sectors off (the default) every function here is the Standard-Model one, operation for operation.
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx

from . import compiling, thermo
from . import dark as dark_mod
from .constants import M_ELECTRON, PI
from .grid import make_eps_grid
from .transport import assemble as A
from .transport import chi_brute, chi_nu, dv_nu
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


def make_decay_tail_norm(grid, eps_split=13.0, feature_weight=0.05):
    """Step-controller error norm that down-weights the high-ε ν-spectrum bins (decay tail).

    The decay injects a ~monochromatic feature at comoving ε=(m_x/2)/T_cm that climbs into ever-
    higher, near-empty ε bins as T cools; a plain relative norm chases those bins' huge relative
    change → dt→0 (atol=1e-30) or executable-dependent step counts (atol floor). Keep tight relative
    control on the populated ε<eps_split bins but weight ε≥eps_split by ``feature_weight``≪1 in the
    error norm, so the controller no longer over-resolves the near-empty feature bins. The spectrum
    still evolves exactly via the RHS; only the step size ignores the feature. Decay path only (it
    reintroduces the SM ν̄-tail roulette in the plain SM solve).

    Parameters
    ----------
    grid : EpsGrid
        ε-grid of the ν spectrum 'f'.
    eps_split : float
        Bins with ε ≥ eps_split (the decay-feature / near-empty region) are down-weighted. Default 13.
    feature_weight : float
        Weight (≪1) on the ε≥eps_split error; 0 = ignore entirely. Default 0.05.

    Returns
    -------
    callable
        norm(error_pytree) -> scalar, NaN→+inf guarded, with the 'f' leaf's high-ε error down-
        weighted. Pass as the phase-2 PIDController ``norm``.
    """
    eps = grid.abscissas
    wf = jnp.where(eps < eps_split, 1.0, feature_weight)        # (N,), weights the ε axis of 'f'

    def norm(x):
        # Re-weight the 'f' leaf's ε bins by wf, then defer to optx.rms_norm + NaN→inf guard (=
        # nanrobust_rms_norm with the high-ε feature bins down-weighted). optx.rms_norm is used
        # rather than a hand-rolled per-leaf RMS, which made the outer-jit compile hang.
        if isinstance(x, dict) and "f" in x:
            x = {**x, "f": x["f"] * wf}                         # broadcast (N,) over (...,N)
        r = optx.rms_norm(x)
        return jnp.where(jnp.isnan(r), jnp.asarray(jnp.inf, r.dtype), r)

    return norm


def rho_nu_from_spectrum(grid, occ, T_cm):
    """ρ_ν [MeV⁴] = (T_cm⁴/2π²) Σ_{flavour, cp} Σ_i w_i ε_i³ f (negative tail excursions clipped)."""
    eps = grid.abscissas
    w = grid.weights
    occ = jnp.maximum(occ, 0.0)
    return T_cm**4 / (2.0 * PI**2) * jnp.sum(w * eps**3 * jnp.sum(occ, axis=(0, 1)))


def high_eps_energy_frac(grid, occ, eps_split, n_top=10):
    """Fraction of ρ_ν carried by the down-weighted high-ε region and the grid-edge top bins.

    A safety monitor for the decay-tail step-control compromise: the phase-2 error norm controls
    error in occupation space, but ρ_ν (what the expansion rate and N_eff consume) is ε³-weighted, so
    a high-ε under-resolution negligible in occupation-RMS can be ε³-amplified into an N_eff bias.
    Two post-solve diagnostics make that risk observable:

      * ``tail`` = ρ_ν fraction in bins with ε ≥ eps_split (the down-weighted region). ≈0.5 % for a
        smooth tail; an O(0.1–1) value flags down-weighting hiding real energy.
      * ``top`` = ρ_ν fraction in the last ``n_top`` bins (grid-edge pile-up indicator). A rising
        value flags the feature edge reaching the ε³-amplified top bins.

    Computed from the final spectrum only, so it has no effect on the integration.

    Parameters
    ----------
    grid : EpsGrid
        ε-grid of the ν spectrum.
    occ : jnp.ndarray
        Final occupation f[3,2,N].
    eps_split : float
        Threshold for the down-weighted region (match make_decay_tail_norm).
    n_top : int
        Number of grid-edge bins for the pile-up fraction (default 10).

    Returns
    -------
    (tail_frac, top_frac) : tuple of float
        ρ_ν fractions, in [0, 1].
    """
    eps = grid.abscissas
    w = grid.weights
    occ = jnp.maximum(occ, 0.0)
    e3 = w * eps**3 * jnp.sum(occ, axis=(0, 1))        # (N,) per-bin ρ_ν integrand (×T_cm⁴/2π²)
    total = jnp.sum(e3)
    safe = jnp.where(total > 0.0, total, 1.0)
    tail = jnp.sum(jnp.where(eps >= eps_split, e3, 0.0)) / safe
    top = jnp.sum(e3[-n_top:]) / safe
    return tail, top


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
                me, qed_nlo, delta_neff_init, rho_extra=()):
    """H and dT_pl/dln a from the plasma EoS, ρ_ν, the inert dark radiation and ρ̇_ν.

    ``rho_extra``: further energy densities [MeV⁴] added to H in order (the dark sectors' ρ_X, ρ_χ).
    """
    rho_pl, P_pl, C = _plasma_eos(T_pl, phie, use_renorm, renorm_pdep, phie_static_zero,
                                  me=me, qed_nlo=qed_nlo)
    rho_nu = rho_nu_from_spectrum(grid, occ, T_cm)
    rho_dr = delta_neff_init * _DR_DNEFF_FACTOR * (7.0 / 8.0) * (PI**2 / 15.0) * T_cm**4
    rho_tot = rho_pl + rho_nu + rho_dr
    for rho_d in rho_extra:
        rho_tot = rho_tot + rho_d
    H = thermo.hubble(rho_tot)
    dT_pl = (-3.0 * (rho_pl + P_pl) - drhonudt / H) / C
    return H, dT_pl


def make_rhs(grid, prec_trans, nunu_kernel, nunubar_kernel, T_cm_init,
             nuratiotol=A.NURATIOTOL, use_renorm=True, use_sched=False, renorm_pdep=True,
             qed_nlo=True, tcm_cutoff=0.015, phie=0.0, eta0=None, q_baryon=0.5,
             delta_neff_init=0.0, me=M_ELECTRON, lag_tail_fix=True,
             dark=None, dark_offgrid_mode="default",
             chi_sector=None, chi_prec=None, chi_nodes=None, chi_brute_kernel=True,
             chi_gate_width=0.5, chi_sig_floor=1.0e-8, chi_sig_decades=2.0,
             chi_smooth_floor=0.0, chi_mask_width=0.0, chi_exact_e3=False,
             chi_gate_shared=False, chi_conserve=False, chi_debug=False):
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
    dark : neutt.dark.BosonDecaySector or None
        Optional dark boson X ↔ νν̄ decay sector (S1 vector Z′ / S3 majoron).  The state gains
        ``'f_x'`` (Nx,); the decay legs are added ungated to the ν RHS (the smooth 1↔2 kernels have
        no quadrature-cancellation noise to gate, and folding them into the SM gate flips ~600 SM
        gates), and ρ_X enters H.  None (default) is the SM path.
    dark_offgrid_mode : str
        Off-grid read of the decay kernels (``interp.interp_log_occ_dynamic``'s ``off_grid_mode``):
        "default", "linear" or "floor".
    chi_sector : neutt.dark.ChiSector or None
        Optional Dirac χ ↔ ν contact sector (S2).  The state gains ``'f_chi'`` (Nx,) and ρ_χ enters
        H.  The χ legs carry the SM-like near-equilibrium cancellation noise, so they pass two gates
        of their own: a noise gate (|net|/frs/prec ≥ nuratiotol, floor ``chi_prec``, a smootherstep
        of width ``chi_gate_width`` e-folds) and, after H, a physical-significance gate on each leg's
        |df/dln a| (smootherstep over ``chi_sig_decades`` decades up from ``chi_sig_floor``; 0
        disables) that keeps a weakly coupled χ from stalling the adaptive stepper.
    chi_prec : (prec_nu, prec_chi) or None
        The χ legs' equilibrium noise floor (``chi_brute.s2_brute_prec`` / ``chi_nu.s2_prec``).
    chi_nodes, chi_brute_kernel
        Quadrature nodes, and the brute-force kernel (``chi_brute``, default; small graph) in place
        of the cut kernels (``chi_nu``, the high-accuracy reference, too large a graph to compile
        inside diffeqsolve).
    chi_smooth_floor, chi_mask_width, chi_exact_e3
        Opt-in smoothings of the brute kernel (C¹ clamp, smooth support mask, exact-support E3
        rescale); default off.
    chi_gate_shared : bool
        One scalar significance weight on both legs (an energy-weighted |Σnet_ν|/Σfrs_ν), which keeps
        the ν↔χ energy identity the per-bin gates break.
    chi_conserve : bool
        Rescale the gated ν injection to the χ energy loss, blended in where the χ is active.
    chi_debug : bool
        Also return the gated χ-sourced pieces ``'_chi_dnu'``/``'_chi_dfchi'`` (for
        :func:`chi_xspecies_sumrule`); never pass that RHS to diffeqsolve.
    """
    phie_at = _phie_fn(phie, eta0, q_baryon, me)
    phie_static_zero = isinstance(phie, (int, float)) and phie == 0.0

    def rhs(lna, state, args):
        T_pl = state["T_pl"]
        occ = state["f"]
        T_cm = T_cm_init * jnp.exp(-lna)
        phie_use = phie_at(T_pl, T_cm)

        rho_extra = ()
        dnu_dark = None
        dfxdt = None
        if dark is not None:
            f_x = state["f_x"]
            meps_x = dark.m_x / T_cm
            occ_c = jnp.clip(occ, A.OCC_FLOOR, 1.0)       # same clamp the SM kernels see
            net_nu_x, _ = dv_nu.nu1_x_decay_rates(
                grid, dark.grid, occ_c, f_x, meps_x, dark.tratio, T_cm, dark.gsq_diag,
                off_grid_mode=dark_offgrid_mode)
            net_x, _ = dv_nu.x1_nu_decay_rates(
                grid, dark.grid, occ_c, f_x, meps_x, dark.tratio, T_cm, dark.gsq_diag,
                spin_dof=dark.spin_dof, off_grid_mode=dark_offgrid_mode)
            dnu_dark = net_nu_x                                           # (3,2,N) s⁻¹, ungated
            dfxdt = jnp.sum(net_x, axis=0)                                # (Nx,) s⁻¹
            _, rho_x, _ = dark_mod.boson_moments(dark.grid, f_x, meps_x * dark.tratio,
                                                 T_cm / dark.tratio, dark.spin_dof)
            rho_extra += (rho_x,)

        if chi_sector is not None:
            # S2 χ ↔ ν 2↔2 contact kernels, in the SM T_cm⁵ convention.
            f_chi = state["f_chi"]
            meps_chi = chi_sector.m_chi / T_cm
            occ_c2 = jnp.clip(occ, A.OCC_FLOOR, 1.0)
            cg, tr, Kf = chi_sector.grid, chi_sector.tratio, chi_sector.Kflav
            _s2 = chi_brute.s2_brute_rates if chi_brute_kernel else chi_nu.s2_rates
            _s2kw = dict(chi_nodes or {})
            if chi_brute_kernel:
                if chi_smooth_floor > 0.0:
                    _s2kw["smooth_floor"] = chi_smooth_floor
                if chi_mask_width > 0.0:
                    _s2kw["mask_w"] = chi_mask_width
                if chi_exact_e3:
                    _s2kw["exact_e3"] = True
            net_nu, frs_nu, net_chi, frs_chi = _s2(grid, occ_c2, f_chi, cg, meps_chi, tr, Kf, **_s2kw)
            if chi_prec is not None and not chi_gate_shared:
                # noise gate (K-independent: the coupling cancels in net/frs)
                prec_nu, prec_chi = chi_prec

                def _smooth_gate(net, frs, prec):
                    pos = frs > 0.0
                    tr = jnp.where(pos, jnp.abs(net) / jnp.where(pos, frs, 1.0) / prec, 0.0)
                    good = tr > 0.0                                     # False for NaN too
                    if chi_gate_width > 0.0:
                        x = jnp.where(good, jnp.log(jnp.where(good, tr, 1.0) / nuratiotol)
                                      / chi_gate_width, -1.0)
                        return _smootherstep(jnp.clip(x, 0.0, 1.0)) * net
                    return jnp.where(tr >= nuratiotol, net, 0.0)

                net_nu = _smooth_gate(net_nu, frs_nu, prec_nu)
                net_chi = _smooth_gate(net_chi, frs_chi, prec_chi)
            dnu_chi = T_cm ** 5 * net_nu                                  # (3,2,N) s⁻¹ (gated)
            dfchidt = T_cm ** 5 * net_chi                                 # (Nx,)   s⁻¹ (gated)
            _, rho_chi, _ = dark_mod.boson_moments(cg, f_chi, meps_chi * tr,
                                                   T_cm / tr, chi_sector.spin_dof)
            rho_extra += (rho_chi,)

        dfdt, drhonudt = A.assemble_rhs(grid, occ, T_cm, T_pl, phie_use, prec_trans,
                                        nunu_kernel, nunubar_kernel, nuratiotol,
                                        m_e_eff=me, use_sched=use_sched, lag_tail_fix=lag_tail_fix)
        if dnu_dark is not None:
            dfdt = dfdt + dnu_dark
        if tcm_cutoff is not None:
            active = T_cm >= tcm_cutoff
            dfdt = jnp.where(active, dfdt, 0.0)
            drhonudt = jnp.where(active, drhonudt, 0.0)
            if dfxdt is not None:
                dfxdt = jnp.where(active, dfxdt, 0.0)
        H, dT_pl = _background(grid, occ, T_pl, T_cm, phie_use, drhonudt, use_renorm, renorm_pdep,
                               phie_static_zero, me, qed_nlo, delta_neff_init, rho_extra)
        out = {"T_pl": dT_pl, "f": dfdt / H}
        if dark is not None:
            out["f_x"] = dfxdt / H
        if chi_sector is not None:
            # physical-significance gate on each leg's own d(state)/dln a (K-dependent)
            dnuchi_dlna = dnu_chi / H                              # (3,2,N)
            dfchi_dlna = dfchidt / H                               # (Nx,)
            if chi_gate_shared:
                # one scalar weight on both legs (keeps W·[Σν + Σχ] = 0): the energy-weighted
                # relative imbalance r = |Σ w ε³ net_ν| / Σ w ε³ frs_ν
                we3 = grid.weights * grid.abscissas ** 3
                nE = jnp.abs(jnp.sum(we3 * jnp.sum(net_nu, axis=(0, 1))))
                fE = jnp.sum(we3 * jnp.sum(frs_nu, axis=(0, 1)))
                r = nE / jnp.where(fE > 0.0, fE, 1.0)
                if chi_sig_floor > 0.0:
                    xs = jnp.clip(jnp.log10(jnp.maximum(r, 1e-300) / chi_sig_floor)
                                  / chi_sig_decades, 0.0, 1.0)
                    W = _smootherstep(xs)
                else:
                    W = 1.0
                g_nu = W
                g_chi = W
            elif chi_sig_floor > 0.0:
                g_nu = _chi_sig_gate_weight(dnuchi_dlna, chi_sig_floor, chi_sig_decades)
                g_chi = _chi_sig_gate_weight(dfchi_dlna, chi_sig_floor, chi_sig_decades)
            else:
                g_nu = 1.0
                g_chi = 1.0
            if tcm_cutoff is not None:
                g_nu = jnp.where(T_cm >= tcm_cutoff, g_nu, 0.0)
                g_chi = jnp.where(T_cm >= tcm_cutoff, g_chi, 0.0)
            gated_dnu = g_nu * dnuchi_dlna                          # (3,2,N)
            gated_dfchi = g_chi * dfchi_dlna                        # (Nx,)
            if chi_conserve:
                # scalar rescale of the ν energy injection to the χ energy loss, blended in only
                # where the χ is active (r ≫ the ~1e-6 equilibrium noise)
                we3 = grid.weights * grid.abscissas ** 3
                rE = jnp.abs(jnp.sum(we3 * jnp.sum(net_nu, axis=(0, 1))))
                fE = jnp.sum(we3 * jnp.sum(frs_nu, axis=(0, 1)))
                r = rE / jnp.where(fE > 0.0, fE, 1.0)
                act = _smootherstep(jnp.clip(jnp.log10(jnp.maximum(r, 1e-300) / 1.0e-4), 0.0, 1.0))
                E_nu = jnp.sum(we3 * jnp.sum(gated_dnu, axis=(0, 1)))
                q1c = cg.abscissas / tr
                Echi = jnp.sqrt(q1c ** 2 + meps_chi ** 2)
                E_chi = (chi_sector.spin_dof / tr ** 3) * jnp.sum(
                    cg.weights * cg.abscissas ** 2 * Echi * gated_dfchi)
                s = jnp.where(jnp.abs(E_nu) > 1e-300, -E_chi / E_nu, 1.0)
                s = jnp.clip(s, 0.3, 3.0)
                gated_dnu = (1.0 + act * (s - 1.0)) * gated_dnu
            out["f"] = out["f"] + gated_dnu
            out["f_chi"] = gated_dfchi
            if chi_debug:
                out["_chi_dnu"] = gated_dnu
                out["_chi_dfchi"] = gated_dfchi
        return out

    return rhs


def make_split_rates(grid, prec_trans, nunu_kernel, nunubar_kernel, T_cm_init,
                     nuratiotol=A.NURATIOTOL, use_renorm=True, use_sched=False,
                     renorm_pdep=True, qed_nlo=True, tcm_cutoff=0.015, phie=0.0,
                     eta0=None, q_baryon=0.5, delta_neff_init=0.0,
                     me=M_ELECTRON, fd_cap_logC=float("inf"), lag_tail_fix=True,
                     fd_decay=False, monotone_read=False, dark=None, separate_dark=False,
                     dark_range_limit=False, dark_offgrid_mode="default", dark_forward_only_tcm=0.0):
    """:func:`make_rhs` in affine form, for the exponential-midpoint step.

    Returns ``split(lna, T_pl, occ) -> (A, B, dT_pl, H)`` with df/dln a = A − B·f (A, B per d(ln a),
    gated, (3, 2, N)); A − B·f equals make_rhs's ``f`` to round-off.  Same arguments as make_rhs, plus
    the off-grid tail closures of the lepton kernels (``fd_cap_logC``, ``fd_decay``,
    ``monotone_read``; see ``interp``), which the QKE solver uses.

    With a boson-decay sector ``dark`` the decay is affine too, so the stiff X decay (−B_X·f_x) is
    integrated exactly: the ν injection A_ν − B_ν·f (``dv_nu.nu1_x_decay_affine``) is added to the SM
    (A, B) and the X equation is A_X − B_X·f_x (``dv_nu.x1_nu_decay_affine``, summed over flavours);
    dT_pl takes only the SM ρ̇_ν (the X ↔ ν exchange is internal) and ρ_X enters H.  It returns
    ``split(lna, T_pl, occ, f_x) -> (A, B, A_X, B_X, dT_pl, H, rho_x)``, or with ``separate_dark``
    the SM and decay pieces apart, ``(A_sm, B_sm, A_ν, B_ν, A_X, B_X, dT_pl, H, rho_x)``.
    ``dark_range_limit`` / ``dark_offgrid_mode`` set the decay kernels' off-grid ν read;
    ``dark_forward_only_tcm`` > 0 drops the inverse decay νν̄ → X below that T_cm [MeV].
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

    if dark is None:
        return split

    def split_dark(lna, T_pl, occ, f_x):
        T_cm = T_cm_init * jnp.exp(-lna)
        phie_use = phie_at(T_pl, T_cm)
        meps_x = dark.m_x / T_cm
        occ_c = jnp.clip(occ, A.OCC_FLOOR, 1.0)               # same clamp the SM/dark kernels see

        a_sm, b_sm, drhonudt = A.assemble_split_rhs(grid, occ, T_cm, T_pl, phie_use, prec_trans,
                                                    nunu_kernel, nunubar_kernel, nuratiotol,
                                                    m_e_eff=me, use_sched=use_sched,
                                                    fd_cap_logC=fd_cap_logC,
                                                    fd_decay=fd_decay, monotone_read=monotone_read,
                                                    lag_tail_fix=lag_tail_fix)
        a_nu_d, b_nu_d = dv_nu.nu1_x_decay_affine(grid, dark.grid, occ_c, f_x, meps_x,
                                                  dark.tratio, T_cm, dark.gsq_diag,
                                                  range_limit=dark_range_limit,
                                                  off_grid_mode=dark_offgrid_mode,
                                                  forward_only_tcm=dark_forward_only_tcm)
        a_x_fl, b_x_fl = dv_nu.x1_nu_decay_affine(grid, dark.grid, occ_c, meps_x, dark.tratio,
                                                  T_cm, dark.gsq_diag, spin_dof=dark.spin_dof,
                                                  range_limit=dark_range_limit,
                                                  off_grid_mode=dark_offgrid_mode,
                                                  forward_only_tcm=dark_forward_only_tcm)
        a_x = jnp.sum(a_x_fl, axis=0)                          # (Nx,)
        b_x = jnp.sum(b_x_fl, axis=0)

        if tcm_cutoff is not None:
            active = T_cm >= tcm_cutoff
            a_sm = jnp.where(active, a_sm, 0.0)
            b_sm = jnp.where(active, b_sm, 0.0)
            a_nu_d = jnp.where(active, a_nu_d, 0.0)
            b_nu_d = jnp.where(active, b_nu_d, 0.0)
            a_x = jnp.where(active, a_x, 0.0)
            b_x = jnp.where(active, b_x, 0.0)
            drhonudt = jnp.where(active, drhonudt, 0.0)

        _, rho_x, _ = dark_mod.boson_moments(dark.grid, f_x, meps_x * dark.tratio,
                                             T_cm / dark.tratio, dark.spin_dof)
        H, dT_pl = _background(grid, occ, T_pl, T_cm, phie_use, drhonudt, use_renorm, renorm_pdep,
                               phie_static_zero, me, qed_nlo, delta_neff_init, (rho_x,))

        if separate_dark:
            return (a_sm / H, b_sm / H, a_nu_d / H, b_nu_d / H, a_x / H, b_x / H, dT_pl, H, rho_x)
        return ((a_sm + a_nu_d) / H, (b_sm + b_nu_d) / H, a_x / H, b_x / H, dT_pl, H, rho_x)

    return split_dark


def _smootherstep(x):
    """C²-smootherstep 6x⁵−15x⁴+10x³ on x already clipped to [0,1] (0/1 derivative-flat ends)."""
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _chi_noise_gate_weight(net, frs, prec, nuratiotol, chi_gate_width):
    """Per-bin noise-gate WEIGHT w∈[0,1] (the multiplier, not w·net) — the weight :func:`make_rhs`'s
    ``_smooth_gate`` applies to the χ legs (test_ratio = |net|/frs/prec ≥ tol, C²-smootherstep over
    ``chi_gate_width`` e-folds, hard step if width=0). Returned separately so the affine split can
    multiply BOTH forf and revf by it (≡ A and B), reproducing ``w·net`` while leaving A/B untouched."""
    pos = frs > 0.0
    tr = jnp.where(pos, jnp.abs(net) / jnp.where(pos, frs, 1.0) / prec, 0.0)
    good = tr > 0.0                                            # False for NaN too
    if chi_gate_width > 0.0:
        x = jnp.where(good, jnp.log(jnp.where(good, tr, 1.0) / nuratiotol) / chi_gate_width, -1.0)
        return _smootherstep(jnp.clip(x, 0.0, 1.0))
    return jnp.where(tr >= nuratiotol, 1.0, 0.0)


def _chi_sig_gate_weight(contrib_dlna, chi_sig_floor, chi_sig_decades):
    """Per-bin physical-significance WEIGHT g∈[0,1] = the exact weight :func:`make_rhs`'s
    ``_sig_gate`` applies, evaluated on the leg's own d(state)/dlna contribution ``contrib_dlna``
    (= |T_cm⁵·net_gated/H|). smootherstep over ``chi_sig_decades`` decades up from ``chi_sig_floor``;
    K-DEPENDENT (the weak-regime stall fix). Multiplying A and B by it keeps A−B·f = g·net."""
    a = jnp.abs(contrib_dlna)
    pos = a > 0.0
    xs = jnp.where(pos, jnp.log10(jnp.where(pos, a, 1.0) / chi_sig_floor) / chi_sig_decades, -1.0)
    return _smootherstep(jnp.clip(xs, 0.0, 1.0))


def _chi_affine_side(net, frs, f1, tcm5, gw_weight):
    """Affine (A_rate, B_rate) [s⁻¹] for one χ kernel side from its gated net/frs and the tracked
    occupation ``f1``. Pure algebra (assemble_split_rhs's): forf=(frs+net)/2∝(1−f1),
    revf=(frs−net)/2∝f1, A = T⁵·forf/(1−f1), B = A + T⁵·revf/f1. The noise-gate weight multiplies
    forf and revf (≡ A and B). g, l clamped ≥0 so a round-off-negative rate can't push the
    exp_step's f below 0 into the OCC_FLOOR cliff (mirrors assemble_split_rhs)."""
    forf = gw_weight * 0.5 * (frs + net)
    revf = gw_weight * 0.5 * (frs - net)
    f1c = jnp.clip(f1, A.OCC_FLOOR, 1.0)                       # the occupation the kernel saw
    g = jnp.maximum(tcm5 * forf / jnp.maximum(1.0 - f1c, A.SPLIT_ONEMF_FLOOR), 0.0)
    l = jnp.maximum(tcm5 * revf / jnp.maximum(f1c, A.SPLIT_OCC_TINY), 0.0)
    return g, g + l, f1c


def chi_xspecies_sumrule(grid, chi_sector, dnu_chi, dfchi, meps):
    """Cross-species ν↔χ NUMBER and ENERGY balance of the χ-sourced RHS contribution.

    Unlike the raw-kernel pytest (which uses ungated rates), this takes the ACTUAL (gated/pinned)
    per-d(ln a) RHS pieces the solve applies, so it CATCHES a ν↔χ energy/number-conservation break
    introduced downstream of the kernel (e.g. the per-bin significance gate's independent ν/χ weights,
    or a faulty energy delivery in a strong-regime solver).

    ν moments Σ w ε^{2,3}·Σ_{fl,cp} dnu_chi; χ moments (g_χ/tratio³) Σ w ε² (·E_χ) dfchi with
    E_χ=√((ε/tratio)²+meps²). For the closed ν↔χ exchange these sums vanish: annihilation transfers
    number 1:1 (Δn_ν=−Δn_χ) and both legs conserve total energy, so ``num_rel`` and ``rho_rel`` ≈ 0
    (down to the kernel's detailed-balance / quadrature floor).

    Parameters:
    -----------
    grid : EpsGrid
    chi_sector : ChiSector
    dnu_chi : (3,2,N) — the χ-sourced ν injection d f_ν/d(ln a) (make_rhs ``out['f']`` minus the SM part).
    dfchi : (Nx,) — d f_χ/d(ln a) (make_rhs ``out['f_chi']``).
    meps : scalar — m_χ/T_cm at the evaluation point (sets E_χ).

    Returns:
    --------
    dict with N_nu, N_chi, num_rel, E_nu, E_chi, rho_rel (relative imbalances ≈0 ⇒ conserved).
    """
    cg, tr, g = chi_sector.grid, chi_sector.tratio, chi_sector.spin_dof
    we2 = grid.weights * grid.abscissas ** 2
    we3 = grid.weights * grid.abscissas ** 3
    dnu_tot = jnp.sum(dnu_chi, axis=(0, 1))           # (N,) over flavor + ν/ν̄
    N_nu = jnp.sum(we2 * dnu_tot)
    E_nu = jnp.sum(we3 * dnu_tot)
    Echi = jnp.sqrt((cg.abscissas / tr) ** 2 + meps ** 2)
    wc2 = cg.weights * cg.abscissas ** 2
    N_chi = (g / tr ** 3) * jnp.sum(wc2 * dfchi)
    E_chi = (g / tr ** 3) * jnp.sum(wc2 * Echi * dfchi)
    num_rel = jnp.abs(N_nu + N_chi) / jnp.maximum(jnp.abs(N_nu) + jnp.abs(N_chi), 1e-300)
    rho_rel = jnp.abs(E_nu + E_chi) / jnp.maximum(jnp.abs(E_nu) + jnp.abs(E_chi), 1e-300)
    return dict(N_nu=N_nu, N_chi=N_chi, num_rel=num_rel, E_nu=E_nu, E_chi=E_chi, rho_rel=rho_rel)


def make_chi_split_rates(grid, prec_trans, nunu_kernel, nunubar_kernel, T_cm_init,
                         chi_sector, chi_prec, nuratiotol=A.NURATIOTOL, use_renorm=True,
                         use_sched=False, renorm_pdep=True, tcm_cutoff=0.015, phie=0.0,
                         delta_neff_init=0.0, me=M_ELECTRON, chi_nodes=None, chi_brute_kernel=True,
                         chi_gate_width=0.5, chi_sig_floor=1.0e-8, chi_sig_decades=2.0):
    """Affine-split RHS for the S2 χ-coupled solve (exponential-midpoint head; strong regime).

    The χ-augmented counterpart of :func:`make_split_rates`: returns the collision RHS for the ν AND
    χ spectra in exact gated affine form ``df/d(ln a) = A − B·f``, so the stiff strong-coupling loss
    rate (large B) is integrated EXACTLY by ``hybrid.exp_step``. The χ legs reuse the SM split algebra
    (forf ∝ (1−f₁), revf ∝ f₁, f₁ the tracked bin) and fold in :func:`make_rhs`'s two χ gates as
    scalar weights on A and B, so ``A − B·f`` reproduces make_rhs's χ-coupled ``out["f"]`` /
    ``out["f_chi"]`` to round-off. χ is a Dirac fermion → the (1−f) blocking and exp_step are the ν ones.

    Same physics/knobs/defaults as :func:`make_rhs`'s χ branch. ``chi_prec`` is the (prec_nu,
    prec_chi) equilibrium noise floor (required — the noise gate needs it).

    Returns:
    --------
    callable
        ``split(lna, T_pl, occ, f_chi) -> (A, B, A_chi, B_chi, A_T, B_T, H)``: A, B (3,2,N) per
        d(ln a) for the ν spectrum (SM + χ-ν legs); A_chi, B_chi (Nx,) for f_χ; A_T, B_T scalars —
        the LINEARLY-IMPLICIT plasma-temperature relaxation (dT_pl/dlna = A_T − B_T·T_pl with B_T =
        −∂dT_pl/∂T_pl ≥ 0, so the head exp_steps T_pl stably through the strong-regime stiffness); and
        H [s⁻¹]. ``A − B·occ`` == make_rhs's ``out["f"]``, ``A_chi − B_chi·f_chi`` == ``out["f_chi"]``,
        ``A_T − B_T·T_pl`` == ``out["T_pl"]`` (all to round-off).
    """
    phie_static_zero = isinstance(phie, (int, float)) and phie == 0.0
    _cn = chi_nodes or {}
    _s2 = chi_brute.s2_brute_rates if chi_brute_kernel else chi_nu.s2_rates
    prec_nu, prec_chi = chi_prec

    def split(lna, T_pl, occ, f_chi):
        T_cm = T_cm_init * jnp.exp(-lna)
        tcm5 = T_cm ** 5

        # ---- SM affine split (identical to make_split_rates; SM legs carry no χ gate) ----
        a_s, b_s, drhonudt = A.assemble_split_rhs(grid, occ, T_cm, T_pl, phie, prec_trans,
                                                  nunu_kernel, nunubar_kernel, nuratiotol,
                                                  m_e_eff=me, use_sched=use_sched)
        if tcm_cutoff is not None:
            active = T_cm >= tcm_cutoff
            a_s = jnp.where(active, a_s, 0.0)
            b_s = jnp.where(active, b_s, 0.0)
            drhonudt = jnp.where(active, drhonudt, 0.0)

        # ---- χ kernel rates (tcmev⁵ convention) ----
        meps_chi = chi_sector.m_chi / T_cm
        occ_c2 = jnp.clip(occ, A.OCC_FLOOR, 1.0)
        cg, tr, Kf = chi_sector.grid, chi_sector.tratio, chi_sector.Kflav
        net_nu, frs_nu, net_chi, frs_chi = _s2(grid, occ_c2, f_chi, cg, meps_chi, tr, Kf, **_cn)

        # noise-gate weights (the K-independent equilibrium-cancellation-noise gate)
        gw_nu = _chi_noise_gate_weight(net_nu, frs_nu, prec_nu, nuratiotol, chi_gate_width)
        gw_chi = _chi_noise_gate_weight(net_chi, frs_chi, prec_chi, nuratiotol, chi_gate_width)

        # affine pieces (per s) for the ν-side χ legs and the χ-side legs
        a_nuchi, b_nuchi, occ_c = _chi_affine_side(net_nu, frs_nu, occ, tcm5, gw_nu)
        a_chi, b_chi, fchi_c = _chi_affine_side(net_chi, frs_chi, f_chi, tcm5, gw_chi)

        # ---- background / H (ρ_χ enters H; χ does not directly heat the plasma) ----
        rho_pl, P_pl, C = _plasma_eos(T_pl, phie, use_renorm, renorm_pdep, phie_static_zero, me=me)
        rho_nu = rho_nu_from_spectrum(grid, occ, T_cm)
        rho_dr = delta_neff_init * _DR_DNEFF_FACTOR * (7.0 / 8.0) * (PI**2 / 15.0) * T_cm**4
        _, rho_chi, _ = dark_mod.boson_moments(cg, f_chi, meps_chi * tr, T_cm / tr,
                                               chi_sector.spin_dof)

        # dT_pl as a function of T_pl alone (occ/f_chi/T_cm frozen → rho_nu/rho_chi/rho_dr fixed);
        # used both for the value and for the T_pl RELAXATION RATE below.
        def _dTpl_at(Tpl):
            rp, Pp, Cc = _plasma_eos(Tpl, phie, use_renorm, renorm_pdep, phie_static_zero, me=me)
            _, _, drho = A.assemble_split_rhs(grid, occ, T_cm, Tpl, phie, prec_trans,
                                              nunu_kernel, nunubar_kernel, nuratiotol,
                                              m_e_eff=me, use_sched=use_sched)
            if tcm_cutoff is not None:
                drho = jnp.where(T_cm >= tcm_cutoff, drho, 0.0)
            Hh = thermo.hubble(rp + rho_nu + rho_dr + rho_chi)
            return (-3.0 * (rp + Pp) - drho / Hh) / Cc

        H = thermo.hubble(rho_pl + rho_nu + rho_dr + rho_chi)
        dT_pl = (-3.0 * (rho_pl + P_pl) - drhonudt / H) / C

        # LINEARLY-IMPLICIT T_pl: the plasma↔ν exchange term −drhonudt/(H·C) makes T_pl STIFF in the
        # strong regime (χ holds ν off the e± detailed-balance manifold, so ∂dT_pl/∂T_pl is huge and
        # explicit T_pl blows up). Recast as a relaxation dT_pl/dlna ≈ A_T − B_T·T_pl with B_T =
        # −∂(dT_pl)/∂T_pl (forward FD; the χ kernel has no T_pl dependence, so it's cheap, SM-EoS
        # only), so exp_step integrates T_pl unconditionally stably. SM detailed balance keeps B_T
        # harmless (drhonudt≈0) — purely the strong-χ cure.
        _dEPS = 1.0e-6
        dT_pert = _dTpl_at(T_pl * (1.0 + _dEPS))
        B_T = jnp.maximum(-(dT_pert - dT_pl) / (T_pl * _dEPS), 0.0)
        A_T = dT_pl + B_T * T_pl

        # ---- per-d(ln a), then the K-DEPENDENT significance gate (= make_rhs's, on each leg's
        # own contribution) folded in as a scalar weight on A and B (keeps A−B·f = g_sig·net) ----
        A_nuchi = a_nuchi / H
        B_nuchi = b_nuchi / H
        A_chi_d = a_chi / H
        B_chi_d = b_chi / H
        if chi_sig_floor > 0.0:
            g_sig_nu = _chi_sig_gate_weight(A_nuchi - B_nuchi * occ_c, chi_sig_floor, chi_sig_decades)
            g_sig_chi = _chi_sig_gate_weight(A_chi_d - B_chi_d * fchi_c, chi_sig_floor, chi_sig_decades)
        else:
            g_sig_nu = 1.0
            g_sig_chi = 1.0
        if tcm_cutoff is not None:
            g_sig_nu = jnp.where(T_cm >= tcm_cutoff, g_sig_nu, 0.0)
            g_sig_chi = jnp.where(T_cm >= tcm_cutoff, g_sig_chi, 0.0)

        A_tot = a_s / H + g_sig_nu * A_nuchi
        B_tot = b_s / H + g_sig_nu * B_nuchi
        A_chi_g = g_sig_chi * A_chi_d
        B_chi_g = g_sig_chi * B_chi_d
        return A_tot, B_tot, A_chi_g, B_chi_g, A_T, B_T, H

    return split


def two_leg_decay_solve(term, solver, state0, lna_handoff, lna_stop, n_save,
                        atol_hi, atol_tail, ctrl_kwargs, max_steps, throw, tail_norm=None,
                        adjoint=None):
    """Two-leg solve for the out-of-equilibrium dark boson decay tail (S1/S3 sectors).

    The decay injects a ~monochromatic feature (E≈m_x/2) into the ν spectrum; in comoving ε=p/T_cm a
    decoupled ν stays at fixed ε, so the feature accumulates at ever-higher ε=(m_x/2)/T_cm as the
    universe cools. Two pathologies one tight-atol Tsit5 can't handle: (1) the feature deposits into
    near-empty high-ε bins, where pure-relative atol chases their huge relative change → dt underflows;
    (2) once ε=(m_x/2)/T_cm exceeds the grid the injection piles up in the boundary bin. ν are
    decoupled below ~1 MeV, so the SM-critical tight-atol transport is entirely above the handoff.
    Phase 1 [0, lna_handoff] keeps atol_hi.

    Phase 2 [lna_handoff, lna_stop] error control: with ``tail_norm`` given (default from
    make_decoupling_solver) keep atol_hi but use ``tail_norm`` to down-weight the high-ε feature bins
    (fusion-robust, no atol floor → no roulette); with ``tail_norm`` None, floor atol to atol_tail
    (works but fusion-roulette-prone). The caller sets lna_stop so the feature edge stays on-grid.
    Trajectories are stitched (duplicated handoff sample dropped) → n_save points.

    ``adjoint`` (default None → diffrax's RecursiveCheckpointAdjoint): pass ``diffrax.DirectAdjoint()``
    to dodge the cold-compile stall — RecursiveCheckpointAdjoint's checkpointed loop has XLA compile
    scaling with max_steps that can hang 10–40 min; DirectAdjoint skips it (bit-identical forward
    solution; cost is reverse-mode AD memory/speed, irrelevant for forward/gradient-free use). Its
    compile cost scales with n_save (SaveAt points), so keep n_save modest. ``tail_norm`` adds its own
    per-step compile cost; the fast combo is DirectAdjoint + the legacy atol floor (tail_norm=None).

    Returns (lna, ys, result, stats1, stats2); ys is the stitched state-trajectory dict.
    """
    _adj = adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint()
    n1 = max(2, n_save // 2)
    n2 = n_save - n1 + 1                                      # +1 so the stitched total is n_save
    sol1 = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=lna_handoff, dt0=1e-4, y0=state0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, lna_handoff, n1)), adjoint=_adj,
        stepsize_controller=diffrax.PIDController(atol=atol_hi, **ctrl_kwargs),
        max_steps=max_steps, throw=throw)
    state_h = jax.tree_util.tree_map(lambda a: a[-1], sol1.ys)
    if tail_norm is None:
        ctrl2, atol2 = ctrl_kwargs, atol_tail                # legacy atol-floor path
    else:
        ctrl2 = {**ctrl_kwargs, "norm": tail_norm}           # hardened down-weighted-norm path
        atol2 = atol_hi                                       # keep the tight floor; the norm does the work
    sol2 = diffrax.diffeqsolve(
        term, solver, t0=lna_handoff, t1=lna_stop, dt0=1e-4, y0=state_h,
        saveat=diffrax.SaveAt(ts=jnp.linspace(lna_handoff, lna_stop, n2)), adjoint=_adj,
        stepsize_controller=diffrax.PIDController(atol=atol2, **ctrl2),
        max_steps=max_steps, throw=throw)
    lna = jnp.concatenate([sol1.ts, sol2.ts[1:]])
    ys = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b[1:]], axis=0),
                                sol1.ys, sol2.ys)
    return lna, ys, sol2.result, sol1.stats, sol2.stats


def decay_to_sm_solve(term_dark, term_sm, solver, state0, lna_decay_handoff, lna_sm_handoff,
                      lna_max, n_save, atol_hi, atol_tail, ctrl_kwargs, max_steps, throw,
                      tail_norm=None, adjoint=None):
    """Three-leg decay→SM hand-off solve for the full-window (post-e±-annihilation) N_eff of a boson
    that decays early, depositing a feature at high comoving ε.

    :func:`two_leg_decay_solve` keeps the redshifting feature ε=(m_x/2)/T_cm on-grid by stopping
    phase 2 where the edge reaches ~0.9·ε_max. For a heavy boson (m_x≳2 → feature ε≳1) that edge hits
    the grid edge at T_cm≈0.1 MeV — before e± annihilation finishes (T_cm≲0.02) — so the two-leg N_eff
    hasn't reached its (11/4)^{4/3} asymptote, and letting a single solve continue chases the feature
    off-grid and grinds.

    The fix exploits: (1) such a boson is depleted by the time its feature reaches the grid edge (the
    thermal majoron is e^{−m_x/T}≈0 by T≈0.1 MeV — the caller must verify rho_x(lna_sm_handoff) ≪
    rho_ν, reported as ``rho_x_final``); (2) decoupled ν sit at fixed comoving ε, so the deposited
    feature is frozen once decays stop injecting. So:

      * Phase 1 [0, lna_decay_handoff] — dark term, tight atol (SM decoupling + bulk decay).
      * Phase 2 [lna_decay_handoff, lna_sm_handoff] — dark term, ``tail_norm`` down-weighted (residual
        decay; feature edge held on-grid by lna_sm_handoff).
      * Phase 3 [lna_sm_handoff, lna_max] — SM-only term (``f_x`` dropped: no more injection),
        ``tail_norm`` down-weighted; e± annihilation finishes with the feature frozen in ε.

    The hard ``f_x`` drop at phase 3 is energy-exact only if the boson is depleted there; the leftover
    rho_x at the hand-off is carried into the output (f_x frozen at its hand-off value) so
    ``rho_x_final`` makes any residual observable.

    ``term_sm`` is :func:`make_rhs` built with ``dark=None`` (state {T_pl, f}); ``term_dark`` is the
    boson-decay RHS (state {T_pl, f, f_x}). ``adjoint`` as in :func:`two_leg_decay_solve`. This
    stitching path works where the dark legs are non-stiff (weak coupling / OOE) and short enough to
    compile; for stiff thermal/overabundant bosons use the exponential/affine head in
    :func:`make_split_rates`.

    Returns (lna, ys, result, stats1, stats2, stats3); ys is the stitched state dict including a
    frozen-in-phase-3 ``f_x`` trajectory → n_save points total.
    """
    _adj = adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint()
    n1 = max(2, n_save // 3)
    n2 = max(2, n_save // 3)
    n3 = n_save + 2 - n1 - n2                                 # +2 → two dropped handoff duplicates
    if tail_norm is None:
        ctrl_tail, atol2 = ctrl_kwargs, atol_tail             # legacy atol-floor path
    else:
        ctrl_tail, atol2 = {**ctrl_kwargs, "norm": tail_norm}, atol_hi   # down-weighted-norm path

    # Phase 1: dark, tight atol through SM decoupling + bulk decay (feature on-grid).
    sol1 = diffrax.diffeqsolve(
        term_dark, solver, t0=0.0, t1=lna_decay_handoff, dt0=1e-4, y0=state0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, lna_decay_handoff, n1)), adjoint=_adj,
        stepsize_controller=diffrax.PIDController(atol=atol_hi, **ctrl_kwargs),
        max_steps=max_steps, throw=throw)
    state_h1 = jax.tree_util.tree_map(lambda a: a[-1], sol1.ys)

    # Phase 2: dark, down-weighted residual decay (feature edge held on-grid).
    sol2 = diffrax.diffeqsolve(
        term_dark, solver, t0=lna_decay_handoff, t1=lna_sm_handoff, dt0=1e-4, y0=state_h1,
        saveat=diffrax.SaveAt(ts=jnp.linspace(lna_decay_handoff, lna_sm_handoff, n2)), adjoint=_adj,
        stepsize_controller=diffrax.PIDController(atol=atol2, **ctrl_tail),
        max_steps=max_steps, throw=throw)
    state_h2 = jax.tree_util.tree_map(lambda a: a[-1], sol2.ys)
    f_x_handoff = state_h2["f_x"]                             # frozen residual carried into phase 3

    # Phase 3: SM-only (drop f_x), finish e± annihilation with the feature frozen in ε.
    state_sm = {k: v for k, v in state_h2.items() if k != "f_x"}
    sol3 = diffrax.diffeqsolve(
        term_sm, solver, t0=lna_sm_handoff, t1=lna_max, dt0=1e-4, y0=state_sm,
        saveat=diffrax.SaveAt(ts=jnp.linspace(lna_sm_handoff, lna_max, n3)), adjoint=_adj,
        stepsize_controller=diffrax.PIDController(atol=atol2, **ctrl_tail),
        max_steps=max_steps, throw=throw)

    lna = jnp.concatenate([sol1.ts, sol2.ts[1:], sol3.ts[1:]])
    # Stitch {T_pl, f} across all three legs; rebuild the f_x trajectory (frozen in phase 3).
    ys = {}
    for k in ("T_pl", "f"):
        ys[k] = jnp.concatenate([sol1.ys[k], sol2.ys[k][1:], sol3.ys[k][1:]], axis=0)
    fx3 = jnp.broadcast_to(f_x_handoff, (n3,) + f_x_handoff.shape)
    ys["f_x"] = jnp.concatenate([sol1.ys["f_x"], sol2.ys["f_x"][1:], fx3[1:]], axis=0)
    return lna, ys, sol3.result, sol1.stats, sol2.stats, sol3.stats


def make_decoupling_solver(grid=None, *, n_save=64, max_steps=50_000,
                           pcoeff=0.4, icoeff=0.3, dcoeff=0.0, solver=None, implicit=False,
                           nuratiotol=A.NURATIOTOL, use_renorm=True, use_sched=True,
                           renorm_pdep=True, qed_nlo=True, tcm_cutoff=0.015, phie=0.0, eta0=None,
                           q_baryon=0.5, xi=(0.0, 0.0, 0.0), sumrules=False, throw=True,
                           lag_tail_fix=True,
                           dark=None, dark_offgrid_mode="default",
                           chi_sector=None, chi_nodes=None,
                           chi_brute_kernel=True, chi_gate_width=0.5,
                           chi_sig_floor=1.0e-4, chi_sig_decades=2.0, chi_smooth_floor=0.0,
                           chi_mask_width=0.0, chi_exact_e3=False, chi_gate_shared=False,
                           chi_conserve=False,
                           decay_tail=False, decay_handoff_tcm=0.20, decay_tail_atol=1.0e-10,
                           decay_stop_safety=0.9, decay_tail_hardened=True,
                           decay_tail_eps_split=13.0, decay_tail_feature_weight=0.05,
                           decay_to_sm=False, decay_sm_handoff_tcm=None,
                           decay_adjoint=None, solve_adjoint=None):
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
    dark, dark_offgrid_mode, chi_sector, chi_nodes, chi_brute_kernel, chi_gate_width, chi_sig_floor,
    chi_sig_decades, chi_smooth_floor, chi_mask_width, chi_exact_e3, chi_gate_shared, chi_conserve
        The dark sectors, as in :func:`make_rhs` (the χ noise floor is built inside ``solve`` from the
        same kernel and nodes as the RHS).  The adaptive χ path defaults ``chi_sig_floor`` to 1e-4,
        above the brute kernel's cancellation jitter.
    decay_tail, decay_handoff_tcm, decay_tail_atol, decay_stop_safety, decay_tail_hardened,
    decay_tail_eps_split, decay_tail_feature_weight
        With a boson-decay ``dark`` sector, :func:`two_leg_decay_solve`: tight control down to
        ``decay_handoff_tcm`` [MeV], then the decay tail with the high-ε bins (ε ≥
        ``decay_tail_eps_split``) down-weighted by ``decay_tail_feature_weight`` in the error norm
        (``make_decay_tail_norm``; or, with ``decay_tail_hardened=False``, an atol floor
        ``decay_tail_atol``), stopping where the feature ε = (m_X/2)/T_cm reaches
        ``decay_stop_safety``·ε_max.
    decay_to_sm, decay_sm_handoff_tcm
        :func:`decay_to_sm_solve` (takes precedence over ``decay_tail``): the two decay legs, then an
        SM-only leg from ``decay_sm_handoff_tcm`` (None: the decay-tail stop) to ``lna_max`` with
        ``f_x`` frozen.  Valid only if the boson is depleted there (``rho_x_final`` ≪ ρ_ν).
    decay_adjoint, solve_adjoint
        diffrax adjoints for the multi-leg decay solves / the single solve (None: diffrax's
        default; ``diffrax.DirectAdjoint()`` avoids a slow compile of the dark single solve).

    Returns
    -------
    (solve, grid)
        ``solve(T_init=10.0, lna_max=7.0, rtol=1e-6, atol=1e-30, dtmax=None, delta_neff_init=0.0,
        me=None, xi=None, f_x0=None, f_chi0=None)`` → dict with ``lna``, ``T_pl``, ``T_cm``,
        ``ratio`` and ``f_traj`` (n_save, 3, 2, N) at the saved states, ``f_final``,
        ``ratio_final``, ``neff_final``, the diffrax solution ``sol``, and ``sumrules`` if requested.
        With ``dark``: ``f_x``, ``f_x_final``, ``rho_x_final``, ``neff_dark_final`` and the
        ``eps_tail_energy_frac``/``eps_top_energy_frac`` monitors (:func:`high_eps_energy_frac`);
        with ``chi_sector``: ``f_chi``, ``f_chi_final``, ``rho_chi_final``, ``neff_chi_final``; on
        the multi-leg decay paths ``sol`` is None and ``decay_result``/``decay_stats``/
        ``decay_num_steps`` report the legs.

        Per-call arguments: ``T_init`` the start T_pl = T_cm [MeV] (neutrinos tightly coupled);
        ``lna_max`` the end of the window (≈7 completes e± annihilation); ``rtol``/``atol`` the
        tolerances — atol = 1e-30 is BURST's pure-relative norm, and raising it leaves the stiff ν̄
        tail bins uncontrolled; ``dtmax`` the step cap (None: 0.08 with the scheduled kernels or a
        dark sector, 0.15 otherwise); ``delta_neff_init`` the inert dark radiation; ``me`` the
        electron mass (None is M_ELECTRON as a compile-time constant); ``xi`` a per-call degeneracy
        (None keeps the baked floor and initial spectrum); ``f_x0``/``f_chi0`` the initial dark
        occupations (None: BE / FD equilibrium at T_init).
    """
    if grid is None:
        grid = make_eps_grid()
    dtmax_default = 0.08 if (use_sched or dark is not None or chi_sector is not None) else 0.15
    if solver is None:
        solver = diffrax.Kvaerno5() if implicit else diffrax.Tsit5()

    nunu_kernel = build_nunu_kernel(grid)
    nunubar_kernel = build_nunubar_kernel(grid)
    prec_trans = A.compute_prec_trans(grid, nunu_kernel, nunubar_kernel, xi=xi,
                                      use_sched=use_sched, lag_tail_fix=lag_tail_fix)
    occ0 = A.equilibrium_occ(grid, xi=xi)

    @jax.jit
    def solve(T_init=10.0, lna_max=7.0, rtol=1e-6, atol=1e-30, dtmax=None, delta_neff_init=0.0,
              me=None, xi=None, f_x0=None, f_chi0=None):
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

        state0 = {"T_pl": T_init, "f": occ_use}
        if dark is not None:
            if f_x0 is None:
                f_x0 = dark_mod.dark_be_equil(dark.grid.abscissas,
                                              dark.m_x / T_init * dark.tratio)
            state0["f_x"] = jnp.asarray(f_x0, dtype=jnp.float64)
        chi_prec = None
        if chi_sector is not None:
            if f_chi0 is None:
                f_chi0 = dark_mod.dark_fd_equil(chi_sector.grid.abscissas,
                                                chi_sector.m_chi / T_init * chi_sector.tratio)
            state0["f_chi"] = jnp.asarray(f_chi0, dtype=jnp.float64)
            # χ noise floor: ν and χ both at T_init (a pure quadrature residual), with the RHS's
            # kernel and nodes; the barrier keeps XLA from constant-folding the cancellation it
            # measures (as in compute_prec_trans).
            meps0 = chi_sector.m_chi / T_init
            f_chi_eq = dark_mod.dark_fd_equil(chi_sector.grid.abscissas, meps0)
            occ_b, kflav_b = jax.lax.optimization_barrier((occ_use, chi_sector.Kflav))
            _prec_fn = chi_brute.s2_brute_prec if chi_brute_kernel else chi_nu.s2_prec
            _prec_kw = dict(chi_nodes or {})
            if chi_brute_kernel and chi_smooth_floor > 0.0:
                _prec_kw["smooth_floor"] = chi_smooth_floor
            if chi_brute_kernel and chi_mask_width > 0.0:
                _prec_kw["mask_w"] = chi_mask_width
            if chi_brute_kernel and chi_exact_e3:
                _prec_kw["exact_e3"] = True
            chi_prec = _prec_fn(grid, occ_b, f_chi_eq, chi_sector.grid, meps0, 1.0, kflav_b,
                                **_prec_kw)

        rhs = make_rhs(grid, prec_use, nunu_kernel, nunubar_kernel, T_init, nuratiotol, use_renorm,
                       use_sched=use_sched, renorm_pdep=renorm_pdep, qed_nlo=qed_nlo,
                       tcm_cutoff=tcm_cutoff, phie=phie, eta0=eta0, q_baryon=q_baryon,
                       delta_neff_init=delta_neff_init, me=me, lag_tail_fix=lag_tail_fix,
                       dark=dark, dark_offgrid_mode=dark_offgrid_mode, chi_sector=chi_sector,
                       chi_prec=chi_prec, chi_nodes=chi_nodes, chi_brute_kernel=chi_brute_kernel,
                       chi_gate_width=chi_gate_width, chi_sig_floor=chi_sig_floor,
                       chi_sig_decades=chi_sig_decades, chi_smooth_floor=chi_smooth_floor,
                       chi_mask_width=chi_mask_width, chi_exact_e3=chi_exact_e3,
                       chi_gate_shared=chi_gate_shared, chi_conserve=chi_conserve)
        ctrl_kwargs = dict(rtol=rtol, dtmax=dtmax, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff,
                           norm=nanrobust_rms_norm)

        decay_result, decay_stats = None, None
        if (decay_to_sm or decay_tail) and dark is not None:
            eps_max = grid.abscissas[-1]
            T_stop = (dark.m_x / 2.0) / (decay_stop_safety * eps_max)   # feature edge on grid
            lna_handoff = jnp.log(T_init / decay_handoff_tcm)
            tail_norm = (make_decay_tail_norm(grid, decay_tail_eps_split, decay_tail_feature_weight)
                         if decay_tail_hardened else None)
            if decay_to_sm:
                sm_rhs = make_rhs(grid, prec_use, nunu_kernel, nunubar_kernel, T_init, nuratiotol,
                                  use_renorm, use_sched=use_sched, renorm_pdep=renorm_pdep,
                                  qed_nlo=qed_nlo, tcm_cutoff=tcm_cutoff, phie=phie, eta0=eta0,
                                  q_baryon=q_baryon, delta_neff_init=delta_neff_init, me=me,
                                  lag_tail_fix=lag_tail_fix)
                T_sm = T_stop if decay_sm_handoff_tcm is None else decay_sm_handoff_tcm
                lna_sm = jnp.minimum(jnp.log(T_init / T_sm), lna_max)
                lna, ys, decay_result, *decay_stats = decay_to_sm_solve(
                    diffrax.ODETerm(rhs), diffrax.ODETerm(sm_rhs), solver, state0, lna_handoff,
                    lna_sm, lna_max, n_save, atol, decay_tail_atol, ctrl_kwargs, max_steps, throw,
                    tail_norm=tail_norm, adjoint=decay_adjoint)
            else:
                lna_stop = jnp.minimum(jnp.log(T_init / T_stop), lna_max)
                lna, ys, decay_result, *decay_stats = two_leg_decay_solve(
                    diffrax.ODETerm(rhs), solver, state0, lna_handoff, lna_stop, n_save,
                    atol, decay_tail_atol, ctrl_kwargs, max_steps, throw, tail_norm=tail_norm,
                    adjoint=decay_adjoint)
            sol = None
        else:
            solve_kw = {} if solve_adjoint is None else {"adjoint": solve_adjoint}
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(rhs), solver, t0=0.0, t1=lna_max, dt0=1e-4,
                y0=state0,
                saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, lna_max, n_save)),
                stepsize_controller=diffrax.PIDController(atol=atol, rtol=rtol, dtmax=dtmax,
                                                          pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff,
                                                          norm=nanrobust_rms_norm),
                max_steps=max_steps, throw=throw, **solve_kw)
            lna, ys = sol.ts, sol.ys

        T_pl = ys["T_pl"]
        f = ys["f"]
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
        if decay_stats is not None:
            out["decay_result"] = decay_result
            out["decay_stats"] = tuple(decay_stats)
            out["decay_num_steps"] = [s["num_steps"] for s in decay_stats]
        rho_1nu_std = (7.0 / 8.0) * (PI**2 / 15.0) * ((4.0 / 11.0) ** (1.0 / 3.0) * T_pl[-1]) ** 4
        if dark is not None:
            f_x = ys["f_x"]
            meps_fin = dark.m_x / T_cm[-1]
            _, rho_x_fin, _ = dark_mod.boson_moments(dark.grid, f_x[-1], meps_fin * dark.tratio,
                                                     T_cm[-1] / dark.tratio, dark.spin_dof)
            out["f_x"] = f_x
            out["f_x_final"] = f_x[-1]
            out["rho_x_final"] = rho_x_fin
            out["neff_dark_final"] = rho_x_fin / rho_1nu_std   # leftover X energy in N_eff units
            tail_frac, top_frac = high_eps_energy_frac(grid, f[-1], decay_tail_eps_split)
            out["eps_tail_energy_frac"] = tail_frac
            out["eps_top_energy_frac"] = top_frac
        if chi_sector is not None:
            f_chi = ys["f_chi"]
            meps_fin = chi_sector.m_chi / T_cm[-1]
            _, rho_chi_fin, _ = dark_mod.boson_moments(chi_sector.grid, f_chi[-1],
                                                       meps_fin * chi_sector.tratio,
                                                       T_cm[-1] / chi_sector.tratio,
                                                       chi_sector.spin_dof)
            out["f_chi"] = f_chi
            out["f_chi_final"] = f_chi[-1]
            out["rho_chi_final"] = rho_chi_fin
            out["neff_chi_final"] = rho_chi_fin / rho_1nu_std
        if sumrules:
            phie_at = _phie_fn(phie, eta0, q_baryon, me)

            def _sr_one(tc, tp, occ):
                return A.compute_sumrules(grid, occ, tc, tp, phie_at(tp, tc), prec_use,
                                          nunu_kernel, nunubar_kernel, nuratiotol, m_e_eff=me,
                                          use_sched=use_sched, lag_tail_fix=lag_tail_fix)

            out["sumrules"] = jax.vmap(_sr_one)(T_cm, T_pl, f)
        return out

    return solve, grid
