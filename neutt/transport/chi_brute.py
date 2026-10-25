"""Brute-force S2 χ↔ν 2↔2 collision kernel for the COUPLED SOLVE (small-graph alternative).

The cut-based kernels in :mod:`chi_nu` are the high-accuracy reference, but their ~40 unrolled
cut-segments make the combined RHS graph too large for diffrax's adaptive while-loop (the compile
blows up to tens of minutes). This is a single masked 4-D Gauss-Legendre quadrature (p2, cosθ2, E3,
φ_a) with the kinematic support enforced by node masking, not analytic cuts. The graph is tiny (one
big reduction) so diffeqsolve compiles in the normal range; the cost moves to runtime (a denser
quadrature), which the GPU absorbs.

Same physics, conventions, and absolute normalization as :func:`chi_nu.s2_rates`. ``s2_brute_rates``
is a drop-in replacement returning ``(net_nu, frs_nu, net_chi, frs_chi)`` in the tcmev⁵ convention
(df = tcmev⁵·net).

Everything is in T_cm units (dimensionless ε = p/T_cm); the χ occupation is queried on its own grid
via ε_χ = q·tratio. Prefactor:
    net = 2/(ħ·16·(2π)⁴) · (1/E1) · Σ_grid (p2²/(E2·a))·⟨|M|²⟩·F·mask     [df = T_cm⁵·net]
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..constants import HBAR, PI
from ..interp import interp_log_occ_dynamic, _smooth_clamp_log

# 2 (φ∈[0,2π]=2×[0,π]) / (ħ·16·(2π)⁴); ⟨|M|²⟩ carries Kflav.
PREF_BRUTE = 2.0 / (HBAR * 16.0 * (2.0 * PI) ** 4)

N_P2 = 20
N_C2 = 20
N_E3 = 20
N_PHI = 12
P2MAX_FAC = 45.0          # outer p2 cutoff in T_cm units (occupations have decayed by here)


def _safe(u):
    return jnp.sqrt(jnp.maximum(u, 0.0))


def _kin(p1, m1, m2, m3, m4, p2, c2, xe, cphi, mask_w=0.0, exact_e3=False):
    """Kinematics + measure on the 4-D grid. Inputs broadcast to (Nbin,Np2,Nc2,Ne3,Nphi).
    Returns (s, t, u, measure, E1, p3, p4, betasq) — measure = mask·(p2²/(E2 a))·E3-jacobian.

    Two OPT-IN smooth-support modes cure the strong-χ kink (as T sweeps through m_χ the support
    boundary crosses fixed E3 nodes; a HARD 0/1 mask makes each crossing a jump → a kink → dt-collapse):

    * ``exact_e3`` (PREFERRED) rescales the E3 nodes to the EXACT physical window [E3⁻, E3⁺] from the
      CM boost (E3± = (E_tot·E3cm ± a·p3cm)/√s), so NO node is ever out of support → no mask, no kink,
      no spurious flux, and better quadrature. The only edge left is the 2-body threshold s=(m3+m4)²,
      where the window width we3→0 smoothly. Strictly more accurate than the hard mask.
    * ``mask_w`` > 0 keeps the [m3, E_tot−m4] window but replaces the 0/1 cut mask with C¹ sigmoid
      tapers on |cosθ|≤1 and E4≥m4. Smooth, but adds a small spurious flux from tapered
      out-of-support nodes — kept for comparison; prefer exact_e3.

    Default (both off) is bit-identical to the original hard-mask kernel.
    """
    E1 = _safe(p1 ** 2 + m1 ** 2)
    E2 = _safe(p2 ** 2 + m2 ** 2)
    a = _safe(p1 ** 2 + p2 ** 2 + 2.0 * p1 * p2 * c2)
    asafe = jnp.where(a > 0.0, a, 1.0)
    # s = (p1+p2)² invariant mass² (= E_tot² − a²); needed early for the exact-support window.
    s = m1 ** 2 + m2 ** 2 + 2.0 * (E1 * E2 - p1 * p2 * c2)
    if exact_e3:
        E_tot = E1 + E2
        thr = (m3 + m4) ** 2
        scl = jnp.maximum(s, thr)                                # clamp into the valid region
        rs = _safe(scl)
        rssafe = jnp.where(rs > 0.0, rs, 1.0)
        E3cm = (scl + m3 ** 2 - m4 ** 2) / (2.0 * rssafe)
        lam = (scl - thr) * (scl - (m3 - m4) ** 2)
        p3cm = _safe(lam) / (2.0 * rssafe)
        half = a * p3cm / rssafe                                 # half-width of the lab E3 window
        cen = E_tot * E3cm / rssafe
        e3lo = cen - half
        e3hi = cen + half
        valid = (s > thr) & (a > 0.0)
    else:
        e3lo = m3
        e3hi = E1 + E2 - m4
        valid = e3hi > e3lo
    E3 = 0.5 * (e3hi - e3lo) * (xe + 1.0) + e3lo
    we3 = 0.5 * (e3hi - e3lo)                                   # dE3 jacobian
    p3 = _safe(E3 ** 2 - m3 ** 2)
    E4 = E1 + E2 - E3
    p4sq = E4 ** 2 - m4 ** 2
    p3safe = jnp.where(p3 > 0.0, p3, 1.0)
    ca = (a ** 2 + p3 ** 2 - p4sq) / (2.0 * asafe * p3safe)
    struct = (valid & (a > 0.0) & (p3 > 0.0)).astype(p1.dtype)
    if exact_e3:
        mask = struct                                          # all nodes physical; we3→0 at thr
    elif mask_w > 0.0:
        s_ca = jax.nn.sigmoid((1.0 - jnp.abs(ca)) / mask_w)         # |cosθ| ≤ 1 (the dominant edge)
        s_e4 = jax.nn.sigmoid((E4 - m4) / (mask_w * jnp.maximum(E1, 1e-3)))  # E4 ≥ m4 ≡ p4² ≥ 0
        mask = struct * s_ca * s_e4
    else:
        mask = struct * ((E4 >= m4) & (p4sq >= 0.0) & (jnp.abs(ca) <= 1.0)).astype(p1.dtype)
    # Mandelstams (s computed above; t on ca AND φ_a)
    c1a = jnp.clip((p1 + p2 * c2) / asafe, -1.0, 1.0)
    s1a = _safe(1.0 - c1a ** 2)
    cacl = jnp.clip(ca, -1.0, 1.0)                              # no-op in support; bounds the taper
    sa = _safe(1.0 - cacl ** 2)
    c13 = c1a * cacl + s1a * sa * cphi
    t = m1 ** 2 + m3 ** 2 - 2.0 * (E1 * E3 - p1 * p3 * c13)
    uu = m1 ** 2 + m2 ** 2 + m3 ** 2 + m4 ** 2 - s - t
    measure = mask * (p2 ** 2 / (E2 * asafe)) * we3
    p4 = _safe(p4sq)
    # cphi enters ONLY through c13 = c1a·cacl + s1a·sa·cphi → t = t0 + β·cphi, uu = uu0 − β·cphi
    # (both linear in cphi), β = 2·p1·p3·s1a·sa. measure, s, E1, p3, p4 and the occupations are
    # cphi-independent. β² is the only extra quantity the analytic φ-integral needs (``analytic_phi``
    # in s2_brute_rates). Unused on the default path ⇒ DCE ⇒ bit-identical.
    betasq = (2.0 * p1 * p3 * s1a * sa) ** 2
    return s, t, uu, measure, E1, p3, p4, betasq


def _Ffac(f1, f2, f3, f4):
    """F = (1−f1)(1−f2) f3 f4 − f1 f2 (1−f3)(1−f4); returns (F_net, F_frs).

    F_net uses the detailed-balance STABLE form F_net = −forf·expm1(log(revf/forf)) with
    log(revf/forf) summed from the individual log factors — algebraically identical to ``forf − revf``
    but it KILLS the catastrophic float cancellation at Γ≫H (the deep-tail x>1.6 regime where
    revf/forf→1 makes the raw subtraction's *sign* round-off noise). ``expm1`` is smooth ⇒ jacfwd-safe
    in the coupled Newton. The direct difference is used where either leg is 0 (the expm1 form is 0/0
    there); the double-``where`` keeps every log finite so no NaN leaks through AD.
    """
    forf = (1.0 - f1) * (1.0 - f2) * f3 * f4
    revf = f1 * f2 * (1.0 - f3) * (1.0 - f4)
    # the expm1 form is only NEEDED near cancellation (revf≈forf); for a dominant leg the direct
    # subtraction is accurate and avoids expm1 overflow. Gate the stable branch to |δ|<1 and clip δ
    # so the (unused) expm1 stays finite for AD.
    safe = (forf > 0.0) & (revf > 0.0)                  # ⟺ all f_i ∈ (0,1) ⇒ every log below finite
    g1 = jnp.where(safe, f1, 0.5); g2 = jnp.where(safe, f2, 0.5)
    g3 = jnp.where(safe, f3, 0.5); g4 = jnp.where(safe, f4, 0.5)
    delta = ((jnp.log(g1) + jnp.log(g2) + jnp.log1p(-g3) + jnp.log1p(-g4))
             - (jnp.log1p(-g1) + jnp.log1p(-g2) + jnp.log(g3) + jnp.log(g4)))   # = log(revf/forf)
    use = safe & (jnp.abs(delta) < 1.0)
    net = jnp.where(use, -forf * jnp.expm1(jnp.clip(delta, -1.0, 1.0)), forf - revf)
    return net, forf + revf


# range_limit=True: cap the order-5 log-interp at its window max so an off-grid read past ε=25 cannot
# overshoot to a spurious O(1) occupation (the grid-edge ε³-amplified pile-up that detonated the
# strong-χ true-shape χ→ν delivery). Opt-in here (default False in interp.py keeps the SM kernels
# Fortran-faithful); the chi_brute kernel is only used by the dark-sector solve.
def _nu_occ_impl(logf_nu_fc, eps_nu, p, smooth_floor=0.0, fd_decay=False, monotone_read=False):
    # fd_decay (off-grid FD-decay) + monotone_read (on-grid 2-node bracket): the lepton-kernel-boundary
    # source fix for the νχ→νχ scatter leg reading ν off-node/off-grid. Both default off ⇒ bit-identical.
    return jnp.exp(interp_log_occ_dynamic(logf_nu_fc, eps_nu, p, smooth_floor=smooth_floor,
                                          range_limit=True, fd_decay=fd_decay,
                                          monotone_read=monotone_read))


def _chi_occ_impl(logf_chi, chi_eps, q, tratio, smooth_floor=0.0):
    return jnp.exp(interp_log_occ_dynamic(logf_chi, chi_eps, q * tratio, clamp_high=True,
                                          smooth_floor=smooth_floor, range_limit=True))


def s2_brute_rates(grid, occ, f_chi, chi_grid, meps, tratio, Kflav,
                   n_p2=N_P2, n_c2=N_C2, n_e3=N_E3, n_phi=N_PHI, p2max_fac=P2MAX_FAC,
                   smooth_floor=0.0, mask_w=0.0, exact_e3=False, return_legs=False,
                   return_nu_legs=False, fd_decay=False, monotone_read=False, analytic_phi=False,
                   **_ignore):
    """Brute-force (net_nu (3,2,N), frs_nu (3,2,N), net_chi (Nx,), frs_chi (Nx,)) — drop-in for
    :func:`chi_nu.s2_rates`. Folds the same multiplicities (ν off χ AND χ̄ → ×2 scatter; χ-side
    sums ν flavor and the ν/ν̄ pair). df = tcmev⁵·net.

    ``return_nu_legs`` (OPT-IN, default off → 4-tuple unchanged) appends the ANNIHILATION-only ν-side
    legs ``(net_nu_a, frs_nu_a)`` — the χχ̄→νν̄ ν-GAIN spectrum WITHOUT the elastic νχ→νχ scatter
    (which up-scatters ν to a spuriously hard ⟨ε⟩). This is the correct shape for the strong-χ
    true-shape χ→ν energy delivery: it peaks at ε~m_χ/T (on-grid) with the right ⟨ε⟩=ΔE/ΔN, so
    energy-normalising it ALSO conserves ν number. Mutually exclusive with ``return_legs``.

    ``smooth_floor`` > 0 (OPT-IN) routes every off-grid occupation read through the C¹ smooth log
    clamp and applies the same floor to the on-grid log f_χ — the contained strong-χ stiffness cure.

    Defaults (all off) are bit-identical to the original hard-mask totals."""
    import numpy as np
    eps_nu = grid.abscissas
    chi_eps = chi_grid.abscissas
    logf_nu = jnp.log(occ)                                      # (3,2,Nnu)
    if smooth_floor > 0.0:
        logf_chi = _smooth_clamp_log(jnp.log(jnp.maximum(f_chi, 1e-300)), smooth_floor, True)
    else:
        logf_chi = jnp.log(jnp.clip(f_chi, 1e-300, 1.0))       # (Nx,)

    # Local occupation readers that fold in the (static) smooth_floor — the call sites below stay
    # unchanged; smooth_floor=0.0 makes these bit-identical to the bare interp clamp (SM-safe).
    def _chi_occ_at(logf_c, ceps, q, tr):
        return _chi_occ_impl(logf_c, ceps, q, tr, smooth_floor)

    def _nu_occ_at(logf_fc, neps, p):
        return _nu_occ_impl(logf_fc, neps, p, smooth_floor, fd_decay, monotone_read)

    # canonical Legendre rules (host)
    xp, wp = np.polynomial.legendre.leggauss(n_p2)
    xc, wc = np.polynomial.legendre.leggauss(n_c2)
    xe, we = np.polynomial.legendre.leggauss(n_e3)
    xf, wf = np.polynomial.legendre.leggauss(n_phi)
    p2max = p2max_fac
    p2 = jnp.asarray(0.5 * p2max * (xp + 1.0))[None, :, None, None, None]
    wp2 = jnp.asarray(0.5 * p2max * wp)[None, :, None, None, None]
    c2 = jnp.asarray(xc)[None, None, :, None, None]
    wc2 = jnp.asarray(wc)[None, None, :, None, None]
    xeb = jnp.asarray(xe)[None, None, None, :, None]
    we3 = jnp.asarray(we)[None, None, None, :, None]
    cphi = jnp.asarray(np.cos(0.5 * PI * (xf + 1.0)))[None, None, None, None, :]   # φ∈[0,π]
    wphi = jnp.asarray(0.5 * PI * wf)[None, None, None, None, :]
    W = wp2 * wc2 * we3 * wphi                                  # (1,Np2,Nc2,Ne3,Nphi)
    # analytic-φ variant (OPT-IN): single cphi=0 node, π folded into the poly (see _process)
    cphi0 = jnp.zeros((1, 1, 1, 1, 1))
    W_an = wp2 * wc2 * we3                                      # (1,Np2,Nc2,Ne3,1)

    m = meps
    Kf = Kflav

    def _process(p1g, m1, m2, m3, m4, poly_fn, ccoef=1.0):
        """Return geometric (measure·W, poly, E1, p2grid, p3, p4) for a tracked-momentum array p1g.

        ``analytic_phi`` (closure flag): replace the n_phi azimuthal quadrature with its EXACT value.
        cphi enters the integrand only through ``poly`` (quadratic in cphi via t=t0+β·cphi,
        uu=uu0−β·cphi), so ∫₀^π poly dφ = π·(poly(cphi=0) + ½·ccoef·β²), ccoef = 1 (scat) / 2 (annih).
        Drops the n_phi axis and is the n_phi→∞ limit — strictly more accurate than GL-n_phi. Default
        off ⇒ bit-identical (betasq DCE'd)."""
        p1 = p1g[:, None, None, None, None]
        if analytic_phi:
            s, t0, uu0, measure, E1, p3, p4, betasq = _kin(p1, m1, m2, m3, m4, p2, c2, xeb, cphi0,
                                                           mask_w=mask_w, exact_e3=exact_e3)
            meas = measure * W_an                              # (N,Np2,Nc2,Ne3,1)
            poly = PI * (poly_fn(s, t0, uu0) + 0.5 * ccoef * betasq)
        else:
            s, t, uu, measure, E1, p3, p4, _bsq = _kin(p1, m1, m2, m3, m4, p2, c2, xeb, cphi,
                                                       mask_w=mask_w, exact_e3=exact_e3)
            meas = measure * W                                 # (N,Np2,Nc2,Ne3,Nphi)
            poly = poly_fn(s, t, uu)
        return meas, poly, E1[:, 0, 0, 0, 0], jnp.broadcast_to(p2, s.shape), p3, p4

    def poly_scat(s, t, uu):    # (s−m²)²+(u−m²)²+2m²t  (×Kflav, ×dof outside)
        return (s - m ** 2) ** 2 + (uu - m ** 2) ** 2 + 2.0 * m ** 2 * t

    def poly_annih(s, t, uu):   # (t−m²)²+(u−m²)²+2m²s
        return (t - m ** 2) ** 2 + (uu - m ** 2) ** 2 + 2.0 * m ** 2 * s

    # ---- ν-side: tracked ν on the ν grid (p1 = eps_nu) ----
    p1nu = eps_nu
    # scattering νχ→νχ: 1=ν,2=χ,3=ν,4=χ ; df_ν uses full Σ = 4·poly
    meas_s, poly_s, E1_s, p2_s, p3_s, p4_s = _process(p1nu, 0.0, m, 0.0, m, poly_scat)
    fchi2_s = _chi_occ_at(logf_chi, chi_eps, p2_s, tratio)     # χ partner: p2 IS its momentum
    fchi4_s = _chi_occ_at(logf_chi, chi_eps, p4_s, tratio)     # χ final:   p4 IS its momentum
    # annihilation νν̄→χχ̄: 1=ν,2=ν̄,3=χ,4=χ̄
    meas_a, poly_a, E1_a, p2_a, p3_a, p4_a = _process(p1nu, 0.0, 0.0, m, m, poly_annih, ccoef=2.0)
    fchi3_a = _chi_occ_at(logf_chi, chi_eps, p3_a, tratio)
    fchi4_a = _chi_occ_at(logf_chi, chi_eps, p4_a, tratio)

    net_nu = jnp.zeros((3, 2, p1nu.shape[0]))
    frs_nu = jnp.zeros((3, 2, p1nu.shape[0]))
    net_nu_a = jnp.zeros((3, 2, p1nu.shape[0]))   # annih-only ν legs (return_nu_legs)
    frs_nu_a = jnp.zeros((3, 2, p1nu.shape[0]))
    inv_e1 = 1.0 / jnp.where(E1_s > 0.0, E1_s, 1.0)             # (N,)
    for fl in range(3):
        cf = 4.0 * Kf[fl]                                      # full Σ = 4 K poly
        for cp in range(2):
            f1 = occ[fl, cp][:, None, None, None, None]
            # scatter labeling 1=ν(p1),2=χ(p2),3=ν(p3),4=χ(p4) → F = fν(p1),fχ(p2),fν(p3),fχ(p4);
            # ×2 (below) for the χ̄ target (identical |M|² by C-inv).
            Fn_s, Ff_s = _Ffac(f1, fchi2_s, _nu_occ_at(logf_nu[fl, cp], eps_nu, p3_s), fchi4_s)
            ns = cf * inv_e1 * jnp.sum(meas_s * poly_s * Fn_s, axis=(1, 2, 3, 4))
            fs = cf * inv_e1 * jnp.sum(meas_s * poly_s * Ff_s, axis=(1, 2, 3, 4))
            # annih: 1=ν(p1),2=ν̄(p2),3=χ(p3),4=χ̄(p4); partner ν̄ at opposite cp
            fnu2a = _nu_occ_at(logf_nu[fl, 1 - cp], eps_nu, p2_a)
            Fn_a, Ff_a = _Ffac(f1, fnu2a, fchi3_a, fchi4_a)
            na = 4.0 * Kf[fl] * inv_e1 * jnp.sum(meas_a * poly_a * Fn_a, axis=(1, 2, 3, 4))
            fa = 4.0 * Kf[fl] * inv_e1 * jnp.sum(meas_a * poly_a * Ff_a, axis=(1, 2, 3, 4))
            net_nu = net_nu.at[fl, cp].set(PREF_BRUTE * (2.0 * ns + na))
            frs_nu = frs_nu.at[fl, cp].set(PREF_BRUTE * (2.0 * fs + fa))
            net_nu_a = net_nu_a.at[fl, cp].set(PREF_BRUTE * na)     # annih ν-gain only (no scatter)
            frs_nu_a = frs_nu_a.at[fl, cp].set(PREF_BRUTE * fa)

    # ---- χ-side: tracked χ on the χ grid (p1 = q1 = chi_eps/tratio in T_cm units) ----
    q1 = chi_eps / tratio
    f_chi1 = f_chi[:, None, None, None, None]
    inv_e1c = 1.0 / jnp.where(_safe(q1 ** 2 + m ** 2) > 0.0, _safe(q1 ** 2 + m ** 2), 1.0)
    # scattering χν→χν: 1=χ,2=ν,3=χ,4=ν ; df_χ uses Σ/2 = 2·poly ; sum over ν cp
    meas_cs, poly_cs, E1_cs, p2_cs, p3_cs, p4_cs = _process(q1, m, 0.0, m, 0.0, poly_scat)
    fchi3_cs = _chi_occ_at(logf_chi, chi_eps, p3_cs, tratio)   # χ final: p3 IS its momentum
    # annihilation χχ̄→νν̄: 1=χ,2=χ̄,3=ν,4=ν̄ ; Σ/2 = 2·poly
    meas_ca, poly_ca, E1_ca, p2_ca, p3_ca, p4_ca = _process(q1, m, m, 0.0, 0.0, poly_annih, ccoef=2.0)
    fchi2_ca = _chi_occ_at(logf_chi, chi_eps, p2_ca, tratio)   # χ̄ partner: p2 IS its momentum

    net_chi = jnp.zeros((q1.shape[0],))
    frs_chi = jnp.zeros((q1.shape[0],))
    # OPT-IN leg separation (return_legs): accumulate the elastic-scatter (number-CONSERVING) and
    # annihilation (number-CHANGING) χ-side contributions separately. Default-off → the returned
    # totals are computed by the same expressions in the same order, so they are bit-identical. Used
    # by the freeze-out diagnostics to confirm Γ_scat≫Γ_annih.
    net_chi_s = jnp.zeros((q1.shape[0],)); net_chi_a = jnp.zeros((q1.shape[0],))
    frs_chi_s = jnp.zeros((q1.shape[0],)); frs_chi_a = jnp.zeros((q1.shape[0],))
    for fl in range(3):
        cf = 2.0 * Kf[fl]                                      # Σ/2 = 2 K poly
        for cp in range(2):
            fnu2 = _nu_occ_at(logf_nu[fl, cp], eps_nu, p2_cs)
            fnu4 = _nu_occ_at(logf_nu[fl, cp], eps_nu, p4_cs)
            Fn, Ff = _Ffac(f_chi1, fnu2, fchi3_cs, fnu4)
            cs_n = PREF_BRUTE * cf * inv_e1c * jnp.sum(meas_cs * poly_cs * Fn, axis=(1, 2, 3, 4))
            cs_f = PREF_BRUTE * cf * inv_e1c * jnp.sum(meas_cs * poly_cs * Ff, axis=(1, 2, 3, 4))
            net_chi = net_chi + cs_n
            frs_chi = frs_chi + cs_f
            if return_legs:
                net_chi_s = net_chi_s + cs_n; frs_chi_s = frs_chi_s + cs_f
        # annih: final ν(cp0) ν̄(cp1) of flavor fl
        fnu3 = _nu_occ_at(logf_nu[fl, 0], eps_nu, p3_ca)
        fnu4 = _nu_occ_at(logf_nu[fl, 1], eps_nu, p4_ca)
        Fn, Ff = _Ffac(f_chi1, fchi2_ca, fnu3, fnu4)
        ca_n = PREF_BRUTE * cf * inv_e1c * jnp.sum(meas_ca * poly_ca * Fn, axis=(1, 2, 3, 4))
        ca_f = PREF_BRUTE * cf * inv_e1c * jnp.sum(meas_ca * poly_ca * Ff, axis=(1, 2, 3, 4))
        net_chi = net_chi + ca_n
        frs_chi = frs_chi + ca_f
        if return_legs:
            net_chi_a = net_chi_a + ca_n; frs_chi_a = frs_chi_a + ca_f

    if return_legs:
        return (net_nu, frs_nu, net_chi, frs_chi,
                net_chi_s, frs_chi_s, net_chi_a, frs_chi_a)
    if return_nu_legs:
        return net_nu, frs_nu, net_chi, frs_chi, net_nu_a, frs_nu_a
    return net_nu, frs_nu, net_chi, frs_chi


# floor for the coupled-solve gate (mirror chi_nu.PREC_FLOOR_CHI / s2_prec, brute kernel)
PREC_FLOOR_CHI = 1.0e-13


def s2_brute_prec(grid, occ_eq, f_chi_eq, chi_grid, meps, tratio, Kflav, **nodes):
    """Brute equilibrium noise floor (prec_nu (3,2,N), prec_chi (Nx,)) = |net|/frs at a common-T
    equilibrium — the gate normalization for the brute coupled-solve path (analogue of
    chi_nu.s2_prec). Must use the same nodes as the RHS s2_brute_rates."""
    net_nu, frs_nu, net_chi, frs_chi = s2_brute_rates(grid, occ_eq, f_chi_eq, chi_grid, meps,
                                                      tratio, Kflav, **nodes)
    prec_nu = jnp.where(frs_nu > 0.0, jnp.abs(net_nu) / jnp.where(frs_nu > 0.0, frs_nu, 1.0),
                        PREC_FLOOR_CHI)
    prec_chi = jnp.where(frs_chi > 0.0, jnp.abs(net_chi) / jnp.where(frs_chi > 0.0, frs_chi, 1.0),
                         PREC_FLOOR_CHI)
    return jnp.maximum(prec_nu, PREC_FLOOR_CHI), jnp.maximum(prec_chi, PREC_FLOOR_CHI)
