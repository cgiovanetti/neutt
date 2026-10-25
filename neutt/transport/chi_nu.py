"""S2 dark-sector χ↔ν 2↔2 collision kernels (Dirac χ + heavy Z′ contact limit).

Scenario S2: a Dirac DM fermion χ (mass m_χ, temperature T_χ) interacts with neutrinos through a
heavy mediator Z′ (m_Z′ ≫ T, never on-shell). In the contact limit the 2↔2 amplitudes are the SM
ν-lepton kernel family at CV=1, CA=0, m_e→m_χ, with K = g_χ²g_ν²/m_Z′⁴. Being a 4-fermion contact
interaction the rate scales as K·T⁵ exactly like the SM weak kernels, so these kernels keep the SM
``tcmev⁵`` convention and the assemble.py machinery (unlike the S1/S3 1↔2 decay in dv_nu.py).

This module holds the ν-side kernels (tracked particle a massless ν, massive χ partner): the
kinematics are identical to the SM nuer1/nuer2/nuepma kernels, so the tile geometry (cut energies,
angular antiderivatives) is reused verbatim with meps = m_χ/T_cm. Only the coupling coefficients
change and the analytic e± occupation ``_fd`` becomes the interpolated tracked f_χ.

Dof / factor-of-2 convention (physically correct):
  |M|² is the same forwards/backwards (CPT). To make a per-bin df_a/dt, the spin-summed |M|² (Σ, all
  external spins) is divided by the *tracked* species' dof g_a and summed (not averaged) over every
  partner's dof. The tracked particle is the neutrino: g_ν = 1 (one coupling helicity), so df_ν/dt
  uses the full spin-sum Σ. (The χ-side kernel tracks χ with g_χ = 2 Dirac spins, so df_χ/dt uses
  Σ/2 — half of this.) Σ for νχ→νχ is 4K[(s−m²)²+(u−m²)²+2m²t]; in the (P1·Q)² + c2 m²(P1·Q) tile
  form:

  νχ→νχ scattering (df_ν, g_ν=1, full Σ):
      Σ = 16K[(P1·Q2)² − m²(P1·Q2)]  +  16K[(P1·Q3)² + m²(P1·Q3)]
        = nuer1 (s-piece, term1=16K, c2=−1) + nuer2 (u-piece, term1=16K, c2=+1)

  term1=16K matches the gold-gated SM legs (nuer1's SM term1=32(2sin²θ+fla)² is the full-Σ df_ν
  normalization), so the same machinery yields the physically-correct full-Σ df_ν for the dark sector.

The χ-side kernels (massive tracked particle, new cut structure) live below.

Single-target (χ) rate. ν also scatters off χ̄ with identical |M|² (C-invariance of the vector
χ current); for a CP-symmetric single per-dof f_χ that doubles the ν-side rate — applied in the
RHS assembly, not here, so this matches the single-target brute-force oracle.
"""
from __future__ import annotations

import jax.numpy as jnp

from ..constants import HBAR, PI
from ..interp import interp_log_occ_dynamic
from ..grid import legendre_canonical, gauss_laguerre_plain
# Reuse the gold-gated SM tile GEOMETRY (cuts, angular antiderivatives, tile tables).
from .nuer1 import _R1, _R2, _topbot, _E2_SEGMENTS, _safe_sqrt_diff
from .nuer2 import (_R2m1, _R2m2, _topbot_r2, _REGIMES as _R2_REGIMES)
from .nuepma import _L1, _L2, _topbot_pma, _REGIMES as _PMA_REGIMES

PREFAC_GEOM = 1.0 / 16.0 / (2.0 * PI) ** 3   # the 1/(2⁴(2π)³) phase-space constant; ×(K/ħ) per flavor


def _chi_occ_at_energy(logf_chi, chi_eps, E_tcm, meps, tratio):
    """Interpolated per-dof f_χ at an energy E (T_cm units). χ momentum q=√(E²−meps²);
    χ-grid query ε_χ = q·tratio (tratio = T_cm/T_χ). Fermion → clamp_high=True."""
    q = _safe_sqrt_diff(E_tcm ** 2 - meps ** 2)
    return jnp.exp(interp_log_occ_dynamic(logf_chi, chi_eps, q * tratio, clamp_high=True))


def _nuside_scatt_r1(grid, occ, f_chi, chi_grid, meps, tratio, Kflav, n_gl, n_lag):
    """s-piece of νχ→νχ (df_ν/dt): nuer1 geometry, term1=16K (full-Σ df_ν dof doubling, g_ν=1 — see
    module header), c2=−1, χ partner. (net,frs) (3,2,N)."""
    eps = grid.abscissas
    p1 = eps
    N = eps.shape[0]
    logf = jnp.log(occ)
    logf_chi = jnp.log(jnp.clip(f_chi, 1e-300, 1.0))
    chi_eps = chi_grid.abscissas
    c2 = -1.0
    # per-flavor coupling: term1[flavor] = 16 K[flavor] = full spin-sum Σ for df_ν (g_ν=1); (3,) net
    # factor. See module DOF note.
    coup = (16.0 * Kflav / HBAR) * PREFAC_GEOM           # (3,)  [s⁻¹ MeV⁻⁵ · (ε-integral)]

    tleg, wleg = legendre_canonical(n_gl)
    xlag, wlag = gauss_laguerre_plain(n_lag)

    ecut3 = jnp.sqrt(p1 ** 2 + meps ** 2)
    denom_a = meps - 2.0 * p1
    ecut1 = meps + 2.0 * p1 ** 2 / jnp.where(denom_a != 0.0, denom_a, 1.0)
    regimeA = p1 < (meps / 2.0)
    binmask = (p1 > 0.0).astype(eps.dtype)
    p1sq = jnp.where(p1 > 0.0, p1 ** 2, 1.0)
    bounds_e2 = {"meps": jnp.broadcast_to(meps, (N,)), "ecut3": ecut3, "ecut1": ecut1}

    net = jnp.zeros((3, 2, N))
    frs = jnp.zeros((3, 2, N))

    for (name, rule, a2k, b2k, regime, subsegs) in _E2_SEGMENTS:
        a2 = bounds_e2[a2k]
        if rule == "leg":
            b2 = bounds_e2[b2k]
            mid2 = 0.5 * (a2 + b2); half2 = 0.5 * (b2 - a2)
            E2 = mid2[:, None] + half2[:, None] * tleg[None, :]
            e2w = wleg[None, :] * half2[:, None]
        else:
            E2 = a2[:, None] + xlag[None, :]
            e2w = jnp.broadcast_to(wlag[None, :], E2.shape)
        E2 = jnp.maximum(E2, meps)
        q2 = _safe_sqrt_diff(E2 ** 2 - meps ** 2)
        d = 2.0 * p1[:, None] + E2 - q2
        etrans2 = 0.5 * (2.0 * p1[:, None] + E2 - q2 + meps ** 2 / jnp.where(d != 0.0, d, 1.0))
        dl = 2.0 * p1[:, None] + E2 + q2
        elim1 = 0.5 * (2.0 * p1[:, None] + E2 + q2 + meps ** 2 / jnp.where(dl != 0.0, dl, 1.0))
        A2 = (p1[:, None] + E2) ** 2 - meps ** 2
        fchi2 = _chi_occ_at_energy(logf_chi, chi_eps, E2, meps, tratio)        # (N,A)
        bounds_e3 = {"meps": jnp.broadcast_to(meps, E2.shape), "e2": E2,
                     "etrans2": etrans2, "elim1": elim1}
        mask = (jnp.ones((N,)) if regime == "all"
                else regimeA.astype(eps.dtype) if regime == "A" else (~regimeA).astype(eps.dtype))

        for (a3k, b3k, tbkind) in subsegs:
            a3 = bounds_e3[a3k]; b3 = bounds_e3[b3k]
            mid3 = 0.5 * (a3 + b3); half3 = 0.5 * (b3 - a3)
            E3 = mid3[:, :, None] + half3[:, :, None] * tleg[None, None, :]
            e3w = wleg[None, None, :] * half3[:, :, None]
            E3 = jnp.maximum(E3, meps)
            q3 = _safe_sqrt_diff(E3 ** 2 - meps ** 2)
            top, bot = _topbot(tbkind, p1[:, None, None], E2[:, :, None], q2[:, :, None], E3, q3)
            I1 = _R1(top, A2[:, :, None]) - _R1(bot, A2[:, :, None])
            I2 = _R2(top, A2[:, :, None]) - _R2(bot, A2[:, :, None])
            w3 = I1 + c2 * meps ** 2 * I2                                      # (N,A,B), term1 folded into coup

            fchi3 = _chi_occ_at_energy(logf_chi, chi_eps, E3, meps, tratio)    # (N,A,B)
            p4 = jnp.maximum(p1[:, None, None] + E2[:, :, None] - E3, 0.0)
            fnu4 = jnp.exp(interp_log_occ_dynamic(logf, eps, p4))              # (3,2,N,A,B)

            fnu1 = occ[:, :, :, None, None]                                    # (m,n,N,1,1)
            fchi2b = fchi2[None, None, :, :, None]                             # (1,1,N,A,1)
            fchi3b = fchi3[None, None]                                         # (1,1,N,A,B)
            forf = (1.0 - fnu1) * (1.0 - fchi2b) * fchi3b * fnu4
            revf = fnu1 * fchi2b * (1.0 - fchi3b) * (1.0 - fnu4)
            Fnet = forf - revf
            Ffrs = forf + revf

            meas = ((mask * binmask)[:, None, None] * (e2w / p1sq[:, None])[:, :, None] * e3w)  # (N,A,B)
            w3b = (w3 * meas)[None, None]                                      # (1,1,N,A,B)
            net = net + coup[:, None, None] * jnp.sum(w3b * Fnet, axis=(3, 4))
            frs = frs + coup[:, None, None] * jnp.sum(w3b * Ffrs, axis=(3, 4))

    return net, frs


def _nuside_scatt_r2(grid, occ, f_chi, chi_grid, meps, tratio, Kflav, n_gl, n_lag):
    """u-piece of νχ→νχ (df_ν/dt): nuer2 geometry, term1=16K (full-Σ df_ν dof doubling), c2=+1,
    χ partner. Returns (net,frs)."""
    eps = grid.abscissas
    p1 = eps
    N = eps.shape[0]
    logf = jnp.log(occ)
    logf_chi = jnp.log(jnp.clip(f_chi, 1e-300, 1.0))
    chi_eps = chi_grid.abscissas
    c2 = +1.0
    coup = (16.0 * Kflav / HBAR) * PREFAC_GEOM           # (3,)  16K = full Σ (df_ν, g_ν=1)

    tleg, wleg = legendre_canonical(n_gl)
    xlag, wlag = gauss_laguerre_plain(n_lag)

    binmask = (p1 > 0.0).astype(eps.dtype)
    p1sq = jnp.where(p1 > 0.0, p1 ** 2, 1.0)
    ecut1 = p1 + meps ** 2 / jnp.where(p1 > 0.0, 4.0 * p1, 1.0)
    ecut2 = p1 + meps * (p1 + meps) / (2.0 * p1 + meps)
    ecut3 = jnp.sqrt(p1 ** 2 + meps ** 2)
    bounds3 = {"meps": jnp.broadcast_to(jnp.asarray(meps, eps.dtype), (N,)),
               "ecut1": ecut1, "ecut2": ecut2, "ecut3": ecut3}
    pcase1 = meps * (jnp.sqrt(5.0) - 1.0) / 4.0
    pcase2 = meps / 2.0 / jnp.sqrt(2.0)
    pcase3 = meps / 2.0
    regmasks = {
        "R1": (p1 < pcase1).astype(eps.dtype),
        "R2": ((p1 >= pcase1) & (p1 < pcase2)).astype(eps.dtype),
        "R3": ((p1 >= pcase2) & (p1 < pcase3)).astype(eps.dtype),
        "R4": (p1 >= pcase3).astype(eps.dtype),
    }
    net = jnp.zeros((3, 2, N))
    frs = jnp.zeros((3, 2, N))

    for rname, segments in _R2_REGIMES.items():
        regmask = regmasks[rname] * binmask
        for (e3rule, a3k, b3k, subsegs) in segments:
            a3 = bounds3[a3k]
            if e3rule == "leg":
                b3 = bounds3[b3k]
                mid3 = 0.5 * (a3 + b3); half3 = 0.5 * (b3 - a3)
                E3 = mid3[:, None] + half3[:, None] * tleg[None, :]
                e3w = wleg[None, :] * half3[:, None]
            else:
                E3 = a3[:, None] + xlag[None, :]
                e3w = jnp.broadcast_to(wlag[None, :], E3.shape)
            E3 = jnp.maximum(E3, meps)
            q3 = _safe_sqrt_diff(E3 ** 2 - meps ** 2)
            d = E3 + q3 - 2.0 * p1[:, None]
            etrans2 = 0.5 * (E3 + q3 - 2.0 * p1[:, None] + meps ** 2 / jnp.where(d != 0.0, d, 1.0))
            db = E3 - q3 - 2.0 * p1[:, None]
            elim1raw = 0.5 * (E3 - q3 - 2.0 * p1[:, None] + meps ** 2 / jnp.where(db != 0.0, db, 1.0))
            elim1 = jnp.where(elim1raw < 0.0, jnp.maximum(E3, etrans2), elim1raw)
            elim2 = etrans2
            fchi3 = _chi_occ_at_energy(logf_chi, chi_eps, E3, meps, tratio)        # (N,K)
            kinemfact1 = e3w / p1sq[:, None]
            Bcoef = (p1[:, None] - E3) ** 2 - meps ** 2
            bounds2 = {"meps": jnp.broadcast_to(jnp.asarray(meps, eps.dtype), E3.shape),
                       "e3": E3, "etrans2": etrans2, "elim1": elim1, "elim2": elim2}

            for (e2rule, a2k, b2k, kind, sel) in subsegs:
                a2 = bounds2[a2k]
                if e2rule == "leg":
                    b2 = bounds2[b2k]
                    mid2 = 0.5 * (a2 + b2); half2 = 0.5 * (b2 - a2)
                    E2 = mid2[:, :, None] + half2[:, :, None] * tleg[None, None, :]
                    e2w = wleg[None, None, :] * half2[:, :, None]
                else:
                    E2 = a2[:, :, None] + xlag[None, None, :]
                    e2w = jnp.broadcast_to(wlag[None, None, :], E2.shape)
                E2 = jnp.maximum(E2, meps)
                q2 = _safe_sqrt_diff(E2 ** 2 - meps ** 2)
                top, bot = _topbot_r2(kind, p1[:, None, None], E3[:, :, None], q3[:, :, None], E2, q2)
                Be = Bcoef[:, :, None]
                I1 = _R2m1(top, Be) - _R2m1(bot, Be)
                I2 = _R2m2(top, Be) - _R2m2(bot, Be)
                w2 = I1 + c2 * meps ** 2 * I2                                      # (N,K,J)
                fchi2 = _chi_occ_at_energy(logf_chi, chi_eps, E2, meps, tratio)    # (N,K,J)
                p4 = jnp.maximum(p1[:, None, None] + E2 - E3[:, :, None], 0.0)
                fnu4 = jnp.exp(interp_log_occ_dynamic(logf, eps, p4))              # (3,2,N,K,J)

                fnu1 = occ[:, :, :, None, None]
                fchi2b = fchi2[None, None]
                fchi3b = fchi3[None, None, :, :, None]
                forf = (1.0 - fnu1) * (1.0 - fchi2b) * fchi3b * fnu4
                revf = fnu1 * fchi2b * (1.0 - fchi3b) * (1.0 - fnu4)
                Fnet = forf - revf
                Ffrs = forf + revf

                if sel is None:
                    extra = jnp.ones(E3.shape)
                else:
                    gap = bounds2[b2k] - a2
                    extra = ((gap < 200.0) if sel == "selleg" else (gap >= 200.0)).astype(eps.dtype)
                meas = (regmask[:, None, None] * (kinemfact1 * extra)[:, :, None] * e2w)  # (N,K,J)
                w2b = (w2 * meas)[None, None]
                net = net + coup[:, None, None] * jnp.sum(w2b * Fnet, axis=(3, 4))
                frs = frs + coup[:, None, None] * jnp.sum(w2b * Ffrs, axis=(3, 4))

    return net, frs


def nuside_scatter_rates(grid, occ, f_chi, chi_grid, meps, tratio, Kflav,
                         n_gl: int = 5, n_lag: int = 5):
    """ν-side df_ν/dt for νχ→νχ scattering [units: ×tcmev⁵ → s⁻¹], single χ target.

    grid, occ : ν ε-grid and occupation (3,2,N).
    f_chi : (Nx,) per-dof χ occupation on chi_grid (FD).
    meps = m_χ/T_cm, tratio = T_cm/T_χ, Kflav = (3,) g_χ²g_ν[α]²/m_Z′⁴ per flavor.
    Returns (net, frs), each (3,2,N). df_ν/dt = tcmev⁵·net.
    """
    n1, f1 = _nuside_scatt_r1(grid, occ, f_chi, chi_grid, meps, tratio, Kflav, n_gl, n_lag)
    n2, f2 = _nuside_scatt_r2(grid, occ, f_chi, chi_grid, meps, tratio, Kflav, n_gl, n_lag)
    return n1 + n2, f1 + f2


def nuside_annih_rates(grid, occ, f_chi, chi_grid, meps, tratio, Kflav, n_gl: int = 5, n_lag: int = 5):
    """ν-side df_ν/dt for νν̄→χχ̄ annihilation [×tcmev⁵ → s⁻¹]. Returns (net,frs), each (3,2,N).

    Reuses the SM nuepma tile geometry (cut energies, _L1/_L2 antiderivatives, the two χ-vs-χ̄ "swap"
    assignments, four p1-regimes) — the tracked ν₁ is massless with a massive final-state pair exactly
    as in νν̄→e⁺e⁻, so the geometry transfers verbatim with meps=m_χ/T_cm. Differences from nuepma:
    (i) the e± occupations become the interpolated tracked f_χ (χ and χ̄ share one CP-symmetric per-dof
    spectrum); (ii) the coupling is the contact form at CV=1, CA=0: term1 = 32 K and c2 = +1 for both
    swaps/flavors, K=g_χ²g_ν[α]²/m_Z′⁴. The nuepma swap-sum evaluates a single tile twice (χ,χ̄ share
    f_χ), so 32K — twice the scattering 16K — reconstructs the full Σ_annih = 4K[(t−m²)²+(u−m²)²+2m²s].
    """
    eps = grid.abscissas
    p1 = eps
    N = eps.shape[0]
    logf = jnp.log(occ)
    logf_chi = jnp.log(jnp.clip(f_chi, 1e-300, 1.0))
    chi_eps = chi_grid.abscissas
    c2 = 1.0
    # per-flavor coupling: term1 = 32 K[flavor] (the SM nuepma literal 32 at CV=1,CA=0), carried as
    # a (3,) factor on the net. The nuepma swap-sum evaluates a SINGLE tile from `eout` TWICE
    # (w_true=w_false since χ,χ̄ share f_chi), so the full Σ_annih is reconstructed with 32K — the
    # factor-of-2 vs the scattering 16K is brute-verified.
    coup = (32.0 * Kflav / HBAR) * PREFAC_GEOM                # (3,)

    tleg, wleg = legendre_canonical(n_gl)
    xlag, wlag = gauss_laguerre_plain(n_lag)

    meps_s = jnp.asarray(meps, eps.dtype)
    binmask = (p1 > 0.0).astype(eps.dtype)
    p1sq = jnp.where(p1 > 0.0, p1 ** 2, 1.0)

    ecut1 = p1 + meps ** 2 / jnp.where(p1 > 0.0, 4.0 * p1, 1.0)
    den2 = 2.0 * p1 - meps
    ecut2 = 0.5 * (2.0 * p1 - meps + meps ** 2 / jnp.where(den2 != 0.0, den2, 1.0))
    bounds_out = {"meps": jnp.broadcast_to(meps_s, (N,)), "ecut1": ecut1, "ecut2": ecut2, "p1": p1}

    pcase1 = meps / 2.0
    pcase2 = meps * (1.0 + jnp.sqrt(5.0)) / 4.0
    pcase3 = meps
    regmasks = {
        "R1": (p1 < pcase1).astype(eps.dtype),
        "R2": ((p1 >= pcase1) & (p1 < pcase2)).astype(eps.dtype),
        "R3": ((p1 >= pcase2) & (p1 < pcase3)).astype(eps.dtype),
        "R4": (p1 >= pcase3).astype(eps.dtype),
    }
    clamp_e2 = {"R1": False, "R2": True, "R3": False, "R4": False}

    net = jnp.zeros((3, 2, N)); frs = jnp.zeros((3, 2, N))

    def _outer_nodes(rule, a, b):
        if rule == "leg":
            mid = 0.5 * (a + b); half = 0.5 * (b - a)
            return mid[:, None] + half[:, None] * tleg[None, :], wleg[None, :] * half[:, None]
        return a[:, None] + xlag[None, :], jnp.broadcast_to(wlag[None, :], (N, xlag.shape[0]))

    def _inner_nodes(rule, a, b):
        if rule == "leg":
            mid = 0.5 * (a + b); half = 0.5 * (b - a)
            return mid[:, :, None] + half[:, :, None] * tleg[None, None, :], wleg[None, None, :] * half[:, :, None]
        return a[:, :, None] + xlag[None, None, :], jnp.broadcast_to(wlag[None, None, :], a.shape + (xlag.shape[0],))

    for rname, segments in _PMA_REGIMES.items():
        regmask = regmasks[rname] * binmask
        for (out_rule, a_out_k, b_out_k, inner_tiles) in segments:
            a_out = bounds_out[a_out_k]
            if out_rule == "genout":
                b_out = bounds_out[b_out_k]
                out_variants = [("leg", a_out, b_out, (b_out - a_out < 200.0).astype(eps.dtype)),
                                ("lag", a_out, None, (b_out - a_out >= 200.0).astype(eps.dtype))]
            elif out_rule == "leg":
                out_variants = [("leg", a_out, bounds_out[b_out_k], jnp.ones((N,)))]
            else:
                out_variants = [("lag", a_out, None, jnp.ones((N,)))]

            for (orule, oa, ob, out_mask) in out_variants:
                eout, eoutw = _outer_nodes(orule, oa, ob)
                eout = jnp.maximum(eout, meps_s)
                qout = _safe_sqrt_diff(eout ** 2 - meps_s ** 2)
                p1c = p1[:, None]
                d1 = 2.0 * p1c - eout - qout
                etrans1 = 0.5 * (2.0 * p1c - eout - qout + meps ** 2 / jnp.where(d1 != 0.0, d1, 1.0))
                d2 = 2.0 * p1c - eout + qout
                etrans2_raw = 0.5 * (2.0 * p1c - eout + qout + meps ** 2 / jnp.where(d2 != 0.0, d2, 1.0))
                etrans2 = jnp.maximum(meps_s, etrans2_raw) if clamp_e2[rname] else etrans2_raw
                C = (p1c - eout) ** 2 - meps ** 2
                kinemfact1 = eoutw / p1sq[:, None]
                fchi_out = _chi_occ_at_energy(logf_chi, chi_eps, eout, meps, tratio)   # (N,J)
                bounds_in = {"meps": jnp.broadcast_to(meps_s, eout.shape), "etrans1": etrans1, "etrans2": etrans2}

                for (in_rule, a_in_k, b_in_k, kind, posguard) in inner_tiles:
                    a_in = bounds_in[a_in_k]
                    if in_rule == "genin":
                        b_in = bounds_in[b_in_k]; gap = b_in - a_in
                        in_variants = [("leg", a_in, b_in, ((gap >= 0.0) & (gap < 200.0)).astype(eps.dtype)),
                                       ("lag", a_in, None, ((gap < 0.0) | (gap >= 200.0)).astype(eps.dtype))]
                    elif in_rule == "leg":
                        in_variants = [("leg", a_in, bounds_in[b_in_k], jnp.ones(eout.shape))]
                    else:
                        m_in = (bounds_in[posguard] > 0.0).astype(eps.dtype) if posguard else jnp.ones(eout.shape)
                        in_variants = [("lag", a_in, None, m_in)]

                    for (irule, ia, ib, in_mask) in in_variants:
                        ein, einw = _inner_nodes(irule, ia, ib)
                        ein = jnp.maximum(ein, meps_s)
                        qin = _safe_sqrt_diff(ein ** 2 - meps_s ** 2)
                        top, bot = _topbot_pma(kind, p1[:, None, None], qout[:, :, None], eout[:, :, None], ein, qin)
                        Ce = C[:, :, None]
                        I1 = _L1(top, Ce) - _L1(bot, Ce)
                        I2 = _L2(top, Ce) - _L2(bot, Ce)
                        w3 = I1 + c2 * meps ** 2 * I2                  # (N,J,K); term1 folded into coup

                        fchi_in = _chi_occ_at_energy(logf_chi, chi_eps, ein, meps, tratio)   # (N,J,K)
                        p4 = jnp.maximum(eout[:, :, None] + ein - p1[:, None, None], 0.0)
                        fnu4 = jnp.exp(interp_log_occ_dynamic(logf, eps, p4))                # (3,2,N,J,K)
                        fnu4p = fnu4[:, ::-1]                          # partner ν̄ at opposite cp

                        fnu1 = occ[:, :, :, None, None]               # (m,n,N,1,1)
                        fc_in = fchi_in[None, None]                   # (1,1,N,J,K)
                        fc_out = fchi_out[None, None, :, :, None]      # (1,1,N,J,1)
                        # χ,χ̄ share f_chi ⇒ both swaps identical: forf = (1-fν1)(1-fν̄4)·f_χ(in)·f_χ(out)
                        forf = (1.0 - fnu1) * (1.0 - fnu4p) * fc_in * fc_out
                        revf = fnu1 * fnu4p * (1.0 - fc_in) * (1.0 - fc_out)

                        meas = ((regmask * out_mask)[:, None, None]
                                * (in_mask * kinemfact1)[:, :, None] * einw)   # (N,J,K)
                        w3b = (w3 * meas)[None, None]                  # (1,1,N,J,K)
                        net = net + coup[:, None, None] * jnp.sum(w3b * (forf - revf), axis=(3, 4))
                        frs = frs + coup[:, None, None] * jnp.sum(w3b * (forf + revf), axis=(3, 4))

    return net, frs


# =====================================================================================
# χ-side kernels (df_χ/dt) — massive tracked particle, new cut structure.
#
# The χ is the tracked species (on chi_grid); ν / χ̄ are the partners. The angular reduction has
# chi-side-specific cuts and cross-term integrands derived from first principles:
#
#   scattering νχ→νχ:  Σ/2 / (2K) = 2(s−m²)² + 2 s t0 + t²     [s+st: y=|q1+p2| cuts;
#                                                                t: z=|q1−q3| cuts]
#     t0 = 2m² − 2 E1 E3 + (q1²+y²−p2²)(y²+q3²−p4²)/(2y²)        (φ-averaged t; p4=E1+p2−E3)
#   annihilation χχ̄→νν̄: Σ/2 / (2K) = [t²−4m²t+6m⁴] + [u0²+½u1²−4m²u0]  [y=|q1−p3| cuts]
#     u0 = m² − 2 E1 E4 + 2(q1·b̂)(p4·b̂),  u1 = 2 q1⊥ p4⊥,  b=p3−q1 (|b|=y)
#
# Prefactor:
#     df_χ/dt = tcmev⁵ · (2 K/ħ)·PREFAC_GEOM · (1/(e1 q1)) · Σ_pieces ∫dE2 ∫dE3 ∫dy integrand · F
# i.e. exactly the ν-side machinery with 16K→2K (df_χ uses Σ/2: g_χ=2 Dirac spins) and the Jacobian
# 1/p1²→1/(e1·q1) (p1 = MOMENTUM q1 of the massive tracked χ, not E1). The Pauli factor F is
# y-INDEPENDENT, so the inner y-integral is purely geometric.
# =====================================================================================

NCHI_GL = 24       # outer (p2, E3) Gauss-Legendre nodes per finite segment
NCHI_Y = 16        # inner (y / z) Gauss-Legendre nodes
NCHI_LAG = 24      # Gauss-Laguerre nodes for the semi-infinite p2 / E3 tails

# The single χ-at-rest bin (ε≈eps_offset → q1→0) is a degenerate thin-shell for the reduction
# (~5% off, not a blow-up); it carries ~1e-10 of the χ number/energy moment, so it is physically
# irrelevant.


def _seg_nodes(a, b, xc, wc):
    """Map canonical Legendre (xc, wc) onto [a, b] (arrays). Weight = 0 where b <= a."""
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    nodes = mid[..., None] + half[..., None] * xc
    w = half[..., None] * wc * (b > a)[..., None]
    return nodes, w


def _lag_nodes(a, xl, wl):
    """Map plain Gauss-Laguerre (xl, wl) onto [a, ∞) (array a). Shift-invariant rule."""
    nodes = a[..., None] + xl
    w = jnp.broadcast_to(wl, nodes.shape)
    return nodes, w


def _qmom(E, meps):
    return _safe_sqrt_diff(E ** 2 - meps ** 2)


def _yint_poly(integ, ylo, yhi, yc, yw):
    """∫_{max(ylo,0)}^{yhi} integ(y) dy by Legendre; integ is a callable of y (broadcasts)."""
    ynodes, yweights = _seg_nodes(jnp.maximum(ylo, 0.0), yhi, yc, yw)
    return jnp.sum(integ(ynodes) * yweights, axis=-1)


def _Ffac_scatter(f_chi1, f_nu2, f_chi3, f_nu4):
    """F = (1−f1)(1−f2) f3 f4 − f1 f2 (1−f3)(1−f4) for νχ→νχ (1=χ,2=ν,3=χ,4=ν)."""
    forf = (1.0 - f_chi1) * (1.0 - f_nu2) * f_chi3 * f_nu4
    revf = f_chi1 * f_nu2 * (1.0 - f_chi3) * (1.0 - f_nu4)
    return forf - revf, forf + revf


def _chiside_scatter_sst(e1, q1, occ, eps_nu, logf_nu, f_chi1, logf_chi, chi_eps, meps, tratio,
                         xc, wc, xl, wl, yc, yw):
    """sst-reduction (y=|q1+p2| cuts 36-38): integrand 2(s−m²)²+2 s t0. Returns (net,frs) (3,N1)."""
    N1 = e1.shape[0]
    net = jnp.zeros((3, N1)); frs = jnp.zeros((3, N1))

    ecut1 = 0.5 * (meps + q1 - e1)                      # (N1,)
    ecut2 = q1
    # the 3 outer-p2 regions: (p2-rule, [plo, phi], [(e3lo_key, e3hi_key, y_rule), ...])
    regions = [
        ("leg", 0.0 * q1, jnp.maximum(ecut1, 0.0),
         [("etr1", "e1", "A"), ("e1", "elim2", "B")]),
        ("leg", jnp.maximum(ecut1, 0.0), jnp.maximum(ecut2, 0.0),
         [("meps", "etr1", "C"), ("etr1", "e1", "A"), ("e1", "elim2", "B")]),
        ("lag", ecut2, None,
         [("meps", "e1", "C"), ("e1", "etr1", "D"), ("etr1", "elim2", "B")]),
    ]
    for (p2rule, plo, phi, e3segs) in regions:
        if p2rule == "leg":
            p2, wp2 = _seg_nodes(plo, phi, xc, wc)      # (N1, Np2)
        else:
            p2, wp2 = _lag_nodes(plo, xl, wl)
        e1b = e1[:, None]; q1b = q1[:, None]            # (N1,1)
        d1 = e1b - q1b + 2.0 * p2
        etr1 = 0.5 * (d1 + meps ** 2 / jnp.where(d1 != 0.0, d1, 1.0))
        dl = e1b + q1b + 2.0 * p2
        elim2 = 0.5 * (dl + meps ** 2 / jnp.where(dl != 0.0, dl, 1.0))
        e3bounds = {"meps": jnp.broadcast_to(jnp.asarray(meps), p2.shape),
                    "etr1": etr1, "e1": jnp.broadcast_to(e1b, p2.shape), "elim2": elim2}
        # ν occupation at p2 (3,2,N1,Np2); χ1 occupation is the tracked bin value (N1,1)
        fnu2 = jnp.exp(interp_log_occ_dynamic(logf_nu, eps_nu, p2))        # (3,2,N1,Np2)
        for (e3lok, e3hik, yrule) in e3segs:
            E3, wE3 = _seg_nodes(e3bounds[e3lok], e3bounds[e3hik], xc, wc)  # (N1,Np2,Ne3)
            q3 = _qmom(E3, meps)
            p2e = p2[:, :, None]; e1e = e1b[:, :, None]; q1e = q1b[:, :, None]
            if yrule == "A":
                ylo, yhi = q1e - p2e, q1e + p2e
            elif yrule == "B":
                ylo, yhi = E3 + q3 - e1e - p2e, e1e + p2e - E3 + q3
            elif yrule == "C":
                ylo, yhi = e1e + p2e - E3 - q3, q1e + p2e
            else:  # "D"
                ylo, yhi = p2e - q1e, e1e + p2e - E3 + q3
            q1sq = e1e ** 2 - meps ** 2
            q3sq = E3 ** 2 - meps ** 2
            p4 = e1e + p2e - E3

            def integ(y):
                y2 = y ** 2
                q1s = q1sq[..., None]; q3s = q3sq[..., None]
                p2b = p2e[..., None]; p4b = p4[..., None]
                s = (e1e[..., None] + p2b) ** 2 - y2
                t0 = (2.0 * meps ** 2 - 2.0 * e1e[..., None] * E3[..., None]
                      + (q1s + y2 - p2b ** 2) * (y2 + q3s - p4b ** 2) / (2.0 * y2))
                val = 2.0 * (s - meps ** 2) ** 2 + 2.0 * s * t0
                # physical geometry masks |cos θ_1a| ≤ 1 and |cos θ_a| ≤ 1 (kill the spurious
                # 1/y² region the cut limits over-include as q1→0). Polynomial form, no division.
                m1a = (q1s + y2 - p2b ** 2) ** 2 <= 4.0 * q1s * y2
                ma = (y2 + q3s - p4b ** 2) ** 2 <= 4.0 * y2 * q3s
                return jnp.where(m1a & ma, val, 0.0)

            val = _yint_poly(integ, ylo, yhi, yc, yw)                       # (N1,Np2,Ne3)
            geom = val * wE3                                               # outer E3 weight
            # χ3 at q3, ν4 at p4 (3,2,...); F per flavor summed over cp
            f_chi3 = jnp.exp(interp_log_occ_dynamic(logf_chi, chi_eps, q3 * tratio, clamp_high=True))
            f_nu4 = jnp.exp(interp_log_occ_dynamic(logf_nu, eps_nu, jnp.maximum(p4, 0.0)))  # (3,2,N1,Np2,Ne3)
            f1b = f_chi1[:, None, None]                                    # (N1,1,1)
            f3b = f_chi3                                                   # (N1,Np2,Ne3)
            fnet, ffrs = _Ffac_scatter(f1b[None, None], fnu2[..., None], f3b[None, None], f_nu4)
            meas = (geom * wp2[:, :, None])[None, None]                    # (1,1,N1,Np2,Ne3)
            net = net + jnp.sum((meas * fnet).sum(axis=1), axis=(2, 3))    # sum cp then p2,E3 -> (3,N1)
            frs = frs + jnp.sum((meas * ffrs).sum(axis=1), axis=(2, 3))
    return net, frs


def _chiside_scatter_t(e1, q1, occ, eps_nu, logf_nu, f_chi1, logf_chi, chi_eps, meps, tratio,
                       xc, wc, xl, wl, yc, yw):
    """t-reduction (z=|q1−q3| cuts 47-51): integrand t²=((e1−E3)²−z²)². Returns (net,frs) (3,N1)."""
    N1 = e1.shape[0]
    net = jnp.zeros((3, N1)); frs = jnp.zeros((3, N1))
    # outer E3 in two regions: [meps, e1] (Legendre, zsign +) and [e1, ∞) (Laguerre, zsign −)
    e3regions = [("leg", jnp.broadcast_to(jnp.asarray(meps), e1.shape), e1, +1.0),
                 ("lag", e1, None, -1.0)]
    for (e3rule, e3lo, e3hi, zsign) in e3regions:
        if e3rule == "leg":
            E3, wE3 = _seg_nodes(e3lo, e3hi, xc, wc)    # (N1,Ne3)
        else:
            E3, wE3 = _lag_nodes(e3lo, xl, wl)
        q3 = _qmom(E3, meps)
        e1e = e1[:, None]; q1e = q1[:, None]            # (N1,1)
        etr1 = 0.5 * (-e1e + q1e + E3 + q3)
        elim1 = 0.5 * (-e1e + q1e + E3 - q3)
        elim2 = 0.5 * (-e1e - q1e + E3 + q3)
        plo_lin = elim1 if zsign > 0 else elim2
        # p2-segments: [plo_lin, etr1] (Legendre, "lin") and [etr1, ∞) (Laguerre, "fix")
        for (p2rule, p2lo, p2hi, zmode) in [("leg", plo_lin, etr1, "lin"), ("lag", etr1, None, "fix")]:
            if p2rule == "leg":
                p2, wp2 = _seg_nodes(jnp.maximum(p2lo, 0.0), p2hi, xc, wc)   # (N1,Ne3,Np2)
            else:
                p2, wp2 = _lag_nodes(p2lo, xl, wl)
            E3e = E3[:, :, None]; q3e = q3[:, :, None]
            e1f = e1e[:, :, None]; q1f = q1e[:, :, None]
            zlo = (q1f - q3e) if zsign > 0 else (q3e - q1f)
            zhi = (e1f + 2.0 * p2 - E3e) if zmode == "lin" else (q1f + q3e)

            def integ(z):
                return ((e1f[..., None] - E3e[..., None]) ** 2 - z ** 2) ** 2

            val = _yint_poly(integ, zlo, zhi, yc, yw)                       # (N1,Ne3,Np2)
            p4 = e1f + p2 - E3e
            f_chi3 = jnp.exp(interp_log_occ_dynamic(logf_chi, chi_eps, q3e * tratio, clamp_high=True))
            fnu2 = jnp.exp(interp_log_occ_dynamic(logf_nu, eps_nu, p2))             # (3,2,N1,Ne3,Np2)
            f_nu4 = jnp.exp(interp_log_occ_dynamic(logf_nu, eps_nu, jnp.maximum(p4, 0.0)))
            f1b = f_chi1[:, None, None]                                     # (N1,1,1)
            fnet, ffrs = _Ffac_scatter(f1b[None, None], fnu2, jnp.broadcast_to(f_chi3, p2.shape)[None, None], f_nu4)
            meas = (val * wE3[:, :, None] * wp2)[None, None]                # (1,1,N1,Ne3,Np2)
            net = net + jnp.sum((meas * fnet).sum(axis=1), axis=(2, 3))     # -> (3,N1)
            frs = frs + jnp.sum((meas * ffrs).sum(axis=1), axis=(2, 3))
    return net, frs


def chiside_scatter_rates(nu_grid, occ, f_chi, chi_grid, meps, tratio, Kflav,
                          n_gl=NCHI_GL, n_y=NCHI_Y, n_lag=NCHI_LAG):
    """χ-side df_χ/dt for νχ→νχ scattering [×tcmev⁵ → s⁻¹], tracked χ on chi_grid.

    occ : ν occupation (3,2,Nnu).  f_chi : (Nx,) per-dof χ occupation on chi_grid.
    meps = m_χ/T_cm, tratio = T_cm/T_χ, Kflav = (3,) g_χ²g_ν[α]²/m_Z′⁴ per flavor.
    Returns (net, frs), each (3, Nx) [per ν-flavor contribution to df_χ; sum over flavor for
    the total].  df_χ/dt = tcmev⁵·net.  χ scatters off both ν and ν̄ (both cp summed here).
    """
    q1 = chi_grid.abscissas / tratio                    # χ momentum in T_cm units, (Nx,)
    e1 = jnp.sqrt(q1 ** 2 + meps ** 2)
    logf_nu = jnp.log(occ)
    logf_chi = jnp.log(jnp.clip(f_chi, 1e-300, 1.0))
    chi_eps = chi_grid.abscissas
    eps_nu = nu_grid.abscissas
    xc, wc = legendre_canonical(n_gl)
    yc, yw = legendre_canonical(n_y)
    xl, wl = gauss_laguerre_plain(n_lag)

    ns, fs = _chiside_scatter_sst(e1, q1, occ, eps_nu, logf_nu, f_chi, logf_chi, chi_eps,
                                  meps, tratio, xc, wc, xl, wl, yc, yw)
    nt, ft = _chiside_scatter_t(e1, q1, occ, eps_nu, logf_nu, f_chi, logf_chi, chi_eps,
                                meps, tratio, xc, wc, xl, wl, yc, yw)
    # prefactor: (2K/ħ)·PREFAC_GEOM / (e1 q1).  Integrand already carries Σ/2/(2K) polynomial.
    coup = (2.0 * Kflav / HBAR) * PREFAC_GEOM            # (3,)
    good = (q1 > 0.0)
    q1safe = jnp.where(good, q1, 1.0)
    invfac = jnp.where(good, 1.0 / (e1 * q1safe), 0.0)   # 1/(e1 q1)
    scale = coup[:, None] * invfac[None, :]             # (3,Nx)
    return scale * (ns + nt), scale * (fs + ft)


# ----- χ-side ANNIHILATION χχ̄→νν̄ (y=|q1−p3| cuts 53-74; tracked χ1, partner χ̄2, finals ν3 ν̄4) -----

# Region lists (port of scripts_dark/chi_side_annih_analytic.py). Each region:
# (p3-rule, p3lo_key, p3hi_key, [(E2lo_key, E2hi_key, y-rule), ...]). "inf" upper ⇒ Laguerre.
_ANNIH_CASE1 = [
    ("leg", "Ec2", "Ec3", [("Et2", "inf", "a")]),
    ("leg", "Ec3", "Ec5", [("Et2", "Et1", "a"), ("Et1", "inf", "b")]),
    ("leg", "Ec5", "Ec4", [("m", "Et2", "c"), ("Et2", "Et1", "a"), ("Et1", "inf", "b")]),
    ("leg", "Ec4", "Ec6", [("m", "Et1", "c"), ("Et1", "Et2", "d"), ("Et2", "inf", "b")]),
    ("lag", "Ec6", None,   [("Et1", "Et2", "d"), ("Et2", "inf", "b")]),
]
_ANNIH_CASE2 = [
    ("leg", "Ec2", "Ec5", [("Et2", "inf", "a")]),
    ("leg", "Ec5", "Ec3", [("m", "Et2", "c"), ("Et2", "inf", "a")]),
    ("leg", "Ec3", "Ec4", [("m", "Et2", "c"), ("Et2", "Et1", "a"), ("Et1", "inf", "b")]),
    ("leg", "Ec4", "Ec6", [("m", "Et1", "c"), ("Et1", "Et2", "d"), ("Et2", "inf", "b")]),
    ("lag", "Ec6", None,   [("Et1", "Et2", "d"), ("Et2", "inf", "b")]),
]


def _annih_case(e1, q1, occ, eps_nu, logf_nu, f_chi1, logf_chi, chi_eps, meps, tratio,
                regions, casemask, xc, wc, xl, wl, yc, yw):
    """One annihilation case (region list) masked by ``casemask`` (N1,). Returns (net,frs) (3,N1)."""
    N1 = e1.shape[0]
    net = jnp.zeros((3, N1)); frs = jnp.zeros((3, N1))
    cm = casemask.astype(e1.dtype)                       # (N1,)
    Ec2 = 0.5 * (e1 - q1); Ec3 = 0.5 * (e1 + q1); Ec4 = e1
    Ec5 = 0.5 * (e1 + meps - q1); Ec6 = 0.5 * (e1 + meps + q1)
    p3keys = {"Ec2": Ec2, "Ec3": Ec3, "Ec4": Ec4, "Ec5": Ec5, "Ec6": Ec6}
    for (p3rule, plok, phik, subs) in regions:
        plo = p3keys[plok]
        if p3rule == "leg":
            p3, wp3 = _seg_nodes(plo, p3keys[phik], xc, wc)     # (N1,Np3)
        else:
            p3, wp3 = _lag_nodes(plo, xl, wl)
        e1b = e1[:, None]; q1b = q1[:, None]
        d1 = 2.0 * p3 - e1b - q1b
        d2 = 2.0 * p3 - e1b + q1b
        Et1 = 0.5 * (d1 + meps ** 2 / jnp.where(d1 != 0.0, d1, 1.0))
        Et2 = 0.5 * (d2 + meps ** 2 / jnp.where(d2 != 0.0, d2, 1.0))
        e2keys = {"m": jnp.broadcast_to(jnp.asarray(meps), p3.shape), "Et1": Et1, "Et2": Et2}
        for (e2lok, e2hik, yrule) in subs:
            e2lo = e2keys[e2lok]
            if e2hik == "inf":
                E2, wE2 = _lag_nodes(e2lo, xl, wl)                # (N1,Np3,Ne2)
            else:
                E2, wE2 = _seg_nodes(e2lo, e2keys[e2hik], xc, wc)
            q2 = _qmom(E2, meps)
            p3e = p3[:, :, None]; q1e = q1b[:, :, None]; e1e = e1b[:, :, None]
            if yrule == "a":
                ylo, yhi = e1e - p3e + E2 - q2, q1e + p3e
            elif yrule == "b":
                ylo, yhi = jnp.broadcast_to(p3e - q1e, E2.shape), jnp.broadcast_to(q1e + p3e, E2.shape)
            elif yrule == "c":
                ylo, yhi = e1e - p3e + E2 - q2, e1e - p3e + E2 + q2
            else:  # "d"
                ylo, yhi = jnp.broadcast_to(p3e - q1e, E2.shape), e1e - p3e + E2 + q2
            q1sq = e1e ** 2 - meps ** 2
            q2sq = E2 ** 2 - meps ** 2
            p4 = e1e + E2 - p3e

            def integ(y):
                y2 = y ** 2
                q1s = q1sq[..., None]; q2s = q2sq[..., None]
                p3b = p3e[..., None]; p4b = p4[..., None]; e1b2 = e1e[..., None]
                M = (e1b2 - p3b) ** 2 - y2                        # = t (clean)
                Ju = M ** 2 - 4.0 * meps ** 2 * M + 6.0 * meps ** 4
                q1bh = (p3b ** 2 - q1s - y2) / (2.0 * y)          # q1·b̂   (p3 massless ⇒ p3 = |p3|)
                p4bh = (q2s - y2 - p4b ** 2) / (2.0 * y)          # p4·b̂
                u0 = meps ** 2 - 2.0 * e1b2 * p4b + 2.0 * q1bh * p4bh
                u1sq = 4.0 * jnp.maximum(q1s - q1bh ** 2, 0.0) * jnp.maximum(p4b ** 2 - p4bh ** 2, 0.0)
                Jt = u0 ** 2 + 0.5 * u1sq - 4.0 * meps ** 2 * u0
                mask = ((p3b ** 2 - q1s - y2) ** 2 <= 4.0 * q1s * y2) & \
                       ((q2s - y2 - p4b ** 2) ** 2 <= 4.0 * p4b ** 2 * y2)
                return jnp.where(mask, Ju + Jt, 0.0)

            val = _yint_poly(integ, ylo, yhi, yc, yw)              # (N1,Np3,Ne2)
            # F: f1=fχ(q1)[bin], f2=fχ(q2), f3=fν(p3), f4=fν̄(p4); per flavor (ν=cp0, ν̄=cp1)
            f_chi2 = jnp.exp(interp_log_occ_dynamic(logf_chi, chi_eps, q2 * tratio, clamp_high=True))
            f_nu3 = jnp.exp(interp_log_occ_dynamic(logf_nu[:, 0, :], eps_nu, jnp.broadcast_to(p3e, E2.shape)))
            f_nu4 = jnp.exp(interp_log_occ_dynamic(logf_nu[:, 1, :], eps_nu, jnp.maximum(p4, 0.0)))
            f1b = f_chi1[:, None, None]                           # (N1,1,1)
            forf = (1.0 - f1b)[None] * (1.0 - f_chi2)[None] * f_nu3 * f_nu4
            revf = f1b[None] * f_chi2[None] * (1.0 - f_nu3) * (1.0 - f_nu4)
            meas = (val * wp3[:, :, None] * wE2)[None]            # (1,N1,Np3,Ne2)
            net = net + (cm[None, :, None, None] * meas * (forf - revf)).sum(axis=(2, 3))
            frs = frs + (cm[None, :, None, None] * meas * (forf + revf)).sum(axis=(2, 3))
    return net, frs


def chiside_annih_rates(nu_grid, occ, f_chi, chi_grid, meps, tratio, Kflav,
                        n_gl=NCHI_GL, n_y=NCHI_Y, n_lag=NCHI_LAG):
    """χ-side df_χ/dt for χχ̄→νν̄ annihilation [×tcmev⁵ → s⁻¹], tracked χ on chi_grid.

    Returns (net, frs), each (3, Nx) [per ν-flavor; sum over flavor for the total]. The two
    angular cases (E1/m_χ ≷ √5/2) are computed and masked per bin. Partner χ̄ uses f_chi (CP).
    """
    q1 = chi_grid.abscissas / tratio
    e1 = jnp.sqrt(q1 ** 2 + meps ** 2)
    logf_nu = jnp.log(occ)
    logf_chi = jnp.log(jnp.clip(f_chi, 1e-300, 1.0))
    chi_eps = chi_grid.abscissas
    eps_nu = nu_grid.abscissas
    xc, wc = legendre_canonical(n_gl)
    yc, yw = legendre_canonical(n_y)
    xl, wl = gauss_laguerre_plain(n_lag)

    sqrt5_2 = 0.5 * jnp.sqrt(5.0)
    mask1 = e1 < sqrt5_2 * meps                          # case 1
    n1, f1 = _annih_case(e1, q1, occ, eps_nu, logf_nu, f_chi, logf_chi, chi_eps, meps, tratio,
                         _ANNIH_CASE1, mask1, xc, wc, xl, wl, yc, yw)
    n2, f2 = _annih_case(e1, q1, occ, eps_nu, logf_nu, f_chi, logf_chi, chi_eps, meps, tratio,
                         _ANNIH_CASE2, ~mask1, xc, wc, xl, wl, yc, yw)
    coup = (2.0 * Kflav / HBAR) * PREFAC_GEOM
    good = (q1 > 0.0)
    q1safe = jnp.where(good, q1, 1.0)
    invfac = jnp.where(good, 1.0 / (e1 * q1safe), 0.0)
    scale = coup[:, None] * invfac[None, :]
    return scale * (n1 + n2), scale * (f1 + f2)


# ----- combined S2 rates + the precision-gate noise floor (for the coupled solve) -----------
#
# The S2 2↔2 kernels are the SM weak-kernel family, so near equilibrium net = forward − reverse
# cancels only to a ~1e-5·frs quadrature residual (not the ~1e-16 of the smooth 1↔2 decay kernels).
# When the coupling keeps χ in equilibrium (frs ≫ H), that residual × T_cm⁵ swamps H and stalls the
# solver — so unlike the decay legs, the S2 legs must be gated. We reuse the SM gate mechanism
# (assemble.assemble_rhs): a per-bin equilibrium noise floor ``prec`` and the test_ratio =
# |net|/frs/prec ≥ nuratiotol decision, applied SEPARATELY from the SM gate (folding the dark frs
# into the SM denominator flips ~600 SM gates).

PREC_FLOOR_CHI = 1.0e-13          # floor where Σfrs ≤ 0 (mirrors assemble.PREC_FLOOR)


def s2_rates(grid, occ, f_chi, chi_grid, meps, tratio, Kflav,
             n_gl=NCHI_GL, n_y=NCHI_Y, n_lag=NCHI_LAG, n_gl_nu=5, n_lag_nu=5):
    """Combined S2 ν-side and χ-side (net, frs) for the coupled solve [×tcmev⁵ → s⁻¹].

    Folds the physical multiplicities: ν scatters off χ AND χ̄ (×2, identical |M|² by C-inv) +
    annihilates with ν̄; the χ-side sums over ν flavor (the kernels already sum cp / the pair).
    Returns (net_nu (3,2,N), frs_nu (3,2,N), net_chi (Nx,), frs_chi (Nx,)). The χ-side node
    counts (n_gl/n_y/n_lag) may be reduced for the coupled solve to shrink the XLA graph
    (the standalone validation uses the 24/16/24 defaults).
    """
    ns_sc, fs_sc = nuside_scatter_rates(grid, occ, f_chi, chi_grid, meps, tratio, Kflav,
                                        n_gl=n_gl_nu, n_lag=n_lag_nu)
    ns_an, fs_an = nuside_annih_rates(grid, occ, f_chi, chi_grid, meps, tratio, Kflav,
                                      n_gl=n_gl_nu, n_lag=n_lag_nu)
    net_nu = 2.0 * ns_sc + ns_an
    frs_nu = 2.0 * fs_sc + fs_an
    cs_sc, cfs_sc = chiside_scatter_rates(grid, occ, f_chi, chi_grid, meps, tratio, Kflav,
                                          n_gl=n_gl, n_y=n_y, n_lag=n_lag)
    cs_an, cfs_an = chiside_annih_rates(grid, occ, f_chi, chi_grid, meps, tratio, Kflav,
                                        n_gl=n_gl, n_y=n_y, n_lag=n_lag)
    net_chi = jnp.sum(cs_sc + cs_an, axis=0)
    frs_chi = jnp.sum(cfs_sc + cfs_an, axis=0)
    return net_nu, frs_nu, net_chi, frs_chi


def s2_prec(grid, occ_eq, f_chi_eq, chi_grid, meps, tratio, Kflav, **nodes):
    """S2 equilibrium noise floor (prec_nu (3,2,N), prec_chi (Nx,)) = |net|/frs at equilibrium.

    Evaluate at ν, χ at a COMMON temperature (true detailed balance) → net is pure quadrature
    residual; the floor normalizes the coupled-solve gate exactly like assemble.compute_prec_trans
    does for the SM kernels. A static precompute at the IC equilibrium (the noise floor is mildly
    meps-dependent but the gate only matters at high T where χ is coupled, which is the IC).
    """
    net_nu, frs_nu, net_chi, frs_chi = s2_rates(grid, occ_eq, f_chi_eq, chi_grid, meps, tratio,
                                                Kflav, **nodes)
    prec_nu = jnp.where(frs_nu > 0.0, jnp.abs(net_nu) / jnp.where(frs_nu > 0.0, frs_nu, 1.0),
                        PREC_FLOOR_CHI)
    prec_chi = jnp.where(frs_chi > 0.0, jnp.abs(net_chi) / jnp.where(frs_chi > 0.0, frs_chi, 1.0),
                         PREC_FLOOR_CHI)
    return jnp.maximum(prec_nu, PREC_FLOOR_CHI), jnp.maximum(prec_chi, PREC_FLOOR_CHI)
