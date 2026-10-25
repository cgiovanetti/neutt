"""Density-matrix annihilation ν(p₁) + ν̄(p₄) ⇌ e⁻ + e⁺: the matrix version of ``nuepma``.

A port of qburst's matrix annihilation collision.  The integration scaffolding (``nuepma``'s
``_REGIMES`` tile table and pcase/ecut/etrans regimes, the ``_L1``/``_L2`` antiderivatives,
``_topbot_pma``, the nodes and the measure) is flavour-independent and shared with the classical
kernel.  Only the coupling and the blocking change: the scalar ``forf``/``revf`` factors become
anticommutators.

Per (cp n, p₁ bin), summed over the L/R index l and the lepton parity nn (nn2 the opposite parity),
in qburst's form::

    for_matrix = Y_l · (I − ρ̄(p₄)) · weightin(l, nn, n)          # all diagonal in flavour
    rev_matrix = Y_l ·      ρ̄(p₄)  · weightin(l, nn, n)
    forfact += c(nn,2)·c(nn2,3) · {for_matrix, I − ρ(p₁)}
    revfact += (1−c(nn,2))(1−c(nn2,3)) · {rev_matrix, ρ(p₁)}
    net = forfact − revfact,   frs = forfact + revfact

with c(nn, ·) the e∓ occupations at E_out (2) and E_in (3), r1 = I1, r2 = meps²·I2 and::

    weightin(l=L) = 2·Y_L·r1·mod(n+nn, 2)   + Y_R·r2
    weightin(l=R) = 2·Y_R·r1·mod(n+nn+1, 2) + Y_L·r2
    Y_L = diag(s + ½, s − ½, s − ½),   Y_R = s·1,   s = sin²θ_W

Y_l and weightin are diagonal, so ``(Y_l·M·weightin)_ab = y_a M_ab w_b`` and the quadrature is
summed before the anticommutators (as in ``collide_matrix.assemble_from_phi``).  The prefactor is
qburst's ``2 G_F ħ/(2π)³`` on the bare Y_l/weightin; on a diagonal ρ, where {A, B} = 2AB, the kernel
reduces per flavour to the classical ``G_F ħ/16/(2π)³ · term1·(I1 + c2·meps²·I2)`` form.

For the ν̄ (cp 1) BURST transposes ρ(p₁) and ρ̄(p₄) and transposes the result back; so does this
kernel (a no-op on the diagonal).  dρ/dt|coll = T_cm⁵·net.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ...constants import GFHBAR, PI, SIN2THW
from ...grid import EpsGrid
from ...grid import legendre_canonical, gauss_laguerre_plain
from .coherent import hermitize
from .collide_matrix import interp_partner_rho
from ..nuepma import (
    _fd,
    _safe_sqrt_diff,
    _L1,
    _L2,
    _topbot_pma,
    _REGIMES,
    _LAG_TAIL_GAP_PMA,
)


def _y_l_matrices(sin2thw: float = SIN2THW):
    """``YL_YR[l, a, b]``, (2, 3, 3): l=0 Y_L = diag(s+½, s−½, s−½), l=1 Y_R = s·1."""
    s = sin2thw
    YL = np.diag([s + 0.5, s - 0.5, s - 0.5]).astype(np.float64)
    YR = np.diag([s, s, s]).astype(np.float64)
    return jnp.asarray(np.stack([YL, YR]))  # (2,3,3)


def _anticomm(X, Y):
    """Anticommutator {X, Y} = XY + YX for batched (..., 3, 3) matrices."""
    XY = jnp.matmul(X, Y)
    return XY + jnp.matmul(Y, X)


def _nuepma_matrix_core(grid: EpsGrid, rho, meps, tcmpl, phie, n_gl: int = 5,
                        n_lag: int = 5, sin2thw: float = SIN2THW,
                        fd_cap_logC: float = float("inf"),
                        fd_decay: bool = False, monotone_read: bool = False,
                        lag_tail_fix: bool = True,
                        lag_tail_gap: float = _LAG_TAIL_GAP_PMA,
                        interp_mode: str = "fraction"):
    """Density-matrix νν̄ → e⁺e⁻: ``(net, frs, A_gain, B_loss)``, each (2, N, 3, 3) complex.

    Parameters
    ----------
    rho : (2, N, 3, 3) complex
        ``rho[cp, bin, a, b]``, cp 0 = ν, 1 = ν̄.
    meps, tcmpl, phie : float
        m_e/T_cm, T_cm/T_pl and the electron degeneracy μ_e/T_pl (may be traced).
    n_gl, n_lag : int
        Gauss–Legendre / Gauss–Laguerre points (BURST: 5 each).
    fd_cap_logC, fd_decay, monotone_read
        Tail closures of the partner read (``neutt.interp``), as in the classical kernel.
    lag_tail_fix, lag_tail_gap
        As in ``nuepma.nuepma_rates``: R2's ``genout`` outer segment switches from Legendre to
        Laguerre at width ``lag_tail_gap`` instead of 200, and R2's ``[ecut2, ∞)`` "a" tail is
        dropped on those bins (the Laguerre tail already covers it).  False is bit-faithful to BURST.
    interp_mode : {"fraction", "linear", "lagrange"}
        Off-grid coherence read (``collide_matrix.interp_partner_rho``); "fraction" is bounded.

    Returns
    -------
    net, frs, A_gain, B_loss
        ``net = forfact − revfact = A_gain − {B_loss, ρ}`` and ``frs = forfact + revfact``, all
        hermitized.  On a diagonal ρ, ``A_gain`` and ``2·B_loss`` are the classical A and B of
        df/dt = A − B·f, so ``nonadiabatic.lstable_collision_step`` reduces to ``hybrid.exp_step``.
        ``B_loss`` is positive-semidefinite on a physical state except at the lowest bin below
        T_cm ≈ 0.1 MeV (``collide_matrix.assemble_from_phi``).  Multiply by T_cm⁵ for s⁻¹.
    """
    rho = jnp.asarray(rho).astype(jnp.complex128)
    eps = grid.abscissas
    p1 = eps                                   # (N,)
    N = eps.shape[0]
    cdtype = rho.dtype
    rdtype = eps.dtype

    YL_YR = _y_l_matrices(sin2thw).astype(cdtype)   # (2,3,3): l=0 Y_L, l=1 Y_R
    YL_YR_diag = jnp.diagonal(YL_YR, axis1=-2, axis2=-1)   # (2,3)
    I3 = jnp.eye(3, dtype=cdtype)

    # qburst's prefactor
    prefac = 2.0 * GFHBAR / (2.0 * PI) ** 3

    tleg, wleg = legendre_canonical(n_gl)
    xlag, wlag = gauss_laguerre_plain(n_lag)

    meps_s = jnp.asarray(meps, rdtype)
    binmask = (p1 > 0.0).astype(rdtype)
    p1sq = jnp.where(p1 > 0.0, p1**2, 1.0)

    # outer-integral cut energies
    ecut1 = p1 + meps**2 / jnp.where(p1 > 0.0, 4.0 * p1, 1.0)
    den2 = 2.0 * p1 - meps
    ecut2 = 0.5 * (2.0 * p1 - meps + meps**2 / jnp.where(den2 != 0.0, den2, 1.0))
    bounds_out = {"meps": jnp.broadcast_to(meps_s, (N,)), "ecut1": ecut1, "ecut2": ecut2,
                  "p1": p1}

    pcase1 = meps / 2.0
    pcase2 = meps * (1.0 + jnp.sqrt(5.0)) / 4.0
    pcase3 = meps
    regmasks = {
        "R1": (p1 < pcase1).astype(rdtype),
        "R2": ((p1 >= pcase1) & (p1 < pcase2)).astype(rdtype),
        "R3": ((p1 >= pcase2) & (p1 < pcase3)).astype(rdtype),
        "R4": (p1 >= pcase3).astype(rdtype),
    }
    clamp_e2 = {"R1": False, "R2": True, "R3": False, "R4": False}

    # genout Legendre→Laguerre switch and the R2 double-count guard, as in nuepma.nuepma_rates
    thr_out = lag_tail_gap if lag_tail_fix else 200.0
    genout_lag = (((ecut2 - ecut1) >= thr_out).astype(rdtype) if lag_tail_fix
                  else jnp.zeros((N,), rdtype))
    log_kw = dict(fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                  monotone_read=monotone_read)

    # Local ρ(p1) and the partner (opposite cp), with BURST's ν̄ transpose: cp 0 pairs ρ_ν with ρ_ν̄,
    # cp 1 pairs ρ_ν̄ᵀ with ρ_νᵀ.  The partner is put matrix-axes-first for the interpolator.
    rho1_T = jnp.swapaxes(rho, -1, -2)
    rho1_cp = jnp.stack([rho[0], rho1_T[1]])                     # (2, N, 3, 3)
    rho_partner_cp = jnp.stack([rho[1], rho1_T[0]])              # (2, N, 3, 3)  matrix axes last
    rho_partner_ab = jnp.transpose(rho_partner_cp, (0, 2, 3, 1))  # (2, 3, 3, N) matrix axes first
    logf_diag_partner = jnp.log(
        jnp.real(rho_partner_cp[:, :, jnp.arange(3), jnp.arange(3)])).transpose(0, 2, 1)  # (2,3,N)

    net = jnp.zeros((2, N, 3, 3), dtype=cdtype)
    frs = jnp.zeros((2, N, 3, 3), dtype=cdtype)
    A_gain = jnp.zeros((2, N, 3, 3), dtype=cdtype)
    B_loss = jnp.zeros((2, N, 3, 3), dtype=cdtype)

    def _outer_nodes(rule, a, b):
        if rule == "leg":
            mid = 0.5 * (a + b); half = 0.5 * (b - a)
            E = mid[:, None] + half[:, None] * tleg[None, :]
            w = wleg[None, :] * half[:, None]
        else:
            E = a[:, None] + xlag[None, :]
            w = jnp.broadcast_to(wlag[None, :], E.shape)
        return E, w

    def _inner_nodes(rule, a, b):
        if rule == "leg":
            mid = 0.5 * (a + b); half = 0.5 * (b - a)
            E = mid[:, :, None] + half[:, :, None] * tleg[None, None, :]
            w = wleg[None, None, :] * half[:, :, None]
        else:
            E = a[:, :, None] + xlag[None, None, :]
            w = jnp.broadcast_to(wlag[None, None, :], E.shape)
        return E, w

    # Y_l parity gate: mod(n+nn, 2) for l=0 (Y_L), mod(n+nn+1, 2) for l=1 (Y_R), with Fortran's
    # 1-based n = cp+1 and nn = parity+1.
    def _gate(cp0, nn0, l):
        n = cp0 + 1
        nn = nn0 + 1
        if l == 0:
            return float((n + nn) % 2)
        return float((n + nn + 1) % 2)

    for rname, segments in _REGIMES.items():
        regmask = regmasks[rname] * binmask                          # (N,)
        for (out_rule, a_out_k, b_out_k, inner_tiles) in segments:
            a_out = bounds_out[a_out_k]
            if out_rule == "genout":
                b_out = bounds_out[b_out_k]
                out_variants = [("leg", a_out, b_out, (b_out - a_out < thr_out).astype(rdtype)),
                                ("lag", a_out, None, (b_out - a_out >= thr_out).astype(rdtype))]
            elif out_rule == "leg":
                out_variants = [("leg", a_out, bounds_out[b_out_k], jnp.ones((N,)))]
            else:
                out_variants = [("lag", a_out, None, jnp.ones((N,)))]

            for (orule, oa, ob, out_mask) in out_variants:
                eout, eoutw = _outer_nodes(orule, oa, ob)            # (N,J)
                eout = jnp.maximum(eout, meps_s)
                qout = _safe_sqrt_diff(eout**2 - meps_s**2)
                p1c = p1[:, None]
                d1 = 2.0 * p1c - eout - qout
                etrans1 = 0.5 * (2.0 * p1c - eout - qout + meps**2 / jnp.where(d1 != 0.0, d1, 1.0))
                d2 = 2.0 * p1c - eout + qout
                etrans2_raw = 0.5 * (2.0 * p1c - eout + qout + meps**2 / jnp.where(d2 != 0.0, d2, 1.0))
                etrans2 = jnp.maximum(meps_s, etrans2_raw) if clamp_e2[rname] else etrans2_raw
                C = (p1c - eout) ** 2 - meps**2                      # (N,J)
                kinemfact1 = eoutw / p1sq[:, None]                   # (N,J)
                femin_out = _fd(eout, tcmpl, phie, 1.0)              # e- at eout (N,J)
                fepl_out = _fd(eout, tcmpl, phie, -1.0)              # e+ at eout (N,J)
                bounds_in = {"meps": jnp.broadcast_to(meps_s, eout.shape),
                             "etrans1": etrans1, "etrans2": etrans2}

                for (in_rule, a_in_k, b_in_k, kind, posguard) in inner_tiles:
                    a_in = bounds_in[a_in_k]
                    if in_rule == "genin":
                        b_in = bounds_in[b_in_k]
                        gap = b_in - a_in
                        in_variants = [
                            ("leg", a_in, b_in, ((gap >= 0.0) & (gap < 200.0)).astype(rdtype)),
                            ("lag", a_in, None, ((gap < 0.0) | (gap >= 200.0)).astype(rdtype))]
                    elif in_rule == "leg":
                        in_variants = [("leg", a_in, bounds_in[b_in_k], jnp.ones(eout.shape))]
                    else:
                        m_in = (bounds_in[posguard] > 0.0).astype(rdtype) if posguard else jnp.ones(eout.shape)
                        in_variants = [("lag", a_in, None, m_in)]

                    # double-count guard: R2's [ecut2,∞) "a" tail is already covered by the
                    # genout Laguerre tail [ecut1,∞) for the wide bins, so drop it there.
                    if lag_tail_fix and rname == "R2" and a_out_k == "ecut2" and kind == "a":
                        in_variants = [(ir, ia, ib, m * (1.0 - genout_lag)[:, None])
                                       for (ir, ia, ib, m) in in_variants]

                    for (irule, ia, ib, in_mask) in in_variants:
                        ein, einw = _inner_nodes(irule, ia, ib)      # (N,J,K)
                        ein = jnp.maximum(ein, meps_s)
                        qin = _safe_sqrt_diff(ein**2 - meps_s**2)
                        top, bot = _topbot_pma(kind, p1[:, None, None], qout[:, :, None],
                                               eout[:, :, None], ein, qin)          # (N,J,K)
                        Ce = C[:, :, None]
                        r1 = _L1(top, Ce) - _L1(bot, Ce)             # I1   (N,J,K)
                        r2 = meps**2 * (_L2(top, Ce) - _L2(bot, Ce))  # meps²·I2 (rint_l_2)

                        femin_in = _fd(ein, tcmpl, phie, 1.0)        # (N,J,K)
                        fepl_in = _fd(ein, tcmpl, phie, -1.0)
                        # cleparray[parity, energy]: parity 0 = e-, 1 = e+; energy 2 = eout, 3 = ein
                        clep2 = jnp.stack([femin_out[:, :, None] + 0.0 * ein,
                                           fepl_out[:, :, None] + 0.0 * ein])    # (parity,N,J,K) at eout
                        clep3 = jnp.stack([femin_in, fepl_in])                   # (parity,N,J,K) at ein

                        p4 = jnp.maximum(eout[:, :, None] + ein - p1[:, None, None], 0.0)  # (N,J,K)

                        # measure, as in the classical kernel
                        meas = (prefac * (regmask * out_mask)[:, None, None]
                                * (in_mask * kinemfact1)[:, :, None] * einw)      # (N,J,K)

                        # Node-sum Φ_for/Φ_rev first; the anticommutators with ρ1 come once per
                        # (cp, bin).  Y_l·M·weightin is the elementwise y_a M_ab w_b.
                        measb = meas[..., None, None]                 # (N,J,K,1,1)
                        for cp0 in range(2):
                            rho4 = interp_partner_rho(
                                rho_partner_ab[cp0], eps, p4, logf_diag_partner[cp0],
                                mode=interp_mode, **log_kw)                            # (N,J,K,3,3)
                            Iminus4 = I3 - rho4                       # (N,J,K,3,3)

                            Phi_for_n = jnp.zeros(p4.shape + (3, 3), dtype=cdtype)
                            Phi_rev_n = jnp.zeros(p4.shape + (3, 3), dtype=cdtype)
                            for nn0 in range(2):       # lepton parity
                                nn2 = 1 - nn0          # opposite parity (Fortran nn2 = 2 − nn/2)
                                cfor = (clep2[nn0] * clep3[nn2])[..., None, None]        # (N,J,K,1,1)
                                crev = ((1.0 - clep2[nn0]) * (1.0 - clep3[nn2]))[..., None, None]
                                for l in range(2):     # L/R
                                    ylv = YL_YR_diag[l]                # (3,) = diag(Y_l)
                                    g = _gate(cp0, nn0, l)
                                    # diag(weightin): l=0: 2 Y_L r1 g + Y_R r2;  l=1: 2 Y_R r1 g + Y_L r2
                                    yav, ybv = ((YL_YR_diag[0], YL_YR_diag[1]) if l == 0
                                                else (YL_YR_diag[1], YL_YR_diag[0]))
                                    wdv = (2.0 * yav * r1[..., None] * g
                                           + ybv * r2[..., None])                 # (N,J,K,3) diag
                                    pair = ylv[:, None] * wdv[..., None, :]        # (N,J,K,3,3) y_a w_b
                                    Phi_for_n = Phi_for_n + cfor * pair * Iminus4
                                    Phi_rev_n = Phi_rev_n + crev * pair * rho4

                            Phi_for = jnp.sum(measb * Phi_for_n, axis=(1, 2))    # (N,3,3)
                            Phi_rev = jnp.sum(measb * Phi_rev_n, axis=(1, 2))
                            # assemble_from_phi, inlined
                            rho1 = rho1_cp[cp0]                       # (N,3,3) local p1-bin matrix
                            A_cp = 2.0 * Phi_for
                            B_cp = Phi_for + Phi_rev
                            net_cp = A_cp - _anticomm(B_cp, rho1)
                            frs_cp = A_cp - _anticomm(Phi_for - Phi_rev, rho1)
                            # ν̄: transpose back
                            if cp0 == 1:
                                net_cp = jnp.swapaxes(net_cp, -1, -2)
                                frs_cp = jnp.swapaxes(frs_cp, -1, -2)
                                A_cp = jnp.swapaxes(A_cp, -1, -2)
                                B_cp = jnp.swapaxes(B_cp, -1, -2)
                            net = net.at[cp0].add(net_cp)
                            frs = frs.at[cp0].add(frs_cp)
                            A_gain = A_gain.at[cp0].add(A_cp)
                            B_loss = B_loss.at[cp0].add(B_cp)

    net = hermitize(net)
    frs = hermitize(frs)
    A_gain = hermitize(A_gain)
    B_loss = hermitize(B_loss)
    return net, frs, A_gain, B_loss


def nuepma_matrix_rates(grid: EpsGrid, rho, meps, tcmpl, phie, **kw):
    """``(net, frs)`` of :func:`_nuepma_matrix_core`, each (2, N, 3, 3); dρ/dt|coll = T_cm⁵·net."""
    net, frs, _, _ = _nuepma_matrix_core(grid, rho, meps, tcmpl, phie, **kw)
    return net, frs


def nuepma_matrix_affine(grid: EpsGrid, rho, meps, tcmpl, phie, **kw):
    """``(A_gain, B_loss)`` with net = A_gain − {B_loss, ρ}, per T_cm⁵ like ``net``."""
    _, _, A_gain, B_loss = _nuepma_matrix_core(grid, rho, meps, tcmpl, phie, **kw)
    return A_gain, B_loss


def nuepma_matrix_dfdt(grid, rho, meps, tcmpl, phie, t_cm_mev, **kw):
    """dρ/dt|coll = T_cm⁵·net from νν̄ ⇌ e⁺e⁻ [s⁻¹], (2, N, 3, 3) complex."""
    net, _ = nuepma_matrix_rates(grid, rho, meps, tcmpl, phie, **kw)
    return t_cm_mev**5 * net


def nuepma_matrix_affine_dfdt(grid, rho, meps, tcmpl, phie, t_cm_mev, **kw):
    """``(A_rate, B_rate) = T_cm⁵·(A_gain, B_loss)`` [s⁻¹], with dρ/dt|coll = A_rate − {B_rate, ρ}.

    The inputs of ``nonadiabatic.lstable_collision_step``.
    """
    A_gain, B_loss = nuepma_matrix_affine(grid, rho, meps, tcmpl, phie, **kw)
    scale = t_cm_mev ** 5
    return scale * A_gain, scale * B_loss
