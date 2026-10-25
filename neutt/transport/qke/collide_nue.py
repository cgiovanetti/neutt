"""Density-matrix ν–e± scattering: the matrix versions of ``nuer1`` (R1) and ``nuer2`` (R2).

.. code::

    dρ(p₁)/dt|coll = T_cm⁵·(A_gain − {B_loss, ρ(p₁)})

All kinematics come from the classical modules: the tile tables (``_E2_SEGMENTS``/
``_E2_SEGMENTS_FIXED``, ``nuer2._regimes``), the antiderivatives (``_R1``/``_R2``,
``_R2m1``/``_R2m2``), the inner-limit maps (``_topbot``, ``_topbot_r2``), the cut energies, the
nodes, the measure and ``lag_tail_fix``.  The new ingredient is the coupling promotion
(``collide_matrix.pair_coupling_tensors``):

.. code::

    term1[nn, m, n]     →  term1_pair[nn, a, b, n]      (g_{L,m}² → g_{L,a} g_{L,b})
    term1·c2[nn, m, n]  →  term2_pair[nn, a, b, n]      (g_{L,m} → ½(g_{L,a} + g_{L,b}))

``Y_L``, ``Y_R`` and the angular weights are diagonal in flavour, so the integrand is the
elementwise ``W_ab·(ρ₄)_ab`` and the 3×3 algebra is one anticommutator per (cp, bin), after the node
sum.

Blocking.  For ν(p₁) + e(E₂) → e(E₃) + ν(p₄) the classical factors are
``forf = (1 − f₁)(1 − f_{e2}) f_{e3} f₄`` and ``revf = f₁ f_{e2}(1 − f_{e3})(1 − f₄)``, so the gain
carries ρ₄ and the loss I − ρ₄ (the reverse of annihilation), and both leptons carry the same parity
nn.  With

.. code::

    Φ_for = Σ_{nodes, nn} meas·(1 − f_{e2}) f_{e3}·(W^{(nn)} ⊙ ρ₄)
    Φ_rev = Σ_{nodes, nn} meas·f_{e2}(1 − f_{e3})·(W^{(nn)} ⊙ (I − ρ₄))

the assembly is ``collide_matrix.assemble_from_phi``.

Normalisation.  ``W^{(nn)} = ½(term1_pair·I1 + term2_pair·meps²·I2)`` with the classical prefactor
``G_F ħ/16/(2π)³``: the ½ cancels the anticommutator's 2, so on a diagonal ρ ``2Φ_for,aa`` is the
classical ``term1·(I1 + c2·meps²·I2)`` gain term for term.

cp.  Scattering keeps cp, so the partner ρ₄ has the cp of ρ(p₁).  BURST's ν̄ transpose is a no-op:
W is symmetric in (a, b), so ``W ⊙ ρᵀ = (W ⊙ ρ)ᵀ``, and ``{Xᵀ, Yᵀ} = {X, Y}ᵀ``
(``tests/test_qke_collide_nue.py::test_transpose_invariance``).

The partner is read with ``collide_matrix.interp_partner_rho`` (default ``"fraction"``: the bounded
coherence fraction on the classical log-space diagonal).
"""

from __future__ import annotations

import jax.numpy as jnp

from ...constants import GFHBAR, PI, SIN2THW
from ...grid import EpsGrid
from ...grid import legendre_canonical, gauss_laguerre_plain
from .collide_matrix import (
    assemble_from_phi,
    interp_partner_rho,
    pair_coupling_tensors,
)
from .coherent import hermitize
from ..nuer1 import (
    _E2_SEGMENTS,
    _E2_SEGMENTS_FIXED,
    _LAG_TAIL_GAP,
    _R1,
    _R2,
    _fd,
    _safe_sqrt_diff,
    _topbot,
)
from ..nuer2 import (
    _R2m1,
    _R2m2,
    _regime_masks as _r2_regime_masks,
    _regimes as _r2_regimes,
    _sel_mask as _r2_sel_mask,
    _topbot_r2,
)

_DIAG3 = jnp.arange(3)

#: Classical scattering prefactor; pairs with the ½ in :func:`_pair_weight`.
_PREFAC = GFHBAR / 16.0 / (2.0 * PI) ** 3


def _unpack_rho(rho):
    """``(rho, rho_ab, logf_diag)``: ρ as complex128, ρ matrix-axes-first (2, 3, 3, N), log diag (2, 3, N)."""
    rho = jnp.asarray(rho).astype(jnp.complex128)
    rho_ab = jnp.transpose(rho, (0, 2, 3, 1))
    logf_diag = jnp.log(jnp.real(rho[:, :, _DIAG3, _DIAG3])).transpose(0, 2, 1)
    return rho, rho_ab, logf_diag


def _pair_weight(t1p, t2p, nn0, cp0, meps, i1, i2):
    """``W^{(nn)}_ab = ½(term1_pair·I1 + term2_pair·meps²·I2)`` at every node: (N, ·, ·, 3, 3), real, symmetric."""
    return 0.5 * (t1p[nn0, :, :, cp0] * i1[..., None, None]
                  + t2p[nn0, :, :, cp0] * meps ** 2 * i2[..., None, None])


def _phi_nodes(t1p, t2p, cp0, meps, i1, i2, rho4, cfor, crev):
    """Node-resolved ``(Φ_for, Φ_rev)``, summed over the lepton parity nn.

    ``cfor[nn] = (1−f_{e2}) f_{e3}`` and ``crev[nn] = f_{e2}(1−f_{e3})``; ``rho4`` is the partner at
    p₄, (N, ·, ·, 3, 3).
    """
    imin4 = jnp.eye(3, dtype=rho4.dtype) - rho4
    phi_for = jnp.zeros_like(rho4)
    phi_rev = jnp.zeros_like(rho4)
    for nn0 in range(2):
        w = _pair_weight(t1p, t2p, nn0, cp0, meps, i1, i2)
        phi_for = phi_for + cfor[nn0][..., None, None] * w * rho4
        phi_rev = phi_rev + crev[nn0][..., None, None] * w * imin4
    return phi_for, phi_rev


def _accumulate(acc, cp0, phi_for, phi_rev, measb, rho1):
    """Node-sum ``Φ``, close the anticommutator once per (cp, bin), and add into ``acc``."""
    net, frs, a_gain, b_loss = acc
    pf = jnp.sum(measb * phi_for, axis=(1, 2))
    pr = jnp.sum(measb * phi_rev, axis=(1, 2))
    net_cp, frs_cp, a_cp, b_cp = assemble_from_phi(pf, pr, rho1)
    return (net.at[cp0].add(net_cp), frs.at[cp0].add(frs_cp),
            a_gain.at[cp0].add(a_cp), b_loss.at[cp0].add(b_cp))


def _zeros4(n):
    z = jnp.zeros((2, n, 3, 3), dtype=jnp.complex128)
    return z, z, z, z


def _finish(acc, hermitian_project):
    """Hermitize the four outputs (removes round-off only) if ``hermitian_project``."""
    if not hermitian_project:
        return acc
    return tuple(hermitize(a) for a in acc)


def nuer1_matrix_core(grid: EpsGrid, rho, meps, tcmpl, phie, n_gl: int = 5, n_lag: int = 5,
                      sin2thw: float = SIN2THW, fd_cap_logC: float = float("inf"), fd_decay: bool = False,
                      monotone_read: bool = False, lag_tail_fix: bool = True,
                      lag_tail_gap: float = _LAG_TAIL_GAP, interp_mode: str = "fraction",
                      hermitian_project: bool = True):
    """Density-matrix ν–e± scattering, R1 integral: ``(net, frs, A_gain, B_loss)``.

    The matrix counterpart of :func:`neutt.transport.nuer1.nuer1_rates`.

    Parameters
    ----------
    rho : (2, N, 3, 3) complex
        ``rho[cp, bin, a, b]``, cp 0 = ν, 1 = ν̄.
    meps, tcmpl, phie : float
        m_e/T_cm, T_cm/T_pl and the electron degeneracy μ_e/T_pl (may be traced).
    n_gl, n_lag : int
        Gauss–Legendre / Gauss–Laguerre points (BURST: 5 each).
    fd_cap_logC, fd_decay, monotone_read
        Tail closures of the partner read; with ``interp_mode="fraction"`` they also act on the
        coherence, through the reconstructed diagonal.
    lag_tail_fix, lag_tail_gap
        As in ``nuer1``: regime A's E₂ cut ecut1 diverges as p₁ → meps/2, and its wide bins get a
        Laguerre tail.  False is BURST's Legendre split.
    interp_mode : {"fraction", "linear", "lagrange"}
        Off-grid coherence read (``collide_matrix.interp_partner_rho``).
    hermitian_project : bool
        Hermitize the outputs; False exposes the raw Hermiticity residual.

    Returns
    -------
    net, frs, A_gain, B_loss : (2, N, 3, 3) complex
        ``net = A_gain − {B_loss, ρ}``; dρ/dt|coll = T_cm⁵·net.
    """
    eps = grid.abscissas
    p1 = eps
    n_bins = eps.shape[0]
    rho, rho_ab, logf_diag = _unpack_rho(rho)
    t1p, t2p = pair_coupling_tensors("nuer1", sin2thw, convention="classical")
    log_kw = dict(fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                  monotone_read=monotone_read)

    tleg, wleg = legendre_canonical(n_gl)
    xlag, wlag = gauss_laguerre_plain(n_lag)

    ecut3 = jnp.sqrt(p1 ** 2 + meps ** 2)
    denom_a = meps - 2.0 * p1
    ecut1 = meps + 2.0 * p1 ** 2 / jnp.where(denom_a != 0.0, denom_a, 1.0)
    regime_a = p1 < (meps / 2.0)
    binmask = (p1 > 0.0).astype(eps.dtype)
    p1sq = jnp.where(p1 > 0.0, p1 ** 2, 1.0)
    bounds_e2 = {"meps": jnp.broadcast_to(meps, (n_bins,)), "ecut3": ecut3, "ecut1": ecut1}

    if lag_tail_fix:
        segments = _E2_SEGMENTS_FIXED
        rega = regime_a.astype(eps.dtype)
        wide = rega * (ecut1 - ecut3 >= lag_tail_gap).astype(eps.dtype)
        regmask_table = {"all": jnp.ones((n_bins,)), "B": (~regime_a).astype(eps.dtype),
                         "A_narrow": rega * (1.0 - wide), "A_wide": wide}
    else:
        segments = _E2_SEGMENTS
        regmask_table = {"all": jnp.ones((n_bins,)), "A": regime_a.astype(eps.dtype),
                         "B": (~regime_a).astype(eps.dtype)}

    acc = _zeros4(n_bins)

    for (_name, rule, a2k, b2k, regime, subsegs) in segments:
        a2 = bounds_e2[a2k]
        if rule == "leg":
            b2 = bounds_e2[b2k]
            mid2 = 0.5 * (a2 + b2)
            half2 = 0.5 * (b2 - a2)
            e2 = mid2[:, None] + half2[:, None] * tleg[None, :]
            e2w = wleg[None, :] * half2[:, None]
        else:
            e2 = a2[:, None] + xlag[None, :]
            e2w = jnp.broadcast_to(wlag[None, :], e2.shape)
        e2 = jnp.maximum(e2, meps)
        q2 = _safe_sqrt_diff(e2 ** 2 - meps ** 2)
        d = 2.0 * p1[:, None] + e2 - q2
        etrans2 = 0.5 * (2.0 * p1[:, None] + e2 - q2 + meps ** 2 / jnp.where(d != 0.0, d, 1.0))
        dl = 2.0 * p1[:, None] + e2 + q2
        elim1 = 0.5 * (2.0 * p1[:, None] + e2 + q2 + meps ** 2 / jnp.where(dl != 0.0, dl, 1.0))
        a2coef = (p1[:, None] + e2) ** 2 - meps ** 2
        fe2 = jnp.stack([_fd(e2, tcmpl, phie, 1.0), _fd(e2, tcmpl, phie, -1.0)])

        bounds_e3 = {"meps": jnp.broadcast_to(meps, e2.shape), "e2": e2,
                     "etrans2": etrans2, "elim1": elim1}
        mask = regmask_table[regime]

        for (a3k, b3k, tbkind) in subsegs:
            a3 = bounds_e3[a3k]
            b3 = bounds_e3[b3k]
            mid3 = 0.5 * (a3 + b3)
            half3 = 0.5 * (b3 - a3)
            e3 = mid3[:, :, None] + half3[:, :, None] * tleg[None, None, :]
            e3w = wleg[None, None, :] * half3[:, :, None]
            e3 = jnp.maximum(e3, meps)
            q3 = _safe_sqrt_diff(e3 ** 2 - meps ** 2)
            top, bot = _topbot(tbkind, p1[:, None, None], e2[:, :, None], q2[:, :, None], e3, q3)
            ae = a2coef[:, :, None]
            i1 = _R1(top, ae) - _R1(bot, ae)
            i2 = _R2(top, ae) - _R2(bot, ae)

            fe3 = jnp.stack([_fd(e3, tcmpl, phie, 1.0), _fd(e3, tcmpl, phie, -1.0)])
            p4 = jnp.maximum(p1[:, None, None] + e2[:, :, None] - e3, 0.0)

            # same parity on both legs (the neutrino scatters off one lepton species)
            cfor = [(1.0 - fe2[nn][:, :, None]) * fe3[nn] for nn in range(2)]
            crev = [fe2[nn][:, :, None] * (1.0 - fe3[nn]) for nn in range(2)]

            meas = ((mask * binmask)[:, None, None] * _PREFAC
                    * (e2w / p1sq[:, None])[:, :, None] * e3w)
            measb = meas[..., None, None]

            for cp0 in range(2):
                rho4 = interp_partner_rho(rho_ab[cp0], eps, p4, logf_diag[cp0],
                                          mode=interp_mode, **log_kw)
                phi_for, phi_rev = _phi_nodes(t1p, t2p, cp0, meps, i1, i2, rho4, cfor, crev)
                acc = _accumulate(acc, cp0, phi_for, phi_rev, measb, rho[cp0])

    return _finish(acc, hermitian_project)


def nuer2_matrix_core(grid: EpsGrid, rho, meps, tcmpl, phie, n_gl: int = 5, n_lag: int = 5,
                      sin2thw: float = SIN2THW, fd_cap_logC: float = float("inf"), fd_decay: bool = False,
                      monotone_read: bool = False, lag_tail_fix: bool = True,
                      lag_tail_gap: float = _LAG_TAIL_GAP, interp_mode: str = "fraction",
                      hermitian_project: bool = True):
    """Density-matrix ν–e± scattering, R2 integral: ``(net, frs, A_gain, B_loss)``.

    The matrix counterpart of :func:`neutt.transport.nuer2.nuer2_rates`: as :func:`nuer1_matrix_core`
    with the outer integral over the outgoing lepton energy E₃ and the CC/NC branches swapped, and
    nuer2's ``lag_tail_fix`` (the wide-R1 Laguerre rule).
    """
    eps = grid.abscissas
    p1 = eps
    n_bins = eps.shape[0]
    rho, rho_ab, logf_diag = _unpack_rho(rho)
    t1p, t2p = pair_coupling_tensors("nuer2", sin2thw, convention="classical")
    log_kw = dict(fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                  monotone_read=monotone_read)

    tleg, wleg = legendre_canonical(n_gl)
    xlag, wlag = gauss_laguerre_plain(n_lag)

    binmask = (p1 > 0.0).astype(eps.dtype)
    p1sq = jnp.where(p1 > 0.0, p1 ** 2, 1.0)

    ecut1 = p1 + meps ** 2 / jnp.where(p1 > 0.0, 4.0 * p1, 1.0)
    ecut2 = p1 + meps * (p1 + meps) / jnp.where(2.0 * p1 + meps > 0.0, 2.0 * p1 + meps, 1.0)
    ecut3 = jnp.sqrt(p1 ** 2 + meps ** 2)
    bounds3 = {"meps": jnp.broadcast_to(jnp.asarray(meps, eps.dtype), (n_bins,)),
               "ecut1": ecut1, "ecut2": ecut2, "ecut3": ecut3}

    regmasks = _r2_regime_masks(p1, meps, ecut1, ecut2, lag_tail_fix, lag_tail_gap)

    acc = _zeros4(n_bins)

    for rname, segments in _r2_regimes(lag_tail_fix).items():
        regmask = regmasks[rname] * binmask
        for (e3rule, a3k, b3k, subsegs) in segments:
            a3 = bounds3[a3k]
            if e3rule == "leg":
                b3 = bounds3[b3k]
                mid3 = 0.5 * (a3 + b3)
                half3 = 0.5 * (b3 - a3)
                e3 = mid3[:, None] + half3[:, None] * tleg[None, :]
                e3w = wleg[None, :] * half3[:, None]
            else:
                e3 = a3[:, None] + xlag[None, :]
                e3w = jnp.broadcast_to(wlag[None, :], e3.shape)
            e3 = jnp.maximum(e3, meps)
            q3 = _safe_sqrt_diff(e3 ** 2 - meps ** 2)
            d = e3 + q3 - 2.0 * p1[:, None]
            etrans2 = 0.5 * (e3 + q3 - 2.0 * p1[:, None] + meps ** 2 / jnp.where(d != 0.0, d, 1.0))
            db = e3 - q3 - 2.0 * p1[:, None]
            elim1raw = 0.5 * (e3 - q3 - 2.0 * p1[:, None]
                              + meps ** 2 / jnp.where(db != 0.0, db, 1.0))
            elim1 = jnp.where(elim1raw < 0.0, jnp.maximum(e3, etrans2), elim1raw)
            elim2 = etrans2
            fe3 = jnp.stack([_fd(e3, tcmpl, phie, 1.0), _fd(e3, tcmpl, phie, -1.0)])
            kinemfact1 = e3w / p1sq[:, None]
            bcoef = (p1[:, None] - e3) ** 2 - meps ** 2

            bounds2 = {"meps": jnp.broadcast_to(jnp.asarray(meps, eps.dtype), e3.shape),
                       "e3": e3, "etrans2": etrans2, "elim1": elim1, "elim2": elim2}

            for (e2rule, a2k, b2k, kind, sel) in subsegs:
                a2 = bounds2[a2k]
                if e2rule == "leg":
                    b2 = bounds2[b2k]
                    mid2 = 0.5 * (a2 + b2)
                    half2 = 0.5 * (b2 - a2)
                    e2 = mid2[:, :, None] + half2[:, :, None] * tleg[None, None, :]
                    e2w = wleg[None, None, :] * half2[:, :, None]
                else:
                    e2 = a2[:, :, None] + xlag[None, None, :]
                    e2w = jnp.broadcast_to(wlag[None, None, :], e2.shape)
                e2 = jnp.maximum(e2, meps)
                q2 = _safe_sqrt_diff(e2 ** 2 - meps ** 2)

                top, bot = _topbot_r2(kind, p1[:, None, None], e3[:, :, None],
                                      q3[:, :, None], e2, q2)
                be = bcoef[:, :, None]
                i1 = _R2m1(top, be) - _R2m1(bot, be)
                i2 = _R2m2(top, be) - _R2m2(bot, be)

                fe2 = jnp.stack([_fd(e2, tcmpl, phie, 1.0), _fd(e2, tcmpl, phie, -1.0)])
                p4 = jnp.maximum(p1[:, None, None] + e2 - e3[:, :, None], 0.0)

                cfor = [(1.0 - fe2[nn]) * fe3[nn][:, :, None] for nn in range(2)]
                crev = [fe2[nn] * (1.0 - fe3[nn][:, :, None]) for nn in range(2)]

                gap = bounds2[b2k] - a2 if sel is not None else None
                extra = _r2_sel_mask(sel, gap, e3, ecut1)

                meas = (_PREFAC * regmask[:, None, None]
                        * (kinemfact1 * extra)[:, :, None] * e2w)
                measb = meas[..., None, None]

                for cp0 in range(2):
                    rho4 = interp_partner_rho(rho_ab[cp0], eps, p4, logf_diag[cp0],
                                              mode=interp_mode, **log_kw)
                    phi_for, phi_rev = _phi_nodes(t1p, t2p, cp0, meps, i1, i2, rho4, cfor, crev)
                    acc = _accumulate(acc, cp0, phi_for, phi_rev, measb, rho[cp0])

    return _finish(acc, hermitian_project)


# ---- public entry points (as collide's nuepma_matrix_*) ----------------------------------------

def nuer1_matrix_rates(grid: EpsGrid, rho, meps, tcmpl, phie, **kw):
    """``(net, frs)`` for the R1 ν–e± scattering matrix collision (each (2, N, 3, 3))."""
    net, frs, _, _ = nuer1_matrix_core(grid, rho, meps, tcmpl, phie, **kw)
    return net, frs


def nuer2_matrix_rates(grid: EpsGrid, rho, meps, tcmpl, phie, **kw):
    """``(net, frs)`` for the R2 ν–e± scattering matrix collision (each (2, N, 3, 3))."""
    net, frs, _, _ = nuer2_matrix_core(grid, rho, meps, tcmpl, phie, **kw)
    return net, frs


def nuer1_matrix_affine(grid: EpsGrid, rho, meps, tcmpl, phie, **kw):
    """Affine pieces ``(A_gain, B_loss)`` of the R1 matrix collision: net = A_gain − {B_loss, ρ}."""
    _, _, a_gain, b_loss = nuer1_matrix_core(grid, rho, meps, tcmpl, phie, **kw)
    return a_gain, b_loss


def nuer2_matrix_affine(grid: EpsGrid, rho, meps, tcmpl, phie, **kw):
    """Affine pieces ``(A_gain, B_loss)`` of the R2 matrix collision: net = A_gain − {B_loss, ρ}."""
    _, _, a_gain, b_loss = nuer2_matrix_core(grid, rho, meps, tcmpl, phie, **kw)
    return a_gain, b_loss


def nuer1_matrix_dfdt(grid, rho, meps, tcmpl, phie, t_cm_mev, **kw):
    """dρ/dt|coll = T_cm⁵·net from R1 ν–e± scattering [s⁻¹]."""
    net, _ = nuer1_matrix_rates(grid, rho, meps, tcmpl, phie, **kw)
    return t_cm_mev ** 5 * net


def nuer2_matrix_dfdt(grid, rho, meps, tcmpl, phie, t_cm_mev, **kw):
    """dρ/dt|coll = T_cm⁵·net from R2 ν–e± scattering [s⁻¹]."""
    net, _ = nuer2_matrix_rates(grid, rho, meps, tcmpl, phie, **kw)
    return t_cm_mev ** 5 * net
