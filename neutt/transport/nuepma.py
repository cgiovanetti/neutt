"""νν̄ ↔ e⁻e⁺ annihilation, BURST's ``trans_nuepma.f90``.

ν(p1) + ν̄(p4) ↔ e⁻ + e⁺, p4 = E_out + E_in − p1.  Both integrated energies are electron energies,
E_out the outer and E_in the inner integral.  The partner ν̄ is read at the opposite ν/ν̄ index, and
both assignments of e⁻ and e⁺ to (E_in, E_out) are summed per node (BURST's ``swapflag``).
Background arguments are as in :mod:`nuer1`.  Four p1-regimes, split at pcase1 = meps/2 <
pcase2 = meps(1 + √5)/4 < pcase3 = meps, divide E_out at ecut1 = p1 + meps²/(4p1),
ecut2 = (2p1 − meps + meps²/(2p1 − meps))/2 and p1, and E_in at etrans1, etrans2 (per E_out node),
into fixed tiles.  BURST's ``genout``/``genin`` segments are Legendre [a, b] for Δ = b − a < 200 and
Laguerre [a, ∞) otherwise.

Two implementations of the same integrand and quadrature: ``nuepma_rates``, the reference (every
regime evaluated and masked; bit-faithful to BURST with ``lag_tail_fix=False``), and
``nuepma_rates_sched``, the scheduled form (``use_sched``: one batched contraction over a
per-regime tile schedule, the genout/genin branches flattened into tiles).

``lag_tail_fix``: ecut2 diverges as p1 → pcase1 from above, so just above pcase1 regime R2's genout
segment [ecut1, ecut2] is wide yet below Δ = 200, and its Legendre rule is under-resolved (the
defect nuer1 has at meps/2).  The fix lowers the genout threshold to ``lag_tail_gap``.  Where genout
then takes the tail [ecut1, ∞), R2's [ecut2, ∞) "a" tile would count E_out ≥ ecut2 twice and is
dropped.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import GFHBAR, PI, SIN2THW
from ..grid import EpsGrid
from ..interp import interp_log_occ_dynamic
from ..grid import legendre_canonical, gauss_laguerre_plain


def _coupling_tensors_pma(sin2thw: float = SIN2THW):
    """term1 and c2 = term2/meps² per (swap, flavour m, ν/ν̄ n), as in calc_weight_nuepma.

    swap 0 is BURST's swapflag = .true. (e⁻ at E_in, e⁺ at E_out), swap 1 the reverse.  (swap 0,
    n = 0) and (swap 1, n = 1) get 32(2s + fla)² and c2 = +2s/(2s + fla), the sign opposite to
    nuer1's; the others get 128s² and (2s + fla)/(2s).
    """
    s = sin2thw
    term1 = np.zeros((2, 3, 2))
    c2 = np.zeros((2, 3, 2))
    for sw in range(2):           # 0=True, 1=False
        for m in range(3):
            fla = 1.0 if m == 0 else -1.0
            for n in range(2):
                branchA = (sw == 0 and n == 0) or (sw == 1 and n == 1)
                if branchA:
                    term1[sw, m, n] = 32.0 * (2.0 * s + fla) ** 2
                    c2[sw, m, n] = 2.0 * s / (2.0 * s + fla)
                else:
                    term1[sw, m, n] = 128.0 * s ** 2
                    c2[sw, m, n] = (2.0 * s + fla) / (2.0 * s)
    return jnp.asarray(term1), jnp.asarray(c2)


def _fd(E, tcmpl, phie, parfact):
    """e∓ occupation 1/(e^{E·tcmpl ∓ phie} + 1) for parfact = ±1, zero above E = 300."""
    arg = E * tcmpl - parfact * phie
    return jnp.where(E <= 300.0, jax.nn.sigmoid(-arg), 0.0)


def _safe_sqrt_diff(u):
    """√max(u, 0) with a zero (not NaN) gradient where u ≤ 0."""
    pos = u > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, u, 1.0)), 0.0)


# Antiderivatives of the angular integrand (BURST's rint_l_1/2), C = (p1 − E_out)² − meps².
def _L1(x, C):
    return 0.25 * (C**2 * x - (2.0 / 3.0) * C * x**3 + 0.2 * x**5)


def _L2(x, C):
    return 0.5 * ((1.0 / 3.0) * x**3 - C * x)


def _topbot_pma(kind, p1, qout, eout, ein, qin):
    """Limits (top, bot) of the analytic angular integral for an E_in tile's formula id."""
    if kind == "a":
        return p1 + qout, eout - p1 + ein - qin
    if kind == "b":
        return p1 + qout, p1 - qout
    if kind == "c":
        return eout - p1 + ein + qin, eout - p1 + ein - qin
    if kind == "d":
        return eout - p1 + ein + qin, p1 - qout
    raise ValueError(kind)


# BURST's segments per regime: (E_out rule, a_out, b_out, E_in tiles), each E_in tile
# (E_in rule, a_in, b_in, angular-limit formula of _topbot_pma, posguard).  Rules: "leg" [a, b],
# "lag" [a, ∞) (b = None), "genout" = leg if b − a < 200 else lag, "genin" = leg if 0 ≤ b − a < 200
# else lag.  E_out bounds: meps, ecut1, ecut2, p1; E_in bounds: meps, etrans1, etrans2 (BURST's
# elim1 = etrans2, elim2 = etrans1).  posguard names a bound that must be > 0 for the tile to count:
# BURST's round-off guards on the plain Laguerre E_in tails.
_REGIMES = {
    # R1: p1 < pcase1  (ecut2 unused).
    "R1": [
        ("lag", "ecut1", None, [("lag", "etrans2", None, "a", "etrans2")]),
    ],
    # R2: pcase1 <= p1 < pcase2  (ecut1 < ecut2).  Middle outer segment is a genout.
    "R2": [
        ("leg", "meps", "ecut1", [("genin", "etrans2", "etrans1", "a", None),
                                  ("lag", "etrans1", None, "b", "etrans1")]),
        ("genout", "ecut1", "ecut2", [("lag", "etrans2", None, "a", "etrans2")]),
        ("lag", "ecut2", None, [("genin", "meps", "etrans2", "c", None),
                                ("lag", "etrans2", None, "a", "etrans2")]),
    ],
    # R3: pcase2 <= p1 < pcase3  (p1 < ecut2 < ecut1).
    "R3": [
        ("leg", "meps", "ecut2", [("genin", "etrans2", "etrans1", "a", None),
                                  ("lag", "etrans1", None, "b", "etrans1")]),
        ("leg", "ecut2", "ecut1", [("leg", "meps", "etrans2", "c", None),
                                   ("genin", "etrans2", "etrans1", "a", None),
                                   ("lag", "etrans1", None, "b", "etrans1")]),
        ("lag", "ecut1", None, [("genin", "meps", "etrans2", "c", None),
                                ("lag", "etrans2", None, "a", "etrans2")]),
    ],
    # R4: p1 >= pcase3  (ecut2 < p1 < ecut1).
    "R4": [
        ("leg", "meps", "ecut2", [("genin", "etrans1", "etrans2", "d", None),
                                  ("lag", "etrans2", None, "b", "etrans2")]),
        ("leg", "ecut2", "p1", [("leg", "meps", "etrans1", "c", None),
                                ("genin", "etrans1", "etrans2", "d", None),
                                ("lag", "etrans2", None, "b", "etrans2")]),
        ("leg", "p1", "ecut1", [("leg", "meps", "etrans2", "c", None),
                                ("genin", "etrans2", "etrans1", "a", None),
                                ("lag", "etrans1", None, "b", "etrans1")]),
        ("lag", "ecut1", None, [("genin", "meps", "etrans2", "c", None),
                                ("lag", "etrans2", None, "a", "etrans2")]),
    ],
}


# The genout threshold on ecut2 − ecut1 [T_cm units] under lag_tail_fix (BURST's is 200); nuer1's
# gap value.
_LAG_TAIL_GAP_PMA = 6.0


def nuepma_rates(grid: EpsGrid, occ, meps, tcmpl, phie, n_gl: int = 5, n_lag: int = 5,
                 range_limit=False, fd_cap_logC: float = float("inf"), fd_decay: bool = False,
                 monotone_read: bool = False,
                 lag_tail_fix: bool = True, lag_tail_gap: float = _LAG_TAIL_GAP_PMA):
    """νν̄ ↔ e⁻e⁺ rates, the reference implementation.

    Arguments and returns as :func:`nuer1.nuer1_rates`, except that ``lag_tail_fix`` lowers R2's
    genout threshold to ``lag_tail_gap`` (module docstring; False keeps BURST's 200 and is
    bit-faithful) and the e⁻/e⁺ sum is over the two swapflag assignments.
    """
    eps = grid.abscissas
    p1 = eps                                   # (N,)
    N = eps.shape[0]
    logf = jnp.log(occ)                        # (3,2,N)
    term1, c2 = _coupling_tensors_pma()        # (2,3,2) each (swap axis)
    prefac = GFHBAR / 16.0 / (2.0 * PI) ** 3

    tleg, wleg = legendre_canonical(n_gl)
    xlag, wlag = gauss_laguerre_plain(n_lag)

    meps_s = jnp.asarray(meps, eps.dtype)
    binmask = (p1 > 0.0).astype(eps.dtype)
    p1sq = jnp.where(p1 > 0.0, p1**2, 1.0)

    # E_out cut energies
    ecut1 = p1 + meps**2 / jnp.where(p1 > 0.0, 4.0 * p1, 1.0)
    den2 = 2.0 * p1 - meps
    ecut2 = 0.5 * (2.0 * p1 - meps + meps**2 / jnp.where(den2 != 0.0, den2, 1.0))
    bounds_out = {"meps": jnp.broadcast_to(meps_s, (N,)), "ecut1": ecut1, "ecut2": ecut2,
                  "p1": p1}

    # regime masks; a bin exactly on a pcase gets zero-width segments (the continuous limit)
    pcase1 = meps / 2.0
    pcase2 = meps * (1.0 + jnp.sqrt(5.0)) / 4.0
    pcase3 = meps
    regmasks = {
        "R1": (p1 < pcase1).astype(eps.dtype),
        "R2": ((p1 >= pcase1) & (p1 < pcase2)).astype(eps.dtype),
        "R3": ((p1 >= pcase2) & (p1 < pcase3)).astype(eps.dtype),
        "R4": (p1 >= pcase3).astype(eps.dtype),
    }
    clamp_e2 = {"R1": False, "R2": True, "R3": False, "R4": False}  # R2 uses max(meps,etrans2)

    # genout_lag flags the bins whose genout takes the [ecut1, ∞) tail under the lowered threshold
    # (only R2 bins use it); with lag_tail_fix=False it is 0 and thr_out is BURST's 200
    thr_out = lag_tail_gap if lag_tail_fix else 200.0
    genout_lag = (((ecut2 - ecut1) >= thr_out).astype(eps.dtype) if lag_tail_fix
                  else jnp.zeros((N,), eps.dtype))

    net = jnp.zeros((3, 2, N))
    frs = jnp.zeros((3, 2, N))

    def _outer_nodes(rule, a, b):
        if rule == "leg":
            mid = 0.5 * (a + b); half = 0.5 * (b - a)
            E = mid[:, None] + half[:, None] * tleg[None, :]          # (N,J)
            w = wleg[None, :] * half[:, None]
        else:  # Laguerre [a, ∞)
            E = a[:, None] + xlag[None, :]
            w = jnp.broadcast_to(wlag[None, :], E.shape)
        return E, w

    def _inner_nodes(rule, a, b):
        if rule == "leg":
            mid = 0.5 * (a + b); half = 0.5 * (b - a)
            E = mid[:, :, None] + half[:, :, None] * tleg[None, None, :]   # (N,J,K)
            w = wleg[None, None, :] * half[:, :, None]
        else:  # Laguerre [a, ∞)
            E = a[:, :, None] + xlag[None, None, :]
            w = jnp.broadcast_to(wlag[None, None, :], E.shape)
        return E, w

    for rname, segments in _REGIMES.items():
        regmask = regmasks[rname] * binmask                          # (N,)
        for (out_rule, a_out_k, b_out_k, inner_tiles) in segments:
            a_out = bounds_out[a_out_k]                              # (N,)
            # expand a genout into (rule, outer-select-mask) variants
            if out_rule == "genout":
                b_out = bounds_out[b_out_k]
                out_variants = [("leg", a_out, b_out, (b_out - a_out < thr_out).astype(eps.dtype)),
                                ("lag", a_out, None, (b_out - a_out >= thr_out).astype(eps.dtype))]
            elif out_rule == "leg":
                out_variants = [("leg", a_out, bounds_out[b_out_k], jnp.ones((N,)))]
            else:
                out_variants = [("lag", a_out, None, jnp.ones((N,)))]

            for (orule, oa, ob, out_mask) in out_variants:
                eout, eoutw = _outer_nodes(orule, oa, ob)            # (N,J)
                eout = jnp.maximum(eout, meps_s)
                qout = _safe_sqrt_diff(eout**2 - meps_s**2)          # (N,J)
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
                    a_in = bounds_in[a_in_k]                         # (N,J)
                    if in_rule == "genin":
                        b_in = bounds_in[b_in_k]
                        gap = b_in - a_in
                        in_variants = [
                            ("leg", a_in, b_in, ((gap >= 0.0) & (gap < 200.0)).astype(eps.dtype)),
                            ("lag", a_in, None, ((gap < 0.0) | (gap >= 200.0)).astype(eps.dtype))]
                    elif in_rule == "leg":
                        in_variants = [("leg", a_in, bounds_in[b_in_k], jnp.ones(eout.shape))]
                    else:  # plain lag, optional posguard
                        m_in = (bounds_in[posguard] > 0.0).astype(eps.dtype) if posguard else jnp.ones(eout.shape)
                        in_variants = [("lag", a_in, None, m_in)]

                    # lag_tail_fix: drop R2's [ecut2, ∞) "a" tile where genout's tail covers it
                    if lag_tail_fix and rname == "R2" and a_out_k == "ecut2" and kind == "a":
                        in_variants = [(ir, ia, ib, m * (1.0 - genout_lag)[:, None])
                                       for (ir, ia, ib, m) in in_variants]

                    for (irule, ia, ib, in_mask) in in_variants:
                        ein, einw = _inner_nodes(irule, ia, ib)      # (N,J,K)
                        ein = jnp.maximum(ein, meps_s)
                        qin = _safe_sqrt_diff(ein**2 - meps_s**2)    # (N,J,K)
                        top, bot = _topbot_pma(kind, p1[:, None, None], qout[:, :, None],
                                               eout[:, :, None], ein, qin)          # (N,J,K)
                        Ce = C[:, :, None]
                        I1 = _L1(top, Ce) - _L1(bot, Ce)
                        I2 = _L2(top, Ce) - _L2(bot, Ce)
                        # weights for both swap assignments: (swap,m,n,N,J,K)
                        w_sw = term1[:, :, :, None, None, None] * (
                            I1[None, None, None] + c2[:, :, :, None, None, None] * meps**2 * I2[None, None, None])
                        w_true = w_sw[0]                              # (m,n,N,J,K)
                        w_false = w_sw[1]

                        femin_in = _fd(ein, tcmpl, phie, 1.0)        # (N,J,K)
                        fepl_in = _fd(ein, tcmpl, phie, -1.0)
                        p4 = jnp.maximum(eout[:, :, None] + ein - p1[:, None, None], 0.0)  # (N,J,K)
                        fnu4 = jnp.exp(interp_log_occ_dynamic(logf, eps, p4, range_limit=range_limit,
                                                              fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                                                              monotone_read=monotone_read))  # (3,2,N,J,K)
                        fnu4p = fnu4[:, ::-1]                          # partner ν̄ at opposite cp

                        fnu1 = occ[:, :, :, None, None]               # (m,n,N,1,1)
                        f_min_in = femin_in[None, None]               # (1,1,N,J,K)
                        f_pl_in = fepl_in[None, None]
                        f_min_out = femin_out[None, None, :, :, None] # (1,1,N,J,1)
                        f_pl_out = fepl_out[None, None, :, :, None]
                        # swap True: e-=ein, e+=eout
                        forf_t = (1.0 - fnu1) * (1.0 - fnu4p) * f_min_in * f_pl_out
                        revf_t = fnu1 * fnu4p * (1.0 - f_min_in) * (1.0 - f_pl_out)
                        # swap False: e-=eout, e+=ein
                        forf_f = (1.0 - fnu1) * (1.0 - fnu4p) * f_min_out * f_pl_in
                        revf_f = fnu1 * fnu4p * (1.0 - f_min_out) * (1.0 - f_pl_in)

                        meas = (prefac * (regmask * out_mask)[:, None, None]
                                * (in_mask * kinemfact1)[:, :, None] * einw)   # (N,J,K)
                        measb = meas[None, None]                       # (1,1,N,J,K)
                        net = net + jnp.sum(
                            measb * (w_true * (forf_t - revf_t) + w_false * (forf_f - revf_f)),
                            axis=(3, 4))                               # (m,n,N)
                        frs = frs + jnp.sum(
                            measb * (w_true * (forf_t + revf_t) + w_false * (forf_f + revf_f)),
                            axis=(3, 4))

    return net, frs


def nuepma_dfdt(grid, occ, meps, tcmpl, phie, t_cm_mev, **kw):
    """df/dt from νν̄ ↔ e⁻e⁺, in s⁻¹ (T_cm⁵·net)."""
    net, _ = nuepma_rates(grid, occ, meps, tcmpl, phie, **kw)
    return t_cm_mev**5 * net


# ---- the scheduled implementation ---------------------------------------------------------------
# Each regime's (E_out-segment × E_in-tile) tiles from _REGIMES, each genout/genin split into its
# Legendre and Laguerre tiles: 1–14 per regime, padded to 14.  A tile is
# (out_rule, a_out, b_out, out_sel, in_rule, a_in, b_in, in_sel, kind, posg, clamp):
#   rule          0 = Legendre [a, b], 1 = Laguerre [a, ∞), b = −1 (a sel = 2 tile keeps b for Δ)
#   a_out, b_out  E_out bound: 0 = meps, 1 = ecut1, 2 = ecut2, 3 = p1
#   out_sel       genout: 0 = none, 1 = on for Δ < thr_out, 2 = on for Δ ≥ thr_out
#   a_in, b_in    E_in bound: 0 = meps, 1 = etrans1, 2 = etrans2
#   in_sel        genin: 0 = none, 1 = on for 0 ≤ Δ < 200, 2 = on otherwise
#   kind          angular limits (_topbot_pma): 0 = a, 1 = b, 2 = c, 3 = d
#   posg          posguard: −1 = none, else the E_in bound key that must be > 0
#   clamp         1: etrans2 = max(meps, etrans2) (all R2 tiles)

_TILES = {
    "R1": [
        (1, 1, -1, 0, 1, 2, -1, 0, 0, 2, 0),
    ],
    "R2": [
        (0, 0, 1, 0, 0, 2, 1, 1, 0, -1, 1), (0, 0, 1, 0, 1, 2, 1, 2, 0, -1, 1),
        (0, 0, 1, 0, 1, 1, -1, 0, 1, 1, 1),
        (0, 1, 2, 1, 1, 2, -1, 0, 0, 2, 1), (1, 1, 2, 2, 1, 2, -1, 0, 0, 2, 1),
        (1, 2, -1, 0, 0, 0, 2, 1, 2, -1, 1), (1, 2, -1, 0, 1, 0, 2, 2, 2, -1, 1),
        (1, 2, -1, 0, 1, 2, -1, 0, 0, 2, 1),
    ],
    "R3": [
        (0, 0, 2, 0, 0, 2, 1, 1, 0, -1, 0), (0, 0, 2, 0, 1, 2, 1, 2, 0, -1, 0),
        (0, 0, 2, 0, 1, 1, -1, 0, 1, 1, 0),
        (0, 2, 1, 0, 0, 0, 2, 0, 2, -1, 0), (0, 2, 1, 0, 0, 2, 1, 1, 0, -1, 0),
        (0, 2, 1, 0, 1, 2, 1, 2, 0, -1, 0), (0, 2, 1, 0, 1, 1, -1, 0, 1, 1, 0),
        (1, 1, -1, 0, 0, 0, 2, 1, 2, -1, 0), (1, 1, -1, 0, 1, 0, 2, 2, 2, -1, 0),
        (1, 1, -1, 0, 1, 2, -1, 0, 0, 2, 0),
    ],
    "R4": [
        (0, 0, 2, 0, 0, 1, 2, 1, 3, -1, 0), (0, 0, 2, 0, 1, 1, 2, 2, 3, -1, 0),
        (0, 0, 2, 0, 1, 2, -1, 0, 1, 2, 0),
        (0, 2, 3, 0, 0, 0, 1, 0, 2, -1, 0), (0, 2, 3, 0, 0, 1, 2, 1, 3, -1, 0),
        (0, 2, 3, 0, 1, 1, 2, 2, 3, -1, 0), (0, 2, 3, 0, 1, 2, -1, 0, 1, 2, 0),
        (0, 3, 1, 0, 0, 0, 2, 0, 2, -1, 0), (0, 3, 1, 0, 0, 2, 1, 1, 0, -1, 0),
        (0, 3, 1, 0, 1, 2, 1, 2, 0, -1, 0), (0, 3, 1, 0, 1, 1, -1, 0, 1, 1, 0),
        (1, 1, -1, 0, 0, 0, 2, 1, 2, -1, 0), (1, 1, -1, 0, 1, 0, 2, 2, 2, -1, 0),
        (1, 1, -1, 0, 1, 2, -1, 0, 0, 2, 0),
    ],
}
_TMAX = 14
_NF = 11


def _build_table():
    """_TILES as a (regime, tile, field) int table and a (regime, tile) 0/1 mask of real tiles."""
    tab = np.zeros((4, _TMAX, _NF), np.int32)
    act = np.zeros((4, _TMAX), np.float64)
    for r, rn in enumerate(["R1", "R2", "R3", "R4"]):
        for t, tile in enumerate(_TILES[rn]):
            tab[r, t] = tile
            act[r, t] = 1.0
    return tab, act


_TAB, _ACT = _build_table()

# The (regime, tile) dropped under lag_tail_fix where genout takes its tail: R2's [ecut2, ∞) "a".
_DCG = np.zeros((4, _TMAX), np.float64)
_DCG[1, 7] = 1.0


def _eval_tiles(grid, occ, logf, meps, tcmpl, phie, A_out, B_out,
                out_rule, out_sel, a_in_k, b_in_k, in_rule, in_sel, kind, posg, clamp, active,
                n_gl=5, n_lag=5, fd_cap_logC=float("inf"), fd_decay=False, monotone_read=False,
                thr_out=200.0):
    """Sum all tiles of all bins in one contraction; returns (net, frs), each (3, 2, N).

    Tile fields are (N, T): A_out, B_out the resolved E_out bounds, the rest int codes, ``active``
    the 0/1 tile mask; ``thr_out`` is the genout threshold.
    """
    eps = grid.abscissas
    p1 = eps
    term1, c2 = _coupling_tensors_pma()
    prefac = GFHBAR / 16.0 / (2.0 * PI) ** 3
    tleg, wleg = legendre_canonical(n_gl)
    xlag, wlag = gauss_laguerre_plain(n_lag)
    binmask = (p1 > 0.0).astype(eps.dtype)
    p1sq = jnp.where(p1 > 0.0, p1 ** 2, 1.0)
    msq = meps ** 2
    meps_s = jnp.asarray(meps, eps.dtype)

    # E_out (outer) nodes, (N, T, J): both rules evaluated, one kept per tile
    is_lago = (out_rule == 1)[:, :, None]
    mido = 0.5 * (A_out + B_out); halfo = 0.5 * (B_out - A_out)
    Eo_leg = mido[:, :, None] + halfo[:, :, None] * tleg[None, None, :]
    ow_leg = wleg[None, None, :] * halfo[:, :, None]
    Eo_lag = A_out[:, :, None] + xlag[None, None, :]
    ow_lag = jnp.broadcast_to(wlag[None, None, :], Eo_lag.shape)
    eout = jnp.where(is_lago, Eo_lag, Eo_leg)
    eoutw = jnp.where(is_lago, ow_lag, ow_leg)
    eout = jnp.maximum(eout, meps_s)
    qout = _safe_sqrt_diff(eout ** 2 - msq)
    p1_3 = p1[:, None, None]
    d1 = 2.0 * p1_3 - eout - qout
    etrans1 = 0.5 * (2.0 * p1_3 - eout - qout + msq / jnp.where(d1 != 0.0, d1, 1.0))
    d2 = 2.0 * p1_3 - eout + qout
    etrans2_raw = 0.5 * (2.0 * p1_3 - eout + qout + msq / jnp.where(d2 != 0.0, d2, 1.0))
    etrans2 = jnp.where(clamp[:, :, None] == 1, jnp.maximum(meps_s, etrans2_raw), etrans2_raw)
    C = (p1_3 - eout) ** 2 - msq                                   # (N,T,J)
    kinemfact1 = eoutw / p1sq[:, None, None]                       # (N,T,J)
    femin_out = _fd(eout, tcmpl, phie, 1.0)                        # e- at eout (N,T,J)
    fepl_out = _fd(eout, tcmpl, phie, -1.0)                        # e+ at eout
    gap_out = B_out - A_out                                        # (N,T)
    out_mask = jnp.where(out_sel == 0, 1.0,
                         jnp.where(out_sel == 1, (gap_out < thr_out).astype(eps.dtype),
                                   (gap_out >= thr_out).astype(eps.dtype)))   # (N,T)

    # E_in bounds and the posguard bound, picked by key from [meps, etrans1, etrans2]
    inb = jnp.stack([jnp.broadcast_to(meps_s, eout.shape), etrans1, etrans2])
    a_in = jnp.take_along_axis(inb, jnp.broadcast_to(a_in_k[None, :, :, None], (1,) + eout.shape), 0)[0]
    b_in_safe = jnp.where(b_in_k < 0, 0, b_in_k)
    b_in = jnp.take_along_axis(inb, jnp.broadcast_to(b_in_safe[None, :, :, None], (1,) + eout.shape), 0)[0]
    posg_safe = jnp.where(posg < 0, 0, posg)
    posg_val = jnp.take_along_axis(inb, jnp.broadcast_to(posg_safe[None, :, :, None], (1,) + eout.shape), 0)[0]

    # E_in (inner) nodes, (N, T, J, K)
    is_lagi = (in_rule == 1)[:, :, None, None]
    midi = 0.5 * (a_in + b_in); halfi = 0.5 * (b_in - a_in)
    Ei_leg = midi[:, :, :, None] + halfi[:, :, :, None] * tleg[None, None, None, :]
    iw_leg = wleg[None, None, None, :] * halfi[:, :, :, None]
    Ei_lag = a_in[:, :, :, None] + xlag[None, None, None, :]
    iw_lag = jnp.broadcast_to(wlag[None, None, None, :], Ei_lag.shape)
    ein = jnp.where(is_lagi, Ei_lag, Ei_leg)
    einw = jnp.where(is_lagi, iw_lag, iw_leg)
    ein = jnp.maximum(ein, meps_s)
    qin = _safe_sqrt_diff(ein ** 2 - msq)
    gap_in = b_in - a_in                                          # (N,T,J)
    posg_mask = jnp.where(posg[:, :, None] < 0, 1.0, (posg_val > 0.0).astype(eps.dtype))   # (N,T,J)
    in_mask = jnp.where(in_sel[:, :, None] == 0, posg_mask,
                        jnp.where(in_sel[:, :, None] == 1,
                                  ((gap_in >= 0.0) & (gap_in < 200.0)).astype(eps.dtype),
                                  ((gap_in < 0.0) | (gap_in >= 200.0)).astype(eps.dtype)))  # (N,T,J)

    # angular limits by kind, (N, T, J, K)
    p1_4 = p1[:, None, None, None]
    qout_4 = qout[:, :, :, None]; eout_4 = eout[:, :, :, None]
    a_top = p1_4 + qout_4; a_bot = eout_4 - p1_4 + ein - qin
    b_top = p1_4 + qout_4; b_bot = p1_4 - qout_4
    c_top = eout_4 - p1_4 + ein + qin; c_bot = eout_4 - p1_4 + ein - qin
    d_top = eout_4 - p1_4 + ein + qin; d_bot = p1_4 - qout_4
    k4 = kind[:, :, None, None]
    top = jnp.where(k4 == 0, a_top, jnp.where(k4 == 1, b_top, jnp.where(k4 == 2, c_top, d_top)))
    bot = jnp.where(k4 == 0, a_bot, jnp.where(k4 == 1, b_bot, jnp.where(k4 == 2, c_bot, d_bot)))
    Ce = C[:, :, :, None]
    I1 = _L1(top, Ce) - _L1(bot, Ce)
    I2 = _L2(top, Ce) - _L2(bot, Ce)
    w_sw = term1[:, :, :, None, None, None, None] * (
        I1[None, None, None] + c2[:, :, :, None, None, None, None] * msq * I2[None, None, None])
    w_true = w_sw[0]; w_false = w_sw[1]                            # (m,n,N,T,J,K)

    femin_in = _fd(ein, tcmpl, phie, 1.0); fepl_in = _fd(ein, tcmpl, phie, -1.0)   # (N,T,J,K)
    p4 = jnp.maximum(eout_4 + ein - p1_4, 0.0)
    fnu4 = jnp.exp(interp_log_occ_dynamic(logf, eps, p4, fd_cap_logC=fd_cap_logC,
                                          fd_decay=fd_decay, monotone_read=monotone_read))  # (3,2,N,T,J,K)
    fnu4p = fnu4[:, ::-1]                                          # partner ν̄ at opposite cp

    fnu1 = occ[:, :, :, None, None, None]                          # (m,n,N,1,1,1)
    f_min_in = femin_in[None, None]; f_pl_in = fepl_in[None, None]                  # (1,1,N,T,J,K)
    f_min_out = femin_out[None, None, :, :, :, None]; f_pl_out = fepl_out[None, None, :, :, :, None]
    forf_t = (1.0 - fnu1) * (1.0 - fnu4p) * f_min_in * f_pl_out
    revf_t = fnu1 * fnu4p * (1.0 - f_min_in) * (1.0 - f_pl_out)
    forf_f = (1.0 - fnu1) * (1.0 - fnu4p) * f_min_out * f_pl_in
    revf_f = fnu1 * fnu4p * (1.0 - f_min_out) * (1.0 - f_pl_in)

    meas = (prefac * (active * binmask[:, None] * out_mask)[:, :, None, None]
            * (in_mask * kinemfact1)[:, :, :, None] * einw)        # (N,T,J,K)
    measb = meas[None, None]                                       # (1,1,N,T,J,K)
    net = jnp.sum(measb * (w_true * (forf_t - revf_t) + w_false * (forf_f - revf_f)), axis=(3, 4, 5))
    frs = jnp.sum(measb * (w_true * (forf_t + revf_t) + w_false * (forf_f + revf_f)), axis=(3, 4, 5))
    return net, frs


def nuepma_rates_sched(grid: EpsGrid, occ, meps, tcmpl, phie, select: bool = True,
                       n_gl: int = 5, n_lag: int = 5, fd_cap_logC: float = float("inf"),
                       fd_decay: bool = False, monotone_read: bool = False,
                       lag_tail_fix: bool = True, lag_tail_gap: float = _LAG_TAIL_GAP_PMA):
    """νν̄ ↔ e⁻e⁺ rates, the scheduled implementation.

    Arguments and returns as :func:`nuepma_rates`.  ``select=True`` gathers each bin's own regime
    row of the schedule (``use_sched=True``); ``select=False`` runs all four regimes' tiles for
    every bin, masked by regime (``use_sched="union"``).
    """
    if n_gl != n_lag:
        raise ValueError("nuepma_rates_sched needs n_gl == n_lag (the Legendre and Laguerre "
                         "tiles share one node axis); use nuepma_rates otherwise")
    eps = grid.abscissas
    p1 = eps
    N = eps.shape[0]
    logf = jnp.log(occ)

    ecut1 = p1 + meps ** 2 / jnp.where(p1 > 0.0, 4.0 * p1, 1.0)
    den2 = 2.0 * p1 - meps
    ecut2 = 0.5 * (2.0 * p1 - meps + meps ** 2 / jnp.where(den2 != 0.0, den2, 1.0))
    boundo = jnp.stack([jnp.broadcast_to(jnp.asarray(meps, eps.dtype), (N,)), ecut1, ecut2, p1])  # (4,N)

    pcase1 = meps / 2.0
    pcase2 = meps * (1.0 + jnp.sqrt(5.0)) / 4.0
    pcase3 = meps
    reg = jnp.where(p1 < pcase1, 0, jnp.where(p1 < pcase2, 1, jnp.where(p1 < pcase3, 2, 3)))

    tab = jnp.asarray(_TAB); actT = jnp.asarray(_ACT); dcgT = jnp.asarray(_DCG)
    nrow = jnp.arange(N)
    if select:
        f = tab[reg]; act = actT[reg]; dcg = dcgT[reg]
    else:
        f = jnp.broadcast_to(tab[None], (N, 4, _TMAX, _NF)).reshape(N, 4 * _TMAX, _NF)
        onehot = (reg[:, None] == jnp.arange(4)[None, :]).astype(eps.dtype)
        act = (onehot[:, :, None] * actT[None]).reshape(N, 4 * _TMAX)
        dcg = jnp.broadcast_to(dcgT[None], (N, 4, _TMAX)).reshape(N, 4 * _TMAX)

    # lag_tail_fix: lower the genout threshold and drop the _DCG tile where genout takes its tail
    thr_out = lag_tail_gap if lag_tail_fix else 200.0
    if lag_tail_fix:
        genout_lag = ((ecut2 - ecut1) >= thr_out).astype(eps.dtype)        # (N,)
        act = act * jnp.where(dcg > 0.0, (1.0 - genout_lag)[:, None], 1.0)

    out_rule = f[..., 0]; a_out_k = f[..., 1]; b_out_k = f[..., 2]; out_sel = f[..., 3]
    in_rule = f[..., 4]; a_in_k = f[..., 5]; b_in_k = f[..., 6]; in_sel = f[..., 7]
    kind = f[..., 8]; posg = f[..., 9]; clamp = f[..., 10]
    A_out = boundo[a_out_k, nrow[:, None]]
    B_out = boundo[jnp.where(b_out_k < 0, 0, b_out_k), nrow[:, None]]

    return _eval_tiles(grid, occ, logf, meps, tcmpl, phie, A_out, B_out,
                       out_rule, out_sel, a_in_k, b_in_k, in_rule, in_sel, kind, posg, clamp, act,
                       n_gl, n_lag, fd_cap_logC, fd_decay, monotone_read, thr_out)
