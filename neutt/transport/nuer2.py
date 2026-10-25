"""ν–e± scattering, BURST's R2 integral (``trans_nuer2.f90``).

The partner of :mod:`nuer1`: the same process ν(p1) + e±(E2) → e±(E3) + ν(p4) and background
arguments, but with E3 the outer and E2 the inner energy integral and the two coupling structures
exchanged.  Four p1-regimes R1–R4 (not the R1/R2 integrals), split at pcase1 = meps(√5 − 1)/4 <
pcase2 = meps/(2√2) < pcase3 = meps/2, cut the energies at ecut1 = p1 + meps²/(4p1),
ecut2 = p1 + meps(p1 + meps)/(2p1 + meps) and ecut3 = √(p1² + meps²) into fixed tiles.  Both
energies have Gauss–Laguerre tails, and a few E2 segments switch, as in BURST, between Legendre
[a, b] for Δ = b − a < 200 and Laguerre [a, ∞) otherwise.  Mutually exclusive regime masks replace
BURST's per-bin branch.

Two implementations of the same integrand and quadrature: ``nuer2_rates``, the reference (every
regime evaluated and masked; bit-faithful to BURST with ``lag_tail_fix=False``), and
``nuer2_rates_sched``, the scheduled form (``use_sched``: one batched contraction over a per-regime
tile schedule).

``lag_tail_fix``: ecut1 diverges as p1 → 0, so at the lowest bins R1's Legendre segment
[ecut2, ecut1] (``trans_nuer2.f90``'s) spans ~meps²/(4p1) — tens of T_cm once meps ≳ 5 — and its
five nodes miss the exponentially falling integrand (nuer1's defect at meps/2, here at p1 → 0).
With the fix, R1 bins with ecut1 − ecut2 ≥ ``lag_tail_gap`` integrate E3 over [ecut2, ∞) with one
Laguerre rule in place of BURST's Legendre [ecut2, ecut1] + Laguerre [ecut1, ∞) pair, keeping the
pair's E2 tiles: below ecut1 the E2 integral ends at elim1 (Legendre for Δ < 200, else its Laguerre
tail), above it elim1 has diverged and the tail [e3, ∞) applies.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import GFHBAR, PI, SIN2THW
from ..grid import EpsGrid
from ..interp import interp_log_occ_dynamic
from ..grid import legendre_canonical, gauss_laguerre_plain
from .nuer1 import _LAG_TAIL_GAP


def _coupling_tensors_r2(sin2thw: float = SIN2THW):
    """term1 and c2 = term2/meps² per (e⁻/e⁺ nn, flavour m, ν/ν̄ n), as in calc_weight_nuer2.

    nuer1's tensors with the n == nn and n ≠ nn entries exchanged.
    """
    s = sin2thw
    term1 = np.zeros((2, 3, 2))
    c2 = np.zeros((2, 3, 2))
    for n in range(2):
        for m in range(3):
            fla = 1.0 if m == 0 else -1.0
            for nn in range(2):
                if n == nn:
                    term1[nn, m, n] = 128.0 * s ** 2
                    c2[nn, m, n] = (2.0 * s + fla) / (2.0 * s)
                else:
                    term1[nn, m, n] = 32.0 * (2.0 * s + fla) ** 2
                    c2[nn, m, n] = -2.0 * s / (2.0 * s + fla)
    return jnp.asarray(term1), jnp.asarray(c2)


def _fd(E, tcmpl, phie, parfact):
    """e∓ occupation 1/(e^{E·tcmpl ∓ phie} + 1) for parfact = ±1, zero above E = 300.

    Written as sigmoid(−arg): the same value, with a stable gradient.
    """
    arg = E * tcmpl - parfact * phie
    return jnp.where(E <= 300.0, jax.nn.sigmoid(-arg), 0.0)


def _safe_sqrt_diff(u):
    """√max(u, 0) with a zero (not NaN) gradient where u ≤ 0."""
    pos = u > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, u, 1.0)), 0.0)


# Antiderivatives of the angular integrand (BURST's r2int_m_1/2), B = (p1 − E3)² − meps².  The
# second has the opposite sign to nuer1's _R2.
def _R2m1(x, B):
    return 0.25 * (B**2 * x - (2.0 / 3.0) * B * x**3 + 0.2 * x**5)


def _R2m2(x, B):
    return 0.5 * ((1.0 / 3.0) * x**3 - B * x)


def _topbot_r2(kind, p1, e3, q3, e2, q2):
    """Limits (top, bot) of the analytic angular integral for an E2 sub-segment's formula id."""
    if kind == "full":
        return p1 - e3 + e2 + q2, p1 - e3 + e2 - q2
    if kind == "pq3":
        return p1 + q3, p1 - q3
    if kind == "mide2":
        return p1 + q3, e3 - p1 - e2 + q2
    if kind == "mida":
        return p1 - e3 + e2 + q2, q3 - p1
    if kind == "q3q3":
        return p1 + q3, q3 - p1
    raise ValueError(kind)


# BURST's segments per regime: (E3 rule, a3, b3, E2 sub-segments), each E2 sub-segment
# (E2 rule, a2, b2, angular-limit formula of _topbot_r2, select).  E3 bounds are per-bin cut
# energies, E2 bounds per E3 node; b = None marks a Laguerre tail [a, ∞).  ``select`` is the
# Δ = b2 − a2 switch: "selleg" is the Legendre [a2, b2] tile, on for Δ < 200, and "sellag" the
# Laguerre [a2, ∞) tile, on for Δ ≥ 200 (its b2 only sets Δ); None is always on.
_REGIMES = {
    # R1: p1 < pcase1   (ecut3 < ecut2 < ecut1)
    "R1": [
        ("leg", "meps", "ecut3", [("leg", "meps", "e3", "full", None),
                                  ("leg", "e3", "etrans2", "pq3", None),
                                  ("leg", "etrans2", "elim1", "mide2", None)]),
        ("leg", "ecut3", "ecut2", [("leg", "meps", "etrans2", "full", None),
                                   ("leg", "etrans2", "e3", "mida", None),
                                   ("leg", "e3", "elim1", "mide2", None)]),
        ("leg", "ecut2", "ecut1", [("leg", "elim2", "e3", "mida", None),
                                   ("leg", "e3", "elim1", "mide2", "selleg"),
                                   ("lag", "e3", "elim1", "mide2", "sellag")]),
        ("lag", "ecut1", None, [("leg", "elim2", "e3", "mida", None),
                                ("lag", "e3", None, "mide2", None)]),
    ],
    # R2: pcase1 <= p1 < pcase2   (ecut3 < ecut1 < ecut2)
    "R2": [
        ("leg", "meps", "ecut3", [("leg", "meps", "e3", "full", None),
                                  ("leg", "e3", "etrans2", "pq3", None),
                                  ("leg", "etrans2", "elim1", "mide2", None)]),
        ("leg", "ecut3", "ecut1", [("leg", "meps", "etrans2", "full", None),
                                   ("leg", "etrans2", "e3", "mida", None),
                                   ("leg", "e3", "elim1", "mide2", "selleg"),
                                   ("lag", "e3", "elim1", "mide2", "sellag")]),
        ("leg", "ecut1", "ecut2", [("leg", "meps", "etrans2", "full", None),
                                   ("leg", "etrans2", "e3", "mida", None),
                                   ("lag", "e3", None, "mide2", None)]),
        ("lag", "ecut2", None, [("leg", "elim2", "e3", "mida", None),
                                ("lag", "e3", None, "q3q3", None)]),
    ],
    # R3: pcase2 <= p1 < pcase3   (ecut1 < ecut3 < ecut2)
    "R3": [
        ("leg", "meps", "ecut1", [("leg", "meps", "e3", "full", None),
                                  ("leg", "e3", "etrans2", "pq3", None),
                                  ("leg", "etrans2", "elim1", "mide2", "selleg"),
                                  ("lag", "etrans2", "elim1", "mide2", "sellag")]),
        ("leg", "ecut1", "ecut3", [("leg", "meps", "e3", "full", None),
                                   ("leg", "e3", "etrans2", "pq3", None),
                                   ("lag", "etrans2", None, "mide2", None)]),
        ("leg", "ecut3", "ecut2", [("leg", "meps", "etrans2", "full", None),
                                   ("leg", "etrans2", "e3", "mida", None),
                                   ("lag", "e3", None, "mide2", None)]),
        ("lag", "ecut2", None, [("leg", "elim2", "e3", "mida", None),
                                ("lag", "e3", None, "q3q3", None)]),
    ],
    # R4: p1 >= pcase3   (ecut1 < ecut3 < ecut2; elim1 unused, etrans2 only for E3 > ecut1)
    "R4": [
        ("leg", "meps", "ecut1", [("leg", "meps", "e3", "full", None),
                                  ("lag", "e3", None, "pq3", None)]),
        ("leg", "ecut1", "ecut3", [("leg", "meps", "e3", "full", None),
                                   ("leg", "e3", "etrans2", "pq3", "selleg"),
                                   ("lag", "e3", "etrans2", "pq3", "sellag"),
                                   ("lag", "etrans2", None, "mide2", None)]),
        ("leg", "ecut3", "ecut2", [("leg", "meps", "etrans2", "full", None),
                                   ("leg", "etrans2", "e3", "mida", None),
                                   ("lag", "e3", None, "mide2", None)]),
        ("lag", "ecut2", None, [("leg", "elim2", "e3", "mida", None),
                                ("lag", "e3", None, "mide2", None)]),
    ],
}

# lag_tail_fix's wide R1 bins: R1's first two segments, then E3 ∈ [ecut2, ∞) by Laguerre with the
# E2 tiles of R1's last two.  "selleg_w" is "selleg" restricted to E3 < ecut1; "sellag_w" its
# complement (the Laguerre tail, which is also the [ecut1, ∞) segment's E2 tile).
_R1_WIDE = _REGIMES["R1"][:2] + [
    ("lag", "ecut2", None, [("leg", "elim2", "e3", "mida", None),
                            ("leg", "e3", "elim1", "mide2", "selleg_w"),
                            ("lag", "e3", "elim1", "mide2", "sellag_w")]),
]


def _regimes(lag_tail_fix):
    """The segment table, with the wide-R1 regime "R1w" under ``lag_tail_fix``."""
    return {**_REGIMES, "R1w": _R1_WIDE} if lag_tail_fix else _REGIMES


def _regime_masks(p1, meps, ecut1, ecut2, lag_tail_fix, lag_tail_gap):
    """Per-bin 0/1 regime masks (mutually exclusive), keyed as :func:`_regimes`."""
    pcase1 = meps * (jnp.sqrt(5.0) - 1.0) / 4.0
    pcase2 = meps / 2.0 / jnp.sqrt(2.0)
    pcase3 = meps / 2.0
    r1 = p1 < pcase1
    masks = {"R1": r1, "R2": (p1 >= pcase1) & (p1 < pcase2), "R3": (p1 >= pcase2) & (p1 < pcase3),
             "R4": p1 >= pcase3}
    if lag_tail_fix:
        wide = r1 & (ecut1 - ecut2 >= lag_tail_gap)
        masks["R1"] = r1 & ~wide
        masks["R1w"] = wide
    return {k: v.astype(p1.dtype) for k, v in masks.items()}


def _sel_mask(sel, gap, e3, ecut1):
    """0/1 weight of an E2 tile per E3 node: its Δ = b2 − a2 switch (``_REGIMES``), and for the
    wide-R1 tiles also E3 < ecut1.  ``gap`` and ``e3`` are (N, K), ``ecut1`` (N,)."""
    if sel is None:
        return jnp.ones(e3.shape)
    if sel == "selleg":
        return (gap < 200.0).astype(e3.dtype)
    if sel == "sellag":
        return (gap >= 200.0).astype(e3.dtype)
    leg = (gap < 200.0) & (e3 < ecut1[:, None])
    return (leg if sel == "selleg_w" else ~leg).astype(e3.dtype)


def nuer2_rates(grid: EpsGrid, occ, meps, tcmpl, phie, n_gl: int = 5, n_lag: int = 5,
                range_limit=False, fd_cap_logC: float = float("inf"), fd_decay: bool = False,
                monotone_read: bool = False,
                lag_tail_fix: bool = True, lag_tail_gap: float = _LAG_TAIL_GAP):
    """ν–e± (R2) rates, the reference implementation.

    Arguments and returns as :func:`nuer1.nuer1_rates`; ``lag_tail_fix``/``lag_tail_gap`` are the
    wide-R1 Laguerre rule and its ecut1 − ecut2 threshold [T_cm units] (module docstring), and
    ``lag_tail_fix=False`` is bit-faithful to BURST.
    """
    eps = grid.abscissas
    p1 = eps                                   # (N,)
    N = eps.shape[0]
    logf = jnp.log(occ)                        # (3,2,N)
    term1, c2 = _coupling_tensors_r2()
    prefac = GFHBAR / 16.0 / (2.0 * PI) ** 3

    tleg, wleg = legendre_canonical(n_gl)
    xlag, wlag = gauss_laguerre_plain(n_lag)

    binmask = (p1 > 0.0).astype(eps.dtype)     # skip the ε=0 output bin
    p1sq = jnp.where(p1 > 0.0, p1**2, 1.0)     # safe denominator

    # per-bin cut energies
    ecut1 = p1 + meps**2 / jnp.where(p1 > 0.0, 4.0 * p1, 1.0)  # = p1 + meps^2/(4 p1)
    ecut2 = p1 + meps * (p1 + meps) / jnp.where(2.0 * p1 + meps > 0.0, 2.0 * p1 + meps, 1.0)
    ecut3 = jnp.sqrt(p1**2 + meps**2)
    bounds3 = {"meps": jnp.broadcast_to(jnp.asarray(meps, eps.dtype), (N,)),
               "ecut1": ecut1, "ecut2": ecut2, "ecut3": ecut3}

    # regime masks; a bin exactly on a pcase gets zero-width segments (the continuous limit)
    regmasks = _regime_masks(p1, meps, ecut1, ecut2, lag_tail_fix, lag_tail_gap)

    net = jnp.zeros((3, 2, N))
    frs = jnp.zeros((3, 2, N))

    for rname, segments in _regimes(lag_tail_fix).items():
        regmask = regmasks[rname] * binmask                   # (N,)
        for (e3rule, a3k, b3k, subsegs) in segments:
            # E3 (outer) nodes and weights, (N, K)
            a3 = bounds3[a3k]                                  # (N,)
            if e3rule == "leg":
                b3 = bounds3[b3k]
                mid3 = 0.5 * (a3 + b3)
                half3 = 0.5 * (b3 - a3)
                E3 = mid3[:, None] + half3[:, None] * tleg[None, :]   # (N, K)
                e3w = wleg[None, :] * half3[:, None]
            else:  # Laguerre tail [a3, ∞)
                E3 = a3[:, None] + xlag[None, :]                      # (N, K)
                e3w = jnp.broadcast_to(wlag[None, :], E3.shape)
            E3 = jnp.maximum(E3, meps)
            q3 = _safe_sqrt_diff(E3**2 - meps**2)                     # (N, K)
            d = E3 + q3 - 2.0 * p1[:, None]
            etrans2 = 0.5 * (E3 + q3 - 2.0 * p1[:, None] + meps**2 / jnp.where(d != 0.0, d, 1.0))
            db = E3 - q3 - 2.0 * p1[:, None]
            elim1raw = 0.5 * (E3 - q3 - 2.0 * p1[:, None] + meps**2 / jnp.where(db != 0.0, db, 1.0))
            elim1 = jnp.where(elim1raw < 0.0, jnp.maximum(E3, etrans2), elim1raw)
            elim2 = etrans2
            fe3 = jnp.stack([_fd(E3, tcmpl, phie, 1.0), _fd(E3, tcmpl, phie, -1.0)])  # (nn,N,K)
            kinemfact1 = e3w / p1sq[:, None]                          # (N, K)
            Bcoef = (p1[:, None] - E3) ** 2 - meps**2                 # (N, K)

            bounds2 = {"meps": jnp.broadcast_to(jnp.asarray(meps, eps.dtype), E3.shape),
                       "e3": E3, "etrans2": etrans2, "elim1": elim1, "elim2": elim2}

            for (e2rule, a2k, b2k, kind, sel) in subsegs:
                a2 = bounds2[a2k]                                     # (N, K)
                if e2rule == "leg":
                    b2 = bounds2[b2k]
                    mid2 = 0.5 * (a2 + b2)
                    half2 = 0.5 * (b2 - a2)
                    E2 = mid2[:, :, None] + half2[:, :, None] * tleg[None, None, :]   # (N,K,J)
                    e2w = wleg[None, None, :] * half2[:, :, None]
                else:  # Laguerre tail [a2, ∞)
                    E2 = a2[:, :, None] + xlag[None, None, :]                         # (N,K,J)
                    e2w = jnp.broadcast_to(wlag[None, None, :], E2.shape)
                E2 = jnp.maximum(E2, meps)
                q2 = _safe_sqrt_diff(E2**2 - meps**2)                                 # (N,K,J)

                top, bot = _topbot_r2(kind, p1[:, None, None], E3[:, :, None],
                                      q3[:, :, None], E2, q2)                          # (N,K,J)
                Be = Bcoef[:, :, None]
                I1 = _R2m1(top, Be) - _R2m1(bot, Be)
                I2 = _R2m2(top, Be) - _R2m2(bot, Be)
                # BURST's weight2 per node, (nn, m, n, N, K, J): term1·(I1 + c2·meps²·I2)
                w2 = term1[:, :, :, None, None, None] * (
                    I1[None, None, None] + c2[:, :, :, None, None, None] * meps**2 * I2[None, None, None])

                fe2 = jnp.stack([_fd(E2, tcmpl, phie, 1.0), _fd(E2, tcmpl, phie, -1.0)])  # (nn,N,K,J)
                p4 = jnp.maximum(p1[:, None, None] + E2 - E3[:, :, None], 0.0)            # (N,K,J)
                fnu4 = jnp.exp(interp_log_occ_dynamic(logf, eps, p4, range_limit=range_limit,
                                                      fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                                                      monotone_read=monotone_read))  # (3,2,N,K,J)

                # occupation factors (get_f_nuer2 = get_f_nuer1), axes (nn, m, n, N, K, J)
                fnu1 = occ[None, :, :, :, None, None]              # (1,m,n,N,1,1)
                fe2b = fe2[:, None, None]                          # (nn,1,1,N,K,J)
                fe3b = fe3[:, None, None, :, :, None]              # (nn,1,1,N,K,1)
                fnu4b = fnu4[None]                                 # (1,m,n,N,K,J)
                forf = (1.0 - fnu1) * (1.0 - fe2b) * fe3b * fnu4b
                revf = fnu1 * fe2b * (1.0 - fe3b) * (1.0 - fnu4b)
                Fnet = forf - revf
                Ffrs = forf + revf

                gap = bounds2[b2k] - a2 if sel is not None else None   # (N,K)
                extra = _sel_mask(sel, gap, E3, ecut1)

                meas = (prefac * regmask[:, None, None]
                        * (kinemfact1 * extra)[:, :, None] * e2w)   # (N,K,J)
                measb = meas[None, None, None]                      # (1,1,1,N,K,J)
                net = net + jnp.sum(measb * w2 * Fnet, axis=(0, 4, 5))   # sum nn,K,J -> (m,n,N)
                frs = frs + jnp.sum(measb * w2 * Ffrs, axis=(0, 4, 5))

    return net, frs


def nuer2_dfdt(grid, occ, meps, tcmpl, phie, t_cm_mev, **kw):
    """df/dt from ν–e± (R2) scattering, in s⁻¹ (T_cm⁵·net)."""
    net, _ = nuer2_rates(grid, occ, meps, tcmpl, phie, **kw)
    return t_cm_mev**5 * net


# ---- the scheduled implementation ---------------------------------------------------------------
# Each regime's (E3-segment × E2-sub-segment) tiles from _REGIMES, 11–12 per regime, padded to 12.
# A tile is (e3_rule, a3, b3, e2_rule, a2, b2, tb, sel):
#   rule     0 = Legendre [a, b], 1 = Laguerre [a, ∞) with b = −1 (a sel = 2 tile keeps b2 for Δ)
#   a3, b3   E3 bound: 0 = meps, 1 = ecut1, 2 = ecut2, 3 = ecut3
#   a2, b2   E2 bound: 0 = meps, 1 = e3, 2 = etrans2, 3 = elim1, 4 = elim2
#   tb       angular limits (_topbot_r2): 0 = full, 1 = pq3, 2 = mide2, 3 = mida, 4 = q3q3
#   sel      Δ = b2 − a2 switch: 0 = always on, 1 = on for Δ < 200, 2 = on for Δ ≥ 200,
#            3 = on for Δ < 200 and E3 < ecut1, 4 = its complement (the wide-R1 tiles)
# "R1w" is the lag_tail_fix row of a wide R1 bin (_R1_WIDE).

_L, _G = 0, 1   # Legendre, Laguerre
_TILES = {
    "R1": [
        (_L, 0, 3, _L, 0, 1, 0, 0), (_L, 0, 3, _L, 1, 2, 1, 0), (_L, 0, 3, _L, 2, 3, 2, 0),
        (_L, 3, 2, _L, 0, 2, 0, 0), (_L, 3, 2, _L, 2, 1, 3, 0), (_L, 3, 2, _L, 1, 3, 2, 0),
        (_L, 2, 1, _L, 4, 1, 3, 0), (_L, 2, 1, _L, 1, 3, 2, 1), (_L, 2, 1, _G, 1, 3, 2, 2),
        (_G, 1, -1, _L, 4, 1, 3, 0), (_G, 1, -1, _G, 1, -1, 2, 0),
    ],
    "R2": [
        (_L, 0, 3, _L, 0, 1, 0, 0), (_L, 0, 3, _L, 1, 2, 1, 0), (_L, 0, 3, _L, 2, 3, 2, 0),
        (_L, 3, 1, _L, 0, 2, 0, 0), (_L, 3, 1, _L, 2, 1, 3, 0), (_L, 3, 1, _L, 1, 3, 2, 1),
        (_L, 3, 1, _G, 1, 3, 2, 2),
        (_L, 1, 2, _L, 0, 2, 0, 0), (_L, 1, 2, _L, 2, 1, 3, 0), (_L, 1, 2, _G, 1, -1, 2, 0),
        (_G, 2, -1, _L, 4, 1, 3, 0), (_G, 2, -1, _G, 1, -1, 4, 0),
    ],
    "R3": [
        (_L, 0, 1, _L, 0, 1, 0, 0), (_L, 0, 1, _L, 1, 2, 1, 0), (_L, 0, 1, _L, 2, 3, 2, 1),
        (_L, 0, 1, _G, 2, 3, 2, 2),
        (_L, 1, 3, _L, 0, 1, 0, 0), (_L, 1, 3, _L, 1, 2, 1, 0), (_L, 1, 3, _G, 2, -1, 2, 0),
        (_L, 3, 2, _L, 0, 2, 0, 0), (_L, 3, 2, _L, 2, 1, 3, 0), (_L, 3, 2, _G, 1, -1, 2, 0),
        (_G, 2, -1, _L, 4, 1, 3, 0), (_G, 2, -1, _G, 1, -1, 4, 0),
    ],
    "R4": [
        (_L, 0, 1, _L, 0, 1, 0, 0), (_L, 0, 1, _G, 1, -1, 1, 0),
        (_L, 1, 3, _L, 0, 1, 0, 0), (_L, 1, 3, _L, 1, 2, 1, 1), (_L, 1, 3, _G, 1, 2, 1, 2),
        (_L, 1, 3, _G, 2, -1, 2, 0),
        (_L, 3, 2, _L, 0, 2, 0, 0), (_L, 3, 2, _L, 2, 1, 3, 0), (_L, 3, 2, _G, 1, -1, 2, 0),
        (_G, 2, -1, _L, 4, 1, 3, 0), (_G, 2, -1, _G, 1, -1, 2, 0),
    ],
}
_TILES["R1w"] = _TILES["R1"][:6] + [
    (_G, 2, -1, _L, 4, 1, 3, 0), (_G, 2, -1, _L, 1, 3, 2, 3), (_G, 2, -1, _G, 1, 3, 2, 4),
]
_TMAX = 12
_NF = 8   # fields per tile


def _build_table():
    """_TILES as a (regime, tile, field) int table and a (regime, tile) 0/1 mask of real tiles;
    rows R1–R4, then R1w."""
    tab = np.zeros((5, _TMAX, _NF), np.int32)
    act = np.zeros((5, _TMAX), np.float64)
    for r, rn in enumerate(["R1", "R2", "R3", "R4", "R1w"]):
        for t, tile in enumerate(_TILES[rn]):
            tab[r, t] = tile
            act[r, t] = 1.0
    return tab, act


_TAB, _ACT = _build_table()


def _eval_tiles(grid, occ, logf, meps, tcmpl, phie,
                A3, B3, e3rule, a2k, b2k, e2rule, tb, sel, active, n_gl=5, n_lag=5,
                fd_cap_logC=float("inf"), fd_decay=False, monotone_read=False, wide_tiles=False):
    """Sum all tiles of all bins in one contraction; returns (net, frs), each (3, 2, N).

    Tile fields are (N, T): A3, B3 the resolved E3 bounds, the rest int codes, ``active`` the 0/1
    tile mask.  ``wide_tiles`` (static) evaluates the sel = 3, 4 switch of the R1w row.
    """
    eps = grid.abscissas
    p1 = eps
    N = eps.shape[0]
    term1, c2 = _coupling_tensors_r2()
    prefac = GFHBAR / 16.0 / (2.0 * PI) ** 3
    tleg, wleg = legendre_canonical(n_gl)
    xlag, wlag = gauss_laguerre_plain(n_lag)
    binmask = (p1 > 0.0).astype(eps.dtype)
    p1sq = jnp.where(p1 > 0.0, p1 ** 2, 1.0)
    msq = meps ** 2

    # E3 (outer) nodes, (N, T, K): both rules evaluated, one kept per tile
    is_lag3 = (e3rule == 1)[:, :, None]
    mid3 = 0.5 * (A3 + B3); half3 = 0.5 * (B3 - A3)
    E3_leg = mid3[:, :, None] + half3[:, :, None] * tleg[None, None, :]
    e3w_leg = wleg[None, None, :] * half3[:, :, None]
    E3_lag = A3[:, :, None] + xlag[None, None, :]
    e3w_lag = jnp.broadcast_to(wlag[None, None, :], E3_lag.shape)
    E3 = jnp.where(is_lag3, E3_lag, E3_leg)
    e3w = jnp.where(is_lag3, e3w_lag, e3w_leg)
    E3 = jnp.maximum(E3, meps)
    q3 = _safe_sqrt_diff(E3 ** 2 - msq)
    p1_3 = p1[:, None, None]
    dd = E3 + q3 - 2.0 * p1_3
    etrans2 = 0.5 * (E3 + q3 - 2.0 * p1_3 + msq / jnp.where(dd != 0.0, dd, 1.0))
    db = E3 - q3 - 2.0 * p1_3
    elim1raw = 0.5 * (E3 - q3 - 2.0 * p1_3 + msq / jnp.where(db != 0.0, db, 1.0))
    elim1 = jnp.where(elim1raw < 0.0, jnp.maximum(E3, etrans2), elim1raw)
    elim2 = etrans2
    Bcoef = (p1_3 - E3) ** 2 - msq                                 # (N,T,K)
    fe3 = jnp.stack([_fd(E3, tcmpl, phie, 1.0), _fd(E3, tcmpl, phie, -1.0)])  # (nn,N,T,K)
    kinemfact1 = e3w / p1sq[:, None, None]                         # (N,T,K)

    # E2 bounds, picked by key from [meps, e3, etrans2, elim1, elim2]
    e2bounds = jnp.stack([jnp.broadcast_to(jnp.asarray(meps, eps.dtype), E3.shape),
                          E3, etrans2, elim1, elim2])             # (5,N,T,K)
    a2idx = jnp.broadcast_to(a2k[None, :, :, None], (1,) + E3.shape)
    a2 = jnp.take_along_axis(e2bounds, a2idx, axis=0)[0]          # (N,T,K)
    b2k_safe = jnp.where(b2k < 0, 0, b2k)
    b2idx = jnp.broadcast_to(b2k_safe[None, :, :, None], (1,) + E3.shape)
    b2 = jnp.take_along_axis(e2bounds, b2idx, axis=0)[0]          # (N,T,K)

    # E2 (inner) nodes, (N, T, K, J)
    is_lag2 = (e2rule == 1)[:, :, None, None]
    mid2 = 0.5 * (a2 + b2); half2 = 0.5 * (b2 - a2)
    E2_leg = mid2[:, :, :, None] + half2[:, :, :, None] * tleg[None, None, None, :]
    e2w_leg = wleg[None, None, None, :] * half2[:, :, :, None]
    E2_lag = a2[:, :, :, None] + xlag[None, None, None, :]
    e2w_lag = jnp.broadcast_to(wlag[None, None, None, :], E2_lag.shape)
    E2 = jnp.where(is_lag2, E2_lag, E2_leg)
    e2w = jnp.where(is_lag2, e2w_lag, e2w_leg)
    E2 = jnp.maximum(E2, meps)
    q2 = _safe_sqrt_diff(E2 ** 2 - msq)                           # (N,T,K,J)

    # angular limits by tb code, (N, T, K, J)
    p1_4 = p1[:, None, None, None]
    e3_4 = E3[:, :, :, None]; q3_4 = q3[:, :, :, None]
    full_t = p1_4 - e3_4 + E2 + q2; full_b = p1_4 - e3_4 + E2 - q2
    pq3_t = p1_4 + q3_4; pq3_b = p1_4 - q3_4
    mide2_b = e3_4 - p1_4 - E2 + q2
    mida_t = p1_4 - e3_4 + E2 + q2; mida_b = q3_4 - p1_4
    q3q3_b = q3_4 - p1_4
    tb4 = tb[:, :, None, None]
    top = jnp.where(tb4 == 0, full_t,
          jnp.where(tb4 == 1, pq3_t,
          jnp.where(tb4 == 2, pq3_t,           # mide2 top = p1+q3
          jnp.where(tb4 == 3, mida_t, pq3_t))))  # q3q3 top = p1+q3
    bot = jnp.where(tb4 == 0, full_b,
          jnp.where(tb4 == 1, pq3_b,
          jnp.where(tb4 == 2, mide2_b,
          jnp.where(tb4 == 3, mida_b, q3q3_b))))
    Be = Bcoef[:, :, :, None]
    I1 = _R2m1(top, Be) - _R2m1(bot, Be)
    I2 = _R2m2(top, Be) - _R2m2(bot, Be)
    w2 = term1[:, :, :, None, None, None, None] * (
        I1[None, None, None] + c2[:, :, :, None, None, None, None] * msq * I2[None, None, None])

    fe2 = jnp.stack([_fd(E2, tcmpl, phie, 1.0), _fd(E2, tcmpl, phie, -1.0)])   # (nn,N,T,K,J)
    p4 = jnp.maximum(p1_4 + E2 - e3_4, 0.0)                                     # (N,T,K,J)
    fnu4 = jnp.exp(interp_log_occ_dynamic(logf, eps, p4, fd_cap_logC=fd_cap_logC,
                                          fd_decay=fd_decay, monotone_read=monotone_read))  # (3,2,N,T,K,J)

    fnu1 = occ[None, :, :, :, None, None, None]        # (1,m,n,N,1,1,1)
    fe2b = fe2[:, None, None]                          # (nn,1,1,N,T,K,J)
    fe3b = fe3[:, None, None, :, :, :, None]           # (nn,1,1,N,T,K,1)
    fnu4b = fnu4[None]                                 # (1,m,n,N,T,K,J)
    forf = (1.0 - fnu1) * (1.0 - fe2b) * fe3b * fnu4b
    revf = fnu1 * fe2b * (1.0 - fe3b) * (1.0 - fnu4b)
    Fnet = forf - revf
    Ffrs = forf + revf

    gap = b2 - a2                                      # (N,T,K)
    sel3 = sel[:, :, None]
    extra = jnp.where(sel3 == 0, 1.0,
                      jnp.where(sel3 == 1, (gap < 200.0).astype(eps.dtype),
                                (gap >= 200.0).astype(eps.dtype)))
    if wide_tiles:
        # the wide-R1 switch: Δ < 200 and below ecut1
        ecut1 = p1 + msq / jnp.where(p1 > 0.0, 4.0 * p1, 1.0)
        leg_w = ((gap < 200.0) & (E3 < ecut1[:, None, None])).astype(eps.dtype)
        extra = jnp.where(sel3 == 3, leg_w, jnp.where(sel3 == 4, 1.0 - leg_w, extra))
    meas = (prefac * (active * binmask[:, None])[:, :, None, None]
            * (kinemfact1 * extra)[:, :, :, None] * e2w)           # (N,T,K,J)
    measb = meas[None, None, None]                                 # (1,1,1,N,T,K,J)
    net = jnp.sum(measb * w2 * Fnet, axis=(0, 4, 5, 6))            # (m,n,N)
    frs = jnp.sum(measb * w2 * Ffrs, axis=(0, 4, 5, 6))
    return net, frs


def nuer2_rates_sched(grid: EpsGrid, occ, meps, tcmpl, phie, select: bool = True,
                      n_gl: int = 5, n_lag: int = 5, fd_cap_logC: float = float("inf"),
                      fd_decay: bool = False, monotone_read: bool = False,
                      lag_tail_fix: bool = True, lag_tail_gap: float = _LAG_TAIL_GAP):
    """ν–e± (R2) rates, the scheduled implementation.

    Arguments and returns as :func:`nuer2_rates`.  ``select=True`` gathers each bin's own regime
    row of the schedule (``use_sched=True``); ``select=False`` runs all four regimes' tiles for
    every bin, masked by regime (``use_sched="union"``).  A wide R1 bin takes the R1w row in place
    of R1's, so the fix adds no tiles.
    """
    if n_gl != n_lag:
        raise ValueError("nuer2_rates_sched needs n_gl == n_lag (the Legendre and Laguerre "
                         "tiles share one node axis); use nuer2_rates otherwise")
    eps = grid.abscissas
    p1 = eps
    N = eps.shape[0]
    logf = jnp.log(occ)

    # cut energies, indexed by the a3/b3 keys
    ecut1 = p1 + meps ** 2 / jnp.where(p1 > 0.0, 4.0 * p1, 1.0)
    ecut2 = p1 + meps * (p1 + meps) / jnp.where(2.0 * p1 + meps > 0.0, 2.0 * p1 + meps, 1.0)
    ecut3 = jnp.sqrt(p1 ** 2 + meps ** 2)
    cutvals = jnp.stack([jnp.broadcast_to(jnp.asarray(meps, eps.dtype), (N,)),
                         ecut1, ecut2, ecut3])                    # (4,N)

    pcase1 = meps * (jnp.sqrt(5.0) - 1.0) / 4.0
    pcase2 = meps / 2.0 / jnp.sqrt(2.0)
    pcase3 = meps / 2.0
    reg = jnp.where(p1 < pcase1, 0, jnp.where(p1 < pcase2, 1,
                    jnp.where(p1 < pcase3, 2, 3)))                # (N,) regime index 0..3

    tab = jnp.asarray(_TAB)        # (5,_TMAX,8), rows R1–R4, R1w
    actT = jnp.asarray(_ACT)       # (5,_TMAX)
    nrow = jnp.arange(N)
    row = reg                      # (N,) schedule row: the regime, or 4 = R1w
    if lag_tail_fix:
        row = jnp.where((reg == 0) & (ecut1 - ecut2 >= lag_tail_gap), 4, reg)

    if select:
        f = tab[row]               # (N,_TMAX,8) gather regime's tiles
        act = actT[row]            # (N,_TMAX)
    else:
        f4 = jnp.broadcast_to(tab[None, :4], (N, 4, _TMAX, 8))
        a4 = jnp.broadcast_to(actT[None, :4], (N, 4, _TMAX))
        if lag_tail_fix:           # a wide R1 bin's R1 block is the R1w row
            w = (row == 4)[:, None, None]
            f4 = f4.at[:, 0].set(jnp.where(w, tab[4][None], tab[0][None]))
            a4 = a4.at[:, 0].set(jnp.where(w[:, :, 0], actT[4][None], actT[0][None]))
        f = f4.reshape(N, 4 * _TMAX, 8)
        onehot = (reg[:, None] == jnp.arange(4)[None, :]).astype(eps.dtype)   # (N,4)
        act = (onehot[:, :, None] * a4).reshape(N, 4 * _TMAX)                 # (N,48)

    e3rule = f[..., 0]; a3k = f[..., 1]; b3k = f[..., 2]
    e2rule = f[..., 3]; a2k = f[..., 4]; b2k = f[..., 5]
    tb = f[..., 6]; sel = f[..., 7]
    A3 = cutvals[a3k, nrow[:, None]]                              # (N,T)
    B3 = cutvals[jnp.where(b3k < 0, 0, b3k), nrow[:, None]]       # (N,T) (unused where lag)

    return _eval_tiles(grid, occ, logf, meps, tcmpl, phie,
                       A3, B3, e3rule, a2k, b2k, e2rule, tb, sel, act, n_gl, n_lag, fd_cap_logC,
                       fd_decay, monotone_read, wide_tiles=lag_tail_fix)
