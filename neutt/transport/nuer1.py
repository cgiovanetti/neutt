"""ν–e± scattering, BURST's R1 integral (``trans_nuer1.f90``).

ν(p1) + e±(E2) → e±(E3) + ν(p4), p4 = p1 + E2 − E3, with E2 the outer and E3 the inner energy
integral and the angular integral done analytically.  The rates depend on the background
(meps = m_e/T_cm, tcmpl = T_cm/T_pl, phie = μ_e/T_pl), so unlike the ν–ν kernels they are jittable
functions, not precomputed tensors.  Two p1-regimes, A: p1 < meps/2 and B: p1 ≥ meps/2, split the
energies at ecut3 = √(p1² + meps²) and (A only) ecut1 = meps + 2p1²/(meps − 2p1) into fixed tiles:
Gauss–Legendre on finite segments, Gauss–Laguerre on [a, ∞) tails.  BURST's per-bin branches become
regime masks, so there is no data-dependent control flow.

Two implementations of the same integrand and quadrature: ``nuer1_rates``, the reference (a loop
over the segments, each masked by regime; bit-faithful to BURST with ``lag_tail_fix=False``), and
``nuer1_rates_sched``, the scheduled form (``use_sched``: the tiles flattened into a per-regime
schedule and evaluated as one batched contraction, collapsing the kernel launches).

``lag_tail_fix``: ecut1 diverges as p1 → meps/2 from below, so BURST's Legendre segment
[ecut3, ecut1] spans a near-infinite interval there and badly under-resolves the rate.  With the
fix, regime-A bins with ecut1 − ecut3 ≥ ``lag_tail_gap`` integrate [ecut3, ∞) with the regime-B
Laguerre tail instead, the continuous p1 → meps/2 limit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import GFHBAR, PI, SIN2THW
from ..grid import EpsGrid
from ..interp import interp_log_occ_dynamic
from ..grid import legendre_canonical, gauss_laguerre_plain


def _coupling_tensors(sin2thw: float = SIN2THW):
    """term1 and c2 = term2/meps² per (e⁻/e⁺ nn, flavour m, ν/ν̄ n), as in calc_weight_nuer1."""
    s = sin2thw
    term1 = np.zeros((2, 3, 2))
    c2 = np.zeros((2, 3, 2))
    for n in range(2):
        for m in range(3):
            fla = 1.0 if m == 0 else -1.0
            for nn in range(2):
                if n == nn:
                    term1[nn, m, n] = 32.0 * (2.0 * s + fla) ** 2
                    c2[nn, m, n] = -2.0 * s / (2.0 * s + fla)
                else:
                    term1[nn, m, n] = 128.0 * s ** 2
                    c2[nn, m, n] = (2.0 * s + fla) / (2.0 * s)
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


# Antiderivatives of the angular integrand (BURST's r1int_m_1/2), A = (p1 + E2)² − meps².
def _R1(x, A):
    return 0.25 * (A**2 * x - (2.0 / 3.0) * A * x**3 + 0.2 * x**5)


def _R2(x, A):
    return 0.5 * (A * x - (1.0 / 3.0) * x**3)


# BURST's E2 segments: (name, E2 rule, a2, b2, regime, E3 sub-segments).  E2 bounds are per-bin cut
# energies, b2 = None marking a Laguerre tail [a2, ∞); each E3 sub-segment is (a3, b3, angular-limit
# formula of _topbot), Legendre, with bounds per E2 node.  Regime "all" applies to A and B.
_E2_SEGMENTS = [
    ("S1", "leg", "meps", "ecut3", "all", [("meps", "e2", "full"),
                                           ("e2", "etrans2", "mid1"),
                                           ("etrans2", "elim1", "mid2")]),
    ("S2", "leg", "ecut3", "ecut1", "A",  [("meps", "etrans2", "full"),
                                           ("etrans2", "e2", "mid3"),
                                           ("e2", "elim1", "mid2")]),
    ("S3", "lag", "ecut1", None,    "A",  [("etrans2", "e2", "mid3"),
                                           ("e2", "elim1", "mid2")]),
    ("S4", "lag", "ecut3", None,    "B",  [("meps", "etrans2", "full"),
                                           ("etrans2", "e2", "mid3"),
                                           ("e2", "elim1", "mid2")]),
]

# The segments with lag_tail_fix: regime A splits by the ``wide`` mask into narrow bins, which keep
# S2 + S3, and wide bins (ecut1 − ecut3 ≥ lag_tail_gap), which take S5 = the regime-B tail S4.
_E2_SEGMENTS_FIXED = [
    ("S1", "leg", "meps", "ecut3", "all",      [("meps", "e2", "full"),
                                                ("e2", "etrans2", "mid1"),
                                                ("etrans2", "elim1", "mid2")]),
    ("S2", "leg", "ecut3", "ecut1", "A_narrow", [("meps", "etrans2", "full"),
                                                ("etrans2", "e2", "mid3"),
                                                ("e2", "elim1", "mid2")]),
    ("S3", "lag", "ecut1", None,    "A_narrow", [("etrans2", "e2", "mid3"),
                                                ("e2", "elim1", "mid2")]),
    ("S4", "lag", "ecut3", None,    "B",        [("meps", "etrans2", "full"),
                                                ("etrans2", "e2", "mid3"),
                                                ("e2", "elim1", "mid2")]),
    ("S5", "lag", "ecut3", None,    "A_wide",   [("meps", "etrans2", "full"),
                                                ("etrans2", "e2", "mid3"),
                                                ("e2", "elim1", "mid2")]),
]

# ecut1 − ecut3 [T_cm units] from which a regime-A bin takes the Laguerre tail; the jump at the
# switch is smallest for gaps of 3–6.  Detailed balance holds exactly on either side.
_LAG_TAIL_GAP = 6.0


def _topbot(kind, p1, e2, q2, e3, q3):
    """Limits (top, bot) of the analytic angular integral for an E3 sub-segment's formula id."""
    if kind == "full":
        return p1 + e2 - e3 + q3, p1 + e2 - e3 - q3
    if kind == "mid1":
        return p1 + q2, p1 - q2
    if kind == "mid2":
        return p1 + q2, e3 + q3 - p1 - e2
    if kind == "mid3":
        return p1 + e2 - e3 + q3, q2 - p1
    raise ValueError(kind)


def nuer1_rates(grid: EpsGrid, occ, meps, tcmpl, phie, n_gl: int = 5, n_lag: int = 5,
                range_limit=False, fd_cap_logC: float = float("inf"), fd_decay: bool = False,
                monotone_read: bool = False,
                lag_tail_fix: bool = True, lag_tail_gap: float = _LAG_TAIL_GAP):
    """ν–e± (R1) rates, the reference implementation.

    Parameters
    ----------
    occ : (3, 2, N) array
        Neutrino occupations (flavour, ν/ν̄, bin).
    meps, tcmpl, phie : float
        m_e/T_cm, T_cm/T_pl and μ_e/T_pl (may be traced).
    n_gl, n_lag : int
        Legendre / Laguerre points (BURST's lgndrbins = lagurbins = 5).
    range_limit, fd_cap_logC, fd_decay, monotone_read
        Off-grid closures of the f(p4) read (see ``interp``); the defaults are BURST's read.
        ``range_limit`` is the dark sectors' (strong-χ) window-max cap.
    lag_tail_fix, lag_tail_gap
        The Laguerre tail for wide regime-A bins and its ecut1 − ecut3 threshold [T_cm units]
        (module docstring); ``lag_tail_fix=False`` is bit-faithful to BURST.

    Returns
    -------
    (net, frs) : (3, 2, N) arrays
        forward − reverse and forward + reverse, summed over e⁻/e⁺.  T_cm⁵·net is df/dt [s⁻¹];
        net is also part of the plasma energy exchange (``leg4`` in ``assemble``).
    """
    eps = grid.abscissas
    p1 = eps                                   # (N,)  output-bin momentum
    N = eps.shape[0]
    logf = jnp.log(occ)                        # (3,2,N)
    term1, c2 = _coupling_tensors()
    prefac = GFHBAR / 16.0 / (2.0 * PI) ** 3

    tleg, wleg = legendre_canonical(n_gl)      # E2 and E3 finite segments
    xlag, wlag = gauss_laguerre_plain(n_lag)   # E2 tails, shifted to [a2, ∞)

    ecut3 = jnp.sqrt(p1**2 + meps**2)                              # (N,)
    denom_a = meps - 2.0 * p1
    ecut1 = meps + 2.0 * p1**2 / jnp.where(denom_a != 0.0, denom_a, 1.0)  # valid in regime A
    regimeA = p1 < (meps / 2.0)                                   # (N,) bool
    binmask = (p1 > 0.0).astype(eps.dtype)                        # skip the ε=0 output bin
    p1sq = jnp.where(p1 > 0.0, p1**2, 1.0)                        # safe denominator

    bounds_e2 = {"meps": jnp.broadcast_to(meps, (N,)), "ecut3": ecut3, "ecut1": ecut1}

    # lag_tail_fix is static: False traces BURST's four segments unchanged (bit-identical)
    if lag_tail_fix:
        segments = _E2_SEGMENTS_FIXED
        regA = regimeA.astype(eps.dtype)
        wide = regA * (ecut1 - ecut3 >= lag_tail_gap).astype(eps.dtype)   # (N,) float
        regmask_table = {"all": jnp.ones((N,)), "B": (~regimeA).astype(eps.dtype),
                         "A_narrow": regA * (1.0 - wide), "A_wide": wide}
    else:
        segments = _E2_SEGMENTS
        regmask_table = {"all": jnp.ones((N,)), "A": regimeA.astype(eps.dtype),
                         "B": (~regimeA).astype(eps.dtype)}

    net = jnp.zeros((3, 2, N))
    frs = jnp.zeros((3, 2, N))

    for (name, rule, a2k, b2k, regime, subsegs) in segments:
        # E2 (outer) nodes and weights, (N, A)
        a2 = bounds_e2[a2k]                                       # (N,)
        if rule == "leg":
            b2 = bounds_e2[b2k]
            mid2 = 0.5 * (a2 + b2)
            half2 = 0.5 * (b2 - a2)
            E2 = mid2[:, None] + half2[:, None] * tleg[None, :]   # (N, n_gl)
            e2w = wleg[None, :] * half2[:, None]
        else:  # Laguerre tail [a2, ∞)
            E2 = a2[:, None] + xlag[None, :]                      # (N, n_lag)
            e2w = jnp.broadcast_to(wlag[None, :], E2.shape)
        E2 = jnp.maximum(E2, meps)                                # clamp (no-op for active tiles)
        q2 = _safe_sqrt_diff(E2**2 - meps**2)          # (N, A)
        d = 2.0 * p1[:, None] + E2 - q2
        etrans2 = 0.5 * (2.0 * p1[:, None] + E2 - q2 + meps**2 / jnp.where(d != 0.0, d, 1.0))
        dl = 2.0 * p1[:, None] + E2 + q2
        elim1 = 0.5 * (2.0 * p1[:, None] + E2 + q2 + meps**2 / jnp.where(dl != 0.0, dl, 1.0))
        A2 = (p1[:, None] + E2) ** 2 - meps**2                    # (N, A)
        # e2 occupation, both lepton parities: fe2[nn, N, A]
        fe2 = jnp.stack([_fd(E2, tcmpl, phie, 1.0), _fd(E2, tcmpl, phie, -1.0)])

        bounds_e3 = {"meps": jnp.broadcast_to(meps, E2.shape), "e2": E2,
                     "etrans2": etrans2, "elim1": elim1}

        mask = regmask_table[regime]

        for (a3k, b3k, tbkind) in subsegs:
            a3 = bounds_e3[a3k]                                   # (N, A)
            b3 = bounds_e3[b3k]
            mid3 = 0.5 * (a3 + b3)
            half3 = 0.5 * (b3 - a3)
            E3 = mid3[:, :, None] + half3[:, :, None] * tleg[None, None, :]  # (N, A, B)
            e3w = wleg[None, None, :] * half3[:, :, None]
            E3 = jnp.maximum(E3, meps)
            q3 = _safe_sqrt_diff(E3**2 - meps**2)
            top, bot = _topbot(tbkind, p1[:, None, None], E2[:, :, None],
                               q2[:, :, None], E3, q3)            # (N, A, B)
            I1 = _R1(top, A2[:, :, None]) - _R1(bot, A2[:, :, None])
            I2 = _R2(top, A2[:, :, None]) - _R2(bot, A2[:, :, None])
            # BURST's weight3 per node, (nn, m, n, N, A, B): term1·(I1 + c2·meps²·I2)
            w3 = term1[:, :, :, None, None, None] * (
                I1[None, None, None] + c2[:, :, :, None, None, None] * meps**2 * I2[None, None, None])

            fe3 = jnp.stack([_fd(E3, tcmpl, phie, 1.0), _fd(E3, tcmpl, phie, -1.0)])  # (nn,N,A,B)
            p4 = jnp.maximum(p1[:, None, None] + E2[:, :, None] - E3, 0.0)            # (N,A,B)
            fnu4 = jnp.exp(interp_log_occ_dynamic(logf, eps, p4, range_limit=range_limit,
                                                  fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                                                  monotone_read=monotone_read))  # (3,2,N,A,B)

            # occupation factors (get_f_nuer1), axes (nn, m, n, N, A, B)
            fnu1 = occ[None, :, :, :, None, None]               # (1,m,n,N,1,1)
            fe2b = fe2[:, None, None, :, :, None]               # (nn,1,1,N,A,1)
            fe3b = fe3[:, None, None]                           # (nn,1,1,N,A,B)
            fnu4b = fnu4[None]                                  # (1,m,n,N,A,B)
            forf = (1.0 - fnu1) * (1.0 - fe2b) * fe3b * fnu4b
            revf = fnu1 * fe2b * (1.0 - fe3b) * (1.0 - fnu4b)
            Fnet = forf - revf
            Ffrs = forf + revf

            # measure: prefac · regime and bin masks · e2w/p1² · e3w
            meas = ((mask * binmask)[:, None, None] * prefac
                    * (e2w / p1sq[:, None])[:, :, None] * e3w)    # (N,A,B)
            measb = meas[None, None, None]                      # (1,1,1,N,A,B)
            net = net + jnp.sum(measb * w3 * Fnet, axis=(0, 4, 5))   # sum nn,A,B -> (m,n,N)
            frs = frs + jnp.sum(measb * w3 * Ffrs, axis=(0, 4, 5))

    return net, frs


def nuer1_dfdt(grid, occ, meps, tcmpl, phie, t_cm_mev, **kw):
    """df/dt from ν–e± (R1) scattering, in s⁻¹ (T_cm⁵·net)."""
    net, _ = nuer1_rates(grid, occ, meps, tcmpl, phie, **kw)
    return t_cm_mev**5 * net


# ---- the scheduled implementation ---------------------------------------------------------------
# Each regime's (E2-segment × E3-sub-segment) tiles from _E2_SEGMENTS: A = S1 + S2 + S3 (8 tiles),
# B = S1 + S4 (6, padded to 8).  A tile is (e2_rule, a2, b2, a3, b3, tb):
#   e2_rule  0 = Legendre [a2, b2], 1 = Laguerre [a2, ∞) with b2 = −1 (E3 is always Legendre)
#   a2, b2   E2 bound: 0 = meps, 1 = ecut3, 2 = ecut1
#   a3, b3   E3 bound: 0 = meps, 1 = e2, 2 = etrans2, 3 = elim1
#   tb       angular limits (_topbot): 0 = full, 1 = mid1, 2 = mid2, 3 = mid3

_L, _G = 0, 1
_TILES = {
    "A": [
        (_L, 0, 1, 0, 1, 0), (_L, 0, 1, 1, 2, 1), (_L, 0, 1, 2, 3, 2),   # S1 (leg, meps..ecut3)
        (_L, 1, 2, 0, 2, 0), (_L, 1, 2, 2, 1, 3), (_L, 1, 2, 1, 3, 2),   # S2 (leg, ecut3..ecut1)
        (_G, 2, -1, 2, 1, 3), (_G, 2, -1, 1, 3, 2),                      # S3 (lag, ecut1)
    ],
    "B": [
        (_L, 0, 1, 0, 1, 0), (_L, 0, 1, 1, 2, 1), (_L, 0, 1, 2, 3, 2),   # S1 (leg, meps..ecut3)
        (_G, 1, -1, 0, 2, 0), (_G, 1, -1, 2, 1, 3), (_G, 1, -1, 1, 3, 2),  # S4 (lag, ecut3)
    ],
}
_TMAX = 8
_NF = 6


def _build_table():
    """_TILES as a (regime, tile, field) int table and a (regime, tile) 0/1 mask of real tiles."""
    tab = np.zeros((2, _TMAX, _NF), np.int32)
    act = np.zeros((2, _TMAX), np.float64)
    for r, rn in enumerate(["A", "B"]):
        for t, tile in enumerate(_TILES[rn]):
            tab[r, t] = tile
            act[r, t] = 1.0
    return tab, act


_TAB, _ACT = _build_table()


def _eval_tiles(grid, occ, logf, meps, tcmpl, phie,
                A2, B2, e2rule, a3k, b3k, tb, active, n_gl=5, n_lag=5,
                fd_cap_logC=float("inf"), fd_decay=False, monotone_read=False):
    """Sum all tiles of all bins in one contraction; returns (net, frs), each (3, 2, N).

    Tile fields are (N, T): A2, B2 the resolved E2 bounds, the rest int codes, ``active`` the 0/1
    tile mask.
    """
    eps = grid.abscissas
    p1 = eps
    term1, c2 = _coupling_tensors()
    prefac = GFHBAR / 16.0 / (2.0 * PI) ** 3
    tleg, wleg = legendre_canonical(n_gl)
    xlag, wlag = gauss_laguerre_plain(n_lag)
    binmask = (p1 > 0.0).astype(eps.dtype)
    p1sq = jnp.where(p1 > 0.0, p1 ** 2, 1.0)
    msq = meps ** 2

    # E2 (outer) nodes, (N, T, K): both rules evaluated, one kept per tile
    is_lag2 = (e2rule == 1)[:, :, None]
    mid2 = 0.5 * (A2 + B2); half2 = 0.5 * (B2 - A2)
    E2_leg = mid2[:, :, None] + half2[:, :, None] * tleg[None, None, :]
    e2w_leg = wleg[None, None, :] * half2[:, :, None]
    E2_lag = A2[:, :, None] + xlag[None, None, :]
    e2w_lag = jnp.broadcast_to(wlag[None, None, :], E2_lag.shape)
    E2 = jnp.where(is_lag2, E2_lag, E2_leg)
    e2w = jnp.where(is_lag2, e2w_lag, e2w_leg)
    E2 = jnp.maximum(E2, meps)
    q2 = _safe_sqrt_diff(E2 ** 2 - msq)
    p1_3 = p1[:, None, None]
    d = 2.0 * p1_3 + E2 - q2
    etrans2 = 0.5 * (2.0 * p1_3 + E2 - q2 + msq / jnp.where(d != 0.0, d, 1.0))
    dl = 2.0 * p1_3 + E2 + q2
    elim1 = 0.5 * (2.0 * p1_3 + E2 + q2 + msq / jnp.where(dl != 0.0, dl, 1.0))
    A2coef = (p1_3 + E2) ** 2 - msq                                # (N,T,K)
    fe2 = jnp.stack([_fd(E2, tcmpl, phie, 1.0), _fd(E2, tcmpl, phie, -1.0)])  # (nn,N,T,K)
    kinemfact = e2w / p1sq[:, None, None]                          # (N,T,K)

    # E3 bounds, picked by key from [meps, e2, etrans2, elim1]
    e3bounds = jnp.stack([jnp.broadcast_to(jnp.asarray(meps, eps.dtype), E2.shape),
                          E2, etrans2, elim1])
    a3 = jnp.take_along_axis(e3bounds, jnp.broadcast_to(a3k[None, :, :, None], (1,) + E2.shape), 0)[0]
    b3 = jnp.take_along_axis(e3bounds, jnp.broadcast_to(b3k[None, :, :, None], (1,) + E2.shape), 0)[0]

    # E3 (inner, Legendre) nodes, (N, T, K, J)
    mid3 = 0.5 * (a3 + b3); half3 = 0.5 * (b3 - a3)
    E3 = mid3[:, :, :, None] + half3[:, :, :, None] * tleg[None, None, None, :]
    e3w = wleg[None, None, None, :] * half3[:, :, :, None]
    E3 = jnp.maximum(E3, meps)
    q3 = _safe_sqrt_diff(E3 ** 2 - msq)                            # (N,T,K,J)

    p1_4 = p1[:, None, None, None]
    e2_4 = E2[:, :, :, None]; q2_4 = q2[:, :, :, None]
    full_t = p1_4 + e2_4 - E3 + q3; full_b = p1_4 + e2_4 - E3 - q3
    mid1_t = p1_4 + q2_4; mid1_b = p1_4 - q2_4
    mid2_b = E3 + q3 - p1_4 - e2_4
    mid3_t = p1_4 + e2_4 - E3 + q3; mid3_b = q2_4 - p1_4
    tb4 = tb[:, :, None, None]
    top = jnp.where(tb4 == 0, full_t, jnp.where(tb4 == 1, mid1_t,
                    jnp.where(tb4 == 2, mid1_t, mid3_t)))          # mid2 top = p1+q2
    bot = jnp.where(tb4 == 0, full_b, jnp.where(tb4 == 1, mid1_b,
                    jnp.where(tb4 == 2, mid2_b, mid3_b)))
    Ae = A2coef[:, :, :, None]
    I1 = _R1(top, Ae) - _R1(bot, Ae)
    I2 = _R2(top, Ae) - _R2(bot, Ae)
    w3 = term1[:, :, :, None, None, None, None] * (
        I1[None, None, None] + c2[:, :, :, None, None, None, None] * msq * I2[None, None, None])

    fe3 = jnp.stack([_fd(E3, tcmpl, phie, 1.0), _fd(E3, tcmpl, phie, -1.0)])  # (nn,N,T,K,J)
    p4 = jnp.maximum(p1_4 + e2_4 - E3, 0.0)
    fnu4 = jnp.exp(interp_log_occ_dynamic(logf, eps, p4, fd_cap_logC=fd_cap_logC,
                                          fd_decay=fd_decay, monotone_read=monotone_read))  # (3,2,N,T,K,J)

    fnu1 = occ[None, :, :, :, None, None, None]
    fe2b = fe2[:, None, None, :, :, :, None]    # (nn,1,1,N,T,K,1)
    fe3b = fe3[:, None, None]                   # (nn,1,1,N,T,K,J)
    fnu4b = fnu4[None]
    forf = (1.0 - fnu1) * (1.0 - fe2b) * fe3b * fnu4b
    revf = fnu1 * fe2b * (1.0 - fe3b) * (1.0 - fnu4b)
    Fnet = forf - revf; Ffrs = forf + revf

    meas = (prefac * (active * binmask[:, None])[:, :, None, None]
            * kinemfact[:, :, :, None] * e3w)                      # (N,T,K,J)
    measb = meas[None, None, None]
    net = jnp.sum(measb * w3 * Fnet, axis=(0, 4, 5, 6))
    frs = jnp.sum(measb * w3 * Ffrs, axis=(0, 4, 5, 6))
    return net, frs


def nuer1_rates_sched(grid: EpsGrid, occ, meps, tcmpl, phie, select: bool = True,
                      n_gl: int = 5, n_lag: int = 5, fd_cap_logC: float = float("inf"),
                      fd_decay: bool = False, monotone_read: bool = False,
                      lag_tail_fix: bool = True, lag_tail_gap: float = _LAG_TAIL_GAP):
    """ν–e± (R1) rates, the scheduled implementation.

    Arguments and returns as :func:`nuer1_rates`.  ``select=True`` gathers each bin's own regime
    row of the schedule (``use_sched=True``); ``select=False`` runs both regimes' tiles for every
    bin, masked by regime (``use_sched="union"``).
    """
    if n_gl != n_lag:
        raise ValueError("nuer1_rates_sched needs n_gl == n_lag (the Legendre and Laguerre "
                         "tiles share one node axis); use nuer1_rates otherwise")
    eps = grid.abscissas
    p1 = eps
    N = eps.shape[0]
    logf = jnp.log(occ)

    ecut3 = jnp.sqrt(p1 ** 2 + meps ** 2)
    denom_a = meps - 2.0 * p1
    ecut1 = meps + 2.0 * p1 ** 2 / jnp.where(denom_a != 0.0, denom_a, 1.0)
    cutvals = jnp.stack([jnp.broadcast_to(jnp.asarray(meps, eps.dtype), (N,)), ecut3, ecut1])  # (3,N)

    reg = (p1 >= meps / 2.0).astype(jnp.int32)                     # 0=A, 1=B
    # lag_tail_fix: a wide regime-A bin's S1 + S5 is exactly the regime-B schedule
    if lag_tail_fix:
        wide = (p1 < meps / 2.0) & (ecut1 - ecut3 >= lag_tail_gap)
        reg = jnp.where(wide, jnp.int32(1), reg)
    tab = jnp.asarray(_TAB); actT = jnp.asarray(_ACT)
    nrow = jnp.arange(N)
    if select:
        f = tab[reg]; act = actT[reg]
    else:
        f = jnp.broadcast_to(tab[None], (N, 2, _TMAX, _NF)).reshape(N, 2 * _TMAX, _NF)
        onehot = (reg[:, None] == jnp.arange(2)[None, :]).astype(eps.dtype)
        act = (onehot[:, :, None] * actT[None]).reshape(N, 2 * _TMAX)

    e2rule = f[..., 0]; a2k = f[..., 1]; b2k = f[..., 2]
    a3k = f[..., 3]; b3k = f[..., 4]; tb = f[..., 5]
    A2 = cutvals[a2k, nrow[:, None]]
    B2 = cutvals[jnp.where(b2k < 0, 0, b2k), nrow[:, None]]

    return _eval_tiles(grid, occ, logf, meps, tcmpl, phie, A2, B2, e2rule, a3k, b3k, tb, act,
                       n_gl, n_lag, fd_cap_logC, fd_decay, monotone_read)
