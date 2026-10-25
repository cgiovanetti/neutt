"""BURST's order-5 log-space occupation interpolation, and the off-grid tail closures.

log f at an off-grid momentum is a 6-point (order-5) Lagrange polynomial in the grid values of
log f, clamped to [−300, 0] (an overshoot above 0 is sent to −300, as in BURST).  For static query
points (the grid-only ν–ν kernels) the windows and weights are precomputed on the host
(``lagrange_window`` + ``apply_log_interp``); the lepton kernels query at traced momenta
(``interp_log_occ_dynamic``).

Past the grid edge the polynomial extrapolates far shallower than a relativistic fermion tail, and
the collision integrals can amplify that into an unphysical tail.  The classical solve is
bit-faithful to BURST and uses none of the closures below; the QKE solve needs them:

- ``fd_decay``: off grid, cap log f at the Fermi–Dirac shape anchored to the live grid-edge value
  (parameter-free, exact at the edge, a no-op on grid);
- ``monotone_read``: clamp the interpolant to the two nodes bracketing the query (kills the
  odd–even mode the alternating Lagrange weights excite); off grid both are the edge node, so the
  read is pinned to f(x_max), and ``fd_decay`` then lowers it to the FD tail (the pair is the
  QKE's "srcfix" closure; ``monotone_read`` alone holds the tail flat);
- ``fd_cap_logC``: the absolute envelope f ≤ C·FD(q), log C finite (``FD_ENVELOPE_C``, env var
  ``NEUTT_FD_ENVELOPE_C``); +inf, the default, is a bitwise no-op.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np

ORDPOLINT = 5           # order-5 polynomial → 6-point window
LOG_OCC_CLAMP = -300.0  # BURST's interp_occfrac clamp value
FD_ENVELOPE_C = float(os.environ.get("NEUTT_FD_ENVELOPE_C", "10.0"))


def _indpolint_start0(abscissas: np.ndarray, x: np.ndarray, ord: int = ORDPOLINT) -> np.ndarray:
    """0-indexed start of the (ord+1)-point window (BURST's indpolint, minus one), vectorized."""
    n = abscissas.shape[0]
    half = ord // 2
    jl = np.searchsorted(abscissas, x, side="right")          # count of nodes ≤ x, in [0, n]
    start1 = np.where(jl < 1 + half, 1, np.where(jl > n - 1 - half, n - ord, jl - half))
    return start1.astype(np.int64) - 1


def lagrange_window(abscissas, query, ord: int = ORDPOLINT):
    """Host-side (idx, weights), each query.shape + (ord+1,), for static query momenta.

    Σ_w weights[..., w]·log f[idx[..., w]] reproduces BURST's polynomial to ~1e-15 relative.  Valid
    for 0 ≤ query ≤ 2·abscissas[-1] (always true for 2 → 2 kinematics).
    """
    abscissas = np.asarray(abscissas, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    m = ord + 1
    start0 = _indpolint_start0(abscissas, q, ord)
    idx = start0[..., None] + np.arange(m, dtype=np.int64)
    nodes = abscissas[idx]
    qe = q[..., None]
    weights = np.ones(nodes.shape, dtype=np.float64)
    for w in range(m):
        for wp in range(m):
            if wp == w:
                continue
            weights[..., w] *= (qe[..., 0] - nodes[..., wp]) / (nodes[..., w] - nodes[..., wp])
    return idx, weights


def apply_log_interp(logf_grid, idx, weights, fd_decay_base=None, bracket=None):
    """Clamped log f at precomputed windows: ``logf_grid`` (..., N) → (...,) + S.

    ``bracket = (lo, hi)`` index arrays enable the monotone read; ``fd_decay_base`` (point-shape S,
    +inf on grid, ``fd_logFD(q) + logaddexp(x_max, 0)`` off grid; built by the ν–ν kernel builders)
    enables the FD-decay closure.  The FD-envelope cap is applied by the caller.
    """
    idx = jnp.asarray(idx)
    weights = jnp.asarray(weights)
    gathered = logf_grid[..., idx]
    out = jnp.sum(weights * gathered, axis=-1)
    if bracket is not None:
        b_lo, b_hi = bracket
        v_lo = logf_grid[..., b_lo]
        v_hi = logf_grid[..., b_hi]
        out = jnp.clip(out, jnp.minimum(v_lo, v_hi), jnp.maximum(v_lo, v_hi))
    if fd_decay_base is not None:
        lead = logf_grid.shape[:-1]
        logf_edge = logf_grid[..., -1].reshape(lead + (1,) * jnp.ndim(fd_decay_base))
        out = jnp.where(jnp.isfinite(fd_decay_base), jnp.minimum(out, logf_edge + fd_decay_base), out)
    # NaN → −300 too (a bare comparison lets NaN through)
    return jnp.where((out < LOG_OCC_CLAMP) | (out > 0.0) | jnp.isnan(out), LOG_OCC_CLAMP, out)


def _lagrange_weights_at(nodes, q):
    """Lagrange weights at q over the (..., m) node windows (jittable)."""
    m = nodes.shape[-1]
    qe = q[..., None]
    w = jnp.ones(nodes.shape, dtype=nodes.dtype)
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            w = w.at[..., a].multiply((qe[..., 0] - nodes[..., b]) / (nodes[..., a] - nodes[..., b]))
    return w


def interp_log_occ_dynamic(logf_grid, abscissas, query, ord: int = ORDPOLINT,
                           fd_cap_logC: float = float("inf"), fd_decay: bool = False,
                           monotone_read: bool = False):
    """log f of a fermion spectrum at traced query momenta, BURST's three branches.

    q ≤ 2·x_max: the order-``ord`` window, clamped; 2·x_max < q < 300: linear in log f through the
    last two nodes (capped at 0); q ≥ 300: −300.  ``logf_grid`` is (..., N), ``query`` any shape S;
    returns (...,) + S.  ``fd_cap_logC``, ``fd_decay`` and ``monotone_read`` are the closures of the
    module docstring.
    """
    abscissas = jnp.asarray(abscissas)
    N = abscissas.shape[0]
    m = ord + 1
    half = ord // 2
    q = jnp.asarray(query)

    jl = jnp.searchsorted(abscissas, q, side="right")
    start1 = jnp.where(jl < 1 + half, 1, jnp.where(jl > N - 1 - half, N - ord, jl - half))
    start0 = jnp.clip(start1 - 1, 0, N - m).astype(jnp.int32)
    idx = start0[..., None] + jnp.arange(m, dtype=jnp.int32)
    nodes = abscissas[idx]
    lw = _lagrange_weights_at(nodes, q)
    gathered = logf_grid[..., idx]
    win = jnp.sum(lw * gathered, axis=-1)
    if monotone_read:
        lo = jnp.clip(jl - 1, 0, N - 1)
        hi = jnp.clip(jl, 0, N - 1)
        b_lo = logf_grid[..., lo]
        b_hi = logf_grid[..., hi]
        win = jnp.clip(win, jnp.minimum(b_lo, b_hi), jnp.maximum(b_lo, b_hi))
    win = jnp.where((win < LOG_OCC_CLAMP) | (win > 0.0) | jnp.isnan(win), LOG_OCC_CLAMP, win)

    lead = logf_grid.shape[:-1]
    x0, x1 = abscissas[-2], abscissas[-1]
    y0 = logf_grid[..., -2].reshape(lead + (1,) * q.ndim)
    y1 = logf_grid[..., -1].reshape(lead + (1,) * q.ndim)
    lin = y0 + (y1 - y0) / (x1 - x0) * (q - x0)
    lin = jnp.where(jnp.isnan(lin), LOG_OCC_CLAMP, jnp.minimum(lin, 0.0))

    xmax = abscissas[-1]
    out = jnp.where(q <= 2.0 * xmax, win, jnp.where(q < 300.0, lin, LOG_OCC_CLAMP))
    out = jnp.minimum(out, fd_cap_logC - jnp.logaddexp(q, 0.0))
    if fd_decay:
        logf_edge = logf_grid[..., -1].reshape(logf_grid.shape[:-1] + (1,) * q.ndim)
        fd_tail = (logf_edge + jnp.logaddexp(xmax, 0.0)) - jnp.logaddexp(q, 0.0)
        out = jnp.where(q > xmax, jnp.minimum(out, fd_tail), out)
    return out
