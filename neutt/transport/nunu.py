"""ν–ν (and ν̄–ν̄) scattering as a static-tensor contraction (BURST's ``scattering_nunu``).

Every momentum in the collision integral is fixed by the ε-grid: ε₁ (output bin i) and ε₂ (partner
bin j) are grid points, p₃ runs over Gauss–Legendre nodes in [0, ε₁ + ε₂] and p₄ = ε₁ + ε₂ − p₃.
The angular and quadrature weights and the Lagrange windows that interpolate f at the off-grid p₃,
p₄ are therefore built once on the host (:func:`build_nunu_kernel`) and contracted against the
occupations at runtime (:func:`nunu_rates`; jittable, differentiable).
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..constants import GFHBAR, PI
from ..grid import EpsGrid
from ..interp import apply_log_interp, lagrange_window
from ..grid import legendre_canonical

# Element budget for the per-chunk gather tensor (3, 2, N, chunk_j, K, M): 4.8 GB at f64.
_CHUNK_ELEM_BUDGET = 6e8


# ---- angular weights (|M|² integrated over angles; host side) -----------------------------------

def _weight_ends(p1, p2, p3):
    """Angular weight on the two outer p₃ segments (BURST's ``calc_weight_nunu_ends``)."""
    s = p1 + p2
    if p3 >= s or p3 == 0.0:
        return 0.0
    first = s**4
    if s >= 2.0 * p3:
        prefac = 4.0 * p3 / 15.0 / p1**3 / p2**3
        second = -(s - p3) * (s - 2.0 * p3) * (s**2 + 3.0 * p3 * (s - p3))
    else:
        prefac = 4.0 * (s - p3) / 15.0 / p1**3 / p2**3
        second = (s - 2.0 * p3) * p3 * (s**2 + 3.0 * p3 * (s - p3))
    fval = prefac * (first + second)
    return fval if fval > 0.0 else 0.0


def _weight_mid(p1, p2):
    """Angular weight on the middle p₃ segment, constant (BURST's ``calc_weight_nunu_mid``)."""
    psmall, plarge = min(p1, p2), max(p1, p2)
    fval = 4.0 / 15.0 / plarge**3 * (psmall**2 + 5.0 * p1 * p2 + 10.0 * plarge**2)
    return fval if fval > 0.0 else 0.0


# ---- kernel ------------------------------------------------------------------------------------

class NuNuKernel(eqx.Module):
    """Static tensors of the ν–ν contraction over (output bin i, partner bin j, slot k[, window]).

    K = 3·n_gl quadrature slots and M = ordpolint + 1 window points.  The ``fd_*`` and ``brk_*``
    fields feed the off-grid closures of :mod:`neutt.interp` (QKE solver only).
    """

    W: jnp.ndarray        # (N, N, K)        prefac·kinemfact1·GL weight·angular weight
    idx3: jnp.ndarray     # (N, N, K, M) int  Lagrange window for p₃
    lw3: jnp.ndarray      # (N, N, K, M)      its weights
    idx4: jnp.ndarray     # (N, N, K, M) int  the same for p₄ = ε₁+ε₂−p₃
    lw4: jnp.ndarray      # (N, N, K, M)
    flafact: jnp.ndarray  # (3, 3)            1 on the diagonal, ½ off-diagonal
    fd_logFD3: jnp.ndarray  # (N, N, K)       log FD(p₃), base of the envelope cap
    fd_logFD4: jnp.ndarray  # (N, N, K)       log FD(p₄)
    fd_decay_base3: jnp.ndarray  # (N, N, K)  FD-decay closure base (+inf on grid)
    fd_decay_base4: jnp.ndarray  # (N, N, K)
    brk_lo3: jnp.ndarray   # (N, N, K) int  monotone read: the grid nodes bracketing p₃ (p₄)
    brk_hi3: jnp.ndarray
    brk_lo4: jnp.ndarray
    brk_hi4: jnp.ndarray
    nbins: int = eqx.field(static=True)
    nslots: int = eqx.field(static=True)


def build_nunu_kernel(grid: EpsGrid, n_gl: int = 5, ordpolint: int = 5) -> NuNuKernel:
    """Build the ν–ν kernel tensors on the host (NumPy).

    p₃ ∈ [0, ε₁ + ε₂] is split at min(ε₁, ε₂) and max(ε₁, ε₂), where the angular weight changes
    form, with ``n_gl`` Gauss–Legendre points per segment (BURST's ``lgndrbins``): K = 3·n_gl
    slots per (i, j).  At i = j the middle segment has zero width and its weights vanish.
    ``ordpolint`` is the log-interpolation order (5, a 6-point window, as in BURST).
    """
    eps = np.asarray(grid.abscissas, dtype=np.float64)
    w = np.asarray(grid.weights, dtype=np.float64)
    N = eps.shape[0]
    K = 3 * n_gl
    M = ordpolint + 1

    tg, wg = legendre_canonical(n_gl)
    tg = np.asarray(tg, dtype=np.float64)
    wg = np.asarray(wg, dtype=np.float64)

    prefac = 128.0 * GFHBAR / 32.0 / (2.0 * PI) ** 3

    W = np.zeros((N, N, K), dtype=np.float64)
    p3 = np.zeros((N, N, K), dtype=np.float64)
    p4 = np.zeros((N, N, K), dtype=np.float64)

    for i in range(N):
        p1 = eps[i]
        if p1 == 0.0:
            continue  # kinemfact1 = 0 at ε₁ = 0
        for j in range(N):
            p2 = eps[j]
            if p2 == 0.0:
                continue
            psmall, plarge = (p2, p1) if i >= j else (p1, p2)
            kinemfact1 = w[j] * p1 * p2**3
            s_tot = p1 + p2
            # three segments: (a, b, is_mid)
            segs = ((0.0, psmall, False), (psmall, plarge, True), (plarge, s_tot, False))
            for sidx, (a, b, is_mid) in enumerate(segs):
                half = 0.5 * (b - a)
                mid = 0.5 * (b + a)
                nodes = mid + half * tg          # rescaled p₃ nodes
                glw = wg * half                  # rescaled GL weights (0 if a == b)
                for k in range(n_gl):
                    slot = sidx * n_gl + k
                    p3v = nodes[k]
                    if is_mid:
                        wt3 = _weight_mid(p1, p2)
                    else:
                        wt3 = _weight_ends(p1, p2, p3v)
                    W[i, j, slot] = prefac * kinemfact1 * glw[k] * wt3
                    p3[i, j, slot] = p3v
                    p4[i, j, slot] = s_tot - p3v

    idx3, lw3 = lagrange_window(eps, p3, ordpolint)
    idx4, lw4 = lagrange_window(eps, p4, ordpolint)

    flafact = 0.5 * np.ones((3, 3), dtype=np.float64) + 0.5 * np.eye(3, dtype=np.float64)
    # envelope cap f ≤ C·FD(p), applied at runtime as min(log f, fd_cap_logC + log FD(p))
    fd_logFD3 = -np.logaddexp(p3, 0.0)
    fd_logFD4 = -np.logaddexp(p4, 0.0)
    # FD decay: off grid, log f ≤ log f(x_max) + log FD(p) − log FD(x_max); +inf on grid
    xmax = eps[-1]
    base_off = np.logaddexp(xmax, 0.0)
    fd_decay_base3 = np.where(p3 > xmax, fd_logFD3 + base_off, np.inf)
    fd_decay_base4 = np.where(p4 > xmax, fd_logFD4 + base_off, np.inf)
    # monotone read: the two grid nodes bracketing p
    jl3 = np.searchsorted(eps, p3, side="right")
    jl4 = np.searchsorted(eps, p4, side="right")
    brk_lo3 = np.clip(jl3 - 1, 0, N - 1); brk_hi3 = np.clip(jl3, 0, N - 1)
    brk_lo4 = np.clip(jl4 - 1, 0, N - 1); brk_hi4 = np.clip(jl4, 0, N - 1)

    return NuNuKernel(
        W=jnp.asarray(W),
        idx3=jnp.asarray(idx3), lw3=jnp.asarray(lw3),
        idx4=jnp.asarray(idx4), lw4=jnp.asarray(lw4),
        flafact=jnp.asarray(flafact),
        fd_logFD3=jnp.asarray(fd_logFD3), fd_logFD4=jnp.asarray(fd_logFD4),
        fd_decay_base3=jnp.asarray(fd_decay_base3), fd_decay_base4=jnp.asarray(fd_decay_base4),
        brk_lo3=jnp.asarray(brk_lo3), brk_hi3=jnp.asarray(brk_hi3),
        brk_lo4=jnp.asarray(brk_lo4), brk_hi4=jnp.asarray(brk_hi4),
        nbins=int(N), nslots=int(K),
    )


# ---- partner-bin (j) chunking ------------------------------------------------------------------
# The interpolation gather is (3, 2, N, N, K, M), ~140 GB at N = 1024, so for large N the j axis is
# summed in blocks under lax.scan (bounding peak memory, not work).  Every j-carrying tensor is
# sliced per block: the kernel tensors (axis 1) and the leg-2 occupation.  j is right-padded to a
# multiple of n_chunks with W = 0, so padded terms vanish and the chunked sum equals the full one
# up to reassociation.  n_chunks = 1 is the unchunked path.


def _auto_n_chunks(N: int, K: int, M: int) -> int:
    """Static n_chunks for which the per-chunk gather (3, 2, N, chunk_j, K, M) fits the budget.

    1 for small N (e.g. 101, 257).
    """
    per_j = 3 * 2 * N * K * M
    chunk_j = max(1, int(_CHUNK_ELEM_BUDGET // per_j))
    return max(1, math.ceil(N / chunk_j))


def _pad_to_chunks(N: int, n_chunks: int):
    """(j_pad, chunk_j): pad length and per-chunk j-size so j_pad = n_chunks·chunk_j >= N."""
    chunk_j = math.ceil(N / n_chunks)
    return n_chunks * chunk_j, chunk_j


def _pad_j(arr, j_pad: int, fill: float):
    """Pad a kernel tensor's j-axis (axis 1) up to j_pad with ``fill`` (right-pad)."""
    n = arr.shape[1]
    if j_pad == n:
        return arr
    pads = [(0, 0)] * arr.ndim
    pads[1] = (0, j_pad - n)
    return jnp.pad(arr, pads, constant_values=fill)


# ---- runtime -----------------------------------------------------------------------------------

def nunu_rates(kernel: NuNuKernel, occ, n_chunks: int | None = None,
               fd_cap_logC: float = float("inf"), range_limit: bool = False,
               fd_decay: bool = False, monotone_read: bool = False):
    """Net and forward+reverse ν–ν rates (net, frs), each (3, 2, N), per T_cm⁵.

    Summed over the partner flavour, the partner bin j and the p₃ quadrature; df/dt = T_cm⁵·net
    [s⁻¹] (:func:`nunu_dfdt`).

    Parameters
    ----------
    occ : jnp.ndarray
        On-grid occupation f[3, 2, N]; legs 1, 2 use it directly, legs 3, 4 interpolate log f.
    n_chunks : int or None
        Partner-bin chunking, a static int: 1 = unchunked, > 1 = that many blocks under
        ``lax.scan``, None = chosen from N and ``_CHUNK_ELEM_BUDGET``.
    fd_cap_logC, fd_decay, monotone_read
        The off-grid closures of :mod:`neutt.interp`, used only by the QKE solver; the defaults are
        bitwise no-ops.
    range_limit : bool
        The window-max cap of :mod:`neutt.interp`, used only by the strong-χ dark path; default off.
    """
    N = kernel.nbins
    if n_chunks is None:
        n_chunks = _auto_n_chunks(N, kernel.W.shape[2], kernel.idx3.shape[3])

    if n_chunks == 1:
        logf = jnp.log(occ)
        fdd3 = kernel.fd_decay_base3 if fd_decay else None
        fdd4 = kernel.fd_decay_base4 if fd_decay else None
        brk3 = (kernel.brk_lo3, kernel.brk_hi3) if monotone_read else None
        brk4 = (kernel.brk_lo4, kernel.brk_hi4) if monotone_read else None
        # interpolated f at p₃, p₄, capped at C·FD(p)
        f3 = jnp.exp(jnp.minimum(apply_log_interp(logf, kernel.idx3, kernel.lw3, range_limit=range_limit, fd_decay_base=fdd3, bracket=brk3), fd_cap_logC + kernel.fd_logFD3))  # leg 3
        f4 = jnp.exp(jnp.minimum(apply_log_interp(logf, kernel.idx4, kernel.lw4, range_limit=range_limit, fd_decay_base=fdd4, bracket=brk4), fd_cap_logC + kernel.fd_logFD4))  # leg 4

        # F-tensor axes: (m=this flavor, p=partner flavor, n=cp, i=out bin, j=partner bin, k=slot)
        f1 = occ[:, None, :, :, None, None]      # (m,1,n,i,1,1)   leg 1: f(ε₁), this flavor
        f2 = occ[None, :, :, None, :, None]      # (1,p,n,1,j,1)   leg 2: f(ε₂), partner flavor
        f3b = f3[:, None]                        # (m,1,n,i,j,k)   leg 3: f(p₃), this flavor
        f4b = f4[None, :]                        # (1,p,n,i,j,k)   leg 4: f(p₄), partner flavor

        gain = (1.0 - f1) * (1.0 - f2) * f3b * f4b
        loss = f1 * f2 * (1.0 - f3b) * (1.0 - f4b)
        fla = kernel.flafact[:, :, None, None, None, None]   # (m,p,1,1,1,1)
        Wb = kernel.W[None, None, None]                      # (1,1,1,i,j,k)

        F_net = fla * (gain - loss)
        F_frs = fla * (gain + loss)
        net = jnp.sum(Wb * F_net, axis=(1, 4, 5))            # sum partner flavor p, bins j, slots k
        frs = jnp.sum(Wb * F_frs, axis=(1, 4, 5))            # -> (m, n, i)
        return net, frs

    # chunked: sum the j axis in n_chunks blocks under scan
    logf = jnp.log(occ)
    fla = kernel.flafact[:, :, None, None, None, None]       # (m,p,1,1,1,1)

    j_pad, chunk_j = _pad_to_chunks(N, n_chunks)
    # pad the j axis to j_pad with W = 0, so padded terms vanish (FD-decay base: +inf, a no-op)
    Wp = _pad_j(kernel.W, j_pad, 0.0).reshape(N, n_chunks, chunk_j, kernel.W.shape[2])
    idx3p = _pad_j(kernel.idx3, j_pad, 0).reshape(N, n_chunks, chunk_j, *kernel.idx3.shape[2:])
    lw3p = _pad_j(kernel.lw3, j_pad, 0.0).reshape(N, n_chunks, chunk_j, *kernel.lw3.shape[2:])
    idx4p = _pad_j(kernel.idx4, j_pad, 0).reshape(N, n_chunks, chunk_j, *kernel.idx4.shape[2:])
    lw4p = _pad_j(kernel.lw4, j_pad, 0.0).reshape(N, n_chunks, chunk_j, *kernel.lw4.shape[2:])
    fd3p = _pad_j(kernel.fd_logFD3, j_pad, 0.0).reshape(N, n_chunks, chunk_j, kernel.fd_logFD3.shape[2])
    fd4p = _pad_j(kernel.fd_logFD4, j_pad, 0.0).reshape(N, n_chunks, chunk_j, kernel.fd_logFD4.shape[2])
    fdd3p = _pad_j(kernel.fd_decay_base3, j_pad, jnp.inf).reshape(N, n_chunks, chunk_j, kernel.fd_decay_base3.shape[2])
    fdd4p = _pad_j(kernel.fd_decay_base4, j_pad, jnp.inf).reshape(N, n_chunks, chunk_j, kernel.fd_decay_base4.shape[2])
    bl3p = _pad_j(kernel.brk_lo3, j_pad, 0).reshape(N, n_chunks, chunk_j, kernel.brk_lo3.shape[2])
    bh3p = _pad_j(kernel.brk_hi3, j_pad, 0).reshape(N, n_chunks, chunk_j, kernel.brk_hi3.shape[2])
    bl4p = _pad_j(kernel.brk_lo4, j_pad, 0).reshape(N, n_chunks, chunk_j, kernel.brk_lo4.shape[2])
    bh4p = _pad_j(kernel.brk_hi4, j_pad, 0).reshape(N, n_chunks, chunk_j, kernel.brk_hi4.shape[2])
    # leg 2 is indexed by j, which is axis 2 of occ
    occ_pad = jnp.pad(occ, [(0, 0), (0, 0), (0, j_pad - N)]) if j_pad != N else occ
    occ_blk = occ_pad.reshape(occ.shape[0], occ.shape[1], n_chunks, chunk_j)  # (p,n,c,jb)

    # chunk axis first, for scan
    Wc = jnp.moveaxis(Wp, 1, 0)            # (c, N, jb, K)
    idx3c = jnp.moveaxis(idx3p, 1, 0)      # (c, N, jb, K, M)
    lw3c = jnp.moveaxis(lw3p, 1, 0)
    idx4c = jnp.moveaxis(idx4p, 1, 0)
    lw4c = jnp.moveaxis(lw4p, 1, 0)
    fd3c = jnp.moveaxis(fd3p, 1, 0)
    fd4c = jnp.moveaxis(fd4p, 1, 0)
    fdd3c = jnp.moveaxis(fdd3p, 1, 0)
    fdd4c = jnp.moveaxis(fdd4p, 1, 0)
    bl3c = jnp.moveaxis(bl3p, 1, 0); bh3c = jnp.moveaxis(bh3p, 1, 0)
    bl4c = jnp.moveaxis(bl4p, 1, 0); bh4c = jnp.moveaxis(bh4p, 1, 0)
    occ_c = jnp.moveaxis(occ_blk, 2, 0)    # (c, p, n, jb)

    f1 = occ[:, None, :, :, None, None]    # (m,1,n,i,1,1)  leg 1: j-independent

    def body(carry, blk):
        net_acc, frs_acc = carry
        (W_b, idx3_b, lw3_b, idx4_b, lw4_b, occ_b, fd3_b, fd4_b, fdd3_b, fdd4_b,
         bl3_b, bh3_b, bl4_b, bh4_b) = blk
        fdd3 = fdd3_b if fd_decay else None
        fdd4 = fdd4_b if fd_decay else None
        brk3 = (bl3_b, bh3_b) if monotone_read else None
        brk4 = (bl4_b, bh4_b) if monotone_read else None
        f3 = jnp.exp(jnp.minimum(apply_log_interp(logf, idx3_b, lw3_b, range_limit=range_limit, fd_decay_base=fdd3, bracket=brk3), fd_cap_logC + fd3_b))  # (m,n,i,jb,k)
        f4 = jnp.exp(jnp.minimum(apply_log_interp(logf, idx4_b, lw4_b, range_limit=range_limit, fd_decay_base=fdd4, bracket=brk4), fd_cap_logC + fd4_b))  # (m,n,i,jb,k)
        f2 = occ_b[None, :, :, None, :, None]                   # (1,p,n,1,jb,1)
        f3b = f3[:, None]                                       # (m,1,n,i,jb,k)
        f4b = f4[None, :]                                       # (1,p,n,i,jb,k)
        gain = (1.0 - f1) * (1.0 - f2) * f3b * f4b
        loss = f1 * f2 * (1.0 - f3b) * (1.0 - f4b)
        Wb = W_b[None, None, None]                              # (1,1,1,i,jb,k)
        net_acc = net_acc + jnp.sum(Wb * (fla * (gain - loss)), axis=(1, 4, 5))
        frs_acc = frs_acc + jnp.sum(Wb * (fla * (gain + loss)), axis=(1, 4, 5))
        return (net_acc, frs_acc), None

    z = jnp.zeros((occ.shape[0], occ.shape[1], N), dtype=occ.dtype)  # (m,n,i)
    (net, frs), _ = jax.lax.scan(body, (z, z),
                                 (Wc, idx3c, lw3c, idx4c, lw4c, occ_c, fd3c, fd4c, fdd3c, fdd4c,
                                  bl3c, bh3c, bl4c, bh4c))
    return net, frs


def nunu_dfdt(kernel: NuNuKernel, occ, t_cm_mev: float, n_chunks: int | None = None):
    """df/dt from ν–ν scattering, in s⁻¹ (T_cm⁵·net)."""
    net, _ = nunu_rates(kernel, occ, n_chunks=n_chunks)
    return t_cm_mev**5 * net
