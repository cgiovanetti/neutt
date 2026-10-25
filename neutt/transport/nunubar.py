"""ν–ν̄ scattering and annihilation as a static-tensor contraction (BURST's ``scattering_nunubar``).

Grid-only like :mod:`neutt.transport.nunu`, with one weight tensor serving two channels: scattering
ν_m ν̄_p → ν_m ν̄_p (net_s, frs_s) and flavour-changing annihilation ν_m ν̄_m → ν_p ν̄_p, p ≠ m
(net_a, frs_a).  The antineutrino partner is the opposite cp (the cp axis reversed).  Relative to
ν–ν the prefactor is ÷16 instead of ÷32, kinemfact1 = w_j·ε₁ with the ε³ measure moved onto p₃,
and p₃ is split once, at ε₂.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from ..constants import GFHBAR, PI
from ..grid import EpsGrid
from ..interp import apply_log_interp, lagrange_window
from ..grid import legendre_canonical
from .nunu import _auto_n_chunks, _pad_to_chunks, _pad_j


# ---- angular weights (host side) ----

def _weight_small(p1, p3):
    """Angular weight for p₃ ∈ [0, ε₂] (BURST's ``calc_weight_nunubar_small``)."""
    psmall, plarge = min(p1, p3), max(p1, p3)
    fval = 4.0 / 15.0 / plarge**3 * (psmall**2 - 5.0 * p1 * p3 + 10.0 * plarge**2)
    return fval if fval > 0.0 else 0.0


def _weight_large(p1, p2, p3):
    """Angular weight for p₃ ∈ [ε₂, ε₁ + ε₂] (BURST's ``calc_weight_nunubar_large``)."""
    if p3 >= p1 + p2 or p3 == 0.0:
        return 0.0
    base = 10.0 * (p1 - p3)**2 + 15.0 * (p1 - p3) * p2 + 6.0 * p2**2
    if p1 >= p3:
        fval = 4.0 * p2**3 / 15.0 / p1**3 / p3**3 * base
    else:
        fval = 4.0 / 15.0 / p1**3 / p3**3 * ((p1 - p3)**5 + p2**3 * base)
    return fval if fval > 0.0 else 0.0


class NuNuBarKernel(eqx.Module):
    """Static tensors of the ν–ν̄ contraction; the layout of ``NuNuKernel``, with K = 2·n_gl."""

    W: jnp.ndarray        # (N, N, K)  prefac·kinemfact1·GL weight·angular weight·p₃³
    idx3: jnp.ndarray     # (N, N, K, M) int
    lw3: jnp.ndarray
    idx4: jnp.ndarray
    lw4: jnp.ndarray
    flafact_s: jnp.ndarray  # (3,3): 1 if m==m' else 1/4   (scattering)
    annfact: jnp.ndarray    # (3,3): 1/4 if m!=m' else 0   (annihilation, flavor-changing)
    fd_logFD3: jnp.ndarray  # (N, N, K)  off-grid closure bases, as in NuNuKernel
    fd_logFD4: jnp.ndarray  # (N, N, K)
    fd_decay_base3: jnp.ndarray  # (N, N, K)
    fd_decay_base4: jnp.ndarray  # (N, N, K)
    brk_lo3: jnp.ndarray   # (N, N, K) int
    brk_hi3: jnp.ndarray
    brk_lo4: jnp.ndarray
    brk_hi4: jnp.ndarray
    nbins: int = eqx.field(static=True)
    nslots: int = eqx.field(static=True)


def build_nunubar_kernel(grid: EpsGrid, n_gl: int = 5, ordpolint: int = 5) -> NuNuBarKernel:
    """Build the ν–ν̄ kernel tensors on the host (NumPy); arguments as ``build_nunu_kernel``.

    p₃ ∈ [0, ε₁ + ε₂] is split at ε₂ (small weight below, large above): K = 2·n_gl slots.
    """
    eps = np.asarray(grid.abscissas, dtype=np.float64)
    w = np.asarray(grid.weights, dtype=np.float64)
    N = eps.shape[0]
    K = 2 * n_gl

    tg, wg = (np.asarray(a, dtype=np.float64) for a in legendre_canonical(n_gl))
    prefac = 128.0 * GFHBAR / 16.0 / (2.0 * PI) ** 3

    W = np.zeros((N, N, K), dtype=np.float64)
    p3 = np.zeros((N, N, K), dtype=np.float64)
    p4 = np.zeros((N, N, K), dtype=np.float64)

    for i in range(N):
        p1 = eps[i]
        if p1 == 0.0:
            continue
        for j in range(N):
            p2 = eps[j]
            if p2 == 0.0:
                continue
            kinemfact1 = w[j] * p1
            s_tot = p1 + p2
            # two segments: (a, b, is_large)
            segs = ((0.0, p2, False), (p2, s_tot, True))
            for sidx, (a, b, is_large) in enumerate(segs):
                half = 0.5 * (b - a)
                mid = 0.5 * (b + a)
                nodes = mid + half * tg
                glw = wg * half
                for k in range(n_gl):
                    slot = sidx * n_gl + k
                    p3v = nodes[k]
                    wt3 = _weight_large(p1, p2, p3v) if is_large else _weight_small(p1, p3v)
                    # BURST's kinemfact2 = glw·weight3·p₃³ (the ε³ measure sits on p₃)
                    W[i, j, slot] = prefac * kinemfact1 * glw[k] * wt3 * p3v**3
                    p3[i, j, slot] = p3v
                    p4[i, j, slot] = s_tot - p3v

    idx3, lw3 = lagrange_window(eps, p3, ordpolint)
    idx4, lw4 = lagrange_window(eps, p4, ordpolint)

    eye = np.eye(3, dtype=np.float64)
    flafact_s = 0.25 * np.ones((3, 3)) + 0.75 * eye      # 1 on diag, 1/4 off
    annfact = 0.25 * (np.ones((3, 3)) - eye)             # 1/4 off diag, 0 on diag
    # off-grid closure bases, as in build_nunu_kernel
    fd_logFD3 = -np.logaddexp(p3, 0.0)
    fd_logFD4 = -np.logaddexp(p4, 0.0)
    xmax = eps[-1]
    base_off = np.logaddexp(xmax, 0.0)
    fd_decay_base3 = np.where(p3 > xmax, fd_logFD3 + base_off, np.inf)
    fd_decay_base4 = np.where(p4 > xmax, fd_logFD4 + base_off, np.inf)
    jl3 = np.searchsorted(eps, p3, side="right"); jl4 = np.searchsorted(eps, p4, side="right")
    brk_lo3 = np.clip(jl3 - 1, 0, N - 1); brk_hi3 = np.clip(jl3, 0, N - 1)
    brk_lo4 = np.clip(jl4 - 1, 0, N - 1); brk_hi4 = np.clip(jl4, 0, N - 1)

    return NuNuBarKernel(
        W=jnp.asarray(W),
        idx3=jnp.asarray(idx3), lw3=jnp.asarray(lw3),
        idx4=jnp.asarray(idx4), lw4=jnp.asarray(lw4),
        flafact_s=jnp.asarray(flafact_s), annfact=jnp.asarray(annfact),
        fd_logFD3=jnp.asarray(fd_logFD3), fd_logFD4=jnp.asarray(fd_logFD4),
        fd_decay_base3=jnp.asarray(fd_decay_base3), fd_decay_base4=jnp.asarray(fd_decay_base4),
        brk_lo3=jnp.asarray(brk_lo3), brk_hi3=jnp.asarray(brk_hi3),
        brk_lo4=jnp.asarray(brk_lo4), brk_hi4=jnp.asarray(brk_hi4),
        nbins=int(N), nslots=int(K),
    )


def nunubar_rates(kernel: NuNuBarKernel, occ, n_chunks: int | None = None,
                  fd_cap_logC: float = float("inf"), range_limit: bool = False,
                  fd_decay: bool = False, monotone_read: bool = False):
    """ν–ν̄ rates (net_s, frs_s, net_a, frs_a), each (3, 2, N), per T_cm⁵.

    Scattering and annihilation, summed over the partner (produced) flavour, the partner bin j and
    the p₃ quadrature; df/dt = T_cm⁵·(net_s + net_a) [s⁻¹].  Arguments as
    :func:`neutt.transport.nunu.nunu_rates`; when chunked, both channels accumulate in one scan.
    """
    N = kernel.nbins
    if n_chunks is None:
        n_chunks = _auto_n_chunks(N, kernel.W.shape[2], kernel.idx3.shape[3])

    if n_chunks == 1:
        logf = jnp.log(occ)
        occ_opp = occ[:, ::-1, :]                                  # opposite-cp occupation
        fdd3 = kernel.fd_decay_base3 if fd_decay else None
        fdd4 = kernel.fd_decay_base4 if fd_decay else None
        brk3 = (kernel.brk_lo3, kernel.brk_hi3) if monotone_read else None
        brk4 = (kernel.brk_lo4, kernel.brk_hi4) if monotone_read else None
        # interpolated f at p₃, p₄, capped at C·FD(p)
        f3 = jnp.exp(jnp.minimum(apply_log_interp(logf, kernel.idx3, kernel.lw3, range_limit=range_limit, fd_decay_base=fdd3, bracket=brk3), fd_cap_logC + kernel.fd_logFD3))  # (flavor,cp,i,j,k)
        f4 = jnp.exp(jnp.minimum(apply_log_interp(logf, kernel.idx4, kernel.lw4, range_limit=range_limit, fd_decay_base=fdd4, bracket=brk4), fd_cap_logC + kernel.fd_logFD4))
        f3o = f3[:, ::-1]                                           # opposite-cp interp at p₃

        Wb = kernel.W[None, None, None]                            # (1,1,1,i,j,k)
        # axes: (m=this flavor, p=partner/produced flavor, n=this cp, i, j, k)
        f1 = occ[:, None, :, :, None, None]                       # (m,1,n,i,1,1)  (m,n) at p₁

        # scattering: (m, n) on legs 1, 4; the partner (p, n̄) on legs 2, 3
        f2s = occ_opp[None, :, :, None, :, None]                  # (1,p,n,1,j,1)  (p,n̄) at p₂
        f3s = f3o[None, :]                                         # (1,p,n,i,j,k)  (p,n̄) at p₃
        f4s = f4[:, None]                                          # (m,1,n,i,j,k)  (m,n) at p₄
        gain_s = (1.0 - f1) * (1.0 - f2s) * f3s * f4s
        loss_s = f1 * f2s * (1.0 - f3s) * (1.0 - f4s)
        flas = kernel.flafact_s[:, :, None, None, None, None]
        net_s = jnp.sum(Wb * (flas * (gain_s - loss_s)), axis=(1, 4, 5))
        frs_s = jnp.sum(Wb * (flas * (gain_s + loss_s)), axis=(1, 4, 5))

        # annihilation: (m, n) + (m, n̄) → (p, n̄) + (p, n), p ≠ m
        f2a = occ_opp[:, None, :, None, :, None]                  # (m,1,n,1,j,1)  (m,n̄) at p₂
        f3a = f3o[None, :]                                         # (1,p,n,i,j,k)  (p,n̄) at p₃
        f4a = f4[None, :]                                          # (1,p,n,i,j,k)  (p,n) at p₄
        gain_a = (1.0 - f1) * (1.0 - f2a) * f3a * f4a
        loss_a = f1 * f2a * (1.0 - f3a) * (1.0 - f4a)
        anna = kernel.annfact[:, :, None, None, None, None]
        net_a = jnp.sum(Wb * (anna * (gain_a - loss_a)), axis=(1, 4, 5))
        frs_a = jnp.sum(Wb * (anna * (gain_a + loss_a)), axis=(1, 4, 5))
        return net_s, frs_s, net_a, frs_a

    # chunked, as in nunu; leg 2 is the opposite-cp occupation, indexed by j
    logf = jnp.log(occ)
    occ_opp = occ[:, ::-1, :]                                 # (p,n,N) opposite-cp
    flas = kernel.flafact_s[:, :, None, None, None, None]
    anna = kernel.annfact[:, :, None, None, None, None]

    j_pad, chunk_j = _pad_to_chunks(N, n_chunks)
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
    # j is axis 2 of occ_opp
    occ_opp_pad = (jnp.pad(occ_opp, [(0, 0), (0, 0), (0, j_pad - N)]) if j_pad != N else occ_opp)
    occ_opp_blk = occ_opp_pad.reshape(occ.shape[0], occ.shape[1], n_chunks, chunk_j)  # (p,n,c,jb)

    Wc = jnp.moveaxis(Wp, 1, 0)
    idx3c = jnp.moveaxis(idx3p, 1, 0)
    lw3c = jnp.moveaxis(lw3p, 1, 0)
    idx4c = jnp.moveaxis(idx4p, 1, 0)
    lw4c = jnp.moveaxis(lw4p, 1, 0)
    fd3c = jnp.moveaxis(fd3p, 1, 0)
    fd4c = jnp.moveaxis(fd4p, 1, 0)
    fdd3c = jnp.moveaxis(fdd3p, 1, 0)
    fdd4c = jnp.moveaxis(fdd4p, 1, 0)
    bl3c = jnp.moveaxis(bl3p, 1, 0); bh3c = jnp.moveaxis(bh3p, 1, 0)
    bl4c = jnp.moveaxis(bl4p, 1, 0); bh4c = jnp.moveaxis(bh4p, 1, 0)
    occ_opp_c = jnp.moveaxis(occ_opp_blk, 2, 0)              # (c,p,n,jb)

    f1 = occ[:, None, :, :, None, None]                      # (m,1,n,i,1,1)  j-independent

    def body(carry, blk):
        net_s_a, frs_s_a, net_a_a, frs_a_a = carry
        (W_b, idx3_b, lw3_b, idx4_b, lw4_b, occ_opp_b, fd3_b, fd4_b, fdd3_b, fdd4_b,
         bl3_b, bh3_b, bl4_b, bh4_b) = blk
        fdd3 = fdd3_b if fd_decay else None
        fdd4 = fdd4_b if fd_decay else None
        brk3 = (bl3_b, bh3_b) if monotone_read else None
        brk4 = (bl4_b, bh4_b) if monotone_read else None
        f3 = jnp.exp(jnp.minimum(apply_log_interp(logf, idx3_b, lw3_b, range_limit=range_limit, fd_decay_base=fdd3, bracket=brk3), fd_cap_logC + fd3_b))  # (m,n,i,jb,k)
        f4 = jnp.exp(jnp.minimum(apply_log_interp(logf, idx4_b, lw4_b, range_limit=range_limit, fd_decay_base=fdd4, bracket=brk4), fd_cap_logC + fd4_b))
        f3o = f3[:, ::-1]                                    # opposite-cp interp at p₃
        Wb = W_b[None, None, None]                           # (1,1,1,i,jb,k)

        # scattering
        f2s = occ_opp_b[None, :, :, None, :, None]           # (1,p,n,1,jb,1)
        f3s = f3o[None, :]
        f4s = f4[:, None]
        gain_s = (1.0 - f1) * (1.0 - f2s) * f3s * f4s
        loss_s = f1 * f2s * (1.0 - f3s) * (1.0 - f4s)
        net_s_a = net_s_a + jnp.sum(Wb * (flas * (gain_s - loss_s)), axis=(1, 4, 5))
        frs_s_a = frs_s_a + jnp.sum(Wb * (flas * (gain_s + loss_s)), axis=(1, 4, 5))

        # annihilation
        f2a = occ_opp_b[:, None, :, None, :, None]           # (m,1,n,1,jb,1)
        f3a = f3o[None, :]
        f4a = f4[None, :]
        gain_a = (1.0 - f1) * (1.0 - f2a) * f3a * f4a
        loss_a = f1 * f2a * (1.0 - f3a) * (1.0 - f4a)
        net_a_a = net_a_a + jnp.sum(Wb * (anna * (gain_a - loss_a)), axis=(1, 4, 5))
        frs_a_a = frs_a_a + jnp.sum(Wb * (anna * (gain_a + loss_a)), axis=(1, 4, 5))
        return (net_s_a, frs_s_a, net_a_a, frs_a_a), None

    z = jnp.zeros((occ.shape[0], occ.shape[1], N), dtype=occ.dtype)
    (net_s, frs_s, net_a, frs_a), _ = jax.lax.scan(
        body, (z, z, z, z),
        (Wc, idx3c, lw3c, idx4c, lw4c, occ_opp_c, fd3c, fd4c, fdd3c, fdd4c,
         bl3c, bh3c, bl4c, bh4c))
    return net_s, frs_s, net_a, frs_a


def nunubar_dfdt(kernel: NuNuBarKernel, occ, t_cm_mev: float, n_chunks: int | None = None):
    """df/dt from ν–ν̄ scattering + annihilation, in s⁻¹ (T_cm⁵·(net_s+net_a))."""
    net_s, _, net_a, _ = nunubar_rates(kernel, occ, n_chunks=n_chunks)
    return t_cm_mev**5 * (net_s + net_a)
