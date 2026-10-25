"""Density-matrix ν–ν and ν–ν̄ collisions: the matrix versions of ``nunu`` and ``nunubar``.

.. code::

    dρ(p₁)/dt|coll = T_cm⁵·(A_gain − {B_loss, ρ(p₁)})

Unlike the ν–e kernels, the Sigl–Raffelt statistical factor has a ``Tr[X]·Y + X·Y``
(identical-particle exchange) structure that couples the partner matrices, so the 3×3 products are
taken per quadrature node.  The outer anticommutator with ρ₁ is still taken once, after the node sum
(``collide_matrix.assemble_from_phi``).

Node forms.  ν–ν: all four legs carry this cp.  ν–ν̄: legs 1, 4 carry this cp, legs 2, 3 the
opposite one (ρ̄).

.. code::

    ν–ν       B₁     = Tr[ρ₃(1−ρ₂)]·ρ₄     + ρ₃(1−ρ₂)ρ₄
              B₂     = Tr[ρ₄(1−ρ₂)]·ρ₃     + ρ₄(1−ρ₂)ρ₃
              B₁ʳᵉᵛ  = Tr[(1−ρ₃)ρ₂]·(1−ρ₄) + (1−ρ₃)ρ₂(1−ρ₄)
              B₂ʳᵉᵛ  = Tr[(1−ρ₄)ρ₂]·(1−ρ₃) + (1−ρ₄)ρ₂(1−ρ₃)

    ν–ν̄       M₁     = Tr[(1−ρ̄₂)ρ̄₃]·ρ₄     + (1−ρ̄₂)ρ̄₃ρ₄
              M₂     = Tr[ρ₄ρ̄₃]·(1−ρ̄₂)     + ρ₄ρ̄₃(1−ρ̄₂)
              M₁ʳᵉᵛ  = Tr[ρ̄₂(1−ρ̄₃)]·(1−ρ₄) + ρ̄₂(1−ρ̄₃)(1−ρ₄)
              M₂ʳᵉᵛ  = Tr[(1−ρ₄)(1−ρ̄₃)]·ρ̄₂ + (1−ρ₄)(1−ρ̄₃)ρ̄₂

    Φ_for = Σ_{j,k} (W/8)·(first + second)ᶠᵒʳ      Φ_rev = Σ_{j,k} (W/8)·(first + second)ʳᵉᵛ

with W the classical weight tensor (``build_nunu_kernel`` / ``build_nunubar_kernel``), prefactor
included.  For ν–ν, qburst integrates p₃ over half the range and symmetrises 3↔4; with the classical
full-range quadrature that is ½(B₁ + B₂) at prefac/4.  ν–ν̄ is prefac/8 with no half-sum.  Both give
W/8.  On a diagonal ρ the kernels reduce to the classical ones, the classical flavour factors
(``flafact = ½ + ½δ`` for ν–ν; ``flafact_s = ¼ + ¾δ`` and ``annfact = ¼(1−δ)`` for ν–ν̄) all
following from the one matrix expression (``tests/test_qke_collide_nunu.py``).  The ν–ν̄ ``net`` is
the classical ``net_s + net_a``.

The ν̄ transpose.  qburst transposes every leg's flavour axes for the ν̄ and transposes net/frs back.
For F built from Hermitian matrices with real coefficients, ``F[ρᵀ]ᵀ = F[ρ]†``, so the round trip is
the identity when F is Hermitian, which the 3↔4 (ν–ν̄: 1↔2) symmetrisation makes it.  The kernels
omit the transpose (``test_transpose_invariance``).

Off-grid reads.  The ν–ν nodes are static, so the classical kernel's precomputed windows
(``idx3/lw3/idx4/lw4``) are reused: the diagonal is exactly the classical ``apply_log_interp`` read,
and the coherence is the bounded fraction ``ρ_ab/√(ρ_aa ρ_bb)`` on the same stencil.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ...interp import apply_log_interp
from ..nunu import _pad_j, _pad_to_chunks
from .collide_matrix import _clamp_unit_modulus, assemble_from_phi, coherence_fraction
from .coherent import hermitize

_DIAG3 = jnp.arange(3)
_EYE3 = jnp.eye(3, dtype=jnp.complex128)

#: Off-grid coherence reads.  There is no "linear" mode: the classical ν–ν path reads no coherence.
INTERP_MODES = ("fraction", "lagrange")

#: Budget, in complex elements, for the (3, 3, N, chunk_j, K, M) partner reads (1.5e8 complex128
#: ≈ 2.4 GB).  Separate from ``nunu._CHUNK_ELEM_BUDGET``, which counts f64 elements of (3, 2, …).
_CHUNK_ELEM_BUDGET = 1.5e8


def _auto_n_chunks(N: int, K: int, M: int) -> int:
    """Static ``n_chunks`` so the gather tensor (3,3,N,chunk_j,K,M) fits the budget."""
    per_j = 3 * 3 * N * K * M
    chunk_j = max(1, int(_CHUNK_ELEM_BUDGET // per_j))
    return max(1, math.ceil(N / chunk_j))


def _mm(X, Y):
    """3×3 matrix product over the trailing axes, as a broadcast contraction.

    Not ``jnp.matmul``/``einsum``: at 3×3 those lower to a batched GEMM, about 9× slower per product
    here, which also cannot fuse with the surrounding node algebra.
    """
    return (X[..., :, :, None] * Y[..., None, :, :]).sum(-2)


def _tr(X):
    """``Tr[X]`` broadcast back to a (…,1,1) scalar factor."""
    return jnp.trace(X, axis1=-2, axis2=-1)[..., None, None]


def _unpack_rho(rho):
    """``(rho, rho_ab, logf_diag)``: ρ as complex128, ρ matrix-axes-first (2, 3, 3, N), log diag (2, 3, N)."""
    rho = jnp.asarray(rho).astype(jnp.complex128)
    rho_ab = jnp.transpose(rho, (0, 2, 3, 1))
    logf_diag = jnp.log(jnp.real(rho[:, :, _DIAG3, _DIAG3])).transpose(0, 2, 1)
    return rho, rho_ab, logf_diag


def _interp_leg(rho_ab, logf_diag, idx, lw, mode, fd_logFD, fd_cap_logC,
                fd_decay_base, bracket):
    """ρ at the kernel's precomputed off-grid windows → ``S + (3, 3)``.

    ``rho_ab`` is (3, 3, N) and ``logf_diag`` (3, N); ``idx``/``lw`` have point shape S plus the
    window axis.  The diagonal is the classical ``apply_log_interp`` read with the same FD cap and
    closures, so it is bit-identical to the classical kernel's.
    """
    f_diag = jnp.exp(jnp.minimum(
        apply_log_interp(logf_diag, idx, lw, fd_decay_base=fd_decay_base, bracket=bracket),
        fd_cap_logC + fd_logFD))                                        # (3,) + S

    if mode == "fraction":
        r = _clamp_unit_modulus(jnp.sum(lw * coherence_fraction(rho_ab)[..., idx], axis=-1))
        sq = jnp.sqrt(jnp.maximum(f_diag, 0.0)).astype(r.dtype)         # (3,) + S
        rho_S = r * sq[:, None] * sq[None, :]                           # (3, 3) + S
    elif mode == "lagrange":
        rho_S = jnp.sum(lw * rho_ab[..., idx], axis=-1)
    else:
        raise ValueError(f"mode must be one of {INTERP_MODES}, got {mode!r}")

    nS = idx.ndim - 1
    out = jnp.transpose(rho_S, tuple(range(2, 2 + nS)) + (0, 1))        # S + (3, 3)
    f_last = jnp.transpose(f_diag, tuple(range(1, 1 + nS)) + (0,))      # S + (3,)
    return out.at[..., _DIAG3, _DIAG3].set(f_last.astype(out.dtype))


def _nunu_node(rho2, rho3, rho4):
    """``(B₁+B₂, B₁ʳᵉᵛ+B₂ʳᵉᵛ)`` for ν–ν, per quadrature node.

    Each sum is Hermitian (real traces; the two triple products are each other's adjoints), so
    ``hermitize`` only removes round-off.
    """
    imr2 = _EYE3 - rho2
    x23 = _mm(rho3, imr2)
    x42 = _mm(rho4, imr2)
    fwd = (_tr(x23) * rho4 + _mm(x23, rho4)
           + _tr(x42) * rho3 + _mm(x42, rho3))
    imr3 = _EYE3 - rho3
    imr4 = _EYE3 - rho4
    y32 = _mm(imr3, rho2)
    y42 = _mm(imr4, rho2)
    rev = (_tr(y32) * imr4 + _mm(y32, imr4)
           + _tr(y42) * imr3 + _mm(y42, imr3))
    return fwd, rev


def _nunubar_node(rho2b, rho3b, rho4):
    """``(M₁+M₂, M₁ʳᵉᵛ+M₂ʳᵉᵛ)`` for ν–ν̄.  ``rho2b``/``rho3b`` are the opposite-cp legs."""
    imr2 = _EYE3 - rho2b
    p = _mm(imr2, rho3b)
    q = _mm(rho4, rho3b)
    fwd = _tr(p) * rho4 + _mm(p, rho4) + _tr(q) * imr2 + _mm(q, imr2)
    imr3 = _EYE3 - rho3b
    imr4 = _EYE3 - rho4
    pr = _mm(rho2b, imr3)
    qr = _mm(imr4, imr3)
    rev = _tr(pr) * imr4 + _mm(pr, imr4) + _tr(qr) * rho2b + _mm(qr, rho2b)
    return fwd, rev


def _core(kernel, rho, kind, n_chunks=None, offdiag_interp="fraction",
          fd_cap_logC=float("inf"), fd_decay=False, monotone_read=False,
          hermitian_project=True):
    """Shared driver for the two ν-only channels (see :func:`nunu_matrix_core`)."""
    if kind not in ("nunu", "nunubar"):
        raise ValueError(f'kind must be "nunu" or "nunubar", got {kind!r}')
    if offdiag_interp not in INTERP_MODES:
        raise ValueError(f"offdiag_interp must be one of {INTERP_MODES}, got {offdiag_interp!r}")

    N, K, M = kernel.nbins, kernel.idx3.shape[2], kernel.idx3.shape[3]
    if n_chunks is None:
        n_chunks = _auto_n_chunks(N, K, M)

    rho, rho_ab, logf_diag = _unpack_rho(rho)

    # Chunk the partner bin j as the classical kernels do; padded bins have W = 0.
    j_pad, chunk_j = _pad_to_chunks(N, n_chunks)

    def blk(a, fill):
        """(N, N, K, …) kernel tensor → (n_chunks, N, chunk_j, K, …), j-padded."""
        return jnp.moveaxis(
            _pad_j(a, j_pad, fill).reshape(N, n_chunks, chunk_j, *a.shape[2:]), 1, 0)

    xs_static = (blk(kernel.W, 0.0),
                 blk(kernel.idx3, 0), blk(kernel.lw3, 0.0),
                 blk(kernel.idx4, 0), blk(kernel.lw4, 0.0),
                 blk(kernel.fd_logFD3, 0.0), blk(kernel.fd_logFD4, 0.0),
                 blk(kernel.fd_decay_base3, jnp.inf), blk(kernel.fd_decay_base4, jnp.inf),
                 blk(kernel.brk_lo3, 0), blk(kernel.brk_hi3, 0),
                 blk(kernel.brk_lo4, 0), blk(kernel.brk_hi4, 0))
    # Leg 2 is on the grid at the partner bin j (rho_ab's last axis), so it is padded and blocked there.
    rho_ab_p = (jnp.pad(rho_ab, [(0, 0)] * 3 + [(0, j_pad - N)]) if j_pad != N else rho_ab)
    rho2_c = jnp.transpose(rho_ab_p.reshape(2, 3, 3, n_chunks, chunk_j),
                           (0, 3, 4, 1, 2))                          # (2, c, jb, 3, 3)

    node = _nunu_node if kind == "nunu" else _nunubar_node
    net = frs = a_gain = b_loss = jnp.zeros((2, N, 3, 3), dtype=jnp.complex128)
    for cp0 in range(2):
        # ν–ν: every leg at this cp.  ν–ν̄: legs 2, 3 at the opposite cp, legs 1, 4 at this one.
        cp23 = cp0 if kind == "nunu" else 1 - cp0

        def body(carry, blk_in, cp0=cp0, cp23=cp23):
            pf, pr = carry
            (W_b, i3, l3, i4, l4, f3, f4, d3, d4, b3l, b3h, b4l, b4h, rho2_b) = blk_in
            rho3 = _interp_leg(rho_ab[cp23], logf_diag[cp23], i3, l3, offdiag_interp, f3,
                               fd_cap_logC,
                               d3 if fd_decay else None,
                               (b3l, b3h) if monotone_read else None)   # (N, jb, K, 3, 3)
            rho4 = _interp_leg(rho_ab[cp0], logf_diag[cp0], i4, l4, offdiag_interp, f4,
                               fd_cap_logC,
                               d4 if fd_decay else None,
                               (b4l, b4h) if monotone_read else None)
            fwd, rev = node(rho2_b[None, :, None], rho3, rho4)          # (1, jb, 1, 3, 3) leg 2
            w = (0.125 * W_b)[..., None, None].astype(jnp.complex128)   # (N, jb, K, 1, 1)
            return (pf + jnp.sum(w * fwd, axis=(1, 2)),
                    pr + jnp.sum(w * rev, axis=(1, 2))), None

        z = jnp.zeros((N, 3, 3), dtype=jnp.complex128)
        (phi_for, phi_rev), _ = jax.lax.scan(body, (z, z), xs_static + (rho2_c[cp23],))

        net_cp, frs_cp, a_cp, b_cp = assemble_from_phi(phi_for, phi_rev, rho[cp0])
        net = net.at[cp0].add(net_cp)
        frs = frs.at[cp0].add(frs_cp)
        a_gain = a_gain.at[cp0].add(a_cp)
        b_loss = b_loss.at[cp0].add(b_cp)

    out = (net, frs, a_gain, b_loss)
    return tuple(hermitize(a) for a in out) if hermitian_project else out


# ---- public API (as collide_nue / collide) -----------------------------------------------------

def nunu_matrix_core(kernel, rho, **kw):
    """ν–ν flavour-matrix collision: ``(net, frs, A_gain, B_loss)``, each ``(2, N, 3, 3)``.

    Parameters
    ----------
    kernel : NuNuKernel
        The classical kernel (:func:`neutt.transport.nunu.build_nunu_kernel`); its kinematics,
        weights and interpolation windows are reused.
    rho : (2, N, 3, 3) complex
        ``rho[cp, bin, a, b]`` (:mod:`.coherent`).
    n_chunks : int or None
        Static partner-bin (j) chunk count: 1 = one block (the classical sum order), None = the
        fewest chunks under ``_CHUNK_ELEM_BUDGET``.
    offdiag_interp : {"fraction", "lagrange"}
        Off-grid coherence read; "fraction" (default) is bounded.
    fd_cap_logC, fd_decay, monotone_read
        Tail closures of the classical interpolant.
    hermitian_project : bool
        Hermitize the outputs; False exposes the raw Hermiticity residual.

    Returns
    -------
    net, frs, A_gain, B_loss : (2, N, 3, 3) complex
        Per T_cm⁵ (the classical prefactor convention); dρ/dt|coll = T_cm⁵·net.
    """
    return _core(kernel, rho, "nunu", **kw)


def nunubar_matrix_core(kernel, rho, **kw):
    """ν–ν̄ flavour-matrix collision, as :func:`nunu_matrix_core` with a ``NuNuBarKernel``.

    One operator covers both classical sub-channels: ``net`` is the classical ``net_s + net_a``.
    """
    return _core(kernel, rho, "nunubar", **kw)


def nunu_matrix_rates(kernel, rho, **kw):
    """``(net, frs)`` only — the classical-kernel-shaped return."""
    net, frs, _, _ = nunu_matrix_core(kernel, rho, **kw)
    return net, frs


def nunubar_matrix_rates(kernel, rho, **kw):
    """``(net, frs)`` only — the classical-kernel-shaped return."""
    net, frs, _, _ = nunubar_matrix_core(kernel, rho, **kw)
    return net, frs


def nunu_matrix_affine(kernel, rho, **kw):
    """``(A_gain, B_loss)`` — the affine form ``net = A_gain − {B_loss, ρ}``."""
    _, _, a, b = nunu_matrix_core(kernel, rho, **kw)
    return a, b


def nunubar_matrix_affine(kernel, rho, **kw):
    """``(A_gain, B_loss)`` — the affine form ``net = A_gain − {B_loss, ρ}``."""
    _, _, a, b = nunubar_matrix_core(kernel, rho, **kw)
    return a, b


def nunu_matrix_dfdt(kernel, rho, t_cm_mev, **kw):
    """dρ/dt|coll = T_cm⁵·net from ν–ν [s⁻¹]."""
    net, _ = nunu_matrix_rates(kernel, rho, **kw)
    return t_cm_mev ** 5 * net


def nunubar_matrix_dfdt(kernel, rho, t_cm_mev, **kw):
    """dρ/dt|coll = T_cm⁵·net from ν–ν̄ [s⁻¹]."""
    net, _ = nunubar_matrix_rates(kernel, rho, **kw)
    return t_cm_mev ** 5 * net
