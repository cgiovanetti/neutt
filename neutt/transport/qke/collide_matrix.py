"""Shared pieces of the density-matrix (Sigl–Raffelt) collision kernels.

Used by ``collide`` (νν̄ → e⁺e⁻), ``collide_nue`` (ν–e±) and ``collide_nunu`` (ν–ν, ν–ν̄):

1. flavour-pair bookkeeping (``PAIR_A``/``PAIR_B``, ``offdiag_only``, ``diag_only``);
2. the pair coupling tensors ``term1_pair``/``term2_pair``: the classical ``(nn, m, n)`` ν–e
   coupling tensors with the flavour index m promoted to a flavour pair (a, b);
3. the off-grid read of the partner density matrix, with a bounded coherence interpolant;
4. the assembly ``Φ_for, Φ_rev → (net, frs, A_gain, B_loss)``.

``Y_L``, ``Y_R`` and the angular weights are all diagonal in flavour, so for the ν–e kernels
``(Σ_l Y_l ρ₄ W_l)_ab = W_ab·(ρ₄)_ab``: the matrix kernel is the classical one with m → (a, b) and
no matrix product inside the integral.  On a diagonal ρ every matrix kernel reduces to its classical
counterpart.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ...constants import SIN2THW
from ...interp import ORDPOLINT, _lagrange_weights_at, interp_log_occ_dynamic
from ..assemble import OCC_FLOOR

N_FLAVOR = 3

# ---- 1. flavour-pair bookkeeping ---------------------------------------------------------------

#: Row/column indices of the three off-diagonal flavour pairs (e,μ), (e,τ), (μ,τ).
PAIR_A, PAIR_B = (np.asarray(a) for a in np.triu_indices(N_FLAVOR, k=1))
N_PAIRS = PAIR_A.size

#: (3, 3) bool, True off the diagonal (the coherence sector).
OFFDIAG_MASK = ~np.eye(N_FLAVOR, dtype=bool)

_DIAG_IDX = jnp.arange(N_FLAVOR)


def offdiag_only(M):
    """Zero the flavour diagonal of ``M[..., 3, 3]``, keeping the coherences."""
    return M.at[..., _DIAG_IDX, _DIAG_IDX].set(0.0)


def diag_only(M):
    """Keep the flavour diagonal of ``M[..., 3, 3]``, zeroing the coherences."""
    d = jnp.diagonal(M, axis1=-2, axis2=-1)
    return jnp.zeros_like(M).at[..., _DIAG_IDX, _DIAG_IDX].set(d)


def g_l_vector(sin2thw: float = SIN2THW) -> np.ndarray:
    """``g_{L,a} = 2 sin²θ_W + fla_a``, with fla = (+1, −1, −1) for (νe, νμ, ντ).

    ``Y_L = ½ diag(g_L)`` and ``Y_R = sin²θ_W·1`` are the ν–e coupling matrices.  ``g_{L,μ} = 2s − 1``
    is negative, so the eμ and eτ products ``g_a g_b`` are negative and the collisional gain
    enhances the eμ/eτ damping.
    """
    fla = np.array([1.0, -1.0, -1.0])
    return 2.0 * sin2thw + fla


# ---- 2. pair coupling tensors ------------------------------------------------------------------

_NUE_KINDS = ("nuer1", "nuer2", "nuepma")


def pair_coupling_tensors(kind: str, sin2thw: float = SIN2THW,
                          convention: str = "matrix"):
    """Flavour-pair coupling tensors ``(term1_pair, term2_pair)`` for one ν–e kernel.

    The classical kernels weight each node by ``term1·(I1 + c2·meps²·I2)``.  ``term2 ≡ term1·c2``
    promotes linearly (a ``c2_pair`` would divide by a small number), so it is returned instead of
    ``c2``.  The promotion

    .. code::

        g_{L,m}²  →  g_{L,a} g_{L,b}          (quadratic term)
        g_{L,m}   →  ½ (g_{L,a} + g_{L,b})    (cross term)
        g_R = 2s  →  g_R                      (flavour-blind)

    gives three branch forms, with ``X_ab ≡ 32 s (g_a + g_b)``:

    ==========  =========================  ================
    branch      ``term1_pair``             ``term2_pair``
    ==========  =========================  ================
    ``LL(−)``   ``32 g_a g_b``             ``−X_ab``
    ``LL(+)``   ``32 g_a g_b``             ``+X_ab``
    ``RR``      ``128 s²``                 ``+X_ab``
    ==========  =========================  ================

    At ``a = b`` these are the classical tensors, for every branch of every kernel
    (:func:`check_pair_tensor_diagonal_reduction`).

    Parameters
    ----------
    kind : {"nuer1", "nuer2", "nuepma"}
        ``nuer1``/``nuer2`` are ``n ↔ nn`` swaps of each other; ``nuepma`` has no ``parfact`` and
        both of its ``meps² I2`` signs are ``+``.
    convention : {"matrix", "classical"}
        Meaning of the leading axis.  ``"matrix"``: the lepton parity ``nn`` of Fortran's
        ``calc_weight_*_mat``.  ``"classical"``: the leading axis of the classical
        ``_coupling_tensors*`` builder, which for ``nuepma`` is ``swapflag = 1 − nn`` (for
        ``nuer1``/``nuer2`` the two conventions coincide).

    Returns
    -------
    term1_pair, term2_pair : (2, 3, 3, 2)
        Indexed ``[nn (or swapflag), a, b, n]``; real and symmetric under ``a ↔ b`` (``Y_R ∝ 1``),
        so ``W ⊙ ρ₄`` and hence ``net`` are Hermitian.

    Notes
    -----
    ``term2_pair`` must multiply its own kernel's ``meps²·I2``.  The three ``I2`` antiderivatives
    differ in sign, each matching its Fortran routine, and pairing with the wrong one flips the cross
    term:

    .. code::

        nuer1 : _R2  (x, A) = ½(A x − x³/3)      (r1int_m_2)
        nuer2 : _R2m2(x, B) = ½(x³/3 − B x)      (r2int_m_2)
        nuepma: _L2  (x, C) = ½(x³/3 − C x)      (rint_l_2)
    """
    if kind not in _NUE_KINDS:
        raise ValueError(f"kind must be one of {_NUE_KINDS}, got {kind!r}")
    if convention not in ("matrix", "classical"):
        raise ValueError(f"convention must be 'matrix' or 'classical', got {convention!r}")

    s = float(sin2thw)
    g = g_l_vector(s)                                   # (3,)
    ll_t1 = 32.0 * np.outer(g, g)                       # 32 g_a g_b
    rr_t1 = np.full((N_FLAVOR, N_FLAVOR), 128.0 * s ** 2)
    x_t2 = 32.0 * s * (g[:, None] + g[None, :])         # 32 s (g_a + g_b)

    # Branch selection from Fortran's calc_weight_*_mat (leading axis = lepton parity nn).  The
    # Y_L lead carries a parity gate and the cross term carries parfact:
    #   nuer1  (parfactn=+1): Y_L gate = mod(n+nn+1,2) = [n==nn];  parfact = +1 iff n==nn
    #   nuer2  (parfactn=-1): Y_L gate = mod(n+nn,2)   = [n!=nn];  parfact = -1 iff n==nn
    #   nuepma (no parfact):  Y_L gate = mod(n+nn,2)   = [n!=nn];  both cross signs +
    # so, with W_ab = Σ_l (Y_l)_a (weightin_l)_b:
    #   nuer1 : n==nn -> LL(-)   n!=nn -> RR(+)
    #   nuer2 : n==nn -> RR(+)   n!=nn -> LL(-)
    #   nuepma: n==nn -> RR(+)   n!=nn -> LL(+)
    term1 = np.zeros((2, N_FLAVOR, N_FLAVOR, 2))
    term2 = np.zeros((2, N_FLAVOR, N_FLAVOR, 2))
    for nn in range(2):          # lepton parity
        for n in range(2):       # cp
            if kind == "nuer1":
                use_ll, ll_sign = (nn == n), -1.0
            elif kind == "nuer2":
                use_ll, ll_sign = (nn != n), -1.0
            else:  # nuepma
                use_ll, ll_sign = (nn != n), +1.0
            term1[nn, :, :, n] = ll_t1 if use_ll else rr_t1
            term2[nn, :, :, n] = (ll_sign if use_ll else +1.0) * x_t2

    if convention == "classical" and kind == "nuepma":
        # the classical nuepma builder is indexed by swapflag = 1 − nn
        term1, term2 = term1[::-1], term2[::-1]
    return jnp.asarray(term1), jnp.asarray(term2)


def check_pair_tensor_diagonal_reduction(kind: str, sin2thw: float = SIN2THW):
    """Max abs difference of ``term{1,2}_pair[nn, a, a, n]`` from the classical ``term1``, ``term1·c2``.

    Pure numpy; both numbers should be round-off.  Uses ``convention="classical"`` so ``nuepma`` is
    compared on the classical builder's leading axis.
    """
    from .. import nuepma as _nuepma, nuer1 as _nuer1, nuer2 as _nuer2

    builder = {"nuer1": _nuer1._coupling_tensors,
               "nuer2": _nuer2._coupling_tensors_r2,
               "nuepma": _nuepma._coupling_tensors_pma}[kind]
    t1_c, c2_c = (np.asarray(a) for a in builder(sin2thw))       # (2, 3, 2)
    t2_c = t1_c * c2_c
    t1_p, t2_p = (np.asarray(a) for a in
                  pair_coupling_tensors(kind, sin2thw, convention="classical"))
    d = np.arange(N_FLAVOR)
    return (np.abs(t1_p[:, d, d, :] - t1_c).max(),
            np.abs(t2_p[:, d, d, :] - t2_c).max())


# ---- 3. off-grid read of the partner density matrix --------------------------------------------

INTERP_MODES = ("fraction", "linear", "lagrange")


def _window_dynamic(abscissas, query, ord: int = ORDPOLINT):
    """``(idx, weights)`` of the order-``ord`` window at traced ``query``, as in ``interp_log_occ_dynamic``.

    The coherence and the diagonal must be read on the same stencil, or the reconstructed ρ_ab in
    :func:`interp_partner_rho` mixes two windows.
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
    return idx, _lagrange_weights_at(abscissas[idx], q)


def interp_lagrange_linear(values, abscissas, query, ord: int = ORDPOLINT):
    """Order-``ord`` Lagrange interpolation of ``values[..., N]`` in linear space.

    Coherences are complex and change sign, so the log-space occupation interpolant does not apply.
    Used on the bounded coherence fraction, and on ρ_ab directly only in the ``"lagrange"`` mode.
    """
    idx, lw = _window_dynamic(abscissas, query, ord)
    return jnp.sum(lw * values[..., idx], axis=-1)


def interp_linear_2pt(values, abscissas, query):
    """2-point linear interpolation of ``values[..., N]`` at traced ``query`` (the ``"linear"`` mode).

    Only the bracket is clamped, so off-grid queries are extrapolated linearly (unbounded).
    """
    abscissas = jnp.asarray(abscissas)
    N = abscissas.shape[0]
    q = jnp.asarray(query)
    hi = jnp.clip(jnp.searchsorted(abscissas, q, side="right"), 1, N - 1)
    lo = hi - 1
    x0, x1 = abscissas[lo], abscissas[hi]
    t = jnp.where(x1 > x0, (q - x0) / jnp.where(x1 > x0, x1 - x0, 1.0), 0.0)
    v0, v1 = values[..., lo], values[..., hi]
    return v0 + (v1 - v0) * t.astype(v0.dtype)


def coherence_fraction(rho_partner):
    """``r_ab = ρ_ab / sqrt(ρ_aa ρ_bb)`` on the grid.

    ``|r_ab| ≤ 1`` for a positive-semidefinite ρ, so after the unit-modulus clamp an interpolated
    coherence is bounded by ``sqrt(f_a f_b)``, unlike an off-grid extrapolation of ρ_ab itself.  Each
    diagonal is floored at ``OCC_FLOOR`` before the product (``OCC_FLOOR²`` underflows to 0), as
    ``assemble`` floors occupations.

    Parameters
    ----------
    rho_partner : (3, 3, N) complex
        Matrix axes first.
    """
    d = jnp.real(rho_partner[_DIAG_IDX, _DIAG_IDX, :])                   # (3, N)
    d = jnp.maximum(d, OCC_FLOOR)                                        # (3, N)
    denom = jnp.sqrt(d[:, None, :] * d[None, :, :])
    return rho_partner / denom.astype(rho_partner.dtype)


def _clamp_unit_modulus(r):
    """Project onto ``|r| ≤ 1`` (a no-op on the fraction of a positive-semidefinite ρ)."""
    mag = jnp.abs(r)
    safe = jnp.where(mag > 0.0, mag, 1.0)
    return r * jnp.where(mag > 1.0, 1.0 / safe, 1.0).astype(r.dtype)


def interp_partner_rho(rho_partner, eps, p4, logf_diag_partner, mode: str = "fraction",
                       ord: int = ORDPOLINT, **log_kw):
    """The partner density matrix ρ̄ at traced momenta ``p4``.

    The diagonal is always the classical log-space read (``interp_log_occ_dynamic``), so on a
    diagonal ρ the matrix kernel reads exactly what the classical kernel does.  ``mode`` sets the
    coherence read:

    ``"fraction"`` (default)
        Interpolate ``r_ab`` (:func:`coherence_fraction`) on the order-``ord`` stencil, clamp
        ``|r_ab| ≤ 1`` and set ``ρ_ab = r_ab·sqrt(f_a f_b)`` from the interpolated diagonals.
        Bounded, and the coherence inherits the diagonal's tail closures.
    ``"linear"``
        2-point linear on ρ_ab (:func:`interp_linear_2pt`); unbounded off grid.
    ``"lagrange"``
        Order-``ord`` Lagrange on ρ_ab; unbounded off grid.

    Parameters
    ----------
    rho_partner : (3, 3, N) complex
        Partner density matrix on the ε-grid, matrix axes first.
    eps : (N,) float
        Grid abscissas.
    p4 : array of shape S
        Query momenta (traced).
    logf_diag_partner : (3, N) float
        log of ``rho_partner``'s diagonal.
    **log_kw
        Tail closures forwarded to ``interp_log_occ_dynamic`` (``fd_decay``, ``monotone_read``,
        ``fd_cap_logC``).

    Returns
    -------
    S + (3, 3) complex
        ρ̄(p4), matrix axes last.
    """
    if mode not in INTERP_MODES:
        raise ValueError(f"mode must be one of {INTERP_MODES}, got {mode!r}")

    f4_diag = jnp.exp(interp_log_occ_dynamic(logf_diag_partner, eps, p4, **log_kw))  # (3,) + S

    if mode == "linear":
        rho4_ab = interp_linear_2pt(rho_partner, eps, p4)                 # (3, 3) + S
    elif mode == "lagrange":
        rho4_ab = interp_lagrange_linear(rho_partner, eps, p4, ord)
    else:
        r4 = _clamp_unit_modulus(
            interp_lagrange_linear(coherence_fraction(rho_partner), eps, p4, ord))
        sq = jnp.sqrt(jnp.maximum(f4_diag, 0.0)).astype(r4.dtype)         # (3,) + S
        rho4_ab = r4 * sq[:, None] * sq[None, :]                          # (3, 3) + S

    nS = len(jnp.shape(p4))
    rho4 = jnp.transpose(rho4_ab, tuple(range(2, 2 + nS)) + (0, 1))       # S + (3, 3)
    f4_last = jnp.transpose(f4_diag, tuple(range(1, 1 + nS)) + (0,))      # S + (3,)
    return rho4.at[..., _DIAG_IDX, _DIAG_IDX].set(f4_last.astype(rho4.dtype))


# ---- 4. Φ → (net, frs, A_gain, B_loss) ---------------------------------------------------------

def anticomm(X, Y):
    """Anticommutator ``{X, Y} = XY + YX`` for batched ``(..., 3, 3)`` matrices."""
    XY = jnp.matmul(X, Y)
    return XY + jnp.matmul(Y, X)


def assemble_from_phi(phi_for, phi_rev, rho1):
    """``(net, frs, A_gain, B_loss)`` from the node-summed ``Φ_for``, ``Φ_rev`` and the local ρ₁.

    ρ₁ is node-independent and the integrand is linear in the partner ρ, so all quadrature precedes
    the 3×3 products.  With ``{X, I} = 2X``:

    .. code::

        forfact = {Φ_for, I − ρ₁},   revfact = {Φ_rev, ρ₁}
        net = forfact − revfact = 2Φ_for − {Φ_for + Φ_rev, ρ₁}
        frs = forfact + revfact = 2Φ_for − {Φ_for − Φ_rev, ρ₁}
        A_gain = 2Φ_for,   B_loss = Φ_for + Φ_rev,   net = A_gain − {B_loss, ρ₁}

    ``B_loss`` is Hermitian, and positive-semidefinite on a physical state (0 ⪯ ρ ⪯ I): each
    node's ``|M|² ≥ 0``, the indefinite ``meps² I2`` interference being bounded by the ``I1`` terms.
    Measured over T_cm = 5 … 0.05 MeV and coherent states rotated up to 2 rad, every kernel's
    B_loss is ⪰ 0 except nuepma's lowest bin (ε = 0.25) below T_cm ≈ 0.1 MeV, where BURST's
    quadrature leaves a negative eigenvalue (4% of that bin's largest at 0.1 MeV, 17% at 0.05 MeV).

    Parameters
    ----------
    phi_for, phi_rev : (..., 3, 3) complex
        Node-summed gain/loss matrices, measure and the other legs' statistical factors included.
    rho1 : (..., 3, 3) complex
        The local density matrix at p₁.
    """
    a_gain = 2.0 * phi_for
    b_loss = phi_for + phi_rev
    net = a_gain - anticomm(b_loss, rho1)
    frs = a_gain - anticomm(phi_for - phi_rev, rho1)
    return net, frs, a_gain, b_loss
