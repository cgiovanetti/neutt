"""The flavour density matrix, the PMNS mixing, and the coherent Hamiltonian H and RHS −i[H, ρ].

State: ρ[cp, bin, a, b], shape (2, N, 3, 3) complex — the batch axes (cp = ν/ν̄, bin) first and
the flavour axes last, so matmul and the commutator broadcast over (cp, bin).  The diagonal is
the classical occupation, ρ[cp, i, m, m] ≡ f[m, cp, i]; the off-diagonal entries are the flavour
coherences, zero at the initial condition.  (BURST stores the same content as dens_mat(a, b, bin).)

Hamiltonian, per (cp, bin) at p1 = ε_i·T_cm, parfact = +1 (ν) / −1 (ν̄):

    H = m2_weak[:, :, cp]/(2 p1 ℏ)  +  mata  +  parfact·p1·mats  +  sia  +  parfact·p1·sis

with m2_weak = U·diag(0, Δm²₂₁, Δm²₃₁)·Uᵀ (real PMNS U = R23·R13·R12, trace/3 removed,
m2_weak(ν̄) = −m2_weak(ν)), the matter potentials mata(1,1) = √2 G_F n_p/ℏ and
mats(1,1) = −(8√2/3)(G_F/M_W²/ℏ) ρ_{e±}, and the ν–ν potentials sia = √2 G_F T³/(2π²)·nudiff/ℏ
and sis = −(8√2/3)(G_F/M_Z²/ℏ) T⁴/(2π²)·nusum.  The solvers pass sia = sis = 0.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ...constants import (DELTA_M2_21, DELTA_M2_31, GFERMI, GFW2HBAR, GFZ2HBAR, HBAR, PI,
                          SIN2_THETA12, SIN2_THETA13, SIN2_THETA23)


# ==== the density-matrix state ==================================================================


def diag_to_rho(f):
    """Embed the classical spectrum ``f`` (3, 2, N) as the complex density matrix (2, N, 3, 3).

    The coherences are zero (decohered flavour states): the QKE initial condition.
    """
    f = jnp.asarray(f)
    nflavor, ncp, nbins = f.shape
    fp = jnp.transpose(f, (1, 2, 0))  # (cp, bin, flavor)
    rho = jnp.zeros((ncp, nbins, nflavor, nflavor), dtype=jnp.complex128)
    diag_idx = jnp.arange(nflavor)
    rho = rho.at[:, :, diag_idx, diag_idx].set(fp.astype(jnp.complex128))
    return rho


def rho_to_diag(rho):
    """The real diagonal of ``rho`` (2, N, 3, 3) as ``f`` (3, 2, N); the coherences are dropped."""
    rho = jnp.asarray(rho)
    ncp, nbins, nflavor, _ = rho.shape
    diag_idx = jnp.arange(nflavor)
    diag = jnp.real(rho[:, :, diag_idx, diag_idx])  # (cp, bin, flavor)
    return jnp.transpose(diag, (2, 0, 1))  # (flavor, cp, bin)


def hermitize(rho):
    """The Hermitian part ½(ρ + ρ†) over the last two (flavour) axes.

    Removes the anti-Hermitian round-off of the RHS and the propagators; bit-identical on a
    Hermitian input.
    """
    rho = jnp.asarray(rho)
    return 0.5 * (rho + jnp.conj(jnp.swapaxes(rho, -1, -2)))


def commutator(A, B):
    """[A, B] = A·B − B·A, batched over leading axes (matmul on the last two)."""
    return jnp.matmul(A, B) - jnp.matmul(B, A)


# ==== PMNS mixing and m2_weak ===================================================================


def _rotation(n: int, i: int, j: int, sin_theta: float) -> np.ndarray:
    """The real rotation R(i, j) (0-based) by the angle whose sine is ``sin_theta``."""
    cos_theta = np.sqrt(1.0 - sin_theta**2)
    R = np.eye(n, dtype=np.float64)
    R[i, i] = cos_theta
    R[j, j] = cos_theta
    R[i, j] = sin_theta
    R[j, i] = -sin_theta
    return R


def pmns_matrix(
    s12: float = np.sqrt(SIN2_THETA12),
    s13: float = np.sqrt(SIN2_THETA13),
    s23: float = np.sqrt(SIN2_THETA23),
    nnu: int = 3,
) -> np.ndarray:
    """The real PMNS matrix U = R23·R13·R12 (3 flavours) from the sines of the mixing angles.

    Defaults are √(PDG sin²θ).  As in BURST, the rotations (1,2), (1,3), (2,3) are applied in that
    order by left-multiplication.
    """
    if nnu != 3:
        raise NotImplementedError("pmns_matrix currently supports nnu=3 only.")
    # BURST's sin_matrix(i, j), 1-based: (1,2) → s12, (1,3) → s13, (2,3) → s23
    sins = {(0, 1): s12, (0, 2): s13, (1, 2): s23}
    U = np.eye(nnu, dtype=np.float64)
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        U = _rotation(nnu, i, j, sins[(i, j)]) @ U  # left-multiply: U <- R . U
    return U


def m2_mass_matrix(
    dm2_21: float = DELTA_M2_21,
    dm2_31: float = DELTA_M2_31,
    normal_hierarchy: bool = True,
    nnu: int = 3,
) -> np.ndarray:
    """The mass-basis mass-squared matrix diag(0, Δm²₂₁, Δm²₃₁) [MeV²].

    The inverted hierarchy (nnu = 3) flips the sign of the (3,3) entry.
    """
    M2 = np.zeros((nnu, nnu), dtype=np.float64)
    M2[1, 1] = dm2_21
    M2[2, 2] = dm2_31
    if (not normal_hierarchy) and nnu == 3:
        M2[2, 2] = -M2[2, 2]
    return M2


def m2_weak(
    s12: float = np.sqrt(SIN2_THETA12),
    s13: float = np.sqrt(SIN2_THETA13),
    s23: float = np.sqrt(SIN2_THETA23),
    dm2_21: float = DELTA_M2_21,
    dm2_31: float = DELTA_M2_31,
    normal_hierarchy: bool = True,
    nnu: int = 3,
) -> np.ndarray:
    """The flavour-basis mass-squared matrix U·M2·Uᵀ − (Tr/nnu)·1, shape (nnu, nnu, 2) [MeV²].

    The last axis is cp: ``[..., 0]`` for ν and ``[..., 1] = −[..., 0]`` for ν̄.  The trace part
    commutes with ρ and drops out of the commutator; it is removed, as BURST does, for a bit-for-bit
    match.
    """
    U = pmns_matrix(s12, s13, s23, nnu)
    M2 = m2_mass_matrix(dm2_21, dm2_31, normal_hierarchy, nnu)
    Mw = U @ M2 @ U.T  # transpose (real U)
    nu_avg = np.trace(Mw) / nnu
    Mw = Mw - nu_avg * np.eye(nnu, dtype=np.float64)
    # symmetrize against round-off
    Mw = 0.5 * (Mw + Mw.T)
    out = np.empty((nnu, nnu, 2), dtype=np.float64)
    out[:, :, 0] = Mw
    out[:, :, 1] = -Mw
    return out


# ==== the coherent Hamiltonian and RHS ===========================================================

_SQRT2 = jnp.sqrt(2.0)
_8SQRT2_3 = 8.0 * _SQRT2 / 3.0


def matter_terms(therm_corr, npro, nflavor: int = 3):
    """The matter potentials (mata, mats), each (nflavor, nflavor) complex.

    Only the (e, e) entry is nonzero (charged current on e±): mata = √2 G_F n_p/ℏ (asymmetric, MSW)
    from the proton (≈ net e⁻) density ``npro`` [MeV³], and mats = −(8√2/3)(G_F/M_W²/ℏ) ρ_{e±}
    (thermal) from the e± energy density ``therm_corr`` [MeV⁴].
    """
    mata = jnp.zeros((nflavor, nflavor), dtype=jnp.complex128)
    mats = jnp.zeros((nflavor, nflavor), dtype=jnp.complex128)
    mata = mata.at[0, 0].set(_SQRT2 * GFERMI * npro / HBAR)
    mats = mats.at[0, 0].set(-GFW2HBAR * _8SQRT2_3 * therm_corr)
    return mata, mats


def self_interaction_terms(tcm, nudiff, nusum):
    """The ν–ν potentials (sia, sis), each (nflavor, nflavor) complex; zero inputs give zero.

    ``tcm`` is T_cm [MeV]; ``nudiff``/``nusum`` are the Hermitian flavour matrices of the ν − ν̄
    number-density differences and ν + ν̄ sums, already time-averaged.
    """
    sia = _SQRT2 * GFERMI * tcm**3 / 2.0 / PI**2 * jnp.asarray(nudiff) / HBAR
    sis = -GFZ2HBAR * _8SQRT2_3 * tcm**4 / 2.0 / PI**2 * jnp.asarray(nusum)
    return sia, sis


def hamiltonian(eps, tcm, m2_weak, mata_mat, mats_mat, sia_mat, sis_mat):
    """The coherent Hamiltonian H[cp, bin, a, b], (2, N, 3, 3) complex [s⁻¹] (module docstring).

    ``eps`` (N,) are the grid abscissas (p1 = ε_i·tcm, ``tcm`` = T_cm [MeV]); ``m2_weak`` is
    :func:`m2_weak` (3, 3, 2); the four (3, 3) potentials come from :func:`matter_terms` and
    :func:`self_interaction_terms`.
    """
    eps = jnp.asarray(eps)
    m2w = jnp.asarray(m2_weak).astype(jnp.complex128)          # (3,3,2)
    p1 = eps[:, None, None] * tcm                               # (N,1,1)
    parfact = jnp.asarray([1.0, -1.0])                          # ν:+1, ν̄:−1

    # vacuum m2_weak/(2 p1 ℏ); p1 = 0 (ε = 0) is guarded against 1/0, and callers mask that bin
    p1_safe = jnp.where(p1 > 0.0, p1, 1.0)
    m2w_cp = jnp.transpose(m2w, (2, 0, 1))                      # (2,3,3)
    H_vac = m2w_cp[:, None, :, :] / (2.0 * p1_safe[None] * HBAR)  # (2,N,3,3)

    # mata, sia: no parfact, no p1; mats, sis: parfact·p1
    H_mata = jnp.broadcast_to(mata_mat, (2, eps.shape[0], 3, 3))
    H_mats = parfact[:, None, None, None] * p1[None] * mats_mat[None, None]
    H_sia = jnp.broadcast_to(sia_mat, (2, eps.shape[0], 3, 3))
    H_sis = parfact[:, None, None, None] * p1[None] * sis_mat[None, None]

    return H_vac + H_mata + H_mats + H_sia + H_sis


def coherent_rhs(rho, eps, tcm, m2_weak, *, therm_corr=0.0, npro=0.0,
                 nudiff=None, nusum=None, hermitian_project=True):
    """dρ/dt|_coh = −i[H, ρ] [s⁻¹] for ρ (2, N, 3, 3), zero in the ε = 0 bins.

    Builds the potentials and H from ``therm_corr``, ``npro`` and ``nudiff``/``nusum`` ((3, 3)
    complex; None = no ν–ν term).  ``hermitian_project`` hermitizes the result.
    """
    rho = jnp.asarray(rho)
    nflavor = rho.shape[-1]
    mata_mat, mats_mat = matter_terms(therm_corr, npro, nflavor)
    if nudiff is None:
        nudiff = jnp.zeros((nflavor, nflavor), dtype=jnp.complex128)
    if nusum is None:
        nusum = jnp.zeros((nflavor, nflavor), dtype=jnp.complex128)
    sia_mat, sis_mat = self_interaction_terms(tcm, nudiff, nusum)
    H = hamiltonian(eps, tcm, m2_weak, mata_mat, mats_mat, sia_mat, sis_mat)
    drho = -1j * commutator(H, rho)
    if hermitian_project:
        drho = hermitize(drho)
    # no coherent evolution at p1 = 0
    eps = jnp.asarray(eps)
    mask = (eps > 0.0)[None, :, None, None]
    drho = jnp.where(mask, drho, 0.0)
    return drho
