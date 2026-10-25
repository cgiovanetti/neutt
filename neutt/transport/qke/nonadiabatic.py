"""The coherent step of the QKE solve: propagators for ρ̇ = −i[H, ρ] − dephasing + source.

Per (cp = ν/ν̄, bin) the 3×3 density matrix rotates at the eigenvalue splittings ω of H, with ω·dt
up to ~1e10 rad per macro step at low ε and late times.  Every integrator here is exact for
coefficients frozen over the step (the stiff phase sits inside an exponential or comes straight
from the eigenvalues), so the error is the frozen-coefficient (Magnus-1) error, set by the slow
collision and cosmological scales.  This replaces qburst's operator-split corotating scheme.

Sections: the exact unitary propagator, a Strang driver and the L-stable matrix-collision step;
the Liouvillian, the frozen-diagonal off-diagonal solve and its quasi-steady state, and the
coherent substep of ``full_coupled`` (dense, sourced, Strang, secular, and ``hybrid``, the
default); the moving eigenframe of the step; diagnostics.

Adiabaticity: ρ_off^ss = −L_oo⁻¹ L_od ρ_diag (:func:`qss_offdiag`) is the coherence that holds the
given diagonal stationary; as Γ/ω → 0 it is the off-diagonal of Σ_i c_i P_i (P_i the eigenprojectors
of H, c = M⁻¹ρ_diag, M_ai = ``|U_ai|²``).  For the default PMNS angles M is nearly singular (smallest
singular value ≈ 1e-3, near μ–τ symmetry), so a diagonal that is not of the form M·p maps to a large
ρ_off^ss: it is a diagnostic, not a closure.  The instantaneous ``|ρ_off|`` aliases at collision-scale
steps; ``|ρ_off^ss|``, the transient ‖ρ_off − ρ_off^ss‖ (it decays monotonically at Γ) and
:func:`off_diag_amps` do not, and are what to compare across step counts.

Vec convention: row-major, vec(ρ)[3a+b] = ρ_ab, so vec(AρB) = kron(A, Bᵀ)·vec(ρ).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .coherent import hermitize


# ==== the exact unitary (corotating-frame / Magnus) propagator and the Strang drivers =============
# Per (cp, bin): ρ(t+Δt) = U ρ U†, U = V e^{−iΛΔt} V† = e^{−iHΔt} for piecewise-constant H — exact
# at any stiffness (BURST's corotating frame + Magnus, realised as one exact propagator).


def _h_inert(n):
    """Stand-in for H = 0 cells in an eigendecomposition: diag(0, 1, …, n−1), nondegenerate, so
    eigh's derivative is finite."""
    return jnp.diag(jnp.arange(n, dtype=jnp.float64)).astype(jnp.complex128)


def eigendecompose(H):
    """(eigvals[..., 3] ascending, V[..., 3, 3], eigenvectors as columns) of the hermitized H."""
    H = jnp.asarray(H)
    H = hermitize(H)
    evals, V = jnp.linalg.eigh(H)
    return evals, V


def magnus_propagator(evals, V, dt):
    """U = V e^{−iΛ dt} V† (..., 3, 3) from H's eigenpairs (Λ [s⁻¹], V); dt [s] scalar or (...,)."""
    evals = jnp.asarray(evals)
    phase = jnp.exp(-1j * evals * jnp.asarray(dt)[..., None])
    Vp = V * phase[..., None, :]
    return jnp.matmul(Vp, jnp.conj(jnp.swapaxes(V, -1, -2)))


def propagate_coherent_H(rho, H, dt):
    """ρ → UρU†, U = e^{−iH dt}, for frozen H: exact at any stiffness; returns the hermitized ρ.

    ρ and H are (..., 3, 3) complex, batched over e.g. (cp, bin); dt is a scalar [s].  Cells with
    H = 0 (masked bins) get U = I, with an inert matrix in the eigendecomposition so its derivative
    stays finite there.
    """
    H = jnp.asarray(H)
    zero = (jnp.max(jnp.abs(H), axis=(-2, -1)) == 0.0)[..., None, None]
    n = H.shape[-1]
    evals, V = eigendecompose(jnp.where(zero, _h_inert(n), H))
    U = jnp.where(zero, jnp.eye(n, dtype=jnp.complex128), magnus_propagator(evals, V, dt))
    Udag = jnp.conj(jnp.swapaxes(U, -1, -2))
    rho_new = jnp.matmul(jnp.matmul(U, jnp.asarray(rho)), Udag)
    return hermitize(rho_new)


def strang_step(rho, H_fn, collision_fn, t, dt):
    """Strang step: ½ explicit-Euler collision, exact coherent step at H(t + dt/2), ½ collision.

    The collision halves are not stiff-stable: once Γ·dt ≳ 1 a half step can drive a tail occupation
    negative, the partner log-interpolation then hits the occupation floor and the rates overflow.
    Use it only for a resolved collision; the production solve (``full_coupled``) advances the stiff
    collision with the L-stable exponential-midpoint step.

    ``rho`` is (2, N, 3, 3); ``H_fn(t)`` and ``collision_fn(rho, t)`` return H and dρ/dt|_coll
    [s⁻¹] of the same shape; t, dt [s].
    """
    rho = jnp.asarray(rho)
    rho = hermitize(rho + 0.5 * dt * collision_fn(rho, t))
    rho = propagate_coherent_H(rho, H_fn(t + 0.5 * dt), dt)
    rho = hermitize(rho + 0.5 * dt * collision_fn(rho, t + dt))
    return rho


def lstable_collision_step(rho, A_rate, B_rate, dt):
    """Exact, L-stable step of the affine matrix collision dρ/dt = A − (Bρ + ρB), frozen (A, B).

    ``rho``, ``A_rate``, ``B_rate`` are (..., 3, 3), the rates in s⁻¹ (e.g.
    ``collide.nuepma_matrix_affine_dfdt``), A and B Hermitian, B ⪰ 0 (true of a physical state up to
    the quadrature residual at nuepma's lowest bin, ``collide_matrix.assemble_from_phi``); dt [s].
    The clamp of B's eigenvalues at 0 removes that residual and round-off.  In the eigenbasis of
    B = V diag(b) V†, with ρ̃ = V†ρV and Ã = V†AV,

        ρ̃_ij(t+dt) = ρ̃_ij(t)·e^{−(b_i+b_j)dt} + Ã_ij·dt·φ((b_i+b_j)dt),   φ(z) = (1 − e^{−z})/z,

    i.e. ρ(t+dt) = ρ_ss + e^{−B dt}(ρ(t) − ρ_ss)e^{−B dt} with Bρ_ss + ρ_ss B = A.  A convex blend
    of ρ and the bounded attractor ρ_ss, so Hermiticity- and positivity-preserving and stable for
    any dt.  On a diagonal ρ it reduces bit-for-bit to the scalar ``hybrid.exp_step`` with rate
    2b_i; degenerate-B blocks have constant decay and φ, so no spurious coherence is sourced.
    """
    rho = jnp.asarray(rho)
    # The barrier stops XLA constant-folding this eigh on the host when B_rate is a compile-time
    # constant; the host-folded eigendecomposition is wrong on GPU.
    B_in = jax.lax.optimization_barrier(hermitize(B_rate))
    b, V = jnp.linalg.eigh(B_in)
    b = jnp.maximum(b, 0.0)                              # B ⪰ 0 (docstring)
    Vd = jnp.conj(jnp.swapaxes(V, -1, -2))
    A_t = jnp.matmul(jnp.matmul(Vd, jnp.asarray(A_rate)), V)   # Ã
    rho_t = jnp.matmul(jnp.matmul(Vd, rho), V)                 # ρ̃
    z = (b[..., :, None] + b[..., None, :]) * dt              # (b_i + b_j)·dt ≥ 0
    decay = jnp.exp(-z)
    # φ(z) = (1 − e^{−z})/z, Taylor near z = 0
    phi = jnp.where(z > 1e-8, -jnp.expm1(-z) / jnp.where(z == 0.0, 1.0, z), 1.0 - 0.5 * z)
    rho_t_new = rho_t * decay + A_t * (dt * phi)
    rho_new = jnp.matmul(jnp.matmul(V, rho_t_new), Vd)
    return hermitize(rho_new)


# ==== the coherent substep: Liouvillian, QSS baseline, and the step integrators ==================

# Off-diagonal / diagonal components of the row-major vec(ρ).
_OFF_IDX = jnp.array([1, 2, 3, 5, 6, 7])   # (0,1)(0,2)(1,0)(1,2)(2,0)(2,1)
_DIAG_IDX = jnp.array([0, 4, 8])           # (0,0)(1,1)(2,2)
# The (a, b) of each off component, and the index of its Hermitian partner within _OFF_IDX.
_OFF_AB = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
_OFF_CONJ_PARTNER = jnp.array([2, 4, 0, 5, 1, 3])


def _kron(a, b):
    """Batched Kronecker product of (...,m,m) and (...,n,n) → (...,mn,mn)."""
    *lead, m, _ = a.shape
    n = b.shape[-1]
    out = a[..., :, None, :, None] * b[..., None, :, None, :]
    return out.reshape(*lead, m * n, m * n)


def liouvillian(H, gamma_diag):
    """The (..., 9, 9) Liouvillian of ρ̇ = −i[H, ρ] − {B, ρ}, B = diag(Γ_a/2), on row-major vec(ρ).

    H (..., 3, 3) [s⁻¹]; ``gamma_diag`` the per-flavour total classical loss Γ_a (..., 3) [s⁻¹].
    {B, ρ}_ab = ½(Γ_a + Γ_b) ρ_ab damps every component, the diagonal included: the generator of the
    frozen-diagonal off-diagonal solve.  The full solve uses :func:`coherent_liouvillian`.
    """
    H = jnp.asarray(H).astype(jnp.complex128)
    *lead, _, _ = H.shape
    I3 = jnp.eye(3, dtype=jnp.complex128)
    I3b = jnp.broadcast_to(I3, (*lead, 3, 3))
    Ht = jnp.swapaxes(H, -1, -2)
    # vec([H,ρ]) = (kron(H,I) − kron(I,Hᵀ)) vec(ρ)
    L_comm = _kron(H, I3b) - _kron(I3b, Ht)
    # vec({B,ρ}) = (kron(B,I) + kron(I,Bᵀ)) vec(ρ)
    B = jnp.zeros((*lead, 3, 3), dtype=jnp.complex128)
    idx = jnp.arange(3)
    B = B.at[..., idx, idx].set(0.5 * jnp.asarray(gamma_diag).astype(jnp.complex128))
    L_damp = _kron(B, I3b) + _kron(I3b, jnp.swapaxes(B, -1, -2))
    return -1j * L_comm - L_damp


def offdiag_blocks(H, gamma_diag):
    """The (off, off) and (off, diag) blocks of :func:`liouvillian`.

    Returns ``(L_oo, L_od)``, (..., 6, 6) and (..., 6, 3) complex: with the diagonal frozen,
    ρ̇_off = L_oo ρ_off + s with the coherent source s = L_od ρ_diag (``_OFF_IDX`` order).
    """
    L = liouvillian(H, gamma_diag)
    L_oo = L[..., _OFF_IDX[:, None], _OFF_IDX[None, :]]
    L_od = L[..., _OFF_IDX[:, None], _DIAG_IDX[None, :]]
    return L_oo, L_od


def qss_offdiag(L_oo, L_od, rho_diag):
    """The adiabatic (quasi-steady) coherence ρ_off^ss = −L_oo⁻¹ L_od ρ_diag, (..., 6) complex.

    ``rho_diag`` is the real per-flavour occupation (..., 3).  An all-zero L_oo block (a fully
    masked bin, e.g. ε = 0) gets the identity, so the solve returns 0 there (the QSS of a zero
    source) and leaves the other bins bit-exact.
    """
    L_oo = jnp.asarray(L_oo)
    s = jnp.einsum('...ij,...j->...i', L_od, jnp.asarray(rho_diag).astype(jnp.complex128))
    scale = jnp.max(jnp.abs(L_oo), axis=(-2, -1), keepdims=True)
    L_reg = L_oo + (scale == 0.0) * jnp.eye(L_oo.shape[-1], dtype=L_oo.dtype)
    # batched solve: b needs a trailing axis (JAX ≥ 0.5)
    return -jnp.linalg.solve(L_reg, s[..., None]).squeeze(-1)


def evolve_offdiag(L_oo, s, rho_off, dt, max_squarings=40):
    """Exact advance of ρ̇_off = L_oo ρ_off + s over dt for frozen (L_oo, s).

    expm of the 7×7 augmented matrix [[L_oo, s], [0, 0]]·dt applied to [ρ_off; 1] gives
    e^{L_oo dt} ρ_off + dt φ₁(L_oo dt) s, φ₁(z) = (e^z − 1)/z: no inverse, and the stiff −iω phase
    is inside the exponential.

    ``max_squarings=40`` (JAX's default is 16): at late times and low ε, ω·dt reaches 1e9–1e10 rad
    per step and needs ~30+ squarings; with 16, expm silently clamps, overflows and returns NaN.
    The phase round-off at that ω·dt (~1e-7) is harmless for the phase-invariant envelopes.

    L_oo (..., 6, 6), s = L_od ρ_diag and rho_off (..., 6), complex; dt [s] scalar or (...,).
    """
    L_oo = jnp.asarray(L_oo)
    *lead, n, _ = L_oo.shape
    M = jnp.zeros((*lead, n + 1, n + 1), dtype=jnp.complex128)
    M = M.at[..., :n, :n].set(L_oo)
    M = M.at[..., :n, n].set(jnp.asarray(s))
    dt = jnp.asarray(dt)
    Mdt = (M * dt[..., None, None]) if dt.ndim else M * dt
    expM = jax.scipy.linalg.expm(Mdt, max_squarings=max_squarings)
    y0 = jnp.concatenate([jnp.asarray(rho_off),
                          jnp.ones((*lead, 1), dtype=jnp.complex128)], axis=-1)
    y = jnp.einsum('...ij,...j->...i', expM, y0)
    return y[..., :n]


def coherent_liouvillian(H, gamma_diag):
    """The (..., 9, 9) generator L_coh = −i[H, ·] − Λ_off of the full solve's coherent substep.

    H (..., 3, 3) [s⁻¹]; ``gamma_diag`` the per-flavour total (ungated) classical relaxation rate
    Γ_a (..., 3) [s⁻¹]; L_coh acts on row-major vec(ρ).  Λ_off dephases the off-diagonal components
    at Γ_ab = ½(Γ_a + Γ_b) and leaves the diagonal undamped: its collisional loss lives in the
    classical-collision substep, and damping it here would double-count.  L_coh is a Lindblad
    pure-dephasing channel ⊕ the unitary −i[H, ·], CPTP for Γ_a ≥ 0 (the Hadamard factor
    e^{−Γ_ab t} = u uᵀ + diag(1 − u_a²), u_a = e^{−Γ_a t/2}, is PSD), so ``expm(L_coh dt)``
    conserves Tr ρ per bin (number) and keeps ρ ⪰ 0 (positivity).
    """
    H = jnp.asarray(H).astype(jnp.complex128)
    *lead, _, _ = H.shape
    I3 = jnp.eye(3, dtype=jnp.complex128)
    I3b = jnp.broadcast_to(I3, (*lead, 3, 3))
    Ht = jnp.swapaxes(H, -1, -2)
    L_comm = _kron(H, I3b) - _kron(I3b, Ht)               # vec([H,ρ]) = L_comm vec(ρ)
    # Λ_off: a diagonal superoperator, Γ_ab on the off components and 0 on the diagonal ones.
    g = jnp.asarray(gamma_diag).astype(jnp.complex128)
    gab = 0.5 * (g[..., :, None] + g[..., None, :])       # Γ_ab
    lam = gab.reshape(*lead, 9)
    lam = lam.at[..., _DIAG_IDX].set(0.0)
    L = -1j * L_comm
    d = jnp.arange(9)
    L = L.at[..., d, d].add(-lam)
    return L


def coherent_step(H, gamma_diag, rho, dt, max_squarings=40):
    """ρ(t + dt) = expm(L_coh dt) ρ(t) for frozen (H, Γ): the dense coherent substep, hermitized.

    H and the Hermitian ρ are (..., 3, 3), H in s⁻¹; ``gamma_diag`` Γ_a (..., 3) [s⁻¹]; dt [s]
    scalar or (...,).  Populations and coherences evolve together, so the coherences' back-reaction
    on the populations, (−i[H, ρ])_aa = −2 Σ_b H_ab Im ρ_ab, is included.  Positivity- and
    number-conserving by construction (:func:`coherent_liouvillian`); ``max_squarings=40`` as in
    :func:`evolve_offdiag`.
    """
    rho = jnp.asarray(rho).astype(jnp.complex128)
    *lead, _, _ = rho.shape
    L = coherent_liouvillian(H, gamma_diag)
    dt = jnp.asarray(dt)
    Ldt = (L * dt[..., None, None]) if dt.ndim else L * dt
    expL = jax.scipy.linalg.expm(Ldt, max_squarings=max_squarings)
    v = rho.reshape(*lead, 9)
    v = jnp.einsum('...ij,...j->...i', expL, v)
    return hermitize(v.reshape(*lead, 3, 3))


def coherent_step_sourced(H, gamma_diag, rho, dt, src, max_squarings=40):
    """The dense coherent substep with the collision as a constant rate: ρ̇ = L_coh ρ + src.

    For frozen (H, Γ, src), expm of the 10×10 augmented matrix [[L_coh, src], [0, 0]]·dt gives::

        ρ(t + dt) = expm(L_coh dt) ρ(t) + dt φ₁(L_coh dt) src,   φ₁(z) = (e^z − 1)/z

    ``src`` (..., 3, 3) is the collision's dρ/dt [s⁻¹]; the other arguments are as in
    :func:`coherent_step`.  A collision applied as a jump before :func:`coherent_step` moves the
    coherence's quasi-steady target discontinuously each step, leaving a lag that freezes in once
    Γ → 0 and converges only at first order.  As a rate the diagonal ramps during the step, the
    fixed point for constant (L_coh, src) is exactly −L_coh⁻¹ src, and the lag is O(dt²).
    H = Γ = 0 gives ρ + dt·src (the classical collision); src = 0 is bit-identical to
    :func:`coherent_step`.
    """
    rho = jnp.asarray(rho).astype(jnp.complex128)
    *lead, _, _ = rho.shape
    L = coherent_liouvillian(H, gamma_diag)
    s = jnp.asarray(src).astype(jnp.complex128).reshape(*lead, 9)
    M = jnp.zeros((*lead, 10, 10), dtype=jnp.complex128)
    M = M.at[..., :9, :9].set(L)
    M = M.at[..., :9, 9].set(s)
    dt = jnp.asarray(dt)
    Mdt = (M * dt[..., None, None]) if dt.ndim else M * dt
    expM = jax.scipy.linalg.expm(Mdt, max_squarings=max_squarings)
    y = jnp.concatenate([rho.reshape(*lead, 9),
                         jnp.ones((*lead, 1), dtype=jnp.complex128)], axis=-1)
    y = jnp.einsum('...ij,...j->...i', expM, y)
    return hermitize(y[..., :9].reshape(*lead, 3, 3))


def coherent_step_strang(H, gamma_diag, rho, dt):
    """Strang split of :func:`coherent_step`, D(dt/2) ∘ U(dt) ∘ D(dt/2); same contract.

    U(dt) is the unitary part, ρ → UρU† with U = e^{−iH dt} from a 3×3 eigendecomposition
    (:func:`propagate_coherent_H`): ω enters as a pure phase, so there is no 9×9 expm and no
    squaring.  D(t) is the dephasing Λ_off as a Hadamard product, ρ_ab → ρ_ab e^{−Γ_ab t} (a ≠ b),
    diagonal untouched.

    Each factor is CPTP, so number and positivity are conserved exactly.  Γ = 0 is bit-identical to
    :func:`propagate_coherent_H`, H = 0 to :func:`coherent_step`.  The deviation from the dense step
    is the O(dt²) BCH commutator [−i[H, ·], Λ_off], whose size depends on H, so vacuum checks do not
    cover the matter (MSW) case.
    """
    rho = jnp.asarray(rho).astype(jnp.complex128)
    *lead, _, _ = rho.shape
    g = jnp.asarray(gamma_diag).astype(jnp.float64)
    gab = 0.5 * (g[..., :, None] + g[..., None, :])           # Γ_ab
    dt = jnp.asarray(dt)
    dt_bc = dt[..., None, None] if dt.ndim else dt
    idx = jnp.arange(3)
    half = jnp.exp(-gab * (0.5 * dt_bc))
    half = half.at[..., idx, idx].set(1.0)                    # unit diagonal → diagonal undamped
    rho = rho * half                                          # ½ dephase  D(dt/2)
    rho = propagate_coherent_H(rho, H, dt)                    # exact unitary  U(dt) ρ U†
    rho = jnp.asarray(rho) * half                             # ½ dephase  D(dt/2)
    return hermitize(rho)


def coherent_step_sourced_strang(H, gamma_diag, rho, dt, src):
    """:func:`coherent_step_sourced` with an exact rotation: D(dt/2) ∘ S(dt) ∘ D(dt/2).

    The dense augmented ``expm`` is not unitary at large ω·dt: it moves occupations by
    ≈ 1e-17·ω·dt per step, and ω·dt reaches 1e8–1e10 at low ε and late times.  Here D(t) is the
    flavour-basis dephasing of :func:`coherent_step_strang` and S(dt) the rotation plus source,
    solved exactly in the eigenbasis of H (H = V Λ V†, ρ' = V†ρV, s' = V†src V,
    θ_ij = (λ_i − λ_j) dt)::

        ρ'_ij → e^{−iθ_ij} ρ'_ij + dt φ₁(−iθ_ij) s'_ij,   φ₁(z) = (e^z − 1)/z

    i.e. ρ → UρU† + ∫₀^dt U(τ) src U(τ)† dτ with U(τ) = e^{−iHτ}.  With z = −iθ,
    e^z − 1 = −2 sin²(θ/2) − i sin θ has no cancellation, so φ₁ is accurate from θ = 0 to θ ~ 1e10,
    and the phase has unit modulus to round-off: no squaring anywhere.

    Error: the split (BCH) error between the dephasing and (H, src), deterministic and convergent,
    but only first order while ω·dt ≫ 1 (:func:`coherent_step_sourced_secular` treats that regime).
    src = 0 gives the D·U·D of :func:`coherent_step_strang`; Γ = 0 is exact (no split).
    Tr ρ(dt) = Tr ρ + dt Tr src.
    """
    rho = jnp.asarray(rho).astype(jnp.complex128)
    src = jnp.asarray(src).astype(jnp.complex128)
    g = jnp.asarray(gamma_diag).astype(jnp.float64)
    gab = 0.5 * (g[..., :, None] + g[..., None, :])           # Γ_ab
    dt = jnp.asarray(dt)
    dt_bc = dt[..., None, None] if dt.ndim else dt
    idx = jnp.arange(3)
    half = jnp.exp(-gab * (0.5 * dt_bc))
    half = half.at[..., idx, idx].set(1.0)                    # diagonal undamped (no double count)

    rho = rho * half                                          # ½ dephase  D(dt/2)
    # H = V diag(λ) V†; an H = 0 cell (masked bin) gets an inert matrix and λ = 0, so the step there
    # is ρ + dt·src with a finite derivative
    H = jnp.asarray(H)
    zero = jnp.max(jnp.abs(H), axis=(-2, -1)) == 0.0
    evals, V = eigendecompose(jnp.where(zero[..., None, None], _h_inert(H.shape[-1]), H))
    evals = jnp.where(zero[..., None], 0.0, evals)
    Vd = jnp.conj(jnp.swapaxes(V, -1, -2))
    rp = jnp.matmul(jnp.matmul(Vd, rho), V)                   # ρ' in the H eigenbasis
    sp = jnp.matmul(jnp.matmul(Vd, src), V)
    theta = (evals[..., :, None] - evals[..., None, :]) * dt_bc   # θ_ij
    rot = jnp.cos(theta) - 1j * jnp.sin(theta)                # e^{−iθ}, |·| = 1
    num = -2.0 * jnp.sin(0.5 * theta) ** 2 - 1j * jnp.sin(theta)  # e^{−iθ} − 1, no cancellation
    nz = theta != 0.0
    phi1 = jnp.where(nz, num / jnp.where(nz, -1j * theta, 1.0), 1.0 + 0.0j)
    rp = rot * rp + dt_bc * phi1 * sp                         # exact unitary + source  S(dt)
    rho = jnp.matmul(jnp.matmul(V, rp), Vd)
    rho = rho * half                                          # ½ dephase  D(dt/2)
    return hermitize(rho)


# ω_max·dt above which coherent_step_sourced_hybrid takes the secular step (when also Γ_max·dt ≤ 1).
# Below it the dense round-off (≈ 1e-17·ω·dt per step) is ≲ 1e-14; above it the secular error,
# O(Γdt·(Γ/ω)²) on the populations and O(Γ/ω·Γdt) on the coherences, is negligible.
HYBRID_WDT_SWITCH = 1.0e3


def _phi1(w):
    """φ₁(w) = (e^w − 1)/w for complex w, accurate from w = 0 to |Im w| ~ 1e10.

    e^w − 1 = e^x(−2 sin²(y/2) + i sin y) + expm1(x) for w = x + iy: no cancellation at small |w|,
    and the phase comes from sin/cos, so |e^{iy}| = 1 to round-off at any y.
    """
    x, y = jnp.real(w), jnp.imag(w)
    num = jnp.exp(x) * (-2.0 * jnp.sin(0.5 * y) ** 2 + 1j * jnp.sin(y)) + jnp.expm1(x)
    nz = w != 0.0
    return jnp.where(nz, num / jnp.where(nz, w, 1.0), 1.0 + 0.0j)


def coherent_step_sourced_secular(H, gamma_diag, rho, dt, src):
    """:func:`coherent_step_sourced` for ω·dt ≫ 1: the secular step plus the first non-secular term.

    Where ω·dt ≫ 1 the rotation averages the flavour-basis dephasing, so the step is expanded in
    Γ/ω rather than in dt.  In the eigenbasis of H (ρ' = V†ρV, s' = V†src V, row-major m = (ij),
    ω_m = λ_i − λ_j) it solves ρ̇' = −iω∘ρ' − D'ρ' + s', D' the dephasing Λ_off in that basis.
    Split the generator into diag(z), z_m = −iω_m − D'_mm on coherences and 0 on populations, and
    P = −D' with its coherence diagonal removed; the first Magnus term in the interaction picture
    of diag(z) over [0, dt] gives::

        Ω1_mn = P_mn · dt φ₁((z_n − z_m) dt),      σ_m = s'_m · dt φ₁(−z_m dt),
        ρ'(dt) = e^{z dt} ∘ top[ expm([[Ω1, σ], [0, 0]]) · [ρ'(0); 1] ].

    The population block (z_n − z_m = 0) is exponentiated exactly (the rotating-wave conversion
    rate); each coherence keeps its exact phase e^{−iω dt} and secular decay; the non-secular
    couplings enter averaged over the step through φ₁(±iω dt) ~ 1/(ω dt), which keeps the O(Γ/ω)
    coherence slaved to the populations.  ``expm`` sees only ‖Ω1‖ ≲ Γ dt, never ω·dt.

    Error (the next Magnus term): O(Γ dt·(Γ/ω)²) per step on populations for real H (the O(Γ/ω) rate
    correction ∝ Im(D'_pc D'_cp) vanishes then, e.g. δ_CP = 0 without self-interaction) and
    O(Γ/ω·Γ dt) on coherences.  Exact for Γ = 0 and for H diagonal in the flavour basis (D' is then
    diagonal).  Tr ρ(dt) = Tr ρ + dt Tr src to round-off (Σ_p D'_pn = 0).

    Valid for Γ dt ≲ 1: beyond it the expansion fails and the interaction-picture factors
    (∝ e^{Γ dt}) overflow the ``expm``.  :func:`coherent_step_sourced_hybrid` uses this step only
    where ω_max·dt > ``HYBRID_WDT_SWITCH`` and Γ_max·dt ≤ 1.
    """
    rho = jnp.asarray(rho).astype(jnp.complex128)
    src = jnp.asarray(src).astype(jnp.complex128)
    *lead, _, _ = rho.shape
    g = jnp.asarray(gamma_diag).astype(jnp.float64)
    lam = 0.5 * (g[..., :, None] + g[..., None, :]) * (1.0 - jnp.eye(3))  # Γ_ab, 0 on the diagonal
    dt = jnp.asarray(dt, dtype=jnp.float64)
    dt9 = dt[..., None] if dt.ndim else dt                                # broadcasts on (...,9)
    dt99 = dt[..., None, None] if dt.ndim else dt                         # broadcasts on (...,9,9)

    evals, V = eigendecompose(H)
    Vc = jnp.conj(V)
    Vd = jnp.swapaxes(Vc, -1, -2)
    # D'_{ij,kl} = Σ_ab conj(V_ai) V_bj Γ_ab V_ak conj(V_bl)
    X = Vc[..., :, :, None] * V[..., :, None, :]                          # (...,a,i,k)
    Y = V[..., :, :, None] * Vc[..., :, None, :]                          # (...,b,j,l)
    Dp = jnp.einsum('...ab,...aik,...bjl->...ijkl', lam.astype(jnp.complex128), X, Y)
    Dp = Dp.reshape(*lead, 9, 9)
    coh = ~jnp.eye(3, dtype=bool).reshape(9)                              # coherence components
    omega = (evals[..., :, None] - evals[..., None, :]).reshape(*lead, 9)  # ω_m = λ_i − λ_j
    d = jnp.where(coh, jnp.real(jnp.diagonal(Dp, axis1=-2, axis2=-1)), 0.0)  # secular coherence decay
    z = -1j * omega - d
    P = -Dp + d[..., :, None] * jnp.eye(9)                                # coherence diagonal removed
    Om1 = P * dt99 * _phi1((z[..., None, :] - z[..., :, None]) * dt99)
    sp = jnp.matmul(jnp.matmul(Vd, src), V).reshape(*lead, 9)
    sig = sp * dt9 * _phi1(-z * dt9)

    M = jnp.zeros((*lead, 10, 10), dtype=jnp.complex128)
    M = M.at[..., :9, :9].set(Om1)
    M = M.at[..., :9, 9].set(sig)
    expM = jax.scipy.linalg.expm(M)                  # ‖M‖ ≲ Γ dt + ‖σ‖: default squarings suffice
    rp = jnp.matmul(jnp.matmul(Vd, rho), V).reshape(*lead, 9)
    y = jnp.einsum('...ij,...j->...i', expM,
                   jnp.concatenate([rp, jnp.ones((*lead, 1), dtype=jnp.complex128)], axis=-1))
    th = omega * dt9
    rot = jnp.exp(-d * dt9) * (jnp.cos(th) - 1j * jnp.sin(th))           # e^{z dt}, exact phase
    rp = (rot * y[..., :9]).reshape(*lead, 3, 3)
    return hermitize(jnp.matmul(jnp.matmul(V, rp), Vd))


def coherent_step_sourced_hybrid(H, gamma_diag, rho, dt, src, wdt_switch=None):
    """:func:`coherent_step_sourced`, switched per cell to the secular step where ω·dt is large.

    Takes :func:`coherent_step_sourced_secular` where ω_max·dt > ``wdt_switch`` (default
    ``HYBRID_WDT_SWITCH``; ω_max = λ_max − λ_min of H) and Γ_max·dt ≤ 1, the dense
    :func:`coherent_step_sourced` elsewhere.  Both are accurate across the switch (dense round-off
    ≲ 1e-17·ω·dt, secular error O(Γ/ω)), so the switch is a numerical choice, not a physics
    threshold.  The Γ_max·dt ≤ 1 guard keeps the secular step in its domain; it binds only where
    collisions dominate a step, where the dense step is accurate.

    Both branches run on every cell; the unselected one gets inert inputs (dt = 0 for the dense one;
    H = diag(0, 1, 2), Γ = 0, src = 0 for the secular one), so it stays cheap, finite and
    gradient-safe (no eigh derivative at a degenerate H, e.g. the ε = 0 node where H = 0).
    """
    K = HYBRID_WDT_SWITCH if wdt_switch is None else wdt_switch
    H = jnp.asarray(H).astype(jnp.complex128)
    rho = jnp.asarray(rho).astype(jnp.complex128)
    src = jnp.asarray(src).astype(jnp.complex128)
    g = jnp.asarray(gamma_diag).astype(jnp.float64)
    lead = rho.shape[:-2]
    dt = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), lead)
    ev = jnp.linalg.eigvalsh(hermitize(jax.lax.stop_gradient(H)))
    dt_ng = jax.lax.stop_gradient(dt)
    sec = (((ev[..., -1] - ev[..., 0]) * dt_ng > K)
           & (jnp.max(jax.lax.stop_gradient(g), axis=-1) * dt_ng <= 1.0))   # (...) bool
    s2, s3 = sec[..., None, None], sec[..., None]
    H_inert = jnp.diag(jnp.array([0.0, 1.0, 2.0], dtype=jnp.complex128))
    out_sec = coherent_step_sourced_secular(jnp.where(s2, H, H_inert), jnp.where(s3, g, 0.0),
                                            rho, dt, jnp.where(s2, src, 0.0))
    out_dense = coherent_step_sourced(H, gamma_diag, rho, jnp.where(sec, 0.0, dt), src)
    return jnp.where(s2, out_sec, out_dense)


# ---- ν–ν pinned (2026-10-22): the substepped mean field of full_coupled is the only user of the explicit
# (E, Φ) propagators below; they are commented out with it (see full_coupled's module header).
# def _secular_propagators(H, g, dt):
#     """(E', Phi') of :func:`coherent_step_sourced_secular` as flavour-basis 9×9 matrices (see
#     :func:`sourced_propagators_hybrid`).  Same algebra as the step function, with the state and the source
#     kept symbolic: in the eigenbasis  ρ'(dt) = e^{z dt} ∘ [e^{Ω1} ρ' + φ₁(Ω1) (σ_fac ∘ s')],  and
#     Q[(ij),(ab)] = conj(V_ai) V_bj maps vec(ρ) to vec(V†ρV)."""
#     *lead, _, _ = H.shape
#     lam = 0.5 * (g[..., :, None] + g[..., None, :]) * (1.0 - jnp.eye(3))
#     dt9 = dt[..., None]
#     dt99 = dt[..., None, None]
#     evals, V = eigendecompose(H)
#     Vc = jnp.conj(V)
#     X = Vc[..., :, :, None] * V[..., :, None, :]
#     Y = V[..., :, :, None] * Vc[..., :, None, :]
#     Dp = jnp.einsum('...ab,...aik,...bjl->...ijkl', lam.astype(jnp.complex128), X, Y).reshape(*lead, 9, 9)
#     coh = ~jnp.eye(3, dtype=bool).reshape(9)
#     omega = (evals[..., :, None] - evals[..., None, :]).reshape(*lead, 9)
#     d = jnp.where(coh, jnp.real(jnp.diagonal(Dp, axis1=-2, axis2=-1)), 0.0)
#     z = -1j * omega - d
#     P = -Dp + d[..., :, None] * jnp.eye(9)
#     Om1 = P * dt99 * _phi1((z[..., None, :] - z[..., :, None]) * dt99)
#     I9 = jnp.eye(9, dtype=jnp.complex128)
#     M = jnp.zeros((*lead, 18, 18), dtype=jnp.complex128)
#     M = M.at[..., :9, :9].set(Om1)
#     M = M.at[..., :9, 9:].set(jnp.broadcast_to(I9, (*lead, 9, 9)))
#     Pm = jax.scipy.linalg.expm(M)                     # [[e^{Ω1}, φ₁(Ω1)], [0, I]]
#     eOm, phiOm = Pm[..., :9, :9], Pm[..., :9, 9:]
#     th = omega * dt9
#     rot = jnp.exp(-d * dt9) * (jnp.cos(th) - 1j * jnp.sin(th))          # e^{z dt}, exact phase
#     sig_fac = dt9 * _phi1(-z * dt9)
#     Q = jnp.einsum('...ai,...bj->...ijab', Vc, V).reshape(*lead, 9, 9)  # vec(V†ρV) = Q vec(ρ); unitary
#     Qd = jnp.conj(jnp.swapaxes(Q, -1, -2))
#     E = Qd @ (rot[..., :, None] * eOm) @ Q
#     Phi = Qd @ (rot[..., :, None] * phiOm * sig_fac[..., None, :]) @ Q
#     return E, Phi
#
#
# def sourced_propagators_hybrid(H, gamma_diag, dt, wdt_switch=None, dense_only=False):
#     """The affine map of :func:`coherent_step_sourced_hybrid` as explicit per-cell matrices.
#
#     Returns ``(E, Phi)``, each (..., 9, 9) complex, such that per cell
#     ``vec(ρ(dt)) = E · vec(ρ) + Phi · vec(src)`` equals ``coherent_step_sourced_hybrid(H, Γ, ρ, dt, src)``
#     (before its final hermitize) for every ρ and src: E is the frozen-(H, Γ) propagator and Phi the
#     source response, ``dt φ₁(L dt)`` in the dense regime (augmented 18×18 ``expm``) and the secular
#     equivalent above the ω·dt switch (:func:`_secular_propagators`).  Built once, they advance any number
#     of states and sources through the same frozen step at a 9×9 matvec each — the resolved ν–ν mean-field
#     substepping in ``full_coupled`` uses them with dt = h = dt_macro / n_sub.
#
#     ``dense_only=True`` (``coherent_method="expm"``) skips the secular branch.
#     """
#     K = HYBRID_WDT_SWITCH if wdt_switch is None else wdt_switch
#     H = jnp.asarray(H).astype(jnp.complex128)
#     g = jnp.asarray(gamma_diag).astype(jnp.float64)
#     lead = H.shape[:-2]
#     dt = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), lead)
#     I9 = jnp.eye(9, dtype=jnp.complex128)
#
#     def _dense(dt_):
#         L = coherent_liouvillian(H, g)
#         M = jnp.zeros((*lead, 18, 18), dtype=jnp.complex128)
#         M = M.at[..., :9, :9].set(L * dt_[..., None, None])
#         M = M.at[..., :9, 9:].set(I9 * dt_[..., None, None])
#         P = jax.scipy.linalg.expm(M, max_squarings=40)   # [[e^{L dt}, dt φ₁(L dt)], [0, I]]
#         return P[..., :9, :9], P[..., :9, 9:]
#
#     if dense_only:
#         return _dense(dt)
#     ev = jnp.linalg.eigvalsh(hermitize(jax.lax.stop_gradient(H)))
#     dt_ng = jax.lax.stop_gradient(dt)
#     sec = (((ev[..., -1] - ev[..., 0]) * dt_ng > K)
#            & (jnp.max(jax.lax.stop_gradient(g), axis=-1) * dt_ng <= 1.0))
#     s2, s3 = sec[..., None, None], sec[..., None]
#     E_d, P_d = _dense(jnp.where(sec, 0.0, dt))
#     H_inert = jnp.diag(jnp.array([0.0, 1.0, 2.0], dtype=jnp.complex128))
#     E_s, P_s = _secular_propagators(jnp.where(s2, H, H_inert), jnp.where(s3, g, 0.0), dt)
#     return jnp.where(s2, E_s, E_d), jnp.where(s2, P_s, P_d)


# Moving eigenframe of the macro step.  Where the thermal potential rotates the eigenbasis of H
# (most at high ε), a frozen H_mid changes direction at every step boundary and leaves the coherence
# behind at an effectively random phase (ω·dt ≫ 1); the kicks random-walk into bin-to-bin structure
# that falls only as 1/n_steps.  The step instead runs in the frame R(τ) = e^{−iK(τ − dt/2)}, with
# e^{−iK dt} = V_end V_start† carrying the eigenvectors of H(t_start) onto those of H(t_end).  There
# the generator is H_mid − K (−K the geometric term) and the eigenbasis is nearly fixed, so the
# coherence is transported adiabatically:
#     ρ(dt) = R̃ · step(H_mid − K, R̃ ρ R̃†) · R̃†,   R̃ = e^{−iK dt/2}.
# The frame equals the flavour basis at the midpoint, where Γ and the collision source are
# evaluated; K = 0 when the eigenvectors do not move (vacuum), which recovers the plain step.


def eigenframe_generator(H_start, H_end, dt, max_angle=0.5):
    """Hermitian K (..., 3, 3) in the flavour basis with e^{−iK dt} = V_end V_start†.

    V_start, V_end are the eigenvectors of ``H_start``, ``H_end`` in eigenvalue order, the columns
    of V_end phase-aligned to V_start.  K = 0 where either H is 0 (no eigenbasis) and, as a guard,
    where the frame turns by more than ``max_angle`` rad in one step (an eigenvalue crossing
    reorders the eigenvectors); the step there falls back to the frozen-H one.  Gradient-safe: eigh
    never sees the degenerate H = 0.
    """
    H_start = jnp.asarray(H_start).astype(jnp.complex128)
    H_end = jnp.asarray(H_end).astype(jnp.complex128)
    nz = (jnp.max(jnp.abs(H_start), axis=(-2, -1)) > 0.0) & (jnp.max(jnp.abs(H_end), axis=(-2, -1)) > 0.0)
    H_inert = jnp.diag(jnp.array([0.0, 1.0, 2.0], dtype=jnp.complex128))
    _, Va = jnp.linalg.eigh(jnp.where(nz[..., None, None], H_start, H_inert))
    _, Vb = jnp.linalg.eigh(jnp.where(nz[..., None, None], H_end, H_inert))
    c = jnp.sum(jnp.conj(Va) * Vb, axis=-2)                          # ⟨Va_k|Vb_k⟩
    Vb = Vb * (jnp.conj(c) / jnp.maximum(jnp.abs(c), 1e-300))[..., None, :]
    W = Vb @ jnp.conj(jnp.swapaxes(Va, -1, -2))                     # e^{−iK dt}, near identity
    X = 0.5j * (W - jnp.conj(jnp.swapaxes(W, -1, -2)))               # sin(K dt), Hermitian
    X3 = X @ X @ X
    Kdt = X + X3 / 6.0 + 0.075 * (X3 @ X @ X)                         # arcsin series (small per-step angle)
    ok = nz & (jnp.sum(jnp.abs(jax.lax.stop_gradient(X)) ** 2, axis=(-2, -1)) <= jnp.sin(max_angle) ** 2)
    dt = jnp.asarray(dt, dtype=jnp.float64)
    return jnp.where(ok[..., None, None], Kdt / (dt[..., None, None] if dt.ndim else dt), 0.0)


def unitary_exp(K, t):
    """e^{−iKt} (..., 3, 3) for Hermitian K (small ‖K t‖; dense expm, gradient-safe at K = 0)."""
    return jax.scipy.linalg.expm(-1j * jnp.asarray(K).astype(jnp.complex128) * t)


def rho_from_offdiag(rho_off, rho_diag):
    """The Hermitian ρ (..., 3, 3) from the off-diagonal vector (..., 6) and the diagonal (..., 3).

    Hermitized, so a small anti-Hermitian part of an evolved off-vector is projected out.
    """
    rho_off = jnp.asarray(rho_off)
    *lead, _ = rho_off.shape
    rho = jnp.zeros((*lead, 3, 3), dtype=jnp.complex128)
    rho = rho.at[..., _DIAG_IDX // 3, _DIAG_IDX % 3].set(
        jnp.asarray(rho_diag).astype(jnp.complex128))
    for c, (a, b) in enumerate(_OFF_AB):
        rho = rho.at[..., a, b].set(rho_off[..., c])
    return hermitize(rho)


def off_diag_amps(rho, V=None, H=None):
    """The mass-basis oscillation-amplitude envelope of ρ (qburst's get_off_diag_amps).

    For each eigen-pair k < kk, ``left = V† P_k V`` and ``right = V† P_kk V`` give the two
    counter-rotating components a1 = left·ρ·right and a2 = right·ρ·left.  For each flavour pair
    m < mm,

        real_amp = √((Re a1 + Re a2)² + (−Im a1 + Im a2)²),
        imag_amp = √((Im a1 + Im a2)² + (Re a1 − Re a2)²),

    and amps[m, mm] = amps[mm, m] = Σ_pairs (real_amp + i·imag_amp), (..., 3, 3) complex.  For the
    real (δ_CP = 0) PMNS this depends only on the eigenbasis coherence magnitudes: phase-invariant,
    so aliasing-immune.  ``V`` is qburst's cob_matrix, the flavour→mass change of basis
    (ρ_mass = V ρ V†, rows the conjugated eigenvectors), so ``left`` is the flavour-basis
    eigenprojector.  Without it, V = V_H† from :func:`eigendecompose` of ``H`` (eigenvectors as
    columns).
    """
    rho = jnp.asarray(rho)
    if V is None:
        if H is None:
            raise ValueError("off_diag_amps needs either V (eigenvectors) or H.")
        _, V_H = eigendecompose(H)
        V = jnp.conj(jnp.swapaxes(V_H, -1, -2))
    Vd = jnp.conj(jnp.swapaxes(V, -1, -2))
    *lead, _, _ = rho.shape
    amps = jnp.zeros((*lead, 3, 3), dtype=jnp.complex128)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for (k, kk) in pairs:
        Pk = jnp.zeros((3, 3), dtype=jnp.complex128).at[k, k].set(1.0)
        Pkk = jnp.zeros((3, 3), dtype=jnp.complex128).at[kk, kk].set(1.0)
        left = jnp.matmul(jnp.matmul(Vd, Pk), V)
        right = jnp.matmul(jnp.matmul(Vd, Pkk), V)
        a1 = jnp.matmul(jnp.matmul(left, rho), right)
        a2 = jnp.matmul(jnp.matmul(right, rho), left)
        for (m, mm) in pairs:
            re1, im1 = jnp.real(a1[..., m, mm]), jnp.imag(a1[..., m, mm])
            re2, im2 = jnp.real(a2[..., m, mm]), jnp.imag(a2[..., m, mm])
            real_amp = jnp.sqrt((re1 + re2) ** 2 + (-im1 + im2) ** 2)
            imag_amp = jnp.sqrt((im1 + im2) ** 2 + (re1 - re2) ** 2)
            val = real_amp + 1j * imag_amp
            amps = amps.at[..., m, mm].add(val)
            amps = amps.at[..., mm, m].add(val)
    return amps
