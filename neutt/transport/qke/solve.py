"""A plain QKE driver: Strang-step ρ[cp,N,3,3] over a given background, plus ρ moments.

``evolve`` uses ``nonadiabatic.strang_step``: half an explicit collision step, the exact coherent
unitary for the midpoint H (so the ~1e5–1e6 s⁻¹ oscillation never bounds dt), half a collision
step.  The Hamiltonian comes from ``coherent.py`` (vacuum + matter; the ν–ν terms are pinned to
zero) and the collision is an optional callable.  The explicit collision half steps suit a resolved,
non-stiff collision only; the production solve is ``full_coupled.py``.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp

from ...constants import PI
from . import coherent, nonadiabatic


def number_density(grid, rho, T_cm):
    """Per-flavour ν number density n[cp, a] = T_cm³/(2π²) Σ_i w_i ε_i² ρ[cp,i,a,a] [MeV³]."""
    eps = grid.abscissas
    w = grid.weights
    diag = jnp.real(jnp.diagonal(rho, axis1=-2, axis2=-1))  # (cp, N, flavor)
    return T_cm**3 / (2.0 * PI**2) * jnp.einsum("i,i,cif->cf", w, eps**2, diag)


def energy_density(grid, rho, T_cm):
    """Total ν energy density ρ_E = T_cm⁴/(2π²) Σ_{cp,flavor,i} w_i ε_i³ ρ[cp,i,a,a] [MeV⁴]."""
    eps = grid.abscissas
    w = grid.weights
    diag = jnp.real(jnp.diagonal(rho, axis1=-2, axis2=-1))  # (cp, N, flavor)
    return T_cm**4 / (2.0 * PI**2) * jnp.sum(w[None, :, None] * eps[None, :, None]**3 * diag)


def nudiff_nusum(grid, rho, diff_pow: int = 2, sum_pow: int = 3):
    """The ν–ν source moments (nudiff, nusum), each (3,3) complex.

        nudiff = Σ_i w_i ε_i^diff_pow (ρ_ν − ρ_ν̄)[i]   (drives sia)
        nusum  = Σ_i w_i ε_i^sum_pow  (ρ_ν + ρ_ν̄)[i]   (drives sis)

    With no lepton asymmetry the solvers' state is ρ̄ = ρ* (conjugate convention), not ρ̄ = ρ, so
    nudiff = 2i Σ_i w_i ε_i^diff_pow Im ρ_ν[i]: its diagonal vanishes, its off-diagonal does not.
    """
    eps = grid.abscissas
    w = grid.weights
    nu, nb = rho[0], rho[1]                          # (N,3,3) each
    we_d = (w * eps**diff_pow)[:, None, None]
    we_s = (w * eps**sum_pow)[:, None, None]
    nudiff = jnp.sum(we_d * (nu - nb), axis=0)
    nusum = jnp.sum(we_s * (nu + nb), axis=0)
    return nudiff, nusum


def coherence_norm(rho):
    """Total off-diagonal (coherence) magnitude ``Σ_{a≠b} |ρ[cp,i,a,b]|`` — a decoherence diagnostic."""
    mask = (1.0 - jnp.eye(rho.shape[-1]))[None, None]
    return jnp.sum(jnp.abs(rho) * mask)


def make_coherent_H_fn(grid, m2_weak, bg_fn):
    """H_fn(t, rho=None) → the coherent Hamiltonian H[cp,N,3,3] [s⁻¹] on the background ``bg_fn``.

    Parameters
    ----------
    grid : EpsGrid
    m2_weak : (3,3,2)
        Weak-basis mass-squared matrix (``coherent.m2_weak()``).
    bg_fn : callable(t) -> dict
        'T_cm' [MeV], and optionally 'therm_corr' (e± energy density) and 'npro' (proton density).

    The ν–ν terms are zero (pinned, as in ``full_coupled``); ``rho`` is accepted and unused.  H is 0
    in the ε = 0 bin, as in ``full_coupled``.
    """
    eps = grid.abscissas
    eps_pos = (eps > 0.0)[None, :, None, None]

    def H_fn(t, rho=None):
        bg = bg_fn(t)
        T_cm = bg["T_cm"]
        mata, mats = coherent.matter_terms(bg.get("therm_corr", 0.0), bg.get("npro", 0.0))
        # ν–ν pinned (see full_coupled's module header):
        # if with_self_interaction and rho is not None:
        #     nudiff, nusum = nudiff_nusum(grid, rho)
        #     sia, sis = coherent.self_interaction_terms(T_cm, nudiff, nusum)
        z = jnp.zeros((3, 3), dtype=jnp.complex128)
        sia, sis = z, z
        return jnp.where(eps_pos, coherent.hamiltonian(eps, T_cm, m2_weak, mata, mats, sia, sis), 0.0)

    return H_fn


def evolve(rho0, t_grid, H_fn, collision_fn=None):
    """Strang-step ρ over ``t_grid``; returns the trajectory (M, cp, N, 3, 3), rho0 first.

    Parameters
    ----------
    rho0 : (cp, N, 3, 3) complex
    t_grid : (M,) increasing times [s]
    H_fn : callable(t) or callable(t, rho) -> (cp, N, 3, 3) [s⁻¹]
        A two-argument H_fn gets the step's starting ρ, so a ρ-dependent H is frozen per step.
    collision_fn : callable(rho, t) -> (cp, N, 3, 3) [s⁻¹], or None for no collision
    """
    if collision_fn is None:
        collision_fn = lambda rho, t: jnp.zeros_like(rho)
    try:
        _takes_rho = len(inspect.signature(H_fn).parameters) >= 2
    except (ValueError, TypeError):
        _takes_rho = False

    def step(rho, idx):
        t0 = t_grid[idx]
        t1 = t_grid[idx + 1]
        dt = t1 - t0
        H_t = (lambda tt: H_fn(tt, rho)) if _takes_rho else H_fn
        rho_new = nonadiabatic.strang_step(rho, H_t, collision_fn, t0, dt)
        return rho_new, rho_new

    rho_final, traj = jax.lax.scan(step, rho0, jnp.arange(t_grid.shape[0] - 1))
    return jnp.concatenate([rho0[None], traj], axis=0)
