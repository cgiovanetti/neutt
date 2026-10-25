"""Quantum-kinetic neutrino transport: the 3×3 flavour density matrix ρ per (ν/ν̄, energy bin),
evolved under i dρ/dt = [H, ρ] + i C[ρ].

- ``coherent``      — the density-matrix state, PMNS mixing, the Hamiltonian H and −i[H, ρ].
- ``nonadiabatic``  — the coherent-step propagators (exact unitary, dense expm, Strang, hybrid
                      secular), the moving eigenframe, and the adiabatic (QSS) coherence.
- ``collide_matrix``, ``collide``, ``collide_nue``, ``collide_nunu`` — the density-matrix
                      (Sigl–Raffelt) collision operator: shared helpers, νν̄→e⁺e⁻, ν–e±, ν–ν/ν–ν̄.
- ``full_coupled``  — ``make_full_coupled_solver``: the production {T_pl, ρ} solve.
- ``solve``         — a standalone Strang driver over a given background (no plasma coupling).
"""

from . import coherent, nonadiabatic, solve, collide, collide_matrix, collide_nue, collide_nunu  # noqa: F401
from . import full_coupled  # noqa: F401
