"""The Standard-Model neutrino collision integrals, recast as precomputed kernel tensors contracted
against the occupations at runtime, and their assembly into the gated RHS (``assemble``).

- ``nunu``    — ν–ν and ν̄–ν̄ scattering (grid-only kernel).
- ``nunubar`` — ν–ν̄ scattering and νν̄ → ν′ν̄′ (grid-only kernel).
- ``nuer1``, ``nuer2`` — ν–e± scattering, BURST's R1 and R2 integrals.
- ``nuepma``  — νν̄ → e⁺e⁻.
- ``qke``     — the flavour-density-matrix (QKE) sector.

The dark-sector kernels are ``dv_nu`` (boson X ↔ νν̄ decay), ``chi_nu`` (χ–ν scattering and
χχ̄ ↔ νν̄, the cut-based reference) and ``chi_brute`` (the same rates by brute-force quadrature,
a small graph for the coupled solves).

Each lepton kernel has a reference implementation (bit-faithful to BURST) and a scheduled one
(one batched contraction over a per-regime tile schedule), selected by ``use_sched``.
"""

from . import nunu, nunubar, nuer1, nuer2, nuepma  # noqa: F401
