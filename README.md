# neutt

`neutt` is the differentiable JAX/GPU port of the neutrino-transport sector of **BURST**: `N_eff`
and the neutrino spectral distortions through weak decoupling, with finite-temperature QED and
the flavour-mixing (QKE) refinement, and (coupled to **LINX**) the light-element abundances.

## Installation

```bash
pip install neutt                 # CPU
pip install "neutt[cuda12]"       # NVIDIA GPU (CUDA 12 build of jaxlib)
```

`neutt` is pure Python. It needs float64, which it switches on in JAX when imported, so import
`neutt` before creating any JAX array.

## Quickstart

```python
from neutt.decoupling import make_decoupling_solver

solve, grid = make_decoupling_solver()    # builds the collision kernels
out = solve()                             # compiles on the first call, then solves
print(float(out["neff_final"]))           # 3.043322 (Standard Model, NLO QED)
```

Every solver has this shape: `make_*_solver(...)` builds once and returns a jitted `solve`, whose
arguments are traced, so calling it again with new values does not recompile.

## Documentation

- **Reference manual:** [`docs/burst_manual.pdf`](docs/burst_manual.pdf), covering every module,
  option and solver, with the physics and the numerical-design rationale.
- **API reference:** the Sphinx site built from `docs/sphinx/` (`pip install "neutt[docs]"`, then
  `sphinx-build docs/sphinx docs/sphinx/_build`).
- **Quickstart:** [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md).

## Releasing

`.github/workflows/release.yml` builds the sdist and wheel on every `v*` tag and publishes them to
PyPI by trusted publishing. Register this repository on PyPI as a trusted publisher for `neutt`
(workflow `release.yml`, environment `pypi`), set the version in `neutt/__init__.py`, then push a
tag such as `v0.1.0`.

## License

MIT (see `LICENSE`).
