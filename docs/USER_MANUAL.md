# neutt — quickstart

`neutt` is the JAX/GPU port of BURST's neutrino-transport sector (the
weak-decoupling calculation of $N_{\rm eff}$ and the neutrino spectral
distortions), plus its coupling to **LINX** for light-element abundances.

**This file is only a quickstart. The full reference is
[`docs/burst_manual.pdf`](burst_manual.pdf)** — it documents every module, option
and solver, with the physics and the numerical-design rationale, and is meant to be
usable on its own. When in doubt, read the manual.

<!-- sphinx: the quickstart page includes this file from here -->

Set `$BURST_ROOT` to your BURST checkout and `$LINX_ROOT` to your LINX checkout.

## Environment (Perlmutter)

Everything that touches JAX runs on a GPU **compute** node, never the login node:

```bash
salloc --nodes=1 --qos=interactive --time=04:00:00 --constraint=gpu --gpus=1 --account=m3166_g
module load conda && conda activate actdr6
export PYTHONPATH=$BURST_ROOT          # add :$LINX_ROOT for coupled BBN
```

Each fresh shell must re-run `module load conda && conda activate actdr6`.

## 1. Standalone N_eff (transport only, no BBN)

```python
from neutt.decoupling import make_decoupling_solver   # importing neutt enables float64
solve, grid = make_decoupling_solver()    # builds the kernels
out = solve()                             # compiles once (~40–80 s), then solves
print(float(out["neff_final"]))           # 3.043322  (NLO QED, the default)
# make_decoupling_solver(qed_nlo=False)[0]() -> 3.044276  (LO; 3.044225 at BURST's sin^2 theta_W = 0.23)
```

Every solver has this shape: `make_*_solver(...)` builds the kernels and returns a jitted
`solve`; call `solve` as often as you like (its arguments are traced, so changing them does
not recompile; the neutt banner prints whenever something does compile). For repeated solves
(a scan / MCMC) the hybrid solver is 2.4× faster:

```python
from neutt.hybrid import make_hybrid_solver
solve, grid = make_hybrid_solver()        # compiles on the first call
out = solve(); out2 = solve(delta_neff_init=0.1)   # ~2.8 s warm; traced: the first keyword call compiles once, later values reuse it
```

## 2. Coupled BBN (abundances + N_eff)

```python
import neutt                                              # FIRST: enables float64
import pandas as pd; pd.options.future.infer_string = False  # cobaya/BAO compatibility
from linx.coupled_bbn import make_coupled_bbn

run, info = make_coupled_bbn(max_steps=8192)   # build ONCE, reuse for a whole chain
res, neff = run(delta_neff_init=0.0)                 # ~80 s maiden (cold), ~4.2 s warm
Yp, DH = float(res[5] * 4.0), float(res[2] / res[1])
print(neff, Yp, DH)        # N_eff 3.043322 (NLO); Yp ~0.2470, D/H ~2.44e-5 (network-dependent)
```

`run(...)` is serialized-MCMC safe: the physics scalars (`delta_neff_init`, `eta_fac`,
`tau_n_fac`, `nuclear_rates_q`, `me`, `T_init`, `xi`) are traced — nothing
recompiles per call.

## Choosing a solver / mode

| you want | use |
|---|---|
| production coupled BBN (any energy-exchanging physics) | `make_coupled_bbn()` — `transport="full"`, hybrid solver, **~4.2 s** warm |
| SM-grade fast path (SM, or SM + inert dark radiation only) | `make_coupled_bbn(transport="semi")` — **~2.4 s**; *never* for physics that exchanges energy with the neutrinos |
| flavour-mixing ΔN_eff | `make_coupled_bbn(transport="qke")` |
| a reference / arbiter number | `make_coupled_bbn(solver_kwargs={"hybrid": False, "rtol": 1e-8})` |

See the manual for the full option matrix, the tolerance knobs, and the troubleshooting
table. The dark-sector extensions live on the `dark` branch.

## Canonical numbers (Standard Model)

| quantity | value | configuration |
|---|---|---|
| N_eff (NLO QED — default) | 3.0433222 | standalone, 10 MeV start |
| N_eff (LO QED) | 3.0442761 | `qed_nlo=False` |
| N_eff (LO QED, BURST-faithful) | 3.0442252 | `qed_nlo=False` at BURST's sin²θ_W = 0.23 |
| N_eff (decoupling only) | 3.034029 | `use_renorm=False` |
| coupled N_eff (NLO) | 3.0433222 | `make_coupled_bbn(solver_kwargs={"hybrid": False, "rtol": 1e-8})` |
| Yp, D/H | ≈0.2470, ≈2.44e-5 | network-dependent (`key_recommended`) |

## Tests

```bash
cd $BURST_ROOT && PYTHONPATH=$PWD srun -n1 -c32 --gpus-per-task=1 \
  python -m pytest tests/ -q          # the first run pays a ~10 min JAX compile (not a hang)
```
