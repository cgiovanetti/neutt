"""Pre-warm the persistent JAX compile cache for the production solver (1 rank), so the 4-rank
cobaya launch loads compiled executables instead of 4 ranks contending on cache writes."""
import neutt  # float64
import jax
import jax.numpy as jnp

from neutt.grid import make_eps_grid
from neutt import dark
from linx.coupled_bbn import make_coupled_bbn
from linx.nuclear import NuclearRates
import linx.const as lc

nu = make_eps_grid(257, 25.0)
xg = make_eps_grid(257, 25.0, eps_offset=0.0025, bin1_weight_frac=0.25)
tmpl = dark.make_majoron_sector(m_j=1.0, g=7e-12, grid=xg)
run, _ = make_coupled_bbn(
    nuclear_net="key_PRIMAT_2023", transport="full", dark=tmpl, grid=nu,
    solver_kwargs={"hybrid": False, "atol": 1e-30, "rtol": 1e-6, "dtmax": 0.08,
                   "lna_max": 4.5, "max_steps": 2000})
q0 = jnp.zeros(len(NuclearRates(nuclear_net="key_PRIMAT_2023").reactions))
abund, neff = run(dark=tmpl, f_x0=1e-3 * dark.dark_be_equil(xg.abscissas, 1.0 / 4.0),
                  T_init=4.0, eta_fac=0.02239 / lc.Omegabh2, nuclear_rates_q=q0)
jax.block_until_ready(abund)
print(f"cache warmed: Yp={float(4*abund[5]):.5f} Neff={float(neff):.4f}", flush=True)
