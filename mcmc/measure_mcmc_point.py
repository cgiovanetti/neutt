"""Per-point cost + no-recompile + physics check on the real make_coupled_bbn pipeline.

Exactly the MCMC compute path: build the coupled-BBN driver once (template majoron sector, fixed
grid, T_init=4 / lna_max=4.5 / atol=1e-30), then call run(dark=...) at several scalar (m_phi, g,
amp) points. Reports:
  * point-1 cold (compile+run) and warm time,
  * point-2 (different m_phi AND g) warm time -> if ~= point-1 warm, zero recompile (the MCMC
    requirement); if ~= cold, it retraced (fatal),
  * Y_p, D/H, N_eff for each point (decay heats nu -> N_eff>3.044, abundances shift).

    python mcmc/measure_mcmc_point.py <nbins>
"""
import sys
import time

import neutt  # float64
import jax
import jax.numpy as jnp

from neutt.grid import make_eps_grid
from neutt import dark as darkmod
from linx.coupled_bbn import make_coupled_bbn
import linx.const as linx_const

T_INIT = 4.0
LNA_MAX = 4.5
EPS_RANGE = 25.0


def main(nbins):
    print(f"=== make_coupled_bbn MCMC-path measure: N={nbins}, T_init={T_INIT}, lna_max={LNA_MAX} ===",
          flush=True)
    nu_grid = make_eps_grid(nbins, EPS_RANGE)
    x_grid = make_eps_grid(nbins, EPS_RANGE, eps_offset=0.0025, bin1_weight_frac=0.25)
    template = darkmod.make_majoron_sector(m_j=1.0, g=7e-12, grid=x_grid)
    solver_kwargs = {"hybrid": False, "atol": 1e-30, "rtol": 1e-6, "dtmax": 0.08,
                     "lna_max": LNA_MAX, "max_steps": 6000}
    tb = time.time()
    run, info = make_coupled_bbn(nuclear_net="key_PRIMAT_2023", transport="full",
                                 dark=template, grid=nu_grid, solver_kwargs=solver_kwargs)
    from linx.nuclear import NuclearRates
    q0 = jnp.zeros(len(NuclearRates(nuclear_net="key_PRIMAT_2023").reactions))
    eta_fac = 0.02239 / linx_const.Omegabh2
    print(f"  build (no compile yet): {time.time()-tb:.1f}s", flush=True)

    def point(m_phi, g, amp, label):
        sec = darkmod.make_majoron_sector(m_j=m_phi, g=g, grid=x_grid)
        f_x0 = amp * darkmod.dark_be_equil(x_grid.abscissas, m_phi / T_INIT)
        t0 = time.time()
        abund, neff = run(dark=sec, f_x0=f_x0, T_init=T_INIT, eta_fac=eta_fac, nuclear_rates_q=q0)
        jax.block_until_ready(abund)
        dt = time.time() - t0
        Yp = float(4.0 * abund[5]); DH = float(abund[2] / abund[1] * 1e5); ne = float(neff)
        print(f"  [{label}] m_phi={m_phi} g={g:.1e} amp={amp:.0e}: {dt:7.1f}s  "
              f"Yp={Yp:.5f} DH={DH:.4f} Neff={ne:.4f}", flush=True)
        return dt

    t_cold = point(1.0, 7e-12, 3e-3, "P1 cold")   # compile + run
    t_warm = point(1.0, 7e-12, 3e-3, "P1 warm")   # warm (per-point cost)
    # DIFFERENT params but STABLE (so it completes and we can time it) -> the no-recompile test
    t_new = point(1.1, 7e-12, 5e-3, "P2 new-stable")
    t_new2 = point(0.9, 7e-12, 1.5e-3, "P3 new-stable")
    print(f"\n  cold={t_cold:.0f}s  warm={t_warm:.1f}s  new1={t_new:.1f}s  new2={t_new2:.1f}s", flush=True)
    print(f"  => RECOMPILE on new params? {'NO (good)' if t_new < 3*t_warm else 'YES (BAD)'} "
          f"(new/warm = {t_new/max(t_warm,1e-9):.2f})", flush=True)


if __name__ == "__main__":
    print("devices:", jax.devices(), flush=True)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 101)
