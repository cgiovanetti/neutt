"""Map where the dark solve is stable across the MCMC prior, and test whether implicit fixes the
stiff (late-decay) corner. N=101.

The collapse is the scalar's stiff Gamma/H>>1 final-dregs tail entering the window for slowly
decaying params (small g or small m_phi). We:
  A) EXPLICIT (Tsit5) sweep at fixed g=7e-12 over m_phi and amp -> stable region (try/except;
     a max_steps crash is caught and reported as FAIL, as cobaya stop_at_error=false would).
  B) IMPLICIT (Kvaerno5) on the worst corners + the P2 long-lifetime point -> does it complete,
     and how fast (the per-point MCMC cost if we switch to implicit)?

    python mcmc/sweep_stability.py {explicit|implicit}
"""
import sys
import time

import neutt  # float64
import jax

from neutt.grid import make_eps_grid
from neutt import dark as darkmod
from linx.coupled_bbn import make_coupled_bbn
from linx.nuclear import NuclearRates
import linx.const as linx_const

import os
T_INIT, LNA_MAX, EPS_RANGE = 4.0, 4.5, 25.0
N = int(os.environ.get("SWEEP_N", "101"))


def build(implicit):
    nu_grid = make_eps_grid(N, EPS_RANGE)
    x_grid = make_eps_grid(N, EPS_RANGE, eps_offset=0.0025, bin1_weight_frac=0.25)
    template = darkmod.make_majoron_sector(m_j=1.0, g=7e-12, grid=x_grid)
    sk = {"hybrid": False, "implicit": implicit, "atol": 1e-30, "rtol": 1e-6,
          "dtmax": 0.08, "lna_max": LNA_MAX, "max_steps": 8000}
    run, _ = make_coupled_bbn(nuclear_net="key_PRIMAT_2023", transport="full",
                              dark=template, grid=nu_grid, solver_kwargs=sk)
    q0 = jax.numpy.zeros(len(NuclearRates(nuclear_net="key_PRIMAT_2023").reactions))
    eta = 0.02239 / linx_const.Omegabh2
    return run, x_grid, q0, eta


def trypoint(run, x_grid, q0, eta, m_phi, g, amp, label):
    sec = darkmod.make_majoron_sector(m_j=m_phi, g=g, grid=x_grid)
    f_x0 = amp * darkmod.dark_be_equil(x_grid.abscissas, m_phi / T_INIT)
    t0 = time.time()
    try:
        abund, neff = run(dark=sec, f_x0=f_x0, T_init=T_INIT, eta_fac=eta, nuclear_rates_q=q0)
        jax.block_until_ready(abund)
        dt = time.time() - t0
        Yp = float(4.0 * abund[5]); ne = float(neff)
        print(f"  [{label}] m={m_phi} g={g:.0e} amp={amp:.0e}: OK  {dt:6.1f}s  Yp={Yp:.5f} Neff={ne:.4f}",
              flush=True)
        return True
    except Exception as e:
        print(f"  [{label}] m={m_phi} g={g:.0e} amp={amp:.0e}: FAIL {time.time()-t0:6.1f}s  "
              f"{type(e).__name__}: {str(e)[:60]}", flush=True)
        return False


def main(mode):
    implicit = (mode == "implicit")
    print(f"=== {mode} solver, N={N}, lna_max={LNA_MAX} ===", flush=True)
    run, xg, q0, eta = build(implicit)
    if not implicit:
        # EXPLICIT: the PRODUCTION prior corners (m_phi in [0.6,1.4] x amp in [1e-4,1e-1]) at this N
        for m in [0.6, 1.0, 1.4]:
            for amp in [1e-4, 1e-1]:
                trypoint(run, xg, q0, eta, m, 7e-12, amp, f"m{m}_a{amp:.0e}")
    else:
        # IMPLICIT on the stiff corners + the long-lifetime P2 point
        trypoint(run, xg, q0, eta, 1.0, 7e-12, 3e-3, "easy")     # warm baseline
        trypoint(run, xg, q0, eta, 0.6, 7e-12, 1e-1, "stiff-smallm")
        trypoint(run, xg, q0, eta, 1.6, 3e-12, 1e-2, "P2-longlife")


if __name__ == "__main__":
    print("devices:", jax.devices(), flush=True)
    main(sys.argv[1] if len(sys.argv) > 1 else "explicit")
