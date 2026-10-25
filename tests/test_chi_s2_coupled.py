"""S2 chi sector in the coupled decoupling solve: pin-safety + gate + end-to-end gates.

The S2 2↔2 legs are gated SEPARATELY from the SM legs (their own equilibrium noise floor) — they
carry the SM kernel family's ~1e-5·frs cancellation noise, so ungated they stall the solve
(scripts_dark/smoke_chi_coupled.py history). These gates lock in:

1. PIN-SAFETY (RHS level, cheap): make_rhs(chi_sector=None) adds no 'f_chi' key; with chi_sector
   set + the χ at the common-T equilibrium IC, the gate zeros the dark legs so the SM {T_pl, f}
   are bit-close to the SM-only RHS (the dark legs do not perturb the pinned SM path).
2. GATE PASSES SIGNAL: an out-of-equilibrium χ yields a nonzero gated df_χ.
3. END-TO-END (slow): a gated coupled evolve from the equilibrium IC runs to completion with a
   finite, sane N_eff and a finite f_χ.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from neutt.grid import make_eps_grid
from neutt.dark import make_chi_sector, dark_fd_equil
from neutt.decoupling import (make_rhs, make_decoupling_solver, build_nunu_kernel,
                            build_nunubar_kernel)
from neutt.transport import assemble as A
from neutt.transport import chi_brute

T_INIT = 10.0
# The coupled solve uses the BRUTE kernel (chi_brute_kernel=True) — the cut kernels' ~40 segments
# make the diffeqsolve graph too large to compile. The gate's prec floor and the RHS rates MUST
# use the same kernel (the test_ratio = |net|/frs/prec is only consistent at equal quadrature).


@pytest.fixture(scope="module")
def setup():
    grid = make_eps_grid()
    knu = build_nunu_kernel(grid)
    knub = build_nunubar_kernel(grid)
    prec = A.compute_prec_trans(grid, knu, knub, use_sched=True)
    occ0 = A.equilibrium_occ(grid)
    chi = make_chi_sector(m_chi=1.0, m_zp=200.0, g_chi=1.0, g_nu=1.0, tratio=1.0)
    meps0 = chi.m_chi / T_INIT
    f_chi_eq = dark_fd_equil(chi.grid.abscissas, meps0)
    chi_prec = chi_brute.s2_brute_prec(grid, occ0, f_chi_eq, chi.grid, meps0, 1.0, chi.Kflav)
    return grid, knu, knub, prec, occ0, chi, f_chi_eq, chi_prec


def test_no_fchi_key_when_none(setup):
    grid, knu, knub, prec, occ0, *_ = setup
    rhs = make_rhs(grid, prec, knu, knub, T_INIT, use_sched=True)
    o = jax.jit(rhs)(0.0, {"T_pl": jnp.asarray(T_INIT), "f": occ0}, None)
    assert set(o.keys()) == {"T_pl", "f"}


def test_pin_safety_gate_off_at_equilibrium(setup):
    """χ at the common-T equilibrium IC → gate OFF → SM {T_pl, f} unperturbed by the dark legs."""
    grid, knu, knub, prec, occ0, chi, f_chi_eq, chi_prec = setup
    rhs_sm = make_rhs(grid, prec, knu, knub, T_INIT, use_sched=True)
    rhs_chi = make_rhs(grid, prec, knu, knub, T_INIT, use_sched=True, chi_sector=chi, chi_prec=chi_prec)
    o_sm = jax.jit(rhs_sm)(0.0, {"T_pl": jnp.asarray(T_INIT), "f": occ0}, None)
    o_chi = jax.jit(rhs_chi)(0.0, {"T_pl": jnp.asarray(T_INIT), "f": occ0, "f_chi": f_chi_eq}, None)
    sm_scale = float(np.max(np.abs(np.asarray(o_sm["f"])))) + 1e-300
    df = float(np.max(np.abs(np.asarray(o_chi["f"]) - np.asarray(o_sm["f"]))))
    assert df < 10.0 * sm_scale, f"gate leaks at equilibrium: df={df:.3e} vs SM scale {sm_scale:.3e}"
    assert float(np.max(np.abs(np.asarray(o_chi["f_chi"])))) == 0.0  # fully gated


def test_gate_passes_out_of_equilibrium_signal(setup):
    grid, knu, knub, prec, occ0, chi, f_chi_eq, chi_prec = setup
    rhs_chi = make_rhs(grid, prec, knu, knub, T_INIT, use_sched=True, chi_sector=chi, chi_prec=chi_prec)
    f_chi_hot = dark_fd_equil(chi.grid.abscissas, chi.m_chi / (1.5 * T_INIT))   # χ hotter
    o = jax.jit(rhs_chi)(0.0, {"T_pl": jnp.asarray(T_INIT), "f": occ0, "f_chi": f_chi_hot}, None)
    assert float(np.max(np.abs(np.asarray(o["f_chi"])))) > 0.0
    assert np.all(np.isfinite(np.asarray(o["f_chi"])))


@pytest.mark.slow
def test_coupled_evolve_weak_completes():
    """The weak (heavy-mediator) χ coupled solve integrates the FULL window without stalling.

    Regression guard for the K-dependent physical-significance gate (2026-06-18). Before it, the
    full-window solve stalled at max_steps even for a decoupled χ: the noise gate is K-INDEPENDENT
    so it leaves the legs open (the χ-ν system is genuinely out of equilibrium, test_ratio~1e9),
    and their tiny coarse-brute-jittery df forced ~3e-5 steps. The significance gate gates the
    negligible legs off by |df/dlna|. m_Z′=1 TeV (g=1) → χ decoupled within the window → explicit
    Tsit5 completes in ~1500 SM-like steps, N_eff ≈ SM. (throw=True default ⇒ this RAISES if it
    stalls, which is exactly the regression we guard against.)
    """
    chi = make_chi_sector(m_chi=1.0, m_zp=1.0e6, g_chi=1.0, g_nu=1.0, tratio=1.0)
    solve, _ = make_decoupling_solver(n_save=16, use_sched=True, chi_sector=chi)
    out = solve(T_init=T_INIT, lna_max=7.0)
    fc = np.asarray(out["f_chi_final"])
    neff = float(out["neff_final"])
    assert np.all(np.isfinite(fc))
    assert np.isfinite(neff) and 2.9 < neff < 3.2, f"N_eff={neff}"   # ≈ SM 3.044 (χ decoupled)
    assert np.isfinite(float(out["rho_chi_final"]))
    assert np.all(np.isfinite(np.asarray(out["f_chi"])))   # full trajectory finite
