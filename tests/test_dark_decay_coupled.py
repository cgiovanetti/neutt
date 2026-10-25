"""Coupled solve with a dark boson decay sector (scenarios S1 vector Z' / S3 majoron).

Cheap end-to-end gates (short lna windows; the full production-shape N_eff qualification runs
live in the docs record, not pytest):
1. PIN SAFETY: a zero-coupling, zero-abundance dark sector reproduces the SM solve.
2. MAJORON HEATS NEUTRINOS: an overabundant J with fast decay raises N_eff above SM and its
   number density falls.
3. Finiteness/shape of the dark outputs.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from neutt import dark
from neutt.decoupling import make_decoupling_solver

LNA = 1.5
KW = dict(T_init=4.0, lna_max=LNA, n_save=8, max_steps=40_000, use_sched=False,
          sumrules=False)
_PER_CALL = ("T_init", "lna_max", "f_x0")


def _solve_once(**kw):
    """Build the solver from the build-time keywords and run it once with the per-call ones."""
    call = {k: kw.pop(k) for k in _PER_CALL if k in kw}
    solve, _ = make_decoupling_solver(**kw)
    return solve(**call)


def test_zero_dark_sector_rhs_bitwise_pin_safe():
    """gsq=0, f_x0=0: the RHS is BITWISE identical to the SM RHS (the rigorous pin guard).

    Trajectory-level comparison is controller-path noise: the extra f_x=0 state component
    enters the PID error norm's RMS denominator, perturbing the step sequence at ~1e-9 —
    no physics content (covered by the loose check in the next test).
    """
    from neutt.grid import make_eps_grid
    from neutt.transport import assemble as A
    from neutt.transport.nunu import build_nunu_kernel
    from neutt.transport.nunubar import build_nunubar_kernel
    from neutt.decoupling import make_rhs

    grid = make_eps_grid()
    nunu_k = build_nunu_kernel(grid)
    nunubar_k = build_nunubar_kernel(grid)
    prec = A.compute_prec_trans(grid, nunu_k, nunubar_k)
    sec = dark.make_majoron_sector(m_j=1.0, g=0.0)

    rhs_sm = make_rhs(grid, prec, nunu_k, nunubar_k, 4.0)
    rhs_dk = make_rhs(grid, prec, nunu_k, nunubar_k, 4.0, dark=sec)

    occ = A.equilibrium_occ(grid) * 1.02            # slightly off-equilibrium state
    st_sm = {"T_pl": jnp.asarray(3.7), "f": occ}
    st_dk = dict(st_sm, f_x=jnp.zeros(sec.grid.abscissas.shape))
    out_sm = rhs_sm(0.3, st_sm, None)
    out_dk = rhs_dk(0.3, st_dk, None)
    assert np.array_equal(np.asarray(out_sm["f"]), np.asarray(out_dk["f"]))
    assert float(out_sm["T_pl"]) == float(out_dk["T_pl"])
    assert float(jnp.max(jnp.abs(out_dk["f_x"]))) == 0.0


def test_zero_dark_sector_trajectory_close():
    """gsq=0, f_x0=0 full short solve: same physics to controller-path noise (<1e-7)."""
    ref = _solve_once(**KW)
    sec = dark.make_majoron_sector(m_j=1.0, g=0.0)
    res = _solve_once(dark=sec, f_x0=jnp.zeros(sec.grid.abscissas.shape), **KW)
    r0 = float(ref["ratio_final"])
    r1 = float(res["ratio_final"])
    assert abs(r1 - r0) / r0 < 1e-7, f"zero dark sector shifted T_pl/T_cm: {r0} vs {r1}"
    assert float(jnp.max(jnp.abs(res["f_x_final"]))) == 0.0
    assert float(res["neff_dark_final"]) == 0.0


def test_majoron_decay_heats_neutrinos():
    """Overabundant fast-decaying majoron: n_J falls, nu sector gains energy vs SM."""
    # m_J = 5 MeV at T_init = 4: the J starts semi-nonrelativistic, so inverse decays are
    # Boltzmann-killed as T drops (m/T: 1.25 -> 5.6 over the window) and the decay COMPLETES.
    # (An m_J << T majoron instead reaches quasi-equilibrium via inverse decays and persists
    # until m/T >> 1 — correct physics, but needs a longer window than a unit test wants.)
    # COUPLING CHOICE: Gamma ~ 55/s keeps max(Gamma/H) <~ 100 (460-step solve, minutes).
    # The strong-coupling corner (g ~ 3e-9, Gamma/H ~ 10^3) has NOT been re-measured since
    # the 2026-06-12 gate fixes (its earlier "impractical on both solvers" verdicts were
    # measured on the gate-flicker-broken RHS and are void) — re-qualify before relying
    # on it; the affine/expMid head remains the principled route for extreme Gamma/H.
    sec = dark.make_majoron_sector(m_j=5.0, g=3.5e-10)
    f0 = 5.0 * dark.dark_be_equil(sec.grid.abscissas, 5.0 / 4.0)   # 5x BE (overabundant)
    ref = _solve_once(**KW)
    res = _solve_once(dark=sec, f_x0=f0, **KW)

    # J number density (per comoving volume ~ /T_cm^3 units) must drop hard
    # (measured 2026-06-12: n1/n0 = 0.0579 — the survivor fraction is the m/T = 5.6
    # late-time tail still draining at lna_max; 0.08 bounds it with headroom)
    n0 = float(jnp.sum(sec.grid.weights * sec.grid.abscissas**2 * f0))
    n1 = float(jnp.sum(sec.grid.weights * sec.grid.abscissas**2 * res["f_x_final"]))
    assert n1 < 0.08 * n0, f"majoron failed to decay: n1/n0 = {n1 / n0:.3e}"

    # the decay energy lands in the nu sector -> N_eff above the SM run's
    neff_sm = float(ref["neff_final"])
    neff_dk = float(res["neff_final"])
    assert neff_dk > neff_sm + 0.01, f"no nu heating: {neff_dk} vs SM {neff_sm}"

    # outputs finite and shaped
    assert np.all(np.isfinite(np.asarray(res["f_x"])))
    assert res["f_x"].shape == (KW["n_save"], sec.grid.abscissas.shape[0])
    assert np.isfinite(float(res["rho_x_final"]))


def test_vector_sector_runs():
    """S1 vector Z' path (spin_dof=3) runs and stays finite with a quiescent sector."""
    sec = dark.make_vector_decay_sector(m_zp=1.0, g_nu=1.0e-10)
    f0 = dark.dark_be_equil(sec.grid.abscissas, 1.0 / 4.0)
    res = _solve_once(dark=sec, f_x0=f0, **KW)
    assert np.all(np.isfinite(np.asarray(res["f_x_final"])))
    assert np.isfinite(float(res["neff_final"]))

# NOTE: cheap pytest gates for the 3-leg DECAY->SM hand-off (decay_to_sm=True) were drafted but NOT
# landed (2026-06-23): with the default RecursiveCheckpoint adjoint the 3-leg compiles TWO dark legs,
# each hitting the stochastic 10-40 min compile hang -> the gates time out in the 25-min debug window.
# Cheap coverage needs DirectAdjoint + tiny n_save and a GPU iteration to confirm; deferred with the
# stiff-dark-leg solver work. See docs/THERMAL_MAJORON_NEFF_TODO.md.
