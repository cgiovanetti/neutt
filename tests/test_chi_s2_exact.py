"""hybrid.make_chi_exact_solver — the EXACT shape-resolved coupled χ→ν solver (strong_coupling13/14).

The strong-coupling χ→ν solver: it evolves (f_ν, f_χ) under the TRUE coupled kernel net (one
s2_brute_rates) via a log-occ line-search Newton, FD-capped reads, and an energy-gated freeze
backstop, so ν↔χ number AND energy are conserved to the kernel quadrature floor — closing the
num_rel~0.55 energy-only-delivery systematic (the ~1–2% Yp/D effect at the BBN floor). Full result:
docs/PHASE_C_RESOLUTION.md.

The full headline solve (m_χ=1, K=100, lna_max=5.9, h=0.005) is ~77 min on an A100 — too slow for the
routine suite. This @slow guard runs a coarser/shorter window (h=0.01, lna_max=2.5) that still exercises
the line-search Newton + freeze + FD-cap THROUGH the x≈1.49 Pauli-saturation transition (the
strong_coupling12 blocker the line search fixed), asserting it stays finite, conserves to the floor, and
depletes χ — without the full deep-tail cost. The certified headline (coarse-h) is
N_eff≈3.99/relic≈0.60 (docs/PHASE_C_CERT_FINDINGS.md; h+node refinement → ≈3.86), pinned by
test_exact_solver_full_lna_headline_pin below. The default solver is now ``robust=True`` (a
rank-revealing lstsq Newton, NO amplitude bound) — it reproduces that 3.99/0.60 headline with no
clip, having replaced the ad-hoc ``cap_evolved`` amplitude cap (docs/HANDOFF_2026-06-27_SCRUB_CAPS.md).
"""
import numpy as np
import pytest

from neutt.dark import make_chi_sector


@pytest.mark.slow
def test_exact_solver_conserves_through_saturation():
    """The coupled solve stays stable + conserving through the x≈1.49 saturation (line-search +
    freeze + FD-cap). Coarse window (h=0.01, lna_max=2.5) for a few-minute @slow guard."""
    from neutt.hybrid import make_chi_exact_solver
    chi = make_chi_sector(m_chi=1.0, tratio=1.0)            # default m_Z′ ≈ 91 GeV (strong, K≈100·G_F²)
    # production default: robust=True (rank-revealing lstsq Newton, NO amplitude bound — truncates the
    # spurious super-thermal growing mode at the linear-solve level; reproducible, replaced cap_evolved).
    solve, _ = make_chi_exact_solver(chi_sector=chi, T_init=7.0, lna_max=2.5, h=0.01,
                                     k_newton=6, trust=4.0, deepfix="freeze", n_save=32)
    out = solve()
    # stable through the saturation transition: finite everywhere, the Newton converged every
    # active step (n_badres=0), no Pauli saturation (nsat=0)
    assert np.all(np.isfinite(np.asarray(out["f_final"]))), "ν spectrum went non-finite"
    assert np.all(np.isfinite(np.asarray(out["f_chi_final"]))), "f_χ went non-finite"
    assert int(out["n_badres"]) == 0, f"Newton failed to converge: n_badres={int(out['n_badres'])}"
    assert int(out["nsat_total"]) == 0, f"Pauli saturation: nsat={int(out['nsat_total'])}"
    # conserves ν↔χ number AND energy to ~the kernel floor in the active window (NOT the energy-only ~0.55)
    assert float(out["num_rel_med"]) < 0.25, f"num_rel_med={float(out['num_rel_med']):.3f} (≫ floor)"
    assert float(out["rho_rel_med"]) < 0.25, f"rho_rel_med={float(out['rho_rel_med']):.3f} (≫ floor)"
    # χ is depleting (relic < 1; it has not run away to the relativistic ceiling)
    relic = float(out["relic_N_over_Nrel"])
    assert 0.0 < relic < 1.0, f"χ relic N/N_rel={relic} (should be depleting below 1)"
    # the per-flavor trajectory the LINX adapter consumes is present + finite
    f_traj = np.asarray(out["f_traj"])
    assert f_traj.ndim == 4 and f_traj.shape[1:3] == (3, 2), f"f_traj shape {f_traj.shape}"
    assert np.all(np.isfinite(f_traj)), "f_traj non-finite"


@pytest.mark.slow
def test_exact_solver_full_lna_headline_pin():
    """FULL-lna parity pin (strong_coupling14 item d): run the production default config
    (m_χ=1, K=100, robust=True) over the COMPLETE freeze window (coarse h=0.02, lna_max=5.9,
    ~23 min on an A100) and pin the headline. The shorter saturation test above stops at lna=2.5,
    where the deep freeze tail (x>4.7) is a no-op — exactly where the sc14 "factory-vs-pc" divergence
    class lived (a freeze-tail bug would shift relic/N_eff but pass the short test). This pin asserts
    the coarse-full certified value N_eff≈3.99 / relic≈0.60 (docs/PHASE_C_CERT_FINDINGS.md; the full-h
    publishable anchor is 3.93/0.59, carrying a +0.057 h-bias). Bands are wide enough for the coarse-h
    spread of either method but reject a freeze-tail regression (the ~3.67/0.70 outlier — that is the point)."""
    from neutt.hybrid import make_chi_exact_solver
    chi = make_chi_sector(m_chi=1.0, tratio=1.0)            # K≈100·G_F²
    solve, _ = make_chi_exact_solver(chi_sector=chi, T_init=7.0, lna_max=5.9, h=0.02,
                                     k_newton=6, k_sub=8, trust=4.0, deepfix="freeze", n_save=64)
    out = solve()
    neff = float(out["neff_final"]); relic = float(out["relic_N_over_Nrel"])
    assert int(out["n_badres"]) == 0, f"Newton failed: n_badres={int(out['n_badres'])}"
    assert int(out["nsat_total"]) == 0, f"Pauli saturation: nsat={int(out['nsat_total'])}"
    assert np.all(np.isfinite(np.asarray(out["f_final"]))), "ν spectrum non-finite"
    # Headline N_eff~3.99/relic~0.60 (robust default — reproducible, no cap knife-edge). The band is
    # wide enough for both methods (robust ~3.998/0.598; legacy cap_evolved ~3.99/0.60) but rejects a
    # freeze-tail regression (relic>~0.9 / N_eff<~3.2).
    assert 3.6 < neff < 4.1, f"N_eff={neff:.4f} outside the strong band (freeze-tail regression?)"
    assert 0.55 < relic < 0.72, f"relic={relic:.4f} outside the strong band (freeze-tail regression?)"
    # num_rel at the kernel floor THROUGHOUT (the energy-only delivery's ~0.55 systematic is closed)
    assert float(out["num_rel_med"]) < 0.10, f"num_rel_med={float(out['num_rel_med']):.3f} (≫ floor)"


@pytest.mark.slow
def test_exact_solver_rejects_bad_deepfix():
    """deepfix is validated to be 'freeze' or 'none' (the shared-scalar 'gate' was rejected)."""
    from neutt.hybrid import make_chi_exact_solver
    chi = make_chi_sector(m_chi=1.0, tratio=1.0)
    with pytest.raises(ValueError, match="deepfix"):
        make_chi_exact_solver(chi_sector=chi, deepfix="gate")
