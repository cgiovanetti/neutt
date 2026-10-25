"""S2 (Dirac χ + heavy Z′) 2↔2 transport kernels: physics gates (chi_nu.py).

The four kernels (νχ→νχ and χχ̄→νν̄, ν-side df_ν and χ-side df_χ) are validated bin-by-bin vs an
independent NumPy brute-force oracle in scripts_dark/validate_{chiside,nuside_annih}.py (≤0.16%
of local scale; the absolute prefactor 2K/(ħ·16(2π)³)·1/(e1 q1) is first-principles + brute-
verified). These pytest gates lock in the conservation/robustness properties used by the coupled
solve (decoupling.py chi_sector):

1. DETAILED BALANCE (normalization-blind): common T → net ~ 0 for all four kernels.
2. CROSS-SPECIES ENERGY balance: energy lost by ν = energy gained by the χ sector (Dirac g_χ=4).
3. CROSS-SPECIES NUMBER balance (annihilation): g_χ=4 makes χ-side = −ν-side; scattering is
   number-conserving per side.
4. Out-of-equilibrium sign (hotter χ → net annihilation χχ̄→νν̄ heats ν).
5. frs ≥ 0, jit/finiteness/differentiability.

The Dirac g_χ=4 (particle+antiparticle × 2 spins, CP-symmetric per-dof f_χ) is REQUIRED for the
balances — verified empirically in scripts_dark/diag_xspecies_balance.py.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from neutt.grid import make_eps_grid
from neutt.dark import make_chi_grid
from neutt.transport.chi_nu import (nuside_scatter_rates, nuside_annih_rates,
                                        chiside_scatter_rates, chiside_annih_rates)

T_CM = 2.0
M_CHI = 1.0
M_ZP = 30.0
K = 1.0 / M_ZP ** 4
MEPS = M_CHI / T_CM
KFLAV = jnp.asarray([K, K, K])
G_CHI = 4.0          # Dirac: particle+antiparticle × 2 spins (CP-symmetric per-dof f_χ)


def _grids():
    return make_eps_grid(), make_chi_grid()


def _occ_nu(grid, T):
    e = grid.abscissas
    return jnp.broadcast_to(1.0 / (jnp.exp(e * T_CM / T) + 1.0), (3, 2, e.shape[0]))


def _f_chi(cg, T):
    e = cg.abscissas
    return 1.0 / (jnp.exp(jnp.sqrt(e ** 2 + (M_CHI / T) ** 2)) + 1.0)


_KERNELS = {"chi_scatter": chiside_scatter_rates, "chi_annih": chiside_annih_rates,
            "nu_scatter": nuside_scatter_rates, "nu_annih": nuside_annih_rates}


# --- 1. detailed balance (common T → net ~ 0) -----------------------------------------------

@pytest.mark.parametrize("name", list(_KERNELS))
def test_detailed_balance(name):
    grid, cg = _grids()
    occ = _occ_nu(grid, T_CM)
    f_chi = _f_chi(cg, T_CM)                      # χ at the SAME T as ν → detailed balance
    net, frs = _KERNELS[name](grid, occ, f_chi, cg, MEPS, 1.0, KFLAV)
    rel = float(jnp.max(jnp.abs(net)) / jnp.maximum(jnp.max(jnp.abs(frs)), 1e-300))
    assert rel < 1e-2, f"{name} detailed balance net/frs = {rel:.3e}"


# --- 2/3. cross-species energy + number balance (Dirac g_χ=4) -------------------------------

def _nets(T_chi):
    grid, cg = _grids()
    tratio = T_CM / T_chi
    occ = _occ_nu(grid, T_CM)
    f_chi = _f_chi(cg, T_chi)
    ns_sc, _ = nuside_scatter_rates(grid, occ, f_chi, cg, MEPS, tratio, KFLAV)
    ns_an, _ = nuside_annih_rates(grid, occ, f_chi, cg, MEPS, tratio, KFLAV)
    cs_sc, _ = chiside_scatter_rates(grid, occ, f_chi, cg, MEPS, tratio, KFLAV)
    cs_an, _ = chiside_annih_rates(grid, occ, f_chi, cg, MEPS, tratio, KFLAV)
    return grid, cg, tratio, (np.asarray(ns_sc), np.asarray(ns_an),
                              np.asarray(cs_sc), np.asarray(cs_an))


def test_energy_balance():
    """ν energy lost = χ-sector energy gained: nu_en + g_χ·chi_en_perdof ≈ 0 (g_χ=4)."""
    grid, cg, tratio, (ns_sc, ns_an, cs_sc, cs_an) = _nets(1.4 * T_CM)
    wn = np.asarray(grid.weights); wc = np.asarray(cg.weights)
    eps = np.asarray(grid.abscissas); ceps = np.asarray(cg.abscissas)
    Echi = np.sqrt((ceps / tratio) ** 2 + MEPS ** 2)
    dnu = 2.0 * ns_sc + ns_an                     # ν off χ AND χ̄ (scatter ×2) + annih
    dchi = (cs_sc + cs_an).sum(axis=0)
    nu_en = float(np.sum(wn * eps ** 3 * dnu.sum(axis=(0, 1))))
    chi_en = G_CHI / tratio ** 3 * float(np.sum(wc * ceps ** 2 * Echi * dchi))
    rel = abs(nu_en + chi_en) / max(abs(nu_en), abs(chi_en), 1e-300)
    assert rel < 0.02, f"energy balance: nu={nu_en:.4e} chi={chi_en:.4e} rel={rel:.3e}"


def test_number_balance_annih():
    """Annihilation makes 1 ν+1 ν̄ from χχ̄: g_χ·chi_num_perdof = −nu_num (g_χ=4)."""
    grid, cg, tratio, (ns_sc, ns_an, cs_sc, cs_an) = _nets(1.4 * T_CM)
    wn = np.asarray(grid.weights); wc = np.asarray(cg.weights)
    eps = np.asarray(grid.abscissas); ceps = np.asarray(cg.abscissas)
    nu_num = float(np.sum(wn * eps ** 2 * ns_an.sum(axis=(0, 1))))
    chi_num = G_CHI / tratio ** 3 * float(np.sum(wc * ceps ** 2 * cs_an.sum(axis=0)))
    rel = abs(nu_num + chi_num) / max(abs(nu_num), abs(chi_num), 1e-300)
    assert rel < 0.02, f"annih number balance: nu={nu_num:.4e} chi={chi_num:.4e} rel={rel:.3e}"


def test_scattering_conserves_number():
    """Elastic νχ→νχ: total ν number and total χ number unchanged (∫ net ≈ 0 vs the annih scale)."""
    grid, cg, tratio, (ns_sc, ns_an, cs_sc, cs_an) = _nets(1.4 * T_CM)
    wn = np.asarray(grid.weights); wc = np.asarray(cg.weights)
    eps = np.asarray(grid.abscissas); ceps = np.asarray(cg.abscissas)
    sc_nu = abs(float(np.sum(wn * eps ** 2 * ns_sc.sum(axis=(0, 1)))))
    sc_chi = abs(float(np.sum(wc * ceps ** 2 * cs_sc.sum(axis=0))))
    annih_scale = abs(float(np.sum(wn * eps ** 2 * ns_an.sum(axis=(0, 1)))))
    assert sc_nu < 1e-3 * annih_scale, f"ν scatter number not conserved: {sc_nu:.3e} vs {annih_scale:.3e}"
    assert sc_chi < 1e-3 * annih_scale, f"χ scatter number not conserved: {sc_chi:.3e} vs {annih_scale:.3e}"


# --- 4. out of equilibrium sign --------------------------------------------------------------

def test_out_of_equilibrium_sign():
    """Hotter χ bath (T_χ > T_cm) → net annihilation χχ̄→νν̄: χ loses number, ν gains."""
    grid, cg, tratio, (ns_sc, ns_an, cs_sc, cs_an) = _nets(1.5 * T_CM)
    assert float(np.max(np.abs(ns_an))) > 0.0 and float(np.max(np.abs(cs_an))) > 0.0
    # χ annihilates away (net < 0), ν created (net > 0)
    assert float(np.sum(cs_an)) < 0.0
    assert float(np.sum(ns_an)) > 0.0


# --- 5. robustness ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(_KERNELS))
def test_frs_nonneg(name):
    grid, cg = _grids()
    occ = _occ_nu(grid, T_CM)
    f_chi = _f_chi(cg, 1.4 * T_CM)
    _, frs = _KERNELS[name](grid, occ, f_chi, cg, MEPS, T_CM / (1.4 * T_CM), KFLAV)
    assert float(jnp.min(frs)) >= -1e-20, f"{name} frs has negative entries: min={float(jnp.min(frs)):.3e}"


@pytest.mark.parametrize("name", list(_KERNELS))
def test_jit_and_finite(name):
    grid, cg = _grids()
    occ = _occ_nu(grid, T_CM)
    f_chi = _f_chi(cg, 1.4 * T_CM)
    fn = jax.jit(lambda o, fc: _KERNELS[name](grid, o, fc, cg, MEPS, T_CM / (1.4 * T_CM), KFLAV))
    net, frs = fn(occ, f_chi)
    assert np.all(np.isfinite(np.asarray(net))) and np.all(np.isfinite(np.asarray(frs)))


@pytest.mark.slow
def test_differentiable():
    """grad of df_χ wrt f_chi is finite (the coupled solve differentiates it)."""
    grid, cg = _grids()
    occ = _occ_nu(grid, T_CM)

    def loss(fc):
        net, _ = chiside_annih_rates(grid, occ, fc, cg, MEPS, T_CM / (1.4 * T_CM), KFLAV)
        return jnp.sum(net ** 2)

    g = jax.grad(loss)(_f_chi(cg, 1.4 * T_CM))
    assert np.all(np.isfinite(np.asarray(g)))
