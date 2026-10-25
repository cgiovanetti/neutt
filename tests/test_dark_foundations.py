"""Phase 5 dark-sector foundations: grids, equilibrium occupations, parameters.

Oracle-independent: validated by analytic thermal moments (the standard Fermi/Bose integrals)
and the occupation formulae, since the dark sector has no compiling Fortran reference.
"""
import numpy as np
import jax.numpy as jnp
import pytest

import neutt
from neutt import dark
from neutt.grid import make_eps_grid


def test_massless_occupations_match_formula():
    eps = jnp.linspace(0.1, 10.0, 50)
    fd = dark.dark_fd_equil(eps, 0.0)
    be = dark.dark_be_equil(eps, 0.0)
    assert np.allclose(np.asarray(fd), 1.0 / (np.exp(np.asarray(eps)) + 1.0), atol=1e-14)
    assert np.allclose(np.asarray(be), 1.0 / (np.exp(np.asarray(eps)) - 1.0), atol=1e-12)


def test_massive_fd_raises_threshold():
    """A massive fermion is Boltzmann-suppressed relative to massless at the same eps."""
    eps = jnp.linspace(0.1, 5.0, 40)
    assert np.all(np.asarray(dark.dark_fd_equil(eps, 3.0)) < np.asarray(dark.dark_fd_equil(eps, 0.0)))


def test_fd_energy_density_massless_limit():
    """Massless FD energy density rho = (T^4/2pi^2) int eps^3 f deps -> 7pi^4/120 * (1/pi^2) per dof.

    Using a fine Boole grid: int eps^3/(e^eps+1) deps = 7 pi^4 / 120.
    """
    g = make_eps_grid(nbins=401, eps_range=60.0)
    eps = g.abscissas
    f = dark.dark_fd_equil(eps, 0.0)
    integral = float(jnp.sum(g.weights * eps**3 * f))
    assert abs(integral / (7.0 * np.pi**4 / 120.0) - 1.0) < 1e-4


def test_be_energy_density_massless_limit():
    """Massless BE: int eps^3/(e^eps-1) deps = pi^4/15 (offset grid avoids the eps=0 pole)."""
    g = make_eps_grid(nbins=401, eps_range=60.0, eps_offset=1e-3)
    eps = g.abscissas
    f = dark.dark_be_equil(eps, 0.0)
    integral = float(jnp.sum(g.weights * eps**3 * f))
    assert abs(integral / (np.pi**4 / 15.0) - 1.0) < 1e-3


def test_dark_params_defaults():
    p = dark.DarkParams()
    assert p.m_chi == 1.0 and p.mass_darkz == 91187.6
    assert np.allclose(np.asarray(p.g_dv_nu_diag()), [1.0, 1.0, 1.0])  # flavor-diagonal identity
    # off-diagonal couplings are zero in the default identity matrix
    assert float(p.g_dv_nu[0, 1, 0]) == 0.0


def test_dark_grids_offsets():
    assert dark.make_chi_grid().eps_offset == dark.DARK_EPS_OFFSET
    assert dark.make_dv_grid().nbins == dark.DARK_NBINS
