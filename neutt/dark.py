"""Dark-sector foundations: grids, equilibrium occupations, and parameters.

BURST's dark sector adds two species interacting with neutrinos through a dark mediator: chi (a
dark-matter fermion, FD, mass m_chi, at T_chi) and dv (a dark vector boson, BE, mass m_dv, at
T_dv). Both are discretized on a linear ε-grid in their own temperature units (ε = p/T_species);
the kernels carry the unit conversion via meps = mass/T_cm and tratio = T_cm/T_species. The BE
occupation diverges as ε,m→0, so use eps_offset > 0.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .grid import make_eps_grid, EpsGrid

# Defaults from the BURST config .ini files.
M_CHI_DEFAULT = 1.0          # MeV   (chi fermion mass)
M_DV_DEFAULT = 1.0           # MeV   (dark vector mass)
MASS_DARKZ_DEFAULT = 91187.6  # MeV  (dark Z' mediator, ~ the SM Z)

# dark-grid defaults
DARK_NBINS = 101
DARK_EPS_RANGE = 25.0
DARK_EPS_OFFSET = 0.0025
CHI_BIN1_WEIGHT_FRAC = 0.5
DV_BIN1_WEIGHT_FRAC = 0.25


def _identity_coupling():
    """Flavor-diagonal coupling matrix g[flavor, flavor, cp] = identity (the .ini default)."""
    g = np.zeros((3, 3, 2), dtype=np.float64)
    for m in range(3):
        g[m, m, 0] = 1.0
        g[m, m, 1] = 1.0
    return jnp.asarray(g)


class DarkParams(eqx.Module):
    """Container for the dark-sector couplings, masses, and temperature ratios.

    An ``equinox.Module`` whose every field is a dynamic jnp leaf, so an instance can be
    vmap/grad/jit-swept directly (a dark-mass or coupling scan compiles once). Construct with
    floats (promoted to float64).

    Attributes
    ----------
    m_chi, m_dv, mass_darkz : jnp.ndarray
        chi, dark-vector, and dark-Z' mediator masses [MeV].
    g_darkz_chi : jnp.ndarray
        dark-Z'–chi coupling, shape (2,).
    g_darkz_nu, g_dv_nu : jnp.ndarray
        (3,3,2) flavor×flavor×cp coupling matrices (default flavor-diagonal identity). Kernels use
        only the diagonal product g[m,m,0]·g[m,m,1] per flavor (off-diagonal needs the full matrix).
    tchi_over_tcm, tdv_over_tcm : jnp.ndarray
        Species temperature ratios T_species/T_cm (escale=1 ⇒ shares the comoving ν temperature).
    """

    m_chi: jnp.ndarray
    m_dv: jnp.ndarray
    mass_darkz: jnp.ndarray
    g_darkz_chi: jnp.ndarray
    g_darkz_nu: jnp.ndarray
    g_dv_nu: jnp.ndarray
    tchi_over_tcm: jnp.ndarray
    tdv_over_tcm: jnp.ndarray

    def __init__(self, m_chi=M_CHI_DEFAULT, m_dv=M_DV_DEFAULT, mass_darkz=MASS_DARKZ_DEFAULT,
                 g_darkz_chi=(1.0, 1.0), g_darkz_nu=None, g_dv_nu=None,
                 tchi_over_tcm=1.1, tdv_over_tcm=1.1):
        self.m_chi = jnp.asarray(m_chi, dtype=jnp.float64)
        self.m_dv = jnp.asarray(m_dv, dtype=jnp.float64)
        self.mass_darkz = jnp.asarray(mass_darkz, dtype=jnp.float64)
        self.g_darkz_chi = jnp.asarray(np.asarray(g_darkz_chi, dtype=np.float64))
        self.g_darkz_nu = _identity_coupling() if g_darkz_nu is None else jnp.asarray(g_darkz_nu)
        self.g_dv_nu = _identity_coupling() if g_dv_nu is None else jnp.asarray(g_dv_nu)
        self.tchi_over_tcm = jnp.asarray(tchi_over_tcm, dtype=jnp.float64)
        self.tdv_over_tcm = jnp.asarray(tdv_over_tcm, dtype=jnp.float64)

    def g_dv_nu_diag(self):
        """Per-flavor diagonal dv-nu coupling product g[m,m,0]*g[m,m,1], shape (3,)."""
        return self.g_dv_nu[jnp.arange(3), jnp.arange(3), 0] * self.g_dv_nu[jnp.arange(3), jnp.arange(3), 1]


class BosonDecaySector(eqx.Module):
    """A dark boson X ↔ ν ν̄ decay sector for the coupled solve (scenarios S1 and S3).

    Couplings/masses are dynamic jnp leaves (sweepable); ``spin_dof`` is static (it selects
    the kernel's polarization average at trace time). ``gsq_diag`` is the per-flavor
    |M|²-coefficient: Σ|M|² = 2·gsq·m_x² (S1 vector: gsq = g_ν²; S3 majoron: gsq = g²/2 in
    the CGJPS convention of arXiv:2401.00687).

    Attributes
    ----------
    m_x : jnp.ndarray
        Boson mass [MeV].
    gsq_diag : jnp.ndarray
        (3,) per-flavor coupling-squared.
    tratio : jnp.ndarray
        T_cm/T_X grid-unit ratio (a static unit label for the X ε-grid; both temperatures
        redshift together, so it is constant along the comoving evolution).
    grid : EpsGrid
        The X ε-grid (ε = p/T_X).
    spin_dof : int (static)
        X dof: 3 (vector) or 1 (scalar).
    """

    m_x: jnp.ndarray
    gsq_diag: jnp.ndarray
    tratio: jnp.ndarray
    grid: EpsGrid
    spin_dof: int = eqx.field(static=True)

    def __init__(self, m_x, gsq_diag, spin_dof, grid=None, tratio=1.0):
        self.m_x = jnp.asarray(m_x, dtype=jnp.float64)
        self.gsq_diag = jnp.asarray(gsq_diag, dtype=jnp.float64) * jnp.ones(3)
        self.tratio = jnp.asarray(tratio, dtype=jnp.float64)
        self.grid = make_dv_grid() if grid is None else grid
        self.spin_dof = int(spin_dof)


def make_vector_decay_sector(m_zp=M_DV_DEFAULT, g_nu=1.0, grid=None, tratio=1.0):
    """Scenario S1: vector DM Z' with L = g_nu Z'_mu nubar ga^mu P_L nu (per-flavor g_nu).

    Gamma(Z' -> nu nubar) = g_nu² m/(24π) per flavor. ``g_nu`` may be a scalar (universal)
    or a (3,) per-flavor array.
    """
    g = jnp.asarray(g_nu, dtype=jnp.float64)
    return BosonDecaySector(m_zp, g**2, spin_dof=3, grid=grid, tratio=tratio)


def make_majoron_sector(m_j, g, grid=None, tratio=1.0):
    """Scenario S3: majoron J in the CGJPS (arXiv:2401.00687) convention.

    ``g`` is their universal 2-component coupling (L = -(g/2) J nu nu + h.c.); the kernel's
    Dirac-pair gsq = g²/2 reproduces their per-flavor width g² m_J/(16π) and
    tau_J = 16π/(3 g² m_J). ``g`` may be scalar or (3,).
    """
    g = jnp.asarray(g, dtype=jnp.float64)
    return BosonDecaySector(m_j, g**2 / 2.0, spin_dof=1, grid=grid, tratio=tratio)


class ChiSector(eqx.Module):
    """A Dirac-χ ↔ ν 2↔2 CONTACT sector for the coupled solve (scenario S2: heavy Z′ mediator).

    Unlike the S1/S3 1↔2 :class:`BosonDecaySector`, S2 is a 4-fermion contact interaction whose
    rate scales as K·T⁵ exactly like the SM weak kernels, so its kernels
    (neutt.transport.chi_nu) keep the SM ``tcmev⁵`` convention. The χ is a Dirac fermion with
    g = 4 internal dof (particle+antiparticle × 2 spins); a single CP-symmetric per-dof f_χ is
    tracked (``spin_dof=4`` feeds :func:`boson_moments` for ρ_χ/P_χ).

    Attributes
    ----------
    m_chi : jnp.ndarray
        χ mass [MeV].
    Kflav : jnp.ndarray
        (3,) per-flavor contact coupling K = g_χ² g_ν[α]² / m_Z′⁴ [MeV⁻⁴] (dynamic, sweepable).
    tratio : jnp.ndarray
        T_cm/T_χ grid-unit ratio (constant along the comoving evolution).
    grid : EpsGrid
        The χ ε-grid (ε = p/T_χ).
    spin_dof : int (static)
        χ dof for the moments: 4 (Dirac: particle+antiparticle × 2 spins, CP-symmetric f_χ).
    """

    m_chi: jnp.ndarray
    Kflav: jnp.ndarray
    tratio: jnp.ndarray
    grid: EpsGrid
    spin_dof: int = eqx.field(static=True)

    def __init__(self, m_chi, Kflav, tratio=1.0, grid=None, spin_dof=4):
        self.m_chi = jnp.asarray(m_chi, dtype=jnp.float64)
        self.Kflav = jnp.asarray(Kflav, dtype=jnp.float64) * jnp.ones(3)
        self.tratio = jnp.asarray(tratio, dtype=jnp.float64)
        self.grid = make_chi_grid() if grid is None else grid
        self.spin_dof = int(spin_dof)


def make_chi_sector(m_chi=M_CHI_DEFAULT, m_zp=MASS_DARKZ_DEFAULT, g_chi=1.0, g_nu=1.0,
                    tratio=1.0, grid=None):
    """Scenario S2: Dirac DM χ + heavy Z′ in the contact limit. K = g_χ² g_ν² / m_Z′⁴ per flavor.

    ``g_nu`` may be a scalar (universal) or a (3,) per-flavor array; ``g_chi`` scalar.
    """
    g_nu = jnp.asarray(g_nu, dtype=jnp.float64) * jnp.ones(3)
    g_chi = jnp.asarray(g_chi, dtype=jnp.float64)
    Kflav = g_chi ** 2 * g_nu ** 2 / jnp.asarray(m_zp, dtype=jnp.float64) ** 4
    return ChiSector(m_chi, Kflav, tratio=tratio, grid=grid, spin_dof=4)


def boson_moments(grid: EpsGrid, f_x, m_over_T, T_x, spin_dof):
    """(n, rho, P) of a boson species from its per-dof spectrum on its own ε-grid.

    n = g T³/(2π²) ∫ ε² f dε ;  rho = g T⁴/(2π²) ∫ ε² e f dε ;  P = g T⁴/(6π²) ∫ ε⁴/e f dε
    with e = √(ε² + (m/T)²). Units: T_x in MeV → n [MeV³], rho/P [MeV⁴].
    """
    eps = grid.abscissas
    w = grid.weights
    e = jnp.sqrt(eps**2 + m_over_T**2)
    pref = spin_dof / (2.0 * jnp.pi**2)
    n = pref * T_x**3 * jnp.sum(w * eps**2 * f_x)
    rho = pref * T_x**4 * jnp.sum(w * eps**2 * e * f_x)
    P = pref / 3.0 * T_x**4 * jnp.sum(w * eps**4 / jnp.where(e > 0.0, e, 1.0) * f_x)
    return n, rho, P


def make_chi_grid(nbins=DARK_NBINS, eps_range=DARK_EPS_RANGE, eps_offset=DARK_EPS_OFFSET,
                  bin1_weight_frac=CHI_BIN1_WEIGHT_FRAC) -> EpsGrid:
    """chi eps-grid (eps = p/T_chi)."""
    return make_eps_grid(nbins, eps_range, eps_offset, bin1_weight_frac)


def make_dv_grid(nbins=DARK_NBINS, eps_range=DARK_EPS_RANGE, eps_offset=DARK_EPS_OFFSET,
                 bin1_weight_frac=DV_BIN1_WEIGHT_FRAC) -> EpsGrid:
    """dv eps-grid (eps = p/T_dv)."""
    return make_eps_grid(nbins, eps_range, eps_offset, bin1_weight_frac)


def dark_fd_equil(eps, m_over_T, xi=0.0):
    """Equilibrium FD spectrum of a massive fermion on its own grid: 1/(exp(sqrt(eps^2+(m/T)^2)-xi)+1).

    eps = p/T_species (the species' own temperature units); m_over_T = mass/T_species.
    """
    ex = jnp.sqrt(eps**2 + m_over_T**2)
    return 1.0 / (jnp.exp(ex - xi) + 1.0)


def dark_be_equil(eps, m_over_T, xi=0.0):
    """Equilibrium BE spectrum of a massive boson on its own grid: 1/(exp(sqrt(eps^2+(m/T)^2)-xi)-1).

    eps = p/T_species; m_over_T = mass/T_species. Diverges as (eps,m)->0 -> use eps_offset>0.
    """
    ex = jnp.sqrt(eps**2 + m_over_T**2)
    return 1.0 / (jnp.exp(ex - xi) - 1.0)
