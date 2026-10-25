"""The comoving-momentum grid, its quadrature, equilibrium occupations and moments.

Every distribution lives on ε = p/T_cm.  The grid is linear in ε with composite Boole's-rule
weights, as in BURST, so ∫ g(ε) dε ≈ Σ_i w_i g(ε_i).  The collision integrals' inner integrals
use fixed Gauss rules, precomputed on the host and rescaled to their (traced) intervals.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_genlaguerre

from .constants import BR_COEFFS, BR_END, PI

LOG_OCC_FLOOR = -300.0  # clamp floor on log(occupation)


# ---- the ε-grid --------------------------------------------------------------------------------
class EpsGrid(eqx.Module):
    """A linear ε-grid with composite-Boole weights (Σ w_i = eps_range)."""

    abscissas: jnp.ndarray
    weights: jnp.ndarray
    nbins: int = eqx.field(static=True)
    eps_range: float = eqx.field(static=True)
    eps_offset: float = eqx.field(static=True)


def make_eps_grid(nbins: int = 101, eps_range: float = 25.0, eps_offset: float = 0.0,
                  bin1_weight_frac: float = 0.0) -> EpsGrid:
    """BURST's ε-grid: ``nbins`` (= 4n+1) points on [eps_offset, eps_offset + eps_range].

    ``bin1_weight_frac`` adds eps_offset·frac to the first weight (BURST's offset term).
    """
    if (nbins - 1) % 4 != 0:
        raise ValueError(f"Boole's rule requires nbins = 4n+1; got nbins={nbins}.")
    h = eps_range / (nbins - 1)
    i = np.arange(nbins)
    absc = i * h + eps_offset
    w = np.empty(nbins, dtype=np.float64)
    j = np.arange(1, nbins - 1)
    w[1:-1] = h * np.asarray(BR_COEFFS)[(j - 1) % 4]       # interior: cycle the Boole coefficients
    w[0] = h * BR_END + eps_offset * bin1_weight_frac       # end points carry 14/45
    w[-1] = h * BR_END
    return EpsGrid(abscissas=jnp.asarray(absc), weights=jnp.asarray(w), nbins=int(nbins),
                   eps_range=float(eps_range), eps_offset=float(eps_offset))


# ---- Gauss rules -------------------------------------------------------------------------------
def legendre_canonical(n: int):
    """Gauss–Legendre nodes/weights on [−1, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(x), jnp.asarray(w)


def gauss_laguerre(n: int, alpha: float = 0.0):
    """Generalized Gauss–Laguerre: ∫_0^∞ x^alpha e^{−x} f dx ≈ Σ w_i f(x_i)."""
    x, w = roots_genlaguerre(n, alpha)
    return jnp.asarray(np.asarray(x, dtype=np.float64)), jnp.asarray(np.asarray(w, dtype=np.float64))


def gauss_laguerre_plain(n: int):
    """Laguerre nodes with e^{x} folded into the weights: Σ w_i f(x_i) ≈ ∫_0^∞ f(x) dx (BURST's)."""
    x, w = roots_genlaguerre(n, 0.0)
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return jnp.asarray(x), jnp.asarray(w * np.exp(x))


def rescale_legendre(nodes_canon, weights_canon, a, b):
    """Map the canonical [−1, 1] Legendre rule to [a, b] (jittable; a, b may be traced)."""
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    return mid + half * nodes_canon, half * weights_canon


def gauss_legendre(n: int, a: float = -1.0, b: float = 1.0):
    """The canonical Legendre rule rescaled to a static [a, b]."""
    xc, wc = legendre_canonical(n)
    return rescale_legendre(xc, wc, a, b)


# ---- equilibrium occupations -------------------------------------------------------------------
def fermi_dirac(eps, escale_ratio: float = 1.0, xi: float = 0.0):
    """1/(exp(ε/escale_ratio − ξ) + 1), escale_ratio = T_species/T_cm."""
    return 1.0 / (jnp.exp(eps / escale_ratio - xi) + 1.0)


def bose_einstein(eps, escale_ratio: float = 1.0, xi: float = 0.0):
    """1/(exp(ε/escale_ratio − ξ) − 1), zero outside arg ∈ (0, 300] (no ε → 0 divergence)."""
    arg = eps / escale_ratio - xi
    f = 1.0 / (jnp.exp(jnp.where(arg > 0.0, arg, 1.0)) - 1.0)
    return jnp.where((arg > 0.0) & (arg <= 300.0), f, 0.0)


def log_occupation(occ):
    """log(occ) clamped to [LOG_OCC_FLOOR, 0], as BURST's log-space interpolation does."""
    return jnp.clip(jnp.log(occ), LOG_OCC_FLOOR, 0.0)


# ---- moments -----------------------------------------------------------------------------------
def integrate(grid: EpsGrid, fvals):
    """∫ f(ε) dε ≈ Σ_i w_i f_i."""
    return jnp.sum(grid.weights * fvals)


def moment(grid: EpsGrid, occ, k: int):
    """∫ ε^k occ(ε) dε."""
    return jnp.sum(grid.weights * grid.abscissas**k * occ)


def number_density(grid: EpsGrid, occ, g: float = 1.0, t_cm: float = 1.0):
    """n = g/(2π²) T_cm³ ∫ ε² occ dε (massless)."""
    return g / (2.0 * PI**2) * t_cm**3 * moment(grid, occ, 2)


def energy_density(grid: EpsGrid, occ, g: float = 1.0, t_cm: float = 1.0):
    """ρ = g/(2π²) T_cm⁴ ∫ ε³ occ dε (massless)."""
    return g / (2.0 * PI**2) * t_cm**4 * moment(grid, occ, 3)
