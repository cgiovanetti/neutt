"""Dark boson X <-> nu nubar decay kernels: blind physics validation + ANALYTIC normalization.

Gates (2026-06-12 rebuild — the original Phase 5 port validated detailed balance only, which is
normalization-blind; the dimensional closure was unresolved and WRONG by meps^2*tcmev (+ a
factor 3 on the nu side masked in the old number-conservation test by a missing dof factor)):

1. DETAILED BALANCE (normalization-blind): common temperature -> net ~ 0 for both kernels.
2. DILUTE DECAY (absolute, quadrature-exact): f_nu = 0 -> d f_X/dt = -Gamma*(m/E)*f_X per bin,
   Gamma_vector = gsq*m/(24 pi hbar), Gamma_scalar = gsq*m/(8 pi hbar).
3. CROSS-SPECIES NUMBER + ENERGY BALANCE (relative normalization of the two kernels, WITH the
   spin_dof factor in the X-side integrals).
4. Scalar/vector twin ratio = 3 at equal gsq.
5. jit/finiteness/differentiability.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import neutt
from neutt import dark
from neutt.constants import PI, HBAR
from neutt.grid import make_eps_grid
from neutt.transport.dv_nu import (x1_nu_decay_rates, nu1_x_decay_rates,
                                       dv1_nu_decay_rates, j1_nu_decay_rates,
                                       x1_nu_decay_affine, nu1_x_decay_affine)

# --- backgrounds ----------------------------------------------------------------------------
T_CM = 10.0
M_X = 1.0
GSQ = jnp.ones(3)


def _nu_equilibrium(nu_grid):
    """Massless FD nu spectrum at T_cm (eps = p/T_cm): f[3,2,N] = 1/(exp(eps)+1)."""
    f = dark.dark_fd_equil(nu_grid.abscissas, 0.0)
    return jnp.broadcast_to(f, (3, 2, f.shape[0]))


def _x_equilibrium(x_grid, m_over_Tx):
    """BE X spectrum at T_X (eps = p/T_X)."""
    return dark.dark_be_equil(x_grid.abscissas, m_over_Tx)


# --- detailed balance (normalization-blind primary) ------------------------------------------

@pytest.mark.parametrize("spin_dof", [3, 1])
def test_detailed_balance_x1(spin_dof):
    """tratio=1, nu & X both in equilibrium at T_cm -> d f_X/dt net ~ 0 (to interp precision)."""
    nu_grid = make_eps_grid()
    x_grid = dark.make_dv_grid()
    meps = M_X / T_CM
    f_nu = _nu_equilibrium(nu_grid)
    f_x = _x_equilibrium(x_grid, meps)             # tratio=1 -> m/T_X = m/T_cm = meps
    net, frs = x1_nu_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, 1.0, T_CM, GSQ,
                                 spin_dof=spin_dof)
    rel = float(jnp.max(jnp.abs(net)) / jnp.max(jnp.abs(frs)))
    assert rel < 1e-2, f"x1 detailed balance net/frs = {rel:.3e}"


def test_detailed_balance_nu1():
    """tratio=1, equilibrium -> d f_nu/dt net ~ 0."""
    nu_grid = make_eps_grid()
    x_grid = dark.make_dv_grid()
    meps = M_X / T_CM
    f_nu = _nu_equilibrium(nu_grid)
    f_x = _x_equilibrium(x_grid, meps)
    net, frs = nu1_x_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, 1.0, T_CM, GSQ)
    rel = float(jnp.max(jnp.abs(net)) / jnp.max(jnp.abs(frs)))
    assert rel < 1e-2, f"nu1 detailed balance net/frs = {rel:.3e}"


# --- affine split: net == A - B*f (feeds the exponential-midpoint dark head) ------------------

@pytest.mark.parametrize("spin_dof", [3, 1])
def test_affine_split_x1(spin_dof):
    """x1_nu_decay_affine: net_X == A_X - B_X*f_X (round-off close) on an OUT-OF-EQ state."""
    nu_grid = make_eps_grid()
    x_grid = dark.make_dv_grid()
    meps, tratio = M_X / T_CM, 0.9
    # genuinely out of equilibrium so net != 0 (perturbed nu + over-abundant X)
    f_nu = _nu_equilibrium(nu_grid) * 1.3
    f_x = _x_equilibrium(x_grid, meps * tratio) * 2.0
    net, _ = x1_nu_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, tratio, T_CM, GSQ,
                               spin_dof=spin_dof)
    a_x, b_x = x1_nu_decay_affine(nu_grid, x_grid, f_nu, meps, tratio, T_CM, GSQ,
                                  spin_dof=spin_dof)
    recon = a_x - b_x * f_x[None, :]
    scale = float(jnp.max(jnp.abs(net))) + 1e-300
    assert float(jnp.max(jnp.abs(recon - net))) / scale < 1e-10, "x1 affine != net"


def test_affine_split_nu1():
    """nu1_x_decay_affine: net_nu == A_nu - B_nu*f_nu (round-off close) on an OUT-OF-EQ state."""
    nu_grid = make_eps_grid()
    x_grid = dark.make_dv_grid()
    meps, tratio = M_X / T_CM, 0.9
    f_nu = _nu_equilibrium(nu_grid) * 1.3
    f_x = _x_equilibrium(x_grid, meps * tratio) * 2.0
    net, _ = nu1_x_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, tratio, T_CM, GSQ)
    a_nu, b_nu = nu1_x_decay_affine(nu_grid, x_grid, f_nu, f_x, meps, tratio, T_CM, GSQ)
    recon = a_nu - b_nu * f_nu
    scale = float(jnp.max(jnp.abs(net))) + 1e-300
    assert float(jnp.max(jnp.abs(recon - net))) / scale < 1e-10, "nu1 affine != net"


# --- ANALYTIC absolute normalization (dilute decay) ------------------------------------------

@pytest.mark.parametrize("spin_dof,gam_denom", [(3, 24.0), (1, 8.0)])
def test_dilute_decay_rate_exact(spin_dof, gam_denom):
    """f_nu = 0: d f_X/dt = -Gamma*(m/E)*f_X per bin EXACTLY (the Legendre integrand is constant).

    Gamma = gsq*m_X/(24 pi hbar) [vector] or /(8 pi hbar) [scalar]; m/E = time dilation at the
    bin momentum. Per flavor gsq=1, three flavors sum to 3x. This pins the absolute units the
    original Phase 5 port left unresolved.
    """
    nu_grid = make_eps_grid()
    x_grid = dark.make_dv_grid()
    meps = M_X / T_CM
    tratio = 0.9
    f_nu = jnp.zeros((3, 2, nu_grid.abscissas.shape[0]))
    f_x = _x_equilibrium(x_grid, meps * tratio) * 0.37     # arbitrary amplitude
    net, frs = x1_nu_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, tratio, T_CM, GSQ,
                                 spin_dof=spin_dof)
    net_tot = np.asarray(jnp.sum(net, axis=0))             # flavor sum (3 flavors x gsq=1)

    q = np.asarray(x_grid.abscissas)
    e = np.sqrt(q**2 + (meps * tratio) ** 2)               # E/T_X
    gamma = 3.0 * M_X / (gam_denom * PI * HBAR)            # 3 flavors, gsq=1 [1/s]
    expected = -gamma * (meps * tratio / e) * np.asarray(f_x)   # m/E = meps*tratio/e
    mask = q > 0
    rel = np.max(np.abs(net_tot[mask] - expected[mask]) /
                 np.maximum(np.abs(expected[mask]), 1e-300))
    assert rel < 1e-10, f"dilute decay rate mismatch (spin_dof={spin_dof}): rel={rel:.3e}"
    # frs must equal |net| here (forfact = 0)
    assert np.allclose(np.asarray(jnp.sum(frs, axis=0))[mask], -expected[mask], rtol=1e-10)


# --- cross-species number AND energy balance (relative normalization) ------------------------

@pytest.mark.parametrize("spin_dof", [3, 1])
def test_number_and_energy_balance(spin_dof):
    """Each decay makes one nu + one nubar carrying E_X:
        spin_dof * (1/tratio^3) int q^2 net_X  = - int p^2 net_nu[cp0]          (number)
        spin_dof * (1/tratio^4) int q^2 E net_X = - int p^3 (net_nu[cp0]+[cp1])  (energy)
    (common T_cm^3/2pi^2 resp. T_cm^4/2pi^2 prefactors divided out). The X-side dof factor is
    REQUIRED: f_X is per dof while the kernel's 1/spin_dof average is per decay event — the old
    test passed without it only because the nu-side kernel wrongly carried the polarization
    average too (two canceling errors, fixed 2026-06-12).
    """
    nu_grid = make_eps_grid()
    x_grid = dark.make_dv_grid()
    meps = M_X / T_CM
    tratio = T_CM / 11.0
    f_nu = _nu_equilibrium(nu_grid)
    f_x = _x_equilibrium(x_grid, M_X / 11.0)

    net_x, _ = x1_nu_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, tratio, T_CM, GSQ,
                                 spin_dof=spin_dof)        # (3,Nx)
    net_nu, _ = nu1_x_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, tratio, T_CM, GSQ)  # (3,2,N)

    q = x_grid.abscissas
    e = jnp.sqrt(q**2 + (meps * tratio) ** 2)
    p = nu_grid.abscissas

    x_num = spin_dof / tratio**3 * float(
        jnp.sum(x_grid.weights * q**2 * jnp.sum(net_x, axis=0)))
    nu_num = float(jnp.sum(nu_grid.weights * p**2 * jnp.sum(net_nu[:, 0, :], axis=0)))
    rel_n = abs(x_num + nu_num) / max(abs(x_num), abs(nu_num), 1e-30)
    assert rel_n < 0.05, f"number balance: x={x_num:.4e} nu={nu_num:.4e} rel={rel_n:.3e}"

    x_en = spin_dof / tratio**4 * float(
        jnp.sum(x_grid.weights * q**2 * e * jnp.sum(net_x, axis=0)))
    nu_en = float(jnp.sum(nu_grid.weights * p**3 *
                          jnp.sum(net_nu[:, 0, :] + net_nu[:, 1, :], axis=0)))
    rel_e = abs(x_en + nu_en) / max(abs(x_en), abs(nu_en), 1e-30)
    assert rel_e < 0.05, f"energy balance: x={x_en:.4e} nu={nu_en:.4e} rel={rel_e:.3e}"


# --- scalar/vector twin ----------------------------------------------------------------------

def test_scalar_is_3x_vector():
    """At equal gsq the scalar (J) X-side rate is exactly 3x the vector (dv) one (1 vs 3 dof);
    the nu-side kernel is identical by construction."""
    nu_grid = make_eps_grid()
    x_grid = dark.make_dv_grid()
    meps = M_X / T_CM
    f_nu = _nu_equilibrium(nu_grid)
    f_x = _x_equilibrium(x_grid, meps * 0.9) * 0.5
    net_v, _ = dv1_nu_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, 0.9, T_CM, GSQ)
    net_s, _ = j1_nu_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, 0.9, T_CM, GSQ)
    assert np.allclose(np.asarray(net_s), 3.0 * np.asarray(net_v), rtol=1e-12)


# --- out of equilibrium / robustness ----------------------------------------------------------

def test_out_of_equilibrium_nonzero_and_signed():
    """A hotter X bath (T_X > T_cm) drives net decay X -> nu nubar: X loses, nu gains."""
    nu_grid = make_eps_grid()
    x_grid = dark.make_dv_grid()
    meps = M_X / T_CM
    tratio = T_CM / 11.0                            # T_X = 11 > T_cm = 10
    f_nu = _nu_equilibrium(nu_grid)                 # nu at T_cm
    f_x = _x_equilibrium(x_grid, M_X / 11.0)        # X at T_X = 11 (hotter)
    net_x, _ = x1_nu_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, tratio, T_CM, GSQ)
    net_nu, _ = nu1_x_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, tratio, T_CM, GSQ)
    assert float(jnp.max(jnp.abs(net_x))) > 0.0
    assert float(jnp.max(jnp.abs(net_nu))) > 0.0
    assert float(jnp.sum(net_x)) < 0.0             # net decay -> X net < 0
    assert float(jnp.sum(net_nu)) > 0.0            # neutrinos gain


def test_jit_and_finite():
    nu_grid = make_eps_grid()
    x_grid = dark.make_dv_grid()
    meps = M_X / T_CM
    f_nu = _nu_equilibrium(nu_grid)
    f_x = _x_equilibrium(x_grid, meps)
    f = jax.jit(lambda fn, fx: x1_nu_decay_rates(nu_grid, x_grid, fn, fx, meps, 0.9, T_CM, GSQ))
    net, frs = f(f_nu, f_x)
    assert np.all(np.isfinite(np.asarray(net))) and np.all(np.isfinite(np.asarray(frs)))


@pytest.mark.slow
def test_differentiable():
    """grad of the decay rate wrt the nu spectrum is finite (the coupled solve diff'ates it)."""
    nu_grid = make_eps_grid()
    x_grid = dark.make_dv_grid()
    meps = M_X / T_CM
    f_x = _x_equilibrium(x_grid, meps)

    def loss(f_nu):
        net, _ = nu1_x_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, 0.9, T_CM, GSQ)
        return jnp.sum(net**2)

    g = jax.grad(loss)(_nu_equilibrium(nu_grid))
    assert np.all(np.isfinite(np.asarray(g)))
