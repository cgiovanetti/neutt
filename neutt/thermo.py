"""Plasma thermodynamics: the e±/γ equation of state, its finite-T QED corrections, the
charge-neutral electron degeneracy, and the decoupling background clock.

Natural units (MeV, MeV⁴).  ρ_γ = (π²/15)T⁴, P = ρ/3; ρ_{e±} = (g/2π²)∫p²E[f₋+f₊]dp with g = 2,
E = √(p²+m²), f∓ = 1/(exp((E∓μ)/T)+1), evaluated in x = p/T, φ = μ/T with BURST's plain
Gauss–Laguerre rules.  The free-gas EoS keeps the bare electron mass; finite-T QED enters as
BURST's additive δρ/δP/δ(dρ/dT) built from the thermal self-energies δm_e², δm_γ² (folding a mass
shift into the free gas instead would double-count).  φ_e = μ_e/T_pl is fixed by charge
neutrality with the baryons, n_{e⁻} − n_{e⁺} = q_baryon·n_b, independently of the ν degeneracy ξ.
"""

#   import jax, neutt; from neutt.thermo import Thermo, ETA0_DEFAULT
#   th = Thermo(); f = jax.jit(lambda t: th.qed_plasma_deltas(t, th.solve_phie(t, 1.02, ETA0_DEFAULT), nlo=True))
#   with jax.log_compiles(True): print(f(1.0)); print(f(0.9))

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from .constants import ALPHA_FS, HBAR, M_ELECTRON, MPL, PI, ZETA3, _FD_CUTOFF, _EXP_MAX, ETA0_DEFAULT, N_GAMMA, _N_NU_DEFAULT
from .grid import gauss_laguerre_plain, legendre_canonical


# ==== closed forms ===============================================================================

def rho_gamma(T):
    """Photon energy density (π²/15)T⁴ (g = 2)."""
    return PI**2 / 15.0 * T**4


def pressure_gamma(T):
    """Photon pressure: ρ_γ/3."""
    return rho_gamma(T) / 3.0


# (T_γ/T_cm)³ today for instantaneous e± annihilation: carries η0 (quoted at today's photon
# temperature) onto the comoving T_cm.  The true ratio differs by ~0.2%, irrelevant for φ_e.
_ELEVEN_OVER_FOUR = 11.0 / 4.0


def nb_over_Tcm3(eta0):
    """n_b/T_cm³ for the present baryon-to-photon ratio ``eta0``: η0·N_GAMMA·(11/4)."""
    return eta0 * N_GAMMA * _ELEVEN_OVER_FOUR




def rho_nu(T_nu, n_nu=_N_NU_DEFAULT):
    """Massless thermal neutrino energy density n_nu·(7/8)(π²/15)T_ν⁴ [MeV⁴]."""
    return n_nu * 7.0 / 8.0 * PI**2 / 15.0 * T_nu**4


def hubble(rho_tot):
    """H = √(8πρ_tot/(3 MPL²))/ħ [1/s] for ρ_tot in MeV⁴."""
    return jnp.sqrt(8.0 * PI * rho_tot / (3.0 * MPL**2)) / HBAR


def neff(T_nu, T_gamma, n_nu=_N_NU_DEFAULT):
    """N_eff = (8/7)(11/4)^{4/3} ρ_ν/ρ_γ for thermal neutrinos at T_ν."""
    return (8.0 / 7.0) * (11.0 / 4.0) ** (4.0 / 3.0) * rho_nu(T_nu, n_nu) / rho_gamma(T_gamma)


# ==== occupations ================================================================================

def _fd(arg, phi):
    """Fermi–Dirac occupation 1/(exp(arg − phi) + 1), arg = E/T (no cutoff).

    0 where exp(arg − phi) overflows, with the exponent masked there so autodiff tangents stay finite;
    a NaN argument stays NaN.
    """
    x = arg - phi
    ok = ~(x > _EXP_MAX)
    return jnp.where(ok, 1.0 / (jnp.exp(jnp.where(ok, x, 0.0)) + 1.0), 0.0)


def _fd_equil(E, phie):
    """Fermi–Dirac 1/(exp(E − phie) + 1), zero for E > 300 (BURST's cutoff).

    The cutoff also zeroes the exp(E)·f² second-derivative integrands in the tail, as BURST does.
    """
    keep = E <= _FD_CUTOFF
    f = 1.0 / (jnp.exp(jnp.where(keep, E, 0.0) - phie) + 1.0)   # masked: finite AD tangents
    return jnp.where(keep, f, 0.0)


def _be(x):
    """Bose–Einstein occupation 1/(exp(x) − 1), zero outside (0, 300]."""
    f = 1.0 / (jnp.exp(x) - 1.0)
    return jnp.where((x > 0.0) & (x <= _FD_CUTOFF), f, 0.0)


# ==== the plasma model ===========================================================================

class Thermo(eqx.Module):
    """e±/γ plasma thermodynamics on fixed Gauss rules, built once here and shared by every method.

    ``N_qed`` plain Gauss–Laguerre nodes for φ_e and the QED corrections, ``N_free`` for the
    free-gas e± EoS, and the ``N_lg`` Legendre + ``N_lag`` Laguerre inner rules of the
    momentum-dependent self-energy (BURST: 25, 25, 50, 100).
    """

    N_qed: int = eqx.field(static=True)
    N_free: int = eqx.field(static=True)
    N_lg: int = eqx.field(static=True)
    N_lag: int = eqx.field(static=True)
    x_qed: jnp.ndarray
    w_qed: jnp.ndarray
    x_free: jnp.ndarray
    w_free: jnp.ndarray
    t_lg: jnp.ndarray
    w_lg: jnp.ndarray
    t_lag: jnp.ndarray
    w_lag: jnp.ndarray

    def __init__(self, N_qed=80, N_free=100, N_lg=50, N_lag=100):
        self.N_qed, self.N_free, self.N_lg, self.N_lag = int(N_qed), int(N_free), int(N_lg), int(N_lag)
        self.x_qed, self.w_qed = gauss_laguerre_plain(self.N_qed)
        self.x_free, self.w_free = gauss_laguerre_plain(self.N_free)
        self.t_lg, self.w_lg = legendre_canonical(self.N_lg)
        self.t_lag, self.w_lag = gauss_laguerre_plain(self.N_lag)

    # ---- free-gas e± and γ ----------------------------------------------------------------------

    def rho_electron(self, T, phi=0.0, m=M_ELECTRON):
        """e⁻ + e⁺ energy density [MeV⁴]: (T⁴/π²)∫x²(E/T)[f₋+f₊]dx over x = p/T, with phi = μ/T;
        ``m`` [MeV] is the bare electron mass by default."""
        x, w = self.x_free, self.w_free
        z = m / T
        e = jnp.sqrt(x**2 + z**2)               # E/T
        occ = _fd(e, phi) + _fd(e, -phi)        # electrons + positrons
        return T**4 / PI**2 * jnp.sum(w * x**2 * e * occ)

    def pressure_electron(self, T, phi=0.0, m=M_ELECTRON):
        """e⁻ + e⁺ pressure [MeV⁴]: (g/2π²)∫p²(p²/3E)[f₋+f₊]dp, g = 2; arguments as rho_electron."""
        x, w = self.x_free, self.w_free
        z = m / T
        e = jnp.sqrt(x**2 + z**2)
        occ = _fd(e, phi) + _fd(e, -phi)
        return T**4 / (3.0 * PI**2) * jnp.sum(w * x**4 / e * occ)

    def rho_plasma(self, T, phi=0.0, m=M_ELECTRON):
        """Free-gas photon + e± energy density [MeV⁴] (bare electron mass by default).

        Finite-T QED is not a shift of ``m``; the caller adds the δρ of qed_plasma_deltas.
        """
        return rho_gamma(T) + self.rho_electron(T, phi, m)

    def pressure_plasma(self, T, phi=0.0, m=M_ELECTRON):
        """Free-gas photon + e± pressure [MeV⁴]; QED δP is added by the caller, as for rho_plasma."""
        return pressure_gamma(T) + self.pressure_electron(T, phi, m)

    def drho_plasma_dT(self, T, m=M_ELECTRON):
        """dρ_plasma/dT [MeV³] at φ = 0, analytic: 4π²T³/15 + (2T³/π²)∫x²(x²+z²)f(1−f)dx, z = m/T.

        The fixed-p derivative of the continuum integral; it matches autodiff of the quadrature
        (drho_plasma_dT_autodiff) to quadrature accuracy, not bitwise.
        """
        x, w = self.x_free, self.w_free
        z = m / T
        e = jnp.sqrt(x**2 + z**2)
        f = _fd(e, 0.0)
        drho_e = 2.0 * T**3 / PI**2 * jnp.sum(w * x**2 * (x**2 + z**2) * f * (1.0 - f))
        return 4.0 * PI**2 * T**3 / 15.0 + drho_e

    def drho_plasma_dT_phi(self, T, phi, m=M_ELECTRON):
        """dρ_plasma/dT [MeV³] at fixed φ, by autodiff of the quadrature (no dφ/dT feedback)."""
        return jax.grad(lambda t: self.rho_plasma(t, phi, m))(T)

    def drho_plasma_dT_autodiff(self, T):
        """Autodiff of the quadrature at φ = 0: the cross-check for drho_plasma_dT."""
        return jax.grad(lambda t: self.rho_plasma(t, 0.0))(T)

    # ---- charge-neutral electron degeneracy φ_e --------------------------------------------------

    def net_electron_number_over_T3(self, phi, mt):
        """Net electron number (n_{e⁻} − n_{e⁺})/T_pl³ at degeneracy φ, mt = m_e/T_pl.

        (1/π²)∫x²[f(E−φ) − f(E+φ)]dx (g = 2, x = p/T_pl, E = √(x²+mt²)), with the difference written
        as sinh φ/(cosh E + cosh φ): the naive subtraction cancels catastrophically at φ ~ 1e-9.
        E > 300 is zeroed (BURST's cutoff).
        """
        x, w = self.x_qed, self.w_qed
        E = jnp.sqrt(x**2 + mt**2)
        in_range = E < _FD_CUTOFF
        cE = jnp.cosh(jnp.where(in_range, E, 0.0))          # finite (cosh(0)=1 in masked bins)
        diff = jnp.sinh(phi) / (cE + jnp.cosh(phi))         # f(E−φ)−f(E+φ), cancellation-free
        diff = jnp.where(in_range, diff, 0.0)               # BURST's E > 300 cutoff
        return jnp.sum(w * x**2 * diff) / PI**2

    def _dnet_dphi_over_T3(self, phi, mt):
        """∂/∂φ of net_electron_number_over_T3, for Newton.

        d/dφ [sinh φ/(cosh E + cosh φ)] = (cosh E cosh φ + 1)/(cosh E + cosh φ)² ≥ 0, so the net
        number is monotone in φ.
        """
        x, w = self.x_qed, self.w_qed
        E = jnp.sqrt(x**2 + mt**2)
        in_range = E < _FD_CUTOFF
        cE = jnp.cosh(jnp.where(in_range, E, 0.0))
        cphi = jnp.cosh(phi)
        integ = x**2 * (cE * cphi + 1.0) / (cE + cphi) ** 2
        return jnp.sum(w * jnp.where(in_range, integ, 0.0)) / PI**2

    def solve_phie(self, tplmev, tcmpl, eta0, q_baryon=0.5, m_e=M_ELECTRON, n_newton=6):
        """Charge-neutral electron degeneracy φ_e = μ_e/T_pl at T_pl = tplmev [MeV], tcmpl = T_cm/T_pl.

        Solves (n_{e⁻} − n_{e⁺})/T_pl³ = q_baryon·nb_over_Tcm3(eta0)·tcmpl³ with ``n_newton`` unrolled
        Newton steps (6 is ample) from an asinh seed; φ_e ≥ 0 for q_baryon > 0.  Valid while
        T_pl > m_e/300 ≈ 1.7 keV (from T_init = 10 MeV, ln a ≲ 8.8): below it BURST's E/T > 300
        occupation cutoff removes every quadrature node and the result is NaN.

        Parameters
        ----------
        eta0 : float
            Present baryon-to-photon ratio (LINX ``const.eta0`` ≈ 6.1e-10).
        q_baryon : float
            Net charge per baryon Σ Z_i Y_i.  The default 0.5 is the high-T n/p ≈ 1 value; it drifts
            (to ~0.87) only below where φ_e affects N_eff.
        """
        mt = m_e / tplmev
        target = q_baryon * nb_over_Tcm3(eta0) * tcmpl**3        # (n_{e⁻}−n_{e⁺})/T_pl³ required

        # seed: net ≈ slope(0)·sinh φ for φ ≪ 1
        slope0 = self._dnet_dphi_over_T3(0.0, mt)
        phi = jnp.arcsinh(target / slope0)

        for _ in range(n_newton):
            f = self.net_electron_number_over_T3(phi, mt) - target
            fp = self._dnet_dphi_over_T3(phi, mt)
            phi = phi - f / fp
        return jnp.where(slope0 > 0.0, phi, jnp.nan)   # NaN past the validity limit: fail loudly

    def phie_linear(self, tplmev, tcmpl, eta0, q_baryon=0.5, m_e=M_ELECTRON):
        """BURST's linearized charge-neutrality φ_e = target/(∂net/∂φ at φ = 0), for validation.

        Equals solve_phie to machine precision at high T (φ_e ~ 1e-9).
        """
        mt = m_e / tplmev
        target = q_baryon * nb_over_Tcm3(eta0) * tcmpl**3
        return target / self._dnet_dphi_over_T3(0.0, mt)

    # ---- finite-T QED mass renormalization (δm_e², δm_γ²) ----------------------------------------

    def get_eg_mass_renorm(self, tplmev, phie, m_e=M_ELECTRON):
        """Momentum-independent thermal QED mass corrections (BURST's get_eg_mass_renorm).

        Returns (dmeeps2, dmedteps, dmedt2, dmgeps2, dmgdteps, dmgdt2), BURST's out-args in order:
        δm_e²/T_pl² and its first and second T-derivative terms, then the same for δm_γ²
        (dimensionless).  ``phie`` = μ_e/T_pl; ``m_e`` [MeV] may be traced.
        """
        x, dx = self.x_qed, self.w_qed

        a = ALPHA_FS
        mt = m_e / tplmev                      # m_e / T_pl
        E = jnp.sqrt(x**2 + mt**2)             # E / T_pl

        occ1 = _fd_equil(E, phie)             # e- (etadeg = +phie)
        occ2 = _fd_equil(E, -phie)            # e+ (etadeg = -phie)
        occ_sum = occ1 + occ2

        intval = jnp.sum(dx * x**2 / E * occ_sum)
        intvaldt = jnp.sum(dx * (2.0 * x**2 + mt**2) / E * occ_sum)
        # 2nd-derivative integrand mt²·exp(E)·(e^{−φ}occ1² + e^{φ}occ2²): mask the exp argument too, so
        # the cutoff tail stays 0 instead of inf·0 = NaN once exp(E) overflows (E ≳ 709).
        in_range = E <= _FD_CUTOFF
        exp_E = jnp.exp(jnp.where(in_range, E, 0.0))
        intvaldt2 = mt**2 * jnp.sum(
            jnp.where(in_range,
                      dx * exp_E * (jnp.exp(-phie) * occ1**2 + jnp.exp(phie) * occ2**2), 0.0)
        )

        two_a_over_pi = 2.0 * a / PI
        four_a_over_pi = 4.0 * a / PI

        dmeeps2 = 2.0 * PI * a / 3.0 + two_a_over_pi * intval
        dmedteps = 4.0 * PI * a / 3.0 + two_a_over_pi * intvaldt
        dmedt2 = 4.0 * PI * a / 3.0 + two_a_over_pi * (intvaldt + intvaldt2)

        dmgeps2 = four_a_over_pi * intval
        dmgdteps = four_a_over_pi * intvaldt
        dmgdt2 = four_a_over_pi * (intvaldt + intvaldt2)

        return dmeeps2, dmedteps, dmedt2, dmgeps2, dmgdteps, dmgdt2

    def delta_me2(self, tplmev, phie=0.0, m_e=M_ELECTRON):
        """Electron thermal mass² δm_e² [MeV²] = dmeeps2·T_pl²."""
        return self.get_eg_mass_renorm(tplmev, phie, m_e=m_e)[0] * tplmev**2

    def delta_mg2(self, tplmev, phie=0.0, m_e=M_ELECTRON):
        """Photon thermal mass² δm_γ² [MeV²] = dmgeps2·T_pl²."""
        return self.get_eg_mass_renorm(tplmev, phie, m_e=m_e)[3] * tplmev**2

    def m_e_eff(self, tplmev, phie=0.0, m_e=M_ELECTRON):
        """Thermal effective electron mass √(m_e² + δm_e²) [MeV].

        Not used by the solvers: the EoS and the lepton kernels take the bare mass, and finite-T QED
        enters the EoS additively.
        """
        return jnp.sqrt(m_e**2 + self.delta_me2(tplmev, phie, m_e=m_e))

    # ---- momentum-dependent electron self-energy (BURST's second thermo loop) --------------------

    def get_e_mass_pdep(self, tplmev, phie, peps, m_e=M_ELECTRON):
        """Momentum-dependent correction to (m_e/T_pl)² and its derivatives at peps = p/T_pl.

        Port of BURST's get_e_mass_pdep: per outer momentum peps the inner integral over y = k/T_pl
        uses the ``N_lg`` Legendre rule reflected about the integrable 1/(peps − y) singularity to
        cover [0, 2·peps] and the ``N_lag`` Laguerre rule shifted onto [2·peps, ∞) (BURST's
        principal-value construction).  ``peps`` is a scalar or (K,).  Returns (dmep1..dmep5) shaped
        like ``peps``: δm_e²(p)/T², d/dT, d/dp, d²/dT², d²/dpdT (BURST's order), zero at peps = 0.
        """
        t_lg, w_lg_c = self.t_lg, self.w_lg
        t_lag, w_lag_c = self.t_lag, self.w_lag

        peps = jnp.asarray(peps)
        scalar_in = peps.ndim == 0
        p = jnp.atleast_1d(peps)[:, None]      # (K,1) outer momenta

        mt = m_e / tplmev

        # Inner abscissas/weights, broadcast over outer nodes -> (K, 2*n_lg + n_lag).
        y_lg1 = 0.5 * p * (1.0 + t_lg[None, :])         # Legendre [0, peps]      (j=1)
        y_lg2 = 2.0 * p - y_lg1                         # reflected [peps, 2peps] (j=2, same weights)
        w_lg = 0.5 * p * w_lg_c[None, :]
        y_lag = 2.0 * p + t_lag[None, :]                # shifted Laguerre [2peps, inf)
        w_lag = jnp.broadcast_to(w_lag_c[None, :], y_lag.shape)

        y = jnp.concatenate([y_lg1, y_lg2, y_lag], axis=1)
        dy = jnp.concatenate([w_lg, w_lg, w_lag], axis=1)

        E = jnp.sqrt(y**2 + mt**2)
        occ1 = _fd_equil(E, phie)
        occ2 = _fd_equil(E, -phie)
        osum = occ1 + occ2
        # exp(E)·(e^{−φ}occ1² + e^{φ}occ2²), exp argument masked as in get_eg_mass_renorm
        in_range = E <= _FD_CUTOFF
        exp_E = jnp.exp(jnp.where(in_range, E, 0.0))
        osq = jnp.where(in_range,
                        exp_E * (jnp.exp(-phie) * occ1**2 + jnp.exp(phie) * occ2**2), 0.0)

        # 1/(peps ± y) factors: no node sits on y = peps, and the reflected pairs cancel the pole.
        d = p - y
        ip = 1.0 / (p + y)
        im = jnp.where(d != 0.0, 1.0 / jnp.where(d != 0.0, d, 1.0), 0.0)
        lg = jnp.log((p + y) / jnp.abs(jnp.where(d != 0.0, d, 1.0)))
        lg = jnp.where(d != 0.0, lg, 0.0)

        y_in = y < 300.0                                 # BURST's y < 300 guard (intval4/5 only)

        intval = jnp.sum(dy * y / E * lg * osum, axis=1)
        intval2 = jnp.sum(dy * E * (ip + im) * osum, axis=1)
        intval3 = jnp.sum(dy * y / E * (ip - im) * osum, axis=1)
        intval4 = jnp.sum(jnp.where(y_in, dy * (y**2 + mt**2) * (ip + im) * osq, 0.0), axis=1)
        intval5 = jnp.sum(jnp.where(y_in, dy * y * (ip - im) * osq, 0.0), axis=1)

        pref = -(mt**2) * ALPHA_FS / PI
        nz = jnp.atleast_1d(peps) != 0.0                 # peps==0 -> dmep = 0
        dmep1 = jnp.where(nz, pref * intval, 0.0)
        dmep2 = jnp.where(nz, pref * (intval + intval2), 0.0)
        dmep3 = jnp.where(nz, pref * intval3, 0.0)
        dmep4 = jnp.where(nz, pref * intval4, 0.0)
        dmep5 = jnp.where(nz, pref * intval5, 0.0)

        if scalar_in:
            return dmep1[0], dmep2[0], dmep3[0], dmep4[0], dmep5[0]
        return dmep1, dmep2, dmep3, dmep4, dmep5

    # ---- finite-T QED corrections to the e±/γ EoS ------------------------------------------------

    def qed_eos_corrections(self, tplmev, phie=0.0, m_e=M_ELECTRON):
        """Momentum-independent O(α) QED corrections to the e±/γ EoS (BURST's glag terms).

        Returns (d_rho_gamma, d_P_gamma, d_drho_gamma_dT, d_rho_e, d_P_e, d_drho_e_dT): δρ, δP [MeV⁴]
        and δ(dρ/dT) [MeV³], photon then e±, additive to the bare-mass free-gas EoS.  Arguments as
        get_eg_mass_renorm, whose self-energies are evaluated on the same rule.
        """
        x, dx = self.x_qed, self.w_qed

        dmeeps2, dmedteps, dmedt2, dmgeps2, dmgdteps, dmgdt2 = self.get_eg_mass_renorm(
            tplmev, phie, m_e=m_e
        )

        mt = m_e / tplmev                      # bare m_e / T_pl
        exm = jnp.sqrt(x**2 + mt**2)           # E / T_pl
        occ_be = _be(x)                        # photon
        occ1 = _fd_equil(exm, phie)                  # e- (etadeg = +phie)
        occ2 = _fd_equil(exm, -phie)                 # e+ (etadeg = -phie)
        occ_e = occ1 + occ2

        pref4 = tplmev**4 / PI**2              # ρ, P normalization [MeV⁴] (BURST's mev4tocgs dropped)
        pref3 = tplmev**3 / PI**2             # dρ/dT normalization [MeV³]

        # --- photon correction pieces ---
        d_rho_gamma = pref4 * jnp.sum(-dx / 4.0 * x * (dmgeps2 + dmgdteps) * occ_be)
        d_P_gamma = pref4 * jnp.sum(-dx / 4.0 * x * dmgeps2 * occ_be)
        d_drho_gamma_dT = pref3 * jnp.sum(
            -dx / 4.0 * x * (2.0 * dmgeps2 + 4.0 * dmgdteps + dmgdt2) * occ_be
        )

        # --- electron+positron correction pieces ---
        d_rho_e = pref4 * jnp.sum(
            -dx / 4.0 * (exm * dmeeps2 + x**2 / exm * dmedteps) * occ_e
        )
        d_P_e = pref4 * jnp.sum(-dx / 4.0 * x**2 / exm * dmeeps2 * occ_e)
        # exp(exm)·occ², exp argument masked against inf·0 = NaN (exm > 709 below T_pl ≈ 7.2e-4 MeV)
        in_range = exm <= _FD_CUTOFF
        exp_exm = jnp.exp(jnp.where(in_range, exm, 0.0))
        occ_sq = jnp.where(in_range,
                           exp_exm * (jnp.exp(-phie) * occ1**2 + jnp.exp(phie) * occ2**2), 0.0)
        d_drho_e_dT = pref3 * jnp.sum(
            -dx / 4.0 * (2.0 * (2.0 * x**2 + mt**2) / exm * dmedteps + x**2 / exm * dmedt2) * occ_e
            - dx / 4.0 * exm**2 * dmeeps2 * occ_sq
        )

        return (d_rho_gamma, d_P_gamma, d_drho_gamma_dT, d_rho_e, d_P_e, d_drho_e_dT)

    def qed_eos_pdep_corrections(self, tplmev, phie=0.0, m_e=M_ELECTRON):
        """Momentum-dependent e± QED EoS corrections (BURST's second thermo loop).

        Evaluates get_e_mass_pdep at every node of the ``N_qed`` rule over x = p/T_pl and
        accumulates its dmep terms into (d_rho_e, d_P_e [MeV⁴], d_drho_e_dT [MeV³]), additive to
        qed_eos_corrections' e± terms.  BURST's dρ_e/dφ_e term (glag6) is not ported; it is
        negligible at φ_e ~ 1e-10.
        """
        x, dx = self.x_qed, self.w_qed

        mt = m_e / tplmev
        exm = jnp.sqrt(x**2 + mt**2)
        occ1 = _fd_equil(exm, phie)
        occ2 = _fd_equil(exm, -phie)
        occ_e = occ1 + occ2
        # exp(exm)·occ², exp argument masked as in qed_eos_corrections
        in_range = exm <= _FD_CUTOFF
        exp_exm = jnp.exp(jnp.where(in_range, exm, 0.0))
        occ_sq = jnp.where(in_range,
                           exp_exm * (jnp.exp(-phie) * occ1**2 + jnp.exp(phie) * occ2**2), 0.0)

        dmep1, dmep2, dmep3, dmep4, dmep5 = self.get_e_mass_pdep(tplmev, phie, x, m_e=m_e)

        pref4 = tplmev**4 / PI**2
        pref3 = tplmev**3 / PI**2

        d_rho_e = pref4 * jnp.sum(-dx / 4.0 * (x / exm * dmep2 + exm * dmep3) * occ_e)
        d_P_e = pref4 * jnp.sum(-dx / 4.0 * x / exm * dmep1 * occ_e)
        d_drho_e_dT = pref3 * jnp.sum(
            -dx / 4.0 * (2.0 * x / exm * dmep2 + x / exm * dmep4 + 2.0 * exm * dmep5) * occ_e
            - dx / 4.0 * exm**2 * dmep3 * occ_sq
        )
        return d_rho_e, d_P_e, d_drho_e_dT

    def qed_eos_e3_pressure(self, tplmev, m_e=M_ELECTRON):
        """NLO O(e³) finite-T QED interaction pressure δP_{e³} [MeV⁴], which the O(α) terms omit.

        Port of LINX's ``explicit_P3`` (Bennett et al., arXiv:1911.04504, eq. 4.24):

            δP_{e³} = (e³/12π⁴) T_pl⁴ J^{3/2},   J = Σ_i w_i (2x_i² + mt²)/E_i · 2/(e^{E_i} + 1),

        with e³ = (4πα)^{3/2}, x = p/T_pl, mt = m_e/T_pl, E = √(x² + mt²).  BURST's E > 300 occupation
        cutoff makes δP exactly 0, with zero T-derivatives, once every node is cut (T_pl < m_e/300); the
        autodiff derivatives of qed_eos_e3_deltas would otherwise be NaN below T_pl ≈ 7e-4 MeV.
        """
        x, dx = self.x_qed, self.w_qed
        mt = m_e / tplmev
        exm = jnp.sqrt(x**2 + mt**2)
        in_range = exm <= _FD_CUTOFF                          # exp argument masked against inf/inf
        fd = jnp.where(in_range, 2.0 / (jnp.exp(jnp.where(in_range, exm, 0.0)) + 1.0), 0.0)
        jint = jnp.sum(dx * (2.0 * x**2 + mt**2) / exm * fd)  # dimensionless J
        e3 = (4.0 * PI * ALPHA_FS) ** 1.5                     # e³
        # J^{3/2} at J = 0 has an infinite second derivative (inf·0 = NaN): evaluate it at J > 0 only
        pos = jint > 0.0
        jint_pos = jnp.where(pos, jint, 1.0)
        return jnp.where(pos, e3 / (12.0 * PI**4) * tplmev**4 * jint_pos**1.5, 0.0)

    def qed_eos_e3_deltas(self, tplmev, m_e=M_ELECTRON):
        """NLO O(e³) QED corrections (δρ, δP, δ(dρ/dT)) [MeV⁴, MeV⁴, MeV³], additive to the O(α) ones.

        δρ = T dP/dT − P and δ(dρ/dT) = T d²P/dT², with the T-derivatives by forward-mode autodiff of
        qed_eos_e3_pressure (so m_e may be traced).
        """

        def _P(T):
            return self.qed_eos_e3_pressure(T, m_e=m_e)

        def _dP(T):
            return jax.jvp(_P, (T,), (jnp.ones_like(T),))[1]

        Tp = jnp.asarray(tplmev)
        P = _P(Tp)
        dP = _dP(Tp)
        d2P = jax.jvp(_dP, (Tp,), (jnp.ones_like(Tp),))[1]
        drho = Tp * dP - P
        dC = Tp * d2P
        return drho, P, dC

    def qed_plasma_deltas(self, tplmev, phie=0.0, pdep=True, m_e=M_ELECTRON, nlo=False):
        """Photon + e± QED corrections (δρ, δP, δ(dρ/dT)) [MeV⁴, MeV⁴, MeV³], the RHS's one call.

        Add to rho_plasma, pressure_plasma and drho_plasma_dT.

        Parameters
        ----------
        pdep : bool
            Include the momentum-dependent terms (qed_eos_pdep_corrections), as BURST does.
        nlo : bool
            Add the NLO O(e³) term (qed_eos_e3_deltas).  False reproduces BURST (O(α) only); the
            solvers default to True, the order of the N_eff = 3.044 literature.
        """
        d = self.qed_eos_corrections(tplmev, phie, m_e=m_e)
        rho, P, drho = d[0] + d[3], d[1] + d[4], d[2] + d[5]
        if pdep:
            dp = self.qed_eos_pdep_corrections(tplmev, phie, m_e=m_e)
            rho, P, drho = rho + dp[0], P + dp[1], drho + dp[2]
        if nlo:
            e3 = self.qed_eos_e3_deltas(tplmev, m_e=m_e)
            rho, P, drho = rho + e3[0], P + e3[1], drho + e3[2]
        return rho, P, drho

    def rho_plasma_qed_delta(self, tplmev, phie=0.0, pdep=True, m_e=M_ELECTRON, nlo=False):
        """QED δρ_γ + δρ_e [MeV⁴], to add to rho_plasma; O(α), plus O(e³) with nlo=True."""
        return self.qed_plasma_deltas(tplmev, phie, pdep, m_e=m_e, nlo=nlo)[0]

    def pressure_plasma_qed_delta(self, tplmev, phie=0.0, pdep=True, m_e=M_ELECTRON, nlo=False):
        """QED δP_γ + δP_e [MeV⁴], to add to pressure_plasma; O(α), plus O(e³) with nlo=True."""
        return self.qed_plasma_deltas(tplmev, phie, pdep, m_e=m_e, nlo=nlo)[1]

    def drho_plasma_dT_qed_delta(self, tplmev, phie=0.0, pdep=True, m_e=M_ELECTRON, nlo=False):
        """QED δ(dρ_γ/dT) + δ(dρ_e/dT) [MeV³], to add to drho_plasma_dT; O(α), + O(e³) if nlo."""
        return self.qed_plasma_deltas(tplmev, phie, pdep, m_e=m_e, nlo=nlo)[2]

    # ---- the decoupling background clock ---------------------------------------------------------

    def _rhs(self, lna, T, args):
        """dT_pl/dln a = −3(ρ_pl + P_pl)/(dρ_pl/dT) at φ = 0: plasma entropy conservation."""
        rho = self.rho_plasma(T, 0.0)
        P = self.pressure_plasma(T, 0.0)
        drhodT = self.drho_plasma_dT(T)
        return -3.0 * (rho + P) / drhodT

    def evolve_background(self, T_init=100.0, lna_max=10.0, n_save=256, rtol=1e-10, atol=1e-12):
        """Instantaneous-decoupling reference: T_pl over ln a ∈ [0, lna_max] (free gas, no ν heating).

        T_init [MeV] defaults to 100 so e± start fully relativistic; lna_max = 10 ends at T_ν ≈ 5 keV.
        Returns a dict of lna, T_pl, T_nu = T_init·e^{−ln a}, ratio = T_pl/T_nu (each ``n_save`` long),
        ratio_final and neff_final.
        """
        term = diffrax.ODETerm(self._rhs)
        saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, lna_max, n_save))
        controller = diffrax.PIDController(rtol=rtol, atol=atol)
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=0.0,
            t1=lna_max,
            dt0=1e-3,
            y0=jnp.asarray(T_init, dtype=jnp.float64),
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=100_000,
        )
        lna = sol.ts
        T_pl = sol.ys
        T_nu = T_init * jnp.exp(-lna)
        return {
            "lna": lna,
            "T_pl": T_pl,
            "T_nu": T_nu,
            "ratio": T_pl / T_nu,
            "neff_final": neff(T_nu[-1], T_pl[-1]),
            "ratio_final": T_pl[-1] / T_nu[-1],
        }

    def comoving_plasma_entropy(self, T_pl, lna):
        """Comoving plasma entropy s_pl·a³ = (ρ_pl + P_pl)/T_pl·e^{3 ln a} at φ = 0.

        T_pl, lna: scalars or a trajectory.  Constant along evolve_background's solution.
        """
        rho = jax.vmap(lambda t: self.rho_plasma(t, 0.0))(jnp.atleast_1d(T_pl))
        P = jax.vmap(lambda t: self.pressure_plasma(t, 0.0))(jnp.atleast_1d(T_pl))
        s = (rho + P) / jnp.atleast_1d(T_pl)
        return jnp.reshape(s * jnp.exp(3.0 * jnp.atleast_1d(lna)), jnp.shape(T_pl))
