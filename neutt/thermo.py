# import jax, neutt; from neutt.thermo import Thermo, ETA0_DEFAULT
# th = Thermo(); f = jax.jit(lambda t: th.qed_plasma_deltas(t, th.solve_phie(t, 1.02, ETA0_DEFAULT), nlo=True))
# with jax.log_compiles(True): print(f(1.0)); print(f(0.9))

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from .constants import ALPHA_FS, HBAR, M_ELECTRON, MPL, ZETA3, _FD_CUTOFF, _EXP_MAX, ETA0_DEFAULT, N_GAMMA, _N_NU_DEFAULT, T0CMB
from .grid import gauss_laguerre_plain, legendre_canonical


# ==== closed forms ===============================================================================

def rho_gamma(T):
    """
    Photon energy density.

    Parameters
    ----------
    T : float or array
        Photon temperature in MeV.

    Returns
    -------
    float or array
        Units of MeV^4.
    """
    return jnp.pi**2 / 15.0 * T**4


def pressure_gamma(T):
    """
    Photon pressure, rho_gamma/3.

    Parameters
    ----------
    T : float or array
        Photon temperature in MeV.

    Returns
    -------
    float or array
        Units of MeV^4.
    """
    return rho_gamma(T) / 3.0


def nb_over_Tcm3(eta0, ratio_today):
    """
    Comoving baryon number density n_b/T_cm^3.

    Set by today's baryon-to-photon ratio and the ratio of photon to comoving
    temperature after e+e- annihilation; constant throughout the evolution.

    Parameters
    ----------
    eta0 : float
        Present baryon-to-photon ratio.
    ratio_today : float
        T_gamma/T_cm after e+e- annihilation.

    Returns
    -------
    float
        Dimensionless.
    """
    n_b0 = eta0 * N_GAMMA * T0CMB**3
    T_cm0 = T0CMB / ratio_today
    return n_b0 / T_cm0**3


def rho_nu(T_nu, n_nu=_N_NU_DEFAULT):
    """
    Energy density of massless thermal neutrinos.

    Parameters
    ----------
    T_nu : float or array
        Neutrino temperature in MeV.
    n_nu : float, optional
        Number of neutrino species, each with its antineutrino. Default is 3.

    Returns
    -------
    float or array
        Units of MeV^4.
    """
    return n_nu * 7.0 / 8.0 * jnp.pi**2 / 15.0 * T_nu**4


def hubble(rho_tot):
    """
    Hubble rate.

    Parameters
    ----------
    rho_tot : float or array
        Total energy density in MeV^4.

    Returns
    -------
    float or array
        Units of s^-1.
    """
    return jnp.sqrt(8.0 * jnp.pi * rho_tot / (3.0 * MPL**2)) / HBAR


def neff(T_nu, T_gamma, n_nu=_N_NU_DEFAULT):
    """
    Effective number of neutrino species for thermal neutrinos at temperature T_nu.

    Parameters
    ----------
    T_nu : float or array
        Neutrino temperature in MeV.
    T_gamma : float or array
        Photon temperature in MeV.
    n_nu : float, optional
        Number of neutrino species, each with its antineutrino. Default is 3.

    Returns
    -------
    float or array
    """
    return (8.0 / 7.0) * (11.0 / 4.0) ** (4.0 / 3.0) * rho_nu(T_nu, n_nu) / rho_gamma(T_gamma)


# ==== occupations ================================================================================

def _fd(arg, phi):
    """
    Fermi-Dirac occupation 1/(exp(arg - phi) + 1).

    Zero where the exponential would overflow, with the exponent masked there so that
    autodiff tangents stay finite.

    Parameters
    ----------
    arg : float or array
        Energy over temperature E/T.
    phi : float
        Degeneracy mu/T.

    Returns
    -------
    float or array
        Dimensionless.
    """
    x = arg - phi
    ok = ~(x > _EXP_MAX)
    return jnp.where(ok, 1.0 / (jnp.exp(jnp.where(ok, x, 0.0)) + 1.0), 0.0)


def _fd_equil(E, phie):
    """
    Fermi-Dirac occupation 1/(exp(E - phie) + 1) with the occupation cutoff.

    Zero for E above the cutoff, with the exponent masked there so that autodiff tangents
    stay finite.

    Parameters
    ----------
    E : float or array
        Energy over temperature E/T.
    phie : float
        Degeneracy mu/T.

    Returns
    -------
    float or array
        Dimensionless.
    """
    keep = E <= _FD_CUTOFF
    f = 1.0 / (jnp.exp(jnp.where(keep, E, 0.0) - phie) + 1.0)   # masked: finite AD tangents
    return jnp.where(keep, f, 0.0)


def _be(x):
    """
    Bose-Einstein occupation 1/(exp(x) - 1) with the occupation cutoff.

    Zero for x <= 0 and for x above the cutoff.

    Parameters
    ----------
    x : float or array
        Energy over temperature E/T.

    Returns
    -------
    float or array
        Dimensionless.
    """
    f = 1.0 / (jnp.exp(x) - 1.0)
    return jnp.where((x > 0.0) & (x <= _FD_CUTOFF), f, 0.0)


# ==== the plasma model ===========================================================================

# Standard-Model T_gamma/T_cm after e+e- annihilation, from this code's reference solve.
RATIO_TODAY_SM = 1.3979608


class Thermo(eqx.Module):
    """
    Thermodynamics of the photon and e+e- plasma on fixed quadrature rules.

    The Gauss rules used by the energy-density, degeneracy and QED integrals are built
    once at construction and shared by every method.

    Parameters
    ----------
    N_qed : int, optional
        Gauss-Laguerre nodes for the electron degeneracy and the QED corrections.
        Default is 80.
    N_free : int, optional
        Gauss-Laguerre nodes for the free-gas equation of state. Default is 100.
    N_lg : int, optional
        Gauss-Legendre nodes of the inner integral of the momentum-dependent
        self-energy. Default is 50.
    N_lag : int, optional
        Gauss-Laguerre nodes of the inner integral of the momentum-dependent
        self-energy. Default is 100.
    ratio_today : float, optional
        T_gamma/T_cm after e+e- annihilation, which converts eta0 to the comoving
        baryon density. Default is the Standard-Model value.

    Notes
    -----
    The free gas uses the bare electron mass. Finite-temperature QED enters as
    additive corrections to rho, P and drho/dT (``qed_plasma_deltas``) built from the
    thermal self-energies; folding the thermal mass into the free gas as well would
    double count.
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
    ratio_today: float

    def __init__(self, N_qed=80, N_free=100, N_lg=50, N_lag=100, ratio_today=RATIO_TODAY_SM):
        self.N_qed, self.N_free, self.N_lg, self.N_lag = int(N_qed), int(N_free), int(N_lg), int(N_lag)
        self.x_qed, self.w_qed = gauss_laguerre_plain(self.N_qed)
        self.x_free, self.w_free = gauss_laguerre_plain(self.N_free)
        self.t_lg, self.w_lg = legendre_canonical(self.N_lg)
        self.t_lag, self.w_lag = gauss_laguerre_plain(self.N_lag)
        self.ratio_today = ratio_today

    def nb_tcm3(self, eta0):
        """
        Comoving baryon number density n_b/T_cm^3 at this model's ``ratio_today``.

        Parameters
        ----------
        eta0 : float
            Present baryon-to-photon ratio.

        Returns
        -------
        float
            Dimensionless.
        """
        return nb_over_Tcm3(eta0, self.ratio_today)

    # ---- free-gas e± and γ ----------------------------------------------------------------------

    def rho_electron(self, T, phi=0.0, m=M_ELECTRON):
        """
        Energy density of electrons and positrons.

        Parameters
        ----------
        T : float
            Plasma temperature in MeV.
        phi : float, optional
            Electron degeneracy mu_e/T. Default is 0.
        m : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        float
            Units of MeV^4.
        """
        x, w = self.x_free, self.w_free
        z = m / T
        e = jnp.sqrt(x**2 + z**2)               # E/T
        occ = _fd(e, phi) + _fd(e, -phi)        # electrons + positrons
        return T**4 / jnp.pi**2 * jnp.sum(w * x**2 * e * occ)

    def pressure_electron(self, T, phi=0.0, m=M_ELECTRON):
        """
        Pressure of electrons and positrons.

        Parameters
        ----------
        T : float
            Plasma temperature in MeV.
        phi : float, optional
            Electron degeneracy mu_e/T. Default is 0.
        m : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        float
            Units of MeV^4.
        """
        x, w = self.x_free, self.w_free
        z = m / T
        e = jnp.sqrt(x**2 + z**2)
        occ = _fd(e, phi) + _fd(e, -phi)
        return T**4 / (3.0 * jnp.pi**2) * jnp.sum(w * x**4 / e * occ)

    def rho_plasma(self, T, phi=0.0, m=M_ELECTRON):
        """
        Energy density of the photon and e+e- gas, without QED corrections.

        Parameters
        ----------
        T : float
            Plasma temperature in MeV.
        phi : float, optional
            Electron degeneracy mu_e/T. Default is 0.
        m : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        float
            Units of MeV^4.
        """
        return rho_gamma(T) + self.rho_electron(T, phi, m)

    def pressure_plasma(self, T, phi=0.0, m=M_ELECTRON):
        """
        Pressure of the photon and e+e- gas, without QED corrections.

        Parameters
        ----------
        T : float
            Plasma temperature in MeV.
        phi : float, optional
            Electron degeneracy mu_e/T. Default is 0.
        m : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        float
            Units of MeV^4.
        """
        return pressure_gamma(T) + self.pressure_electron(T, phi, m)

    def drho_plasma_dT(self, T, m=M_ELECTRON):
        """
        Heat capacity drho/dT of the photon and e+e- gas at zero degeneracy, without QED
        corrections.

        Parameters
        ----------
        T : float
            Plasma temperature in MeV.
        m : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        float
            Units of MeV^3.
        """
        x, w = self.x_free, self.w_free
        z = m / T
        e = jnp.sqrt(x**2 + z**2)
        f = _fd(e, 0.0)
        drho_e = 2.0 * T**3 / jnp.pi**2 * jnp.sum(w * x**2 * (x**2 + z**2) * f * (1.0 - f))
        return 4.0 * jnp.pi**2 * T**3 / 15.0 + drho_e

    def drho_plasma_dT_phi(self, T, phi, m=M_ELECTRON):
        """
        Heat capacity drho/dT of the photon and e+e- gas at fixed degeneracy, without QED
        corrections, by autodiff of ``rho_plasma``.

        Parameters
        ----------
        T : float
            Plasma temperature in MeV.
        phi : float
            Electron degeneracy mu_e/T, held fixed in the derivative.
        m : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        float
            Units of MeV^3.
        """
        return jax.grad(lambda t: self.rho_plasma(t, phi, m))(T)

    # ---- charge-neutral electron degeneracy φ_e --------------------------------------------------

    def net_electron_number_over_T3(self, phi, mt):
        """
        Net electron number density (n_e- - n_e+)/T^3.

        The difference of occupations is written as sinh(phi)/(cosh(E) + cosh(phi)), which
        does not cancel at small phi.

        Parameters
        ----------
        phi : float
            Electron degeneracy mu_e/T.
        mt : float
            Electron mass over temperature m_e/T.

        Returns
        -------
        float
            Dimensionless.
        """
        x, w = self.x_qed, self.w_qed
        E = jnp.sqrt(x**2 + mt**2)
        in_range = E < _FD_CUTOFF
        cE = jnp.cosh(jnp.where(in_range, E, 0.0))          # finite (cosh(0)=1 in masked bins)
        diff = jnp.sinh(phi) / (cE + jnp.cosh(phi))         # f(E−φ)−f(E+φ), cancellation-free
        diff = jnp.where(in_range, diff, 0.0)               # occupation cutoff
        return jnp.sum(w * x**2 * diff) / jnp.pi**2

    def _dnet_dphi_over_T3(self, phi, mt):
        """
        Derivative of ``net_electron_number_over_T3`` with respect to phi.

        Positive for all phi, so the net electron number is monotone in phi and the Newton
        iteration of ``solve_phie`` is well posed.

        Parameters
        ----------
        phi : float
            Electron degeneracy mu_e/T.
        mt : float
            Electron mass over temperature m_e/T.

        Returns
        -------
        float
            Dimensionless.
        """
        x, w = self.x_qed, self.w_qed
        E = jnp.sqrt(x**2 + mt**2)
        in_range = E < _FD_CUTOFF
        cE = jnp.cosh(jnp.where(in_range, E, 0.0))
        cphi = jnp.cosh(phi)
        integ = x**2 * (cE * cphi + 1.0) / (cE + cphi) ** 2
        return jnp.sum(w * jnp.where(in_range, integ, 0.0)) / jnp.pi**2

    def solve_phie(self, tplmev, tcmpl, eta0, q_baryon=0.5, m_e=M_ELECTRON, n_newton=6):
        """
        Electron degeneracy phi_e = mu_e/T_pl from charge neutrality with the baryons.

        Solves n_e- - n_e+ = q_baryon n_b by Newton iteration from an asinh seed; derivatives
        are taken through the root by the implicit function theorem.

        Parameters
        ----------
        tplmev : float
            Plasma temperature in MeV.
        tcmpl : float
            Ratio of comoving to plasma temperature T_cm/T_pl.
        eta0 : float
            Present baryon-to-photon ratio.
        q_baryon : float, optional
            Net charge per baryon, sum_i Z_i Y_i. Default is 0.5.
        m_e : float, optional
            Electron mass in MeV. Default is the bare mass.
        n_newton : int, optional
            Number of unrolled Newton steps. Default is 6.

        Returns
        -------
        float
            Dimensionless. NaN below T_pl = m_e/_FD_CUTOFF, where the occupation cutoff
            removes every quadrature node.
        """
        mt = m_e / tplmev
        target = q_baryon * self.nb_tcm3(eta0) * tcmpl**3        # (n_{e⁻}−n_{e⁺})/T_pl³ required

        def residual(phi):
            return self.net_electron_number_over_T3(phi, mt) - target

        def newton(f, phi):
            for _ in range(n_newton):
                phi = phi - f(phi) / self._dnet_dphi_over_T3(phi, mt)
            return phi

        # seed: net ≈ slope(0)·sinh φ for φ ≪ 1; derivatives go through the root implicitly
        slope0 = self._dnet_dphi_over_T3(0.0, mt)
        phi0 = jnp.arcsinh(target / slope0)
        phi = jax.lax.custom_root(residual, phi0, newton, lambda g, y: y / g(1.0))
        return jnp.where(slope0 > 0.0, phi, jnp.nan)   # NaN past the validity limit: fail loudly

    # ---- finite-T QED mass renormalization (δm_e², δm_γ²) ----------------------------------------

    def get_eg_mass_renorm(self, tplmev, phie, m_e=M_ELECTRON):
        """
        Momentum-independent thermal self-energies of the electron and the photon.

        Parameters
        ----------
        tplmev : float
            Plasma temperature in MeV.
        phie : float
            Electron degeneracy mu_e/T_pl.
        m_e : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        tuple of float
            (dmeeps2, dmedteps, dmedt2, dmgeps2, dmgdteps, dmgdt2): delta m_e^2/T_pl^2 and
            its first- and second-derivative terms with respect to T_pl, then the same for
            delta m_gamma^2. Dimensionless.
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

        two_a_over_pi = 2.0 * a / jnp.pi
        four_a_over_pi = 4.0 * a / jnp.pi

        dmeeps2 = 2.0 * jnp.pi * a / 3.0 + two_a_over_pi * intval
        dmedteps = 4.0 * jnp.pi * a / 3.0 + two_a_over_pi * intvaldt
        dmedt2 = 4.0 * jnp.pi * a / 3.0 + two_a_over_pi * (intvaldt + intvaldt2)

        dmgeps2 = four_a_over_pi * intval
        dmgdteps = four_a_over_pi * intvaldt
        dmgdt2 = four_a_over_pi * (intvaldt + intvaldt2)

        return dmeeps2, dmedteps, dmedt2, dmgeps2, dmgdteps, dmgdt2

    def delta_me2(self, tplmev, phie=0.0, m_e=M_ELECTRON):
        """
        Electron thermal mass squared, delta m_e^2.

        Parameters
        ----------
        tplmev : float
            Plasma temperature in MeV.
        phie : float, optional
            Electron degeneracy mu_e/T_pl. Default is 0.
        m_e : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        float
            Units of MeV^2.
        """
        return self.get_eg_mass_renorm(tplmev, phie, m_e=m_e)[0] * tplmev**2

    def delta_mg2(self, tplmev, phie=0.0, m_e=M_ELECTRON):
        """
        Photon thermal mass squared, delta m_gamma^2.

        Parameters
        ----------
        tplmev : float
            Plasma temperature in MeV.
        phie : float, optional
            Electron degeneracy mu_e/T_pl. Default is 0.
        m_e : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        float
            Units of MeV^2.
        """
        return self.get_eg_mass_renorm(tplmev, phie, m_e=m_e)[3] * tplmev**2

    # ---- momentum-dependent electron self-energy -------------------------------------------------

    def get_e_mass_pdep(self, tplmev, phie, peps, m_e=M_ELECTRON):
        """
        Momentum-dependent thermal correction to the electron mass and its derivatives.

        The inner integral over the loop momentum is split at the integrable singularity
        at k = p: a Gauss-Legendre rule reflected about it covers [0, 2p] and a shifted
        Gauss-Laguerre rule covers [2p, infinity).

        Parameters
        ----------
        tplmev : float
            Plasma temperature in MeV.
        phie : float
            Electron degeneracy mu_e/T_pl.
        peps : float or array
            Electron momentum p/T_pl, a scalar or of shape (K,).
        m_e : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        tuple of float or array
            (dmep1, dmep2, dmep3, dmep4, dmep5), shaped like ``peps``: delta m_e^2(p)/T_pl^2 and
            its derivative terms with respect to T_pl, p, T_pl twice, and p and T_pl.
            Dimensionless; zero at peps = 0.
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

        y_in = y < 300.0                                 # occupation cutoff (intval4/5 only)

        intval = jnp.sum(dy * y / E * lg * osum, axis=1)
        intval2 = jnp.sum(dy * E * (ip + im) * osum, axis=1)
        intval3 = jnp.sum(dy * y / E * (ip - im) * osum, axis=1)
        intval4 = jnp.sum(jnp.where(y_in, dy * (y**2 + mt**2) * (ip + im) * osq, 0.0), axis=1)
        intval5 = jnp.sum(jnp.where(y_in, dy * y * (ip - im) * osq, 0.0), axis=1)

        pref = -(mt**2) * ALPHA_FS / jnp.pi
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
        """
        Momentum-independent O(alpha) QED corrections to the plasma equation of state.

        Parameters
        ----------
        tplmev : float
            Plasma temperature in MeV.
        phie : float, optional
            Electron degeneracy mu_e/T_pl. Default is 0.
        m_e : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        tuple of float
            (d_rho_gamma, d_P_gamma, d_drho_gamma_dT, d_rho_e, d_P_e, d_drho_e_dT): corrections
            to the energy density and pressure in MeV^4 and to drho/dT in MeV^3, for the
            photons and then for the e+e- gas, to be added to the free-gas values.
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

        pref4 = tplmev**4 / jnp.pi**2              # ρ, P normalization [MeV⁴]
        pref3 = tplmev**3 / jnp.pi**2             # dρ/dT normalization [MeV³]

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
        """
        Momentum-dependent O(alpha) QED corrections to the e+e- equation of state.

        Parameters
        ----------
        tplmev : float
            Plasma temperature in MeV.
        phie : float, optional
            Electron degeneracy mu_e/T_pl. Default is 0.
        m_e : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        tuple of float
            (d_rho_e, d_P_e, d_drho_e_dT): corrections to the e+e- energy density and
            pressure in MeV^4 and to drho/dT in MeV^3, to be added to those of
            ``qed_eos_corrections``.

        Notes
        -----
        The derivative of the correction with respect to phi_e is omitted; it is negligible
        for phi_e ~ 1e-9.
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

        pref4 = tplmev**4 / jnp.pi**2
        pref3 = tplmev**3 / jnp.pi**2

        d_rho_e = pref4 * jnp.sum(-dx / 4.0 * (x / exm * dmep2 + exm * dmep3) * occ_e)
        d_P_e = pref4 * jnp.sum(-dx / 4.0 * x / exm * dmep1 * occ_e)
        d_drho_e_dT = pref3 * jnp.sum(
            -dx / 4.0 * (2.0 * x / exm * dmep2 + x / exm * dmep4 + 2.0 * exm * dmep5) * occ_e
            - dx / 4.0 * exm**2 * dmep3 * occ_sq
        )
        return d_rho_e, d_P_e, d_drho_e_dT

    def qed_eos_e3_pressure(self, tplmev, m_e=M_ELECTRON):
        """
        Next-to-leading-order O(e^3) QED correction to the plasma pressure.

        Equation 4.24 of Bennett et al., arXiv:1911.04504, as implemented in LINX:
        delta P = (e^3/12 pi^4) T_pl^4 J^(3/2) with J the quadrature of
        (2x^2 + mt^2)/E * 2/(exp(E) + 1) over x = p/T_pl.

        Parameters
        ----------
        tplmev : float
            Plasma temperature in MeV.
        m_e : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        float
            Units of MeV^4. Exactly zero, with zero derivatives, once the occupation cutoff
            removes every quadrature node.
        """
        x, dx = self.x_qed, self.w_qed
        mt = m_e / tplmev
        exm = jnp.sqrt(x**2 + mt**2)
        in_range = exm <= _FD_CUTOFF                          # exp argument masked against inf/inf
        fd = jnp.where(in_range, 2.0 / (jnp.exp(jnp.where(in_range, exm, 0.0)) + 1.0), 0.0)
        jint = jnp.sum(dx * (2.0 * x**2 + mt**2) / exm * fd)  # dimensionless J
        e3 = (4.0 * jnp.pi * ALPHA_FS) ** 1.5                     # e³
        # J^{3/2} at J = 0 has an infinite second derivative (inf·0 = NaN): evaluate it at J > 0 only
        pos = jint > 0.0
        jint_pos = jnp.where(pos, jint, 1.0)
        return jnp.where(pos, e3 / (12.0 * jnp.pi**4) * tplmev**4 * jint_pos**1.5, 0.0)

    def qed_eos_e3_deltas(self, tplmev, m_e=M_ELECTRON):
        """
        Next-to-leading-order O(e^3) QED corrections to the plasma equation of state.

        delta rho = T dP/dT - P and delta(drho/dT) = T d^2P/dT^2, with the derivatives of
        ``qed_eos_e3_pressure`` by forward-mode autodiff.

        Parameters
        ----------
        tplmev : float
            Plasma temperature in MeV.
        m_e : float, optional
            Electron mass in MeV. Default is the bare mass.

        Returns
        -------
        tuple of float
            (delta_rho, delta_P, delta_drho_dT) in MeV^4, MeV^4 and MeV^3, to be added to
            the O(alpha) corrections.
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
        """
        Total QED corrections to the plasma energy density, pressure and heat capacity.

        Parameters
        ----------
        tplmev : float
            Plasma temperature in MeV.
        phie : float, optional
            Electron degeneracy mu_e/T_pl. Default is 0.
        pdep : bool, optional
            Include the momentum-dependent O(alpha) terms. Default is True.
        m_e : float, optional
            Electron mass in MeV. Default is the bare mass.
        nlo : bool, optional
            Include the O(e^3) term. Default is False.

        Returns
        -------
        tuple of float
            (delta_rho, delta_P, delta_drho_dT) in MeV^4, MeV^4 and MeV^3, to be added to
            ``rho_plasma``, ``pressure_plasma`` and ``drho_plasma_dT``.
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
