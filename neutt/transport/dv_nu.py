"""Dark-boson ↔ neutrino 1↔2 decay collision kernels.

Process X ↔ ν + ν̄ for a massive dark boson X: the dark vector dv/Z′ (scenario S1,
``spin_dof=3``) or the majoron J (scenario S3, ``spin_dof=1``). Matrix element:

    Σ|M|² = 2·gsq·m_X²   summed over all dof
    (S1: gsq = g_ν²  per flavor; S3 Dirac-pair convention: gsq = g_CGJPS²/2)

Two kernels, one per evolving species (per-dof occupations f):
  - :func:`x1_nu_decay_rates` → d f_X/dt:
        C_X(q) = (Σ|M|²/spin_dof) / (16π E q ħ) · ∫dE₂ F     [1/g_X average]
    Gauss-Legendre over the finite daughter-ν range; dilute limit reproduces the exact
    −Γ·(m/E)·f_X with Γ_vector = gsq·m/(24π), Γ_scalar = gsq·m/(8π).
  - :func:`nu1_x_decay_rates` → d f_ν/dt:
        C_ν(p) = Σ|M|² / (16π p² ħ) · ∫dE_X F                [ν: 1 physical helicity]
    Gauss-Laguerre over the semi-infinite parent-X tail.

Statistical factors (forfact/revfact below); F = forfact − revfact directionally per kernel.
Both return (net, frs) in s⁻¹ directly (a 1↔2 rate scales as gsq·m²/E, not G²T⁵, unlike the SM
kernels' per-tcmev⁵ convention; a gate-folding caller must divide by tcmev⁵ before summing with the
SM legs). Off-grid occupations are interpolated in log space; the X (boson) spectrum without the
fermionic clamp (BE may exceed 1). At a common temperature net → 0 (detailed balance, the primary
normalization-blind gate).
"""

from __future__ import annotations

import jax.numpy as jnp

from ..constants import PI, HBAR
from ..interp import interp_log_occ_dynamic
from ..grid import log_occupation
from ..grid import legendre_canonical, gauss_laguerre_plain

NLEG_DEFAULT = 20    # Gauss-Legendre nodes for the finite (x1_nu) outer integral
NLAG_DEFAULT = 20    # Gauss-Laguerre nodes for the semi-infinite (nu1_x) outer tail


def _safe_sqrt_diff(a2, b2):
    """sqrt(max(a2 - b2, 0)) with a finite gradient at the zero boundary (double-where)."""
    d = a2 - b2
    safe = jnp.where(d > 0.0, d, 1.0)
    return jnp.where(d > 0.0, jnp.sqrt(safe), 0.0)


def x1_nu_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, tratio, tcmev, gsq_diag,
                      spin_dof=3, nleg=NLEG_DEFAULT, off_grid_mode="default"):
    """d f_X/dt rate for X ↔ ν ν̄, per X dof [s⁻¹].

    Parameters
    ----------
    nu_grid, x_grid : EpsGrid
        ε-grids for neutrinos (T_cm units) and X (T_X units).
    f_nu : jnp.ndarray
        (3,2,Nnu) neutrino occupation [flavor, cp, bin].
    f_x : jnp.ndarray
        (Nx,) dark-boson occupation per dof (BE may exceed 1).
    meps, tratio, tcmev : float
        m_X/T_cm, T_cm/T_X, and T_cm [MeV].
    gsq_diag : jnp.ndarray
        (3,) per-flavor coupling-squared (Σ|M|² = 2·gsq·m_X²).
    spin_dof : int (static)
        X dof: 3 (vector dv/Z′) or 1 (scalar J).
    nleg : int
        Gauss-Legendre node count for the outer integral.

    Returns
    -------
    tuple of jnp.ndarray
        (net, frs), each (3, Nx) [flavor, X bin], s⁻¹. Sum over flavors for the total.
    """
    xc, wc = legendre_canonical(nleg)                       # canonical [-1,1]
    q1 = x_grid.abscissas                                   # (Nx,) T_X units
    e1 = jnp.sqrt(q1**2 + (meps * tratio) ** 2)             # E_X/T_X
    a = (e1 - q1) / (2.0 * tratio)                          # p2 lower (T_cm)
    b = (e1 + q1) / (2.0 * tratio)                          # p2 upper
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    p2 = mid[:, None] + half[:, None] * xc[None, :]         # (Nx, nleg)
    w = half[:, None] * wc[None, :]
    p3 = (e1 / tratio)[:, None] - p2                        # (Nx, nleg) partner nubar momentum

    logf = log_occupation(f_nu)                             # (3,2,Nnu)
    # Do not pass fd_cap_logC here: the decay deposits a genuinely super-FD injection into f_ν that
    # the QKE FD-envelope cap (f ≤ C·FD) would clip. This kernel stays at the BURST-faithful no-cap
    # default; the QKE pump the cap fixes lives in the SM lepton kernels, not here.
    f2 = jnp.exp(interp_log_occ_dynamic(logf[:, 0, :], nu_grid.abscissas, p2,
                                        off_grid_mode=off_grid_mode))   # nu  (3,Nx,nleg)
    f3 = jnp.exp(interp_log_occ_dynamic(logf[:, 1, :], nu_grid.abscissas, p3,
                                        off_grid_mode=off_grid_mode))   # nubar (3,Nx,nleg)
    fx = f_x[None, :, None]                                 # (1,Nx,1)

    forfact = (1.0 + fx) * f2 * f3                          # nu nubar -> X
    revfact = fx * (1.0 - f2) * (1.0 - f3)                  # X -> nu nubar
    kinemfact1 = w / (q1[:, None] * e1[:, None])            # (Nx,nleg)

    # C_X = (2 gsq m²/spin_dof)/(16π E q ħ)·T_cm Σw F  →  grid units below [s⁻¹]
    prefac = gsq_diag[:, None, None] * meps**2 * tcmev * tratio**2 / (
        8.0 * spin_dof * PI * HBAR)
    net = jnp.sum(prefac * (forfact - revfact) * kinemfact1[None], axis=-1)   # (3,Nx)
    frs = jnp.sum(prefac * (forfact + revfact) * kinemfact1[None], axis=-1)

    # zero out a degenerate eps=0 X bin (q1=0 -> 1/q1 blows up); offset>0 grids never hit this.
    good = (q1 > 0.0)[None, :]
    return jnp.where(good, net, 0.0), jnp.where(good, frs, 0.0)


def x1_nu_decay_affine(nu_grid, x_grid, f_nu, meps, tratio, tcmev, gsq_diag,
                       spin_dof=3, nleg=NLEG_DEFAULT, range_limit=False,
                       off_grid_mode="default", forward_only_tcm=0.0):
    """Per-X-bin affine split of :func:`x1_nu_decay_rates`: ``net_X = A_X − B_X·f_X`` [s⁻¹].

    The X rate is exactly linear in the per-bin f_X (constant over the daughter-ν nodes, so it
    factors out): A_X = Σ prefac·kinem·f₂f₃ (inverse-decay source ≥0), B_X = Σ prefac·kinem·
    (1−f₂−f₃) (decay loss). Feeds the exponential-midpoint head, which integrates the stiff
    −B_X·f_X exactly. Same setup as :func:`x1_nu_decay_rates` (no f_X argument); returns
    (A_X, B_X), each (3, Nx) [flavor, X bin].
    """
    xc, wc = legendre_canonical(nleg)
    q1 = x_grid.abscissas
    e1 = jnp.sqrt(q1**2 + (meps * tratio) ** 2)
    a = (e1 - q1) / (2.0 * tratio)
    b = (e1 + q1) / (2.0 * tratio)
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    p2 = mid[:, None] + half[:, None] * xc[None, :]
    w = half[:, None] * wc[None, :]
    p3 = (e1 / tratio)[:, None] - p2

    logf = log_occupation(f_nu)
    f2 = jnp.exp(interp_log_occ_dynamic(logf[:, 0, :], nu_grid.abscissas, p2,
                                        range_limit=range_limit,
                                        off_grid_mode=off_grid_mode))   # (3,Nx,nleg)
    f3 = jnp.exp(interp_log_occ_dynamic(logf[:, 1, :], nu_grid.abscissas, p3,
                                        range_limit=range_limit,
                                        off_grid_mode=off_grid_mode))
    kinemfact1 = w / (q1[:, None] * e1[:, None])
    prefac = gsq_diag[:, None, None] * meps**2 * tcmev * tratio**2 / (
        8.0 * spin_dof * PI * HBAR)
    a_x = jnp.sum(prefac * (f2 * f3) * kinemfact1[None], axis=-1)                 # (3,Nx)
    b_x = jnp.sum(prefac * (1.0 - f2 - f3) * kinemfact1[None], axis=-1)
    # Forward-decay-only gate: below forward_only_tcm drop the inverse decay νν̄→X (physically
    # ∝e^{−m/T}→0 at low T, and the source of the off-grid ill-conditioning that samples ν at
    # ε≈E_X/T_cm). a_x → 0 and b_x → the exact forward-decay Pauli loss K(1−f₂)(1−f₃) ≥0, so the X
    # only depletes (no inverse-decay runaway). forward_only_tcm=0.0 (default) ⇒ bit-identical.
    b_x_fwd = jnp.sum(prefac * ((1.0 - f2) * (1.0 - f3)) * kinemfact1[None], axis=-1)
    fwd = tcmev < forward_only_tcm
    a_x = jnp.where(fwd, 0.0, a_x)
    b_x = jnp.where(fwd, b_x_fwd, b_x)
    good = (q1 > 0.0)[None, :]
    return jnp.where(good, a_x, 0.0), jnp.where(good, b_x, 0.0)


def nu1_x_decay_rates(nu_grid, x_grid, f_nu, f_x, meps, tratio, tcmev, gsq_diag,
                      nlag=NLAG_DEFAULT, off_grid_mode="default"):
    """d f_ν/dt rate for X ↔ ν ν̄ [s⁻¹] (no spin_dof: the X inside is summed over pols).

    Parameters as :func:`x1_nu_decay_rates`, except ``nlag`` is the Gauss-Laguerre node count for
    the semi-infinite tail. The X is a boson, so its log-occupation is interpolated without the
    fermionic clamp (else super-unity BE occupations get zeroed).

    Returns
    -------
    tuple of jnp.ndarray
        (net, frs), each (3, 2, Nnu) [flavor, cp, ν bin], s⁻¹.
    """
    xl, wl = gauss_laguerre_plain(nlag)                     # plain Laguerre on [0, inf)
    p1 = nu_grid.abscissas                                  # (Nnu,) T_cm
    p1s = jnp.where(p1 > 0.0, p1, 1.0)                      # guard 1/p1 at eps=0
    a = (p1s + meps**2 / (4.0 * p1s)) * tratio              # e3 lower bound (T_X), Laguerre [a,inf)
    e3 = a[:, None] + xl[None, :]                           # (Nnu, nlag) T_X units
    w = wl[None, :]                                         # plain-Laguerre weights (shift-invariant)
    q3 = _safe_sqrt_diff(e3**2, (meps * tratio) ** 2)       # X momentum (T_X), >= 0
    p2 = e3 / tratio - p1[:, None]                          # (Nnu,nlag) partner nubar momentum (T_cm)

    # X is a boson: BE occupation exceeds 1 at small momentum, so interpolate without the
    # fermionic log f <= 0 clamp (clamp_high=False) -- else super-unity occupations get zeroed.
    logf_x = jnp.log(jnp.clip(f_x, 1e-300, None))           # (Nx,) floor only (BE has no upper bound)
    # Off-grid X read: the parent-X integral samples f_x above the X grid edge (non-relativistic
    # parent ⇒ log f_x ∝ −q²/(2mT)). off_grid_mode="floor" is the physically correct zero-above-edge
    # read; "default" (order-5 poly) overshoots up and can inject a spurious ε³-amplified high-ε ν tail.
    # The explicit path keeps f_x smooth, so the extrapolation is benign and the default is
    # bit-identical; the knob is a robustness opt-in for non-rel-parent runs. "default" ⇒ bit-identical.
    fx3 = jnp.exp(interp_log_occ_dynamic(logf_x, x_grid.abscissas, q3, clamp_high=False,
                                         off_grid_mode=off_grid_mode))

    logf = log_occupation(f_nu)                             # (3,2,Nnu)
    f2 = jnp.exp(interp_log_occ_dynamic(logf, nu_grid.abscissas, p2,
                                        off_grid_mode=off_grid_mode))             # (3,2,Nnu,nlag) at p2
    f2_partner = f2[:, ::-1, :, :]                          # cp reversal: partner nubar (nn = 1-n)
    f1 = f_nu[:, :, :, None]                                # (3,2,Nnu,1) particle-1 nu at the bin

    fx = fx3[None, None, :, :]                              # (1,1,Nnu,nlag)
    forfact = (1.0 - f1) * (1.0 - f2_partner) * fx          # X -> nu nubar (produces nu)
    revfact = f1 * f2_partner * (1.0 + fx)                  # nu nubar -> X (destroys nu)
    kinemfact1 = (w / p1s[:, None] ** 2)[None, None]        # (1,1,Nnu,nlag)

    # C_ν = (2 gsq m²)/(16π p² ħ)·∫dE_X F  →  grid units below [s⁻¹]
    prefac = gsq_diag[:, None, None, None] * meps**2 * tcmev / (8.0 * PI * tratio * HBAR)
    net = jnp.sum(prefac * (forfact - revfact) * kinemfact1, axis=-1)        # (3,2,Nnu)
    frs = jnp.sum(prefac * (forfact + revfact) * kinemfact1, axis=-1)

    good = (p1 > 0.0)[None, None, :]
    return jnp.where(good, net, 0.0), jnp.where(good, frs, 0.0)


def nu1_x_decay_affine(nu_grid, x_grid, f_nu, f_x, meps, tratio, tcmev, gsq_diag,
                       nlag=NLAG_DEFAULT, range_limit=False, off_grid_mode="default",
                       forward_only_tcm=0.0):
    """Per-ν-bin affine split of :func:`nu1_x_decay_rates`: ``net_ν = A_ν − B_ν·f1`` [s⁻¹].

    Linear in the local bin occupation f1=f_ν[fl,cp,bin] (the partner-ν f₂ₚ and parent-X f_X₃ are
    frozen sources): A_ν = Σ prefac·kinem·(1−f₂ₚ)·f_X₃ (the X→νν̄ injection ≥0), B_ν = Σ prefac·
    kinem·(f_X₃+f₂ₚ). The injection lives in A_ν (a source — ν have no fast dark loss), so the head's
    exp_step keeps ν second-order accurate while the X equation carries the stiffness. Same setup/args
    as :func:`nu1_x_decay_rates`; returns (A_ν, B_ν), each (3, 2, Nnu).
    """
    xl, wl = gauss_laguerre_plain(nlag)
    p1 = nu_grid.abscissas
    p1s = jnp.where(p1 > 0.0, p1, 1.0)
    a = (p1s + meps**2 / (4.0 * p1s)) * tratio
    e3 = a[:, None] + xl[None, :]
    w = wl[None, :]
    q3 = _safe_sqrt_diff(e3**2, (meps * tratio) ** 2)
    p2 = e3 / tratio - p1[:, None]

    logf_x = jnp.log(jnp.clip(f_x, 1e-300, None))
    # Off-grid X read: the parent-X integral samples f_x up to ~3-4× the X grid edge. The X is
    # non-relativistic (log f_x ∝ −q²/(2mT)), so its occupation off-grid is ≈0 and a ν daughter at
    # ε≫(m/2)/T is kinematically forbidden; off_grid_mode="default" (order-5 poly) overshoots those
    # reads up, injecting a spurious ε³-amplified high-ε ν tail. "floor" (zero above x_max) is the
    # physically correct read for the non-rel parent. Default "default" ⇒ bit-identical.
    fx3 = jnp.exp(interp_log_occ_dynamic(logf_x, x_grid.abscissas, q3, clamp_high=False,
                                         off_grid_mode=off_grid_mode))
    # (Removed the ad-hoc `fx3 = min(fx3, max(f_x))` boson-read fence — cap-refactor 2026-06-28.) It
    # guarded a NaN from the boson interp's UPWARD extrapolation when the per-bin exp_step distorts f_x
    # into a rising-edge tail. Two production defaults now make it redundant: (1) the off-grid read
    # defaults to "floor" (zero above ε_max — no extrapolation to blow up; physically correct for the
    # non-rel decaying parent), and (2) `monotone_x=True` clamps f_x monotone-decreasing (no rising
    # edge for the interp to amplify). It was already a no-op for a smooth/decreasing f_x (so dropping
    # it is bit-identical on every physical state, incl. test_dv_decay's affine-identity gate).
    logf = log_occupation(f_nu)
    f2 = jnp.exp(interp_log_occ_dynamic(logf, nu_grid.abscissas, p2, range_limit=range_limit,
                                        off_grid_mode=off_grid_mode))
    f2_partner = f2[:, ::-1, :, :]
    fx = fx3[None, None, :, :]
    kinemfact1 = (w / p1s[:, None] ** 2)[None, None]
    prefac = gsq_diag[:, None, None, None] * meps**2 * tcmev / (8.0 * PI * tratio * HBAR)
    a_nu = jnp.sum(prefac * ((1.0 - f2_partner) * fx) * kinemfact1, axis=-1)      # (3,2,Nnu)
    b_nu = jnp.sum(prefac * (fx + f2_partner) * kinemfact1, axis=-1)
    # Forward-decay-only gate (see x1_nu_decay_affine): below forward_only_tcm the ν loss is only the
    # forward Pauli factor (1−f₁) ⇒ b_ν → a_ν (the inverse-decay loss ∝ f₂ₚ is dropped), giving the
    # exact forward X→νν̄ injection. forward_only_tcm=0.0 (default) ⇒ bit-identical.
    fwd = tcmev < forward_only_tcm
    b_nu = jnp.where(fwd, a_nu, b_nu)
    good = (p1 > 0.0)[None, None, :]
    return jnp.where(good, a_nu, 0.0), jnp.where(good, b_nu, 0.0)


def dv1_nu_decay_rates(nu_grid, dv_grid, f_nu, f_dv, meps, tratio, tcmev, gsq_diag,
                       nleg=NLEG_DEFAULT):
    """Vector parent (dv/Z′, scenario S1): :func:`x1_nu_decay_rates` with spin_dof=3."""
    return x1_nu_decay_rates(nu_grid, dv_grid, f_nu, f_dv, meps, tratio, tcmev, gsq_diag,
                             spin_dof=3, nleg=nleg)


def j1_nu_decay_rates(nu_grid, j_grid, f_nu, f_j, meps, tratio, tcmev, gsq_diag,
                      nleg=NLEG_DEFAULT):
    """Scalar parent (majoron J, scenario S3): :func:`x1_nu_decay_rates` with spin_dof=1.

    ``gsq_diag`` is the Dirac-pair-convention coupling-squared = g_CGJPS²/2 per flavor.
    """
    return x1_nu_decay_rates(nu_grid, j_grid, f_nu, f_j, meps, tratio, tcmev, gsq_diag,
                             spin_dof=1, nleg=nleg)


# ν-side: identical for vector and scalar parents (the parent is summed over its dof).
nu1_dv_decay_rates = nu1_x_decay_rates
nu1_j_decay_rates = nu1_x_decay_rates
