"""The transport RHS: the five SM collision kernels, summed and gated (BURST's ``trans_evolve``).

Each kernel (ν–ν, ν–ν̄, and the ν–e± kernels nuer1, nuer2, nuepma) returns, per flavour m, cp n and
bin i, a net rate (gain − loss) and a forward-plus-reverse rate frs (gain + loss), both per T_cm⁵.
Near equilibrium Σnet is a cancellation between nearly equal forward and reverse rates and is
dominated by round-off, so BURST gates it::

    test_ratio = |Σnet| / Σfrs / prec_trans[m, n, i]       (0 where Σfrs = 0)
    df/dt = T_cm⁵·Σnet  if test_ratio ≥ nuratiotol (= 3), else 0

``prec_trans`` is that round-off floor, ``|Σnet|/Σfrs`` at the equilibrium spectrum
(:func:`compute_prec_trans`).  Only the ν–e± kernels exchange energy with the plasma, so
``drhonudt`` sums only their gated nets.  Rates are per second; ``neutt.decoupling`` divides by H.

``use_sched`` selects the ν–e± implementation at trace time: False (reference), True (scheduled,
per-bin regime select) or "union" (scheduled, regime-masked).  The scheduled modes change the gate
sums at round-off level, and full solves with them need dtmax ≲ 0.1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..constants import M_ELECTRON, PI
from ..grid import EpsGrid
from .nunu import nunu_rates
from .nunubar import nunubar_rates
from .nuer1 import nuer1_rates
from .nuer2 import nuer2_rates
from .nuepma import nuepma_rates
from .nuer1 import nuer1_rates_sched
from .nuer2 import nuer2_rates_sched
from .nuepma import nuepma_rates_sched

NURATIOTOL = 3.0          # gate threshold
PREC_FLOOR = 1.0e-13      # noise floor where Σfrs ≤ 0
OCC_FLOOR = 1.0e-300      # occupations are clamped to [OCC_FLOOR, 1] so log f stays finite


def equilibrium_occ(grid: EpsGrid, xi=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    """Equilibrium occupation f0[m, n, i] = 1/(exp(ε_i − s_n·ξ_m) + 1), s = (+1, −1) for (ν, ν̄).

    ``xi`` (3,) is the per-flavour degeneracy and may be traced (vmap/grad over ξ without
    recompiling).  As in BURST, the ε = 0 bin is exactly ½ at ξ = 0.
    """
    eps = grid.abscissas
    xi = jnp.asarray(xi, dtype=jnp.float64)
    parfact = jnp.asarray([1.0, -1.0])
    arg = eps[None, None, :] - parfact[None, :, None] * xi[:, None, None]   # (3,2,N)
    return 1.0 / (jnp.exp(arg) + 1.0)


def _all_kernel_rates(grid, occ, meps, tcmpl, phie, nunu_kernel, nunubar_kernel, use_sched=False,
                      fd_cap_logC=float("inf"), lag_tail_fix=True,
                      fd_decay=False, monotone_read=False):
    """(numer, denom, leg4), each (3, 2, N): Σnet and Σfrs over the five kernels, and the ν–e± nets.

    leg4 (BURST's process slot 4) is the part of Σnet that exchanges energy with the plasma.
    Arguments as :func:`_kernel_rate_groups`.
    """
    groups = _kernel_rate_groups(grid, occ, meps, tcmpl, phie,
                                 nunu_kernel, nunubar_kernel, use_sched,
                                 fd_cap_logC=fd_cap_logC,
                                 lag_tail_fix=lag_tail_fix, fd_decay=fd_decay,
                                 monotone_read=monotone_read)
    (net_nn, frs_nn, net_s, frs_s, net_a, frs_a,
     net_r1, frs_r1, net_r2, frs_r2, net_pma, frs_pma) = groups
    numer = net_nn + net_s + net_a + net_r1 + net_r2 + net_pma
    denom = frs_nn + frs_s + frs_a + frs_r1 + frs_r2 + frs_pma
    leg4 = net_r1 + net_r2 + net_pma           # plasma energy-exchange channel
    return numer, denom, leg4


def _kernel_rate_groups(grid: EpsGrid, occ, meps, tcmpl, phie,
                        nunu_kernel, nunubar_kernel, use_sched=False,
                        fd_cap_logC=float("inf"), lag_tail_fix=True,
                        fd_decay=False, monotone_read=False):
    """Per-kernel (net, frs) pairs, the terms of the gate sums and the sum rules.

    Returns (net_nn, frs_nn, net_s, frs_s, net_a, frs_a, net_r1, frs_r1, net_r2, frs_r2, net_pma,
    frs_pma), each (3, 2, N): ν–ν, ν–ν̄ scattering, ν–ν̄ annihilation, nuer1, nuer2, nuepma.

    The occupation is first clamped to [OCC_FLOOR, 1].  The clamp is a no-op for a physical
    occupation; for a bin overshot below zero it keeps log f finite (the NaN would otherwise be
    masked to zero by the gate, freezing the bin), at the price of an RHS discontinuous at f = 0.

    ``fd_cap_logC``, ``fd_decay`` and ``monotone_read`` are the off-grid closures of
    :mod:`neutt.interp`, used only by the QKE solver (the defaults are bitwise no-ops).
    ``lag_tail_fix`` is the Gauss–Laguerre tail rule of nuer1/nuer2/nuepma.
    """
    occ = jnp.clip(occ, OCC_FLOOR, 1.0)
    net_nn, frs_nn = nunu_rates(nunu_kernel, occ, fd_cap_logC=fd_cap_logC,
                                fd_decay=fd_decay, monotone_read=monotone_read)
    net_s, frs_s, net_a, frs_a = nunubar_rates(nunubar_kernel, occ, fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                                               monotone_read=monotone_read)
    if use_sched:
        select = use_sched != "union"
        net_r1, frs_r1 = nuer1_rates_sched(grid, occ, meps, tcmpl, phie, select=select,
                                           fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                                           monotone_read=monotone_read, lag_tail_fix=lag_tail_fix)
        net_r2, frs_r2 = nuer2_rates_sched(grid, occ, meps, tcmpl, phie, select=select,
                                           fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                                           monotone_read=monotone_read, lag_tail_fix=lag_tail_fix)
        net_pma, frs_pma = nuepma_rates_sched(grid, occ, meps, tcmpl, phie, select=select,
                                              fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                                              monotone_read=monotone_read, lag_tail_fix=lag_tail_fix)
    else:
        net_r1, frs_r1 = nuer1_rates(grid, occ, meps, tcmpl, phie, fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                                     monotone_read=monotone_read, lag_tail_fix=lag_tail_fix)
        net_r2, frs_r2 = nuer2_rates(grid, occ, meps, tcmpl, phie, fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                                     monotone_read=monotone_read, lag_tail_fix=lag_tail_fix)
        net_pma, frs_pma = nuepma_rates(grid, occ, meps, tcmpl, phie, fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                                        monotone_read=monotone_read, lag_tail_fix=lag_tail_fix)
    return (net_nn, frs_nn, net_s, frs_s, net_a, frs_a,
            net_r1, frs_r1, net_r2, frs_r2, net_pma, frs_pma)


def compute_prec_trans_at(grid: EpsGrid, nunu_kernel, nunubar_kernel, occ_eq, meps, tcmpl,
                          phie=0.0, use_sched=False, lag_tail_fix=True):
    """Noise floor prec[3, 2, N] = ``|Σnet|/Σfrs`` of the five kernels at the equilibrium ``occ_eq``.

    At the given (meps, tcmpl, phie); :func:`compute_prec_trans` is BURST's static point.
    ``PREC_FLOOR`` where Σfrs ≤ 0; the ε = 0 bin is 0.  ``lag_tail_fix`` as in :func:`assemble_rhs`,
    so the floor is measured with the same kernels as the gated rates.
    """
    numer, denom, _ = _all_kernel_rates(grid, occ_eq, meps, tcmpl, phie,
                                        nunu_kernel, nunubar_kernel, use_sched,
                                        lag_tail_fix=lag_tail_fix)
    prec = jnp.where(denom > 0.0, jnp.abs(numer) / jnp.where(denom > 0.0, denom, 1.0), PREC_FLOOR)
    return jnp.where(grid.abscissas > 0.0, prec, 0.0)


def compute_prec_trans(grid: EpsGrid, nunu_kernel, nunubar_kernel, xi=(0.0, 0.0, 0.0),
                       use_sched=False, lag_tail_fix=True):
    """BURST's static noise floor prec_trans[3, 2, N]: equilibrium, meps = 0, tcmpl = 1, phie = 0.

    It depends only on the grid and ``xi`` (per-flavour degeneracy, (3,), may be traced).  With a
    static ``xi``, XLA would constant-fold it inside a jitted solve, which mishandles the
    forward−reverse cancellation it measures (prec → 1, inverting the gate).  ``optimization_barrier`` makes the
    equilibrium occupation an opaque runtime value; at runtime it is an identity.
    """
    occ0 = jax.lax.optimization_barrier(equilibrium_occ(grid, xi))
    return compute_prec_trans_at(grid, nunu_kernel, nunubar_kernel, occ0, 0.0, 1.0, 0.0, use_sched,
                                 lag_tail_fix)


def assemble_rhs(grid: EpsGrid, occ, tcmev, tplmev, phie, prec_trans,
                 nunu_kernel, nunubar_kernel, nuratiotol: float = NURATIOTOL,
                 m_e_eff: float = M_ELECTRON, use_sched=False,
                 gate_width: float = 0.0, lag_tail_fix=True):
    """Gated df/dt [s⁻¹] and plasma energy-exchange rate drhonudt [MeV⁴/s] (``trans_evolve``).

    Parameters
    ----------
    occ : jnp.ndarray
        Occupation f[3, 2, N] (flavour, cp, bin).
    tcmev, tplmev : float
        T_cm and T_pl [MeV].
    phie : float
        Electron degeneracy μ_e/T_pl.
    prec_trans : jnp.ndarray
        Noise floor from :func:`compute_prec_trans`.
    nunu_kernel, nunubar_kernel : eqx.Module
        The grid-only ν–ν and ν–ν̄ kernels.
    m_e_eff : float
        Electron mass in the ν–e± kernels [MeV].
    use_sched : bool or str
        ν–e± implementation: False (reference), True (scheduled, per-bin select) or "union"
        (scheduled, regime-masked).
    gate_width : float
        0 = BURST's hard gate; > 0 replaces it by a sigmoid in log(test_ratio/nuratiotol) of this
        width (a continuous RHS).
    lag_tail_fix : bool
        Gauss–Laguerre tail rule in nuer1/nuer2/nuepma (False = bit-faithful to BURST).

    Returns
    -------
    doccfracdt : jnp.ndarray
        df/dt, (3, 2, N) [s⁻¹].
    drhonudt : float
        Rate at which the neutrinos gain energy from the plasma [MeV⁴/s].
    """
    meps = m_e_eff / tcmev
    tcmpl = tcmev / tplmev
    numer, denom, leg4 = _all_kernel_rates(grid, occ, meps, tcmpl, phie,
                                           nunu_kernel, nunubar_kernel, use_sched,
                                           lag_tail_fix=lag_tail_fix)

    test_ratio = jnp.where(denom != 0.0,
                           jnp.abs(numer) / jnp.where(denom != 0.0, denom, 1.0)
                           / jnp.where(prec_trans > 0.0, prec_trans, 1.0),
                           0.0)
    eps = grid.abscissas
    w = grid.weights
    if gate_width > 0.0:
        # NaN or zero test_ratio gets weight 0, as in the hard gate (NaN ≥ tol is False)
        pos = test_ratio > 0.0                               # False for NaN too
        logr = jnp.log(jnp.where(pos, test_ratio, 1.0) / nuratiotol)
        gate_w = jnp.where(pos, jax.nn.sigmoid(logr / gate_width), 0.0)
        doccfracdt = tcmev**5 * jnp.where(gate_w > 0.0, gate_w * numer, 0.0)
        leg4_g = jnp.sum(jnp.where(gate_w > 0.0, gate_w * leg4, 0.0), axis=(0, 1))
    else:
        gate = test_ratio >= nuratiotol                      # (3,2,N) bool
        doccfracdt = tcmev**5 * jnp.where(gate, numer, 0.0)   # s⁻¹
        leg4_g = jnp.sum(jnp.where(gate, leg4, 0.0), axis=(0, 1))  # (N,)

    # T_cm⁹ = T_cm⁵ (rate) × T_cm⁴ (energy density)
    drhonudt = tcmev**9 / (2.0 * PI**2) * jnp.sum(w * eps**3 * leg4_g)

    return doccfracdt, drhonudt


SPLIT_OCC_TINY = 1.0e-100  # floor on f in revf/f; prevents 0/0 at empty bins
SPLIT_ONEMF_FLOOR = 1.0e-3  # floor on 1 − f in forf/(1 − f); exact for f ≤ 0.999 (FD to ξ ≈ 6.9)


def assemble_split_rhs(grid: EpsGrid, occ, tcmev, tplmev, phie, prec_trans,
                       nunu_kernel, nunubar_kernel, nuratiotol: float = NURATIOTOL,
                       m_e_eff: float = M_ELECTRON, use_sched=False,
                      fd_cap_logC=float("inf"), lag_tail_fix=True,
                      fd_decay=False, monotone_read=False):
    """The gated RHS in affine form, df/dt = A − B·f with A, B ≥ 0 [s⁻¹], for the exponential step.

    Every kernel's forward rate carries (1 − f) and its reverse rate f, so from Σnet and Σfrs::

        forf = (Σfrs + Σnet)/2 = (1 − f)·g,    revf = (Σfrs − Σnet)/2 = f·l,    A = g,  B = g + l,

    and A − B·f equals :func:`assemble_rhs`'s ``doccfracdt`` to round-off.  The step
    f′ = f·e^{−Bh} + (A/B)(1 − e^{−Bh}) is exact for frozen (A, B), positivity-preserving and has
    no stability limit on h; g and l are clamped at 0 so round-off-negative rates cannot break
    positivity.  The gate (hard only) and ``drhonudt`` are those of :func:`assemble_rhs`.

    Arguments as :func:`assemble_rhs` without ``gate_width``, plus the off-grid closures
    (``fd_cap_logC``, ``fd_decay``, ``monotone_read``; see :func:`_kernel_rate_groups`).
    Returns (A, B, drhonudt).
    """
    meps = m_e_eff / tcmev
    tcmpl = tcmev / tplmev
    numer, denom, leg4 = _all_kernel_rates(grid, occ, meps, tcmpl, phie,
                                           nunu_kernel, nunubar_kernel, use_sched,
                                           fd_cap_logC=fd_cap_logC,
                                           lag_tail_fix=lag_tail_fix, fd_decay=fd_decay,
                                           monotone_read=monotone_read)

    test_ratio = jnp.where(denom != 0.0,
                           jnp.abs(numer) / jnp.where(denom != 0.0, denom, 1.0)
                           / jnp.where(prec_trans > 0.0, prec_trans, 1.0),
                           0.0)
    gate = test_ratio >= nuratiotol                          # (3,2,N) bool

    forf = jnp.where(gate, 0.5 * (denom + numer), 0.0)
    revf = jnp.where(gate, 0.5 * (denom - numer), 0.0)
    occ_c = jnp.clip(occ, OCC_FLOOR, 1.0)                    # the occupation the kernels saw
    g = jnp.maximum(tcmev**5 * forf / jnp.maximum(1.0 - occ_c, SPLIT_ONEMF_FLOOR), 0.0)
    l = jnp.maximum(tcmev**5 * revf / jnp.maximum(occ_c, SPLIT_OCC_TINY), 0.0)

    leg4_g = jnp.sum(jnp.where(gate, leg4, 0.0), axis=(0, 1))  # (N,)
    eps = grid.abscissas
    w = grid.weights
    drhonudt = tcmev**9 / (2.0 * PI**2) * jnp.sum(w * eps**3 * leg4_g)

    return g, g + l, drhonudt


def decoherence_rate(grid: EpsGrid, occ, tcmev, tplmev, phie,
                     nunu_kernel, nunubar_kernel, m_e_eff: float = M_ELECTRON,
                     use_sched=False, fd_cap_logC=float("inf"),
                     fd_decay=False, monotone_read=False, lag_tail_fix=True):
    """Ungated collisional decoherence rate Γ[3, 2, N] [s⁻¹], the QKE off-diagonal damping.

    Every weak coupling is flavour-diagonal, so on a flavour-diagonal bath the matrix collision
    term damps the coherences as dρ_ab/dt = −½(Γ_a + Γ_b)ρ_ab with

        Γ_a = g_a + l_a,    g_a = T_cm⁵·forf_a/(1 − f_a),    l_a = T_cm⁵·revf_a/f_a,

    the ``B`` of :func:`assemble_split_rhs` without the gate.  The gate suppresses the
    forward−reverse cancellation in Σnet; Γ is dominated by the cancellation-free Σfrs and is
    largest where the gate would zero it (near equilibrium), so gating it would send Γ → 0 at late
    times.  Arguments as :func:`assemble_split_rhs`, without ``prec_trans`` and ``nuratiotol``.
    """
    meps = m_e_eff / tcmev
    tcmpl = tcmev / tplmev
    numer, denom, _ = _all_kernel_rates(grid, occ, meps, tcmpl, phie,
                                        nunu_kernel, nunubar_kernel, use_sched,
                                        fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                                        monotone_read=monotone_read, lag_tail_fix=lag_tail_fix)
    forf = 0.5 * (denom + numer)
    revf = 0.5 * (denom - numer)
    occ_c = jnp.clip(occ, OCC_FLOOR, 1.0)                    # the occupation the kernels saw
    g = jnp.maximum(tcmev**5 * forf / jnp.maximum(1.0 - occ_c, SPLIT_ONEMF_FLOOR), 0.0)
    l = jnp.maximum(tcmev**5 * revf / jnp.maximum(occ_c, SPLIT_OCC_TINY), 0.0)
    return g + l


def decoherence_rate_split(grid: EpsGrid, occ, tcmev, tplmev, phie,
                           nunu_kernel, nunubar_kernel, m_e_eff: float = M_ELECTRON,
                           use_sched=False, fd_cap_logC=float("inf"),
                           fd_decay=False, monotone_read=False, lag_tail_fix=True):
    """:func:`decoherence_rate` split into its ν–e± and ν–ν halves, (gamma_nue, gamma_nunu).

    nue = nuer1 + nuer2 + nuepma, nunu = ν–ν + ν–ν̄; the sum reproduces :func:`decoherence_rate` up
    to the per-group clamps at 0 (no-ops for a physical state).  Used when the QKE solver evolves
    the ν–e± coherences with the matrix collision operator (``collision="matrix_nue"``), which
    already contains their damping; only the ν–ν half is then added as a scalar rate.
    """
    meps = m_e_eff / tcmev
    tcmpl = tcmev / tplmev
    groups = _kernel_rate_groups(grid, occ, meps, tcmpl, phie,
                                 nunu_kernel, nunubar_kernel, use_sched,
                                 fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                                 monotone_read=monotone_read, lag_tail_fix=lag_tail_fix)
    (net_nn, frs_nn, net_s, frs_s, net_a, frs_a,
     net_r1, frs_r1, net_r2, frs_r2, net_pma, frs_pma) = groups
    occ_c = jnp.clip(occ, OCC_FLOOR, 1.0)

    def _gl(numer, denom):
        forf = 0.5 * (denom + numer)
        revf = 0.5 * (denom - numer)
        g = jnp.maximum(tcmev**5 * forf / jnp.maximum(1.0 - occ_c, SPLIT_ONEMF_FLOOR), 0.0)
        l = jnp.maximum(tcmev**5 * revf / jnp.maximum(occ_c, SPLIT_OCC_TINY), 0.0)
        return g + l

    return (_gl(net_r1 + net_r2 + net_pma, frs_r1 + frs_r2 + frs_pma),
            _gl(net_nn + net_s + net_a, frs_nn + frs_s + frs_a))


def compute_sumrules(grid: EpsGrid, occ, tcmev, tplmev, phie, prec_trans,
                     nunu_kernel, nunubar_kernel, nuratiotol: float = NURATIOTOL,
                     m_e_eff: float = M_ELECTRON, use_sched=False, lag_tail_fix=True):
    """BURST's conservation monitors sumrule1–5 at one state, summed over the gated bins.

    sumrule1/2 are the ν-only number/energy balances (≈ 0 up to quadrature error); 3–5 monitor ν–ν̄
    (CP) symmetry.  One full kernel evaluation, so for post-hoc diagnostics, not the solver RHS.
    Arguments as :func:`assemble_rhs`.  Returns a dict of sumrule{k} = net/frs (0 where frs = 0,
    a divide BURST leaves unguarded) and the raw sumrule{k}net, sumrule{k}frs.
    """
    meps = m_e_eff / tcmev
    tcmpl = tcmev / tplmev
    (net_nn, frs_nn, net_s, frs_s, net_a, frs_a,
     net_r1, frs_r1, net_r2, frs_r2, net_pma, frs_pma) = _kernel_rate_groups(
        grid, occ, meps, tcmpl, phie, nunu_kernel, nunubar_kernel, use_sched,
        lag_tail_fix=lag_tail_fix)

    numer = net_nn + net_s + net_a + net_r1 + net_r2 + net_pma
    denom = frs_nn + frs_s + frs_a + frs_r1 + frs_r2 + frs_pma
    test_ratio = jnp.where(denom != 0.0,
                           jnp.abs(numer) / jnp.where(denom != 0.0, denom, 1.0)
                           / jnp.where(prec_trans > 0.0, prec_trans, 1.0),
                           0.0)
    gate = test_ratio >= nuratiotol                       # (3,2,N)

    eps = grid.abscissas
    w = grid.weights
    pref = 1.0 / (2.0 * PI**2)
    w2 = pref * w * eps**2                                # (N,)
    w3 = pref * w * eps**3
    parfact = jnp.array([1.0, -1.0])[None, :, None]       # +1 ν, −1 ν̄

    def gsum(weights_1d, field):
        return jnp.sum(weights_1d * jnp.where(gate, field, 0.0))

    nu_net = net_nn + net_s + net_a                       # ν-only processes
    nu_frs = frs_nn + frs_s + frs_a
    out = {
        "sumrule1net": gsum(w2, nu_net), "sumrule1frs": gsum(w2, nu_frs),
        "sumrule2net": gsum(w3, nu_net), "sumrule2frs": gsum(w3, nu_frs),
        "sumrule3net": gsum(w2, parfact * (net_a + net_pma)),
        "sumrule3frs": gsum(w2, frs_a + frs_pma),
        "sumrule4net": gsum(w2, parfact * net_a), "sumrule4frs": gsum(w2, frs_a),
        "sumrule5net": gsum(w2, parfact * net_pma), "sumrule5frs": gsum(w2, frs_pma),
    }
    for k in range(1, 6):
        net, frs = out[f"sumrule{k}net"], out[f"sumrule{k}frs"]
        out[f"sumrule{k}"] = jnp.where(frs != 0.0, net / jnp.where(frs != 0.0, frs, 1.0), 0.0)
    return out
