"""The full coupled QKE solve: {T_pl, ρ[2,N,3,3]} → N_eff with the coherences' back-reaction.

The full density matrix is evolved, so the coherences and their back-reaction on the occupations
are carried together; nothing is assumed about adiabaticity.  Each ln a macro-step is operator
split:

1. Collision and plasma: the classical affine-split rates (``decoupling.make_split_rates``,
   df/dln a = A − B·f over the five kernels, plus the dT_pl ODE), advanced on the diagonal with the
   L-stable exponential-midpoint step (``hybrid.exp_step``).
2. Coherent substep: ρ ← exp(L_coh·dt)ρ per bin (``coherent_method``), with L_coh = −i[H,·] minus
   an off-diagonal-only dephasing at Γ_ab = ½(Γ_a + Γ_b).  The generator is trace- and
   positivity-preserving, so it conserves Σ_a ρ_aa per bin and keeps the diagonal ≥ 0.

``collision`` selects the operator on the coherences: scalar damping (default), which is exact as a
loss but has no gain, or the density-matrix (Sigl–Raffelt) operator A_gain − {B_loss, ρ}.

``mixing=False`` sets H = 0 and skips the coherent substep at trace time, so the solve reduces
bit-identically to the classical decoupling solve.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ...constants import M_ELECTRON, PI, SIN2THW   # noqa: F401  (PI: pinned ν–ν code)
from ...grid import EpsGrid, make_eps_grid       # noqa: F401  (EpsGrid: pinned ν–ν code)
from ...interp import FD_ENVELOPE_C
from ...thermo import ETA0_DEFAULT, Thermo
from .. import assemble as A
from ..nunu import build_nunu_kernel
from ..nunubar import build_nunubar_kernel
from . import coherent
from . import nonadiabatic as na
from .coherent import hermitize
from . import collide as qcoll
from . import collide_nue as qnue
from . import collide_nunu as qnunu
from .collide_matrix import offdiag_only
from ... import compiling


# ==== ν–ν self-interaction: PINNED (2026-10-22) ======================================================
# The ν–ν refractive terms are zero in this version: ``solve`` passes H_sia = H_sis = 0 to
# coherent.hamiltonian (which still implements both, oracle-tested).  The self-interaction code is
# commented out in place, not deleted: the helpers below, the moment set-up in the factory, and the
# substepped mean-field branch of the coherent substep.  It is the resolved substepped mean field
# (knobs self_interaction, si_substeps, si_theta, si_omega_star, si_substeps_max; diagnostics si_gain,
# si_gain_hist, si_nsub_hist) and the fine log-ε coherent state grid that resolves its collective mode
# (coh_grid, coh_eps_low).  Uncommenting restores the mean field on the Boole grid.  coh_grid also needs
# its restriction/prolongation at the collision edges and the pre-pin make_split_rates option
# return_energy=True (not in this version); the complete pre-pin code is branch selfint-prepin of the
# BURST repository (jaxburst names).  Physics status:
# docs/HANDOFF_2026-10-12_QKE_STRIATIONS_EXPLAINED.md (branch dark).
#
# def _si_pack(nudiff, nusum):
#     """(nudiff, nusum) Hermitian 3×3 → 18 real unknowns (diag ×3, then Re/Im of 01, 02, 12)."""
#     def _v(M):
#         return jnp.stack([M[0, 0].real, M[1, 1].real, M[2, 2].real,
#                           M[0, 1].real, M[0, 1].imag, M[0, 2].real, M[0, 2].imag,
#                           M[1, 2].real, M[1, 2].imag])
#     return jnp.concatenate([_v(nudiff), _v(nusum)])
#
#
# def _si_unpack(x):
#     """Inverse of :func:`_si_pack`."""
#     def _m(v):
#         return jnp.array([[v[0], v[3] + 1j * v[4], v[5] + 1j * v[6]],
#                           [v[3] - 1j * v[4], v[1], v[7] + 1j * v[8]],
#                           [v[5] - 1j * v[6], v[7] - 1j * v[8], v[2]]], dtype=jnp.complex128)
#     return _m(x[:9]), _m(x[9:])
#
#
# # ---- ν–ν self-interaction ---------------------------------------------------------------------
# # self_interaction: False (no ν-ν terms) | True (both terms) | "second_order_only".
# #   H_sia ∝ √2 G_F (n_ν − n_ν̄)  first order in G_F (nonzero in the symmetric universe: ρ̄ = ρ*
# #           ⇒ ∝ Σ w ε² Im ρ), the mean-field term of the flavour-pendulum literature;
# #   H_sis ∝ (G_F/M_Z²) (ρ+ρ̄)    second order; |δN_eff| ≲ 3e-7.
# # "second_order_only" is the standard literature treatment (Froustey–Pitrou–Volpe 2020 solve the
# # ν equation in the ATAO scheme, where ρ is real and n_ν − n_ν̄ ≡ 0).  Both moments enter the
# # coherent substep through the RESOLVED SUBSTEPPED mean field below (2026-09-24): the per-bin
# # propagation over the macro step stays the exact frozen-(H₀,Γ) one, and only the bilinear coupling
# # −i[H_si(m(ρ)), ρ] is advanced in si_substeps substeps with the 18 moments implicit-midpoint (an
# # 18×18 linear solve each).  Rejected alternatives, all measured on scripts_qke/selfint/mwe_meanfield.py:
# # a per-macro-step-frozen mean field (exponential Euler on the coupling; spurious onset 6.8 → 5.9 →
# # 5.15 MeV for dlna 1e-5 → 3.3e-6 → 1e-6, 2026-09-21); the macro-step trapezoid (implicit midpoint on
# # an UNRESOLVED coupling, μ_eff·dt ~ 1e2–1e3: self-sustained 15–25 % transient coherence at
# # dlna = 1.4e-4, 2026-09-23); an exact-in-step LINEARISED coupling (re-linearising once per macro
# # step drops −i[H_si(δm), δρ], a 25 % effect while the collective mode swings within a step — it
# # diverges in ~40 steps for any substep count, 2026-09-24).  Genuine effect of H_sia over the full
# # solve: δN_eff = −3e-7 (vacuum-only, N=101, n=23333 and 70000).  The model's fixed-T linear
# # instability of the mean field below ~4.5 MeV (+100…+600 s⁻¹ at N=101) is a momentum-grid artefact
# # (single-bin modes whose rates fall with N) — docs/HANDOFF_2026-09-24_QKE_SI_SUBSTEP.md (branch dark).
# SELF_INTERACTION_MODES = ("off", "full", "second_order_only")
# # coh_grid: drive T_pl from the state grid's own energy-exchange integral (HANDOFF_2026-09-28 §8 D2, branch dark).  Diagnostic
# # switch, read at trace time; False leaves the plasma on the Boole grid's leg-4 integral as in the coarse solve.
# _COH_GRID_PLASMA_BUDGET = True
#
#
# def resolve_self_interaction(self_interaction):
#     """Normalise the ``self_interaction`` knob: ``False`` → ``"off"``, ``True`` → ``"full"``,
#     ``"second_order_only"`` as is; anything else raises ``ValueError``."""
#     if isinstance(self_interaction, (bool, np.bool_)):
#         return "full" if self_interaction else "off"
#     if self_interaction not in SELF_INTERACTION_MODES:
#         raise ValueError(f"self_interaction must be False, True or one of {SELF_INTERACTION_MODES}, "
#                          f"got {self_interaction!r}")
#     return self_interaction
#
#
# def _lagrange_weights(x_nodes, x):
#     """Lagrange interpolation weights of the nodes ``x_nodes`` at ``x`` (numpy, static)."""
#     x_nodes = np.asarray(x_nodes, float)
#     wts = np.ones_like(x_nodes)
#     for m in range(x_nodes.size):
#         for l in range(x_nodes.size):
#             if l != m:
#                 wts[m] *= (x - x_nodes[l]) / (x_nodes[m] - x_nodes[l])
#     return wts
#
#
# def _make_coh_grid(grid, coh_grid):
#     """The fine coherent state grid and its exchange operators with the Boole collision ``grid``.
#
#     ``coh_grid = ("log", N_coh, eps_min)``: trapezoid rule in ln ε on N_coh nodes over [eps_min, ε_max] — uniform
#     relative frequency resolution, spectrally accurate on the Lorentzian resonant layer of the ν–ν mode (1e4 nodes
#     give its growth rate to 0.5 %, uniform grids need ≳ 1e5).  Returns ``(sgrid, R, P)`` with R (N_c, N_coh) the
#     restriction (cubic Lagrange in ln ε at the Boole nodes; the ε = 0 node takes the first fine value) and P
#     (N_coh, N_c) the prolongation of RATES (cubic Lagrange in ε on the positive Boole nodes plus the anchor (0, 0)
#     — collision rates and Γ vanish ∝ p at p → 0; the Boole ε = 0 column is dropped).
#     """
#     kind, n_coh, eps_min = coh_grid
#     if kind != "log":
#         raise ValueError(f'coh_grid must be ("log", N_coh, eps_min), got {coh_grid!r}')
#     n_coh, eps_min = int(n_coh), float(eps_min)
#     eps_c = np.asarray(grid.abscissas, float)
#     eps_max = float(eps_c[-1])
#     first_pos = float(eps_c[eps_c > 0.0][0])
#     if not (n_coh >= 8 and 0.0 < eps_min < first_pos):
#         raise ValueError(f"coh_grid needs N_coh >= 8 and 0 < eps_min < {first_pos}, got {coh_grid!r}")
#     u = np.linspace(np.log(eps_min), np.log(eps_max), n_coh)
#     du = u[1] - u[0]
#     e = np.exp(u)
#     wq = e * du
#     wq[0] *= 0.5
#     wq[-1] *= 0.5
#     sgrid = EpsGrid(abscissas=jnp.asarray(e), weights=jnp.asarray(wq), nbins=n_coh,
#                     eps_range=eps_max - eps_min, eps_offset=eps_min)
#     R = np.zeros((eps_c.size, n_coh))
#     for j, ec in enumerate(eps_c):
#         if ec <= eps_min:
#             R[j, 0] = 1.0
#             continue
#         uc = np.log(ec)
#         k = int(np.clip(np.searchsorted(u, uc) - 2, 0, n_coh - 4))
#         R[j, k:k + 4] = _lagrange_weights(u[k:k + 4], uc)
#     pos = np.nonzero(eps_c > 0.0)[0]
#     X = np.concatenate([[0.0], eps_c[pos]])                 # anchor (0, 0) + the positive Boole nodes
#     cols = np.concatenate([[-1], pos])                      # −1: the anchor has no column
#     P = np.zeros((n_coh, eps_c.size))
#     for q, eq in enumerate(e):
#         k = int(np.clip(np.searchsorted(X, eq) - 2, 0, X.size - 4))
#         wts = _lagrange_weights(X[k:k + 4], eq)
#         for m in range(4):
#             if cols[k + m] >= 0:
#                 P[q, cols[k + m]] = wts[m]
#     return sgrid, R, P
#
# The pinned knobs' documentation (formerly in make_full_coupled_solver's docstring):
# self_interaction : {False, True, "second_order_only"}
#     The ν-ν self-interaction (refractive) Hamiltonian.  Static; requires ``mixing=True`` (no-op
#     otherwise).  ``False`` (default): no ν-ν terms, bit-identical to the vacuum/matter solve.
#     ``True``: both terms, H_sia ∝ √2 G_F (n_ν − n_ν̄) and H_sis ∝ (G_F/M_Z²)(ρ+ρ̄).
#     ``"second_order_only"``: H_sis alone — the literature (ATAO) treatment, for comparison.
#     H_sia is NOT zero in the symmetric universe (ρ̄ = ρ* ⇒ ∝ Σ w ε² Im ρ); it is the mean-field
#     term of the flavour-pendulum system, and the coded form reproduces that system's analytic
#     growth rates to 0.1% (scripts_qke/selfint/pendulum_benchmark.py).  Its genuine effect on
#     N_eff over the full solve is −3e-7 (vacuum-only); with it the ν/ν̄-antisymmetric
#     (lepton-number) sector is weakly unstable below ~6 MeV and saturates at |ρ̄ − ρ*| ~ 1e-7,
#     so coherence-level quantities there are only statistically meaningful — N_eff is not
#     affected (docs/HANDOFF_2026-09-21_QKE_SELFINT.md (branch dark)).

#     The source moments ``nudiff = Σ w ε²(ρ−ρ̄)`` / ``nusum = Σ w ε³(ρ+ρ̄)`` (18 complex numbers,
#     linear in ρ) enter the coherent substep through a resolved substepping of the coupling: the
#     per-bin frozen-(H₀,Γ) propagation over the step is exact (E = e^{L₀h}, Φ = hφ₁(L₀h) with
#     h = dt/``si_substeps``, from the same dense/secular machinery as the hybrid step), and per
#     substep the bilinear term −i[H_si(m(ρ)), ρ] is linearised around the substep and the
#     moments taken implicit-midpoint — an 18×18 linear solve, no Newton.  The coupling's own
#     rate μ_eff = 2μΣwε²δf ~ 1e7 s⁻¹ (μ = √2 G_F T³/2π²ℏ) is what must be resolved:
#     ``result["si_gain"]`` is the max over the run of the per-substep loop gain of the
#     first-order block, ‖½(MΦB)_nudiff‖_F (``si_gain_hist`` per step; 0 in
#     "second_order_only"); the MWE gives the same |H_sia| to 1 % for gains 38, 9.6 and 2.4
#     (16, 64, 256 substeps) at 4 MeV, dlna = 1.4e-4, matching the macro-step trapezoid at
#     dlna/4, so the gain need not be ≲ 1.  Cost ≈ si_substeps × a few small kernels; at 64 about 1/10 of the former
#     Newton (18 tangents × up to 8 iterations through the coherent step).  Why not the
#     macro-step trapezoid or an exact-in-step linearised coupling: module header.
# si_substeps : int
#     Substeps of the mean-field coupling per macro step (16: the value validated end to end on the
#     notebook configuration, n = 50 000, N_eff 3.04399172; 64 explodes at 0.37 MeV — handoff 09-24 §5).
#     With ``si_theta`` it is the minimum of the schedule.
# coh_grid : None | ("log", N_coh, eps_min)
#     None (default): the state ρ lives on ``grid`` (as before 2026-09-28).  ("log", N_coh, eps_min): the
#     state lives on a fine log-ε trapezoid grid of N_coh nodes on [eps_min, ε_max of ``grid``] and ``grid``
#     becomes the collision EVALUATION grid.  The coherent sector needs it: with the first-order ν–ν mean field
#     the CP-symmetric state has a collective instability resonant with a momentum layer ~2e-4 wide in ε
#     (ε_r·T ≈ 0.3 MeV) that the Boole grid aliases — ×21 in growth rate at 1.3 MeV; 1e4 log nodes are within
#     0.5 % of the continuum, 3e4 exact (docs/HANDOFF_2026-09-28_QKE_RESOLVE_MODE.md (branch dark) §7–8).  Per macro step the
#     populations are restricted to ``grid`` (cubic in ln ε on ln f) for the kernels and the midpoint evaluation;
#     the collision update ln(f_coll/f), Γ and the matrix-collision gains are prolonged back (cubic in ε through
#     (0, 0)) and the update is applied multiplicatively, so the state grid's populations stay positive and follow
#     the Boole grid's gated increments; T_pl is driven by the state grid's quadrature of the same
#     instantaneous rate, so the ν↔plasma budget has one integrand on both sides.  The populations must live on
#     the fine grid too (populations frozen ⇒ no mode).  N_eff,
#     moments and frames are on the state grid, which the factory returns as its grid (and ``result["eps"]``,
#     ``result["w"]``).  Cost: ~3 ms per coupling substep at 1e4 nodes on an A100.
# si_theta : None | float
#     None (default): the fixed ``si_substeps``.  A float: per macro step the coupling substep count is the
#     smallest level si_substeps·2^k ≤ ``si_substeps_max`` with ω*(T)·dt/n ≤ si_theta, ω*(T) the collective
#     frequency of the ν–ν mode (``si_omega_star``), selected by a lax.switch over the compiled levels.  The
#     substep error is second order: −1.6 % (−0.5 %) in the mode's growth rate at 0.3 (0.15), −0.7 % (−0.2 %)
#     in its frequency.  ``result["si_nsub_hist"]`` records the count used per step.
# si_omega_star : (w0, T0, p)
#     ω*(T) = w0·(T/T0)^p [s⁻¹, MeV]; the default fits the dispersion relation's roots (2.06e5 / 1.745e5 /
#     1.43e5 s⁻¹ at 1.8 / 1.304 / 0.8 MeV).
# si_substeps_max : int
#     Cap of the substep schedule (default 1024).


def make_full_coupled_solver(grid=None, *, thermo=None, mixing=True, include_matter=False,
                             include_matter_asym=False, use_renorm=True,
                             renorm_pdep=True, qed_nlo=True, use_sched="union", tcm_cutoff=0.015,
                             n_steps=6000,
                             phie=0.0, eta0=None, q_baryon=0.5,
                             xi=(0.0, 0.0, 0.0), normal_hierarchy=True,
                             gamma_ungated=True, coh_eps_cut=None, n_frames=600,
                             tail_closure="srcfix", fd_cap_C=None,
                             coherent_method="hybrid", collision_coupling="sourced",
                             collision="damping"):
    # ν–ν pinned (module header): the keyword arguments coh_eps_low=None, self_interaction=False,
    # si_substeps=16, coh_grid=None, si_theta=None, si_omega_star=(1.745e5, 1.304, 0.45), si_substeps_max=1024.
    """Compile-once full coupled-QKE solver; returns ``(solve, grid)``.

    Every argument here is static; ``solve`` takes the swept scalars and does not recompile across
    their values.  The ν–ν self-interaction terms are zero (pinned; see the
    module header).

    Parameters
    ----------
    grid : EpsGrid, optional
        Default ``make_eps_grid()``.
    mixing : bool
        Include the vacuum oscillation Hamiltonian; False reduces exactly to the classical solve.
    use_renorm, renorm_pdep, qed_nlo, tcm_cutoff
        Forwarded to ``decoupling.make_split_rates``.
    use_sched : {"union", True, False}
        Lepton kernels: "union" (default; regime-masked, bit-faithful, fastest here), True (per-bin
        regime gather) or False (the reference kernels).
    xi, normal_hierarchy, coh_eps_cut
        The (3,) flavour degeneracies (they set prec_trans and the initial FD occupation), the mass
        ordering, and the ε above which H is zeroed (None: no cut).
    phie, eta0, q_baryon
        Electron degeneracy: a float, or ``"auto"`` for charge neutrality at the live T_pl (as in
        ``decoupling.make_rhs``).  It reaches the matter potential, the collision and Γ.
    include_matter : bool
        Add the thermal (C-even, G_F²) MSW potential from the e± energy density.
    include_matter_asym : bool
        Add the asymmetric ("solar", linear-G_F) MSW potential √2 G_F n_p on the ee entry, with
        n_p = q_baryon·thermo.nb_tcm3(eta0)·T_cm³.  It is the one ξ = 0 term that breaks ν ↔ ν̄, and
        seeds an η_b-sized flavour asymmetry ρ_aa(ν) − ρ_aa(ν̄) (read it from ``diag_fr`` or
        ``rho_final``).  Default off, bit-identical to the solve without it.
    n_steps : int
        Fixed ln a steps (the scan length; the cost is linear in it).  The default 6000 is for
        absolute N_eff/Y_p (order-2 error ≈1.5e-5); the mixing ΔN_eff is converged from 2000, where
        the baseline error cancels in the difference.  Keep n_steps ≳ 500: at ≲ 250 the explicit
        T_pl step goes unstable, which ``solve`` detects (``decoupling.plasma_history_ok``) and
        raises on.
    gamma_ungated : bool
        The decoherence rate Γ_a: True (default) the ungated total throughput g_a + l_a
        (``assemble.decoherence_rate``); False the gated transport rate B·H, a comparison mode
        allowed only with ``collision="damping"``.
    n_frames : int
        Number of downsampled frames of the coherences to return; the per-step trajectory is not
        kept.
    tail_closure : {"srcfix", "cap"}
        The off-grid tail closure, without which the coherence-induced lepton-kernel tail pump sends
        the solve to NaN.  It acts on both kernel edges (the collision and Γ) and in both mixing
        branches, so it cancels in the mixing ΔN_eff.  "srcfix" (default): the parameter-free
        ``fd_decay`` + ``monotone_read`` closures, no envelope cap.  "cap": the envelope f ≤ C·FD,
        C = ``fd_cap_C``.  See the manual, "The off-grid tail closure".
    fd_cap_C : float or None
        C for ``tail_closure="cap"`` (None: ``interp.FD_ENVELOPE_C``); ignored by "srcfix".
    coherent_method : {"hybrid", "expm", "strang"}
        The coherent-substep integrator.  "hybrid" (default,
        ``nonadiabatic.coherent_step_sourced_hybrid``): per bin, the secular step where
        ω_max·dt > ``nonadiabatic.HYBRID_WDT_SWITCH`` and Γ_max·dt ≤ 1, the dense expm elsewhere;
        exact phases and no split error.  "expm": the dense 9×9 expm(L_coh·dt), which is not unitary
        at large phase (≈1e-17·ω·dt per step, and ω·dt reaches 1e10), so the low-ε coherences become
        round-off noise.  "strang": the corotating Strang split (an exact 3×3 unitary between half
        dephasing kicks), free of round-off but with a split error (1.5e-6 in ΔN_eff at
        n_steps = 2000).  The sourced coupling uses the ``*_sourced*`` variant of each.
    collision : {"damping", "matrix", "matrix_full", "matrix_nue"}
        The collision operator on the coherences (see the manual, "The collision acting on the
        coherences").
        "damping" (default): dρ_ab/dt = −½(Γ_a + Γ_b)ρ_ab, exact as a loss but with no gain.
        "matrix": all five SM channels through the Sigl–Raffelt operator A_gain − {B_loss, ρ}.  Each
        channel's scalar damping is replaced by 2·diag(B_loss) (equal on a diagonal bath) and the
        momentum-nonlocal remainder enters the sourced step as a source; the diagonal stays on the
        classical kernels.
        "matrix_full": as "matrix", and the diagonal is advanced from the same operator (the
        classical A − B·f plus the O(``|ρ_off|²``) coherence feedback) with the same exp_step and the
        classical precision-gate decision, so it differs from "matrix" in the operator only.
        "matrix_nue": the three ν–e channels on the matrix operator, ν–ν and ν–ν̄ on damping.
        The matrix modes require ``collision_coupling="sourced"`` and ``gamma_ungated=True``.
    collision_coupling : {"sourced", "jump"}
        How the collision's diagonal update reaches the coherent substep.  "sourced" (default): as a
        constant rate src = (f_coll − f)/dt inside the exponential, so the quasi-steady target of
        ρ_off moves continuously and a solve on the QSS stays on it.  "jump": the collision is
        applied first and the coherent step runs on the result; ρ_off then trails a target that
        jumps every step, a first-order lag that freezes in after decoupling (Im ρ_eτ ≈ 2.5e-7
        against 5.5e-11 sourced, n_steps = 1e4).  N_eff barely sees the difference.

    ``mixing=False`` skips the coherent substep in every mode, so all of these reduce
    bit-identically to the classical solve there.

    Returns
    -------
    (solve, EpsGrid)
        ``solve(T_init=10.0, lna_max=7.0, delta_neff_init=0.0, me=None, xi=None,
        matter_asym_scale=1.0)``.  ``me=None`` uses M_ELECTRON; ``xi=None`` the build-time
        degeneracies, while a (3,) ``xi`` recomputes prec_trans and the initial occupation inside
        the jit; ``matter_asym_scale`` multiplies n_p (0: no ν–ν̄ asymmetry, 1: physical η_b).  It
        returns a dict with the final state ('neff_final', 'T_pl_final', 'T_cm_final',
        'ratio_final', 'f_final' (3,2,N), 'rho_final' (2,N,3,3)), per-step histories ('lna_left',
        'T_pl_hist', 'T_cm_hist', 'num_drift', 'min_f'), the NaN flag 'any_nan', the closure
        provenance 'fd_cap_C'/'fd_cap_logC', and the frames ('frame_idx', 'lna_fr', 'T_cm_fr',
        'T_cm_end_fr', 'off_fr', 'ss_fr', 'diag_fr'); see the comments at the return.
    """
    import neutt.decoupling as D
    from neutt.hybrid import exp_step

    # # Resolve + validate before any kernel build, so a bad mode fails in milliseconds.
    # si_mode = resolve_self_interaction(self_interaction)
    # si_on = si_mode != "off"
    # if not (isinstance(si_substeps, (int, np.integer)) and si_substeps >= 1):
    #     raise ValueError(f"si_substeps must be a positive int, got {si_substeps!r}")
    # if si_on and coherent_method == "strang":
    #     raise ValueError('self_interaction requires coherent_method "hybrid" or "expm": the substepped '
    #                      'mean field is built on the sourced dense/secular propagators, not the strang split')
    # if si_theta is not None:
    #     if not (isinstance(si_theta, (int, float)) and si_theta > 0.0):
    #         raise ValueError(f"si_theta must be None or a positive float, got {si_theta!r}")
    #     if not (isinstance(si_substeps_max, (int, np.integer)) and si_substeps_max >= si_substeps):
    #         raise ValueError(f"si_substeps_max must be an int >= si_substeps, got {si_substeps_max!r}")
    #     if len(si_omega_star) != 3:
    #         raise ValueError("si_omega_star must be (w0, T0, p)")

    if grid is None:
        grid = make_eps_grid()
    if thermo is None:
        thermo = Thermo()
    eps = grid.abscissas
    w = grid.weights                       # noqa: F841  (the pinned ν–ν moments use it)
    N = eps.shape[0]
    # # ---- state grid (2026-09-28): the Boole grid, or the fine log-ε coherent grid (coh_grid) ----
    # # Every ρ-shaped array is (…, N, …) on the STATE grid; the collision kernels see the Boole grid through the
    # # restriction R (cubic in ln ε) and hand their rates back through the prolongation P (cubic in ε, through
    # # (0, 0) below the first positive Boole node).  Without coh_grid both are the identity and N is the Boole N.
    # if coh_grid is None:
    #     sgrid, R_op, P_op = grid, None, None
    # else:
    #     sgrid, R_np, P_np = _make_coh_grid(grid, coh_grid)
    #     R_op, P_op = jnp.asarray(R_np), jnp.asarray(P_np)
    # eps = sgrid.abscissas
    # w = sgrid.weights
    # N = eps.shape[0]
    #
    # def _restrict(x):
    #     """(…, N) → (…, N_c) along the last axis."""
    #     return x if R_op is None else jnp.einsum('cq,...q->...c', R_op, x)
    #
    # def _prolong(x):
    #     """(…, N_c) → (…, N) along the last axis."""
    #     return x if P_op is None else jnp.einsum('qc,...c->...q', P_op, x)
    #
    # def _prolong_rate(x):
    #     """A non-negative rate (…, N_c) → (…, N): prolonged and clipped at 0 (cubic overshoot guard)."""
    #     return x if P_op is None else jnp.maximum(_prolong(x), 0.0)
    #
    # def _restrict_occ(f3):
    #     """Populations (…, N) → (…, N_c): cubic in ln ε on ln f — positive by construction and exact on the
    #     exponential tail (a cubic on f itself can turn a 1e-11 tail value negative, which the kernels' tail
    #     closures then feed to a log)."""
    #     if R_op is None:
    #         return f3
    #     return jnp.exp(jnp.einsum('cq,...q->...c', R_op, jnp.log(jnp.maximum(f3, 1e-300))))
    #
    # def _restrict1(x):
    #     """Bin axis 1: (a, N, …) → (a, N_c, …)."""
    #     return x if R_op is None else jnp.einsum('cq,aq...->ac...', R_op, x)
    #
    # def _prolong1(x):
    #     """Bin axis 1: (a, N_c, …) → (a, N, …)."""
    #     return x if P_op is None else jnp.einsum('qc,ac...->aq...', P_op, x)
    #
    # def _prolong_rate1(x):
    #     return x if P_op is None else jnp.maximum(_prolong1(x), 0.0)
    # φ_e at the live state: charge neutrality for phie="auto", else the float as given
    phie_auto = isinstance(phie, str) and phie == "auto"
    eta0_default = ETA0_DEFAULT if eta0 is None else eta0

    def _phie_at(T_pl, T_cm, me, eta0):
        return (thermo.solve_phie(T_pl, T_cm / T_pl, eta0, q_baryon, m_e=me)
                if phie_auto else phie)
    # # self-interaction source-moment weights (diff_pow=2 number-like, sum_pow=3 energy-like)
    # we_diff = (w * eps**2)[:, None, None]    # (N,1,1) for nudiff = Σ w ε²(ρ−ρ̄)
    # we_sum = (w * eps**3)[:, None, None]     # (N,1,1) for nusum  = Σ w ε³(ρ+ρ̄)

    # ---- tail closure (``tail_closure``): the three settings both kernel edges below take ----
    if tail_closure == "srcfix":
        fd_decay, monotone_read = True, True
        fd_cap_C_used, fd_cap_logC = float("inf"), float("inf")
    elif tail_closure == "cap":
        fd_decay, monotone_read = False, False
        fd_cap_C_used = float(FD_ENVELOPE_C if fd_cap_C is None else fd_cap_C)   # active C (provenance)
        fd_cap_logC = float(np.log(fd_cap_C_used))
    else:
        raise ValueError("tail_closure must be 'srcfix' or 'cap'; got %r" % (tail_closure,))
    if coherent_method not in ("expm", "strang", "hybrid"):
        raise ValueError("coherent_method must be 'expm', 'strang' or 'hybrid'; got %r"
                         % (coherent_method,))

    # ---- static ingredients, built once ----
    nunu_kernel = build_nunu_kernel(grid)
    nunubar_kernel = build_nunubar_kernel(grid)
    prec_trans = A.compute_prec_trans(grid, nunu_kernel, nunubar_kernel, xi=xi, use_sched=use_sched)
    occ0 = A.equilibrium_occ(grid, xi=xi)                          # (3,2,N) diagonal IC

    m2w = (jnp.asarray(coherent.m2_weak(normal_hierarchy=normal_hierarchy)) if mixing
           else jnp.zeros((3, 3, 2)))
    z33 = jnp.zeros((3, 3), dtype=jnp.complex128)
    # Coherence support: ε > 0, optionally ε ≤ coh_eps_cut.  The deep tail carries negligible
    # coherence (vacuum H ∝ Δm²/2p) and energy, so zeroing H there is inert, and it removes a
    # back-reaction seed the lepton kernels can amplify into a tail pump.
    # # ν–ν pinned: coh_eps_low insulates the FASTEST bins (vacuum ω ∝ 1/ε) instead, the low-ε mirror of
    # # coh_eps_cut: with the ν–ν mean field on, the ε = 0.25 bin of the 101-bin grid drives a lepton-number
    # # (ν/ν̄-antisymmetric) instability of the bulk below ~1.5 MeV (growth 14 → 1e3 s⁻¹, saturating at
    # # |ρ̄−ρ*| ~ 1e-7 and leaving Θ_a ~ 1e-12 noisy); with H = 0 for ε < 0.5 it does not occur (2026-09-24,
    # # docs/HANDOFF_2026-09-24_QKE_SI_SUBSTEP.md (branch dark) §7).  Those bins carry ~1 % of the number moment w ε²; in the
    # # continuum the ε → 0 region has zero weight and infinite frequency, i.e. it averages out of the mean
    # # field, which a discretised bin cannot do.  Default None (off); 0.5 is the validated value.
    # _lo = 0.0 if coh_eps_low is None else float(coh_eps_low)      (and "& (eps >= _lo)" in both masks)
    if coh_eps_cut is not None:
        eps_coh = ((eps > 0.0) & (eps <= coh_eps_cut))[None, :, None, None]
    else:
        eps_coh = (eps > 0.0)[None, :, None, None]
    idx3 = jnp.arange(3)

    # # ν–ν mean field (static): the 18 complex moments of ρ — nudiff (9) then nusum (9); the nudiff block is
    # # zero in "second_order_only", whose H_si has no first-order term.  Applied directly (2026-09-28; the former
    # # dense (18, 2N·9) matrix was O(N²) to build — a round-off-level change in the summation order).
    # I18 = jnp.eye(18, dtype=jnp.complex128)
    # m2w_zero = jnp.zeros((3, 3, 2))
    #
    # def _moments(r):
    #     """(2,N,3,3) → (18,) complex."""
    #     nusum = jnp.sum(we_sum * (r[0] + r[1]), axis=0).reshape(9)
    #     nudiff = (jnp.zeros(9, jnp.complex128) if si_mode == "second_order_only"
    #               else jnp.sum(we_diff * (r[0] - r[1]), axis=0).reshape(9))
    #     return jnp.concatenate([nudiff, nusum])
    #
    # def _moments_cols(Rm):
    #     """The moments of each of the 18 columns of Rm (2,N,9,18) → (18,18)."""
    #     return jax.vmap(lambda k: _moments(Rm[..., k].reshape(2, N, 3, 3)), out_axes=1)(jnp.arange(18))
    #
    # # Coupling-substep schedule (HANDOFF_2026-09-28 §8 D3, branch dark): static levels si_substeps·2^k ≤ si_substeps_max,
    # # one lax.switch branch each, chosen per macro step from ω*(T)·dt/si_theta; si_theta=None → the fixed count.
    # si_levels = [si_substeps]
    # if si_theta is not None:
    #     while si_levels[-1] < si_substeps_max:
    #         si_levels.append(2 * si_levels[-1])
    # si_levels_j = jnp.asarray(si_levels, dtype=jnp.int32)

    stride = max(1, n_steps // n_frames)
    frame_idx = jnp.arange(0, n_steps, stride)

    def _hamiltonian(T_cm, T_pl, me, eta0, sia=z33, sis=z33, asym_scale=1.0):
        """Coherent H (2,N,3,3) [s⁻¹]: vacuum [+ thermal MSW] [+ asymmetric MSW]; ν–ν terms zero."""
        if include_matter:
            # rho_electron returns e⁻+e⁺, so a single call is the full e± density.
            therm = thermo.rho_electron(T_pl, _phie_at(T_pl, T_cm, me, eta0), me)
            _, mats = coherent.matter_terms(therm, 0.0)
        else:
            mats = z33
        if include_matter_asym:
            # The same matrix enters ν and ν̄: in this conjugate (Sigl–Raffelt) convention the ν̄
            # sign flip sits on the vacuum and thermal terms, not on this one (the lab-frame
            # equivalent flips the asymmetric term instead; Nötzold–Raffelt).  asym_scale (traced)
            # multiplies n_p: 0 null, 1 physical η_b, > 1 an enhanced source.
            npro = asym_scale * q_baryon * thermo.nb_tcm3(eta0) * T_cm**3   # n_p = q_baryon·n_b [MeV³]
            mata, _ = coherent.matter_terms(0.0, npro)
        else:
            mata = z33
        H = coherent.hamiltonian(eps, T_cm, m2w, mata, mats, sia, sis)  # (2,N,3,3)
        return jnp.where(eps_coh, H, 0.0)                              # 0 off the coherence support

    if collision not in ("damping", "matrix_nue", "matrix", "matrix_full"):
        raise ValueError('collision must be "damping", "matrix_nue", "matrix" or "matrix_full", '
                         f'got {collision!r}')
    if collision != "damping" and collision_coupling != "sourced":
        # without a source the gain is dropped and only the replaced loss kept: a wrong damping
        raise ValueError(f'collision={collision!r} requires collision_coupling="sourced" '
                         f'(got {collision_coupling!r}): the nonlocal coherence gain enters as a '
                         'source and the jump/strang paths have nowhere to put it.')
    if collision != "damping" and not gamma_ungated:
        raise ValueError(f'collision={collision!r} is incompatible with gamma_ungated=False: the '
                         'matrix path takes its nu-e rate from B_loss, not from the gated '
                         'transport rate.')

    @jax.jit
    def solve(T_init=10.0, lna_max=7.0, delta_neff_init=0.0, me=None, xi=None, matter_asym_scale=1.0,
              eta0=None):
        compiling()
        if me is None:
            me = M_ELECTRON
        if eta0 is None:
            eta0 = eta0_default
        T_init = jnp.asarray(T_init, dtype=jnp.float64)
        _asym_scale = jnp.asarray(matter_asym_scale, dtype=jnp.float64)
        # a passed xi recomputes both from the traced value (compute_prec_trans's
        # optimization_barrier keeps that jit-safe), so an xi scan shares one compile
        if xi is None:
            prec_use, occ_use = prec_trans, occ0
        else:
            _xi = jnp.asarray(xi, dtype=jnp.float64)
            prec_use = A.compute_prec_trans(grid, nunu_kernel, nunubar_kernel, xi=_xi,
                                            use_sched=use_sched)
            occ_use = A.equilibrium_occ(grid, xi=_xi)

        split = D.make_split_rates(grid, prec_use, nunu_kernel, nunubar_kernel, T_init,
                                   use_renorm=use_renorm, use_sched=use_sched,
                                   renorm_pdep=renorm_pdep, qed_nlo=qed_nlo, tcm_cutoff=tcm_cutoff,
                                   phie=phie, eta0=eta0, q_baryon=q_baryon,
                                   delta_neff_init=delta_neff_init, me=me, fd_cap_logC=fd_cap_logC,
                                   fd_decay=fd_decay, monotone_read=monotone_read, thermo=thermo)
        dlna = lna_max / n_steps

        rho0 = na.rho_from_offdiag(jnp.zeros((2, N, 6), dtype=jnp.complex128),
                                   jnp.transpose(occ_use, (1, 2, 0)))   # (2,N,3,3) decohered IC

        n_fr = frame_idx.shape[0]
        # frame buffers (off, ss, diag), carried through the scan and written on frame steps only,
        # so no per-step (n_steps, 2, N, 6) stack is built
        buf0 = (jnp.zeros((n_fr, 2, N, 6), dtype=jnp.complex128),
                jnp.zeros((n_fr, 2, N, 6), dtype=jnp.complex128),
                jnp.zeros((n_fr, 2, N, 3), dtype=jnp.float64))

        def step(carry, scan_in):
            T_pl, rho, bufs = carry
            lna0, i = scan_in
            f_s = jnp.real(rho[:, :, idx3, idx3])                # (2,N,3) diagonal occupations
            f = jnp.transpose(f_s, (2, 0, 1))                    # (3,2,N) classical-port layout
            # f = _restrict_occ(…)  — ν–ν pinned: coh_grid restriction to the collision grid

            # --- (1) collision: exponential midpoint on {T_pl, f} over the full step ---
            a1, b1, dT1, _ = split(lna0, T_pl, f)
            f_half = exp_step(f, a1, b1, 0.5 * dlna)
            T_half = T_pl + 0.5 * dlna * dT1
            a2, b2, dT2, Hub = split(lna0 + 0.5 * dlna, T_half, f_half)
            T_cm_mid = T_init * jnp.exp(-(lna0 + 0.5 * dlna))
            dt = dlna / Hub                                                  # seconds
            f_coll = exp_step(f, a2, b2, dlna)                               # (3,2,N) collided populations
            f_coll_s = jnp.transpose(f_coll, (1, 2, 0))                      # (2,N,3) the collided diagonal
            # ν–ν pinned: with coh_grid the state grid takes the update from the collision grid —
            # if coh_grid is not None:
            #     # the state grid takes the Boole grid's collision update as the prolonged LOG-RATIO ln(f_coll/f)
            #     # (cubic in ε through 0 at ε = 0), applied multiplicatively: positive by construction, exact for an
            #     # exponential relaxation, and it carries the gated increments as they are — prolonging the affine
            #     # rates (A, B) instead re-derives rates across BURST's per-bin precision gate, where they are
            #     # discontinuous in ε, and moved N_eff by 4e-5 (2026-09-28)
            #     lr = jnp.log(jnp.maximum(f_coll, 1e-300)) - jnp.log(jnp.maximum(f, 1e-300))
            #     f_coll_s = f_s * jnp.exp(jnp.transpose(_prolong(lr), (1, 2, 0)))
            #     if _COH_GRID_PLASMA_BUDGET:
            #         # D2: the plasma's energy loss is the state grid's quadrature of the same INSTANTANEOUS
            #         # midpoint rate that dT_pl already uses (the leg-4 Boole integral behind drhonudt), so the
            #         # ν↔plasma budget has one integrand on both sides; an effective-rate version instead mixes two
            #         # time discretisations and moves N_eff by O(dlna²) (1.4e-4 at n = 300).
            #         rate2 = _prolong((a2 - b2 * f_half) * Hub)                # (3,2,N) s⁻¹ instantaneous, midpoint
            #         drho_s = T_cm_mid ** 4 / (2.0 * PI ** 2) * jnp.sum((w * eps ** 3)[None, None, :] * rate2)
            #         dT2 = dT2 + (sp2[4] - drho_s) / (Hub * sp2[5])           # (drhonudt_coarse − drhonudt_state)/(H C)
            dfdt_s = (f_coll_s - f_s) / dt                                   # (2,N,3) s⁻¹ the effective rate
            T_new = T_pl + dlna * dT2

            # write the collided diagonal back into ρ (off-diagonal untouched by the collision)
            rho_c = rho.at[:, :, idx3, idx3].set(f_coll_s.astype(jnp.complex128))

            num_before = jnp.sum(jnp.real(rho_c[:, :, idx3, idx3]), axis=-1)  # (2,N) per-bin number
            f_coll_mat = None       # the matrix-collided diagonal (2,N,3); matrix_full only

            # --- (2) coherent substep ---
            if mixing:
                # def _H_of(nudiff, nusum):
                #     """Full coherent H for given source moments (nusum ∝ ρ+ρ̄ feeds H_sis; nudiff ∝ ρ−ρ̄
                #     feeds H_sia, which does NOT vanish at ξ=0: ρ̄=ρ* ⇒ ∝ Im ρ — module header)."""
                #     sia_m, sis_m = coherent.self_interaction_terms(T_cm_mid, nudiff, nusum)
                #     return _hamiltonian(T_cm_mid, T_new, me, sia_m, sis_m,
                #                         asym_scale=_asym_scale)                 # (2,N,3,3) s⁻¹
                #
                # def _H_si_basis(m):
                #     """H_si alone (2,N,3,3) for one basis moment m ∈ C¹⁸ — H_si is linear in m."""
                #     sia_m, sis_m = coherent.self_interaction_terms(T_cm_mid, m[:9].reshape(3, 3),
                #                                                    m[9:].reshape(3, 3))
                #     return jnp.where(eps_coh, coherent.hamiltonian(eps, T_cm_mid, m2w_zero, z33, z33,
                #                                                    sia_m, sis_m), 0.0)
                #
                H = _hamiltonian(T_cm_mid, T_new, me, eta0, asym_scale=_asym_scale)  # (2,N,3,3) s⁻¹, H_si = 0
                # Γ and the matrix collision are evaluated at the midpoint state
                phie_mid = _phie_at(T_half, T_cm_mid, me, eta0)
                coh_gain = None     # the matrix collision's off-diagonal source (2,N,3,3)
                if collision != "damping":
                    # Matrix collision dρ_ab/dt|coll = A_gain,ab − {B_loss, ρ}_ab.  Each matrix
                    # channel's scalar damping is replaced (not supplemented) by 2·diag(B), which
                    # equals the ungated throughput Γ_a on a diagonal bath, so the loss is
                    # unchanged and the new physics is the momentum-nonlocal gain, a source.
                    rho_mid = rho_c.at[:, :, idx3, idx3].set(
                        jnp.transpose(f_half, (1, 2, 0)).astype(jnp.complex128))   # (2,N,3,3)
                    # sin2thw = the classical kernels' SIN2THW, so a matrix mode changes the
                    # operator and not the coupling
                    kw_c = dict(fd_cap_logC=fd_cap_logC,
                                fd_decay=fd_decay, monotone_read=monotone_read)
                    n_tot = jnp.zeros_like(rho_mid)
                    b_tot = jnp.zeros_like(rho_mid)
                    for _core in (qcoll._nuepma_matrix_core, qnue.nuer1_matrix_core,
                                  qnue.nuer2_matrix_core):
                        _n, _, _, _b = _core(grid, rho_mid, me / T_cm_mid, T_cm_mid / T_half,
                                             phie_mid, sin2thw=SIN2THW, **kw_c)
                        n_tot = n_tot + _n
                        b_tot = b_tot + _b
                    tc5 = T_cm_mid ** 5                                          # rates -> s⁻¹
                    if collision in ("matrix", "matrix_full"):
                        # ν–ν and ν–ν̄ too, so no scalar damping survives; same quadrature as the
                        # diagonal collision (use_sched selects lepton kernels only)
                        for _k, _core in ((nunu_kernel, qnunu.nunu_matrix_core),
                                          (nunubar_kernel, qnunu.nunubar_matrix_core)):
                            _n, _, _, _b = _core(_k, rho_mid, **kw_c)
                            n_tot = n_tot + _n
                            b_tot = b_tot + _b
                        bd = jnp.real(b_tot[:, :, idx3, idx3])                   # (2,N,3)
                        gamma_c = 2.0 * tc5 * bd
                    else:
                        # matrix_nue: ν–ν and ν–ν̄ stay on damping, added from the split rate;
                        # n_tot and B_loss cover the ν–e sector alone
                        _, g_nunu = A.decoherence_rate_split(
                            grid, f_half, T_cm_mid, T_half, phie_mid,
                            nunu_kernel, nunubar_kernel, m_e_eff=me, use_sched=use_sched,
                            fd_cap_logC=fd_cap_logC, fd_decay=fd_decay,
                            monotone_read=monotone_read)                         # (3,2,N) s⁻¹
                        bd = jnp.real(b_tot[:, :, idx3, idx3])                   # (2,N,3)
                        gamma_c = 2.0 * tc5 * bd + jnp.transpose(g_nunu, (1, 2, 0))
                    # gamma applies exactly −(B_aa+B_bb)ρ_ab inside the exponential, so the source
                    # is the rest of net_ab: net + (B_aa+B_bb)ρ, not offdiag(A_gain), since the
                    # anticommutator's (ρ_aa+ρ_bb)B_ab term is first order in the coherence.  The
                    # off-diagonal RHS is then T_cm⁵·net_ab exactly at the frozen state.
                    bsum = (bd[:, :, :, None] + bd[:, :, None, :]).astype(rho_c.dtype)
                    src_full = tc5 * (n_tot + bsum * rho_mid)                    # (2,N,3,3) s⁻¹
                    coh_gain = offdiag_only(src_full)
                    gamma = gamma_c
                    if collision == "matrix_full":
                        # The diagonal from the same operator,
                        #     dρ_aa/dt = A_gain,aa − 2 B_aa ρ_aa − Σ_{c≠a} 2 Re(B_ac ρ_ca),
                        # the classical affine A − B·f plus the O(``|ρ_off|²``) coherence feedback.  The
                        # diagonal of src_full is that split's A and gamma_c its B = 2·T_cm⁵·B_aa;
                        # both go to the same exp_step as the classical collision, so the operator
                        # split is unchanged.  The precision gate stays on the diagonal (it kills
                        # forward−reverse cancellation noise) and off the coherences (gating them
                        # would switch their damping off at freeze-out).  b2 > 0 is exactly the
                        # classical gate decision at this midpoint, tcm_cutoff included (nothing
                        # else zeroes the classical g + l), so both modes open the same bins.
                        dgate = jnp.transpose(b2 > 0.0, (1, 2, 0))               # (2,N,3) bool
                        a_mat = jnp.where(dgate, jnp.real(src_full[:, :, idx3, idx3]), 0.0)
                        b_mat = jnp.where(dgate, gamma_c, 0.0)                   # (2,N,3) s⁻¹
                        f_coll_mat = exp_step(f_s, a_mat / Hub,
                                              b_mat / Hub, dlna)                 # (2,N,3)
                elif gamma_ungated:
                    gsrc = A.decoherence_rate(grid, f_half, T_cm_mid, T_half, phie_mid,
                                              nunu_kernel, nunubar_kernel, m_e_eff=me,
                                              use_sched=use_sched,
                                              fd_cap_logC=fd_cap_logC,
                                              fd_decay=fd_decay, monotone_read=monotone_read)  # (3,2,N) s⁻¹
                    gamma = jnp.transpose(gsrc, (1, 2, 0))       # (2,N,3)
                else:
                    gsrc = b2 * Hub                                              # (3,2,N) s⁻¹ gated
                    gamma = jnp.transpose(gsrc, (1, 2, 0))       # (2,N,3)
                if collision_coupling == "sourced":
                    # The collision as a constant rate (``collision_coupling``), built from the same
                    # f_coll the jump path writes: the two differ in when the diagonal changes,
                    # not by how much.  matrix_full hands in the matrix diagonal's step instead.
                    src = jnp.zeros_like(rho)
                    dfdt = dfdt_s if f_coll_mat is None else (f_coll_mat - f_s) / dt   # (2,N,3) s⁻¹
                    src = src.at[:, :, idx3, idx3].set(dfdt.astype(jnp.complex128))
                    if coh_gain is not None:
                        # the nonlocal gain does not depend on ρ_ab at this bin: a source too
                        src = src + coh_gain

                def _advance(H, r):
                    """Advance r over dt under the frozen H with the chosen ``coherent_method``."""
                    if collision_coupling == "sourced":
                        if coherent_method == "strang":
                            return na.coherent_step_sourced_strang(H, gamma, r, dt, src)
                        elif coherent_method == "hybrid":
                            return na.coherent_step_sourced_hybrid(H, gamma, r, dt, src)
                        return na.coherent_step_sourced(H, gamma, r, dt, src)   # (2,N,3,3)
                    elif coherent_method == "hybrid":
                        return na.coherent_step_sourced_hybrid(H, gamma, r, dt,
                                                               jnp.zeros_like(r))
                    elif coherent_method == "strang":
                        return na.coherent_step_strang(H, gamma, r, dt)      # (2,N,3,3)
                    return na.coherent_step(H, gamma, r, dt)                 # dense expm (reference)

                # Moving eigenframe: K rotates the eigenbasis of H₀(step start) onto that of H₀(step
                # end) and the step runs in that frame (generator H_mid − K), so the coherence is
                # carried adiabatically instead of being kicked by a frozen H at every boundary.
                # K = 0 in vacuum.  See the manual, "The moving eigenframe of the step".
                K_fr = na.eigenframe_generator(
                    _hamiltonian(T_init * jnp.exp(-lna0), T_pl, me, eta0, asym_scale=_asym_scale),
                    _hamiltonian(T_init * jnp.exp(-(lna0 + dlna)), T_new, me, eta0, asym_scale=_asym_scale), dt)
                R_fr = na.unitary_exp(K_fr, 0.5 * dt)
                R_frd = jnp.conj(jnp.swapaxes(R_fr, -1, -2))
                rho_in = R_fr @ (rho if collision_coupling == "sourced" else rho_c) @ R_frd

                # if si_on:
                #     # Resolved substepping of the ν–ν mean-field coupling (2026-09-24,
                #     # docs/HANDOFF_2026-09-24_QKE_SI_SUBSTEP.md (branch dark)).  The per-bin propagation over the macro
                #     # step is the exact frozen-(H₀,Γ) one — E = e^{L₀h}, Φ = hφ₁(L₀h) from the same
                #     # dense/secular machinery as coherent_step_sourced_hybrid, h = dt/n_sub, so
                #     # Eⁿ = e^{L₀dt} — and only the bilinear coupling −i[H_si(m(ρ)), ρ] is substepped.  Over a
                #     # substep it is linearised around the substep (B(ρ) = −i[H_si(·), ρ] is linear in ρ:
                #     # predictor with B(ρ_k), corrector with B at the substep midpoint) and the 18 complex
                #     # moments m are implicit-midpoint, one 18×18 linear solve:
                #     #     (I − ½G) m₁ = M(Eρ₀ + Φc) + ½G m₀,   G = M Φ B,   ρ₁ = Eρ₀ + Φc + ΦB·½(m₀+m₁).
                #     # No Newton, no Jacobian tangents.  Over a substep δm, δρ are small, so the
                #     # linearisation holds; over a MACRO step it does not (the collective mode swings by
                #     # O(1) within dt), which is why re-linearising once per step diverges and why the
                #     # macro-step trapezoid — implicit midpoint on a coupling with μ_eff·dt ~ 1e2–1e3 —
                #     # self-sustained a 15–25 % transient coherence (module header).  The substep count n_sub
                #     # comes from the static levels si_levels (si_theta: ω*(T)·h ≤ si_theta, second-order
                #     # error −1.6 % / −0.5 % in the collective mode's growth rate at 0.3 / 0.15; fixed
                #     # si_substeps otherwise), one lax.switch branch per level.
                #     # ρ_start = the state the coherent substep starts from: ρ under the sourced coupling
                #     # (the collision enters as the constant rate c = src), else ρ_c with no source.
                #     H0 = _hamiltonian(T_cm_mid, T_new, me, asym_scale=_asym_scale)   # no ν–ν terms
                #     # in the step's eigenframe (above); the ν–ν coupling is evaluated on the frame state,
                #     # an O(K·dt) rotation of ρ, i.e. a second-order error per step
                #     rho_s = rho_in
                #     src_s = src if collision_coupling == "sourced" else jnp.zeros_like(rho_c)
                #     H0K = H0 - K_fr
                #     Hb = jax.vmap(_H_si_basis)(I18)                                  # (18,2,N,3,3)
                #
                #     def _Bof(r9):
                #         """B(ρ): the 18 columns −i[H_si(e_m), ρ] as a (2,N,9,18) matrix."""
                #         r = r9.reshape(2, N, 3, 3)
                #         C = -1j * (Hb @ r[None] - r[None] @ Hb)
                #         return jnp.moveaxis(C.reshape(18, 2, N, 9), 0, -1)
                #
                #     def _si_run(n_sub):
                #         def run(_):
                #             h = dt / n_sub
                #             E, Phi = na.sourced_propagators_hybrid(
                #                 H0K, gamma, h, dense_only=(coherent_method == "expm"))   # (2,N,9,9) each
                #             r_c = jnp.einsum('...ij,...j->...i', Phi, src_s.reshape(2, N, 9))   # Φ c  (2,N,9)
                #
                #             def _sub(c, _):
                #                 r9, uacc, gmax = c
                #                 Er = jnp.einsum('...ij,...j->...i', E, r9) + r_c           # Eρ₀ + Φc
                #                 ua = _moments(r9.reshape(2, N, 3, 3))                          # m₀
                #                 MEr = _moments(Er.reshape(2, N, 3, 3))
                #
                #                 def _solve_with(B):
                #                     R = jnp.einsum('...ij,...jk->...ik', Phi, B)           # Φ B  (2,N,9,18)
                #                     G = _moments_cols(R)                                    # (18,18)
                #                     ub = jnp.linalg.solve(I18 - 0.5 * G, MEr + 0.5 * (G @ ua))   # m₁
                #                     um = 0.5 * (ua + ub)
                #                     return Er + jnp.einsum('...ij,j->...i', R, um), um, G
                #                 B0 = _Bof(r9)
                #                 r_pred, _, _ = _solve_with(B0)                             # predictor
                #                 r1, um, G = _solve_with(0.5 * (B0 + _Bof(r_pred)))         # corrector
                #                 # loop gain of the FIRST-ORDER block (nudiff → H_sia → nudiff); the nusum
                #                 # block is inert physics with large moment weights and would swamp it
                #                 return (r1, uacc + um,
                #                         jnp.maximum(gmax, 0.5 * jnp.linalg.norm(G[:9, :9]))), None
                #             (r_f, uacc, si_gain_k), _ = jax.lax.scan(
                #                 _sub, (rho_s.reshape(2, N, 9), jnp.zeros(18, jnp.complex128), 0.0),
                #                 None, length=n_sub)
                #             return r_f, uacc / n_sub, si_gain_k               # end state, step-average moments
                #         return run
                #     if len(si_levels) == 1:
                #         level = jnp.int32(0)
                #         r_f, ubar, si_gain = _si_run(si_levels[0])(None)
                #     else:
                #         w0s, T0s, ps = si_omega_star
                #         n_req = w0s * (T_cm_mid / T0s) ** ps * dt / si_theta
                #         level = jnp.clip(jnp.ceil(jnp.log2(jnp.maximum(n_req / si_levels[0], 1.0))),
                #                          0, len(si_levels) - 1).astype(jnp.int32)
                #         r_f, ubar, si_gain = jax.lax.switch(level, [_si_run(n) for n in si_levels], None)
                #     n_sub_used = si_levels_j[level]
                #     rho_new = hermitize(R_fr @ r_f.reshape(2, N, 3, 3) @ R_frd)
                #     H = _H_of(ubar[:9].reshape(3, 3), ubar[9:].reshape(3, 3))         # for the frame QSS
                rho_new = hermitize(R_fr @ _advance(H - K_fr, rho_in) @ R_frd)
            else:
                rho_new = rho_c                                                  # exact classical reduction

            num_after = jnp.sum(jnp.real(rho_new[:, :, idx3, idx3]), axis=-1)
            # number conservation is measured against the collided diagonal that fed src
            # (f_coll_mat under matrix_full)
            num_ref = (num_before if f_coll_mat is None
                       else jnp.sum(f_coll_mat, axis=-1))           # (2,N)
            num_drift = jnp.max(jnp.abs(num_after - num_ref))   # number-conservation residual
            min_f = jnp.min(jnp.real(rho_new[:, :, idx3, idx3]))   # positivity diagnostic
            # diagnostic only: a NaN in the fixed-step scan would reach neff_final unflagged
            step_nan = jnp.any(~jnp.isfinite(rho_new)) | (~jnp.isfinite(T_new))

            # Frames, at slot i // stride on frame steps: the off-diagonal and its adiabatic QSS on
            # the same diagonal.  The QSS (a 9×9 Liouvillian and a 6×6 solve per bin) is only
            # computed on frame steps (lax.cond on a uniform predicate) and reuses the substep's H.
            rho_diag = jnp.real(rho_new[:, :, idx3, idx3])         # (2,N,3)
            off_vec = jnp.stack([rho_new[:, :, a, b] for (a, b) in na._OFF_AB], axis=-1)  # (2,N,6)
            is_fr = (i % stride) == 0
            if mixing:
                def _frame_qss(_):
                    L_oo, L_od = na.offdiag_blocks(H, gamma)
                    return na.qss_offdiag(L_oo, L_od, rho_diag)    # (2,N,6) adiabatic baseline
                ss_vec = jax.lax.cond(is_fr, _frame_qss,
                                      lambda _: jnp.zeros((2, N, 6), dtype=jnp.complex128), None)
            else:
                ss_vec = jnp.zeros((2, N, 6), dtype=jnp.complex128)
            slot = i // stride
            off_buf, ss_buf, diag_buf = bufs
            off_buf = off_buf.at[slot].set(jnp.where(is_fr, off_vec, off_buf[slot]))
            ss_buf = ss_buf.at[slot].set(jnp.where(is_fr, ss_vec, ss_buf[slot]))
            diag_buf = diag_buf.at[slot].set(jnp.where(is_fr, rho_diag, diag_buf[slot]))

            diagn = (num_drift, min_f, T_new, T_cm_mid, step_nan)
            return (T_new, rho_new, (off_buf, ss_buf, diag_buf)), diagn

        lna_left = jnp.linspace(0.0, lna_max, n_steps + 1)[:-1]
        (T_pl_f, rho_f, (off_fr, ss_fr, diag_fr)), diags = jax.lax.scan(
            step, (T_init, rho0, buf0), (lna_left, jnp.arange(n_steps)))

        f_f = jnp.transpose(jnp.real(rho_f[:, :, idx3, idx3]), (2, 0, 1))   # (3,2,N)
        T_cm_f = T_init * jnp.exp(-lna_max)
        neff = D.neff_from_spectrum(grid, f_f, T_cm_f, T_pl_f)

        num_drift, min_f, T_pl_hist, T_cm_hist, step_nan_hist = diags
        neff = eqx.error_if(neff, ~D.plasma_history_ok(T_pl_hist, T_init, T_pl_f / T_cm_f),
                            "fixed-step QKE solve unstable: the plasma temperature left its physical range "
                            "(T_pl above T_init or T_pl/T_cm outside [0.99, 1.45]); increase n_steps")
        any_nan = jnp.any(step_nan_hist)       # True iff any step produced a non-finite ρ/T_pl
        return {
            "neff_final": neff,
            "T_pl_final": T_pl_f,
            "T_cm_final": T_cm_f,
            "ratio_final": T_pl_f / T_cm_f,
            "f_final": f_f,
            "rho_final": rho_f,
            "lna_left": lna_left,
            "T_pl_hist": T_pl_hist,
            "T_cm_hist": T_cm_hist,
            "num_drift": num_drift,            # per-step number-conservation residual (≈ 0)
            "min_f": min_f,                    # per-step min occupation (≥ 0)
            "any_nan": any_nan,                # any non-finite ρ/T_pl over the scan
            # ν–ν pinned: "si_gain", "si_gain_hist", "si_nsub_hist" (the mean-field substep diagnostics)
            "fd_cap_logC": fd_cap_logC,        # active FD-envelope cap exponent (provenance)
            "fd_cap_C": fd_cap_C_used,         # active FD-envelope cap constant C (provenance)
            # downsampled frames: off/ss/diag_fr are end-of-step states, lna_fr the step's start
            "frame_idx": frame_idx,
            "lna_fr": lna_left[frame_idx],
            "T_cm_fr": T_cm_hist[frame_idx],    # step-midpoint T_cm
            # end-of-step T_cm (= T_cm_fr·e^{−dlna/2}), the one that pairs with diag_fr
            "T_cm_end_fr": T_init * jnp.exp(-(lna_left[frame_idx] + dlna)),
            "off_fr": off_fr,                  # (frames,2,N,6) off-diagonal, na._OFF_AB order
            "ss_fr": ss_fr,                    # (frames,2,N,6) adiabatic QSS on the same diagonal
            "diag_fr": diag_fr,                # (frames,2,N,3) diagonal occupations
        }

    return solve, grid
