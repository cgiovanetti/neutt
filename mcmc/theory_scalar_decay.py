"""Cobaya theory: BBN abundances from an out-of-equilibrium scalar (majoron) → ν decay.

Wraps LINX `make_coupled_bbn(dark=...)` (full neutt transport + LINX nuclear network). The
transport ε-grid + dark scenario + solver are built once in initialize() (static); each MCMC point
varies only the scalar's dynamic leaves (mass m_phi, coupling g, abundance amp) via run(dark=...),
reusing the compiled solve with no recompile.

Sampled inputs (defaults in the yaml):
    omega_b      -- baryon density (-> LINX eta_fac)
    m_phi        -- scalar mass [MeV]
    log10_amp    -- log10 of the frozen-in scalar abundance (x BE shape); injection strength
    log10_g      -- log10 of the majoron coupling (sets the decay epoch); optional (else g_fixed)
Outputs: BBN_Yp, BBN_DH, N_eff_BBN.

The LINX nuclear-rate nuisances + tau_n are fixed at central values for this short demo chain; add
them back as sampled params for a full analysis.
"""
from __future__ import annotations

import math
import os
import sys

from cobaya.theory import Theory
from cobaya.log import LoggedError
from cobaya.component import ComponentNotInstalledError


class scalar_decay_bbn(Theory):
    # ---- paths (prepended to sys.path so neutt / LINX / linx-transport import) ----
    neutt_path = None        # path to the neutt checkout
    linx_transport_path = None  # path to the dark-capable LINX coupling checkout
    linx_repo_path = None       # path to the LINX checkout

    # ---- build-time (STATIC) transport + solver config ----
    # GRID: N=101, eps_range=30. Larger N explodes the phase-2 step count (the redshifting
    # monochromatic decay feature sweeps ever-finer bins), but observables are N-converged to 5
    # digits N=101<->257, so N=101 is ~10x faster at no accuracy cost. eps_range>=35 over-resolves
    # the SM thermal tail in phase 1; 30 is the deepest high-T-safe grid.
    nbins = 101
    eps_range = 30.0
    T_init = 4.0                # MeV; avoids the stiff 10-MeV start (no hybrid needed for dark)
    lna_max = 5.5              # caps the two-leg phase-2 feature-edge stop (lna_stop=min(computed,lna_max))
    nuclear_net = "key_PRIMAT_2023"
    # Two-leg out-of-eq DECAY-tail solve (decay_tail=True): phase 1 keeps solver_atol=1e-30
    # (SM-faithful pure-relative norm) through decoupling; phase 2 floors atol to decay_tail_atol
    # for the decoupled decay tail, whose redshifting monochromatic feature over-resolves a
    # pure-relative tol (too small an lna_max truncates the late decay at the step collapse).
    solver_atol = 1.0e-30
    solver_implicit = False
    solver_dtmax = 0.08
    solver_rtol = 1.0e-6
    solver_max_steps = 50_000   # decay-tail phase 2 can take ~hundreds-thousands of steps; bounds rare grind
    # use_sched=False (the union kernels): use_sched=True (make_coupled_bbn's default) cascades the
    # Tsit5 stepper on the dark-perturbed spectrum and grinds to max_steps. Forwarded to make_coupled_bbn.
    solver_use_sched = False
    # two-leg decay-tail knobs (forwarded to make_transport_solver via make_coupled_bbn)
    decay_tail = True
    decay_handoff_tcm = 0.20    # MeV; phase1->2 handoff (above the decay epoch, below nu-decoupling relevance)
    decay_tail_atol = 1.0e-10   # phase-2 atol floor (accuracy-neutral: floored bins carry ~1e-3 of rho_nu)
    decay_stop_safety = 0.9     # phase 2 stops where the feature edge eps=(m_x/2)/T_cm reaches 0.9*eps_max
    # ---- compile-stall fix + recommended production combo ----
    # use_direct_adjoint=True swaps diffrax's default RecursiveCheckpointAdjoint (whose XLA compile
    # stochastically hangs 10-40 min CPU-bound, a chain-killer on any recompile) for DirectAdjoint:
    # the forward solution is unchanged, only reverse-mode AD differs (irrelevant for a gradient-free
    # Metropolis chain). Paired with decay_tail_hardened=False (the legacy atol floor): the hardened
    # down-weighted norm has its own ~350s per-step pytree-reconstruction compile cost that
    # DirectAdjoint does not remove, and the feature is step-size-converged either way. Tradeoff:
    # ~2x warm. n_save=64 (not make_coupled_bbn's 256 default): DirectAdjoint compile scales with
    # #SaveAt(ts) pts and 64 is N-converged for the smooth H.
    use_direct_adjoint = True
    decay_tail_hardened = False
    n_save = 64

    # ---- scenario defaults (used when a param is not sampled) ----
    g_fixed = 7.0e-12          # majoron coupling -> decay near ~100 keV
    m_phi_fixed = 1.0          # MeV
    # robustness guard: outside the explicit-solver range (m_phi >~ 1.2) the under-resolved high-eps
    # decay feature can return a FINITE-but-garbage N_eff (e.g. 24.5) rather than a NaN/throw, so
    # cobaya's stop_at_error does not catch it. A point whose transport output is non-finite or
    # unphysically large is rejected here (calculate -> False) so a stray sample becomes a rejected
    # point instead of polluting the posterior. Physical N_eff for this neutrinophilic scalar decay
    # is ~3.04-3.4; the ceiling has huge margin yet catches the ~20+ blow-ups.
    neff_ceiling = 6.0
    # Energy-space safety BACKSTOP: the phase-2 step controller measures OCCUPATION error but ρ_ν is
    # ε³-weighted, so an under-resolved high-ε feature can be ε³-amplified into a garbage N_eff (the
    # m_φ≳1.2 grid-edge pile-up puts most of ρ_ν in the spurious feature). out's eps_tail_energy_frac
    # = ρ_ν fraction in the down-weighted ε≥13 region (≈0.5% at m_φ=1.0). Set GENEROUSLY (0.2 = 40x
    # the nominal tail) so it ONLY catches the catastrophic pile-up, not valid high-amp/edge points
    # whose feature legitimately carries a few % of ρ_ν. Tighten only after inspecting the per-point
    # eps_tail_energy_frac derived output across the prior.
    eps_tail_frac_ceiling = 0.2

    def initialize(self):
        for p in (self.neutt_path, self.linx_transport_path):
            if p:
                p = os.path.abspath(os.path.expanduser(p))
                if p not in sys.path:
                    sys.path.insert(0, p)
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp
            import diffrax
            import neutt  # float64
            from neutt.grid import make_eps_grid
            from neutt import dark as darkmod
            from linx.coupled_bbn import make_coupled_bbn
            import linx.const as linx_const
            from linx.nuclear import NuclearRates
        except Exception as e:
            raise ComponentNotInstalledError(
                self.log, "Could not import neutt/LINX/JAX. Original error: %r", e)

        self._jnp = jnp
        self._dark = darkmod
        self._const = linx_const

        # build-time STATIC grids (never varied per call)
        self._nu_grid = make_eps_grid(self.nbins, self.eps_range)
        self._x_grid = make_eps_grid(self.nbins, self.eps_range, eps_offset=0.0025,
                                     bin1_weight_frac=0.25)
        # template majoron sector: fixes spin_dof(=1) + grid structure for the compiled solve
        template = darkmod.make_majoron_sector(m_j=self.m_phi_fixed, g=self.g_fixed,
                                               grid=self._x_grid)
        solver_kwargs = {"hybrid": False, "implicit": bool(self.solver_implicit),
                         "atol": float(self.solver_atol), "rtol": float(self.solver_rtol),
                         "dtmax": float(self.solver_dtmax), "lna_max": float(self.lna_max),
                         "max_steps": int(self.solver_max_steps),
                         # two-leg out-of-eq decay-tail solve (captures the full late decay)
                         "decay_tail": bool(self.decay_tail),
                         "decay_handoff_tcm": float(self.decay_handoff_tcm),
                         "decay_tail_atol": float(self.decay_tail_atol),
                         "decay_stop_safety": float(self.decay_stop_safety),
                         # compile-stall fix + recommended combo (see the class-attr note above)
                         "decay_tail_hardened": bool(self.decay_tail_hardened),
                         "decay_adjoint": (diffrax.DirectAdjoint()
                                           if self.use_direct_adjoint else None)}
        self._run, self._info = make_coupled_bbn(
            nuclear_net=self.nuclear_net, transport="full", dark=template,
            grid=self._nu_grid, n_save=int(self.n_save),
            use_sched=bool(self.solver_use_sched),
            solver_kwargs=solver_kwargs)

        # fixed nuclear nuisances (central values) for this short chain
        self._n_q = len(NuclearRates(nuclear_net=self.nuclear_net).reactions)
        self._q0 = jnp.zeros(self._n_q)

    def get_can_support_params(self):
        return ["omega_b", "m_phi", "log10_amp", "log10_g"]

    def get_can_provide(self):
        return ["BBN_Yp", "BBN_DH", "N_eff_BBN",
                "eps_tail_energy_frac", "rho_x_over_rho_nu_final"]

    def calculate(self, state, want_derived=True, **params):
        jnp = self._jnp
        if "omega_b" not in params:
            raise LoggedError(self.log, "scalar_decay_bbn requires 'omega_b'.")
        omega_b = params["omega_b"]
        m_phi = float(params.get("m_phi", self.m_phi_fixed))
        g = 10.0 ** params["log10_g"] if "log10_g" in params else self.g_fixed
        amp = 10.0 ** params.get("log10_amp", math.log10(3e-3))

        # per-point sector: SAME x_grid + spin_dof as the template -> reuses the compiled solve
        sec = self._dark.make_majoron_sector(m_j=m_phi, g=g, grid=self._x_grid)
        f_x0 = amp * self._dark.dark_be_equil(self._x_grid.abscissas, m_phi / self.T_init)
        eta_fac = omega_b / self._const.Omegabh2

        (abund, neff, diag) = self._run(dark=sec, f_x0=f_x0, T_init=self.T_init,
                                        eta_fac=eta_fac, nuclear_rates_q=self._q0,
                                        return_diag=True)
        # LINX 8-vector: (Yn, Yp, Yd, Yt, YHe3, Ya, YLi7, YBe7)
        Yp = float(4.0 * abund[5])
        DH = float((abund[2] / abund[1]) * 1.0e5)
        Neff = float(neff)
        # transport safety metrics (the ε³ energy-space monitor + leftover-ρ_X gate)
        eps_tail = float(diag.get("eps_tail_energy_frac", 0.0))
        rho_x_ratio = float(diag.get("rho_x_over_rho_nu_final", 0.0))

        # reject non-finite / unphysical transport output (see neff_ceiling note above) AND points
        # where real ρ_ν energy is hiding in the down-weighted high-ε bins (see eps_tail_frac_ceiling)
        if not (math.isfinite(Yp) and math.isfinite(DH) and math.isfinite(Neff)) \
                or not (0.0 < Neff < self.neff_ceiling) or not (0.0 < Yp < 0.5) \
                or not (eps_tail < self.eps_tail_frac_ceiling):
            self.log.warning(
                "scalar_decay_bbn: rejecting unphysical point "
                "(m_phi=%.4g Yp=%.4g DH=%.4g N_eff=%.4g eps_tail=%.3g)",
                m_phi, Yp, DH, Neff, eps_tail)
            return False

        state["BBN_Yp"] = Yp
        state["BBN_DH"] = DH
        state["N_eff_BBN"] = Neff
        if want_derived:
            d = state.setdefault("derived", {})
            d["BBN_Yp"] = Yp
            d["BBN_DH"] = DH
            d["N_eff_BBN"] = Neff
            d["eps_tail_energy_frac"] = eps_tail
            d["rho_x_over_rho_nu_final"] = rho_x_ratio
        return True
