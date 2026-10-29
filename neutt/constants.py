"""Physical and mathematical constants (MeV, s), bit-identical to BURST's except sin²θ_W."""

# --- physical constants ---
HBAR = 6.582119514e-22       # MeV·s
ALPHA_FS = 7.2973525698e-03  # fine-structure constant
MPL = 1.220932e22            # Planck mass [MeV]
M_ELECTRON = 0.510998928     # [MeV]

# --- weak interactions ---
GFHBAR = 0.20668190937       # G_F²/ℏ [s⁻¹ MeV⁻⁵], the collision-rate scale
GFERMI = 1.1663787e-11       # G_F [MeV⁻²]
SIN2THW = 0.23122            # sin²θ_W (MS-bar at M_Z), in the ν–e kernels and the QKE sector
SIN2THW_BURST = 0.23         # BURST's value; only the Fortran-parity oracle tests use it
_MASS_W = 80379.0            # M_W [MeV]
_MASS_Z = 91187.6            # M_Z [MeV]
GFW2HBAR = GFERMI / _MASS_W**2 / HBAR   # G_F/M_W²/ℏ [MeV⁻⁵ s⁻¹], thermal MSW potential
GFZ2HBAR = GFERMI / _MASS_Z**2 / HBAR   # G_F/M_Z²/ℏ [MeV⁻⁵ s⁻¹], second-order ν–ν potential

# --- neutrino mixing (normal ordering) ---
SIN2_THETA12 = 0.307
SIN2_THETA13 = 0.0212
SIN2_THETA23 = 0.417
DELTA_M2_21 = 7.53e-17       # [MeV²]
DELTA_M2_31 = 2.59e-15       # [MeV²]

# --- mathematical constants ---
PI = 3.1415926535897932
ZETA3 = 1.202056903159594

# composite Boole's rule: interior weights (×h) cycled over panels, end points 14/45
BR_COEFFS = (64.0 / 45.0, 24.0 / 45.0, 64.0 / 45.0, 28.0 / 45.0)
BR_END = 14.0 / 45.0

# Occupation cutoff hardcoded at E/T = 300.
_FD_CUTOFF = 300.0