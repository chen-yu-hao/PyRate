"""Physical constants and unit factors used throughout PyRate.

The defining SI constants are exact.  Keeping them in one module avoids the
silent mixture of cgs, atomic, and SI units that is common in legacy kinetics
codes.
"""

from __future__ import annotations

import math

# Exact SI defining constants.
BOLTZMANN_J_K = 1.380_649e-23
PLANCK_J_S = 6.626_070_15e-34
HBAR_J_S = PLANCK_J_S / (2.0 * math.pi)
AVOGADRO_MOL = 6.022_140_76e23
LIGHT_SPEED_M_S = 299_792_458.0
GAS_CONSTANT_J_MOL_K = BOLTZMANN_J_K * AVOGADRO_MOL

# Unit conversions.
AMU_KG = 1.660_539_066_60e-27
ANGSTROM_M = 1.0e-10
BOHR_M = 5.291_772_109_03e-11
HARTREE_J = 4.359_744_722_207_1e-18
HARTREE_KJ_MOL = HARTREE_J * AVOGADRO_MOL / 1000.0
WAVENUMBER_TO_J = PLANCK_J_S * LIGHT_SPEED_M_S * 100.0
WAVENUMBER_TO_KJ_MOL = WAVENUMBER_TO_J * AVOGADRO_MOL / 1000.0
CAL_TO_J = 4.184
BAR_PA = 100_000.0
ATM_PA = 101_325.0

DEFAULT_STANDARD_PRESSURE_PA = BAR_PA


def standard_concentration_mol_m3(
    temperature_k: float, pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA
) -> float:
    """Return the ideal-gas standard concentration ``p/(R T)`` in mol m⁻³."""

    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if pressure_pa <= 0.0:
        raise ValueError("pressure must be positive")
    return pressure_pa / (GAS_CONSTANT_J_MOL_K * temperature_k)


def standard_volume_per_molecule_m3(
    temperature_k: float, pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA
) -> float:
    """Return the ideal-gas volume per molecule at the standard pressure."""

    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if pressure_pa <= 0.0:
        raise ValueError("pressure must be positive")
    return BOLTZMANN_J_K * temperature_k / pressure_pa

# Available constant sets are explicit, avoiding silent changes to legacy runs.
# The current physics stack uses module-level CODATA/SI constants throughout;
# selecting a different profile is rejected until every kernel accepts it.
CONSTANTS_PROFILES: tuple[str, ...] = ("codata2018",)


def validate_constants_profile(profile: str) -> str:
    """Validate a named, reproducible constants profile."""

    normalized = profile.strip().lower()
    if normalized not in CONSTANTS_PROFILES:
        raise ValueError(
            f"unsupported constants_profile {profile!r}; available: {CONSTANTS_PROFILES}"
        )
    return normalized
