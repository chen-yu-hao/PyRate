"""Rigid-rotor/harmonic-oscillator molecular partition functions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import (
    AMU_KG,
    ANGSTROM_M,
    AVOGADRO_MOL,
    BOLTZMANN_J_K,
    DEFAULT_STANDARD_PRESSURE_PA,
    GAS_CONSTANT_J_MOL_K,
    PLANCK_J_S,
    WAVENUMBER_TO_KJ_MOL,
    standard_volume_per_molecule_m3,
)
from .geometry import rotor_geometry
from .models import Species
from .native import harmonic_log_partition


@dataclass(frozen=True, slots=True)
class PartitionComponents:
    """Natural logarithms of dimensionless molecular partition functions."""

    translational: float
    rotational: float
    vibrational: float
    electronic: float

    @property
    def total(self) -> float:
        return self.translational + self.rotational + self.vibrational + self.electronic

    def as_dict(self) -> dict[str, float]:
        return {
            "log_q_trans": self.translational,
            "log_q_rot": self.rotational,
            "log_q_vib": self.vibrational,
            "log_q_elec": self.electronic,
            "log_q_total": self.total,
        }


def log_translational_partition(
    mass_amu: float,
    temperature_k: float,
    standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA,
) -> float:
    """Log ideal-gas translational partition function at a standard pressure."""

    if mass_amu <= 0.0:
        raise ValueError("mass_amu must be positive")
    if temperature_k <= 0.0 or standard_pressure_pa <= 0.0:
        raise ValueError("temperature and standard pressure must be positive")
    mass_kg = mass_amu * AMU_KG
    volume = standard_volume_per_molecule_m3(temperature_k, standard_pressure_pa)
    log_density_factor = 1.5 * math.log(
        2.0 * math.pi * mass_kg * BOLTZMANN_J_K * temperature_k / PLANCK_J_S**2
    )
    return log_density_factor + math.log(volume)


def log_rotational_partition(species: Species, temperature_k: float) -> float:
    """Log classical rigid-rotor partition function including symmetry number."""

    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    geometry = rotor_geometry(species)
    sigma = float(species.symmetry_number)
    if geometry.rotor_type == "atomic":
        return 0.0

    conversion = AMU_KG * ANGSTROM_M**2
    moments = tuple(moment * conversion for moment in geometry.moments_amu_a2)
    prefactor = 8.0 * math.pi**2 * BOLTZMANN_J_K * temperature_k / PLANCK_J_S**2

    if geometry.rotor_type == "linear":
        nonzero = [moment for moment in moments if moment > 1.0e-52]
        if not nonzero:
            raise ValueError(f"linear species {species.name!r} has zero rotational inertia")
        inertia = sum(nonzero) / len(nonzero)
        q_rot = prefactor * inertia / sigma
    else:
        if any(moment <= 1.0e-52 for moment in moments):
            raise ValueError(f"nonlinear species {species.name!r} requires three nonzero moments")
        q_rot = math.sqrt(math.pi) * (prefactor**1.5) * math.sqrt(
            moments[0] * moments[1] * moments[2]
        ) / sigma
    if q_rot <= 0.0 or not math.isfinite(q_rot):
        raise ValueError(f"invalid rotational partition function for {species.name!r}")
    return math.log(q_rot)


def log_vibrational_partition(species: Species, temperature_k: float) -> float:
    """Log harmonic vibrational partition function, excluding zero-point energy."""

    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    return harmonic_log_partition(species.positive_frequencies_cm1, temperature_k)


def zero_point_energy_kj_mol(species: Species) -> float:
    """Return harmonic zero-point energy from positive frequencies."""

    if not species.include_zpe:
        return 0.0
    return 0.5 * WAVENUMBER_TO_KJ_MOL * sum(species.positive_frequencies_cm1)


def log_electronic_partition(species: Species, temperature_k: float) -> float:
    """Log electronic partition function relative to its lowest state."""

    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    inverse_rt = 1000.0 / (GAS_CONSTANT_J_MOL_K * temperature_k)
    terms = [
        state.degeneracy * math.exp(-state.energy_kj_mol * inverse_rt)
        for state in species.electronic_states
    ]
    partition = sum(terms)
    if partition <= 0.0:
        raise ValueError(f"electronic partition function vanished for {species.name!r}")
    return math.log(partition)


def partition_components(
    species: Species,
    temperature_k: float,
    standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA,
) -> PartitionComponents:
    """Return all standard-state partition-function components."""

    return PartitionComponents(
        translational=log_translational_partition(
            species.mass_amu, temperature_k, standard_pressure_pa
        ),
        rotational=log_rotational_partition(species, temperature_k),
        vibrational=log_vibrational_partition(species, temperature_k),
        electronic=log_electronic_partition(species, temperature_k),
    )


def standard_gibbs_energy_kj_mol(
    species: Species,
    temperature_k: float,
    standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA,
) -> float:
    """Return the RRHO molecular standard Gibbs energy.

    The result includes the supplied electronic energy and, by default, the
    harmonic zero-point energy.  The finite-temperature contribution is
    ``-R T ln(q)`` with the translational partition function evaluated at the
    ideal-gas volume per molecule corresponding to ``standard_pressure_pa``.
    """

    log_partition = partition_components(
        species, temperature_k, standard_pressure_pa
    ).total
    thermal_kj_mol = (
        GAS_CONSTANT_J_MOL_K * temperature_k * log_partition / 1000.0
    )
    return species.electronic_energy_kj_mol + zero_point_energy_kj_mol(species) - thermal_kj_mol


def molecular_partition_function(
    species: Species,
    temperature_k: float,
    standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA,
) -> float:
    """Return the full dimensionless standard-state molecular partition function."""

    return math.exp(partition_components(species, temperature_k, standard_pressure_pa).total)


def frequency_energy_j_per_molecule(frequency_cm1: float) -> float:
    """Convert a wavenumber to one quantum of energy per molecule."""

    return abs(frequency_cm1) * WAVENUMBER_TO_KJ_MOL * 1000.0 / AVOGADRO_MOL
