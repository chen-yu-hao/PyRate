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
    LIGHT_SPEED_M_S,
    PLANCK_J_S,
    WAVENUMBER_TO_KJ_MOL,
    standard_volume_per_molecule_m3,
)
from .geometry import rotor_geometry
from .models import HinderedRotor, Species
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


def _scaled_bessel_i0(u: float) -> float:
    """Return exp(-u) * I_0(u) for u >= 0 using Abramowitz & Stegun / Polyrate BES0."""

    x = abs(float(u))
    if x < 3.75:
        t = (x / 3.75) ** 2
        i0 = 1.0 + t * (
            3.5156229
            + t
            * (
                3.0899424
                + t
                * (
                    1.2067492
                    + t * (0.2659732 + t * (0.0360768 + t * 0.0045813))
                )
            )
        )
        return math.exp(-x) * i0
    t = 3.75 / x
    poly = (
        0.39894228
        + t
        * (
            0.01328592
            + t
            * (
                0.00225319
                + t
                * (
                    -0.00157565
                    + t
                    * (
                        0.00916281
                        + t
                        * (
                            -0.02057706
                            + t
                            * (
                                0.02635537
                                + t * (-0.01647633 + t * 0.00392377)
                            )
                        )
                    )
                )
            )
        )
    )
    return poly / math.sqrt(x)


def morse_mode_log_partition(
    frequency_cm1: float, anharmonicity_cm1: float, temperature_k: float
) -> float:
    """Return log partition function for a 1D Morse oscillator relative to v=0."""

    if frequency_cm1 <= 0.0:
        return 0.0
    if anharmonicity_cm1 <= 0.0:
        return harmonic_log_partition((frequency_cm1,), temperature_k)
    v_max = max(0, int(math.floor(frequency_cm1 / (2.0 * anharmonicity_cm1) - 0.5)))
    hc_over_kt = PLANCK_J_S * LIGHT_SPEED_M_S * 100.0 / (BOLTZMANN_J_K * temperature_k)
    total = 1.0
    for v in range(1, min(v_max, 5000) + 1):
        delta_cm1 = v * frequency_cm1 - v * (v + 1) * anharmonicity_cm1
        if delta_cm1 <= 0.0:
            break
        term = math.exp(-delta_cm1 * hc_over_kt)
        total += term
        if term < 1.0e-15 * total:
            break
    return math.log(total)


def hindered_rotor_log_correction(rotor: HinderedRotor, temperature_k: float) -> float:
    """Return log(Q_HR / Q_HO) for a 1D hindered internal rotor (Pitzer-Gwinn / Polyrate BES0)."""

    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    u = (rotor.barrier_kj_mol * 1000.0) / (2.0 * GAS_CONSTANT_J_MOL_K * temperature_k)
    bessel_factor = _scaled_bessel_i0(u)
    q_class_ho = BOLTZMANN_J_K * temperature_k / (
        PLANCK_J_S * LIGHT_SPEED_M_S * 100.0 * rotor.frequency_cm1
    )
    if rotor.moment_amu_angstrom2 is not None:
        inertia_kg_m2 = rotor.moment_amu_angstrom2 * AMU_KG * (ANGSTROM_M**2)
        q_fr = (
            math.sqrt(2.0 * math.pi * inertia_kg_m2 * BOLTZMANN_J_K * temperature_k)
            / (float(rotor.symmetry_number) * PLANCK_J_S)
        )
    elif u > 0.0:
        q_fr = q_class_ho * math.sqrt(2.0 * math.pi * u)
    else:
        q_fr = q_class_ho
    ratio = (q_fr / q_class_ho) * bessel_factor
    return math.log(max(ratio, 1.0e-300))


def log_vibrational_partition(species: Species, temperature_k: float) -> float:
    """Log vibrational partition function (with optional Morse anharmonicity and hindered rotors)."""

    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    frequencies = species.positive_frequencies_cm1
    if species.anharmonicities_cm1:
        total = 0.0
        for idx, freq in enumerate(frequencies):
            x_e = (
                species.anharmonicities_cm1[idx]
                if idx < len(species.anharmonicities_cm1)
                else 0.0
            )
            total += morse_mode_log_partition(freq, x_e, temperature_k)
    else:
        total = harmonic_log_partition(frequencies, temperature_k)

    for rotor in species.hindered_rotors:
        total += hindered_rotor_log_correction(rotor, temperature_k)
    return total


def zero_point_energy_kj_mol(species: Species) -> float:
    """Return vibrational zero-point energy from positive frequencies (including anharmonicity)."""

    if not species.include_zpe:
        return 0.0
    frequencies = species.positive_frequencies_cm1
    zpe_cm1 = 0.5 * sum(frequencies)
    if species.anharmonicities_cm1:
        for idx, freq in enumerate(frequencies):
            if idx < len(species.anharmonicities_cm1) and species.anharmonicities_cm1[idx] > 0.0:
                zpe_cm1 -= 0.25 * min(species.anharmonicities_cm1[idx], 0.5 * freq)
    return WAVENUMBER_TO_KJ_MOL * zpe_cm1


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
