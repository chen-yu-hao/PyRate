from __future__ import annotations

import math

from pyrate.constants import (
    BOLTZMANN_J_K,
    LIGHT_SPEED_M_S,
    PLANCK_J_S,
    WAVENUMBER_TO_KJ_MOL,
)
from pyrate.models import Species
from pyrate.partition import (
    log_rotational_partition,
    log_vibrational_partition,
    partition_components,
    zero_point_energy_kj_mol,
)


def test_harmonic_partition_single_mode() -> None:
    species = Species(
        name="oscillator",
        mass_amu_override=10.0,
        rotor_type="atomic",
        frequencies_cm1=(1000.0,),
    )
    temperature = 300.0
    x = PLANCK_J_S * LIGHT_SPEED_M_S * 100.0 * 1000.0 / (
        BOLTZMANN_J_K * temperature
    )
    expected = -math.log1p(-math.exp(-x))
    assert math.isclose(
        log_vibrational_partition(species, temperature), expected, rel_tol=1.0e-13
    )


def test_zero_point_energy() -> None:
    species = Species(
        name="two modes",
        mass_amu_override=10.0,
        rotor_type="atomic",
        frequencies_cm1=(500.0, 1000.0, -200.0),
    )
    assert math.isclose(
        zero_point_energy_kj_mol(species),
        0.5 * 1500.0 * WAVENUMBER_TO_KJ_MOL,
        rel_tol=1.0e-14,
    )


def test_rotational_symmetry_number_divides_partition() -> None:
    common = dict(
        mass_amu_override=28.0,
        rotor_type="linear",
        moments_of_inertia_amu_a2=(10.0,),
    )
    asymmetric = Species(name="AB", symmetry_number=1, **common)
    symmetric = Species(name="A2", symmetry_number=2, **common)
    difference = log_rotational_partition(asymmetric, 300.0) - log_rotational_partition(
        symmetric, 300.0
    )
    assert math.isclose(difference, math.log(2.0), rel_tol=1.0e-13)


def test_partition_components_are_finite() -> None:
    species = Species(
        name="nonlinear",
        mass_amu_override=30.0,
        rotor_type="nonlinear",
        moments_of_inertia_amu_a2=(10.0, 12.0, 15.0),
        frequencies_cm1=(300.0, 800.0, 1200.0),
    )
    components = partition_components(species, 500.0)
    assert math.isfinite(components.total)
    assert components.translational > 0.0
