from __future__ import annotations

import math

from pyrate.constants import (
    BOLTZMANN_J_K,
    LIGHT_SPEED_M_S,
    PLANCK_J_S,
    WAVENUMBER_TO_KJ_MOL,
)
from pyrate.elements import atomic_mass_amu
from pyrate.models import Atom, HinderedRotor, Species
from pyrate.partition import (
    hindered_rotor_log_correction,
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


def test_polyrate_mass_convention_and_isotopes() -> None:
    assert math.isclose(atomic_mass_amu("C", mass_convention="polyrate"), 12.0, rel_tol=1.0e-15)
    assert math.isclose(atomic_mass_amu("C", mass_convention="iupac"), 12.011, rel_tol=1.0e-12)
    assert math.isclose(atomic_mass_amu("2H"), 2.01410177812, rel_tol=1.0e-12)
    assert math.isclose(atomic_mass_amu("13C"), 13.00335483507, rel_tol=1.0e-12)
    assert math.isclose(atomic_mass_amu("37Cl"), 36.965902602, rel_tol=1.0e-12)

    atom_poly = Atom("Cl", 0.0, 0.0, 0.0, mass_convention="polyrate")
    atom_iupac = Atom("Cl", 0.0, 0.0, 0.0, mass_convention="iupac")
    assert math.isclose(atom_poly.mass_amu, 34.96885, rel_tol=1.0e-12)
    assert math.isclose(atom_iupac.mass_amu, 35.45, rel_tol=1.0e-12)


def test_frequency_scale_floor_and_morse_anharmonicity() -> None:
    harmonic = Species(
        name="harm",
        mass_amu_override=20.0,
        rotor_type="atomic",
        frequencies_cm1=(50.0, 1000.0),
        frequency_scale=0.9,
        frequency_floor_cm1=100.0,
    )
    assert harmonic.positive_frequencies_cm1 == (100.0, 900.0)

    anharmonic = Species(
        name="morse",
        mass_amu_override=20.0,
        rotor_type="atomic",
        frequencies_cm1=(1000.0,),
        anharmonicities_cm1=(25.0,),
    )
    zpe_harm = 0.5 * 1000.0 * WAVENUMBER_TO_KJ_MOL
    zpe_morse = (0.5 * 1000.0 - 0.25 * 25.0) * WAVENUMBER_TO_KJ_MOL
    assert math.isclose(zero_point_energy_kj_mol(anharmonic), zpe_morse, rel_tol=1.0e-13)
    assert zero_point_energy_kj_mol(anharmonic) < zpe_harm
    # Anharmonic spacing lowers excited vibrational levels and increases Q_vib
    harm_single = Species(
        name="harm1", mass_amu_override=20.0, rotor_type="atomic", frequencies_cm1=(1000.0,)
    )
    assert log_vibrational_partition(anharmonic, 800.0) > log_vibrational_partition(harm_single, 800.0)


def test_hindered_rotor_pitzer_gwinn_correction() -> None:
    rotor = HinderedRotor(
        frequency_cm1=120.0,
        barrier_kj_mol=8.0,
        symmetry_number=3,
        moment_amu_angstrom2=3.2,
        replaces_lowest_frequency=True,
    )
    species_hr = Species(
        name="ethane-like",
        mass_amu_override=30.0,
        rotor_type="nonlinear",
        moments_of_inertia_amu_a2=(10.0, 25.0, 25.0),
        frequencies_cm1=(120.0, 900.0, 1400.0),
        hindered_rotors=(rotor,),
    )
    species_harm = Species(
        name="ethane-harm",
        mass_amu_override=30.0,
        rotor_type="nonlinear",
        moments_of_inertia_amu_a2=(10.0, 25.0, 25.0),
        frequencies_cm1=(120.0, 900.0, 1400.0),
    )
    corr = hindered_rotor_log_correction(rotor, 500.0)
    assert math.isfinite(corr)
    assert not math.isclose(
        log_vibrational_partition(species_hr, 500.0),
        log_vibrational_partition(species_harm, 500.0),
    )

