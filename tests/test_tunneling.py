from __future__ import annotations

import math

from pyrate.constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL
from pyrate.models import Species, TunnelingProfile
from pyrate.native import thermal_wkb_kappa, wigner_kappa
from pyrate.tunneling import profile_transmission_probability, transition_state_wigner_kappa


def test_wigner_wrapper() -> None:
    ts = Species(
        name="ts",
        mass_amu_override=10.0,
        rotor_type="atomic",
        frequencies_cm1=(500.0, -1000.0),
    )
    assert math.isclose(
        transition_state_wigner_kappa(ts, 300.0),
        wigner_kappa(1000.0, 300.0),
        rel_tol=1.0e-14,
    )


def test_flat_profile_has_unit_correction() -> None:
    coordinates = (-ANGSTROM_M, 0.0, ANGSTROM_M)
    potentials = (0.0, 0.0, 0.0)
    assert thermal_wkb_kappa(coordinates, potentials, 300.0, AMU_KG, 100) == 1.0


def test_barrier_profile_has_tunneling_enhancement() -> None:
    profile = TunnelingProfile(
        coordinates_angstrom=(-1.0, -0.5, 0.0, 0.5, 1.0),
        energies_kj_mol=(0.0, 12.0, 25.0, 12.0, 0.0),
        reduced_mass_amu=1.0,
        energy_grid_points=200,
    )
    low = profile_transmission_probability(profile, 1.0)
    high = profile_transmission_probability(profile, 20.0)
    assert 0.0 <= low < high < 1.0

    energies_j = tuple(value * 1000.0 / AVOGADRO_MOL for value in profile.energies_kj_mol)
    kappa = thermal_wkb_kappa(
        tuple(value * ANGSTROM_M for value in profile.coordinates_angstrom),
        energies_j,
        300.0,
        AMU_KG,
        200,
    )
    assert kappa > 1.0
    assert math.isfinite(kappa)


def test_uniform_wkb_reproduces_bell_parabolic_barrier() -> None:
    from pyrate.constants import BOLTZMANN_J_K, HBAR_J_S, LIGHT_SPEED_M_S

    omega = 2 * math.pi * LIGHT_SPEED_M_S * 100 * 1000.0
    height = 200e3 / AVOGADRO_MOL
    half_width = math.sqrt(2 * height / (AMU_KG * omega**2))
    n = 4001
    x = tuple(-half_width + 2 * half_width * i / (n - 1) for i in range(n))
    v = tuple(height - 0.5 * AMU_KG * omega**2 * xi * xi for xi in x)
    for temperature in (600.0, 1500.0):
        u = HBAR_J_S * omega / (BOLTZMANN_J_K * temperature)
        bell = (u / 2) / math.sin(u / 2)
        assert math.isclose(thermal_wkb_kappa(x, v, temperature, AMU_KG, 2000), bell, rel_tol=1e-5)


def test_primitive_form_is_still_available_and_larger_at_high_temperature() -> None:
    x = tuple(i * 1e-12 for i in range(-200, 201))
    v = tuple(4e-20 * max(0.0, 1 - (xi / 2e-10) ** 2) for xi in x)
    uniform = thermal_wkb_kappa(x, v, 1500.0, AMU_KG, 400)
    primitive = thermal_wkb_kappa(x, v, 1500.0, AMU_KG, 400, "primitive")
    assert 1.0 < uniform < primitive


def test_uniform_probability_is_half_at_barrier_top() -> None:
    profile = TunnelingProfile(
        coordinates_angstrom=(-1.0, 0.0, 1.0), energies_kj_mol=(0.0, 25.0, 0.0), reduced_mass_amu=1.0
    )
    assert math.isclose(profile_transmission_probability(profile, 25.0), 0.5)
    low, high = profile_transmission_probability(profile, 20.0), profile_transmission_probability(profile, 30.0)
    assert math.isclose(low + high, 1.0, rel_tol=1e-12)
