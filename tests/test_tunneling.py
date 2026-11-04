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
