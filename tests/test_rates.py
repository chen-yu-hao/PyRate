from __future__ import annotations

import math

from pyrate.constants import (
    BOLTZMANN_J_K,
    DEFAULT_STANDARD_PRESSURE_PA,
    PLANCK_J_S,
    standard_concentration_mol_m3,
)
from pyrate.models import PathPoint, Reaction, ReactionPath, Species
from pyrate.rates import cvt_rate_si, tst_rate_from_activation_gibbs_si


def model_species(name: str, energy: float) -> Species:
    return Species(
        name=name,
        mass_amu_override=30.0,
        rotor_type="nonlinear",
        moments_of_inertia_amu_a2=(10.0, 15.0, 20.0),
        electronic_energy_kj_mol=energy,
        frequencies_cm1=(500.0, 800.0, 1200.0),
    )


def test_zero_barrier_unimolecular_eyring_prefactor() -> None:
    temperature = 300.0
    rate = tst_rate_from_activation_gibbs_si(
        0.0, temperature, 1, DEFAULT_STANDARD_PRESSURE_PA
    )
    assert math.isclose(
        rate, BOLTZMANN_J_K * temperature / PLANCK_J_S, rel_tol=1.0e-14
    )


def test_zero_barrier_bimolecular_has_standard_concentration_factor() -> None:
    temperature = 300.0
    rate = tst_rate_from_activation_gibbs_si(
        0.0, temperature, 2, DEFAULT_STANDARD_PRESSURE_PA
    )
    expected = (BOLTZMANN_J_K * temperature / PLANCK_J_S) / standard_concentration_mol_m3(
        temperature
    )
    assert math.isclose(rate, expected, rel_tol=1.0e-14)


def test_cvt_finds_refined_free_energy_maximum() -> None:
    reactant = model_species("reactant", 0.0)
    transition_state = model_species("ts", 50.0)
    coordinates = (-1.0, -0.5, 0.0, 0.5, 1.0)
    points = tuple(
        PathPoint(
            coordinate=s,
            dividing_surface=model_species("path", 60.0 - 8.0 * (s - 0.25) ** 2),
        )
        for s in coordinates
    )
    reaction = Reaction(
        reactants=(reactant,),
        transition_state=transition_state,
        path=ReactionPath(points),
    )
    result = cvt_rate_si(reaction, 400.0)
    assert math.isclose(result.coordinate, 0.25, abs_tol=2.0e-3)
    assert result.fit_points == 5
    assert result.activation_gibbs_kj_mol > result.sampled_activation_gibbs_kj_mol
