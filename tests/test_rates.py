from __future__ import annotations

import math
from collections.abc import Callable

import pytest

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


def quadratic_path_reaction(
    coordinates: tuple[float, ...], energy: Callable[[float], float]
) -> Reaction:
    points = tuple(
        PathPoint(coordinate=s, dividing_surface=model_species("path", energy(s)))
        for s in coordinates
    )
    return Reaction(
        reactants=(model_species("reactant", 0.0),),
        transition_state=model_species("ts", 50.0),
        path=ReactionPath(points),
    )


def test_cvt_finds_refined_free_energy_maximum() -> None:
    # Path points differ only in electronic energy, so ΔG(s) is exactly
    # quadratic and the three-point parabola recovers the true bottleneck.
    reaction = quadratic_path_reaction(
        (-1.0, -0.5, 0.0, 0.5, 1.0), lambda s: 60.0 - 8.0 * (s - 0.25) ** 2
    )
    result = cvt_rate_si(reaction, 400.0)
    assert math.isclose(result.coordinate, 0.25, abs_tol=1.0e-12)
    assert result.fit_points == 3
    assert result.sampled_coordinate == 0.0
    assert result.activation_gibbs_kj_mol > result.sampled_activation_gibbs_kj_mol
    assert math.isclose(
        result.activation_gibbs_kj_mol - result.sampled_activation_gibbs_kj_mol,
        8.0 * 0.25**2,
        rel_tol=1.0e-10,
    )


def test_cvt_opt_in_five_point_fit() -> None:
    reaction = quadratic_path_reaction(
        (-1.0, -0.5, 0.0, 0.5, 1.0), lambda s: 60.0 - 8.0 * (s - 0.25) ** 2
    )
    default = cvt_rate_si(reaction, 400.0)
    quartic = cvt_rate_si(reaction, 400.0, max_fit_points=5)
    assert quartic.fit_points == 5
    assert math.isclose(quartic.coordinate, 0.25, abs_tol=1.0e-12)
    assert math.isclose(quartic.rate_si, default.rate_si, rel_tol=1.0e-12)


def test_cvt_falls_back_to_sampled_bottleneck_when_fit_exceeds_bound() -> None:
    # A steep rise into a wide plateau: the parabola would overshoot the drop to
    # the lower neighbour, so the sampled dividing surface is kept.
    reaction = quadratic_path_reaction(
        (-0.01, 0.0, 1.0), {-0.01: 59.0, 0.0: 60.0, 1.0: 60.0}.__getitem__
    )
    result = cvt_rate_si(reaction, 300.0)
    assert result.fit_points == 1
    assert result.coordinate == result.sampled_coordinate == 0.0
    assert result.activation_gibbs_kj_mol == result.sampled_activation_gibbs_kj_mol
    expected = tst_rate_from_activation_gibbs_si(
        result.sampled_activation_gibbs_kj_mol, 300.0, 1, DEFAULT_STANDARD_PRESSURE_PA
    )
    assert math.isclose(result.rate_si, expected, rel_tol=1.0e-15)


def test_cvt_rejects_invalid_fit_points() -> None:
    reaction = quadratic_path_reaction((-1.0, 0.0, 1.0), lambda s: 60.0 - s * s)
    with pytest.raises(ValueError, match="max_points"):
        cvt_rate_si(reaction, 300.0, max_fit_points=7)
