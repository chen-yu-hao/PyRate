from __future__ import annotations

import math

from pyrate.calculation import calculate_rates
from pyrate.constants import (
    BOLTZMANN_J_K,
    DEFAULT_STANDARD_PRESSURE_PA,
    PLANCK_J_S,
    standard_concentration_mol_m3,
)
from pyrate.models import CalculationOptions, PathPoint, Reaction, ReactionPath, Species
from pyrate.rates import (
    conventional_tst_rate_si,
    cus_rate_si,
    cvt_rate_si,
    muvt_and_us_rates_si,
    reaction_equilibrium_constants,
    tst_rate_from_activation_gibbs_si,
)


def model_species(name: str, energy: float, freqs: tuple[float, ...] = (500.0, 800.0, 1200.0)) -> Species:
    return Species(
        name=name,
        mass_amu_override=30.0,
        rotor_type="nonlinear",
        moments_of_inertia_amu_a2=(10.0, 15.0, 20.0),
        electronic_energy_kj_mol=energy,
        frequencies_cm1=freqs,
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


def test_cus_muvt_and_us_on_double_barrier_path() -> None:
    reactant = model_species("reactant", 0.0)
    product = model_species("product", -10.0)
    transition_state = model_species("ts", 45.0, (500.0, 800.0, 1200.0, -900.0))
    # Double-peaked profile with an intermediate complex well at s = 0.0
    coords_and_energies = (
        (-1.0, 20.0),
        (-0.5, 50.0),
        (0.0, 35.0),
        (0.5, 48.0),
        (1.0, 15.0),
    )
    points = tuple(
        PathPoint(coordinate=s, dividing_surface=model_species(f"p-{i}", e))
        for i, (s, e) in enumerate(coords_and_energies)
    )
    reaction = Reaction(
        reactants=(reactant,),
        transition_state=transition_state,
        products=(product,),
        path=ReactionPath(points),
    )
    cvt_res = cvt_rate_si(reaction, 400.0)
    cus_res = cus_rate_si(reaction, 400.0)
    assert cus_res.has_intermediate_well
    assert 0.0 < cus_res.rate_si < cvt_res.rate_si

    _, k_tst = conventional_tst_rate_si(reaction, 400.0)
    k_muvt, k_us = muvt_and_us_rates_si(reaction, 400.0)
    assert 0.0 < k_us <= k_muvt <= k_tst * 1.001

    _, kp, kc = reaction_equilibrium_constants(reaction, 400.0)
    assert kp > 0.0 and kc > 0.0

    calc = calculate_rates(
        reaction,
        CalculationOptions(temperatures_k=(400.0,), method="all", tunneling="muomt"),
    )
    row = calc.rows[0]
    assert row.k_cus is not None and row.k_muvt is not None and row.k_us is not None
    assert row.equilibrium_kc is not None and row.k_reverse_final is not None
    assert math.isclose(row.k_final / row.k_reverse_final, row.equilibrium_kc, rel_tol=1.0e-12)

