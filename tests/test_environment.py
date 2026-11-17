"""Tests for solvent, surface, collision and pressure-dependent helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyrate.environment import (
    CollisionModel,
    SSQRRKModel,
    SolvationCorrection,
    SurfaceModel,
    pressure_dependent_rate_si,
    reaction_solvation_delta_g_kj_mol,
    solve_master_equation,
    solvation_free_energy_kj_mol,
    surface_correction_kj_mol,
)


def test_born_solvation_is_zero_in_vacuum_and_stabilizes_charge() -> None:
    assert solvation_free_energy_kj_mol(charge_e=1.0, radius_angstrom=2.0, dielectric=1.0) == pytest.approx(0.0)
    value = solvation_free_energy_kj_mol(charge_e=1.0, radius_angstrom=2.0, dielectric=80.0)
    assert value < -80.0
    assert math.isfinite(value)


def test_reaction_solvation_delta_uses_transition_state_minus_reactants() -> None:
    reactant = SolvationCorrection(charge_e=1.0, radius_angstrom=2.0, dielectric=20.0)
    ts = SolvationCorrection(charge_e=1.0, radius_angstrom=1.0, dielectric=20.0)
    assert reaction_solvation_delta_g_kj_mol(ts, [reactant]) == pytest.approx(
        ts.free_energy_kj_mol() - reactant.free_energy_kj_mol()
    )


def test_langmuir_coverage_and_surface_correction() -> None:
    model = SurfaceModel(adsorption_free_energy_kj_mol=-20.0)
    low = model.coverage(500.0, pressure_pa=1.0)
    high = model.coverage(500.0, pressure_pa=1.0e7)
    assert 0.0 <= low < high <= 1.0
    assert model.rate_correction(500.0, coverage=0.5) == pytest.approx(0.25)
    assert surface_correction_kj_mol(model, 500.0, 0.0) == pytest.approx(-20.0)


def test_collision_frequency_scales_with_pressure() -> None:
    model = CollisionModel()
    low = model.frequency_s_inv(300.0, 1.0e4)
    high = model.frequency_s_inv(300.0, 2.0e4)
    assert low > 0.0
    assert high / low == pytest.approx(2.0)


def _ssqrrk() -> SSQRRKModel:
    return SSQRRKModel(
        energy_grid_kj_mol=(0.0, 10.0, 20.0, 30.0),
        density_of_states=(1.0, 2.0, 3.0, 4.0),
        microcanonical_rates_s_inv=(0.0, 1.0e2, 2.0e3, 1.0e4),
        collision_frequency_s_inv=2.0e5,
        temperature_k=600.0,
        energy_transfer_kj_mol=8.0,
    )


def test_ssqrrk_matrix_is_population_conserving_with_reaction_sink() -> None:
    model = _ssqrrk()
    matrix = model.rate_matrix()
    rates = np.asarray(model.microcanonical_rates_s_inv)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix.sum(axis=1), -rates)
    assert np.all(np.diag(matrix) < 0.0)
    assert model.effective_rate_s_inv() > 0.0


def test_master_equation_population_and_pressure_limits() -> None:
    model = _ssqrrk()
    result = solve_master_equation(model, time_s=(0.0, 1.0e-8, 1.0e-6))
    assert result.populations.shape == (3, 4)
    assert np.all(result.populations >= 0.0)
    assert np.all(result.survival_probability <= 1.0 + 1.0e-10)
    low = pressure_dependent_rate_si(model, 1.0)
    high = pressure_dependent_rate_si(model, 1.0e9)
    assert 0.0 < low < high < model.effective_rate_s_inv()


def test_invalid_environment_inputs_fail_early() -> None:
    with pytest.raises(ValueError):
        SolvationCorrection(charge_e=1.0)
    with pytest.raises(ValueError):
        SurfaceModel(-5.0, sticking_coefficient=1.1)
    with pytest.raises(ValueError):
        CollisionModel().frequency_s_inv(300.0, 0.0)
    with pytest.raises(ValueError):
        SSQRRKModel((0.0,), (1.0,), (1.0,), 1.0e5, 300.0)
