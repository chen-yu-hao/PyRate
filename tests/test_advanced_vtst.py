from __future__ import annotations

import math

import pytest

from pyrate.advanced_vtst import (
    DualLevelCorrection,
    apply_dual_level_correction,
    dual_level_correction,
    harmonic_vibrational_free_energy_kj_mol,
    intrinsic_coordinate_free_energies,
    iv_tst_rate_si,
    log_eyring_rate_si,
    sample_vrc_loose_transition_state,
    vrc_generalized_free_energy_kj_mol,
    vrc_vtst_rate_si,
)
from pyrate.constants import BOLTZMANN_J_K, PLANCK_J_S


def test_dual_level_correction_interpolates_delta_energy() -> None:
    correction = DualLevelCorrection(
        coordinates=(-1.0, 0.0, 1.0),
        low_level_energies_kj_mol=(0.0, 10.0, 0.0),
        high_level_energies_kj_mol=(1.0, 13.0, 2.0),
    )
    assert correction.corrections_kj_mol == (1.0, 3.0, 2.0)
    assert correction.corrected_energies_kj_mol == (1.0, 13.0, 2.0)
    assert correction.at(0.5) == pytest.approx(2.5)
    assert correction.corrected_at(0.5) == pytest.approx(7.5)
    assert dual_level_correction((0.0, 5.0), (1.0, 8.0)) == (1.0, 3.0)
    assert apply_dual_level_correction((0.0, 5.0), (1.0, 8.0)) == (1.0, 8.0)
    with pytest.raises(ValueError):
        correction.at(2.0)


def test_irc_frequency_free_energy_omits_imaginary_mode() -> None:
    free_energy = harmonic_vibrational_free_energy_kj_mol((1000.0, -500.0), 300.0)
    positive_only = harmonic_vibrational_free_energy_kj_mol((1000.0,), 300.0)
    assert free_energy == pytest.approx(positive_only)
    points = intrinsic_coordinate_free_energies(
        (-1.0, 0.0, 1.0),
        (0.0, 5.0, 0.0),
        ((1000.0, -500.0), (), (1000.0,)),
        300.0,
    )
    assert points[0].frequencies_cm1 == (1000.0, -500.0)
    assert points[0].potential_energy_kj_mol > 0.0


def test_ivtst_refines_quadratic_intrinsic_bottleneck() -> None:
    coordinates = (-1.0, -0.5, 0.0, 0.5, 1.0)
    free_energies = tuple(20.0 - 4.0 * (coordinate - 0.2) ** 2 for coordinate in coordinates)
    result = iv_tst_rate_si(coordinates, free_energies, 400.0)
    assert result.coordinate == pytest.approx(0.2, abs=1.0e-12)
    assert result.activation_gibbs_kj_mol == pytest.approx(20.0, abs=1.0e-12)
    assert result.rate_si > 0.0
    assert result.sample_index == 2


def test_ivtst_bimolecular_standard_state() -> None:
    result = iv_tst_rate_si((0.0, 1.0), (0.0, 0.0), 300.0, molecularity=2)
    expected = math.exp(log_eyring_rate_si(0.0, 300.0, 2))
    assert result.rate_si == pytest.approx(expected)


def test_vrc_constant_potential_reduces_to_eyring_rate() -> None:
    result = vrc_vtst_rate_si(
        (-2.0, 2.0), lambda _coordinate: 12.0, 300.0, quadrature_points=24
    )
    expected = (BOLTZMANN_J_K * 300.0 / PLANCK_J_S) * math.exp(
        -12_000.0 / (8.31446261815324 * 300.0)
    )
    assert result.partition_integral == pytest.approx(
        math.exp(-12.0 / (8.31446261815324 / 1000.0 * 300.0))
    )
    assert result.rate_si == pytest.approx(expected, rel=1.0e-12)
    assert result.sample_count == 24


def test_vrc_phase_space_factor_lowers_effective_barrier() -> None:
    plain = vrc_vtst_rate_si((0.0, 1.0), lambda _coordinate: 10.0, 300.0)
    weighted = vrc_vtst_rate_si((0.0, 1.0), lambda _coordinate: (10.0, 2.0), 300.0)
    assert weighted.effective_activation_gibbs_kj_mol < plain.effective_activation_gibbs_kj_mol
    assert weighted.rate_si == pytest.approx(2.0 * plain.rate_si, rel=1.0e-12)
    assert vrc_generalized_free_energy_kj_mol(10.0, 2.0, 300.0) < 10.0


def test_vrc_input_validation() -> None:
    with pytest.raises(ValueError):
        sample_vrc_loose_transition_state((1.0, 1.0), lambda x: x)
    with pytest.raises(ValueError):
        iv_tst_rate_si((0.0, 1.0), (0.0,), 300.0)
    with pytest.raises(ValueError):
        harmonic_vibrational_free_energy_kj_mol((1000.0,), 0.0)
