from __future__ import annotations

import math

import numpy as np
import pytest

from pyrate.constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL, LIGHT_SPEED_M_S
from pyrate.path_frequencies import (
    coordinate_frequencies,
    hessian_coordinate_frequencies,
    intrinsic_coordinate_frequencies,
)


def _quadratic_energy_profile(target_cm1: float = 1000.0, mass_amu: float = 1.0):
    angular = 2.0 * math.pi * LIGHT_SPEED_M_S * 100.0 * target_cm1
    force = mass_amu * AMU_KG * angular**2
    coordinates = np.asarray([-0.23, -0.11, 0.0, 0.19, 0.37])
    energies = 0.5 * force * (coordinates * ANGSTROM_M) ** 2 * AVOGADRO_MOL / 1000.0
    return coordinates, energies


def test_coordinate_frequency_from_uneven_quadratic_profile() -> None:
    coordinates, energies = _quadratic_energy_profile()
    profile = coordinate_frequencies(coordinates, energies, 1.0, coordinate_unit="angstrom")
    assert profile.source == "energy-profile"
    assert np.allclose(profile.frequencies_cm1, 1000.0, rtol=1e-10, atol=1e-8)
    assert profile.imaginary_frequencies_cm1 == ()
    assert profile.barrier_index == 0
    assert intrinsic_coordinate_frequencies is coordinate_frequencies


def test_inverted_profile_reports_signed_imaginary_frequency() -> None:
    coordinates, energies = _quadratic_energy_profile(target_cm1=720.0)
    profile = coordinate_frequencies(coordinates, -energies, 1.0)
    assert np.allclose(profile.frequencies_cm1, -720.0, rtol=1e-10, atol=1e-8)
    assert len(profile.imaginary_frequencies_cm1) == len(coordinates)
    assert profile.real_frequencies_cm1 == ()


def test_point_dependent_effective_mass_changes_local_frequency() -> None:
    coordinates, energies = _quadratic_energy_profile()
    masses = [1.0, 1.0, 4.0, 4.0, 4.0]
    profile = coordinate_frequencies(coordinates, energies, masses)
    assert profile.effective_masses_amu == tuple(masses)
    assert np.allclose(profile.frequencies_cm1[:2], 1000.0, rtol=1e-10, atol=1e-8)
    assert np.allclose(profile.frequencies_cm1[2:], 500.0, rtol=1e-10, atol=1e-8)


def test_hessian_projection_matches_analytic_frequency_and_is_scale_invariant() -> None:
    coordinates, _ = _quadratic_energy_profile(target_cm1=850.0)
    angular = 2.0 * math.pi * LIGHT_SPEED_M_S * 100.0 * 850.0
    force = AMU_KG * angular**2
    hessian = np.diag([force, 0.0, 0.0])
    hessians = np.repeat(hessian[None, :, :], len(coordinates), axis=0)
    tangents = np.repeat(np.asarray([[1.0, 0.0, 0.0]]), len(coordinates), axis=0)
    profile = hessian_coordinate_frequencies(
        hessians,
        tangents * 7.5,
        [1.0],
        coordinates=coordinates,
        hessian_unit="j/m^2",
    )
    assert profile.source == "cartesian-hessian-projection"
    assert np.allclose(profile.frequencies_cm1, 850.0, rtol=1e-12, atol=1e-10)


def test_frequency_input_validation() -> None:
    with pytest.raises(ValueError, match="at least three"):
        coordinate_frequencies([0.0, 1.0], [0.0, 1.0], 1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        coordinate_frequencies([0.0, 0.5, 0.5], [0.0, 1.0, 0.0], 1.0)
    with pytest.raises(ValueError, match="matching lengths"):
        coordinate_frequencies([0.0, 1.0, 2.0], [0.0, 1.0], 1.0)
    with pytest.raises(ValueError, match="unsupported coordinate unit"):
        coordinate_frequencies(
            [0.0, 1.0, 2.0], [0.0, 1.0, 4.0], 1.0, coordinate_unit="foo"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="symmetric"):
        hessian_coordinate_frequencies(
            np.repeat(np.asarray([[[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]), 3, axis=0),
            np.repeat(np.asarray([[1.0, 0.0, 0.0]]), 3, axis=0),
            [1.0],
        )
    with pytest.raises(ValueError, match="unsupported coordinate unit"):
        hessian_coordinate_frequencies(
            np.repeat(np.eye(3)[None, :, :], 3, axis=0),
            np.repeat(np.asarray([[1.0, 0.0, 0.0]]), 3, axis=0),
            [1.0],
            coordinate_unit="foo",  # type: ignore[arg-type]
        )
