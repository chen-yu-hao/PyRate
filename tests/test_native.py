from __future__ import annotations

import math

import numpy as np

from pyrate.constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL
from pyrate.native import (
    backend_info,
    beyer_swinehart,
    eckart_transmission,
    harmonic_log_partition,
    refine_maximum,
    refine_minimum,
    thermal_eckart_kappa,
    thermal_wkb_kappa,
    thermal_wkb_profile_kappa,
    wkb_exponent_profile,
)


def test_backend_reports_version() -> None:
    info = backend_info()
    assert info["backend"] in {"cpp", "python"}
    assert isinstance(info["version"], str)


def test_refine_maximum_uses_local_polynomial() -> None:
    x = np.linspace(-1.0, 1.0, 5)
    y = 3.0 - 2.0 * (x - 0.2) ** 2
    result = refine_maximum(x, y)
    assert math.isclose(result.coordinate, 0.2, abs_tol=2.0e-3)
    assert math.isclose(result.value, 3.0, abs_tol=1.0e-5)
    assert result.fit_points == 5


def test_refine_maximum_near_boundary_uses_three_point_fitmax() -> None:
    # Maximum at index 1 triggers Polyrate FITMAX 3-point parabolic fit
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    y = 4.0 - 3.0 * (x - (-0.45)) ** 2
    result = refine_maximum(x, y)
    assert result.fit_points == 3
    assert math.isclose(result.coordinate, -0.45, abs_tol=2.0e-3)
    assert math.isclose(result.value, 4.0, abs_tol=1.0e-4)


def test_refine_minimum_finds_intermediate_well() -> None:
    x = np.linspace(-1.0, 1.0, 5)
    y = 1.5 + 4.0 * (x + 0.15) ** 2
    result = refine_minimum(x, y)
    assert math.isclose(result.coordinate, -0.15, abs_tol=2.0e-3)
    assert math.isclose(result.value, 1.5, abs_tol=1.0e-4)


def test_beyer_swinehart_known_small_case() -> None:
    density = beyer_swinehart((100.0, 200.0), 100.0, 6)
    assert density.tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_harmonic_kernel_ignores_imaginary_frequency() -> None:
    assert math.isclose(
        harmonic_log_partition((500.0, -1000.0), 300.0),
        harmonic_log_partition((500.0,), 300.0),
        rel_tol=1.0e-15,
    )


def test_variable_mass_wkb_profile_enhances_tunneling() -> None:
    coords = tuple(s * ANGSTROM_M for s in (-1.0, -0.5, 0.0, 0.5, 1.0))
    kj_to_j = 1000.0 / AVOGADRO_MOL
    pots = tuple(v * kj_to_j for v in (0.0, 15.0, 30.0, 15.0, 0.0))
    masses_uniform = tuple(AMU_KG for _ in coords)
    masses_reduced = tuple(0.5 * AMU_KG for _ in coords)

    theta_uniform = wkb_exponent_profile(coords, pots, 10.0 * kj_to_j, masses_uniform)
    theta_reduced = wkb_exponent_profile(coords, pots, 10.0 * kj_to_j, masses_reduced)
    assert math.isclose(theta_reduced, theta_uniform / math.sqrt(2.0), rel_tol=1.0e-12)

    kappa_uniform = thermal_wkb_kappa(coords, pots, 300.0, AMU_KG, 200)
    kappa_profile_uniform = thermal_wkb_profile_kappa(coords, pots, 300.0, masses_uniform, 200)
    kappa_profile_reduced = thermal_wkb_profile_kappa(coords, pots, 300.0, masses_reduced, 200)
    assert math.isclose(kappa_uniform, kappa_profile_uniform, rel_tol=1.0e-12)
    assert kappa_profile_reduced > kappa_uniform > 1.0


def test_eckart_transmission_and_thermal_kappa() -> None:
    kj_to_j = 1000.0 / AVOGADRO_MOL
    v1_j = 25.0 * kj_to_j
    v2_j = 30.0 * kj_to_j
    p_low = eckart_transmission(5.0 * kj_to_j, v1_j, v2_j, 1200.0)
    p_top = eckart_transmission(v1_j, v1_j, v2_j, 1200.0)
    p_high = eckart_transmission(60.0 * kj_to_j, v1_j, v2_j, 1200.0)
    assert 0.0 < p_low < p_top < p_high <= 1.0

    kappa = thermal_eckart_kappa(v1_j, v2_j, 1200.0, 300.0, 240)
    assert kappa > 1.0
    assert math.isfinite(kappa)

