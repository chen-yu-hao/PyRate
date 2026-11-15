from __future__ import annotations

import math

import numpy as np

from pyrate.constants import (
    AMU_KG,
    ANGSTROM_M,
    AVOGADRO_MOL,
    BOLTZMANN_J_K,
    LIGHT_SPEED_M_S,
    PLANCK_J_S,
)
from pyrate import native as native_mod
from pyrate.native import (
    backend_info,
    beyer_swinehart,
    eckart_transmission,
    gauss_legendre_quadrature,
    harmonic_log_partition,
    lct_integrate_amplitudes,
    lct_uniformize_probabilities,
    neville_interpolate,
    refine_maximum,
    refine_minimum,
    thermal_boltz_probabilities_kappa,
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


def test_eckart_matches_parabolic_and_wigner_limits() -> None:
    """Guard the Johnston-Heicklen prefactors against analytic limits."""

    h_c = PLANCK_J_S * LIGHT_SPEED_M_S * 100.0
    barrier_j = 100.0e3 / AVOGADRO_MOL
    frequency_cm1 = 1000.0
    quantum_j = h_c * frequency_cm1

    # Just below a high barrier the Eckart potential is locally parabolic:
    # P(E) -> 1 / (1 + exp(2 pi (V - E) / (h nu*))).
    assert math.isclose(
        eckart_transmission(barrier_j, barrier_j, barrier_j, frequency_cm1), 0.5, rel_tol=5.0e-2
    )
    energy_j = barrier_j - 0.25 * quantum_j
    parabolic = 1.0 / (1.0 + math.exp(2.0 * math.pi * (barrier_j - energy_j) / quantum_j))
    assert math.isclose(
        eckart_transmission(energy_j, barrier_j, barrier_j, frequency_cm1),
        parabolic,
        rel_tol=1.0e-1,
    )

    # High temperature: kappa -> 1 + (h nu* / k_B T)^2 / 24 (Wigner).
    ratio = quantum_j / (BOLTZMANN_J_K * 3000.0)
    kappa_hot = thermal_eckart_kappa(barrier_j, barrier_j, frequency_cm1, 3000.0, 2000)
    assert math.isclose(kappa_hot - 1.0, ratio * ratio / 24.0, rel_tol=0.3)

    # Independent midpoint-rule evaluation of the textbook formula.
    assert math.isclose(
        thermal_eckart_kappa(barrier_j, barrier_j, frequency_cm1, 300.0, 2000),
        3.2462,
        rel_tol=2.0e-3,
    )
    asym = thermal_eckart_kappa(
        40.0e3 / AVOGADRO_MOL, 60.0e3 / AVOGADRO_MOL, 1500.0, 300.0, 2000
    )
    assert math.isclose(asym, 14.521, rel_tol=2.0e-3)


def test_lct_and_boltz_kernels_exactness_and_cpp_python_parity(monkeypatch) -> None:
    # 1. Gauss-Legendre integrates x^5 + 3 x^4 - 2 x^2 + 1 on [-1.5, 2.5] to machine precision
    nodes, weights = gauss_legendre_quadrature(-1.5, 2.5, 6)
    poly_int = sum(
        w * (x**5 + 3.0 * x**4 - 2.0 * x**2 + 1.0) for x, w in zip(nodes, weights, strict=True)
    )

    def antideriv(x: float) -> float:
        return x**6 / 6.0 + 3.0 * x**5 / 5.0 - 2.0 * x**3 / 3.0 + x

    assert math.isclose(poly_int, antideriv(2.5) - antideriv(-1.5), rel_tol=1.0e-13)

    # 2. Neville interpolation reproduces a cubic polynomial exactly from 4 nodes
    xs = (-0.8, -0.2, 0.3, 1.1)
    ys = tuple(2.0 * x**3 - 1.5 * x**2 + 0.7 * x - 0.4 for x in xs)
    x_eval = 0.55
    expected_y = 2.0 * x_eval**3 - 1.5 * x_eval**2 + 0.7 * x_eval - 0.4
    assert math.isclose(neville_interpolate(xs, ys, x_eval), expected_y, rel_tol=1.0e-14)

    # 3. LCSINT + LCPROB/LCGSRT + BOLTZ overflow-safe log-space integration
    st = (-1.2, -0.9, -0.6, -0.3, -0.05)
    esv = (0.004, 0.008, 0.012, 0.016, 0.020)
    sinx = (0.95, 0.96, 0.98, 0.99, 1.0)
    thet = (8.5, 5.2, 2.6, 0.8, 0.0)
    tau = (120.0, 115.0, 110.0, 105.0, 100.0)
    redm_au = 44000.0

    amps = lct_integrate_amplitudes(st, esv, sinx, thet, tau, redm_au, 0, 40, 4)
    assert len(amps) == 5
    assert all(a >= 0.0 for a in amps)
    assert any(a > 0.0 for a in amps[:-1])

    prim_ground = (amps[0], amps[1], amps[2], amps[3], max(amps[3], 0.25))
    prim_cumul = (amps[0], amps[1], amps[2] * 1.3, amps[3] * 1.5, max(amps[3], 0.25) * 1.4)
    unif_matrix, sorted_probs = lct_uniformize_probabilities([prim_ground, prim_cumul])
    assert len(unif_matrix) == 2
    assert len(unif_matrix[0]) == 5
    assert len(sorted_probs) == 4
    assert math.isclose(unif_matrix[0][-1], 0.5, rel_tol=1.0e-15)
    assert all(0.0 <= p <= 1.0 for p in sorted_probs)
    assert all(s >= g for s, g in zip(sorted_probs, unif_matrix[0][:-1], strict=True))

    # Extreme low-temperature BOLTZ overflow test: at T = 25 K, beta * V_max = 721.64 > 709.78
    # (naive exp/sinh overflows IEEE-754 float64!), while log-space BOLTZ stays exact and finite.
    kj_to_j = 1000.0 / AVOGADRO_MOL
    e_nodes_kj, e_weights_kj = gauss_legendre_quadrature(0.0, 150.0, 24)
    e_nodes_j = tuple(val * kj_to_j for val in e_nodes_kj)
    e_weights_j = tuple(val * kj_to_j for val in e_weights_kj)
    probs = tuple(1.0 / (1.0 + math.exp(4.6 * (150.0 - e))) for e in e_nodes_kj)
    kappa_cold = thermal_boltz_probabilities_kappa(
        e_nodes_j, e_weights_j, probs, 150.0 * kj_to_j, 25.0
    )
    assert math.isfinite(kappa_cold) and kappa_cold > 1.0e10

    # Direct C++ vs pure-Python fallback parity check when C++ extension is compiled
    if native_mod._cpp is not None:
        monkeypatch.setattr(native_mod, "HAS_NATIVE", True)
        cpp_nodes, cpp_weights = gauss_legendre_quadrature(-1.5, 2.5, 17)
        cpp_nev = neville_interpolate(xs, ys, x_eval)
        cpp_amps = lct_integrate_amplitudes(st, esv, sinx, thet, tau, redm_au, 0, 40, 4)
        cpp_mat, cpp_tot = lct_uniformize_probabilities([prim_ground, prim_cumul])
        cpp_kap = thermal_boltz_probabilities_kappa(
            e_nodes_j, e_weights_j, probs, 150.0 * kj_to_j, 25.0
        )

        monkeypatch.setattr(native_mod, "HAS_NATIVE", False)
        py_nodes, py_weights = gauss_legendre_quadrature(-1.5, 2.5, 17)
        py_nev = neville_interpolate(xs, ys, x_eval)
        py_amps = lct_integrate_amplitudes(st, esv, sinx, thet, tau, redm_au, 0, 40, 4)
        py_mat, py_tot = lct_uniformize_probabilities([prim_ground, prim_cumul])
        py_kap = thermal_boltz_probabilities_kappa(
            e_nodes_j, e_weights_j, probs, 150.0 * kj_to_j, 25.0
        )

        assert np.allclose(py_nodes, cpp_nodes, rtol=1.0e-13, atol=1.0e-14)
        assert np.allclose(py_weights, cpp_weights, rtol=1.0e-13, atol=1.0e-14)
        assert math.isclose(py_nev, cpp_nev, rel_tol=1.0e-14)
        assert np.allclose(py_amps, cpp_amps, rtol=1.0e-12, atol=1.0e-15)
        assert np.allclose(py_mat, cpp_mat, rtol=1.0e-13, atol=1.0e-15)
        assert np.allclose(py_tot, cpp_tot, rtol=1.0e-13, atol=1.0e-15)
        assert math.isclose(py_kap, cpp_kap, rel_tol=1.0e-13)
