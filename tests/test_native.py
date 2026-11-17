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


def test_project_symmetric_matrix_annihilates_basis_and_matches_fallback(monkeypatch) -> None:
    import pyrate.native as native_mod

    rng = np.random.default_rng(42)
    raw = rng.normal(size=(9, 9))
    mat = 0.5 * (raw + raw.T)
    q, _ = np.linalg.qr(rng.normal(size=(9, 4)))

    projected = native_mod.project_symmetric_matrix(mat, q)
    expected = np.eye(9) - q @ q.T
    expected = expected @ mat @ expected
    expected = 0.5 * (expected + expected.T)

    assert np.allclose(projected, projected.T, atol=1.0e-14)
    assert np.allclose(projected @ q, 0.0, atol=1.0e-13)
    assert np.allclose(projected, expected, atol=1.0e-13)

    # Empty basis just symmetrizes input matrix
    sym_only = native_mod.project_symmetric_matrix(raw, np.zeros((9, 0)))
    assert np.allclose(sym_only, 0.5 * (raw + raw.T), atol=1.0e-14)

    if native_mod.HAS_NATIVE:
        monkeypatch.setattr(native_mod, "HAS_NATIVE", False)
        py_projected = native_mod.project_symmetric_matrix(mat, q)
        assert np.allclose(projected, py_projected, atol=1.0e-13)


def test_page_mciver_lqa_step_matches_fine_ode_and_taylor_limit(monkeypatch) -> None:
    import pyrate.native as native_mod

    rng = np.random.default_rng(7)
    raw = rng.normal(size=(6, 6))
    hess = 0.5 * (raw + raw.T)
    evals, evecs = np.linalg.eigh(hess)
    # Include positive, negative (saddle), and exact-zero eigenvalues
    evals[0] = -1.2
    evals[2] = 0.0
    evals[4] = 1.0e-8
    grad = np.array([0.08, -0.05, 0.03, 0.02, -0.04, 0.06], dtype=float)
    step_s = 0.02

    dx, end_tan, u_star = native_mod.page_mciver_lqa_step(grad, evals, evecs, step_s)
    assert u_star > 0.0
    assert math.isclose(float(np.linalg.norm(end_tan)), 1.0, rel_tol=1.0e-12)

    # Fine reference Euler integration of dx/du = -(g + H_eff x) on [0, u_star]
    h_eff = evecs @ np.diag(evals) @ evecs.T
    n_sub = 4000
    du = u_star / n_sub
    x_ref = np.zeros(6, dtype=float)
    s_ref = 0.0
    for _ in range(n_sub):
        v_mid = -(grad + h_eff @ x_ref)
        x_mid = x_ref + 0.5 * du * v_mid
        v_step = -(grad + h_eff @ x_mid)
        s_ref += float(np.linalg.norm(v_step)) * du
        x_ref = x_ref + du * v_step

    assert math.isclose(s_ref, step_s, rel_tol=1.0e-5)
    assert np.allclose(dx, x_ref, atol=1.0e-6)
    g_end = grad + h_eff @ dx
    expected_tan = -g_end / float(np.linalg.norm(g_end))
    assert np.allclose(end_tan, expected_tan, atol=1.0e-5)

    # Zero Hessian limit -> straight steepest descent
    dx_zero, tan_zero, u_zero = native_mod.page_mciver_lqa_step(
        grad, np.zeros(6), np.eye(6), step_s
    )
    g_unit = -grad / float(np.linalg.norm(grad))
    assert np.allclose(dx_zero, step_s * g_unit, atol=1.0e-13)
    assert np.allclose(tan_zero, g_unit, atol=1.0e-13)
    assert math.isclose(u_zero, step_s / float(np.linalg.norm(grad)), rel_tol=1.0e-10)

    # Zero gradient returns zero step cleanly
    z_dx, z_tan, z_u = native_mod.page_mciver_lqa_step(np.zeros(4), np.ones(4), np.eye(4), step_s)
    assert np.allclose(z_dx, 0.0) and np.allclose(z_tan, 0.0) and z_u == 0.0

    if native_mod.HAS_NATIVE:
        monkeypatch.setattr(native_mod, "HAS_NATIVE", False)
        py_dx, py_tan, py_u = native_mod.page_mciver_lqa_step(grad, evals, evecs, step_s)
        assert np.allclose(dx, py_dx, atol=1.0e-12)
        assert np.allclose(end_tan, py_tan, atol=1.0e-12)
        assert math.isclose(u_star, py_u, rel_tol=1.0e-12)


def test_leps_kernel_gradient_matches_finite_difference_and_fallback(monkeypatch) -> None:
    import pyrate.native as native_mod

    bohr_ang = 0.529177106
    k2h = 1.0 / 627.5095
    de = (106.4 * k2h, 90.24 * k2h, 52.09 * k2h)
    re = (1.275 / bohr_ang, 1.414 / bohr_ang, 2.136 / bohr_ang)
    beta = (1.867 * bohr_ang, 1.851 * bohr_ang, 1.923 * bohr_ang)
    sato = (0.02, 0.02, 0.0)
    # Non-collinear 3D geometry
    coords = np.array(
        [[0.1, -0.2, -1.65], [0.25, 0.15, -0.05], [-0.15, 0.3, 1.42]],
        dtype=float,
    )

    energy, grad_au = native_mod.leps_energy_and_gradient(coords, de, re, beta, sato, bohr_ang)
    assert np.allclose(np.sum(grad_au, axis=0), 0.0, atol=1.0e-13)

    h_bohr = 1.0e-5
    h_ang = h_bohr * bohr_ang
    fd_grad = np.zeros_like(grad_au)
    for atom_i in range(3):
        for axis in range(3):
            cp = coords.copy()
            cm = coords.copy()
            cp[atom_i, axis] += h_ang
            cm[atom_i, axis] -= h_ang
            ep, _ = native_mod.leps_energy_and_gradient(cp, de, re, beta, sato, bohr_ang)
            em, _ = native_mod.leps_energy_and_gradient(cm, de, re, beta, sato, bohr_ang)
            fd_grad[atom_i, axis] = (ep - em) / (2.0 * h_bohr)

    assert np.allclose(grad_au, fd_grad, rtol=1.0e-8, atol=1.0e-10)

    if native_mod.HAS_NATIVE:
        monkeypatch.setattr(native_mod, "HAS_NATIVE", False)
        py_e, py_g = native_mod.leps_energy_and_gradient(coords, de, re, beta, sato, bohr_ang)
        assert math.isclose(energy, py_e, rel_tol=1.0e-13)
        assert np.allclose(grad_au, py_g, atol=1.0e-13)

