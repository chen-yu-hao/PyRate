from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pyrate.constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL, BOLTZMANN_J_K, HBAR_J_S
from pyrate.models import Species, TunnelingProfile
from pyrate.native import semiclassical_transmission, thermal_wkb_kappa, wigner_kappa
from pyrate.tunneling import (
    profile_transmission_probability,
    profile_wkb_kappa,
    transition_state_wigner_kappa,
)

KJ_MOL = 1000.0 / AVOGADRO_MOL  # J per molecule

# Symmetric Eckart barrier V(x) = V0 sech^2(x/a) of docs/roadmap.md N1.
ECKART_V0 = 40.0 * KJ_MOL
ECKART_A = 0.35 * ANGSTROM_M
ECKART_MU = AMU_KG
# γ(E) = 2θ(E) = C (1 - sqrt(E/V0)) with C = 2π a sqrt(2 μ V0) / ħ.
ECKART_C = 2.0 * math.pi * ECKART_A * math.sqrt(2.0 * ECKART_MU * ECKART_V0) / HBAR_J_S
ECKART_TEMPERATURES = (200.0, 300.0, 500.0, 1000.0)


def _eckart_grid(points: int, half_width: float = 12.0) -> tuple[list[float], list[float]]:
    x = np.linspace(-half_width, half_width, points) * ECKART_A
    return x.tolist(), (ECKART_V0 / np.cosh(x / ECKART_A) ** 2).tolist()


def _eckart_exact_probability(energy_j: float) -> float:
    if energy_j > ECKART_V0:
        return 1.0 - _eckart_exact_probability(2.0 * ECKART_V0 - energy_j)
    return 1.0 / (1.0 + math.exp(ECKART_C * (1.0 - math.sqrt(energy_j / ECKART_V0))))


def _eckart_reference_kappa(temperature_k: float, nodes: int = 400) -> float:
    """Independent uniform-semiclassical κ from the analytic θ(E), numpy only.

    With δ = V0 - E and E0 = 0 (both asymptotes of sech²),
    κ = β∫_0^{V0}[P(V0-δ) e^{βδ} + (1-P(V0-δ)) e^{-βδ}] dδ + e^{-βV0}.
    The substitution E = V0 s² removes the sqrt(E) endpoint singularity and
    makes γ = C(1 - s) linear, so Gauss–Legendre in s converges to machine
    precision.
    """

    beta_v0 = ECKART_V0 / (BOLTZMANN_J_K * temperature_k)
    abscissae, weights = np.polynomial.legendre.leggauss(nodes)
    s = 0.5 * (abscissae + 1.0)
    gamma = ECKART_C * (1.0 - s)
    damping = np.exp(-gamma)
    excess = beta_v0 * (1.0 - s * s)  # βδ
    integrand = s * (
        np.exp(excess - gamma - np.log1p(damping)) + np.exp(-excess) / (1.0 + damping)
    )
    return float(beta_v0 * np.sum(weights * integrand) + math.exp(-beta_v0))


def _endoergic_profile(plateau_points: int) -> tuple[list[float], list[float]]:
    """Barrier rising to an exactly flat product plateau at ~20 kJ/mol."""

    x = np.linspace(-2.0, 1.5, 71)
    energies = 20.0 * (1.0 + np.tanh(2.0 * x)) / 2.0 + 30.0 * np.exp(-(x**2) / 0.2)
    spacing = x[1] - x[0]
    plateau = x[-1] + spacing * np.arange(1, plateau_points + 1)
    coordinates = np.concatenate([x, plateau]) * ANGSTROM_M
    potentials = np.concatenate([energies, np.full(plateau_points, energies[-1])]) * KJ_MOL
    return coordinates.tolist(), potentials.tolist()


def _simple_profile() -> tuple[list[float], list[float]]:
    coordinates = [value * ANGSTROM_M for value in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    potentials = [value * KJ_MOL for value in (0.0, 12.0, 25.0, 12.0, 0.0)]
    return coordinates, potentials


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


def test_profile_without_barrier_above_both_asymptotes_has_unit_correction() -> None:
    coordinates = [value * ANGSTROM_M for value in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    downhill = [value * KJ_MOL for value in (10.0, 10.0, 4.0, -3.0, -8.0)]
    assert thermal_wkb_kappa(coordinates, downhill, 300.0, AMU_KG, 100) == 1.0
    assert semiclassical_transmission(coordinates, downhill, 9.0 * KJ_MOL, AMU_KG) == 0.0
    assert semiclassical_transmission(coordinates, downhill, 10.5 * KJ_MOL, AMU_KG) == 1.0


def test_barrier_profile_has_tunneling_enhancement() -> None:
    profile = TunnelingProfile(
        coordinates_angstrom=(-1.0, -0.5, 0.0, 0.5, 1.0),
        energies_kj_mol=(0.0, 12.0, 25.0, 12.0, 0.0),
        reduced_mass_amu=1.0,
        energy_grid_points=200,
    )
    low = profile_transmission_probability(profile, 1.0)
    high = profile_transmission_probability(profile, 20.0)
    assert 0.0 <= low < high < 0.5
    assert profile_transmission_probability(profile, 25.0) == 0.5

    kappa = profile_wkb_kappa(profile, 300.0)
    assert kappa > 1.0
    assert math.isfinite(kappa)


def test_eckart_reference_is_converged() -> None:
    for temperature in ECKART_TEMPERATURES:
        assert math.isclose(
            _eckart_reference_kappa(temperature, 200),
            _eckart_reference_kappa(temperature, 800),
            rel_tol=1.0e-13,
        )


@pytest.mark.parametrize("temperature", ECKART_TEMPERATURES)
def test_eckart_kappa_matches_uniform_semiclassical_reference(
    temperature: float, native: Any
) -> None:
    """Fine piecewise-linear Eckart grid vs the analytic-θ reference, ≤1e-6.

    Grid: 8001 uniform points on ±12a (V(±12a)/V0 = 1.5e-10) and 16000 Simpson
    intervals.  Measured max |rel. error| over the four temperatures:
    1001×2000: 8.8e-6, 4001×8000: 2.6e-7, 8001×16000: 6.9e-8 (≤1.8e-7 for
    neighbouring grid sizes), 16001×32000: 1.6e-8.  The residual is not a
    smooth power law: on a piecewise-linear profile γ(E) carries an
    oscillatory error (std ∝ h^1.5, much smaller mean) from the segment that
    contains each turning point, which the energy integral averages out.
    Runs on the C++ kernel only (the Python kernel would take ~1.5 min); the
    parity tests pin ``py_thermal_wkb_kappa`` to it.
    """

    coordinates, potentials = _eckart_grid(8001)
    kappa = native.thermal_wkb_kappa(coordinates, potentials, temperature, ECKART_MU, 16000)
    assert math.isclose(kappa, _eckart_reference_kappa(temperature), rel_tol=1.0e-6)


@pytest.mark.parametrize("temperature", ECKART_TEMPERATURES)
def test_eckart_kappa_coarse_grid_on_active_backend(temperature: float) -> None:
    # 1001 points and 400 intervals: measured max error 5.6e-5 (see above).
    coordinates, potentials = _eckart_grid(1001)
    kappa = thermal_wkb_kappa(coordinates, potentials, temperature, ECKART_MU, 400)
    assert math.isclose(kappa, _eckart_reference_kappa(temperature), rel_tol=1.0e-4)


@pytest.mark.parametrize("fraction", [0.05, 0.3, 0.7, 0.99, 1.01, 1.5])
def test_eckart_transmission_probability(fraction: float) -> None:
    # Pointwise γ(E) on 8001 points has an oscillatory error ≲5e-4 (see above).
    coordinates, potentials = _eckart_grid(8001)
    energy = fraction * ECKART_V0
    probability = semiclassical_transmission(coordinates, potentials, energy, ECKART_MU)
    assert math.isclose(probability, _eckart_exact_probability(energy), rel_tol=1.0e-3)


def test_transmission_probability_shape() -> None:
    coordinates, potentials = _simple_profile()
    barrier = max(potentials)
    threshold = max(potentials[0], potentials[-1])

    def probability(energy: float) -> float:
        return semiclassical_transmission(coordinates, potentials, energy, AMU_KG)

    assert probability(threshold - 1.0e-3 * KJ_MOL) == 0.0
    assert 0.0 < probability(threshold) < 1.0e-6
    assert probability(barrier) == 0.5
    for offset in (0.1, 3.0, 12.0, 24.9):
        below = probability(barrier - offset * KJ_MOL)
        above = probability(barrier + offset * KJ_MOL)
        assert math.isclose(below + above, 1.0, rel_tol=0.0, abs_tol=1.0e-15)
    assert probability(2.0 * barrier - threshold) == 1.0
    assert probability(3.0 * barrier) == 1.0

    scan = [probability(value) for value in np.linspace(-5.0, 55.0, 601) * KJ_MOL]
    assert all(right >= left for left, right in zip(scan, scan[1:], strict=False))


def test_endoergic_profile_threshold_is_product_asymptote() -> None:
    coordinates, potentials = _endoergic_profile(40)
    threshold = potentials[-1]
    assert threshold > potentials[0]
    for energy_kj in (-1.0, 5.0, 15.0, threshold / KJ_MOL - 1.0e-6):
        energy = energy_kj * KJ_MOL
        assert semiclassical_transmission(coordinates, potentials, energy, AMU_KG) == 0.0
    assert semiclassical_transmission(coordinates, potentials, threshold, AMU_KG) > 0.0

    profile = TunnelingProfile(
        coordinates_angstrom=tuple(value / ANGSTROM_M for value in coordinates),
        energies_kj_mol=tuple(value / KJ_MOL for value in potentials),
        reduced_mass_amu=1.0,
    )
    assert profile_transmission_probability(profile, 5.0) == 0.0
    assert profile_transmission_probability(profile, 15.0) == 0.0


@pytest.mark.parametrize("temperature", [150.0, 300.0, 1000.0])
def test_endoergic_kappa_independent_of_plateau_length(temperature: float) -> None:
    short = _endoergic_profile(40)
    long = _endoergic_profile(80)
    kappa_short = thermal_wkb_kappa(*short, temperature, AMU_KG, 400)
    kappa_long = thermal_wkb_kappa(*long, temperature, AMU_KG, 400)
    assert kappa_short > 1.0
    assert math.isclose(kappa_short, kappa_long, rel_tol=1.0e-10)


def test_kappa_limits_and_monotonicity() -> None:
    coordinates, potentials = _simple_profile()
    assert abs(thermal_wkb_kappa(coordinates, potentials, 1.0e5, AMU_KG, 200) - 1.0) < 1.0e-3
    temperatures = (20.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 5000.0, 1.0e4)
    kappas = [thermal_wkb_kappa(coordinates, potentials, t, AMU_KG, 200) for t in temperatures]
    assert all(value > 1.0 and math.isfinite(value) for value in kappas)
    assert all(low_t > high_t for low_t, high_t in zip(kappas, kappas[1:], strict=False))


def test_kappa_overflow_returns_infinity() -> None:
    coordinates, potentials = _simple_profile()
    assert thermal_wkb_kappa(coordinates, potentials, 1.0, AMU_KG, 200) == math.inf


def test_kappa_is_invariant_to_energy_origin() -> None:
    coordinates, potentials = _simple_profile()
    shifted = [value + 100.0 * KJ_MOL for value in potentials]
    assert math.isclose(
        thermal_wkb_kappa(coordinates, potentials, 300.0, AMU_KG, 200),
        thermal_wkb_kappa(coordinates, shifted, 300.0, AMU_KG, 200),
        rel_tol=1.0e-10,
    )


def test_semiclassical_transmission_validates_input() -> None:
    coordinates, potentials = _simple_profile()
    with pytest.raises(ValueError):
        semiclassical_transmission(coordinates, potentials[:-1], 0.0, AMU_KG)
    with pytest.raises(ValueError):
        semiclassical_transmission(coordinates[::-1], potentials, 0.0, AMU_KG)
    with pytest.raises(ValueError):
        semiclassical_transmission(coordinates, potentials, 0.0, 0.0)
