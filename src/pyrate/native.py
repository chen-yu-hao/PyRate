"""Stable Python interface to optional C++ numerical kernels.

The package remains usable from a source checkout when the extension has not
been built.  Production installations compile :mod:`pyrate._kernels`; the
fallbacks below deliberately mirror the native algorithms for reproducibility.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .constants import BOLTZMANN_J_K, HBAR_J_S, LIGHT_SPEED_M_S, PLANCK_J_S

_DISABLE_NATIVE = os.environ.get("PYRATE_DISABLE_NATIVE", "").lower() in {"1", "true", "yes"}

try:
    if _DISABLE_NATIVE:
        raise ImportError("native kernels disabled by PYRATE_DISABLE_NATIVE")
    from . import _kernels as _cpp
except ImportError:  # pragma: no cover - exercised explicitly through fallback tests
    _cpp = None

HAS_NATIVE = _cpp is not None


@dataclass(frozen=True, slots=True)
class RefinedMaximum:
    coordinate: float
    value: float
    sample_index: int
    fit_points: int


def backend_info() -> dict[str, str | bool]:
    return {
        "native": HAS_NATIVE,
        "backend": "cpp" if HAS_NATIVE else "python",
        "version": _cpp.native_version() if HAS_NATIVE else "python-fallback-0.2",
    }


def harmonic_log_partition(frequencies_cm1: Iterable[float], temperature_k: float) -> float:
    frequencies = tuple(float(value) for value in frequencies_cm1)
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if HAS_NATIVE:
        return float(_cpp.harmonic_log_partition(frequencies, float(temperature_k)))
    total = 0.0
    for frequency in frequencies:
        if frequency <= 0.0:
            continue
        ratio = (
            PLANCK_J_S
            * LIGHT_SPEED_M_S
            * 100.0
            * frequency
            / (BOLTZMANN_J_K * temperature_k)
        )
        total -= math.log1p(-math.exp(-ratio))
    return total


def wigner_kappa(imaginary_frequency_cm1: float, temperature_k: float) -> float:
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if HAS_NATIVE:
        return float(_cpp.wigner_kappa(float(imaginary_frequency_cm1), float(temperature_k)))
    ratio = (
        PLANCK_J_S
        * LIGHT_SPEED_M_S
        * 100.0
        * abs(imaginary_frequency_cm1)
        / (BOLTZMANN_J_K * temperature_k)
    )
    return 1.0 + ratio * ratio / 24.0


def refine_maximum(x: Sequence[float], y: Sequence[float]) -> RefinedMaximum:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if x_values.size == 0:
        raise ValueError("x and y cannot be empty")
    if np.any(np.diff(x_values) <= 0.0):
        raise ValueError("x must be strictly increasing")
    if HAS_NATIVE:
        coordinate, value, sample_index, fit_points = _cpp.refine_maximum(
            x_values.tolist(), y_values.tolist()
        )
        return RefinedMaximum(
            float(coordinate), float(value), int(sample_index), int(fit_points)
        )

    maximum_index = int(np.argmax(y_values))
    if x_values.size < 3 or maximum_index in {0, x_values.size - 1}:
        return RefinedMaximum(
            float(x_values[maximum_index]), float(y_values[maximum_index]), maximum_index, 1
        )
    can_use_five = bool(x_values.size >= 5 and 1 < maximum_index < x_values.size - 2)
    count = 5 if can_use_five else 3
    start = maximum_index - count // 2
    local_x = x_values[start : start + count]
    local_y = y_values[start : start + count]
    center = float(x_values[maximum_index])
    scale = max(abs(float(local_x[0]) - center), abs(float(local_x[-1]) - center))
    if scale <= 0.0:
        return RefinedMaximum(center, float(y_values[maximum_index]), maximum_index, 1)
    t = (local_x - center) / scale
    coefficients = np.polynomial.polynomial.polyfit(t, local_y, count - 1)
    derivative = np.polynomial.polynomial.polyder(coefficients)
    roots = np.polynomial.polynomial.polyroots(derivative)
    candidates = [float(t[0]), float(t[-1]), 0.0]
    candidates.extend(
        float(root.real)
        for root in roots
        if abs(float(root.imag)) < 1.0e-10 and t[0] <= root.real <= t[-1]
    )
    values = [float(np.polynomial.polynomial.polyval(candidate, coefficients)) for candidate in candidates]
    best = int(np.argmax(values))
    best_value = values[best]
    local_span = float(np.ptp(local_y))
    if not math.isfinite(best_value) or best_value > y_values[maximum_index] + 2.0 * max(
        1.0, local_span
    ):
        return RefinedMaximum(center, float(y_values[maximum_index]), maximum_index, 1)
    return RefinedMaximum(
        center + scale * candidates[best], best_value, maximum_index, count
    )


def refine_minimum(x: Sequence[float], y: Sequence[float]) -> RefinedMaximum:
    """Refine a sampled minimum using a local 3-point or 5-point fit (by negating y)."""

    neg_y = [-float(value) for value in y]
    refined = refine_maximum(x, neg_y)
    return RefinedMaximum(
        coordinate=refined.coordinate,
        value=-refined.value,
        sample_index=refined.sample_index,
        fit_points=refined.fit_points,
    )


def _linear_sqrt_integral(u0: float, u1: float) -> float:
    if u0 <= 0.0 and u1 <= 0.0:
        return 0.0
    delta = u1 - u0
    scale = max(1.0, abs(u0), abs(u1))
    if abs(delta) <= np.finfo(float).eps * scale:
        return math.sqrt(max(0.0, u0))
    if u0 > 0.0 and u1 > 0.0:
        return (2.0 / 3.0) * (u1**1.5 - u0**1.5) / delta
    if u0 > 0.0:
        return (2.0 / 3.0) * u0**1.5 / (u0 - u1)
    return (2.0 / 3.0) * u1**1.5 / (u1 - u0)


def wkb_exponent(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    energy_j_per_molecule: float,
    reduced_mass_kg: float,
) -> float:
    coordinates = tuple(float(value) for value in coordinates_m)
    potentials = tuple(float(value) for value in potentials_j_per_molecule)
    if len(coordinates) != len(potentials) or len(coordinates) < 2:
        raise ValueError("coordinates and potentials require matching lengths of at least 2")
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("coordinates must be strictly increasing")
    if reduced_mass_kg <= 0.0:
        raise ValueError("reduced mass must be positive")
    if HAS_NATIVE:
        return float(
            _cpp.wkb_exponent(
                coordinates, potentials, float(energy_j_per_molecule), float(reduced_mass_kg)
            )
        )
    integral = 0.0
    for index in range(1, len(coordinates)):
        dx = coordinates[index] - coordinates[index - 1]
        u0 = potentials[index - 1] - energy_j_per_molecule
        u1 = potentials[index] - energy_j_per_molecule
        integral += dx * _linear_sqrt_integral(u0, u1)
    return 2.0 * math.sqrt(2.0 * reduced_mass_kg) * integral / HBAR_J_S


def wkb_exponent_profile(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    energy_j_per_molecule: float,
    effective_masses_kg: Sequence[float],
) -> float:
    coordinates = tuple(float(value) for value in coordinates_m)
    potentials = tuple(float(value) for value in potentials_j_per_molecule)
    masses = tuple(float(value) for value in effective_masses_kg)
    if len(coordinates) != len(potentials) or len(coordinates) != len(masses) or len(coordinates) < 2:
        raise ValueError(
            "coordinates, potentials, and effective_masses require matching lengths >= 2"
        )
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("coordinates must be strictly increasing")
    if any(m <= 0.0 for m in masses):
        raise ValueError("effective masses must be positive")
    if HAS_NATIVE:
        return float(
            _cpp.wkb_exponent_profile(
                coordinates, potentials, float(energy_j_per_molecule), masses
            )
        )
    integral = 0.0
    for index in range(1, len(coordinates)):
        dx = coordinates[index] - coordinates[index - 1]
        u0 = 2.0 * masses[index - 1] * (potentials[index - 1] - energy_j_per_molecule)
        u1 = 2.0 * masses[index] * (potentials[index] - energy_j_per_molecule)
        integral += dx * _linear_sqrt_integral(u0, u1)
    return 2.0 * integral / HBAR_J_S


def _safe_exp(exponent: float) -> float:
    if exponent < -745.0:
        return 0.0
    if exponent > 709.0:
        return math.inf
    return math.exp(exponent)


def _thermal_sag_integral_fallback(
    potentials: tuple[float, ...],
    temperature_k: float,
    energy_grid_points: int,
    exponent_fn,
) -> float:
    threshold = max(potentials[0], potentials[-1])
    vmax = max(potentials)
    vm = vmax - threshold
    if vm <= 0.0:
        return 1.0
    count = energy_grid_points + energy_grid_points % 2
    beta = 1.0 / (BOLTZMANN_J_K * temperature_k)
    step = vm / count

    def integrand(energy: float) -> float:
        gamma = exponent_fn(energy)
        delta = max(0.0, beta * (vmax - energy))
        if delta == 0.0:
            return 0.0
        exp_num = _safe_exp(delta - gamma)
        if exp_num == 0.0:
            return 0.0
        reflection_factor = -math.expm1(-2.0 * delta)
        kemble_denom = 1.0 + _safe_exp(-gamma)
        return beta * exp_num * (reflection_factor / kemble_denom)

    total = integrand(threshold) + integrand(vmax)
    for index in range(1, count):
        total += (2.0 if index % 2 == 0 else 4.0) * integrand(threshold + index * step)
    return 1.0 + step * total / 3.0


def thermal_wkb_kappa(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    temperature_k: float,
    reduced_mass_kg: float,
    energy_grid_points: int = 400,
) -> float:
    coordinates = tuple(float(value) for value in coordinates_m)
    potentials = tuple(float(value) for value in potentials_j_per_molecule)
    if len(coordinates) != len(potentials) or len(coordinates) < 3:
        raise ValueError("coordinates and potentials require matching lengths of at least 3")
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("coordinates must be strictly increasing")
    if temperature_k <= 0.0 or reduced_mass_kg <= 0.0:
        raise ValueError("temperature and reduced mass must be positive")
    if energy_grid_points < 20:
        raise ValueError("energy_grid_points must be at least 20")
    if HAS_NATIVE:
        return float(
            _cpp.thermal_wkb_kappa(
                coordinates,
                potentials,
                float(temperature_k),
                float(reduced_mass_kg),
                int(energy_grid_points),
            )
        )
    return _thermal_sag_integral_fallback(
        potentials,
        temperature_k,
        energy_grid_points,
        lambda energy: wkb_exponent(coordinates, potentials, energy, reduced_mass_kg),
    )


def thermal_wkb_profile_kappa(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    temperature_k: float,
    effective_masses_kg: Sequence[float],
    energy_grid_points: int = 400,
) -> float:
    coordinates = tuple(float(value) for value in coordinates_m)
    potentials = tuple(float(value) for value in potentials_j_per_molecule)
    masses = tuple(float(value) for value in effective_masses_kg)
    if len(coordinates) != len(potentials) or len(coordinates) != len(masses) or len(coordinates) < 3:
        raise ValueError(
            "coordinates, potentials, and effective_masses require matching lengths >= 3"
        )
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("coordinates must be strictly increasing")
    if temperature_k <= 0.0 or any(m <= 0.0 for m in masses):
        raise ValueError("temperature and effective masses must be positive")
    if energy_grid_points < 20:
        raise ValueError("energy_grid_points must be at least 20")
    if HAS_NATIVE:
        return float(
            _cpp.thermal_wkb_profile_kappa(
                coordinates,
                potentials,
                float(temperature_k),
                masses,
                int(energy_grid_points),
            )
        )
    return _thermal_sag_integral_fallback(
        potentials,
        temperature_k,
        energy_grid_points,
        lambda energy: wkb_exponent_profile(coordinates, potentials, energy, masses),
    )


def eckart_transmission(
    energy_j_per_molecule: float,
    barrier_forward_j_per_molecule: float,
    barrier_reverse_j_per_molecule: float,
    imaginary_frequency_cm1: float,
) -> float:
    if HAS_NATIVE:
        return float(
            _cpp.eckart_transmission(
                float(energy_j_per_molecule),
                float(barrier_forward_j_per_molecule),
                float(barrier_reverse_j_per_molecule),
                float(imaginary_frequency_cm1),
            )
        )
    if barrier_forward_j_per_molecule <= 0.0 or barrier_reverse_j_per_molecule <= 0.0:
        threshold = max(0.0, barrier_forward_j_per_molecule - barrier_reverse_j_per_molecule)
        return 1.0 if energy_j_per_molecule >= threshold else 0.0
    threshold = max(0.0, barrier_forward_j_per_molecule - barrier_reverse_j_per_molecule)
    if energy_j_per_molecule <= threshold:
        return 0.0
    u_star = PLANCK_J_S * LIGHT_SPEED_M_S * 100.0 * abs(imaginary_frequency_cm1)
    if u_star <= 0.0:
        return 1.0 if energy_j_per_molecule >= barrier_forward_j_per_molecule else 0.0
    inv_sqrt_sum = (1.0 / math.sqrt(barrier_forward_j_per_molecule)) + (
        1.0 / math.sqrt(barrier_reverse_j_per_molecule)
    )
    two_pi_alpha = (2.0 * math.pi / u_star) * math.sqrt(energy_j_per_molecule) / inv_sqrt_sum
    two_pi_beta = (
        (2.0 * math.pi / u_star)
        * math.sqrt(
            max(
                0.0,
                energy_j_per_molecule
                - barrier_forward_j_per_molecule
                + barrier_reverse_j_per_molecule,
            )
        )
        / inv_sqrt_sum
    )
    a_plus = two_pi_alpha + two_pi_beta
    a_minus = abs(two_pi_alpha - two_pi_beta)
    d_arg = (
        16.0
        * barrier_forward_j_per_molecule
        * barrier_reverse_j_per_molecule
        / (u_star * u_star)
        - 1.0
    )
    if d_arg >= 0.0:
        two_pi_delta = 0.5 * math.pi * math.sqrt(d_arg)
        scale = max(a_plus, two_pi_delta)
        num = (_safe_exp(a_plus - scale) + _safe_exp(-a_plus - scale)) - (
            _safe_exp(a_minus - scale) + _safe_exp(-a_minus - scale)
        )
        den = (_safe_exp(a_plus - scale) + _safe_exp(-a_plus - scale)) + (
            _safe_exp(two_pi_delta - scale) + _safe_exp(-two_pi_delta - scale)
        )
        return min(1.0, max(0.0, num / den))
    cos_term = 2.0 * math.cos(0.5 * math.pi * math.sqrt(-d_arg))
    scale = a_plus
    num = (1.0 + _safe_exp(-2.0 * a_plus)) - (
        _safe_exp(a_minus - scale) + _safe_exp(-a_minus - scale)
    )
    den = (1.0 + _safe_exp(-2.0 * a_plus)) + cos_term * _safe_exp(-scale)
    return min(1.0, max(0.0, num / den))


def thermal_eckart_kappa(
    barrier_forward_j_per_molecule: float,
    barrier_reverse_j_per_molecule: float,
    imaginary_frequency_cm1: float,
    temperature_k: float,
    energy_grid_points: int = 400,
) -> float:
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if energy_grid_points < 20:
        raise ValueError("energy_grid_points must be at least 20")
    if HAS_NATIVE:
        return float(
            _cpp.thermal_eckart_kappa(
                float(barrier_forward_j_per_molecule),
                float(barrier_reverse_j_per_molecule),
                float(imaginary_frequency_cm1),
                float(temperature_k),
                int(energy_grid_points),
            )
        )
    threshold = max(0.0, barrier_forward_j_per_molecule - barrier_reverse_j_per_molecule)
    vm = barrier_forward_j_per_molecule - threshold
    if vm <= 0.0 or abs(imaginary_frequency_cm1) <= 0.0:
        return 1.0
    count = energy_grid_points + energy_grid_points % 2
    beta = 1.0 / (BOLTZMANN_J_K * temperature_k)
    e_upper = barrier_forward_j_per_molecule + 16.0 * BOLTZMANN_J_K * temperature_k
    step = (e_upper - threshold) / count

    def integrand(energy: float) -> float:
        prob = eckart_transmission(
            energy,
            barrier_forward_j_per_molecule,
            barrier_reverse_j_per_molecule,
            imaginary_frequency_cm1,
        )
        if prob <= 0.0:
            return 0.0
        return beta * prob * _safe_exp(beta * (barrier_forward_j_per_molecule - energy))

    total = integrand(threshold) + integrand(e_upper)
    for index in range(1, count):
        total += (2.0 if index % 2 == 0 else 4.0) * integrand(threshold + index * step)
    return step * total / 3.0 + _safe_exp(beta * (barrier_forward_j_per_molecule - e_upper))


def beyer_swinehart(
    frequencies_cm1: Iterable[float], grain_cm1: float, grain_count: int
) -> np.ndarray:
    frequencies = tuple(float(value) for value in frequencies_cm1)
    if grain_cm1 <= 0.0 or grain_count <= 0:
        raise ValueError("grain_cm1 and grain_count must be positive")
    if HAS_NATIVE:
        return np.asarray(_cpp.beyer_swinehart(frequencies, grain_cm1, grain_count), dtype=float)
    density = np.zeros(grain_count, dtype=float)
    density[0] = 1.0
    for frequency in frequencies:
        if frequency <= 0.0:
            continue
        step = max(1, int(round(frequency / grain_cm1)))
        for index in range(step, grain_count):
            density[index] += density[index - step]
    return density

