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
        "version": _cpp.native_version() if HAS_NATIVE else "python-fallback-0.1",
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
    count = min(5, int(x_values.size))
    start = max(0, maximum_index - count // 2)
    start = min(start, int(x_values.size) - count)
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


def _safe_exp(value: float) -> float:
    if value < -745.0:
        return 0.0
    if value > 709.0:
        return math.inf
    return math.exp(value)


def thermal_wkb_kappa(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    temperature_k: float,
    reduced_mass_kg: float,
    energy_grid_points: int = 400,
    transmission: str = "uniform",
) -> float:
    """Canonical 1D semiclassical tunneling coefficient.

    ``transmission="uniform"`` (default) uses Kemble's uniform semiclassical
    probability ``P = 1/(1 + e^{2θ})`` with nonclassical reflection above the
    barrier top, which is exact for a parabolic barrier.  ``"primitive"`` keeps
    the earlier ``P = e^{-2θ}`` (``P = 1`` above the top) form, which
    overestimates κ at high temperature (κ − 1 ≈ u/2π).
    """
    if transmission not in {"uniform", "primitive"}:
        raise ValueError("transmission must be 'uniform' or 'primitive'")
    uniform = transmission == "uniform"
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
                uniform,
            )
        )

    shifted = tuple(value - potentials[0] for value in potentials)
    barrier = max(shifted)
    if barrier <= 0.0:
        return 1.0
    count = energy_grid_points + energy_grid_points % 2
    beta = 1.0 / (BOLTZMANN_J_K * temperature_k)
    step = barrier / count

    def integrand(energy: float) -> float:
        exponent = wkb_exponent(coordinates, shifted, energy, reduced_mass_kg)
        if uniform:
            a = beta * (barrier - energy)
            log_one_plus = (exponent + math.log1p(math.exp(-exponent)) if exponent > 0.0
                            else math.log1p(math.exp(exponent)))
            return beta * (_safe_exp(a - log_one_plus) - _safe_exp(-a - log_one_plus))
        log_value = beta * (barrier - energy) - exponent
        if log_value < -745.0:
            return 0.0
        if log_value > 709.0:
            return math.inf
        return beta * math.exp(log_value)

    total = integrand(0.0) + integrand(barrier)
    for index in range(1, count):
        total += (2.0 if index % 2 == 0 else 4.0) * integrand(index * step)
    return 1.0 + step * total / 3.0


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
