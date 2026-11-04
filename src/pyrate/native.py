"""Stable Python interface to optional C++ numerical kernels.

The package remains usable from a source checkout when the extensions have not
been built. Production installations compile :mod:`pyrate._kernels` and
:mod:`pyrate._wkb_kernels`; the fallbacks below deliberately mirror the native
algorithms for reproducibility.
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

try:
    if _DISABLE_NATIVE:
        raise ImportError("native kernels disabled by PYRATE_DISABLE_NATIVE")
    from . import _wkb_kernels as _wkb_cpp
except ImportError:  # pragma: no cover - exercised explicitly through fallback tests
    _wkb_cpp = None

# A native installation is considered complete only when both extension modules
# are available. This avoids silently routing only part of one calculation
# through compiled code when a stale or partial build is imported.
HAS_NATIVE = _cpp is not None and _wkb_cpp is not None


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
        "version": (
            f"{_cpp.native_version()}+{_wkb_cpp.native_version()}"
            if HAS_NATIVE
            else "python-fallback-0.2"
        ),
    }


def _require_finite(values: Iterable[float], name: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")


def harmonic_log_partition(frequencies_cm1: Iterable[float], temperature_k: float) -> float:
    frequencies = tuple(float(value) for value in frequencies_cm1)
    if not math.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature must be positive and finite")
    _require_finite(frequencies, "frequencies_cm1")
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
    if not math.isfinite(imaginary_frequency_cm1):
        raise ValueError("imaginary frequency must be finite")
    if not math.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature must be positive and finite")
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
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("x and y must contain only finite values")
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
    values = [
        float(np.polynomial.polynomial.polyval(candidate, coefficients))
        for candidate in candidates
    ]
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
    """Integrate ``sqrt(max(u(t), 0))`` for a linear segment on ``[0, 1]``.

    The formulas are algebraically equivalent to the usual difference of
    three-halves powers but are scaled to the local energy. This is essential
    for molecular barriers, whose per-molecule energies are typically around
    ``1e-20`` joule and must never be compared with an absolute ``1.0`` scale.
    """

    if u0 <= 0.0 and u1 <= 0.0:
        return 0.0
    if u0 > 0.0 and u1 > 0.0:
        scale = max(u0, u1)
        root0 = math.sqrt(u0 / scale)
        root1 = math.sqrt(u1 / scale)
        return (
            (2.0 / 3.0)
            * math.sqrt(scale)
            * (root0 * root0 + root0 * root1 + root1 * root1)
            / (root0 + root1)
        )
    if u0 > 0.0:
        scale = max(u0, -u1)
        positive = u0 / scale
        negative = -u1 / scale
        return (
            (2.0 / 3.0)
            * math.sqrt(scale)
            * positive**1.5
            / (positive + negative)
        )
    scale = max(-u0, u1)
    negative = -u0 / scale
    positive = u1 / scale
    return (
        (2.0 / 3.0)
        * math.sqrt(scale)
        * positive**1.5
        / (positive + negative)
    )


def _wkb_exponent_python(
    coordinates: Sequence[float],
    potentials: Sequence[float],
    energy: float,
    reduced_mass: float,
) -> float:
    integral = 0.0
    for index in range(1, len(coordinates)):
        dx = coordinates[index] - coordinates[index - 1]
        u0 = potentials[index - 1] - energy
        u1 = potentials[index] - energy
        integral += dx * _linear_sqrt_integral(u0, u1)
    return 2.0 * math.sqrt(2.0 * reduced_mass) * integral / HBAR_J_S


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
    _require_finite(coordinates, "coordinates_m")
    _require_finite(potentials, "potentials_j_per_molecule")
    if not math.isfinite(energy_j_per_molecule):
        raise ValueError("energy must be finite")
    if not math.isfinite(reduced_mass_kg) or reduced_mass_kg <= 0.0:
        raise ValueError("reduced mass must be positive and finite")
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("coordinates must be strictly increasing")
    if HAS_NATIVE:
        return float(
            _wkb_cpp.wkb_exponent(
                coordinates, potentials, float(energy_j_per_molecule), float(reduced_mass_kg)
            )
        )
    return _wkb_exponent_python(
        coordinates, potentials, energy_j_per_molecule, reduced_mass_kg
    )


def _log_simpson_wkb_below_barrier(
    coordinates: Sequence[float],
    shifted_potentials: Sequence[float],
    temperature: float,
    reduced_mass: float,
    count: int,
    barrier: float,
) -> float:
    """Return log of the below-barrier canonical WKB contribution."""

    beta_barrier = barrier / (BOLTZMANN_J_K * temperature)
    log_values: list[float] = []
    weights: list[float] = []
    for index in range(count + 1):
        fraction = index / count
        energy = barrier * fraction
        exponent = _wkb_exponent_python(
            coordinates, shifted_potentials, energy, reduced_mass
        )
        log_values.append(beta_barrier * (1.0 - fraction) - exponent)
        weights.append(1.0 if index in {0, count} else (2.0 if index % 2 == 0 else 4.0))

    maximum = max(log_values)
    weighted_sum = math.fsum(
        weight * math.exp(log_value - maximum)
        for weight, log_value in zip(weights, log_values)
    )
    return (
        math.log(beta_barrier)
        - math.log(3.0 * count)
        + maximum
        + math.log(weighted_sum)
    )


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
    _require_finite(coordinates, "coordinates_m")
    _require_finite(potentials, "potentials_j_per_molecule")
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("coordinates must be strictly increasing")
    if not math.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if not math.isfinite(reduced_mass_kg) or reduced_mass_kg <= 0.0:
        raise ValueError("reduced mass must be positive and finite")
    if energy_grid_points < 20:
        raise ValueError("energy_grid_points must be at least 20")
    if HAS_NATIVE:
        return float(
            _wkb_cpp.thermal_wkb_kappa(
                coordinates,
                potentials,
                float(temperature_k),
                float(reduced_mass_kg),
                int(energy_grid_points),
            )
        )

    shifted = tuple(value - potentials[0] for value in potentials)
    barrier = max(shifted)
    if barrier <= 0.0:
        return 1.0
    count = energy_grid_points + energy_grid_points % 2
    log_below = _log_simpson_wkb_below_barrier(
        coordinates,
        shifted,
        temperature_k,
        reduced_mass_kg,
        count,
        barrier,
    )
    if log_below > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return math.inf
    if log_below < -745.0:
        return 1.0
    return 1.0 + math.exp(log_below)


def beyer_swinehart(
    frequencies_cm1: Iterable[float], grain_cm1: float, grain_count: int
) -> np.ndarray:
    frequencies = tuple(float(value) for value in frequencies_cm1)
    _require_finite(frequencies, "frequencies_cm1")
    if not math.isfinite(grain_cm1) or grain_cm1 <= 0.0 or grain_count <= 0:
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
