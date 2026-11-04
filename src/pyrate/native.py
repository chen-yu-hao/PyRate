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
    """Refine a *local* CVT bottleneck without extrapolating past adjacent samples.

    A three-point parabola is used even when a wider path is available.  The
    stationary point is accepted only for a concave parabola, within the
    immediate bracketing nodes, and when it does not reduce the sampled maximum.
    This avoids spurious global quartic overshoot on noisy/free-energy paths.
    """

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if x_values.size == 0 or not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("x and y must be nonempty and finite")
    if np.any(np.diff(x_values) <= 0.0):
        raise ValueError("x must be strictly increasing")
    if HAS_NATIVE:
        coordinate, value, sample_index, fit_points = _cpp.refine_maximum(
            x_values.tolist(), y_values.tolist()
        )
        return RefinedMaximum(
            float(coordinate), float(value), int(sample_index), int(fit_points)
        )

    index = int(np.argmax(y_values))
    sampled = RefinedMaximum(float(x_values[index]), float(y_values[index]), index, 1)
    if index == 0 or index == x_values.size - 1:
        return sampled

    left, center, right = (float(v) for v in x_values[index - 1:index + 2])
    yl, yc, yr = (float(v) for v in y_values[index - 1:index + 2])
    h_left, h_right = center - left, right - center
    slope_left = (yc - yl) / h_left
    slope_right = (yr - yc) / h_right
    curvature = 2.0 * (slope_right - slope_left) / (h_left + h_right)
    if curvature >= 0.0 or not math.isfinite(curvature):
        return sampled
    slope_center = (h_right * slope_left + h_left * slope_right) / (h_left + h_right)
    displacement = -slope_center / curvature
    peak = center + displacement
    value = yc + slope_center * displacement + 0.5 * curvature * displacement**2
    if not (left <= peak <= right and math.isfinite(value) and value >= yc):
        return sampled
    return RefinedMaximum(peak, value, index, 3)


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


def _log_semiclassical_probability(action: float) -> float:
    """Stable log of smooth transmission for signed semiclassical action."""

    if action >= 0.0:
        return -action - math.log1p(math.exp(-action))
    return -math.log1p(math.exp(action))


def thermal_wkb_kappa(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    temperature_k: float,
    reduced_mass_kg: float,
    energy_grid_points: int = 400,
) -> float:
    """Canonical transmission divided by the classical step-barrier result.

    The energy origin is the larger asymptotic endpoint (threshold); energies
    below this threshold have no open reactant/product scattering channel.
    The probability is smoothed through the barrier top rather than forcing
    transmission to unity (which would omit nonclassical reflection).
    """

    coordinates = tuple(float(value) for value in coordinates_m)
    potentials = tuple(float(value) for value in potentials_j_per_molecule)
    if len(coordinates) != len(potentials) or len(coordinates) < 3:
        raise ValueError("coordinates and potentials require matching lengths of at least 3")
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("coordinates must be strictly increasing")
    if not all(map(math.isfinite, coordinates + potentials)):
        raise ValueError("coordinates and potentials must be finite")
    if not math.isfinite(temperature_k) or not math.isfinite(reduced_mass_kg) or temperature_k <= 0 or reduced_mass_kg <= 0:
        raise ValueError("temperature and reduced mass must be finite and positive")
    if energy_grid_points < 20:
        raise ValueError("energy_grid_points must be at least 20")
    if HAS_NATIVE:
        return float(
            _wkb_cpp.thermal_wkb_kappa(
                coordinates, potentials, float(temperature_k), float(reduced_mass_kg),
                int(energy_grid_points),
            )
        )
    reference = potentials[0]
    shifted = tuple(value - reference for value in potentials)
    threshold = max(0.0, shifted[-1])
    barrier = max(shifted)
    if barrier <= threshold:
        return 1.0
    beta = 1.0 / (BOLTZMANN_J_K * temperature_k)
    height = barrier - threshold
    # Integrate in reduced thermal energy u = beta*(E-threshold).
    # Include a thermal tail above the barrier, which captures reflection.
    upper_u = beta * height + 40.0
    intervals = energy_grid_points + energy_grid_points % 2
    du = upper_u / intervals

    peak = max(range(len(shifted)), key=shifted.__getitem__)
    omega = 0.0
    if 0 < peak < len(shifted) - 1:
        x0, x1, x2 = coordinates[peak - 1 : peak + 2]
        v0, v1, v2 = shifted[peak - 1 : peak + 2]
        curvature = 2.0 * (
            (v2 - v1) / (x2 - x1) - (v1 - v0) / (x1 - x0)
        ) / (x2 - x0)
        if curvature < 0.0:
            omega = math.sqrt(-curvature / reduced_mass_kg)

    def log_integrand(u: float) -> float:
        energy = threshold + u / beta
        if energy < barrier:
            action = wkb_exponent(coordinates, shifted, energy, reduced_mass_kg)
        else:
            action = (
                -2.0 * math.pi * (energy - barrier) / (HBAR_J_S * omega)
                if omega > 0.0
                else -math.inf
            )
        return -u + _log_semiclassical_probability(action)

    # Log-sum-exp preserves canonical transmission for high barriers without
    # clipping the physical rate or underflowing the barrier-top contribution.
    log_terms = [log_integrand(index * du) for index in range(intervals + 1)]
    maximum = max(log_terms)
    if maximum == -math.inf:
        return 0.0
    total = math.fsum(
        (1.0 if i in (0, intervals) else (4.0 if i % 2 else 2.0))
        * math.exp(value - maximum)
        for i, value in enumerate(log_terms)
    )
    log_kappa = beta * height + math.log(du / 3.0) + maximum + math.log(total)
    if log_kappa > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return math.inf
    return math.exp(log_kappa)


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
