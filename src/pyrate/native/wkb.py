"""One-dimensional WKB barrier transmission kernels."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..constants import BOLTZMANN_J_K, HBAR_J_S
from ._loader import cpp

_EPSILON = 2.220446049250313e-16


def _linear_sqrt_integral(u0: float, u1: float) -> float:
    if u0 <= 0.0 and u1 <= 0.0:
        return 0.0
    delta = u1 - u0
    scale = max(1.0, abs(u0), abs(u1))
    if abs(delta) <= _EPSILON * scale:
        return math.sqrt(max(0.0, u0))
    if u0 > 0.0 and u1 > 0.0:
        return (2.0 / 3.0) * (math.pow(u1, 1.5) - math.pow(u0, 1.5)) / delta
    if u0 > 0.0:
        return (2.0 / 3.0) * math.pow(u0, 1.5) / (u0 - u1)
    return (2.0 / 3.0) * math.pow(u1, 1.5) / (u1 - u0)


def _validate_profile(
    coordinates_m: Sequence[float], potentials_j: Sequence[float], minimum_points: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    coordinates = tuple(float(value) for value in coordinates_m)
    potentials = tuple(float(value) for value in potentials_j)
    if len(coordinates) != len(potentials) or len(coordinates) < minimum_points:
        raise ValueError(
            "coordinates and potentials require matching lengths of at least "
            f"{minimum_points}"
        )
    if any(right <= left for left, right in zip(coordinates, coordinates[1:], strict=False)):
        raise ValueError("coordinates must be strictly increasing")
    return coordinates, potentials


def py_wkb_exponent(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    energy_j_per_molecule: float,
    reduced_mass_kg: float,
) -> float:
    integral = 0.0
    for index in range(1, len(coordinates_m)):
        dx = coordinates_m[index] - coordinates_m[index - 1]
        u0 = potentials_j_per_molecule[index - 1] - energy_j_per_molecule
        u1 = potentials_j_per_molecule[index] - energy_j_per_molecule
        integral += dx * _linear_sqrt_integral(u0, u1)
    return 2.0 * math.sqrt(2.0 * reduced_mass_kg) * integral / HBAR_J_S


def wkb_exponent(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    energy_j_per_molecule: float,
    reduced_mass_kg: float,
) -> float:
    coordinates, potentials = _validate_profile(coordinates_m, potentials_j_per_molecule, 2)
    if reduced_mass_kg <= 0.0:
        raise ValueError("reduced mass must be positive")
    if cpp is not None:
        return float(
            cpp.wkb_exponent(
                coordinates, potentials, float(energy_j_per_molecule), float(reduced_mass_kg)
            )
        )
    return py_wkb_exponent(
        coordinates, potentials, float(energy_j_per_molecule), float(reduced_mass_kg)
    )


def py_thermal_wkb_kappa(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    temperature_k: float,
    reduced_mass_kg: float,
    energy_grid_points: int = 400,
) -> float:
    shifted = tuple(value - potentials_j_per_molecule[0] for value in potentials_j_per_molecule)
    barrier = max(shifted)
    if barrier <= 0.0:
        return 1.0
    count = energy_grid_points + energy_grid_points % 2
    beta = 1.0 / (BOLTZMANN_J_K * temperature_k)
    step = barrier / count

    def integrand(energy: float) -> float:
        exponent = py_wkb_exponent(coordinates_m, shifted, energy, reduced_mass_kg)
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


def thermal_wkb_kappa(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    temperature_k: float,
    reduced_mass_kg: float,
    energy_grid_points: int = 400,
) -> float:
    coordinates, potentials = _validate_profile(coordinates_m, potentials_j_per_molecule, 3)
    if temperature_k <= 0.0 or reduced_mass_kg <= 0.0:
        raise ValueError("temperature and reduced mass must be positive")
    if energy_grid_points < 20:
        raise ValueError("energy_grid_points must be at least 20")
    if cpp is not None:
        return float(
            cpp.thermal_wkb_kappa(
                coordinates,
                potentials,
                float(temperature_k),
                float(reduced_mass_kg),
                int(energy_grid_points),
            )
        )
    return py_thermal_wkb_kappa(
        coordinates, potentials, float(temperature_k), float(reduced_mass_kg),
        int(energy_grid_points),
    )
