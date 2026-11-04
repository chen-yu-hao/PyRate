"""One-dimensional semiclassical barrier transmission kernels.

``wkb_exponent`` returns the barrier-penetration integral ``γ(E) = 2θ(E)`` of a
piecewise-linear profile.  ``semiclassical_transmission`` and
``thermal_wkb_kappa`` use the uniform semiclassical probability of Polyrate's
ZCT/SCT methods with nonclassical reflection above the barrier:

* ``V^AG = max(V)`` and threshold ``E0 = max(V[0], V[-1])``;
* ``P(E) = 0`` for ``E < E0``;
* ``P(E) = 1 / (1 + exp(γ(E)))`` for ``E0 <= E <= V^AG``;
* ``P(E) = 1 - P(2 V^AG - E)`` for ``V^AG < E < 2 V^AG - E0``;
* ``P(E) = 1`` for ``E >= 2 V^AG - E0``.

``κ(T) = β exp(β V^AG) ∫_{E0}^{∞} P(E) exp(-βE) dE`` is the correction relative
to the classical threshold at ``V^AG``.
"""

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


def _safe_exp(exponent: float) -> float:
    # Same clamp as pyrate::safe_exp in cpp/common.hpp.
    if exponent < -745.0:
        return 0.0
    if exponent > 709.0:
        return math.inf
    return math.exp(exponent)


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


def py_semiclassical_transmission(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    energy_j_per_molecule: float,
    reduced_mass_kg: float,
) -> float:
    barrier = max(potentials_j_per_molecule)
    threshold = max(potentials_j_per_molecule[0], potentials_j_per_molecule[-1])
    energy = energy_j_per_molecule
    if energy < threshold:
        return 0.0
    if energy <= barrier:
        # P = 1/(1+e^γ) written as e^-γ/(1+e^-γ): γ >= 0, so nothing overflows.
        exponent = py_wkb_exponent(
            coordinates_m, potentials_j_per_molecule, energy, reduced_mass_kg
        )
        damping = math.exp(-exponent)
        return damping / (1.0 + damping)
    if energy < 2.0 * barrier - threshold:
        # Nonclassical reflection: P(E) = 1 - P(2V^AG - E) = 1/(1+e^-γ(2V^AG-E)).
        exponent = py_wkb_exponent(
            coordinates_m, potentials_j_per_molecule, 2.0 * barrier - energy, reduced_mass_kg
        )
        return 1.0 / (1.0 + math.exp(-exponent))
    return 1.0


def semiclassical_transmission(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    energy_j_per_molecule: float,
    reduced_mass_kg: float,
) -> float:
    """Return the uniform semiclassical transmission probability ``P(E)``.

    Energies share the absolute scale of ``potentials_j_per_molecule``.
    """

    coordinates, potentials = _validate_profile(coordinates_m, potentials_j_per_molecule, 2)
    if reduced_mass_kg <= 0.0:
        raise ValueError("reduced mass must be positive")
    if cpp is not None:
        return float(
            cpp.semiclassical_transmission(
                coordinates, potentials, float(energy_j_per_molecule), float(reduced_mass_kg)
            )
        )
    return py_semiclassical_transmission(
        coordinates, potentials, float(energy_j_per_molecule), float(reduced_mass_kg)
    )


def py_thermal_wkb_kappa(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    temperature_k: float,
    reduced_mass_kg: float,
    energy_grid_points: int = 400,
) -> float:
    # Measure energies from the barrier top, so E = V^AG - δ becomes E = -δ.
    barrier = max(potentials_j_per_molecule)
    shifted = tuple(value - barrier for value in potentials_j_per_molecule)
    depth = -max(shifted[0], shifted[-1])  # Δ = V^AG - E0 >= 0
    if depth <= 0.0:
        return 1.0
    count = energy_grid_points + energy_grid_points % 2
    beta = 1.0 / (BOLTZMANN_J_K * temperature_k)
    step = depth / count

    # κ = β ∫_0^Δ [P(V^AG-δ) e^{βδ} + (1-P(V^AG-δ)) e^{-βδ}] dδ + e^{-βΔ}.
    # The first term is evaluated as exp(βδ - log(1+e^γ)) so that deep
    # tunneling at low temperature neither overflows nor underflows.
    def integrand(index: int) -> float:
        delta = depth if index == count else index * step
        exponent = py_wkb_exponent(coordinates_m, shifted, -delta, reduced_mass_kg)
        damping = math.exp(-exponent)
        scaled = beta * delta
        log_tunneling = scaled - (exponent + math.log1p(damping))
        reflected = math.exp(-scaled) / (1.0 + damping)
        return _safe_exp(log_tunneling) + reflected

    total = integrand(0) + integrand(count)
    for index in range(1, count):
        total += (2.0 if index % 2 == 0 else 4.0) * integrand(index)
    return (beta * step) * total / 3.0 + math.exp(-beta * depth)


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
