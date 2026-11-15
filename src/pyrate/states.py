"""Microcanonical state-counting and variational (μVT / US) helpers."""

from __future__ import annotations

import math

import numpy as np

from .constants import (
    AMU_KG,
    ANGSTROM_M,
    BOLTZMANN_J_K,
    GAS_CONSTANT_J_MOL_K,
    LIGHT_SPEED_M_S,
    PLANCK_J_S,
    WAVENUMBER_TO_KJ_MOL,
)
from .geometry import rotor_geometry
from .models import ReactionPath, Species
from .native import beyer_swinehart
from .partition import zero_point_energy_kj_mol


def vibrational_density_of_states(
    frequencies_cm1: tuple[float, ...] | list[float],
    grain_cm1: float,
    maximum_energy_cm1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return energy grains and Beyer–Swinehart state counts.

    Frequencies at or below zero are ignored.
    """

    if maximum_energy_cm1 < 0.0:
        raise ValueError("maximum_energy_cm1 cannot be negative")
    grain_count = int(maximum_energy_cm1 // grain_cm1) + 1
    energies = np.arange(grain_count, dtype=float) * grain_cm1
    density = beyer_swinehart(frequencies_cm1, grain_cm1, grain_count)
    return energies, density


def vibrational_sum_of_states(
    frequencies_cm1: tuple[float, ...] | list[float],
    grain_cm1: float,
    maximum_energy_cm1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return energy grains and cumulative Beyer–Swinehart vibrational sum of states N_vib(E)."""

    energies, density = vibrational_density_of_states(
        frequencies_cm1, grain_cm1, maximum_energy_cm1
    )
    return energies, np.cumsum(density)


def rovibrational_sum_of_states(
    species: Species,
    energies_cm1: np.ndarray,
    grain_cm1: float,
    *,
    reference_vag_kj_mol: float = 0.0,
) -> np.ndarray:
    """Return the rovibrational-electronic sum of states N^GT(E) on an energy grid.

    ``energies_cm1`` is measured above ``reference_vag_kj_mol``.  At each grain,
    the Beyer–Swinehart vibrational density of states is convolved with the
    rigid-rotor rotational sum of states and weighted by the ground electronic
    degeneracy.
    """

    vag_kj_mol = species.electronic_energy_kj_mol + zero_point_energy_kj_mol(species)
    shift_cm1 = max(0.0, (vag_kj_mol - reference_vag_kj_mol) / WAVENUMBER_TO_KJ_MOL)
    shift_grains = int(round(shift_cm1 / grain_cm1))
    n_grains = int(energies_cm1.size)
    if shift_grains >= n_grains:
        return np.zeros(n_grains, dtype=float)

    active_grains = n_grains - shift_grains
    vib_density = beyer_swinehart(
        species.positive_frequencies_cm1, grain_cm1, active_grains
    )
    geometry = rotor_geometry(species)
    sigma = float(species.symmetry_number)
    g_elec = float(species.electronic_states[0].degeneracy) if species.electronic_states else 1.0

    if geometry.rotor_type == "atomic":
        rovib_sum_active = g_elec * np.cumsum(vib_density)
    else:
        conversion = AMU_KG * (ANGSTROM_M**2)
        moments = tuple(m * conversion for m in geometry.moments_amu_a2)
        grain_j = PLANCK_J_S * LIGHT_SPEED_M_S * 100.0 * grain_cm1
        # Evaluate N_rot(E_rot) on the active energy grid (with half-grain offset for
        # accurate Laplace transform recovery at low E_rot).
        k_indices = np.arange(active_grains, dtype=float) + 0.5
        e_rot_j = k_indices * grain_j
        prefactor = 8.0 * (math.pi**2) / (PLANCK_J_S**2)
        if geometry.rotor_type == "linear":
            nonzero = [m for m in moments if m > 1.0e-52]
            inertia = sum(nonzero) / len(nonzero) if nonzero else 1.0e-47
            n_rot = (prefactor * inertia / sigma) * e_rot_j
        else:
            prod_i = max(1.0e-150, moments[0] * moments[1] * moments[2])
            n_rot = (
                (4.0 * math.sqrt(2.0) * math.pi / (3.0 * sigma * (PLANCK_J_S**3)))
                * math.sqrt(prod_i)
                * np.power(e_rot_j, 1.5)
            )
        rovib_sum_active = g_elec * np.convolve(vib_density, n_rot)[:active_grains]

    result = np.zeros(n_grains, dtype=float)
    result[shift_grains:] = rovib_sum_active
    return result


def _unified_statistical_sum_of_states_1d(n_of_s: np.ndarray) -> float:
    """Compute Miller's microcanonical unified statistical N^US(E) across s at fixed E."""

    positive = n_of_s[n_of_s > 0.0]
    if positive.size == 0:
        return 0.0
    n_min_global = float(np.min(positive))
    if np.any(n_of_s <= 0.0):
        return n_min_global

    # Local minima of N(E, s) correspond to bottlenecks (maxima of 1 / N(E, s)).
    inv_n = 1.0 / n_of_s
    max_indices: list[int] = []
    for idx in range(1, int(inv_n.size) - 1):
        if inv_n[idx] > inv_n[idx - 1] and inv_n[idx] >= inv_n[idx + 1]:
            max_indices.append(idx)
    if len(max_indices) <= 1:
        return n_min_global

    inv_us = sum(float(inv_n[idx]) for idx in max_indices)
    for k in range(len(max_indices) - 1):
        left = max_indices[k]
        right = max_indices[k + 1]
        min_inv = float(np.min(inv_n[left : right + 1]))
        inv_us -= min_inv
    return 1.0 / inv_us if inv_us > 0.0 else n_min_global


def microcanonical_path_factors(
    path: ReactionPath,
    temperature_k: float,
    *,
    grain_cm1: float = 25.0,
) -> tuple[float, float]:
    """Return ``(gamma_muvt, gamma_us)`` relative to the minimum-flux dividing surface.

    ``gamma_muvt = k^muVT(T) / k^CVT(T) <= 1`` and ``gamma_us = k^US(T) / k^CVT(T) <= 1``
    are computed by Boltzmann-integrating the microcanonical sum of states
    ``N^GT(E, s)`` along the reaction path.
    """

    vag_values = [
        pt.dividing_surface.electronic_energy_kj_mol
        + zero_point_energy_kj_mol(pt.dividing_surface)
        for pt in path.points
    ]
    vag_min = min(vag_values)
    vag_max = max(vag_values)
    kt_cm1 = (BOLTZMANN_J_K * temperature_k) / (PLANCK_J_S * LIGHT_SPEED_M_S * 100.0)
    span_cm1 = max(0.0, (vag_max - vag_min) / WAVENUMBER_TO_KJ_MOL)
    max_energy_cm1 = min(40000.0, max(2000.0, span_cm1 + 14.0 * kt_cm1))
    effective_grain = max(grain_cm1, max_energy_cm1 / 1200.0)

    energies_cm1, _ = vibrational_density_of_states((), effective_grain, max_energy_cm1)
    n_matrix = np.vstack(
        [
            rovibrational_sum_of_states(
                pt.dividing_surface,
                energies_cm1,
                effective_grain,
                reference_vag_kj_mol=vag_min,
            )
            for pt in path.points
        ]
    )
    # Boltzmann weights exp(-(E - E_max_barrier) / k_B T) to avoid underflow
    shift_ref_cm1 = span_cm1
    boltzmann = np.exp(-np.clip((energies_cm1 - shift_ref_cm1) / kt_cm1, -700.0, 700.0))

    # Thermal integral for each fixed dividing surface s_k
    surface_integrals = n_matrix @ boltzmann
    min_surface_integral = float(np.min(surface_integrals))
    if min_surface_integral <= 0.0:
        return 1.0, 1.0

    n_muvt = np.min(n_matrix, axis=0)
    muvt_integral = float(np.dot(n_muvt, boltzmann))
    gamma_muvt = min(1.0, max(0.0, muvt_integral / min_surface_integral))

    n_us = np.array(
        [_unified_statistical_sum_of_states_1d(n_matrix[:, col]) for col in range(n_matrix.shape[1])],
        dtype=float,
    )
    us_integral = float(np.dot(n_us, boltzmann))
    gamma_us = min(gamma_muvt, max(0.0, us_integral / min_surface_integral))
    return gamma_muvt, gamma_us

