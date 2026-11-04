"""Microcanonical state-counting helpers."""

from __future__ import annotations

import numpy as np

from .native import beyer_swinehart


def vibrational_density_of_states(
    frequencies_cm1: tuple[float, ...] | list[float],
    grain_cm1: float,
    maximum_energy_cm1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return energy grains and Beyer–Swinehart state counts.

    This is the numerical foundation needed by future μVT/US and pressure-
    dependent modules.  Frequencies at or below zero are ignored.
    """

    if maximum_energy_cm1 < 0.0:
        raise ValueError("maximum_energy_cm1 cannot be negative")
    grain_count = int(maximum_energy_cm1 // grain_cm1) + 1
    energies = np.arange(grain_count, dtype=float) * grain_cm1
    density = beyer_swinehart(frequencies_cm1, grain_cm1, grain_count)
    return energies, density
