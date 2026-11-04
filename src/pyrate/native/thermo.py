"""Harmonic partition-function and Wigner kernels."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from ..constants import BOLTZMANN_J_K, LIGHT_SPEED_M_S, PLANCK_J_S
from ._loader import cpp


def py_harmonic_log_partition(frequencies_cm1: Sequence[float], temperature_k: float) -> float:
    total = 0.0
    for frequency in frequencies_cm1:
        if frequency <= 0.0:
            continue
        ratio = PLANCK_J_S * LIGHT_SPEED_M_S * 100.0 * frequency / (BOLTZMANN_J_K * temperature_k)
        total -= math.log1p(-math.exp(-ratio))
    return total


def harmonic_log_partition(frequencies_cm1: Iterable[float], temperature_k: float) -> float:
    frequencies = tuple(float(value) for value in frequencies_cm1)
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if cpp is not None:
        return float(cpp.harmonic_log_partition(frequencies, float(temperature_k)))
    return py_harmonic_log_partition(frequencies, float(temperature_k))


def py_wigner_kappa(imaginary_frequency_cm1: float, temperature_k: float) -> float:
    ratio = (
        PLANCK_J_S
        * LIGHT_SPEED_M_S
        * 100.0
        * abs(imaginary_frequency_cm1)
        / (BOLTZMANN_J_K * temperature_k)
    )
    return 1.0 + ratio * ratio / 24.0


def wigner_kappa(imaginary_frequency_cm1: float, temperature_k: float) -> float:
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if cpp is not None:
        return float(cpp.wigner_kappa(float(imaginary_frequency_cm1), float(temperature_k)))
    return py_wigner_kappa(float(imaginary_frequency_cm1), float(temperature_k))
