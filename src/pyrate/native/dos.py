"""Beyer–Swinehart state counting."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from ._loader import cpp


def py_beyer_swinehart(
    frequencies_cm1: Sequence[float], grain_cm1: float, grain_count: int
) -> npt.NDArray[np.float64]:
    density = np.zeros(grain_count, dtype=np.float64)
    density[0] = 1.0
    for frequency in frequencies_cm1:
        if frequency <= 0.0:
            continue
        step = max(1, int(round(frequency / grain_cm1)))
        for index in range(step, grain_count):
            density[index] += density[index - step]
    return density


def beyer_swinehart(
    frequencies_cm1: Iterable[float], grain_cm1: float, grain_count: int
) -> npt.NDArray[np.float64]:
    frequencies = tuple(float(value) for value in frequencies_cm1)
    if grain_cm1 <= 0.0 or grain_count <= 0:
        raise ValueError("grain_cm1 and grain_count must be positive")
    if cpp is not None:
        return np.asarray(
            cpp.beyer_swinehart(frequencies, float(grain_cm1), int(grain_count)),
            dtype=np.float64,
        )
    return py_beyer_swinehart(frequencies, float(grain_cm1), int(grain_count))
