"""Local refinement of a sampled variational maximum (used by CVT)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._loader import cpp


@dataclass(frozen=True, slots=True)
class RefinedMaximum:
    coordinate: float
    value: float
    sample_index: int
    fit_points: int


def _validate(x: Sequence[float], y: Sequence[float]) -> tuple[list[float], list[float]]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if x_values.size == 0:
        raise ValueError("x and y cannot be empty")
    if np.any(np.diff(x_values) <= 0.0):
        raise ValueError("x must be strictly increasing")
    return [float(value) for value in x_values], [float(value) for value in y_values]


def py_refine_maximum(x: Sequence[float], y: Sequence[float]) -> RefinedMaximum:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
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
    return RefinedMaximum(center + scale * candidates[best], best_value, maximum_index, count)


def refine_maximum(x: Sequence[float], y: Sequence[float]) -> RefinedMaximum:
    x_list, y_list = _validate(x, y)
    if cpp is not None:
        coordinate, value, sample_index, fit_points = cpp.refine_maximum(x_list, y_list)
        return RefinedMaximum(float(coordinate), float(value), int(sample_index), int(fit_points))
    return py_refine_maximum(x_list, y_list)
