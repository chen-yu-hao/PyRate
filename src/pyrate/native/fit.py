"""Local refinement of a sampled variational maximum (used by CVT).

With ``i`` the index of the first sampled maximum and
``B = y[i] - min(y[i-1], y[i+1])``, every refined result satisfies

* ``x[i-1] <= coordinate <= x[i+1]``;
* ``y[i] <= value <= y[i] + B``.

A fit that would violate either bound is never reported.  The default fit is
the parabola through ``(i-1, i, i+1)``; ``max_points=4`` or ``5`` opts in to the
interpolating polynomial through that many points, which falls back to the
parabola and finally to the sample itself (``fit_points == 1``).

``py_refine_maximum`` mirrors ``cpp/fit.cpp`` operation by operation so that the
two backends agree to rounding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._loader import cpp

MIN_FIT_POINTS = 3
MAX_FIT_POINTS = 5
_BISECTION_STEPS = 100


@dataclass(frozen=True, slots=True)
class RefinedMaximum:
    """Refined maximum; ``fit_points == 1`` means the sample itself."""

    coordinate: float
    value: float
    sample_index: int
    fit_points: int


def _validate(
    x: Sequence[float], y: Sequence[float], max_points: int
) -> tuple[list[float], list[float]]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if x_values.size == 0:
        raise ValueError("x and y cannot be empty")
    if not (np.all(np.isfinite(x_values)) and np.all(np.isfinite(y_values))):
        raise ValueError("x and y must be finite")
    if np.any(np.diff(x_values) <= 0.0):
        raise ValueError("x must be strictly increasing")
    if not MIN_FIT_POINTS <= max_points <= MAX_FIT_POINTS:
        raise ValueError("max_points must be 3, 4 or 5")
    return [float(value) for value in x_values], [float(value) for value in y_values]


def _horner(coefficients: Sequence[float], t: float) -> float:
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * t + coefficient
    return result


def _derivative(coefficients: Sequence[float]) -> list[float]:
    return [float(degree) * coefficients[degree] for degree in range(1, len(coefficients))]


def _admit(
    coordinate: float, value: float, x: Sequence[float], y: Sequence[float], i: int
) -> tuple[float, float] | None:
    """Accept a refined point only inside the bracket and below the drop bound."""

    ceiling = y[i] + (y[i] - min(y[i - 1], y[i + 1]))
    if not (math.isfinite(coordinate) and math.isfinite(value)):
        return None
    if not (value >= y[i] and value <= ceiling):
        return None
    # The fits keep the coordinate inside the bracket analytically; the clamp
    # only absorbs rounding in x[i] + offset.
    return min(max(coordinate, x[i - 1]), x[i + 1]), value


def _quadratic_vertex(x: Sequence[float], y: Sequence[float], i: int) -> tuple[float, float] | None:
    """Vertex of the parabola through ``(i-1, i, i+1)``.

    Writing ``p(x[i] + t) = y[i] + b t + a t²`` with spacings ``h_l, h_r`` and
    drops ``d_l = y[i] - y[i-1] > 0`` (``i`` is the first maximum) and
    ``d_r = y[i] - y[i+1] >= 0``::

        a = -(d_l h_r + d_r h_l) / H,  b = (d_l h_r² - d_r h_l²) / H,
        H = h_l h_r (h_l + h_r),

    so ``a < 0``, the vertex ``t* = -b / (2a)`` lies in ``[-h_l/2, h_r/2]`` and
    ``p(t*) = y[i] + b t*/2 >= y[i]``.
    """

    h_left = x[i] - x[i - 1]
    h_right = x[i + 1] - x[i]
    drop_left = y[i] - y[i - 1]
    drop_right = y[i] - y[i + 1]
    numerator = drop_left * h_right * h_right - drop_right * h_left * h_left
    weight = drop_left * h_right + drop_right * h_left
    spacing = h_left * h_right * (h_left + h_right)
    if not (weight > 0.0 and spacing > 0.0):
        return None
    offset = numerator / (2.0 * weight)
    slope = numerator / spacing
    return _admit(x[i] + offset, y[i] + 0.5 * slope * offset, x, y, i)


def _interpolating_polynomial(t: Sequence[float], values: Sequence[float]) -> list[float] | None:
    """Monomial coefficients (lowest degree first) via Newton divided differences."""

    count = len(t)
    divided = list(values)
    for level in range(1, count):
        for k in range(count - 1, level - 1, -1):
            denominator = t[k] - t[k - level]
            if not denominator > 0.0:
                return None
            divided[k] = (divided[k] - divided[k - 1]) / denominator
    coefficients = [divided[count - 1]]
    for k in range(count - 2, -1, -1):
        # coefficients <- coefficients * (t - t[k]) + divided[k]
        expanded = [0.0] * (len(coefficients) + 1)
        for degree in range(len(expanded)):
            shifted = coefficients[degree - 1] if degree >= 1 else 0.0
            scaled = t[k] * coefficients[degree] if degree < len(coefficients) else 0.0
            expanded[degree] = shifted - scaled
        expanded[0] = expanded[0] + divided[k]
        coefficients = expanded
    return coefficients


def _interior_roots(coefficients: Sequence[float], lower: float, upper: float) -> list[float]:
    """Real roots of a polynomial of degree <= 2 strictly inside ``(lower, upper)``."""

    constant = coefficients[0] if len(coefficients) > 0 else 0.0
    linear = coefficients[1] if len(coefficients) > 1 else 0.0
    quadratic = coefficients[2] if len(coefficients) > 2 else 0.0
    found: list[float] = []
    if quadratic != 0.0:
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant >= 0.0:
            q = -0.5 * (linear + math.copysign(math.sqrt(discriminant), linear))
            found.append(q / quadratic)
            if q != 0.0:
                found.append(constant / q)
    elif linear != 0.0:
        found.append(-constant / linear)
    return [root for root in found if lower < root < upper]


def _polynomial_maximum(
    x: Sequence[float], y: Sequence[float], i: int, count: int
) -> tuple[float, float] | None:
    """Highest local maximum inside ``[x[i-1], x[i+1]]`` of the interpolant
    through ``count`` points around ``i``.

    The roots of ``p''`` split the bracket into pieces on which ``p'`` is
    monotonic; each + to - sign change of ``p'`` is located by bisection.
    """

    start = i - count // 2 if i >= count // 2 else 0
    if start + count > len(x):
        start = len(x) - count
    scale = max(x[i] - x[start], x[start + count - 1] - x[i])
    if not (math.isfinite(scale) and scale > 0.0):
        return None
    t = [(x[start + k] - x[i]) / scale for k in range(count)]
    coefficients = _interpolating_polynomial(t, y[start : start + count])
    if coefficients is None:
        return None
    slope = _derivative(coefficients)
    curvature = _derivative(slope)

    lower = t[i - 1 - start]
    upper = t[i + 1 - start]
    breaks = [lower, *sorted(_interior_roots(curvature, lower, upper)), upper]

    best: float | None = None
    best_value = y[i]
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        if not (_horner(slope, left) > 0.0 and _horner(slope, right) <= 0.0):
            continue
        for _ in range(_BISECTION_STEPS):
            middle = 0.5 * (left + right)
            if middle <= left or middle >= right:
                break
            if _horner(slope, middle) > 0.0:
                left = middle
            else:
                right = middle
        root = 0.5 * (left + right)
        value = _horner(coefficients, root)
        if value > best_value:
            best = root
            best_value = value
    if best is None:
        return None
    return _admit(x[i] + scale * best, best_value, x, y, i)


def py_refine_maximum(
    x: Sequence[float], y: Sequence[float], *, max_points: int = MIN_FIT_POINTS
) -> RefinedMaximum:
    """Pure-Python reference implementation (inputs assumed validated)."""

    x_values = [float(value) for value in x]
    y_values = [float(value) for value in y]
    i = max(range(len(y_values)), key=y_values.__getitem__)
    sample = RefinedMaximum(x_values[i], y_values[i], i, 1)
    if len(x_values) < 3 or i in {0, len(x_values) - 1}:
        return sample
    count = min(max_points, len(x_values))
    if count > 3:
        polynomial = _polynomial_maximum(x_values, y_values, i, count)
        if polynomial is not None:
            return RefinedMaximum(polynomial[0], polynomial[1], i, count)
    parabola = _quadratic_vertex(x_values, y_values, i)
    if parabola is not None:
        return RefinedMaximum(parabola[0], parabola[1], i, 3)
    return sample


def refine_maximum(
    x: Sequence[float], y: Sequence[float], *, max_points: int = MIN_FIT_POINTS
) -> RefinedMaximum:
    """Refine the first sampled maximum of ``y(x)``; see the module docstring."""

    x_list, y_list = _validate(x, y, max_points)
    if cpp is not None:
        coordinate, value, sample_index, fit_points = cpp.refine_maximum(
            x_list, y_list, int(max_points)
        )
        return RefinedMaximum(float(coordinate), float(value), int(sample_index), int(fit_points))
    return py_refine_maximum(x_list, y_list, max_points=int(max_points))
