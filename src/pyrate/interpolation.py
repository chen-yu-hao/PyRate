"""Cubic-spline interpolation of reaction-path profiles.

Reaction paths are usually stored on a much coarser grid than the one needed
to converge semiclassical action integrals.  The helpers here interpolate
path properties with a C2 cubic spline (not-a-knot end conditions) before
they are integrated; with fewer than four points they fall back to a natural
spline.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class CubicSpline:
    """Interpolating cubic spline through strictly increasing nodes."""

    def __init__(self, x: Sequence[float], y: Sequence[float]) -> None:
        nodes = np.asarray(x, dtype=float)
        values = np.asarray(y, dtype=float)
        if nodes.ndim != 1 or nodes.shape != values.shape:
            raise ValueError("x and y must be one-dimensional arrays of equal length")
        if nodes.size < 2:
            raise ValueError("a spline requires at least two points")
        steps = np.diff(nodes)
        if np.any(steps <= 0.0):
            raise ValueError("x must be strictly increasing")
        self._x = nodes
        self._y = values
        self._h = steps
        self._m = self._second_derivatives(nodes, values, steps)

    @staticmethod
    def _second_derivatives(x: np.ndarray, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        count = x.size
        if count == 2:
            return np.zeros(2, dtype=float)
        matrix = np.zeros((count, count), dtype=float)
        rhs = np.zeros(count, dtype=float)
        slopes = np.diff(y) / h
        for row in range(1, count - 1):
            matrix[row, row - 1] = h[row - 1]
            matrix[row, row] = 2.0 * (h[row - 1] + h[row])
            matrix[row, row + 1] = h[row]
            rhs[row] = 6.0 * (slopes[row] - slopes[row - 1])
        if count >= 4:
            # Not-a-knot: the third derivative is continuous at the first and
            # last interior nodes.
            matrix[0, 0], matrix[0, 1], matrix[0, 2] = h[1], -(h[0] + h[1]), h[0]
            matrix[-1, -3], matrix[-1, -2], matrix[-1, -1] = h[-1], -(h[-2] + h[-1]), h[-2]
        else:
            matrix[0, 0] = 1.0
            matrix[-1, -1] = 1.0
        return np.linalg.solve(matrix, rhs)

    def __call__(self, points: Sequence[float] | float) -> np.ndarray:
        query = np.atleast_1d(np.asarray(points, dtype=float))
        index = np.clip(np.searchsorted(self._x, query, side="right") - 1, 0, self._x.size - 2)
        h = self._h[index]
        left = (self._x[index + 1] - query) / h
        right = (query - self._x[index]) / h
        return (
            left * self._y[index]
            + right * self._y[index + 1]
            + ((left**3 - left) * self._m[index] + (right**3 - right) * self._m[index + 1])
            * h
            * h
            / 6.0
        )


def refined_grid(x: Sequence[float], subdivisions: int) -> np.ndarray:
    """Return ``x`` with every interval split into ``subdivisions`` equal parts."""

    nodes = np.asarray(x, dtype=float)
    if subdivisions < 1:
        raise ValueError("subdivisions must be at least 1")
    if subdivisions == 1 or nodes.size < 2:
        return nodes.copy()
    fractions = np.arange(subdivisions, dtype=float) / subdivisions
    inner = nodes[:-1, None] + np.diff(nodes)[:, None] * fractions[None, :]
    return np.concatenate([inner.ravel(), nodes[-1:]])
