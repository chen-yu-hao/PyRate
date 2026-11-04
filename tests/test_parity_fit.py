"""Guarantees and C++/Python parity for CVT maximum refinement (CVT area).

With ``i`` the first sampled maximum and ``B = y[i] - min(y[i-1], y[i+1])``,
``refine_maximum`` must return the sample (``fit_points == 1``) at an endpoint or
for fewer than three points, and otherwise keep ``x[i-1] <= coordinate <= x[i+1]``
and ``y[i] <= value <= y[i] + B``.  The property tests check this on 12 000
random cases per backend and mode; the cases are generated vectorised with
fixed seeds.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pyrate.native.fit import py_refine_maximum

Result = tuple[float, float, int, int]
Backend = Callable[[list[float], list[float], int], Result]

GRIDS = ("uniform", "non-uniform")
VALUES = ("smooth", "noise", "plateau")
SIZES = (2, 3, 4, 5, 7)
CASES_PER_BATCH = 400  # 2 grids x 3 value families x 5 sizes x 400 = 12 000 cases


def _python_backend(x: list[float], y: list[float], max_points: int) -> Result:
    result = py_refine_maximum(x, y, max_points=max_points)
    return result.coordinate, result.value, result.sample_index, result.fit_points


@pytest.fixture(params=["python", "cpp"])
def backend(request: pytest.FixtureRequest) -> Backend:
    if request.param == "python":
        return _python_backend
    cpp: Any = request.getfixturevalue("native")

    def run(x: list[float], y: list[float], max_points: int) -> Result:
        coordinate, value, index, points = cpp.refine_maximum(x, y, max_points)
        return float(coordinate), float(value), int(index), int(points)

    return run


def _grid(rng: np.random.Generator, kind: str, count: int, size: int) -> np.ndarray:
    origin = rng.uniform(-10.0, 10.0, (count, 1))
    if kind == "uniform":
        step = 10.0 ** rng.uniform(-3.0, 1.0, (count, 1))
        return origin + step * np.arange(size)
    # Neighbouring spacings differ by up to six decades.
    spacing = 10.0 ** rng.uniform(-3.0, 3.0, (count, size - 1))
    return origin + np.concatenate([np.zeros((count, 1)), np.cumsum(spacing, axis=1)], axis=1)


def _values(rng: np.random.Generator, kind: str, x: np.ndarray) -> np.ndarray:
    count, size = x.shape
    unit = 10.0 ** rng.uniform(-6.0, 6.0, (count, 1))  # unit independence
    offset = rng.normal(0.0, 1.0e3, (count, 1)) * unit  # mostly negative or positive
    if kind == "smooth":
        span = np.maximum(x[:, -1:] - x[:, :1], 1.0e-300)
        u = (x - x[:, :1]) / span
        peak = rng.uniform(-0.2, 1.2, (count, 1))
        skew = rng.uniform(-3.0, 3.0, (count, 1))
        noise = rng.uniform(0.0, 0.3, (count, 1)) * rng.normal(size=(count, size))
        shape = -((u - peak) ** 2) + skew * (u - peak) ** 3 + noise
    elif kind == "noise":
        shape = rng.normal(size=(count, size))
    else:  # plateaus and ties, including ties at the maximum
        shape = rng.integers(-2, 2, (count, size)).astype(np.float64)
    return offset + unit * shape


def _random_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    batches = []
    for seed, (grid, values, size) in enumerate(
        (grid, values, size) for grid in GRIDS for values in VALUES for size in SIZES
    ):
        rng = np.random.default_rng(1000 + seed)
        x = _grid(rng, grid, CASES_PER_BATCH, size)
        assert np.all(np.diff(x, axis=1) > 0.0)
        batches.append((grid, x, _values(rng, values, x)))
    return batches


CASES = _random_cases()


def _assert_cases(condition: np.ndarray, x: np.ndarray, y: np.ndarray, what: str) -> None:
    if not np.all(condition):
        row = int(np.flatnonzero(~condition)[0])
        raise AssertionError(f"{what} violated for x={x[row].tolist()} y={y[row].tolist()}")


def _run(backend: Backend, x: np.ndarray, y: np.ndarray, max_points: int) -> np.ndarray:
    return np.array(
        [backend(xs, ys, max_points) for xs, ys in zip(x.tolist(), y.tolist(), strict=True)]
    )


@pytest.mark.parametrize("max_points", [3, 4, 5])
def test_refine_maximum_guarantees_on_random_cases(backend: Backend, max_points: int) -> None:
    refined_cases = interior_cases = 0
    for grid, x, y in CASES:
        results = _run(backend, x, y, max_points)
        coordinate, value = results[:, 0], results[:, 1]
        index, points = results[:, 2].astype(int), results[:, 3].astype(int)
        rows = np.arange(x.shape[0])
        size = x.shape[1]
        expected = np.argmax(y, axis=1)  # first maximum
        _assert_cases(index == expected, x, y, "sample index")
        sample_x, sample_y = x[rows, expected], y[rows, expected]

        # Guarantee 1, and fallbacks report the sample itself.
        edge = (expected == 0) | (expected == size - 1)
        _assert_cases(~edge | (points == 1), x, y, "endpoint fallback")
        sampled = points == 1
        _assert_cases(~sampled | ((coordinate == sample_x) & (value == sample_y)), x, y, "sample")
        allowed = (points == 1) | (points == 3) | (points == min(max_points, size))
        _assert_cases(allowed, x, y, "fit_points")
        if size < 3:
            continue

        left = np.clip(expected - 1, 0, size - 1)
        right = np.clip(expected + 1, 0, size - 1)
        inside = ~edge
        # Guarantee 2: bracketed by the neighbouring samples.
        bracket = (x[rows, left] <= coordinate) & (coordinate <= x[rows, right])
        _assert_cases(~inside | bracket, x, y, "bracket")
        # Guarantee 3: y[i] <= value <= y[i] + B.
        ceiling = sample_y + (sample_y - np.minimum(y[rows, left], y[rows, right]))
        _assert_cases(~inside | ((sample_y <= value) & (value <= ceiling)), x, y, "bound")

        # On (near-)uniform grids the parabola overshoots by at most B/8, so an
        # interior maximum is always refined.
        if grid == "uniform":
            _assert_cases(~inside | ~sampled, x, y, "uniform-grid refinement")
        interior_cases += int(np.count_nonzero(inside))
        refined_cases += int(np.count_nonzero(inside & ~sampled))
    # The guards must not turn refinement off altogether.
    assert refined_cases > 0.5 * interior_cases


def _quadratic_cases(
    seed: int, max_ratio: float, count: int = 500
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    batches = []
    for size in (3, 5, 7, 9):
        if max_ratio > 1.0:
            spacing = rng.uniform(1.0, max_ratio, (count, size - 1))
        else:
            spacing = np.ones((count, size - 1))
        grid = np.concatenate([np.zeros((count, 1)), np.cumsum(spacing, axis=1)], axis=1)
        x = -1.0 + 2.0 * grid / grid[:, -1:]
        vertex = rng.uniform(x[:, 1], x[:, -2])
        top = rng.choice([-1.0, 1.0], count) * rng.uniform(1.0, 50.0, count)
        curvature = rng.uniform(0.5, 20.0, count)
        y = top[:, None] - curvature[:, None] * (x - vertex[:, None]) ** 2
        batches.append((x, y, vertex, top))
    return batches


@pytest.mark.parametrize("max_points", [3, 4, 5])
@pytest.mark.parametrize("max_ratio", [1.0, 4.0], ids=["uniform", "ratio<=4"])
def test_refine_maximum_is_exact_on_quadratic_data(
    backend: Backend, max_points: int, max_ratio: float
) -> None:
    # Guarantee 4: neighbouring spacings within a factor 2 + 2*sqrt(2) are always
    # accepted, so the refinement must be exact.
    for x, y, vertex, top in _quadratic_cases(11, max_ratio):
        results = _run(backend, x, y, max_points)
        assert np.all(results[:, 3] == min(max_points, x.shape[1]))
        assert np.max(np.abs(results[:, 0] - vertex)) <= 1.0e-12
        assert np.max(np.abs(results[:, 1] - top) / np.abs(top)) <= 1.0e-12


@pytest.mark.parametrize("max_points", [3, 5])
def test_quadratic_data_on_extreme_grids_is_exact_or_sampled(
    backend: Backend, max_points: int
) -> None:
    # Spacing ratios up to 1e3 may trip the drop bound; whatever is accepted
    # is still the true vertex, up to cancellation in the small drops.
    for x, y, vertex, top in _quadratic_cases(12, 1.0e3):
        results = _run(backend, x, y, max_points)
        fitted = results[:, 3] > 1
        assert np.count_nonzero(fitted) > 0.5 * fitted.size
        assert np.max(np.abs(results[fitted, 0] - vertex[fitted]), initial=0.0) <= 1.0e-10
        error = np.abs(results[fitted, 1] - top[fitted]) / np.abs(top[fitted])
        assert np.max(error, initial=0.0) <= 1.0e-10


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("max_points", [3, 4, 5])
def test_refine_maximum_parity_on_smooth_profiles(seed: int, max_points: int, native: Any) -> None:
    cpp = native
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1.0, 1.0, 9))
    peak = rng.uniform(-0.5, 0.5)
    y = 50.0 - 12.0 * (x - peak) ** 2 + 0.3 * (x - peak) ** 3
    coordinate, value, index, points = cpp.refine_maximum(x.tolist(), y.tolist(), max_points)
    reference = py_refine_maximum(x.tolist(), y.tolist(), max_points=max_points)
    assert (index, points) == (reference.sample_index, reference.fit_points)
    assert math.isclose(coordinate, reference.coordinate, rel_tol=1.0e-9, abs_tol=1.0e-9)
    assert math.isclose(value, reference.value, rel_tol=1.0e-12)


@pytest.mark.parametrize("max_points", [3, 4, 5])
def test_refine_maximum_parity_on_random_cases(max_points: int, native: Any) -> None:
    cpp = native
    for _, x, y in CASES:
        for xs, ys in zip(x.tolist(), y.tolist(), strict=True):
            coordinate, value, index, points = cpp.refine_maximum(xs, ys, max_points)
            reference = py_refine_maximum(xs, ys, max_points=max_points)
            assert (index, points) == (reference.sample_index, reference.fit_points)
            assert math.isclose(coordinate, reference.coordinate, rel_tol=1.0e-9, abs_tol=1.0e-9)
            # Relative to the data scale, so values that happen to sit near zero
            # do not demand more than 1e-12 of the profile.
            scale = max(abs(item) for item in ys)
            assert math.isclose(value, reference.value, rel_tol=1.0e-12, abs_tol=1.0e-12 * scale)


def test_refine_maximum_parity_on_audit_counter_example(native: Any) -> None:
    x = [-0.294, -0.234, 0.117, 0.849, 0.919]
    y = [2.99, 0.081, 3.404, 2.422, 0.784]
    for max_points in (3, 4, 5):
        reference = py_refine_maximum(x, y, max_points=max_points)
        assert (reference.sample_index, reference.fit_points) == (2, 3)
        assert native.refine_maximum(x, y, max_points) == (
            pytest.approx(reference.coordinate, rel=1.0e-12),
            pytest.approx(reference.value, rel=1.0e-12),
            2,
            3,
        )


@pytest.mark.parametrize(
    ("x", "y", "extra"),
    [
        ([0.0, 1.0, 2.0], [0.0, math.nan, 1.0], ()),
        ([0.0, 1.0, 1.0], [0.0, 1.0, 0.0], ()),
        ([0.0, 1.0], [0.0], ()),
        ([], [], ()),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], (2,)),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], (6,)),
    ],
)
def test_native_refine_maximum_validates_like_python(
    x: list[float], y: list[float], extra: tuple[int, ...], native: Any
) -> None:
    with pytest.raises(ValueError):
        native.refine_maximum(x, y, *extra)
