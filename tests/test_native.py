from __future__ import annotations

import math

import numpy as np
import pytest

from pyrate.native import (
    RefinedMaximum,
    backend_info,
    beyer_swinehart,
    harmonic_log_partition,
    refine_maximum,
)


def test_backend_reports_version() -> None:
    info = backend_info()
    assert info["backend"] in {"cpp", "python"}
    assert isinstance(info["version"], str)


def test_refine_maximum_parabola_is_exact_on_quadratic_data() -> None:
    x = np.linspace(-1.0, 1.0, 5)
    y = 3.0 - 2.0 * (x - 0.2) ** 2
    result = refine_maximum(x, y)
    assert (result.sample_index, result.fit_points) == (2, 3)
    assert math.isclose(result.coordinate, 0.2, abs_tol=1.0e-12)
    assert math.isclose(result.value, 3.0, rel_tol=1.0e-12)


@pytest.mark.parametrize("max_points", [3, 4, 5])
def test_refine_maximum_is_exact_on_non_uniform_quadratic_data(max_points: int) -> None:
    x = np.array([-1.3, -0.9, -0.75, -0.1, 0.35, 0.5, 1.6])
    y = -40.0 - 7.5 * (x + 0.31) ** 2
    result = refine_maximum(x, y, max_points=max_points)
    assert (result.sample_index, result.fit_points) == (3, max_points)
    assert math.isclose(result.coordinate, -0.31, abs_tol=1.0e-12)
    assert math.isclose(result.value, -40.0, rel_tol=1.0e-12)


def test_refine_maximum_returns_sample_at_endpoints_and_short_grids() -> None:
    assert refine_maximum([0.0, 1.0, 2.0], [3.0, 2.0, 1.0]) == RefinedMaximum(0.0, 3.0, 0, 1)
    assert refine_maximum([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == RefinedMaximum(2.0, 3.0, 2, 1)
    assert refine_maximum([0.0, 1.0], [1.0, 0.5], max_points=5) == RefinedMaximum(
        0.0, 1.0, 0, 1
    )
    assert refine_maximum([4.0], [-2.0]) == RefinedMaximum(4.0, -2.0, 0, 1)


def test_refine_maximum_tie_uses_first_maximum_and_midpoint_vertex() -> None:
    # Uniform plateau: the parabola peaks halfway to the tied neighbour and
    # overshoots by d_l / 8, well inside B = d_l.
    result = refine_maximum([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 1.0, 0.0])
    assert (result.sample_index, result.fit_points) == (1, 3)
    assert math.isclose(result.coordinate, 1.5, rel_tol=1.0e-15)
    assert math.isclose(result.value, 1.125, rel_tol=1.0e-15)


def test_refine_maximum_rejects_parabola_beyond_drop_bound() -> None:
    # Steep rise into a wide plateau: the parabola would peak at x = 0.5 with
    # an overshoot of 24.75 against B = 1, so the sample is returned instead.
    result = refine_maximum([-0.01, 0.0, 1.0], [-1.0, 0.0, 0.0])
    assert result == RefinedMaximum(0.0, 0.0, 1, 1)


@pytest.mark.parametrize("max_points", [3, 4, 5])
def test_refine_maximum_audit_counter_example(max_points: int) -> None:
    # roadmap N3: the former quartic accepted s = 0.460 with an overshoot of
    # 6.37 (value 9.78) although B = 3.404 - 0.081 = 3.323.
    x = [-0.294, -0.234, 0.117, 0.849, 0.919]
    y = [2.99, 0.081, 3.404, 2.422, 0.784]
    result = refine_maximum(x, y, max_points=max_points)
    assert (result.sample_index, result.fit_points) == (2, 3)
    assert x[1] <= result.coordinate <= x[3]
    assert y[2] <= result.value <= y[2] + (y[2] - min(y[1], y[3]))
    # Independent check: vertex of the interpolating parabola through i-1, i, i+1.
    a, b, c = np.polyfit(x[1:4], y[1:4], 2)
    assert math.isclose(result.coordinate, -b / (2.0 * a), rel_tol=1.0e-12)
    assert math.isclose(result.value, c - b * b / (4.0 * a), rel_tol=1.0e-12)


@pytest.mark.parametrize(
    ("x", "y", "kwargs", "message"),
    [
        ([0.0, 1.0, 2.0], [0.0, math.nan, 1.0], {}, "finite"),
        ([0.0, math.inf, 2.0], [0.0, 1.0, 0.0], {}, "finite"),
        ([0.0, 1.0, 1.0], [0.0, 1.0, 0.0], {}, "strictly increasing"),
        ([0.0, 1.0], [0.0], {}, "equal length"),
        ([], [], {}, "empty"),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], {"max_points": 2}, "max_points"),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], {"max_points": 6}, "max_points"),
    ],
)
def test_refine_maximum_rejects_invalid_input(
    x: list[float], y: list[float], kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        refine_maximum(x, y, **kwargs)


def test_beyer_swinehart_known_small_case() -> None:
    density = beyer_swinehart((100.0, 200.0), 100.0, 6)
    assert density.tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_harmonic_kernel_ignores_imaginary_frequency() -> None:
    assert math.isclose(
        harmonic_log_partition((500.0, -1000.0), 300.0),
        harmonic_log_partition((500.0,), 300.0),
        rel_tol=1.0e-15,
    )
