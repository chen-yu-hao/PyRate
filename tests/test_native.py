from __future__ import annotations

import math

import numpy as np

from pyrate.native import backend_info, beyer_swinehart, harmonic_log_partition, refine_maximum


def test_backend_reports_version() -> None:
    info = backend_info()
    assert info["backend"] in {"cpp", "python"}
    assert isinstance(info["version"], str)


def test_refine_maximum_uses_local_polynomial() -> None:
    x = np.linspace(-1.0, 1.0, 5)
    y = 3.0 - 2.0 * (x - 0.2) ** 2
    result = refine_maximum(x, y)
    assert math.isclose(result.coordinate, 0.2, abs_tol=2.0e-3)
    assert math.isclose(result.value, 3.0, abs_tol=1.0e-5)


def test_beyer_swinehart_known_small_case() -> None:
    density = beyer_swinehart((100.0, 200.0), 100.0, 6)
    assert density.tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_harmonic_kernel_ignores_imaginary_frequency() -> None:
    assert math.isclose(
        harmonic_log_partition((500.0, -1000.0), 300.0),
        harmonic_log_partition((500.0,), 300.0),
        rel_tol=1.0e-15,
    )
