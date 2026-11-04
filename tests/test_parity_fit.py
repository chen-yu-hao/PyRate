"""C++ vs pure-Python parity for CVT maximum refinement (owned by the CVT area)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pyrate.native.fit import py_refine_maximum


@pytest.mark.parametrize("seed", range(25))
def test_refine_maximum_parity_on_smooth_profiles(seed: int, native: Any) -> None:
    cpp = native
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1.0, 1.0, 9))
    peak = rng.uniform(-0.5, 0.5)
    y = 50.0 - 12.0 * (x - peak) ** 2 + 0.3 * (x - peak) ** 3
    coordinate, value, index, points = cpp.refine_maximum(x.tolist(), y.tolist())
    reference = py_refine_maximum(x.tolist(), y.tolist())
    assert (index, points) == (reference.sample_index, reference.fit_points)
    assert math.isclose(coordinate, reference.coordinate, rel_tol=1.0e-9, abs_tol=1.0e-9)
    assert math.isclose(value, reference.value, rel_tol=1.0e-12)
