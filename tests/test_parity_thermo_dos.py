"""C++ vs pure-Python parity for partition-function and state-counting kernels."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pyrate.native.dos import py_beyer_swinehart
from pyrate.native.thermo import py_harmonic_log_partition, py_wigner_kappa


@pytest.mark.parametrize("temperature", [10.0, 200.0, 298.15, 1000.0, 5000.0])
def test_harmonic_log_partition_parity(temperature: float, native: Any) -> None:
    cpp = native
    rng = np.random.default_rng(1)
    frequencies = [*rng.uniform(20.0, 4000.0, 30).tolist(), -1500.0, 0.0]
    assert math.isclose(
        cpp.harmonic_log_partition(frequencies, temperature),
        py_harmonic_log_partition(frequencies, temperature),
        rel_tol=1.0e-13,
        abs_tol=1.0e-300,
    )


@pytest.mark.parametrize("frequency", [-2500.0, -800.0, 300.0])
@pytest.mark.parametrize("temperature", [50.0, 300.0, 2000.0])
def test_wigner_parity(frequency: float, temperature: float, native: Any) -> None:
    cpp = native
    assert math.isclose(
        cpp.wigner_kappa(frequency, temperature),
        py_wigner_kappa(frequency, temperature),
        rel_tol=1.0e-14,
    )


def test_beyer_swinehart_parity_is_exact(native: Any) -> None:
    cpp = native
    frequencies = (300.0, 455.5, 700.0, 1010.0, 1499.4, -900.0)
    native = np.asarray(cpp.beyer_swinehart(frequencies, 10.0, 2000))
    python = py_beyer_swinehart(frequencies, 10.0, 2000)
    assert np.array_equal(native, python)
