"""C++ vs pure-Python parity for WKB tunneling kernels (owned by the tunneling area)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pyrate.constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL
from pyrate.native.wkb import py_thermal_wkb_kappa, py_wkb_exponent


def _profile() -> tuple[list[float], list[float]]:
    x = np.linspace(-1.5, 1.5, 61)
    energies_kj = 30.0 * np.exp(-(x**2) / 0.2) - 5.0 * (1.0 + np.tanh(2.0 * x)) / 2.0
    return (x * ANGSTROM_M).tolist(), (energies_kj * 1000.0 / AVOGADRO_MOL).tolist()


@pytest.mark.parametrize("fraction", [0.05, 0.3, 0.7, 0.99])
def test_wkb_exponent_parity(fraction: float, native: Any) -> None:
    cpp = native
    coordinates, potentials = _profile()
    energy = fraction * max(potentials)
    assert math.isclose(
        cpp.wkb_exponent(coordinates, potentials, energy, AMU_KG),
        py_wkb_exponent(coordinates, potentials, energy, AMU_KG),
        rel_tol=1.0e-12,
    )


@pytest.mark.parametrize("temperature", [150.0, 300.0, 1000.0])
def test_thermal_wkb_kappa_parity(temperature: float, native: Any) -> None:
    cpp = native
    coordinates, potentials = _profile()
    assert math.isclose(
        cpp.thermal_wkb_kappa(coordinates, potentials, temperature, AMU_KG, 300),
        py_thermal_wkb_kappa(coordinates, potentials, temperature, AMU_KG, 300),
        rel_tol=1.0e-11,
    )
