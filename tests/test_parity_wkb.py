"""C++ vs pure-Python parity for WKB tunneling kernels (owned by the tunneling area).

Contract (docs/contracts.md, A1): ``P(E)`` ≤1e-12 and ``κ`` ≤1e-10 relative.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pyrate.constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL
from pyrate.native.wkb import (
    py_semiclassical_transmission,
    py_thermal_wkb_kappa,
    py_wkb_exponent,
)


def _profile(product_shift_kj_mol: float = -5.0) -> tuple[list[float], list[float]]:
    """Asymmetric barrier; exoergic by default, endoergic for a positive shift."""

    x = np.linspace(-1.5, 1.5, 61)
    energies_kj = 30.0 * np.exp(-(x**2) / 0.2) + product_shift_kj_mol * (
        1.0 + np.tanh(2.0 * x)
    ) / 2.0
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


@pytest.mark.parametrize("shift", [-5.0, 12.0])
@pytest.mark.parametrize(
    "fraction", [-0.5, -1.0e-9, 0.0, 0.02, 0.3, 0.7, 0.999, 1.0, 1.001, 1.3, 1.98, 2.0, 3.0]
)
def test_semiclassical_transmission_parity(fraction: float, shift: float, native: Any) -> None:
    # Energies are placed relative to [E0, 2V^AG - E0]: fraction 0 is E0,
    # 1 is the barrier top V^AG and 2 is 2V^AG - E0.
    cpp = native
    coordinates, potentials = _profile(shift)
    threshold = max(potentials[0], potentials[-1])
    energy = threshold + fraction * (max(potentials) - threshold)
    expected = py_semiclassical_transmission(coordinates, potentials, energy, AMU_KG)
    assert math.isclose(
        cpp.semiclassical_transmission(coordinates, potentials, energy, AMU_KG),
        expected,
        rel_tol=1.0e-12,
    )
    if fraction < 0.0:
        assert expected == 0.0


@pytest.mark.parametrize("shift", [-5.0, 12.0])
@pytest.mark.parametrize("temperature", [40.0, 150.0, 300.0, 1000.0])
def test_thermal_wkb_kappa_parity(temperature: float, shift: float, native: Any) -> None:
    cpp = native
    coordinates, potentials = _profile(shift)
    assert math.isclose(
        cpp.thermal_wkb_kappa(coordinates, potentials, temperature, AMU_KG, 300),
        py_thermal_wkb_kappa(coordinates, potentials, temperature, AMU_KG, 300),
        rel_tol=1.0e-10,
    )


def test_thermal_wkb_kappa_overflow_parity(native: Any) -> None:
    cpp = native
    coordinates, potentials = _profile()
    assert cpp.thermal_wkb_kappa(coordinates, potentials, 1.0, AMU_KG, 300) == math.inf
    assert py_thermal_wkb_kappa(coordinates, potentials, 1.0, AMU_KG, 300) == math.inf
