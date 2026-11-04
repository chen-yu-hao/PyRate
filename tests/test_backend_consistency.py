from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import pytest

from pyrate.native import backend_info


_SNAPSHOT_SCRIPT = r"""
import json

from pyrate.constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL
from pyrate.native import (
    backend_info,
    beyer_swinehart,
    harmonic_log_partition,
    refine_maximum,
    thermal_wkb_kappa,
    wigner_kappa,
    wkb_exponent,
)

coordinates = tuple(
    value * ANGSTROM_M for value in (-1.0, -0.4, 0.0, 0.6, 1.1)
)
potentials = tuple(
    value * 1000.0 / AVOGADRO_MOL for value in (0.0, 13.0, 31.0, 11.0, -2.0)
)
energy = 6.0 * 1000.0 / AVOGADRO_MOL
refined = refine_maximum(
    (-1.0, -0.5, 0.0, 0.5, 1.0),
    (1.2, 2.4, 3.1, 2.8, 1.0),
)

print(json.dumps({
    "backend": backend_info(),
    "harmonic": harmonic_log_partition((350.0, 800.0, 1500.0), 425.0),
    "wigner": wigner_kappa(1234.5, 425.0),
    "refined": [
        refined.coordinate,
        refined.value,
        refined.sample_index,
        refined.fit_points,
    ],
    "wkb": wkb_exponent(coordinates, potentials, energy, 1.2 * AMU_KG),
    "thermal_wkb": thermal_wkb_kappa(
        coordinates, potentials, 425.0, 1.2 * AMU_KG, 240
    ),
    "density": beyer_swinehart((100.0, 175.0, 300.0), 25.0, 80).tolist(),
}))
"""


def _snapshot(*, disable_native: bool) -> dict[str, Any]:
    environment = os.environ.copy()
    if disable_native:
        environment["PYRATE_DISABLE_NATIVE"] = "1"
    else:
        environment.pop("PYRATE_DISABLE_NATIVE", None)
    process = subprocess.run(
        [sys.executable, "-c", _SNAPSHOT_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(process.stdout)


def test_cpp_and_python_fallback_are_consistent_for_every_kernel() -> None:
    if not backend_info()["native"]:
        pytest.skip("compiled extensions are not installed")

    native = _snapshot(disable_native=False)
    fallback = _snapshot(disable_native=True)

    assert native["backend"]["backend"] == "cpp"
    assert fallback["backend"]["backend"] == "python"

    for key in ("harmonic", "wigner", "wkb", "thermal_wkb"):
        assert native[key] == pytest.approx(fallback[key], rel=5.0e-13, abs=1.0e-13)

    assert native["refined"][:2] == pytest.approx(
        fallback["refined"][:2], rel=5.0e-10, abs=1.0e-10
    )
    assert native["refined"][2:] == fallback["refined"][2:]
    assert native["density"] == fallback["density"]
