"""P0 numerical invariants shared by the Python and C++ backends."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

from pyrate.constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL, HBAR_J_S
from pyrate.elements import isotope_mass_amu
from pyrate.models import Atom, Reaction, Species, TunnelingProfile
from pyrate.native import HAS_NATIVE, refine_maximum, thermal_wkb_kappa, wkb_exponent
from pyrate.tunneling import profile_transmission_probability


def test_cvt_local_parabola_is_bracketed() -> None:
    samples = [-2.0, -1.0, 0.0, 1.0, 2.0]
    values = [0.0, 2.0, 3.0, 2.0, -10.0]
    answer = refine_maximum(samples, values)
    assert -1.0 <= answer.coordinate <= 1.0
    assert answer.value >= 3.0
    assert answer.fit_points == 3


def test_cvt_edge_peak_never_extrapolates() -> None:
    answer = refine_maximum([0.0, 1.0, 2.0], [3.0, 2.0, 1.0])
    assert (answer.coordinate, answer.value, answer.fit_points) == (0.0, 3.0, 1)


def test_cvt_rejects_nonfinite_points() -> None:
    with pytest.raises(ValueError):
        refine_maximum([0.0, 1.0, 2.0], [1.0, math.nan, 0.0])


def test_isotope_mass_is_neutral_nuclidic_not_atomic_weight() -> None:
    assert isotope_mass_amu("C", 12) == 12.0
    assert math.isclose(isotope_mass_amu("H", 2), 2.01410177812)
    atom = Atom("C", 0.0, 0.0, 0.0, isotope=13)
    assert math.isclose(float(atom.mass_amu), 13.00335483507)
    with pytest.raises(ValueError):
        isotope_mass_amu("C", 99)


def test_constants_profile_fails_closed() -> None:
    ts = Species("ts", mass_amu_override=1.0, rotor_type="atomic")
    with pytest.raises(ValueError, match="constants_profile"):
        Reaction(reactants=(ts,), transition_state=ts, constants_profile="legacy")


def test_reflection_below_and_above_barrier() -> None:
    profile = TunnelingProfile(
        (-1.0, -0.5, 0.0, 0.5, 1.0),
        (0.0, 12.0, 25.0, 12.0, 0.0),
        1.0,
    )
    assert 0.0 < profile_transmission_probability(profile, 20.0) < 0.5
    assert math.isclose(profile_transmission_probability(profile, 25.0), 0.5)
    assert 0.5 < profile_transmission_probability(profile, 30.0) < 1.0


def test_asymmetric_product_threshold() -> None:
    x = tuple(n * ANGSTROM_M for n in (-2.0, -1.0, 0.0, 1.0, 2.0))
    energy = tuple(v * 1000.0 / AVOGADRO_MOL for v in (0.0, 10.0, 30.0, 20.0, 15.0))
    kappa = thermal_wkb_kappa(x, energy, 350.0, AMU_KG, 600)
    assert math.isfinite(kappa) and kappa > 0.0


def test_wkb_action_decreases_with_energy() -> None:
    x = tuple(n * ANGSTROM_M for n in (-1.0, 0.0, 1.0))
    v = tuple(n * 1000.0 / AVOGADRO_MOL for n in (0.0, 20.0, 0.0))
    low = wkb_exponent(x, v, 0.0, AMU_KG)
    high = wkb_exponent(x, v, 10_000.0 / AVOGADRO_MOL, AMU_KG)
    assert low > high > 0.0


def test_symmetric_eckart_analytic_probability() -> None:
    """Compare semiclassical transmission with an exact sech² (Eckart) barrier.

    The approximation is not exact: a 12% envelope is applied and documented.
    The benchmark catches exponent normalization and unit-factor regressions.
    """

    mass = AMU_KG
    inverse_width = 1.0e10  # 1 / angstrom
    height = 25_000.0 / AVOGADRO_MOL
    energy = 10_000.0 / AVOGADRO_MOL
    wave = math.pi * math.sqrt(2 * mass * energy) / (HBAR_J_S * inverse_width)
    shape = math.sqrt(8 * mass * height / (HBAR_J_S**2 * inverse_width**2) - 1)
    exact = math.sinh(wave)**2 / (
        math.sinh(wave)**2 + math.cosh(math.pi * shape / 2)**2
    )
    coordinates = tuple(-5.0 + 10.0 * i / 400 for i in range(401))
    potential = tuple(25.0 / math.cosh(s)**2 for s in coordinates)
    profile = TunnelingProfile(coordinates, potential, 1.0)
    actual = profile_transmission_probability(profile, 10.0)
    assert math.isclose(actual, exact, rel_tol=0.12)


def test_triangular_barrier_action_fixed_regression() -> None:
    """Closed-form action for a triangular barrier, including unit conversion."""

    x = (-ANGSTROM_M, 0.0, ANGSTROM_M)
    v0 = 25_000.0 / AVOGADRO_MOL
    action = wkb_exponent(x, (0.0, v0, 0.0), 0.0, AMU_KG)
    exact = (8.0 / 3.0) * ANGSTROM_M * math.sqrt(2.0 * AMU_KG * v0) / HBAR_J_S
    assert math.isclose(action, exact, rel_tol=1.0e-12)
    assert math.isclose(action, 29.691126198605367, rel_tol=2.0e-10)


def test_backend_consistency_subprocess() -> None:
    """Compare native and disabled-native numerical results in separate imports."""

    if not HAS_NATIVE:
        pytest.skip("native extension is not present in this installation")
    code = """
import json
from pyrate.native import refine_maximum, thermal_wkb_kappa, wkb_exponent
from pyrate.constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL
x = [-1., -.5, 0., .5, 1.]
y = [0., 12., 25., 12., 0.]
p = [v*1000/AVOGADRO_MOL for v in y]
coord = [v*ANGSTROM_M for v in x]
maximum = refine_maximum(x, [-2., 1., 3., 2., 0.])
print(json.dumps([maximum.coordinate, maximum.value,
    wkb_exponent(coord, p, 10_000/AVOGADRO_MOL, AMU_KG),
    thermal_wkb_kappa(coord, p, 350., AMU_KG, 400)]))
"""
    def execute(disabled: bool) -> list[float]:
        env = dict(os.environ)
        if disabled:
            env["PYRATE_DISABLE_NATIVE"] = "1"
        else:
            env.pop("PYRATE_DISABLE_NATIVE", None)
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return list(json.loads(result.stdout))

    native = execute(False)
    fallback = execute(True)
    assert native == pytest.approx(fallback, rel=2.0e-9, abs=1.0e-11)
