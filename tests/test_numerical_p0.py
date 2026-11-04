"""P0 numerical invariants shared by the Python and C++ backends."""

from __future__ import annotations

import math

import pytest

from pyrate.constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL, BOLTZMANN_J_K
from pyrate.elements import isotope_mass_amu
from pyrate.native import refine_maximum, thermal_wkb_kappa, wkb_exponent
from pyrate.models import Atom, Reaction, Species


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
    from pyrate.models import TunnelingProfile
    from pyrate.tunneling import profile_transmission_probability
    profile = TunnelingProfile((-1.0, -0.5, 0.0, 0.5, 1.0),
                               (0.0, 12.0, 25.0, 12.0, 0.0), 1.0)
    assert 0.0 < profile_transmission_probability(profile, 20.0) < 0.5
    assert math.isclose(profile_transmission_probability(profile, 25.0), 0.5)
    assert 0.5 < profile_transmission_probability(profile, 30.0) < 1.0


def test_asymmetric_product_threshold() -> None:
    x = tuple(n * ANGSTROM_M for n in (-2.0, -1.0, 0.0, 1.0, 2.0))
    energy = tuple(v * 1000.0 / AVOGADRO_MOL for v in (0., 10., 30., 20., 15.))
    kappa = thermal_wkb_kappa(x, energy, 350.0, AMU_KG, 600)
    assert math.isfinite(kappa) and kappa > 0.0


def test_wkb_action_decreases_with_energy() -> None:
    x = tuple(n * ANGSTROM_M for n in (-1., 0., 1.))
    v = tuple(n * 1000. / AVOGADRO_MOL for n in (0., 20., 0.))
    low = wkb_exponent(x, v, 0.0, AMU_KG)
    high = wkb_exponent(x, v, 10_000.0 / AVOGADRO_MOL, AMU_KG)
    assert low > high > 0.0
