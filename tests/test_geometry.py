from __future__ import annotations

import math

from pyrate.geometry import principal_moments_amu_a2, rotor_geometry
from pyrate.models import Atom, Species


def test_h2_is_linear() -> None:
    species = Species(
        name="H2",
        atoms=(
            Atom("H", -0.37, 0.0, 0.0),
            Atom("H", 0.37, 0.0, 0.0),
        ),
        frequencies_cm1=(4401.0,),
        symmetry_number=2,
    )
    moments = principal_moments_amu_a2(species.atoms)
    assert moments[0] == 0.0
    assert math.isclose(moments[1], moments[2], rel_tol=1.0e-12)
    assert rotor_geometry(species).rotor_type == "linear"


def test_supplied_moments_without_geometry() -> None:
    species = Species(
        name="model",
        mass_amu_override=28.0,
        moments_of_inertia_amu_a2=(10.0, 15.0, 20.0),
        rotor_type="auto",
    )
    assert rotor_geometry(species).rotor_type == "nonlinear"


def test_atomic_model_without_geometry() -> None:
    species = Species(name="H", mass_amu_override=1.0, rotor_type="atomic")
    geometry = rotor_geometry(species)
    assert geometry.rotor_type == "atomic"
    assert geometry.moments_amu_a2 == (0.0, 0.0, 0.0)
