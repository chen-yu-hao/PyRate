from __future__ import annotations

import math

import pytest

from pyrate.elements import atomic_mass_amu
from pyrate.geometry import principal_moments_amu_a2, rotor_geometry
from pyrate.models import Atom, PathPoint, Species


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


def test_atom_default_mass_is_isotopic() -> None:
    atom = Atom("Cl", 0.0, 0.0, 0.0)
    assert atom.mass_model == "isotopic"
    assert atom.mass == atomic_mass_amu("Cl", "isotopic") == 34.968852694


def test_atom_average_mass_model() -> None:
    atom = Atom("Cl", 0.0, 0.0, 0.0, mass_model="Average")
    assert atom.mass_model == "average"
    assert atom.mass == 35.45


def test_atom_explicit_mass_overrides_mass_model() -> None:
    for model in ("isotopic", "average"):
        atom = Atom("Cl", 0.0, 0.0, 0.0, mass_amu=36.965902573, mass_model=model)
        assert atom.mass == 36.965902573
        assert atom.mass_model == model


def test_atom_positional_label_still_works() -> None:
    atom = Atom("H", 0.0, 0.0, 0.0, None, "H1")
    assert atom.label == "H1"
    assert atom.mass == atomic_mass_amu("H")


def test_atom_rejects_unknown_mass_model() -> None:
    with pytest.raises(ValueError, match="mass_model"):
        Atom("H", 0.0, 0.0, 0.0, mass_model="monoisotopic")
    with pytest.raises(ValueError, match="mass_model"):
        Atom.from_mapping({"symbol": "H", "xyz": [0.0, 0.0, 0.0]}, mass_model="exact")


def test_from_mapping_threads_mass_model() -> None:
    data = {
        "name": "HCl",
        "coordinate": 0.0,
        "atoms": [
            {"symbol": "H", "xyz": [0.0, 0.0, -1.2]},
            {"symbol": "Cl", "xyz": [0.0, 0.0, 0.07]},
        ],
    }
    default = Species.from_mapping(data)
    average = Species.from_mapping(data, mass_model="average")
    point = PathPoint.from_mapping(data, 0, mass_model="average")
    assert [atom.mass for atom in default.atoms] == [1.0078250319, 34.968852694]
    assert [atom.mass for atom in average.atoms] == [1.00784, 35.45]
    assert point.dividing_surface.atoms == average.atoms
    # The heavier average masses raise the HCl moment of inertia slightly.
    largest_default = principal_moments_amu_a2(default.atoms)[2]
    largest_average = principal_moments_amu_a2(average.atoms)[2]
    assert 1.0 < largest_average / largest_default < 1.001
    with pytest.raises(ValueError, match="mass_model"):
        Species.from_mapping(
            {"name": "X", "mass_amu": 4.0, "rotor_type": "atomic"}, mass_model="bogus"
        )
