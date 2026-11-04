from __future__ import annotations

import sys

import pytest

from pyrate.input import LoadedCalculation, load_toml, parse_toml_data
from pyrate.legacy import parse_legacy_text
from pyrate.models import Atom

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib

# Every species kind carries atoms without mass_amu; the reactant's Cl carries
# an explicit 37Cl mass that must survive any mass_model.
MASS_MODEL_TOML = """
[calculation]
temperatures_k = [300.0]
method = "cvt"
{mass_model_line}

[[reactants]]
name = "HCl"
frequencies_cm1 = [2990.0]
atoms = [
  {{symbol = "H", xyz = [0.0, 0.0, -1.2]}},
  {{symbol = "Cl", xyz = [0.0, 0.0, 0.07], mass_amu = 36.965902573}},
]

[[reactants]]
name = "H"
rotor_type = "atomic"
atoms = [{{symbol = "H", xyz = [0.0, 0.0, 0.0]}}]

[transition_state]
name = "TS"
frequencies_cm1 = [500.0, 500.0, 1400.0]
imaginary_frequency_cm1 = 1200.0
atoms = [
  {{symbol = "H", xyz = [0.0, 0.0, -2.2]}},
  {{symbol = "H", xyz = [0.0, 0.0, -1.2]}},
  {{symbol = "Cl", xyz = [0.0, 0.0, 0.07]}},
]

[[products]]
name = "H2"
atoms = [
  {{symbol = "H", xyz = [-0.37, 0.0, 0.0]}},
  {{symbol = "H", xyz = [0.37, 0.0, 0.0]}},
]

[[products]]
name = "Cl"
rotor_type = "atomic"
atoms = [{{symbol = "Cl", xyz = [0.0, 0.0, 0.0]}}]

[path]
{path_points}
"""
PATH_POINT = """
[[path.points]]
coordinate = {coordinate}
frequencies_cm1 = [500.0, 500.0, 1400.0]
atoms = [
  {{symbol = "H", xyz = [0.0, 0.0, -2.2]}},
  {{symbol = "D", xyz = [0.0, 0.0, -1.2]}},
  {{symbol = "Cl", xyz = [0.0, 0.0, 0.07]}},
]
"""
EXPECTED = {
    "isotopic": {"H": 1.0078250319, "D": 2.01410177784, "Cl": 34.968852694},
    "average": {"H": 1.00784, "D": 2.01410177812, "Cl": 35.45},
}


def _load_mass_model_case(mass_model_line: str) -> LoadedCalculation:
    points = "".join(PATH_POINT.format(coordinate=value) for value in (-0.5, 0.0, 0.5))
    text = MASS_MODEL_TOML.format(mass_model_line=mass_model_line, path_points=points)
    return parse_toml_data(tomllib.loads(text))


def _all_atoms(loaded: LoadedCalculation) -> list[tuple[str, Atom]]:
    reaction = loaded.reaction
    species = [*reaction.reactants, reaction.transition_state, *reaction.products]
    assert reaction.path is not None
    species.extend(point.dividing_surface for point in reaction.path.points)
    return [(item.name, atom) for item in species for atom in item.atoms]


@pytest.mark.parametrize(
    ("line", "model"),
    [
        ("", "isotopic"),
        ('mass_model = "isotopic"', "isotopic"),
        ('mass_model = "average"', "average"),
        ('mass_model = "Average"', "average"),
    ],
)
def test_mass_model_reaches_every_species_kind(line: str, model: str) -> None:
    loaded = _load_mass_model_case(line)
    assert loaded.mass_model == model
    assert loaded.summary()["mass_model"] == model
    atoms = _all_atoms(loaded)
    kinds = {name for name, _ in atoms}
    assert {"HCl", "H", "TS", "H2", "Cl", "path-0", "path-1", "path-2"} <= kinds
    explicit = [atom for name, atom in atoms if name == "HCl" and atom.symbol == "Cl"]
    assert [atom.mass for atom in explicit] == [36.965902573]
    for _, atom in atoms:
        assert atom.mass_model == model
        if atom not in explicit:
            assert atom.mass == EXPECTED[model][atom.symbol]
    reaction = loaded.reaction
    expected_hcl = EXPECTED[model]["H"] + 36.965902573
    assert reaction.reactants[0].mass_amu == pytest.approx(expected_hcl, rel=1.0e-15)


@pytest.mark.parametrize("value", ['"monoisotopic"', '""', "1", "true"])
def test_unknown_mass_model_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match=r"\[calculation\] mass_model must be"):
        _load_mass_model_case(f"mass_model = {value}")


def test_legacy_parser_preserves_sections_blocks_and_unknowns() -> None:
    document = parse_legacy_text(
        """
*General
 TITLE
 H2 + H -> H + H2
 END
 ATOMS
  1 H
  2 H
  3 H
 END
 MYSTERY  17 alpha
*Tunnel
 ZCT
 SCT
*Rate
 TST
 CVT
 TEMP
  200.0
  300.0
 END
"""
    )
    assert document.title == "H2 + H -> H + H2"
    assert document.temperatures_k == (200.0, 300.0)
    assert document.atom_table == ((1, "H"), (2, "H"), (3, "H"))
    assert document.section("GENERAL").get("MYSTERY").arguments == ("17", "alpha")
    assert document.requested_methods() == ("TST", "CVT", "ZCT", "SCT")
    assert any("SCT" in warning for warning in document.compatibility_warnings())


def test_example_toml_loads(project_root) -> None:
    loaded = load_toml(project_root / "examples" / "harmonic_cvt.toml")
    assert loaded.options.method == "cvt"
    assert loaded.options.tunneling == "wigner"
    assert loaded.reaction.path is not None
    assert len(loaded.reaction.path.points) == 5
    assert loaded.summary()["mass_model"] == "isotopic"
