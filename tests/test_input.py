from __future__ import annotations

import pytest

from pyrate.input import load_toml
from pyrate.legacy import parse_legacy_text


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


@pytest.mark.parametrize(
    ("fu40_line", "keyword"),
    [
        ("geomunit msbohr", "GEOMUNIT"),
        ("hessunit msang", "HESSUNIT"),
        ("gradunit si", "GRADUNIT"),
        ("frequnit hz", "FREQUNIT"),
    ],
)
def test_legacy_fu40_refuses_units_it_cannot_honour(fu40_line: str, keyword: str) -> None:
    from pyrate.input import legacy_to_calculation
    from pyrate.legacy import parse_legacy_text

    dat_text = "*General\n  ATOMS\n    1 H\n    2 H\n  END\n"
    fu40_text = f"*gen40\n  {fu40_line}\n"
    with pytest.raises(ValueError, match=keyword):
        legacy_to_calculation(parse_legacy_text(dat_text), fu40_text=fu40_text)
