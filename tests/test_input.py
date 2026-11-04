from __future__ import annotations

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
