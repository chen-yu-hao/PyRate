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


def test_legacy_lctopt_and_toml_lct_options_parsing() -> None:
    from pyrate.input import parse_toml_data

    doc = parse_legacy_text(
        """
*General
 TITLE
 Cl+HBr LCG3 with LCTOPT and PRVIB
 END
 ATOMS
  1 Cl
  2 H
  3 Br
 END
*Path
 SCALEMASS 24.3259
 SRANGE
  slp  3.0
  slm -3.0
 END
*Tunnel
 ZCT
 SCT
 EXCITED 3
 LCTOPT
  lcgmethod 3
  ngtheta 140
  ngamp 40
  vadavg
 END
 LCTGRID
  state 0
   3
 END
 QUAD
  nqe 20
  nsegboltz 2
 END
*Rate
 TST
 CVT
 PRVIB
  sx  2.0
  sy -2.0
 END
 TEMP
  200.0 300.0
 END
"""
    )
    assert doc.requested_methods() == ("TST", "CVT", "ZCT", "SCT", "LCG3")
    assert doc.compatibility_warnings() == ()

    parsed = parse_toml_data(
        {
            "calculation": {
                "name": "LCT TOML check",
                "temperatures_k": [200.0, 300.0],
                "method": "cvt",
                "tunneling": "lcg3",
            },
            "reactants": [
                {"name": "A", "mass_amu": 20.0, "rotor_type": "atomic", "electronic_energy_kj_mol": 0.0}
            ],
            "transition_state": {
                "name": "TS",
                "mass_amu": 20.0,
                "rotor_type": "atomic",
                "electronic_energy_kj_mol": 30.0,
                "frequencies_cm1": [900.0, -1000.0],
            },
            "tunneling": {
                "excited_states": 2,
                "energy_quadrature_points": 24,
                "theta_quadrature_points": 60,
                "amplitude_quadrature_points": 30,
                "vad_average": True,
            },
        }
    )
    assert parsed.options.tunneling == "lcg3"
    assert parsed.options.lct_options.version == 3
    assert parsed.options.lct_options.excited_states == 2
    assert parsed.options.lct_options.energy_quadrature_points == 24
    assert parsed.options.lct_options.theta_quadrature_points == 60
    assert parsed.options.lct_options.amplitude_quadrature_points == 30
    assert parsed.options.lct_options.vad_average is True
