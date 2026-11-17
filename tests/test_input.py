from __future__ import annotations

import json
import math
import shlex
import sys
from pathlib import Path

import pytest

from pyrate.direct_dynamics import ExternalCommandPes
from pyrate.elements import atomic_mass_amu
from pyrate.input import legacy_to_calculation, load_toml, parse_toml_data
from pyrate.legacy import LegacySemanticError, parse_legacy_text, validate_legacy_semantics


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
    ("snippet", "expected_section", "expected_keyword", "expected_line"),
    [
        ("*TUNNEL\n  LCG3\n", "TUNNEL", "LCG3", 2),
        ("*TUNNEL\n  LCG4\n", "TUNNEL", "LCG4", 2),
        ("*RATE\n  ISPE\n", "RATE", "ISPE", 2),
        ("*RATE\n  ICVT\n", "RATE", "ICVT", 2),
        ("*RATE\n  VRCTST\n", "RATE", "VRCTST", 2),
        ("*GENERAL\n  SOLV\n", "GENERAL", "SOLV", 2),
        ("*PATH\n  RODS\n", "PATH", "RODS", 2),
        ("*GENERAL\n  INPUNIT FURLONG\n", "GENERAL", "INPUNIT", 2),
        ("*PATH\n  SCALEMASS -1.0\n", "PATH", "SCALEMASS", 2),
        ("*RATE\n  BOGUSKEY\n", "RATE", "BOGUSKEY", 2),
        ("*EXOTIC\n  FOO\n", "EXOTIC", None, 1),
    ],
)
def test_validate_legacy_semantics_structured_errors(
    snippet: str,
    expected_section: str,
    expected_keyword: str | None,
    expected_line: int,
) -> None:
    doc = parse_legacy_text(snippet)
    with pytest.raises(LegacySemanticError) as exc_info:
        validate_legacy_semantics(doc)
    err = exc_info.value
    assert err.section == expected_section
    assert err.keyword == expected_keyword
    assert err.line_number == expected_line


def test_legacy_mass_overrides_isotopes_and_freqscale() -> None:
    # Inline MASS and block MASS with explicit isotope symbols (2H, 37Cl) and FREQSCALE
    dat_text = """
*GENERAL
  ATOMS
    1 37Cl
    2 2H
    3 Br
  END
  INPUNIT ANG
  MASS 1 36.9659
  MASS
    2 2.01410178
  END
*ENERGETICS
  POTENTIAL BUILTIN:cl_hbr_leps
*REACT1
  GEOM
    1 0.0 0.0 0.0
  END
  ELEC
    2 0.0
  END
  SPECIES ATOMIC
*REACT2
  GEOM
    2 0.0 0.0 0.0
    3 0.0 0.0 1.414
  END
  FREQSCALE 0.96
  SPECIES LINRP
*START
  GEOM
    1 0.0 0.0 -1.93
    2 0.0 0.0 -0.05
    3 0.0 0.0  1.41
  END
  ELEC
    2 0.0
  END
  FREQSCALE 0.98
  SPECIES LINTS
*RATE
  TST
  NOCVT
  TEMP
    300.0
  END
"""
    doc = parse_legacy_text(dat_text)
    assert doc.atom_table == ((1, "37Cl"), (2, "2H"), (3, "Br"))
    assert doc.mass_overrides == {1: 36.9659, 2: 2.01410178}
    loaded = legacy_to_calculation(doc)
    ts = loaded.reaction.transition_state
    r2 = loaded.reaction.reactants[1]
    assert ts.atoms[0].symbol == "Cl"
    assert ts.atoms[1].symbol == "H"
    assert math.isclose(ts.atoms[0].mass_amu or 0.0, 36.9659, rel_tol=1.0e-9)
    assert math.isclose(ts.atoms[1].mass_amu or 0.0, 2.01410178, rel_tol=1.0e-9)
    assert math.isclose(
        ts.atoms[2].mass_amu or 0.0,
        atomic_mass_amu("Br", mass_convention="polyrate"),
        rel_tol=1.0e-9,
    )
    assert math.isclose(r2.frequency_scale, 0.96, rel_tol=1.0e-12)
    assert math.isclose(ts.frequency_scale, 0.98, rel_tol=1.0e-12)

    # Without explicit MASS lines, 37Cl and 2H still resolve their isotope masses
    dat_iso_only = dat_text.replace("  MASS 1 36.9659\n", "").replace(
        "  MASS\n    2 2.01410178\n  END\n", ""
    )
    loaded_iso = legacy_to_calculation(parse_legacy_text(dat_iso_only))
    ts_iso = loaded_iso.reaction.transition_state
    assert math.isclose(
        ts_iso.atoms[0].mass_amu or 0.0,
        atomic_mass_amu("37Cl", mass_convention="polyrate"),
        rel_tol=1.0e-9,
    )
    assert math.isclose(
        ts_iso.atoms[1].mass_amu or 0.0,
        atomic_mass_amu("2H", mass_convention="polyrate"),
        rel_tol=1.0e-9,
    )


def test_toml_potential_mep_path_integration_and_command_restart(tmp_path: Path) -> None:
    # 1) Mutual exclusion between [mep] and [path.integration]
    base_data = {
        "calculation": {"temperatures_k": [300.0], "method": "cvt", "tunneling": "zct"},
        "potential": {"type": "h3_leps"},
        "reactants": [
            {"name": "H", "atoms": [{"symbol": "H", "xyz": [0.0, 0.0, 0.0]}]},
            {
                "name": "H2",
                "rotor_type": "linear",
                "symmetry_number": 2,
                "atoms": [
                    {"symbol": "H", "xyz": [0.0, 0.0, 0.0]},
                    {"symbol": "H", "xyz": [0.0, 0.0, 0.7414]},
                ],
            },
        ],
        "transition_state": {
            "name": "TS",
            "rotor_type": "linear",
            "atoms": [
                {"symbol": "H", "xyz": [0.0, 0.0, -0.929]},
                {"symbol": "H", "xyz": [0.0, 0.0, 0.0]},
                {"symbol": "H", "xyz": [0.0, 0.0, 0.929]},
            ],
        },
    }
    both = dict(base_data)
    both["mep"] = {"method": "pm", "s_min_bohr": -0.3, "s_max_bohr": 0.3, "step_size_bohr": 0.1}
    both["path"] = {
        "integration": {
            "method": "pm",
            "s_min_bohr": -0.3,
            "s_max_bohr": 0.3,
            "step_size_bohr": 0.1,
        }
    }
    with pytest.raises(ValueError, match="either \\[mep\\] or \\[path\\.integration\\]"):
        parse_toml_data(both)

    # 2) [path.integration] equivalence with [mep]
    via_path = dict(base_data)
    via_path["path"] = both["path"]
    loaded_pi = parse_toml_data(via_path)
    assert loaded_pi.mep_result is not None
    assert loaded_pi.reaction.path is not None
    assert len(loaded_pi.reaction.path.points) >= 5

    # 3) TOML [potential] type = "command" with base_dir-relative restart_file
    driver = tmp_path / "h3_driver.py"
    driver.write_text(
        """
import json
import sys
from pathlib import Path
import numpy as np
from pyrate.pes import h3_leps_surface

req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pes = h3_leps_surface()
ev = pes.evaluate(
    tuple(req["symbols"]),
    np.asarray(req["coords_angstrom"], dtype=float),
    need_gradient=bool(req["need_gradient"] or req["need_hessian"]),
    need_hessian=bool(req["need_hessian"]),
)
payload = {"energy_kj_mol": ev.energy_kj_mol}
if req["need_gradient"] and ev.gradient_kj_mol_angstrom is not None:
    payload["gradient_kj_mol_angstrom"] = ev.gradient_kj_mol_angstrom.tolist()
if req["need_hessian"] and ev.hessian_kj_mol_angstrom2 is not None:
    payload["hessian_kj_mol_angstrom2"] = ev.hessian_kj_mol_angstrom2.tolist()
Path(sys.argv[2]).write_text(json.dumps(payload), encoding="utf-8")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    quoted_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(driver))}"
    toml_file = tmp_path / "cmd_job.toml"
    toml_file.write_text(
        f"""
[calculation]
name = "cmd-h3"
temperatures_k = [300.0]
method = "tst"
tunneling = "none"

[potential]
type = "command"
command = {json.dumps(quoted_cmd)}
restart_file = "cache/restart.json"
supermolecule = true

[[reactants]]
name = "H"
atoms = [{{ symbol = "H", xyz = [0.0, 0.0, 0.0] }}]

[[reactants]]
name = "H2"
rotor_type = "linear"
symmetry_number = 2
atoms = [
  {{ symbol = "H", xyz = [0.0, 0.0, 0.0] }},
  {{ symbol = "H", xyz = [0.0, 0.0, 0.7414] }},
]

[transition_state]
name = "H3-TS"
rotor_type = "linear"
atoms = [
  {{ symbol = "H", xyz = [0.0, 0.0, -0.929] }},
  {{ symbol = "H", xyz = [0.0, 0.0, 0.0] }},
  {{ symbol = "H", xyz = [0.0, 0.0, 0.929] }},
]
""",
        encoding="utf-8",
    )
    loaded_cmd = load_toml(toml_file)
    assert loaded_cmd.reaction.transition_state.imaginary_frequencies_cm1
    restart_expected = tmp_path / "cache" / "restart.json"
    assert restart_expected.is_file()
    # Verify ExternalCommandPes parsing matches base_dir directly too
    pes_direct = ExternalCommandPes(command=quoted_cmd, restart_file=restart_expected)
    assert pes_direct.cache.entries

