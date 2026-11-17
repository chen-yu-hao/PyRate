from __future__ import annotations

import json
import math
import shlex
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pyrate", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def test_cli_runs_example(project_root) -> None:
    process = _run_cli(
        "run",
        str(project_root / "examples" / "harmonic_cvt.toml"),
        "--format",
        "json",
    )
    payload = json.loads(process.stdout)
    assert payload["method"] == "cvt"
    assert len(payload["rows"]) == 4
    assert all(row["k_final"] > 0.0 for row in payload["rows"])


def test_cli_runs_cl_hbr_mep_example(project_root) -> None:
    process = _run_cli(
        "run",
        str(project_root / "examples" / "cl_hbr_mep.toml"),
        "--format",
        "json",
    )
    payload = json.loads(process.stdout)
    assert payload["method"] == "cvt"
    assert payload["tunneling"] == "sct"
    assert len(payload["rows"]) == 5
    for row in payload["rows"]:
        assert row["cvt_coordinate"] < -0.1
        assert row["k_cvt"] < row["k_tst"]
        assert row["tunneling_correction"] > 1.0


def test_cli_pes_and_export_fu40_roundtrip(tmp_path: Path) -> None:
    dat_tmpl = """
*GENERAL
  ATOMS
    1 Cl
    2 H
    3 Br
  END
  INPUNIT ANG
*ENERGETICS
  POTENTIAL __POT__
*OPTIMIZATION
  OPTMIN
  OPTTS
*SECOND
  CURV
  CENTRAL
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
    3 0.0 0.0 1.4146
  END
  SPECIES LINRP
*PROD1
  GEOM
    1 0.0 0.0 0.0
    2 0.0 0.0 1.2746
  END
  SPECIES LINRP
*PROD2
  GEOM
    3 0.0 0.0 0.0
  END
  ELEC
    2 0.0
  END
  SPECIES ATOMIC
*START
  GEOM
    1 0.0 0.0 -1.930
    2 0.0 0.0 -0.055
    3 0.0 0.0  1.410
  END
  ELEC
    2 0.0
  END
  SPECIES LINTS
*PATH
  PM
  CURV
  SSTEP 0.06
  SRANGE
    -0.6 0.6
  END
*TUNNEL
  SCT
*RATE
  CVT
  TEMP
    298.15 500.0
  END
"""
    dat_hook = tmp_path / "cl_hbr_hook.dat"
    dat_u40 = tmp_path / "cl_hbr_u40.dat"
    fu40_out = tmp_path / "exported.fu40"
    dat_hook.write_text(dat_tmpl.replace("__POT__", "HOOKS"), encoding="utf-8")
    dat_u40.write_text(dat_tmpl.replace("__POT__", "UNIT40"), encoding="utf-8")

    direct_proc = _run_cli(
        "run",
        str(dat_hook),
        "--pes",
        "cl_hbr_leps",
        "--export-fu40",
        str(fu40_out),
        "--format",
        "json",
    )
    assert fu40_out.is_file()
    reloaded_proc = _run_cli(
        "run",
        str(dat_u40),
        "--fu40",
        str(fu40_out),
        "--format",
        "json",
    )
    direct_rows = json.loads(direct_proc.stdout)["rows"]
    reload_rows = json.loads(reloaded_proc.stdout)["rows"]
    for row_a, row_b in zip(direct_rows, reload_rows, strict=True):
        assert math.isclose(row_a["k_tst"], row_b["k_tst"], rel_tol=1.0e-6)
        assert math.isclose(row_a["k_cvt"], row_b["k_cvt"], rel_tol=1.0e-6)
        assert math.isclose(row_a["k_final"], row_b["k_final"], rel_tol=1.0e-6)


def test_cli_external_command_pes_and_restart_cache(tmp_path: Path) -> None:
    counter = tmp_path / "count.txt"
    driver = tmp_path / "driver.py"
    restart = tmp_path / "restart.json"
    toml_file = tmp_path / "job.toml"

    driver.write_text(
        f"""
import json
import sys
from pathlib import Path
import numpy as np
from pyrate.pes import h3_leps_surface

cnt = Path({str(counter)!r})
cnt.write_text(str((int(cnt.read_text()) if cnt.exists() else 0) + 1))

req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pes = h3_leps_surface()
syms = tuple(req["symbols"])
coords = np.asarray(req["coords_angstrom"], dtype=float)
n = len(syms)
# Evaluate arbitrary 1/2/3-atom fragment by placing non-interacting spectators at 200 A
if n == 1:
    eval_coords = np.vstack([coords, [[0.0, 0.0, 200.0], [0.0, 200.0, 200.0]]])
elif n == 2:
    eval_coords = np.vstack([coords, [[0.0, 0.0, 200.0]]])
else:
    eval_coords = coords
ev = pes.evaluate(
    ("H", "H", "H"),
    eval_coords,
    need_gradient=bool(req["need_gradient"] or req["need_hessian"]),
    need_hessian=bool(req["need_hessian"]),
)
out = {{"energy_kj_mol": ev.energy_kj_mol}}
if req["need_gradient"] and ev.gradient_kj_mol_angstrom is not None:
    out["gradient_kj_mol_angstrom"] = ev.gradient_kj_mol_angstrom[:n].tolist()
if req["need_hessian"] and ev.hessian_kj_mol_angstrom2 is not None:
    out["hessian_kj_mol_angstrom2"] = ev.hessian_kj_mol_angstrom2[: 3 * n, : 3 * n].tolist()
Path(sys.argv[2]).write_text(json.dumps(out), encoding="utf-8")
""".strip()
        + "\n",
        encoding="utf-8",
    )

    toml_file.write_text(
        """
[calculation]
name = "cli-restart"
temperatures_k = [300.0]
method = "tst"
tunneling = "none"

[[reactants]]
name = "H"
atoms = [{ symbol = "H", xyz = [0.0, 0.0, 0.0] }]

[[reactants]]
name = "H2"
rotor_type = "linear"
symmetry_number = 2
atoms = [
  { symbol = "H", xyz = [0.0, 0.0, 0.0] },
  { symbol = "H", xyz = [0.0, 0.0, 0.7414] },
]

[transition_state]
name = "TS"
rotor_type = "linear"
atoms = [
  { symbol = "H", xyz = [0.0, 0.0, -0.929] },
  { symbol = "H", xyz = [0.0, 0.0, 0.0] },
  { symbol = "H", xyz = [0.0, 0.0, 0.929] },
]
""",
        encoding="utf-8",
    )

    pes_arg = f"command:{shlex.quote(sys.executable)} {shlex.quote(str(driver))}"
    first = _run_cli(
        "run",
        str(toml_file),
        "--pes",
        pes_arg,
        "--restart",
        str(restart),
        "--format",
        "json",
    )
    calls_after_first = int(counter.read_text())
    assert calls_after_first > 0
    assert restart.is_file()

    second = _run_cli(
        "run",
        str(toml_file),
        "--pes",
        pes_arg,
        "--restart",
        str(restart),
        "--format",
        "json",
    )
    assert int(counter.read_text()) == calls_after_first
    assert json.loads(first.stdout)["rows"] == json.loads(second.stdout)["rows"]


def test_cli_validate_strict_reports_unsupported_and_accepts_valid(tmp_path: Path) -> None:
    bad_dat = tmp_path / "lcg4.dat"
    bad_dat.write_text(
        "*GENERAL\n  ATOMS\n    1 H\n  END\n*TUNNEL\n  LCG4\n",
        encoding="utf-8",
    )
    bad = _run_cli("validate", str(bad_dat), "--strict", check=False)
    assert bad.returncode == 2
    assert "*TUNNEL LCG4 line 6" in bad.stderr

    ok_dat = tmp_path / "valid.dat"
    ok_dat.write_text(
        """
*GENERAL
  ATOMS
    1 H
    2 H
    3 H
  END
*ENERGETICS
  POTENTIAL BUILTIN:h3_leps
*RATE
  TST
  TEMP
    300.0
  END
""",
        encoding="utf-8",
    )
    ok = _run_cli("validate", str(ok_dat), "--strict", check=False)
    assert ok.returncode == 0
    assert "strict validation passed" in ok.stdout

