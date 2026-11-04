from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pyrate.cli import main

LEGACY = """
*GENERAL
 TITLE
 test reaction
 END
*RATE
 CVT
 TEMP
  300.0
 END
*TUNNEL
 SCT
"""


def test_module_entry_point_runs_example(project_root: Path) -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrate",
            "run",
            str(project_root / "examples" / "harmonic_cvt.toml"),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(process.stdout)
    assert payload["method"] == "cvt"
    assert len(payload["rows"]) == 4
    assert all(row["k_final"] > 0.0 for row in payload["rows"])


@pytest.mark.parametrize("output_format", ["table", "csv", "json"])
def test_run_formats(
    project_root: Path, capsys: pytest.CaptureFixture[str], output_format: str
) -> None:
    example = project_root / "examples" / "wkb_tst.toml"
    assert main(["run", str(example), "--format", output_format]) == 0
    text = capsys.readouterr().out
    if output_format == "json":
        assert json.loads(text)["tunneling"] == "wkb"
    elif output_format == "csv":
        assert text.splitlines()[0].startswith("temperature_k,")
        assert len(text.splitlines()) == 4
    else:
        assert "k_final" in text and "cm^3 molecule^-1 s^-1" in text


def test_run_rate_unit_override_and_output_file(project_root: Path, tmp_path: Path) -> None:
    example = project_root / "examples" / "wkb_tst.toml"
    target = tmp_path / "rates.json"
    argv = ["run", str(example), "--format", "json", "--rate-unit", "L mol^-1 s^-1"]
    assert main([*argv, "-o", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert {row["rate_unit"] for row in payload["rows"]} == {"L mol^-1 s^-1"}


def test_inspect_and_validate_legacy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "case.dat"
    source.write_text(LEGACY, encoding="utf-8")
    assert main(["inspect", str(source), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["temperatures_k"] == [300.0]
    assert summary["requested_methods"] == ["CVT", "SCT"]
    assert main(["validate", str(source)]) == 0
    assert "SCT" in capsys.readouterr().out


def test_inspect_and_validate_toml(project_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    example = project_root / "examples" / "harmonic_cvt.toml"
    assert main(["inspect", str(example)]) == 0
    assert "path_points: 5" in capsys.readouterr().out
    assert main(["validate", str(example)]) == 0
    assert "4 temperatures" in capsys.readouterr().out


def test_native_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["native-info"]) == 0
    assert json.loads(capsys.readouterr().out)["backend"] in {"cpp", "python"}


def test_running_legacy_input_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "case.dat"
    source.write_text(LEGACY, encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["run", str(source)])
    assert excinfo.value.code == 2
    assert "legacy Polyrate files" in capsys.readouterr().err
