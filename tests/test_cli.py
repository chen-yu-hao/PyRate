from __future__ import annotations

import json
import subprocess
import sys


def test_cli_runs_example(project_root) -> None:
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


def test_cli_emits_and_compares_staged_numerical_trace(project_root, tmp_path) -> None:
    input_path = project_root / "examples" / "harmonic_cvt.toml"
    trace_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrate",
            "trace",
            str(input_path),
            "--action-grid-points",
            "9",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    trace = json.loads(trace_process.stdout)
    assert trace["schema"] == "pyrate-numerical-trace-v1"
    stages = {point["stage"] for point in trace["points"]}
    assert {"partition", "zpe", "activation_gibbs", "kappa", "rate"} <= stages

    reference = tmp_path / "reference.json"
    actual = tmp_path / "actual.json"
    reference.write_text(json.dumps(trace), encoding="utf-8")
    actual.write_text(json.dumps(trace), encoding="utf-8")
    compare_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrate",
            "compare-trace",
            str(reference),
            str(actual),
            "--strict",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(compare_process.stdout)
    assert report["success"] is True
    assert report["counts"]["failed"] == 0
    assert report["counts"]["missing"] == 0
