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
