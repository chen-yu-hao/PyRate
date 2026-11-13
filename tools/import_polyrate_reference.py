#!/usr/bin/env python3
"""Create a provenance manifest for a local Polyrate reference distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Permit execution from an uninstalled source checkout.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyrate.legacy import LegacyParseError, load_legacy  # noqa: E402
from pyrate.reference import load_fu15_summary  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_non_polyrate_data_file(path: Path, root: Path) -> bool:
    rel_parts = [part.lower() for part in path.relative_to(root).parts]
    return "poten" in rel_parts or "util" in rel_parts or "pot" in rel_parts[:-1]


def discover_inputs(root: Path) -> list[Path]:
    candidates = [
        path
        for suffix in ("*.dat", "*.fu5")
        for path in root.rglob(suffix)
        if path.is_file() and not _is_non_polyrate_data_file(path, root)
    ]
    # Prefer one path per physical file and stable ordering.
    return sorted(set(candidates), key=lambda path: str(path.relative_to(root)))


def build_manifest(root: Path) -> dict[str, Any]:
    outputs_by_stem: dict[str, list[Path]] = {}
    for output in root.rglob("*.fu15"):
        outputs_by_stem.setdefault(output.stem.lower(), []).append(output)

    cases: list[dict[str, Any]] = []
    for input_path in discover_inputs(root):
        relative = input_path.relative_to(root)
        case: dict[str, Any] = {
            "name": input_path.stem,
            "input": str(relative),
            "input_sha256": sha256(input_path),
        }
        try:
            document = load_legacy(input_path)
            sections_upper = {s.name.upper() for s in document.sections}
            is_auxiliary = sections_upper.issubset({"VTSTIC", "IVTST"})
            case.update(
                {
                    "role": "auxiliary_unit50" if is_auxiliary else "calculation_input",
                    "title": document.title,
                    "temperatures_k": list(document.temperatures_k),
                    "requested_methods": list(document.requested_methods()),
                    "parse_warnings": list(document.compatibility_warnings()),
                }
            )
        except (OSError, UnicodeError, LegacyParseError) as exc:
            case["parse_error"] = str(exc)

        outputs = outputs_by_stem.get(input_path.stem.lower(), [])
        if outputs:
            output = sorted(outputs, key=str)[0]
            case["fu15"] = str(output.relative_to(root))
            case["fu15_sha256"] = sha256(output)
            if output.stat().st_size == 0:
                case["reference_status"] = "empty_fu15_output"
            else:
                try:
                    table = load_fu15_summary(output)
                    case["reference"] = {
                        "direction": table.direction,
                        "unit": table.unit,
                        "columns": list(table.columns),
                        "temperatures_k": [row.temperature_k for row in table.rows],
                        "values": [list(row.values) for row in table.rows],
                    }
                except ValueError as exc:
                    case["fu15_parse_error"] = str(exc)
        cases.append(case)

    return {
        "schema": "pyrate-polyrate-reference-manifest-v1",
        "root_name": root.name,
        "case_count": len(cases),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="extracted Polyrate distribution root")
    parser.add_argument(
        "--output", type=Path, default=Path("reference/manifest.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    manifest = build_manifest(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {manifest['case_count']} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
