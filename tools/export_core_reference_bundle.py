#!/usr/bin/env python3
"""Export the small official-file bundle needed by the core legacy candidates.

The bundle is a CI artifact, not a vendored copy of the Polyrate distribution.
It preserves exact paths, bytes-as-text, sizes, and SHA-256 digests so parser
work can be reviewed without weakening provenance or conflating inspection with
numerical reproduction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_polyrate_reference import (
    locate_distribution_root,
    read_text_candidate,
    relative,
    sha256_file,
)

CORE_CLASSIFICATION = "core_methods_but_legacy_semantics_blocked"


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    text = read_text_candidate(path, allow_empty=True)
    return {
        "path": relative(path, root),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "text": text,
        "text_status": "readable" if text is not None else "binary_or_too_large",
    }


def build_bundle(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    distribution_root = locate_distribution_root(root)
    case_ids = tuple(
        report.get("cases_by_classification", {}).get(CORE_CLASSIFICATION, ())
    )
    cases: list[dict[str, Any]] = []

    for case_id in case_ids:
        input_matches = sorted((distribution_root / "testrun").rglob(f"{case_id}.dat"))
        if len(input_matches) != 1:
            raise RuntimeError(
                f"expected one primary input for {case_id}, found {len(input_matches)}"
            )
        primary = input_matches[0]
        sidecars = sorted(
            path
            for path in primary.parent.iterdir()
            if path.is_file() and path.stem == case_id
        )
        oracle = distribution_root / "testo" / f"{case_id}.fu15"
        files = list(sidecars)
        if oracle.is_file():
            files.append(oracle)
        cases.append(
            {
                "case_id": case_id,
                "primary_input": relative(primary, root),
                "files": [_file_record(path, root) for path in files],
            }
        )

    return {
        "schema": "pyrate-polyrate-core-reference-bundle-v1",
        "classification": CORE_CLASSIFICATION,
        "case_count": len(cases),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="extracted Polyrate distribution root")
    parser.add_argument("report", type=Path, help="audit report.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    bundle = build_bundle(args.root.resolve(), report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"core_case_count": bundle["case_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
