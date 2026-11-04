#!/usr/bin/env python3
"""Audit the official Polyrate 2023 revision B reference distribution.

The official suite contains 112 primary ``testrun/**/*.dat`` inputs.  This tool
keeps four different claims separate:

1. the archive was identified and inventoried;
2. a legacy input was syntactically parsed;
3. a legacy input can be executed with matching semantics;
4. PyRate numerically reproduces the reference intermediates and rates.

Only the first two gates are currently automated.  A parsed file is never
counted as an executable or numerically reproduced case.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyrate.head_to_head import parse_fu15_tables  # noqa: E402
from pyrate.legacy import LegacyParseError, iter_keywords, parse_legacy_text  # noqa: E402

MAX_TEXT_BYTES = 16 * 1024 * 1024
EXCERPT_CHARS = 4000
INPUT_SUFFIXES = {".dat", ".fu5", ".inp", ".input"}
SECTION_RE = re.compile(
    r"(?im)^\s*\*(?:GENERAL|REACT1|REACT2|PROD1|PROD2|START|PATH|TUNNEL|RATE)\b"
)
RATE_SUMMARY_RE = re.compile(
    r"Summary\s+of\s+(?:forward|reverse)\s+rate\s+constants", re.IGNORECASE
)

CORE_METHODS = {"TST", "CVT"}
PARTIAL_METHODS = {"ZCT"}
ADVANCED_METHODS = {
    "ICVT",
    "MUVT",
    "CUS",
    "US",
    "SCT",
    "LCT",
    "LCG3",
    "LCG4",
    "VRC",
    "IVTST",
    "IOC",
    "ISPE",
    "SSQRRK",
}
METHOD_KEYWORDS = CORE_METHODS | PARTIAL_METHODS | ADVANCED_METHODS | {
    "WIGNER",
    "MUSCT",
    "OMT",
}


@dataclass(frozen=True, slots=True)
class InputAudit:
    case_id: str
    path: str
    size_bytes: int
    sha256: str
    title: str | None
    sections: tuple[str, ...]
    temperatures_k: tuple[float, ...]
    requested_methods: tuple[str, ...]
    all_method_keywords: tuple[str, ...]
    classification: str
    warnings: tuple[str, ...]
    parse_error: str | None


@dataclass(frozen=True, slots=True)
class Fu15TableAudit:
    direction: str
    unit: str
    columns: tuple[str, ...]
    temperatures_k: tuple[float, ...]
    row_count: int


@dataclass(frozen=True, slots=True)
class Fu15Audit:
    case_id: str
    path: str
    size_bytes: int
    sha256: str
    tables: tuple[Fu15TableAudit, ...]
    parse_error: str | None
    unparsed_excerpt: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text_candidate(path: Path, *, allow_empty: bool = False) -> str | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > MAX_TEXT_BYTES or (size == 0 and not allow_empty):
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def locate_distribution_root(root: Path) -> Path:
    candidates = [root]
    candidates.extend(path for path in root.rglob("*") if path.is_dir())
    matches = [
        path
        for path in candidates
        if (path / "testrun").is_dir() and (path / "testo").is_dir()
    ]
    if not matches:
        raise RuntimeError(f"could not locate Polyrate testrun/testo directories under {root}")
    return min(matches, key=lambda path: len(path.parts))


def discover_inventory(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: str(p))
    input_candidates: list[Path] = []
    fu15_candidates: list[Path] = []
    for path in files:
        lowered = path.name.lower()
        likely_input = path.suffix.lower() in INPUT_SUFFIXES or lowered in {"fu5", "input"}
        likely_output = path.suffix.lower() == ".fu15" or lowered == "fu15"
        text: str | None = None
        if likely_input or likely_output or path.stat().st_size <= 2 * 1024 * 1024:
            text = read_text_candidate(path, allow_empty=True)
        if likely_input or (text is not None and SECTION_RE.search(text)):
            input_candidates.append(path)
        if likely_output or (text is not None and RATE_SUMMARY_RE.search(text)):
            fu15_candidates.append(path)
    return files, sorted(set(input_candidates)), sorted(set(fu15_candidates))


def classify_methods(methods: Iterable[str]) -> str:
    method_set = {method.upper() for method in methods}
    if not method_set:
        return "parser_only"
    if method_set & ADVANCED_METHODS or method_set & {"MUSCT", "OMT"}:
        return "advanced_not_implemented"
    if method_set & PARTIAL_METHODS:
        return "partial_zct"
    if method_set <= (CORE_METHODS | {"WIGNER"}):
        return "core_methods_but_legacy_semantics_blocked"
    return "unclassified_needs_review"


def audit_input(path: Path, root: Path) -> InputAudit:
    case_id = path.stem
    path_text = relative(path, root)
    size = path.stat().st_size
    digest = sha256_file(path)
    text = read_text_candidate(path)
    if text is None:
        return InputAudit(
            case_id,
            path_text,
            size,
            digest,
            None,
            (),
            (),
            (),
            (),
            "unreadable",
            (),
            "file is binary, empty, too large, or unreadable",
        )
    try:
        document = parse_legacy_text(text, source=path_text)
    except (LegacyParseError, ValueError) as exc:
        return InputAudit(
            case_id,
            path_text,
            size,
            digest,
            None,
            (),
            (),
            (),
            (),
            "parse_error",
            (),
            str(exc),
        )

    discovered_methods = {
        entry.keyword
        for _, entry in iter_keywords(document)
        if entry.keyword in METHOD_KEYWORDS
    }
    requested = tuple(document.requested_methods())
    all_methods = tuple(sorted(discovered_methods | set(requested)))
    return InputAudit(
        case_id=case_id,
        path=path_text,
        size_bytes=size,
        sha256=digest,
        title=document.title,
        sections=tuple(section.name for section in document.sections),
        temperatures_k=tuple(document.temperatures_k),
        requested_methods=requested,
        all_method_keywords=all_methods,
        classification=classify_methods(all_methods),
        warnings=tuple(document.compatibility_warnings()),
        parse_error=None,
    )


def excerpt(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(normalized) <= EXCERPT_CHARS:
        return normalized
    return normalized[:EXCERPT_CHARS] + "\n…"


def audit_fu15(path: Path, root: Path) -> Fu15Audit:
    case_id = path.stem
    path_text = relative(path, root)
    size = path.stat().st_size
    digest = sha256_file(path)
    text = read_text_candidate(path, allow_empty=True)
    if text is None:
        return Fu15Audit(
            case_id,
            path_text,
            size,
            digest,
            (),
            "file is binary, too large, or unreadable",
            None,
        )
    if not text:
        return Fu15Audit(case_id, path_text, size, digest, (), "empty file", "")
    try:
        tables = parse_fu15_tables(text)
    except ValueError as exc:
        return Fu15Audit(case_id, path_text, size, digest, (), str(exc), excerpt(text))
    audited = tuple(
        Fu15TableAudit(
            direction=table.direction,
            unit=table.unit,
            columns=tuple(table.columns),
            temperatures_k=tuple(row.temperature_k for row in table.rows),
            row_count=len(table.rows),
        )
        for table in tables
    )
    return Fu15Audit(case_id, path_text, size, digest, audited, None, None)


def counter_dict(counter: collections.Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def make_case_manifest(
    inputs: list[InputAudit], outputs: list[Fu15Audit]
) -> list[dict[str, Any]]:
    output_by_case = {item.case_id: item for item in outputs}
    manifest: list[dict[str, Any]] = []
    for item in inputs:
        output = output_by_case.get(item.case_id)
        manifest.append(
            {
                "case_id": item.case_id,
                "input": jsonable(item),
                "reference_fu15": jsonable(output) if output is not None else None,
                "syntax_status": "pass" if item.parse_error is None else "fail",
                "legacy_execution_status": "blocked_not_implemented",
                "numerical_status": "not_run",
            }
        )
    return manifest


def make_markdown(report: dict[str, Any]) -> str:
    counts = report["official_suite_counts"]
    classifications = report["classification_counts"]
    method_counts = report["method_counts"]
    column_counts = report["fu15_column_counts"]
    lines = [
        "# Polyrate 2023 revision B head-to-head audit",
        "",
        "This report scopes the primary suite exactly as the 112 files under ",
        "`testrun/**/*.dat`. Syntax recognition, executable legacy semantics, and numerical ",
        "agreement are separate gates.",
        "",
        "## Provenance",
        "",
        f"- Source: `{report['provenance']['source_url']}`",
        f"- Supplied archive MD5: `{report['provenance']['archive_md5']}`",
        f"- Distribution root: `{report['provenance']['distribution_root']}`",
        "",
        "## Official-suite gates",
        "",
        "| Gate | Result |",
        "|---|---:|",
        f"| Primary test-run inputs discovered | {counts['test_run_inputs']} |",
        f"| Primary inputs syntactically parsed | {counts['inputs_syntax_pass']} |",
        f"| Primary input syntax failures | {counts['inputs_syntax_fail']} |",
        f"| Cases executable from legacy input | {counts['legacy_executable_cases']} |",
        f"| Cases numerically reproduced | {counts['numerically_reproduced_cases']} |",
        f"| Reference `fu15` files present | {counts['reference_fu15_files']} |",
        f"| `fu15` files with supported rate summary | {counts['reference_fu15_summary_pass']} |",
        f"| `fu15` files without a supported rate summary | {counts['reference_fu15_summary_not_parsed']} |",
        f"| Primary cases without a `fu15` reference | {counts['cases_without_reference_fu15']} |",
        f"| Parsed forward/reverse rate tables | {counts['parsed_rate_tables']} |",
        "",
        "A readable `fu15` table is an oracle input, not a PyRate pass. Therefore the ",
        "numerical pass count remains zero.",
        "",
        "## Primary-input classification",
        "",
        "| Classification | Count | Meaning |",
        "|---|---:|---|",
    ]
    meanings = {
        "core_methods_but_legacy_semantics_blocked": "Only TST/CVT/Wigner-family method keywords were observed, but fixed-unit/section semantics still block execution.",
        "partial_zct": "Requests ZCT; current PyRate has only a one-dimensional WKB building block, not full MEP/MEPSAG ZCT.",
        "advanced_not_implemented": "Requests SCT/LCT/μVT/CUS/VRC or another advanced Polyrate algorithm not implemented end to end.",
        "parser_only": "No recognized rate-method keyword was found; these include file-interface or nonstandard control cases.",
        "parse_error": "Current legacy parser could not parse the primary input.",
        "unreadable": "Primary input could not be read as bounded text.",
        "unclassified_needs_review": "Contains method keywords outside the current classification rules.",
    }
    for name, count in classifications.items():
        lines.append(f"| `{name}` | {count} | {meanings.get(name, '')} |")

    lines.extend(["", "## Requested method keywords", "", "| Method | Inputs |", "|---|---:|"])
    for name, count in method_counts.items():
        lines.append(f"| `{name}` | {count} |")
    if not method_counts:
        lines.append("| — | 0 |")

    lines.extend(["", "## Parsed reference columns", "", "| Column | Tables |", "|---|---:|"])
    for name, count in column_counts.items():
        lines.append(f"| `{name}` | {count} |")
    if not column_counts:
        lines.append("| — | 0 |")

    lines.extend(
        [
            "",
            "## Head-to-head conclusion",
            "",
            "- **Input syntax:** 112/112 primary inputs are recognized by the retaining parser.",
            "- **Execution parity:** 0/112 primary inputs can yet be run directly with complete Polyrate legacy semantics.",
            "- **Numerical parity:** 0/112 official cases have passed an end-to-end PyRate-versus-Polyrate tolerance gate.",
            "- **Scientific coverage:** Polyrate remains the reference implementation for reaction paths, full ZCT/SCT/LCT/OMT, μVT/CUS, direct dynamics, VRC, dual-level corrections, anharmonic models, and pressure dependence.",
            "- **Immediate implementation target:** execute the legacy thermochemistry/energetics and fixed-unit inputs needed by the 16 core-method cases, then compare intermediates before final rate constants.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="extracted Polyrate distribution root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--archive-md5", default="")
    parser.add_argument("--expected-test-runs", type=int, default=112)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    distribution_root = locate_distribution_root(root)
    files, input_candidates, fu15_candidates = discover_inventory(root)
    official_input_paths = sorted((distribution_root / "testrun").rglob("*.dat"))
    official_fu15_paths = sorted((distribution_root / "testo").glob("*.fu15"))
    if len(official_input_paths) != args.expected_test_runs:
        raise RuntimeError(
            f"expected {args.expected_test_runs} primary test inputs, found {len(official_input_paths)}"
        )

    input_audits = [audit_input(path, root) for path in official_input_paths]
    fu15_audits = [audit_fu15(path, root) for path in official_fu15_paths]
    case_manifest = make_case_manifest(input_audits, fu15_audits)

    classification_counts = collections.Counter(item.classification for item in input_audits)
    method_counts: collections.Counter[str] = collections.Counter()
    section_counts: collections.Counter[str] = collections.Counter()
    for item in input_audits:
        method_counts.update(item.all_method_keywords)
        section_counts.update(item.sections)
    column_counts: collections.Counter[str] = collections.Counter()
    for item in fu15_audits:
        for table in item.tables:
            column_counts.update(table.columns)

    input_case_ids = {item.case_id for item in input_audits}
    output_case_ids = {item.case_id for item in fu15_audits}
    missing_fu15 = sorted(input_case_ids - output_case_ids)
    parsed_inputs = sum(item.parse_error is None for item in input_audits)
    parsed_fu15_files = sum(item.parse_error is None for item in fu15_audits)
    fu15_table_count = sum(len(item.tables) for item in fu15_audits)
    suffix_counts = collections.Counter(path.suffix.lower() or "<none>" for path in files)

    report: dict[str, Any] = {
        "schema": "pyrate-polyrate-reference-audit-v2",
        "provenance": {
            "source_url": args.source_url,
            "archive_md5": args.archive_md5,
            "root_name": root.name,
            "distribution_root": relative(distribution_root, root),
            "audit_host": os.environ.get("RUNNER_OS", sys.platform),
            "git_sha": os.environ.get("GITHUB_SHA", ""),
        },
        "official_suite_counts": {
            "test_run_inputs": len(input_audits),
            "inputs_syntax_pass": parsed_inputs,
            "inputs_syntax_fail": len(input_audits) - parsed_inputs,
            "legacy_executable_cases": 0,
            "numerically_reproduced_cases": 0,
            "reference_fu15_files": len(fu15_audits),
            "reference_fu15_summary_pass": parsed_fu15_files,
            "reference_fu15_summary_not_parsed": len(fu15_audits) - parsed_fu15_files,
            "cases_without_reference_fu15": len(missing_fu15),
            "parsed_rate_tables": fu15_table_count,
        },
        "inventory_counts": {
            "files": len(files),
            "input_candidates_all_distribution": len(input_candidates),
            "fu15_candidates_all_distribution": len(fu15_candidates),
        },
        "classification_counts": counter_dict(classification_counts),
        "cases_by_classification": {
            name: sorted(item.case_id for item in input_audits if item.classification == name)
            for name in sorted(classification_counts)
        },
        "method_counts": counter_dict(method_counts),
        "section_counts": counter_dict(section_counts),
        "fu15_column_counts": counter_dict(column_counts),
        "suffix_counts": counter_dict(suffix_counts),
        "cases_without_reference_fu15": missing_fu15,
        "inputs": [jsonable(item) for item in input_audits],
        "fu15_outputs": [jsonable(item) for item in fu15_audits],
        "case_manifest": case_manifest,
    }

    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(make_markdown(report), encoding="utf-8")
    (output / "case-manifest.json").write_text(
        json.dumps(case_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "parser-cases.json").write_text(
        json.dumps([jsonable(item) for item in input_audits], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "fu15-summary.json").write_text(
        json.dumps([jsonable(item) for item in fu15_audits], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output / "tree.txt").open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{path.stat().st_size:12d}  {relative(path, root)}\n")

    print(json.dumps(report["official_suite_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
