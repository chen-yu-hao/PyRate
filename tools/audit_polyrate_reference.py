#!/usr/bin/env python3
"""Inventory and audit an extracted Polyrate 2023 reference distribution.

The audit is intentionally conservative.  It measures what the current PyRate
parser and ``fu15`` reader can recognize, classifies cases by requested methods,
and records provenance.  It does not count a case as numerically reproduced
unless a separate rate comparison has actually been executed.
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
class CandidateFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class InputAudit:
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
    path: str
    size_bytes: int
    sha256: str
    tables: tuple[Fu15TableAudit, ...]
    parse_error: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text_candidate(path: Path) -> str | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= 0 or size > MAX_TEXT_BYTES:
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


def discover(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: str(p))
    inputs: list[Path] = []
    outputs: list[Path] = []
    for path in files:
        lowered = path.name.lower()
        likely_input = path.suffix.lower() in INPUT_SUFFIXES or lowered in {"fu5", "input"}
        likely_output = path.suffix.lower() == ".fu15" or lowered == "fu15"
        text: str | None = None
        if likely_input or likely_output or path.stat().st_size <= 2 * 1024 * 1024:
            text = read_text_candidate(path)
        if likely_input or (text is not None and SECTION_RE.search(text)):
            inputs.append(path)
        if likely_output or (text is not None and RATE_SUMMARY_RE.search(text)):
            outputs.append(path)
    return files, sorted(set(inputs)), sorted(set(outputs))


def classify_methods(methods: Iterable[str]) -> str:
    method_set = {method.upper() for method in methods}
    if not method_set:
        return "parser_only"
    if method_set & ADVANCED_METHODS or method_set & {"MUSCT", "OMT"}:
        return "advanced_not_implemented"
    if method_set & PARTIAL_METHODS:
        return "partial_zct"
    if method_set <= (CORE_METHODS | {"WIGNER"}):
        return "core_candidate_needs_semantic_translation"
    return "unclassified_needs_review"


def audit_input(path: Path, root: Path) -> InputAudit:
    path_text = relative(path, root)
    size = path.stat().st_size
    digest = sha256_file(path)
    text = read_text_candidate(path)
    if text is None:
        return InputAudit(
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


def audit_fu15(path: Path, root: Path) -> Fu15Audit:
    path_text = relative(path, root)
    size = path.stat().st_size
    digest = sha256_file(path)
    text = read_text_candidate(path)
    if text is None:
        return Fu15Audit(
            path_text,
            size,
            digest,
            (),
            "file is binary, empty, too large, or unreadable",
        )
    try:
        tables = parse_fu15_tables(text)
    except ValueError as exc:
        return Fu15Audit(path_text, size, digest, (), str(exc))
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
    return Fu15Audit(path_text, size, digest, audited, None)


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


def make_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    classifications = report["classification_counts"]
    method_counts = report["method_counts"]
    column_counts = report["fu15_column_counts"]
    lines = [
        "# Polyrate 2023 revision B reference-suite audit",
        "",
        "This report audits the official extracted distribution against the current PyRate ",
        "parser and result readers. It is an inventory/readiness report, **not a claim that ",
        "the official numerical cases have been reproduced**.",
        "",
        "## Provenance",
        "",
        f"- Source: `{report['provenance']['source_url']}`",
        f"- Supplied archive MD5: `{report['provenance']['archive_md5']}`",
        f"- Audit root: `{report['provenance']['root_name']}`",
        "",
        "## Inventory",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Extracted files | {counts['files']} |",
        f"| Input candidates | {counts['input_candidates']} |",
        f"| Inputs parsed | {counts['inputs_parsed']} |",
        f"| Input parse failures | {counts['input_parse_failures']} |",
        f"| `fu15` candidates | {counts['fu15_candidates']} |",
        f"| `fu15` files parsed | {counts['fu15_files_parsed']} |",
        f"| `fu15` parse failures | {counts['fu15_parse_failures']} |",
        f"| Parsed rate tables | {counts['fu15_tables']} |",
        f"| Numerically reproduced official cases | {counts['numerically_reproduced_cases']} |",
        "",
        "## Case classification",
        "",
        "| Classification | Count | Meaning |",
        "|---|---:|---|",
    ]
    meanings = {
        "core_candidate_needs_semantic_translation": "Only current core method keywords were observed; legacy semantics still need translation before execution.",
        "partial_zct": "Requests ZCT; current PyRate has only a one-dimensional WKB building block.",
        "advanced_not_implemented": "Requests one or more advanced Polyrate algorithms not implemented in PyRate.",
        "parser_only": "No recognized rate-method keyword was found.",
        "parse_error": "Current legacy parser could not parse the file.",
        "unreadable": "Candidate could not be read as bounded text.",
        "unclassified_needs_review": "Contains method keywords outside the current classification rules.",
    }
    for name, count in classifications.items():
        lines.append(f"| `{name}` | {count} | {meanings.get(name, '')} |")

    lines.extend(["", "## Requested method keywords", "", "| Method | Inputs |", "|---|---:|"])
    for name, count in method_counts.items():
        lines.append(f"| `{name}` | {count} |")
    if not method_counts:
        lines.append("| — | 0 |")

    lines.extend(["", "## Parsed `fu15` columns", "", "| Column | Tables |", "|---|---:|"])
    for name, count in column_counts.items():
        lines.append(f"| `{name}` | {count} |")
    if not column_counts:
        lines.append("| — | 0 |")

    lines.extend(
        [
            "",
            "## Head-to-head conclusion",
            "",
            "- **Scientific coverage:** Polyrate remains the reference implementation for the full suite. ",
            "  PyRate currently covers a harmonic CTST/CVT core, Wigner correction, and a partial ",
            "  one-dimensional WKB/ZCT building block.",
            "- **Reference evidence:** Parsing a file or reading its summary is not numerical agreement. ",
            "  The official-pass count remains zero until the same physical model is executed and ",
            "  intermediate quantities plus final rates satisfy declared tolerances.",
            "- **Next gate:** translate the core-candidate legacy inputs into executable typed models, ",
            "  then compare partition functions, free-energy profiles, bottleneck positions, tunneling ",
            "  actions, and final rates before promoting any compatibility status.",
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
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    files, input_paths, fu15_paths = discover(root)
    input_audits = [audit_input(path, root) for path in input_paths]
    fu15_audits = [audit_fu15(path, root) for path in fu15_paths]

    classification_counts = collections.Counter(item.classification for item in input_audits)
    method_counts: collections.Counter[str] = collections.Counter()
    for item in input_audits:
        method_counts.update(item.all_method_keywords)
    column_counts: collections.Counter[str] = collections.Counter()
    for item in fu15_audits:
        for table in item.tables:
            column_counts.update(table.columns)

    parsed_inputs = sum(item.parse_error is None for item in input_audits)
    parsed_fu15_files = sum(item.parse_error is None for item in fu15_audits)
    fu15_table_count = sum(len(item.tables) for item in fu15_audits)
    suffix_counts = collections.Counter(path.suffix.lower() or "<none>" for path in files)

    report: dict[str, Any] = {
        "schema": "pyrate-polyrate-reference-audit-v1",
        "provenance": {
            "source_url": args.source_url,
            "archive_md5": args.archive_md5,
            "root_name": root.name,
            "audit_host": os.environ.get("RUNNER_OS", sys.platform),
            "git_sha": os.environ.get("GITHUB_SHA", ""),
        },
        "counts": {
            "files": len(files),
            "input_candidates": len(input_audits),
            "inputs_parsed": parsed_inputs,
            "input_parse_failures": len(input_audits) - parsed_inputs,
            "fu15_candidates": len(fu15_audits),
            "fu15_files_parsed": parsed_fu15_files,
            "fu15_parse_failures": len(fu15_audits) - parsed_fu15_files,
            "fu15_tables": fu15_table_count,
            "numerically_reproduced_cases": 0,
        },
        "classification_counts": counter_dict(classification_counts),
        "method_counts": counter_dict(method_counts),
        "fu15_column_counts": counter_dict(column_counts),
        "suffix_counts": counter_dict(suffix_counts),
        "inputs": [jsonable(item) for item in input_audits],
        "fu15_outputs": [jsonable(item) for item in fu15_audits],
    }

    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(make_markdown(report), encoding="utf-8")
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

    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
