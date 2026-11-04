#!/usr/bin/env python3
"""Execute the currently mapped core legacy subset against official ``fu15`` tables."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyrate.calculation import calculate_rates  # noqa: E402
from pyrate.input import load_legacy_calculation  # noqa: E402
from pyrate.legacy import load_legacy  # noqa: E402
from pyrate.legacy_execution import analyze_legacy_execution  # noqa: E402
from pyrate.reference import ReferenceRateTable, parse_fu15_tables  # noqa: E402

SCHEMA = "pyrate-core-legacy-execution-validation-v1"
_NUMBER_TOKEN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?$"
)


def _materialize(case: Mapping[str, Any], directory: Path) -> tuple[Path, str | None]:
    primary = Path(str(case["primary_input"])).name
    reference_text: str | None = None
    for file_data in case.get("files", []):
        if not isinstance(file_data, Mapping):
            continue
        text = file_data.get("text")
        path_value = file_data.get("path")
        if not isinstance(text, str) or not isinstance(path_value, str):
            continue
        name = Path(path_value).name
        (directory / name).write_text(text, encoding="utf-8")
        if name.lower().endswith(".fu15"):
            reference_text = text
    return directory / primary, reference_text


def _forward_tst_table(text: str) -> ReferenceRateTable:
    tables = parse_fu15_tables(text)
    for table in tables:
        if table.direction == "forward" and any(
            column.upper() == "TST" for column in table.columns
        ):
            return table
    raise ValueError("no forward TST reference table")


def _forward_tst_printed_tokens(text: str) -> dict[float, str]:
    """Return the literal TST token printed for each forward-table temperature."""

    lines = text.splitlines()
    for start, line in enumerate(lines):
        lowered = line.lower()
        is_forward = (
            "summary of forward rate constants" in lowered
            or "forward tst rate constants" in lowered
            or "final rate constants" in lowered
        )
        if not is_forward:
            continue
        header_index: int | None = None
        for index in range(start + 1, min(len(lines), start + 24)):
            if "T(K)" in lines[index].upper().replace(" ", ""):
                header_index = index
                break
        if header_index is None:
            continue
        headers = lines[header_index].split()
        try:
            tst_index = tuple(item.upper() for item in headers[1:]).index("TST")
        except ValueError:
            continue
        tokens_by_temperature: dict[float, str] = {}
        for row_line in lines[header_index + 1 :]:
            tokens = row_line.split()
            if not tokens:
                if tokens_by_temperature:
                    break
                continue
            if len(tokens) != len(headers):
                if tokens_by_temperature:
                    break
                continue
            if not all(_NUMBER_TOKEN.fullmatch(token) for token in tokens):
                if tokens_by_temperature:
                    break
                continue
            temperature = float(tokens[0].replace("D", "E").replace("d", "e"))
            tokens_by_temperature[temperature] = tokens[tst_index + 1]
        if tokens_by_temperature:
            return tokens_by_temperature
    return {}


def _printed_half_interval(token: str | None) -> float:
    """Return half a decimal unit in the last printed digit."""

    if token is None:
        return 0.0
    try:
        value = Decimal(token.replace("D", "E").replace("d", "e"))
    except InvalidOperation as exc:
        raise ValueError(f"invalid printed numeric token {token!r}") from exc
    if not value.is_finite():
        raise ValueError(f"non-finite printed numeric token {token!r}")
    quantum = Decimal(1).scaleb(value.as_tuple().exponent)
    return float(abs(quantum) / 2)


def _compare_printed_values(
    table: ReferenceRateTable,
    result_rows: list[Mapping[str, Any]],
    printed_tokens: Mapping[float, str],
    relative_tolerance: float,
) -> dict[str, Any]:
    column_index = tuple(column.upper() for column in table.columns).index("TST")
    actual_by_temperature = {
        float(row["temperature_k"]): float(row["k_tst"])
        for row in result_rows
    }
    points: list[dict[str, Any]] = []
    errors: list[float] = []
    limits: list[float] = []
    for reference_row in table.rows:
        expected = reference_row.values[column_index]
        actual = actual_by_temperature.get(reference_row.temperature_k)
        printed_token = printed_tokens.get(reference_row.temperature_k)
        rounding_half_interval = _printed_half_interval(printed_token)
        if actual is None:
            points.append(
                {
                    "temperature_k": reference_row.temperature_k,
                    "expected": expected,
                    "printed_token": printed_token,
                    "actual": None,
                    "absolute_error": None,
                    "relative_error": None,
                    "rounding_half_interval": rounding_half_interval,
                    "allowed_absolute_error": None,
                    "status": "missing",
                }
            )
            continue
        absolute_error = abs(actual - expected)
        allowed_absolute_error = (
            relative_tolerance * abs(expected) + rounding_half_interval
        )
        if expected == 0.0:
            relative_error = 0.0 if actual == 0.0 else math.inf
            effective_relative_limit = math.inf
        else:
            relative_error = absolute_error / abs(expected)
            effective_relative_limit = allowed_absolute_error / abs(expected)
        errors.append(relative_error)
        limits.append(effective_relative_limit)
        points.append(
            {
                "temperature_k": reference_row.temperature_k,
                "expected": expected,
                "printed_token": printed_token,
                "actual": actual,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "rounding_half_interval": rounding_half_interval,
                "allowed_absolute_error": allowed_absolute_error,
                "effective_relative_limit": effective_relative_limit,
                "status": "pass" if absolute_error <= allowed_absolute_error else "fail",
            }
        )
    missing = sum(point["status"] == "missing" for point in points)
    failed = sum(point["status"] == "fail" for point in points)
    return {
        "reference_unit": table.unit,
        "comparison_mode": "printed-values-with-rounding-interval",
        "unit_context": (
            "compact TST fu15 omits its unit; the executable legacy mapper emits "
            "the Polyrate bimolecular cm^3 molecule^-1 s^-1 convention"
        ),
        "tolerance_policy": (
            "absolute error <= relative_tolerance*abs(reference) + "
            "half of one unit in the last printed decimal digit"
        ),
        "compared": len(errors),
        "missing": missing,
        "failed": failed,
        "max_relative_error": max(errors, default=math.inf),
        "max_effective_relative_limit": max(limits, default=math.inf),
        "points": points,
        "relative_tolerance": relative_tolerance,
        "success": len(errors) > 0 and missing == 0 and failed == 0,
    }


def validate_case(case: Mapping[str, Any], relative_tolerance: float) -> dict[str, Any]:
    case_id = str(case.get("case_id", "unknown"))
    with tempfile.TemporaryDirectory(prefix=f"pyrate-{case_id}-") as temporary:
        directory = Path(temporary)
        input_path, reference_text = _materialize(case, directory)
        document = load_legacy(input_path)
        analysis = analyze_legacy_execution(document, input_path)
        payload: dict[str, Any] = {
            "case_id": case_id,
            "analysis": analysis.as_dict(),
            "execution_status": "blocked",
            "printed_rate_status": "not_run",
        }
        if not analysis.executable:
            return payload

        loaded = load_legacy_calculation(input_path)
        result = calculate_rates(loaded.reaction, loaded.options, name=loaded.name)
        payload["execution_status"] = "pass"
        payload["result_unit"] = result.rows[0].rate_unit
        payload["temperature_count"] = len(result.rows)
        if reference_text is None:
            payload["printed_rate_status"] = "missing_reference"
            return payload

        table = _forward_tst_table(reference_text)
        comparison = _compare_printed_values(
            table,
            [row.as_dict() for row in result.rows],
            _forward_tst_printed_tokens(reference_text),
            relative_tolerance,
        )
        payload["comparison"] = comparison
        payload["printed_rate_status"] = (
            "pass" if comparison["success"] else "fail"
        )
        return payload


def _update_audit_outputs(
    audit_report_path: Path,
    case_manifest_path: Path,
    execution_report: Mapping[str, Any],
) -> None:
    """Merge execution evidence back into the canonical 112-case audit."""

    audit = json.loads(audit_report_path.read_text(encoding="utf-8"))
    manifest = audit.get("case_manifest")
    if not isinstance(manifest, list):
        raise ValueError("audit report has no case_manifest array")
    raw_results = execution_report.get("cases")
    if not isinstance(raw_results, list):
        raise ValueError("execution report has no cases array")
    results = {
        str(item.get("case_id")): item
        for item in raw_results
        if isinstance(item, Mapping)
    }
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        result = results.get(str(entry.get("case_id")))
        if result is None:
            continue
        analysis = result.get("analysis")
        if isinstance(analysis, Mapping):
            entry["semantic_status"] = analysis.get("semantic_status", "unknown")
            entry["execution_blockers"] = list(analysis.get("blockers", []))
        entry["legacy_execution_status"] = result.get("execution_status", "blocked")
        printed_status = result.get("printed_rate_status", "not_run")
        entry["printed_rate_status"] = printed_status
        entry["intermediate_validation_status"] = "not_run"
        entry["numerical_status"] = (
            "partial_final_rate_only" if printed_status == "pass" else "not_run"
        )
        comparison = result.get("comparison")
        if isinstance(comparison, Mapping):
            entry["numerical_comparison"] = {
                "compared": comparison.get("compared"),
                "failed": comparison.get("failed"),
                "max_relative_error": comparison.get("max_relative_error"),
                "relative_tolerance": comparison.get("relative_tolerance"),
                "comparison_mode": comparison.get("comparison_mode"),
            }

    executable = int(execution_report.get("executable_cases", 0))
    printed_matches = int(execution_report.get("printed_rate_matches", 0))
    counts = audit.get("official_suite_counts")
    if not isinstance(counts, dict):
        raise ValueError("audit report has no official_suite_counts object")
    counts["legacy_executable_cases"] = executable
    counts["printed_rate_matches"] = printed_matches
    counts["intermediate_quantity_validated_cases"] = 0
    counts["numerically_reproduced_cases"] = 0
    audit["case_manifest"] = manifest
    audit["legacy_execution_validation"] = {
        "schema": execution_report.get("schema"),
        "case_count": execution_report.get("case_count"),
        "executable_cases": executable,
        "printed_rate_matches": printed_matches,
        "printed_rate_failures": execution_report.get("printed_rate_failures", 0),
        "relative_tolerance": execution_report.get("relative_tolerance"),
    }
    audit_report_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    case_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    markdown_path = audit_report_path.with_suffix(".md")
    if not markdown_path.is_file():
        return
    markdown = markdown_path.read_text(encoding="utf-8")
    substitutions = (
        (
            r"(?m)^\| Cases executable from legacy input \| \d+ \|$",
            f"| Cases executable from legacy input | {executable} |",
        ),
        (
            (
                r"(?m)^\| Cases numerically reproduced \| \d+ \|$"
                r"(?:\n\| Cases matching printed final TST rates \| \d+ \|)?"
            ),
            (
                "| Cases numerically reproduced | 0 |\n"
                f"| Cases matching printed final TST rates | {printed_matches} |"
            ),
        ),
        (
            r"(?m)^- \*\*Execution parity:\*\*.*$",
            (
                f"- **Execution parity:** {executable}/112 primary inputs run through "
                "the mapped legacy solver path."
            ),
        ),
        (
            r"(?m)^- \*\*Numerical parity:\*\*.*$",
            (
                "- **Numerical parity:** 0/112 cases have complete staged "
                "intermediate-quantity and final-rate validation."
            ),
        ),
    )
    for pattern, replacement in substitutions:
        markdown = re.sub(pattern, replacement, markdown)
    execution_line = (
        f"- **Execution parity:** {executable}/112 primary inputs run through "
        "the mapped legacy solver path."
    )
    printed_line = (
        f"- **Printed final-rate parity:** {printed_matches}/112 official cases "
        "match their printed forward TST tables."
    )
    if printed_line not in markdown:
        markdown = markdown.replace(execution_line, execution_line + "\n" + printed_line)
    markdown = markdown.replace(
        "A readable `fu15` table is an oracle input, not a PyRate pass. Therefore the \n"
        "numerical pass count remains zero.",
        "A readable `fu15` table is only an oracle input. A printed-rate match is \n"
        "recorded only after the legacy deck executes and its mapped rate column \n"
        "satisfies the declared tolerance, including the table's rounding interval. \n"
        "It is not complete numerical reproduction until staged intermediate \n"
        "quantities are also validated.",
    )
    markdown_path.write_text(markdown, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--expected-executable", type=int, default=0)
    parser.add_argument("--expected-matches", type=int, default=0)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--case-manifest", type=Path)
    args = parser.parse_args()

    if args.relative_tolerance < 0.0 or not math.isfinite(args.relative_tolerance):
        parser.error("--relative-tolerance must be finite and non-negative")
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    cases = bundle.get("cases")
    if not isinstance(cases, list):
        raise ValueError("core reference bundle has no cases array")
    results = [
        validate_case(case, args.relative_tolerance)
        for case in cases
        if isinstance(case, Mapping)
    ]
    executable = sum(item["execution_status"] == "pass" for item in results)
    matches = sum(item["printed_rate_status"] == "pass" for item in results)
    failed = sum(item["printed_rate_status"] == "fail" for item in results)
    report = {
        "schema": SCHEMA,
        "case_count": len(results),
        "executable_cases": executable,
        "printed_rate_matches": matches,
        "printed_rate_failures": failed,
        "relative_tolerance": args.relative_tolerance,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    if (args.audit_report is None) != (args.case_manifest is None):
        parser.error("--audit-report and --case-manifest must be supplied together")
    if args.audit_report is not None and args.case_manifest is not None:
        _update_audit_outputs(args.audit_report, args.case_manifest, report)

    if executable < args.expected_executable:
        raise SystemExit(
            f"expected at least {args.expected_executable} executable cases, found {executable}"
        )
    if matches < args.expected_matches:
        raise SystemExit(
            f"expected at least {args.expected_matches} printed-rate matches, found {matches}"
        )
    if failed:
        raise SystemExit(f"{failed} executable legacy cases failed numerical comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
