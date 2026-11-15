"""Parsers and comparators for Polyrate fixed-format reference summaries."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?"


@dataclass(frozen=True, slots=True)
class ReferenceRateRow:
    temperature_k: float
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ReferenceRateTable:
    direction: str
    unit: str
    columns: tuple[str, ...]
    rows: tuple[ReferenceRateRow, ...]

    def column(self, name: str) -> tuple[float, ...]:
        try:
            index = tuple(column.upper() for column in self.columns).index(name.upper())
        except ValueError as exc:
            raise KeyError(name) from exc
        return tuple(row.values[index] for row in self.rows)


def _fortran_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


_SUMMARY_PATTERNS = (
    re.compile(
        r"Summary\s+of\s+(forward|reverse)\s+rate\s+constants\s*\((.*?)\)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(Forward|Reverse)\s+TST\s+rate\s+constants\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"FINAL\s+RATE\s+CONSTANTS\s+IVTST-\d+\s*\((.*?)\)",
        re.IGNORECASE,
    ),
)


def _parse_subtable(
    lines: list[str], start_index: int
) -> tuple[tuple[str, ...], list[ReferenceRateRow], int] | None:
    header_index: int | None = None
    for index in range(start_index, min(len(lines), start_index + 20)):
        stripped = lines[index].strip()
        if any(pat.search(lines[index]) for pat in _SUMMARY_PATTERNS) and index > start_index:
            return None
        if "T(K)" in stripped.upper().replace(" ", ""):
            header_index = index
            break
    if header_index is None:
        return None

    header_tokens = lines[header_index].split()
    if not header_tokens or header_tokens[0].upper() != "T(K)":
        return None
    columns = tuple(header_tokens[1:])
    rows: list[ReferenceRateRow] = []
    row_pattern = re.compile(rf"^\s*({_NUMBER})(?:\s+({_NUMBER}))+\s*$")
    next_index = len(lines)
    for offset, line in enumerate(lines[header_index + 1 :], start=header_index + 1):
        stripped = line.strip()
        if not stripped:
            if rows:
                next_index = offset + 1
                break
            continue
        if not row_pattern.match(line):
            if rows:
                next_index = offset
                break
            continue
        tokens = stripped.split()
        if len(tokens) != len(columns) + 1:
            if rows:
                next_index = offset
                break
            continue
        rows.append(
            ReferenceRateRow(
                temperature_k=_fortran_float(tokens[0]),
                values=tuple(_fortran_float(token) for token in tokens[1:]),
            )
        )
    if not rows:
        return None
    return columns, rows, next_index


def parse_fu15_summary(text: str) -> ReferenceRateTable:
    """Parse the first forward/reverse rate table in a Polyrate ``fu15`` file."""

    lines = text.splitlines()
    summary_index: int | None = None
    direction = ""
    unit = ""
    for index, line in enumerate(lines):
        m0 = _SUMMARY_PATTERNS[0].search(line)
        if m0:
            summary_index = index
            direction = m0.group(1).lower()
            unit = m0.group(2).strip()
            break
        m1 = _SUMMARY_PATTERNS[1].search(line)
        if m1:
            summary_index = index
            direction = m1.group(1).lower()
            unit = "cm**3/molecule-1 s-1"
            break
        m2 = _SUMMARY_PATTERNS[2].search(line)
        if m2:
            summary_index = index
            direction = "forward"
            unit = m2.group(1).strip()
            break
    if summary_index is None:
        raise ValueError("no Polyrate rate summary found")

    first = _parse_subtable(lines, summary_index + 1)
    if first is None:
        raise ValueError("rate summary has no valid T(K) subtable")
    columns, rows, next_index = first

    # Polyrate often prints a continuation subtable (e.g. CVT/COMT, CVT/mOMT)
    # immediately below the main table under the same summary header.
    while next_index < len(lines):
        continuation = _parse_subtable(lines, next_index)
        if continuation is None:
            break
        cont_cols, cont_rows, next_index = continuation
        if [r.temperature_k for r in cont_rows] != [r.temperature_k for r in rows]:
            break
        columns = columns + cont_cols
        rows = [
            ReferenceRateRow(
                temperature_k=r1.temperature_k,
                values=r1.values + r2.values,
            )
            for r1, r2 in zip(rows, cont_rows)
        ]

    return ReferenceRateTable(
        direction=direction, unit=unit, columns=columns, rows=tuple(rows)
    )


def load_fu15_summary(path: str | Path) -> ReferenceRateTable:
    return parse_fu15_summary(Path(path).read_text(encoding="utf-8", errors="replace"))


def relative_errors(actual: Iterable[float], expected: Iterable[float]) -> tuple[float, ...]:
    actual_values = tuple(actual)
    expected_values = tuple(expected)
    if len(actual_values) != len(expected_values):
        raise ValueError("actual and expected sequences must have equal length")
    errors: list[float] = []
    for actual_value, expected_value in zip(actual_values, expected_values):
        if expected_value == 0.0:
            errors.append(abs(actual_value - expected_value))
        else:
            errors.append(abs(actual_value / expected_value - 1.0))
    return tuple(errors)


def assert_rates_close(
    actual: Iterable[float],
    expected: Iterable[float],
    *,
    relative_tolerance: float,
) -> None:
    errors = relative_errors(actual, expected)
    maximum = max(errors, default=0.0)
    if not math.isfinite(maximum) or maximum > relative_tolerance:
        raise AssertionError(
            f"maximum relative rate error {maximum:.6g} exceeds {relative_tolerance:.6g}"
        )
