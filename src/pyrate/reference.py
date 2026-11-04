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


def parse_fu15_summary(text: str) -> ReferenceRateTable:
    """Parse the first forward/reverse rate table in a Polyrate ``fu15`` file."""

    lines = text.splitlines()
    summary_index: int | None = None
    direction = ""
    unit = ""
    summary_pattern = re.compile(
        r"Summary\s+of\s+(forward|reverse)\s+rate\s+constants\s*\((.*?)\)",
        re.IGNORECASE,
    )
    for index, line in enumerate(lines):
        match = summary_pattern.search(line)
        if match:
            summary_index = index
            direction = match.group(1).lower()
            unit = match.group(2).strip()
            break
    if summary_index is None:
        raise ValueError("no Polyrate rate summary found")

    header_index: int | None = None
    for index in range(summary_index + 1, min(len(lines), summary_index + 20)):
        if "T(K)" in lines[index].upper().replace(" ", ""):
            header_index = index
            break
    if header_index is None:
        raise ValueError("rate summary has no T(K) header")

    header_tokens = lines[header_index].split()
    if not header_tokens or header_tokens[0].upper() != "T(K)":
        raise ValueError("unrecognized rate-summary header")
    columns = tuple(header_tokens[1:])
    rows: list[ReferenceRateRow] = []
    row_pattern = re.compile(rf"^\s*({_NUMBER})(?:\s+({_NUMBER}))+\s*$")
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if not row_pattern.match(line):
            if rows:
                break
            continue
        tokens = stripped.split()
        if len(tokens) != len(columns) + 1:
            if rows:
                break
            continue
        rows.append(
            ReferenceRateRow(
                temperature_k=_fortran_float(tokens[0]),
                values=tuple(_fortran_float(token) for token in tokens[1:]),
            )
        )
    if not rows:
        raise ValueError("rate summary contains no numeric rows")
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
