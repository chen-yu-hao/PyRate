"""Parsers and comparators for Polyrate fixed-format reference summaries."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?"
_TABLE_START = re.compile(
    r"^\s*(?:"
    r"Summary\s+of\s+(?:forward|reverse)\s+rate\s+constants\s*\("
    r"|FINAL\s+RATE\s+CONSTANTS(?:\s+IVTST-\d+)?\s*\("
    r"|(?:Forward|Reverse)\s+TST\s+rate\s+constants\s*:"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
_SUMMARY_HEADER = re.compile(
    r"Summary\s+of\s+(forward|reverse)\s+rate\s+constants\s*\((.*?)\)",
    re.IGNORECASE,
)
_FINAL_HEADER = re.compile(
    r"FINAL\s+RATE\s+CONSTANTS(?:\s+IVTST-\d+)?\s*\((.*?)\)",
    re.IGNORECASE,
)
_DIRECTIONAL_TST_HEADER = re.compile(
    r"\b(Forward|Reverse)\s+TST\s+rate\s+constants\s*:",
    re.IGNORECASE,
)


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


def _table_metadata(lines: list[str]) -> tuple[int, str, str]:
    """Return header line, direction and unit for the first supported rate table."""

    for index, line in enumerate(lines):
        summary = _SUMMARY_HEADER.search(line)
        if summary:
            return index, summary.group(1).lower(), summary.group(2).strip()

        final = _FINAL_HEADER.search(line)
        if final:
            # In Polyrate's compact legacy output, unqualified FINAL RATE
            # CONSTANTS blocks are the forward-direction result tables.
            return index, "forward", final.group(1).strip()

        directional_tst = _DIRECTIONAL_TST_HEADER.search(line)
        if directional_tst:
            # These compact TST-only blocks omit the unit entirely. Preserve
            # that fact so comparison code must obtain a unit from input
            # context rather than silently guessing reaction order.
            return index, directional_tst.group(1).lower(), "unspecified"

    raise ValueError("no Polyrate rate table found")


def parse_fu15_summary(text: str) -> ReferenceRateTable:
    """Parse the first supported rate table in a Polyrate ``fu15`` file.

    Supported layouts are the standard forward/reverse ``Summary`` block, the
    compact ``FINAL RATE CONSTANTS``/``IVTST-n`` block, and the compact
    ``Forward TST rate constants`` block. The last layout has no printed unit,
    so its returned unit is ``"unspecified"`` and numerical comparison remains
    deliberately blocked until the caller supplies reaction-order context.
    """

    lines = text.splitlines()
    summary_index, direction, unit = _table_metadata(lines)

    header_index: int | None = None
    for index in range(summary_index + 1, min(len(lines), summary_index + 20)):
        if "T(K)" in lines[index].upper().replace(" ", ""):
            header_index = index
            break
    if header_index is None:
        raise ValueError("rate table has no T(K) header")

    header_tokens = lines[header_index].split()
    if not header_tokens or header_tokens[0].upper() != "T(K)":
        raise ValueError("unrecognized rate-table header")
    columns = tuple(header_tokens[1:])
    if not columns:
        raise ValueError("rate table has no method columns")

    rows: list[ReferenceRateRow] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        tokens = stripped.split()
        if len(tokens) != len(columns) + 1:
            if rows:
                break
            continue
        if not all(re.fullmatch(_NUMBER, token) for token in tokens):
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
        raise ValueError("rate table contains no numeric rows")
    return ReferenceRateTable(
        direction=direction, unit=unit, columns=columns, rows=tuple(rows)
    )


def parse_fu15_tables(text: str) -> tuple[ReferenceRateTable, ...]:
    """Parse every supported rate table while preserving block order."""

    starts = [match.start() for match in _TABLE_START.finditer(text)]
    if not starts:
        raise ValueError("no Polyrate rate table found")
    return tuple(
        parse_fu15_summary(text[start:stop])
        for start, stop in zip(starts, starts[1:] + [len(text)])
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
