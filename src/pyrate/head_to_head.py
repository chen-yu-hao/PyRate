"""Conservative, unit-aware Polyrate `fu15` vs PyRate JSON comparisons.

This is a *result comparator*, not a legacy input translator. Only named,
explicitly mapped columns are tested; missing values never count as passing.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .constants import AVOGADRO_MOL
from .reference import ReferenceRateTable, parse_fu15_summary


_SUMMARY = re.compile(
    r"(?:^\s*Summary\s+of\s+(?:forward|reverse)\s+rate\s+constants\s*\("
    r"|^\s*(?:Forward|Reverse)\s+TST\s+rate\s+constants\s*:"
    r"|FINAL\s+RATE\s+CONSTANTS\s+IVTST-\d+\s*\()",
    flags=re.IGNORECASE | re.MULTILINE,
)
# Convert to m^3 mol^-1 s^-1 (or to s^-1 for first-order rates).
_UNIT_FACTORS: dict[str, tuple[str, float]] = {
    "s-1": ("first-order", 1.0),
    "sec-1": ("first-order", 1.0),
    "1s": ("first-order", 1.0),
    "1sec": ("first-order", 1.0),
    "m3mol-1s-1": ("second-order", 1.0),
    "cm3mol-1s-1": ("second-order", 1.0e-6),
    "lmol-1s-1": ("second-order", 1.0e-3),
    "cm3molecule-1s-1": ("second-order", 1.0e-6 * AVOGADRO_MOL),
    "cm3molecule-1sec-1": ("second-order", 1.0e-6 * AVOGADRO_MOL),
    "cm3moleculesec": ("second-order", 1.0e-6 * AVOGADRO_MOL),
    "cm3molecule-sec": ("second-order", 1.0e-6 * AVOGADRO_MOL),
}
DEFAULT_COLUMN_MAP: Mapping[str, str] = {"TST": "k_tst", "CVT": "k_cvt"}
EXTENDED_COLUMN_MAP: Mapping[str, str] = {
    "TST": "k_tst",
    "CVT": "k_cvt",
    "CUS": "k_cus",
    "MUVT": "k_muvt",
    "MCVT": "k_muvt",
    "US": "k_us",
}
_ALLOWED_FIELDS = frozenset(
    {
        "k_tst",
        "k_cvt",
        "k_cus",
        "k_muvt",
        "k_us",
        "k_final",
        "k_reverse_final",
    }
)


def parse_fu15_tables(text: str) -> tuple[ReferenceRateTable, ...]:
    """Find every forward/reverse summary table without losing later blocks."""

    starts = [match.start() for match in _SUMMARY.finditer(text)]
    if not starts:
        raise ValueError("no Polyrate rate summary found")
    tables = tuple(
        parse_fu15_summary(text[start:stop])
        for start, stop in zip(starts, starts[1:] + [len(text)])
    )
    return tables


def _unit_factor(unit: str) -> tuple[str, float]:
    # Polyrate uses `cm**3/molecule-1 s-1`; PyRate uses
    # `cm^3 molecule^-1 s^-1`. Canonicalize both spellings.
    key = re.sub(r"\s+", "", unit.lower())
    key = key.replace("**", "^").replace("³", "^3")
    key = key.replace("^", "").replace("/", "").replace("·", "")
    try:
        return _UNIT_FACTORS[key]
    except KeyError as exc:
        raise ValueError(f"unrecognized rate unit {unit!r}; refusing comparison") from exc


@dataclass(frozen=True, slots=True)
class ComparisonPoint:
    reference_column: str
    pyrate_field: str | None
    temperature_k: float
    status: str
    reference_value: float
    pyrate_value: float | None = None
    relative_error: float | None = None


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    reference_direction: str
    reference_unit: str
    relative_tolerance: float
    compared: int
    passed: int
    failed: int
    unavailable: int
    unmapped: int
    success: bool
    points: tuple[ComparisonPoint, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["points"] = [asdict(point) for point in self.points]
        return payload


def compare_fu15_to_pyrate(
    reference: ReferenceRateTable,
    pyrate_result: Mapping[str, Any],
    *,
    column_map: Mapping[str, str] = DEFAULT_COLUMN_MAP,
    relative_tolerance: float = 5.0e-3,
    temperature_tolerance_k: float = 1.0e-3,
    strict: bool = False,
) -> ComparisonReport:
    """Compare rate columns by matching temperature and physical units.

    `strict` requires every reference column to be mapped and available.
    Even in non-strict mode at least one point must be compared, and any
    mapped-but-unavailable point makes the report unsuccessful.

    `CVT/ZCT` is deliberately *not* mapped by default: the current one-
    dimensional WKB kernel is not a validated Polyrate ZCT implementation.
    """

    if relative_tolerance < 0 or not math.isfinite(relative_tolerance):
        raise ValueError("relative_tolerance must be finite and non-negative")
    if temperature_tolerance_k < 0 or not math.isfinite(temperature_tolerance_k):
        raise ValueError("temperature_tolerance_k must be finite and non-negative")
    reference_kind, reference_factor = _unit_factor(reference.unit)
    rows = pyrate_result.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("PyRate JSON must contain a non-empty rows array")
    normalized_mapping = {key.upper(): value for key, value in column_map.items()}
    invalid_fields = sorted(set(normalized_mapping.values()) - _ALLOWED_FIELDS)
    if invalid_fields:
        raise ValueError(f"unsupported PyRate fields in column_map: {invalid_fields}")
    by_temperature: list[tuple[float, Mapping[str, Any], float]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("PyRate rate rows must be objects")
        temp = float(row["temperature_k"])
        kind, factor = _unit_factor(str(row["rate_unit"]))
        if kind != reference_kind:
            raise ValueError("first-order and second-order rate units cannot be compared")
        if not math.isfinite(temp) or temp <= 0:
            raise ValueError("invalid PyRate temperature")
        if any(abs(temp - old_temp) <= temperature_tolerance_k for old_temp, _, _ in by_temperature):
            raise ValueError("ambiguous duplicate PyRate temperatures")
        by_temperature.append((temp, row, factor))

    points: list[ComparisonPoint] = []
    for column_index, column in enumerate(reference.columns):
        field = normalized_mapping.get(column.upper())
        for reference_row in reference.rows:
            temperature = reference_row.temperature_k
            expected = reference_row.values[column_index]
            if field is None:
                points.append(ComparisonPoint(column, None, temperature, "unmapped", expected))
                continue
            candidates = [
                (row, factor)
                for temp, row, factor in by_temperature
                if abs(temp - temperature) <= temperature_tolerance_k
            ]
            if len(candidates) != 1:
                points.append(ComparisonPoint(column, field, temperature, "missing", expected))
                continue
            row, factor = candidates[0]
            value = row.get(field)
            if value is None:
                points.append(ComparisonPoint(column, field, temperature, "unavailable", expected))
                continue
            actual = float(value)
            expected_si = expected * reference_factor
            actual_si = actual * factor
            if not math.isfinite(actual_si) or not math.isfinite(expected_si):
                relative_error = None
            elif expected_si == 0:
                relative_error = 0.0 if actual_si == 0 else None
            else:
                relative_error = abs(actual_si / expected_si - 1.0)
            status = (
                "pass" if relative_error is not None and relative_error <= relative_tolerance
                else "fail"
            )
            points.append(ComparisonPoint(
                column, field, temperature, status, expected,
                actual if math.isfinite(actual) else None, relative_error,
            ))

    passed = sum(point.status == "pass" for point in points)
    failed = sum(point.status == "fail" for point in points)
    unavailable = sum(point.status in {"missing", "unavailable"} for point in points)
    unmapped = sum(point.status == "unmapped" for point in points)
    compared = passed + failed
    return ComparisonReport(
        reference_direction=reference.direction,
        reference_unit=reference.unit,
        relative_tolerance=relative_tolerance,
        compared=compared,
        passed=passed,
        failed=failed,
        unavailable=unavailable,
        unmapped=unmapped,
        success=compared > 0 and failed == 0 and unavailable == 0 and (not strict or unmapped == 0),
        points=tuple(points),
    )
