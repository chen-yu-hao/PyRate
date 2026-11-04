"""Readers for Polyrate fixed unit files 30, 31, and 40.

Polyrate uses three related interchange formats for stationary-point and
reaction-path data.  Unit 30 is positional, while units 31 and 40 use section
and keyword syntax.  The readers in this module intentionally separate two
jobs:

* lossless-enough parsing into typed sections and entries; and
* conversion of the common energetic, frequency, Hessian, geometry, and path
  fields into a format-independent model.

Unsupported/unknown keywords are retained rather than discarded.  This is
important for a scientific compatibility layer: successful parsing must not be
mistaken for executable semantics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .constants import HARTREE_KJ_MOL, WAVENUMBER_TO_KJ_MOL

Scalar = int | float | str

_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[EeDd][+-]?\d+)?$"
)
_UNIT_FROM_SUFFIX = {".fu30": 30, ".fu31": 31, ".fu40": 40}


def _parse_token(token: str) -> Scalar:
    token = token.strip()
    if token.upper() in {"NAN", "+NAN", "-NAN", "INF", "+INF", "-INF", "INFINITY", "+INFINITY", "-INFINITY"}:
        raise ValueError(f"non-finite number in fixed-unit input: {token!r}")
    if _INTEGER_RE.fullmatch(token):
        return int(token)
    if _NUMBER_RE.fullmatch(token):
        value = float(token.replace("D", "E").replace("d", "e"))
        if not math.isfinite(value):
            raise ValueError(f"non-finite number in fixed-unit input: {token!r}")
        return value
    return token


def _parse_row(line: str) -> tuple[Scalar, ...]:
    return tuple(_parse_token(token) for token in line.replace(",", " ").split())


def _numeric_values(rows: Iterable[Sequence[Scalar]]) -> tuple[float, ...]:
    values: list[float] = []
    for row in rows:
        for value in row:
            if isinstance(value, str):
                raise ValueError(f"expected numeric fixed-unit value, found {value!r}")
            values.append(float(value))
    return tuple(values)


def _integer_values(rows: Iterable[Sequence[Scalar]]) -> tuple[int, ...]:
    values: list[int] = []
    for row in rows:
        for value in row:
            if isinstance(value, str) or not float(value).is_integer():
                raise ValueError(f"expected integer fixed-unit value, found {value!r}")
            values.append(int(value))
    return tuple(values)


def _strip_comment(line: str) -> tuple[str, str]:
    body, marker, comment = line.partition("#")
    return body.rstrip(), comment.strip() if marker else ""


def _nonempty(lines: Sequence[str], index: int) -> int:
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def _is_numeric_line(line: str) -> bool:
    tokens = line.replace(",", " ").split()
    return bool(tokens) and all(_NUMBER_RE.fullmatch(token) for token in tokens)


@dataclass(frozen=True, slots=True)
class FixedUnitEntry:
    """One scalar keyword or terminated numeric/text block."""

    keyword: str
    values: tuple[Scalar, ...] = ()
    rows: tuple[tuple[Scalar, ...], ...] = ()
    source_line: int = 0
    comment: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "keyword", self.keyword.upper())
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "rows", tuple(tuple(row) for row in self.rows))

    @property
    def is_block(self) -> bool:
        return bool(self.rows)

    def flattened_numeric(self) -> tuple[float, ...]:
        rows: tuple[Sequence[Scalar], ...]
        if self.rows:
            rows = self.rows
        else:
            rows = (self.values,)
        return _numeric_values(rows)

    def flattened_integers(self) -> tuple[int, ...]:
        rows: tuple[Sequence[Scalar], ...]
        if self.rows:
            rows = self.rows
        else:
            rows = (self.values,)
        return _integer_values(rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "values": list(self.values),
            "rows": [list(row) for row in self.rows],
            "source_line": self.source_line,
            "comment": self.comment,
        }


@dataclass(frozen=True, slots=True)
class FixedUnitSection:
    """A named section; names may repeat for reaction-path points."""

    name: str
    entries: tuple[FixedUnitEntry, ...] = ()
    source_line: int = 0
    comment: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", self.name.upper())
        object.__setattr__(self, "entries", tuple(self.entries))

    def find(self, keyword: str) -> tuple[FixedUnitEntry, ...]:
        target = keyword.upper()
        return tuple(entry for entry in self.entries if entry.keyword == target)

    def first(self, keyword: str) -> FixedUnitEntry | None:
        matches = self.find(keyword)
        return matches[0] if matches else None

    def scalar(self, keyword: str, default: Scalar | None = None) -> Scalar | None:
        entry = self.first(keyword)
        if entry is None or not entry.values:
            return default
        return entry.values[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_line": self.source_line,
            "comment": self.comment,
            "entries": [entry.as_dict() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class FixedUnitStationaryPoint:
    """Format-independent stationary-point data available in a fixed-unit file."""

    role: str
    source_section: str
    coordinate_au: float | None = None
    potential_hartree: float | None = None
    active_mode_indices: tuple[int, ...] = ()
    frequencies: tuple[float, ...] = ()
    frequency_unit: str | None = None
    imaginary_frequency: float | None = None
    hessian_values: tuple[float, ...] = ()
    hessian_form: str | None = None
    hessian_unit: str | None = None

    @property
    def potential_kj_mol(self) -> float | None:
        if self.potential_hartree is None:
            return None
        return self.potential_hartree * HARTREE_KJ_MOL

    @property
    def frequencies_cm1(self) -> tuple[float, ...]:
        unit = (self.frequency_unit or "").lower()
        if not self.frequencies:
            return ()
        if unit in {"waven", "wavenumber", "cm-1", "cm^-1"}:
            return self.frequencies
        if unit in {"au", "hartree"}:
            factor = HARTREE_KJ_MOL / WAVENUMBER_TO_KJ_MOL
            return tuple(value * factor for value in self.frequencies)
        raise ValueError(
            f"cannot convert frequencies for {self.role}: unknown unit {self.frequency_unit!r}"
        )

    @property
    def imaginary_frequency_cm1(self) -> float | None:
        if self.imaginary_frequency is None:
            return None
        unit = (self.frequency_unit or "").lower()
        if unit in {"waven", "wavenumber", "cm-1", "cm^-1"}:
            return self.imaginary_frequency
        if unit in {"au", "hartree"}:
            return self.imaginary_frequency * HARTREE_KJ_MOL / WAVENUMBER_TO_KJ_MOL
        raise ValueError(
            f"cannot convert imaginary frequency for {self.role}: "
            f"unknown unit {self.frequency_unit!r}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source_section": self.source_section,
            "coordinate_au": self.coordinate_au,
            "potential_hartree": self.potential_hartree,
            "potential_kj_mol": self.potential_kj_mol,
            "active_mode_indices": list(self.active_mode_indices),
            "frequencies": list(self.frequencies),
            "frequency_unit": self.frequency_unit,
            "frequencies_cm1": list(self.frequencies_cm1) if self.frequencies else [],
            "imaginary_frequency": self.imaginary_frequency,
            "imaginary_frequency_cm1": (
                self.imaginary_frequency_cm1
                if self.imaginary_frequency is not None
                else None
            ),
            "hessian_values": list(self.hessian_values),
            "hessian_form": self.hessian_form,
            "hessian_unit": self.hessian_unit,
        }


@dataclass(frozen=True, slots=True)
class FixedUnitPathPoint:
    """Common fields for a reaction-path point."""

    coordinate_au: float
    potential_hartree: float
    geometry_rows: tuple[tuple[float, ...], ...] = ()
    gradient_rows: tuple[tuple[float, ...], ...] = ()
    active_mode_indices: tuple[int, ...] = ()
    hessian_values: tuple[float, ...] = ()
    source_section: str = ""

    @property
    def potential_kj_mol(self) -> float:
        return self.potential_hartree * HARTREE_KJ_MOL

    def as_dict(self) -> dict[str, Any]:
        return {
            "coordinate_au": self.coordinate_au,
            "potential_hartree": self.potential_hartree,
            "potential_kj_mol": self.potential_kj_mol,
            "geometry_rows": [list(row) for row in self.geometry_rows],
            "gradient_rows": [list(row) for row in self.gradient_rows],
            "active_mode_indices": list(self.active_mode_indices),
            "hessian_values": list(self.hessian_values),
            "source_section": self.source_section,
        }


@dataclass(frozen=True, slots=True)
class FixedUnitDocument:
    """Parsed fixed-unit input with common conversion helpers."""

    unit: int
    source_name: str
    title: str
    sections: tuple[FixedUnitSection, ...]
    header_flags: tuple[int, ...] = ()
    control_flags: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.unit not in {30, 31, 40}:
            raise ValueError("fixed-unit document must be unit 30, 31, or 40")
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "header_flags", tuple(self.header_flags))
        object.__setattr__(self, "control_flags", tuple(self.control_flags))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def sections_named(self, name: str) -> tuple[FixedUnitSection, ...]:
        target = name.upper()
        return tuple(section for section in self.sections if section.name == target)

    @property
    def section_names(self) -> tuple[str, ...]:
        return tuple(section.name for section in self.sections)

    def _general(self) -> FixedUnitSection | None:
        names = ("GEN40", "GENERAL", "HEADER30")
        for name in names:
            matches = self.sections_named(name)
            if matches:
                return matches[0]
        return None

    def _general_scalar(self, keyword: str) -> str | None:
        section = self._general()
        if section is None:
            return None
        value = section.scalar(keyword)
        return None if value is None else str(value)

    @property
    def frequency_unit(self) -> str | None:
        if self.unit == 30:
            return "au"
        value = self._general_scalar("FREQUNIT")
        return value.lower() if value is not None else None

    @property
    def hessian_form(self) -> str | None:
        value = self._general_scalar("HESSFORM")
        if value is not None:
            return value.lower()
        if self.unit == 30:
            return str(self.metadata.get("hessian_form")) if self.metadata.get("hessian_form") else None
        return "packed" if self.unit == 31 else None

    @property
    def hessian_unit(self) -> str | None:
        value = self._general_scalar("HESSUNIT")
        if value is not None:
            return value.lower()
        return str(self.metadata.get("hessian_unit")) if self.metadata.get("hessian_unit") else None

    def stationary_points(self) -> tuple[FixedUnitStationaryPoint, ...]:
        roles = {
            "R140": "reactant1",
            "R240": "reactant2",
            "P140": "product1",
            "P240": "product2",
            "SADDLE40": "saddle",
            "REACT1": "reactant1",
            "REACT2": "reactant2",
            "PROD1": "product1",
            "PROD2": "product2",
            "SADDLE": "saddle",
        }
        result: list[FixedUnitStationaryPoint] = []
        for section in self.sections:
            role = roles.get(section.name)
            if role is None:
                continue
            coordinate: float | None = None
            potential: float | None = None
            sv = section.first("S_V")
            if sv is not None and len(sv.values) >= 2:
                coordinate = float(sv.values[0])
                potential = float(sv.values[1])
            else:
                value = section.scalar("VVALUE")
                if value is None:
                    if role == "saddle":
                        value = section.scalar("ENESAD")
                    elif role.startswith("product"):
                        value = section.scalar("ENERXN")
                if value is not None and not isinstance(value, str):
                    potential = float(value)

            ifreq = section.first("IFREQ")
            indices = ifreq.flattened_integers() if ifreq is not None else ()
            freq = section.first("FREQ")
            frequencies = freq.flattened_numeric() if freq is not None else ()
            wstar_entry = section.first("WSTAR")
            imaginary: float | None = None
            if wstar_entry is not None:
                if wstar_entry.values and not isinstance(wstar_entry.values[0], str):
                    imaginary = float(wstar_entry.values[0])
                elif wstar_entry.rows:
                    values = wstar_entry.flattened_numeric()
                    imaginary = values[0] if values else None
            hessian = section.first("HESSIAN")
            hessian_values = hessian.flattened_numeric() if hessian is not None else ()
            result.append(
                FixedUnitStationaryPoint(
                    role=role,
                    source_section=section.name,
                    coordinate_au=coordinate,
                    potential_hartree=potential,
                    active_mode_indices=indices,
                    frequencies=frequencies,
                    frequency_unit=self.frequency_unit,
                    imaginary_frequency=imaginary,
                    hessian_values=hessian_values,
                    hessian_form=self.hessian_form,
                    hessian_unit=self.hessian_unit,
                )
            )
        return tuple(result)

    def path_points(self) -> tuple[FixedUnitPathPoint, ...]:
        result: list[FixedUnitPathPoint] = []
        for section in self.sections:
            if section.name not in {"POINT30", "POINT40"}:
                continue
            if section.name == "POINT40":
                coordinate_value = section.scalar("SMEP")
                potential_value = section.scalar("VMEP")
            else:
                sv = section.first("S_V")
                if sv is None or len(sv.values) < 2:
                    continue
                coordinate_value, potential_value = sv.values[:2]
            if (
                coordinate_value is None
                or potential_value is None
                or isinstance(coordinate_value, str)
                or isinstance(potential_value, str)
            ):
                continue
            geom = section.first("GEOM")
            grads = section.first("GRADS")
            ifreq = section.first("IFREQ")
            hessian = section.first("HESSIAN")
            result.append(
                FixedUnitPathPoint(
                    coordinate_au=float(coordinate_value),
                    potential_hartree=float(potential_value),
                    geometry_rows=(
                        tuple(tuple(float(value) for value in row) for row in geom.rows)
                        if geom is not None
                        else ()
                    ),
                    gradient_rows=(
                        tuple(tuple(float(value) for value in row) for row in grads.rows)
                        if grads is not None
                        else ()
                    ),
                    active_mode_indices=(
                        ifreq.flattened_integers() if ifreq is not None else ()
                    ),
                    hessian_values=(
                        hessian.flattened_numeric() if hessian is not None else ()
                    ),
                    source_section=section.name,
                )
            )
        return tuple(result)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "source_name": self.source_name,
            "title": self.title,
            "header_flags": list(self.header_flags),
            "control_flags": list(self.control_flags),
            "metadata": dict(self.metadata),
            "sections": [section.as_dict() for section in self.sections],
            "stationary_points": [point.as_dict() for point in self.stationary_points()],
            "path_points": [point.as_dict() for point in self.path_points()],
        }


def _entry(
    keyword: str,
    *,
    values: Sequence[Scalar] = (),
    rows: Sequence[Sequence[Scalar]] = (),
    source_line: int,
    comment: str = "",
) -> FixedUnitEntry:
    return FixedUnitEntry(
        keyword=keyword,
        values=tuple(values),
        rows=tuple(tuple(row) for row in rows),
        source_line=source_line,
        comment=comment,
    )


class _SectionBuilder:
    __slots__ = ("name", "source_line", "comment", "entries")

    def __init__(self, name: str, source_line: int, comment: str = "") -> None:
        self.name = name
        self.source_line = source_line
        self.comment = comment
        self.entries: list[FixedUnitEntry] = []

    def freeze(self) -> FixedUnitSection:
        return FixedUnitSection(
            name=self.name,
            entries=tuple(self.entries),
            source_line=self.source_line,
            comment=self.comment,
        )


def _parse_keyword_unit(text: str, unit: int, source_name: str) -> FixedUnitDocument:
    lines = text.splitlines()
    block_keywords = (
        {"HESSIAN"}
        if unit == 31
        else {
            "TITLE",
            "FREQ",
            "IFREQ",
            "HESSIAN",
            "GEOM",
            "GRADS",
            "GRADSB1",
            "GRADSB2",
        }
    )
    sections: list[_SectionBuilder] = []
    current: _SectionBuilder | None = None
    index = 0
    while index < len(lines):
        raw = lines[index]
        body, comment = _strip_comment(raw)
        stripped = body.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("*"):
            name = stripped[1:].split()[0]
            current = _SectionBuilder(name, index + 1, comment)
            sections.append(current)
            index += 1
            continue
        if current is None:
            current = _SectionBuilder("GENERAL", index + 1)
            sections.append(current)

        tokens = stripped.split()
        keyword = tokens[0].upper()
        if keyword in block_keywords and len(tokens) == 1:
            block_line = index + 1
            rows: list[tuple[Scalar, ...]] = []
            index += 1
            while index < len(lines):
                block_body, _ = _strip_comment(lines[index])
                block_stripped = block_body.strip()
                if block_stripped.upper() == "END":
                    break
                if block_stripped:
                    if keyword == "TITLE":
                        rows.append((block_stripped,))
                    else:
                        rows.append(_parse_row(block_stripped))
                index += 1
            if index >= len(lines):
                raise ValueError(
                    f"unterminated {keyword} block in {source_name} at line {index + 1}"
                )
            current.entries.append(
                _entry(
                    keyword,
                    rows=rows,
                    source_line=block_line,
                    comment=comment,
                )
            )
            index += 1
            continue

        current.entries.append(
            _entry(
                keyword,
                values=tuple(_parse_token(token) for token in tokens[1:]),
                source_line=index + 1,
                comment=comment,
            )
        )
        index += 1

    frozen = tuple(section.freeze() for section in sections)
    title = ""
    for section in frozen:
        title_entry = section.first("TITLE")
        if title_entry is not None:
            title = " ".join(str(row[0]) for row in title_entry.rows)
            break
    return FixedUnitDocument(
        unit=unit,
        source_name=source_name,
        title=title,
        sections=frozen,
    )


def _unit30_role(line: str) -> str | None:
    upper = line.upper()
    iop = re.search(r"\bIOP\s*=\s*([1-4])\s*,\s*S\s*,\s*V\s*=", upper)
    if iop:
        return {"1": "REACT1", "2": "REACT2", "3": "PROD1", "4": "PROD2"}[
            iop.group(1)
        ]
    if upper.startswith("S AND V"):
        match = re.search(r"\b(REACTANT|PRODUCT)\s*#?\s*([12])\b", upper)
        if match:
            prefix = "REACT" if match.group(1) == "REACTANT" else "PROD"
            return prefix + match.group(2)
    if upper.startswith("SADDLE POINT") and "IMAGINARY" not in upper:
        return "SADDLE"
    return None


def _collect_numeric_rows(
    lines: Sequence[str], index: int
) -> tuple[list[tuple[Scalar, ...]], int]:
    rows: list[tuple[Scalar, ...]] = []
    index = _nonempty(lines, index)
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if not _is_numeric_line(stripped):
            break
        rows.append(_parse_row(stripped))
        index += 1
    return rows, index


def _parse_unit30(text: str, source_name: str) -> FixedUnitDocument:
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError(f"unit 30 file {source_name} is shorter than its three-line header")
    title = lines[0].strip()
    header_flags = _integer_values((_parse_row(lines[1]),))
    control_flags = _integer_values((_parse_row(lines[2]),))
    if len(header_flags) != 8:
        raise ValueError(
            f"unit 30 file {source_name} requires 8 header flags, found {len(header_flags)}"
        )
    if len(control_flags) != 32:
        raise ValueError(
            f"unit 30 file {source_name} requires 32 control flags, found {len(control_flags)}"
        )

    builders: list[_SectionBuilder] = []
    header = _SectionBuilder("HEADER30", 1)
    header.entries.extend(
        (
            _entry("TITLE", values=(title,), source_line=1),
            _entry("HEADER_FLAGS", values=header_flags, source_line=2),
            _entry("CONTROL_FLAGS", values=control_flags, source_line=3),
        )
    )
    builders.append(header)
    current: _SectionBuilder | None = None
    in_path = False
    index = 3

    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        upper = stripped.upper()

        if upper.startswith("BEGIN POINTS ALONG REACTION COORDINATE"):
            in_path = True
            path = _SectionBuilder("PATH30", index + 1)
            builders.append(path)
            next_index = _nonempty(lines, index + 1)
            if next_index < len(lines) and _is_numeric_line(lines[next_index].strip()):
                values = _parse_row(lines[next_index].strip())
                path.entries.append(_entry("NPTS", values=values, source_line=next_index + 1))
                index = next_index + 1
            else:
                index += 1
            current = path
            continue

        if in_path and upper.replace(" ", "") == "S,V=":
            current = _SectionBuilder("POINT30", index + 1)
            builders.append(current)
            rows, next_index = _collect_numeric_rows(lines, index + 1)
            if not rows:
                raise ValueError(f"missing S,V values in {source_name} at line {index + 1}")
            current.entries.append(
                _entry("S_V", values=rows[0], source_line=index + 2)
            )
            index = next_index
            continue

        role = _unit30_role(stripped)
        if role is not None:
            current = _SectionBuilder(role, index + 1)
            builders.append(current)
            rows, next_index = _collect_numeric_rows(lines, index + 1)
            if not rows:
                raise ValueError(
                    f"missing S,V values after {role} in {source_name} at line {index + 1}"
                )
            current.entries.append(
                _entry("S_V", values=rows[0], source_line=index + 2)
            )
            if len(rows) > 1:
                current.entries.append(
                    _entry("IFREQ", rows=(rows[1],), source_line=index + 3)
                )
                if len(rows) > 2:
                    current.entries.append(
                        _entry("UNLABELED", rows=rows[2:], source_line=index + 4)
                    )
            index = next_index
            continue

        if current is None:
            current = header

        if upper.startswith("SADDLE POINT IMAGINARY FREQUENCY"):
            rows, next_index = _collect_numeric_rows(lines, index + 1)
            current.entries.append(
                _entry("WSTAR", rows=rows, source_line=index + 1)
            )
            index = next_index
            continue
        if upper.startswith("FRECUENC") or upper.startswith("FREQUENC"):
            rows, next_index = _collect_numeric_rows(lines, index + 1)
            current.entries.append(
                _entry("FREQ", rows=rows, source_line=index + 1)
            )
            index = next_index
            continue
        if "HESSIAN" in upper or upper.startswith("F MATRIX"):
            rows, next_index = _collect_numeric_rows(lines, index + 1)
            current.entries.append(
                _entry("HESSIAN", rows=rows, source_line=index + 1)
            )
            index = next_index
            continue
        if upper == "X=":
            rows, next_index = _collect_numeric_rows(lines, index + 1)
            atom_count = header_flags[0]
            current.entries.append(
                _entry("GEOM", rows=rows[:atom_count], source_line=index + 1)
            )
            if len(rows) > atom_count:
                current.entries.append(
                    _entry("UNLABELED", rows=rows[atom_count:], source_line=index + 1 + atom_count)
                )
            index = next_index
            continue
        if upper == "DX=":
            rows, next_index = _collect_numeric_rows(lines, index + 1)
            atom_count = header_flags[0]
            current.entries.append(
                _entry("GRADS", rows=rows[:atom_count], source_line=index + 1)
            )
            if len(rows) > atom_count:
                current.entries.append(
                    _entry("IFREQ", rows=(rows[atom_count],), source_line=index + 2 + atom_count)
                )
            if len(rows) > atom_count + 1:
                current.entries.append(
                    _entry("UNLABELED", rows=rows[atom_count + 1 :], source_line=index + 3 + atom_count)
                )
            index = next_index
            continue
        if upper.replace(" ", "") == "S,DX=":
            rows, next_index = _collect_numeric_rows(lines, index + 1)
            current.entries.append(
                _entry("GRAD_SAMPLE", rows=rows, source_line=index + 1)
            )
            index = next_index
            continue

        if not _is_numeric_line(stripped):
            current.entries.append(
                _entry("NOTE", values=(stripped,), source_line=index + 1)
            )
            index += 1
            continue

        rows, next_index = _collect_numeric_rows(lines, index)
        current.entries.append(
            _entry("UNLABELED", rows=rows, source_line=index + 1)
        )
        index = next_index

    metadata: dict[str, Any] = {}
    hessian_notes = " ".join(
        str(value)
        for builder in builders
        for entry in builder.entries
        if entry.keyword == "NOTE"
        for value in entry.values
    ).upper()
    if "PACKED" in hessian_notes:
        metadata["hessian_form"] = "packed"
    elif "UNSCALED HESSIAN" in hessian_notes:
        metadata["hessian_form"] = "full_or_packed_legacy"
    if "MASS SCALED" in title.upper() or " F SCALED" in title.upper():
        metadata["hessian_unit"] = "mass_scaled_atomic"
    elif "HESSIAN" in title.upper():
        metadata["hessian_unit"] = "atomic"

    return FixedUnitDocument(
        unit=30,
        source_name=source_name,
        title=title,
        sections=tuple(builder.freeze() for builder in builders),
        header_flags=header_flags,
        control_flags=control_flags,
        metadata=metadata,
    )


def parse_fixed_unit_text(
    text: str,
    unit: int,
    *,
    source_name: str = "<memory>",
) -> FixedUnitDocument:
    """Parse text from Polyrate fixed unit 30, 31, or 40."""

    if unit == 30:
        return _parse_unit30(text, source_name)
    if unit in {31, 40}:
        return _parse_keyword_unit(text, unit, source_name)
    raise ValueError("unit must be 30, 31, or 40")


def parse_fixed_unit(path: str | Path, *, unit: int | None = None) -> FixedUnitDocument:
    """Read and parse a fixed-unit file, inferring its unit from the suffix."""

    source = Path(path)
    resolved_unit = unit
    if resolved_unit is None:
        resolved_unit = _UNIT_FROM_SUFFIX.get(source.suffix.lower())
    if resolved_unit is None:
        raise ValueError(
            f"cannot infer fixed unit from {source.name!r}; pass unit=30, 31, or 40"
        )
    return parse_fixed_unit_text(
        source.read_text(encoding="utf-8", errors="strict"),
        resolved_unit,
        source_name=str(source),
    )


def summarize_fixed_unit(document: FixedUnitDocument) -> dict[str, Any]:
    """Return a compact semantic summary suitable for CLI/CI diagnostics."""

    stationary = document.stationary_points()
    path = document.path_points()
    return {
        "unit": document.unit,
        "source_name": document.source_name,
        "title": document.title,
        "section_count": len(document.sections),
        "section_names": list(document.section_names),
        "stationary_roles": [point.role for point in stationary],
        "stationary_count": len(stationary),
        "path_point_count": len(path),
        "frequency_unit": document.frequency_unit,
        "hessian_form": document.hessian_form,
        "hessian_unit": document.hessian_unit,
    }
