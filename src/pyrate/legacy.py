"""Lossless-enough parser for Polyrate-style sectioned input files.

The legacy language is keyword driven and permits many program-specific hooks.
PyRate therefore separates syntactic parsing from semantic support: every
section and keyword is retained, while supported fields are exposed through
small typed accessors.  Unknown input is never silently discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class LegacyParseError(ValueError):
    """Raised when a section or block is structurally malformed."""


_BLOCK_KEYWORDS = {
    "TITLE",
    "ATOMS",
    "GEOM",
    "CONSTANT",
    "VANHAR",
    "ELEC",
    "EXFIRST",
    "EXSECOND",
    "SRANGE",
    "QUAD",
    "TEMP",
    "GSPEC",
    "ANALYSIS",
    "SPECIESDATA",
    "FREQ",
    "FREQUENCIES",
    "MOMENT",
    "MOMENTS",
    "IFREQ",
    "HESSIAN",
    "GRADS",
    "FORCES",
    "GRADSB1",
    "GRADSB2",
    "FORCESB1",
    "FORCESB2",
    "BFS",
    "3RDDERIV",
    "4THDERIV",
    "SCTOPT",
    "EACT",
}

_SUPPORTED_SECTIONS = {
    "GENERAL",
    "ENERGETICS",
    "SECOND",
    "OPTIMIZATION",
    "REACT1",
    "REACT2",
    "PROD1",
    "PROD2",
    "START",
    "PATH",
    "TUNNEL",
    "RATE",
    "GEN40",
    "R140",
    "R240",
    "P140",
    "P240",
    "SADDLE40",
    "POINT40",
}


@dataclass(frozen=True, slots=True)
class LegacyEntry:
    keyword: str
    arguments: tuple[str, ...] = ()
    body: tuple[str, ...] = ()
    line_number: int = 0
    raw_line: str = ""

    @property
    def is_block(self) -> bool:
        return bool(self.body) or self.keyword in _BLOCK_KEYWORDS

    def as_dict(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "arguments": list(self.arguments),
            "body": list(self.body),
            "line_number": self.line_number,
        }


@dataclass(slots=True)
class LegacySection:
    name: str
    entries: list[LegacyEntry] = field(default_factory=list)
    line_number: int = 0

    def get_all(self, keyword: str) -> tuple[LegacyEntry, ...]:
        normalized = keyword.upper()
        return tuple(entry for entry in self.entries if entry.keyword == normalized)

    def get(self, keyword: str) -> LegacyEntry | None:
        entries = self.get_all(keyword)
        return entries[-1] if entries else None

    def has(self, keyword: str) -> bool:
        return self.get(keyword) is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "line_number": self.line_number,
            "entries": [entry.as_dict() for entry in self.entries],
        }


@dataclass(slots=True)
class LegacyInput:
    sections: list[LegacySection]
    source: str | None = None
    comments: tuple[tuple[int, str], ...] = ()

    def section_all(self, name: str) -> tuple[LegacySection, ...]:
        normalized = name.upper().lstrip("*")
        return tuple(section for section in self.sections if section.name == normalized)

    def section(self, name: str) -> LegacySection | None:
        sections = self.section_all(name)
        return sections[-1] if sections else None

    @property
    def title(self) -> str | None:
        general = self.section("GENERAL")
        if general is None:
            return None
        entry = general.get("TITLE")
        if entry is None:
            return None
        if entry.body:
            return " ".join(line.strip() for line in entry.body if line.strip()).strip()
        return " ".join(entry.arguments).strip() or None

    @property
    def temperatures_k(self) -> tuple[float, ...]:
        rate = self.section("RATE")
        if rate is None:
            return ()
        entry = rate.get("TEMP")
        if entry is None:
            return ()
        tokens: list[str] = list(entry.arguments)
        for line in entry.body:
            tokens.extend(line.split())
        values: list[float] = []
        for token in tokens:
            try:
                values.append(float(token.replace("D", "E").replace("d", "e")))
            except ValueError:
                continue
        return tuple(values)

    @property
    def atom_table(self) -> tuple[tuple[int, str], ...]:
        general = self.section("GENERAL")
        if general is None:
            return ()
        entry = general.get("ATOMS")
        if entry is None:
            return ()
        atoms: list[tuple[int, str]] = []
        for line in entry.body:
            tokens = line.split()
            if len(tokens) >= 2:
                try:
                    atoms.append((int(tokens[0]), tokens[1]))
                except ValueError:
                    continue
        return tuple(atoms)

    def requested_methods(self) -> tuple[str, ...]:
        methods: list[str] = []
        rate = self.section("RATE")
        tunnel = self.section("TUNNEL")
        for keyword in ("TST", "CVT", "ICVT", "MUVT", "CUS", "US"):
            if rate is not None and rate.has(keyword):
                methods.append(keyword)
        for keyword in ("ZCT", "SCT", "LCT", "LCG3", "LCG4"):
            if tunnel is not None and tunnel.has(keyword):
                methods.append(keyword)
        return tuple(methods)

    def compatibility_warnings(self) -> tuple[str, ...]:
        warnings: list[str] = []
        unsupported_sections = sorted(
            {section.name for section in self.sections if section.name not in _SUPPORTED_SECTIONS}
        )
        if unsupported_sections:
            warnings.append(
                "parsed but not executable sections: " + ", ".join(unsupported_sections)
            )
        unsupported_methods = [
            method
            for method in self.requested_methods()
            if method in {"ICVT", "LCT", "LCG3", "LCG4"}
        ]
        if unsupported_methods:
            warnings.append(
                "requested methods not implemented in the current solver: "
                + ", ".join(unsupported_methods)
            )
        if "SCT" in self.requested_methods() and not self.section_all("POINT40"):
            warnings.append("SCT requires reaction-path curvature or unit40 path data")
        if self.section("RATE") is None and self.section("GEN40") is None:
            warnings.append("no *RATE section")
        if not self.temperatures_k and self.section("GEN40") is None:
            warnings.append("no temperature grid found in *RATE/TEMP")
        return tuple(warnings)

    def summary(self) -> dict[str, Any]:
        return {
            "format": "polyrate-legacy",
            "source": self.source,
            "title": self.title,
            "sections": [section.name for section in self.sections],
            "atoms": [list(atom) for atom in self.atom_table],
            "temperatures_k": list(self.temperatures_k),
            "requested_methods": list(self.requested_methods()),
            "warnings": list(self.compatibility_warnings()),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "sections": [section.as_dict() for section in self.sections],
            "comments": [list(item) for item in self.comments],
        }


def _strip_comment(line: str) -> tuple[str, str | None]:
    if "#" not in line:
        return line.rstrip(), None
    content, comment = line.split("#", 1)
    return content.rstrip(), comment.rstrip()


def parse_legacy_text(text: str, *, source: str | None = None) -> LegacyInput:
    """Parse a Polyrate-style input string."""

    lines = text.splitlines()
    sections: list[LegacySection] = []
    comments: list[tuple[int, str]] = []
    current: LegacySection | None = None
    index = 0

    while index < len(lines):
        line_number = index + 1
        original = lines[index]
        content, comment = _strip_comment(original)
        if comment is not None:
            comments.append((line_number, comment))
        stripped = content.strip()
        index += 1
        if not stripped:
            continue

        if stripped.startswith("*"):
            name = stripped[1:].split()[0].upper()
            if not name:
                raise LegacyParseError(f"empty section marker at line {line_number}")
            current = LegacySection(name=name, line_number=line_number)
            sections.append(current)
            continue

        if current is None:
            raise LegacyParseError(
                f"keyword outside a section at line {line_number}: {stripped!r}"
            )

        tokens = stripped.split()
        keyword = tokens[0].upper()
        arguments = tuple(tokens[1:])
        body: list[str] = []
        if keyword in _BLOCK_KEYWORDS:
            found_end = False
            while index < len(lines):
                body_line_number = index + 1
                body_original = lines[index]
                body_content, body_comment = _strip_comment(body_original)
                if body_comment is not None:
                    comments.append((body_line_number, body_comment))
                index += 1
                if body_content.strip().upper() == "END":
                    found_end = True
                    break
                body.append(body_content.rstrip())
            if not found_end:
                raise LegacyParseError(
                    f"unterminated {keyword} block beginning at line {line_number}"
                )
        elif keyword == "END":
            # A redundant END outside a recognized block is retained instead of
            # being treated as a section terminator; sections end at the next '*'.
            arguments = ()

        current.entries.append(
            LegacyEntry(
                keyword=keyword,
                arguments=arguments,
                body=tuple(body),
                line_number=line_number,
                raw_line=original,
            )
        )

    if not sections:
        raise LegacyParseError("no *SECTION markers found")
    return LegacyInput(sections=sections, source=source, comments=tuple(comments))


def load_legacy(path: str | Path) -> LegacyInput:
    source = Path(path)
    return parse_legacy_text(source.read_text(encoding="utf-8"), source=str(source))


def iter_keywords(document: LegacyInput) -> Iterable[tuple[str, LegacyEntry]]:
    for section in document.sections:
        for entry in section.entries:
            yield section.name, entry
