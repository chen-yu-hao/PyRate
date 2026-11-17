"""Lossless-enough parser for Polyrate-style sectioned input files.

The legacy language is keyword driven and permits many program-specific hooks.
PyRate therefore separates syntactic parsing from semantic support: every
section and keyword is retained, while supported fields are exposed through
small typed accessors.  Unknown input is never silently discarded.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class LegacyParseError(ValueError):
    """Raised when a section or block is structurally malformed."""


class LegacySemanticError(ValueError):
    """Raised when a Polyrate input requests unsupported or contradictory semantics."""

    def __init__(
        self,
        message: str,
        *,
        section: str,
        keyword: str | None = None,
        line_number: int = 0,
    ) -> None:
        self.section = section
        self.keyword = keyword
        self.line_number = int(line_number)
        self.message = message
        loc = f"*{section}" + (f" {keyword}" if keyword else "")
        if self.line_number > 0:
            loc = f"{loc} line {self.line_number}"
        super().__init__(f"[{loc}] {message}")


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

_UNSUPPORTED_FEATURE_HINTS: dict[str, str] = {
    "ISPE": "dual-level ISPE interpolation is not supported",
    "DUAL": "dual-level calculations are not supported",
    "DUALLEVEL": "dual-level calculations are not supported",
    "IOC": "internal optimization correction (IOC) is not supported",
    "IVTST": "IVTST-0 / IVTST-M interpolation is not supported",
    "VTSTIC": "IVTST / VTST-IC interpolation is not supported",
    "VRCTST": "VRC-VTST variable-reaction-coordinate sampling is not supported",
    "VRC": "VRC-VTST is not supported",
    "ICVT": "ICVT (removed in Polyrate 2023) is not supported",
    "LCT": "large-curvature tunneling (LCT) is not supported",
    "LCG3": "large-curvature tunneling (LCG3) is not supported",
    "LCG4": "large-curvature tunneling (LCG4) is not supported",
    "LSAG": "large-curvature tunneling (LSAG) is not supported",
    "LAT": "large-curvature tunneling (LAT) is not supported",
    "SOLV": "implicit solvent / VSXC models are not supported",
    "RODS": "RODS (reoriented dividing surface) is not supported",
    "FU30": "legacy unit30 potential files are not supported; use .fu40 or analytical/direct PES",
    "UNIT30": "legacy unit30 potential files are not supported; use .fu40 or analytical/direct PES",
    "FU50": "dual-level unit50 files are not supported",
    "UNIT50": "dual-level unit50 files are not supported",
}

_SUPPORTED_SECTION_KEYWORDS: dict[str, set[str]] = {
    "GENERAL": {
        "TITLE",
        "ATOMS",
        "INPUNIT",
        "MASS",
        "NOPRINT",
        "PRINT",
        "STOP",
        "WRITEFU40",
        "NOFU40",
        "SUPERMOL",
        "NOSUPERMOL",
    },
    "ENERGETICS": {"POTENTIAL", "VSCALE", "EREF"},
    "SECOND": {
        "FIRST",
        "CENTRAL",
        "FORWARD",
        "DX",
        "HESSIAN",
        "CURV",
        "CURVILINEAR",
        "RECTILINEAR",
        "RECT",
        "PROJECT",
        "NOPROJECT",
    },
    "OPTIMIZATION": {
        "OPTMIN",
        "OPTTS",
        "NOOPT",
        "NOOPTMIN",
        "NOOPTTS",
        "BFGS",
        "PRFO",
        "EF",
        "NR",
        "GTOL",
        "MAXSTEP",
        "NSTEP",
    },
    "REACT1": {"GEOM", "SPECIES", "ELEC", "FREQSCALE", "SCALEFREQ", "STATUS", "INIT"},
    "REACT2": {"GEOM", "SPECIES", "ELEC", "FREQSCALE", "SCALEFREQ", "STATUS", "INIT"},
    "PROD1": {"GEOM", "SPECIES", "ELEC", "FREQSCALE", "SCALEFREQ", "STATUS", "INIT"},
    "PROD2": {"GEOM", "SPECIES", "ELEC", "FREQSCALE", "SCALEFREQ", "STATUS", "INIT"},
    "START": {"GEOM", "SPECIES", "ELEC", "FREQSCALE", "SCALEFREQ", "STATUS", "INIT", "SIGN"},
    "PATH": {
        "SCALEMASS",
        "SSTEP",
        "INH",
        "INP",
        "SRANGE",
        "SMAX",
        "SMIN",
        "PM",
        "RPM",
        "LQA",
        "ESD",
        "ES1",
        "EULER",
        "SADDLE",
        "NOSADDLE",
        "SIGN",
        "CURV",
        "CURVILINEAR",
        "RECTILINEAR",
        "RECT",
        "PROJECT",
        "NOPROJECT",
        "RPMCOORD",
        "PRCOORD",
        "CURVATURE",
        "NOCURV",
        "PRINT",
        "NOPRINT",
        "GRRESTART",
        "RSTTOL",
        "OPTM",
        "OPTTS",
        "FREQSCALE",
        "RPHB",
    },
    "TUNNEL": {
        "ZCT",
        "MEPSAG",
        "SCT",
        "CD-SCSAG",
        "CDSCSAG",
        "WIGNER",
        "ECKART",
        "NONE",
        "NOTUNNEL",
        "OMT",
        "COMT",
        "MUOMT",
        "MOMT",
        "KAPVA",
        "BOLTZ",
        "QUAD",
        "SSTEP",
    },
    "RATE": {
        "TST",
        "CVT",
        "CUS",
        "MUVT",
        "UVT",
        "US",
        "NOCVT",
        "NOTST",
        "SIGMAF",
        "SIGMAR",
        "TEMP",
        "BOTHK",
        "FORWARDK",
        "REVERSEK",
        "PRESSURE",
        "PSTD",
        "ANALYSIS",
        "PRINT",
        "NOPRINT",
    },
    "POINT40": {
        "SMEP",
        "SVALUE",
        "VMEP",
        "VVALUE",
        "GEOM",
        "FREQ",
        "FREQUENCIES",
        "GRADS",
        "FORCES",
        "HESSIAN",
        "BFS",
    },
}

_STATIONARY40_KWS = {
    "VVALUE",
    "VMEP",
    "GEOM",
    "FREQ",
    "FREQUENCIES",
    "WSTAR",
    "GRADS",
    "FORCES",
    "HESSIAN",
}
_SUPPORTED_SECTION_KEYWORDS["GEN40"] = {
    "GEOMUNIT",
    "FREQSOURCE",
    "FREQUNIT",
    "HESSFORM",
    "GRADS",
    "NOGRADS",
}
_SUPPORTED_SECTION_KEYWORDS["R140"] = set(_STATIONARY40_KWS)
_SUPPORTED_SECTION_KEYWORDS["R240"] = set(_STATIONARY40_KWS)
_SUPPORTED_SECTION_KEYWORDS["P140"] = set(_STATIONARY40_KWS)
_SUPPORTED_SECTION_KEYWORDS["P240"] = set(_STATIONARY40_KWS)
_SUPPORTED_SECTION_KEYWORDS["SADDLE40"] = set(_STATIONARY40_KWS)


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

    @property
    def mass_overrides(self) -> dict[int, float]:
        general = self.section("GENERAL")
        if general is None:
            return {}
        overrides: dict[int, float] = {}
        for entry in general.get_all("MASS"):
            tokens: list[str] = list(entry.arguments)
            for line in entry.body:
                tokens.extend(line.split())
            for i in range(0, len(tokens) - 1, 2):
                idx = int(tokens[i])
                mass = float(tokens[i + 1].replace("D", "E").replace("d", "e"))
                overrides[idx] = mass
        return overrides

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
            ener = self.section("ENERGETICS")
            pot_entry = ener.get("POTENTIAL") if ener else None
            pot_arg = pot_entry.arguments[0].upper() if pot_entry and pot_entry.arguments else ""
            if pot_arg not in {"HOOKS", "DIRECT", "PES"} and not pot_arg.startswith("BUILTIN:"):
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


def _parse_num_token(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def validate_legacy_semantics(document: LegacyInput) -> None:
    """Validate that a parsed legacy document uses strictly supported Polyrate semantics.

    Raises :class:`LegacySemanticError` carrying ``section``, ``keyword``, and
    ``line_number`` for any unsupported section, keyword, or contradictory option.
    """

    for section in document.sections:
        sec_name = section.name
        if sec_name not in _SUPPORTED_SECTION_KEYWORDS:
            hint = _UNSUPPORTED_FEATURE_HINTS.get(
                sec_name, f"unsupported Polyrate section *{sec_name}"
            )
            raise LegacySemanticError(
                hint, section=sec_name, keyword=None, line_number=section.line_number
            )
        allowed = _SUPPORTED_SECTION_KEYWORDS[sec_name]
        for entry in section.entries:
            kw = entry.keyword
            if kw not in allowed:
                hint = _UNSUPPORTED_FEATURE_HINTS.get(
                    kw, f"unsupported keyword {kw} in *{sec_name}"
                )
                raise LegacySemanticError(
                    hint, section=sec_name, keyword=kw, line_number=entry.line_number
                )
            for arg in entry.arguments:
                arg_u = arg.upper()
                if arg_u in _UNSUPPORTED_FEATURE_HINTS:
                    raise LegacySemanticError(
                        _UNSUPPORTED_FEATURE_HINTS[arg_u],
                        section=sec_name,
                        keyword=kw,
                        line_number=entry.line_number,
                    )

            if sec_name == "GENERAL":
                if kw == "INPUNIT" and entry.arguments:
                    unit_u = entry.arguments[0].upper()
                    if unit_u not in {"AU", "BOHR", "ANG", "ANGSTROM"}:
                        raise LegacySemanticError(
                            f"unsupported coordinate unit {unit_u!r}; expected AU/BOHR/ANG",
                            section=sec_name,
                            keyword=kw,
                            line_number=entry.line_number,
                        )
                elif kw == "MASS":
                    toks: list[str] = list(entry.arguments)
                    for line in entry.body:
                        toks.extend(line.split())
                    if not toks or len(toks) % 2 != 0:
                        raise LegacySemanticError(
                            "MASS requires pairs of <atom_index> <mass_amu>",
                            section=sec_name,
                            keyword=kw,
                            line_number=entry.line_number,
                        )
                    for idx_t in range(0, len(toks), 2):
                        try:
                            _ = int(toks[idx_t])
                            mval = _parse_num_token(toks[idx_t + 1])
                        except ValueError as exc:
                            raise LegacySemanticError(
                                f"invalid MASS specification near {toks[idx_t]!r}",
                                section=sec_name,
                                keyword=kw,
                                line_number=entry.line_number,
                            ) from exc
                        if not math.isfinite(mval) or mval <= 0.0:
                            raise LegacySemanticError(
                                "isotopic mass must be positive",
                                section=sec_name,
                                keyword=kw,
                                line_number=entry.line_number,
                            )

            elif sec_name == "ENERGETICS" and kw == "POTENTIAL":
                pot_arg = entry.arguments[0].upper() if entry.arguments else ""
                valid_pot = pot_arg in {"UNIT40", "FU40", "HOOKS", "DIRECT", "PES"} or (
                    pot_arg.startswith("BUILTIN:")
                )
                if not valid_pot:
                    raise LegacySemanticError(
                        f"unsupported POTENTIAL source {pot_arg!r}; expected UNIT40, HOOKS, "
                        "or BUILTIN:<name>",
                        section=sec_name,
                        keyword=kw,
                        line_number=entry.line_number,
                    )

            elif sec_name == "SECOND" and kw == "DX" and entry.arguments:
                try:
                    dx = _parse_num_token(entry.arguments[0])
                except ValueError as exc:
                    raise LegacySemanticError(
                        f"invalid DX value {entry.arguments[0]!r}",
                        section=sec_name,
                        keyword=kw,
                        line_number=entry.line_number,
                    ) from exc
                if not math.isfinite(dx) or dx <= 0.0:
                    raise LegacySemanticError(
                        "DX finite-difference step must be positive",
                        section=sec_name,
                        keyword=kw,
                        line_number=entry.line_number,
                    )

            elif sec_name == "PATH":
                if kw in {"SCALEMASS", "SSTEP"} and entry.arguments:
                    try:
                        val = _parse_num_token(entry.arguments[0])
                    except ValueError as exc:
                        raise LegacySemanticError(
                            f"invalid numeric argument for {kw}: {entry.arguments[0]!r}",
                            section=sec_name,
                            keyword=kw,
                            line_number=entry.line_number,
                        ) from exc
                    if not math.isfinite(val) or val <= 0.0:
                        raise LegacySemanticError(
                            f"{kw} must be positive",
                            section=sec_name,
                            keyword=kw,
                            line_number=entry.line_number,
                        )
                elif kw in {"INH", "INP"} and entry.arguments:
                    try:
                        ival = int(entry.arguments[0])
                    except ValueError as exc:
                        raise LegacySemanticError(
                            f"invalid integer argument for {kw}: {entry.arguments[0]!r}",
                            section=sec_name,
                            keyword=kw,
                            line_number=entry.line_number,
                        ) from exc
                    if ival < 1:
                        raise LegacySemanticError(
                            f"{kw} must be at least 1",
                            section=sec_name,
                            keyword=kw,
                            line_number=entry.line_number,
                        )
                elif kw == "RPMCOORD" and entry.arguments:
                    rc = entry.arguments[0].upper()
                    valid_rc = {
                        "CURV",
                        "CURVILINEAR",
                        "CART",
                        "CARTESIAN",
                        "RECT",
                        "RECTILINEAR",
                    }
                    if rc not in valid_rc:
                        raise LegacySemanticError(
                            f"unsupported RPMCOORD {rc!r}; expected CURV or RECT/CART",
                            section=sec_name,
                            keyword=kw,
                            line_number=entry.line_number,
                        )



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
        if keyword in _BLOCK_KEYWORDS or (keyword == "MASS" and not arguments):
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
