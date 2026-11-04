"""Executable mapping for the conservative Polyrate legacy subset.

The retaining parser in :mod:`pyrate.legacy` answers whether a deck is
syntactically readable.  This module answers a stricter question: whether all
physical data required by the current PyRate solver can be mapped without
inventing values.  The first executable subset is conventional harmonic TST
whose stationary-point energies and frequencies are supplied by fixed unit 30,
31, or 40 files.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .constants import ANGSTROM_M, BOHR_M, HARTREE_KJ_MOL
from .fixed_units import (
    FixedUnitDocument,
    FixedUnitStationaryPoint,
    parse_fixed_unit,
    summarize_fixed_unit,
)
from .legacy import LegacyInput, LegacySection
from .models import Atom, CalculationOptions, ElectronicState, Reaction, Species

_POTENTIAL_UNIT_RE = re.compile(r"UNIT\s*(30|31|40)", re.IGNORECASE)
_ROLE_TO_SECTION = {
    "reactant1": "REACT1",
    "reactant2": "REACT2",
    "product1": "PROD1",
    "product2": "PROD2",
    "saddle": "START",
}
_UNSUPPORTED_METHODS = frozenset(
    {"CVT", "ICVT", "MUVT", "CUS", "US", "ZCT", "SCT", "LCT", "LCG3", "LCG4"}
)


@dataclass(frozen=True, slots=True)
class LegacyExecutionAnalysis:
    """Independent syntax-to-semantics execution assessment."""

    semantic_status: str
    execution_status: str
    fixed_unit: int | None
    fixed_unit_path: str | None
    blockers: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def executable(self) -> bool:
        return self.execution_status == "ready"

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantic_status": self.semantic_status,
            "execution_status": self.execution_status,
            "executable": self.executable,
            "fixed_unit": self.fixed_unit,
            "fixed_unit_path": self.fixed_unit_path,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ConvertedLegacyCalculation:
    """Typed solver inputs produced from one executable legacy deck."""

    name: str
    reaction: Reaction
    options: CalculationOptions
    raw: Mapping[str, Any]
    analysis: LegacyExecutionAnalysis


class LegacyExecutionError(ValueError):
    """Raised when a parsed legacy input still lacks executable semantics."""

    def __init__(self, analysis: LegacyExecutionAnalysis) -> None:
        self.analysis = analysis
        detail = "; ".join(analysis.blockers) or "legacy input is not executable"
        super().__init__(detail)


def _potential_unit(document: LegacyInput) -> int | None:
    energetics = document.section("ENERGETICS")
    if energetics is None:
        return None
    entry = energetics.get("POTENTIAL")
    if entry is None:
        return None
    text = " ".join(entry.arguments)
    match = _POTENTIAL_UNIT_RE.search(text)
    return int(match.group(1)) if match is not None else None


def _fixed_unit_path(source_path: str | Path, unit: int) -> Path:
    source = Path(source_path)
    candidates = (
        source.with_suffix(f".fu{unit}"),
        source.parent / f"fu{unit}",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _section_geometry_atom_count(section: LegacySection | None) -> int:
    if section is None:
        return 0
    geometry = section.get("GEOM")
    if geometry is None:
        return 0
    count = 0
    for line in geometry.body:
        tokens = line.replace(",", " ").split()
        if len(tokens) < 4:
            continue
        try:
            int(tokens[0])
            tuple(
                float(token.replace("D", "E").replace("d", "e"))
                for token in tokens[1:4]
            )
        except ValueError:
            continue
        count += 1
    return count


def _has_nonzero_electronic_energy(section: LegacySection | None) -> bool:
    if section is None:
        return False
    entry = section.get("ELEC")
    if entry is None:
        return False
    for line in entry.body:
        tokens = line.replace(",", " ").split()
        if len(tokens) < 2:
            continue
        try:
            value = float(tokens[1].replace("D", "E").replace("d", "e"))
        except ValueError:
            return True
        if value != 0.0:
            return True
    return False


def _requested_wigner(document: LegacyInput) -> bool:
    return any(
        section is not None and section.has("WIGNER")
        for section in (document.section("RATE"), document.section("TUNNEL"))
    )


def analyze_legacy_execution(
    document: LegacyInput,
    source_path: str | Path | None = None,
) -> LegacyExecutionAnalysis:
    """Report whether the current solver can execute ``document`` without guesses."""

    blockers: list[str] = []
    warnings: list[str] = []
    unit = _potential_unit(document)
    fixed_path: Path | None = None
    fixed_document: FixedUnitDocument | None = None

    rate = document.section("RATE")
    if rate is None:
        blockers.append("missing *RATE section")
    elif not rate.has("TST"):
        blockers.append("current legacy execution requires *RATE/TST")

    unsupported = sorted(set(document.requested_methods()) & _UNSUPPORTED_METHODS)
    if unsupported:
        blockers.append("requested methods not executable: " + ", ".join(unsupported))

    if not document.temperatures_k:
        blockers.append("missing *RATE/TEMP temperature grid")

    if unit is None:
        blockers.append("*ENERGETICS/POTENTIAL must select UNIT30, UNIT31, or UNIT40")
    elif source_path is None:
        blockers.append(f"UNIT{unit} execution requires the legacy input path")
    else:
        fixed_path = _fixed_unit_path(source_path, unit)
        if not fixed_path.is_file():
            blockers.append(f"fixed unit file not found: {fixed_path}")
        else:
            try:
                fixed_document = parse_fixed_unit(fixed_path, unit=unit)
            except (OSError, ValueError) as exc:
                blockers.append(f"cannot parse {fixed_path.name}: {exc}")

    atom_table = dict(document.atom_table)
    if not atom_table:
        blockers.append("missing *GENERAL/ATOMS table")

    required_roles: list[str] = ["reactant1"]
    if _section_geometry_atom_count(document.section("REACT2")):
        required_roles.append("reactant2")
    required_roles.append("saddle")

    if fixed_document is not None:
        stationary = {point.role: point for point in fixed_document.stationary_points()}
        for role in required_roles:
            section_name = _ROLE_TO_SECTION[role]
            section = document.section(section_name)
            atom_count = _section_geometry_atom_count(section)
            if atom_count == 0:
                blockers.append(f"missing executable geometry in *{section_name}")
                continue
            point = stationary.get(role)
            if point is None:
                blockers.append(f"UNIT{fixed_document.unit} has no {role} stationary-point record")
                continue
            if atom_count > 1 and not point.frequencies:
                blockers.append(
                    f"{role} supplies a Hessian but no direct frequencies; "
                    "legacy Hessian-to-mode execution is not implemented yet"
                )
            elif point.frequencies:
                try:
                    point.frequencies_cm1
                except ValueError as exc:
                    blockers.append(f"cannot convert {role} frequencies: {exc}")
            if point.potential_hartree is None:
                blockers.append(f"UNIT{fixed_document.unit} has no potential for {role}")
            if _has_nonzero_electronic_energy(section):
                blockers.append(
                    f"*{section_name}/ELEC contains a nonzero excitation energy; "
                    "its legacy energy-unit semantics are not mapped yet"
                )
        saddle = stationary.get("saddle")
        if _requested_wigner(document):
            if saddle is None or saddle.imaginary_frequency is None:
                blockers.append(
                    "Wigner execution requires a fixed-unit saddle imaginary frequency"
                )
            else:
                try:
                    saddle.imaginary_frequency_cm1
                except ValueError as exc:
                    blockers.append(f"cannot convert saddle imaginary frequency: {exc}")

    for section_name in ("REACT1", "REACT2", "PROD1", "PROD2", "START"):
        section = document.section(section_name)
        if section is not None and (section.has("VANHAR") or section.has("TOR")):
            blockers.append(
                f"*{section_name} requests anharmonic/torsional semantics not implemented"
            )

    if rate is not None and rate.has("BOTHK"):
        warnings.append("forward rate is executable; reverse-rate emission is not implemented yet")

    deduplicated_blockers = tuple(dict.fromkeys(blockers))
    deduplicated_warnings = tuple(dict.fromkeys(warnings))
    if not deduplicated_blockers:
        semantic_status = "mapped"
        execution_status = "ready"
    elif unit is not None:
        semantic_status = "partial"
        execution_status = "blocked"
    else:
        semantic_status = "unmapped"
        execution_status = "blocked"
    return LegacyExecutionAnalysis(
        semantic_status=semantic_status,
        execution_status=execution_status,
        fixed_unit=unit,
        fixed_unit_path=str(fixed_path) if fixed_path is not None else None,
        blockers=deduplicated_blockers,
        warnings=deduplicated_warnings,
    )


def _input_length_scale_angstrom(document: LegacyInput) -> float:
    general = document.section("GENERAL")
    entry = general.get("INPUNIT") if general is not None else None
    if entry is None or not entry.arguments:
        return 1.0
    unit = entry.arguments[0].strip().upper()
    if unit in {"AU", "BOHR", "A0"}:
        return BOHR_M / ANGSTROM_M
    if unit in {"ANG", "ANGSTROM", "ANGSTROMS"}:
        return 1.0
    raise ValueError(f"unsupported legacy coordinate unit {entry.arguments[0]!r}")


def _atoms_by_index(document: LegacyInput) -> dict[int, str]:
    result = dict(document.atom_table)
    if not result:
        raise ValueError("legacy input has no atom table")
    return result


def _section_atoms(
    document: LegacyInput,
    section_name: str,
    atom_symbols: Mapping[int, str],
) -> tuple[Atom, ...]:
    section = document.section(section_name)
    if section is None:
        raise ValueError(f"legacy input has no *{section_name} section")
    geometry = section.get("GEOM")
    if geometry is None:
        raise ValueError(f"legacy input has no *{section_name}/GEOM block")
    scale = _input_length_scale_angstrom(document)
    atoms: list[Atom] = []
    seen: set[int] = set()
    for line in geometry.body:
        tokens = line.replace(",", " ").split()
        if len(tokens) < 4:
            continue
        try:
            atom_index = int(tokens[0])
            coordinates = tuple(
                float(token.replace("D", "E").replace("d", "e")) * scale
                for token in tokens[1:4]
            )
        except ValueError as exc:
            raise ValueError(
                f"invalid *{section_name}/GEOM row {line!r}"
            ) from exc
        if atom_index in seen:
            raise ValueError(f"duplicate atom {atom_index} in *{section_name}/GEOM")
        try:
            symbol = atom_symbols[atom_index]
        except KeyError as exc:
            raise ValueError(
                f"atom {atom_index} in *{section_name}/GEOM is absent from *GENERAL/ATOMS"
            ) from exc
        seen.add(atom_index)
        atoms.append(
            Atom(
                symbol=symbol,
                label=f"{symbol}{atom_index}",
                x_angstrom=coordinates[0],
                y_angstrom=coordinates[1],
                z_angstrom=coordinates[2],
            )
        )
    if not atoms:
        raise ValueError(f"*{section_name}/GEOM contains no atom rows")
    return tuple(atoms)


def _electronic_states(section: LegacySection) -> tuple[ElectronicState, ...]:
    entry = section.get("ELEC")
    if entry is None:
        return (ElectronicState(),)
    states: list[ElectronicState] = []
    for line in entry.body:
        tokens = line.replace(",", " ").split()
        if len(tokens) < 2:
            continue
        try:
            degeneracy = int(tokens[0])
            energy_hartree = float(tokens[1].replace("D", "E").replace("d", "e"))
        except ValueError as exc:
            raise ValueError(f"invalid ELEC row {line!r}") from exc
        states.append(
            ElectronicState(
                degeneracy=degeneracy,
                energy_kj_mol=energy_hartree * HARTREE_KJ_MOL,
            )
        )
    return tuple(states) if states else (ElectronicState(),)


def _rotor_type(section: LegacySection) -> str:
    entry = section.get("SPECIES")
    if entry is None or not entry.arguments:
        return "auto"
    token = entry.arguments[0].upper().replace("-", "")
    if "ATOMIC" in token:
        return "atomic"
    if "NONLIN" in token:
        return "nonlinear"
    if "LIN" in token:
        return "linear"
    return "auto"


def _rate_factor(document: LegacyInput, keyword: str) -> float:
    rate = document.section("RATE")
    entry = rate.get(keyword) if rate is not None else None
    if entry is None or not entry.arguments:
        return 1.0
    try:
        value = float(entry.arguments[0].replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ValueError(f"invalid *RATE/{keyword} value") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"*RATE/{keyword} must be finite and positive")
    return value


def _build_species(
    document: LegacyInput,
    section_name: str,
    role: str,
    point: FixedUnitStationaryPoint,
    atom_symbols: Mapping[int, str],
    *,
    forward_symmetry_factor: float = 1.0,
    reverse_symmetry_factor: float = 1.0,
) -> Species:
    section = document.section(section_name)
    if section is None:
        raise ValueError(f"missing *{section_name} section")
    frequencies = list(point.frequencies_cm1)
    if role == "saddle" and point.imaginary_frequency_cm1 is not None:
        frequencies.append(-abs(point.imaginary_frequency_cm1))
    metadata: dict[str, Any] = {
        "legacy_section": section_name,
        "fixed_unit_role": role,
        "fixed_unit_section": point.source_section,
    }
    if role == "saddle":
        metadata["reaction_path_degeneracy"] = forward_symmetry_factor
        metadata["reverse_reaction_path_degeneracy"] = reverse_symmetry_factor
    return Species(
        name=role,
        atoms=_section_atoms(document, section_name, atom_symbols),
        electronic_energy_kj_mol=point.potential_kj_mol or 0.0,
        frequencies_cm1=tuple(frequencies),
        symmetry_number=1,
        electronic_states=_electronic_states(section),
        rotor_type=_rotor_type(section),
        metadata=metadata,
    )


def convert_legacy_to_models(
    document: LegacyInput,
    source_path: str | Path,
) -> ConvertedLegacyCalculation:
    """Convert the executable legacy subset to typed PyRate solver models."""

    source = Path(source_path)
    analysis = analyze_legacy_execution(document, source)
    if not analysis.executable:
        raise LegacyExecutionError(analysis)
    assert analysis.fixed_unit is not None
    assert analysis.fixed_unit_path is not None
    fixed = parse_fixed_unit(analysis.fixed_unit_path, unit=analysis.fixed_unit)
    stationary = {point.role: point for point in fixed.stationary_points()}
    atom_symbols = _atoms_by_index(document)
    sigma_forward = _rate_factor(document, "SIGMAF")
    sigma_reverse = _rate_factor(document, "SIGMAR")

    reactant_roles = ["reactant1"]
    if _section_geometry_atom_count(document.section("REACT2")):
        reactant_roles.append("reactant2")
    reactants = tuple(
        _build_species(
            document,
            _ROLE_TO_SECTION[role],
            role,
            stationary[role],
            atom_symbols,
        )
        for role in reactant_roles
    )
    transition_state = _build_species(
        document,
        "START",
        "saddle",
        stationary["saddle"],
        atom_symbols,
        forward_symmetry_factor=sigma_forward,
        reverse_symmetry_factor=sigma_reverse,
    )

    product_roles = [
        role
        for role in ("product1", "product2")
        if role in stationary
        and _section_geometry_atom_count(document.section(_ROLE_TO_SECTION[role]))
    ]
    products = tuple(
        _build_species(
            document,
            _ROLE_TO_SECTION[role],
            role,
            stationary[role],
            atom_symbols,
        )
        for role in product_roles
    )

    reaction = Reaction(
        reactants=reactants,
        transition_state=transition_state,
        products=products,
    )
    tunneling = "wigner" if _requested_wigner(document) else "none"
    output_unit = (
        "s^-1" if reaction.molecularity == 1 else "cm^3 molecule^-1 s^-1"
    )
    options = CalculationOptions(
        temperatures_k=document.temperatures_k,
        method="tst",
        tunneling=tunneling,
        output_rate_unit=output_unit,
    )
    name = document.title or source.stem
    raw: dict[str, Any] = {
        "format": "polyrate-legacy-executable-subset",
        "source": str(source),
        "legacy": document.as_dict(),
        "fixed_unit": summarize_fixed_unit(fixed),
        "execution": analysis.as_dict(),
        "forward_symmetry_factor": sigma_forward,
        "reverse_symmetry_factor": sigma_reverse,
    }
    return ConvertedLegacyCalculation(
        name=name,
        reaction=reaction,
        options=options,
        raw=raw,
        analysis=analysis,
    )
