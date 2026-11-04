"""Staged numerical traces and error reports for Polyrate validation.

A final rate constant can agree for the wrong reasons. This module therefore
records and compares the calculation in physical stages: molecular partition
functions, zero-point energies, activation free energies, WKB actions,
transmission coefficients, and rates. The format is JSON-serializable and is
intended to be populated either by PyRate or by a parser for a reference
Polyrate output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .calculation import calculate_rates
from .constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL
from .models import CalculationOptions, Reaction, Species
from .native import wkb_exponent
from .partition import (
    partition_components,
    standard_gibbs_energy_kj_mol,
    zero_point_energy_kj_mol,
)
from .rates import activation_gibbs_kj_mol
from .tunneling import profile_wkb_kappa, transition_state_wigner_kappa

TRACE_SCHEMA = "pyrate-numerical-trace-v1"
COMPARISON_SCHEMA = "pyrate-numerical-comparison-v1"

STAGES = (
    "partition",
    "zpe",
    "free_energy",
    "activation_gibbs",
    "action",
    "kappa",
    "rate",
)


@dataclass(frozen=True, slots=True)
class TraceTolerance:
    absolute: float
    relative: float

    def __post_init__(self) -> None:
        if self.absolute < 0.0 or self.relative < 0.0:
            raise ValueError("trace tolerances cannot be negative")

    def limit(self, expected: float) -> float:
        return self.absolute + self.relative * abs(expected)

    def as_dict(self) -> dict[str, float]:
        return {"absolute": self.absolute, "relative": self.relative}


DEFAULT_TRACE_TOLERANCES: dict[str, TraceTolerance] = {
    "partition": TraceTolerance(absolute=1.0e-10, relative=1.0e-10),
    "zpe": TraceTolerance(absolute=1.0e-6, relative=1.0e-8),
    "free_energy": TraceTolerance(absolute=1.0e-5, relative=1.0e-8),
    "activation_gibbs": TraceTolerance(absolute=1.0e-5, relative=1.0e-8),
    "action": TraceTolerance(absolute=1.0e-8, relative=1.0e-7),
    "kappa": TraceTolerance(absolute=1.0e-8, relative=1.0e-5),
    "rate": TraceTolerance(absolute=0.0, relative=1.0e-4),
}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("trace axes must be finite")
    return result


def _axis_token(value: float | None) -> str:
    return "-" if value is None else format(value, ".17g")


@dataclass(frozen=True, slots=True)
class TracePoint:
    stage: str
    quantity: str
    value: float
    unit: str
    temperature_k: float | None = None
    coordinate: float | None = None
    energy_kj_mol: float | None = None

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unsupported trace stage {self.stage!r}")
        if not self.quantity.strip():
            raise ValueError("trace quantity cannot be empty")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("trace values must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "temperature_k", _optional_float(self.temperature_k))
        object.__setattr__(self, "coordinate", _optional_float(self.coordinate))
        object.__setattr__(self, "energy_kj_mol", _optional_float(self.energy_kj_mol))

    @property
    def locator(self) -> str:
        return "|".join(
            (
                self.stage,
                self.quantity,
                f"T={_axis_token(self.temperature_k)}",
                f"s={_axis_token(self.coordinate)}",
                f"E={_axis_token(self.energy_kj_mol)}",
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "quantity": self.quantity,
            "value": self.value,
            "unit": self.unit,
            "temperature_k": self.temperature_k,
            "coordinate": self.coordinate,
            "energy_kj_mol": self.energy_kj_mol,
            "locator": self.locator,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TracePoint":
        return cls(
            stage=str(data["stage"]),
            quantity=str(data["quantity"]),
            value=float(data["value"]),
            unit=str(data["unit"]),
            temperature_k=data.get("temperature_k"),
            coordinate=data.get("coordinate"),
            energy_kj_mol=data.get("energy_kj_mol"),
        )


@dataclass(frozen=True, slots=True)
class NumericalTrace:
    name: str
    points: tuple[TracePoint, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = TRACE_SCHEMA

    def __post_init__(self) -> None:
        points = tuple(self.points)
        locators = [point.locator for point in points]
        if len(locators) != len(set(locators)):
            duplicates = sorted(
                locator for locator in set(locators) if locators.count(locator) > 1
            )
            raise ValueError(f"duplicate trace locators: {duplicates[:3]}")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "metadata": dict(self.metadata),
            "points": [point.as_dict() for point in self.points],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NumericalTrace":
        schema = str(data.get("schema", ""))
        if schema != TRACE_SCHEMA:
            raise ValueError(f"unsupported numerical trace schema {schema!r}")
        return cls(
            name=str(data.get("name", "reference")),
            metadata=dict(data.get("metadata", {})),
            points=tuple(TracePoint.from_dict(item) for item in data.get("points", ())),
            schema=schema,
        )


def _species_entries(reaction: Reaction) -> tuple[tuple[str, Species, float | None], ...]:
    entries: list[tuple[str, Species, float | None]] = []
    for index, species in enumerate(reaction.reactants):
        entries.append((f"reactant[{index}].{species.name}", species, None))
    entries.append(
        (f"transition_state.{reaction.transition_state.name}", reaction.transition_state, None)
    )
    for index, species in enumerate(reaction.products):
        entries.append((f"product[{index}].{species.name}", species, None))
    if reaction.path is not None:
        for index, point in enumerate(reaction.path.points):
            entries.append(
                (
                    f"path[{index}].{point.dividing_surface.name}",
                    point.dividing_surface,
                    point.coordinate,
                )
            )
    return tuple(entries)


def _append_species_trace(
    points: list[TracePoint],
    key: str,
    species: Species,
    coordinate: float | None,
    temperature: float,
    standard_pressure_pa: float,
) -> None:
    components = partition_components(species, temperature, standard_pressure_pa)
    for component, value in components.as_dict().items():
        points.append(
            TracePoint(
                stage="partition",
                quantity=f"{key}.{component}",
                value=value,
                unit="ln(q)",
                temperature_k=temperature,
                coordinate=coordinate,
            )
        )
    points.append(
        TracePoint(
            stage="free_energy",
            quantity=f"{key}.standard_gibbs",
            value=standard_gibbs_energy_kj_mol(
                species, temperature, standard_pressure_pa
            ),
            unit="kJ mol^-1",
            temperature_k=temperature,
            coordinate=coordinate,
        )
    )


def build_numerical_trace(
    reaction: Reaction,
    options: CalculationOptions,
    *,
    name: str = "calculation",
    action_grid_points: int = 33,
) -> NumericalTrace:
    """Build a deterministic staged trace for one PyRate calculation."""

    if action_grid_points < 2:
        raise ValueError("action_grid_points must be at least 2")
    if len(set(options.temperatures_k)) != len(options.temperatures_k):
        raise ValueError("numerical traces require unique temperatures")
    points: list[TracePoint] = []
    species_entries = _species_entries(reaction)

    for key, species, coordinate in species_entries:
        points.append(
            TracePoint(
                stage="zpe",
                quantity=f"{key}.zpe",
                value=zero_point_energy_kj_mol(species),
                unit="kJ mol^-1",
                coordinate=coordinate,
            )
        )

    result = calculate_rates(reaction, options, name=name)
    rows = {row.temperature_k: row for row in result.rows}

    for temperature in options.temperatures_k:
        for key, species, coordinate in species_entries:
            _append_species_trace(
                points,
                key,
                species,
                coordinate,
                temperature,
                reaction.standard_pressure_pa,
            )

        points.append(
            TracePoint(
                stage="activation_gibbs",
                quantity="tst.delta_g",
                value=activation_gibbs_kj_mol(
                    reaction, reaction.transition_state, temperature
                ),
                unit="kJ mol^-1",
                temperature_k=temperature,
            )
        )
        if reaction.path is not None:
            for path_point in reaction.path.points:
                points.append(
                    TracePoint(
                        stage="activation_gibbs",
                        quantity="path.delta_g",
                        value=activation_gibbs_kj_mol(
                            reaction, path_point.dividing_surface, temperature
                        ),
                        unit="kJ mol^-1",
                        temperature_k=temperature,
                        coordinate=path_point.coordinate,
                    )
                )

        if options.tunneling == "none":
            kappa = 1.0
        elif options.tunneling == "wigner":
            kappa = transition_state_wigner_kappa(
                reaction.transition_state, temperature
            )
        else:
            assert options.tunneling_profile is not None
            kappa = profile_wkb_kappa(options.tunneling_profile, temperature)
        points.append(
            TracePoint(
                stage="kappa",
                quantity=options.tunneling,
                value=kappa,
                unit="dimensionless",
                temperature_k=temperature,
            )
        )

        row = rows[temperature]
        rate_values = (
            ("tst", row.k_tst),
            ("cvt", row.k_cvt),
            ("final", row.k_final),
        )
        for quantity, value in rate_values:
            if value is None:
                continue
            points.append(
                TracePoint(
                    stage="rate",
                    quantity=quantity,
                    value=value,
                    unit=row.rate_unit,
                    temperature_k=temperature,
                )
            )

    if options.tunneling_profile is not None:
        profile = options.tunneling_profile
        coordinates_m = tuple(
            value * ANGSTROM_M for value in profile.coordinates_angstrom
        )
        reference = profile.energies_kj_mol[0]
        shifted_energies_kj_mol = tuple(
            value - reference for value in profile.energies_kj_mol
        )
        shifted_energies_j = tuple(
            value * 1000.0 / AVOGADRO_MOL for value in shifted_energies_kj_mol
        )
        barrier = max(shifted_energies_kj_mol)
        reduced_mass = profile.reduced_mass_amu * AMU_KG
        for index in range(action_grid_points):
            energy_kj_mol = barrier * index / (action_grid_points - 1)
            action = wkb_exponent(
                coordinates_m,
                shifted_energies_j,
                energy_kj_mol * 1000.0 / AVOGADRO_MOL,
                reduced_mass,
            )
            points.append(
                TracePoint(
                    stage="action",
                    quantity="wkb.gamma",
                    value=action,
                    unit="dimensionless",
                    energy_kj_mol=energy_kj_mol,
                )
            )

    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    points.sort(
        key=lambda point: (
            stage_order[point.stage],
            point.quantity,
            -math.inf if point.temperature_k is None else point.temperature_k,
            -math.inf if point.coordinate is None else point.coordinate,
            -math.inf if point.energy_kj_mol is None else point.energy_kj_mol,
        )
    )
    return NumericalTrace(
        name=name,
        points=tuple(points),
        metadata={
            "method": options.method,
            "tunneling": options.tunneling,
            "molecularity": reaction.molecularity,
            "standard_pressure_pa": reaction.standard_pressure_pa,
            "action_grid_points": action_grid_points,
        },
    )


@dataclass(frozen=True, slots=True)
class TraceErrorPoint:
    locator: str
    stage: str
    quantity: str
    unit: str
    status: str
    expected: float | None
    actual: float | None
    absolute_error: float | None
    relative_error: float | None
    tolerance: TraceTolerance

    def as_dict(self) -> dict[str, Any]:
        return {
            "locator": self.locator,
            "stage": self.stage,
            "quantity": self.quantity,
            "unit": self.unit,
            "status": self.status,
            "expected": self.expected,
            "actual": self.actual,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
            "tolerance": self.tolerance.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class TraceComparisonReport:
    reference_name: str
    actual_name: str
    points: tuple[TraceErrorPoint, ...]
    strict: bool

    @property
    def passed(self) -> int:
        return sum(point.status == "pass" for point in self.points)

    @property
    def failed(self) -> int:
        return sum(
            point.status in {"fail", "unit_mismatch"} for point in self.points
        )

    @property
    def missing(self) -> int:
        return sum(point.status == "missing" for point in self.points)

    @property
    def unexpected(self) -> int:
        return sum(point.status == "unexpected" for point in self.points)

    @property
    def success(self) -> bool:
        if self.failed or self.missing:
            return False
        return not self.strict or self.unexpected == 0

    def stage_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for point in self.points:
            stage = summary.setdefault(
                point.stage,
                {
                    "pass": 0,
                    "fail": 0,
                    "unit_mismatch": 0,
                    "missing": 0,
                    "unexpected": 0,
                },
            )
            stage[point.status] += 1
        return summary

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPARISON_SCHEMA,
            "reference_name": self.reference_name,
            "actual_name": self.actual_name,
            "strict": self.strict,
            "success": self.success,
            "counts": {
                "passed": self.passed,
                "failed": self.failed,
                "missing": self.missing,
                "unexpected": self.unexpected,
            },
            "stage_summary": self.stage_summary(),
            "points": [point.as_dict() for point in self.points],
        }


def _resolve_tolerances(
    overrides: Mapping[str, TraceTolerance] | None,
) -> dict[str, TraceTolerance]:
    tolerances = dict(DEFAULT_TRACE_TOLERANCES)
    if overrides:
        for stage, tolerance in overrides.items():
            if stage not in STAGES:
                raise ValueError(f"unknown trace stage {stage!r}")
            tolerances[stage] = tolerance
    return tolerances


def compare_numerical_traces(
    reference: NumericalTrace,
    actual: NumericalTrace,
    *,
    tolerances: Mapping[str, TraceTolerance] | None = None,
    strict: bool = False,
) -> TraceComparisonReport:
    """Compare two traces point-by-point with stage-specific tolerances."""

    resolved = _resolve_tolerances(tolerances)
    reference_by_locator = {point.locator: point for point in reference.points}
    actual_by_locator = {point.locator: point for point in actual.points}
    results: list[TraceErrorPoint] = []

    for locator, expected in reference_by_locator.items():
        tolerance = resolved[expected.stage]
        observed = actual_by_locator.get(locator)
        if observed is None:
            results.append(
                TraceErrorPoint(
                    locator=locator,
                    stage=expected.stage,
                    quantity=expected.quantity,
                    unit=expected.unit,
                    status="missing",
                    expected=expected.value,
                    actual=None,
                    absolute_error=None,
                    relative_error=None,
                    tolerance=tolerance,
                )
            )
            continue
        if observed.unit != expected.unit:
            results.append(
                TraceErrorPoint(
                    locator=locator,
                    stage=expected.stage,
                    quantity=expected.quantity,
                    unit=f"{expected.unit} != {observed.unit}",
                    status="unit_mismatch",
                    expected=expected.value,
                    actual=observed.value,
                    absolute_error=None,
                    relative_error=None,
                    tolerance=tolerance,
                )
            )
            continue
        absolute_error = abs(observed.value - expected.value)
        relative_error = (
            absolute_error / abs(expected.value) if expected.value != 0.0 else None
        )
        status = (
            "pass"
            if absolute_error <= tolerance.limit(expected.value)
            else "fail"
        )
        results.append(
            TraceErrorPoint(
                locator=locator,
                stage=expected.stage,
                quantity=expected.quantity,
                unit=expected.unit,
                status=status,
                expected=expected.value,
                actual=observed.value,
                absolute_error=absolute_error,
                relative_error=relative_error,
                tolerance=tolerance,
            )
        )

    for locator, observed in actual_by_locator.items():
        if locator in reference_by_locator:
            continue
        results.append(
            TraceErrorPoint(
                locator=locator,
                stage=observed.stage,
                quantity=observed.quantity,
                unit=observed.unit,
                status="unexpected",
                expected=None,
                actual=observed.value,
                absolute_error=None,
                relative_error=None,
                tolerance=resolved[observed.stage],
            )
        )

    results.sort(key=lambda point: point.locator)
    return TraceComparisonReport(
        reference_name=reference.name,
        actual_name=actual.name,
        points=tuple(results),
        strict=strict,
    )
