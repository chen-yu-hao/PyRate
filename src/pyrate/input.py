"""Modern TOML input loading and validated legacy execution mapping."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .constants import BAR_PA
from .legacy import LegacyInput, load_legacy
from .legacy_execution import convert_legacy_to_models
from .models import (
    CalculationOptions,
    PathPoint,
    Reaction,
    ReactionPath,
    Species,
    TunnelingProfile,
)


@dataclass(frozen=True, slots=True)
class LoadedCalculation:
    name: str
    reaction: Reaction
    options: CalculationOptions
    raw: Mapping[str, Any]
    source: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "format": (
                "pyrate-toml"
                if self.source is None or self.source.endswith(".toml")
                else "polyrate-legacy-executable-subset"
            ),
            "source": self.source,
            "name": self.name,
            "reactants": [species.name for species in self.reaction.reactants],
            "transition_state": self.reaction.transition_state.name,
            "products": [species.name for species in self.reaction.products],
            "molecularity": self.reaction.molecularity,
            "temperatures_k": list(self.options.temperatures_k),
            "method": self.options.method,
            "tunneling": self.options.tunneling,
            "path_points": len(self.reaction.path.points) if self.reaction.path else 0,
            "standard_pressure_pa": self.reaction.standard_pressure_pa,
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a TOML table")
    return value


def parse_toml_data(
    data: Mapping[str, Any], *, source: str | None = None
) -> LoadedCalculation:
    calculation = _mapping(data.get("calculation", {}), "calculation")
    name = str(calculation.get("name", "PyRate calculation"))

    reactant_tables = data.get("reactants")
    if not isinstance(reactant_tables, list) or not reactant_tables:
        raise ValueError("input requires one or two [[reactants]] tables")
    reactants = tuple(
        Species.from_mapping(_mapping(table, f"reactants[{index}]"), f"reactant-{index + 1}")
        for index, table in enumerate(reactant_tables)
    )

    transition_table = _mapping(data.get("transition_state"), "transition_state")
    transition_state = Species.from_mapping(transition_table, "transition-state")

    product_tables = data.get("products", [])
    if not isinstance(product_tables, list):
        raise ValueError("products must use [[products]] array tables")
    products = tuple(
        Species.from_mapping(_mapping(table, f"products[{index}]"), f"product-{index + 1}")
        for index, table in enumerate(product_tables)
    )

    path_model: ReactionPath | None = None
    path_table = data.get("path")
    if path_table is not None:
        path_mapping = _mapping(path_table, "path")
        point_tables = path_mapping.get("points")
        if not isinstance(point_tables, list):
            raise ValueError("[path] requires one or more [[path.points]] tables")
        points = tuple(
            PathPoint.from_mapping(
                _mapping(table, f"path.points[{index}]"), index
            )
            for index, table in enumerate(point_tables)
        )
        path_model = ReactionPath(
            points=points,
            coordinate_unit=str(path_mapping.get("coordinate_unit", "angstrom")).lower(),
        )

    pressure_pa: float
    if calculation.get("standard_pressure_pa") is not None:
        pressure_pa = float(calculation["standard_pressure_pa"])
    else:
        pressure_pa = float(calculation.get("standard_pressure_bar", 1.0)) * BAR_PA

    reaction = Reaction(
        reactants=reactants,
        transition_state=transition_state,
        products=products,
        path=path_model,
        standard_pressure_pa=pressure_pa,
    )

    temperatures = calculation.get("temperatures_k")
    if not isinstance(temperatures, list):
        raise ValueError("[calculation] requires temperatures_k = [...] ")

    tunneling_table = _mapping(data.get("tunneling", {}), "tunneling")
    tunneling_method = str(
        calculation.get("tunneling", tunneling_table.get("method", "none"))
    ).lower()
    tunneling_profile: TunnelingProfile | None = None
    if tunneling_method in {"wkb", "zct"}:
        profile_data = tunneling_table.get("profile", tunneling_table)
        tunneling_profile = TunnelingProfile.from_mapping(
            _mapping(profile_data, "tunneling.profile")
        )

    options = CalculationOptions(
        temperatures_k=tuple(float(value) for value in temperatures),
        method=str(calculation.get("method", "tst")),
        tunneling=tunneling_method,
        output_rate_unit=str(calculation.get("rate_unit", "auto")),
        tunneling_profile=tunneling_profile,
    )
    return LoadedCalculation(
        name=name, reaction=reaction, options=options, raw=data, source=source
    )


def load_toml(path: str | Path) -> LoadedCalculation:
    source = Path(path)
    with source.open("rb") as handle:
        data = tomllib.load(handle)
    return parse_toml_data(data, source=str(source))


def load_legacy_calculation(path: str | Path) -> LoadedCalculation:
    """Load a legacy deck only when all required semantics are mapped."""

    source = Path(path)
    document = load_legacy(source)
    converted = convert_legacy_to_models(document, source)
    return LoadedCalculation(
        name=converted.name,
        reaction=converted.reaction,
        options=converted.options,
        raw=converted.raw,
        source=str(source),
    )


def load_calculation(path: str | Path) -> LoadedCalculation:
    """Load either native TOML or the executable Polyrate legacy subset."""

    source = Path(path)
    if source.suffix.lower() == ".toml":
        return load_toml(source)
    return load_legacy_calculation(source)


def load_input(path: str | Path) -> LoadedCalculation | LegacyInput:
    """Load TOML as a calculation and legacy input as a retaining syntax tree."""

    source = Path(path)
    if source.suffix.lower() == ".toml":
        return load_toml(source)
    return load_legacy(source)
