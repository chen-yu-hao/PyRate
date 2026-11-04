"""Typed domain models for molecular partition functions and rate calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .constants import DEFAULT_STANDARD_PRESSURE_PA, validate_constants_profile
from .elements import atomic_mass_amu, canonical_symbol, isotope_mass_amu


def _as_float_tuple(values: Iterable[float] | None) -> tuple[float, ...]:
    if values is None:
        return ()
    return tuple(float(value) for value in values)


@dataclass(frozen=True, slots=True)
class Atom:
    """An atom with Cartesian coordinates in ångström and mass in amu."""

    symbol: str
    x_angstrom: float
    y_angstrom: float
    z_angstrom: float
    mass_amu: float | None = None
    label: str | None = None
    isotope: int | None = None

    def __post_init__(self) -> None:
        symbol = canonical_symbol(self.symbol)
        object.__setattr__(self, "symbol", symbol)
        mass = (isotope_mass_amu(symbol, self.isotope) if self.isotope is not None\n                else atomic_mass_amu(symbol)) if self.mass_amu is None else float(self.mass_amu)
        if mass <= 0.0:
            raise ValueError("atom mass must be positive")
        object.__setattr__(self, "mass_amu", mass)
        object.__setattr__(self, "x_angstrom", float(self.x_angstrom))
        object.__setattr__(self, "y_angstrom", float(self.y_angstrom))
        object.__setattr__(self, "z_angstrom", float(self.z_angstrom))

    @property
    def coordinates_angstrom(self) -> tuple[float, float, float]:
        return (self.x_angstrom, self.y_angstrom, self.z_angstrom)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Atom":
        xyz = data.get("xyz") or data.get("coordinates_angstrom")
        if xyz is None or len(xyz) != 3:
            raise ValueError("each atom requires xyz = [x, y, z]")
        return cls(
            symbol=str(data.get("symbol") or data.get("element") or data.get("label")),
            label=str(data["label"]) if data.get("label") is not None else None,
            x_angstrom=float(xyz[0]),
            y_angstrom=float(xyz[1]),
            z_angstrom=float(xyz[2]),
            mass_amu=float(data["mass_amu"]) if data.get("mass_amu") is not None else None,\n            isotope=int(data["isotope"]) if data.get("isotope") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class ElectronicState:
    """An electronic state relative to the ground state."""

    degeneracy: int = 1
    energy_kj_mol: float = 0.0

    def __post_init__(self) -> None:
        if self.degeneracy <= 0:
            raise ValueError("electronic-state degeneracy must be positive")
        if self.energy_kj_mol < 0.0:
            raise ValueError("electronic-state relative energy cannot be negative")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ElectronicState":
        return cls(
            degeneracy=int(data.get("degeneracy", 1)),
            energy_kj_mol=float(data.get("energy_kj_mol", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class Species:
    """Molecular information required by a rigid-rotor/harmonic-oscillator model.

    ``electronic_energy_kj_mol`` is the potential/electronic energy without
    vibrational zero-point energy.  Positive vibrational frequencies contribute
    zero-point energy when ``include_zpe`` is true.  Negative frequencies are
    excluded from the vibrational partition function and may be used to mark a
    transition-state reaction coordinate.
    """

    name: str
    atoms: tuple[Atom, ...] = ()
    mass_amu_override: float | None = None
    electronic_energy_kj_mol: float = 0.0
    frequencies_cm1: tuple[float, ...] = ()
    symmetry_number: int = 1
    electronic_states: tuple[ElectronicState, ...] = (ElectronicState(),)
    moments_of_inertia_amu_a2: tuple[float, ...] | None = None
    rotor_type: str = "auto"
    include_zpe: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("species name cannot be empty")
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "frequencies_cm1", _as_float_tuple(self.frequencies_cm1))
        object.__setattr__(self, "electronic_states", tuple(self.electronic_states))
        if not self.electronic_states:
            object.__setattr__(self, "electronic_states", (ElectronicState(),))
        if self.symmetry_number <= 0:
            raise ValueError("symmetry_number must be positive")
        if self.rotor_type not in {"auto", "atomic", "linear", "nonlinear"}:
            raise ValueError("rotor_type must be auto, atomic, linear, or nonlinear")
        if self.moments_of_inertia_amu_a2 is not None:
            moments = _as_float_tuple(self.moments_of_inertia_amu_a2)
            if len(moments) not in {1, 2, 3}:
                raise ValueError("moments_of_inertia_amu_a2 must contain 1, 2, or 3 values")
            if any(moment < 0.0 for moment in moments):
                raise ValueError("moments of inertia cannot be negative")
            object.__setattr__(self, "moments_of_inertia_amu_a2", moments)
        if self.mass_amu_override is not None and self.mass_amu_override <= 0.0:
            raise ValueError("mass_amu must be positive")
        if not self.atoms and self.mass_amu_override is None:
            raise ValueError(
                f"species {self.name!r} requires atoms or mass_amu"
            )
        if not self.atoms and self.moments_of_inertia_amu_a2 is None and self.rotor_type != "atomic":
            raise ValueError(
                f"species {self.name!r} without atoms requires moments_of_inertia_amu_a2 "
                "or rotor_type = 'atomic'"
            )

    @property
    def mass_amu(self) -> float:
        if self.mass_amu_override is not None:
            return float(self.mass_amu_override)
        return sum(float(atom.mass_amu) for atom in self.atoms)

    @property
    def imaginary_frequencies_cm1(self) -> tuple[float, ...]:
        return tuple(value for value in self.frequencies_cm1 if value < 0.0)

    @property
    def positive_frequencies_cm1(self) -> tuple[float, ...]:
        return tuple(value for value in self.frequencies_cm1 if value > 0.0)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default_name: str = "species") -> "Species":
        atoms = tuple(Atom.from_mapping(item) for item in data.get("atoms", ()))
        states_data = data.get("electronic_states")
        if states_data is None:
            states = (
                ElectronicState(
                    degeneracy=int(data.get("electronic_degeneracy", 1)),
                    energy_kj_mol=0.0,
                ),
            )
        else:
            states = tuple(ElectronicState.from_mapping(item) for item in states_data)

        frequencies = list(data.get("frequencies_cm1", ()))
        imaginary = data.get("imaginary_frequency_cm1")
        if imaginary is not None:
            imag_value = float(imaginary)
            frequencies.append(-abs(imag_value))

        return cls(
            name=str(data.get("name", default_name)),
            atoms=atoms,
            mass_amu_override=(float(data["mass_amu"]) if data.get("mass_amu") is not None else None),
            electronic_energy_kj_mol=float(
                data.get("electronic_energy_kj_mol", data.get("energy_kj_mol", 0.0))
            ),
            frequencies_cm1=tuple(float(value) for value in frequencies),
            symmetry_number=int(data.get("symmetry_number", 1)),
            electronic_states=states,
            moments_of_inertia_amu_a2=(
                tuple(float(value) for value in data["moments_of_inertia_amu_a2"])
                if data.get("moments_of_inertia_amu_a2") is not None
                else None
            ),
            rotor_type=str(data.get("rotor_type", "auto")).lower(),
            include_zpe=bool(data.get("include_zpe", True)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class PathPoint:
    """A generalized transition state located at reaction coordinate ``s``."""

    coordinate: float
    dividing_surface: Species

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate", float(self.coordinate))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], index: int) -> "PathPoint":
        return cls(
            coordinate=float(data["coordinate"]),
            dividing_surface=Species.from_mapping(data, default_name=f"path-{index}"),
        )


@dataclass(frozen=True, slots=True)
class ReactionPath:
    """Ordered reaction-path dividing surfaces."""

    points: tuple[PathPoint, ...]
    coordinate_unit: str = "angstrom"

    def __post_init__(self) -> None:
        points = tuple(sorted(self.points, key=lambda item: item.coordinate))
        if len(points) < 3:
            raise ValueError("a CVT reaction path requires at least three points")
        coordinates = [point.coordinate for point in points]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("reaction-path coordinates must be unique")
        object.__setattr__(self, "points", points)
        if self.coordinate_unit not in {"angstrom", "bohr", "dimensionless"}:
            raise ValueError("coordinate_unit must be angstrom, bohr, or dimensionless")


@dataclass(frozen=True, slots=True)
class TunnelingProfile:
    """One-dimensional barrier profile for WKB/ZCT-style transmission."""

    coordinates_angstrom: tuple[float, ...]
    energies_kj_mol: tuple[float, ...]
    reduced_mass_amu: float
    energy_grid_points: int = 400

    def __post_init__(self) -> None:
        coordinates = _as_float_tuple(self.coordinates_angstrom)
        energies = _as_float_tuple(self.energies_kj_mol)
        if len(coordinates) != len(energies) or len(coordinates) < 3:
            raise ValueError("tunneling profile requires matching arrays with at least 3 points")
        if any(b <= a for a, b in zip(coordinates, coordinates[1:])):
            raise ValueError("tunneling coordinates must be strictly increasing")
        if self.reduced_mass_amu <= 0.0:
            raise ValueError("reduced_mass_amu must be positive")
        if self.energy_grid_points < 20:
            raise ValueError("energy_grid_points must be at least 20")
        object.__setattr__(self, "coordinates_angstrom", coordinates)
        object.__setattr__(self, "energies_kj_mol", energies)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TunnelingProfile":
        return cls(
            coordinates_angstrom=tuple(float(value) for value in data["coordinates_angstrom"]),
            energies_kj_mol=tuple(float(value) for value in data["energies_kj_mol"]),
            reduced_mass_amu=float(data["reduced_mass_amu"]),
            energy_grid_points=int(data.get("energy_grid_points", 400)),
        )


@dataclass(frozen=True, slots=True)
class Reaction:
    """A gas-phase reaction and its transition-state model."""

    reactants: tuple[Species, ...]
    transition_state: Species
    products: tuple[Species, ...] = ()
    path: ReactionPath | None = None
    standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA\n    constants_profile: str = "codata2018"

    def __post_init__(self) -> None:
        reactants = tuple(self.reactants)
        products = tuple(self.products)
        if not reactants:
            raise ValueError("reaction requires at least one reactant")
        if len(reactants) > 2:
            raise NotImplementedError("the current public API supports uni- and bimolecular rates")
        object.__setattr__(self, "constants_profile", validate_constants_profile(self.constants_profile))\n        if self.standard_pressure_pa <= 0.0:
            raise ValueError("standard pressure must be positive")
        if len(self.transition_state.imaginary_frequencies_cm1) > 1:
            raise ValueError("transition state may contain at most one imaginary frequency")
        object.__setattr__(self, "reactants", reactants)
        object.__setattr__(self, "products", products)

    @property
    def molecularity(self) -> int:
        return len(self.reactants)


@dataclass(frozen=True, slots=True)
class CalculationOptions:
    """Runtime choices independent of molecular data."""

    temperatures_k: tuple[float, ...]
    method: str = "tst"
    tunneling: str = "none"
    output_rate_unit: str = "auto"
    tunneling_profile: TunnelingProfile | None = None

    def __post_init__(self) -> None:
        temperatures = tuple(float(value) for value in self.temperatures_k)
        if not temperatures or any(value <= 0.0 for value in temperatures):
            raise ValueError("temperatures must contain positive values")
        method = self.method.lower()
        tunneling = self.tunneling.lower()
        if method not in {"tst", "cvt"}:
            raise ValueError("method must be tst or cvt")
        if tunneling not in {"none", "wigner", "wkb", "zct"}:
            raise ValueError("tunneling must be none, wigner, wkb, or zct")
        if method == "cvt" and self.output_rate_unit == "":
            raise ValueError("output_rate_unit cannot be empty")
        if tunneling in {"wkb", "zct"} and self.tunneling_profile is None:
            raise ValueError("wkb/zct tunneling requires a tunneling profile")
        object.__setattr__(self, "temperatures_k", temperatures)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "tunneling", tunneling)
