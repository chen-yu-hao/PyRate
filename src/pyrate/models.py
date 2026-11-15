"""Typed domain models for molecular partition functions and rate calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .constants import DEFAULT_STANDARD_PRESSURE_PA
from .elements import atomic_mass_amu, canonical_symbol


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
    mass_convention: str = "iupac"

    def __post_init__(self) -> None:
        raw_symbol = self.symbol
        symbol = canonical_symbol(raw_symbol)
        object.__setattr__(self, "symbol", symbol)
        mass = (
            atomic_mass_amu(raw_symbol, mass_convention=self.mass_convention)
            if self.mass_amu is None
            else float(self.mass_amu)
        )
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
    def from_mapping(cls, data: Mapping[str, Any], *, default_mass_convention: str = "iupac") -> "Atom":
        xyz = data.get("xyz") or data.get("coordinates_angstrom")
        if xyz is None or len(xyz) != 3:
            raise ValueError("each atom requires xyz = [x, y, z]")
        return cls(
            symbol=str(data.get("symbol") or data.get("element") or data.get("label")),
            label=str(data["label"]) if data.get("label") is not None else None,
            x_angstrom=float(xyz[0]),
            y_angstrom=float(xyz[1]),
            z_angstrom=float(xyz[2]),
            mass_amu=float(data["mass_amu"]) if data.get("mass_amu") is not None else None,
            mass_convention=str(data.get("mass_convention", default_mass_convention)),
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
class HinderedRotor:
    """One-dimensional hindered internal rotor specification."""

    frequency_cm1: float
    barrier_kj_mol: float
    symmetry_number: int = 1
    moment_amu_angstrom2: float | None = None
    replaces_lowest_frequency: bool = False

    def __post_init__(self) -> None:
        if self.frequency_cm1 <= 0.0:
            raise ValueError("hindered rotor frequency_cm1 must be positive")
        if self.barrier_kj_mol < 0.0:
            raise ValueError("hindered rotor barrier_kj_mol cannot be negative")
        if self.symmetry_number <= 0:
            raise ValueError("hindered rotor symmetry_number must be positive")
        if self.moment_amu_angstrom2 is not None and self.moment_amu_angstrom2 <= 0.0:
            raise ValueError("hindered rotor moment_amu_angstrom2 must be positive")
        object.__setattr__(self, "frequency_cm1", float(self.frequency_cm1))
        object.__setattr__(self, "barrier_kj_mol", float(self.barrier_kj_mol))
        object.__setattr__(self, "symmetry_number", int(self.symmetry_number))
        if self.moment_amu_angstrom2 is not None:
            object.__setattr__(self, "moment_amu_angstrom2", float(self.moment_amu_angstrom2))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "HinderedRotor":
        return cls(
            frequency_cm1=float(data["frequency_cm1"]),
            barrier_kj_mol=float(data.get("barrier_kj_mol", 0.0)),
            symmetry_number=int(data.get("symmetry_number", 1)),
            moment_amu_angstrom2=(
                float(data["moment_amu_angstrom2"])
                if data.get("moment_amu_angstrom2") is not None
                else None
            ),
            replaces_lowest_frequency=bool(data.get("replaces_lowest_frequency", False)),
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
    frequency_scale: float = 1.0
    frequency_floor_cm1: float | None = None
    anharmonicities_cm1: tuple[float, ...] = ()
    hindered_rotors: tuple[HinderedRotor, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("species name cannot be empty")
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "frequencies_cm1", _as_float_tuple(self.frequencies_cm1))
        object.__setattr__(self, "anharmonicities_cm1", _as_float_tuple(self.anharmonicities_cm1))
        object.__setattr__(self, "hindered_rotors", tuple(self.hindered_rotors))
        object.__setattr__(self, "electronic_states", tuple(self.electronic_states))
        if not self.electronic_states:
            object.__setattr__(self, "electronic_states", (ElectronicState(),))
        if self.symmetry_number <= 0:
            raise ValueError("symmetry_number must be positive")
        if self.frequency_scale <= 0.0:
            raise ValueError("frequency_scale must be positive")
        object.__setattr__(self, "frequency_scale", float(self.frequency_scale))
        if self.frequency_floor_cm1 is not None:
            if self.frequency_floor_cm1 < 0.0:
                raise ValueError("frequency_floor_cm1 cannot be negative")
            object.__setattr__(self, "frequency_floor_cm1", float(self.frequency_floor_cm1))
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
        return tuple(value * self.frequency_scale for value in self.frequencies_cm1 if value < 0.0)

    @property
    def positive_frequencies_cm1(self) -> tuple[float, ...]:
        scaled = [value * self.frequency_scale for value in self.frequencies_cm1 if value > 0.0]
        if self.frequency_floor_cm1 is not None:
            floor = self.frequency_floor_cm1
            scaled = [max(value, floor) for value in scaled]
        return tuple(scaled)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default_name: str = "species") -> "Species":
        mass_convention = str(data.get("mass_convention", "iupac"))
        atoms = tuple(
            Atom.from_mapping(item, default_mass_convention=mass_convention)
            for item in data.get("atoms", ())
        )
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

        rotors_data = data.get("hindered_rotors", ())
        rotors = tuple(HinderedRotor.from_mapping(item) for item in rotors_data)

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
            frequency_scale=float(data.get("frequency_scale", 1.0)),
            frequency_floor_cm1=(
                float(data["frequency_floor_cm1"])
                if data.get("frequency_floor_cm1") is not None
                else None
            ),
            anharmonicities_cm1=tuple(float(value) for value in data.get("anharmonicities_cm1", ())),
            hindered_rotors=rotors,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class PathPoint:
    """A generalized transition state located at reaction coordinate ``s``."""

    coordinate: float
    dividing_surface: Species
    effective_mass_amu: float | None = None
    curvatures_bohr_inv: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate", float(self.coordinate))
        if self.effective_mass_amu is not None:
            if self.effective_mass_amu <= 0.0:
                raise ValueError("effective_mass_amu must be positive")
            object.__setattr__(self, "effective_mass_amu", float(self.effective_mass_amu))
        object.__setattr__(self, "curvatures_bohr_inv", _as_float_tuple(self.curvatures_bohr_inv))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], index: int) -> "PathPoint":
        return cls(
            coordinate=float(data["coordinate"]),
            dividing_surface=Species.from_mapping(data, default_name=f"path-{index}"),
            effective_mass_amu=(
                float(data["effective_mass_amu"])
                if data.get("effective_mass_amu") is not None
                else None
            ),
            curvatures_bohr_inv=tuple(
                float(value) for value in data.get("curvatures_bohr_inv", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class ReactionPath:
    """Ordered reaction-path dividing surfaces."""

    points: tuple[PathPoint, ...]
    coordinate_unit: str = "angstrom"
    reduced_mass_amu: float = 1.0

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
        if self.reduced_mass_amu <= 0.0:
            raise ValueError("reduced_mass_amu must be positive")
        object.__setattr__(self, "reduced_mass_amu", float(self.reduced_mass_amu))


@dataclass(frozen=True, slots=True)
class TunnelingProfile:
    """One-dimensional barrier profile for WKB/ZCT/SCT-style transmission."""

    coordinates_angstrom: tuple[float, ...]
    energies_kj_mol: tuple[float, ...]
    reduced_mass_amu: float
    energy_grid_points: int = 400
    effective_masses_amu: tuple[float, ...] | None = None

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
        if self.effective_masses_amu is not None:
            eff_masses = _as_float_tuple(self.effective_masses_amu)
            if len(eff_masses) != len(coordinates):
                raise ValueError("effective_masses_amu must match coordinates_angstrom length")
            if any(m <= 0.0 for m in eff_masses):
                raise ValueError("effective_masses_amu values must be positive")
            object.__setattr__(self, "effective_masses_amu", eff_masses)
        object.__setattr__(self, "coordinates_angstrom", coordinates)
        object.__setattr__(self, "energies_kj_mol", energies)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TunnelingProfile":
        return cls(
            coordinates_angstrom=tuple(float(value) for value in data["coordinates_angstrom"]),
            energies_kj_mol=tuple(float(value) for value in data["energies_kj_mol"]),
            reduced_mass_amu=float(data["reduced_mass_amu"]),
            energy_grid_points=int(data.get("energy_grid_points", 400)),
            effective_masses_amu=(
                tuple(float(value) for value in data["effective_masses_amu"])
                if data.get("effective_masses_amu") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class Reaction:
    """A gas-phase reaction and its transition-state model."""

    reactants: tuple[Species, ...]
    transition_state: Species
    products: tuple[Species, ...] = ()
    path: ReactionPath | None = None
    standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA

    def __post_init__(self) -> None:
        reactants = tuple(self.reactants)
        products = tuple(self.products)
        if not reactants:
            raise ValueError("reaction requires at least one reactant")
        if len(reactants) > 2:
            raise NotImplementedError("the current public API supports uni- and bimolecular rates")
        if self.standard_pressure_pa <= 0.0:
            raise ValueError("standard pressure must be positive")
        if len(self.transition_state.imaginary_frequencies_cm1) > 1:
            raise ValueError("transition state may contain at most one imaginary frequency")
        object.__setattr__(self, "reactants", reactants)
        object.__setattr__(self, "products", products)

    @property
    def molecularity(self) -> int:
        return len(self.reactants)


_SUPPORTED_METHODS = {"tst", "cvt", "cus", "muvt", "us", "all"}
_SUPPORTED_TUNNELING = {
    "none",
    "wigner",
    "wkb",
    "zct",
    "mepsag",
    "sct",
    "cd-scsag",
    "eckart",
    "omt",
    "comt",
    "muomt",
}


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
        if method not in _SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {sorted(_SUPPORTED_METHODS)}")
        if tunneling not in _SUPPORTED_TUNNELING:
            raise ValueError(f"tunneling must be one of {sorted(_SUPPORTED_TUNNELING)}")
        if self.output_rate_unit == "":
            raise ValueError("output_rate_unit cannot be empty")
        if tunneling == "wkb" and self.tunneling_profile is None:
            raise ValueError("wkb tunneling requires a tunneling profile")
        object.__setattr__(self, "temperatures_k", temperatures)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "tunneling", tunneling)

