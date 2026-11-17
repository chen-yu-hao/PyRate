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
    intrinsic_frequencies_cm1: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate", float(self.coordinate))
        if self.effective_mass_amu is not None:
            if self.effective_mass_amu <= 0.0:
                raise ValueError("effective_mass_amu must be positive")
            object.__setattr__(self, "effective_mass_amu", float(self.effective_mass_amu))
        object.__setattr__(self, "curvatures_bohr_inv", _as_float_tuple(self.curvatures_bohr_inv))
        frequencies = _as_float_tuple(self.intrinsic_frequencies_cm1)
        if any(value == 0.0 for value in frequencies):
            raise ValueError("intrinsic coordinate frequencies cannot be zero")
        object.__setattr__(self, "intrinsic_frequencies_cm1", frequencies)

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
            intrinsic_frequencies_cm1=tuple(
                float(value)
                for value in data.get(
                    "intrinsic_frequencies_cm1",
                    data.get("coordinate_frequencies_cm1", ()),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ReactionPath:
    """Ordered reaction-path dividing surfaces."""

    points: tuple[PathPoint, ...]
    coordinate_unit: str = "angstrom"
    reduced_mass_amu: float = 1.0
    coordinate_system: str = "cartesian"

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
        coordinate_system = str(self.coordinate_system).strip().lower().replace("-", "_")
        if coordinate_system not in {
            "cartesian",
            "intrinsic",
            "mass_scaled",
            "dimensionless",
            "curvilinear",
            "curv1",
            "curv2",
            "curv3",
        }:
            raise ValueError(
                "coordinate_system must be cartesian, curvilinear, curv1, curv2, curv3, "
                "intrinsic, mass_scaled, or dimensionless"
            )
        object.__setattr__(self, "coordinate_system", coordinate_system)


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
class DualLevelCorrection:
    """High-minus-low electronic energies along a reaction path.

    ``method`` follows the Polyrate terminology: ``ispe`` interpolates
    single-point corrections on the full path, ``ivtst`` uses the corrected
    variational path, and ``ioc`` applies the correction at one optimized
    dividing surface to all path points.
    """

    method: str = "ivtst"
    low_level_energies_kj_mol: tuple[float, ...] = ()
    high_level_energies_kj_mol: tuple[float, ...] = ()
    energy_coordinates: tuple[float, ...] = ()
    coordinates: tuple[float, ...] = ()
    energy_shift_kj_mol: float = 0.0
    reference_coordinate: float | None = None
    interpolation: str = "linear"
    enabled: bool = True

    def __post_init__(self) -> None:
        method = str(self.method).lower().replace("-", "")
        if method == "ic":
            method = "ioc"
        if method in {"ic", "ioc"}:
            method = "ioc"
        if method in {"ivtst0", "ivtst-0", "ivtst1", "ivtst-1", "ivtstm", "ivtst-m"}:
            method = method.replace("-", "")
        if method not in {"ispe", "ivtst", "ioc", "ivtst0", "ivtst1", "ivtstm"}:
            raise ValueError("dual-level method must be ispe, ivtst, ivtst-0, ivtst-1, ivtst-m, or ioc")
        interpolation = str(self.interpolation).lower()
        if interpolation not in {"linear", "cubic"}:
            raise ValueError("dual-level interpolation must be linear or cubic")
        low = _as_float_tuple(self.low_level_energies_kj_mol)
        high = _as_float_tuple(self.high_level_energies_kj_mol)
        coordinates = _as_float_tuple(self.energy_coordinates or self.coordinates)
        if bool(low) != bool(high) or (low and len(low) != len(high)):
            raise ValueError("dual-level low/high energy arrays must be supplied together")
        if coordinates and len(coordinates) != len(low):
            raise ValueError("energy_coordinates must match dual-level energy arrays")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "low_level_energies_kj_mol", low)
        object.__setattr__(self, "high_level_energies_kj_mol", high)
        object.__setattr__(self, "energy_coordinates", coordinates)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "energy_shift_kj_mol", float(self.energy_shift_kj_mol))
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(self, "enabled", bool(self.enabled))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DualLevelCorrection":
        return cls(
            method=str(data.get("method", data.get("model", "ivtst"))),
            low_level_energies_kj_mol=tuple(
                float(v) for v in data.get("low_level_energies_kj_mol", data.get("low_energies_kj_mol", ()))
            ),
            high_level_energies_kj_mol=tuple(
                float(v) for v in data.get("high_level_energies_kj_mol", data.get("high_energies_kj_mol", ()))
            ),
            energy_coordinates=tuple(float(v) for v in data.get("energy_coordinates", data.get("coordinates", ()))),
            coordinates=tuple(float(v) for v in data.get("coordinates", data.get("energy_coordinates", ()))),
            energy_shift_kj_mol=float(data.get("energy_shift_kj_mol", data.get("correction_kj_mol", 0.0))),
            reference_coordinate=(
                float(data["reference_coordinate"])
                if data.get("reference_coordinate") is not None
                else None
            ),
            interpolation=str(data.get("interpolation", "linear")),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True, slots=True)
class VrcSample:
    """One deterministic VRC loose-transition-state quadrature facet."""

    coordinate: float = 0.0
    activation_gibbs_kj_mol: float = 0.0
    energy_kj_mol: float | None = None
    weight: float = 1.0
    degeneracy: float = 1.0
    frequencies_cm1: tuple[float, ...] = ()
    separation_angstrom: float | None = None
    orientation: float | None = None

    def __post_init__(self) -> None:
        if self.weight <= 0.0 or self.degeneracy <= 0.0:
            raise ValueError("VRC sample weight and degeneracy must be positive")
        object.__setattr__(self, "coordinate", float(self.coordinate))
        energy = self.activation_gibbs_kj_mol if self.energy_kj_mol is None else self.energy_kj_mol
        object.__setattr__(self, "activation_gibbs_kj_mol", float(energy))
        object.__setattr__(self, "energy_kj_mol", float(energy))
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "degeneracy", float(self.degeneracy))
        object.__setattr__(self, "frequencies_cm1", _as_float_tuple(self.frequencies_cm1))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "VrcSample":
        return cls(
            coordinate=float(data.get("coordinate", data.get("separation_angstrom", 0.0))),
            activation_gibbs_kj_mol=float(data.get("activation_gibbs_kj_mol", data.get("energy_kj_mol", 0.0))),
            energy_kj_mol=(float(data["energy_kj_mol"]) if data.get("energy_kj_mol") is not None else None),
            weight=float(data.get("weight", 1.0)),
            degeneracy=float(data.get("degeneracy", 1.0)),
            frequencies_cm1=tuple(float(v) for v in data.get("frequencies_cm1", ())),
            separation_angstrom=(float(data["separation_angstrom"]) if data.get("separation_angstrom") is not None else None),
            orientation=(float(data["orientation"]) if data.get("orientation") is not None else None),
        )


@dataclass(frozen=True, slots=True)
class VrcModel:
    """VRC-VTST loose dividing-surface quadrature configuration."""

    samples: tuple[VrcSample, ...]
    symmetry_factor: float = 1.0
    sampling_method: str = "deterministic"
    enabled: bool = True

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples:
            raise ValueError("VRC model requires at least one sample")
        if self.symmetry_factor <= 0.0:
            raise ValueError("VRC symmetry_factor must be positive")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "symmetry_factor", float(self.symmetry_factor))
        object.__setattr__(self, "enabled", bool(self.enabled))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "VrcModel":
        raw = data.get("samples", data.get("facets", ()))
        if not isinstance(raw, (list, tuple)):
            raise ValueError("vrc.samples must be an array")
        return cls(
            samples=tuple(VrcSample.from_mapping(item) for item in raw),
            symmetry_factor=float(data.get("symmetry_factor", data.get("symmetry_number", 1.0))),
            sampling_method=str(data.get("sampling_method", "deterministic")),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True, slots=True)
class PressureDependence:
    """SS-QRRK/master-equation pressure grid and collision model."""

    pressures_pa: tuple[float, ...]
    collider: str = "M"
    collision_frequency_s_1_at_1bar: float = 1.0e9
    collision_frequency_s1_at_1bar: float | None = None
    energy_transfer_cm1: float = 200.0
    low_pressure_rate_si: float = 0.0
    falloff_model: str = "troe"
    f_cent: float = 0.6
    energy_grain_cm1: float = 25.0
    maximum_energy_kj_mol: float = 250.0
    collision_efficiency: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        raw_pressures = self.pressures_pa
        if isinstance(raw_pressures, (int, float)):
            raw_pressures = (raw_pressures,)
        pressures = _as_float_tuple(raw_pressures)
        if not pressures or any(value <= 0.0 for value in pressures):
            raise ValueError("pressure grid must contain positive values")
        collision_frequency = (
            self.collision_frequency_s1_at_1bar
            if self.collision_frequency_s1_at_1bar is not None
            else self.collision_frequency_s_1_at_1bar
        )
        if collision_frequency <= 0.0:
            raise ValueError("collision frequency must be positive")
        if self.energy_transfer_cm1 <= 0.0 or self.energy_grain_cm1 <= 0.0:
            raise ValueError("energy transfer and grain must be positive")
        if self.collision_efficiency <= 0.0:
            raise ValueError("collision_efficiency must be positive")
        model = str(self.falloff_model).lower().replace("_", "-")
        if model == "strong-collision":
            model = "strong_collision"
        if model not in {"troe", "lindemann", "strong_collision", "ss-qrrk"}:
            raise ValueError("falloff_model must be troe, lindemann, strong_collision, or ss-qrrk")
        object.__setattr__(self, "pressures_pa", pressures)
        object.__setattr__(self, "collision_frequency_s_1_at_1bar", float(collision_frequency))
        object.__setattr__(self, "collision_frequency_s1_at_1bar", float(collision_frequency))
        object.__setattr__(self, "energy_transfer_cm1", float(self.energy_transfer_cm1))
        object.__setattr__(self, "low_pressure_rate_si", float(self.low_pressure_rate_si))
        object.__setattr__(self, "falloff_model", model)
        object.__setattr__(self, "f_cent", float(self.f_cent))
        object.__setattr__(self, "collision_efficiency", float(self.collision_efficiency))
        object.__setattr__(self, "enabled", bool(self.enabled))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PressureDependence":
        pressures = data.get("pressures_pa")
        if pressures is None:
            pressures = tuple(float(v) * 1.0e5 for v in data.get("pressures_bar", ()))
        elif isinstance(pressures, (int, float)):
            pressures = (float(pressures),)
        return cls(
            pressures_pa=tuple(float(v) for v in pressures),
            collider=str(data.get("collider", data.get("bath_gas", "M"))),
            collision_frequency_s_1_at_1bar=float(data.get("collision_frequency_s_1_at_1bar", data.get("collision_frequency_s^-1_at_1bar", 1.0e9))),
            collision_frequency_s1_at_1bar=(
                float(data["collision_frequency_s1_at_1bar"])
                if data.get("collision_frequency_s1_at_1bar") is not None else None
            ),
            energy_transfer_cm1=float(data.get("energy_transfer_cm1", data.get("average_energy_transfer_cm1", 200.0))),
            low_pressure_rate_si=float(data.get("low_pressure_rate_si", 0.0)),
            falloff_model=str(data.get("falloff_model", "troe")),
            f_cent=float(data.get("f_cent", 0.6)),
            energy_grain_cm1=float(data.get("energy_grain_cm1", 25.0)),
            maximum_energy_kj_mol=float(data.get("maximum_energy_kj_mol", 250.0)),
            collision_efficiency=float(data.get("collision_efficiency", 1.0)),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentModel:
    """Solvent or surface activation free-energy model."""

    phase: str = "gas"
    solvent: str | None = None
    species_solvation_kj_mol: Mapping[str, float] = field(default_factory=dict)
    dielectric: float | None = None
    activation_free_energy_kj_mol: float = 0.0
    surface_site_density_m2: float | None = None
    adsorption_energy_kj_mol: float = 0.0
    orientation_factor: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        phase = str(self.phase).lower()
        if phase not in {"gas", "solution", "solvent", "surface"}:
            raise ValueError("environment phase must be gas, solution, solvent, or surface")
        if self.dielectric is not None and self.dielectric <= 0.0:
            raise ValueError("dielectric must be positive")
        if self.surface_site_density_m2 is not None and self.surface_site_density_m2 <= 0.0:
            raise ValueError("surface_site_density_m2 must be positive")
        if self.orientation_factor <= 0.0:
            raise ValueError("orientation_factor must be positive")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "species_solvation_kj_mol", dict(self.species_solvation_kj_mol))
        object.__setattr__(self, "activation_free_energy_kj_mol", float(self.activation_free_energy_kj_mol))
        object.__setattr__(self, "adsorption_energy_kj_mol", float(self.adsorption_energy_kj_mol))
        object.__setattr__(self, "orientation_factor", float(self.orientation_factor))
        object.__setattr__(self, "enabled", bool(self.enabled))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EnvironmentModel":
        energies = data.get("species_solvation_kj_mol", data.get("solvation_free_energies_kj_mol", {}))
        return cls(
            phase=str(data.get("phase", "gas")),
            solvent=(str(data["solvent"]) if data.get("solvent") is not None else None),
            species_solvation_kj_mol={str(k): float(v) for k, v in dict(energies).items()},
            dielectric=(float(data["dielectric"]) if data.get("dielectric") is not None else None),
            activation_free_energy_kj_mol=float(data.get("activation_free_energy_kj_mol", 0.0)),
            surface_site_density_m2=(float(data["surface_site_density_m2"]) if data.get("surface_site_density_m2") is not None else None),
            adsorption_energy_kj_mol=float(data.get("adsorption_energy_kj_mol", 0.0)),
            orientation_factor=float(data.get("orientation_factor", 1.0)),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True, slots=True)
class Reaction:
    """A gas-phase reaction and its transition-state model."""

    reactants: tuple[Species, ...]
    transition_state: Species
    products: tuple[Species, ...] = ()
    path: ReactionPath | None = None
    standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA
    dual_level: DualLevelCorrection | None = None
    vrc: VrcModel | None = None
    pressure_dependence: PressureDependence | None = None
    environment: EnvironmentModel | None = None

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


_SUPPORTED_METHODS = {
    "tst", "cvt", "cus", "muvt", "us", "ivtst", "vrc_vtst", "pressure",
    "master_equation", "all",
}
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
    pressures_pa: tuple[float, ...] = ()
    coordinate_system: str | None = None
    pressure_method: str = "troe"

    def __post_init__(self) -> None:
        temperatures = tuple(float(value) for value in self.temperatures_k)
        if not temperatures or any(value <= 0.0 for value in temperatures):
            raise ValueError("temperatures must contain positive values")
        method = self.method.lower().replace("-", "_")
        tunneling = self.tunneling.lower()
        if method not in _SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {sorted(_SUPPORTED_METHODS)}")
        if tunneling not in _SUPPORTED_TUNNELING:
            raise ValueError(f"tunneling must be one of {sorted(_SUPPORTED_TUNNELING)}")
        if self.output_rate_unit == "":
            raise ValueError("output_rate_unit cannot be empty")
        if tunneling == "wkb" and self.tunneling_profile is None:
            raise ValueError("wkb tunneling requires a tunneling profile")
        raw_pressures = self.pressures_pa
        if isinstance(raw_pressures, (int, float)):
            raw_pressures = (raw_pressures,)
        pressures = _as_float_tuple(raw_pressures)
        if any(value <= 0.0 for value in pressures):
            raise ValueError("pressures_pa must contain positive values")
        object.__setattr__(self, "temperatures_k", temperatures)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "tunneling", tunneling)
        object.__setattr__(self, "pressures_pa", pressures)
        object.__setattr__(self, "pressure_method", str(self.pressure_method).lower())

