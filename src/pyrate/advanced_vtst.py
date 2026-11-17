"""Advanced variational transition-state-theory building blocks.

This module deliberately has no dependency on :mod:`pyrate.models` or the rate
orchestration layer.  It accepts numerical path data and therefore can be used
by both the TOML/legacy front ends and by direct-dynamics adapters.  The
implementations are small, deterministic reference algorithms; applications
that have a profiled native kernel can replace the sampling callback without
changing the result dataclasses.

The three public pieces are:

``DualLevelCorrection``
    Interpolates a high-level minus low-level electronic energy correction
    along a low-level path.

``iv_tst_rate_si``
    Finds the largest generalized free-energy barrier on an intrinsic-reaction
    coordinate and evaluates the Eyring rate at that bottleneck.

``vrc_vtst_rate_si``
    Integrates a loose-transition-state phase-space weight with Gauss--Legendre
    quadrature.  Energies and weights may come from an analytic potential or a
    direct-dynamics sampling callback.

All energies are in kJ mol-1, frequencies in cm-1, coordinates in an arbitrary
but consistent unit, and returned rates use s-1 for a unimolecular process or
m3 mol-1 s-1 for a bimolecular process.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

from .constants import (
    BOLTZMANN_J_K,
    DEFAULT_STANDARD_PRESSURE_PA,
    GAS_CONSTANT_J_MOL_K,
    PLANCK_J_S,
    standard_concentration_mol_m3,
)

# The public callback type accepts a coordinate and returns either an energy or
# ``(energy, phase_space_weight)``.  The latter is useful for direct VRC
# adapters that already evaluate the loose-TS density of states.
PotentialCallback = Callable[[float], float | tuple[float, float]]
WeightCallback = Callable[[float], float]


_R_KJ_MOL_K = GAS_CONSTANT_J_MOL_K / 1000.0


def _finite_tuple(values: Iterable[float], name: str, *, minimum: int = 1) -> tuple[float, ...]:
    """Convert an iterable to a finite tuple and validate its length."""

    result = tuple(float(value) for value in values)
    if len(result) < minimum:
        raise ValueError(f"{name} requires at least {minimum} values")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_coordinates(
    coordinates: Sequence[float], name: str = "coordinates"
) -> tuple[float, ...]:
    values = _finite_tuple(coordinates, name, minimum=2)
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def _validate_temperature(temperature_k: float) -> float:
    value = float(temperature_k)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("temperature_k must be finite and positive")
    return value


def _validate_molecularity(molecularity: int) -> int:
    value = int(molecularity)
    if value < 1:
        raise ValueError("molecularity must be a positive integer")
    return value


def _logsumexp(values: Sequence[float]) -> float:
    """Stable log(sum(exp(values))) for a non-empty finite sequence."""

    if not values:
        raise ValueError("logsumexp requires at least one value")
    largest = max(values)
    if math.isinf(largest) and largest < 0.0:
        return -math.inf
    return largest + math.log(sum(math.exp(value - largest) for value in values))


def _safe_exp(log_value: float) -> float:
    if log_value > 709.0:
        return math.inf
    if log_value < -745.0:
        return 0.0
    return math.exp(log_value)


@dataclass(frozen=True, slots=True)
class DualLevelCorrection:
    """High-minus-low-level correction defined on a reaction-coordinate grid.

    ``high_level_energies_kj_mol`` may be sparse in real workflows, but the
    class stores values on one common grid so that interpolation is explicit
    and reproducible.  ``at`` linearly interpolates the correction and rejects
    extrapolation by default; this prevents a high-level correction from being
    silently applied outside the region that was calculated at the expensive
    level.
    """

    coordinates: tuple[float, ...]
    low_level_energies_kj_mol: tuple[float, ...]
    high_level_energies_kj_mol: tuple[float, ...]

    def __post_init__(self) -> None:
        coordinates = _validate_coordinates(self.coordinates)
        low = _finite_tuple(self.low_level_energies_kj_mol, "low_level_energies_kj_mol", minimum=2)
        high = _finite_tuple(
            self.high_level_energies_kj_mol, "high_level_energies_kj_mol", minimum=2
        )
        if len(coordinates) != len(low) or len(low) != len(high):
            raise ValueError("coordinates and energy arrays must have identical lengths")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "low_level_energies_kj_mol", low)
        object.__setattr__(self, "high_level_energies_kj_mol", high)

    @property
    def corrections_kj_mol(self) -> tuple[float, ...]:
        """Return ``E_high - E_low`` at every supplied coordinate."""

        return tuple(
            high - low
            for low, high in zip(
                self.low_level_energies_kj_mol, self.high_level_energies_kj_mol
            )
        )

    @property
    def corrections(self) -> tuple[float, ...]:
        """Short alias for :attr:`corrections_kj_mol`."""

        return self.corrections_kj_mol

    @property
    def energy_corrections_kj_mol(self) -> tuple[float, ...]:
        """Alias for :attr:`corrections_kj_mol`."""

        return self.corrections_kj_mol

    @property
    def corrected_energies(self) -> tuple[float, ...]:
        """Short alias for :attr:`corrected_energies_kj_mol`."""

        return self.corrected_energies_kj_mol

    @property
    def corrected_energies_kj_mol(self) -> tuple[float, ...]:
        """Return low-level energies after applying the dual-level correction."""

        return tuple(
            low + correction
            for low, correction in zip(self.low_level_energies_kj_mol, self.corrections_kj_mol)
        )

    def at(self, coordinate: float, *, extrapolate: bool = False) -> float:
        """Return the interpolated correction at ``coordinate`` in kJ mol-1."""

        x = float(coordinate)
        if not math.isfinite(x):
            raise ValueError("coordinate must be finite")
        left = self.coordinates[0]
        right = self.coordinates[-1]
        if not extrapolate and not left <= x <= right:
            raise ValueError(
                f"coordinate {x:g} is outside [{left:g}, {right:g}]; "
                "pass extrapolate=True to permit linear extrapolation"
            )
        # Locate the segment without requiring numpy; the path is usually short.
        if x <= left:
            index = 0
        elif x >= right:
            index = len(self.coordinates) - 2
        else:
            index = next(
                idx
                for idx, (x0, x1) in enumerate(zip(self.coordinates, self.coordinates[1:]))
                if x0 <= x <= x1
            )
        x0 = self.coordinates[index]
        x1 = self.coordinates[index + 1]
        c0 = self.corrections_kj_mol[index]
        c1 = self.corrections_kj_mol[index + 1]
        fraction = (x - x0) / (x1 - x0)
        return c0 + fraction * (c1 - c0)

    def correction_at(self, coordinate: float, *, extrapolate: bool = False) -> float:
        """Alias for :meth:`at`."""

        return self.at(coordinate, extrapolate=extrapolate)

    def corrected_at(self, coordinate: float, *, extrapolate: bool = False) -> float:
        """Return the corrected low-level energy at ``coordinate``."""

        correction = self.at(coordinate, extrapolate=extrapolate)
        # Linear interpolation of low-level energy follows the same segment as
        # ``at``; this keeps the returned profile continuous when the correction
        # is sampled on a denser grid.
        x = float(coordinate)
        if x <= self.coordinates[0]:
            index = 0
        elif x >= self.coordinates[-1]:
            index = len(self.coordinates) - 2
        else:
            index = next(
                idx
                for idx, (x0, x1) in enumerate(zip(self.coordinates, self.coordinates[1:]))
                if x0 <= x <= x1
            )
        x0 = self.coordinates[index]
        x1 = self.coordinates[index + 1]
        e0 = self.low_level_energies_kj_mol[index]
        e1 = self.low_level_energies_kj_mol[index + 1]
        low = e0 + (x - x0) / (x1 - x0) * (e1 - e0)
        return low + correction


def dual_level_correction(
    low_level_energies_kj_mol: Iterable[float],
    high_level_energies_kj_mol: Iterable[float],
) -> tuple[float, ...]:
    """Return pointwise high-minus-low corrections.

    This compact function is convenient when a caller already owns the path
    coordinates.  Use :class:`DualLevelCorrection` when interpolation is
    needed.
    """

    low = _finite_tuple(low_level_energies_kj_mol, "low_level_energies_kj_mol", minimum=1)
    high = _finite_tuple(high_level_energies_kj_mol, "high_level_energies_kj_mol", minimum=1)
    if len(low) != len(high):
        raise ValueError("low-level and high-level arrays must have identical lengths")
    return tuple(high_value - low_value for low_value, high_value in zip(low, high))


def apply_dual_level_correction(
    low_level_energies_kj_mol: Iterable[float],
    high_level_energies_kj_mol: Iterable[float],
) -> tuple[float, ...]:
    """Return the dual-level corrected path (equivalent to the high-level path).

    The explicit name documents the operation in input adapters and avoids the
    common mistake of adding the high-level energy itself to the low-level
    energy instead of adding ``high - low``.
    """

    low = _finite_tuple(low_level_energies_kj_mol, "low_level_energies_kj_mol", minimum=1)
    correction = dual_level_correction(low, high_level_energies_kj_mol)
    return tuple(value + delta for value, delta in zip(low, correction))


@dataclass(frozen=True, slots=True)
class IntrinsicCoordinatePoint:
    """One point on an intrinsic reaction coordinate (IRC) profile."""

    coordinate: float
    potential_energy_kj_mol: float
    frequencies_cm1: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        coordinate = float(self.coordinate)
        energy = float(self.potential_energy_kj_mol)
        if not math.isfinite(coordinate) or not math.isfinite(energy):
            raise ValueError("IRC coordinates and energies must be finite")
        frequencies = (
            _finite_tuple(self.frequencies_cm1, "frequencies_cm1", minimum=0)
            if self.frequencies_cm1
            else ()
        )
        if any(frequency == 0.0 for frequency in frequencies):
            raise ValueError("IRC frequencies cannot be zero")
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "potential_energy_kj_mol", energy)
        object.__setattr__(self, "frequencies_cm1", frequencies)

    @property
    def free_energy_kj_mol(self) -> float:
        """Free energy represented by this point (including vibrational terms)."""

        return self.potential_energy_kj_mol

    @property
    def energy_kj_mol(self) -> float:
        """Alias for :attr:`potential_energy_kj_mol` used by path adapters."""

        return self.potential_energy_kj_mol

    @property
    def positive_frequencies_cm1(self) -> tuple[float, ...]:
        """Return real vibrational frequencies; imaginary modes are omitted."""

        return tuple(value for value in self.frequencies_cm1 if value > 0.0)


def harmonic_vibrational_free_energy_kj_mol(
    frequencies_cm1: Iterable[float], temperature_k: float
) -> float:
    """Return harmonic vibrational free energy including zero-point energy.

    Negative IRC modes are omitted, because they represent the reaction
    coordinate rather than a bound transverse vibration.  The expression is
    evaluated with ``log1p``/``expm1`` to remain stable for very low-frequency
    modes and high temperatures.
    """

    temperature = _validate_temperature(temperature_k)
    frequencies = (
        _finite_tuple(frequencies_cm1, "frequencies_cm1", minimum=0)
        if frequencies_cm1
        else ()
    )
    total = 0.0
    wavenumber_to_kj_mol = 0.011_962_656_5638
    for frequency in frequencies:
        if frequency <= 0.0:
            continue
        theta_over_t = wavenumber_to_kj_mol * frequency / (_R_KJ_MOL_K * temperature)
        # F = 1/2 h nu + R T ln(1 - exp(-h nu / kT)).
        thermal_log = math.log(-math.expm1(-theta_over_t))
        total += 0.5 * wavenumber_to_kj_mol * frequency + _R_KJ_MOL_K * temperature * thermal_log
    return total


def intrinsic_coordinate_free_energies(
    coordinates: Iterable[float],
    potential_energies_kj_mol: Iterable[float],
    frequencies_cm1: Sequence[Iterable[float]] | None,
    temperature_k: float,
) -> tuple[IntrinsicCoordinatePoint, ...]:
    """Build temperature-dependent IRC free energies from potentials/frequencies."""

    coords = _validate_coordinates(tuple(coordinates), "coordinates")
    energies = _finite_tuple(potential_energies_kj_mol, "potential_energies_kj_mol", minimum=2)
    if len(coords) != len(energies):
        raise ValueError("coordinates and potential energies must have identical lengths")
    if frequencies_cm1 is None:
        frequency_rows: tuple[tuple[float, ...], ...] = tuple(() for _ in coords)
    else:
        if len(frequencies_cm1) != len(coords):
            raise ValueError("frequencies_cm1 must contain one row per coordinate")
        frequency_rows = tuple(
            _finite_tuple(row, f"frequencies_cm1[{index}]", minimum=0) if row else ()
            for index, row in enumerate(frequencies_cm1)
        )
    # Validate temperature once before constructing points.
    temperature = _validate_temperature(temperature_k)
    del temperature
    return tuple(
        IntrinsicCoordinatePoint(
            coordinate=coordinate,
            potential_energy_kj_mol=(
                energy + harmonic_vibrational_free_energy_kj_mol(row, temperature_k)
            ),
            frequencies_cm1=row,
        )
        for coordinate, energy, row in zip(coords, energies, frequency_rows)
    )


def _refine_peak(
    coordinates: Sequence[float], values: Sequence[float], index: int
) -> tuple[float, float]:
    """Safely refine an interior sampled maximum with a local parabola."""

    if index <= 0 or index >= len(coordinates) - 1:
        return coordinates[index], values[index]
    x0, x1, x2 = coordinates[index - 1 : index + 2]
    y0, y1, y2 = values[index - 1 : index + 2]
    denominator = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denominator == 0.0:
        return x1, y1
    # Newton form converted to a vertex; this works on nonuniform grids.
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denominator
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denominator
    if not math.isfinite(a) or not math.isfinite(b) or a >= 0.0:
        return x1, y1
    vertex = -b / (2.0 * a)
    if not x0 <= vertex <= x2:
        return x1, y1
    value = a * vertex * vertex + b * vertex + (y0 - a * x0 * x0 - b * x0)
    if not math.isfinite(value) or value < y1:
        return x1, y1
    # Keep a local interpolation from producing an unphysical overshoot.
    cap = y1 + max(y1 - min(y0, y2), 0.0)
    return vertex, min(value, cap)


@dataclass(frozen=True, slots=True)
class IVTSTResult:
    """Result of intrinsic-coordinate variational transition-state theory."""

    coordinate: float
    activation_gibbs_kj_mol: float
    rate_si: float
    sampled_coordinate: float
    sampled_activation_gibbs_kj_mol: float
    sample_index: int
    temperature_k: float
    molecularity: int
    standard_pressure_pa: float
    free_energies_kj_mol: tuple[float, ...]

    @property
    def bottleneck_coordinate(self) -> float:
        """Alias used by result consumers that handle CVT and IVTST alike."""

        return self.coordinate

    @property
    def rate_constant_si(self) -> float:
        """Alias for the SI rate constant."""

        return self.rate_si

    @property
    def k_ivtst(self) -> float:
        """Compact alias used in rate tables."""

        return self.rate_si


def log_eyring_rate_si(
    activation_gibbs_kj_mol: float,
    temperature_k: float,
    molecularity: int = 1,
    standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA,
) -> float:
    """Return a numerically stable Eyring rate in SI molar units."""

    temperature = _validate_temperature(temperature_k)
    activation = float(activation_gibbs_kj_mol)
    if not math.isfinite(activation):
        raise ValueError("activation_gibbs_kj_mol must be finite")
    order = _validate_molecularity(molecularity)
    pressure = float(standard_pressure_pa)
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("standard_pressure_pa must be finite and positive")
    log_rate = math.log(BOLTZMANN_J_K * temperature / PLANCK_J_S)
    log_rate -= activation * 1000.0 / (GAS_CONSTANT_J_MOL_K * temperature)
    if order != 1:
        log_rate += (1 - order) * math.log(standard_concentration_mol_m3(temperature, pressure))
    return log_rate


def iv_tst_rate_si(
    coordinates: Iterable[float],
    free_energies_kj_mol: Iterable[float],
    temperature_k: float,
    *,
    reactant_free_energy_kj_mol: float = 0.0,
    molecularity: int = 1,
    standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA,
    refine: bool = True,
) -> IVTSTResult:
    """Evaluate IVTST along an intrinsic reaction coordinate.

    At each coordinate the generalized transition-state rate is proportional
    to ``exp(-G‡/RT)``.  Variational theory therefore selects the *largest*
    activation free energy (the bottleneck), with a guarded three-point
    parabolic refinement when possible.
    """

    coords = _validate_coordinates(tuple(coordinates), "coordinates")
    free_energies = _finite_tuple(free_energies_kj_mol, "free_energies_kj_mol", minimum=2)
    if len(coords) != len(free_energies):
        raise ValueError("coordinates and free energies must have identical lengths")
    temperature = _validate_temperature(temperature_k)
    reference = float(reactant_free_energy_kj_mol)
    if not math.isfinite(reference):
        raise ValueError("reactant_free_energy_kj_mol must be finite")
    activation = tuple(value - reference for value in free_energies)
    index = max(range(len(activation)), key=activation.__getitem__)
    coordinate = coords[index]
    value = activation[index]
    if refine:
        coordinate, value = _refine_peak(coords, activation, index)
    log_rate = log_eyring_rate_si(value, temperature, molecularity, standard_pressure_pa)
    return IVTSTResult(
        coordinate=coordinate,
        activation_gibbs_kj_mol=value,
        rate_si=_safe_exp(log_rate),
        sampled_coordinate=coords[index],
        sampled_activation_gibbs_kj_mol=activation[index],
        sample_index=index,
        temperature_k=temperature,
        molecularity=_validate_molecularity(molecularity),
        standard_pressure_pa=float(standard_pressure_pa),
        free_energies_kj_mol=free_energies,
    )


# Common spelling variants used in chemistry codes and legacy adapters.
iv_tst = iv_tst_rate_si
# ``ivtst`` is the spelling used by Polyrate input decks.
ivtst_rate_si = iv_tst_rate_si
intrinsic_variational_tst = iv_tst_rate_si


@dataclass(frozen=True, slots=True)
class VRCSample:
    """One quadrature sample for a VRC loose transition state."""

    coordinate: float
    energy_kj_mol: float
    phase_space_weight: float
    quadrature_weight: float

    def __post_init__(self) -> None:
        values = (
            self.coordinate,
            self.energy_kj_mol,
            self.phase_space_weight,
            self.quadrature_weight,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("VRC sample values must be finite")
        if self.phase_space_weight <= 0.0 or self.quadrature_weight <= 0.0:
            raise ValueError("VRC phase-space and quadrature weights must be positive")

    @property
    def generalized_free_energy_kj_mol(self) -> float:
        """Return ``E - R T ln(phase_space_weight)`` (temperature supplied by caller)."""

        # This property is intentionally temperature-free; use
        # ``vrc_generalized_free_energy`` when a numerical value is required.
        return self.energy_kj_mol


def _gauss_legendre_nodes(order: int) -> tuple[tuple[float, float], ...]:
    """Return nodes/weights on [-1, 1] without requiring SciPy."""

    if order < 2 or order > 256:
        raise ValueError("quadrature_points must be between 2 and 256")
    # NumPy is already a hard dependency of PyRate.  Import lazily so this
    # standalone module remains cheap to import in lightweight adapters.
    try:
        from numpy.polynomial.legendre import leggauss
    except ImportError as exc:  # pragma: no cover - numpy is a project dependency
        raise RuntimeError("VRC quadrature requires numpy") from exc
    nodes, weights = leggauss(order)
    return tuple((float(node), float(weight)) for node, weight in zip(nodes, weights))


def _evaluate_potential(
    callback: PotentialCallback, coordinate: float, phase_space_fn: WeightCallback | None
) -> tuple[float, float]:
    value = callback(coordinate)
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                "potential_fn tuple results must be "
                "(energy_kj_mol, phase_space_weight)"
            )
        energy, weight = float(value[0]), float(value[1])
    else:
        energy, weight = float(value), 1.0
    if phase_space_fn is not None:
        weight *= float(phase_space_fn(coordinate))
    if not math.isfinite(energy) or not math.isfinite(weight):
        raise ValueError("VRC potential and phase-space weights must be finite")
    if weight <= 0.0:
        raise ValueError("VRC phase-space weights must be positive")
    return energy, weight


def sample_vrc_loose_transition_state(
    coordinate_bounds: tuple[float, float],
    potential_fn: PotentialCallback,
    *,
    phase_space_fn: WeightCallback | None = None,
    quadrature_points: int = 32,
) -> tuple[VRCSample, ...]:
    """Sample a loose VRC transition state by Gauss--Legendre quadrature.

    The returned quadrature weights are normalized to sum to one.  Normalizing
    makes a constant potential produce the usual Eyring rate independent of the
    arbitrary coordinate interval; phase-space weights remain explicit and are
    not normalized, so they can change the loose-TS rate as intended.
    """

    if len(coordinate_bounds) != 2:
        raise ValueError("coordinate_bounds must be (lower, upper)")
    lower, upper = (float(value) for value in coordinate_bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("coordinate_bounds must be finite with upper > lower")
    if not callable(potential_fn):
        raise TypeError("potential_fn must be callable")
    nodes_weights = _gauss_legendre_nodes(int(quadrature_points))
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    samples: list[VRCSample] = []
    total_weight = sum(weight for _, weight in nodes_weights)
    for node, quadrature_weight in nodes_weights:
        coordinate = midpoint + half_width * node
        energy, phase_weight = _evaluate_potential(potential_fn, coordinate, phase_space_fn)
        samples.append(
            VRCSample(
                coordinate=coordinate,
                energy_kj_mol=energy,
                phase_space_weight=phase_weight,
                quadrature_weight=quadrature_weight / total_weight,
            )
        )
    return tuple(samples)


def vrc_generalized_free_energy_kj_mol(
    energy_kj_mol: float, phase_space_weight: float, temperature_k: float
) -> float:
    """Convert a loose-TS energy and dimensionless phase-space weight to G."""

    temperature = _validate_temperature(temperature_k)
    energy = float(energy_kj_mol)
    weight = float(phase_space_weight)
    if not math.isfinite(energy) or not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("energy must be finite and phase_space_weight must be positive")
    return energy - _R_KJ_MOL_K * temperature * math.log(weight)


@dataclass(frozen=True, slots=True)
class VRCVTSTResult:
    """Integrated VRC-VTST rate and quadrature diagnostics."""

    rate_si: float
    effective_activation_gibbs_kj_mol: float
    partition_integral: float
    temperature_k: float
    molecularity: int
    standard_pressure_pa: float
    samples: tuple[VRCSample, ...]

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def activation_gibbs_kj_mol(self) -> float:
        """Alias matching :class:`IVTSTResult`."""

        return self.effective_activation_gibbs_kj_mol

    @property
    def rate_constant_si(self) -> float:
        """Alias for the SI rate constant."""

        return self.rate_si

    @property
    def k_vrc_vtst(self) -> float:
        """Compact alias used in rate tables."""

        return self.rate_si

    @property
    def bottleneck_coordinate(self) -> float:
        """Coordinate of the largest Boltzmann-weighted contribution."""

        return max(
            self.samples,
            key=lambda sample: sample.quadrature_weight
            * sample.phase_space_weight
            * math.exp(-sample.energy_kj_mol / (_R_KJ_MOL_K * self.temperature_k)),
        ).coordinate


@dataclass(frozen=True, slots=True)
class VRCVTSTModel:
    """Reusable VRC loose-transition-state model.

    A model can be backed by a direct-dynamics ``potential_fn`` or by a
    tabulated ``coordinates``/``energies_kj_mol`` profile.  The latter is
    linearly interpolated and is useful for deterministic tests and for
    legacy path data.  ``phase_space_weights`` are optional tabulated loose-TS
    degeneracies and are interpolated with the same grid.
    """

    coordinate_bounds: tuple[float, float] | None = None
    potential_fn: PotentialCallback | None = None
    coordinates: tuple[float, ...] | None = None
    energies_kj_mol: tuple[float, ...] | None = None
    phase_space_weights: tuple[float, ...] | None = None
    phase_space_fn: WeightCallback | None = None
    energy_kj_mol: float | None = None
    quadrature_points: int = 32

    def __post_init__(self) -> None:
        if self.phase_space_fn is not None and not callable(self.phase_space_fn):
            raise TypeError("phase_space_fn must be callable")
        if self.potential_fn is None:
            if self.coordinates is None and self.energies_kj_mol is None:
                if self.energy_kj_mol is None or self.coordinate_bounds is None:
                    raise ValueError(
                        "VRCVTSTModel requires potential_fn, a tabulated profile, "
                        "or energy_kj_mol with coordinate_bounds"
                    )
                energy = float(self.energy_kj_mol)
                if not math.isfinite(energy):
                    raise ValueError("energy_kj_mol must be finite")
                bounds = self.coordinate_bounds
                object.__setattr__(self, "energy_kj_mol", energy)
                object.__setattr__(self, "coordinate_bounds", bounds)
            elif self.coordinates is None or self.energies_kj_mol is None:
                raise ValueError(
                    "VRCVTSTModel requires potential_fn or coordinates and energies_kj_mol"
                )
            else:
                coords = _validate_coordinates(self.coordinates, "coordinates")
                energies = _finite_tuple(self.energies_kj_mol, "energies_kj_mol", minimum=2)
                if len(coords) != len(energies):
                    raise ValueError("coordinates and energies_kj_mol must have identical lengths")
                object.__setattr__(self, "coordinates", coords)
                object.__setattr__(self, "energies_kj_mol", energies)
                if self.phase_space_weights is None:
                    weights = tuple(1.0 for _ in coords)
                else:
                    weights = _finite_tuple(
                        self.phase_space_weights, "phase_space_weights", minimum=2
                    )
                    if len(weights) != len(coords):
                        raise ValueError("phase_space_weights must match coordinates")
                    if any(weight <= 0.0 for weight in weights):
                        raise ValueError("phase_space_weights must be positive")
                object.__setattr__(self, "phase_space_weights", weights)
                bounds = (
                    (coords[0], coords[-1])
                    if self.coordinate_bounds is None
                    else self.coordinate_bounds
                )
                if bounds[0] < coords[0] or bounds[1] > coords[-1]:
                    raise ValueError(
                        "coordinate_bounds must lie within the tabulated VRC profile"
                    )
        else:
            if not callable(self.potential_fn):
                raise TypeError("potential_fn must be callable")
            if self.coordinate_bounds is None:
                raise ValueError("coordinate_bounds is required with potential_fn")
            bounds = self.coordinate_bounds
        if bounds is None:
            raise ValueError("coordinate_bounds is required")
        if len(bounds) != 2:
            raise ValueError("coordinate_bounds must be (lower, upper)")
        lower, upper = (float(value) for value in bounds)
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError("coordinate_bounds must be finite with upper > lower")
        object.__setattr__(self, "coordinate_bounds", (lower, upper))
        if int(self.quadrature_points) < 2 or int(self.quadrature_points) > 256:
            raise ValueError("quadrature_points must be between 2 and 256")
        object.__setattr__(self, "quadrature_points", int(self.quadrature_points))

    def _tabulated_callback(self, coordinate: float) -> tuple[float, float]:
        assert self.coordinates is not None and self.energies_kj_mol is not None
        assert self.phase_space_weights is not None
        x = float(coordinate)
        if x < self.coordinates[0] or x > self.coordinates[-1]:
            raise ValueError("VRC coordinate is outside the tabulated profile")
        if x == self.coordinates[-1]:
            index = len(self.coordinates) - 2
        else:
            index = next(
                idx
                for idx, (x0, x1) in enumerate(zip(self.coordinates, self.coordinates[1:]))
                if x0 <= x <= x1
            )
        x0, x1 = self.coordinates[index], self.coordinates[index + 1]
        fraction = (x - x0) / (x1 - x0)
        energy = self.energies_kj_mol[index] + fraction * (
            self.energies_kj_mol[index + 1] - self.energies_kj_mol[index]
        )
        weight = self.phase_space_weights[index] + fraction * (
            self.phase_space_weights[index + 1] - self.phase_space_weights[index]
        )
        return energy, weight

    def callback(self) -> PotentialCallback:
        """Return a callback suitable for :func:`vrc_vtst_rate_si`."""

        if self.potential_fn is not None:
            return self.potential_fn
        if self.energy_kj_mol is not None and self.coordinates is None:
            energy = self.energy_kj_mol
            return lambda _coordinate: energy
        return self._tabulated_callback

    def calculate(
        self,
        temperature_k: float,
        *,
        reactant_free_energy_kj_mol: float = 0.0,
        molecularity: int = 1,
        standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA,
    ) -> VRCVTSTResult:
        """Alias for :meth:`rate_si` used by generic model runners."""

        return self.rate_si(
            temperature_k,
            reactant_free_energy_kj_mol=reactant_free_energy_kj_mol,
            molecularity=molecularity,
            standard_pressure_pa=standard_pressure_pa,
        )

    def rate_si(
        self,
        temperature_k: float,
        *,
        reactant_free_energy_kj_mol: float = 0.0,
        molecularity: int = 1,
        standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA,
    ) -> VRCVTSTResult:
        """Evaluate this model at one temperature."""

        assert self.coordinate_bounds is not None
        return vrc_vtst_rate_si(
            self.coordinate_bounds,
            self.callback(),
            temperature_k,
            phase_space_fn=self.phase_space_fn,
            quadrature_points=self.quadrature_points,
            reactant_free_energy_kj_mol=reactant_free_energy_kj_mol,
            molecularity=molecularity,
            standard_pressure_pa=standard_pressure_pa,
        )


def vrc_vtst_rate_si(
    coordinate_bounds: tuple[float, float],
    potential_fn: PotentialCallback,
    temperature_k: float,
    *,
    phase_space_fn: WeightCallback | None = None,
    quadrature_points: int = 32,
    reactant_free_energy_kj_mol: float = 0.0,
    molecularity: int = 1,
    standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA,
) -> VRCVTSTResult:
    """Evaluate a one-dimensional VRC-VTST loose-transition-state integral.

    ``potential_fn`` returns the loose-TS energy and optionally a
    dimensionless phase-space weight.  The normalized quadrature computes
    ``I = <w(s) exp(-G(s)/(RT))>``; the effective activation free energy is
    ``-RT ln(I) - G_reactant``.  Log-sum-exp arithmetic keeps deep barriers
    and strong phase-space factors finite.
    """

    temperature = _validate_temperature(temperature_k)
    order = _validate_molecularity(molecularity)
    samples = sample_vrc_loose_transition_state(
        coordinate_bounds,
        potential_fn,
        phase_space_fn=phase_space_fn,
        quadrature_points=quadrature_points,
    )
    beta = 1.0 / (_R_KJ_MOL_K * temperature)
    log_terms = tuple(
        math.log(sample.quadrature_weight)
        + math.log(sample.phase_space_weight)
        - beta * sample.energy_kj_mol
        for sample in samples
    )
    log_integral = _logsumexp(log_terms)
    integral = _safe_exp(log_integral)
    reactant = float(reactant_free_energy_kj_mol)
    if not math.isfinite(reactant):
        raise ValueError("reactant_free_energy_kj_mol must be finite")
    effective = -log_integral / beta - reactant
    log_rate = log_eyring_rate_si(effective, temperature, order, standard_pressure_pa)
    return VRCVTSTResult(
        rate_si=_safe_exp(log_rate),
        effective_activation_gibbs_kj_mol=effective,
        partition_integral=integral,
        temperature_k=temperature,
        molecularity=order,
        standard_pressure_pa=float(standard_pressure_pa),
        samples=samples,
    )


vrc_vtst = vrc_vtst_rate_si
vrc_loose_ts_rate_si = vrc_vtst_rate_si


__all__ = [
    "DualLevelCorrection",
    "IVTSTResult",
    "IntrinsicCoordinatePoint",
    "VRCVTSTModel",
    "PotentialCallback",
    "VRCSample",
    "VRCVTSTResult",
    "apply_dual_level_correction",
    "dual_level_correction",
    "harmonic_vibrational_free_energy_kj_mol",
    "intrinsic_coordinate_free_energies",
    "intrinsic_variational_tst",
    "iv_tst",
    "iv_tst_rate_si",
    "ivtst_rate_si",
    "log_eyring_rate_si",
    "sample_vrc_loose_transition_state",
    "vrc_generalized_free_energy_kj_mol",
    "vrc_loose_ts_rate_si",
    "vrc_vtst",
    "vrc_vtst_rate_si",
]
