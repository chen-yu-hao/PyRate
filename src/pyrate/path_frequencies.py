r"""Intrinsic reaction-coordinate frequencies on IRC/MEP paths.

The reaction-coordinate (or *path-coordinate*) frequency is the local harmonic
frequency obtained by projecting the potential onto the one-dimensional path.
For a tabulated path this is

.. math::

   \omega(s)^2 = \frac{1}{\mu}\frac{d^2 V}{d q^2},

where ``q`` is the physical path coordinate in metres and ``V`` is the energy
per molecule.  A negative second derivative is retained as a negative
wavenumber, which is the convention used for an imaginary transition-state
mode throughout PyRate.

Two independent data sources are supported:

* :func:`coordinate_frequencies` differentiates a potential/energy profile on
  an unevenly spaced path grid.  It uses the exact quadratic finite-difference
  stencil for each local triplet, including one-sided endpoint stencils.
* :func:`hessian_coordinate_frequencies` projects Cartesian Hessians onto
  supplied path tangents.  Tangents may have arbitrary scale because the
  scale cancels between the projected force constant and the path mass.

All public functions validate dimensions and units before doing numerical work;
this is deliberate because a plausible but incorrectly scaled IRC frequency can
otherwise silently corrupt a tunnelling calculation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from .constants import (
    AMU_KG,
    ANGSTROM_M,
    AVOGADRO_MOL,
    BOHR_M,
    HARTREE_J,
    LIGHT_SPEED_M_S,
)

FloatArray = npt.NDArray[np.float64]
CoordinateUnit = Literal["angstrom", "bohr", "m", "meter", "metre"]
HessianUnit = Literal["hartree/bohr^2", "j/m^2", "kj/mol/angstrom^2"]


_LENGTH_SCALE_M: dict[str, float] = {
    "angstrom": ANGSTROM_M,
    "bohr": BOHR_M,
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
}

_HESSIAN_SCALE_TO_J_M2: dict[str, float] = {
    "j/m^2": 1.0,
    "hartree/bohr^2": HARTREE_J / BOHR_M**2,
    "kj/mol/angstrom^2": 1000.0 / AVOGADRO_MOL / ANGSTROM_M**2,
}


def _as_vector(values: Sequence[float], name: str) -> FloatArray:
    array = cast(FloatArray, np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _length_scale(unit: str) -> float:
    normalized = str(unit).strip().lower()
    try:
        return _LENGTH_SCALE_M[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_LENGTH_SCALE_M))
        raise ValueError(
            f"unsupported coordinate unit {unit!r}; expected one of {supported}"
        ) from exc


def _quadratic_second_derivative(x: FloatArray, y: FloatArray) -> float:
    """Return ``d²y/dx²`` for the unique quadratic through three points."""

    if x.size != 3 or y.size != 3:
        raise ValueError("a quadratic stencil requires exactly three points")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("quadratic stencil coordinates must be strictly increasing")
    # Fit in the local coordinate frame to avoid loss of precision for large
    # absolute IRC coordinates.  The coefficient of u² is invariant under the
    # translation, and the second derivative is 2*a.
    u = x - x[1]
    coefficients = np.polynomial.polynomial.polyfit(u, y, deg=2)
    return float(2.0 * coefficients[2])


def _second_derivatives(coordinates: FloatArray, values: FloatArray) -> FloatArray:
    """Differentiate values on a strictly increasing, possibly uneven grid."""

    count = coordinates.size
    if count < 3:
        raise ValueError("at least three path points are required")
    if values.size != count:
        raise ValueError("coordinates and values must have matching lengths")
    if np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("path coordinates must be strictly increasing")

    result = np.empty(count, dtype=float)
    result[0] = _quadratic_second_derivative(coordinates[:3], values[:3])
    result[-1] = _quadratic_second_derivative(coordinates[-3:], values[-3:])
    steps = np.diff(coordinates)
    # For an interior point i, h0 = x_i - x_{i-1} and h1 = x_{i+1}-x_i.
    h0 = steps[:-1]
    h1 = steps[1:]
    result[1:-1] = 2.0 * (
        h0 * values[2:] - (h0 + h1) * values[1:-1] + h1 * values[:-2]
    ) / (h0 * h1 * (h0 + h1))
    return result


def _signed_wavenumber(curvature_j_m2: FloatArray, mass_amu: float) -> FloatArray:
    if mass_amu <= 0.0 or not np.isfinite(mass_amu):
        raise ValueError("reduced_mass_amu must be a finite positive number")
    mass_kg = mass_amu * AMU_KG
    angular = np.sign(curvature_j_m2) * np.sqrt(np.abs(curvature_j_m2) / mass_kg)
    return cast(FloatArray, angular / (2.0 * np.pi * LIGHT_SPEED_M_S * 100.0))


@dataclass(frozen=True, slots=True)
class CoordinateFrequencyProfile:
    """Frequency profile along an IRC/MEP coordinate.

    ``coordinates`` retain the input coordinate unit.  ``curvatures`` are the
    second derivatives of the energy in ``kJ mol⁻¹`` with respect to the numeric
    coordinate values (for example, ``kJ mol⁻¹ Å⁻²`` for ``coordinate_unit``
    ``"angstrom"``).  ``frequencies_cm1`` are signed wavenumbers; negative
    entries denote imaginary modes.
    """

    coordinates: tuple[float, ...]
    frequencies_cm1: tuple[float, ...]
    curvatures_kj_mol_per_coordinate2: tuple[float, ...]
    reduced_mass_amu: float
    coordinate_unit: str
    source: str = "energy-profile"
    effective_masses_amu: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        coordinates = tuple(float(value) for value in self.coordinates)
        frequencies = tuple(float(value) for value in self.frequencies_cm1)
        curvatures = tuple(
            float(value) for value in self.curvatures_kj_mol_per_coordinate2
        )
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "frequencies_cm1", frequencies)
        object.__setattr__(self, "curvatures_kj_mol_per_coordinate2", curvatures)
        if any(not np.isfinite(value) for value in coordinates + frequencies + curvatures):
            raise ValueError("coordinates, frequencies and curvatures must be finite")
        if len(self.coordinates) < 3:
            raise ValueError("a coordinate-frequency profile requires at least three points")
        if len(self.coordinates) != len(self.frequencies_cm1):
            raise ValueError("coordinates and frequencies_cm1 must have matching lengths")
        if len(self.coordinates) != len(self.curvatures_kj_mol_per_coordinate2):
            raise ValueError("coordinates and curvatures must have matching lengths")
        if any(right <= left for left, right in zip(self.coordinates, self.coordinates[1:])):
            raise ValueError("profile coordinates must be strictly increasing")
        if self.reduced_mass_amu <= 0.0 or not np.isfinite(self.reduced_mass_amu):
            raise ValueError("reduced_mass_amu must be positive")
        if self.effective_masses_amu is not None:
            masses = tuple(float(value) for value in self.effective_masses_amu)
            if len(masses) != len(self.coordinates):
                raise ValueError("effective_masses_amu must match coordinates length")
            if any(value <= 0.0 or not np.isfinite(value) for value in masses):
                raise ValueError("effective_masses_amu must contain positive finite values")
            object.__setattr__(self, "effective_masses_amu", masses)

    @property
    def imaginary_frequencies_cm1(self) -> tuple[float, ...]:
        """Signed negative wavenumbers at path points with negative curvature."""

        return tuple(value for value in self.frequencies_cm1 if value < 0.0)

    @property
    def real_frequencies_cm1(self) -> tuple[float, ...]:
        """Positive wavenumbers at path points with positive curvature."""

        return tuple(value for value in self.frequencies_cm1 if value > 0.0)

    @property
    def barrier_index(self) -> int:
        """Index of the largest tabulated path-coordinate value, by curvature sign.

        A reaction-coordinate barrier is commonly represented by the most
        negative local frequency.  If no point is imaginary, the point with the
        smallest signed frequency is returned.
        """

        frequencies = np.asarray(self.frequencies_cm1, dtype=float)
        minimum = float(np.min(frequencies))
        # A quadratic potential produces the same curvature at every point;
        # tiny endpoint round-off should not move the reported representative
        # from the first point to the last one.
        candidates = np.flatnonzero(
            np.isclose(frequencies, minimum, rtol=1.0e-12, atol=1.0e-9)
        )
        return int(candidates[0])


def coordinate_frequencies(
    coordinates: Sequence[float],
    energies_kj_mol: Sequence[float],
    reduced_mass_amu: float | Sequence[float],
    *,
    coordinate_unit: CoordinateUnit = "angstrom",
) -> CoordinateFrequencyProfile:
    """Compute local IRC/MEP frequencies from a tabulated energy profile.

    Parameters
    ----------
    coordinates, energies_kj_mol:
        Matching, finite arrays with at least three strictly increasing path
        coordinates and energies in ``kJ mol⁻¹``.
    reduced_mass_amu:
        Effective path mass.  It may be a scalar constant for a profile; a
        point-dependent mass can be handled by calling this function for each
        segment or by projecting Hessians with
        :func:`hessian_coordinate_frequencies`.
    coordinate_unit:
        Physical length unit of the numeric coordinate, one of ``angstrom``,
        ``bohr`` or metres (``m``, ``meter``, ``metre``).

    Returns
    -------
    CoordinateFrequencyProfile
        Signed wavenumbers in ``cm⁻¹``.  A negative curvature produces a
        negative wavenumber (an imaginary coordinate frequency).
    """

    x = _as_vector(coordinates, "coordinates")
    energies = _as_vector(energies_kj_mol, "energies_kj_mol")
    if x.size != energies.size:
        raise ValueError("coordinates and energies_kj_mol must have matching lengths")
    if x.size < 3:
        raise ValueError("at least three path points are required")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("coordinates must be strictly increasing")
    scale_m = _length_scale(coordinate_unit)
    if isinstance(reduced_mass_amu, Real):
        mass_values = np.full(x.size, float(reduced_mass_amu), dtype=np.float64)
    else:
        mass_values = _as_vector(cast(Sequence[float], reduced_mass_amu), "reduced_mass_amu")
        if mass_values.size != x.size:
            raise ValueError("reduced_mass_amu sequence must match coordinates length")
    if np.any(mass_values <= 0.0):
        raise ValueError("reduced_mass_amu values must be positive")
    curvature_kj = _second_derivatives(x, energies)
    # d²V/dq_m² = (d²V/dq_numeric²) / scale².  Convert the molar energy to
    # energy per molecule before dividing by the path mass.
    curvature_j_m2 = curvature_kj * 1000.0 / AVOGADRO_MOL / (scale_m**2)
    # A path-point mass changes only the local denominator; use the vectorized
    # form rather than repeatedly calling the scalar helper.
    angular = np.sign(curvature_j_m2) * np.sqrt(
        np.abs(curvature_j_m2) / (mass_values * AMU_KG)
    )
    frequencies = angular / (2.0 * np.pi * LIGHT_SPEED_M_S * 100.0)
    return CoordinateFrequencyProfile(
        coordinates=tuple(float(value) for value in x),
        frequencies_cm1=tuple(float(value) for value in frequencies),
        curvatures_kj_mol_per_coordinate2=tuple(float(value) for value in curvature_kj),
        reduced_mass_amu=float(np.mean(mass_values)),
        coordinate_unit=str(coordinate_unit).lower(),
        effective_masses_amu=tuple(float(value) for value in mass_values),
    )


def _hessian_to_si(hessian: FloatArray, unit: str) -> FloatArray:
    normalized = str(unit).strip().lower()
    try:
        scale = _HESSIAN_SCALE_TO_J_M2[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_HESSIAN_SCALE_TO_J_M2))
        raise ValueError(f"unsupported Hessian unit {unit!r}; expected one of {supported}") from exc
    return hessian * scale


def hessian_coordinate_frequencies(
    hessians: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    tangents: Sequence[Sequence[float]] | np.ndarray,
    masses_amu: Sequence[float],
    *,
    coordinates: Sequence[float] | None = None,
    coordinate_unit: CoordinateUnit = "angstrom",
    hessian_unit: HessianUnit = "j/m^2",
) -> CoordinateFrequencyProfile:
    """Project Cartesian Hessians onto IRC tangents and return frequencies.

    ``tangents[i]`` is a Cartesian direction for path point ``i``.  Its norm
    and sign are immaterial: the same scale appears in the projected force
    constant and effective path mass, so their ratio is invariant.  The
    tangent dimension is ``3 * len(masses_amu)`` and masses are in amu.

    Hessians may be supplied in ``j/m²``, ``hartree/bohr²`` or
    ``kJ mol⁻¹ Å⁻²``.  If ``coordinates`` is omitted, integer point indices
    ``0, …, n-1`` are used; supplying coordinates is recommended when the
    result is to be combined with another path profile.
    """

    matrix = np.asarray(hessians, dtype=float)
    tangent_array = np.asarray(tangents, dtype=float)
    masses = _as_vector(masses_amu, "masses_amu")
    # Validate the coordinate label even when the caller asks for generated
    # integer point indices (``coordinates=None``).  The label is retained in
    # the returned profile and must remain a known physical unit.
    _length_scale(coordinate_unit)
    if matrix.ndim == 2:
        matrix = matrix[None, ...]
    if tangent_array.ndim == 1:
        tangent_array = tangent_array[None, ...]
    if matrix.ndim != 3 or tangent_array.ndim != 2:
        raise ValueError("hessians must be [n, d, d] and tangents must be [n, d]")
    point_count, dimension, dimension2 = matrix.shape
    if dimension != dimension2 or tangent_array.shape != (point_count, dimension):
        raise ValueError("each Hessian and tangent must have matching square dimensions")
    if dimension != 3 * masses.size:
        raise ValueError("Hessian dimension must equal 3 * len(masses_amu)")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(tangent_array)):
        raise ValueError("hessians and tangents must contain only finite values")
    if np.any(np.linalg.norm(tangent_array, axis=1) <= 0.0):
        raise ValueError("each tangent must be non-zero")
    if not np.allclose(matrix, np.swapaxes(matrix, 1, 2), rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("each Hessian must be symmetric")

    mass_vector_kg = np.repeat(masses, 3) * AMU_KG
    frequencies: list[float] = []
    curvatures: list[float] = []
    for hessian, tangent in zip(matrix, tangent_array):
        hessian_si = _hessian_to_si(hessian, hessian_unit)
        # The force constant and path mass both scale as tangent²; no arbitrary
        # tangent normalization is necessary.  This is equivalent to a
        # one-dimensional mass-weighted Hessian Rayleigh quotient.
        effective_mass = float(np.dot(mass_vector_kg, tangent * tangent))
        projected_force = float(tangent @ hessian_si @ tangent)
        angular = np.sign(projected_force) * np.sqrt(abs(projected_force) / effective_mass)
        frequencies.append(angular / (2.0 * np.pi * LIGHT_SPEED_M_S * 100.0))
        # Report the curvature in the supplied Hessian unit's common public unit
        # (kJ mol⁻¹ Å⁻²), independent of the coordinate label.
        curvatures.append(projected_force * AVOGADRO_MOL * ANGSTROM_M**2 / 1000.0)

    if coordinates is None:
        coordinates_array = np.arange(point_count, dtype=float)
    else:
        coordinates_array = _as_vector(coordinates, "coordinates")
        if coordinates_array.size != point_count:
            raise ValueError("coordinates length must match the number of Hessians")
        if point_count >= 2 and np.any(np.diff(coordinates_array) <= 0.0):
            raise ValueError("coordinates must be strictly increasing")
    # The profile dataclass requires three points because an IRC profile is
    # normally differentiated; Hessian projection can legitimately be a single
    # stationary point, so pad only the metadata-free one-point case with an
    # explicit, clear error instead of manufacturing values.
    if point_count < 3:
        raise ValueError("at least three Hessian/tangent points are required for a profile")
    return CoordinateFrequencyProfile(
        coordinates=tuple(float(value) for value in coordinates_array),
        frequencies_cm1=tuple(frequencies),
        curvatures_kj_mol_per_coordinate2=tuple(curvatures),
        reduced_mass_amu=float(
            np.mean(
                np.sum((tangent_array * tangent_array) * masses.repeat(3), axis=1)
            )
        ),
        coordinate_unit=str(coordinate_unit).lower(),
        source="cartesian-hessian-projection",
        effective_masses_amu=tuple(
            float(np.sum((tangent * tangent) * masses.repeat(3)))
            for tangent in tangent_array
        ),
    )


def projected_coordinate_frequency_cm1(
    hessians: Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[float]] | np.ndarray,
    tangents: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    masses_amu: Sequence[float],
    *,
    hessian_unit: HessianUnit = "j/m^2",
) -> tuple[float, ...] | float:
    """Return only projected signed wavenumbers from one or more Hessians.

    This compact helper is useful to callers that already own path coordinates
    and only need the frequency values.  A single Hessian/tangent pair returns
    a scalar; batched inputs return a tuple in path order.  For coordinates and
    curvature metadata use :func:`hessian_coordinate_frequencies`.
    """

    matrix = np.asarray(hessians, dtype=float)
    tangent_array = np.asarray(tangents, dtype=float)
    if matrix.ndim == 2:
        matrix = matrix[None, ...]
    if tangent_array.ndim == 1:
        tangent_array = tangent_array[None, ...]
    masses = _as_vector(masses_amu, "masses_amu")
    if matrix.ndim != 3 or tangent_array.ndim != 2 or matrix.shape[0] != tangent_array.shape[0]:
        raise ValueError("hessians and tangents must contain the same number of points")
    if matrix.shape[1:] != (3 * masses.size, 3 * masses.size):
        raise ValueError("Hessian dimension must equal 3 * len(masses_amu)")
    mass_vector_kg = np.repeat(masses, 3) * AMU_KG
    values: list[float] = []
    for hessian, tangent in zip(matrix, tangent_array):
        if tangent.shape != (matrix.shape[1],) or not np.all(np.isfinite(tangent)):
            raise ValueError("each tangent must have one finite value per Cartesian coordinate")
        if np.linalg.norm(tangent) <= 0.0:
            raise ValueError("each tangent must be non-zero")
        if not np.allclose(hessian, hessian.T, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("each Hessian must be symmetric")
        projected_force = float(tangent @ _hessian_to_si(hessian, hessian_unit) @ tangent)
        effective_mass = float(np.dot(mass_vector_kg, tangent * tangent))
        angular = np.sign(projected_force) * np.sqrt(abs(projected_force) / effective_mass)
        values.append(float(angular / (2.0 * np.pi * LIGHT_SPEED_M_S * 100.0)))
    return values[0] if len(values) == 1 else tuple(values)


# Descriptive aliases used by different Polyrate/IRC workflows.
intrinsic_coordinate_frequencies = coordinate_frequencies
irc_coordinate_frequencies = coordinate_frequencies
path_coordinate_frequencies = coordinate_frequencies
coordinate_frequencies_from_hessian = hessian_coordinate_frequencies
