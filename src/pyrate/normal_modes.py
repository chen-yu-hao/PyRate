"""Cartesian Hessian mass weighting and normal-mode analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from .constants import AMU_KG, ANGSTROM_M, BOHR_M, HARTREE_J, LIGHT_SPEED_M_S

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NormalModeResult:
    """Normal-mode frequencies and mass-weighted eigenvectors."""

    frequencies_cm1: FloatArray
    eigenvectors_mass_weighted: FloatArray
    rigid_mode_count: int

    @property
    def imaginary_frequencies_cm1(self) -> FloatArray:
        return self.frequencies_cm1[self.frequencies_cm1 < 0.0]

    @property
    def real_frequencies_cm1(self) -> FloatArray:
        return self.frequencies_cm1[self.frequencies_cm1 > 0.0]


def _hessian_to_si(
    hessian: FloatArray,
    unit: Literal["hartree/bohr^2", "j/m^2", "kj/mol/angstrom^2"],
) -> FloatArray:
    if unit == "hartree/bohr^2":
        return hessian * HARTREE_J / BOHR_M**2
    if unit == "j/m^2":
        return hessian.copy()
    if unit == "kj/mol/angstrom^2":
        from .constants import AVOGADRO_MOL

        return hessian * 1000.0 / AVOGADRO_MOL / ANGSTROM_M**2
    raise ValueError(f"unsupported Hessian unit {unit!r}")


def _rigid_basis(
    masses_kg: FloatArray, coordinates_m: FloatArray
) -> tuple[FloatArray, int]:
    atom_count = masses_kg.size
    center = np.sum(coordinates_m * masses_kg[:, None], axis=0) / masses_kg.sum()
    centered = coordinates_m - center
    vectors: list[FloatArray] = []

    square_roots = np.sqrt(masses_kg)
    for axis in np.eye(3):
        vectors.append((square_roots[:, None] * axis[None, :]).reshape(-1))
    for axis in np.eye(3):
        displacement = np.cross(np.broadcast_to(axis, centered.shape), centered)
        vectors.append((square_roots[:, None] * displacement).reshape(-1))

    basis_matrix = np.column_stack(vectors)
    u, singular_values, _ = np.linalg.svd(basis_matrix, full_matrices=False)
    tolerance = max(basis_matrix.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    return u[:, :rank], rank


def normal_modes(
    hessian: Sequence[Sequence[float]] | FloatArray,
    masses_amu: Sequence[float],
    *,
    coordinates_angstrom: Sequence[Sequence[float]] | FloatArray | None = None,
    hessian_unit: Literal[
        "hartree/bohr^2", "j/m^2", "kj/mol/angstrom^2"
    ] = "hartree/bohr^2",
    remove_rigid_modes: bool = True,
    zero_threshold_cm1: float = 10.0,
) -> NormalModeResult:
    """Diagonalize a Cartesian Hessian and return signed frequencies.

    A negative Hessian eigenvalue is represented by a negative wavenumber.
    When coordinates are supplied, mass-weighted translations and rotations are
    projected before diagonalization.  NumPy's optimized symmetric LAPACK
    backend performs the dense eigensolve; all chemistry-specific preparation
    remains explicit here for auditability.
    """

    masses = np.asarray(masses_amu, dtype=float)
    if masses.ndim != 1 or masses.size == 0 or np.any(masses <= 0.0):
        raise ValueError("masses_amu must be a non-empty vector of positive masses")
    dimension = int(3 * masses.size)
    matrix = np.asarray(hessian, dtype=float)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"Hessian must have shape {(dimension, dimension)}")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("Hessian must be symmetric")

    hessian_si = _hessian_to_si(matrix, hessian_unit)
    coordinate_masses = np.repeat(masses * AMU_KG, 3)
    mass_weighted = hessian_si / np.sqrt(
        coordinate_masses[:, None] * coordinate_masses[None, :]
    )

    rigid_count = 0
    if remove_rigid_modes:
        if coordinates_angstrom is None:
            raise ValueError("coordinates_angstrom are required to remove rigid modes")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (masses.size, 3):
            raise ValueError(f"coordinates must have shape {(masses.size, 3)}")
        basis, rigid_count = _rigid_basis(masses * AMU_KG, coordinates * ANGSTROM_M)
        projector = np.eye(dimension) - basis @ basis.T
        mass_weighted = projector.T @ mass_weighted @ projector
        mass_weighted = 0.5 * (mass_weighted + mass_weighted.T)

    eigenvalues, eigenvectors = np.linalg.eigh(mass_weighted)
    angular_frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    frequencies = angular_frequencies / (2.0 * np.pi * LIGHT_SPEED_M_S * 100.0)
    frequencies[np.abs(frequencies) < zero_threshold_cm1] = 0.0

    if remove_rigid_modes:
        keep = np.abs(frequencies) >= zero_threshold_cm1
        frequencies = frequencies[keep]
        eigenvectors = eigenvectors[:, keep]

    return NormalModeResult(
        frequencies_cm1=frequencies,
        eigenvectors_mass_weighted=eigenvectors,
        rigid_mode_count=rigid_count,
    )
