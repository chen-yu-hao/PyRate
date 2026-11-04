"""Geometry utilities for rigid-rotor partition functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .models import Atom, Species

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RotorGeometry:
    """Principal moments and the inferred rotor class."""

    rotor_type: str
    moments_amu_a2: tuple[float, float, float]


def center_of_mass_angstrom(atoms: tuple[Atom, ...]) -> FloatArray:
    if not atoms:
        raise ValueError("at least one atom is required")
    masses = np.asarray([atom.mass_amu for atom in atoms], dtype=float)
    coordinates = np.asarray([atom.coordinates_angstrom for atom in atoms], dtype=float)
    return np.sum(coordinates * masses[:, None], axis=0) / masses.sum()


def inertia_tensor_amu_a2(atoms: tuple[Atom, ...]) -> FloatArray:
    """Return the center-of-mass inertia tensor in amu Å²."""

    if not atoms:
        raise ValueError("at least one atom is required")
    center = center_of_mass_angstrom(atoms)
    tensor = np.zeros((3, 3), dtype=float)
    for atom in atoms:
        vector = np.asarray(atom.coordinates_angstrom, dtype=float) - center
        radius_squared = float(np.dot(vector, vector))
        tensor += float(atom.mass_amu) * (
            radius_squared * np.eye(3, dtype=float) - np.outer(vector, vector)
        )
    return tensor


def principal_moments_amu_a2(atoms: tuple[Atom, ...]) -> tuple[float, float, float]:
    values = np.linalg.eigvalsh(inertia_tensor_amu_a2(atoms))
    values[np.abs(values) < 1.0e-12] = 0.0
    values = np.maximum(values, 0.0)
    return tuple(float(value) for value in np.sort(values))  # type: ignore[return-value]


def classify_rotor(
    moments_amu_a2: tuple[float, float, float], atom_count: int, tolerance: float = 1.0e-7
) -> str:
    if atom_count <= 1:
        return "atomic"
    smallest, _, largest = moments_amu_a2
    if largest <= 0.0:
        return "atomic"
    if smallest / largest <= tolerance:
        return "linear"
    return "nonlinear"


def rotor_geometry(species: Species) -> RotorGeometry:
    if species.moments_of_inertia_amu_a2 is None:
        if species.rotor_type == "atomic" and not species.atoms:
            moments = (0.0, 0.0, 0.0)
        else:
            moments = principal_moments_amu_a2(species.atoms)
    else:
        supplied = sorted(species.moments_of_inertia_amu_a2)
        if len(supplied) == 1:
            moments = (0.0, supplied[0], supplied[0])
        elif len(supplied) == 2:
            moments = (0.0, supplied[0], supplied[1])
        else:
            moments = (supplied[0], supplied[1], supplied[2])

    rotor_type = species.rotor_type
    if rotor_type == "auto":
        inferred_atom_count = len(species.atoms)
        if inferred_atom_count == 0 and any(moment > 0.0 for moment in moments):
            inferred_atom_count = 2
        rotor_type = classify_rotor(moments, inferred_atom_count)
    return RotorGeometry(rotor_type=rotor_type, moments_amu_a2=moments)
