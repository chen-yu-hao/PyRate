from __future__ import annotations

import math

import numpy as np
import pytest

from pyrate.constants import AMU_KG, AVOGADRO_MOL, BOHR_ANGSTROM, LIGHT_SPEED_M_S
from pyrate.interpolation import CubicSpline, refined_grid
from pyrate.models import PathPoint, ReactionPath, Species
from pyrate.tunneling import path_tunneling_profile, profile_wkb_kappa


def test_not_a_knot_spline_reproduces_cubic_polynomials() -> None:
    nodes = np.array([-1.0, -0.4, 0.1, 0.7, 1.5, 2.0])

    def cubic(x: np.ndarray) -> np.ndarray:
        return 2.0 - 3.0 * x + 0.5 * x**2 + 1.25 * x**3

    spline = CubicSpline(nodes, cubic(nodes))
    query = np.linspace(-1.0, 2.0, 41)
    assert np.allclose(spline(query), cubic(query), rtol=0.0, atol=1.0e-11)
    assert np.allclose(spline(nodes), cubic(nodes), rtol=0.0, atol=1.0e-12)


def test_refined_grid_keeps_nodes_and_rejects_bad_input() -> None:
    fine = refined_grid((0.0, 1.0, 3.0), 4)
    assert fine.size == 9
    assert np.allclose(fine[::4], (0.0, 1.0, 3.0))
    assert np.all(np.diff(fine) > 0.0)
    assert np.allclose(refined_grid((0.0, 1.0), 1), (0.0, 1.0))
    with pytest.raises(ValueError):
        refined_grid((0.0, 1.0), 0)
    with pytest.raises(ValueError):
        CubicSpline((0.0, 0.0, 1.0), (1.0, 2.0, 3.0))


def _eckart_path(point_count: int) -> ReactionPath:
    """Symmetric Eckart barrier, 60 kJ/mol and 1500i cm^-1 at mu = 1 amu."""

    barrier_kj_mol = 60.0
    omega = 2.0 * math.pi * LIGHT_SPEED_M_S * 100.0 * 1500.0
    width_m = math.sqrt(2.0 * barrier_kj_mol * 1000.0 / AVOGADRO_MOL / (AMU_KG * omega**2))
    width_bohr = width_m / (BOHR_ANGSTROM * 1.0e-10)
    coordinates = np.linspace(-6.0 * width_bohr, 6.0 * width_bohr, point_count)
    points = tuple(
        PathPoint(
            coordinate=float(s),
            dividing_surface=Species(
                name=f"gts-{index}",
                mass_amu_override=10.0,
                rotor_type="atomic",
                electronic_energy_kj_mol=barrier_kj_mol / math.cosh(s / width_bohr) ** 2,
            ),
        )
        for index, s in enumerate(coordinates)
    )
    return ReactionPath(points=points, coordinate_unit="bohr", reduced_mass_amu=1.0)


@pytest.mark.parametrize("temperature_k", [200.0, 300.0, 600.0])
def test_spline_refinement_converges_zct_on_a_coarse_path(temperature_k: float) -> None:
    converged = profile_wkb_kappa(
        path_tunneling_profile(_eckart_path(1537), subdivisions=1), temperature_k
    )
    coarse = _eckart_path(49)  # 0.18 bohr between stored points
    linear = profile_wkb_kappa(path_tunneling_profile(coarse, subdivisions=1), temperature_k)
    splined = profile_wkb_kappa(path_tunneling_profile(coarse), temperature_k)
    assert abs(splined / converged - 1.0) < 2.0e-3
    assert abs(splined / converged - 1.0) < 0.2 * abs(linear / converged - 1.0)


def test_curvature_components_must_match_frequencies() -> None:
    def surface(index: int, energy: float) -> Species:
        return Species(
            name=f"gts-{index}",
            mass_amu_override=15.0,
            rotor_type="atomic",
            electronic_energy_kj_mol=energy,
            frequencies_cm1=(1500.0, 900.0, 400.0),
        )

    energies = (0.0, 20.0, 35.0, 18.0, 1.0)
    points = tuple(
        PathPoint(
            coordinate=-1.0 + 0.5 * index,
            dividing_surface=surface(index, energy),
            curvatures_bohr_inv=(0.3, 0.2),
        )
        for index, energy in enumerate(energies)
    )
    path = ReactionPath(points=points, coordinate_unit="bohr", reduced_mass_amu=1.0)
    with pytest.raises(ValueError, match="curvature components"):
        path_tunneling_profile(path, method="sct")


def test_curvatures_stay_aligned_when_a_mode_turns_imaginary() -> None:
    """A non-positive frequency must drop its own curvature, not shift the others."""

    def path_with(frequencies: tuple[float, ...], curvatures: tuple[float, ...]) -> ReactionPath:
        energies = (0.0, 20.0, 35.0, 18.0, 1.0)
        return ReactionPath(
            points=tuple(
                PathPoint(
                    coordinate=-1.0 + 0.5 * index,
                    dividing_surface=Species(
                        name=f"gts-{index}",
                        mass_amu_override=15.0,
                        rotor_type="atomic",
                        electronic_energy_kj_mol=energy,
                        frequencies_cm1=frequencies,
                        include_zpe=False,
                    ),
                    curvatures_bohr_inv=curvatures,
                )
                for index, energy in enumerate(energies)
            ),
            coordinate_unit="bohr",
            reduced_mass_amu=1.0,
        )

    with_imaginary = path_tunneling_profile(
        path_with((-150.0, 1500.0, 400.0), (5.0, 0.3, 0.1)), method="sct", subdivisions=1
    )
    reference = path_tunneling_profile(
        path_with((1500.0, 400.0), (0.3, 0.1)), method="sct", subdivisions=1
    )
    assert with_imaginary.effective_masses_amu is not None
    assert reference.effective_masses_amu is not None
    assert np.allclose(with_imaginary.effective_masses_amu, reference.effective_masses_amu)
