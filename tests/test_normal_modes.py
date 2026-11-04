from __future__ import annotations

import math

import numpy as np

from pyrate.constants import AMU_KG, LIGHT_SPEED_M_S
from pyrate.normal_modes import normal_modes


def test_single_cartesian_oscillator_without_projection() -> None:
    # Three Cartesian coordinates for one atom.  Only x has a force constant
    # chosen to produce 1000 cm^-1; y and z remain zero modes.
    target_cm1 = 1000.0
    omega = 2.0 * math.pi * LIGHT_SPEED_M_S * 100.0 * target_cm1
    force_constant = AMU_KG * omega**2
    hessian = np.diag([force_constant, 0.0, 0.0])
    result = normal_modes(
        hessian,
        [1.0],
        hessian_unit="j/m^2",
        remove_rigid_modes=False,
    )
    assert np.count_nonzero(result.frequencies_cm1) == 1
    assert math.isclose(result.frequencies_cm1[-1], target_cm1, rel_tol=1.0e-12)


def test_diatomic_rigid_projection_has_five_rigid_modes() -> None:
    # The numerical Hessian is zero; this test targets rank detection of the
    # translational/rotational basis for a linear molecule.
    result = normal_modes(
        np.zeros((6, 6)),
        [1.0, 1.0],
        coordinates_angstrom=[[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]],
        hessian_unit="j/m^2",
        remove_rigid_modes=True,
    )
    assert result.rigid_mode_count == 5
    assert result.frequencies_cm1.size == 0
