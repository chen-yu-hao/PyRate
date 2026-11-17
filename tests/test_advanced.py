from __future__ import annotations

import math

import numpy as np

from pyrate.advanced import (
    SSQRRKModel,
    SolventCorrection,
    SurfaceCorrection,
    VRCWeightedSample,
    curvilinear_frequencies,
    pressure_dependent_rate,
    rrkm_microcanonical_rate,
    vrc_vtst_rate,
)
from pyrate.calculation import calculate_rates
from pyrate.models import (
    CalculationOptions,
    DualLevelCorrection as ModelDualLevelCorrection,
    EnvironmentModel,
    PathPoint,
    PressureDependence,
    Reaction,
    ReactionPath,
    Species,
    VrcModel,
    VrcSample,
)


def _species(name: str, energy: float) -> Species:
    return Species(
        name=name,
        mass_amu_override=30.0,
        rotor_type="nonlinear",
        moments_of_inertia_amu_a2=(10.0, 15.0, 20.0),
        electronic_energy_kj_mol=energy,
        frequencies_cm1=(500.0, 800.0, 1200.0),
    )


def test_curvilinear_wilson_gf_returns_signed_modes() -> None:
    frequencies = curvilinear_frequencies(
        [[4.0, 0.0], [0.0, -1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [1.0],
    )
    assert frequencies == (-1.0, 2.0)


def test_pressure_and_rrkm_helpers_are_finite_and_monotonic() -> None:
    rates = pressure_dependent_rate((1.0e3, 1.0e5, 1.0e7), 1.0e-28, 1.0e5, 300.0)
    assert np.all(np.isfinite(rates))
    assert rates[0] < rates[-1] <= 1.0e5
    micro = rrkm_microcanonical_rate((0.0, 100.0), (0.0, 10.0), (0.0, 2.0), grain_cm1=10.0)
    assert micro[0] == 0.0 and micro[1] > 0.0


def test_vrc_weighted_quadrature_selects_smallest_flux() -> None:
    result = vrc_vtst_rate(
        [
            VRCWeightedSample(0.0, 40.0, weight=1.0),
            VRCWeightedSample(1.0, 35.0, weight=1.0),
        ],
        300.0,
    )
    assert result.bottleneck_coordinate == 0.0
    assert result.rate_si > 0.0


def test_calculation_integrates_dual_level_vrc_pressure_and_environment() -> None:
    path = ReactionPath(
        tuple(
            PathPoint(coordinate=x, dividing_surface=_species(f"p{x}", 60.0 - 6.0 * x * x))
            for x in (-1.0, 0.0, 1.0)
        ),
        coordinate_system="curv1",
    )
    reaction = Reaction(
        reactants=(_species("A", 0.0),),
        transition_state=_species("TS", 50.0),
        path=path,
        dual_level=ModelDualLevelCorrection(
            method="ispe",
            coordinates=(-1.0, 0.0, 1.0),
            low_level_energies_kj_mol=(60.0, 60.0, 60.0),
            high_level_energies_kj_mol=(61.0, 63.0, 61.0),
        ),
        vrc=VrcModel((VrcSample(0.0, 55.0), VrcSample(1.0, 56.0))),
        pressure_dependence=PressureDependence((1.0e3, 1.0e5)),
        environment=EnvironmentModel(
            phase="solvent", species_solvation_kj_mol={"TS": -5.0, "A": -2.0}
        ),
    )
    result = calculate_rates(
        reaction,
        CalculationOptions(
            temperatures_k=(300.0,), method="all", pressures_pa=(1.0e3, 1.0e5)
        ),
    )
    row = result.rows[0]
    assert row.k_ivtst is not None and row.k_vrc_vtst is not None
    assert len(row.pressure_rates) == 2
    assert row.intrinsic_coordinate_system == "curv1"
    assert math.isfinite(row.environment_correction_kj_mol)


def test_environment_correction_helpers() -> None:
    a, ts = _species("A", 0.0), _species("TS", 20.0)
    solvent = SolventCorrection({"A": -2.0, "TS": -5.0})
    surface = SurfaceCorrection(site_density_m2=1.0e19, binding_energies_kj_mol={"TS": -3.0})
    assert solvent.correction_kj_mol(ts, (a,)) < 0.0
    assert surface.rate_multiplier(300.0, ts, (a,)) > 0.0

