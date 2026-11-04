from __future__ import annotations

from dataclasses import replace

import pytest

from pyrate.models import CalculationOptions, Reaction, Species, TunnelingProfile
from pyrate.numerical_trace import (
    NumericalTrace,
    TracePoint,
    build_numerical_trace,
    compare_numerical_traces,
)


def _reaction() -> Reaction:
    reactant = Species(
        name="R",
        mass_amu_override=2.0,
        rotor_type="atomic",
        electronic_energy_kj_mol=0.0,
        frequencies_cm1=(600.0,),
    )
    transition_state = Species(
        name="TS",
        mass_amu_override=2.0,
        rotor_type="atomic",
        electronic_energy_kj_mol=35.0,
        frequencies_cm1=(400.0, -1000.0),
    )
    product = Species(
        name="P",
        mass_amu_override=2.0,
        rotor_type="atomic",
        electronic_energy_kj_mol=-5.0,
        frequencies_cm1=(700.0,),
    )
    return Reaction(
        reactants=(reactant,),
        transition_state=transition_state,
        products=(product,),
    )


def test_trace_contains_partition_zpe_activation_kappa_and_rates() -> None:
    trace = build_numerical_trace(
        _reaction(),
        CalculationOptions(
            temperatures_k=(300.0, 600.0),
            method="tst",
            tunneling="wigner",
        ),
        name="wigner-case",
    )

    stages = {point.stage for point in trace.points}
    assert {"partition", "zpe", "free_energy", "activation_gibbs", "kappa", "rate"} <= stages
    assert "action" not in stages
    assert len([point for point in trace.points if point.stage == "kappa"]) == 2
    assert NumericalTrace.from_dict(trace.as_dict()) == trace


def test_trace_comparison_localizes_error_to_physical_stage() -> None:
    reference = build_numerical_trace(
        _reaction(),
        CalculationOptions(
            temperatures_k=(300.0,),
            method="tst",
            tunneling="wigner",
        ),
        name="reference",
    )
    changed_points = tuple(
        replace(point, value=point.value * 1.02)
        if point.stage == "kappa"
        else point
        for point in reference.points
    )
    actual = NumericalTrace(name="actual", points=changed_points, metadata=reference.metadata)

    report = compare_numerical_traces(reference, actual)

    assert not report.success
    assert report.failed == 1
    assert report.missing == 0
    assert report.stage_summary()["kappa"]["fail"] == 1
    assert report.stage_summary()["partition"]["fail"] == 0


def test_trace_comparison_reports_missing_and_strict_unexpected_points() -> None:
    reference = NumericalTrace(
        name="reference",
        points=(TracePoint("zpe", "R.zpe", 1.0, "kJ mol^-1"),),
    )
    actual = NumericalTrace(
        name="actual",
        points=(TracePoint("rate", "extra", 2.0, "s^-1", temperature_k=300.0),),
    )

    relaxed = compare_numerical_traces(reference, actual)
    strict = compare_numerical_traces(reference, actual, strict=True)

    assert relaxed.missing == 1
    assert relaxed.unexpected == 1
    assert not relaxed.success
    assert not strict.success


def test_wkb_trace_contains_energy_resolved_action_grid() -> None:
    profile = TunnelingProfile(
        coordinates_angstrom=(-1.0, 0.0, 1.0),
        energies_kj_mol=(0.0, 25.0, 0.0),
        reduced_mass_amu=1.0,
        energy_grid_points=100,
    )
    trace = build_numerical_trace(
        _reaction(),
        CalculationOptions(
            temperatures_k=(300.0,),
            method="tst",
            tunneling="wkb",
            tunneling_profile=profile,
        ),
        action_grid_points=7,
    )

    actions = [point for point in trace.points if point.stage == "action"]
    assert len(actions) == 7
    assert actions[0].energy_kj_mol == 0.0
    assert actions[-1].energy_kj_mol == 25.0
    assert actions[-1].value == pytest.approx(0.0, abs=1.0e-14)
