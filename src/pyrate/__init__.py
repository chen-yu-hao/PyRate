"""PyRate: readable Python orchestration with native C++ kinetics kernels."""

from __future__ import annotations

from .calculation import CalculationResult, RateRow, calculate_rates
from .fixed_units import (
    FixedUnitDocument,
    FixedUnitEntry,
    FixedUnitPathPoint,
    FixedUnitSection,
    FixedUnitStationaryPoint,
    parse_fixed_unit,
    parse_fixed_unit_text,
    summarize_fixed_unit,
)
from .input import LoadedCalculation, load_input, load_toml, parse_toml_data
from .models import (
    Atom,
    CalculationOptions,
    ElectronicState,
    PathPoint,
    Reaction,
    ReactionPath,
    Species,
    TunnelingProfile,
)
from .native import HAS_NATIVE, backend_info
from .normal_modes import NormalModeResult, normal_modes
from .numerical_trace import (
    DEFAULT_TRACE_TOLERANCES,
    NumericalTrace,
    TraceComparisonReport,
    TracePoint,
    TraceTolerance,
    build_numerical_trace,
    compare_numerical_traces,
)
from .rates import conventional_tst_rate_si, cvt_rate_si
from .states import vibrational_density_of_states

__all__ = [
    "Atom",
    "CalculationOptions",
    "CalculationResult",
    "DEFAULT_TRACE_TOLERANCES",
    "ElectronicState",
    "FixedUnitDocument",
    "FixedUnitEntry",
    "FixedUnitPathPoint",
    "FixedUnitSection",
    "FixedUnitStationaryPoint",
    "HAS_NATIVE",
    "LoadedCalculation",
    "NormalModeResult",
    "NumericalTrace",
    "PathPoint",
    "RateRow",
    "Reaction",
    "ReactionPath",
    "Species",
    "TraceComparisonReport",
    "TracePoint",
    "TraceTolerance",
    "TunnelingProfile",
    "backend_info",
    "build_numerical_trace",
    "calculate_rates",
    "compare_numerical_traces",
    "conventional_tst_rate_si",
    "cvt_rate_si",
    "load_input",
    "load_toml",
    "normal_modes",
    "parse_fixed_unit",
    "parse_fixed_unit_text",
    "parse_toml_data",
    "summarize_fixed_unit",
    "vibrational_density_of_states",
]

__version__ = "0.1.0"
