"""PyRate: readable Python orchestration with native C++ kinetics kernels."""

from __future__ import annotations

from .calculation import CalculationResult, RateRow, calculate_rates
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
from .states import vibrational_density_of_states
from .rates import conventional_tst_rate_si, cvt_rate_si

__all__ = [
    "Atom",
    "CalculationOptions",
    "CalculationResult",
    "ElectronicState",
    "HAS_NATIVE",
    "LoadedCalculation",
    "NormalModeResult",
    "PathPoint",
    "RateRow",
    "Reaction",
    "ReactionPath",
    "Species",
    "TunnelingProfile",
    "backend_info",
    "calculate_rates",
    "conventional_tst_rate_si",
    "cvt_rate_si",
    "load_input",
    "load_toml",
    "normal_modes",
    "parse_toml_data",
    "vibrational_density_of_states",
]

__version__ = "0.1.0"
