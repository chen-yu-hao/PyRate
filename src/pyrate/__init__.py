"""PyRate: readable Python orchestration with native C++ kinetics kernels."""

from __future__ import annotations

from .calculation import CalculationResult, RateRow, calculate_rates
from .input import (
    LoadedCalculation,
    legacy_to_calculation,
    load_input,
    load_toml,
    parse_toml_data,
)
from .models import (
    Atom,
    CalculationOptions,
    ElectronicState,
    HinderedRotor,
    PathPoint,
    Reaction,
    ReactionPath,
    Species,
    TunnelingProfile,
)
from .native import HAS_NATIVE, backend_info
from .normal_modes import NormalModeResult, normal_modes
from .rates import (
    CusRate,
    conventional_tst_rate_si,
    cus_rate_si,
    cvt_rate_si,
    muvt_and_us_rates_si,
    reaction_equilibrium_constants,
)
from .states import (
    rovibrational_sum_of_states,
    vibrational_density_of_states,
    vibrational_sum_of_states,
)
from .tunneling import (
    cag_kappa,
    path_tunneling_profile,
    profile_muomt_kappa,
    reaction_eckart_kappa,
)

__all__ = [
    "Atom",
    "CalculationOptions",
    "CalculationResult",
    "CusRate",
    "ElectronicState",
    "HAS_NATIVE",
    "HinderedRotor",
    "LoadedCalculation",
    "NormalModeResult",
    "PathPoint",
    "RateRow",
    "Reaction",
    "ReactionPath",
    "Species",
    "TunnelingProfile",
    "backend_info",
    "cag_kappa",
    "calculate_rates",
    "conventional_tst_rate_si",
    "cus_rate_si",
    "cvt_rate_si",
    "legacy_to_calculation",
    "load_input",
    "load_toml",
    "muvt_and_us_rates_si",
    "normal_modes",
    "parse_toml_data",
    "path_tunneling_profile",
    "profile_muomt_kappa",
    "reaction_eckart_kappa",
    "reaction_equilibrium_constants",
    "rovibrational_sum_of_states",
    "vibrational_density_of_states",
    "vibrational_sum_of_states",
]

__version__ = "0.2.0"
