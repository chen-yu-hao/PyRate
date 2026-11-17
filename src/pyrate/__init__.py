"""PyRate: readable Python orchestration with native C++ kinetics kernels."""

from __future__ import annotations

from .calculation import CalculationResult, RateRow, calculate_rates
from .advanced_vtst import (
    DualLevelCorrection,
    IVTSTResult,
    VRCVTSTModel,
    VRCVTSTResult,
    apply_dual_level_correction,
    intrinsic_coordinate_free_energies,
    iv_tst_rate_si,
    ivtst_rate_si,
    vrc_vtst_rate_si,
)
from .environment import (
    CollisionModel,
    MasterEquation,
    SSQRRKModel,
    SolvationCorrection,
    SolventModel,
    SurfaceModel,
    pressure_dependent_rate_si,
    solve_master_equation,
    solvation_free_energy_kj_mol,
    surface_correction_kj_mol,
)
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
from .path_frequencies import (
    CoordinateFrequencyProfile,
    coordinate_frequencies,
    hessian_coordinate_frequencies,
    irc_coordinate_frequencies,
    projected_coordinate_frequency_cm1,
)
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
    "CoordinateFrequencyProfile",
    "CollisionModel",
    "CusRate",
    "DualLevelCorrection",
    "ElectronicState",
    "HAS_NATIVE",
    "HinderedRotor",
    "IVTSTResult",
    "LoadedCalculation",
    "MasterEquation",
    "NormalModeResult",
    "PathPoint",
    "RateRow",
    "Reaction",
    "ReactionPath",
    "Species",
    "SSQRRKModel",
    "SolvationCorrection",
    "SolventModel",
    "SurfaceModel",
    "TunnelingProfile",
    "VRCVTSTModel",
    "VRCVTSTResult",
    "apply_dual_level_correction",
    "backend_info",
    "cag_kappa",
    "calculate_rates",
    "coordinate_frequencies",
    "conventional_tst_rate_si",
    "cus_rate_si",
    "cvt_rate_si",
    "hessian_coordinate_frequencies",
    "intrinsic_coordinate_free_energies",
    "irc_coordinate_frequencies",
    "iv_tst_rate_si",
    "ivtst_rate_si",
    "legacy_to_calculation",
    "load_input",
    "load_toml",
    "muvt_and_us_rates_si",
    "normal_modes",
    "parse_toml_data",
    "path_tunneling_profile",
    "profile_muomt_kappa",
    "pressure_dependent_rate_si",
    "projected_coordinate_frequency_cm1",
    "reaction_eckart_kappa",
    "reaction_equilibrium_constants",
    "rovibrational_sum_of_states",
    "vibrational_density_of_states",
    "vibrational_sum_of_states",
    "solvation_free_energy_kj_mol",
    "solve_master_equation",
    "surface_correction_kj_mol",
    "vrc_vtst_rate_si",
]

__version__ = "0.2.0"
