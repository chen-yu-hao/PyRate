"""Conventional and variational transition-state-theory rate constants."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import (
    AVOGADRO_MOL,
    BOLTZMANN_J_K,
    GAS_CONSTANT_J_MOL_K,
    PLANCK_J_S,
    standard_concentration_mol_m3,
)
from .models import Reaction, Species
from .native import RefinedMaximum, refine_maximum, refine_minimum
from .partition import standard_gibbs_energy_kj_mol
from .states import microcanonical_path_factors


@dataclass(frozen=True, slots=True)
class CvtRate:
    """Variational bottleneck and its rate constant in SI molar units."""

    coordinate: float
    activation_gibbs_kj_mol: float
    rate_si: float
    sampled_coordinate: float
    sampled_activation_gibbs_kj_mol: float
    fit_points: int


@dataclass(frozen=True, slots=True)
class CusRate:
    """Canonical Unified Statistical (CUS) rate constant and extrema."""

    rate_si: float
    maxima_coordinates: tuple[float, ...]
    maxima_gibbs_kj_mol: tuple[float, ...]
    minima_coordinates: tuple[float, ...]
    minima_gibbs_kj_mol: tuple[float, ...]

    @property
    def has_intermediate_well(self) -> bool:
        return len(self.minima_coordinates) > 0


def activation_gibbs_kj_mol(
    reaction: Reaction, dividing_surface: Species, temperature_k: float
) -> float:
    """Return the standard Gibbs free energy of activation."""

    transition_gibbs = standard_gibbs_energy_kj_mol(
        dividing_surface, temperature_k, reaction.standard_pressure_pa
    )
    reactant_gibbs = sum(
        standard_gibbs_energy_kj_mol(
            reactant, temperature_k, reaction.standard_pressure_pa
        )
        for reactant in reaction.reactants
    )
    return transition_gibbs - reactant_gibbs


def reaction_equilibrium_constants(
    reaction: Reaction, temperature_k: float
) -> tuple[float, float, float]:
    """Return ``(delta_g_rxn_kj_mol, K_p, K_c_si)`` when products are defined."""

    if not reaction.products:
        raise ValueError("reaction_equilibrium_constants requires reaction.products")
    reactant_gibbs = sum(
        standard_gibbs_energy_kj_mol(r, temperature_k, reaction.standard_pressure_pa)
        for r in reaction.reactants
    )
    product_gibbs = sum(
        standard_gibbs_energy_kj_mol(p, temperature_k, reaction.standard_pressure_pa)
        for p in reaction.products
    )
    delta_g_kj_mol = product_gibbs - reactant_gibbs
    log_kp = -delta_g_kj_mol * 1000.0 / (GAS_CONSTANT_J_MOL_K * temperature_k)
    kp = math.exp(max(-700.0, min(700.0, log_kp)))
    delta_nu = len(reaction.products) - len(reaction.reactants)
    c0 = standard_concentration_mol_m3(temperature_k, reaction.standard_pressure_pa)
    kc_si = kp * (c0**delta_nu)
    return delta_g_kj_mol, kp, kc_si


def log_tst_rate_si(
    activation_gibbs_kj_mol_value: float,
    temperature_k: float,
    molecularity: int,
    standard_pressure_pa: float,
) -> float:
    """Natural log of an Eyring rate in SI molar concentration units.

    Unimolecular output is s⁻¹.  Bimolecular output is m³ mol⁻¹ s⁻¹.
    """

    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if molecularity not in {1, 2}:
        raise NotImplementedError("only uni- and bimolecular rates are supported")
    log_prefactor = math.log(BOLTZMANN_J_K * temperature_k / PLANCK_J_S)
    log_boltzmann = -activation_gibbs_kj_mol_value * 1000.0 / (
        GAS_CONSTANT_J_MOL_K * temperature_k
    )
    concentration = standard_concentration_mol_m3(
        temperature_k, standard_pressure_pa
    )
    return log_prefactor + log_boltzmann + (1 - molecularity) * math.log(concentration)


def tst_rate_from_activation_gibbs_si(
    activation_gibbs_kj_mol_value: float,
    temperature_k: float,
    molecularity: int,
    standard_pressure_pa: float,
) -> float:
    log_rate = log_tst_rate_si(
        activation_gibbs_kj_mol_value,
        temperature_k,
        molecularity,
        standard_pressure_pa,
    )
    if log_rate > 709.0:
        return math.inf
    if log_rate < -745.0:
        return 0.0
    return math.exp(log_rate)


def conventional_tst_rate_si(reaction: Reaction, temperature_k: float) -> tuple[float, float]:
    """Return ``(ΔG‡, k_TST)`` for the designated transition state."""

    delta_g = activation_gibbs_kj_mol(
        reaction, reaction.transition_state, temperature_k
    )
    rate = tst_rate_from_activation_gibbs_si(
        delta_g,
        temperature_k,
        reaction.molecularity,
        reaction.standard_pressure_pa,
    )
    return delta_g, rate


def cvt_rate_si(reaction: Reaction, temperature_k: float) -> CvtRate:
    """Return the canonical variational rate from a reaction-path grid.

    The generalized free energy of activation is maximized along the path,
    which is equivalent to minimizing the generalized TST rate.  A local
    polynomial fit using up to five path points refines the sampled maximum.
    """

    if reaction.path is None:
        raise ValueError("CVT requires reaction.path")
    coordinates = [point.coordinate for point in reaction.path.points]
    activation_gibbs = [
        activation_gibbs_kj_mol(
            reaction, point.dividing_surface, temperature_k
        )
        for point in reaction.path.points
    ]
    refined: RefinedMaximum = refine_maximum(coordinates, activation_gibbs)
    sampled_index = refined.sample_index
    rate = tst_rate_from_activation_gibbs_si(
        refined.value,
        temperature_k,
        reaction.molecularity,
        reaction.standard_pressure_pa,
    )
    return CvtRate(
        coordinate=refined.coordinate,
        activation_gibbs_kj_mol=refined.value,
        rate_si=rate,
        sampled_coordinate=coordinates[sampled_index],
        sampled_activation_gibbs_kj_mol=activation_gibbs[sampled_index],
        fit_points=refined.fit_points,
    )


def cus_rate_si(
    reaction: Reaction,
    temperature_k: float,
    *,
    max_bottlenecks: int = 10,
) -> CusRate:
    """Return the Canonical Unified Statistical (CUS) rate constant (Polyrate cusspl).

    When a single free-energy maximum exists along ``s``, ``k^CUS(T) == k^CVT(T)``.
    When multiple local maxima separated by intermediate wells exist,
    ``1 / k^CUS = sum_i (1 / k^GT(s_max,i)) - sum_j (1 / k^GT(s_min,j))``.
    """

    if reaction.path is None:
        raise ValueError("CUS requires reaction.path")
    coordinates = [point.coordinate for point in reaction.path.points]
    activation_gibbs = [
        activation_gibbs_kj_mol(reaction, point.dividing_surface, temperature_k)
        for point in reaction.path.points
    ]
    n_pts = len(coordinates)
    local_max_indices: list[int] = []
    for idx in range(1, n_pts - 1):
        if (
            activation_gibbs[idx] > activation_gibbs[idx - 1]
            and activation_gibbs[idx] >= activation_gibbs[idx + 1]
        ):
            local_max_indices.append(idx)

    if len(local_max_indices) <= 1:
        cvt = cvt_rate_si(reaction, temperature_k)
        return CusRate(
            rate_si=cvt.rate_si,
            maxima_coordinates=(cvt.coordinate,),
            maxima_gibbs_kj_mol=(cvt.activation_gibbs_kj_mol,),
            minima_coordinates=(),
            minima_gibbs_kj_mol=(),
        )

    # Keep top `max_bottlenecks` local maxima ordered by coordinate (matching polyag.F:2773).
    if len(local_max_indices) > max_bottlenecks:
        local_max_indices = sorted(
            sorted(local_max_indices, key=lambda i: activation_gibbs[i], reverse=True)[
                :max_bottlenecks
            ]
        )

    max_coords: list[float] = []
    max_gibbs: list[float] = []
    for idx in local_max_indices:
        window_x = coordinates[idx - 1 : idx + 2]
        window_y = activation_gibbs[idx - 1 : idx + 2]
        refined_max = refine_maximum(window_x, window_y)
        max_coords.append(refined_max.coordinate)
        max_gibbs.append(refined_max.value)

    min_coords: list[float] = []
    min_gibbs: list[float] = []
    for k in range(len(local_max_indices) - 1):
        left_idx = local_max_indices[k]
        right_idx = local_max_indices[k + 1]
        min_idx = min(
            range(left_idx, right_idx + 1), key=lambda i: activation_gibbs[i]
        )
        if 0 < min_idx < n_pts - 1:
            window_x = coordinates[min_idx - 1 : min_idx + 2]
            window_y = activation_gibbs[min_idx - 1 : min_idx + 2]
            refined_min = refine_minimum(window_x, window_y)
            min_coords.append(refined_min.coordinate)
            min_gibbs.append(refined_min.value)
        else:
            min_coords.append(coordinates[min_idx])
            min_gibbs.append(activation_gibbs[min_idx])

    inv_cus = 0.0
    for g_max in max_gibbs:
        k_max = tst_rate_from_activation_gibbs_si(
            g_max, temperature_k, reaction.molecularity, reaction.standard_pressure_pa
        )
        if k_max > 0.0:
            inv_cus += 1.0 / k_max
    for g_min in min_gibbs:
        k_min = tst_rate_from_activation_gibbs_si(
            g_min, temperature_k, reaction.molecularity, reaction.standard_pressure_pa
        )
        if k_min > 0.0:
            inv_cus -= 1.0 / k_min

    cus_rate = (
        1.0 / inv_cus
        if inv_cus > 0.0
        else cvt_rate_si(reaction, temperature_k).rate_si
    )
    return CusRate(
        rate_si=cus_rate,
        maxima_coordinates=tuple(max_coords),
        maxima_gibbs_kj_mol=tuple(max_gibbs),
        minima_coordinates=tuple(min_coords),
        minima_gibbs_kj_mol=tuple(min_gibbs),
    )


def muvt_and_us_rates_si(
    reaction: Reaction,
    temperature_k: float,
    *,
    cvt_result: CvtRate | None = None,
) -> tuple[float, float]:
    """Return ``(k_muvt_si, k_us_si)`` using microcanonical sum-of-states minimization."""

    if reaction.path is None:
        raise ValueError("muVT/US requires reaction.path")
    cvt = cvt_result if cvt_result is not None else cvt_rate_si(reaction, temperature_k)
    gamma_muvt, gamma_us = microcanonical_path_factors(reaction.path, temperature_k)
    return cvt.rate_si * gamma_muvt, cvt.rate_si * gamma_us



def canonical_rate_unit(molecularity: int) -> str:
    if molecularity == 1:
        return "s^-1"
    if molecularity == 2:
        return "m^3 mol^-1 s^-1"
    return f"m^{3 * (molecularity - 1)} mol^{1 - molecularity} s^-1"


def normalize_rate_unit(unit: str, molecularity: int) -> str:
    normalized = unit.strip().lower().replace("³", "^3").replace("·", " ")
    aliases = {
        "auto": canonical_rate_unit(molecularity),
        "si": canonical_rate_unit(molecularity),
        "s-1": "s^-1",
        "s^-1": "s^-1",
        "m3 mol-1 s-1": "m^3 mol^-1 s^-1",
        "m^3 mol^-1 s^-1": "m^3 mol^-1 s^-1",
        "cm3 mol-1 s-1": "cm^3 mol^-1 s^-1",
        "cm^3 mol^-1 s^-1": "cm^3 mol^-1 s^-1",
        "l mol-1 s-1": "L mol^-1 s^-1",
        "l mol^-1 s^-1": "L mol^-1 s^-1",
        "cm3 molecule-1 s-1": "cm^3 molecule^-1 s^-1",
        "cm^3 molecule^-1 s^-1": "cm^3 molecule^-1 s^-1",
    }
    try:
        result = aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported rate unit {unit!r}") from exc
    if molecularity == 1 and result != "s^-1":
        raise ValueError("unimolecular rates must use s^-1")
    if molecularity == 2 and result == "s^-1":
        raise ValueError("bimolecular rates require a second-order unit")
    return result


def convert_rate_from_si(value: float, molecularity: int, unit: str) -> tuple[float, str]:
    """Convert a rate from s⁻¹ or m³ mol⁻¹ s⁻¹ to a requested unit."""

    target = normalize_rate_unit(unit, molecularity)
    if molecularity == 1:
        return value, target
    factors = {
        "m^3 mol^-1 s^-1": 1.0,
        "cm^3 mol^-1 s^-1": 1.0e6,
        "L mol^-1 s^-1": 1.0e3,
        "cm^3 molecule^-1 s^-1": 1.0e6 / AVOGADRO_MOL,
    }
    return value * factors[target], target
