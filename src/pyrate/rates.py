"""Conventional and variational transition-state-theory rate constants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .constants import (
    AVOGADRO_MOL,
    BOLTZMANN_J_K,
    GAS_CONSTANT_J_MOL_K,
    PLANCK_J_S,
    standard_concentration_mol_m3,
)
from .models import Reaction, Species
from .native import RefinedMaximum, refine_maximum
from .partition import standard_gibbs_energy_kj_mol


@dataclass(frozen=True, slots=True)
class CvtRate:
    """Variational bottleneck and its rate constant in SI molar units."""

    coordinate: float
    activation_gibbs_kj_mol: float
    rate_si: float
    sampled_coordinate: float
    sampled_activation_gibbs_kj_mol: float
    fit_points: int


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


def forward_reaction_path_degeneracy(reaction: Reaction) -> float:
    """Return the multiplicative forward-path degeneracy.

    Modern inputs normally leave this at one.  The legacy converter records
    Polyrate ``SIGMAF`` explicitly in transition-state metadata rather than
    folding it into a molecular rotational symmetry number or electronic
    degeneracy.  This keeps the molecular partition functions physically
    interpretable while reproducing the legacy rate convention.
    """

    raw: Any = reaction.transition_state.metadata.get(
        "reaction_path_degeneracy", 1.0
    )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("reaction_path_degeneracy must be a real number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("reaction_path_degeneracy must be finite and positive")
    return value


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
    return delta_g, rate * forward_reaction_path_degeneracy(reaction)


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
    ) * forward_reaction_path_degeneracy(reaction)
    return CvtRate(
        coordinate=refined.coordinate,
        activation_gibbs_kj_mol=refined.value,
        rate_si=rate,
        sampled_coordinate=coordinates[sampled_index],
        sampled_activation_gibbs_kj_mol=activation_gibbs[sampled_index],
        fit_points=refined.fit_points,
    )


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
