"""B5 (CI part): conventional TST against an independent Eyring implementation,
standard-state invariance, and unit conversion, using ``examples/wkb_tst.toml``.

The literature H + H2 reproduction needs transcribed LSTH saddle-point data and is
kept out of CI until those values are checked against the original papers.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

from pyrate import load_toml
from pyrate.constants import (
    ATM_PA,
    AVOGADRO_MOL,
    BOLTZMANN_J_K,
    GAS_CONSTANT_J_MOL_K,
    PLANCK_J_S,
    WAVENUMBER_TO_J,
)
from pyrate.rates import conventional_tst_rate_si, convert_rate_from_si

from .common import check, rel

ROOT = Path(__file__).resolve().parents[1]


def independent_rate(reaction, t: float) -> float:
    """k = (kT/h) · (Q‡/V) / Π(Q_R/V) · exp(−ΔE0/kT), molar SI units, written from scratch."""

    kt = BOLTZMANN_J_K * t

    def q_per_volume(sp) -> tuple[float, float]:
        m = sp.mass_amu * 1.66053906660e-27
        q = (2 * math.pi * m * kt / PLANCK_J_S**2) ** 1.5
        if sp.rotor_type == "linear":
            inertia = sp.moments_of_inertia_amu_a2[0] * 1.66053906660e-27 * 1e-20
            q *= 8 * math.pi**2 * inertia * kt / PLANCK_J_S**2 / sp.symmetry_number
        for nu in sp.positive_frequencies_cm1:
            q /= 1 - math.exp(-nu * WAVENUMBER_TO_J / kt)
        q *= sum(s.degeneracy * math.exp(-s.energy_kj_mol * 1000 / (GAS_CONSTANT_J_MOL_K * t))
                 for s in sp.electronic_states)
        e0 = sp.electronic_energy_kj_mol * 1000 / AVOGADRO_MOL + 0.5 * WAVENUMBER_TO_J * sum(sp.positive_frequencies_cm1)
        return q, e0

    qt, et = q_per_volume(reaction.transition_state)
    ratio, e_r = qt, 0.0
    for r in reaction.reactants:
        q, e = q_per_volume(r)
        ratio /= q
        e_r += e
    return kt / PLANCK_J_S * ratio * math.exp(-(et - e_r) / kt) * AVOGADRO_MOL  # m³ mol⁻¹ s⁻¹


def run() -> dict:
    reaction = load_toml(ROOT / "examples" / "wkb_tst.toml").reaction
    atm = dataclasses.replace(reaction, standard_pressure_pa=ATM_PA)
    checks, data = [], {}
    worst_formula = worst_pressure = worst_unit = 0.0
    for t in (200.0, 300.0, 500.0, 1000.0, 2000.0):
        _, k = conventional_tst_rate_si(reaction, t)
        _, k_atm = conventional_tst_rate_si(atm, t)
        ref = independent_rate(reaction, t)
        worst_formula = max(worst_formula, rel(k, ref))
        worst_pressure = max(worst_pressure, rel(k_atm, k))
        cm3, _ = convert_rate_from_si(k, 2, "cm^3 molecule^-1 s^-1")
        worst_unit = max(worst_unit, rel(cm3, k * 1e6 / AVOGADRO_MOL))
        data[str(t)] = {"k_m3_mol_s": k, "k_cm3_molecule_s": cm3, "independent": ref}
    checks.append(check("k_TST vs independent Eyring formula (max rel)", worst_formula, 1e-10))
    checks.append(check("k_TST invariant to p° (1 bar vs 1 atm, max rel)", worst_pressure, 1e-12))
    checks.append(check("unit conversion to cm³ molecule⁻¹ s⁻¹ (max rel)", worst_unit, 1e-14))
    return {
        "case": "b05_tst_consistency",
        "title": "Conventional TST: independent formula, standard-state invariance, units",
        "references": ["Eyring, J. Chem. Phys. 3, 107 (1935)", "Evans & Polanyi, Trans. Faraday Soc. 31, 875 (1935)"],
        "checks": checks,
        "data": data,
    }
