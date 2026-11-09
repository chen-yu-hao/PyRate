"""B3: RRHO standard entropies vs NIST-JANAF and the H2 + D2 ⇌ 2HD equilibrium.

S° = −∂G°/∂T is taken by a central difference of ``standard_gibbs_energy_kj_mol``.
Masses follow the JANAF convention (natural-abundance atomic weights) for the
entropy comparison and isotopic masses for the isotope-exchange equilibrium.
"""

from __future__ import annotations

import math

from pyrate.constants import ATM_PA, BAR_PA, GAS_CONSTANT_J_MOL_K, WAVENUMBER_TO_KJ_MOL
from pyrate.models import Atom, ElectronicState, Species
from pyrate.partition import standard_gibbs_energy_kj_mol

from .common import check

W = WAVENUMBER_TO_KJ_MOL
M_H, M_D = 1.00782503, 2.01410178


def entropy(species: Species, t: float, h: float = 1.0e-3) -> float:
    return -(standard_gibbs_energy_kj_mol(species, t + h) - standard_gibbs_energy_kj_mol(species, t - h)) / (2 * h) * 1000.0


def diatomic(name: str, symbol: str, r_e: float, w_e: float, sigma: int, g: int = 1,
             masses: tuple[float, float] | None = None) -> Species:
    m1, m2 = masses if masses else (None, None)
    return Species(name, atoms=(Atom(symbol, 0, 0, 0, mass_amu=m1), Atom(symbol, r_e, 0, 0, mass_amu=m2)),
                   symmetry_number=sigma, frequencies_cm1=(w_e,), electronic_states=(ElectronicState(g, 0.0),))


def water() -> Species:
    r, th = 0.9578, math.radians(104.48)
    s, c = math.sin(th / 2), math.cos(th / 2)
    return Species("H2O", atoms=(Atom("O", 0, 0, 0, mass_amu=15.9994), Atom("H", r * s, r * c, 0),
                                 Atom("H", -r * s, r * c, 0)),
                   symmetry_number=2, frequencies_cm1=(3657.0, 1595.0, 3756.0))


# JANAF 4th ed. (1998), S°(298.15 K, 1 bar) in J mol⁻¹ K⁻¹; check against the tables.
JANAF_S298 = {"O": 161.059, "Cl": 165.190, "N2": 191.609, "O2": 205.147, "H2O": 188.834}


def run() -> dict:
    species = {
        "O": Species("O", mass_amu_override=15.9994, rotor_type="atomic",
                     electronic_states=(ElectronicState(5, 0.0), ElectronicState(3, 158.265 * W),
                                        ElectronicState(1, 226.977 * W))),
        "Cl": Species("Cl", mass_amu_override=35.453, rotor_type="atomic",
                      electronic_states=(ElectronicState(4, 0.0), ElectronicState(2, 882.35 * W))),
        "N2": diatomic("N2", "N", 1.09768, 2358.57, 2),
        "O2": diatomic("O2", "O", 1.20752, 1580.19, 2, g=3),
        "H2O": water(),
    }
    checks, table = [], {"entropy_298": {}, "entropy_T": {}, "k_h2_d2_hd": {}}
    for name, sp in species.items():
        s = entropy(sp, 298.15)
        table["entropy_298"][name] = {"pyrate": s, "janaf": JANAF_S298[name], "diff": s - JANAF_S298[name]}
        threshold = 0.01 if sp.rotor_type == "atomic" else 0.2
        checks.append(check(f"S°(298.15) {name} |PyRate − JANAF| (J/mol/K)", abs(s - JANAF_S298[name]), threshold))
        table["entropy_T"][name] = {str(t): entropy(sp, t) for t in (298.15, 500.0, 1000.0, 2000.0, 3000.0)}

    h2 = diatomic("H2", "H", 0.74144, 4401.21, 2, masses=(M_H, M_H))
    d2 = diatomic("D2", "H", 0.74144, 3115.50, 2, masses=(M_D, M_D))
    hd = diatomic("HD", "H", 0.74144, 3813.15, 1, masses=(M_H, M_D))

    def k_eq(t: float, p: float = BAR_PA) -> float:
        dg = (2 * standard_gibbs_energy_kj_mol(hd, t, p) - standard_gibbs_energy_kj_mol(h2, t, p)
              - standard_gibbs_energy_kj_mol(d2, t, p))
        return math.exp(-dg * 1000.0 / (GAS_CONSTANT_J_MOL_K * t))

    for t in (200.0, 298.15, 500.0, 1000.0, 3000.0, 10000.0):
        table["k_h2_d2_hd"][str(t)] = k_eq(t)
    checks.append(check("K(10000 K) → σ-ratio limit 4 (rel)", abs(k_eq(10000.0) / 4 - 1), 5e-3))
    checks.append(check("K independent of p° (1 bar vs 1 atm, rel)", abs(k_eq(298.15, ATM_PA) / k_eq(298.15) - 1), 1e-12))
    checks.append(check("K(298.15 K)", k_eq(298.15), None))
    return {
        "case": "b03_thermochemistry",
        "title": "RRHO standard entropies vs NIST-JANAF; H2 + D2 ⇌ 2HD equilibrium constant",
        "references": ["Chase, NIST-JANAF Thermochemical Tables, 4th ed. (1998)",
                       "Urey, J. Chem. Soc. 1947, 562", "Bigeleisen & Mayer, J. Chem. Phys. 15, 261 (1947)",
                       "Huber & Herzberg, Constants of Diatomic Molecules (1979)"],
        "checks": checks,
        "data": table,
    }
