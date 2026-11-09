"""B2: H2O/HDO/D2O harmonic frequencies and the Teller–Redlich product rule.

Valence force field of Hoy, Mills & Strey, Mol. Phys. 24, 1265 (1972) (values to be
checked against the paper).  The Cartesian Hessian is built analytically as
H = Bᵀ F B with Wilson B-matrix rows for two bonds and the bond angle, then fed to
``pyrate.normal_modes``.  An independent GF-matrix solve in internal coordinates is
the reference.
"""

from __future__ import annotations

import math

import numpy as np

from pyrate.constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL, LIGHT_SPEED_M_S
from pyrate.geometry import principal_moments_amu_a2
from pyrate.models import Atom
from pyrate.normal_modes import normal_modes

from .common import check, rel

# 1 mdyn/Å = 100 N/m = 100 J/m² → per mol and per Å²:
MDYN_A_TO_KJ_MOL_A2 = 100.0 * AVOGADRO_MOL / 1000.0 * ANGSTROM_M**2

R_E = 0.9578
THETA_E = math.radians(104.48)
F_RR, F_RR2, F_RT, F_TT = 8.454, -0.101, 0.228, 0.697  # mdyn/Å, mdyn/Å, mdyn/rad, mdyn Å/rad²
M_O, M_H, M_D = 15.99491462, 1.00782503, 2.01410178
EXPERIMENT_HARMONIC = {  # Benedict, Gailar & Plyler (1956); check against the paper
    "H2O": (1648.5, 3832.2, 3942.5),
    "D2O": (1206.4, 2763.8, 2888.8),
}


def geometry() -> np.ndarray:
    s, c = math.sin(THETA_E / 2), math.cos(THETA_E / 2)
    return np.array([[0.0, 0.0, 0.0], [R_E * s, R_E * c, 0.0], [-R_E * s, R_E * c, 0.0]])


def b_matrix(x: np.ndarray) -> np.ndarray:
    """Wilson B matrix (3 × 9) for r1(O–H1), r2(O–H2), θ(H1–O–H2)."""

    o, h1, h2 = x
    b = np.zeros((3, 9))
    r1v, r2v = h1 - o, h2 - o
    r1, r2 = np.linalg.norm(r1v), np.linalg.norm(r2v)
    e1, e2 = r1v / r1, r2v / r2
    b[0, 0:3], b[0, 3:6] = -e1, e1
    b[1, 0:3], b[1, 6:9] = -e2, e2
    cos_t = float(np.dot(e1, e2))
    sin_t = math.sqrt(1.0 - cos_t * cos_t)
    s1 = (cos_t * e1 - e2) / (r1 * sin_t)
    s2 = (cos_t * e2 - e1) / (r2 * sin_t)
    b[2, 3:6], b[2, 6:9], b[2, 0:3] = s1, s2, -(s1 + s2)
    return b


def force_matrix() -> np.ndarray:
    k = MDYN_A_TO_KJ_MOL_A2
    return k * np.array([[F_RR, F_RR2, F_RT], [F_RR2, F_RR, F_RT], [F_RT, F_RT, F_TT]])


def gf_frequencies(masses: tuple[float, float, float]) -> np.ndarray:
    b = b_matrix(geometry())
    inv_m = np.repeat(1.0 / np.asarray(masses), 3)
    g = b @ np.diag(inv_m) @ b.T  # amu⁻¹ (Å or rad)
    eig = np.sort(np.linalg.eigvals(g @ force_matrix()).real)  # kJ mol⁻¹ Å⁻² amu⁻¹
    omega = np.sqrt(eig * 1000.0 / AVOGADRO_MOL / ANGSTROM_M**2 / AMU_KG)
    return omega / (2 * math.pi * LIGHT_SPEED_M_S * 100.0)


def cartesian_frequencies(masses: tuple[float, float, float], rotate: bool = False):
    x = geometry()
    hess = b_matrix(x).T @ force_matrix() @ b_matrix(x)
    if rotate:  # random rigid rotation of frame + Hessian
        rng = np.random.default_rng(7)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        x = x @ q.T + rng.normal(size=3)
        big = np.kron(np.eye(3), q)
        hess = big @ hess @ big.T
    return normal_modes(hess, masses, coordinates_angstrom=x, hessian_unit="kj/mol/angstrom^2")


def moments(masses: tuple[float, float, float]) -> tuple[float, float, float]:
    atoms = tuple(Atom("H", *xyz, mass_amu=m) for xyz, m in zip(geometry(), masses))
    return principal_moments_amu_a2(atoms)


def run() -> dict:
    sets = {"H2O": (M_O, M_H, M_H), "HDO": (M_O, M_H, M_D), "D2O": (M_O, M_D, M_D)}
    checks, table = [], {}
    for label, m in sets.items():
        result = cartesian_frequencies(m)
        f = np.sort(result.frequencies_cm1)
        gf = np.sort(gf_frequencies(m))
        table[label] = {"cartesian_cm1": f.tolist(), "gf_cm1": gf.tolist()}
        checks.append(check(f"{label}: max rel diff Cartesian vs GF", float(np.max(np.abs(f / gf - 1))), 1e-8))
        checks.append(check(f"{label}: rigid modes removed", float(result.rigid_mode_count), 6.0, mode="eq"))
        if label in EXPERIMENT_HARMONIC:
            dev = float(np.max(np.abs(f - np.sort(EXPERIMENT_HARMONIC[label]))))
            checks.append(check(f"{label}: max |ω − ω_exp| (cm⁻¹)", dev, 3.0))
    rotated = np.sort(cartesian_frequencies(sets["H2O"], rotate=True).frequencies_cm1)
    checks.append(check("H2O: rotated-frame invariance (max rel)",
                        float(np.max(np.abs(rotated / np.sort(table["H2O"]["cartesian_cm1"]) - 1))), 1e-10))

    for pair in (("H2O", "D2O"), ("H2O", "HDO")):
        a, b = sets[pair[0]], sets[pair[1]]
        lhs = float(np.prod(np.asarray(table[pair[1]]["cartesian_cm1"]) / np.asarray(table[pair[0]]["cartesian_cm1"])))
        rhs = float(np.prod([(ma / mb) ** 1.5 for ma, mb in zip(a, b)]) * (sum(b) / sum(a)) ** 1.5
                    * math.sqrt(np.prod(moments(b)) / np.prod(moments(a))))
        table[f"teller_redlich_{pair[0]}_{pair[1]}"] = {"lhs": lhs, "rhs": rhs}
        checks.append(check(f"Teller–Redlich {pair[0]}/{pair[1]} rel diff", rel(lhs, rhs), 1e-9))
    return {
        "case": "b02_water_normal_modes",
        "title": "H2O isotopologue harmonic frequencies and Teller–Redlich product rule",
        "references": ["Hoy, Mills & Strey, Mol. Phys. 24, 1265 (1972)",
                       "Benedict, Gailar & Plyler, J. Chem. Phys. 24, 1139 (1956)",
                       "Redlich, Z. Phys. Chem. B 28, 371 (1935)"],
        "checks": checks,
        "data": table,
    }
