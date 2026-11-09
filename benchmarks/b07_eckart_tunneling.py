"""B7: symmetric Eckart barrier V = V0 sech²(x/a).

* WKB action vs the closed form θ(E) = π a √(2μ) (√V0 − √E)/ħ (grid convergence);
* canonical κ(T) of PyRate's 1D WKB vs the exact quantum Eckart result
  (Landau–Lifshitz transmission), a uniform semiclassical reference
  (Kemble P = 1/(1+e^{2θ}) with nonclassical reflection), Bell's parabolic
  formula and Wigner's correction.

PyRate's default (uniform) kernel must reproduce the independent uniform
reference and, for a parabolic barrier, Bell's exact result.  The legacy
"primitive" form (P = e^{−2θ}, P = 1 above the top) is reported for comparison;
it overestimates κ at high temperature (κ − 1 ≈ u/2π instead of O(u²)).
"""

from __future__ import annotations

import math

import numpy as np

from pyrate.constants import AMU_KG, AVOGADRO_MOL, BOLTZMANN_J_K, HBAR_J_S, LIGHT_SPEED_M_S
from pyrate.native import thermal_wkb_kappa, wigner_kappa, wkb_exponent

from .common import check

V0_KJ, MU_AMU, NU_IM = 40.0, 1.0, 1500.0
MU = MU_AMU * AMU_KG
V0 = V0_KJ * 1e3 / AVOGADRO_MOL
OMEGA = 2 * math.pi * LIGHT_SPEED_M_S * 100 * NU_IM
A = math.sqrt(2 * V0 / (MU * OMEGA**2))
TEMPERATURES = (200.0, 250.0, 300.0, 400.0, 600.0, 1000.0, 2000.0, 5000.0)


def profile(n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    x = np.linspace(-12 * A, 12 * A, n)
    return tuple(x), tuple(V0 / np.cosh(x / A) ** 2)


def gamma_exact(e: float) -> float:
    return 2 * math.pi * A * math.sqrt(2 * MU) * (math.sqrt(V0) - math.sqrt(e)) / HBAR_J_S


def p_exact(e: float) -> float:
    k = math.sqrt(2 * MU * e) / HBAR_J_S
    d = 8 * MU * V0 * A * A / HBAR_J_S**2 - 1.0
    x = math.pi * k * A
    y = 0.5 * math.pi * math.sqrt(d)
    # sinh²x / (sinh²x + cosh²y), evaluated stably
    return 1.0 / (1.0 + (math.cosh(y) / math.sinh(x)) ** 2) if x > 0 else 0.0


def _boltzmann_integral(p, t: float, n: int = 40001) -> float:
    beta = 1 / (BOLTZMANN_J_K * t)
    e = np.linspace(1e-6 * V0, V0 + 50 / beta, n)
    vals = np.array([p(x) for x in e]) * np.exp(-beta * (e - V0)) * beta
    return float(np.sum((vals[1:] + vals[:-1]) * np.diff(e)) / 2)


def kappa_exact(t: float) -> float:
    return _boltzmann_integral(p_exact, t)


def kappa_uniform(t: float) -> float:
    def p(e: float) -> float:
        if e <= V0:
            return 1 / (1 + math.exp(gamma_exact(e)))
        if e >= 2 * V0:
            return 1.0
        return 1 - 1 / (1 + math.exp(gamma_exact(2 * V0 - e)))
    return _boltzmann_integral(p, t)


def run() -> dict:
    checks, data = [], {"parameters": {"V0_kJ_mol": V0_KJ, "mu_amu": MU_AMU, "nu_imag_cm1": NU_IM, "a_m": A}}

    errors = {}
    for n in (501, 1001, 2001, 4001, 8001):
        x, v = profile(n)
        errors[n] = max(abs(wkb_exponent(x, v, f * V0, MU) / gamma_exact(f * V0) - 1) for f in (0.1, 0.5, 0.9))
    data["action_rel_error_vs_points"] = {str(k): val for k, val in errors.items()}
    order = math.log2(errors[4001] / errors[8001])
    # Turning points fall between grid nodes, so convergence is irregular (~O(h^1.5)).
    checks.append(check("WKB action: observed order 4001→8001 points", order, None))
    checks.append(check("WKB action rel error, 8001 points", errors[8001], 1e-4))

    x, v = profile(4001)
    rows = {}
    for t in TEMPERATURES:
        u = HBAR_J_S * OMEGA / (BOLTZMANN_J_K * t)
        rows[str(t)] = {
            "u": u,
            "wigner": wigner_kappa(NU_IM, t),
            "bell_parabolic": (u / 2) / math.sin(u / 2) if u < 2 * math.pi else None,
            "exact_eckart": kappa_exact(t),
            "uniform_semiclassical": kappa_uniform(t),
            "pyrate_wkb": thermal_wkb_kappa(x, v, t, MU, 2000),
            "pyrate_wkb_primitive": thermal_wkb_kappa(x, v, t, MU, 2000, "primitive"),
        }
    data["kappa"] = rows
    worst = max(abs(r["pyrate_wkb"] / r["uniform_semiclassical"] - 1) for r in rows.values())
    checks.append(check("κ_PyRate vs independent uniform semiclassical reference (max rel)", worst, 1e-4))

    # Parabolic barrier: uniform semiclassical κ equals Bell's (u/2)/sin(u/2).
    w = 2 * math.pi * LIGHT_SPEED_M_S * 100 * 1000.0
    vp = 200e3 / AVOGADRO_MOL
    xm = math.sqrt(2 * vp / (MU * w * w))
    xp = np.linspace(-xm, xm, 2001)
    bell = {}
    for t in (600.0, 1000.0, 2000.0):
        u = HBAR_J_S * w / (BOLTZMANN_J_K * t)
        k = thermal_wkb_kappa(tuple(xp), tuple(vp - 0.5 * MU * w * w * xp**2), t, MU, 400)
        bell[str(t)] = {"pyrate": k, "bell": (u / 2) / math.sin(u / 2)}
    data["parabolic_bell"] = bell
    checks.append(check("parabolic barrier: κ_PyRate vs Bell (max rel)",
                        max(abs(r["pyrate"] / r["bell"] - 1) for r in bell.values()), 1e-4))  # piecewise-linear grid error
    hot = rows["5000.0"]
    checks.append(check("Wigner vs Bell at 5000 K (rel)", abs(hot["wigner"] / hot["bell_parabolic"] - 1), 1e-3))
    for t in ("300.0", "1000.0", "2000.0"):
        r = rows[t]
        checks.append(check(f"κ_PyRate/κ_exact at {float(t):g} K", r["pyrate_wkb"] / r["exact_eckart"], None))
        checks.append(check(f"κ_primitive/κ_exact at {float(t):g} K (legacy form)",
                            r["pyrate_wkb_primitive"] / r["exact_eckart"], None))
    return {
        "case": "b07_eckart_tunneling",
        "title": "Eckart barrier: WKB action and canonical tunneling coefficients",
        "references": ["Eckart, Phys. Rev. 35, 1303 (1930)", "Wigner, Z. Phys. Chem. B 19, 203 (1932)",
                       "Bell, Trans. Faraday Soc. 55, 1 (1959)",
                       "Johnston & Heicklen, J. Phys. Chem. 66, 532 (1962)",
                       "Garrett & Truhlar, J. Phys. Chem. 83, 2921 (1979)"],
        "checks": checks,
        "data": data,
    }
