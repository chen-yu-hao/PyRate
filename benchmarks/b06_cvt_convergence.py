"""B6a: CVT on a manufactured, analytically known free-energy profile.

V_MEP(s) = V0 sech²(s/L); one transverse mode ω⊥(s) = ω_R − Δω sech²((s − s0)/L')
with s0 ≠ 0 and moments of inertia linear in s, so the variational bottleneck
moves with temperature.  The reference ΔG(s, T) is written independently of
PyRate and maximised by golden-section search.  PyRate sees only N sampled
path points.
"""

from __future__ import annotations

import math

import numpy as np

from pyrate.constants import (
    BOLTZMANN_J_K,
    GAS_CONSTANT_J_MOL_K,
    WAVENUMBER_TO_J,
    WAVENUMBER_TO_KJ_MOL,
)
from pyrate.models import PathPoint, Reaction, ReactionPath, Species
from pyrate.rates import (
    activation_gibbs_kj_mol,
    conventional_tst_rate_si,
    cvt_rate_si,
    tst_rate_from_activation_gibbs_si,
)

from .common import check

V0, L = 50.0, 0.6
OMEGA_R, D_OMEGA, S0, L2 = 1400.0, 500.0, 0.25, 0.35
BASE = (450.0, 800.0, 1100.0, 2900.0)
I0, I1 = (10.0, 15.0, 20.0), (0.8, 1.2, 1.5)
REACTANT_FREQS = BASE + (1600.0, 1000.0)
REACTANT_I = (9.0, 14.0, 19.0)
TEMPERATURES = (200.0, 300.0, 500.0, 1000.0, 2000.0)


def v(s: float) -> float:
    return V0 / math.cosh(s / L) ** 2


def omega(s: float) -> float:
    return OMEGA_R - D_OMEGA / math.cosh((s - S0) / L2) ** 2


def moments(s: float) -> tuple[float, float, float]:
    return tuple(a + b * s for a, b in zip(I0, I1))  # type: ignore[return-value]


def path_species(s: float) -> Species:
    return Species(f"s={s:+.5f}", mass_amu_override=30.0, rotor_type="nonlinear",
                   moments_of_inertia_amu_a2=moments(s), electronic_energy_kj_mol=v(s),
                   frequencies_cm1=BASE + (1600.0, omega(s)))


REACTANT = Species("R", mass_amu_override=30.0, rotor_type="nonlinear",
                   moments_of_inertia_amu_a2=REACTANT_I, frequencies_cm1=REACTANT_FREQS)


def reference_dg(s: float, t: float) -> float:
    """Independent RRHO ΔG(s, T) in kJ/mol (translation cancels: equal masses)."""

    rt = GAS_CONSTANT_J_MOL_K * t / 1000.0
    kt = BOLTZMANN_J_K * t

    def lnqvib(freqs):
        return -sum(math.log(1 - math.exp(-f * WAVENUMBER_TO_J / kt)) for f in freqs)

    freqs = BASE + (1600.0, omega(s))
    zpe = 0.5 * WAVENUMBER_TO_KJ_MOL * (sum(freqs) - sum(REACTANT_FREQS))
    lnrot = 0.5 * (math.log(np.prod(moments(s))) - math.log(np.prod(REACTANT_I)))
    return v(s) + zpe - rt * (lnrot + lnqvib(freqs) - lnqvib(REACTANT_FREQS))


def reference_maximum(t: float) -> tuple[float, float]:
    grid = np.linspace(-1.0, 1.0, 4001)
    vals = [reference_dg(x, t) for x in grid]
    i = int(np.argmax(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    g = (math.sqrt(5) - 1) / 2
    for _ in range(200):
        c, d = b - g * (b - a), a + g * (b - a)
        if reference_dg(c, t) > reference_dg(d, t):
            b = d
        else:
            a = c
    s = 0.5 * (a + b)
    return s, reference_dg(s, t)


def reaction(n: int) -> Reaction:
    points = tuple(PathPoint(float(s), path_species(float(s))) for s in np.linspace(-1.0, 1.0, n))
    return Reaction(reactants=(REACTANT,), transition_state=path_species(0.0), path=ReactionPath(points))


def run() -> dict:
    checks, data = [], {"reference": {}, "convergence": {}}
    refs = {t: reference_maximum(t) for t in TEMPERATURES}
    data["reference"] = {str(t): {"s_star": s, "dG_star": g} for t, (s, g) in refs.items()}

    # Independent formula vs PyRate's ΔG at an arbitrary dividing surface.
    probe = max(abs(activation_gibbs_kj_mol(reaction(9), path_species(0.37), t) - reference_dg(0.37, t))
                for t in TEMPERATURES)
    checks.append(check("ΔG(s,T): PyRate vs independent formula (kJ/mol)", probe, 1e-9))

    ns = (9, 17, 33, 65, 129)
    for n in ns:
        rx = reaction(n)
        rows = {}
        for t in TEMPERATURES:
            c = cvt_rate_si(rx, t)
            s_ref, g_ref = refs[t]
            rows[str(t)] = {"s_cvt": c.coordinate, "dG_cvt": c.activation_gibbs_kj_mol,
                            "err_s": abs(c.coordinate - s_ref), "err_dG": abs(c.activation_gibbs_kj_mol - g_ref),
                            "fit_points": c.fit_points}
        data["convergence"][str(n)] = rows

    # Observed order of the ΔG error between successive halvings, worst over T.
    orders = []
    for t in TEMPERATURES:
        errs = [data["convergence"][str(n)][str(t)]["err_dG"] for n in ns]
        for e1, e2 in zip(errs[:-2], errs[1:-1]):  # skip finest pair (round-off floor)
            if e1 > 1e-11 and e2 > 1e-11:
                orders.append(math.log2(e1 / e2))
    data["observed_orders_dG"] = orders
    checks.append(check("median observed convergence order of ΔG_CVT", float(np.median(orders)), None))
    checks.append(check("ΔG_CVT error at N=129, worst T (kJ/mol)",
                        max(r["err_dG"] for r in data["convergence"]["129"].values()), 1e-6))
    checks.append(check("s_CVT error at N=129, worst T (Å)",
                        max(r["err_s"] for r in data["convergence"]["129"].values()), 1e-3))

    # Variational principle: k_CVT ≤ every sampled generalized-TST rate and ≤ k_TST.
    rx = reaction(33)
    worst = -math.inf
    for t in TEMPERATURES:
        k_cvt = cvt_rate_si(rx, t).rate_si
        k_min = min(tst_rate_from_activation_gibbs_si(activation_gibbs_kj_mol(rx, p.dividing_surface, t), t, 1, rx.standard_pressure_pa)
                    for p in rx.path.points)
        _, k_tst = conventional_tst_rate_si(rx, t)
        worst = max(worst, k_cvt / k_min - 1, k_cvt / k_tst - 1)
    checks.append(check("variational bound max(k_CVT/k_GT,min − 1, k_CVT/k_TST − 1)", worst, 1e-12))

    data["s_cvt_vs_T_N33"] = {str(t): cvt_rate_si(rx, t).coordinate for t in TEMPERATURES}
    return {
        "case": "b06_cvt_convergence",
        "title": "CVT bottleneck location and free energy on a manufactured profile",
        "references": ["Truhlar & Garrett, Annu. Rev. Phys. Chem. 35, 159 (1984)"],
        "checks": checks,
        "data": data,
    }
