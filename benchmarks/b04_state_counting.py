"""B4: Beyer–Swinehart state counts — exact enumeration, Laplace consistency with
the harmonic partition function, and the Whitten–Rabinovitch approximation."""

from __future__ import annotations

import itertools
import math

import numpy as np

from pyrate.constants import BOLTZMANN_J_K, WAVENUMBER_TO_J
from pyrate.models import Species
from pyrate.partition import log_vibrational_partition
from pyrate.states import vibrational_density_of_states

from .common import check

# Generic, non-commensurate 12-mode set (cm⁻¹) of a small polyatomic.
MODES_12 = (263.3, 945.7, 1129.1, 1129.1, 1414.9, 1429.4, 1429.4, 2166.2, 2965.8, 3014.1, 3014.1, 3061.0)


def brute_force(freqs: tuple[float, ...], grain: float, emax: float) -> np.ndarray:
    counts = np.zeros(int(emax // grain) + 1)
    for n in itertools.product(*(range(int(emax // f) + 1) for f in freqs)):
        e = sum(a * b for a, b in zip(n, freqs))
        if e <= emax:
            counts[int(round(e / grain))] += 1
    return counts


def whitten_rabinovitch_sum(freqs: tuple[float, ...], energy: float) -> float:
    s = len(freqs)
    ez = 0.5 * sum(freqs)
    ep = energy / ez
    if ep < 1.0:
        w = 1.0 / (5.00 * ep + 2.73 * math.sqrt(ep) + 3.51)
    else:
        w = math.exp(-2.4191 * ep**0.25)
    nu = np.asarray(freqs)
    beta = (s - 1) / s * float(np.mean(nu**2) / np.mean(nu) ** 2)
    a = 1.0 - beta * w
    return (energy + a * ez) ** s / (math.factorial(s) * float(np.prod(nu)))


def run() -> dict:
    checks, data = [], {}
    small = (500.0, 800.0, 1200.0)
    _, bs = vibrational_density_of_states(small, 1.0, 6000.0)
    checks.append(check("exact enumeration: max |BS − brute force|", float(np.max(np.abs(bs - brute_force(small, 1.0, 6000.0)))), 0.0, mode="eq"))

    sp = Species("model", mass_amu_override=40.0, rotor_type="atomic", frequencies_cm1=MODES_12)
    laplace = {}
    for t in (300.0, 1000.0):
        q_ho = math.exp(log_vibrational_partition(sp, t))
        kt_cm = BOLTZMANN_J_K * t / WAVENUMBER_TO_J
        laplace[str(t)] = {}
        for grain in (50.0, 20.0, 10.0, 5.0, 1.0):
            e, n = vibrational_density_of_states(MODES_12, grain, 40.0 * kt_cm + 40000.0)
            laplace[str(t)][str(grain)] = float(np.sum(n * np.exp(-e / kt_cm)) / q_ho - 1.0)
        checks.append(check(f"Laplace rel error at {t:g} K, grain 1 cm⁻¹ (frequency rounding)",
                            abs(laplace[str(t)]["1.0"]), None))
        # Exact identity once frequencies are rounded to the grain the algorithm uses.
        grain = 10.0
        rounded = tuple(grain * max(1, round(f / grain)) for f in MODES_12)
        q_rounded = math.exp(log_vibrational_partition(
            Species("r", mass_amu_override=40.0, rotor_type="atomic", frequencies_cm1=rounded), t))
        e, n = vibrational_density_of_states(MODES_12, grain, 60.0 * kt_cm + 60000.0)
        checks.append(check(f"Laplace identity with grain-rounded modes at {t:g} K (rel)",
                            abs(float(np.sum(n * np.exp(-e / kt_cm))) / q_rounded - 1.0), 1e-10))
    data["laplace_rel_error_vs_grain"] = laplace

    e, n = vibrational_density_of_states(MODES_12, 1.0, 40000.0)
    cumulative = np.cumsum(n)
    wr = {}
    for energy in (5000.0, 10000.0, 20000.0, 40000.0):
        exact = float(cumulative[int(energy)])
        wr[str(energy)] = {"N_BS": exact, "N_WR": whitten_rabinovitch_sum(MODES_12, energy),
                           "ratio": whitten_rabinovitch_sum(MODES_12, energy) / exact}
        checks.append(check(f"N_WR/N_BS at E = {energy:g} cm⁻¹", wr[str(energy)]["ratio"], None))
    data["whitten_rabinovitch"] = wr
    return {
        "case": "b04_state_counting",
        "title": "Beyer–Swinehart state counting",
        "references": ["Beyer & Swinehart, Commun. ACM 16, 379 (1973)",
                       "Stein & Rabinovitch, J. Chem. Phys. 58, 2438 (1973)",
                       "Whitten & Rabinovitch, J. Chem. Phys. 38, 2466 (1963)"],
        "checks": checks,
        "data": data,
    }
