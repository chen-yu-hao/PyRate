"""Tunneling corrections used by canonical rate calculations."""

from __future__ import annotations

import math

from .constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL
from .models import Species, TunnelingProfile
from .native import thermal_wkb_kappa, wigner_kappa, wkb_exponent


def transition_state_wigner_kappa(transition_state: Species, temperature_k: float) -> float:
    """Return Wigner's parabolic-barrier correction for one imaginary mode."""

    imaginary = transition_state.imaginary_frequencies_cm1
    if len(imaginary) != 1:
        raise ValueError(
            "Wigner tunneling requires exactly one imaginary transition-state frequency"
        )
    return wigner_kappa(imaginary[0], temperature_k)


def profile_wkb_kappa(profile: TunnelingProfile, temperature_k: float) -> float:
    """Return a canonical 1D WKB correction for a supplied barrier profile.

    The first profile energy is the forward-reactant reference.  This kernel is
    the zero-curvature, one-dimensional building block; it does not yet include
    Polyrate's multidimensional corner-cutting SCT or LCT algorithms.
    """

    coordinates_m = tuple(value * ANGSTROM_M for value in profile.coordinates_angstrom)
    energies_j = tuple(
        value * 1000.0 / AVOGADRO_MOL for value in profile.energies_kj_mol
    )
    return thermal_wkb_kappa(
        coordinates_m,
        energies_j,
        temperature_k,
        profile.reduced_mass_amu * AMU_KG,
        profile.energy_grid_points,
    )


def profile_transmission_probability(
    profile: TunnelingProfile, energy_kj_mol: float
) -> float:
    """Return a bounded, reflection-aware semiclassical transmission probability."""

    coordinates_m = tuple(value * ANGSTROM_M for value in profile.coordinates_angstrom)
    reference = profile.energies_kj_mol[0]
    energies_j = tuple(
        (value - reference) * 1000.0 / AVOGADRO_MOL
        for value in profile.energies_kj_mol
    )
    energy_j = (energy_kj_mol - reference) * 1000.0 / AVOGADRO_MOL
    barrier = max(energies_j)
    threshold = max(energies_j[0], energies_j[-1])
    if energy_j <= threshold:
        return 0.0
    if energy_j >= barrier:
        peak = max(range(len(energies_j)), key=energies_j.__getitem__)
        if peak == 0 or peak == len(energies_j) - 1:
            return 1.0
        xl = coordinates_m[peak] - coordinates_m[peak - 1]
        xr = coordinates_m[peak + 1] - coordinates_m[peak]
        curvature = 2.0 * ((energies_j[peak + 1] - energies_j[peak]) / xr - (energies_j[peak] - energies_j[peak - 1]) / xl) / (xl + xr)
        omega = math.sqrt(-curvature / (profile.reduced_mass_amu * AMU_KG)) if curvature < 0 else 0.0
        if omega == 0.0:
            return 1.0
        from .constants import HBAR_J_S
        action = -2.0 * math.pi * (energy_j - barrier) / (HBAR_J_S * omega)
        return 1.0 / (1.0 + math.exp(action))
    exponent = wkb_exponent(
        coordinates_m, energies_j, energy_j, profile.reduced_mass_amu * AMU_KG
    )
    transmission = math.exp(-exponent) if exponent < 745.0 else 0.0
    return transmission / (1.0 + transmission)
