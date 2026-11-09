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
        profile.transmission,
    )


def profile_transmission_probability(
    profile: TunnelingProfile, energy_kj_mol: float
) -> float:
    """Return the 1D semiclassical transmission probability at one energy.

    With ``profile.transmission == "uniform"`` this is ``1/(1 + e^{γ})`` below
    the barrier top and ``1 − P(2V − E)`` above it; ``"primitive"`` returns
    ``e^{−γ}`` below and 1 above the top.
    """

    coordinates_m = tuple(value * ANGSTROM_M for value in profile.coordinates_angstrom)
    reference = profile.energies_kj_mol[0]
    energies_j = tuple(
        (value - reference) * 1000.0 / AVOGADRO_MOL
        for value in profile.energies_kj_mol
    )
    energy_j = (energy_kj_mol - reference) * 1000.0 / AVOGADRO_MOL
    barrier = max(energies_j)
    mass = profile.reduced_mass_amu * AMU_KG
    if profile.transmission == "primitive":
        if energy_j >= barrier:
            return 1.0
        exponent = wkb_exponent(coordinates_m, energies_j, energy_j, mass)
        return math.exp(-exponent) if exponent < 745.0 else 0.0

    def below(e: float) -> float:
        exponent = wkb_exponent(coordinates_m, energies_j, e, mass)
        return 1.0 / (1.0 + math.exp(exponent)) if exponent < 709.0 else 0.0

    if energy_j <= barrier:
        return below(energy_j)
    if energy_j >= 2.0 * barrier:
        return 1.0
    return 1.0 - below(2.0 * barrier - energy_j)
