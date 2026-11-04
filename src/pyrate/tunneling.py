"""Tunneling corrections used by canonical rate calculations."""

from __future__ import annotations

from .constants import AMU_KG, ANGSTROM_M, AVOGADRO_MOL
from .models import Species, TunnelingProfile
from .native import semiclassical_transmission, thermal_wkb_kappa, wigner_kappa


def transition_state_wigner_kappa(transition_state: Species, temperature_k: float) -> float:
    """Return Wigner's parabolic-barrier correction for one imaginary mode."""

    imaginary = transition_state.imaginary_frequencies_cm1
    if len(imaginary) != 1:
        raise ValueError(
            "Wigner tunneling requires exactly one imaginary transition-state frequency"
        )
    return wigner_kappa(imaginary[0], temperature_k)


def profile_wkb_kappa(profile: TunnelingProfile, temperature_k: float) -> float:
    """Return the canonical 1D uniform semiclassical correction for a profile.

    ``κ(T) = β exp(βV^AG) ∫_{E0}^{∞} P(E) exp(-βE) dE`` with ``V^AG = max(V)``
    and threshold ``E0 = max(V[0], V[-1])``; ``P(E)`` is
    :func:`profile_transmission_probability`.  This kernel is the
    zero-curvature, one-dimensional building block: Polyrate's full ZCT needs
    ``V_a^G(s)`` along the MEP, and the multidimensional corner-cutting SCT and
    LCT algorithms are not included.
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
    """Return the uniform semiclassical transmission probability ``P(E)``.

    ``energy_kj_mol`` is on the same absolute scale as ``profile.energies_kj_mol``.
    ``P = 0`` below ``E0 = max(V[0], V[-1])``, ``1/(1+exp(γ(E)))`` up to the
    barrier top ``V^AG``, ``1 - P(2V^AG - E)`` above it (nonclassical
    reflection) and ``1`` from ``2V^AG - E0`` on.
    """

    coordinates_m = tuple(value * ANGSTROM_M for value in profile.coordinates_angstrom)
    reference = profile.energies_kj_mol[0]
    energies_j = tuple(
        (value - reference) * 1000.0 / AVOGADRO_MOL
        for value in profile.energies_kj_mol
    )
    energy_j = (energy_kj_mol - reference) * 1000.0 / AVOGADRO_MOL
    return semiclassical_transmission(
        coordinates_m, energies_j, energy_j, profile.reduced_mass_amu * AMU_KG
    )
