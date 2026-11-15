"""Tunneling corrections used by canonical and variational rate calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .constants import (
    AMU_KG,
    ANGSTROM_M,
    AVOGADRO_MOL,
    BOHR_ANGSTROM,
    GAS_CONSTANT_J_MOL_K,
    HBAR_J_S,
    LIGHT_SPEED_M_S,
)
from .lct import (
    LctResult,
    _prepare_mep,
    can_compute_lct,
    comt_kappa_from_result,
    lct_kappa,
    lct_kappa_from_result,
    muomt_kappa_from_result,
    reaction_lct_result,
    sct_kappa_from_result,
    zct_kappa_from_result,
)
from .models import Reaction, ReactionPath, Species, TunnelingProfile
from .native import (
    refine_maximum,
    thermal_boltz_probabilities_kappa,
    thermal_eckart_kappa,
    thermal_wkb_kappa,
    thermal_wkb_profile_kappa,
    wigner_kappa,
    wkb_exponent,
    wkb_exponent_profile,
)
from .partition import zero_point_energy_kj_mol


def transition_state_wigner_kappa(transition_state: Species, temperature_k: float) -> float:
    """Return Wigner's parabolic-barrier correction for one imaginary mode."""

    imaginary = transition_state.imaginary_frequencies_cm1
    if len(imaginary) != 1:
        raise ValueError(
            "Wigner tunneling requires exactly one imaginary transition-state frequency"
        )
    return wigner_kappa(imaginary[0], temperature_k)


def profile_wkb_kappa(profile: TunnelingProfile, temperature_k: float) -> float:
    """Return a canonical 1D WKB/ZCT or variable-mass SCT correction for a barrier profile."""

    coordinates_m = tuple(value * ANGSTROM_M for value in profile.coordinates_angstrom)
    energies_j = tuple(
        value * 1000.0 / AVOGADRO_MOL for value in profile.energies_kj_mol
    )
    if profile.effective_masses_amu is not None:
        masses_kg = tuple(m * AMU_KG for m in profile.effective_masses_amu)
        return thermal_wkb_profile_kappa(
            coordinates_m,
            energies_j,
            temperature_k,
            masses_kg,
            profile.energy_grid_points,
        )
    return thermal_wkb_kappa(
        coordinates_m,
        energies_j,
        temperature_k,
        profile.reduced_mass_amu * AMU_KG,
        profile.energy_grid_points,
    )


def _kemble_probability(gamma: float) -> float:
    if gamma > 709.0:
        return 0.0
    if gamma < -709.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(gamma))


def profile_transmission_probability(
    profile: TunnelingProfile, energy_kj_mol: float
) -> float:
    """Return the Kemble uniform semiclassical transmission probability P(E)."""

    coordinates_m = tuple(value * ANGSTROM_M for value in profile.coordinates_angstrom)
    energies_j = tuple(
        value * 1000.0 / AVOGADRO_MOL for value in profile.energies_kj_mol
    )
    threshold_j = max(energies_j[0], energies_j[-1])
    barrier_j = max(energies_j)
    energy_j = energy_kj_mol * 1000.0 / AVOGADRO_MOL
    if energy_j < threshold_j:
        return 0.0
    if energy_j <= barrier_j:
        if profile.effective_masses_amu is not None:
            masses_kg = tuple(m * AMU_KG for m in profile.effective_masses_amu)
            gamma = wkb_exponent_profile(coordinates_m, energies_j, energy_j, masses_kg)
        else:
            gamma = wkb_exponent(
                coordinates_m, energies_j, energy_j, profile.reduced_mass_amu * AMU_KG
            )
        return _kemble_probability(gamma)
    # Nonclassical parabolic reflection above the barrier (E in (V_max, 2*V_max - E_0])
    reflected_energy_j = 2.0 * barrier_j - energy_j
    if reflected_energy_j <= threshold_j:
        return 1.0
    if profile.effective_masses_amu is not None:
        masses_kg = tuple(m * AMU_KG for m in profile.effective_masses_amu)
        gamma = wkb_exponent_profile(
            coordinates_m, energies_j, reflected_energy_j, masses_kg
        )
    else:
        gamma = wkb_exponent(
            coordinates_m,
            energies_j,
            reflected_energy_j,
            profile.reduced_mass_amu * AMU_KG,
        )
    return 1.0 - _kemble_probability(gamma)


def _path_coordinates_angstrom(path: ReactionPath) -> tuple[float, ...]:
    scale = BOHR_ANGSTROM if path.coordinate_unit == "bohr" else 1.0
    return tuple(point.coordinate * scale for point in path.points)


def path_adiabatic_ground_potentials_kj_mol(path: ReactionPath) -> tuple[float, ...]:
    """Return V_a^G(s) = V_MEP(s) + ZPE(s) along the reaction path in kJ/mol."""

    return tuple(
        point.dividing_surface.electronic_energy_kj_mol
        + zero_point_energy_kj_mol(point.dividing_surface)
        for point in path.points
    )


def _compute_cdscsag_effective_masses_amu(path: ReactionPath) -> tuple[float, ...]:
    """Compute Polyrate CD-SCSAG (SCT) effective masses mu_eff(s) along a reaction path."""

    mu_0 = path.reduced_mass_amu
    explicit = [point.effective_mass_amu for point in path.points]
    if all(m is not None for m in explicit):
        return tuple(float(m) for m in explicit if m is not None)

    coords_bohr = [
        point.coordinate / BOHR_ANGSTROM
        if path.coordinate_unit == "angstrom"
        else point.coordinate
        for point in path.points
    ]
    n_pts = len(path.points)
    turning_points_bohr: list[float] = []
    bkap_values: list[float] = []

    mu_kg = mu_0 * AMU_KG
    bohr_m = BOHR_ANGSTROM * ANGSTROM_M

    fallback_bcur: tuple[tuple[float, ...], ...] | None = None
    atom_counts = [len(pt.dividing_surface.atoms) for pt in path.points]
    if (
        any(not pt.curvatures_bohr_inv for pt in path.points)
        and all(count >= 2 for count in atom_counts)
        and len(set(atom_counts)) == 1
        and any(len(pt.dividing_surface.positive_frequencies_cm1) > 0 for pt in path.points)
    ):
        synth_rxn = Reaction(
            reactants=(path.points[0].dividing_surface,),
            transition_state=path.points[len(path.points) // 2].dividing_surface,
            products=(path.points[-1].dividing_surface,),
            path=path,
        )
        mep = _prepare_mep(synth_rxn)
        fallback_bcur = tuple(
            tuple(float(val) for val in mep.bcur[idx]) for idx in range(len(path.points))
        )

    for pt_idx, point in enumerate(path.points):
        curvatures = (
            point.curvatures_bohr_inv
            if point.curvatures_bohr_inv
            else (fallback_bcur[pt_idx] if fallback_bcur is not None else ())
        )
        freqs = point.dividing_surface.positive_frequencies_cm1
        if not curvatures or not freqs:
            turning_points_bohr.append(0.0)
            bkap_values.append(0.0)
            continue
        m_count = min(len(curvatures), len(freqs))
        b_sq_sum = 0.0
        denom_sum = 0.0
        for idx in range(m_count):
            b_m = curvatures[idx]
            omega_s_inv = 2.0 * math.pi * LIGHT_SPEED_M_S * 100.0 * freqs[idx]
            if omega_s_inv <= 0.0:
                continue
            # Dimensionless turning-point inverse length in bohr^-1:
            # t_m^-2 = (mu * omega_m / hbar) in bohr^-2
            alpha_bohr_inv2 = (mu_kg * omega_s_inv / HBAR_J_S) * (bohr_m**2)
            b_sq_sum += b_m * b_m
            denom_sum += (b_m * b_m) * (alpha_bohr_inv2**2)
        bkap = math.sqrt(b_sq_sum)
        tp = math.sqrt(bkap) / (denom_sum**0.25) if (bkap > 0.0 and denom_sum > 0.0) else 0.0
        turning_points_bohr.append(tp)
        bkap_values.append(bkap)

    effective_masses: list[float] = []
    for i, point in enumerate(path.points):
        if point.effective_mass_amu is not None:
            effective_masses.append(float(point.effective_mass_amu))
            continue
        bkap = bkap_values[i]
        tp = abs(turning_points_bohr[i])
        if bkap <= 0.0 or tp <= 0.0:
            effective_masses.append(mu_0)
            continue
        if 0 < i < n_pts - 1:
            dt_ds = (abs(turning_points_bohr[i + 1]) - abs(turning_points_bohr[i - 1])) / (
                coords_bohr[i + 1] - coords_bohr[i - 1]
            )
        elif i == 0:
            dt_ds = (abs(turning_points_bohr[1]) - abs(turning_points_bohr[0])) / (
                coords_bohr[1] - coords_bohr[0]
            )
        else:
            dt_ds = (abs(turning_points_bohr[-1]) - abs(turning_points_bohr[-2])) / (
                coords_bohr[-1] - coords_bohr[-2]
            )
        arg = tp * bkap
        xarg = -2.0 * arg - arg * arg + dt_ds * dt_ds
        factor = math.exp(xarg) if xarg < 0.0 else 1.0
        effective_masses.append(max(1.0e-6 * mu_0, mu_0 * factor))

    return tuple(effective_masses)


def path_tunneling_profile(
    path: ReactionPath,
    *,
    method: str = "zct",
    energy_grid_points: int = 400,
) -> TunnelingProfile:
    """Build a ZCT (MEPSAG) or SCT (CD-SCSAG) TunnelingProfile from a ReactionPath."""

    coordinates_angstrom = _path_coordinates_angstrom(path)
    energies_kj_mol = path_adiabatic_ground_potentials_kj_mol(path)
    normalized = method.lower()
    effective_masses = (
        _compute_cdscsag_effective_masses_amu(path)
        if normalized in {"sct", "cd-scsag"}
        else None
    )
    return TunnelingProfile(
        coordinates_angstrom=coordinates_angstrom,
        energies_kj_mol=energies_kj_mol,
        reduced_mass_amu=path.reduced_mass_amu,
        energy_grid_points=energy_grid_points,
        effective_masses_amu=effective_masses,
    )


def cag_kappa(
    reaction: Reaction,
    temperature_k: float,
    *,
    variational_coordinate: float | None = None,
) -> float:
    """Return Polyrate's Canonical Adiabatic Ground-State (CAG) factor kappa^CAG(T).

    In Polyrate 2023 (polyhl.F:1723 and polyrr.F:rate),
    ``kappa^CAG = exp((V_a^G(s_*) - V_a^AG) / (R * T))``, where ``V_a^AG`` is the
    maximum of the vibrationally adiabatic ground-state potential along ``s`` and
    ``s_*`` is ``0`` for conventional TST or ``s_*^CVT(T)`` for CVT.
    """

    if reaction.path is None:
        return 1.0
    coords = [point.coordinate for point in reaction.path.points]
    vag = list(path_adiabatic_ground_potentials_kj_mol(reaction.path))
    vag_max = refine_maximum(coords, vag).value
    target_s = 0.0 if variational_coordinate is None else float(variational_coordinate)
    # Linearly interpolate V_a^G(target_s) on the reaction path grid
    if target_s <= coords[0]:
        vag_star = vag[0]
    elif target_s >= coords[-1]:
        vag_star = vag[-1]
    else:
        vag_star = vag[0]
        for idx in range(1, len(coords)):
            if coords[idx - 1] <= target_s <= coords[idx]:
                span = coords[idx] - coords[idx - 1]
                weight = (target_s - coords[idx - 1]) / span if span > 0.0 else 0.0
                vag_star = vag[idx - 1] + weight * (vag[idx] - vag[idx - 1])
                break
    diff_kj_mol = vag_star - vag_max
    exponent = (diff_kj_mol * 1000.0) / (GAS_CONSTANT_J_MOL_K * temperature_k)
    return math.exp(max(-700.0, min(700.0, exponent)))


def reaction_eckart_kappa(
    reaction: Reaction,
    temperature_k: float,
    *,
    energy_grid_points: int = 400,
) -> float:
    """Return the thermal Eckart transmission coefficient for a reaction."""

    imaginary = reaction.transition_state.imaginary_frequencies_cm1
    if len(imaginary) != 1:
        raise ValueError(
            "Eckart tunneling requires exactly one imaginary transition-state frequency"
        )
    v_reactants = sum(
        sp.electronic_energy_kj_mol + zero_point_energy_kj_mol(sp)
        for sp in reaction.reactants
    )
    v_ts = (
        reaction.transition_state.electronic_energy_kj_mol
        + zero_point_energy_kj_mol(reaction.transition_state)
    )
    v1_kj_mol = v_ts - v_reactants
    if v1_kj_mol <= 0.0:
        return 1.0
    if reaction.products:
        v_products = sum(
            sp.electronic_energy_kj_mol + zero_point_energy_kj_mol(sp)
            for sp in reaction.products
        )
        v2_kj_mol = v_ts - v_products
    elif reaction.path is not None:
        vag = path_adiabatic_ground_potentials_kj_mol(reaction.path)
        v2_kj_mol = max(vag) - vag[-1]
    else:
        v2_kj_mol = v1_kj_mol
    if v2_kj_mol <= 0.0:
        return 1.0
    v1_j = v1_kj_mol * 1000.0 / AVOGADRO_MOL
    v2_j = v2_kj_mol * 1000.0 / AVOGADRO_MOL
    return thermal_eckart_kappa(
        v1_j, v2_j, abs(imaginary[0]), temperature_k, energy_grid_points
    )


def profile_muomt_kappa(
    zct_profile: TunnelingProfile,
    sct_profile: TunnelingProfile,
    temperature_k: float,
    *,
    lct_probabilities: Sequence[float] | None = None,
) -> float:
    """Return microcanonical OMT (muOMT) transmission factor on 1D barrier profiles."""

    energies_kj = zct_profile.energies_kj_mol
    threshold_kj = max(energies_kj[0], energies_kj[-1])
    vmax_kj = max(energies_kj)
    vm_kj = vmax_kj - threshold_kj
    if vm_kj <= 0.0:
        return 1.0
    count = zct_profile.energy_grid_points + zct_profile.energy_grid_points % 2
    step_kj = vm_kj / count
    scale_j = 1000.0 / AVOGADRO_MOL

    grid_e_j: list[float] = []
    grid_w_j: list[float] = []
    grid_p: list[float] = []
    if lct_probabilities is not None and len(lct_probabilities) != count + 1:
        raise ValueError(
            f"lct_probabilities length ({len(lct_probabilities)}) "
            f"must match Simpson grid ({count + 1})"
        )
    for idx in range(count + 1):
        e_kj = threshold_kj + idx * step_kj
        simpson_mult = 1.0 if (idx == 0 or idx == count) else (2.0 if idx % 2 == 0 else 4.0)
        w_kj = (step_kj / 3.0) * simpson_mult
        p_zct = profile_transmission_probability(zct_profile, e_kj)
        p_sct = profile_transmission_probability(sct_profile, e_kj)
        p_lct = float(lct_probabilities[idx]) if lct_probabilities is not None else 0.0
        grid_e_j.append(e_kj * scale_j)
        grid_w_j.append(w_kj * scale_j)
        grid_p.append(max(p_zct, p_sct, p_lct))

    return thermal_boltz_probabilities_kappa(
        grid_e_j, grid_w_j, grid_p, vmax_kj * scale_j, temperature_k
    )


__all__ = [
    "LctResult",
    "cag_kappa",
    "can_compute_lct",
    "comt_kappa_from_result",
    "lct_kappa",
    "lct_kappa_from_result",
    "muomt_kappa_from_result",
    "path_adiabatic_ground_potentials_kj_mol",
    "path_tunneling_profile",
    "profile_muomt_kappa",
    "profile_transmission_probability",
    "profile_wkb_kappa",
    "reaction_eckart_kappa",
    "reaction_lct_result",
    "sct_kappa_from_result",
    "transition_state_wigner_kappa",
    "zct_kappa_from_result",
]


