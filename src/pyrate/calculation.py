"""High-level orchestration for a complete temperature-grid calculation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .models import CalculationOptions, Reaction
from .advanced_vtst import (
    DualLevelCorrection,
    VRCVTSTModel,
    iv_tst_rate_si,
    vrc_vtst_rate_si,
)
from .environment import (
    CollisionModel,
    SSQRRKModel,
    SolvationCorrection,
    SurfaceModel,
    pressure_dependent_rate_si,
    reaction_solvation_delta_g_kj_mol,
)
from .path_frequencies import coordinate_frequencies
from .rates import (
    activation_gibbs_kj_mol,
    conventional_tst_rate_si,
    convert_rate_from_si,
    cus_rate_si,
    cvt_rate_si,
    muvt_and_us_rates_si,
    normalize_rate_unit,
    reaction_equilibrium_constants,
    tst_rate_from_activation_gibbs_si,
)
from .tunneling import (
    cag_kappa,
    path_tunneling_profile,
    profile_muomt_kappa,
    profile_wkb_kappa,
    reaction_eckart_kappa,
    transition_state_wigner_kappa,
)


@dataclass(frozen=True, slots=True)
class RateRow:
    temperature_k: float
    activation_gibbs_tst_kj_mol: float
    k_tst: float
    activation_gibbs_cvt_kj_mol: float | None
    cvt_coordinate: float | None
    cvt_sample_coordinate: float | None
    cvt_fit_points: int | None
    k_cvt: float | None
    tunneling_correction: float
    k_final: float
    rate_unit: str
    k_cus: float | None = None
    k_muvt: float | None = None
    k_us: float | None = None
    cag_tst: float | None = None
    cag_cvt: float | None = None
    kappa_wigner: float | None = None
    kappa_zct: float | None = None
    kappa_sct: float | None = None
    kappa_eckart: float | None = None
    kappa_comt: float | None = None
    kappa_muomt: float | None = None
    equilibrium_kc: float | None = None
    k_reverse_final: float | None = None
    k_ivtst: float | None = None
    k_vrc_vtst: float | None = None
    coordinate_frequency_cm1: float | None = None
    dual_level_correction_kj_mol: float | None = None
    solvent_correction_kj_mol: float | None = None
    surface_rate_correction: float | None = None
    pressure_pa: float | None = None
    k_pressure: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalculationResult:
    name: str
    method: str
    tunneling: str
    molecularity: int
    standard_pressure_pa: float
    rows: tuple[RateRow, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "tunneling": self.tunneling,
            "molecularity": self.molecularity,
            "standard_pressure_pa": self.standard_pressure_pa,
            "rows": [row.as_dict() for row in self.rows],
        }


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _dual_level_profile(
    reaction: Reaction, options: CalculationOptions
) -> tuple[DualLevelCorrection | None, tuple[float, ...] | None]:
    """Build a dual-level correction from options and the reaction path.

    The TOML form may provide only the high-level anchor energies; low-level
    energies then come from the path's dividing-surface electronic energies.
    A fully constructed :class:`DualLevelCorrection` can be passed directly.
    """

    value = options.dual_level
    if value is None:
        return None, None
    if isinstance(value, DualLevelCorrection):
        return value, value.corrected_energies_kj_mol
    if reaction.path is None:
        raise ValueError("dual-level correction requires reaction.path")
    if not isinstance(value, Mapping):
        raise TypeError("dual_level must be a DualLevelCorrection or mapping")
    coordinates = tuple(
        float(point.coordinate) for point in reaction.path.points
    )
    low = tuple(
        float(point.dividing_surface.electronic_energy_kj_mol)
        for point in reaction.path.points
    )
    configured_coordinates = value.get("coordinates")
    if configured_coordinates is not None:
        coordinates = tuple(float(item) for item in configured_coordinates)
    configured_low = value.get("low_level_energies_kj_mol")
    if configured_low is not None:
        low = tuple(float(item) for item in configured_low)
    high_raw = value.get("high_level_energies_kj_mol", value.get("high_level_energies"))
    if high_raw is None:
        raise ValueError("dual_level requires high_level_energies_kj_mol")
    correction = DualLevelCorrection(
        coordinates=coordinates,
        low_level_energies_kj_mol=low,
        high_level_energies_kj_mol=tuple(float(item) for item in high_raw),
    )
    return correction, correction.corrected_energies_kj_mol


def _solvation_delta(options: CalculationOptions) -> float:
    value = options.solvent
    if value is None:
        return 0.0
    if isinstance(value, SolvationCorrection):
        return value.free_energy_kj_mol()
    if not isinstance(value, Mapping):
        raise TypeError("solvent must be a SolvationCorrection or mapping")

    def make(item: Any) -> SolvationCorrection:
        if isinstance(item, SolvationCorrection):
            return item
        if isinstance(item, Mapping):
            return SolvationCorrection(**dict(item))
        raise TypeError("solvent species corrections must be mappings")

    ts = make(value.get("transition_state", value.get("ts", value)))
    reactants = value.get("reactants", ())
    if isinstance(reactants, Mapping):
        reactants = (reactants,)
    return reaction_solvation_delta_g_kj_mol(ts, (make(item) for item in reactants))


def _surface_factor(
    options: CalculationOptions, temperature_k: float
) -> float | None:
    value = options.surface
    if value is None:
        return None
    if isinstance(value, SurfaceModel):
        return value.rate_correction(temperature_k)
    if not isinstance(value, Mapping):
        raise TypeError("surface must be a SurfaceModel or mapping")
    data = dict(value)
    concentration = data.pop("concentration_m", None)
    pressure = data.pop("pressure_pa", None)
    coverage = data.pop("coverage", None)
    return SurfaceModel(**data).rate_correction(
        temperature_k,
        concentration_m=float(concentration) if concentration is not None else None,
        pressure_pa=float(pressure) if pressure is not None else None,
        coverage=float(coverage) if coverage is not None else None,
    )


def _path_coordinate_frequency(
    reaction: Reaction,
) -> float | None:
    if reaction.path is None:
        return None
    points = reaction.path.points
    provided = [point.coordinate_frequency_cm1 for point in points]
    if all(value is not None for value in provided):
        index = min(
            range(len(points)),
            key=lambda idx: abs(points[idx].dividing_surface.electronic_energy_kj_mol),
        )
        return float(provided[index])  # type: ignore[arg-type]
    if any(point.coordinate_frequency_cm1 is not None for point in points):
        return None
    try:
        profile = coordinate_frequencies(
            [point.coordinate for point in points],
            [point.dividing_surface.electronic_energy_kj_mol for point in points],
            [
                point.effective_mass_amu or reaction.path.reduced_mass_amu
                for point in points
            ],
            coordinate_unit=reaction.path.coordinate_unit,
        )
    except (ValueError, TypeError):
        return None
    return float(profile.frequencies_cm1[profile.barrier_index])


def _pressure_rate(
    options: CalculationOptions, temperature_k: float
) -> tuple[float | None, float | None]:
    value = options.pressure
    if value is None:
        return None, None
    if isinstance(value, SSQRRKModel):
        # A standalone SS-QRRK model carries the collision frequency but not a
        # pressure label; treat it as the reference 1 bar model.
        return pressure_dependent_rate_si(value, 100_000.0), 100_000.0
    if not isinstance(value, Mapping):
        raise TypeError("pressure must be an SSQRRKModel or mapping")
    data = dict(value)
    pressure = float(data.pop("pressure_pa", data.pop("pressure", 100_000.0)))
    pressures = data.pop("pressures_pa", None)
    if pressures is not None:
        pressure = float(tuple(pressures)[0])
    grid = data.get("energy_grid_kj_mol")
    dos = data.get("density_of_states")
    micro = data.get("microcanonical_rates_s_inv")
    if grid is None or dos is None or micro is None:
        raise ValueError(
            "pressure mapping requires energy_grid_kj_mol, density_of_states, "
            "and microcanonical_rates_s_inv"
        )
    collision = data.pop("collision_frequency_s_inv", None)
    energy_transfer = float(data.pop("energy_transfer_kj_mol", 10.0))
    if collision is None:
        collision_model = CollisionModel(
            bath_gas_mass_amu=float(data.pop("bath_gas_mass_amu", 28.0134)),
            collision_diameter_angstrom=float(data.pop("collision_diameter_angstrom", 3.5)),
            energy_transfer_kj_mol=energy_transfer,
            efficiency=float(data.pop("efficiency", 1.0)),
        )
        # The falloff closure scales a reference collision frequency by
        # ``pressure / 1 bar``.  Evaluate the hard-sphere model at that
        # reference pressure so low-pressure rates remain linear in p.
        collision = collision_model.frequency_s_inv(temperature_k, 100_000.0)
    model = SSQRRKModel(
        energy_grid_kj_mol=tuple(float(item) for item in grid),
        density_of_states=tuple(float(item) for item in dos),
        microcanonical_rates_s_inv=tuple(float(item) for item in micro),
        collision_frequency_s_inv=float(collision),
        temperature_k=float(data.pop("temperature_k", temperature_k)),
        energy_transfer_kj_mol=energy_transfer,
    )
    return pressure_dependent_rate_si(model, pressure), pressure


def calculate_rates(
    reaction: Reaction,
    options: CalculationOptions,
    *,
    name: str = "PyRate calculation",
) -> CalculationResult:
    """Calculate gas-phase and optional advanced VTST/environment rates."""

    if options.method in {"cvt", "cus", "muvt", "us", "ivtst", "all"} and reaction.path is None:
        raise ValueError(f"method = {options.method!r} requires a reaction path")
    if options.method in {"vrc", "vrc-vtst"} and options.vrc_vtst is None:
        raise ValueError("method = 'vrc-vtst' requires options.vrc_vtst")
    if (
        options.tunneling in {"zct", "mepsag", "sct", "cd-scsag", "omt", "comt", "muomt"}
        and options.tunneling_profile is None
        and reaction.path is None
    ):
        raise ValueError(
            f"tunneling = {options.tunneling!r} requires either a tunneling_profile or a reaction path"
        )

    rate_unit = normalize_rate_unit(options.output_rate_unit, reaction.molecularity)
    rows: list[RateRow] = []

    dual_profile, _ = _dual_level_profile(reaction, options)
    solvent_delta = _solvation_delta(options)
    coordinate_frequency = _path_coordinate_frequency(reaction)

    zct_profile = (
        options.tunneling_profile
        if options.tunneling_profile is not None
        else (
            path_tunneling_profile(reaction.path, method="zct")
            if reaction.path is not None
            else None
        )
    )
    sct_profile = (
        path_tunneling_profile(reaction.path, method="sct")
        if reaction.path is not None
        else options.tunneling_profile
    )

    for temperature in options.temperatures_k:
        delta_g_tst, k_tst_si = conventional_tst_rate_si(reaction, temperature)
        if solvent_delta:
            delta_g_tst += solvent_delta
            k_tst_si = tst_rate_from_activation_gibbs_si(
                delta_g_tst,
                temperature,
                reaction.molecularity,
                reaction.standard_pressure_pa,
            )
        k_tst, _ = convert_rate_from_si(
            k_tst_si, reaction.molecularity, rate_unit
        )

        delta_g_cvt: float | None = None
        cvt_coordinate: float | None = None
        sample_coordinate: float | None = None
        cvt_fit_points: int | None = None
        k_cvt_si: float | None = None
        k_cvt: float | None = None
        k_cus_si: float | None = None
        k_cus: float | None = None
        k_muvt_si: float | None = None
        k_muvt: float | None = None
        k_us_si: float | None = None
        k_us: float | None = None
        cag_tst_val: float | None = None
        cag_cvt_val: float | None = None
        k_ivtst_si: float | None = None
        k_ivtst: float | None = None
        k_vrc_vtst_si: float | None = None
        k_vrc_vtst: float | None = None
        dual_correction_val: float | None = None
        surface_factor_val = _surface_factor(options, temperature)
        pressure_rate_si, pressure_pa_val = _pressure_rate(options, temperature)

        if reaction.path is not None and options.method in {"ivtst", "all"}:
            coordinates = tuple(point.coordinate for point in reaction.path.points)
            free_energies = tuple(
                activation_gibbs_kj_mol(
                    reaction, point.dividing_surface, temperature
                )
                for point in reaction.path.points
            )
            if dual_profile is not None:
                corrections = tuple(dual_profile.at(item) for item in coordinates)
                dual_correction_val = max(corrections, key=abs) if corrections else 0.0
                free_energies = tuple(
                    value + correction
                    for value, correction in zip(free_energies, corrections)
                )
            ivtst = iv_tst_rate_si(
                coordinates,
                free_energies,
                temperature,
                molecularity=reaction.molecularity,
                standard_pressure_pa=reaction.standard_pressure_pa,
            )
            k_ivtst_si = ivtst.rate_si
            k_ivtst, _ = convert_rate_from_si(
                k_ivtst_si, reaction.molecularity, rate_unit
            )

        if options.method in {"vrc", "vrc-vtst", "all"} and options.vrc_vtst is not None:
            config = options.vrc_vtst
            if isinstance(config, VRCVTSTModel):
                vrc = config.rate_si(
                    temperature,
                    molecularity=reaction.molecularity,
                    standard_pressure_pa=reaction.standard_pressure_pa,
                )
                k_vrc_vtst_si = vrc.rate_si
                k_vrc_vtst, _ = convert_rate_from_si(
                    k_vrc_vtst_si, reaction.molecularity, rate_unit
                )
            else:
                if not isinstance(config, Mapping):
                    raise TypeError("vrc_vtst must be a VRCVTSTModel or mapping")
                bounds_raw = config.get("coordinate_bounds")
                coords_raw = config.get("coordinates")
                energies_raw = config.get(
                    "energies_kj_mol", config.get("potential_energies_kj_mol")
                )
                if bounds_raw is None:
                    if coords_raw is None:
                        raise ValueError("vrc_vtst requires coordinate_bounds or coordinates")
                    coords_raw = tuple(float(item) for item in coords_raw)
                    bounds_raw = (coords_raw[0], coords_raw[-1])
                if energies_raw is not None and coords_raw is not None:
                    coords_profile = tuple(float(item) for item in coords_raw)
                    energy_profile = tuple(float(item) for item in energies_raw)
                    if len(coords_profile) != len(energy_profile):
                        raise ValueError("VRC coordinates and energies must have matching lengths")

                    def potential_fn(coordinate: float) -> float:
                        if coordinate <= coords_profile[0]:
                            return energy_profile[0]
                        if coordinate >= coords_profile[-1]:
                            return energy_profile[-1]
                        for left, right, e_left, e_right in zip(
                            coords_profile,
                            coords_profile[1:],
                            energy_profile,
                            energy_profile[1:],
                        ):
                            if left <= coordinate <= right:
                                fraction = (coordinate - left) / (right - left)
                                return e_left + fraction * (e_right - e_left)
                        return energy_profile[-1]
                else:
                    potential_value = float(
                        config.get(
                            "energy_kj_mol", config.get("potential_energy_kj_mol", 0.0)
                        )
                    )

                    def potential_fn(_coordinate: float) -> float:
                        return potential_value

                phase_space = config.get("phase_space_weights")
                phase_fn = None
                if phase_space is not None and coords_raw is not None:
                    phase_profile = tuple(float(item) for item in phase_space)
                    coord_profile = tuple(float(item) for item in coords_raw)
                    if len(phase_profile) != len(coord_profile):
                        raise ValueError("VRC phase_space_weights must match coordinates")

                    def phase_fn(coordinate: float) -> float:
                        if coordinate <= coord_profile[0]:
                            return phase_profile[0]
                        if coordinate >= coord_profile[-1]:
                            return phase_profile[-1]
                        for left, right, p_left, p_right in zip(
                            coord_profile,
                            coord_profile[1:],
                            phase_profile,
                            phase_profile[1:],
                        ):
                            if left <= coordinate <= right:
                                fraction = (coordinate - left) / (right - left)
                                return p_left + fraction * (p_right - p_left)
                        return phase_profile[-1]

                vrc = vrc_vtst_rate_si(
                    (float(bounds_raw[0]), float(bounds_raw[1])),
                    potential_fn,
                    temperature,
                    phase_space_fn=phase_fn,
                    quadrature_points=int(config.get("quadrature_points", 32)),
                    molecularity=reaction.molecularity,
                    standard_pressure_pa=reaction.standard_pressure_pa,
                )
                k_vrc_vtst_si = vrc.rate_si
                k_vrc_vtst, _ = convert_rate_from_si(
                    k_vrc_vtst_si, reaction.molecularity, rate_unit
                )

        if reaction.path is not None:
            cvt = cvt_rate_si(reaction, temperature)
            delta_g_cvt = cvt.activation_gibbs_kj_mol
            cvt_coordinate = cvt.coordinate
            sample_coordinate = cvt.sampled_coordinate
            cvt_fit_points = cvt.fit_points
            k_cvt_si = cvt.rate_si
            k_cvt, _ = convert_rate_from_si(
                k_cvt_si, reaction.molecularity, rate_unit
            )

            cus = cus_rate_si(reaction, temperature)
            k_cus_si = cus.rate_si
            k_cus, _ = convert_rate_from_si(
                k_cus_si, reaction.molecularity, rate_unit
            )

            k_muvt_si, k_us_si = muvt_and_us_rates_si(
                reaction, temperature, cvt_result=cvt
            )
            k_muvt, _ = convert_rate_from_si(
                k_muvt_si, reaction.molecularity, rate_unit
            )
            k_us, _ = convert_rate_from_si(
                k_us_si, reaction.molecularity, rate_unit
            )

            cag_tst_val = cag_kappa(reaction, temperature, variational_coordinate=0.0)
            cag_cvt_val = cag_kappa(
                reaction, temperature, variational_coordinate=cvt_coordinate
            )

        # Apply the free-energy solvent correction to every path rate.
        if solvent_delta:
            solvent_factor = math.exp(
                -solvent_delta * 1000.0 / (8.31446261815324 * temperature)
            )
            if k_cvt_si is not None:
                k_cvt_si *= solvent_factor
                k_cvt, _ = convert_rate_from_si(k_cvt_si, reaction.molecularity, rate_unit)
            if k_cus_si is not None:
                k_cus_si *= solvent_factor
                k_cus, _ = convert_rate_from_si(k_cus_si, reaction.molecularity, rate_unit)
            if k_muvt_si is not None:
                k_muvt_si *= solvent_factor
                k_muvt, _ = convert_rate_from_si(k_muvt_si, reaction.molecularity, rate_unit)
            if k_us_si is not None:
                k_us_si *= solvent_factor
                k_us, _ = convert_rate_from_si(k_us_si, reaction.molecularity, rate_unit)
            if k_ivtst_si is not None:
                k_ivtst_si *= solvent_factor
                k_ivtst, _ = convert_rate_from_si(k_ivtst_si, reaction.molecularity, rate_unit)
            if k_vrc_vtst_si is not None:
                k_vrc_vtst_si *= solvent_factor
                k_vrc_vtst, _ = convert_rate_from_si(k_vrc_vtst_si, reaction.molecularity, rate_unit)

        kappa_wigner_val: float | None = None
        if len(reaction.transition_state.imaginary_frequencies_cm1) == 1:
            kappa_wigner_val = transition_state_wigner_kappa(
                reaction.transition_state, temperature
            )

        kappa_zct_val: float | None = None
        if zct_profile is not None:
            kappa_zct_val = profile_wkb_kappa(zct_profile, temperature)

        kappa_sct_val: float | None = None
        if sct_profile is not None:
            kappa_sct_val = profile_wkb_kappa(sct_profile, temperature)

        kappa_comt_val: float | None = None
        if kappa_zct_val is not None and kappa_sct_val is not None:
            kappa_comt_val = max(kappa_zct_val, kappa_sct_val)

        kappa_muomt_val: float | None = None
        if (
            zct_profile is not None
            and sct_profile is not None
            and options.tunneling in {"omt", "comt", "muomt"}
        ):
            kappa_muomt_val = profile_muomt_kappa(zct_profile, sct_profile, temperature)

        kappa_eckart_val: float | None = None
        if options.tunneling == "eckart" or (
            options.method == "all"
            and len(reaction.transition_state.imaginary_frequencies_cm1) == 1
        ):
            kappa_eckart_val = reaction_eckart_kappa(reaction, temperature)

        if options.method == "cvt" and k_cvt_si is not None:
            base_rate_si = k_cvt_si
        elif options.method == "cus" and k_cus_si is not None:
            base_rate_si = k_cus_si
        elif options.method == "muvt" and k_muvt_si is not None:
            base_rate_si = k_muvt_si
        elif options.method == "us" and k_us_si is not None:
            base_rate_si = k_us_si
        elif options.method == "ivtst" and k_ivtst_si is not None:
            base_rate_si = k_ivtst_si
        elif options.method in {"vrc", "vrc-vtst"} and k_vrc_vtst_si is not None:
            base_rate_si = k_vrc_vtst_si
        elif options.method == "all" and dual_profile is not None and k_ivtst_si is not None:
            base_rate_si = k_ivtst_si
        elif options.method == "all" and k_cvt_si is not None:
            base_rate_si = k_cvt_si
        else:
            base_rate_si = k_tst_si

        if pressure_rate_si is not None:
            base_rate_si = pressure_rate_si
        if surface_factor_val is not None:
            base_rate_si *= surface_factor_val

        cag_active = (
            (cag_cvt_val if options.method != "tst" else cag_tst_val)
            if (reaction.path is not None and options.tunneling_profile is None)
            else 1.0
        ) or 1.0

        if options.tunneling == "none":
            kappa = 1.0
        elif options.tunneling == "wigner":
            kappa = transition_state_wigner_kappa(
                reaction.transition_state, temperature
            )
        elif options.tunneling == "eckart":
            assert kappa_eckart_val is not None
            kappa = kappa_eckart_val
        elif options.tunneling in {"sct", "cd-scsag"}:
            assert kappa_sct_val is not None
            kappa = cag_active * kappa_sct_val
        elif options.tunneling in {"omt", "comt"}:
            assert kappa_comt_val is not None
            kappa = cag_active * kappa_comt_val
        elif options.tunneling == "muomt":
            assert kappa_muomt_val is not None
            kappa = cag_active * kappa_muomt_val
        elif options.tunneling in {"zct", "mepsag"}:
            assert kappa_zct_val is not None
            kappa = cag_active * kappa_zct_val
        else:
            assert zct_profile is not None
            kappa = profile_wkb_kappa(zct_profile, temperature)

        k_final_si = base_rate_si * kappa
        k_final, _ = convert_rate_from_si(
            k_final_si, reaction.molecularity, rate_unit
        )

        kc_si_val: float | None = None
        k_rev_final: float | None = None
        if reaction.products:
            _, _, kc_si_val = reaction_equilibrium_constants(reaction, temperature)
            if kc_si_val > 0.0:
                k_rev_si = k_final_si / kc_si_val
                rev_molecularity = len(reaction.products)
                if rev_molecularity in {1, 2}:
                    rev_unit = (
                        rate_unit
                        if rev_molecularity == reaction.molecularity
                        else "auto"
                    )
                    k_rev_final, _ = convert_rate_from_si(
                        k_rev_si, rev_molecularity, rev_unit
                    )

        rows.append(
            RateRow(
                temperature_k=temperature,
                activation_gibbs_tst_kj_mol=delta_g_tst,
                k_tst=k_tst,
                activation_gibbs_cvt_kj_mol=delta_g_cvt,
                cvt_coordinate=cvt_coordinate,
                cvt_sample_coordinate=sample_coordinate,
                cvt_fit_points=cvt_fit_points,
                k_cvt=k_cvt,
                tunneling_correction=kappa,
                k_final=k_final,
                rate_unit=rate_unit,
                k_cus=k_cus,
                k_muvt=k_muvt,
                k_us=k_us,
                cag_tst=cag_tst_val,
                cag_cvt=cag_cvt_val,
                kappa_wigner=kappa_wigner_val,
                kappa_zct=kappa_zct_val,
                kappa_sct=kappa_sct_val,
                kappa_eckart=kappa_eckart_val,
                kappa_comt=kappa_comt_val,
                kappa_muomt=kappa_muomt_val,
                equilibrium_kc=kc_si_val,
                k_reverse_final=k_rev_final,
                k_ivtst=k_ivtst,
                k_vrc_vtst=k_vrc_vtst,
                coordinate_frequency_cm1=coordinate_frequency,
                dual_level_correction_kj_mol=dual_correction_val,
                solvent_correction_kj_mol=(
                    solvent_delta if options.solvent is not None else None
                ),
                surface_rate_correction=surface_factor_val,
                pressure_pa=pressure_pa_val,
                k_pressure=(
                    convert_rate_from_si(
                        pressure_rate_si, reaction.molecularity, rate_unit
                    )[0]
                    if pressure_rate_si is not None
                    else None
                ),
            )
        )

    return CalculationResult(
        name=name,
        method=options.method,
        tunneling=options.tunneling,
        molecularity=reaction.molecularity,
        standard_pressure_pa=reaction.standard_pressure_pa,
        rows=tuple(rows),
    )

