"""High-level orchestration for a complete temperature-grid calculation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from .models import CalculationOptions, Reaction
from .advanced import (
    DualLevelCorrection as AdvancedDualLevelCorrection,
    VRCWeightedSample,
    apply_dual_level_correction,
    path_curvilinear_frequencies,
    pressure_dependent_rate,
    vrc_vtst_rate,
)
from .constants import AVOGADRO_MOL, BOLTZMANN_J_K
from .rates import (
    conventional_tst_rate_si,
    convert_rate_from_si,
    cus_rate_si,
    cvt_rate_si,
    muvt_and_us_rates_si,
    normalize_rate_unit,
    reaction_equilibrium_constants,
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
    ivtst_coordinate: float | None = None
    k_vrc_vtst: float | None = None
    k_pressure: float | None = None
    pressure_pa: float | None = None
    pressure_rates: tuple[float, ...] = ()
    pressure_grid_pa: tuple[float, ...] = ()
    environment_correction_kj_mol: float = 0.0
    intrinsic_coordinate_system: str | None = None
    intrinsic_frequency_count: int | None = None

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


def calculate_rates(
    reaction: Reaction,
    options: CalculationOptions,
    *,
    name: str = "PyRate calculation",
) -> CalculationResult:
    """Calculate TST/CVT/CUS/muVT/US rates and tunneling corrections."""

    if options.method in {"cvt", "cus", "muvt", "us", "ivtst", "all"} and reaction.path is None:
        raise ValueError(f"method = {options.method!r} requires a reaction path")
    if options.method == "ivtst" and reaction.dual_level is None:
        raise ValueError("method = 'ivtst' requires reaction.dual_level configuration")
    if options.method in {"vrc_vtst"} and reaction.vrc is None:
        raise ValueError("method = 'vrc_vtst' requires reaction.vrc configuration")
    if options.method in {"pressure", "master_equation"} and reaction.pressure_dependence is None:
        raise ValueError(
            f"method = {options.method!r} requires reaction.pressure_dependence configuration"
        )
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
        ivtst_coordinate: float | None = None
        k_vrc_si: float | None = None
        k_vrc: float | None = None
        intrinsic_system: str | None = None
        intrinsic_count: int | None = None

        if reaction.path is not None:
            intrinsic_system = str(
                options.coordinate_system or getattr(reaction.path, "coordinate_system", "cartesian")
            )
            intrinsic = path_curvilinear_frequencies(reaction.path)
            intrinsic_count = max((len(values) for values in intrinsic.values()), default=0)

        if reaction.path is not None and reaction.dual_level is not None:
            model = reaction.dual_level
            if getattr(model, "enabled", True):
                coordinates = tuple(point.coordinate for point in reaction.path.points)
                low = tuple(
                    getattr(model, "low_level_energies_kj_mol", ())
                    or getattr(model, "low_energies_kj_mol", ())
                )
                high = tuple(
                    getattr(model, "high_level_energies_kj_mol", ())
                    or getattr(model, "high_energies_kj_mol", ())
                )
                if low and high and len(low) == 1:
                    shift = high[0] - low[0]
                    shifted_points = tuple(
                        replace(
                            point,
                            dividing_surface=replace(
                                point.dividing_surface,
                                electronic_energy_kj_mol=point.dividing_surface.electronic_energy_kj_mol + shift,
                            ),
                        )
                        for point in reaction.path.points
                    )
                    ivtst = cvt_rate_si(replace(reaction, path=replace(reaction.path, points=shifted_points)), temperature)
                    k_ivtst_si = ivtst.rate_si
                    k_ivtst, _ = convert_rate_from_si(k_ivtst_si, reaction.molecularity, rate_unit)
                    ivtst_coordinate = ivtst.coordinate
                elif low and high:
                    source = tuple(
                        getattr(model, "coordinates", ())
                        or getattr(model, "low_coordinates", ())
                    ) or coordinates
                    high_source = tuple(getattr(model, "high_coordinates", ())) or source
                    method_name = str(getattr(model, "method", "ispe")).lower()
                    if method_name == "ivtst":
                        method_name = "ivtst-m"
                    advanced_model = AdvancedDualLevelCorrection(
                        low_coordinates=source,
                        low_energies_kj_mol=low,
                        high_coordinates=high_source,
                        high_energies_kj_mol=high,
                        method=method_name,
                    )
                    corrected_path = apply_dual_level_correction(reaction.path, advanced_model)
                    corrected_reaction = replace(reaction, path=corrected_path)
                    ivtst = cvt_rate_si(corrected_reaction, temperature)
                    k_ivtst_si = ivtst.rate_si
                    k_ivtst, _ = convert_rate_from_si(k_ivtst_si, reaction.molecularity, rate_unit)
                    ivtst_coordinate = ivtst.coordinate
                else:
                    shift = float(getattr(model, "energy_shift_kj_mol", 0.0))
                    if shift:
                        shifted_points = tuple(
                            replace(
                                point,
                                dividing_surface=replace(
                                    point.dividing_surface,
                                    electronic_energy_kj_mol=point.dividing_surface.electronic_energy_kj_mol + shift,
                                ),
                            )
                            for point in reaction.path.points
                        )
                        ivtst = cvt_rate_si(replace(reaction, path=replace(reaction.path, points=shifted_points)), temperature)
                        k_ivtst_si = ivtst.rate_si
                        k_ivtst, _ = convert_rate_from_si(k_ivtst_si, reaction.molecularity, rate_unit)
                        ivtst_coordinate = ivtst.coordinate

        if reaction.vrc is not None and getattr(reaction.vrc, "enabled", True):
            samples = tuple(
                VRCWeightedSample(
                    coordinate=float(sample.coordinate),
                    energy_kj_mol=float(
                        getattr(sample, "energy_kj_mol", getattr(sample, "activation_gibbs_kj_mol", 0.0))
                    ),
                    weight=float(sample.weight),
                    frequencies_cm1=tuple(getattr(sample, "frequencies_cm1", ())),
                    degeneracy=float(getattr(sample, "degeneracy", 1.0)),
                    angular_factor=float(getattr(sample, "orientation", 1.0) or 1.0),
                )
                for sample in reaction.vrc.samples
            )
            vrc = vrc_vtst_rate(samples, temperature)
            k_vrc_si = vrc.rate_si / float(getattr(reaction.vrc, "symmetry_factor", 1.0))
            if reaction.molecularity == 2:
                k_vrc_si /= temperature * BOLTZMANN_J_K / reaction.standard_pressure_pa * AVOGADRO_MOL
            k_vrc, _ = convert_rate_from_si(k_vrc_si, reaction.molecularity, rate_unit)

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
        elif options.method == "all" and k_cvt_si is not None:
            base_rate_si = k_cvt_si
        elif options.method == "ivtst" and k_ivtst_si is not None:
            base_rate_si = k_ivtst_si
        elif options.method == "vrc_vtst" and k_vrc_si is not None:
            base_rate_si = k_vrc_si
        else:
            base_rate_si = k_tst_si

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

        environment_correction = 0.0
        if reaction.environment is not None and getattr(reaction.environment, "enabled", True):
            mapping = dict(getattr(reaction.environment, "species_solvation_kj_mol", {}) or {})
            environment_correction = mapping.get(reaction.transition_state.name, 0.0) - sum(
                mapping.get(reactant.name, 0.0) for reactant in reaction.reactants
            )
            environment_correction += float(
                getattr(reaction.environment, "activation_free_energy_kj_mol", 0.0)
            )
            if getattr(reaction.environment, "phase", "gas") == "surface":
                environment_correction += float(getattr(reaction.environment, "adsorption_energy_kj_mol", 0.0) or 0.0)
                orientation = float(getattr(reaction.environment, "orientation_factor", 1.0))
                environment_correction -= 8.31446261815324 * temperature * math.log(orientation) / 1000.0
            k_final_si *= math.exp(-environment_correction * 1000.0 / (8.31446261815324 * temperature))

        pressure_rates: tuple[float, ...] = ()
        pressure_grid: tuple[float, ...] = ()
        pressure_rate_selected: float | None = None
        pressure_selected: float | None = None
        pressure_model = reaction.pressure_dependence
        if pressure_model is not None and getattr(pressure_model, "enabled", True):
            pressure_grid = tuple(options.pressures_pa or pressure_model.pressures_pa)
            if pressure_grid:
                concentration_1bar = 1.0e5 / (BOLTZMANN_J_K * temperature)
                reduced_1bar = min(1.0, float(getattr(pressure_model, "energy_transfer_cm1", 200.0)) / 1000.0)
                k_zero = float(getattr(pressure_model, "low_pressure_rate_si", 0.0) or 0.0)
                if k_zero <= 0.0:
                    k_zero = k_final_si * reduced_1bar / max(concentration_1bar, 1.0e-300)
                pressure_rates = tuple(float(v) for v in pressure_dependent_rate(
                    pressure_grid,
                    k_zero,
                    k_final_si,
                    temperature,
                    f_cent=float(getattr(pressure_model, "f_cent", 0.6)),
                    collision_efficiency=float(getattr(pressure_model, "collision_efficiency", 1.0)),
                ))
                pressure_selected = pressure_grid[0]
                pressure_rate_selected = pressure_rates[0]
                if options.method in {"pressure", "master_equation"} or options.pressures_pa:
                    k_final_si = pressure_rate_selected
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
                ivtst_coordinate=ivtst_coordinate,
                k_vrc_vtst=k_vrc,
                k_pressure=convert_rate_from_si(pressure_rate_selected, reaction.molecularity, rate_unit)[0] if pressure_rate_selected is not None else None,
                pressure_pa=pressure_selected,
                pressure_rates=pressure_rates,
                pressure_grid_pa=pressure_grid,
                environment_correction_kj_mol=environment_correction,
                intrinsic_coordinate_system=intrinsic_system,
                intrinsic_frequency_count=intrinsic_count,
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

