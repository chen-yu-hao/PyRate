"""High-level orchestration for a complete temperature-grid calculation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .models import CalculationOptions, Reaction
from .rates import (
    conventional_tst_rate_si,
    convert_rate_from_si,
    cvt_rate_si,
    normalize_rate_unit,
)
from .tunneling import profile_wkb_kappa, transition_state_wigner_kappa


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalculationResult:
    name: str
    method: str
    tunneling: str
    molecularity: int
    standard_pressure_pa: float
    constants_profile: str
    rows: tuple[RateRow, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "tunneling": self.tunneling,
            "molecularity": self.molecularity,
            "standard_pressure_pa": self.standard_pressure_pa,
            "constants_profile": self.constants_profile,
            "rows": [row.as_dict() for row in self.rows],
        }


def calculate_rates(
    reaction: Reaction,
    options: CalculationOptions,
    *,
    name: str = "PyRate calculation",
) -> CalculationResult:
    """Calculate CTST/CVT rates and optional tunneling corrections."""

    if options.method == "cvt" and reaction.path is None:
        raise ValueError("method = 'cvt' requires a reaction path")
    rate_unit = normalize_rate_unit(options.output_rate_unit, reaction.molecularity)
    rows: list[RateRow] = []

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

        if options.tunneling == "none":
            kappa = 1.0
        elif options.tunneling == "wigner":
            kappa = transition_state_wigner_kappa(
                reaction.transition_state, temperature
            )
        else:
            assert options.tunneling_profile is not None
            kappa = profile_wkb_kappa(options.tunneling_profile, temperature)

        base_rate_si = (
            k_cvt_si if options.method == "cvt" and k_cvt_si is not None else k_tst_si
        )
        k_final, _ = convert_rate_from_si(
            base_rate_si * kappa, reaction.molecularity, rate_unit
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
