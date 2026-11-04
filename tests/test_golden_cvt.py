"""Golden regression for ``examples/harmonic_cvt.toml`` (CTST + CVT + Wigner).

Owned by the CVT area.  Values must only change together with a documented
algorithmic change; update them in the same commit and explain why.

History: the default refinement changed from a five-point quartic to the
guarded three-point parabola (N3).  k_TST and the Wigner kappa are unchanged;
s_CVT and k_CVT moved (k_CVT -19 % at 200 K, -5.2 % at 1000 K).  The previous
quartic values are kept in ``QUARTIC`` and are reproduced by the opt-in
``max_fit_points=5`` mode.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pyrate import LoadedCalculation, RateRow, calculate_rates, load_toml
from pyrate.rates import cvt_rate_si

# (T/K, k_TST, s_CVT, k_CVT, kappa_Wigner, k_final), unit s^-1.
GOLDEN = [
    (200.0, 8683.72285883704, 0.24087983757288556, 5026.136346262487,
     3.70490310023356, 18621.348131464463),
    (300.0, 10856097.031341055, 0.23793954604474563, 7602430.754250872,
     2.20217915565936, 16741914.539354937),
    (500.0, 4044702408.2133303, 0.2297885395282545, 3311429551.5958476,
     1.4327844960373697, 4744564921.24651),
    (1000.0, 484056211789.81866, 0.20282175601198627, 447476150310.47565,
     1.1081961240093423, 495891335360.691),
]
# (T/K, s_CVT, k_CVT) of the former five-point quartic, unit s^-1.
QUARTIC = [
    (200.0, 0.33073915924838737, 6179.64169897881),
    (300.0, 0.32626288130688863, 8757452.729704592),
    (500.0, 0.31291507255767564, 3625729569.458448),
    (1000.0, 0.25742166573586395, 472004362581.7078),
]
RTOL = 1.0e-9


@pytest.fixture(scope="module")
def job(request: pytest.FixtureRequest) -> LoadedCalculation:
    root = Path(request.config.rootpath)
    return load_toml(root / "examples" / "harmonic_cvt.toml")


@pytest.fixture(scope="module")
def rows(job: LoadedCalculation) -> list[RateRow]:
    return list(calculate_rates(job.reaction, job.options, name=job.name).rows)


@pytest.mark.parametrize("expected", GOLDEN, ids=lambda item: f"{item[0]:.0f}K")
def test_harmonic_cvt_golden(rows: list[RateRow], expected: tuple[float, ...]) -> None:
    temperature, k_tst, s_cvt, k_cvt, kappa, k_final = expected
    row = next(item for item in rows if item.temperature_k == temperature)
    assert row.rate_unit == "s^-1"
    assert math.isclose(row.k_tst, k_tst, rel_tol=RTOL)
    assert math.isclose(row.cvt_coordinate, s_cvt, rel_tol=RTOL)
    assert math.isclose(row.k_cvt, k_cvt, rel_tol=RTOL)
    assert math.isclose(row.tunneling_correction, kappa, rel_tol=RTOL)
    assert math.isclose(row.k_final, k_final, rel_tol=RTOL)


@pytest.mark.parametrize("expected", QUARTIC, ids=lambda item: f"{item[0]:.0f}K")
def test_opt_in_five_point_mode_reproduces_former_quartic(
    job: LoadedCalculation, expected: tuple[float, float, float]
) -> None:
    temperature, s_cvt, k_cvt = expected
    result = cvt_rate_si(job.reaction, temperature, max_fit_points=5)
    assert result.fit_points == 5
    assert math.isclose(result.coordinate, s_cvt, rel_tol=RTOL)
    assert math.isclose(result.rate_si, k_cvt, rel_tol=RTOL)
