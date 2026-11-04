"""Golden regression for ``examples/harmonic_cvt.toml`` (CTST + CVT + Wigner).

Owned by the CVT area.  Values must only change together with a documented
algorithmic change; update them in the same commit and explain why.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pyrate import RateRow, calculate_rates, load_toml

# (T/K, k_TST, s_CVT, k_CVT, kappa_Wigner, k_final), unit s^-1.
GOLDEN = [
    (200.0, 8683.72285883704, 0.33073915924838737, 6179.64169897881,
     3.70490310023356, 22894.973688879178),
    (300.0, 10856097.031341055, 0.32626288130688863, 8757452.729704592,
     2.20217915565936, 19285479.85802762),
    (500.0, 4044702408.2133303, 0.31291507255767564, 3625729569.458448,
     1.4327844960373697, 5194889113.944312),
    (1000.0, 484056211789.81866, 0.25742166573586395, 472004362581.7078,
     1.1081961240093423, 523073405128.5488),
]
RTOL = 1.0e-9


@pytest.fixture(scope="module")
def rows(request: pytest.FixtureRequest) -> list[RateRow]:
    root = Path(request.config.rootpath)
    job = load_toml(root / "examples" / "harmonic_cvt.toml")
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
