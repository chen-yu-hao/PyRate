"""Golden regression for ``examples/wkb_tst.toml`` (bimolecular CTST + 1D WKB).

Owned by the tunneling area.  Values must only change together with a documented
algorithmic change; update them in the same commit and explain why.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pyrate import RateRow, calculate_rates, load_toml

# (T/K, k_TST, kappa_WKB, k_final), unit cm^3 molecule^-1 s^-1.
# kappa_WKB uses the uniform semiclassical P(E) = 1/(1+exp(2θ)) with
# nonclassical reflection and threshold E0 = max(V[0], V[-1]) (docs/roadmap.md
# N1/N2).  The former P = exp(-2θ) kernel gave 3.62053852245742,
# 1.9056968481115728 and 1.3636612822103644 (13-16% higher).
GOLDEN = [
    (200.0, 3.467999767240354e-21, 3.1368579522983135, 1.0878622648436604e-20),
    (300.0, 4.560347853274816e-18, 1.6051878689965948, 7.320215052481398e-18),
    (500.0, 1.421224173779944e-15, 1.176293782452655, 1.6717771590887596e-15),
]
RTOL = 1.0e-9


@pytest.fixture(scope="module")
def rows(request: pytest.FixtureRequest) -> list[RateRow]:
    root = Path(request.config.rootpath)
    job = load_toml(root / "examples" / "wkb_tst.toml")
    return list(calculate_rates(job.reaction, job.options, name=job.name).rows)


@pytest.mark.parametrize("expected", GOLDEN, ids=lambda item: f"{item[0]:.0f}K")
def test_wkb_tst_golden(rows: list[RateRow], expected: tuple[float, ...]) -> None:
    temperature, k_tst, kappa, k_final = expected
    row = next(item for item in rows if item.temperature_k == temperature)
    assert row.rate_unit == "cm^3 molecule^-1 s^-1"
    assert row.k_cvt is None
    assert math.isclose(row.k_tst, k_tst, rel_tol=RTOL)
    assert math.isclose(row.tunneling_correction, kappa, rel_tol=RTOL)
    assert math.isclose(row.k_final, k_final, rel_tol=RTOL)
