"""Regular CI guard: every self-contained paper benchmark must have no failing check."""

from __future__ import annotations

import importlib

import pytest

from benchmarks.run_all import CASES


@pytest.mark.parametrize("case", CASES)
def test_paper_benchmark_case_passes(case: str) -> None:
    result = importlib.import_module(f"benchmarks.{case}").run()
    failing = [c for c in result["checks"] if c["status"] == "fail"]
    assert not failing, failing
