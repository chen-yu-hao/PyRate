"""Shared helpers for the paper benchmark cases.

Every case module exposes ``run() -> dict`` returning::

    {"case": str, "title": str, "references": [...],
     "checks": [{"name", "value", "threshold", "status"}],   # status: pass|fail|info
     "data": {...}}                                           # tables for the paper
"""

from __future__ import annotations

import math
from typing import Any


def check(name: str, value: float, threshold: float | None, *, mode: str = "le") -> dict[str, Any]:
    """Build one check record.  ``threshold=None`` marks an informational number."""

    if threshold is None:
        status = "info"
    elif not math.isfinite(value):
        status = "fail"
    elif mode == "le":
        status = "pass" if value <= threshold else "fail"
    elif mode == "eq":
        status = "pass" if value == threshold else "fail"
    else:  # pragma: no cover - programming error
        raise ValueError(mode)
    return {"name": name, "value": value, "threshold": threshold, "status": status}


def rel(a: float, b: float) -> float:
    return abs(a / b - 1.0)
