from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pyrate.native._loader import cpp


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def native() -> Any:
    """The compiled extension; parity tests are skipped when it is unavailable."""

    if cpp is None:
        pytest.skip("native extension not available (or disabled)")
    return cpp
