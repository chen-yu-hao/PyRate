"""Load the optional compiled extension exactly once.

``cpp`` is the :mod:`pyrate._kernels` module or ``None``.  Setting
``PYRATE_DISABLE_NATIVE=1`` forces the pure-Python fallbacks.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

_DISABLE_NATIVE = os.environ.get("PYRATE_DISABLE_NATIVE", "").lower() in {"1", "true", "yes"}


def _load() -> Any | None:
    if _DISABLE_NATIVE:
        return None
    try:
        return importlib.import_module("pyrate._kernels")
    except ImportError:  # pragma: no cover - exercised by the fallback CI job
        return None


cpp: Any | None = _load()
HAS_NATIVE = cpp is not None


def backend_info() -> dict[str, str | bool]:
    return {
        "native": HAS_NATIVE,
        "backend": "cpp" if cpp is not None else "python",
        "version": str(cpp.native_version()) if cpp is not None else "python-fallback-0.2",
    }
