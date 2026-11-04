"""Stable Python interface to optional C++ numerical kernels.

Each numerical area has its own submodule (``thermo``, ``fit``, ``wkb``,
``dos``) mirroring one C++ translation unit in ``cpp/``.  Every public kernel
validates its inputs once and dispatches to :mod:`pyrate._kernels` when the
extension is available; the ``py_*`` functions are the pure-Python reference
implementations used as fallbacks and by the cross-backend parity tests.
"""

from __future__ import annotations

from ._loader import HAS_NATIVE, backend_info
from .dos import beyer_swinehart
from .fit import RefinedMaximum, refine_maximum
from .thermo import harmonic_log_partition, wigner_kappa
from .wkb import thermal_wkb_kappa, wkb_exponent

__all__ = [
    "HAS_NATIVE",
    "RefinedMaximum",
    "backend_info",
    "beyer_swinehart",
    "harmonic_log_partition",
    "refine_maximum",
    "thermal_wkb_kappa",
    "wigner_kappa",
    "wkb_exponent",
]
