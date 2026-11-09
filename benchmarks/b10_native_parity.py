"""B10: C++ kernels vs Python fallback on random inputs, plus timings.

The fallback branch is reached in-process by temporarily setting
``pyrate.native.HAS_NATIVE = False``.  When the extension is not built the case
reports itself as skipped.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from pyrate import native

from .common import check


@contextmanager
def fallback():
    saved = native.HAS_NATIVE
    native.HAS_NATIVE = False
    try:
        yield
    finally:
        native.HAS_NATIVE = saved


def both(fn, *args):
    a = fn(*args)
    with fallback():
        b = fn(*args)
    return a, b


def relerr(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = np.maximum(np.abs(b), 1e-300)
    return float(np.max(np.abs(a - b) / scale))


def timed(fn, *args, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def run() -> dict:
    base = {"case": "b10_native_parity", "title": "Native C++ kernels vs Python fallback",
            "references": [], "data": {"backend": native.backend_info()}}
    if not native.HAS_NATIVE:
        base["checks"] = [check("native extension available", 0.0, None)]
        return base
    rng = np.random.default_rng(2026)
    worst = {k: 0.0 for k in ("harmonic_log_partition", "wigner_kappa", "refine_maximum",
                              "wkb_exponent", "thermal_wkb_kappa", "beyer_swinehart")}
    for _ in range(300):
        freqs = tuple(rng.uniform(50, 4000, rng.integers(1, 30)))
        t = float(rng.uniform(50, 3000))
        worst["harmonic_log_partition"] = max(worst["harmonic_log_partition"], relerr(*both(native.harmonic_log_partition, freqs, t)))
        worst["wigner_kappa"] = max(worst["wigner_kappa"], relerr(*both(native.wigner_kappa, float(rng.uniform(100, 3000)), t)))
        n = int(rng.integers(5, 40))
        x = np.sort(rng.uniform(-1, 1, n))
        y = -(x - rng.uniform(-0.5, 0.5)) ** 2 + 0.01 * rng.normal(size=n)
        a, b = both(native.refine_maximum, x, y)
        worst["refine_maximum"] = max(worst["refine_maximum"], relerr([a.coordinate, a.value], [b.coordinate, b.value]),
                                      float(a.sample_index != b.sample_index or a.fit_points != b.fit_points))
        xs = tuple(np.linspace(-1e-10, 1e-10, int(rng.integers(5, 60))))
        height = float(rng.uniform(1e-20, 1e-19))
        vs = tuple(height * np.exp(-(np.asarray(xs) / 4e-11) ** 2))
        mu = float(rng.uniform(1, 10)) * 1.66e-27
        e = float(rng.uniform(0.01, 0.99)) * height
        worst["wkb_exponent"] = max(worst["wkb_exponent"], relerr(*both(native.wkb_exponent, xs, vs, e, mu)))
    for _ in range(20):
        xs = tuple(np.linspace(-1e-10, 1e-10, 41))
        vs = tuple(float(rng.uniform(1e-20, 1e-19)) * np.exp(-(np.asarray(xs) / 4e-11) ** 2))
        worst["thermal_wkb_kappa"] = max(worst["thermal_wkb_kappa"], relerr(*both(
            native.thermal_wkb_kappa, xs, vs, float(rng.uniform(150, 2000)), 1.67e-27, 200)))
        freqs = tuple(rng.uniform(100, 3500, 8))
        a, b = both(native.beyer_swinehart, freqs, 10.0, 3000)
        worst["beyer_swinehart"] = max(worst["beyer_swinehart"], float(np.max(np.abs(a - b))))
    # refine_maximum: C++ and NumPy use different least-squares solvers for the
    # local polynomial, so agreement is limited to ~1e-12 rather than bitwise.
    thresholds = {"beyer_swinehart": 0.0, "refine_maximum": 1e-10}
    checks = [check(f"{k}: max rel diff C++ vs Python", v, thresholds.get(k, 1e-12),
                    mode="eq" if k == "beyer_swinehart" else "le") for k, v in worst.items()]

    freqs = tuple(rng.uniform(100, 3500, 30))
    xs = tuple(np.linspace(-2e-10, 2e-10, 401))
    vs = tuple(4e-20 * np.maximum(0.0, 1 - (np.asarray(xs) / 2e-10) ** 2))
    timings = {}
    for name, fn, args in (("beyer_swinehart(30 modes, 1e5 grains)", native.beyer_swinehart, (freqs, 1.0, 100_000)),
                           ("thermal_wkb_kappa(401 pts, 400 E)", native.thermal_wkb_kappa, (xs, vs, 300.0, 1.67e-27, 400))):
        t_cpp = timed(fn, *args)
        with fallback():
            t_py = timed(fn, *args, repeats=1)
        timings[name] = {"cpp_s": t_cpp, "python_s": t_py, "speedup": t_py / t_cpp}
        checks.append(check(f"speedup {name}", t_py / t_cpp, None))
    base["data"]["timings"] = timings
    base["data"]["max_rel_diff"] = worst
    base["checks"] = checks
    return base
