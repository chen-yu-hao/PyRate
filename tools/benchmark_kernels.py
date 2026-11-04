#!/usr/bin/env python3
"""Small reproducible benchmark for native and Python numerical kernels."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import timeit
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_current_backend(repeats: int) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from pyrate.native import backend_info, beyer_swinehart, thermal_wkb_kappa

    coordinates = [index * 1.0e-12 for index in range(401)]
    center = coordinates[len(coordinates) // 2]
    potentials = [4.0e-20 * max(0.0, 1.0 - ((x - center) / 2.0e-10) ** 2) for x in coordinates]

    dos_time = timeit.timeit(
        lambda: beyer_swinehart((300, 450, 700, 1000, 1200, 1500), 1.0, 20_000),
        number=repeats,
    )
    wkb_time = timeit.timeit(
        lambda: thermal_wkb_kappa(coordinates, potentials, 300.0, 1.67e-27, 400),
        number=repeats,
    )
    print(backend_info())
    print(f"Beyer-Swinehart: {dos_time / repeats:.6f} s/call")
    print(f"thermal WKB:     {wkb_time / repeats:.6f} s/call")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--both", action="store_true", help="run native and fallback in separate processes"
    )
    args = parser.parse_args()
    if args.both and os.environ.get("PYRATE_BENCH_CHILD") != "1":
        for disabled in ("0", "1"):
            environment = os.environ.copy()
            environment["PYRATE_BENCH_CHILD"] = "1"
            environment["PYRATE_DISABLE_NATIVE"] = disabled
            subprocess.run(
                [sys.executable, __file__, "--repeats", str(args.repeats)],
                env=environment,
                check=True,
            )
        return 0
    run_current_backend(args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
