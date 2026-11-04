#!/usr/bin/env python3
"""Small reproducible benchmark for native and Python numerical kernels.

When PyRate is installed, the benchmark deliberately imports that installation
instead of unconditionally prepending ``src``.  Prepending ``src`` would shadow
an installed ``pyrate._kernels`` extension and silently benchmark the Python
fallback twice.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import timeit
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_backend() -> tuple[Any, Any, Any, str]:
    try:
        import pyrate
    except ModuleNotFoundError:
        sys.path.insert(0, str(ROOT / "src"))
        import pyrate

    from pyrate.native import backend_info, beyer_swinehart, thermal_wkb_kappa

    return backend_info, beyer_swinehart, thermal_wkb_kappa, str(Path(pyrate.__file__).resolve())


def run_current_backend(repeats: int) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    backend_info, beyer_swinehart, thermal_wkb_kappa, module_path = load_backend()

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
    return {
        "backend": backend_info(),
        "module_path": module_path,
        "repeats": repeats,
        "beyer_swinehart_seconds_per_call": dos_time / repeats,
        "thermal_wkb_seconds_per_call": wkb_time / repeats,
    }


def print_result(result: dict[str, Any]) -> None:
    print(result["backend"])
    print(f"module:           {result['module_path']}")
    print(f"Beyer-Swinehart: {result['beyer_swinehart_seconds_per_call']:.6f} s/call")
    print(f"thermal WKB:     {result['thermal_wkb_seconds_per_call']:.6f} s/call")


def run_child(repeats: int, *, disable_native: bool) -> dict[str, Any]:
    environment = os.environ.copy()
    if disable_native:
        environment["PYRATE_DISABLE_NATIVE"] = "1"
    else:
        environment.pop("PYRATE_DISABLE_NATIVE", None)
    completed = subprocess.run(
        [sys.executable, __file__, "--repeats", str(repeats), "--json"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--both", action="store_true", help="run native and fallback in separate processes"
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    args = parser.parse_args()

    if args.both:
        native = run_child(args.repeats, disable_native=False)
        fallback = run_child(args.repeats, disable_native=True)
        if native["backend"]["native"] is not True:
            raise RuntimeError(
                "native benchmark requested, but pyrate._kernels was not imported; "
                f"loaded package from {native['module_path']}"
            )
        if fallback["backend"]["native"] is not False:
            raise RuntimeError("fallback benchmark unexpectedly loaded native kernels")
        combined = {
            "native": native,
            "fallback": fallback,
            "speedup": {
                "beyer_swinehart": fallback["beyer_swinehart_seconds_per_call"]
                / native["beyer_swinehart_seconds_per_call"],
                "thermal_wkb": fallback["thermal_wkb_seconds_per_call"]
                / native["thermal_wkb_seconds_per_call"],
            },
        }
        if args.json:
            print(json.dumps(combined, sort_keys=True))
        else:
            print("native")
            print_result(native)
            print("fallback")
            print_result(fallback)
            print(
                "speedup: "
                f"Beyer-Swinehart {combined['speedup']['beyer_swinehart']:.2f}x; "
                f"thermal WKB {combined['speedup']['thermal_wkb']:.2f}x"
            )
        return 0

    result = run_current_backend(args.repeats)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
