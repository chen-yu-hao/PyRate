#!/usr/bin/env python3
"""Compare Polyrate `fu15` reference columns with a PyRate JSON result.

Run from a source checkout or an installed environment:
  python tools/compare_polyrate_rates.py reference.fu15 pyrate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Works from a checkout without package installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyrate.head_to_head import compare_fu15_to_pyrate, parse_fu15_tables  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Polyrate .fu15 reference file")
    parser.add_argument("result", type=Path, help="PyRate --format json result")
    parser.add_argument("--direction", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--column", action="append", default=[], metavar="REF=FIELD",
                        help="explicit map, e.g. CVT=k_cvt; repeatable (defaults TST, CVT)")
    parser.add_argument("--rtol", type=float, default=5.0e-3)
    parser.add_argument("--strict", action="store_true", help="fail on any unmapped reference column")
    parser.add_argument("--output", type=Path, help="optional JSON report file")
    args = parser.parse_args()

    try:
        tables = parse_fu15_tables(args.reference.read_text(encoding="utf-8", errors="replace"))
        selected = [table for table in tables if table.direction == args.direction]
        if len(selected) != 1:
            raise ValueError(
                f"expected exactly one {args.direction} summary, found {len(selected)}; "
                "manual selection is required for ambiguous files"
            )
        result = json.loads(args.result.read_text(encoding="utf-8"))
        mapping = {"TST": "k_tst", "CVT": "k_cvt"}
        for pair in args.column:
            if pair.count("=") != 1 or not all(piece.strip() for piece in pair.split("=")):
                raise ValueError(f"invalid column mapping {pair!r}; expected REF=FIELD")
            column, field = pair.split("=", 1)
            mapping[column.strip().upper()] = field.strip()
        report = compare_fu15_to_pyrate(
            selected[0], result, column_map=mapping,
            relative_tolerance=args.rtol, strict=args.strict,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"comparison error: {exc}\n")

    rendered = json.dumps(report.as_dict(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
