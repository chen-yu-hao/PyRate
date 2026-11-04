"""Command-line interface for PyRate."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .calculation import CalculationResult, calculate_rates
from .input import LoadedCalculation, load_input
from .legacy import LegacyInput
from .native import backend_info
from .numerical_trace import (
    NumericalTrace,
    build_numerical_trace,
    compare_numerical_traces,
)


def _format_optional(value: float | int | None, specifier: str = ".6g") -> str:
    if value is None:
        return "-"
    return format(value, specifier)


def format_table(result: CalculationResult) -> str:
    headers = [
        "T / K",
        "ΔG‡ TST / kJ mol⁻¹",
        "k_TST",
        "s_CVT",
        "ΔG‡ CVT / kJ mol⁻¹",
        "k_CVT",
        "κ",
        "k_final",
    ]
    rows: list[list[str]] = []
    for row in result.rows:
        rows.append(
            [
                f"{row.temperature_k:.3f}",
                f"{row.activation_gibbs_tst_kj_mol:.6f}",
                f"{row.k_tst:.8e}",
                _format_optional(row.cvt_coordinate, ".6f"),
                _format_optional(row.activation_gibbs_cvt_kj_mol, ".6f"),
                _format_optional(row.k_cvt, ".8e"),
                f"{row.tunneling_correction:.8g}",
                f"{row.k_final:.8e}",
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = [
        f"{result.name}",
        f"method={result.method}; tunneling={result.tunneling}; unit={result.rows[0].rate_unit}",
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.rjust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines) + "\n"


def format_csv(result: CalculationResult) -> str:
    stream = io.StringIO()
    fieldnames = list(result.rows[0].as_dict())
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in result.rows:
        writer.writerow(row.as_dict())
    return stream.getvalue()


def _write_output(text: str, output: str | None) -> None:
    if output is None or output == "-":
        sys.stdout.write(text)
        return
    Path(output).write_text(text, encoding="utf-8")


def _solvable_input(path: str) -> LoadedCalculation:
    loaded = load_input(path)
    if isinstance(loaded, LegacyInput):
        raise ValueError(
            "legacy Polyrate files can currently be inspected and validated, but the solver "
            "requires a PyRate TOML file because legacy potential hooks do not contain a "
            "portable energy/frequency model"
        )
    return loaded


def _override_rate_unit(
    loaded: LoadedCalculation, rate_unit: str | None
) -> LoadedCalculation:
    if rate_unit is None:
        return loaded
    return replace(
        loaded,
        options=replace(loaded.options, output_rate_unit=rate_unit),
    )


def _run_command(args: argparse.Namespace) -> int:
    loaded = _override_rate_unit(_solvable_input(args.input), args.rate_unit)
    result = calculate_rates(
        loaded.reaction,
        loaded.options,
        name=loaded.name,
    )
    if args.format == "json":
        text = json.dumps(result.as_dict(), indent=2, ensure_ascii=False) + "\n"
    elif args.format == "csv":
        text = format_csv(result)
    else:
        text = format_table(result)
    _write_output(text, args.output)
    return 0


def _trace_command(args: argparse.Namespace) -> int:
    loaded = _override_rate_unit(_solvable_input(args.input), args.rate_unit)
    trace = build_numerical_trace(
        loaded.reaction,
        loaded.options,
        name=loaded.name,
        action_grid_points=args.action_grid_points,
    )
    _write_output(
        json.dumps(trace.as_dict(), indent=2, ensure_ascii=False) + "\n",
        args.output,
    )
    return 0


def _compare_trace_command(args: argparse.Namespace) -> int:
    reference = NumericalTrace.from_dict(
        json.loads(Path(args.reference).read_text(encoding="utf-8"))
    )
    actual = NumericalTrace.from_dict(
        json.loads(Path(args.actual).read_text(encoding="utf-8"))
    )
    report = compare_numerical_traces(reference, actual, strict=args.strict)
    _write_output(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n",
        args.output,
    )
    return 0 if report.success else 1


def _inspect_command(args: argparse.Namespace) -> int:
    loaded = load_input(args.input)
    if isinstance(loaded, LegacyInput):
        payload: dict[str, Any] = loaded.as_dict() if args.full else loaded.summary()
    else:
        payload = loaded.summary()
    if args.json:
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    else:
        for key, value in payload.items():
            if isinstance(value, list):
                rendered = ", ".join(str(item) for item in value) or "-"
            else:
                rendered = str(value)
            sys.stdout.write(f"{key}: {rendered}\n")
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    loaded = load_input(args.input)
    if isinstance(loaded, LegacyInput):
        warnings = loaded.compatibility_warnings()
        sys.stdout.write("syntax: valid Polyrate-style input\n")
        if warnings:
            for warning in warnings:
                sys.stdout.write(f"warning: {warning}\n")
        else:
            sys.stdout.write("compatibility: supported subset\n")
    else:
        # Construction performs all structural and physical validation.
        sys.stdout.write("syntax: valid PyRate TOML input\n")
        sys.stdout.write(
            f"calculation: {loaded.name}; {len(loaded.options.temperatures_k)} temperatures\n"
        )
    return 0


def _native_info_command(_: argparse.Namespace) -> int:
    sys.stdout.write(json.dumps(backend_info(), indent=2) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyrate",
        description="Python/C++ transition-state-theory and variational rate calculations",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a PyRate TOML calculation")
    run_parser.add_argument("input", help="input TOML file")
    run_parser.add_argument(
        "--format", choices=("table", "json", "csv"), default="table"
    )
    run_parser.add_argument("-o", "--output", help="output path; default is stdout")
    run_parser.add_argument(
        "--rate-unit",
        help="override output unit (SI, L mol^-1 s^-1, cm^3 molecule^-1 s^-1)",
    )
    run_parser.set_defaults(handler=_run_command)

    trace_parser = subparsers.add_parser(
        "trace",
        help="emit partition, ZPE, free-energy, action, kappa, and rate diagnostics",
    )
    trace_parser.add_argument("input", help="input TOML file")
    trace_parser.add_argument("-o", "--output", help="output JSON path; default is stdout")
    trace_parser.add_argument(
        "--action-grid-points",
        type=int,
        default=33,
        help="number of energy points used for an energy-resolved WKB action trace",
    )
    trace_parser.add_argument(
        "--rate-unit",
        help="override output rate unit recorded in the trace",
    )
    trace_parser.set_defaults(handler=_trace_command)

    compare_parser = subparsers.add_parser(
        "compare-trace",
        help="compare a reference and actual staged numerical trace",
    )
    compare_parser.add_argument("reference", help="reference trace JSON")
    compare_parser.add_argument("actual", help="actual trace JSON")
    compare_parser.add_argument("-o", "--output", help="comparison JSON path")
    compare_parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when the actual trace contains unreferenced points",
    )
    compare_parser.set_defaults(handler=_compare_trace_command)

    inspect_parser = subparsers.add_parser(
        "inspect", help="summarize a TOML or legacy Polyrate input"
    )
    inspect_parser.add_argument("input")
    inspect_parser.add_argument("--json", action="store_true", help="emit JSON")
    inspect_parser.add_argument(
        "--full", action="store_true", help="include every parsed legacy entry"
    )
    inspect_parser.set_defaults(handler=_inspect_command)

    validate_parser = subparsers.add_parser(
        "validate", help="validate a TOML or legacy Polyrate input"
    )
    validate_parser.add_argument("input")
    validate_parser.set_defaults(handler=_validate_command)

    native_parser = subparsers.add_parser(
        "native-info", help="show the active numerical backend"
    )
    native_parser.set_defaults(handler=_native_info_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, KeyError, NotImplementedError) as exc:
        parser.exit(2, f"pyrate: error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
