"""Command-line interface for PyRate."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .calculation import CalculationResult, calculate_rates
from .direct_dynamics import DirectDynamicsError, ExternalCommandPes
from .input import (
    legacy_to_calculation,
    load_input,
    parse_toml_data,
)
from .legacy import LegacyInput, LegacySemanticError, validate_legacy_semantics
from .native import backend_info
from .pes import PotentialEnergySurface, get_builtin_pes


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
    for cells in rows:
        for index, value in enumerate(cells):
            widths[index] = max(widths[index], len(value))
    lines = [
        f"{result.name}",
        f"method={result.method}; tunneling={result.tunneling}; unit={result.rows[0].rate_unit}",
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.rjust(widths[index]) for index, value in enumerate(cells))
        for cells in rows
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


def _resolve_cli_pes(
    pes_spec: str | None, restart_path: str | None
) -> PotentialEnergySurface | None:
    if pes_spec is None:
        return None
    lower = pes_spec.strip().lower()
    if lower.startswith("command:") or lower.startswith("external:"):
        cmd = pes_spec.split(":", 1)[1].strip()
        return ExternalCommandPes(command=cmd, restart_file=restart_path)
    return get_builtin_pes(pes_spec)


def _run_command(args: argparse.Namespace) -> int:
    pes = _resolve_cli_pes(getattr(args, "pes", None), getattr(args, "restart", None))
    input_path = Path(args.input)
    restart_path = getattr(args, "restart", None)

    if (
        input_path.suffix.lower() == ".toml"
        and pes is None
        and restart_path is not None
    ):
        with input_path.open("rb") as handle:
            raw_data = dict(tomllib.load(handle))
        pot_tbl = raw_data.get("potential")
        if isinstance(pot_tbl, Mapping):
            pot_copy = dict(pot_tbl)
            pot_type = str(pot_copy.get("type", "")).strip().lower()
            if pot_type in {"command", "external", "gaussian", "gamess"}:
                pot_copy["restart_file"] = str(restart_path)
                raw_data["potential"] = pot_copy
        loaded: Any = parse_toml_data(raw_data, source=str(input_path))
    else:
        loaded = load_input(input_path, pes=pes)

    if isinstance(loaded, LegacyInput):
        loaded = legacy_to_calculation(
            loaded,
            fu40_path=getattr(args, "fu40", None),
            pes=pes,
        )

    export_fu40 = getattr(args, "export_fu40", None)
    if export_fu40 is not None:
        loaded.save_fu40(export_fu40)

    if args.rate_unit is not None:
        from dataclasses import replace

        loaded = replace(
            loaded,
            options=replace(loaded.options, output_rate_unit=args.rate_unit),
        )
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
        if getattr(args, "strict", False):
            validate_legacy_semantics(loaded)
        warnings = loaded.compatibility_warnings()
        sys.stdout.write("syntax: valid Polyrate-style input\n")
        if getattr(args, "strict", False):
            sys.stdout.write("compatibility: strict validation passed\n")
        elif warnings:
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

    run_parser = subparsers.add_parser(
        "run", help="run a PyRate TOML or Polyrate .dat+.fu40 calculation"
    )
    run_parser.add_argument("input", help="input TOML or Polyrate .dat file")
    run_parser.add_argument(
        "--fu40", help="companion Polyrate unit-40 (.fu40) electronic-structure file"
    )
    run_parser.add_argument(
        "--pes",
        help=(
            "analytical PES name (e.g. cl_hbr_leps, h3_leps) or external driver "
            "(command:<cmd>)"
        ),
    )
    run_parser.add_argument(
        "--export-fu40",
        help="write an integrated MEP and stationary points to a Polyrate .fu40 file",
    )
    run_parser.add_argument(
        "--restart",
        help="path to JSON restart cache for direct-dynamics PES evaluations",
    )
    run_parser.add_argument(
        "--format", choices=("table", "json", "csv"), default="table"
    )
    run_parser.add_argument("-o", "--output", help="output path; default is stdout")
    run_parser.add_argument(
        "--rate-unit",
        help="override output unit (SI, L mol^-1 s^-1, cm^3 molecule^-1 s^-1)",
    )
    run_parser.set_defaults(handler=_run_command)

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
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="fail with exit code 2 on unsupported legacy Polyrate sections or keywords",
    )
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
    except (
        OSError,
        ValueError,
        KeyError,
        NotImplementedError,
        LegacySemanticError,
        DirectDynamicsError,
    ) as exc:
        parser.exit(2, f"pyrate: error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
