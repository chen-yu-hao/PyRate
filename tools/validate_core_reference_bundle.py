#!/usr/bin/env python3
"""Validate official core fixed-unit files and cross-format equivalence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Iterable

from pyrate.fixed_units import FixedUnitDocument, parse_fixed_unit_text, summarize_fixed_unit


def _maximum_delta(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = tuple(left)
    right_values = tuple(right)
    if len(left_values) != len(right_values):
        raise RuntimeError(
            f"fixed-unit vectors have different lengths: {len(left_values)} != {len(right_values)}"
        )
    return max((abs(a - b) for a, b in zip(left_values, right_values)), default=0.0)


def _case_documents(bundle: dict[str, Any]) -> dict[str, dict[int, FixedUnitDocument]]:
    result: dict[str, dict[int, FixedUnitDocument]] = {}
    for case in bundle["cases"]:
        case_id = str(case["case_id"])
        documents: dict[int, FixedUnitDocument] = {}
        for item in case["files"]:
            suffix = PurePosixPath(item["path"]).suffix.lower()
            unit = {".fu30": 30, ".fu31": 31, ".fu40": 40}.get(suffix)
            if unit is None:
                continue
            text = item.get("text")
            if not isinstance(text, str):
                raise RuntimeError(f"fixed-unit file is not readable text: {item['path']}")
            documents[unit] = parse_fixed_unit_text(
                text, unit, source_name=str(item["path"])
            )
        result[case_id] = documents
    return result


def _stationary_by_role(document: FixedUnitDocument) -> dict[str, Any]:
    return {point.role: point for point in document.stationary_points()}


def validate(bundle: dict[str, Any]) -> dict[str, Any]:
    documents = _case_documents(bundle)
    unit_counts: Counter[int] = Counter()
    stationary_count = 0
    path_point_count = 0
    summaries: list[dict[str, Any]] = []

    required_roles = {"reactant1", "reactant2", "product1", "product2", "saddle"}
    for case_id, by_unit in documents.items():
        for unit, document in by_unit.items():
            summary = summarize_fixed_unit(document)
            summary["case_id"] = case_id
            summaries.append(summary)
            unit_counts[unit] += 1
            roles = {point.role for point in document.stationary_points()}
            if roles != required_roles:
                raise RuntimeError(
                    f"{case_id} unit {unit} stationary roles {sorted(roles)} "
                    f"do not match {sorted(required_roles)}"
                )
            stationary_count += len(roles)
            path_point_count += len(document.path_points())

    expected_counts = {30: 5, 31: 4, 40: 5}
    if dict(unit_counts) != expected_counts:
        raise RuntimeError(
            f"expected official fixed-unit counts {expected_counts}, found {dict(unit_counts)}"
        )

    max_frequency_delta = 0.0
    max_hessian_delta = 0.0
    max_potential_delta = 0.0
    discrepancies: list[dict[str, Any]] = []
    ch5_pairs: list[dict[str, Any]] = []

    for index in range(1, 5):
        case30 = f"ch5fu30tr{index}"
        case40 = f"ch5fu40tr{index}"
        doc30 = documents[case30][30]
        doc40 = documents[case40][40]
        points30 = _stationary_by_role(doc30)
        points40 = _stationary_by_role(doc40)
        pair_frequency_delta = 0.0
        pair_hessian_delta = 0.0
        pair_potential_delta = 0.0
        for role in sorted(required_roles):
            left = points30[role]
            right = points40[role]
            if left.potential_hartree is not None and right.potential_hartree is not None:
                pair_potential_delta = max(
                    pair_potential_delta,
                    abs(left.potential_hartree - right.potential_hartree),
                )
            if left.frequencies and right.frequencies:
                pair_frequency_delta = max(
                    pair_frequency_delta,
                    _maximum_delta(left.frequencies_cm1, right.frequencies_cm1),
                )
            if left.hessian_values and right.hessian_values:
                pair_hessian_delta = max(
                    pair_hessian_delta,
                    _maximum_delta(left.hessian_values, right.hessian_values),
                )
            if left.active_mode_indices != right.active_mode_indices:
                discrepancies.append(
                    {
                        "kind": "official_mode_index_mismatch",
                        "pair": [case30, case40],
                        "role": role,
                        "unit30": list(left.active_mode_indices),
                        "unit40": list(right.active_mode_indices),
                    }
                )
        if pair_potential_delta > 1.0e-12:
            raise RuntimeError(
                f"{case30}/{case40} potential mismatch {pair_potential_delta} hartree"
            )
        if pair_frequency_delta > 0.5:
            raise RuntimeError(
                f"{case30}/{case40} frequency mismatch {pair_frequency_delta} cm^-1"
            )
        if pair_hessian_delta > 1.0e-14:
            raise RuntimeError(
                f"{case30}/{case40} Hessian mismatch {pair_hessian_delta}"
            )
        max_frequency_delta = max(max_frequency_delta, pair_frequency_delta)
        max_hessian_delta = max(max_hessian_delta, pair_hessian_delta)
        max_potential_delta = max(max_potential_delta, pair_potential_delta)
        ch5_pairs.append(
            {
                "unit30_case": case30,
                "unit40_case": case40,
                "max_potential_delta_hartree": pair_potential_delta,
                "max_frequency_delta_cm1": pair_frequency_delta,
                "max_hessian_delta": pair_hessian_delta,
            }
        )

    oh30 = documents["oh3fu30tr1"][30]
    oh40 = documents["oh3fu40tr1"][40]
    stationary30 = _stationary_by_role(oh30)
    stationary40 = _stationary_by_role(oh40)
    oh_stationary_hessian_delta = 0.0
    for role in sorted(required_roles):
        left = stationary30[role]
        right = stationary40[role]
        if left.potential_hartree != right.potential_hartree:
            raise RuntimeError(f"OH3 stationary potential mismatch for {role}")
        oh_stationary_hessian_delta = max(
            oh_stationary_hessian_delta,
            _maximum_delta(left.hessian_values, right.hessian_values),
        )
    path30 = oh30.path_points()
    path40 = oh40.path_points()
    if len(path30) != 4 or len(path40) != 4:
        raise RuntimeError(
            f"OH3 reference path requires 4 points per format, found {len(path30)} and {len(path40)}"
        )
    max_path_coordinate_delta = 0.0
    max_path_potential_delta = 0.0
    max_path_geometry_delta = 0.0
    max_path_gradient_delta = 0.0
    max_path_hessian_delta = 0.0
    for left, right in zip(path30, path40):
        max_path_coordinate_delta = max(
            max_path_coordinate_delta,
            abs(left.coordinate_au - right.coordinate_au),
        )
        max_path_potential_delta = max(
            max_path_potential_delta,
            abs(left.potential_hartree - right.potential_hartree),
        )
        max_path_geometry_delta = max(
            max_path_geometry_delta,
            _maximum_delta(
                (value for row in left.geometry_rows for value in row),
                (value for row in right.geometry_rows for value in row),
            ),
        )
        max_path_gradient_delta = max(
            max_path_gradient_delta,
            _maximum_delta(
                (value for row in left.gradient_rows for value in row),
                (value for row in right.gradient_rows for value in row),
            ),
        )
        max_path_hessian_delta = max(
            max_path_hessian_delta,
            _maximum_delta(left.hessian_values, right.hessian_values),
        )
        if left.active_mode_indices != right.active_mode_indices:
            raise RuntimeError("OH3 path active-mode indices differ between units 30 and 40")
    if max_path_coordinate_delta > 3.0e-11:
        raise RuntimeError(f"OH3 path coordinate mismatch {max_path_coordinate_delta}")
    for name, value in {
        "potential": max_path_potential_delta,
        "geometry": max_path_geometry_delta,
        "gradient": max_path_gradient_delta,
        "hessian": max_path_hessian_delta,
    }.items():
        if value > 1.0e-14:
            raise RuntimeError(f"OH3 path {name} mismatch {value}")

    return {
        "schema": "pyrate-polyrate-fixed-unit-validation-v1",
        "parsed_fixed_unit_files": sum(unit_counts.values()),
        "files_by_unit": {str(key): unit_counts[key] for key in sorted(unit_counts)},
        "stationary_records": stationary_count,
        "path_point_records": path_point_count,
        "ch5_cross_format_pairs": ch5_pairs,
        "oh3_cross_format": {
            "stationary_max_hessian_delta": oh_stationary_hessian_delta,
            "path_point_count_per_format": len(path30),
            "max_path_coordinate_delta_au": max_path_coordinate_delta,
            "max_path_potential_delta_hartree": max_path_potential_delta,
            "max_path_geometry_delta": max_path_geometry_delta,
            "max_path_gradient_delta": max_path_gradient_delta,
            "max_path_hessian_delta": max_path_hessian_delta,
        },
        "max_frequency_delta_cm1": max_frequency_delta,
        "max_hessian_delta": max_hessian_delta,
        "max_potential_delta_hartree": max_potential_delta,
        "official_source_discrepancies": discrepancies,
        "files": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="core-reference-bundle.json")
    parser.add_argument("--output", required=True, help="validation report JSON")
    args = parser.parse_args()

    with open(args.bundle, encoding="utf-8") as handle:
        bundle = json.load(handle)
    report = validate(bundle)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "parsed_fixed_unit_files": report["parsed_fixed_unit_files"],
                "files_by_unit": report["files_by_unit"],
                "path_point_records": report["path_point_records"],
                "official_source_discrepancies": len(
                    report["official_source_discrepancies"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
