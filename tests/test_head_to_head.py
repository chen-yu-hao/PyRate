from __future__ import annotations

import json

import pytest

from pyrate.constants import AVOGADRO_MOL
from pyrate.head_to_head import compare_fu15_to_pyrate, parse_fu15_tables


FU15 = """
 Summary of forward rate constants (cm**3/molecule-1 s-1) :
     T(K) TST CVT CVT/ZCT CVT/SCT
   200.0 1.00D-20 1.50D-20 2.00D-20 3.00D-20
   300.0 2.00D-19 2.50D-19 3.00D-19 4.00D-19

 Summary of reverse rate constants (s-1) :
     T(K) TST CVT
   200.0 1.00D+01 1.20D+01
   300.0 2.00D+01 2.20D+01
"""

COMPACT_FINAL = """
 FINAL RATE CONSTANTS IVTST-1 (cm**3/molecule-sec)
 T(K) TST TST/W TST/ZCT
 200.0 1.00E-20 2.00E-20 3.00E-20
 300.0 2.00E-19 3.00E-19 4.00E-19

 FINAL RATE CONSTANTS (cm**3/molecule-sec)
 T(K) CVT CVT/W CVT/ZCT
 200.0 1.50E-20 2.50E-20 3.50E-20
 300.0 2.50E-19 3.50E-19 4.50E-19
"""

COMPACT_TST = """
 Forward TST rate constants:
 T(K) TST
 200.0 1.00E-20
 300.0 2.00E-19
"""


def _result():
    factor = 1e-6 * AVOGADRO_MOL
    return {"rows": [
        {"temperature_k": 200., "k_tst": 1e-20 * factor,
         "k_cvt": 1.5e-20 * factor, "rate_unit": "m^3 mol^-1 s^-1"},
        {"temperature_k": 300., "k_tst": 2e-19 * factor,
         "k_cvt": 2.5e-19 * factor, "rate_unit": "m^3 mol^-1 s^-1"},
    ]}


def test_all_forward_reverse_tables_are_preserved():
    tables = parse_fu15_tables(FU15)
    assert len(tables) == 2
    assert (tables[0].direction, tables[1].direction) == ("forward", "reverse")
    assert tables[0].column("CVT/SCT") == (3e-20, 4e-19)


def test_compact_final_tables_are_preserved_and_unit_aware():
    tables = parse_fu15_tables(COMPACT_FINAL)
    assert len(tables) == 2
    assert tables[0].columns == ("TST", "TST/W", "TST/ZCT")
    assert tables[1].columns == ("CVT", "CVT/W", "CVT/ZCT")
    report = compare_fu15_to_pyrate(tables[0], _result())
    assert report.success
    assert (report.compared, report.passed, report.unmapped) == (2, 2, 4)


def test_compact_tst_without_unit_is_not_compared_by_guessing():
    table = parse_fu15_tables(COMPACT_TST)[0]
    assert table.unit == "unspecified"
    with pytest.raises(ValueError, match="unrecognized rate unit"):
        compare_fu15_to_pyrate(table, _result())


def test_units_and_explicit_method_mapping():
    report = compare_fu15_to_pyrate(parse_fu15_tables(FU15)[0], _result())
    assert report.success
    assert (report.compared, report.passed, report.failed, report.unmapped) == (4, 4, 0, 4)
    assert max(point.relative_error for point in report.points if point.relative_error is not None) < 1e-12
    assert not compare_fu15_to_pyrate(parse_fu15_tables(FU15)[0], _result(), strict=True).success


def test_mismatch_does_not_pass():
    result = _result()
    result["rows"][0]["k_tst"] *= 1.5
    report = compare_fu15_to_pyrate(parse_fu15_tables(FU15)[0], result)
    assert not report.success
    assert report.failed == 1


def test_missing_temperature_never_passes():
    result = _result()
    result["rows"].pop()
    report = compare_fu15_to_pyrate(parse_fu15_tables(FU15)[0], result)
    assert not report.success
    assert report.unavailable == 2


def test_missing_mapped_field_never_passes():
    result = _result()
    result["rows"][1]["k_cvt"] = None
    report = compare_fu15_to_pyrate(parse_fu15_tables(FU15)[0], result)
    assert not report.success
    assert report.unavailable == 1


def test_mismatched_rate_dimensions_rejected():
    with pytest.raises(ValueError, match="first-order and second-order"):
        compare_fu15_to_pyrate(parse_fu15_tables(FU15)[1], _result())


def test_no_comparable_columns_is_not_success():
    report = compare_fu15_to_pyrate(
        parse_fu15_tables(FU15)[0], _result(), column_map={}
    )
    assert report.compared == 0
    assert not report.success


def test_unknown_unit_is_not_silently_assumed():
    result = _result()
    result["rows"][0]["rate_unit"] = "unknown"
    with pytest.raises(ValueError, match="unrecognized rate unit"):
        compare_fu15_to_pyrate(parse_fu15_tables(FU15)[0], result)


def test_nonfinite_value_fails_with_json_serializable_report():
    result = _result()
    result["rows"][0]["k_tst"] = float("inf")
    report = compare_fu15_to_pyrate(parse_fu15_tables(FU15)[0], result)
    assert not report.success
    assert report.failed == 1
    assert '"relative_error": null' in json.dumps(report.as_dict(), allow_nan=False)
