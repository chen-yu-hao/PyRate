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


def test_parse_all_three_polyrate_2023_fu15_formats_and_subtables():
    text = """
 Forward TST rate constants:

      T(K)           TST

     298.00        1.67E-21
     400.00        4.11E-19

 FINAL RATE CONSTANTS IVTST-0 (cm**3/molecule-sec)

   T(K)      TST      CVT     CVT/ZCT  CVT/SCT
  300.0   1.00E-15  9.00E-16  2.50E-15 4.20E-15

   T(K)    CVT/COMT  CVT/mOMT
  300.0   4.20E-15  4.35E-15
"""
    tables = parse_fu15_tables(text)
    assert len(tables) == 2
    assert tables[0].columns == ("TST",)
    assert len(tables[0].rows) == 2
    assert tables[1].columns == ("TST", "CVT", "CVT/ZCT", "CVT/SCT", "CVT/COMT", "CVT/mOMT")
    assert tables[1].column("CVT/mOMT") == (4.35e-15,)


def test_polyrate_2023_ch5fu40tr4_end_to_end_head_to_head():
    from pyrate.calculation import calculate_rates
    from pyrate.input import legacy_to_calculation
    from pyrate.legacy import parse_legacy_text

    dat_text = """
*General
  TITLE
   CH3 + H-H -> CH4 + H Conventional TST - Data input from Unit 40
  END
  ATOMS
    1 C
    2 H
    3 H
    4 H
    5 H
    6 H
  END
  INPUNIT AU
*ENERGETICS
  POTENTIAL unit40
*REACT1
  GEOM
   1    0.000000   0.000000   0.000000
   2    0.000000   0.000000   2.046574
   3    1.772385   0.000000  -1.023287
   4   -1.772385   0.000000  -1.023287
  END
  ELEC
    2   0.0
  END
*REACT2
  GEOM
   5    0.000000  30.000000   0.000000
   6    0.000000  31.404067   0.000000
  END
  ELEC
    1   0.0
  END
  SPECIES LINRP
*PROD1
  GEOM
   1    0.000000   0.000000   0.000000
   2    0.000000   0.000000   2.065471
   3    1.947372   0.000000  -0.688415
   4   -0.973686  -1.686473  -0.688415
   5   -0.973686   1.686473  -0.688415
  END
  ELEC
    1   0.0
  END
*PROD2
  GEOM
   6  -15.102738  26.158701 -10.677688
  END
  ELEC
    2   0.0
  END
  SPECIES ATOMIC
*START
  GEOM
   1    0.000000   0.000000   0.000000
   2    0.000000   0.000000   2.057913
   3    1.871577   0.000000  -0.855689
   4   -0.935788  -1.620833  -0.855689
   5   -1.275994   2.210086  -0.622108
   6   -2.092979   3.625178  -1.020435
  END
  ELEC
    2   0.0
  END
  SPECIES NONLINTS
*RATE
  BOTHK
  SIGMAF 4
  SIGMAR 4
  TST
  NOCVT
  TEMP
    298. 300. 372. 400. 424. 500. 600. 667. 1000. 1340. 1500. 1550.
  END
"""
    fu40_text = """
*gen40
  geomunit bohr
  freqsource read
  frequnit waven
*r140
  vvalue 0.0
  freq
  3309.974  3309.974  3127.989  1435.989  1435.989  431.997
  end
*r240
  vvalue 0.0
  freq
  4421.966
  end
*p140
  vvalue -0.003984
  freq
  3166.975  3166.975  3166.975  3046.975  1572.988  1572.988
  1366.989  1366.989  1366.989
  end
*p240
  vvalue 0.0
*saddle40
  vvalue 0.018804
  freq
  3235.976  3235.976  3089.977  1763.986  1458.989  1458.550
  1151.991  1151.991  1110.991   533.996   533.996
  end
  wstar 1528.988
"""
    fu15_text = """
 Forward TST rate constants:

      T(K)           TST

     298.00        1.67E-21
     300.00        1.93E-21
     372.00        1.23E-19
     400.00        4.11E-19
     424.00        1.02E-18
     500.00        1.03E-17
     600.00        8.97E-17
     667.00        2.71E-16
    1000.00        8.62E-15
    1340.00        6.12E-14
    1500.00        1.20E-13
    1550.00        1.44E-13
"""
    loaded = legacy_to_calculation(parse_legacy_text(dat_text), fu40_text=fu40_text)
    result = calculate_rates(loaded.reaction, loaded.options, name=loaded.name)
    reference = parse_fu15_tables(fu15_text)[0]
    report = compare_fu15_to_pyrate(reference, result.as_dict(), relative_tolerance=5.0e-3)
    assert report.success
    assert report.passed == 12
    assert report.failed == 0

