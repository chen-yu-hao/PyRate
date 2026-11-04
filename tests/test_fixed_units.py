from __future__ import annotations

import pytest

from pyrate.constants import HARTREE_KJ_MOL, WAVENUMBER_TO_KJ_MOL
from pyrate.fixed_units import parse_fixed_unit_text, summarize_fixed_unit


UNIT40_FREQUENCIES = """
*gen40
  title
    frequency example
  end
  freqsource read
  frequnit au
*r140
  vvalue 0.0
  freq
    1.0D-02 2.0D-02
  end
  ifreq
    1 2
  end
*p140
  vvalue -1.0D-03
*saddle40
  vvalue 2.0D-02
  freq
    5.0D-03
  end
  ifreq
    1
  end
  wstar 7.0D-03
"""


UNIT31_HESSIANS = """
*GENERAL
*REACT1
  HESSIAN
    1.0
    0.1 2.0
  END
*PROD2
  ENERXN -1.0D-01
*SADDLE
  ENESAD 2.5D-02
  HESSIAN
    3.0
  END
"""


UNIT30_PATH = """unit 30 example
2 3 0 -2 0 1 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
IOP = 1, S, V=
0.0 0.0
1
F MATRIX (PACKED)=
1.0 0.0 2.0
SADDLE POINT, V=
0.0 0.02
1 2
F MATRIX (PACKED)=
3.0 0.0 4.0
BEGIN POINTS ALONG REACTION COORDINATE
1
S,V=
-0.1 0.015
X=
0.0 0.0 0.0
1.0 0.0 0.0
DX=
0.1 0.0 0.0
-0.1 0.0 0.0
1 2
F MATRIX (PACKED)=
5.0 0.0 6.0
"""


def test_unit40_frequency_data_convert_to_common_units() -> None:
    document = parse_fixed_unit_text(UNIT40_FREQUENCIES, 40, source_name="example.fu40")
    points = {point.role: point for point in document.stationary_points()}

    assert document.title == "frequency example"
    assert document.frequency_unit == "au"
    assert points["reactant1"].active_mode_indices == (1, 2)
    factor = HARTREE_KJ_MOL / WAVENUMBER_TO_KJ_MOL
    assert points["reactant1"].frequencies_cm1 == pytest.approx((0.01 * factor, 0.02 * factor))
    assert points["product1"].potential_kj_mol == pytest.approx(-0.001 * HARTREE_KJ_MOL)
    assert points["saddle"].imaginary_frequency_cm1 == pytest.approx(0.007 * factor)


def test_unit31_retains_hessians_and_relative_energies() -> None:
    document = parse_fixed_unit_text(UNIT31_HESSIANS, 31, source_name="example.fu31")
    points = {point.role: point for point in document.stationary_points()}

    assert document.hessian_form == "packed"
    assert points["reactant1"].hessian_values == (1.0, 0.1, 2.0)
    assert points["product2"].potential_hartree == pytest.approx(-0.1)
    assert points["saddle"].potential_hartree == pytest.approx(0.025)


def test_unit30_parses_stationary_and_reaction_path_records() -> None:
    document = parse_fixed_unit_text(UNIT30_PATH, 30, source_name="example.fu30")
    points = {point.role: point for point in document.stationary_points()}
    path = document.path_points()

    assert document.header_flags == (2, 3, 0, -2, 0, 1, 0, 0)
    assert points["reactant1"].hessian_values == (1.0, 0.0, 2.0)
    assert points["saddle"].potential_hartree == pytest.approx(0.02)
    assert len(path) == 1
    assert path[0].coordinate_au == pytest.approx(-0.1)
    assert path[0].geometry_rows == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert path[0].gradient_rows == ((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0))
    assert path[0].active_mode_indices == (1, 2)
    assert path[0].hessian_values == (5.0, 0.0, 6.0)


def test_unknown_keyword_is_retained_in_keyword_format() -> None:
    document = parse_fixed_unit_text("*gen40\n futureoption enabled 7\n", 40)
    entry = document.sections[0].first("futureoption")
    assert entry is not None
    assert entry.values == ("enabled", 7)


def test_summary_separates_stationary_and_path_coverage() -> None:
    summary = summarize_fixed_unit(parse_fixed_unit_text(UNIT30_PATH, 30))
    assert summary["unit"] == 30
    assert summary["stationary_count"] == 2
    assert summary["path_point_count"] == 1


def test_unit30_rejects_incomplete_header() -> None:
    with pytest.raises(ValueError, match="8 header flags"):
        parse_fixed_unit_text("title\n1 2\n" + "0 " * 32, 30)


def test_nonfinite_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        parse_fixed_unit_text("*gen40\n maxlpts NaN\n", 40)
