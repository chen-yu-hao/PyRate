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


def test_cl_hbr_leps_curvilinear_mep_fu40_roundtrip_and_fu15_check(tmp_path) -> None:
    import math

    from pyrate.calculation import calculate_rates
    from pyrate.input import legacy_to_calculation
    from pyrate.legacy import parse_legacy_text
    from pyrate.pes import (
        SupermoleculeFragmentPes,
        build_species_from_pes,
        cl_hbr_leps_surface,
        optimize_saddle,
    )

    pes = cl_hbr_leps_surface()
    ts_guess = [[0.0, 0.0, -1.8], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]
    ts_opt = optimize_saddle(pes, ("Cl", "H", "Br"), ts_guess, mass_convention="polyrate")
    assert ts_opt.converged
    assert math.isclose(ts_opt.evaluation.energy_kj_mol / 4.184, 10.9993, abs_tol=1.0e-3)

    ts_freqs = sorted(float(f) for f in ts_opt.normal_mode_result.frequencies_cm1)
    assert math.isclose(ts_freqs[0], -1563.29, abs_tol=0.3)
    assert math.isclose(ts_freqs[1], 305.89, abs_tol=0.3)
    assert math.isclose(ts_freqs[2], 452.31, abs_tol=0.3)
    assert math.isclose(ts_freqs[3], 452.31, abs_tol=0.3)

    hbr_pes = SupermoleculeFragmentPes(pes, ("Cl", "H", "Br"), (1, 2), ts_opt.coords_angstrom)
    sp_hbr, _ = build_species_from_pes(
        hbr_pes,
        "HBr",
        ("H", "Br"),
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.414]],
        mass_convention="polyrate",
        optimize=True,
    )
    assert math.isclose(sp_hbr.positive_frequencies_cm1[0], 2706.94, abs_tol=0.5)

    hcl_pes = SupermoleculeFragmentPes(pes, ("Cl", "H", "Br"), (0, 1), ts_opt.coords_angstrom)
    sp_hcl, _ = build_species_from_pes(
        hcl_pes,
        "HCl",
        ("Cl", "H"),
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.275]],
        mass_convention="polyrate",
        optimize=True,
    )
    assert math.isclose(sp_hcl.electronic_energy_kj_mol / 4.184, -16.160, abs_tol=1.0e-2)
    assert math.isclose(sp_hcl.positive_frequencies_cm1[0], 2988.15, abs_tol=0.5)

    r2_a = sp_hbr.atoms
    p1_a = sp_hcl.atoms
    ts_c = ts_opt.coords_angstrom
    ts_lines = "\n".join(
        f"    {i + 1} {ts_c[i, 0]:.10f} {ts_c[i, 1]:.10f} {ts_c[i, 2]:.10f}"
        for i in range(3)
    )

    def _make_dat(pot: str, coord: str, tunnel: str) -> str:
        return f"""
*GENERAL
  ATOMS
    1 Cl
    2 H
    3 Br
  END
  INPUNIT ANG
*ENERGETICS
  POTENTIAL {pot}
*SECOND
  {coord}
  CENTRAL
*REACT1
  GEOM
    1 0.0 0.0 0.0
  END
  ELEC
    2 0.0
  END
  SPECIES ATOMIC
*REACT2
  GEOM
    2 {r2_a[0].x_angstrom:.10f} {r2_a[0].y_angstrom:.10f} {r2_a[0].z_angstrom:.10f}
    3 {r2_a[1].x_angstrom:.10f} {r2_a[1].y_angstrom:.10f} {r2_a[1].z_angstrom:.10f}
  END
  SPECIES LINRP
*PROD1
  GEOM
    1 {p1_a[0].x_angstrom:.10f} {p1_a[0].y_angstrom:.10f} {p1_a[0].z_angstrom:.10f}
    2 {p1_a[1].x_angstrom:.10f} {p1_a[1].y_angstrom:.10f} {p1_a[1].z_angstrom:.10f}
  END
  SPECIES LINRP
*PROD2
  GEOM
    3 0.0 0.0 0.0
  END
  ELEC
    2 0.0
  END
  SPECIES ATOMIC
*START
  GEOM
{ts_lines}
  END
  ELEC
    2 0.0
  END
  SPECIES LINTS
*PATH
  PM
  {coord}
  SSTEP 0.04
  SRANGE
    -1.0 1.0
  END
*TUNNEL
  {tunnel}
*RATE
  CVT
  TEMP
    200.0 298.15 300.0 500.0 1000.0
  END
"""

    loaded_curv_sct = legacy_to_calculation(
        parse_legacy_text(_make_dat("BUILTIN:cl_hbr_leps", "CURV", "SCT"))
    )
    loaded_curv_zct = legacy_to_calculation(
        parse_legacy_text(_make_dat("BUILTIN:cl_hbr_leps", "CURV", "ZCT"))
    )
    loaded_rect_sct = legacy_to_calculation(
        parse_legacy_text(_make_dat("BUILTIN:cl_hbr_leps", "RECT", "SCT"))
    )

    assert loaded_curv_sct.reaction.path is not None
    assert loaded_rect_sct.reaction.path is not None
    neg_curv = sum(
        1
        for pt in loaded_curv_sct.reaction.path.points
        for f in pt.dividing_surface.frequencies_cm1
        if f < 0.0
    )
    neg_rect = sum(
        1
        for pt in loaded_rect_sct.reaction.path.points
        for f in pt.dividing_surface.frequencies_cm1
        if f < 0.0
    )
    assert neg_curv == 0
    assert neg_rect > 0

    res_sct = calculate_rates(loaded_curv_sct.reaction, loaded_curv_sct.options)
    res_zct = calculate_rates(loaded_curv_zct.reaction, loaded_curv_zct.options)
    for row_sct, row_zct in zip(res_sct.rows, res_zct.rows, strict=True):
        assert row_sct.cvt_coordinate is not None and row_sct.cvt_coordinate < -0.1
        assert row_sct.k_cvt is not None and row_sct.k_cvt < row_sct.k_tst
        assert row_sct.tunneling_correction > row_zct.tunneling_correction > 1.0

    # Roundtrip to .fu40 (both hessian and read modes) and verify rate parity within 1e-6
    for mode in ("hessian", "read"):
        fu40_file = tmp_path / f"cl_hbr_{mode}.fu40"
        loaded_curv_sct.save_fu40(fu40_file, freqsource=mode, hessform="packed")
        reloaded = legacy_to_calculation(
            parse_legacy_text(_make_dat("UNIT40", "CURV", "SCT")),
            fu40_path=fu40_file,
        )
        res_re = calculate_rates(reloaded.reaction, reloaded.options)
        for ra, rb in zip(res_sct.rows, res_re.rows, strict=True):
            assert math.isclose(ra.k_tst, rb.k_tst, rel_tol=1.0e-6)
            assert math.isclose(ra.k_cvt or 0.0, rb.k_cvt or 0.0, rel_tol=1.0e-6)
            assert math.isclose(ra.k_final, rb.k_final, rel_tol=1.0e-6)

    # Verify 3-sig-fig Polyrate .fu15 formatting + unit/column comparator passes all 15 points
    fu15_lines = [
        " Summary of forward rate constants (cm**3/molecule-1 s-1) :",
        "     T(K)      TST       CVT     CVT/SCT",
    ]
    for row in res_sct.rows:
        assert row.k_cvt is not None
        fu15_lines.append(
            f"  {row.temperature_k:7.2f}  {row.k_tst:.2E}  {row.k_cvt:.2E}  {row.k_final:.2E}"
        )
    fu15_table = parse_fu15_tables("\n".join(fu15_lines))[0]
    fu15_report = compare_fu15_to_pyrate(
        fu15_table,
        res_sct.as_dict(),
        column_map={"TST": "k_tst", "CVT": "k_cvt", "CVT/SCT": "k_final"},
        relative_tolerance=5.0e-3,
        strict=True,
    )
    assert fu15_report.success
    assert fu15_report.passed == 15
    assert fu15_report.failed == 0


def test_h3_and_dh2_leps_kie_and_native_fallback_parity(monkeypatch) -> None:
    import math

    import pyrate.native as native_mod
    from pyrate.calculation import calculate_rates
    from pyrate.input import parse_toml_data
    from pyrate.pes import (
        SupermoleculeFragmentPes,
        build_species_from_pes,
        h3_leps_surface,
        optimize_saddle,
    )

    pes = h3_leps_surface()
    ts_guess = [[0.0, 0.0, -0.95], [0.0, 0.0, 0.0], [0.0, 0.0, 0.95]]
    ts_opt = optimize_saddle(pes, ("H", "H", "H"), ts_guess, mass_convention="polyrate")
    assert ts_opt.converged

    h2_pes = SupermoleculeFragmentPes(pes, ("H", "H", "H"), (1, 2), ts_opt.coords_angstrom)
    sp_h2, _ = build_species_from_pes(
        h2_pes,
        "H2",
        ("H", "H"),
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        mass_convention="polyrate",
        optimize=True,
    )
    r_bc = sp_h2.atoms[1].z_angstrom - sp_h2.atoms[0].z_angstrom

    def _run_isotopologue(syms: tuple[str, str, str], r_sym: int, sct_tunnel: str = "sct"):
        ts_xyz = ts_opt.coords_angstrom
        # Polyrate SIGMAF convention: H2 reactant carries symmetry_number=2 and TS uses 1
        # (matching PathPoint symmetry_number=1 along the MEP).
        toml_doc = {
            "calculation": {
                "name": "-".join(syms),
                "temperatures_k": [200.0, 300.0, 600.0],
                "method": "cvt",
                "tunneling": sct_tunnel,
                "rate_unit": "cm^3 molecule^-1 s^-1",
                "mass_convention": "polyrate",
            },
            "reactants": [
                {
                    "name": f"{syms[0]}_atom",
                    "atoms": [{"symbol": syms[0], "xyz": [0.0, 0.0, 0.0]}],
                    "electronic_states": [{"degeneracy": 2, "energy_kj_mol": 0.0}],
                },
                {
                    "name": f"{syms[1]}{syms[2]}",
                    "rotor_type": "linear",
                    "symmetry_number": r_sym,
                    "atoms": [
                        {"symbol": syms[1], "xyz": [0.0, 0.0, 0.0]},
                        {"symbol": syms[2], "xyz": [0.0, 0.0, float(r_bc)]},
                    ],
                },
            ],
            "transition_state": {
                "name": "TS",
                "rotor_type": "linear",
                "symmetry_number": 1,
                "electronic_states": [{"degeneracy": 2, "energy_kj_mol": 0.0}],
                "atoms": [
                    {"symbol": syms[0], "xyz": ts_xyz[0].tolist()},
                    {"symbol": syms[1], "xyz": ts_xyz[1].tolist()},
                    {"symbol": syms[2], "xyz": ts_xyz[2].tolist()},
                ],
            },
            "mep": {
                "method": "pm",
                "step_size_bohr": 0.04,
                "s_min_bohr": -0.8,
                "s_max_bohr": 0.8,
                "coordinate_system": "curvilinear",
            },
        }
        loaded = parse_toml_data(toml_doc, pes=pes)
        return loaded, calculate_rates(loaded.reaction, loaded.options, name=loaded.name)

    loaded_hhh_sct, res_hhh_sct = _run_isotopologue(("H", "H", "H"), 2, "sct")
    _, res_hhh_zct = _run_isotopologue(("H", "H", "H"), 2, "zct")
    loaded_dhh, res_dhh = _run_isotopologue(("D", "H", "H"), 2, "sct")
    loaded_2hhh, res_2hhh = _run_isotopologue(("2H", "H", "H"), 2, "sct")
    _, res_hdd = _run_isotopologue(("H", "D", "D"), 2, "sct")

    for rs, rz in zip(res_hhh_sct.rows, res_hhh_zct.rows, strict=True):
        assert rs.cvt_coordinate is not None and abs(rs.cvt_coordinate) < 1.0e-4
        assert rs.k_cvt is not None and math.isclose(rs.k_cvt, rs.k_tst, rel_tol=1.0e-4)
        assert rs.tunneling_correction > rz.tunneling_correction > 1.0

    # Isotope aliases "2H" and "D" produce identical TS modes and rate constants
    assert (
        loaded_2hhh.reaction.transition_state.frequencies_cm1
        == loaded_dhh.reaction.transition_state.frequencies_cm1
    )
    for r2h, rd in zip(res_2hhh.rows, res_dhh.rows, strict=True):
        assert math.isclose(r2h.k_final, rd.k_final, rel_tol=1.0e-12)

    # Heavy substitution lowers TS frequencies / tunneling; primary H+H2 / H+D2 KIE > 10 at 300 K
    assert abs(loaded_dhh.reaction.transition_state.imaginary_frequencies_cm1[0]) < abs(
        loaded_hhh_sct.reaction.transition_state.imaginary_frequencies_cm1[0]
    )
    assert res_dhh.rows[1].tunneling_correction < res_hhh_sct.rows[1].tunneling_correction
    assert res_hhh_sct.rows[1].k_final / res_hdd.rows[1].k_final > 10.0

    # End-to-end native C++ vs pure Python fallback parity
    if native_mod.HAS_NATIVE:
        monkeypatch.setattr(native_mod, "HAS_NATIVE", False)
        _, res_hhh_py = _run_isotopologue(("H", "H", "H"), 2, "sct")
        for rn, rp in zip(res_hhh_sct.rows, res_hhh_py.rows, strict=True):
            assert math.isclose(rn.k_tst, rp.k_tst, rel_tol=1.0e-8)
            assert math.isclose(rn.k_cvt or 0.0, rp.k_cvt or 0.0, rel_tol=1.0e-8)
            assert math.isclose(rn.tunneling_correction, rp.tunneling_correction, rel_tol=1.0e-8)
            assert math.isclose(rn.k_final, rp.k_final, rel_tol=1.0e-8)


