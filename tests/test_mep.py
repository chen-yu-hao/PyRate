from __future__ import annotations

import math

import numpy as np

from pyrate.calculation import calculate_rates
from pyrate.input import legacy_to_calculation, parse_toml_data
from pyrate.legacy import parse_legacy_text
from pyrate.mep import curvilinear_corrected_hessian, integrate_mep
from pyrate.models import Atom, CalculationOptions, MepOptions, Species
from pyrate.pes import (
    EckartValleySurface,
    MullerBrownSurface,
    SupermoleculeFragmentPes,
    build_species_from_pes,
    cl_hbr_leps_surface,
    h3_leps_surface,
    optimize_saddle,
)


def test_mep_pm_outperforms_coarse_euler_vs_fine_reference() -> None:
    pes = h3_leps_surface()
    guess = np.array([[0.0, 0.0, -0.95], [0.0, 0.0, 0.0], [0.0, 0.0, 0.95]], dtype=float)
    ts_opt = optimize_saddle(pes, ("H", "H", "H"), guess, mass_convention="polyrate")

    fine = integrate_mep(
        pes,
        ("H", "H", "H"),
        ts_opt.coords_angstrom,
        options=MepOptions(
            method="euler",
            step_size_bohr=0.002,
            hessian_stride=10,
            s_min_bohr=-0.6,
            s_max_bohr=0.6,
            mass_convention="polyrate",
        ),
    )
    pm = integrate_mep(
        pes,
        ("H", "H", "H"),
        ts_opt.coords_angstrom,
        options=MepOptions(
            method="pm",
            step_size_bohr=0.04,
            hessian_stride=1,
            s_min_bohr=-0.6,
            s_max_bohr=0.6,
            mass_convention="polyrate",
        ),
    )
    es1 = integrate_mep(
        pes,
        ("H", "H", "H"),
        ts_opt.coords_angstrom,
        options=MepOptions(
            method="es1",
            step_size_bohr=0.04,
            hessian_stride=1,
            s_min_bohr=-0.6,
            s_max_bohr=0.6,
            mass_convention="polyrate",
            stabilization_angle_deg=2.0,
        ),
    )
    euler_coarse = integrate_mep(
        pes,
        ("H", "H", "H"),
        ts_opt.coords_angstrom,
        options=MepOptions(
            method="euler",
            step_size_bohr=0.04,
            hessian_stride=1,
            s_min_bohr=-0.6,
            s_max_bohr=0.6,
            mass_convention="polyrate",
        ),
    )

    e_fine = fine.points[-1].energy_hartree
    e_pm = pm.points[-1].energy_hartree
    e_es1 = es1.points[-1].energy_hartree
    e_euler = euler_coarse.points[-1].energy_hartree

    assert abs(e_pm - e_fine) <= abs(e_euler - e_fine) + 1.0e-8
    assert abs(e_es1 - e_fine) <= abs(e_euler - e_fine) + 1.0e-8
    # Endpoints near s=+-0.6 should agree tightly for PM
    assert abs(e_pm - e_fine) < 1.0e-4


def test_mep_inh_stride_reduces_hessian_evaluations() -> None:
    pes = h3_leps_surface()
    guess = np.array([[0.0, 0.0, -0.95], [0.0, 0.0, 0.0], [0.0, 0.0, 0.95]], dtype=float)
    ts_opt = optimize_saddle(pes, ("H", "H", "H"), guess, mass_convention="polyrate")

    mep_dense = integrate_mep(
        pes,
        ("H", "H", "H"),
        ts_opt.coords_angstrom,
        options=MepOptions(
            method="pm",
            step_size_bohr=0.05,
            hessian_stride=1,
            s_min_bohr=-0.5,
            s_max_bohr=0.5,
            mass_convention="polyrate",
        ),
    )
    mep_strided = integrate_mep(
        pes,
        ("H", "H", "H"),
        ts_opt.coords_angstrom,
        options=MepOptions(
            method="pm",
            step_size_bohr=0.05,
            hessian_stride=2,
            save_stride=2,
            s_min_bohr=-0.5,
            s_max_bohr=0.5,
            mass_convention="polyrate",
        ),
    )
    assert mep_strided.hessian_evaluations < mep_dense.hessian_evaluations
    assert len(mep_strided.points) < len(mep_dense.points)
    assert abs(mep_strided.points[0].energy_hartree - mep_dense.points[0].energy_hartree) < 1.0e-4


def test_mep_nosaddle_both_branches_and_direction_orientation() -> None:
    pes = h3_leps_surface()
    guess = np.array([[0.0, 0.0, -0.95], [0.0, 0.0, 0.0], [0.0, 0.0, 0.95]], dtype=float)
    ts_opt = optimize_saddle(pes, ("H", "H", "H"), guess, mass_convention="polyrate")

    # Displace toward products along TS eigenvector and descend with start_from_saddle=False
    disp = np.array([[0.0, 0.0, 0.04], [0.0, 0.0, -0.04], [0.0, 0.0, 0.0]])
    pos_start = ts_opt.coords_angstrom + disp
    mep_pos = integrate_mep(
        pes,
        ("H", "H", "H"),
        pos_start,
        options=MepOptions(
            method="pm",
            step_size_bohr=0.05,
            s_min_bohr=0.0,
            s_max_bohr=0.4,
            start_from_saddle=False,
            mass_convention="polyrate",
        ),
    )
    assert mep_pos.points[0].coordinate_bohr == 0.0
    assert mep_pos.points[-1].coordinate_bohr > 0.3
    energies_pos = [pt.energy_hartree for pt in mep_pos.points]
    assert all(b <= a + 1.0e-9 for a, b in zip(energies_pos, energies_pos[1:], strict=False))

    mep_neg = integrate_mep(
        pes,
        ("H", "H", "H"),
        pos_start,
        options=MepOptions(
            method="pm",
            step_size_bohr=0.05,
            s_min_bohr=-0.4,
            s_max_bohr=0.0,
            start_from_saddle=False,
            mass_convention="polyrate",
        ),
    )
    assert mep_neg.points[0].coordinate_bohr < -0.3
    assert mep_neg.points[-1].coordinate_bohr == 0.0
    # s goes from -0.4 to 0.0, so energy rises back toward the s=0 starting point
    energies_neg = [pt.energy_hartree for pt in mep_neg.points]
    assert all(b >= a - 1.0e-9 for a, b in zip(energies_neg, energies_neg[1:], strict=False))

    # direction_sign flip mirrors the branch assignment at a saddle
    mep_fwd = integrate_mep(
        pes,
        ("H", "H", "H"),
        ts_opt.coords_angstrom,
        options=MepOptions(
            method="pm", step_size_bohr=0.05, s_min_bohr=-0.3, s_max_bohr=0.3, direction_sign=1
        ),
    )
    mep_rev = integrate_mep(
        pes,
        ("H", "H", "H"),
        ts_opt.coords_angstrom,
        options=MepOptions(
            method="pm", step_size_bohr=0.05, s_min_bohr=-0.3, s_max_bohr=0.3, direction_sign=-1
        ),
    )
    assert np.allclose(
        mep_fwd.points[0].coords_angstrom, mep_rev.points[-1].coords_angstrom, atol=1.0e-5
    )


def test_mep_se3_rotation_translation_invariance_on_muller_brown() -> None:
    mb = MullerBrownSurface()
    ts_guess = np.array([[0.0, 0.0, 0.0], [1.68, 0.0, 0.0], [1.68, 2.12, 0.0]], dtype=float)
    ts_opt = optimize_saddle(mb, ("C", "O", "N"), ts_guess)

    mep_orig = integrate_mep(
        mb,
        ("C", "O", "N"),
        ts_opt.coords_angstrom,
        options=MepOptions(method="pm", step_size_bohr=0.03, s_min_bohr=-0.3, s_max_bohr=0.3),
    )

    rot_seed = np.array([[0.8, -0.4, 0.3], [0.5, 0.9, -0.2], [-0.3, 0.2, 1.1]], dtype=float)
    q_rot, _ = np.linalg.qr(rot_seed)
    shift = np.array([2.5, -1.7, 0.6])
    ts_rot = ts_opt.coords_angstrom @ q_rot.T + shift

    mep_rot = integrate_mep(
        mb,
        ("C", "O", "N"),
        ts_rot,
        options=MepOptions(
            method="pm", step_size_bohr=0.03, s_min_bohr=-0.3, s_max_bohr=0.3
        ),
    )

    e_orig = np.array([pt.energy_hartree for pt in mep_orig.points])
    # Flip if arbitrary eigenvector sign chose opposite orientation
    e_rot = np.array([pt.energy_hartree for pt in mep_rot.points])
    if abs(e_orig[0] - e_rot[-1]) < abs(e_orig[0] - e_rot[0]):
        pts_rot = list(reversed(mep_rot.points))
    else:
        pts_rot = list(mep_rot.points)

    for p1, p2 in zip(mep_orig.points, pts_rot, strict=True):
        assert math.isclose(p1.energy_hartree, p2.energy_hartree, rel_tol=1.0e-7, abs_tol=1.0e-7)
        assert np.allclose(p1.frequencies_cm1, p2.frequencies_cm1, atol=5.0e-2)
        b1 = np.sort(np.abs(p1.curvatures_bohr_inv))
        b2 = np.sort(np.abs(p2.curvatures_bohr_inv))
        assert np.allclose(b1, b2, atol=5.0e-4)


def test_eckart_valley_uncoupled_vs_coupled_sct_physics() -> None:
    # Uncoupled symmetric-mass collinear Eckart valley -> zero path curvature -> SCT == ZCT
    r_eq = 1.2
    ts_coords = np.array([[0.0, 0.0, -r_eq], [0.0, 0.0, 0.0], [0.0, 0.0, r_eq]], dtype=float)
    r_sp = Species(
        name="R",
        atoms=(Atom("H", 0.0, 0.0, -r_eq), Atom("O", 0.0, 0.0, 0.0), Atom("H", 0.0, 0.0, r_eq)),
        electronic_energy_kj_mol=0.0,
        frequencies_cm1=(1000.0, 500.0, 500.0),
    )

    pes_uncoupled = EckartValleySurface(
        v1_kj_mol=40.0, v2_kj_mol=40.0, width_angstrom=0.35, coupling_angstrom=0.0
    )
    mep_unc = integrate_mep(
        pes_uncoupled,
        ("H", "O", "H"),
        ts_coords,
        options=MepOptions(method="pm", step_size_bohr=0.04, s_min_bohr=-1.2, s_max_bohr=1.2),
        reactants=(r_sp,),
    )
    max_curv_unc = max(abs(b) for pt in mep_unc.points for b in pt.curvatures_bohr_inv)
    assert max_curv_unc < 1.0e-5

    rxn_unc = mep_unc.build_reaction(reactants=(r_sp,))
    res_unc_zct = calculate_rates(
        rxn_unc, CalculationOptions(temperatures_k=(250.0, 400.0), method="tst", tunneling="zct")
    )
    res_unc_sct = calculate_rates(
        rxn_unc, CalculationOptions(temperatures_k=(250.0, 400.0), method="tst", tunneling="sct")
    )
    for rz, rs in zip(res_unc_zct.rows, res_unc_sct.rows, strict=True):
        assert math.isclose(
            rs.tunneling_correction, rz.tunneling_correction, rel_tol=1.0e-4
        )

    # Coupled Eckart valley introduces reaction-path curvature => kappa_SCT > kappa_ZCT
    pes_coupled = EckartValleySurface(
        v1_kj_mol=40.0, v2_kj_mol=40.0, width_angstrom=0.35, coupling_angstrom=0.08
    )
    ts_cpl = optimize_saddle(pes_coupled, ("H", "O", "H"), ts_coords).coords_angstrom
    mep_cpl = integrate_mep(
        pes_coupled,
        ("H", "O", "H"),
        ts_cpl,
        options=MepOptions(method="pm", step_size_bohr=0.04, s_min_bohr=-1.2, s_max_bohr=1.2),
        reactants=(r_sp,),
    )
    max_curv_cpl = max(abs(b) for pt in mep_cpl.points for b in pt.curvatures_bohr_inv)
    assert max_curv_cpl > 0.1

    rxn_cpl = mep_cpl.build_reaction(reactants=(r_sp,))
    res_cpl_zct = calculate_rates(
        rxn_cpl, CalculationOptions(temperatures_k=(250.0, 400.0), method="tst", tunneling="zct")
    )
    res_cpl_sct = calculate_rates(
        rxn_cpl, CalculationOptions(temperatures_k=(250.0, 400.0), method="tst", tunneling="sct")
    )
    assert res_cpl_sct.rows[0].tunneling_correction > res_cpl_zct.rows[0].tunneling_correction


def test_curvilinear_corrected_hessian_eliminates_spurious_bends_and_handles_4atom() -> None:
    # 1) Cl + HBr rectilinear vs curvilinear along MEP
    pes = cl_hbr_leps_surface()
    guess = np.array([[0.0, 0.0, -1.8], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=float)
    ts_opt = optimize_saddle(pes, ("Cl", "H", "Br"), guess, mass_convention="polyrate")

    mep_curv = integrate_mep(
        pes,
        ("Cl", "H", "Br"),
        ts_opt.coords_angstrom,
        options=MepOptions(
            method="pm",
            step_size_bohr=0.05,
            s_min_bohr=-0.8,
            s_max_bohr=0.8,
            coordinate_system="curvilinear",
            mass_convention="polyrate",
        ),
    )
    mep_rect = integrate_mep(
        pes,
        ("Cl", "H", "Br"),
        ts_opt.coords_angstrom,
        options=MepOptions(
            method="pm",
            step_size_bohr=0.05,
            s_min_bohr=-0.8,
            s_max_bohr=0.8,
            coordinate_system="rectilinear",
            mass_convention="polyrate",
        ),
    )
    neg_curv = sum(1 for pt in mep_curv.points for f in pt.frequencies_cm1 if f < 0.0)
    neg_rect = sum(1 for pt in mep_rect.points for f in pt.frequencies_cm1 if f < 0.0)
    assert neg_curv == 0
    assert neg_rect > 0

    # 2) Non-linear 4-atom potential purely quadratic in internal (r12, r23, r34, angles, dihedral):
    # off-minimum internal gradient induces Cartesian off-diagonal curvature unless corrected!
    coords_bohr = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.8, 1.9, 0.0], [3.5, 2.2, 1.7]], dtype=float
    )
    masses = [12.0, 14.0, 16.0, 1.0]
    p12 = coords_bohr[0] - coords_bohr[1]
    r12 = float(np.linalg.norm(p12))
    u12 = p12 / r12
    grad = np.zeros(12, dtype=float)
    grad[0:3] = 0.05 * u12
    grad[3:6] = -0.05 * u12
    block = (np.eye(3) - np.outer(u12, u12)) / r12
    # Pure curvilinear stretch-only Hessian in Cartesian coords has gradient*d^2r/dx^2 term
    hess = np.zeros((12, 12), dtype=float)
    hess[0:3, 0:3] = block * 0.05
    hess[3:6, 3:6] = block * 0.05
    hess[0:3, 3:6] = -block * 0.05
    hess[3:6, 0:3] = -block * 0.05

    h_corr = curvilinear_corrected_hessian(coords_bohr, grad, hess, masses)
    # Subtracting g_q * (d^2 r12 / dx^2) removes the artificial non-bond force-constant residue
    assert np.allclose(h_corr, 0.0, atol=1.0e-8)


def test_fu40_roundtrip_and_two_fragment_relative_energies_false(tmp_path) -> None:
    pes = cl_hbr_leps_surface()
    ts_opt = optimize_saddle(
        pes,
        ("Cl", "H", "Br"),
        [[0.0, 0.0, -1.8], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
        mass_convention="polyrate",
    )
    hbr_pes = SupermoleculeFragmentPes(pes, ("Cl", "H", "Br"), (1, 2), ts_opt.coords_angstrom)
    sp_hbr, _ = build_species_from_pes(
        hbr_pes, "HBr", ("H", "Br"), [[0.0, 0.0, 0.0], [0.0, 0.0, 1.414]],
        mass_convention="polyrate", optimize=True,
    )
    hcl_pes = SupermoleculeFragmentPes(pes, ("Cl", "H", "Br"), (0, 1), ts_opt.coords_angstrom)
    sp_hcl, _ = build_species_from_pes(
        hcl_pes, "HCl", ("Cl", "H"), [[0.0, 0.0, 0.0], [0.0, 0.0, 1.275]],
        mass_convention="polyrate", optimize=True,
    )
    p1_atoms = sp_hcl.atoms
    r2_atoms = sp_hbr.atoms

    ts_c = ts_opt.coords_angstrom
    ts_lines = "\n".join(
        f"    {i + 1} {ts_c[i, 0]:.10f} {ts_c[i, 1]:.10f} {ts_c[i, 2]:.10f}" for i in range(3)
    )
    dat_tmpl = f"""
*GENERAL
  ATOMS
    1 Cl
    2 H
    3 Br
  END
  INPUNIT ANG
*ENERGETICS
  POTENTIAL __POT__
*SECOND
  CURV
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
    2 {r2_atoms[0].x_angstrom:.10f} {r2_atoms[0].y_angstrom:.10f} {r2_atoms[0].z_angstrom:.10f}
    3 {r2_atoms[1].x_angstrom:.10f} {r2_atoms[1].y_angstrom:.10f} {r2_atoms[1].z_angstrom:.10f}
  END
  SPECIES LINRP
*PROD1
  GEOM
    1 {p1_atoms[0].x_angstrom:.10f} {p1_atoms[0].y_angstrom:.10f} {p1_atoms[0].z_angstrom:.10f}
    2 {p1_atoms[1].x_angstrom:.10f} {p1_atoms[1].y_angstrom:.10f} {p1_atoms[1].z_angstrom:.10f}
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
  CURV
  SSTEP 0.04
  SRANGE
    -0.8 0.8
  END
*TUNNEL
  SCT
*RATE
  CVT
  TEMP
    200.0 300.0
  END
"""
    loaded_direct = legacy_to_calculation(
        parse_legacy_text(dat_tmpl.replace("__POT__", "BUILTIN:cl_hbr_leps"))
    )
    res_direct = calculate_rates(loaded_direct.reaction, loaded_direct.options)

    for src_mode, hform in (("hessian", "packed"), ("hessian", "full"), ("read", "packed")):
        fu40_path = tmp_path / f"clhbr_{src_mode}_{hform}.fu40"
        loaded_direct.save_fu40(fu40_path, freqsource=src_mode, hessform=hform)
        loaded_re = legacy_to_calculation(
            parse_legacy_text(dat_tmpl.replace("__POT__", "UNIT40")), fu40_path=fu40_path
        )
        res_re = calculate_rates(loaded_re.reaction, loaded_re.options)
        for ra, rb in zip(res_direct.rows, res_re.rows, strict=True):
            assert math.isclose(ra.k_final, rb.k_final, rel_tol=1.0e-6)
            assert math.isclose(ra.k_cvt or 0.0, rb.k_cvt or 0.0, rel_tol=1.0e-7)

    # Check relative_energies=False and two polyatomic fragments preserve non-zero asymptote & shape
    unshifted_pes = cl_hbr_leps_surface(zero_at_reactants=False)
    toml_data = {
        "calculation": {
            "name": "unshifted-ref",
            "temperatures_k": [300.0],
            "method": "tst",
            "tunneling": "none",
            "relative_energies": False,
        },
        "reactants": [
            {
                "name": "HBr_far",
                "atoms": [
                    {"symbol": "H", "xyz": [0.0, 0.0, 0.0]},
                    {"symbol": "Br", "xyz": [0.0, 0.0, 1.414]},
                ],
                "fragment_indices": [1, 2],
            },
            {
                "name": "Cl_atom",
                "atoms": [{"symbol": "Cl", "xyz": [0.0, 0.0, 0.0]}],
                "fragment_indices": [0],
            },
        ],
        "transition_state": {
            "name": "TS",
            "atoms": [
                {"symbol": "Cl", "xyz": ts_opt.coords_angstrom[0].tolist()},
                {"symbol": "H", "xyz": ts_opt.coords_angstrom[1].tolist()},
                {"symbol": "Br", "xyz": ts_opt.coords_angstrom[2].tolist()},
            ],
        },
        "mep": {"method": "pm", "step_size_bohr": 0.1, "s_min_bohr": -0.2, "s_max_bohr": 0.2},
    }
    loaded_unshifted = parse_toml_data(toml_data, pes=unshifted_pes)
    assert loaded_unshifted.reactant_evaluations[0] is not None
    assert loaded_unshifted.reactant_evaluations[0].hessian_kj_mol_angstrom2 is not None
    assert loaded_unshifted.reactant_evaluations[0].hessian_kj_mol_angstrom2.shape == (6, 6)
    assert loaded_unshifted.reactant_evaluations[0].energy_kj_mol < -100.0
    fu40_unshifted = loaded_unshifted.to_fu40_text()
    assert "*R140" in fu40_unshifted and "*R240" in fu40_unshifted
