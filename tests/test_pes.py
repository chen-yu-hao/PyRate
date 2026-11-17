from __future__ import annotations

import math

import numpy as np
import pytest

from pyrate.constants import BOHR_ANGSTROM, HARTREE_KJ_MOL
from pyrate.pes import (
    CallablePes,
    EckartValleySurface,
    ExtendedLepsSurface,
    MullerBrownSurface,
    PolyratePotAdapter,
    SupermoleculeFragmentPes,
    build_pes_from_mapping,
    build_species_from_pes,
    cl_hbr_leps_surface,
    get_builtin_pes,
    h3_leps_surface,
    optimize_minimum,
    optimize_saddle,
)


def _harmonic_diatomic_energy(syms: tuple[str, ...], coords_ang: np.ndarray) -> float:
    del syms
    k = 1200.0  # kJ mol^-1 A^-2
    r_eq = 1.1
    r = float(np.linalg.norm(coords_ang[1] - coords_ang[0]))
    return 0.5 * k * (r - r_eq) ** 2


def _harmonic_diatomic_grad(syms: tuple[str, ...], coords_ang: np.ndarray) -> np.ndarray:
    del syms
    k = 1200.0
    r_eq = 1.1
    d = coords_ang[1] - coords_ang[0]
    r = float(np.linalg.norm(d))
    u = d / r
    f = k * (r - r_eq) * u
    return np.asarray([-f, f], dtype=float)


def test_callable_pes_numerical_derivatives_and_hessian_modes() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]], dtype=float)
    energy_only = CallablePes(_harmonic_diatomic_energy, step_bohr=1.0e-4)
    with_grad = CallablePes(
        _harmonic_diatomic_energy, gradient_fn=_harmonic_diatomic_grad, step_bohr=1.0e-4
    )

    ev_e = energy_only.evaluate(
        ("H", "F"), coords, need_gradient=True, need_hessian=True, hessian_difference="central"
    )
    ev_g = with_grad.evaluate(
        ("H", "F"), coords, need_gradient=True, need_hessian=True, hessian_difference="central"
    )
    ev_fwd = with_grad.evaluate(
        ("H", "F"), coords, need_gradient=True, need_hessian=True, hessian_difference="forward"
    )
    ev_e_fwd = energy_only.evaluate(
        ("H", "F"), coords, need_gradient=True, need_hessian=True, hessian_difference="forward"
    )

    assert ev_e.gradient_kj_mol_angstrom is not None
    assert ev_g.gradient_kj_mol_angstrom is not None
    assert ev_e.hessian_kj_mol_angstrom2 is not None
    assert ev_g.hessian_kj_mol_angstrom2 is not None
    assert ev_fwd.hessian_kj_mol_angstrom2 is not None
    assert ev_e_fwd.hessian_kj_mol_angstrom2 is not None

    assert np.allclose(ev_e.gradient_kj_mol_angstrom, ev_g.gradient_kj_mol_angstrom, atol=1.0e-6)
    assert np.allclose(ev_e.hessian_kj_mol_angstrom2, ev_g.hessian_kj_mol_angstrom2, atol=1.0e-2)
    assert np.allclose(
        ev_g.hessian_kj_mol_angstrom2, ev_fwd.hessian_kj_mol_angstrom2, atol=2.0e-1
    )
    assert np.allclose(
        ev_e.hessian_kj_mol_angstrom2, ev_e_fwd.hessian_kj_mol_angstrom2, atol=5.0e-1
    )


def test_polyrate_pot_adapter_units_and_shapes() -> None:
    k_au = 0.25  # Hartree / bohr^2
    r0_bohr = 2.0
    e_ref_au = -1.1

    def pot_flat(x_bohr: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        pts = x_bohr.reshape((2, 3))
        d = pts[1] - pts[0]
        r = float(np.linalg.norm(d))
        u = d / r
        e = e_ref_au + 0.5 * k_au * (r - r0_bohr) ** 2
        g_pos = k_au * (r - r0_bohr) * u
        g = np.concatenate([-g_pos, g_pos])
        eye3 = np.eye(3)
        uu = np.outer(u, u)
        block = k_au * uu + k_au * (1.0 - r0_bohr / r) * (eye3 - uu)
        hess = np.block([[block, -block], [-block, block]])
        return e, g, hess

    def pot_3d(x_bohr: np.ndarray) -> float:
        assert x_bohr.shape == (2, 3)
        return pot_flat(x_bohr.reshape(-1))[0]

    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.3 * BOHR_ANGSTROM]], dtype=float)
    adapter_flat = PolyratePotAdapter(
        pot_flat, flat_coords=True, reference_energy_hartree=e_ref_au
    )
    adapter_3d = PolyratePotAdapter(
        pot_3d, flat_coords=False, reference_energy_hartree=e_ref_au
    )

    ev_flat = adapter_flat.evaluate(("H", "H"), coords, need_gradient=True, need_hessian=True)
    ev_3d = adapter_3d.evaluate(("H", "H"), coords, need_gradient=True, need_hessian=True)

    expected_e_au = 0.5 * k_au * (0.3**2)
    assert math.isclose(ev_flat.energy_hartree, expected_e_au, rel_tol=1.0e-12)
    assert math.isclose(ev_flat.energy_kj_mol, expected_e_au * HARTREE_KJ_MOL, rel_tol=1.0e-12)
    assert math.isclose(ev_3d.energy_kj_mol, ev_flat.energy_kj_mol, rel_tol=1.0e-12)
    assert ev_flat.gradient_hartree_bohr is not None
    assert ev_3d.gradient_hartree_bohr is not None
    assert np.allclose(ev_flat.gradient_hartree_bohr, ev_3d.gradient_hartree_bohr, atol=1.0e-7)
    assert ev_flat.hessian_hartree_bohr2 is not None
    assert ev_3d.hessian_hartree_bohr2 is not None
    assert np.allclose(ev_flat.hessian_hartree_bohr2, ev_3d.hessian_hartree_bohr2, atol=1.0e-4)


def test_cl_hbr_leps_surface_permutation_isotopes_and_saddle() -> None:
    pes = cl_hbr_leps_surface()
    guess = np.array([[0.0, 0.0, -1.8], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=float)
    ts_opt = optimize_saddle(pes, ("Cl", "H", "Br"), guess, mass_convention="polyrate")

    assert ts_opt.converged
    assert ts_opt.saddle_order == 1
    assert ts_opt.max_gradient_hartree_bohr < 1.0e-6
    barrier_kcal = ts_opt.evaluation.energy_kj_mol / 4.184
    assert math.isclose(barrier_kcal, 10.9993, abs_tol=1.0e-3)

    freqs = sorted(float(f) for f in ts_opt.normal_mode_result.frequencies_cm1)
    assert math.isclose(freqs[0], -1563.29, abs_tol=0.2)
    assert math.isclose(freqs[1], 305.89, abs_tol=0.2)
    assert math.isclose(freqs[2], 452.31, abs_tol=0.2)
    assert math.isclose(freqs[3], 452.31, abs_tol=0.2)

    # Atom permutation invariance + gradient matching
    perm = [2, 0, 1]
    coords_asym = np.array([[0.2, 0.0, -1.7], [0.0, 0.1, -0.05], [-0.1, -0.1, 1.45]], dtype=float)
    ev_std = pes.evaluate(("Cl", "H", "Br"), coords_asym, need_gradient=True)
    ev_perm = pes.evaluate(("Br", "Cl", "H"), coords_asym[perm], need_gradient=True)
    assert math.isclose(ev_std.energy_kj_mol, ev_perm.energy_kj_mol, rel_tol=1.0e-12)
    assert ev_std.gradient_kj_mol_angstrom is not None
    assert ev_perm.gradient_kj_mol_angstrom is not None
    assert np.allclose(ev_perm.gradient_kj_mol_angstrom, ev_std.gradient_kj_mol_angstrom[perm])

    # Isotope substitution keeps BO potential identical and shifts frequency cleanly
    ev_iso = pes.evaluate(("37Cl", "2H", "Br"), coords_asym, need_gradient=True)
    assert math.isclose(ev_std.energy_kj_mol, ev_iso.energy_kj_mol, rel_tol=1.0e-12)
    sp_h, _ = build_species_from_pes(
        pes, "ts-H", ("Cl", "H", "Br"), ts_opt.coords_angstrom,
        is_transition_state=True, mass_convention="polyrate",
    )
    sp_d, _ = build_species_from_pes(
        pes, "ts-D", ("37Cl", "2H", "Br"), ts_opt.coords_angstrom,
        is_transition_state=True, mass_convention="polyrate",
    )
    assert abs(sp_d.imaginary_frequencies_cm1[0]) < abs(sp_h.imaginary_frequencies_cm1[0])

    # Supermolecule fragment optimization of HBr & HCl
    hbr_pes = SupermoleculeFragmentPes(pes, ("Cl", "H", "Br"), (1, 2), ts_opt.coords_angstrom)
    sp_hbr, ev_hbr = build_species_from_pes(
        hbr_pes, "HBr", ("H", "Br"), [[0.0, 0.0, 0.0], [0.0, 0.0, 1.414]],
        mass_convention="polyrate", optimize=True,
    )
    assert abs(sp_hbr.electronic_energy_kj_mol) < 1.0e-6
    assert math.isclose(sp_hbr.positive_frequencies_cm1[0], 2706.94, abs_tol=0.5)
    assert ev_hbr.hessian_kj_mol_angstrom2 is not None
    assert ev_hbr.hessian_kj_mol_angstrom2.shape == (6, 6)

    hcl_pes = SupermoleculeFragmentPes(pes, ("Cl", "H", "Br"), (0, 1), ts_opt.coords_angstrom)
    sp_hcl, _ = build_species_from_pes(
        hcl_pes, "HCl", ("Cl", "H"), [[0.0, 0.0, 0.0], [0.0, 0.0, 1.275]],
        mass_convention="polyrate", optimize=True,
    )
    assert math.isclose(sp_hcl.electronic_energy_kj_mol / 4.184, -16.160, abs_tol=1.0e-2)
    assert math.isclose(sp_hcl.positive_frequencies_cm1[0], 2988.15, abs_tol=0.5)


def test_muller_brown_se3_invariance_and_stationary_points() -> None:
    mb = MullerBrownSurface()
    # Start near MB local/global minimum (-0.558, 1.442) with bond lengths r_x0=2.5, r_y0=1.5
    min_guess = np.array([[0.0, 0.0, 0.0], [1.94, 0.0, 0.0], [1.94, 2.94, 0.0]], dtype=float)
    min_opt = optimize_minimum(mb, ("C", "O", "N"), min_guess)
    assert min_opt.converged
    assert min_opt.saddle_order == 0
    assert min_opt.max_gradient_hartree_bohr < 1.0e-6

    # Saddle region near (-0.822, 0.624)
    ts_guess = np.array([[0.0, 0.0, 0.0], [1.68, 0.0, 0.0], [1.68, 2.12, 0.0]], dtype=float)
    ts_opt = optimize_saddle(mb, ("C", "O", "N"), ts_guess)
    assert ts_opt.converged
    assert ts_opt.saddle_order == 1
    assert ts_opt.evaluation.energy_kj_mol > min_opt.evaluation.energy_kj_mol

    # SE(3) rotation + translation invariance
    theta = 0.73
    rot = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.3],
            [math.sin(theta), math.cos(theta), -0.4],
            [0.1, -0.2, 0.9],
        ]
    )
    q_rot, _ = np.linalg.qr(rot)
    shift = np.array([1.7, -2.3, 0.9])
    ev_orig = mb.evaluate(("C", "O", "N"), ts_opt.coords_angstrom, need_gradient=True)
    ev_rot = mb.evaluate(
        ("C", "O", "N"), ts_opt.coords_angstrom @ q_rot.T + shift, need_gradient=True
    )
    assert math.isclose(ev_orig.energy_kj_mol, ev_rot.energy_kj_mol, rel_tol=1.0e-12)
    assert ev_orig.gradient_kj_mol_angstrom is not None
    assert ev_rot.gradient_kj_mol_angstrom is not None
    assert np.allclose(
        ev_rot.gradient_kj_mol_angstrom, ev_orig.gradient_kj_mol_angstrom @ q_rot.T, atol=1.0e-10
    )


def test_eckart_valley_surface_barrier_and_analytic_gradient() -> None:
    surf = EckartValleySurface(v1_kj_mol=40.0, v2_kj_mol=60.0, coupling_angstrom=0.05)
    # Collinear geometry along z with arbitrary asymmetric bond lengths
    coords = np.array([[0.0, 0.0, -1.35], [0.0, 0.0, 0.0], [0.0, 0.0, 1.15]], dtype=float)
    ev = surf.evaluate(("H", "C", "O"), coords, need_gradient=True)
    num_grad = surf.numerical_gradient(("H", "C", "O"), coords, step_bohr=1.0e-5)
    assert ev.gradient_kj_mol_angstrom is not None
    assert np.allclose(ev.gradient_kj_mol_angstrom, num_grad, atol=1.0e-6)

    # Check saddle location at u = r01 - r12 where Eckart derivative vanishes
    ts_guess = np.array([[0.0, 0.0, -1.2], [0.0, 0.0, 0.0], [0.0, 0.0, 1.2]], dtype=float)
    ts_opt = optimize_saddle(surf, ("H", "C", "O"), ts_guess)
    assert ts_opt.converged
    assert ts_opt.saddle_order == 1


def test_single_atom_and_surface_factory_validation() -> None:
    pes = h3_leps_surface()
    with pytest.raises(ValueError, match="exactly 3 atoms"):
        optimize_minimum(pes, ("H",), [[0.0, 0.0, 0.0]])

    atom_pes = CallablePes(lambda syms, xyz: -5.0, gradient_fn=lambda s, c: np.zeros((1, 3)))
    opt_1 = optimize_minimum(atom_pes, ("Ar",), [[0.0, 0.0, 0.0]])
    assert opt_1.converged
    assert opt_1.steps == 0
    sp_1, _ = build_species_from_pes(atom_pes, "Ar", ("Ar",), [[0.0, 0.0, 0.0]])
    assert sp_1.rotor_type == "atomic"

    # Factory helpers
    assert isinstance(get_builtin_pes("cl_hbr_leps"), ExtendedLepsSurface)
    assert isinstance(get_builtin_pes("h3"), ExtendedLepsSurface)
    assert isinstance(get_builtin_pes("muller-brown"), MullerBrownSurface)
    assert isinstance(get_builtin_pes("eckart"), EckartValleySurface)
    with pytest.raises(ValueError, match="unknown built-in PES"):
        get_builtin_pes("nonexistent")

    custom = build_pes_from_mapping(
        {
            "type": "extended_leps",
            "de_kcal_mol": [109.0, 109.0, 109.0],
            "re_angstrom": [0.74, 0.74, 0.74],
            "beta_angstrom_inv": [1.94, 1.94, 1.94],
            "sato": [0.14, 0.14, 0.14],
            "atom_symbols": ["H", "H", "H"],
        }
    )
    assert isinstance(custom, ExtendedLepsSurface)

    # TS validation fails if more than 1 imaginary frequency exists (3-atom nonlinear -> 3 modes)
    ref_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.2, 0.0]])
    bad_ts_pes = CallablePes(
        lambda s, c: -500.0 * float(np.sum((c - ref_coords) ** 2))
    )
    with pytest.raises(ValueError, match="imaginary frequencies"):
        build_species_from_pes(
            bad_ts_pes,
            "bad-ts",
            ("H", "O", "H"),
            ref_coords,
            is_transition_state=True,
        )
