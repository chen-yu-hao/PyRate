from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import pytest

from pyrate.constants import (
    AMU_KG,
    AMU_TO_AU_MASS,
    AVOGADRO_MOL,
    BOHR_ANGSTROM,
    BOHR_M,
    HARTREE_J,
    LIGHT_SPEED_M_S,
)
from pyrate.models import Atom, ElectronicState, PathPoint, Reaction, ReactionPath, Species


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Extended LEPS parameters for Cl + HBr -> ClH + Br (Kuntz et al., JCP 44, 1168 (1966))
_CKCAL = 627.5095
_CANGS = 0.529177106
_R2 = 1.41421356
_DEI = (106.4, 90.24, 52.09)
_REI = (1.275, 1.414, 2.136)
_BETAI = (1.867, 1.851, 1.923)
_ZI = (0.02, 0.02, 0.0)

_D = tuple(d / _CKCAL for d in _DEI)
_RE = tuple(r / _CANGS for r in _REI)
_BE = tuple(b * _CANGS for b in _BETAI)
_OP3Z = tuple(1.0 + 3.0 * z for z in _ZI)
_TOP3Z = tuple(2.0 * v for v in _OP3Z)
_ZP3 = tuple(z + 3.0 for z in _ZI)
_TZP3 = tuple(2.0 * v for v in _ZP3)
_DO4Z = tuple(_D[i] / (4.0 * (1.0 + _ZI[i])) for i in range(3))
_B = tuple(_BE[i] * _DO4Z[i] * 2.0 for i in range(3))

_HARTREE_TO_KJ_MOL = HARTREE_J * AVOGADRO_MOL / 1000.0
_AU_FREQ_TO_CM1 = (
    1.0
    / (2.0 * math.pi * LIGHT_SPEED_M_S * 100.0)
    * math.sqrt(HARTREE_J / ((AMU_KG / AMU_TO_AU_MASS) * BOHR_M**2))
)


def _clhbr_pot_and_dedr_scalar(
    r1: float, r2: float, r3: float
) -> tuple[float, float, float, float]:
    x0 = math.exp(-_BE[0] * (r1 - _RE[0]))
    x1 = math.exp(-_BE[1] * (r2 - _RE[1]))
    x2 = math.exp(-_BE[2] * (r3 - _RE[2]))

    coul0 = _DO4Z[0] * (_ZP3[0] * x0 - _TOP3Z[0]) * x0
    coul1 = _DO4Z[1] * (_ZP3[1] * x1 - _TOP3Z[1]) * x1
    coul2 = _DO4Z[2] * (_ZP3[2] * x2 - _TOP3Z[2]) * x2

    ex0 = _DO4Z[0] * (_OP3Z[0] * x0 - _TZP3[0]) * x0
    ex1 = _DO4Z[1] * (_OP3Z[1] * x1 - _TZP3[1]) * x1
    ex2 = _DO4Z[2] * (_OP3Z[2] * x2 - _TZP3[2]) * x2

    rad = math.sqrt((ex0 - ex1) ** 2 + (ex1 - ex2) ** 2 + (ex2 - ex0) ** 2)
    energy = coul0 + coul1 + coul2 - rad / _R2
    s = ex0 + ex1 + ex2
    inv_r2_rad = 1.0 / (_R2 * rad)
    d0 = _B[0] * x0 * (
        (3.0 * ex0 - s) * (_OP3Z[0] * x0 - _ZP3[0]) * inv_r2_rad - _ZP3[0] * x0 + _OP3Z[0]
    )
    d1 = _B[1] * x1 * (
        (3.0 * ex1 - s) * (_OP3Z[1] * x1 - _ZP3[1]) * inv_r2_rad - _ZP3[1] * x1 + _OP3Z[1]
    )
    d2 = _B[2] * x2 * (
        (3.0 * ex2 - s) * (_OP3Z[2] * x2 - _ZP3[2]) * inv_r2_rad - _ZP3[2] * x2 + _OP3Z[2]
    )
    return energy, d0, d1, d2


@lru_cache(maxsize=2)
def build_clhbr_leps_fixture(
    save_step_bohr: float = 0.01,
) -> tuple[Reaction, Callable[[np.ndarray], float]]:
    """Construct the collinear Cl + HBr -> ClH + Br Extended LEPS reaction and PES callable."""

    a_cl = Atom("Cl", 0.0, 0.0, 0.0, mass_convention="polyrate")
    a_h = Atom("H", 0.0, 0.0, 0.0, mass_convention="polyrate")
    a_br = Atom("Br", 0.0, 0.0, 0.0, mass_convention="polyrate")
    m_amu = np.array([a_cl.mass_amu, a_h.mass_amu, a_br.mass_amu])
    m_au = m_amu * AMU_TO_AU_MASS
    scalemass_amu = 24.3259

    r_hbr_eq = _RE[1]
    r_hcl_eq = _RE[0]
    e_react_au = _clhbr_pot_and_dedr_scalar(500.0, r_hbr_eq, 500.0 + r_hbr_eq)[0]
    e_prod_au = _clhbr_pot_and_dedr_scalar(r_hcl_eq, 500.0, 500.0 + r_hcl_eq)[0]

    eps = 1.0e-5
    ep = _clhbr_pot_and_dedr_scalar(500.0, r_hbr_eq + eps, 500.0 + r_hbr_eq + eps)[0]
    em = _clhbr_pot_and_dedr_scalar(500.0, r_hbr_eq - eps, 500.0 + r_hbr_eq - eps)[0]
    k_hbr = (ep - 2.0 * e_react_au + em) / (eps * eps)
    freq_hbr_cm1 = math.sqrt(k_hbr / (m_au[1] * m_au[2] / (m_au[1] + m_au[2]))) * _AU_FREQ_TO_CM1

    ep1 = _clhbr_pot_and_dedr_scalar(r_hcl_eq + eps, 500.0, 500.0 + r_hcl_eq + eps)[0]
    em1 = _clhbr_pot_and_dedr_scalar(r_hcl_eq - eps, 500.0, 500.0 + r_hcl_eq - eps)[0]
    k_hcl = (ep1 - 2.0 * e_prod_au + em1) / (eps * eps)
    freq_hcl_cm1 = math.sqrt(k_hcl / (m_au[0] * m_au[1] / (m_au[0] + m_au[1]))) * _AU_FREQ_TO_CM1

    r1_ts = 3.136276420
    r2_ts = 6.061583360 - 3.136276420

    def r12_grad_hess(r1: float, r2: float) -> tuple[float, np.ndarray, np.ndarray]:
        e0, d0, d1, d2 = _clhbr_pot_and_dedr_scalar(r1, r2, r1 + r2)
        g = np.array([d0 + d2, d1 + d2])
        d = 1.0e-5
        _, dp0, dp1, dp2 = _clhbr_pot_and_dedr_scalar(r1 + d, r2, r1 + d + r2)
        _, dm0, dm1, dm2 = _clhbr_pot_and_dedr_scalar(r1 - d, r2, r1 - d + r2)
        _, qp0, qp1, qp2 = _clhbr_pot_and_dedr_scalar(r1, r2 + d, r1 + r2 + d)
        _, qm0, qm1, qm2 = _clhbr_pot_and_dedr_scalar(r1, r2 - d, r1 + r2 - d)
        h = np.array(
            [
                [(dp0 + dp2 - dm0 - dm2) / (2.0 * d), (qp0 + qp2 - qm0 - qm2) / (2.0 * d)],
                [(dp1 + dp2 - dm1 - dm2) / (2.0 * d), (qp1 + qp2 - qm1 - qm2) / (2.0 * d)],
            ]
        )
        h = 0.5 * (h + h.T)
        return e0, g, h

    for _ in range(12):
        _, g, h = r12_grad_hess(r1_ts, r2_ts)
        if float(np.linalg.norm(g)) < 1.0e-14:
            break
        step = np.linalg.solve(h, g)
        r1_ts -= float(step[0])
        r2_ts -= float(step[1])

    m_tot = float(np.sum(m_amu))
    sqrt_m_ratio = np.sqrt(m_amu / scalemass_amu)
    inv_sqrt_mau = 1.0 / np.sqrt(m_au)
    uz = np.sqrt(m_au)
    uz /= float(np.linalg.norm(uz))
    pz = np.eye(3) - np.outer(uz, uz)
    jac = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])

    def r12_to_z(r1: float, r2: float) -> np.ndarray:
        z_raw = np.array([0.0, r1, r1 + r2])
        return z_raw - float(np.dot(m_amu, z_raw)) / m_tot

    def analyze_point(
        r1: float, r2: float, is_ts: bool = False
    ) -> tuple[np.ndarray, float, tuple[float, ...], np.ndarray, np.ndarray]:
        z = r12_to_z(r1, r2)
        e_au, d0, d1, d2 = _clhbr_pot_and_dedr_scalar(r1, r2, r1 + r2)
        gz = np.array([-d0 - d2, d0 - d1, d1 + d2])
        _, _, h12 = r12_grad_hess(r1, r2)
        hz = jac.T @ h12 @ jac
        mw_hz = hz * np.outer(inv_sqrt_mau, inv_sqrt_mau)

        if not is_ts:
            gz_mw = gz * inv_sqrt_mau
            gz_mw = gz_mw - uz * float(np.dot(uz, gz_mw))
            g_unit = gz_mw / float(np.linalg.norm(gz_mw))
            p_trans = pz - np.outer(g_unit, g_unit)
            evals_s, evecs_s = np.linalg.eigh(p_trans @ mw_hz @ p_trans)
            idx_str = int(np.argmax(np.abs(evals_s)))
            ev_str = float(evals_s[idx_str])
            omega_str = math.copysign(math.sqrt(abs(ev_str)), ev_str) * _AU_FREQ_TO_CM1
            vec_str_z = evecs_s[:, idx_str]
            imag_freq = None
        else:
            evals_s, evecs_s = np.linalg.eigh(pz @ mw_hz @ pz)
            idx_imag = int(np.argmin(evals_s))
            idx_str = int(np.argmax(evals_s))
            omega_str = math.sqrt(max(0.0, float(evals_s[idx_str]))) * _AU_FREQ_TO_CM1
            imag_freq = -math.sqrt(abs(float(evals_s[idx_imag]))) * _AU_FREQ_TO_CM1
            vec_str_z = evecs_s[:, idx_str]
            g_unit = evecs_s[:, idx_imag].copy()
            dz_test = g_unit * inv_sqrt_mau
            if (dz_test[1] - dz_test[0]) > 0.0:
                g_unit = -g_unit

        f_theta = -(r1 * r2 / (r1 + r2)) * d2
        g_theta = (
            1.0 / (m_au[0] * r1 * r1)
            + 1.0 / (m_au[2] * r2 * r2)
            + (1.0 / m_au[1]) * (1.0 / r1 + 1.0 / r2) ** 2
        )
        omega_bend = math.sqrt(max(0.0, f_theta * g_theta)) * _AU_FREQ_TO_CM1

        dz_str = vec_str_z * inv_sqrt_mau
        if (dz_str[2] - dz_str[0]) < 0.0:
            vec_str_z = -vec_str_z

        rx = np.sqrt(m_au) * z
        rx = rx - uz * float(np.dot(uz, rx))
        rx /= float(np.linalg.norm(rx))
        bx = np.cross(uz, rx)
        bx /= float(np.linalg.norm(bx))
        if bx[1] < 0.0:
            bx = -bx

        tangent_9d = np.zeros(9)
        tangent_9d[2::3] = g_unit if is_ts else ((1.0 if r1 > r1_ts else -1.0) * g_unit)

        mode_str_9d = np.zeros(9)
        mode_str_9d[2::3] = vec_str_z
        mode_bx_9d = np.zeros(9)
        mode_bx_9d[0::3] = bx
        mode_by_9d = np.zeros(9)
        mode_by_9d[1::3] = bx

        freqs = [omega_str, omega_bend, omega_bend]
        if imag_freq is not None:
            freqs.append(imag_freq)
        return (
            z,
            (e_au - e_react_au) * _HARTREE_TO_KJ_MOL,
            tuple(freqs),
            tangent_9d,
            np.column_stack([mode_str_9d, mode_bx_9d, mode_by_9d]),
        )

    z_ts, _, _, tan_ts, _ = analyze_point(r1_ts, r2_ts, is_ts=True)

    sr0, sr1, sr2 = float(sqrt_m_ratio[0]), float(sqrt_m_ratio[1]), float(sqrt_m_ratio[2])
    u_norm2 = sr0 * sr0 + sr1 * sr1 + sr2 * sr2
    u0, u1, u2 = sr0 / math.sqrt(u_norm2), sr1 / math.sqrt(u_norm2), sr2 / math.sqrt(u_norm2)

    def mep_rhs_scalar(xz0: float, xz1: float, xz2: float) -> tuple[float, float, float]:
        z0, z1, z2 = xz0 / sr0, xz1 / sr1, xz2 / sr2
        r1 = z1 - z0
        r2 = z2 - z1
        _, d0, d1, d2 = _clhbr_pot_and_dedr_scalar(r1, r2, r1 + r2)
        g0 = (-d0 - d2) / sr0
        g1 = (d0 - d1) / sr1
        g2 = (d1 + d2) / sr2
        proj = u0 * g0 + u1 * g1 + u2 * g2
        g0 -= u0 * proj
        g1 -= u1 * proj
        g2 -= u2 * proj
        inv_n = -1.0 / math.sqrt(g0 * g0 + g1 * g1 + g2 * g2)
        return g0 * inv_n, g1 * inv_n, g2 * inv_n

    ds_int = 0.00005
    save_every = max(1, int(round(save_step_bohr / ds_int)))
    n_steps = int(round(3.0 / ds_int))

    def integrate_branch(sign_dir: float) -> list[tuple[float, float, float]]:
        xz0 = sr0 * float(z_ts[0]) + sign_dir * 1.0e-4 * float(tan_ts[2])
        xz1 = sr1 * float(z_ts[1]) + sign_dir * 1.0e-4 * float(tan_ts[5])
        xz2 = sr2 * float(z_ts[2]) + sign_dir * 1.0e-4 * float(tan_ts[8])
        rem = ds_int - 1.0e-4
        k10, k11, k12 = mep_rhs_scalar(xz0, xz1, xz2)
        k20, k21, k22 = mep_rhs_scalar(xz0 + 0.5 * rem * k10, xz1 + 0.5 * rem * k11, xz2 + 0.5 * rem * k12)
        k30, k31, k32 = mep_rhs_scalar(xz0 + 0.5 * rem * k20, xz1 + 0.5 * rem * k21, xz2 + 0.5 * rem * k22)
        k40, k41, k42 = mep_rhs_scalar(xz0 + rem * k30, xz1 + rem * k31, xz2 + rem * k32)
        xz0 += (rem / 6.0) * (k10 + 2.0 * k20 + 2.0 * k30 + k40)
        xz1 += (rem / 6.0) * (k11 + 2.0 * k21 + 2.0 * k31 + k41)
        xz2 += (rem / 6.0) * (k12 + 2.0 * k22 + 2.0 * k32 + k42)
        pts_b: list[tuple[float, float, float]] = []
        h = ds_int
        h2 = 0.5 * h
        h6 = h / 6.0
        for step_idx in range(1, n_steps + 1):
            if step_idx % save_every == 0:
                z0, z1, z2 = xz0 / sr0, xz1 / sr1, xz2 / sr2
                pts_b.append((sign_dir * step_idx * h, z1 - z0, z2 - z1))
            if step_idx == n_steps:
                break
            k10, k11, k12 = mep_rhs_scalar(xz0, xz1, xz2)
            k20, k21, k22 = mep_rhs_scalar(xz0 + h2 * k10, xz1 + h2 * k11, xz2 + h2 * k12)
            k30, k31, k32 = mep_rhs_scalar(xz0 + h2 * k20, xz1 + h2 * k21, xz2 + h2 * k22)
            k40, k41, k42 = mep_rhs_scalar(xz0 + h * k30, xz1 + h * k31, xz2 + h * k32)
            xz0 += h6 * (k10 + 2.0 * k20 + 2.0 * k30 + k40)
            xz1 += h6 * (k11 + 2.0 * k21 + 2.0 * k31 + k41)
            xz2 += h6 * (k12 + 2.0 * k22 + 2.0 * k32 + k42)
        return pts_b

    all_s_r12 = integrate_branch(-1.0)[::-1] + [(0.0, r1_ts, r2_ts)] + integrate_branch(1.0)
    s_list: list[float] = []
    sp_list: list[Species] = []
    tan_list: list[np.ndarray] = []
    modes_list: list[np.ndarray] = []
    ts_elec = (ElectronicState(2, 0.0),)

    for idx, (s_val, r1_p, r2_p) in enumerate(all_s_r12):
        is_ts = abs(s_val) < 1.0e-12
        z_pt, v_kj, freqs, tan_9d, modes_9d = analyze_point(r1_p, r2_p, is_ts=is_ts)
        atoms_pt = (
            Atom("Cl", 0.0, 0.0, float(z_pt[0]) * BOHR_ANGSTROM, mass_convention="polyrate"),
            Atom("H", 0.0, 0.0, float(z_pt[1]) * BOHR_ANGSTROM, mass_convention="polyrate"),
            Atom("Br", 0.0, 0.0, float(z_pt[2]) * BOHR_ANGSTROM, mass_convention="polyrate"),
        )
        sp_list.append(
            Species(
                name="ts" if is_ts else f"mep-{idx}",
                atoms=atoms_pt,
                electronic_energy_kj_mol=v_kj,
                frequencies_cm1=freqs,
                symmetry_number=1,
                rotor_type="linear",
                electronic_states=ts_elec,
            )
        )
        s_list.append(s_val)
        tan_list.append(tan_9d)
        modes_list.append(modes_9d)

    dyds = np.gradient(np.array(tan_list), np.array(s_list), axis=0)
    path_points: list[PathPoint] = []
    ts_species: Species | None = None
    for idx, s_val in enumerate(s_list):
        modes_9d = modes_list[idx]
        curvs = tuple(float(-np.dot(dyds[idx], modes_9d[:, m])) for m in range(3))
        if abs(s_val) < 1.0e-12:
            ts_species = sp_list[idx]
        path_points.append(
            PathPoint(
                coordinate=s_val,
                dividing_surface=sp_list[idx],
                curvatures_bohr_inv=curvs,
                tangent_vector=tuple(float(v) for v in tan_list[idx]),
                normal_mode_vectors=tuple(
                    tuple(float(v) for v in modes_9d[:, m]) for m in range(3)
                ),
            )
        )
    assert ts_species is not None

    r_hbr = Species(
        name="HBr",
        atoms=(
            Atom("H", 0.0, 0.0, 0.0, mass_convention="polyrate"),
            Atom("Br", 0.0, 0.0, r_hbr_eq * BOHR_ANGSTROM, mass_convention="polyrate"),
        ),
        electronic_energy_kj_mol=0.0,
        frequencies_cm1=(freq_hbr_cm1,),
        symmetry_number=1,
        rotor_type="linear",
        electronic_states=(ElectronicState(1, 0.0),),
    )
    r_cl = Species(
        name="Cl",
        atoms=(Atom("Cl", 0.0, 0.0, 0.0, mass_convention="polyrate"),),
        electronic_energy_kj_mol=0.0,
        frequencies_cm1=(),
        symmetry_number=1,
        rotor_type="atomic",
        electronic_states=(
            ElectronicState(4, 0.0),
            ElectronicState(2, 0.00401413 * _HARTREE_TO_KJ_MOL),
        ),
    )
    p_hcl = Species(
        name="HCl",
        atoms=(
            Atom("Cl", 0.0, 0.0, 0.0, mass_convention="polyrate"),
            Atom("H", 0.0, 0.0, r_hcl_eq * BOHR_ANGSTROM, mass_convention="polyrate"),
        ),
        electronic_energy_kj_mol=(e_prod_au - e_react_au) * _HARTREE_TO_KJ_MOL,
        frequencies_cm1=(freq_hcl_cm1,),
        symmetry_number=1,
        rotor_type="linear",
        electronic_states=(ElectronicState(1, 0.0),),
    )
    p_br = Species(
        name="Br",
        atoms=(Atom("Br", 0.0, 0.0, 0.0, mass_convention="polyrate"),),
        electronic_energy_kj_mol=0.0,
        frequencies_cm1=(),
        symmetry_number=1,
        rotor_type="atomic",
        electronic_states=(
            ElectronicState(4, 0.0),
            ElectronicState(2, 0.01679010 * _HARTREE_TO_KJ_MOL),
        ),
    )

    rxn = Reaction(
        reactants=(r_hbr, r_cl),
        transition_state=ts_species,
        products=(p_hcl, p_br),
        path=ReactionPath(
            points=tuple(path_points),
            coordinate_unit="bohr",
            reduced_mass_amu=scalemass_amu,
        ),
    )

    inv_bohr = 1.0 / BOHR_ANGSTROM

    def potential_fn(coords_angstrom: np.ndarray) -> float:
        c00 = float(coords_angstrom[0, 0])
        c01 = float(coords_angstrom[0, 1])
        c02 = float(coords_angstrom[0, 2])
        c10 = float(coords_angstrom[1, 0])
        c11 = float(coords_angstrom[1, 1])
        c12 = float(coords_angstrom[1, 2])
        c20 = float(coords_angstrom[2, 0])
        c21 = float(coords_angstrom[2, 1])
        c22 = float(coords_angstrom[2, 2])
        r1 = math.sqrt((c00 - c10) ** 2 + (c01 - c11) ** 2 + (c02 - c12) ** 2) * inv_bohr
        r2 = math.sqrt((c10 - c20) ** 2 + (c11 - c21) ** 2 + (c12 - c22) ** 2) * inv_bohr
        r3 = math.sqrt((c20 - c00) ** 2 + (c21 - c01) ** 2 + (c22 - c02) ** 2) * inv_bohr
        e_au = _clhbr_pot_and_dedr_scalar(r1, r2, r3)[0]
        return (e_au - e_react_au) * _HARTREE_TO_KJ_MOL

    return rxn, potential_fn
