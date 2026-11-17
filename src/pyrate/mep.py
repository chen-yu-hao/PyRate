"""Minimum Energy Path (MEP) integration, projected Hessian analysis, and ``.fu40`` export."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .constants import (
    AMU_KG,
    BOHR_ANGSTROM,
    BOHR_M,
    DEFAULT_STANDARD_PRESSURE_PA,
    HARTREE_J,
    KJ_MOL_TO_HARTREE,
    LIGHT_SPEED_M_S,
)
from .elements import atomic_mass_amu
from .geometry import classify_rotor, principal_moments_amu_a2
from .models import (
    Atom,
    ElectronicState,
    MepOptions,
    PathPoint,
    Reaction,
    ReactionPath,
    Species,
)
from .native import page_mciver_lqa_step, project_symmetric_matrix
from .normal_modes import _rigid_basis, normal_modes
from .pes import PesEvaluation, PotentialEnergySurface

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MepPointData:
    """Detailed electronic-structure and normal-mode data at one saved MEP point."""

    coordinate_bohr: float
    coords_angstrom: FloatArray
    energy_hartree: float
    gradient_hartree_bohr: FloatArray
    hessian_hartree_bohr2: FloatArray
    tangent_mw: FloatArray
    frequencies_cm1: tuple[float, ...]
    curvatures_bohr_inv: tuple[float, ...]
    mode_vectors_mw: FloatArray
    species: Species
    path_point: PathPoint


@dataclass(frozen=True, slots=True)
class MepResult:
    """Complete result of reaction-path integration from an analytical or external PES."""

    symbols: tuple[str, ...]
    masses_amu: tuple[float, ...]
    scale_mass_amu: float
    rotor_type: str
    points: tuple[MepPointData, ...]
    reaction_path: ReactionPath
    transition_state: Species
    saddle_evaluation: PesEvaluation
    method: str
    hessian_evaluations: int
    gradient_evaluations: int
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def build_reaction(
        self,
        *,
        reactants: Sequence[Species],
        products: Sequence[Species] = (),
        standard_pressure_pa: float = DEFAULT_STANDARD_PRESSURE_PA,
    ) -> Reaction:
        """Assemble a complete ``Reaction`` with the integrated ``reaction_path``."""

        return Reaction(
            reactants=tuple(reactants),
            transition_state=self.transition_state,
            products=tuple(products),
            path=self.reaction_path,
            standard_pressure_pa=standard_pressure_pa,
        )

    def to_fu40_text(
        self,
        *,
        reactants: Sequence[Species] = (),
        products: Sequence[Species] = (),
        hessform: str = "packed",
        freqsource: str = "hessian",
        reactant_evals: Sequence[PesEvaluation | None] | None = None,
        product_evals: Sequence[PesEvaluation | None] | None = None,
    ) -> str:
        """Serialize the MEP and stationary points to a Polyrate 2023 ``.fu40`` file."""

        norm_form = hessform.strip().lower()
        if norm_form not in {"packed", "full"}:
            raise ValueError("hessform must be 'packed' or 'full'")
        norm_src = freqsource.strip().lower()
        if norm_src not in {"hessian", "read"}:
            raise ValueError("freqsource must be 'hessian' or 'read'")

        lines: list[str] = [
            "*GEN40",
            "  GEOMUNIT BOHR",
            f"  FREQSOURCE {norm_src.upper()}",
            "  FREQUNIT WAVEN",
            f"  HESSFORM {norm_form.upper()}",
            "",
        ]

        def _append_matrix_block(keyword: str, values: Sequence[float], per_line: int = 5) -> None:
            lines.append(f"  {keyword}")
            for i in range(0, len(values), per_line):
                chunk = values[i : i + per_line]
                lines.append("    " + "  ".join(f"{v:16.8E}" for v in chunk))
            lines.append("  END")

        def _format_hessian_values(hess_au: FloatArray) -> list[float]:
            dim = hess_au.shape[0]
            vals: list[float] = []
            if norm_form == "packed":
                for row in range(dim):
                    for col in range(row + 1):
                        vals.append(float(hess_au[row, col]))
            else:
                for row in range(dim):
                    for col in range(dim):
                        vals.append(float(hess_au[row, col]))
            return vals

        def _append_stationary(
            sec_name: str,
            sp: Species,
            ev: PesEvaluation | None,
            *,
            is_ts: bool = False,
        ) -> None:
            lines.append(f"*{sec_name}")
            e_au = (
                ev.energy_hartree
                if ev is not None
                else sp.electronic_energy_kj_mol * KJ_MOL_TO_HARTREE
            )
            lines.append(f"  VVALUE {e_au:18.10E}")
            if sp.atoms:
                lines.append("  GEOM")
                for atom in sp.atoms:
                    xb = atom.x_angstrom / BOHR_ANGSTROM
                    yb = atom.y_angstrom / BOHR_ANGSTROM
                    zb = atom.z_angstrom / BOHR_ANGSTROM
                    lines.append(f"    {xb:16.8E}  {yb:16.8E}  {zb:16.8E}")
                lines.append("  END")
            if norm_src == "read" or ev is None or ev.hessian_hartree_bohr2 is None:
                if sp.positive_frequencies_cm1:
                    _append_matrix_block("FREQ", sp.positive_frequencies_cm1, per_line=6)
                if is_ts and sp.imaginary_frequencies_cm1:
                    lines.append(f"  WSTAR {abs(sp.imaginary_frequencies_cm1[0]):16.8F}")
            else:
                if ev.gradient_hartree_bohr is not None:
                    _append_matrix_block(
                        "GRADS", ev.gradient_hartree_bohr.reshape(-1).tolist(), per_line=5
                    )
                _append_matrix_block(
                    "HESSIAN", _format_hessian_values(ev.hessian_hartree_bohr2), per_line=5
                )
            lines.append("")

        r_tags = ("R140", "R240")
        for idx, sp in enumerate(reactants[:2]):
            ev = reactant_evals[idx] if reactant_evals and idx < len(reactant_evals) else None
            _append_stationary(r_tags[idx], sp, ev, is_ts=False)

        p_tags = ("P140", "P240")
        for idx, sp in enumerate(products[:2]):
            ev = product_evals[idx] if product_evals and idx < len(product_evals) else None
            _append_stationary(p_tags[idx], sp, ev, is_ts=False)

        _append_stationary("SADDLE40", self.transition_state, self.saddle_evaluation, is_ts=True)

        for pt in self.points:
            lines.append("*POINT40")
            lines.append(f"  SMEP {pt.coordinate_bohr:16.8E}")
            lines.append(f"  VMEP {pt.energy_hartree:18.10E}")
            lines.append("  GEOM")
            coords_b = pt.coords_angstrom / BOHR_ANGSTROM
            for row in coords_b:
                lines.append(f"    {row[0]:16.8E}  {row[1]:16.8E}  {row[2]:16.8E}")
            lines.append("  END")
            if norm_src == "read":
                pos_freqs = pt.species.positive_frequencies_cm1
                if pos_freqs:
                    _append_matrix_block("FREQ", pos_freqs, per_line=6)
                if pt.curvatures_bohr_inv:
                    _append_matrix_block("BFS", pt.curvatures_bohr_inv, per_line=5)
            else:
                _append_matrix_block(
                    "GRADS", pt.gradient_hartree_bohr.reshape(-1).tolist(), per_line=5
                )
                _append_matrix_block(
                    "HESSIAN", _format_hessian_values(pt.hessian_hartree_bohr2), per_line=5
                )
            lines.append("")

        return "\n".join(lines)

    def save_fu40(
        self,
        path: str | Path,
        *,
        reactants: Sequence[Species] = (),
        products: Sequence[Species] = (),
        hessform: str = "packed",
        freqsource: str = "hessian",
        reactant_evals: Sequence[PesEvaluation | None] | None = None,
        product_evals: Sequence[PesEvaluation | None] | None = None,
    ) -> Path:
        target = Path(path)
        text = self.to_fu40_text(
            reactants=reactants,
            products=products,
            hessform=hessform,
            freqsource=freqsource,
            reactant_evals=reactant_evals,
            product_evals=product_evals,
        )
        target.write_text(text, encoding="utf-8")
        return target


def _fragment_pair_distances_bohr(species: Species) -> list[float]:
    coords = np.asarray(
        [(a.x_angstrom, a.y_angstrom, a.z_angstrom) for a in species.atoms], dtype=float
    ) / BOHR_ANGSTROM
    n = coords.shape[0]
    dists: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(coords[i] - coords[j])))
    return dists


def _map_species_atom_indices(
    species: Species,
    full_symbols: tuple[str, ...],
    used: set[int],
) -> tuple[int, ...] | None:
    if not species.atoms:
        return None
    assigned: list[int] = []
    for atom in species.atoms:
        found = None
        for idx, sym in enumerate(full_symbols):
            if idx not in used and sym == atom.symbol:
                found = idx
                break
        if found is None:
            return None
        used.add(found)
        assigned.append(found)
    return tuple(assigned)


def _fragment_deviation(
    coords_bohr: FloatArray,
    species_list: Sequence[Species],
    index_groups: Sequence[tuple[int, ...] | None],
) -> float | None:
    total_err = 0.0
    count = 0
    for sp, idx_group in zip(species_list, index_groups, strict=True):
        if idx_group is None or len(idx_group) < 2 or len(sp.atoms) != len(idx_group):
            continue
        eq_dists = _fragment_pair_distances_bohr(sp)
        pair_i = 0
        for a in range(len(idx_group)):
            for b in range(a + 1, len(idx_group)):
                d_cur = float(np.linalg.norm(coords_bohr[idx_group[a]] - coords_bohr[idx_group[b]]))
                diff = d_cur - eq_dists[pair_i]
                total_err += diff * diff
                pair_i += 1
                count += 1
    if count == 0:
        return None
    return math.sqrt(total_err / count)


def _determine_tangent_sign(
    t0_mw: FloatArray,
    ts_coords_bohr: FloatArray,
    sigma: FloatArray,
    symbols: tuple[str, ...],
    *,
    direction_sign: int,
    direction_vector_angstrom: FloatArray | None,
    reactants: Sequence[Species],
    products: Sequence[Species],
    reactant_atom_indices: Sequence[tuple[int, ...]] | None,
    product_atom_indices: Sequence[tuple[int, ...]] | None,
) -> float:
    base_sign = 1.0
    if direction_vector_angstrom is not None:
        d_mw = (direction_vector_angstrom.reshape(-1) / BOHR_ANGSTROM) * sigma
        base_sign = 1.0 if float(np.dot(t0_mw, d_mw)) >= 0.0 else -1.0
        return base_sign * float(direction_sign)

    eps = 0.02
    delta_bohr = (eps * t0_mw / sigma).reshape((-1, 3))
    c_plus = ts_coords_bohr + delta_bohr
    c_minus = ts_coords_bohr - delta_bohr

    if reactant_atom_indices is None and reactants:
        used_r: set[int] = set()
        r_groups: list[tuple[int, ...] | None] = [
            _map_species_atom_indices(sp, symbols, used_r) for sp in reactants
        ]
    else:
        r_groups = list(reactant_atom_indices) if reactant_atom_indices is not None else []

    if product_atom_indices is None and products:
        used_p: set[int] = set()
        p_groups: list[tuple[int, ...] | None] = [
            _map_species_atom_indices(sp, symbols, used_p) for sp in products
        ]
    else:
        p_groups = list(product_atom_indices) if product_atom_indices is not None else []

    score_plus = 0.0
    score_minus = 0.0
    has_signal = False

    if products and p_groups:
        dp = _fragment_deviation(c_plus, products, p_groups)
        dm = _fragment_deviation(c_minus, products, p_groups)
        if dp is not None and dm is not None and abs(dp - dm) > 1.0e-12:
            # Plus moves closer to product eq -> dp < dm -> reward +
            score_plus += dm - dp
            has_signal = True

    if reactants and r_groups:
        rp = _fragment_deviation(c_plus, reactants, r_groups)
        rm = _fragment_deviation(c_minus, reactants, r_groups)
        if rp is not None and rm is not None and abs(rp - rm) > 1.0e-12:
            # Plus moves away from reactant eq -> rp > rm -> reward +
            score_plus += rp - rm
            has_signal = True

    if has_signal and abs(score_plus - score_minus) > 1.0e-12:
        base_sign = 1.0 if score_plus > score_minus else -1.0
    else:
        max_idx = int(np.argmax(np.abs(t0_mw)))
        base_sign = 1.0 if t0_mw[max_idx] >= 0.0 else -1.0

    return base_sign * float(direction_sign)


def _eig_to_cm1(evals: FloatArray, scale_mass_amu: float) -> FloatArray:
    si_factor = HARTREE_J / (scale_mass_amu * AMU_KG * BOHR_M * BOHR_M)
    cm1_factor = 1.0 / (2.0 * math.pi * LIGHT_SPEED_M_S * 100.0)
    lam_si = evals * si_factor
    return np.asarray(np.sign(lam_si) * np.sqrt(np.abs(lam_si)) * cm1_factor, dtype=np.float64)


def _rigid_basis_for_coords(
    coords_bohr: FloatArray,
    masses_amu: FloatArray,
    *,
    project_rigid_modes: bool,
) -> tuple[FloatArray, int]:
    dim = coords_bohr.size
    if not project_rigid_modes:
        return np.zeros((dim, 0), dtype=float), 0
    coords_m = coords_bohr.reshape((-1, 3)) * BOHR_M
    masses_kg = masses_amu * AMU_KG
    return _rigid_basis(masses_kg, coords_m)


def _projected_vibrational_modes(
    hess_mw: FloatArray,
    basis_proj: FloatArray,
    *,
    scale_mass_amu: float,
) -> tuple[FloatArray, FloatArray]:
    """Return transverse eigenvalues converted to cm⁻¹ (descending) and columns ``L_m``."""

    dim = hess_mw.shape[0]
    proj_h = project_symmetric_matrix(hess_mw, basis_proj)
    evals, evecs = np.linalg.eigh(proj_h)
    n_proj = basis_proj.shape[1]
    n_vib = dim - n_proj
    if n_vib <= 0:
        return np.zeros(0, dtype=float), np.zeros((dim, 0), dtype=float)

    overlaps = np.sum((basis_proj.T @ evecs) ** 2, axis=0) if n_proj > 0 else np.zeros(dim)
    keep_indices = np.argsort(overlaps)[:n_vib]
    freqs = _eig_to_cm1(evals[keep_indices], scale_mass_amu)
    desc_order = np.argsort(freqs)[::-1]
    sel = keep_indices[desc_order]
    return freqs[desc_order], evecs[:, sel]


def _bond_derivs(
    coords: FloatArray, i: int, j: int
) -> tuple[FloatArray, FloatArray]:
    dim = 3 * coords.shape[0]
    d = coords[i] - coords[j]
    r = float(np.linalg.norm(d))
    grad = np.zeros(dim, dtype=float)
    hess = np.zeros((dim, dim), dtype=float)
    if r <= 1.0e-12:
        return grad, hess
    u = d / r
    grad[3 * i : 3 * i + 3] = u
    grad[3 * j : 3 * j + 3] = -u
    block = (np.eye(3, dtype=float) - np.outer(u, u)) / r
    hess[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = block
    hess[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] = block
    hess[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = -block
    hess[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = -block
    return grad, hess


def _angle_derivs(
    coords: FloatArray, i: int, j: int, k: int
) -> list[tuple[FloatArray, FloatArray]]:
    """Return valence angle or orthogonal linear-bend ``(grad, hess)`` pairs."""

    dim = 3 * coords.shape[0]
    r1 = coords[i] - coords[j]
    r2 = coords[k] - coords[j]
    d1 = float(np.linalg.norm(r1))
    d2 = float(np.linalg.norm(r2))
    if d1 <= 1.0e-12 or d2 <= 1.0e-12:
        return []
    u = r1 / d1
    v = r2 / d2
    c = float(np.clip(float(np.dot(u, v)), -1.0, 1.0))
    s2 = max(0.0, 1.0 - c * c)
    s = math.sqrt(s2)

    sl_i = slice(3 * i, 3 * i + 3)
    sl_j = slice(3 * j, 3 * j + 3)
    sl_k = slice(3 * k, 3 * k + 3)

    if s >= 0.05:
        g_r1 = (c * u - v) / (d1 * s)
        g_r2 = (c * v - u) / (d2 * s)
        grad = np.zeros(dim, dtype=float)
        grad[sl_i] = g_r1
        grad[sl_k] = g_r2
        grad[sl_j] = -(g_r1 + g_r2)

        eye3 = np.eye(3, dtype=float)
        pu = (eye3 - np.outer(u, u)) / d1
        pv = (eye3 - np.outer(v, v)) / d2
        dc_11 = -(np.outer(v, u) + np.outer(u, v) + c * eye3 - 3.0 * c * np.outer(u, u)) / (
            d1 * d1
        )
        dc_22 = -(np.outer(u, v) + np.outer(v, u) + c * eye3 - 3.0 * c * np.outer(v, v)) / (
            d2 * d2
        )
        dc_12 = pu @ pv

        h_11 = -dc_11 / s - (c / s) * np.outer(g_r1, g_r1)
        h_22 = -dc_22 / s - (c / s) * np.outer(g_r2, g_r2)
        h_12 = -dc_12 / s - (c / s) * np.outer(g_r1, g_r2)
        h_11 = 0.5 * (h_11 + h_11.T)
        h_22 = 0.5 * (h_22 + h_22.T)

        hess = np.zeros((dim, dim), dtype=float)
        hess[sl_i, sl_i] = h_11
        hess[sl_k, sl_k] = h_22
        hess[sl_i, sl_k] = h_12
        hess[sl_k, sl_i] = h_12.T
        hess[sl_i, sl_j] = -(h_11 + h_12)
        hess[sl_j, sl_i] = hess[sl_i, sl_j].T
        hess[sl_k, sl_j] = -(h_12.T + h_22)
        hess[sl_j, sl_k] = hess[sl_k, sl_j].T
        hess[sl_j, sl_j] = h_11 + h_12 + h_12.T + h_22
        return [(grad, 0.5 * (hess + hess.T))]

    # Collinear / near-collinear: two orthogonal linear-bend coordinates wrt local axis
    axis = (u - v) if c <= 0.0 else (u + v)
    ax_norm = float(np.linalg.norm(axis))
    e_par = axis / ax_norm if ax_norm > 1.0e-12 else u.copy()
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(e_par, trial))) > 0.8:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    n1 = trial - float(np.dot(trial, e_par)) * e_par
    n1 /= float(np.linalg.norm(n1))
    n2 = np.cross(e_par, n1)
    n2 /= float(np.linalg.norm(n2))

    eye3 = np.eye(3, dtype=float)
    out: list[tuple[FloatArray, FloatArray]] = []
    for n_vec in (n1, n2):
        nu = float(np.dot(n_vec, u))
        nv = float(np.dot(n_vec, v))
        g_r1 = (n_vec - nu * u) / d1
        g_r2 = (n_vec - nv * v) / d2
        grad = np.zeros(dim, dtype=float)
        grad[sl_i] = g_r1
        grad[sl_k] = g_r2
        grad[sl_j] = -(g_r1 + g_r2)

        h_11 = (
            -(np.outer(n_vec, u) + np.outer(u, n_vec) + nu * eye3) + 3.0 * nu * np.outer(u, u)
        ) / (d1 * d1)
        h_22 = (
            -(np.outer(n_vec, v) + np.outer(v, n_vec) + nv * eye3) + 3.0 * nv * np.outer(v, v)
        ) / (d2 * d2)
        hess = np.zeros((dim, dim), dtype=float)
        hess[sl_i, sl_i] = h_11
        hess[sl_k, sl_k] = h_22
        hess[sl_i, sl_j] = -h_11
        hess[sl_j, sl_i] = -h_11
        hess[sl_k, sl_j] = -h_22
        hess[sl_j, sl_k] = -h_22
        hess[sl_j, sl_j] = h_11 + h_22
        out.append((grad, 0.5 * (hess + hess.T)))
    return out


def _dihedral_grad_4(
    p0: FloatArray, p1: FloatArray, p2: FloatArray, p3: FloatArray
) -> FloatArray | None:
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    b2_n = float(np.linalg.norm(b2))
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_sq = float(np.dot(n1, n1))
    n2_sq = float(np.dot(n2, n2))
    b1_n = float(np.linalg.norm(b1))
    b3_n = float(np.linalg.norm(b3))
    if b2_n <= 1.0e-12 or b1_n <= 1.0e-12 or b3_n <= 1.0e-12:
        return None
    if math.sqrt(n1_sq) <= 0.04 * b1_n * b2_n or math.sqrt(n2_sq) <= 0.04 * b2_n * b3_n:
        return None
    g0 = -(b2_n / n1_sq) * n1
    g3 = (b2_n / n2_sq) * n2
    c12 = float(np.dot(b1, b2)) / (b2_n * b2_n)
    c23 = float(np.dot(b3, b2)) / (b2_n * b2_n)
    g1 = -g0 + c12 * g0 - c23 * g3
    g2 = -g3 - c12 * g0 + c23 * g3
    return np.concatenate([g0, g1, g2, g3])


def _dihedral_derivs(
    coords: FloatArray, i: int, j: int, k: int, ell: int
) -> tuple[FloatArray, FloatArray] | None:
    p = np.asarray([coords[i], coords[j], coords[k], coords[ell]], dtype=float)
    g0 = _dihedral_grad_4(p[0], p[1], p[2], p[3])
    if g0 is None:
        return None
    dim = 3 * coords.shape[0]
    h_loc = np.zeros((12, 12), dtype=float)
    step = 1.0e-6
    for c_idx in range(12):
        a_l, xyz = divmod(c_idx, 3)
        pp = p.copy()
        pm = p.copy()
        pp[a_l, xyz] += step
        pm[a_l, xyz] -= step
        gp = _dihedral_grad_4(pp[0], pp[1], pp[2], pp[3])
        gm = _dihedral_grad_4(pm[0], pm[1], pm[2], pm[3])
        if gp is None or gm is None:
            return None
        h_loc[:, c_idx] = (gp - gm) / (2.0 * step)
    h_loc = 0.5 * (h_loc + h_loc.T)

    grad = np.zeros(dim, dtype=float)
    hess = np.zeros((dim, dim), dtype=float)
    atoms4 = (i, j, k, ell)
    for idx_a, at_a in enumerate(atoms4):
        sl_a = slice(3 * at_a, 3 * at_a + 3)
        grad[sl_a] = g0[3 * idx_a : 3 * idx_a + 3]
        for idx_b, at_b in enumerate(atoms4):
            sl_b = slice(3 * at_b, 3 * at_b + 3)
            hess[sl_a, sl_b] = h_loc[3 * idx_a : 3 * idx_a + 3, 3 * idx_b : 3 * idx_b + 3]
    return grad, hess


def curvilinear_corrected_hessian(
    coords_bohr: npt.ArrayLike,
    gradient_hartree_bohr: npt.ArrayLike,
    hessian_hartree_bohr2: npt.ArrayLike,
    masses_amu: Sequence[float] | npt.ArrayLike,
) -> FloatArray:
    """Apply the Jackels–Gu–Truhlar (1995) curvilinear internal coordinate correction.

    Returns ``H_curv = H_x - sum_a g_{q,a} (d^2 q_a / dx^2)`` in Cartesian
    Hartree bohr⁻² using Prim MST bonds, valence/linear bends, and dihedrals
    with inverse-mass weighting ``W = M^-1``.
    """

    coords = np.asarray(coords_bohr, dtype=float)
    n_atoms = coords.shape[0]
    dim = 3 * n_atoms
    hess = np.asarray(hessian_hartree_bohr2, dtype=float).reshape((dim, dim))
    hess = np.asarray(0.5 * (hess + hess.T), dtype=float)
    grad = np.asarray(gradient_hartree_bohr, dtype=float).reshape(dim)
    if n_atoms < 2 or float(np.max(np.abs(grad))) <= 1.0e-14:
        return hess

    m_arr = np.asarray(masses_amu, dtype=float).reshape(n_atoms)
    coord_m = np.repeat(m_arr, 3)
    inv_sqrt_m = 1.0 / np.sqrt(coord_m)

    # Prim Minimum Spanning Tree over interatomic distances
    dist = np.zeros((n_atoms, n_atoms), dtype=float)
    for a in range(n_atoms):
        for b in range(a + 1, n_atoms):
            d_ab = float(np.linalg.norm(coords[a] - coords[b]))
            dist[a, b] = dist[b, a] = d_ab

    in_tree = [False] * n_atoms
    in_tree[0] = True
    mst_edges: list[tuple[int, int]] = []
    for _ in range(n_atoms - 1):
        best_d = math.inf
        best_u, best_v = -1, -1
        for u in range(n_atoms):
            if not in_tree[u]:
                continue
            for v in range(n_atoms):
                if not in_tree[v] and dist[u, v] < best_d:
                    best_d = dist[u, v]
                    best_u, best_v = u, v
        if best_v < 0:
            break
        in_tree[best_v] = True
        pair = (best_u, best_v) if best_u < best_v else (best_v, best_u)
        mst_edges.append(pair)

    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for u, v in mst_edges:
        adj[u].append(v)
        adj[v].append(u)

    b_rows: list[FloatArray] = []
    c_mats: list[FloatArray] = []

    for u, v in mst_edges:
        g_row, c_mat = _bond_derivs(coords, u, v)
        b_rows.append(g_row)
        c_mats.append(c_mat)

    for j in range(n_atoms):
        nbrs = sorted(adj[j])
        for a_idx in range(len(nbrs)):
            for b_idx in range(a_idx + 1, len(nbrs)):
                i_at, k_at = nbrs[a_idx], nbrs[b_idx]
                for g_row, c_mat in _angle_derivs(coords, i_at, j, k_at):
                    b_rows.append(g_row)
                    c_mats.append(c_mat)

    if n_atoms >= 4:
        seen_tors: set[tuple[int, int, int, int]] = set()
        for j, k in mst_edges:
            for u, v in ((j, k), (k, j)):
                for i_at in adj[u]:
                    if i_at == v:
                        continue
                    for l_at in adj[v]:
                        if l_at == u or l_at == i_at:
                            continue
                        canon = min((i_at, u, v, l_at), (l_at, v, u, i_at))
                        if canon in seen_tors:
                            continue
                        seen_tors.add(canon)
                        res = _dihedral_derivs(coords, *canon)
                        if res is not None:
                            b_rows.append(res[0])
                            c_mats.append(res[1])

    # Augment with shortest nonbonded pairs only when MST Z-matrix is rank-deficient
    _, rigid_rank = _rigid_basis_for_coords(coords, m_arr, project_rigid_modes=True)
    target_rank = max(0, dim - rigid_rank)
    if b_rows and target_rank > 0:
        b_tilde_cur = np.asarray(b_rows, dtype=float) * inv_sqrt_m[None, :]
        cur_rank = int(np.linalg.matrix_rank(b_tilde_cur, tol=1.0e-8))
        if cur_rank < target_rank:
            mst_set = set(mst_edges)
            extra_pairs = sorted(
                (dist[a, b], a, b)
                for a in range(n_atoms)
                for b in range(a + 1, n_atoms)
                if (a, b) not in mst_set
            )
            for _, a, b in extra_pairs:
                g_row, c_mat = _bond_derivs(coords, a, b)
                cand_tilde = np.vstack([b_tilde_cur, (g_row * inv_sqrt_m)[None, :]])
                new_rank = int(np.linalg.matrix_rank(cand_tilde, tol=1.0e-8))
                if new_rank > cur_rank:
                    b_rows.append(g_row)
                    c_mats.append(c_mat)
                    b_tilde_cur = cand_tilde
                    cur_rank = new_rank
                    if cur_rank >= target_rank:
                        break

    if not b_rows:
        return hess

    b_mat = np.asarray(b_rows, dtype=float)
    b_tilde = b_mat * inv_sqrt_m[None, :]
    g_tilde = grad * inv_sqrt_m
    g_mat = b_tilde @ b_tilde.T
    g_pinv = np.linalg.pinv(g_mat, rcond=1.0e-10)
    g_q = g_pinv @ (b_tilde @ g_tilde)

    h_curv = hess.copy()
    for coef, c_mat in zip(g_q, c_mats, strict=True):
        if abs(float(coef)) > 1.0e-18:
            h_curv -= float(coef) * c_mat
    return np.asarray(0.5 * (h_curv + h_curv.T), dtype=float)


@dataclass(slots=True)
class _RawStepRecord:
    s_bohr: float
    coords_angstrom: FloatArray
    evaluation: PesEvaluation
    tangent_mw: FloatArray


def integrate_mep(
    pes: PotentialEnergySurface,
    symbols: Sequence[str],
    start_coords_angstrom: npt.ArrayLike,
    *,
    options: MepOptions | None = None,
    masses_amu: Sequence[float] | None = None,
    electronic_states: Sequence[ElectronicState] = (ElectronicState(),),
    symmetry_number: int = 1,
    frequency_scale: float = 1.0,
    frequency_floor_cm1: float | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    reference_energy_kj_mol: float = 0.0,
    reactants: Sequence[Species] = (),
    products: Sequence[Species] = (),
    reactant_atom_indices: Sequence[tuple[int, ...]] | None = None,
    product_atom_indices: Sequence[tuple[int, ...]] | None = None,
    direction_vector_angstrom: npt.ArrayLike | None = None,
) -> MepResult:
    """Integrate the Minimum Energy Path in mass-scaled Cartesian coordinates."""

    opts = options if options is not None else MepOptions()
    sym_tuple = tuple(symbols)
    n_atoms = len(sym_tuple)
    if n_atoms < 2:
        raise ValueError("MEP integration requires at least 2 atoms")
    dim = 3 * n_atoms

    start_coords_ang = np.asarray(start_coords_angstrom, dtype=float)
    if start_coords_ang.shape != (n_atoms, 3):
        raise ValueError(f"start_coords_angstrom must have shape {(n_atoms, 3)}")

    if masses_amu is not None:
        m_arr = np.asarray(masses_amu, dtype=float)
        if m_arr.shape != (n_atoms,) or np.any(m_arr <= 0.0):
            raise ValueError("masses_amu must be positive and match symbols")
    else:
        m_arr = np.asarray(
            [atomic_mass_amu(sym, mass_convention=opts.mass_convention) for sym in sym_tuple],
            dtype=float,
        )

    mu = opts.scale_mass_amu
    coord_masses_amu = np.repeat(m_arr, 3)
    sigma = np.sqrt(coord_masses_amu / mu)

    dir_vec_ang: FloatArray | None = None
    if direction_vector_angstrom is not None:
        dir_vec_ang = np.asarray(direction_vector_angstrom, dtype=float).reshape((n_atoms, 3))

    grad_count = 0
    hess_count = 0

    def _eval_grad(x_mw: FloatArray) -> tuple[PesEvaluation, FloatArray]:
        nonlocal grad_count
        coords_ang = (x_mw / sigma).reshape((n_atoms, 3)) * BOHR_ANGSTROM
        ev = pes.evaluate(
            sym_tuple,
            coords_ang,
            need_gradient=True,
            need_hessian=False,
            charge=charge,
            multiplicity=multiplicity,
            hessian_step_bohr=opts.hessian_step_bohr,
            hessian_difference=opts.hessian_difference,
        )
        grad_count += 1
        assert ev.gradient_hartree_bohr is not None
        g_mw = np.asarray(ev.gradient_hartree_bohr.reshape(-1) / sigma, dtype=np.float64)
        return ev, g_mw

    def _eval_hess(x_mw: FloatArray) -> tuple[PesEvaluation, FloatArray, FloatArray]:
        nonlocal grad_count, hess_count
        coords_ang = (x_mw / sigma).reshape((n_atoms, 3)) * BOHR_ANGSTROM
        ev = pes.evaluate(
            sym_tuple,
            coords_ang,
            need_gradient=True,
            need_hessian=True,
            charge=charge,
            multiplicity=multiplicity,
            hessian_step_bohr=opts.hessian_step_bohr,
            hessian_difference=opts.hessian_difference,
        )
        grad_count += 1
        hess_count += 1
        assert ev.gradient_hartree_bohr is not None and ev.hessian_hartree_bohr2 is not None
        g_mw = np.asarray(ev.gradient_hartree_bohr.reshape(-1) / sigma, dtype=np.float64)
        h_mw = np.asarray(ev.hessian_hartree_bohr2 / np.outer(sigma, sigma), dtype=np.float64)
        return ev, g_mw, h_mw

    start_bohr = start_coords_ang / BOHR_ANGSTROM
    x0_mw = start_bohr.reshape(-1) * sigma

    start_atoms = tuple(
        Atom(
            symbol=sym_tuple[i],
            x_angstrom=float(start_coords_ang[i, 0]),
            y_angstrom=float(start_coords_ang[i, 1]),
            z_angstrom=float(start_coords_ang[i, 2]),
            mass_amu=float(m_arr[i]),
            mass_convention=opts.mass_convention,
        )
        for i in range(n_atoms)
    )
    rotor_type = classify_rotor(principal_moments_amu_a2(start_atoms), n_atoms)

    ev0, g0_mw, h0_mw = _eval_hess(x0_mw)
    basis0, rank0 = _rigid_basis_for_coords(
        start_bohr, m_arr, project_rigid_modes=opts.project_rigid_modes
    )
    proj_h0 = project_symmetric_matrix(h0_mw, basis0)
    evals0, evecs0 = np.linalg.eigh(proj_h0)
    overlaps0 = np.sum((basis0.T @ evecs0) ** 2, axis=0) if rank0 > 0 else np.zeros(dim)
    n_int0 = dim - rank0
    int_idx0 = np.argsort(overlaps0)[:n_int0]
    int_evals0 = evals0[int_idx0]
    int_evecs0 = evecs0[:, int_idx0]
    int_freqs0 = _eig_to_cm1(int_evals0, mu)

    save_every = opts.hessian_stride * opts.save_stride

    if opts.start_from_saddle:
        neg_positions = [i for i, f in enumerate(int_freqs0) if f < -opts.zero_threshold_cm1]
        if not neg_positions:
            raise ValueError(
                "start_from_saddle=True requires a saddle point with an imaginary frequency"
            )
        if len(neg_positions) > 1:
            raise ValueError(
                f"start_from_saddle=True found {len(neg_positions)} imaginary modes; expected 1"
            )
        ts_pos = int(np.argmin(int_evals0))
        lam0 = float(int_evals0[ts_pos])
        t0_raw = int_evecs0[:, ts_pos]
        sign0 = _determine_tangent_sign(
            t0_raw,
            start_bohr,
            sigma,
            sym_tuple,
            direction_sign=opts.direction_sign,
            direction_vector_angstrom=dir_vec_ang,
            reactants=reactants,
            products=products,
            reactant_atom_indices=reactant_atom_indices,
            product_atom_indices=product_atom_indices,
        )
        t0_mw = t0_raw * sign0

        # Compute Page-McIver 2nd-order saddle curvature x''(0) from symmetric +-eps gradients
        eps_curv = min(2.0e-3, 0.2 * opts.step_size_bohr)
        _, gp_curv = _eval_grad(x0_mw + eps_curv * t0_mw)
        _, gm_curv = _eval_grad(x0_mw - eps_curv * t0_mw)
        f_3rd = (gp_curv + gm_curv) / (eps_curv * eps_curv)
        b0_plus = np.column_stack([basis0, t0_mw]) if rank0 > 0 else t0_mw.reshape((-1, 1))
        f_perp = f_3rd - b0_plus @ (b0_plus.T @ f_3rd)
        xpp0 = np.zeros(dim, dtype=float)
        for m in range(n_int0):
            if m == ts_pos:
                continue
            vm = int_evecs0[:, m]
            denom = 2.0 * lam0 - float(int_evals0[m])
            if abs(denom) > 1.0e-12:
                xpp0 += (float(np.dot(vm, f_perp)) / denom) * vm

        h0_kj = (
            ev0.hessian_kj_mol_angstrom2
            if ev0.hessian_kj_mol_angstrom2 is not None
            else np.zeros((dim, dim))
        )
        ts_nm = normal_modes(
            h0_kj,
            m_arr.tolist(),
            coordinates_angstrom=start_coords_ang,
            hessian_unit="kj/mol/angstrom^2",
            remove_rigid_modes=opts.project_rigid_modes,
            zero_threshold_cm1=opts.zero_threshold_cm1,
        )
        ts_freqs_sorted = sorted(float(f) for f in ts_nm.frequencies_cm1)
        ts_pos_f = [f for f in reversed(ts_freqs_sorted) if f > 0.0]
        ts_neg_f = [f for f in ts_freqs_sorted if f < 0.0]
        ts_freqs = tuple(ts_pos_f + ts_neg_f)
        transition_state = Species(
            name="transition-state",
            atoms=start_atoms,
            electronic_energy_kj_mol=ev0.energy_kj_mol - reference_energy_kj_mol,
            frequencies_cm1=ts_freqs,
            symmetry_number=symmetry_number,
            electronic_states=tuple(electronic_states),
            rotor_type=rotor_type,
            frequency_scale=frequency_scale,
            frequency_floor_cm1=frequency_floor_cm1,
        )
    else:
        g0_proj = g0_mw - basis0 @ (basis0.T @ g0_mw)
        g0_norm = float(np.linalg.norm(g0_proj))
        if g0_norm <= 0.0:
            raise ValueError("start_from_saddle=False requires non-zero initial gradient")
        branch0 = 1.0 if opts.s_max_bohr > 0.0 else -1.0
        t0_mw = (-g0_proj / g0_norm) * branch0
        xpp0 = np.zeros(dim, dtype=float)
        transition_state = Species(
            name="start",
            atoms=start_atoms,
            electronic_energy_kj_mol=ev0.energy_kj_mol - reference_energy_kj_mol,
            frequencies_cm1=tuple(
                sorted(
                    (float(f) for f in int_freqs0 if f >= opts.zero_threshold_cm1),
                    reverse=True,
                )
            ),
            symmetry_number=symmetry_number,
            electronic_states=tuple(electronic_states),
            rotor_type=rotor_type,
            frequency_scale=frequency_scale,
            frequency_floor_cm1=frequency_floor_cm1,
        )

    def _single_substep(
        x_cur: FloatArray,
        ev_cur: PesEvaluation,
        g_cur: FloatArray,
        h_ref: FloatArray,
        h_step: float,
    ) -> tuple[FloatArray, PesEvaluation, FloatArray, FloatArray]:
        coords_cur_b = (x_cur / sigma).reshape((n_atoms, 3))
        basis_cur, _ = _rigid_basis_for_coords(
            coords_cur_b, m_arr, project_rigid_modes=opts.project_rigid_modes
        )
        g_proj = g_cur - basis_cur @ (basis_cur.T @ g_cur)
        g_norm = float(np.linalg.norm(g_proj))
        if g_norm <= 1.0e-14:
            return x_cur.copy(), ev_cur, g_cur, np.zeros(dim, dtype=float)

        if opts.method == "euler":
            d_unit = -g_proj / g_norm
            x_next = x_cur + h_step * d_unit
            ev_next, g_next = _eval_grad(x_next)
            return x_next, ev_next, g_next, d_unit

        if opts.method == "es1":
            # Adaptive Euler stabilization: check turning angle; if it exceeds threshold,
            # subdivide [0, h_step] with adaptive step-halving Heun substeps.
            max_cos = math.cos(math.radians(opts.stabilization_angle_deg))
            d0 = -g_proj / g_norm
            x_trial = x_cur + h_step * d0
            ev_trial, g_trial = _eval_grad(x_trial)
            b_t, _ = _rigid_basis_for_coords(
                (x_trial / sigma).reshape((n_atoms, 3)),
                m_arr,
                project_rigid_modes=opts.project_rigid_modes,
            )
            gt_proj = g_trial - b_t @ (b_t.T @ g_trial)
            gt_norm = float(np.linalg.norm(gt_proj))
            if gt_norm <= 1.0e-14:
                return x_trial, ev_trial, g_trial, d0
            dt = -gt_proj / gt_norm
            if float(np.dot(d0, dt)) >= max_cos:
                return x_trial, ev_trial, g_trial, dt

            min_sub_h = h_step / float(max(2, opts.stabilization_substeps))
            sub_h = 0.5 * h_step
            rem_h = h_step
            x_sub = x_cur.copy()
            ev_sub = ev_cur
            g_sub = g_cur
            d_last = dt
            while rem_h > 1.0e-12:
                try_h = min(sub_h, rem_h)
                b_s, _ = _rigid_basis_for_coords(
                    (x_sub / sigma).reshape((n_atoms, 3)),
                    m_arr,
                    project_rigid_modes=opts.project_rigid_modes,
                )
                gp_s = g_sub - b_s @ (b_s.T @ g_sub)
                gn_s = float(np.linalg.norm(gp_s))
                if gn_s <= 1.0e-14:
                    break
                d_s = -gp_s / gn_s
                x_eul = x_sub + try_h * d_s
                ev_eul, g_eul = _eval_grad(x_eul)
                b_e, _ = _rigid_basis_for_coords(
                    (x_eul / sigma).reshape((n_atoms, 3)),
                    m_arr,
                    project_rigid_modes=opts.project_rigid_modes,
                )
                gp_e = g_eul - b_e @ (b_e.T @ g_eul)
                gn_e = float(np.linalg.norm(gp_e))
                d_e = (-gp_e / gn_e) if gn_e > 1.0e-14 else d_s
                if (
                    float(np.dot(d_s, d_e)) < max_cos
                    and try_h > min_sub_h * (1.0 + 1.0e-9)
                ):
                    sub_h = max(0.5 * try_h, min_sub_h)
                    continue
                d_avg = d_s + d_e
                d_avg_n = float(np.linalg.norm(d_avg))
                d_step = (d_avg / d_avg_n) if d_avg_n > 1.0e-14 else d_s
                x_sub = x_sub + try_h * d_step
                ev_sub, g_sub = _eval_grad(x_sub)
                b_n, _ = _rigid_basis_for_coords(
                    (x_sub / sigma).reshape((n_atoms, 3)),
                    m_arr,
                    project_rigid_modes=opts.project_rigid_modes,
                )
                gp_n = g_sub - b_n @ (b_n.T @ g_sub)
                gn_n = float(np.linalg.norm(gp_n))
                d_last = (-gp_n / gn_n) if gn_n > 1.0e-14 else d_step
                rem_h -= try_h
            return x_sub, ev_sub, g_sub, d_last

        # Default: "pm" (Page-McIver LQA)
        proj_h = project_symmetric_matrix(h_ref, basis_cur)
        evals, evecs = np.linalg.eigh(proj_h)
        delta_x, end_tan, _ = page_mciver_lqa_step(g_proj, evals, evecs, h_step)
        if float(np.linalg.norm(delta_x)) <= 1.0e-15:
            delta_x = h_step * (-g_proj / g_norm)
            end_tan = -g_proj / g_norm
        x_next = x_cur + delta_x
        ev_next, g_next = _eval_grad(x_next)
        return x_next, ev_next, g_next, end_tan

    def _integrate_branch(branch_sign: int) -> list[_RawStepRecord]:
        target_len = opts.s_max_bohr if branch_sign > 0 else abs(opts.s_min_bohr)
        if target_len <= 0.0:
            return []

        records: list[_RawStepRecord] = []
        s_acc = 0.0
        step_idx = 0

        if opts.start_from_saddle:
            delta_init = (
                float(opts.saddle_step_bohr)
                if opts.saddle_step_bohr is not None
                else min(5.0e-4, 0.2 * opts.step_size_bohr)
            )
            delta_init = min(delta_init, target_len, opts.step_size_bohr)
            dx_init = (
                (float(branch_sign) * delta_init) * t0_mw
                + 0.5 * (delta_init * delta_init) * xpp0
            )
            x_cur = x0_mw + dx_init
            s_acc = delta_init

            # Finish first step_size_bohr interval cleanly when delta_init < step_size_bohr
            first_target = min(opts.step_size_bohr, target_len)
            rem = first_target - delta_init
            if rem > 1.0e-12:
                ev_tmp, g_tmp = _eval_grad(x_cur)
                x_cur, ev_cur, g_cur, _ = _single_substep(x_cur, ev_tmp, g_tmp, h0_mw, rem)
                s_acc = first_target
            step_idx = 1
            h_cached = h0_mw
            if step_idx % save_every == 0 or abs(s_acc - target_len) <= 1.0e-10:
                ev_cur, g_cur, h_cur = _eval_hess(x_cur)
                h_cached = h_cur
                records.append(
                    _RawStepRecord(
                        s_bohr=float(branch_sign) * s_acc,
                        coords_angstrom=(x_cur / sigma).reshape((n_atoms, 3)) * BOHR_ANGSTROM,
                        evaluation=ev_cur,
                        tangent_mw=np.zeros(dim, dtype=float),  # populated in post-pass
                    )
                )
            else:
                ev_cur, g_cur = _eval_grad(x_cur)
        else:
            x_cur = x0_mw.copy()
            ev_cur = ev0
            g_cur = g0_mw
            h_cached = h0_mw

        while s_acc + 1.0e-10 < target_len and step_idx < opts.max_steps:
            # Check gradient convergence away from saddle
            coords_b = (x_cur / sigma).reshape((n_atoms, 3))
            b_c, _ = _rigid_basis_for_coords(
                coords_b, m_arr, project_rigid_modes=opts.project_rigid_modes
            )
            gp = g_cur - b_c @ (b_c.T @ g_cur)
            gp_norm = float(np.linalg.norm(gp))
            if gp_norm < opts.gradient_tolerance_au and s_acc > 0.05:
                break

            step_h = min(opts.step_size_bohr, target_len - s_acc)
            if opts.method == "pm" and step_idx % opts.hessian_stride == 0 and step_idx >= 1:
                if not (records and abs(abs(records[-1].s_bohr) - s_acc) <= 1.0e-10):
                    ev_cur, g_cur, h_cached = _eval_hess(x_cur)

            x_next, ev_next, g_next, _ = _single_substep(x_cur, ev_cur, g_cur, h_cached, step_h)
            x_cur = x_next
            ev_cur = ev_next
            g_cur = g_next
            s_acc += step_h
            step_idx += 1

            is_end = s_acc + 1.0e-10 >= target_len
            if step_idx % save_every == 0 or is_end:
                ev_cur, g_cur, h_cur = _eval_hess(x_cur)
                h_cached = h_cur
                records.append(
                    _RawStepRecord(
                        s_bohr=float(branch_sign) * s_acc,
                        coords_angstrom=(x_cur / sigma).reshape((n_atoms, 3)) * BOHR_ANGSTROM,
                        evaluation=ev_cur,
                        tangent_mw=np.zeros(dim, dtype=float),
                    )
                )

        # Ensure if terminated early by gradient norm, final point is saved with Hessian
        if records:
            if abs(abs(records[-1].s_bohr) - s_acc) > 1.0e-10:
                ev_cur, g_cur, h_cur = _eval_hess(x_cur)
                records.append(
                    _RawStepRecord(
                        s_bohr=float(branch_sign) * s_acc,
                        coords_angstrom=(x_cur / sigma).reshape((n_atoms, 3)) * BOHR_ANGSTROM,
                        evaluation=ev_cur,
                        tangent_mw=np.zeros(dim, dtype=float),
                    )
                )
        return records

    raw_records: list[_RawStepRecord] = []
    if opts.start_from_saddle:
        rev_records = _integrate_branch(-1)
        fwd_records = _integrate_branch(+1)
        raw_records.extend(reversed(rev_records))
        raw_records.append(
            _RawStepRecord(
                s_bohr=0.0,
                coords_angstrom=start_coords_ang.copy(),
                evaluation=ev0,
                tangent_mw=t0_mw.copy(),
            )
        )
        raw_records.extend(fwd_records)
    else:
        raw_records.append(
            _RawStepRecord(
                s_bohr=0.0,
                coords_angstrom=start_coords_ang.copy(),
                evaluation=ev0,
                tangent_mw=t0_mw.copy(),
            )
        )
        branch = 1 if opts.s_max_bohr > 0.0 else -1
        fwd_records = _integrate_branch(branch)
        if branch < 0:
            raw_records = list(reversed(fwd_records)) + raw_records
        else:
            raw_records.extend(fwd_records)

    raw_records.sort(key=lambda r: r.s_bohr)
    if len(raw_records) < 3:
        raise ValueError(
            f"MEP produced only {len(raw_records)} saved points; "
            "reduce step_size_bohr or expand s_range"
        )

    # Orient tangents consistently from reactant side (low s) to product side (high s)
    idx0 = min(range(len(raw_records)), key=lambda i: abs(raw_records[i].s_bohr))
    tangents: list[FloatArray] = []
    for _i, rec in enumerate(raw_records):
        if opts.start_from_saddle and abs(rec.s_bohr) <= 1.0e-12:
            tangents.append(t0_mw.copy())
            continue
        assert rec.evaluation.gradient_hartree_bohr is not None
        g_mw = rec.evaluation.gradient_hartree_bohr.reshape(-1) / sigma
        b_rig, _ = _rigid_basis_for_coords(
            rec.coords_angstrom / BOHR_ANGSTROM, m_arr, project_rigid_modes=opts.project_rigid_modes
        )
        g_proj = g_mw - b_rig @ (b_rig.T @ g_mw)
        gn = float(np.linalg.norm(g_proj))
        if gn > 1.0e-14:
            # For s < 0 on saddle branch, +gradient points uphill toward s=0 (+s direction).
            # For s > 0, -gradient points downhill away from s=0 (+s direction).
            y_vec = (g_proj / gn) if rec.s_bohr < 0.0 else (-g_proj / gn)
        else:
            y_vec = t0_mw.copy()
        tangents.append(y_vec)

    # Smoothness sign guard stepping outward from idx0
    for i in range(idx0 + 1, len(tangents)):
        if float(np.dot(tangents[i], tangents[i - 1])) < 0.0:
            tangents[i] = -tangents[i]
    for i in range(idx0 - 1, -1, -1):
        if float(np.dot(tangents[i], tangents[i + 1])) < 0.0:
            tangents[i] = -tangents[i]

    # Projected vibrational frequencies and eigenvectors at each saved point
    freqs_per_pt: list[tuple[float, ...]] = []
    modes_per_pt: list[FloatArray] = []
    sigma_outer = np.outer(sigma, sigma)
    for i, rec in enumerate(raw_records):
        assert rec.evaluation.hessian_hartree_bohr2 is not None
        assert rec.evaluation.gradient_hartree_bohr is not None
        coords_b = rec.coords_angstrom / BOHR_ANGSTROM
        if opts.coordinate_system == "curvilinear":
            hess_cart_au = curvilinear_corrected_hessian(
                coords_b,
                rec.evaluation.gradient_hartree_bohr,
                rec.evaluation.hessian_hartree_bohr2,
                m_arr,
            )
        else:
            hess_cart_au = rec.evaluation.hessian_hartree_bohr2
        h_mw = hess_cart_au / sigma_outer
        b_rig, _ = _rigid_basis_for_coords(
            coords_b, m_arr, project_rigid_modes=opts.project_rigid_modes
        )
        y_vec = tangents[i]
        y_perp = y_vec - b_rig @ (b_rig.T @ y_vec)
        yp_n = float(np.linalg.norm(y_perp))
        if yp_n > 1.0e-12:
            y_perp /= yp_n
            tangents[i] = y_perp
            b_proj = (
                np.column_stack([b_rig, y_perp])
                if b_rig.shape[1] > 0
                else y_perp.reshape((-1, 1))
            )
        else:
            b_proj = b_rig
        pt_freqs_arr, pt_evecs = _projected_vibrational_modes(h_mw, b_proj, scale_mass_amu=mu)
        freqs_per_pt.append(tuple(float(v) for v in pt_freqs_arr))
        modes_per_pt.append(pt_evecs)

    # Sign-track vibrational eigenvectors continuously outward from idx0
    if opts.mode_tracking and modes_per_pt[idx0].shape[1] > 0:
        for i in range(idx0 + 1, len(raw_records)):
            prev_m = modes_per_pt[i - 1]
            cur_m = modes_per_pt[i].copy()
            if cur_m.shape == prev_m.shape:
                used_prev: set[int] = set()
                for m in range(cur_m.shape[1]):
                    dots = np.abs(prev_m.T @ cur_m[:, m])
                    best_p = max(
                        (p for p in range(prev_m.shape[1]) if p not in used_prev),
                        key=lambda p: float(dots[p]),
                        default=m,
                    )
                    used_prev.add(best_p)
                    if float(np.dot(prev_m[:, best_p], cur_m[:, m])) < 0.0:
                        cur_m[:, m] = -cur_m[:, m]
            modes_per_pt[i] = cur_m

        for i in range(idx0 - 1, -1, -1):
            next_m = modes_per_pt[i + 1]
            cur_m = modes_per_pt[i].copy()
            if cur_m.shape == next_m.shape:
                used_next: set[int] = set()
                for m in range(cur_m.shape[1]):
                    dots = np.abs(next_m.T @ cur_m[:, m])
                    best_n = max(
                        (p for p in range(next_m.shape[1]) if p not in used_next),
                        key=lambda p: float(dots[p]),
                        default=m,
                    )
                    used_next.add(best_n)
                    if float(np.dot(next_m[:, best_n], cur_m[:, m])) < 0.0:
                        cur_m[:, m] = -cur_m[:, m]
            modes_per_pt[i] = cur_m

    # Compute RPHB reaction-path curvatures B_mF(s) = -<dy/ds, L_m(s)>
    s_arr = np.asarray([r.s_bohr for r in raw_records], dtype=float)
    y_arr = np.asarray(tangents, dtype=float)
    dyds = np.gradient(y_arr, s_arr, axis=0)

    point_data_list: list[MepPointData] = []
    path_points: list[PathPoint] = []
    for i, rec in enumerate(raw_records):
        pt_freqs = freqs_per_pt[i]
        pt_modes = modes_per_pt[i]
        if opts.curvature_method == "none" or pt_modes.shape[1] == 0:
            curvatures: tuple[float, ...] = ()
        else:
            # Align curvatures strictly with positive frequencies
            # (descending order matching tunneling.py).
            pos_indices = [m for m, f in enumerate(pt_freqs) if f > 0.0]
            curvatures = tuple(
                float(-np.dot(dyds[i], pt_modes[:, m])) for m in pos_indices
            )

        pt_atoms = tuple(
            Atom(
                symbol=sym_tuple[a],
                x_angstrom=float(rec.coords_angstrom[a, 0]),
                y_angstrom=float(rec.coords_angstrom[a, 1]),
                z_angstrom=float(rec.coords_angstrom[a, 2]),
                mass_amu=float(m_arr[a]),
                mass_convention=opts.mass_convention,
            )
            for a in range(n_atoms)
        )
        pt_species = Species(
            name="ts" if abs(rec.s_bohr) <= 1.0e-12 else f"mep-{i + 1}",
            atoms=pt_atoms,
            electronic_energy_kj_mol=rec.evaluation.energy_kj_mol - reference_energy_kj_mol,
            frequencies_cm1=pt_freqs,
            symmetry_number=1,
            electronic_states=tuple(electronic_states),
            rotor_type=rotor_type,
            frequency_scale=frequency_scale,
            frequency_floor_cm1=frequency_floor_cm1,
        )
        pp = PathPoint(
            coordinate=rec.s_bohr,
            dividing_surface=pt_species,
            curvatures_bohr_inv=curvatures,
        )
        path_points.append(pp)

        assert rec.evaluation.gradient_hartree_bohr is not None
        assert rec.evaluation.hessian_hartree_bohr2 is not None
        point_data_list.append(
            MepPointData(
                coordinate_bohr=rec.s_bohr,
                coords_angstrom=rec.coords_angstrom,
                energy_hartree=rec.evaluation.energy_hartree,
                gradient_hartree_bohr=rec.evaluation.gradient_hartree_bohr,
                hessian_hartree_bohr2=rec.evaluation.hessian_hartree_bohr2,
                tangent_mw=tangents[i],
                frequencies_cm1=pt_freqs,
                curvatures_bohr_inv=curvatures,
                mode_vectors_mw=pt_modes,
                species=pt_species,
                path_point=pp,
            )
        )

    reaction_path = ReactionPath(
        points=tuple(path_points),
        coordinate_unit="bohr",
        reduced_mass_amu=mu,
    )
    return MepResult(
        symbols=sym_tuple,
        masses_amu=tuple(float(m) for m in m_arr),
        scale_mass_amu=mu,
        rotor_type=rotor_type,
        points=tuple(point_data_list),
        reaction_path=reaction_path,
        transition_state=transition_state,
        saddle_evaluation=ev0,
        method=opts.method,
        hessian_evaluations=hess_count,
        gradient_evaluations=grad_count,
        metadata={
            "s_min_bohr": float(s_arr[0]),
            "s_max_bohr": float(s_arr[-1]),
            "step_size_bohr": opts.step_size_bohr,
            "hessian_stride": opts.hessian_stride,
            "coordinate_system": opts.coordinate_system,
        },
    )
