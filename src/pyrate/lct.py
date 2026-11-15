"""Polyrate-compatible Large-Curvature Tunneling (LCG3 / LCG4) and true muOMT.

Implements the straight-line corner-cutting Large-Curvature Ground-State /
Excited-State approximations Version 3 (LCG3) and Version 4 (LCG4) together
with Microcanonical Optimized Multidimensional Tunneling (muOMT):

- Garrett, Truhlar, Grev & Magnuson, *J. Phys. Chem.* **1980**, *84*, 1730
- Truhlar, Isaacson & Garrett, in *Theory of Chemical Reaction Dynamics*, Vol. 4 (1985)
- Liu, Lu, Gonzalez-Lafont, Truhlar & Garrett, *J. Am. Chem. Soc.* **1993**, *115*, 7806 (muOMT)
- Fernandez-Ramos & Truhlar, *J. Chem. Phys.* **2001**, *114*, 1491 (LCG4)
- Zheng et al., *POLYRATE 2023* (`polyhl.F`: `KAPVA`, `LCG34`, `LCSET`, `LCSTX`,
  `LCNWTN`, `LCZCRT`, `LCVCOR`, `LCGNAD`, `LCTV4`, `LCVEF`, `LCGTHE`, `LCSINT`,
  `LCPROB`, `LCGSRT`; `polymq.F`: `NSTATE`, `OVRLP`; `polyag.F`: `BOLTZ`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .constants import (
    AMU_KG,
    AMU_TO_AU_MASS,
    AVOGADRO_MOL,
    BOHR_ANGSTROM,
    BOHR_M,
    HARTREE_J,
    HARTREE_KJ_MOL,
    WAVENUMBER_TO_J,
)
from .models import LctOptions, Reaction
from .native import (
    gauss_legendre_quadrature,
    lct_integrate_amplitudes,
    lct_uniformize_probabilities,
    neville_interpolate,
    thermal_boltz_probabilities_kappa,
)
from .normal_modes import _rigid_basis
from .partition import zero_point_energy_kj_mol

FloatArray = npt.NDArray[np.float64]

_TOL_UALF_BOHR = 1.0e-7
_TOL_DOTP = 1.0e-8
_OFF_GRID_THETA = 100.0


@dataclass(frozen=True, slots=True)
class LctResult:
    """Precomputed LCT (LCG3 or LCG4) and OMT energy-resolved tunneling probabilities."""

    version: int
    threshold_kj_mol: float
    barrier_kj_mol: float
    energies_kj_mol: tuple[float, ...]
    weights_kj_mol: tuple[float, ...]
    probabilities_lct: tuple[float, ...]
    probabilities_zct: tuple[float, ...]
    probabilities_sct: tuple[float, ...]
    probabilities_muomt: tuple[float, ...]
    max_excited_state: int
    state_uniformized_probabilities: tuple[tuple[float, ...], ...]


def can_compute_lct(reaction: Reaction) -> bool:
    """Return True when ``reaction`` carries atomistic reaction-path geometries for LCT."""

    path = reaction.path
    if path is None or len(path.points) < 3:
        return False
    atom_counts = [len(pt.dividing_surface.atoms) for pt in path.points]
    if any(count < 2 for count in atom_counts) or len(set(atom_counts)) != 1:
        return False
    return any(len(pt.dividing_surface.positive_frequencies_cm1) > 0 for pt in path.points)


class _NaturalCubicSpline:
    """Natural cubic spline S_i(x) = d_i + c_i*dx + b_i*dx^2 + a_i*dx^3 on [x_i, x_{i+1}]."""

    def __init__(self, x: FloatArray, y: FloatArray) -> None:
        n = len(x)
        if n < 2:
            raise ValueError("cubic spline requires at least 2 points")
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.n = n
        if n == 2:
            h0 = float(self.x[1] - self.x[0])
            slope = float(self.y[1] - self.y[0]) / h0 if h0 > 0.0 else 0.0
            self.a = np.zeros(1, dtype=float)
            self.b = np.zeros(1, dtype=float)
            self.c = np.asarray([slope], dtype=float)
            self.d = self.y[:-1].copy()
            return

        h = np.diff(self.x)
        alpha = np.zeros(n, dtype=float)
        for i in range(1, n - 1):
            alpha[i] = (
                3.0 / h[i] * (self.y[i + 1] - self.y[i])
                - 3.0 / h[i - 1] * (self.y[i] - self.y[i - 1])
            )

        l_diag = np.ones(n, dtype=float)
        mu = np.zeros(n, dtype=float)
        z = np.zeros(n, dtype=float)
        for i in range(1, n - 1):
            l_diag[i] = 2.0 * (self.x[i + 1] - self.x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l_diag[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l_diag[i]

        b_full = np.zeros(n, dtype=float)
        c = np.zeros(n - 1, dtype=float)
        a = np.zeros(n - 1, dtype=float)
        for j in range(n - 2, -1, -1):
            b_full[j] = z[j] - mu[j] * b_full[j + 1]
            c[j] = (
                (self.y[j + 1] - self.y[j]) / h[j]
                - h[j] * (b_full[j + 1] + 2.0 * b_full[j]) / 3.0
            )
            a[j] = (b_full[j + 1] - b_full[j]) / (3.0 * h[j])

        self.a = a
        self.b = b_full[:-1]
        self.c = c
        self.d = self.y[:-1].copy()

    def evaluate(self, x_val: float) -> float:
        if x_val <= self.x[0]:
            idx = 0
        elif x_val >= self.x[-1]:
            idx = self.n - 2
        else:
            idx = int(np.searchsorted(self.x, x_val, side="right")) - 1
            idx = max(0, min(self.n - 2, idx))
        dx = x_val - float(self.x[idx])
        return float(
            self.d[idx] + dx * (self.c[idx] + dx * (self.b[idx] + dx * self.a[idx]))
        )

    def derivative(self, x_val: float) -> float:
        if x_val <= self.x[0]:
            idx = 0
        elif x_val >= self.x[-1]:
            idx = self.n - 2
        else:
            idx = int(np.searchsorted(self.x, x_val, side="right")) - 1
            idx = max(0, min(self.n - 2, idx))
        dx = x_val - float(self.x[idx])
        return float(self.c[idx] + dx * (2.0 * self.b[idx] + 3.0 * self.a[idx] * dx))

    def maximum(self) -> tuple[float, float]:
        best_idx = int(np.argmax(self.y))
        best_x = float(self.x[best_idx])
        best_y = float(self.y[best_idx])
        for i in range(self.n - 1):
            h_i = float(self.x[i + 1] - self.x[i])
            aa = 3.0 * float(self.a[i])
            bb = 2.0 * float(self.b[i])
            cc = float(self.c[i])
            candidates: list[float] = []
            if abs(aa) <= 1.0e-18:
                if abs(bb) > 1.0e-18:
                    candidates.append(-cc / bb)
            else:
                disc = bb * bb - 4.0 * aa * cc
                if disc >= 0.0:
                    root_disc = math.sqrt(disc)
                    candidates.append((-bb - root_disc) / (2.0 * aa))
                    candidates.append((-bb + root_disc) / (2.0 * aa))
            for dx in candidates:
                if 0.0 <= dx <= h_i:
                    val = float(
                        self.d[i] + dx * (self.c[i] + dx * (self.b[i] + dx * self.a[i]))
                    )
                    if val > best_y:
                        best_y = val
                        best_x = float(self.x[i]) + dx
        return best_x, best_y

    def _solve_on_segment(self, seg_idx: int, target: float) -> float:
        x_lo = float(self.x[seg_idx])
        x_hi = float(self.x[seg_idx + 1])
        f_lo = self.evaluate(x_lo) - target
        f_hi = self.evaluate(x_hi) - target
        if abs(f_lo) <= 1.0e-15:
            return x_lo
        if abs(f_hi) <= 1.0e-15:
            return x_hi
        lo, hi = x_lo, x_hi
        flo = f_lo
        mid = 0.5 * (lo + hi)
        for _ in range(60):
            f_mid = self.evaluate(mid) - target
            if abs(f_mid) <= 1.0e-14 or abs(hi - lo) <= 1.0e-13:
                return mid
            if (flo < 0.0 and f_mid < 0.0) or (flo > 0.0 and f_mid > 0.0):
                lo = mid
                flo = f_mid
            else:
                hi = mid
            df = self.derivative(mid)
            if abs(df) > 1.0e-14:
                nxt = mid - f_mid / df
                if lo < nxt < hi:
                    mid = nxt
                    continue
            mid = 0.5 * (lo + hi)
        return mid

    def outer_turning_points(self, energy: float, s_max: float) -> tuple[float, float]:
        """Find outermost reactant (<= s_max) and product (>= s_max) turning points."""

        # Left (reactant) turning point: scan from x[0] upward toward s_max, or from s_max outward?
        # Polyrate PSAG / LCTP (polyhl.F:3503-3520): scans from the asymptotic end inward
        # so pre-reactive van der Waals wells do not create premature inner turning points.
        if energy <= float(self.y[0]):
            stp0 = float(self.x[0])
        else:
            stp0 = s_max
            for i in range(self.n - 1):
                if float(self.x[i]) >= s_max:
                    break
                y1 = float(self.y[i])
                y2 = float(self.y[i + 1])
                if (y1 - energy) * (y2 - energy) <= 0.0 and max(y1, y2) >= energy:
                    cand = self._solve_on_segment(i, energy)
                    if cand <= s_max + 1.0e-12:
                        stp0 = min(cand, s_max)
                        break

        if energy <= float(self.y[-1]):
            stp1 = float(self.x[-1])
        else:
            stp1 = s_max
            for i in range(self.n - 2, -1, -1):
                if float(self.x[i + 1]) <= s_max:
                    break
                y1 = float(self.y[i])
                y2 = float(self.y[i + 1])
                if (y1 - energy) * (y2 - energy) <= 0.0 and max(y1, y2) >= energy:
                    cand = self._solve_on_segment(i, energy)
                    if cand >= s_max - 1.0e-12:
                        stp1 = max(cand, s_max)
                        break

        return stp0, stp1


def _locate_and_lagrange4(ss: FloatArray, stp: float) -> tuple[int, tuple[float, ...]]:
    """Return (jnl, weights) matching Polyrate LOCATE + 4-point INTRPL (polyhl.F:1155)."""

    ns = len(ss)
    if ns == 1:
        return 0, (1.0,)
    jj = int(np.searchsorted(ss, stp, side="right")) - 1
    jj = max(0, min(ns - 2, jj))
    nn = min(4, ns)
    jnl = jj - (nn // 2 - 1)
    if jnl < 0:
        jnl = 0
    if jnl + nn > ns:
        jnl = ns - nn
    xa = ss[jnl : jnl + nn]
    weights: list[float] = []
    for a in range(nn):
        xa_val = float(xa[a])
        num = 1.0
        den = 1.0
        for b in range(nn):
            if a == b:
                continue
            xb_val = float(xa[b])
            num *= stp - xb_val
            den *= xa_val - xb_val
        weights.append(num / den if abs(den) > 1.0e-30 else 0.0)
    return jnl, tuple(weights)


def _interp_1d(arr: FloatArray, jnl: int, weights: tuple[float, ...]) -> float:
    total = 0.0
    for offset, w in enumerate(weights):
        total += w * float(arr[jnl + offset])
    return total


def _interp_vec(arr: FloatArray, jnl: int, weights: tuple[float, ...]) -> FloatArray:
    out = np.zeros(arr.shape[1], dtype=float)
    for offset, w in enumerate(weights):
        out += w * arr[jnl + offset]
    return out


def _interp_mat(arr: FloatArray, jnl: int, weights: tuple[float, ...]) -> FloatArray:
    out = np.zeros(arr.shape[1:], dtype=float)
    for offset, w in enumerate(weights):
        out += w * arr[jnl + offset]
    return out


@dataclass(slots=True)
class _PreparedMep:
    n_atoms: int
    dim: int
    n_sd: int
    n_vib: int
    redm: float
    alpha: FloatArray
    s_bohr: FloatArray
    geom: FloatArray  # (n_sd, dim) mass-scaled COM-centered bohr
    dxsv: FloatArray  # (n_sd, dim) unit tangent pointing toward +s
    cofsv: FloatArray  # (n_sd, n_vib, dim) orthonormal transverse modes
    wets: FloatArray  # (n_sd, n_vib) frequencies in hartree
    bcur: FloatArray  # (n_sd, n_vib) signed curvatures in bohr^-1
    vclas: FloatArray  # (n_sd,) classical potential in hartree
    vadib: FloatArray  # (n_sd,) ground adiabatic potential in hartree
    var: float
    vap: float
    ez: float
    iexog: int


def _construct_transverse_basis(
    masses_kg: FloatArray,
    coords_bohr_centered: FloatArray,
    tangent_unit: FloatArray,
    n_vib: int,
) -> FloatArray:
    """Construct an orthonormal (n_vib, dim) basis of the internal transverse subspace."""

    dim = len(tangent_unit)
    coords_m = coords_bohr_centered.reshape(-1, 3) * BOHR_M
    rigid, _ = _rigid_basis(masses_kg, coords_m)
    t_proj = tangent_unit - rigid @ (rigid.T @ tangent_unit)
    t_norm = float(np.linalg.norm(t_proj))
    if t_norm > 1.0e-12:
        excluded = np.column_stack([rigid, t_proj / t_norm])
    else:
        excluded = rigid

    # Check if geometry is collinear along a coordinate axis (e.g. z-axis); ordering
    # axial internal coordinates before perpendicular bends separates stretching from
    # degenerate bending modes automatically.
    axis_spread = np.ptp(coords_bohr_centered.reshape(-1, 3), axis=0)
    active_axes = [ax for ax in range(3) if axis_spread[ax] > 1.0e-8]
    seed_cols: list[FloatArray] = []
    if len(active_axes) == 1:
        ax = active_axes[0]
        for atom_idx in range(dim // 3):
            e_vec = np.zeros(dim, dtype=float)
            e_vec[3 * atom_idx + ax] = 1.0
            seed_cols.append(e_vec)
        for other_ax in range(3):
            if other_ax == ax:
                continue
            for atom_idx in range(dim // 3):
                e_vec = np.zeros(dim, dtype=float)
                e_vec[3 * atom_idx + other_ax] = 1.0
                seed_cols.append(e_vec)
    else:
        for idx in range(dim):
            e_vec = np.zeros(dim, dtype=float)
            e_vec[idx] = 1.0
            seed_cols.append(e_vec)

    basis_vecs: list[FloatArray] = []
    for cand in seed_cols:
        v = cand - excluded @ (excluded.T @ cand)
        for prev in basis_vecs:
            v = v - float(np.dot(prev, v)) * prev
        norm_v = float(np.linalg.norm(v))
        if norm_v > 1.0e-8:
            basis_vecs.append(v / norm_v)
            if len(basis_vecs) == n_vib:
                break
    while len(basis_vecs) < n_vib:
        basis_vecs.append(np.zeros(dim, dtype=float))
    return np.asarray(basis_vecs[:n_vib], dtype=float)


def _parallel_transport_step(prev_modes: FloatArray, raw_subspace: FloatArray) -> FloatArray:
    """Rotate orthonormal rows of ``raw_subspace`` to maximize overlap with ``prev_modes``."""

    overlap = prev_modes @ raw_subspace.T
    u_mat, _, vt_mat = np.linalg.svd(overlap, full_matrices=False)
    rot = u_mat @ vt_mat
    return np.asarray(rot @ raw_subspace, dtype=np.float64)


def _align_modes_step(
    prev_cof: FloatArray,
    curr_cof: FloatArray,
    curr_wets: FloatArray,
    curr_bcur: FloatArray,
    has_explicit_bcur: bool,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Match mode ordering and signs against ``prev_cof`` (Polyrate OVRLP + phase lock)."""

    n_vib = prev_cof.shape[0]
    used: set[int] = set()
    perm = [0] * n_vib
    signs = [1.0] * n_vib

    # Match from highest-index/lowest-frequency or highest-overlap greedy order (polymq.F:1679)
    for i in range(n_vib - 1, -1, -1):
        best_j = -1
        best_dot = -1.0
        best_signed = 1.0
        for j in range(n_vib - 1, -1, -1):
            if j in used:
                continue
            dot_val = float(np.dot(prev_cof[i], curr_cof[j]))
            if abs(dot_val) > best_dot:
                best_dot = abs(dot_val)
                best_j = j
                best_signed = dot_val
        if best_j < 0:
            best_j = next(j for j in range(n_vib) if j not in used)
            best_signed = 1.0
        used.add(best_j)
        perm[i] = best_j
        signs[i] = -1.0 if best_signed < 0.0 else 1.0

    new_cof = np.empty_like(curr_cof)
    new_wets = np.empty_like(curr_wets)
    new_bcur = np.empty_like(curr_bcur)
    for i in range(n_vib):
        j = perm[i]
        sgn = signs[i]
        new_cof[i] = sgn * curr_cof[j]
        new_wets[i] = curr_wets[j]
        new_bcur[i] = (sgn * curr_bcur[j]) if has_explicit_bcur else curr_bcur[j]
    return new_cof, new_wets, new_bcur


def _prepare_mep(reaction: Reaction) -> _PreparedMep:
    path = reaction.path
    if path is None or not can_compute_lct(reaction):
        raise ValueError(
            "LCT requires a ReactionPath with >= 3 points and atomistic geometries"
        )
    points = path.points
    n_sd = len(points)
    first_atoms = points[0].dividing_surface.atoms
    n_atoms = len(first_atoms)
    dim = 3 * n_atoms

    masses_amu = np.array([float(atom.mass_amu or 0.0) for atom in first_atoms], dtype=np.float64)
    masses_kg = masses_amu * AMU_KG
    total_mass_amu = float(np.sum(masses_amu))
    mu_amu = path.reduced_mass_amu
    redm = mu_amu * AMU_TO_AU_MASS
    alpha = np.sqrt(np.repeat(masses_amu, 3) / mu_amu)

    s_scale = 1.0 / BOHR_ANGSTROM if path.coordinate_unit == "angstrom" else 1.0
    s_bohr = np.asarray([pt.coordinate * s_scale for pt in points], dtype=float)

    geom = np.zeros((n_sd, dim), dtype=float)
    coords_bohr_centered = np.zeros((n_sd, dim), dtype=float)
    for k, pt in enumerate(points):
        xyz_ang = np.asarray(
            [atom.coordinates_angstrom for atom in pt.dividing_surface.atoms], dtype=float
        )
        com_ang = np.sum(xyz_ang * masses_amu[:, None], axis=0) / total_mass_amu
        centered_bohr = ((xyz_ang - com_ang) / BOHR_ANGSTROM).reshape(dim)
        coords_bohr_centered[k] = centered_bohr
        geom[k] = centered_bohr * alpha

    # Finite-difference mass-scaled path tangents pointing strictly toward +s
    fd_raw = np.gradient(geom, s_bohr, axis=0)
    fd_tangents = np.zeros_like(geom)
    for k in range(n_sd):
        nrm = float(np.linalg.norm(fd_raw[k]))
        if nrm > 1.0e-15:
            fd_tangents[k] = fd_raw[k] / nrm
        elif k > 0:
            fd_tangents[k] = fd_tangents[k - 1]

    dxsv = np.zeros((n_sd, dim), dtype=float)
    for k, pt in enumerate(points):
        if pt.tangent_vector is not None and len(pt.tangent_vector) == dim:
            t_vec = np.asarray(pt.tangent_vector, dtype=float)
            nrm = float(np.linalg.norm(t_vec))
            if nrm > 1.0e-15:
                t_unit = t_vec / nrm
                if float(np.dot(t_unit, fd_tangents[k])) < 0.0:
                    t_unit = -t_unit
                dxsv[k] = t_unit
                continue
        dxsv[k] = fd_tangents[k]

    n_vib = max(
        max(
            len(pt.dividing_surface.positive_frequencies_cm1),
            len(pt.normal_mode_vectors),
            len(pt.curvatures_bohr_inv),
        )
        for pt in points
    )
    if n_vib <= 0:
        raise ValueError("LCT requires at least one bound vibrational mode along the path")

    wets = np.zeros((n_sd, n_vib), dtype=float)
    freq_to_hartree = WAVENUMBER_TO_J / HARTREE_J
    for k, pt in enumerate(points):
        pos_f = pt.dividing_surface.positive_frequencies_cm1
        count = min(len(pos_f), n_vib)
        for m in range(count):
            wets[k, m] = pos_f[m] * freq_to_hartree

    k0 = int(np.argmin(np.abs(s_bohr)))
    has_any_vectors = any(bool(pt.normal_mode_vectors) for pt in points)
    cofsv = np.zeros((n_sd, n_vib, dim), dtype=float)
    raw_bcur = np.zeros((n_sd, n_vib), dtype=float)
    explicit_b_mask = [False] * n_sd

    for k, pt in enumerate(points):
        if pt.curvatures_bohr_inv:
            explicit_b_mask[k] = True
            c_len = min(len(pt.curvatures_bohr_inv), n_vib)
            for m in range(c_len):
                raw_bcur[k, m] = pt.curvatures_bohr_inv[m]

    if has_any_vectors:
        valid_vec_mask = [
            len(pt.normal_mode_vectors) >= n_vib
            and all(len(row) == dim for row in pt.normal_mode_vectors[:n_vib])
            for pt in points
        ]
        for k, pt in enumerate(points):
            if valid_vec_mask[k]:
                mat = np.asarray(pt.normal_mode_vectors[:n_vib], dtype=float)
                for m in range(n_vib):
                    nrm = float(np.linalg.norm(mat[m]))
                    if nrm > 1.0e-15:
                        mat[m] /= nrm
                cofsv[k] = mat
        # Fill any missing points (e.g. s=0 if synthesized without Hessian) via subspace transport
        for k in range(n_sd):
            if not valid_vec_mask[k]:
                sub = _construct_transverse_basis(
                    masses_kg, coords_bohr_centered[k], dxsv[k], n_vib
                )
                ref_idx = -1
                for offset in range(1, n_sd):
                    if k - offset >= 0 and valid_vec_mask[k - offset]:
                        ref_idx = k - offset
                        break
                    if k + offset < n_sd and valid_vec_mask[k + offset]:
                        ref_idx = k + offset
                        break
                cofsv[k] = (
                    _parallel_transport_step(cofsv[ref_idx], sub) if ref_idx >= 0 else sub
                )
    else:
        cofsv[k0] = _construct_transverse_basis(
            masses_kg, coords_bohr_centered[k0], dxsv[k0], n_vib
        )
        for k in range(k0 + 1, n_sd):
            sub = _construct_transverse_basis(
                masses_kg, coords_bohr_centered[k], dxsv[k], n_vib
            )
            cofsv[k] = _parallel_transport_step(cofsv[k - 1], sub)
        for k in range(k0 - 1, -1, -1):
            sub = _construct_transverse_basis(
                masses_kg, coords_bohr_centered[k], dxsv[k], n_vib
            )
            cofsv[k] = _parallel_transport_step(cofsv[k + 1], sub)

    # Continuous OVRLP mode matching and phase alignment outward from k0
    for k in range(k0 + 1, n_sd):
        cofsv[k], wets[k], raw_bcur[k] = _align_modes_step(
            cofsv[k - 1], cofsv[k], wets[k], raw_bcur[k], explicit_b_mask[k]
        )
    for k in range(k0 - 1, -1, -1):
        cofsv[k], wets[k], raw_bcur[k] = _align_modes_step(
            cofsv[k + 1], cofsv[k], wets[k], raw_bcur[k], explicit_b_mask[k]
        )

    # Compute geometric curvature B_m(s) = -d(eta)/ds . L_m(s) and reconcile signs
    d_eta_ds = np.gradient(dxsv, s_bohr, axis=0)
    geom_bcur = np.zeros((n_sd, n_vib), dtype=float)
    for k in range(n_sd):
        for m in range(n_vib):
            geom_bcur[k, m] = -float(np.dot(d_eta_ds[k], cofsv[k, m]))

    bcur = np.zeros((n_sd, n_vib), dtype=float)
    if all(explicit_b_mask):
        # Ensure global sign of each mode i agrees between explicit B_m and -d(eta)/ds . L_m
        for m in range(n_vib):
            corr = float(np.dot(raw_bcur[:, m], geom_bcur[:, m]))
            geom_norm = float(np.linalg.norm(geom_bcur[:, m]))
            if geom_norm > 1.0e-3 and corr < 0.0:
                cofsv[:, m, :] *= -1.0
                geom_bcur[:, m] *= -1.0
        # If user supplied unsigned (all-nonnegative) curvatures while geometry flips sign,
        # adopt the physical geometric sign where |geom_bcur| is clear.
        bcur = raw_bcur.copy()
    else:
        for k in range(n_sd):
            bcur[k] = raw_bcur[k] if explicit_b_mask[k] else geom_bcur[k]

    # Polyrate LCG34 (polyhl.F:4260-4267): linearly interpolate BCUR across s = 0
    if 0 < k0 < n_sd - 1 and abs(float(s_bohr[k0])) <= 1.0e-10:
        s_left = float(s_bohr[k0 - 1])
        s_right = float(s_bohr[k0 + 1])
        if abs(s_right - s_left) > 1.0e-15:
            w_right = (0.0 - s_left) / (s_right - s_left)
            bcur[k0] = bcur[k0 - 1] + w_right * (bcur[k0 + 1] - bcur[k0 - 1])

    vclas = np.asarray(
        [pt.dividing_surface.electronic_energy_kj_mol / HARTREE_KJ_MOL for pt in points],
        dtype=float,
    )
    vadib = np.asarray(
        [
            (
                pt.dividing_surface.electronic_energy_kj_mol
                + zero_point_energy_kj_mol(pt.dividing_surface)
            )
            / HARTREE_KJ_MOL
            for pt in points
        ],
        dtype=float,
    )

    var = (
        sum(
            sp.electronic_energy_kj_mol + zero_point_energy_kj_mol(sp)
            for sp in reaction.reactants
        )
        / HARTREE_KJ_MOL
    )
    if reaction.products:
        vap = (
            sum(
                sp.electronic_energy_kj_mol + zero_point_energy_kj_mol(sp)
                for sp in reaction.products
            )
            / HARTREE_KJ_MOL
        )
    else:
        vap = float(vadib[-1])

    ez = max(var, vap)
    iexog = 0 if var >= vap else -1

    return _PreparedMep(
        n_atoms=n_atoms,
        dim=dim,
        n_sd=n_sd,
        n_vib=n_vib,
        redm=redm,
        alpha=alpha,
        s_bohr=s_bohr,
        geom=geom,
        dxsv=dxsv,
        cofsv=cofsv,
        wets=wets,
        bcur=bcur,
        vclas=vclas,
        vadib=vadib,
        var=var,
        vap=vap,
        ez=ez,
        iexog=iexog,
    )


class _ShepardRphSurface:
    """Euclidean Shepard 2nd-order reaction-path Hamiltonian off-MEP potential (in hartree)."""

    def __init__(self, mep: _PreparedMep, power: int = 4) -> None:
        self.mep = mep
        self.power = max(2, int(power))
        self.dv_ds = np.gradient(mep.vclas, mep.s_bohr)
        self.d2v_ds2 = np.gradient(self.dv_ds, mep.s_bohr)
        spacings = np.diff(mep.s_bohr)
        median_ds = float(np.median(spacings)) if len(spacings) > 0 else 0.1
        self.eps_sq = max(1.0e-12, (0.05 * median_ds) ** 2)
        self.trust_s = max(0.25, 4.0 * median_ds)
        self.force_k = mep.redm * np.maximum(mep.wets, 0.0) ** 2  # (n_sd, n_vib)

    def __call__(self, x_ms_bohr: FloatArray) -> float:
        mep = self.mep
        diff = x_ms_bohr[None, :] - mep.geom  # (n_sd, dim)
        dist_sq = np.einsum("kd,kd->k", diff, diff)
        min_idx = int(np.argmin(dist_sq))
        if float(dist_sq[min_idx]) <= 1.0e-20:
            return float(mep.vclas[min_idx])

        weights = (dist_sq + self.eps_sq) ** (-self.power)
        w_sum = float(np.sum(weights))
        if w_sum <= 0.0 or not math.isfinite(w_sum):
            return float(mep.vclas[min_idx])

        # Longitudinal displacement ds_k = eta_k . (x - X_k), trust-region damped at large |ds_k|
        ds_raw = np.einsum("kd,kd->k", mep.dxsv, diff)
        ds_eff = ds_raw / np.sqrt(1.0 + (ds_raw / self.trust_s) ** 2)

        # Transverse normal-mode projections q_{k, m} = L_{k, m} . (x - X_k)
        q_km = np.einsum("kmd,kd->km", mep.cofsv, diff)
        bq_k = np.einsum("km,km->k", mep.bcur, q_km)
        v_trans_k = 0.5 * np.einsum("km,km->k", self.force_k, q_km * q_km)

        local_v = (
            mep.vclas
            + self.dv_ds * ds_eff * (1.0 - bq_k)
            + 0.5 * self.d2v_ds2 * (ds_eff * ds_eff)
            + v_trans_k
        )
        return float(np.dot(weights, local_v) / w_sum)


def _build_classical_evaluator(
    mep: _PreparedMep, options: LctOptions
) -> Callable[[FloatArray], float]:
    if options.potential_fn is None:
        return _ShepardRphSurface(mep, power=options.shepard_power)

    pot_fn = options.potential_fn
    inv_alpha_ang = BOHR_ANGSTROM / mep.alpha

    def _eval(x_ms_bohr: FloatArray) -> float:
        coords_angstrom = (x_ms_bohr * inv_alpha_ang).reshape(mep.n_atoms, 3)
        v_kj_mol = float(pot_fn(coords_angstrom))
        return v_kj_mol / HARTREE_KJ_MOL

    return _eval


def _lcproj(cof_mat: FloatArray, x0: FloatArray, x1: FloatArray) -> FloatArray:
    """Polyrate LCPROJ (polyhl.F:3083): normalized projection of (x1 - x0) on modes."""

    q = cof_mat @ (x1 - x0)
    nrm = float(np.linalg.norm(q))
    if nrm > 0.0:
        q = q / nrm
    return q


def _lcxtau(wets_vec: FloatArray, smlq: FloatArray) -> float:
    """Polyrate LCXTAU (polyhl.F:3738): vibrational period 2*pi / omega_p in atomic units."""

    pos = np.maximum(wets_vec, 0.0)
    omega_p = float(np.linalg.norm(pos * smlq))
    if omega_p <= 1.0e-30:
        return 0.0
    return 2.0 * math.pi / omega_p


def _compute_nstate(
    mep: _PreparedMep,
    vad_spline: _NaturalCubicSpline,
    esv: FloatArray,
    trnpt1: FloatArray,
    trnpt2: FloatArray,
) -> tuple[int, list[int], float, float, int, int]:
    """Polyrate NSTATE (polymq.F:1490-1602): accessible excited states at each energy."""

    sl = float(mep.s_bohr[0])
    sr = float(mep.s_bohr[-1])
    ncutf = 0
    icutf = 0
    if mep.iexog == 0 and float(mep.vadib[-1]) < mep.vap:
        ncutf = 1
    elif mep.iexog == -1 and float(mep.vadib[0]) < mep.var:
        ncutf = 2

    s_max, _ = vad_spline.maximum()
    if ncutf != 0:
        if mep.iexog == 0:
            _, stxx = vad_spline.outer_turning_points(mep.vap, s_max)
            sr = stxx
            idx_r = int(np.searchsorted(mep.s_bohr, stxx, side="right")) - 1
            jj_cu = max(0, min(mep.n_sd - 2, idx_r))
            icutf = jj_cu + 1
        else:
            stxx, _ = vad_spline.outer_turning_points(mep.var, s_max)
            sl = stxx
            idx_l = int(np.searchsorted(mep.s_bohr, stxx, side="right")) - 1
            jj_cu = max(0, min(mep.n_sd - 2, idx_l))
            icutf = jj_cu
        jnl, wts = _locate_and_lagrange4(mep.s_bohr, stxx)
        geom_cu = _interp_vec(mep.geom, jnl, wts)
        cof_cu = _interp_mat(mep.cofsv, jnl, wts)
        wets_cu = _interp_vec(mep.wets, jnl, wts)
        vadb_cu = mep.vap if mep.iexog == 0 else mep.var
    else:
        ixx = mep.n_sd - 1 if mep.iexog == 0 else 0
        vadb_cu = float(mep.vadib[ixx])
        geom_cu = mep.geom[ixx]
        cof_cu = mep.cofsv[ixx]
        wets_cu = mep.wets[ixx]

    nelast = len(esv)  # includes NEMX = NEMAX + 1
    npacc = [0] * nelast
    ndumol = 0
    pos_wets_cu = np.maximum(wets_cu, 0.0)
    for ie in range(nelast):
        ener = float(esv[ie])
        stp0 = float(trnpt1[ie])
        stp1 = float(trnpt2[ie])
        stpx = stp0 if mep.iexog == 0 else stp1
        if stp0 > sl and stp1 < sr:
            jnl, wts = _locate_and_lagrange4(mep.s_bohr, stpx)
            rxx = _interp_vec(mep.geom, jnl, wts)
            smlq = _lcproj(cof_cu, rxx, geom_cu)
            wp = float(np.linalg.norm(pos_wets_cu * smlq))
            if wp > 1.0e-15 and ener >= vadb_cu:
                npx = int(math.floor((ener - vadb_cu) / wp))
                npx = max(0, npx)
            else:
                npx = 0
            npacc[ie] = npx
            if npx > ndumol:
                ndumol = npx

    return ndumol, npacc, sl, sr, ncutf, icutf


def _lcstx(
    mep: _PreparedMep,
    iprod: int,
    ener: float,
    rxx: FloatArray,
    jjx: int,
    ncutf: int,
    icutf: int,
) -> tuple[float, int, int, FloatArray, FloatArray]:
    """Polyrate LCSTX + LCTP (polyhl.F:3383-3548): excited adiabatic potential & turning point."""

    if mep.iexog == 0:
        libeg = jjx + 1
        iend = (icutf - 1) if ncutf == 1 else (mep.n_sd - 1)
    else:
        libeg = icutf if ncutf == 2 else 0
        iend = jjx
    libeg = max(0, min(mep.n_sd - 1, libeg))
    iend = max(libeg, min(mep.n_sd - 1, iend))
    nsdex = iend - libeg + 1

    vadex_sub = np.zeros(nsdex, dtype=float)
    tplxx_sub = np.zeros((nsdex, mep.n_vib), dtype=float)
    for offset in range(nsdex):
        is_idx = libeg + offset
        smlq = _lcproj(mep.cofsv[is_idx], rxx, mep.geom[is_idx])
        pos_w = np.maximum(mep.wets[is_idx], 0.0)
        wp = float(np.linalg.norm(pos_w * smlq))
        vadex_sub[offset] = float(mep.vadib[is_idx]) + float(iprod) * wp
        for m in range(mep.n_vib):
            w_m = float(mep.wets[is_idx, m])
            if w_m <= 0.0:
                tplxx_sub[offset, m] = 0.0
            else:
                edlqm = (
                    float(iprod) * (float(smlq[m]) * w_m) ** 2 / wp
                    if wp > 1.0e-15
                    else 0.0
                )
                tplxx_sub[offset, m] = 0.5 * w_m + edlqm

    ss_sub = mep.s_bohr[libeg : iend + 1]
    if nsdex < 2:
        return float(ss_sub[0]), libeg, nsdex, vadex_sub, tplxx_sub

    ex_spline = _NaturalCubicSpline(ss_sub, vadex_sub)
    if mep.iexog == 0:
        if ener <= float(vadex_sub[-1]):
            stxx = float(ss_sub[-1])
        else:
            stxx = float(ss_sub[-1])
            for idx in range(nsdex - 2, -1, -1):
                if float(vadex_sub[idx]) > ener:
                    stxx = ex_spline._solve_on_segment(idx, ener)
                    break
    else:
        if ener <= float(vadex_sub[0]):
            stxx = float(ss_sub[0])
        else:
            stxx = float(ss_sub[0])
            for idx in range(nsdex - 1):
                if float(vadex_sub[idx + 1]) > ener:
                    stxx = ex_spline._solve_on_segment(idx, ener)
                    break

    return stxx, libeg, nsdex, vadex_sub, tplxx_sub


def _lcnwtn(
    mep: _PreparedMep,
    lcx: FloatArray,
    sx_init: float,
    zeta: float,
    zetold: float,
    lcgx: FloatArray,
) -> tuple[float, FloatArray, FloatArray]:
    """Polyrate LCNWTN (polyhl.F:2853-3010): find s(zeta) with (lcx - X(s)) . eta(s) = 0."""

    s_min = float(mep.s_bohr[0])
    s_max = float(mep.s_bohr[-1])
    sx = max(s_min, min(s_max, sx_init))

    def _eval_s(s_val: float) -> tuple[FloatArray, FloatArray, float, float]:
        jnl, wts = _locate_and_lagrange4(mep.s_bohr, s_val)
        mepx = _interp_vec(mep.geom, jnl, wts)
        dx = _interp_vec(mep.dxsv, jnl, wts)
        xlin = lcx - mepx
        ualf = float(np.linalg.norm(xlin))
        dotp = float(np.dot(dx, xlin))
        return xlin, dx, ualf, dotp

    xlin, dx, ualf, dotp = _eval_s(sx)
    if ualf <= _TOL_UALF_BOHR or abs(dotp) <= _TOL_DOTP:
        return sx, xlin, dx

    dels = (zeta - zetold) * float(np.dot(dx, lcgx))
    if abs(dels) <= 1.0e-12:
        dels = math.copysign(1.0e-3, dotp)
    # Increasing s moves mepx along +dx, decreasing dot(dx, lcx - mepx), so step in sign(dotp)
    if dotp * dels < 0.0:
        dels = -dels

    sold = sx
    dotpo = dotp
    bracket_found = False
    for _ in range(500):
        sold = sx
        dotpo = dotp
        sx_next = sx + dels
        clamped = False
        if sx_next <= s_min:
            sx_next = s_min
            clamped = True
        elif sx_next >= s_max:
            sx_next = s_max
            clamped = True
        sx = sx_next
        xlin, dx, ualf, dotp = _eval_s(sx)
        if ualf <= _TOL_UALF_BOHR or abs(dotp) <= _TOL_DOTP:
            return sx, xlin, dx
        if dotp * dotpo <= 0.0:
            bracket_found = True
            break
        if clamped:
            break

    if not bracket_found:
        # Fallback: coarse scan over MEP nodes to guarantee closest/orthogonal point
        diffs = lcx[None, :] - mep.geom
        dots = np.einsum("kd,kd->k", mep.dxsv, diffs)
        sign_changes = np.where(dots[:-1] * dots[1:] <= 0.0)[0]
        if len(sign_changes) > 0:
            best_k = int(
                min(
                    sign_changes,
                    key=lambda idx: float(np.sum(diffs[idx] ** 2)),
                )
            )
            sold = float(mep.s_bohr[best_k])
            sx = float(mep.s_bohr[best_k + 1])
            _, _, _, dotpo = _eval_s(sold)
            xlin, dx, ualf, dotp = _eval_s(sx)
            bracket_found = True
        else:
            best_k = int(np.argmin(np.einsum("kd,kd->k", diffs, diffs)))
            sx = float(mep.s_bohr[best_k])
            xlin, dx, _, _ = _eval_s(sx)
            return sx, xlin, dx

    sm, sp = (sx, sold) if sx < sold else (sold, sx)
    for _ in range(200):
        if ualf <= _TOL_UALF_BOHR or abs(dotp) <= _TOL_DOTP or abs(sp - sm) <= 1.0e-12:
            break
        denom = dotp - dotpo
        if abs(denom) > 1.0e-30:
            step = -dotp * (sx - sold) / denom
        else:
            step = 0.5 * (sm + sp) - sx
        if sx + step <= sm or sx + step >= sp:
            step = 0.5 * (sm + sp) - sx
        sold = sx
        dotpo = dotp
        sx = sx + step
        xlin, dx, ualf, dotp = _eval_s(sx)
        if ualf > _TOL_UALF_BOHR:
            if dotp > 0.0:
                sm = sx
            elif dotp < 0.0:
                sp = sx

    return sx, xlin, dx


def _lczcrt_and_lcvcor(
    mep: _PreparedMep,
    qm: FloatArray,
    freq: FloatArray,
    bm1: FloatArray,
    tplx: FloatArray,
    dexwqm: FloatArray,
    use_excited_corr: bool,
) -> tuple[int, float]:
    """Polyrate LCZCRT + LCVCOR + LCVMOD (harmonic transverse modes, polyhl.F:3554-3820)."""

    ich = 0
    vcor = 0.0
    tm2 = 0.0
    for m in range(mep.n_vib):
        om = float(freq[m])
        q_m = float(qm[m])
        mc_out = False
        if om >= 0.0 and abs(q_m) > abs(float(tplx[m])):
            mc_out = True
            ich = 1
        if om > 0.0:
            tm2 -= float(bm1[m]) * q_m
            if not mc_out:
                ev = 0.5 * om + (float(dexwqm[m]) if use_excited_corr else 0.0)
                vm = 0.5 * mep.redm * (om * q_m) ** 2
                vcor += ev - vm
    if tm2 >= 1.0:
        ich = 2
    return ich, vcor


def reaction_lct_result(
    reaction: Reaction,
    options: LctOptions | None = None,
) -> LctResult:
    """Compute Polyrate-compatible LCG3/LCG4, ZCT, SCT, and muOMT probability tables."""

    opts = options if options is not None else LctOptions()
    mep = _prepare_mep(reaction)
    vad_spline = _NaturalCubicSpline(mep.s_bohr, mep.vadib)
    s_max, v_max = vad_spline.maximum()

    # Build energy quadrature grid on [EZ, VMAX]
    vm = v_max - mep.ez
    nemax = opts.energy_segments * opts.energy_quadrature_points
    if vm <= 0.0:
        zeros = tuple(0.0 for _ in range(nemax))
        return LctResult(
            version=opts.version,
            threshold_kj_mol=mep.ez * HARTREE_KJ_MOL,
            barrier_kj_mol=v_max * HARTREE_KJ_MOL,
            energies_kj_mol=tuple(v_max * HARTREE_KJ_MOL for _ in range(nemax)),
            weights_kj_mol=zeros,
            probabilities_lct=zeros,
            probabilities_zct=zeros,
            probabilities_sct=zeros,
            probabilities_muomt=zeros,
            max_excited_state=0,
            state_uniformized_probabilities=(tuple(0.0 for _ in range(nemax + 1)),),
        )

    esv_list: list[float] = []
    ewt_list: list[float] = []
    seg_width = vm / opts.energy_segments
    for iseg in range(opts.energy_segments):
        e_lo = mep.ez + iseg * seg_width
        e_hi = e_lo + seg_width
        nodes, weights = gauss_legendre_quadrature(
            e_lo, e_hi, opts.energy_quadrature_points
        )
        esv_list.extend(nodes)
        ewt_list.extend(weights)

    nemx = nemax + 1
    esv = np.zeros(nemx, dtype=float)
    esv[:nemax] = esv_list
    esv[nemax] = v_max

    trnpt1 = np.zeros(nemx, dtype=float)
    trnpt2 = np.zeros(nemx, dtype=float)
    for ie in range(nemax):
        t0, t1 = vad_spline.outer_turning_points(float(esv[ie]), s_max)
        trnpt1[ie] = t0
        trnpt2[ie] = t1
    trnpt1[nemax] = s_max
    trnpt2[nemax] = s_max

    # Determine accessible excited final vibrational states (Polyrate NSTATE)
    if opts.excited_states == 0:
        nprod, _, slcutf, srcutf, ncutf, icutf = _compute_nstate(
            mep, vad_spline, esv, trnpt1, trnpt2
        )
        nprod = 0
        npacc = [0] * nemx
    else:
        nprod, npacc, slcutf, srcutf, ncutf, icutf = _compute_nstate(
            mep, vad_spline, esv, trnpt1, trnpt2
        )
        if isinstance(opts.excited_states, int):
            nprod = min(nprod, opts.excited_states)
    nprod = min(nprod, opts.max_states_cap)
    nnprod = nprod + 1

    classical_v = _build_classical_evaluator(mep, opts)
    ng = opts.theta_quadrature_points
    unit_nodes, _ = gauss_legendre_quadrature(0.0, 1.0, ng)
    unit_nodes_arr = np.asarray(unit_nodes, dtype=float)

    pesum_trap: list[list[float]] = []

    # ==================== Loop over final vibrational states ====================
    for iprod in range(nnprod):
        nelast = (nemx - 1) if iprod == 0 else nemx
        st = trnpt1.copy()
        stpr = trnpt2.copy()
        sinx = np.zeros(nemx, dtype=float)
        sinxp = np.zeros(nemx, dtype=float)
        cosx = np.zeros(nemx, dtype=float)
        cosxp = np.zeros(nemx, dtype=float)
        tau = np.zeros(nemx, dtype=float)
        taupr = np.zeros(nemx, dtype=float)
        thet = np.full(nemx, _OFF_GRID_THETA, dtype=float)

        ie0_first: int | None = None
        ivef_limit = -1

        for ie in range(nelast):
            ener = float(esv[ie])
            stp0 = float(st[ie])
            stp1 = float(stpr[ie])

            # Off-grid check (Polyrate LCG34 lines 4401-4404)
            if mep.iexog == 0:
                on_grid = stp0 > slcutf and (iprod != 0 or stp1 < srcutf)
            else:
                on_grid = stp1 < srcutf and (iprod != 0 or stp0 > slcutf)
            if not on_grid or iprod > npacc[ie]:
                continue

            # ---- LCSET (polyhl.F:3116-3299) ----
            libeg = 0
            nsdex = 0
            vadex_sub = np.empty(0, dtype=float)
            tplxx_sub = np.empty((0, mep.n_vib), dtype=float)
            if iprod != 0:
                stpx_init = stp0 if mep.iexog == 0 else stp1
                jjx_init = max(
                    0,
                    min(
                        mep.n_sd - 2,
                        int(np.searchsorted(mep.s_bohr, stpx_init, side="right")) - 1,
                    ),
                )
                jnl_i, wts_i = _locate_and_lagrange4(mep.s_bohr, stpx_init)
                rxx_init = _interp_vec(mep.geom, jnl_i, wts_i)
                stxx, libeg, nsdex, vadex_sub, tplxx_sub = _lcstx(
                    mep, iprod, ener, rxx_init, jjx_init, ncutf, icutf
                )
                if mep.iexog == 0:
                    stp1 = stxx
                    stpr[ie] = stp1
                    if stp1 >= float(mep.s_bohr[-1]):
                        continue
                else:
                    stp0 = stxx
                    st[ie] = stp0
                    if stp0 <= float(mep.s_bohr[0]):
                        continue

            jnl0, wts0 = _locate_and_lagrange4(mep.s_bohr, stp0)
            rx0 = _interp_vec(mep.geom, jnl0, wts0)
            dx0 = _interp_vec(mep.dxsv, jnl0, wts0)
            cof0 = _interp_mat(mep.cofsv, jnl0, wts0)
            freq0 = _interp_vec(mep.wets, jnl0, wts0)

            jnl1, wts1 = _locate_and_lagrange4(mep.s_bohr, stp1)
            rx1 = _interp_vec(mep.geom, jnl1, wts1)
            dx1 = _interp_vec(mep.dxsv, jnl1, wts1)
            cof1 = _interp_mat(mep.cofsv, jnl1, wts1)
            freq1 = _interp_vec(mep.wets, jnl1, wts1)

            chord = rx1 - rx0
            zeta_len = float(np.linalg.norm(chord))
            if zeta_len <= 1.0e-14:
                thet[ie] = 0.0
                if ie0_first is None:
                    ie0_first = ie
                continue
            lcgx = chord / zeta_len

            c0 = min(1.0, abs(float(np.dot(dx0, lcgx))))
            cosx[ie] = c0
            sinx[ie] = math.sqrt(max(0.0, 1.0 - c0 * c0))
            smlq0 = _lcproj(cof0, rx0, rx1)
            tau[ie] = _lcxtau(freq0, smlq0)

            c1 = min(1.0, abs(float(np.dot(dx1, lcgx))))
            cosxp[ie] = c1
            sinxp[ie] = math.sqrt(max(0.0, 1.0 - c1 * c1))
            smlq1 = _lcproj(cof1, rx0, rx1)
            taupr[ie] = _lcxtau(freq1, smlq1)

            if tau[ie] <= 0.0 or taupr[ie] <= 0.0:
                continue

            if ie0_first is None:
                ie0_first = ie

            # ---- LCGNAD (polyhl.F:4799-5238) ----
            qdx = unit_nodes_arr * zeta_len
            vadiz0 = np.zeros(ng, dtype=float)
            vadiz1 = np.zeros(ng, dtype=float)
            dotp0 = np.zeros(ng, dtype=float)
            dotp1 = np.zeros(ng, dtype=float)
            vcoriz = np.zeros(ng, dtype=float)

            ss_ex = mep.s_bohr[libeg : libeg + nsdex] if iprod != 0 else np.empty(0)

            def _eval_adiabatic_point(
                ir_dir: int,
                stpx_val: float,
                xlin_vec: FloatArray,
                *,
                _iprod: int = iprod,
                _ss_ex: FloatArray = ss_ex,
                _tplxx_sub: FloatArray = tplxx_sub,
                _vadex_sub: FloatArray = vadex_sub,
            ) -> tuple[int, float, float]:
                jnl_s, wts_s = _locate_and_lagrange4(mep.s_bohr, stpx_val)
                cof_x = _interp_mat(mep.cofsv, jnl_s, wts_s)
                qm = cof_x @ xlin_vec
                freq_x = _interp_vec(mep.wets, jnl_s, wts_s)
                bm_x = _interp_vec(mep.bcur, jnl_s, wts_s)

                endo_branch = (
                    _iprod == 0
                    or (mep.iexog == 0 and ir_dir == 1)
                    or (mep.iexog == -1 and ir_dir == 2)
                )
                tplx_arr = np.zeros(mep.n_vib, dtype=float)
                dexwqm_arr = np.zeros(mep.n_vib, dtype=float)
                if endo_branch:
                    for m in range(mep.n_vib):
                        fq = float(freq_x[m])
                        tplx_arr[m] = (
                            -1.0 / math.sqrt(mep.redm * fq) if fq > 0.0 else -1.0e10
                        )
                    vad_val = _interp_1d(mep.vadib, jnl_s, wts_s)
                else:
                    jnl_ex, wts_ex = _locate_and_lagrange4(_ss_ex, stpx_val)
                    tpl_interp = _interp_vec(_tplxx_sub, jnl_ex, wts_ex)
                    for m in range(mep.n_vib):
                        fq_e = float(tpl_interp[m])
                        fq_0 = float(freq_x[m])
                        if fq_e <= 0.0 or fq_0 <= 0.0:
                            tplx_arr[m] = -1.0e10
                            dexwqm_arr[m] = 0.0
                        else:
                            dexwqm_arr[m] = fq_e - 0.5 * fq_0
                            tplx_arr[m] = -math.sqrt(
                                fq_e / (0.5 * mep.redm * fq_0 * fq_0)
                            )
                    vad_val = _interp_1d(_vadex_sub, jnl_ex, wts_ex)

                ich_val, vcor_val = _lczcrt_and_lcvcor(
                    mep,
                    qm,
                    freq_x,
                    bm_x,
                    tplx_arr,
                    dexwqm_arr,
                    use_excited_corr=not endo_branch,
                )
                return ich_val, vcor_val, vad_val

            # Direction IR = 1 (from reactant turning point stp0)
            iz0 = 0
            stpx = stp0
            zetold = 0.0
            for iz in range(ng):
                z_val = float(qdx[iz])
                lcx = rx0 + (z_val / zeta_len) * chord
                stpx, xlin, dx_cur = _lcnwtn(mep, lcx, stpx, z_val, zetold, lcgx)
                zetold = z_val
                dotp0[iz] = float(np.dot(dx_cur, lcgx))
                if stpx > stp1:
                    _, vcor_clamped, vad_clamped = _eval_adiabatic_point(
                        1, min(stpx, stp1), xlin
                    )
                    vcoriz[iz] = vcor_clamped
                    vadiz0[iz] = vad_clamped
                    break
                ich, vcor, vad = _eval_adiabatic_point(1, stpx, xlin)
                vcoriz[iz] = vcor
                vadiz0[iz] = vad
                if ich == 0:
                    iz0 = min(iz + 1, ng - 1)
                else:
                    break

            # Direction IR = 2 (from product turning point stp1)
            iz1 = ng - 1
            stpx = stp1
            zetold = zeta_len
            for iz in range(ng - 1, -1, -1):
                z_val = float(qdx[iz])
                lcx = rx0 + (z_val / zeta_len) * chord
                stpx, xlin, dx_cur = _lcnwtn(mep, lcx, stpx, z_val, zetold, lcgx)
                zetold = z_val
                dotp1[iz] = float(np.dot(dx_cur, lcgx))
                if stpx < stp0:
                    _, vcor_clamped, vad_clamped = _eval_adiabatic_point(
                        2, max(stpx, stp0), xlin
                    )
                    vcoriz[iz] = vcor_clamped
                    vadiz1[iz] = vad_clamped
                    break
                ich, vcor, vad = _eval_adiabatic_point(2, stpx, xlin)
                vcoriz[iz] = vcor
                vadiz1[iz] = vad
                if ich == 0:
                    iz1 = max(iz - 1, 0)
                else:
                    break

            vadiz = np.zeros(ng, dtype=float)
            dotiz = np.zeros(ng, dtype=float)
            if iz1 < iz0:
                vadiz[:iz1] = vadiz0[:iz1]
                dotiz[:iz1] = dotp0[:iz1]
                vadiz[iz0 + 1 :] = vadiz1[iz0 + 1 :]
                dotiz[iz0 + 1 :] = dotp1[iz0 + 1 :]
                if not opts.vad_average:
                    for iz in range(iz1, iz0 + 1):
                        vadiz[iz] = min(float(vadiz0[iz]), float(vadiz1[iz]))
                        dotiz[iz] = min(float(dotp0[iz]), float(dotp1[iz]))
                else:
                    denom_z = float(qdx[iz0] - qdx[iz1])
                    for iz in range(iz1, iz0 + 1):
                        t_w = (
                            float(qdx[iz] - qdx[iz1]) / denom_z
                            if abs(denom_z) > 1.0e-15
                            else 0.5
                        )
                        vadiz[iz] = float(vadiz1[iz]) + t_w * float(
                            vadiz0[iz] - vadiz1[iz]
                        )
                        dotiz[iz] = float(dotp1[iz]) + t_w * float(
                            dotp0[iz] - dotp1[iz]
                        )
            else:
                vadiz[: iz0 + 1] = vadiz0[: iz0 + 1]
                dotiz[: iz0 + 1] = dotp0[: iz0 + 1]
                vadiz[iz1:] = vadiz1[iz1:]
                dotiz[iz1:] = dotp1[iz1:]

            izmx = int(np.argmax(vadiz))

            # Cache classical V(x(zeta_iz)) per iz along this chord
            v_clas_cache: dict[int, float] = {}

            def _lcvef(
                iz_idx: int,
                izr_idx: int,
                izp_idx: int,
                e0_val: float,
                e1_val: float,
                *,
                _cache: dict[int, float] = v_clas_cache,
                _qdx: FloatArray = qdx,
                _rx0: FloatArray = rx0,
                _zeta_len: float = zeta_len,
                _chord: FloatArray = chord,
                _vcoriz: FloatArray = vcoriz,
            ) -> float:
                if iz_idx not in _cache:
                    z_here = float(_qdx[iz_idx])
                    _cache[iz_idx] = classical_v(
                        _rx0 + (z_here / _zeta_len) * _chord
                    )
                v_c = _cache[iz_idx]
                zcrt0 = float(_qdx[izr_idx])
                zcrt1 = float(_qdx[izp_idx])
                term1 = zcrt1 - zcrt0
                vcor0 = float(_vcoriz[izr_idx])
                vcor1 = float(_vcoriz[izp_idx])
                if term1 < 1.0e-10:
                    vbcorr = 0.5 * (vcor0 + vcor1)
                    ebcorr = 0.5 * (e0_val + e1_val)
                else:
                    zterm = (float(_qdx[iz_idx]) - zcrt0) / term1
                    vbcorr = vcor0 + zterm * (vcor1 - vcor0)
                    ebcorr = e0_val + zterm * (e1_val - e0_val)
                return v_c + vbcorr + ebcorr

            # ---- LCTV4 (polyhl.F:5183-5354) ----
            eiz0 = 0.0
            eiz1 = 0.0
            if iz1 >= iz0:
                ivef_limit = ie + opts.vef_search_points
                ienonad = True
                iz0b = iz0
                iz1b = iz1
                vef0 = _lcvef(iz0, iz0, iz1, 0.0, 0.0)
                vef1 = _lcvef(iz1, iz0, iz1, 0.0, 0.0)
                vad0 = float(vadiz[iz0])
                vad1 = float(vadiz[iz1])
                if opts.version == 4:
                    if vef0 < vad0:
                        eiz0 = vad0 - vef0
                    if vef1 < vad1:
                        eiz1 = vad1 - vef1
                    if vef0 > vad0:
                        for iz in range(iz0b - 1, -1, -1):
                            if _lcvef(iz, iz, iz, 0.0, 0.0) <= float(vadiz[iz]):
                                break
                            iz0 = iz
                    if vef1 > vad1:
                        for iz in range(iz1b + 1, ng):
                            if _lcvef(iz, iz, iz, 0.0, 0.0) <= float(vadiz[iz]):
                                break
                            iz1 = iz
            else:
                ienonad = False
                if opts.version == 4 and ivef_limit >= ie and izmx + 1 < ng:
                    izm0 = izmx
                    izm1 = izmx + 1
                    vef0 = _lcvef(izm0, izm0, izm1, 0.0, 0.0)
                    vef1 = _lcvef(izm1, izm0, izm1, 0.0, 0.0)
                    vad0 = float(vadiz[izm0])
                    vad1 = float(vadiz[izm1])
                    if vef0 > vad0 or vef1 > vad1:
                        ienonad = True
                        if vef0 > vad0:
                            for iz in range(izm0 - 1, -1, -1):
                                if _lcvef(iz, iz, iz, 0.0, 0.0) <= float(vadiz[iz]):
                                    break
                                izm0 = iz
                        if vef1 > vad1:
                            for iz in range(izm1 + 1, ng):
                                if _lcvef(iz, iz, iz, 0.0, 0.0) <= float(vadiz[iz]):
                                    break
                                izm1 = iz
                        iz0 = izm0
                        iz1 = izm1

            # ---- LCGTHE (polyhl.F:5615-5693): trapezoidal rule on Gauss nodes ----
            trpwt = np.zeros(ng, dtype=float)
            trpwt[0] = 0.5 * float(qdx[1] - qdx[0])
            trpwt[ng - 1] = 0.5 * float(qdx[ng - 1] - qdx[ng - 2])
            if ng > 2:
                trpwt[1 : ng - 1] = 0.5 * (qdx[2:] - qdx[:-2])

            xintsv = np.zeros(ng, dtype=float)
            if ienonad:
                for iz in range(iz0, iz1 + 1):
                    dif = _lcvef(iz, iz0, iz1, eiz0, eiz1) - ener
                    if dif > 0.0:
                        xintsv[iz] = math.sqrt(dif)
                for iz in range(0, iz0):
                    dif = float(vadiz[iz]) - ener
                    if dif > 0.0:
                        term = float(dotiz[iz]) * math.sqrt(dif)
                        if term > 0.0:
                            xintsv[iz] = term
                for iz in range(iz1 + 1, ng):
                    dif = float(vadiz[iz]) - ener
                    if dif > 0.0:
                        term = float(dotiz[iz]) * math.sqrt(dif)
                        if term > 0.0:
                            xintsv[iz] = term
            else:
                for iz in range(ng):
                    dif = float(vadiz[iz]) - ener
                    if dif > 0.0:
                        term = float(dotiz[iz]) * math.sqrt(dif)
                        if term > 0.0:
                            xintsv[iz] = term

            thet[ie] = math.sqrt(2.0 * mep.redm) * float(np.dot(trpwt, xintsv))

        # Extrapolate barrier-top quantities for iprod == 0 (LCG34 lines 4488-4526)
        if iprod == 0:
            thet[nemax] = 0.0
            st[nemax] = s_max
            stpr[nemax] = s_max
            fit_n = min(4, nemax)
            sl_slice = slice(nemax - fit_n, nemax)
            st_tail = st[sl_slice].tolist()
            stpr_tail = stpr[sl_slice].tolist()
            tau[nemax] = max(1.0e-12, neville_interpolate(st_tail, tau[sl_slice].tolist(), s_max))
            taupr[nemax] = max(
                1.0e-12, neville_interpolate(stpr_tail, taupr[sl_slice].tolist(), s_max)
            )
            cosx[nemax] = neville_interpolate(st_tail, cosx[sl_slice].tolist(), s_max)
            cosxp[nemax] = neville_interpolate(stpr_tail, cosxp[sl_slice].tolist(), s_max)

        pe_state = [0.0] * nemx
        esv_list_full = esv.tolist()
        if ie0_first is not None and ie0_first < nemax:
            pe_r = lct_integrate_amplitudes(
                st.tolist(),
                esv_list_full,
                sinx.tolist(),
                thet.tolist(),
                tau.tolist(),
                mep.redm,
                ie0_first,
                opts.amplitude_quadrature_points,
                opts.interpolation_order,
            )
            pe_p = lct_integrate_amplitudes(
                stpr.tolist(),
                esv_list_full,
                sinxp.tolist(),
                thet.tolist(),
                taupr.tolist(),
                mep.redm,
                ie0_first,
                opts.amplitude_quadrature_points,
                opts.interpolation_order,
            )
            for ie in range(ie0_first, nemax):
                amp_sum = pe_r[ie] + pe_p[ie]
                val = amp_sum * amp_sum
                if iprod == 0:
                    csr = max(0.0, float(cosx[ie]))
                    csp = max(0.0, float(cosxp[ie]))
                    costm = 0.5 * (csr + csp)
                    th_val = float(thet[ie])
                    adi_term = (
                        costm * costm * math.exp(-2.0 * th_val)
                        if th_val < 350.0
                        else 0.0
                    )
                    val += adi_term
                pe_state[ie] = val

            fit_n = min(4, nemax)
            e_sub = esv[nemax - fit_n : nemax].tolist()
            p_sub = pe_state[nemax - fit_n : nemax]
            pe_state[nemax] = max(0.0, neville_interpolate(e_sub, p_sub, float(esv[nemax])))

        if iprod == 0:
            pesum_trap.append(pe_state)
        else:
            prev_sum = pesum_trap[-1]
            pesum_trap.append([prev_sum[ie] + pe_state[ie] for ie in range(nemx)])

    unif_rows, probs_lct = lct_uniformize_probabilities(pesum_trap)

    # Evaluate ZCT and SCT probabilities on the exact same quadrature energy grid (Polyrate PSAG)
    from .tunneling import _compute_cdscsag_effective_masses_amu

    assert reaction.path is not None
    cdscmu_au = np.asarray(
        [
            m * AMU_TO_AU_MASS
            for m in _compute_cdscsag_effective_masses_amu(reaction.path)
        ],
        dtype=float,
    )
    cdscmu_spline = _NaturalCubicSpline(mep.s_bohr, cdscmu_au)
    sag_nodes, sag_weights = gauss_legendre_quadrature(0.0, 1.0, 41)

    probs_zct_list: list[float] = []
    probs_sct_list: list[float] = []
    probs_muomt_list: list[float] = []
    sqrt_two_redm = math.sqrt(2.0 * mep.redm)
    sqrt_two = math.sqrt(2.0)
    for ie in range(nemax):
        ener_au = float(esv[ie])
        sl_tp = float(trnpt1[ie])
        sr_tp = float(trnpt2[ie])
        span = sr_tp - sl_tp
        if span <= 1.0e-15:
            p_zct = 0.5
            p_sct = 0.5
        else:
            sum_zct = 0.0
            sum_sct = 0.0
            for u_node, u_wt in zip(sag_nodes, sag_weights, strict=True):
                s_here = sl_tp + u_node * span
                diff_v = vad_spline.evaluate(s_here) - ener_au
                if diff_v > 0.0:
                    root_v = math.sqrt(diff_v)
                    sum_zct += u_wt * root_v
                    mu_here = max(0.0, cdscmu_spline.evaluate(s_here))
                    sum_sct += u_wt * root_v * math.sqrt(mu_here)
            theta_zct = span * sqrt_two_redm * sum_zct
            theta_sct = span * sqrt_two * sum_sct
            p_zct = 0.0 if theta_zct > 350.0 else 1.0 / (1.0 + math.exp(2.0 * theta_zct))
            p_sct = 0.0 if theta_sct > 350.0 else 1.0 / (1.0 + math.exp(2.0 * theta_sct))
        p_lct = probs_lct[ie]
        probs_zct_list.append(p_zct)
        probs_sct_list.append(p_sct)
        probs_muomt_list.append(max(p_zct, p_sct, p_lct))

    energies_kj = tuple(float(esv[ie]) * HARTREE_KJ_MOL for ie in range(nemax))
    weights_kj = tuple(float(ewt_list[ie]) * HARTREE_KJ_MOL for ie in range(nemax))

    return LctResult(
        version=opts.version,
        threshold_kj_mol=mep.ez * HARTREE_KJ_MOL,
        barrier_kj_mol=v_max * HARTREE_KJ_MOL,
        energies_kj_mol=energies_kj,
        weights_kj_mol=weights_kj,
        probabilities_lct=probs_lct,
        probabilities_zct=tuple(probs_zct_list),
        probabilities_sct=tuple(probs_sct_list),
        probabilities_muomt=tuple(probs_muomt_list),
        max_excited_state=nprod,
        state_uniformized_probabilities=unif_rows,
    )


def _boltz_from_kj(
    energies_kj_mol: tuple[float, ...],
    weights_kj_mol: tuple[float, ...],
    probabilities: tuple[float, ...],
    barrier_kj_mol: float,
    temperature_k: float,
) -> float:
    scale = 1000.0 / AVOGADRO_MOL
    e_j = tuple(val * scale for val in energies_kj_mol)
    w_j = tuple(val * scale for val in weights_kj_mol)
    return thermal_boltz_probabilities_kappa(
        e_j, w_j, probabilities, barrier_kj_mol * scale, temperature_k
    )


def lct_kappa_from_result(result: LctResult, temperature_k: float) -> float:
    """Return the thermal LCT (LCG3 or LCG4) transmission coefficient from ``LctResult``."""

    return _boltz_from_kj(
        result.energies_kj_mol,
        result.weights_kj_mol,
        result.probabilities_lct,
        result.barrier_kj_mol,
        temperature_k,
    )


def zct_kappa_from_result(result: LctResult, temperature_k: float) -> float:
    """Return the thermal ZCT transmission coefficient integrated on ``result``'s grid."""

    return _boltz_from_kj(
        result.energies_kj_mol,
        result.weights_kj_mol,
        result.probabilities_zct,
        result.barrier_kj_mol,
        temperature_k,
    )


def sct_kappa_from_result(result: LctResult, temperature_k: float) -> float:
    """Return the thermal SCT transmission coefficient integrated on ``result``'s grid."""

    return _boltz_from_kj(
        result.energies_kj_mol,
        result.weights_kj_mol,
        result.probabilities_sct,
        result.barrier_kj_mol,
        temperature_k,
    )


def muomt_kappa_from_result(result: LctResult, temperature_k: float) -> float:
    """Return the true thermal muOMT transmission coefficient from ``LctResult``."""

    return _boltz_from_kj(
        result.energies_kj_mol,
        result.weights_kj_mol,
        result.probabilities_muomt,
        result.barrier_kj_mol,
        temperature_k,
    )


def comt_kappa_from_result(result: LctResult, temperature_k: float) -> float:
    """Return the Canonical OMT transmission coefficient max(ZCT, SCT, LCT) from ``LctResult``."""

    return max(
        zct_kappa_from_result(result, temperature_k),
        sct_kappa_from_result(result, temperature_k),
        lct_kappa_from_result(result, temperature_k),
    )


def lct_kappa(
    reaction: Reaction,
    temperature_k: float,
    options: LctOptions | None = None,
) -> float:
    """Compute the canonical LCT (LCG3 or LCG4) transmission coefficient at ``temperature_k``."""

    return lct_kappa_from_result(reaction_lct_result(reaction, options), temperature_k)
