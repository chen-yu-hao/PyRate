"""Potential energy surface abstractions, analytical surfaces, and stationary-point tools."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .constants import (
    BOHR_ANGSTROM,
    HARTREE_KJ_MOL,
    KJ_MOL_TO_HARTREE,
)
from .elements import atomic_mass_amu
from .geometry import classify_rotor, principal_moments_amu_a2
from .models import Atom, ElectronicState, Species
from .native import leps_energy_and_gradient, project_symmetric_matrix
from .normal_modes import _rigid_basis, normal_modes

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PesEvaluation:
    """Single-point potential energy and optional analytical/numerical derivatives."""

    energy_kj_mol: float
    gradient_kj_mol_angstrom: FloatArray | None = None
    hessian_kj_mol_angstrom2: FloatArray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def energy_hartree(self) -> float:
        return float(self.energy_kj_mol * KJ_MOL_TO_HARTREE)

    @property
    def gradient_hartree_bohr(self) -> FloatArray | None:
        if self.gradient_kj_mol_angstrom is None:
            return None
        return np.asarray(
            self.gradient_kj_mol_angstrom * (KJ_MOL_TO_HARTREE * BOHR_ANGSTROM),
            dtype=float,
        )

    @property
    def hessian_hartree_bohr2(self) -> FloatArray | None:
        if self.hessian_kj_mol_angstrom2 is None:
            return None
        return np.asarray(
            self.hessian_kj_mol_angstrom2 * (KJ_MOL_TO_HARTREE * BOHR_ANGSTROM * BOHR_ANGSTROM),
            dtype=float,
        )


def _normalize_coords(symbols: Sequence[str], coords_angstrom: npt.ArrayLike) -> FloatArray:
    n_atoms = len(symbols)
    if n_atoms == 0:
        raise ValueError("symbols cannot be empty")
    coords: FloatArray = np.asarray(coords_angstrom, dtype=np.float64)
    if coords.shape != (n_atoms, 3):
        raise ValueError(f"coords_angstrom must have shape {(n_atoms, 3)}, got {coords.shape}")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords_angstrom contains non-finite values")
    return np.array(coords, dtype=np.float64, copy=True)


class PotentialEnergySurface(ABC):
    """Unified interface for analytical and direct-dynamics potential energy surfaces."""

    def __init__(
        self,
        *,
        step_bohr: float = 1.0e-4,
        hessian_difference: str = "central",
    ) -> None:
        if step_bohr <= 0.0:
            raise ValueError("finite-difference step_bohr must be positive")
        hdiff = hessian_difference.strip().lower()
        if hdiff not in {"central", "forward", "first"}:
            raise ValueError("hessian_difference must be 'central' or 'forward'")
        self.default_step_bohr = float(step_bohr)
        self.default_hessian_difference = "forward" if hdiff in {"forward", "first"} else "central"

    @abstractmethod
    def _evaluate_raw(
        self,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> PesEvaluation:
        """Compute native PES energy and whichever derivative orders the surface supports."""

    def numerical_gradient(
        self,
        symbols: Sequence[str],
        coords_angstrom: npt.ArrayLike,
        *,
        step_bohr: float | None = None,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> FloatArray:
        """Compute central finite-difference Cartesian gradient in kJ mol⁻¹ Å⁻¹."""

        sym_tuple = tuple(symbols)
        coords = _normalize_coords(sym_tuple, coords_angstrom)
        h_ang = (self.default_step_bohr if step_bohr is None else float(step_bohr)) * BOHR_ANGSTROM
        if h_ang <= 0.0:
            raise ValueError("step_bohr must be positive")
        n_atoms = len(sym_tuple)
        grad = np.zeros((n_atoms, 3), dtype=float)
        for atom_idx in range(n_atoms):
            for cart in range(3):
                disp_p = coords.copy()
                disp_m = coords.copy()
                disp_p[atom_idx, cart] += h_ang
                disp_m[atom_idx, cart] -= h_ang
                ep = self._evaluate_raw(
                    sym_tuple,
                    disp_p,
                    need_gradient=False,
                    need_hessian=False,
                    charge=charge,
                    multiplicity=multiplicity,
                ).energy_kj_mol
                em = self._evaluate_raw(
                    sym_tuple,
                    disp_m,
                    need_gradient=False,
                    need_hessian=False,
                    charge=charge,
                    multiplicity=multiplicity,
                ).energy_kj_mol
                grad[atom_idx, cart] = (ep - em) / (2.0 * h_ang)
        return grad

    def numerical_hessian(
        self,
        symbols: Sequence[str],
        coords_angstrom: npt.ArrayLike,
        *,
        step_bohr: float | None = None,
        difference: str | None = None,
        center_eval: PesEvaluation | None = None,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> FloatArray:
        """Compute finite-difference Cartesian Hessian in kJ mol⁻¹ Å⁻²."""

        sym_tuple = tuple(symbols)
        coords = _normalize_coords(sym_tuple, coords_angstrom)
        h_bohr = self.default_step_bohr if step_bohr is None else float(step_bohr)
        if h_bohr <= 0.0:
            raise ValueError("step_bohr must be positive")
        h_ang = h_bohr * BOHR_ANGSTROM
        diff_mode = (self.default_hessian_difference if difference is None else difference).lower()
        if diff_mode in {"first", "forward"}:
            diff_mode = "forward"
        elif diff_mode != "central":
            raise ValueError("difference must be 'central' or 'forward'")

        dim = 3 * len(sym_tuple)
        hess = np.zeros((dim, dim), dtype=float)

        if center_eval is None:
            center_eval = self._evaluate_raw(
                sym_tuple,
                coords,
                need_gradient=True,
                need_hessian=False,
                charge=charge,
                multiplicity=multiplicity,
            )

        if center_eval.gradient_kj_mol_angstrom is not None:
            g0 = center_eval.gradient_kj_mol_angstrom.reshape(-1)
            for j in range(dim):
                atom_j, cart_j = divmod(j, 3)
                disp_p = coords.copy()
                disp_p[atom_j, cart_j] += h_ang
                gp_eval = self._evaluate_raw(
                    sym_tuple,
                    disp_p,
                    need_gradient=True,
                    need_hessian=False,
                    charge=charge,
                    multiplicity=multiplicity,
                )
                assert gp_eval.gradient_kj_mol_angstrom is not None
                gp = gp_eval.gradient_kj_mol_angstrom.reshape(-1)
                if diff_mode == "forward":
                    hess[:, j] = (gp - g0) / h_ang
                else:
                    disp_m = coords.copy()
                    disp_m[atom_j, cart_j] -= h_ang
                    gm_eval = self._evaluate_raw(
                        sym_tuple,
                        disp_m,
                        need_gradient=True,
                        need_hessian=False,
                        charge=charge,
                        multiplicity=multiplicity,
                    )
                    assert gm_eval.gradient_kj_mol_angstrom is not None
                    gm = gm_eval.gradient_kj_mol_angstrom.reshape(-1)
                    hess[:, j] = (gp - gm) / (2.0 * h_ang)
            return np.asarray(0.5 * (hess + hess.T), dtype=float)

        # Energy-only finite-difference Hessian fallback (second-order accurate for central).
        e0 = center_eval.energy_kj_mol
        inv_h2 = 1.0 / (h_ang * h_ang)
        if diff_mode == "forward":
            e_plus = np.zeros(dim, dtype=float)
            for i in range(dim):
                ai, ci = divmod(i, 3)
                disp = coords.copy()
                disp[ai, ci] += h_ang
                e_plus[i] = self._evaluate_raw(
                    sym_tuple, disp, need_gradient=False, need_hessian=False,
                    charge=charge, multiplicity=multiplicity,
                ).energy_kj_mol
            for i in range(dim):
                ai, ci = divmod(i, 3)
                disp2 = coords.copy()
                disp2[ai, ci] += 2.0 * h_ang
                e2 = self._evaluate_raw(
                    sym_tuple, disp2, need_gradient=False, need_hessian=False,
                    charge=charge, multiplicity=multiplicity,
                ).energy_kj_mol
                hess[i, i] = (e2 - 2.0 * e_plus[i] + e0) * inv_h2
                for j in range(i + 1, dim):
                    aj, cj = divmod(j, 3)
                    disp_ij = coords.copy()
                    disp_ij[ai, ci] += h_ang
                    disp_ij[aj, cj] += h_ang
                    e_ij = self._evaluate_raw(
                        sym_tuple, disp_ij, need_gradient=False, need_hessian=False,
                        charge=charge, multiplicity=multiplicity,
                    ).energy_kj_mol
                    val = (e_ij - e_plus[i] - e_plus[j] + e0) * inv_h2
                    hess[i, j] = hess[j, i] = val
            return hess

        # Central energy-only second differences
        for i in range(dim):
            ai, ci = divmod(i, 3)
            disp_p = coords.copy()
            disp_m = coords.copy()
            disp_p[ai, ci] += h_ang
            disp_m[ai, ci] -= h_ang
            ep = self._evaluate_raw(
                sym_tuple, disp_p, need_gradient=False, need_hessian=False,
                charge=charge, multiplicity=multiplicity,
            ).energy_kj_mol
            em = self._evaluate_raw(
                sym_tuple, disp_m, need_gradient=False, need_hessian=False,
                charge=charge, multiplicity=multiplicity,
            ).energy_kj_mol
            hess[i, i] = (ep - 2.0 * e0 + em) * inv_h2
            for j in range(i + 1, dim):
                aj, cj = divmod(j, 3)
                dpp = coords.copy()
                dpm = coords.copy()
                dmp = coords.copy()
                dmm = coords.copy()
                dpp[ai, ci] += h_ang
                dpp[aj, cj] += h_ang
                dpm[ai, ci] += h_ang
                dpm[aj, cj] -= h_ang
                dmp[ai, ci] -= h_ang
                dmp[aj, cj] += h_ang
                dmm[ai, ci] -= h_ang
                dmm[aj, cj] -= h_ang
                epp = self._evaluate_raw(
                    sym_tuple, dpp, need_gradient=False, need_hessian=False,
                    charge=charge, multiplicity=multiplicity,
                ).energy_kj_mol
                epm = self._evaluate_raw(
                    sym_tuple, dpm, need_gradient=False, need_hessian=False,
                    charge=charge, multiplicity=multiplicity,
                ).energy_kj_mol
                emp = self._evaluate_raw(
                    sym_tuple, dmp, need_gradient=False, need_hessian=False,
                    charge=charge, multiplicity=multiplicity,
                ).energy_kj_mol
                emm = self._evaluate_raw(
                    sym_tuple, dmm, need_gradient=False, need_hessian=False,
                    charge=charge, multiplicity=multiplicity,
                ).energy_kj_mol
                val = 0.25 * (epp - epm - emp + emm) * inv_h2
                hess[i, j] = hess[j, i] = val
        return hess

    def evaluate(
        self,
        symbols: Sequence[str],
        coords_angstrom: npt.ArrayLike,
        *,
        need_gradient: bool = False,
        need_hessian: bool = False,
        charge: int = 0,
        multiplicity: int = 1,
        hessian_step_bohr: float | None = None,
        hessian_difference: str | None = None,
    ) -> PesEvaluation:
        """Evaluate energy, filling missing gradient/Hessian numerically when requested."""

        sym_tuple = tuple(symbols)
        coords = _normalize_coords(sym_tuple, coords_angstrom)
        raw = self._evaluate_raw(
            sym_tuple,
            coords,
            need_gradient=need_gradient or need_hessian,
            need_hessian=need_hessian,
            charge=charge,
            multiplicity=multiplicity,
        )
        grad = raw.gradient_kj_mol_angstrom
        hess = raw.hessian_kj_mol_angstrom2
        n_atoms = len(sym_tuple)
        if grad is not None:
            grad = np.asarray(grad, dtype=float).reshape((n_atoms, 3))
        if hess is not None:
            dim = 3 * n_atoms
            hess = np.asarray(hess, dtype=float).reshape((dim, dim))
            hess = np.asarray(0.5 * (hess + hess.T), dtype=float)

        if need_gradient and grad is None:
            grad = self.numerical_gradient(
                sym_tuple,
                coords,
                step_bohr=hessian_step_bohr,
                charge=charge,
                multiplicity=multiplicity,
            )
        if need_hessian and hess is None:
            center = PesEvaluation(
                energy_kj_mol=raw.energy_kj_mol,
                gradient_kj_mol_angstrom=raw.gradient_kj_mol_angstrom,
                hessian_kj_mol_angstrom2=None,
                metadata=raw.metadata,
            )
            hess = self.numerical_hessian(
                sym_tuple,
                coords,
                step_bohr=hessian_step_bohr,
                difference=hessian_difference,
                center_eval=center,
                charge=charge,
                multiplicity=multiplicity,
            )
        return PesEvaluation(
            energy_kj_mol=float(raw.energy_kj_mol),
            gradient_kj_mol_angstrom=grad,
            hessian_kj_mol_angstrom2=hess,
            metadata=raw.metadata,
        )


class CallablePes(PotentialEnergySurface):
    """Wrap Python callables returning energies and derivatives in kJ mol⁻¹ / Å."""

    def __init__(
        self,
        energy_fn: Callable[[tuple[str, ...], FloatArray], float],
        *,
        gradient_fn: Callable[[tuple[str, ...], FloatArray], npt.ArrayLike] | None = None,
        hessian_fn: Callable[[tuple[str, ...], FloatArray], npt.ArrayLike] | None = None,
        step_bohr: float = 1.0e-4,
        hessian_difference: str = "central",
    ) -> None:
        super().__init__(step_bohr=step_bohr, hessian_difference=hessian_difference)
        self._energy_fn = energy_fn
        self._gradient_fn = gradient_fn
        self._hessian_fn = hessian_fn

    def _evaluate_raw(
        self,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> PesEvaluation:
        del charge, multiplicity
        energy = float(self._energy_fn(symbols, coords_angstrom))
        grad: FloatArray | None = None
        hess: FloatArray | None = None
        if (need_gradient or need_hessian) and self._gradient_fn is not None:
            grad = np.asarray(self._gradient_fn(symbols, coords_angstrom), dtype=float).reshape(
                (len(symbols), 3)
            )
        if need_hessian and self._hessian_fn is not None:
            dim = 3 * len(symbols)
            hess = np.asarray(self._hessian_fn(symbols, coords_angstrom), dtype=float).reshape(
                (dim, dim)
            )
        return PesEvaluation(
            energy_kj_mol=energy,
            gradient_kj_mol_angstrom=grad,
            hessian_kj_mol_angstrom2=hess,
        )


class PolyratePotAdapter(PotentialEnergySurface):
    """Adapt a Polyrate/Fortran-style callable operating in Bohr and Hartree.

    The wrapped callable receives ``coords_bohr`` (shape ``(3N,)`` when
    ``flat_coords=True``, else ``(N, 3)``) and may return either ``energy_hartree``,
    ``(energy_hartree, grad_hartree_bohr)``, or
    ``(energy_hartree, grad_hartree_bohr, hess_hartree_bohr2)``.
    """

    def __init__(
        self,
        pot_fn: Callable[[FloatArray], Any],
        *,
        flat_coords: bool = True,
        bohr_angstrom: float = BOHR_ANGSTROM,
        reference_energy_hartree: float = 0.0,
        step_bohr: float = 1.0e-4,
        hessian_difference: str = "central",
    ) -> None:
        super().__init__(step_bohr=step_bohr, hessian_difference=hessian_difference)
        self._pot_fn = pot_fn
        self.flat_coords = bool(flat_coords)
        self.bohr_angstrom = float(bohr_angstrom)
        self.reference_energy_hartree = float(reference_energy_hartree)

    def _evaluate_raw(
        self,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> PesEvaluation:
        del need_gradient, need_hessian, charge, multiplicity
        n_atoms = len(symbols)
        coords_bohr = coords_angstrom / self.bohr_angstrom
        arg = coords_bohr.reshape(-1) if self.flat_coords else coords_bohr
        out = self._pot_fn(arg)

        grad_kj: FloatArray | None = None
        hess_kj: FloatArray | None = None
        if isinstance(out, tuple):
            e_au = float(out[0])
            if len(out) >= 2 and out[1] is not None:
                grad_au = np.asarray(out[1], dtype=float).reshape((n_atoms, 3))
                grad_kj = grad_au * (HARTREE_KJ_MOL / self.bohr_angstrom)
            if len(out) >= 3 and out[2] is not None:
                dim = 3 * n_atoms
                hess_au = np.asarray(out[2], dtype=float).reshape((dim, dim))
                hess_kj = hess_au * (HARTREE_KJ_MOL / (self.bohr_angstrom * self.bohr_angstrom))
        else:
            e_au = float(out)

        return PesEvaluation(
            energy_kj_mol=(e_au - self.reference_energy_hartree) * HARTREE_KJ_MOL,
            gradient_kj_mol_angstrom=grad_kj,
            hessian_kj_mol_angstrom2=hess_kj,
        )


class ExtendedLepsSurface(PotentialEnergySurface):
    """3-atom Extended London–Eyring–Polanyi–Sato (LEPS) analytical surface.

    Diatomic pairs follow ``0: (atom0, atom1)``, ``1: (atom1, atom2)``, ``2: (atom0, atom2)``.
    Exact analytical gradients are evaluated via native C++/Python kernels.
    """

    def __init__(
        self,
        *,
        de_kcal_mol: tuple[float, float, float],
        re_angstrom: tuple[float, float, float],
        beta_angstrom_inv: tuple[float, float, float],
        sato: tuple[float, float, float],
        kcal_mol_to_hartree: float = 1.0 / 627.5095,
        bohr_angstrom: float = BOHR_ANGSTROM,
        atom_symbols: tuple[str, str, str] | None = None,
        zero_at_reactants: bool = True,
        reference_energy_hartree: float | None = None,
        step_bohr: float = 1.0e-5,
    ) -> None:
        super().__init__(step_bohr=step_bohr, hessian_difference="central")
        self.bohr_angstrom = float(bohr_angstrom)
        self.kcal_mol_to_hartree = float(kcal_mol_to_hartree)
        self.de_hartree: tuple[float, float, float] = (
            float(de_kcal_mol[0]) * self.kcal_mol_to_hartree,
            float(de_kcal_mol[1]) * self.kcal_mol_to_hartree,
            float(de_kcal_mol[2]) * self.kcal_mol_to_hartree,
        )
        self.re_bohr: tuple[float, float, float] = (
            float(re_angstrom[0]) / self.bohr_angstrom,
            float(re_angstrom[1]) / self.bohr_angstrom,
            float(re_angstrom[2]) / self.bohr_angstrom,
        )
        self.beta_bohr_inv: tuple[float, float, float] = (
            float(beta_angstrom_inv[0]) * self.bohr_angstrom,
            float(beta_angstrom_inv[1]) * self.bohr_angstrom,
            float(beta_angstrom_inv[2]) * self.bohr_angstrom,
        )
        self.sato: tuple[float, float, float] = (float(sato[0]), float(sato[1]), float(sato[2]))
        self.atom_symbols = atom_symbols

        if reference_energy_hartree is not None:
            self.reference_energy_hartree = float(reference_energy_hartree)
        elif zero_at_reactants:
            # Asymptotic A + BC(r = re_BC) reference energy
            asym_bohr = np.array(
                [
                    [0.0, 0.0, -500.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, self.re_bohr[1]],
                ],
                dtype=float,
            )
            e_ref, _ = leps_energy_and_gradient(
                asym_bohr * self.bohr_angstrom,
                self.de_hartree,
                self.re_bohr,
                self.beta_bohr_inv,
                self.sato,
                self.bohr_angstrom,
            )
            self.reference_energy_hartree = float(e_ref)
        else:
            self.reference_energy_hartree = 0.0

    def _permutation(self, symbols: tuple[str, ...]) -> tuple[int, int, int]:
        if len(symbols) != 3:
            raise ValueError(f"ExtendedLepsSurface requires exactly 3 atoms, got {len(symbols)}")
        if self.atom_symbols is None or tuple(symbols) == self.atom_symbols:
            return (0, 1, 2)
        from .elements import atomic_number

        try:
            z_in = [atomic_number(sym) for sym in symbols]
            z_ref = [atomic_number(sym) for sym in self.atom_symbols]
        except ValueError:
            return (0, 1, 2)
        if z_in == z_ref:
            return (0, 1, 2)
        if sorted(z_in) == sorted(z_ref) and len(set(z_ref)) == 3:
            idx_map = {z: i for i, z in enumerate(z_in)}
            return (idx_map[z_ref[0]], idx_map[z_ref[1]], idx_map[z_ref[2]])
        return (0, 1, 2)

    def _evaluate_raw(
        self,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> PesEvaluation:
        del charge, multiplicity, need_hessian
        perm = self._permutation(symbols)
        ordered_coords = coords_angstrom[list(perm)]
        e_au, grad_au = leps_energy_and_gradient(
            ordered_coords,
            self.de_hartree,
            self.re_bohr,
            self.beta_bohr_inv,
            self.sato,
            self.bohr_angstrom,
        )
        grad_kj: FloatArray | None = None
        if need_gradient:
            unpermuted_au = np.zeros_like(grad_au)
            for canonical_idx, user_idx in enumerate(perm):
                unpermuted_au[user_idx] = grad_au[canonical_idx]
            grad_kj = np.asarray(unpermuted_au * (HARTREE_KJ_MOL / self.bohr_angstrom), dtype=float)
        return PesEvaluation(
            energy_kj_mol=(e_au - self.reference_energy_hartree) * HARTREE_KJ_MOL,
            gradient_kj_mol_angstrom=grad_kj,
            hessian_kj_mol_angstrom2=None,
            metadata={"surface": "extended_leps"},
        )


def cl_hbr_leps_surface(*, zero_at_reactants: bool = True) -> ExtendedLepsSurface:
    """Extended LEPS surface for ``Cl + HBr -> HCl + Br`` (Kuntz et al., JCP 44, 1168 (1966)).

    Matches the exact Polyrate 2023 ``clhbrtr1``/``clhbrtr3`` parameters and Bohr/Hartree factors.
    """

    return ExtendedLepsSurface(
        de_kcal_mol=(106.4, 90.24, 52.09),
        re_angstrom=(1.275, 1.414, 2.136),
        beta_angstrom_inv=(1.867, 1.851, 1.923),
        sato=(0.02, 0.02, 0.0),
        kcal_mol_to_hartree=1.0 / 627.5095,
        bohr_angstrom=0.529177106,
        atom_symbols=("Cl", "H", "Br"),
        zero_at_reactants=zero_at_reactants,
        step_bohr=1.0e-5,
    )


def h3_leps_surface(*, zero_at_reactants: bool = True) -> ExtendedLepsSurface:
    """Symmetric Extended LEPS surface for collinear ``H + H2 -> H2 + H``."""

    return ExtendedLepsSurface(
        de_kcal_mol=(109.458, 109.458, 109.458),
        re_angstrom=(0.7414, 0.7414, 0.7414),
        beta_angstrom_inv=(1.942, 1.942, 1.942),
        sato=(0.145, 0.145, 0.145),
        kcal_mol_to_hartree=1.0 / 627.5095,
        bohr_angstrom=0.529177106,
        atom_symbols=("H", "H", "H"),
        zero_at_reactants=zero_at_reactants,
        step_bohr=1.0e-5,
    )


class MullerBrownSurface(PotentialEnergySurface):
    """SE(3)-invariant 3-atom Cartesian embedding of the Müller–Brown benchmark surface.

    The bond lengths ``r_01`` and ``r_12`` map smoothly into the standard Müller–Brown
    ``(x, y)`` plane via ``x = r_01 - r_x0``, ``y = r_12 - r_y0``, while a harmonic dot-product
    restraint constrains the angle around 90 degrees so the full 9D Hessian has 3 real (or 2 real +
    1 imaginary) vibrations and 6 exact rigid modes.
    """

    _A = (-200.0, -100.0, -170.0, 15.0)
    _a = (-1.0, -1.0, -6.5, 0.7)
    _b = (0.0, 0.0, 11.0, 0.6)
    _c = (-10.0, -10.0, -6.5, 0.7)
    _x0 = (1.0, 0.0, -0.5, -1.0)
    _y0 = (0.0, 0.5, 1.5, 1.0)

    def __init__(
        self,
        *,
        r_x0_angstrom: float = 2.5,
        r_y0_angstrom: float = 1.5,
        energy_scale_kj_mol: float = 0.5,
        bend_force_kj_mol: float = 200.0,
        reference_kj_mol: float | None = None,
    ) -> None:
        super().__init__(step_bohr=1.0e-5, hessian_difference="central")
        self.r_x0_angstrom = float(r_x0_angstrom)
        self.r_y0_angstrom = float(r_y0_angstrom)
        self.energy_scale_kj_mol = float(energy_scale_kj_mol)
        self.bend_force_kj_mol = float(bend_force_kj_mol)
        self.reference_kj_mol = float(
            reference_kj_mol
            if reference_kj_mol is not None
            else self._mb_energy_and_grad(-0.55822363, 1.44172584)[0] * self.energy_scale_kj_mol
        )

    @classmethod
    def _mb_energy_and_grad(cls, x: float, y: float) -> tuple[float, float, float]:
        energy = 0.0
        dvdx = 0.0
        dvdy = 0.0
        for idx in range(4):
            dx = x - cls._x0[idx]
            dy = y - cls._y0[idx]
            term = cls._A[idx] * math.exp(
                cls._a[idx] * dx * dx + cls._b[idx] * dx * dy + cls._c[idx] * dy * dy
            )
            energy += term
            dvdx += term * (2.0 * cls._a[idx] * dx + cls._b[idx] * dy)
            dvdy += term * (cls._b[idx] * dx + 2.0 * cls._c[idx] * dy)
        return energy, dvdx, dvdy

    def _evaluate_raw(
        self,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> PesEvaluation:
        del charge, multiplicity, need_hessian
        if len(symbols) != 3:
            raise ValueError("MullerBrownSurface requires 3 atoms")
        v01 = coords_angstrom[0] - coords_angstrom[1]
        v21 = coords_angstrom[2] - coords_angstrom[1]
        r01 = float(np.linalg.norm(v01))
        r12 = float(np.linalg.norm(v21))
        if r01 <= 1.0e-12 or r12 <= 1.0e-12:
            raise ValueError("MullerBrownSurface atomic separation collapsed to zero")
        u01 = v01 / r01
        u21 = v21 / r12

        x = r01 - self.r_x0_angstrom
        y = r12 - self.r_y0_angstrom
        mb_e, mb_dx, mb_dy = self._mb_energy_and_grad(x, y)
        cos_theta = float(np.dot(u01, u21))
        v_bend = 0.5 * self.bend_force_kj_mol * (cos_theta * cos_theta)
        energy = mb_e * self.energy_scale_kj_mol - self.reference_kj_mol + v_bend

        grad: FloatArray | None = None
        if need_gradient:
            grad = np.zeros((3, 3), dtype=float)
            dedr01 = mb_dx * self.energy_scale_kj_mol
            dedr12 = mb_dy * self.energy_scale_kj_mol
            grad[0] += dedr01 * u01
            grad[1] -= dedr01 * u01
            grad[2] += dedr12 * u21
            grad[1] -= dedr12 * u21

            # Bending derivative: d(cos_theta)/dR
            d_cos_dr0 = (u21 - cos_theta * u01) / r01
            d_cos_dr2 = (u01 - cos_theta * u21) / r12
            pref = self.bend_force_kj_mol * cos_theta
            grad[0] += pref * d_cos_dr0
            grad[2] += pref * d_cos_dr2
            grad[1] -= pref * (d_cos_dr0 + d_cos_dr2)

        return PesEvaluation(
            energy_kj_mol=energy,
            gradient_kj_mol_angstrom=grad,
            hessian_kj_mol_angstrom2=None,
            metadata={"surface": "muller_brown", "x": x, "y": y},
        )


class EckartValleySurface(PotentialEnergySurface):
    """Analytical 3-atom collinear Eckart barrier + coupled harmonic stretch & bend PES."""

    def __init__(
        self,
        *,
        v1_kj_mol: float = 40.0,
        v2_kj_mol: float = 55.0,
        width_angstrom: float = 0.45,
        r_eq_angstrom: float = 1.2,
        stretch_force_kj_mol_a2: float = 4500.0,
        bend_force_kj_mol_a: float = 250.0,
        coupling_angstrom: float = 0.0,
    ) -> None:
        super().__init__(step_bohr=1.0e-5, hessian_difference="central")
        self.v1_kj_mol = float(v1_kj_mol)
        self.v2_kj_mol = float(v2_kj_mol)
        self.width_angstrom = float(width_angstrom)
        self.r_eq_angstrom = float(r_eq_angstrom)
        self.stretch_force = float(stretch_force_kj_mol_a2)
        self.bend_force = float(bend_force_kj_mol_a)
        self.coupling_angstrom = float(coupling_angstrom)

    def _eckart_1d(self, u: float) -> tuple[float, float]:
        # Eckart parameterization: u = r01 - r12
        # V(u) = A*y/(1+y) + B*y/(1+y)^2 with y = exp(u / width)
        a_coef = self.v1_kj_mol - self.v2_kj_mol
        b_coef = (math.sqrt(self.v1_kj_mol) + math.sqrt(self.v2_kj_mol)) ** 2
        z = max(-300.0, min(300.0, u / self.width_angstrom))
        ez = math.exp(z)
        denom = 1.0 + ez
        s = ez / denom  # y/(1+y) = 1/(1+e^-z)
        ds_dz = s * (1.0 - s)
        val = a_coef * s + b_coef * s * (1.0 - s)
        dval_dz = a_coef * ds_dz + b_coef * (1.0 - 2.0 * s) * ds_dz
        return val, dval_dz / self.width_angstrom

    def _evaluate_raw(
        self,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> PesEvaluation:
        del charge, multiplicity, need_hessian
        if len(symbols) != 3:
            raise ValueError("EckartValleySurface requires 3 atoms")
        d01 = coords_angstrom[0] - coords_angstrom[1]
        d12 = coords_angstrom[1] - coords_angstrom[2]
        d02 = coords_angstrom[0] - coords_angstrom[2]
        r01 = float(np.linalg.norm(d01))
        r12 = float(np.linalg.norm(d12))
        r02 = float(np.linalg.norm(d02))
        if r01 <= 1.0e-12 or r12 <= 1.0e-12 or r02 <= 1.0e-12:
            raise ValueError("EckartValleySurface atomic separation collapsed to zero")
        u01 = d01 / r01
        u12 = d12 / r12
        u02 = d02 / r02

        u_rxn = r01 - r12
        v_eck, dv_eck_du = self._eckart_1d(u_rxn)

        arg_c = max(-50.0, min(50.0, u_rxn / self.width_angstrom))
        sech = 1.0 / math.cosh(arg_c)
        sech2 = sech * sech
        dsech2_du = (-2.0 / self.width_angstrom) * sech2 * math.tanh(arg_c)
        q_sym = (r01 + r12 - 2.0 * self.r_eq_angstrom) - self.coupling_angstrom * sech2
        v_stretch = 0.5 * self.stretch_force * q_sym * q_sym

        # Collinear bend invariant: r01 + r12 - r02 >= 0, vanishes identically when collinear
        delta_col = max(0.0, r01 + r12 - r02)
        v_bend = self.bend_force * delta_col
        energy = v_eck + v_stretch + v_bend

        grad: FloatArray | None = None
        if need_gradient:
            grad = np.zeros((3, 3), dtype=float)
            dq_dr01 = 1.0 - self.coupling_angstrom * dsech2_du
            dq_dr12 = 1.0 + self.coupling_angstrom * dsech2_du
            de_dr01 = dv_eck_du + self.stretch_force * q_sym * dq_dr01 + self.bend_force
            de_dr12 = -dv_eck_du + self.stretch_force * q_sym * dq_dr12 + self.bend_force
            de_dr02 = -self.bend_force

            grad[0] += de_dr01 * u01 + de_dr02 * u02
            grad[1] += -de_dr01 * u01 + de_dr12 * u12
            grad[2] += -de_dr12 * u12 - de_dr02 * u02

        return PesEvaluation(
            energy_kj_mol=energy,
            gradient_kj_mol_angstrom=grad,
            hessian_kj_mol_angstrom2=None,
            metadata={"surface": "eckart_valley"},
        )


class SupermoleculeFragmentPes(PotentialEnergySurface):
    """Evaluate a fragment inside a full-system analytical PES with rigid far spectators.

    Useful when reactants or products (e.g. ``HBr`` or ``Cl``) are evaluated on a
    fixed-atomicity 3-atom surface such as :class:`ExtendedLepsSurface`.
    """

    def __init__(
        self,
        full_pes: PotentialEnergySurface,
        full_symbols: Sequence[str],
        fragment_indices: Sequence[int],
        spectator_coords_angstrom: npt.ArrayLike,
        *,
        separation_angstrom: float = 200.0,
        energy_offset_kj_mol: float = 0.0,
    ) -> None:
        super().__init__(
            step_bohr=full_pes.default_step_bohr,
            hessian_difference=full_pes.default_hessian_difference,
        )
        self.full_pes = full_pes
        self.full_symbols = tuple(full_symbols)
        n_full = len(self.full_symbols)
        self.fragment_indices = tuple(int(i) for i in fragment_indices)
        if not self.fragment_indices or len(set(self.fragment_indices)) != len(
            self.fragment_indices
        ):
            raise ValueError("fragment_indices must be non-empty and unique")
        if any(idx < 0 or idx >= n_full for idx in self.fragment_indices):
            raise ValueError("fragment_indices out of bounds for full_symbols")
        frag_set = set(self.fragment_indices)
        self.spectator_indices = tuple(i for i in range(n_full) if i not in frag_set)
        self.spectator_coords_angstrom = _normalize_coords(
            self.full_symbols, spectator_coords_angstrom
        )
        self.separation_angstrom = float(separation_angstrom)
        self.energy_offset_kj_mol = float(energy_offset_kj_mol)

    def _evaluate_raw(
        self,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> PesEvaluation:
        if len(symbols) != len(self.fragment_indices):
            raise ValueError(
                f"fragment expected {len(self.fragment_indices)} atoms, got {len(symbols)}"
            )
        full_coords = np.array(self.spectator_coords_angstrom, dtype=float, copy=True)
        for local_i, full_i in enumerate(self.fragment_indices):
            full_coords[full_i] = coords_angstrom[local_i]

        if self.spectator_indices:
            frag_center = np.mean(coords_angstrom, axis=0)
            spec_pts = full_coords[list(self.spectator_indices)]
            spec_center = np.mean(spec_pts, axis=0)
            delta = spec_center - frag_center
            dist = float(np.linalg.norm(delta))
            direction = delta / dist if dist > 1.0e-8 else np.array([0.0, 0.0, 1.0], dtype=float)
            shifted = spec_pts - spec_center + (frag_center + self.separation_angstrom * direction)
            for local_s, full_s in enumerate(self.spectator_indices):
                full_coords[full_s] = shifted[local_s]

        raw = self.full_pes._evaluate_raw(
            self.full_symbols,
            full_coords,
            need_gradient=need_gradient,
            need_hessian=need_hessian,
            charge=charge,
            multiplicity=multiplicity,
        )
        grad_kj: FloatArray | None = None
        if raw.gradient_kj_mol_angstrom is not None:
            grad_kj = np.asarray(
                raw.gradient_kj_mol_angstrom[list(self.fragment_indices)], dtype=float
            ).copy()
        hess_kj: FloatArray | None = None
        if raw.hessian_kj_mol_angstrom2 is not None:
            dofs = [3 * i + c for i in self.fragment_indices for c in range(3)]
            hess_kj = np.asarray(
                raw.hessian_kj_mol_angstrom2[np.ix_(dofs, dofs)], dtype=float
            ).copy()
        return PesEvaluation(
            energy_kj_mol=raw.energy_kj_mol - self.energy_offset_kj_mol,
            gradient_kj_mol_angstrom=grad_kj,
            hessian_kj_mol_angstrom2=hess_kj,
            metadata={**raw.metadata, "fragment": True},
        )



def get_builtin_pes(name: str, **kwargs: Any) -> PotentialEnergySurface:
    """Construct a built-in analytical PES by canonical name."""

    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"cl_hbr_leps", "clhbr", "clhbr_leps"}:
        return cl_hbr_leps_surface(**kwargs)
    if normalized in {"h3_leps", "h3", "hh2_leps"}:
        return h3_leps_surface(**kwargs)
    if normalized in {"leps", "extended_leps"}:
        return ExtendedLepsSurface(**kwargs)
    if normalized in {"muller_brown", "mueller_brown"}:
        return MullerBrownSurface(**kwargs)
    if normalized in {"eckart_valley", "eckart"}:
        return EckartValleySurface(**kwargs)
    raise ValueError(f"unknown built-in PES {name!r}")


def build_pes_from_mapping(
    table: Mapping[str, Any],
    *,
    base_dir: str | None = None,
) -> PotentialEnergySurface:
    """Construct an analytical or direct-dynamics PES from a TOML ``[potential]`` mapping."""

    pes_type = str(table.get("type", "")).strip().lower().replace("-", "_")
    if not pes_type:
        raise ValueError("[potential] table requires 'type'")

    if pes_type in {"cl_hbr_leps", "clhbr", "clhbr_leps"}:
        return cl_hbr_leps_surface(zero_at_reactants=bool(table.get("zero_at_reactants", True)))
    if pes_type in {"h3_leps", "h3", "hh2_leps"}:
        return h3_leps_surface(zero_at_reactants=bool(table.get("zero_at_reactants", True)))
    if pes_type in {"leps", "extended_leps"}:
        de = tuple(float(v) for v in table["de_kcal_mol"])
        re = tuple(float(v) for v in table["re_angstrom"])
        beta = tuple(float(v) for v in table["beta_angstrom_inv"])
        sato = tuple(float(v) for v in table["sato"])
        if len(de) != 3 or len(re) != 3 or len(beta) != 3 or len(sato) != 3:
            raise ValueError("LEPS parameters must each have 3 floats")
        syms_raw = table.get("atom_symbols")
        syms = (str(syms_raw[0]), str(syms_raw[1]), str(syms_raw[2])) if syms_raw else None
        return ExtendedLepsSurface(
            de_kcal_mol=(de[0], de[1], de[2]),
            re_angstrom=(re[0], re[1], re[2]),
            beta_angstrom_inv=(beta[0], beta[1], beta[2]),
            sato=(sato[0], sato[1], sato[2]),
            kcal_mol_to_hartree=float(table.get("kcal_mol_to_hartree", 1.0 / 627.5095)),
            bohr_angstrom=float(table.get("bohr_angstrom", BOHR_ANGSTROM)),
            atom_symbols=syms,
            zero_at_reactants=bool(table.get("zero_at_reactants", True)),
            reference_energy_hartree=(
                float(table["reference_energy_hartree"])
                if table.get("reference_energy_hartree") is not None
                else None
            ),
        )
    if pes_type in {"muller_brown", "mueller_brown"}:
        return MullerBrownSurface(
            r_x0_angstrom=float(table.get("r_x0_angstrom", 2.5)),
            r_y0_angstrom=float(table.get("r_y0_angstrom", 1.5)),
            energy_scale_kj_mol=float(table.get("energy_scale_kj_mol", 0.5)),
            bend_force_kj_mol=float(table.get("bend_force_kj_mol", 200.0)),
        )
    if pes_type in {"eckart_valley", "eckart"}:
        return EckartValleySurface(
            v1_kj_mol=float(table.get("v1_kj_mol", 40.0)),
            v2_kj_mol=float(table.get("v2_kj_mol", 55.0)),
            width_angstrom=float(table.get("width_angstrom", 0.45)),
            r_eq_angstrom=float(table.get("r_eq_angstrom", 1.2)),
            stretch_force_kj_mol_a2=float(table.get("stretch_force_kj_mol_a2", 4500.0)),
            bend_force_kj_mol_a=float(table.get("bend_force_kj_mol_a", 250.0)),
            coupling_angstrom=float(table.get("coupling_angstrom", 0.0)),
        )
    if pes_type in {"command", "external", "gaussian", "gamess"}:
        from .direct_dynamics import direct_dynamics_from_mapping

        return direct_dynamics_from_mapping(table, base_dir=base_dir)
    raise ValueError(f"unsupported [potential] type {pes_type!r}")


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Result from optimizing a minimum or index-1 saddle point on a PES."""

    coords_angstrom: FloatArray
    evaluation: PesEvaluation
    normal_mode_result: Any
    converged: bool
    steps: int
    max_gradient_hartree_bohr: float
    saddle_order: int


def _resolve_masses_amu(
    symbols: Sequence[str],
    masses_amu: Sequence[float] | None,
    mass_convention: str,
) -> FloatArray:
    if masses_amu is not None:
        arr = np.asarray(masses_amu, dtype=float)
        if arr.shape != (len(symbols),):
            raise ValueError("masses_amu length must match symbols")
        return arr
    return np.asarray(
        [atomic_mass_amu(sym, mass_convention=mass_convention) for sym in symbols],
        dtype=float,
    )


def _internal_projection(
    coords_angstrom: FloatArray,
    masses_amu: FloatArray,
) -> tuple[FloatArray, int]:
    from .constants import AMU_KG, ANGSTROM_M

    return _rigid_basis(masses_amu * AMU_KG, coords_angstrom * ANGSTROM_M)


def optimize_minimum(
    pes: PotentialEnergySurface,
    symbols: Sequence[str],
    initial_coords_angstrom: npt.ArrayLike,
    *,
    masses_amu: Sequence[float] | None = None,
    mass_convention: str = "iupac",
    gtol_au: float = 1.0e-6,
    max_steps: int = 200,
    max_step_bohr: float = 0.25,
    charge: int = 0,
    multiplicity: int = 1,
) -> OptimizationResult:
    """Optimize a local minimum using projected BFGS with backtracking line search."""

    sym_tuple = tuple(symbols)
    coords_bohr = _normalize_coords(sym_tuple, initial_coords_angstrom).reshape(-1) / BOHR_ANGSTROM
    masses = _resolve_masses_amu(sym_tuple, masses_amu, mass_convention)
    dim = coords_bohr.size

    if len(sym_tuple) == 1:
        ev = pes.evaluate(
            sym_tuple,
            coords_bohr.reshape((1, 3)) * BOHR_ANGSTROM,
            need_gradient=True,
            need_hessian=True,
            charge=charge,
            multiplicity=multiplicity,
        )
        assert ev.hessian_kj_mol_angstrom2 is not None
        nm = normal_modes(
            ev.hessian_kj_mol_angstrom2,
            masses.tolist(),
            coordinates_angstrom=coords_bohr.reshape((1, 3)) * BOHR_ANGSTROM,
            hessian_unit="kj/mol/angstrom^2",
            remove_rigid_modes=True,
        )
        return OptimizationResult(
            coords_angstrom=coords_bohr.reshape((1, 3)) * BOHR_ANGSTROM,
            evaluation=ev,
            normal_mode_result=nm,
            converged=True,
            steps=0,
            max_gradient_hartree_bohr=0.0,
            saddle_order=0,
        )

    inv_h = np.eye(dim, dtype=float)
    converged = False
    ev = pes.evaluate(
        sym_tuple,
        coords_bohr.reshape((-1, 3)) * BOHR_ANGSTROM,
        need_gradient=True,
        charge=charge,
        multiplicity=multiplicity,
    )
    assert ev.gradient_hartree_bohr is not None
    grad = ev.gradient_hartree_bohr.reshape(-1)
    basis, _ = _internal_projection(coords_bohr.reshape((-1, 3)) * BOHR_ANGSTROM, masses)
    proj_grad = grad - basis @ (basis.T @ grad)
    max_g = float(np.max(np.abs(proj_grad)))
    steps_taken = 0

    for step_idx in range(max_steps):
        if max_g <= gtol_au:
            converged = True
            break
        steps_taken = step_idx + 1
        direction = -(inv_h @ proj_grad)
        direction = direction - basis @ (basis.T @ direction)
        if float(np.dot(direction, proj_grad)) >= 0.0:
            direction = -proj_grad
            inv_h = np.eye(dim, dtype=float)
        dir_norm = float(np.linalg.norm(direction))
        if dir_norm > max_step_bohr:
            direction *= max_step_bohr / dir_norm

        alpha = 1.0
        slope = float(np.dot(proj_grad, direction))
        accepted = False
        e0 = ev.energy_hartree
        next_ev = ev
        next_coords = coords_bohr
        for _ in range(25):
            cand = coords_bohr + alpha * direction
            cand_ev = pes.evaluate(
                sym_tuple,
                cand.reshape((-1, 3)) * BOHR_ANGSTROM,
                need_gradient=True,
                charge=charge,
                multiplicity=multiplicity,
            )
            if cand_ev.energy_hartree <= e0 + 1.0e-4 * alpha * slope:
                next_ev = cand_ev
                next_coords = cand
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            alpha = 1.0e-3 / max(1.0, float(np.linalg.norm(proj_grad)))
            next_coords = coords_bohr - alpha * proj_grad
            next_ev = pes.evaluate(
                sym_tuple,
                next_coords.reshape((-1, 3)) * BOHR_ANGSTROM,
                need_gradient=True,
                charge=charge,
                multiplicity=multiplicity,
            )

        assert next_ev.gradient_hartree_bohr is not None
        s_vec = next_coords - coords_bohr
        new_basis, _ = _internal_projection(next_coords.reshape((-1, 3)) * BOHR_ANGSTROM, masses)
        new_grad = next_ev.gradient_hartree_bohr.reshape(-1)
        new_proj = new_grad - new_basis @ (new_basis.T @ new_grad)
        y_vec = new_proj - proj_grad
        ys = float(np.dot(y_vec, s_vec))
        if ys > 1.0e-12 * float(np.linalg.norm(s_vec) * np.linalg.norm(y_vec)):
            rho = 1.0 / ys
            i_mat = np.eye(dim, dtype=float)
            term_l = i_mat - rho * np.outer(s_vec, y_vec)
            term_r = i_mat - rho * np.outer(y_vec, s_vec)
            inv_h = term_l @ inv_h @ term_r + rho * np.outer(s_vec, s_vec)
            inv_h = 0.5 * (inv_h + inv_h.T)

        coords_bohr = next_coords
        ev = next_ev
        basis = new_basis
        proj_grad = new_proj
        max_g = float(np.max(np.abs(proj_grad)))

    if max_g <= gtol_au:
        converged = True

    final_coords_ang = coords_bohr.reshape((-1, 3)) * BOHR_ANGSTROM
    final_ev = pes.evaluate(
        sym_tuple,
        final_coords_ang,
        need_gradient=True,
        need_hessian=True,
        charge=charge,
        multiplicity=multiplicity,
    )
    assert final_ev.hessian_kj_mol_angstrom2 is not None
    nm = normal_modes(
        final_ev.hessian_kj_mol_angstrom2,
        masses.tolist(),
        coordinates_angstrom=final_coords_ang,
        hessian_unit="kj/mol/angstrom^2",
        remove_rigid_modes=True,
    )
    saddle_order = int(np.count_nonzero(nm.frequencies_cm1 < -1.0))
    return OptimizationResult(
        coords_angstrom=final_coords_ang,
        evaluation=final_ev,
        normal_mode_result=nm,
        converged=converged,
        steps=steps_taken,
        max_gradient_hartree_bohr=max_g,
        saddle_order=saddle_order,
    )


def optimize_saddle(
    pes: PotentialEnergySurface,
    symbols: Sequence[str],
    initial_coords_angstrom: npt.ArrayLike,
    *,
    masses_amu: Sequence[float] | None = None,
    mass_convention: str = "iupac",
    gtol_au: float = 1.0e-6,
    max_steps: int = 120,
    trust_radius_bohr: float = 0.15,
    mode_vector_bohr: Sequence[float] | npt.ArrayLike | None = None,
    charge: int = 0,
    multiplicity: int = 1,
) -> OptimizationResult:
    """Locate an index-1 saddle point via Partitioned RFO (Baker's eigenvector-following)."""

    sym_tuple = tuple(symbols)
    if len(sym_tuple) < 2:
        raise ValueError("saddle-point optimization requires at least 2 atoms")
    coords_bohr = _normalize_coords(sym_tuple, initial_coords_angstrom).reshape(-1) / BOHR_ANGSTROM
    masses = _resolve_masses_amu(sym_tuple, masses_amu, mass_convention)
    tracked_mode = (
        np.asarray(mode_vector_bohr, dtype=float).reshape(-1)
        if mode_vector_bohr is not None
        else None
    )
    if tracked_mode is not None:
        t_norm = float(np.linalg.norm(tracked_mode))
        if t_norm > 0.0:
            tracked_mode = tracked_mode / t_norm

    converged = False
    steps_taken = 0
    max_g = math.inf
    final_ev = pes.evaluate(
        sym_tuple,
        coords_bohr.reshape((-1, 3)) * BOHR_ANGSTROM,
        need_gradient=True,
        need_hessian=True,
        charge=charge,
        multiplicity=multiplicity,
    )

    for step_idx in range(max_steps):
        ev = final_ev if step_idx == 0 else pes.evaluate(
            sym_tuple,
            coords_bohr.reshape((-1, 3)) * BOHR_ANGSTROM,
            need_gradient=True,
            need_hessian=True,
            charge=charge,
            multiplicity=multiplicity,
        )
        final_ev = ev
        assert ev.gradient_hartree_bohr is not None and ev.hessian_hartree_bohr2 is not None
        grad = ev.gradient_hartree_bohr.reshape(-1)
        hess = ev.hessian_hartree_bohr2

        coords_cur_ang = coords_bohr.reshape((-1, 3)) * BOHR_ANGSTROM
        basis, rigid_rank = _internal_projection(coords_cur_ang, masses)
        proj_grad = grad - basis @ (basis.T @ grad)
        max_g = float(np.max(np.abs(proj_grad)))
        if max_g <= gtol_au:
            converged = True
            break

        steps_taken = step_idx + 1
        proj_hess = project_symmetric_matrix(hess, basis)
        evals, evecs = np.linalg.eigh(proj_hess)

        # Isolate the orthogonal internal subspace of dimension 3N - rigid_rank
        overlaps = (
            np.sum((basis.T @ evecs) ** 2, axis=0) if rigid_rank > 0 else np.zeros(evals.size)
        )
        n_int = evals.size - rigid_rank
        int_indices = np.argsort(overlaps)[:n_int]
        int_indices = int_indices[np.argsort(evals[int_indices])]
        int_evals = evals[int_indices]
        int_evecs = evecs[:, int_indices]

        if tracked_mode is None:
            ts_local = 0
        else:
            mode_dots = np.abs(int_evecs.T @ tracked_mode)
            ts_local = int(np.argmax(mode_dots))
        tracked_mode = int_evecs[:, ts_local].copy()

        g_int = int_evecs.T @ proj_grad
        lam_ts = float(int_evals[ts_local])
        g_ts = float(g_int[ts_local])
        # Maximize along TS eigenvector via 1D RFO shift
        mu_p = 0.5 * (lam_ts + math.sqrt(lam_ts * lam_ts + 4.0 * g_ts * g_ts))
        denom_p = lam_ts - mu_p
        step_ts = -g_ts / denom_p if abs(denom_p) > 1.0e-12 else math.copysign(0.05, g_ts)

        step_vec = step_ts * int_evecs[:, ts_local]

        # Minimize along all remaining internal eigenvectors via RFO
        min_mask = np.ones(n_int, dtype=bool)
        min_mask[ts_local] = False
        if np.any(min_mask):
            min_evals = int_evals[min_mask]
            min_evecs = int_evecs[:, min_mask]
            g_min = g_int[min_mask]
            m_dim = min_evals.size
            rfo_mat = np.zeros((m_dim + 1, m_dim + 1), dtype=float)
            np.fill_diagonal(rfo_mat[:m_dim, :m_dim], min_evals)
            rfo_mat[:m_dim, m_dim] = g_min
            rfo_mat[m_dim, :m_dim] = g_min
            rfo_vals, _ = np.linalg.eigh(rfo_mat)
            mu_n = min(0.0, float(rfo_vals[0]) - 1.0e-12)
            for k in range(m_dim):
                denom_k = float(min_evals[k]) - mu_n
                if abs(denom_k) < 1.0e-12:
                    denom_k = 1.0e-12 if denom_k >= 0.0 else -1.0e-12
                step_vec += (-float(g_min[k]) / denom_k) * min_evecs[:, k]

        step_norm = float(np.linalg.norm(step_vec))
        if step_norm > trust_radius_bohr:
            step_vec *= trust_radius_bohr / step_norm
        coords_bohr = coords_bohr + step_vec

    final_coords_ang = coords_bohr.reshape((-1, 3)) * BOHR_ANGSTROM
    final_ev = pes.evaluate(
        sym_tuple,
        final_coords_ang,
        need_gradient=True,
        need_hessian=True,
        charge=charge,
        multiplicity=multiplicity,
    )
    assert final_ev.hessian_kj_mol_angstrom2 is not None
    nm = normal_modes(
        final_ev.hessian_kj_mol_angstrom2,
        masses.tolist(),
        coordinates_angstrom=final_coords_ang,
        hessian_unit="kj/mol/angstrom^2",
        remove_rigid_modes=True,
    )
    saddle_order = int(np.count_nonzero(nm.frequencies_cm1 < -1.0))
    return OptimizationResult(
        coords_angstrom=final_coords_ang,
        evaluation=final_ev,
        normal_mode_result=nm,
        converged=converged,
        steps=steps_taken,
        max_gradient_hartree_bohr=max_g,
        saddle_order=saddle_order,
    )


def build_species_from_pes(
    pes: PotentialEnergySurface,
    name: str,
    symbols: Sequence[str],
    coords_angstrom: npt.ArrayLike,
    *,
    symmetry_number: int = 1,
    electronic_states: Sequence[ElectronicState] = (ElectronicState(),),
    charge: int = 0,
    multiplicity: int = 1,
    reference_energy_kj_mol: float = 0.0,
    mass_convention: str = "iupac",
    masses_amu: Sequence[float] | None = None,
    is_transition_state: bool = False,
    rotor_type: str = "auto",
    frequency_scale: float = 1.0,
    frequency_floor_cm1: float | None = None,
    optimize: bool = False,
    gtol_au: float = 1.0e-6,
    hessian_step_bohr: float | None = None,
    hessian_difference: str | None = None,
    zero_threshold_cm1: float = 10.0,
) -> tuple[Species, PesEvaluation]:
    """Compute Hessian/frequencies from a PES (optionally optimizing) and return a ``Species``."""

    sym_tuple = tuple(symbols)
    coords = _normalize_coords(sym_tuple, coords_angstrom)
    resolved_masses = _resolve_masses_amu(sym_tuple, masses_amu, mass_convention)

    if optimize:
        opt = (
            optimize_saddle(
                pes,
                sym_tuple,
                coords,
                masses_amu=resolved_masses.tolist(),
                mass_convention=mass_convention,
                gtol_au=gtol_au,
                charge=charge,
                multiplicity=multiplicity,
            )
            if is_transition_state
            else optimize_minimum(
                pes,
                sym_tuple,
                coords,
                masses_amu=resolved_masses.tolist(),
                mass_convention=mass_convention,
                gtol_au=gtol_au,
                charge=charge,
                multiplicity=multiplicity,
            )
        )
        coords = opt.coords_angstrom
        ev = opt.evaluation
    else:
        ev = pes.evaluate(
            sym_tuple,
            coords,
            need_gradient=False,
            need_hessian=len(sym_tuple) > 1,
            charge=charge,
            multiplicity=multiplicity,
            hessian_step_bohr=hessian_step_bohr,
            hessian_difference=hessian_difference,
        )

    atoms = tuple(
        Atom(
            symbol=sym_tuple[i],
            x_angstrom=float(coords[i, 0]),
            y_angstrom=float(coords[i, 1]),
            z_angstrom=float(coords[i, 2]),
            mass_amu=float(resolved_masses[i]),
            mass_convention=mass_convention,
        )
        for i in range(len(sym_tuple))
    )

    resolved_rotor = rotor_type.lower()
    if resolved_rotor == "auto":
        moments = principal_moments_amu_a2(atoms)
        resolved_rotor = classify_rotor(moments, len(atoms))

    if len(atoms) == 1:
        freqs_cm1: tuple[float, ...] = ()
    else:
        assert ev.hessian_kj_mol_angstrom2 is not None
        nm = normal_modes(
            ev.hessian_kj_mol_angstrom2,
            resolved_masses.tolist(),
            coordinates_angstrom=coords,
            hessian_unit="kj/mol/angstrom^2",
            remove_rigid_modes=True,
            zero_threshold_cm1=zero_threshold_cm1,
        )
        sorted_freqs = sorted(float(f) for f in nm.frequencies_cm1)
        pos_desc = [f for f in reversed(sorted_freqs) if f > 0.0]
        neg = [f for f in sorted_freqs if f < 0.0]
        if is_transition_state and len(neg) > 1:
            raise ValueError(
                f"transition state {name!r} has {len(neg)} imaginary frequencies; "
                "expected at most 1"
            )
        freqs_cm1 = tuple(pos_desc + neg)

    species = Species(
        name=name,
        atoms=atoms,
        electronic_energy_kj_mol=ev.energy_kj_mol - reference_energy_kj_mol,
        frequencies_cm1=freqs_cm1,
        symmetry_number=symmetry_number,
        electronic_states=tuple(electronic_states),
        rotor_type=resolved_rotor,
        frequency_scale=frequency_scale,
        frequency_floor_cm1=frequency_floor_cm1,
    )
    return species, ev
