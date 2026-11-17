"""Subprocess direct-dynamics providers, restart caching, and QM I/O parsers."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .constants import BOHR_ANGSTROM, HARTREE_KJ_MOL
from .elements import atomic_number
from .pes import PesEvaluation, PotentialEnergySurface

FloatArray = npt.NDArray[np.float64]


class DirectDynamicsError(RuntimeError):
    """Raised when an external electronic-structure invocation fails or returns invalid output."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        command: tuple[str, ...] = (),
        returncode: int | None = None,
        stdout_tail: str = "",
        stderr_tail: str = "",
    ) -> None:
        self.provider = provider
        self.command = command
        self.returncode = returncode
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail
        super().__init__(message)


def _fortran_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _normalize_command(command: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, str):
        parts = tuple(shlex.split(command))
    else:
        parts = tuple(str(part) for part in command)
    if not parts:
        raise ValueError("direct-dynamics command cannot be empty")
    return parts


class RestartCache:
    """Persistent JSON geometry-and-derivative cache (Gaussrate ``GRRESTART``/``RSTTOL`` model)."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        rst_tol_bohr: float = 1.0e-8,
        scale_mass_amu: float | None = None,
    ) -> None:
        if rst_tol_bohr < 0.0:
            raise ValueError("rst_tol_bohr cannot be negative")
        self.path = Path(path) if path is not None else None
        self.rst_tol_bohr = float(rst_tol_bohr)
        self.scale_mass_amu = float(scale_mass_amu) if scale_mass_amu is not None else None
        self.entries: list[dict[str, Any]] = []
        self.hits = 0
        self.misses = 0
        if self.path is not None and self.path.is_file():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        file_scale = payload.get("scale_mass_amu")
        if (
            self.scale_mass_amu is not None
            and file_scale is not None
            and abs(float(file_scale) - self.scale_mass_amu)
            > 1.0e-9 * max(1.0, self.scale_mass_amu)
        ):
            raise ValueError(
                f"restart file {self.path} has scale_mass_amu={file_scale}, "
                f"expected {self.scale_mass_amu}"
            )
        if self.scale_mass_amu is None and file_scale is not None:
            self.scale_mass_amu = float(file_scale)
        raw_entries = payload.get("entries", [])
        if isinstance(raw_entries, list):
            self.entries = list(raw_entries)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": 1,
            "rst_tol_bohr": self.rst_tol_bohr,
            "scale_mass_amu": self.scale_mass_amu,
            "entries": self.entries,
        }
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            os.replace(tmp_name, self.path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def validate_scale_mass(self, scale_mass_amu: float) -> None:
        val = float(scale_mass_amu)
        if self.scale_mass_amu is not None:
            if abs(self.scale_mass_amu - val) > 1.0e-9 * max(1.0, val):
                raise ValueError(
                    f"restart cache scale_mass_amu={self.scale_mass_amu} conflicts with {val}"
                )
        self.scale_mass_amu = val

    def _find_index(
        self,
        symbols: tuple[str, ...],
        coords_bohr: FloatArray,
        charge: int,
        multiplicity: int,
    ) -> int | None:
        sym_list = list(symbols)
        for idx, entry in enumerate(self.entries):
            if entry.get("symbols") != sym_list:
                continue
            if (
                int(entry.get("charge", 0)) != charge
                or int(entry.get("multiplicity", 1)) != multiplicity
            ):
                continue
            cached_coords = np.asarray(entry["coords_bohr"], dtype=float)
            if cached_coords.shape != coords_bohr.shape:
                continue
            if float(np.max(np.abs(cached_coords - coords_bohr))) <= self.rst_tol_bohr:
                return idx
        return None

    def lookup(
        self,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> PesEvaluation | None:
        coords_bohr = coords_angstrom / BOHR_ANGSTROM
        idx = self._find_index(symbols, coords_bohr, charge, multiplicity)
        if idx is None:
            self.misses += 1
            return None
        entry = self.entries[idx]
        if need_gradient and entry.get("gradient_kj_mol_angstrom") is None:
            self.misses += 1
            return None
        if need_hessian and entry.get("hessian_kj_mol_angstrom2") is None:
            self.misses += 1
            return None
        self.hits += 1
        grad = (
            np.asarray(entry["gradient_kj_mol_angstrom"], dtype=float)
            if entry.get("gradient_kj_mol_angstrom") is not None
            else None
        )
        hess = (
            np.asarray(entry["hessian_kj_mol_angstrom2"], dtype=float)
            if entry.get("hessian_kj_mol_angstrom2") is not None
            else None
        )
        return PesEvaluation(
            energy_kj_mol=float(entry["energy_kj_mol"]),
            gradient_kj_mol_angstrom=grad,
            hessian_kj_mol_angstrom2=hess,
            metadata={"cached": True, **dict(entry.get("metadata", {}))},
        )

    def store(
        self,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        evaluation: PesEvaluation,
        *,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> None:
        coords_bohr = coords_angstrom / BOHR_ANGSTROM
        idx = self._find_index(symbols, coords_bohr, charge, multiplicity)
        record: dict[str, Any] = {
            "symbols": list(symbols),
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "coords_bohr": coords_bohr.tolist(),
            "energy_kj_mol": float(evaluation.energy_kj_mol),
            "gradient_kj_mol_angstrom": (
                evaluation.gradient_kj_mol_angstrom.tolist()
                if evaluation.gradient_kj_mol_angstrom is not None
                else None
            ),
            "hessian_kj_mol_angstrom2": (
                evaluation.hessian_kj_mol_angstrom2.tolist()
                if evaluation.hessian_kj_mol_angstrom2 is not None
                else None
            ),
            "metadata": {
                k: v
                for k, v in evaluation.metadata.items()
                if k != "cached" and isinstance(v, (str, int, float, bool, type(None)))
            },
        }
        if idx is not None:
            prev = self.entries[idx]
            if (
                record["gradient_kj_mol_angstrom"] is None
                and prev.get("gradient_kj_mol_angstrom") is not None
            ):
                record["gradient_kj_mol_angstrom"] = prev["gradient_kj_mol_angstrom"]
            if (
                record["hessian_kj_mol_angstrom2"] is None
                and prev.get("hessian_kj_mol_angstrom2") is not None
            ):
                record["hessian_kj_mol_angstrom2"] = prev["hessian_kj_mol_angstrom2"]
            self.entries[idx] = record
        else:
            self.entries.append(record)
        self.save()


class DirectDynamicsProvider(PotentialEnergySurface):
    """Base class for subprocess electronic-structure interfaces with scratch isolation & cache."""

    provider_name: str = "direct_dynamics"

    def __init__(
        self,
        *,
        command: str | Sequence[str],
        restart_file: str | Path | None = None,
        rst_tol_bohr: float = 1.0e-8,
        scale_mass_amu: float | None = None,
        timeout_s: float = 300.0,
        env: Mapping[str, str] | None = None,
        step_bohr: float = 1.0e-4,
        hessian_difference: str = "central",
        scratch_dir: str | Path | None = None,
    ) -> None:
        super().__init__(step_bohr=step_bohr, hessian_difference=hessian_difference)
        self.command = _normalize_command(command)
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = float(timeout_s)
        self.env = dict(env) if env is not None else None
        self.scratch_dir = Path(scratch_dir) if scratch_dir is not None else None
        self.cache = RestartCache(
            restart_file, rst_tol_bohr=rst_tol_bohr, scale_mass_amu=scale_mass_amu
        )

    def _run_subprocess(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path,
        extra_env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        merged_env = os.environ.copy()
        if self.env:
            merged_env.update(self.env)
        if extra_env:
            merged_env.update(extra_env)
        try:
            proc = subprocess.run(
                list(cmd),
                cwd=str(cwd),
                env=merged_env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise DirectDynamicsError(
                f"{self.provider_name} timed out after {self.timeout_s}s: {' '.join(cmd)}",
                provider=self.provider_name,
                command=tuple(cmd),
            ) from exc
        except OSError as exc:
            raise DirectDynamicsError(
                f"{self.provider_name} failed to execute {cmd[0]!r}: {exc}",
                provider=self.provider_name,
                command=tuple(cmd),
            ) from exc

        if proc.returncode != 0:
            stderr_tail = "\n".join(proc.stderr.splitlines()[-15:])
            stdout_tail = "\n".join(proc.stdout.splitlines()[-15:])
            raise DirectDynamicsError(
                f"{self.provider_name} exited with status {proc.returncode}: "
                f"{stderr_tail or stdout_tail}",
                provider=self.provider_name,
                command=tuple(cmd),
                returncode=proc.returncode,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
            )
        return proc

    def _execute_in_workdir(
        self,
        workdir: Path,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int,
        multiplicity: int,
    ) -> PesEvaluation:
        raise NotImplementedError

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
        cached = self.cache.lookup(
            symbols,
            coords_angstrom,
            need_gradient=need_gradient,
            need_hessian=need_hessian,
            charge=charge,
            multiplicity=multiplicity,
        )
        if cached is not None:
            return cached

        parent = str(self.scratch_dir) if self.scratch_dir is not None else None
        if self.scratch_dir is not None:
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pyrate_dd_", dir=parent) as tmp_dir:
            evaluation = self._execute_in_workdir(
                Path(tmp_dir),
                symbols,
                coords_angstrom,
                need_gradient=need_gradient,
                need_hessian=need_hessian,
                charge=charge,
                multiplicity=multiplicity,
            )
        self.cache.store(
            symbols,
            coords_angstrom,
            evaluation,
            charge=charge,
            multiplicity=multiplicity,
        )
        return evaluation


class ExternalCommandPes(DirectDynamicsProvider):
    """Generic JSON subprocess direct-dynamics adapter.

    Writes ``pyrate_request.json`` in an isolated scratch directory and invokes ``command``.
    Substitutes ``{input}`` / ``{output}`` tokens if present in ``command`` (otherwise
    appends both paths as trailing positional arguments), and sets ``PYRATE_DD_INPUT`` /
    ``PYRATE_DD_OUTPUT`` environment variables.
    """

    provider_name = "external_command"

    def _execute_in_workdir(
        self,
        workdir: Path,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int,
        multiplicity: int,
    ) -> PesEvaluation:
        req_path = workdir / "pyrate_request.json"
        resp_path = workdir / "pyrate_response.json"
        request_payload = {
            "symbols": list(symbols),
            "coords_angstrom": coords_angstrom.tolist(),
            "coords_bohr": (coords_angstrom / BOHR_ANGSTROM).tolist(),
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "need_gradient": bool(need_gradient),
            "need_hessian": bool(need_hessian),
        }
        req_path.write_text(json.dumps(request_payload, indent=2) + "\n", encoding="utf-8")

        has_placeholders = any("{input}" in tok or "{output}" in tok for tok in self.command)
        if has_placeholders:
            cmd = [
                tok.replace("{input}", str(req_path)).replace("{output}", str(resp_path))
                for tok in self.command
            ]
        else:
            cmd = [*self.command, str(req_path), str(resp_path)]

        proc = self._run_subprocess(
            cmd,
            cwd=workdir,
            extra_env={"PYRATE_DD_INPUT": str(req_path), "PYRATE_DD_OUTPUT": str(resp_path)},
        )

        if resp_path.is_file():
            raw_text = resp_path.read_text(encoding="utf-8")
        elif proc.stdout.strip().startswith("{"):
            raw_text = proc.stdout
        else:
            raise DirectDynamicsError(
                f"external command did not produce {resp_path.name} or JSON stdout",
                provider=self.provider_name,
                command=tuple(cmd),
                stdout_tail="\n".join(proc.stdout.splitlines()[-10:]),
            )

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise DirectDynamicsError(
                f"external command produced invalid JSON: {exc}",
                provider=self.provider_name,
                command=tuple(cmd),
            ) from exc

        n_atoms = len(symbols)
        dim = 3 * n_atoms
        if "energy_kj_mol" in payload:
            energy_kj = float(payload["energy_kj_mol"])
        elif "energy_hartree" in payload:
            energy_kj = float(payload["energy_hartree"]) * HARTREE_KJ_MOL
        else:
            raise DirectDynamicsError(
                "external response missing 'energy_hartree' or 'energy_kj_mol'",
                provider=self.provider_name,
                command=tuple(cmd),
            )

        grad_kj: FloatArray | None = None
        if payload.get("gradient_kj_mol_angstrom") is not None:
            grad_kj = np.asarray(
                payload["gradient_kj_mol_angstrom"], dtype=float
            ).reshape((n_atoms, 3))
        elif payload.get("gradient_hartree_bohr") is not None:
            grad_au = np.asarray(
                payload["gradient_hartree_bohr"], dtype=float
            ).reshape((n_atoms, 3))
            grad_kj = grad_au * (HARTREE_KJ_MOL / BOHR_ANGSTROM)

        hess_kj: FloatArray | None = None
        if payload.get("hessian_kj_mol_angstrom2") is not None:
            hess_kj = _unpack_hessian_payload(payload["hessian_kj_mol_angstrom2"], dim)
        elif payload.get("hessian_hartree_bohr2") is not None:
            hess_au = _unpack_hessian_payload(payload["hessian_hartree_bohr2"], dim)
            hess_kj = hess_au * (HARTREE_KJ_MOL / (BOHR_ANGSTROM * BOHR_ANGSTROM))

        raw_meta = payload.get("metadata")
        meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
        return PesEvaluation(
            energy_kj_mol=energy_kj,
            gradient_kj_mol_angstrom=grad_kj,
            hessian_kj_mol_angstrom2=hess_kj,
            metadata={"provider": self.provider_name, **meta},
        )


def _unpack_hessian_payload(raw: Any, dim: int) -> FloatArray:
    arr = np.asarray(raw, dtype=float)
    if arr.shape == (dim, dim):
        return np.asarray(0.5 * (arr + arr.T), dtype=float)
    flat = arr.reshape(-1)
    if flat.size == dim * dim:
        mat = flat.reshape((dim, dim))
        return np.asarray(0.5 * (mat + mat.T), dtype=float)
    packed_len = dim * (dim + 1) // 2
    if flat.size == packed_len:
        mat = np.zeros((dim, dim), dtype=float)
        idx = 0
        for row in range(dim):
            for col in range(row + 1):
                mat[row, col] = mat[col, row] = float(flat[idx])
                idx += 1
        return mat
    raise ValueError(f"unexpected Hessian size {flat.size} for dimension {dim}")


def _read_fchk_block(lines: Sequence[str], start_idx: int, count: int) -> tuple[list[float], int]:
    values: list[float] = []
    idx = start_idx
    while idx < len(lines) and len(values) < count:
        for tok in lines[idx].split():
            values.append(_fortran_float(tok))
        idx += 1
    if len(values) < count:
        raise ValueError(f"truncated Gaussian .fchk block: expected {count}, got {len(values)}")
    return values[:count], idx


def parse_gaussian_fchk(text: str, *, n_atoms: int | None = None) -> PesEvaluation:
    """Parse Gaussian formatted checkpoint (``.fchk``) energy, gradient, and force constants."""

    lines = text.splitlines()
    parsed_atoms = n_atoms
    energy_au: float | None = None
    grad_au: FloatArray | None = None
    hess_au: FloatArray | None = None

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("Number of atoms"):
            tokens = line[40:].split() or line.split()
            parsed_atoms = int(tokens[-1])
            idx += 1
        elif line.startswith("Total Energy"):
            tokens = line[40:].split() or line.split()
            energy_au = _fortran_float(tokens[-1])
            idx += 1
        elif line.startswith("Cartesian Gradient"):
            m_count = re.search(r"N=\s*(\d+)", line)
            count = int(m_count.group(1)) if m_count else (3 * parsed_atoms if parsed_atoms else 0)
            vals, idx = _read_fchk_block(lines, idx + 1, count)
            grad_au = np.asarray(vals, dtype=float).reshape((-1, 3))
        elif line.startswith("Cartesian Force Constants"):
            m_count = re.search(r"N=\s*(\d+)", line)
            if not m_count:
                raise ValueError("Gaussian .fchk Cartesian Force Constants missing N= count")
            count = int(m_count.group(1))
            vals, idx = _read_fchk_block(lines, idx + 1, count)
            dim = (
                parsed_atoms * 3
                if parsed_atoms is not None
                else int(round((-1 + math.sqrt(1 + 8 * count)) / 2))
            )
            if dim * (dim + 1) // 2 != count:
                raise ValueError(
                    f"inconsistent packed force constants length {count} for dim {dim}"
                )
            hess_au = np.zeros((dim, dim), dtype=float)
            k = 0
            for row in range(dim):
                for col in range(row + 1):
                    hess_au[row, col] = hess_au[col, row] = vals[k]
                    k += 1
        else:
            idx += 1

    if energy_au is None:
        raise ValueError("Gaussian .fchk missing 'Total Energy'")
    if parsed_atoms is not None and grad_au is not None and grad_au.shape != (parsed_atoms, 3):
        raise ValueError("Gaussian .fchk gradient atom count mismatch")

    return PesEvaluation(
        energy_kj_mol=energy_au * HARTREE_KJ_MOL,
        gradient_kj_mol_angstrom=(
            np.asarray(grad_au * (HARTREE_KJ_MOL / BOHR_ANGSTROM), dtype=float)
            if grad_au is not None
            else None
        ),
        hessian_kj_mol_angstrom2=(
            np.asarray(hess_au * (HARTREE_KJ_MOL / (BOHR_ANGSTROM * BOHR_ANGSTROM)), dtype=float)
            if hess_au is not None
            else None
        ),
        metadata={"provider": "gaussian", "source": "fchk"},
    )


def parse_gaussian_log(text: str) -> PesEvaluation:
    """Validate normal termination and extract total energy from a Gaussian ``.log`` file."""

    if "Error termination" in text:
        tail = "\n".join(text.splitlines()[-10:])
        raise DirectDynamicsError(
            f"Gaussian reported error termination:\n{tail}",
            provider="gaussian",
        )
    if "Normal termination" not in text:
        raise DirectDynamicsError(
            "Gaussian .log missing 'Normal termination'",
            provider="gaussian",
        )
    matches = re.findall(r"SCF Done:\s+E\([^)]+\)\s*=\s*([-+]?\d+\.\d+(?:[EDed][-+]?\d+)?)", text)
    if not matches:
        compact = text.replace("\n", "").replace(" ", "")
        matches = re.findall(r"\\HF=([-+]?\d+\.\d+(?:[EDed][-+]?\d+)?)", compact)
    if not matches:
        raise DirectDynamicsError(
            "Gaussian .log missing SCF/total energy and no .fchk file was generated",
            provider="gaussian",
        )
    energy_au = _fortran_float(matches[-1])
    return PesEvaluation(
        energy_kj_mol=energy_au * HARTREE_KJ_MOL,
        metadata={"provider": "gaussian", "source": "log"},
    )


class GaussianProvider(DirectDynamicsProvider):
    """Subprocess Gaussian (``g16``/``g09``) direct-dynamics provider."""

    provider_name = "gaussian"

    def __init__(
        self,
        *,
        command: str | Sequence[str] = ("g16",),
        method: str = "HF",
        basis: str = "STO-3G",
        mem: str | None = None,
        nprocshared: int | None = None,
        extra_keywords: Sequence[str] = (),
        extra_sections: Sequence[str] = (),
        fchk_command: str | Sequence[str] | None = ("formchk",),
        restart_file: str | Path | None = None,
        rst_tol_bohr: float = 1.0e-8,
        scale_mass_amu: float | None = None,
        timeout_s: float = 600.0,
        env: Mapping[str, str] | None = None,
        scratch_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            command=command,
            restart_file=restart_file,
            rst_tol_bohr=rst_tol_bohr,
            scale_mass_amu=scale_mass_amu,
            timeout_s=timeout_s,
            env=env,
            scratch_dir=scratch_dir,
        )
        self.method = method
        self.basis = basis
        self.mem = mem
        self.nprocshared = nprocshared
        self.extra_keywords = tuple(extra_keywords)
        self.extra_sections = tuple(extra_sections)
        self.fchk_command = _normalize_command(fchk_command) if fchk_command else None

    def build_gjf(
        self,
        symbols: Sequence[str],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int = 0,
        multiplicity: int = 1,
        checkpoint_name: str = "pyrate_dd.chk",
    ) -> str:
        lines: list[str] = [f"%chk={checkpoint_name}"]
        if self.mem:
            lines.append(f"%mem={self.mem}")
        if self.nprocshared is not None:
            lines.append(f"%nprocshared={int(self.nprocshared)}")

        route_tokens = [f"#p {self.method}/{self.basis}"]
        if need_hessian:
            route_tokens.append("Freq=NoRaman")
        elif need_gradient:
            route_tokens.append("Force")
        route_tokens.extend(["Units=AU", "NoSymm", "FChk"])
        route_tokens.extend(self.extra_keywords)
        lines.append(" ".join(route_tokens))
        lines.append("")
        lines.append("PyRate direct-dynamics evaluation")
        lines.append("")
        lines.append(f"{int(charge)} {int(multiplicity)}")

        coords_bohr = coords_angstrom / BOHR_ANGSTROM
        for idx, sym in enumerate(symbols):
            x, y, z = coords_bohr[idx]
            lines.append(f" {sym:<4s} {x:18.10f} {y:18.10f} {z:18.10f}")
        lines.append("")
        for section in self.extra_sections:
            stripped = section.strip("\n")
            if stripped:
                lines.append(stripped)
                lines.append("")
        return "\n".join(lines) + "\n"

    def _execute_in_workdir(
        self,
        workdir: Path,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int,
        multiplicity: int,
    ) -> PesEvaluation:
        gjf_path = workdir / "gaussian.gjf"
        chk_name = "pyrate_dd.chk"
        chk_path = workdir / chk_name
        gjf_text = self.build_gjf(
            symbols,
            coords_angstrom,
            need_gradient=need_gradient,
            need_hessian=need_hessian,
            charge=charge,
            multiplicity=multiplicity,
            checkpoint_name=chk_name,
        )
        gjf_path.write_text(gjf_text, encoding="utf-8")

        cmd = [*self.command, str(gjf_path)]
        proc = self._run_subprocess(cmd, cwd=workdir)

        log_candidates = [workdir / "gaussian.log", workdir / "gaussian.gjf.log"]
        log_text = ""
        for cand in log_candidates:
            if cand.is_file():
                log_text = cand.read_text(encoding="utf-8", errors="replace")
                break
        if not log_text and proc.stdout.strip():
            log_text = proc.stdout

        if log_text:
            if "Error termination" in log_text:
                raise DirectDynamicsError(
                    "Gaussian execution terminated with an error",
                    provider=self.provider_name,
                    command=tuple(cmd),
                    stdout_tail="\n".join(log_text.splitlines()[-15:]),
                )

        fchk_candidates = [
            workdir / "Test.FChk",
            workdir / "gaussian.fchk",
            workdir / "pyrate_dd.fchk",
        ]
        fchk_path = next((p for p in fchk_candidates if p.is_file()), None)
        if fchk_path is None and chk_path.is_file() and self.fchk_command is not None:
            target_fchk = workdir / "pyrate_dd.fchk"
            self._run_subprocess([*self.fchk_command, str(chk_path), str(target_fchk)], cwd=workdir)
            if target_fchk.is_file():
                fchk_path = target_fchk

        if fchk_path is not None:
            return parse_gaussian_fchk(
                fchk_path.read_text(encoding="utf-8", errors="replace"),
                n_atoms=len(symbols),
            )
        if log_text:
            return parse_gaussian_log(log_text)
        raise DirectDynamicsError(
            "Gaussian run produced neither .fchk nor .log output",
            provider=self.provider_name,
            command=tuple(cmd),
        )


def _extract_fixed_or_free_floats(tail: str) -> list[float]:
    # GAMESS punch $HESS formats each row chunk as 5 columns of width 15 (E15.8) or space-separated.
    tokens = re.findall(r"[-+]?\d+\.\d+(?:[EDed][-+]?\d+)?", tail)
    return [_fortran_float(tok) for tok in tokens]


def parse_gamess_output(
    dat_text: str,
    *,
    n_atoms: int,
    log_text: str | None = None,
) -> PesEvaluation:
    """Parse GAMESS ``.dat`` (PUNCH) ``$GRAD`` / ``$HESS`` blocks and ``.log`` termination."""

    if log_text:
        if "TERMINATED ABNORMALLY" in log_text or (
            "ABNORMALLY" in log_text and "TERMINATED NORMALLY" not in log_text
        ):
            raise DirectDynamicsError(
                "GAMESS execution terminated abnormally",
                provider="gamess",
                stdout_tail="\n".join(log_text.splitlines()[-15:]),
            )

    dim = 3 * n_atoms
    energy_au: float | None = None
    grad_au: FloatArray | None = None
    hess_au: FloatArray | None = None

    grad_match = re.search(r"\$GRAD\b(.*?)\$END", dat_text, flags=re.IGNORECASE | re.DOTALL)
    if grad_match:
        grad_lines = [ln.strip() for ln in grad_match.group(1).splitlines() if ln.strip()]
        grad_rows: list[list[float]] = []
        for ln in grad_lines:
            if ln.upper().startswith("E=") or re.match(r"^E\s*=", ln, flags=re.IGNORECASE):
                nums = _extract_fixed_or_free_floats(ln)
                if nums:
                    energy_au = nums[0]
                continue
            nums = _extract_fixed_or_free_floats(ln)
            if len(nums) >= 3:
                grad_rows.append(nums[-3:])
        if len(grad_rows) >= n_atoms:
            grad_au = np.asarray(grad_rows[:n_atoms], dtype=float)

    hess_match = re.search(r"\$HESS\b(.*?)\$END", dat_text, flags=re.IGNORECASE | re.DOTALL)
    if hess_match:
        hess_lines = [ln for ln in hess_match.group(1).splitlines() if ln.strip()]
        hess_mat = np.zeros((dim, dim), dtype=float)
        data_lines: list[str] = []
        for ln in hess_lines:
            if "ENERGY IS" in ln.upper():
                nums = _extract_fixed_or_free_floats(ln)
                if nums and energy_au is None:
                    energy_au = nums[0]
                continue
            data_lines.append(ln)

        col_offset = 0
        row_in_block = 0
        for ln in data_lines:
            # Fixed GAMESS format puts row (I2) + blk (I3) in first 5 chars, values in [5:].
            if len(ln) > 5 and ln[:5].strip() and all(ch.isdigit() or ch == " " for ch in ln[:5]):
                vals = _extract_fixed_or_free_floats(ln[5:])
            else:
                toks = ln.split()
                if len(toks) >= 3 and toks[0].isdigit() and toks[1].isdigit():
                    vals = [_fortran_float(t) for t in toks[2:]]
                else:
                    vals = _extract_fixed_or_free_floats(ln)
            if not vals:
                continue
            for idx_val, val in enumerate(vals):
                col = col_offset + idx_val
                if row_in_block < dim and col < dim:
                    hess_mat[row_in_block, col] = val
            row_in_block += 1
            if row_in_block >= dim:
                row_in_block = 0
                col_offset += len(vals)
        hess_au = np.asarray(0.5 * (hess_mat + hess_mat.T), dtype=float)

    if energy_au is None and log_text:
        m_e = re.findall(
            r"(?:FINAL\s+[\w\-]+\s+ENERGY\s+IS|TOTAL\s+ENERGY\s*=)\s*([-+]?\d+\.\d+(?:[EDed][-+]?\d+)?)",
            log_text,
            flags=re.IGNORECASE,
        )
        if m_e:
            energy_au = _fortran_float(m_e[-1])

    if energy_au is None:
        raise DirectDynamicsError(
            "GAMESS output did not contain an energy value in .dat or .log",
            provider="gamess",
        )

    return PesEvaluation(
        energy_kj_mol=energy_au * HARTREE_KJ_MOL,
        gradient_kj_mol_angstrom=(
            np.asarray(grad_au * (HARTREE_KJ_MOL / BOHR_ANGSTROM), dtype=float)
            if grad_au is not None
            else None
        ),
        hessian_kj_mol_angstrom2=(
            np.asarray(hess_au * (HARTREE_KJ_MOL / (BOHR_ANGSTROM * BOHR_ANGSTROM)), dtype=float)
            if hess_au is not None
            else None
        ),
        metadata={"provider": "gamess"},
    )


class GamessProvider(DirectDynamicsProvider):
    """Subprocess GAMESS direct-dynamics provider (``.inp`` -> ``.dat`` / ``.log``)."""

    provider_name = "gamess"

    def __init__(
        self,
        *,
        command: str | Sequence[str] = ("rungms",),
        scftyp: str = "RHF",
        contrl: Mapping[str, Any] | None = None,
        basis: Mapping[str, Any] | str | None = None,
        extra_groups: Sequence[str] = (),
        restart_file: str | Path | None = None,
        rst_tol_bohr: float = 1.0e-8,
        scale_mass_amu: float | None = None,
        timeout_s: float = 600.0,
        env: Mapping[str, str] | None = None,
        scratch_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            command=command,
            restart_file=restart_file,
            rst_tol_bohr=rst_tol_bohr,
            scale_mass_amu=scale_mass_amu,
            timeout_s=timeout_s,
            env=env,
            scratch_dir=scratch_dir,
        )
        self.scftyp = scftyp
        self.contrl = dict(contrl) if contrl else {}
        self.basis = basis if basis is not None else {"GBASIS": "STO", "NGAUSS": 3}
        self.extra_groups = tuple(extra_groups)

    def build_inp(
        self,
        symbols: Sequence[str],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> str:
        runtyp = "HESSIAN" if need_hessian else ("GRADIENT" if need_gradient else "ENERGY")
        contrl_items = [
            f"SCFTYP={self.scftyp}",
            f"RUNTYP={runtyp}",
            "UNITS=BOHR",
            f"ICHARG={int(charge)}",
            f"MULT={int(multiplicity)}",
        ]
        for k, v in self.contrl.items():
            key_u = str(k).upper()
            if key_u not in {"SCFTYP", "RUNTYP", "UNITS", "ICHARG", "MULT"}:
                contrl_items.append(f"{key_u}={v}")

        lines = [" $CONTRL " + " ".join(contrl_items) + " $END"]
        if isinstance(self.basis, str):
            basis_line = self.basis.strip()
            if not basis_line.upper().startswith("$BASIS"):
                basis_line = f" $BASIS {basis_line} $END"
            else:
                basis_line = f" {basis_line}"
            lines.append(basis_line)
        else:
            b_items = [f"{str(k).upper()}={v}" for k, v in self.basis.items()]
            lines.append(" $BASIS " + " ".join(b_items) + " $END")
        for grp in self.extra_groups:
            stripped = grp.strip()
            if stripped:
                lines.append(f" {stripped}")

        lines.extend([" $DATA", "PyRate direct-dynamics evaluation", "C1"])
        coords_bohr = coords_angstrom / BOHR_ANGSTROM
        for idx, sym in enumerate(symbols):
            z = atomic_number(sym)
            x, y, zc = coords_bohr[idx]
            lines.append(f" {sym:<4s} {float(z):5.1f} {x:18.10f} {y:18.10f} {zc:18.10f}")
        lines.append(" $END")
        return "\n".join(lines) + "\n"

    def _execute_in_workdir(
        self,
        workdir: Path,
        symbols: tuple[str, ...],
        coords_angstrom: FloatArray,
        *,
        need_gradient: bool,
        need_hessian: bool,
        charge: int,
        multiplicity: int,
    ) -> PesEvaluation:
        inp_path = workdir / "gamess.inp"
        inp_text = self.build_inp(
            symbols,
            coords_angstrom,
            need_gradient=need_gradient,
            need_hessian=need_hessian,
            charge=charge,
            multiplicity=multiplicity,
        )
        inp_path.write_text(inp_text, encoding="utf-8")
        cmd = [*self.command, str(inp_path)]
        proc = self._run_subprocess(cmd, cwd=workdir)

        dat_path = workdir / "gamess.dat"
        dat_text = (
            dat_path.read_text(encoding="utf-8", errors="replace") if dat_path.is_file() else ""
        )
        log_path = workdir / "gamess.log"
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.is_file()
            else proc.stdout
        )
        if not dat_text and not log_text:
            raise DirectDynamicsError(
                "GAMESS run produced neither .dat nor .log output",
                provider=self.provider_name,
                command=tuple(cmd),
            )
        return parse_gamess_output(dat_text, n_atoms=len(symbols), log_text=log_text)


def _resolve_optional_path(value: Any, *, base_dir: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute() and base_dir is not None:
        return Path(base_dir) / path
    return path


def direct_dynamics_from_mapping(
    table: Mapping[str, Any],
    *,
    base_dir: str | None = None,
) -> DirectDynamicsProvider:
    """Instantiate ``ExternalCommandPes``, ``GaussianProvider``, or ``GamessProvider`` from TOML."""

    pes_type = str(table.get("type", "")).strip().lower()
    restart_file = _resolve_optional_path(table.get("restart_file"), base_dir=base_dir)
    rst_tol_bohr = float(table.get("rst_tol_bohr", 1.0e-8))
    scale_mass_amu = (
        float(table["scale_mass_amu"]) if table.get("scale_mass_amu") is not None else None
    )
    timeout_s = float(table.get("timeout_s", 300.0))
    step_bohr = float(table.get("step_bohr", 1.0e-4))
    hessian_difference = str(table.get("hessian_difference", "central"))
    env = dict(table["env"]) if isinstance(table.get("env"), Mapping) else None

    if pes_type in {"command", "external"}:
        if "command" not in table:
            raise ValueError("[potential] with type='command' requires 'command'")
        return ExternalCommandPes(
            command=table["command"],
            restart_file=restart_file,
            rst_tol_bohr=rst_tol_bohr,
            scale_mass_amu=scale_mass_amu,
            timeout_s=timeout_s,
            env=env,
            step_bohr=step_bohr,
            hessian_difference=hessian_difference,
        )
    if pes_type == "gaussian":
        return GaussianProvider(
            command=table.get("command", ("g16",)),
            method=str(table.get("method", "HF")),
            basis=str(table.get("basis", "STO-3G")),
            mem=str(table["mem"]) if table.get("mem") is not None else None,
            nprocshared=int(table["nprocshared"]) if table.get("nprocshared") is not None else None,
            extra_keywords=tuple(str(k) for k in table.get("extra_keywords", ())),
            extra_sections=tuple(str(s) for s in table.get("extra_sections", ())),
            fchk_command=table.get("fchk_command", ("formchk",)),
            restart_file=restart_file,
            rst_tol_bohr=rst_tol_bohr,
            scale_mass_amu=scale_mass_amu,
            timeout_s=timeout_s,
            env=env,
        )
    if pes_type == "gamess":
        return GamessProvider(
            command=table.get("command", ("rungms",)),
            scftyp=str(table.get("scftyp", "RHF")),
            contrl=table.get("contrl"),
            basis=table.get("basis"),
            extra_groups=tuple(str(g) for g in table.get("extra_groups", ())),
            restart_file=restart_file,
            rst_tol_bohr=rst_tol_bohr,
            scale_mass_amu=scale_mass_amu,
            timeout_s=timeout_s,
            env=env,
        )
    raise ValueError(f"unsupported direct-dynamics provider type {pes_type!r}")
