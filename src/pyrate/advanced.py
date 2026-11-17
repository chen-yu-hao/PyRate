"""Advanced reaction-rate helpers.

This module contains small, dependency-light implementations of the advanced
models that are useful when a complete Polyrate input file is not available:

* Wilson ``GF`` generalized normal modes in curvilinear coordinates;
* interpolated dual-level (IOC/ISPE/IVTST) reaction-path corrections;
* deterministic weighted VRC-VTST faceted dividing surfaces;
* RRKM/SS-QRRK and Lindemann/Troe pressure falloff helpers; and
* explicit solvent and gas--surface free-energy corrections.

The functions deliberately operate on arrays and the public :mod:`pyrate.models`
objects.  They do not change the existing CTST/CVT implementation and can be
used independently by callers, input adapters, or future native kernels.
Energies exposed by this module are in kJ mol⁻¹ unless a function explicitly
states otherwise; rates are in s⁻¹ for unimolecular quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .constants import (
    AMU_KG,
    ANGSTROM_M,
    BOLTZMANN_J_K,
    GAS_CONSTANT_J_MOL_K,
    LIGHT_SPEED_M_S,
    PLANCK_J_S,
    WAVENUMBER_TO_J,
)
from .models import PathPoint, ReactionPath, Species


# ---------------------------------------------------------------------------
# Curvilinear-coordinate generalized normal modes


def _as_matrix(values: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _mass_vector(masses: Sequence[float] | np.ndarray, n_cart: int) -> np.ndarray:
    values = np.asarray(masses, dtype=float).reshape(-1)
    if values.size * 3 == n_cart:
        values = np.repeat(values, 3)
    if values.size != n_cart:
        raise ValueError("masses must contain one value per Cartesian component or atom")
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("masses must be finite and positive")
    return values


def _symmetric_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenpairs of a matrix after symmetrizing round-off noise."""

    return np.linalg.eigh((matrix + matrix.T) * 0.5)


def curvilinear_frequencies(
    force_constants: Sequence[Sequence[float]] | np.ndarray,
    wilson_b: Sequence[Sequence[float]] | np.ndarray,
    masses_amu: Sequence[float] | np.ndarray,
    *,
    hessian_is_cartesian: bool | None = None,
    frequency_scale_cm1: float | None = None,
    remove_reaction_coordinate: bool = False,
    imag_tol: float = 1.0e-12,
) -> tuple[float, ...]:
    """Calculate generalized vibrational frequencies with Wilson's ``GF`` method.

    Parameters
    ----------
    force_constants:
        Force-constant matrix in internal coordinates (``m × m``), or a
        Cartesian Hessian (``3N × 3N``) when ``hessian_is_cartesian=True``.
        Internal force constants may be supplied in any self-consistent energy
        unit; use ``frequency_scale_cm1`` to map ``sqrt(lambda)`` to cm⁻¹.
    wilson_b:
        Wilson ``B = dq/dx`` matrix (``m × 3N``).  Rows correspond to
        non-redundant or redundant stretches, bends, and torsions.
    masses_amu:
        One mass per atom or Cartesian component.  For an atom list the values
        are repeated three times internally.
    hessian_is_cartesian:
        If omitted it is inferred from the matrix dimensions.  A Cartesian
        Hessian is projected into the internal-coordinate tangent space using a
        mass-weighted least-squares Jacobian.
    frequency_scale_cm1:
        Conversion from ``sqrt(lambda)`` to wavenumber.  For a Hessian in SI
        force constants this is normally ``1/(2πc)`` with ``c`` in cm s⁻¹;
        for pre-scaled Wilson matrices the default ``1`` is convenient.
    remove_reaction_coordinate:
        Drop the lowest-magnitude eigenvalue, useful for a path-point Hessian
        whose tangential mode has been retained in the projected matrix.

    Negative eigenvalues are represented as negative frequencies.  Frequencies
    are returned in ascending order and zero modes below ``imag_tol`` are
    omitted.  The implementation covers ``curv1``/``curv2`` and redundant
    ``curv3`` inputs; coordinate selection and the Wilson ``B`` matrix remain
    the caller's responsibility.
    """

    b = np.asarray(wilson_b, dtype=float)
    if b.ndim != 2 or b.shape[0] == 0 or b.shape[1] == 0:
        raise ValueError("wilson_b must be a non-empty m x 3N matrix")
    if not np.all(np.isfinite(b)):
        raise ValueError("wilson_b contains non-finite values")
    f = np.asarray(force_constants, dtype=float)
    if f.ndim != 2 or f.shape[0] != f.shape[1]:
        raise ValueError("force_constants must be a square matrix")
    m, n_cart = b.shape
    masses = _mass_vector(masses_amu, n_cart)
    if hessian_is_cartesian is None:
        hessian_is_cartesian = f.shape == (n_cart, n_cart)
    if hessian_is_cartesian:
        if f.shape != (n_cart, n_cart):
            raise ValueError("Cartesian Hessian must have shape (3N, 3N)")
        # J maps internal displacements into the mass-weighted Cartesian
        # tangent space.  The symmetric pseudo-inverse remains well behaved
        # for redundant B matrices (curv3).
        inv_mass = np.diag(1.0 / masses)
        g = b @ inv_mass @ b.T
        g_pinv = np.linalg.pinv((g + g.T) * 0.5, rcond=1e-12)
        j = inv_mass @ b.T @ g_pinv
        f_internal = j.T @ f @ j
    else:
        if f.shape != (m, m):
            raise ValueError("internal force constants must have shape (m, m)")
        f_internal = f

    # G is the Wilson inverse-mass metric.  Solve via G^{-1/2} F G^{-1/2}
    # instead of forming a generally non-symmetric G F product.
    g = b @ np.diag(1.0 / masses) @ b.T
    g = (g + g.T) * 0.5
    g_values, g_vectors = _symmetric_eigh(g)
    keep = g_values > 1.0e-14 * max(1.0, float(np.max(np.abs(g_values))))
    if not np.any(keep):
        return ()
    g_inv_sqrt = (g_vectors[:, keep] / np.sqrt(g_values[keep])) @ g_vectors[:, keep].T
    projected = g_inv_sqrt @ ((f_internal + f_internal.T) * 0.5) @ g_inv_sqrt
    eigenvalues = _symmetric_eigh(projected)[0]
    if remove_reaction_coordinate and eigenvalues.size:
        eigenvalues = np.delete(eigenvalues, int(np.argmin(np.abs(eigenvalues))))
    conversion = 1.0 if frequency_scale_cm1 is None else float(frequency_scale_cm1)
    if conversion <= 0.0 or not math.isfinite(conversion):
        raise ValueError("frequency_scale_cm1 must be finite and positive")
    frequencies = conversion * np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    frequencies[np.abs(frequencies) < imag_tol] = 0.0
    return tuple(float(value) for value in np.sort(frequencies) if abs(value) >= imag_tol)


# Common aliases used by input adapters and downstream clients.
wilson_gf_frequencies = curvilinear_frequencies
curvilinear_coordinate_frequencies = curvilinear_frequencies
intrinsic_coordinate_frequencies = curvilinear_frequencies


def path_curvilinear_frequencies(
    path: ReactionPath,
    *,
    frequency_scale_cm1: float | None = None,
    remove_reaction_coordinate: bool = False,
) -> dict[float, tuple[float, ...]]:
    """Evaluate curvilinear frequencies stored in path-point metadata.

    A path point can carry ``B_matrix``/``wilson_b`` and ``force_constants`` or
    ``hessian`` entries in ``Species.metadata``.  Points without these fields
    are returned with their existing positive frequencies, which makes this
    helper safe for mixed precomputed/explicit paths.
    """

    result: dict[float, tuple[float, ...]] = {}
    for point in path.points:
        metadata: Mapping[str, Any] = point.dividing_surface.metadata
        b = metadata.get("wilson_b", metadata.get("B_matrix"))
        h = metadata.get("force_constants", metadata.get("hessian"))
        if b is None or h is None:
            result[point.coordinate] = point.dividing_surface.positive_frequencies_cm1
            continue
        masses = metadata.get("masses_amu")
        if masses is None:
            masses = [atom.mass_amu for atom in point.dividing_surface.atoms]
        result[point.coordinate] = curvilinear_frequencies(
            h,
            b,
            masses,
            frequency_scale_cm1=frequency_scale_cm1,
            remove_reaction_coordinate=remove_reaction_coordinate,
            hessian_is_cartesian=metadata.get("hessian_is_cartesian"),
        )
    return result


# ---------------------------------------------------------------------------
# Dual-level path interpolation (IOC, ISPE, IVTST)


def _strict_arrays(x: Iterable[float], y: Iterable[float], *, name: str) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(tuple(float(value) for value in x), dtype=float)
    ya = np.asarray(tuple(float(value) for value in y), dtype=float)
    if xa.ndim != 1 or ya.ndim != 1 or xa.size != ya.size or xa.size < 2:
        raise ValueError(f"{name} coordinates and values must have equal length >= 2")
    order = np.argsort(xa)
    xa, ya = xa[order], ya[order]
    if np.any(~np.isfinite(xa)) or np.any(~np.isfinite(ya)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(np.diff(xa) <= 0.0):
        raise ValueError(f"{name} coordinates must be strictly increasing")
    return xa, ya


def _linear_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    slopes = np.empty_like(y)
    slopes[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    slopes[0] = (y[1] - y[0]) / (x[1] - x[0])
    slopes[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return slopes


@dataclass(frozen=True, slots=True)
class DualLevelCorrection:
    """High-minus-low electronic-energy corrections along a reaction path.

    ``method`` accepts ``ioc`` (piecewise linear optimized correction),
    ``ispe``/``ivtst-m`` (mapped cubic interpolation), ``ivtst-0`` (piecewise
    constant nearest-point correction), or ``ivtst-1`` (cubic Hermite using
    supplied ``high_gradients``).  Values are electronic energies in kJ mol⁻¹.
    """

    low_coordinates: tuple[float, ...]
    low_energies_kj_mol: tuple[float, ...]
    high_coordinates: tuple[float, ...]
    high_energies_kj_mol: tuple[float, ...]
    method: str = "ispe"
    high_gradients_kj_mol_per_coordinate: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        lx, ly = _strict_arrays(self.low_coordinates, self.low_energies_kj_mol, name="low-level")
        hx, hy = _strict_arrays(self.high_coordinates, self.high_energies_kj_mol, name="high-level")
        method = self.method.lower()
        if method == "ivtst":
            method = "ivtst-m"
        if method not in {"ioc", "ispe", "ivtst-0", "ivtst0", "ivtst-1", "ivtst1", "ivtst-m", "ivtstm"}:
            raise ValueError("method must be ioc, ispe, ivtst-0, ivtst-1, or ivtst-m")
        gradients = self.high_gradients_kj_mol_per_coordinate
        if gradients is not None and len(gradients) != len(hx):
            raise ValueError("high_gradients must match high-level coordinates")
        object.__setattr__(self, "low_coordinates", tuple(float(v) for v in lx))
        object.__setattr__(self, "low_energies_kj_mol", tuple(float(v) for v in ly))
        object.__setattr__(self, "high_coordinates", tuple(float(v) for v in hx))
        object.__setattr__(self, "high_energies_kj_mol", tuple(float(v) for v in hy))
        object.__setattr__(self, "method", method)
        if gradients is not None:
            object.__setattr__(self, "high_gradients_kj_mol_per_coordinate", tuple(float(v) for v in gradients))

    @property
    def corrections_kj_mol(self) -> tuple[float, ...]:
        low = np.asarray(self.low_energies_kj_mol)
        high_on_low = np.asarray([self.high_energy_at(x) for x in self.low_coordinates])
        return tuple(float(v) for v in high_on_low - low)

    def high_energy_at(self, coordinate: float) -> float:
        """Interpolate the high-level energy at one path coordinate."""

        x = float(coordinate)
        hx = np.asarray(self.high_coordinates)
        hy = np.asarray(self.high_energies_kj_mol)
        method = self.method
        if method in {"ivtst-0", "ivtst0"}:
            return float(hy[int(np.argmin(np.abs(hx - x)))])
        if method in {"ivtst-1", "ivtst1"} and self.high_gradients_kj_mol_per_coordinate is not None:
            slopes = np.asarray(self.high_gradients_kj_mol_per_coordinate)
            i = int(np.clip(np.searchsorted(hx, x) - 1, 0, hx.size - 2))
            h = hx[i + 1] - hx[i]
            t = np.clip((x - hx[i]) / h, 0.0, 1.0)
            h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
            h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
            return float(h00 * hy[i] + h10 * h * slopes[i] + h01 * hy[i + 1] + h11 * h * slopes[i + 1])
        # ISPE and IVTST-M use mapped interpolation; linear interpolation is
        # monotone and deterministic, avoiding overshoots in sparse legacy data.
        return float(np.interp(x, hx, hy, left=hy[0], right=hy[-1]))

    def correction_at(self, coordinate: float) -> float:
        return self.high_energy_at(coordinate) - float(np.interp(float(coordinate), self.low_coordinates, self.low_energies_kj_mol, left=self.low_energies_kj_mol[0], right=self.low_energies_kj_mol[-1]))

    def corrected_energy(self, coordinate: float, low_energy: float | None = None) -> float:
        if low_energy is None:
            low_energy = float(np.interp(float(coordinate), self.low_coordinates, self.low_energies_kj_mol))
        return float(low_energy) + self.correction_at(coordinate)

    def corrected_path(self, path: ReactionPath) -> ReactionPath:
        """Return a copy of ``path`` with interpolated electronic energies."""

        points: list[PathPoint] = []
        for point in path.points:
            species = point.dividing_surface
            corrected = self.corrected_energy(point.coordinate, species.electronic_energy_kj_mol)
            points.append(replace(point, dividing_surface=replace(species, electronic_energy_kj_mol=corrected)))
        return replace(path, points=tuple(points))


def apply_dual_level_correction(path: ReactionPath, correction: DualLevelCorrection) -> ReactionPath:
    """Functional alias for :meth:`DualLevelCorrection.corrected_path`."""

    return correction.corrected_path(path)


@dataclass(frozen=True, slots=True)
class IVTSTResult:
    """High-level corrected variational rate diagnostics."""

    rate_si: float
    coordinate: float
    activation_gibbs_kj_mol: float
    correction_kj_mol: float
    method: str


def ivtst_rate_si(reaction: Any, temperature_k: float) -> IVTSTResult:
    """Evaluate a dual-level corrected variational rate for a reaction object."""

    if reaction.path is None:
        raise ValueError("IVTST requires reaction.path")
    model = getattr(reaction, "dual_level", None)
    if model is None:
        raise ValueError("IVTST requires reaction.dual_level")
    low = tuple(getattr(model, "low_energies_kj_mol", ()) or getattr(model, "low_level_energies_kj_mol", ()))
    high = tuple(getattr(model, "high_energies_kj_mol", ()) or getattr(model, "high_level_energies_kj_mol", ()))
    if len(low) != len(high) or len(low) < 2:
        raise ValueError("dual-level correction requires at least two low/high energies")
    path_coords = tuple(point.coordinate for point in reaction.path.points)
    low_coords = tuple(getattr(model, "low_coordinates", ()) or getattr(model, "coordinates", ())) or path_coords
    high_coords = tuple(getattr(model, "high_coordinates", ())) or low_coords
    method = str(getattr(model, "method", "ispe")).lower()
    if method == "ivtst":
        method = "ivtst-m"
    correction = DualLevelCorrection(
        low_coordinates=low_coords,
        low_energies_kj_mol=low,
        high_coordinates=high_coords,
        high_energies_kj_mol=high,
        method=method,
    )
    corrected = replace(reaction, path=correction.corrected_path(reaction.path))
    from .rates import cvt_rate_si

    result = cvt_rate_si(corrected, temperature_k)
    index = min(range(len(path_coords)), key=lambda i: abs(path_coords[i] - result.coordinate))
    return IVTSTResult(
        rate_si=result.rate_si,
        coordinate=result.coordinate,
        activation_gibbs_kj_mol=result.activation_gibbs_kj_mol,
        correction_kj_mol=correction.correction_at(path_coords[index]),
        method=method,
    )


# ---------------------------------------------------------------------------
# Weighted VRC-VTST faceted dividing surfaces


def _vibrational_q(frequencies_cm1: Sequence[float], temperature_k: float) -> float:
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    kt_wavenumber = BOLTZMANN_J_K * temperature_k / WAVENUMBER_TO_J
    log_q = 0.0
    for frequency in frequencies_cm1:
        if frequency <= 0.0:
            continue
        x = float(frequency) / kt_wavenumber
        if x < 700.0:
            log_q -= math.log1p(-math.exp(-x))
    return math.exp(min(700.0, log_q))


@dataclass(frozen=True, slots=True)
class VRCWeightedSample:
    """One weighted VRC dividing-surface sample or quadrature facet."""

    coordinate: float
    energy_kj_mol: float
    weight: float = 1.0
    frequencies_cm1: tuple[float, ...] = ()
    degeneracy: float = 1.0
    angular_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0.0 or self.degeneracy <= 0.0 or self.angular_factor < 0.0:
            raise ValueError("VRC weights and degeneracy must be non-negative")
        if not math.isfinite(self.energy_kj_mol):
            raise ValueError("VRC energy must be finite")
        object.__setattr__(self, "frequencies_cm1", tuple(float(v) for v in self.frequencies_cm1))


@dataclass(frozen=True, slots=True)
class VRCResult:
    temperature_k: float
    rate_si: float
    bottleneck_coordinate: float
    weighted_flux: float
    sample_rates_si: tuple[float, ...]


def vrc_vtst_rate(
    samples: Sequence[VRCWeightedSample],
    temperature_k: float,
    *,
    reactant_partition: float = 1.0,
    energy_reference_kj_mol: float = 0.0,
    prefactor_s_inv: float | None = None,
) -> VRCResult:
    """Compute a deterministic weighted canonical VRC-VTST rate.

    Samples are grouped by coordinate.  For each facet, the Boltzmann-weighted
    flux is evaluated from the vibrational state factor and angular/Monte-Carlo
    weight.  The variational bottleneck is the coordinate with the smallest
    positive flux.  A normalized weighted quadrature over all samples at that
    coordinate gives ``rate_si``.  ``prefactor_s_inv`` defaults to ``kBT/h``.
    """

    if not samples:
        raise ValueError("VRC requires at least one sample")
    if temperature_k <= 0.0 or reactant_partition <= 0.0:
        raise ValueError("temperature and reactant_partition must be positive")
    active = [sample for sample in samples if sample.weight > 0.0 and sample.angular_factor > 0.0]
    if not active:
        return VRCResult(float(temperature_k), 0.0, float(samples[0].coordinate), 0.0, tuple(0.0 for _ in samples))
    prefactor = float(prefactor_s_inv) if prefactor_s_inv is not None else BOLTZMANN_J_K * temperature_k / PLANCK_J_S
    if prefactor < 0.0 or not math.isfinite(prefactor):
        raise ValueError("prefactor_s_inv must be finite and non-negative")
    rt = GAS_CONSTANT_J_MOL_K * temperature_k / 1000.0
    sample_rates: list[float] = []
    for sample in samples:
        q = _vibrational_q(sample.frequencies_cm1, temperature_k)
        exponent = -((sample.energy_kj_mol - energy_reference_kj_mol) / rt)
        boltzmann = math.exp(max(-745.0, min(700.0, exponent)))
        sample_rates.append(prefactor * sample.weight * sample.degeneracy * sample.angular_factor * q * boltzmann / reactant_partition)
    coordinates = sorted({sample.coordinate for sample in active})
    fluxes: dict[float, float] = {}
    for coordinate in coordinates:
        selected = [i for i, sample in enumerate(samples) if sample.coordinate == coordinate]
        weights = sum(samples[i].weight for i in selected if samples[i].weight > 0.0)
        fluxes[coordinate] = sum(sample_rates[i] for i in selected) / max(weights, 1.0e-300)
    bottleneck = min(fluxes, key=fluxes.get)
    indices = [i for i, sample in enumerate(samples) if sample.coordinate == bottleneck]
    total_weight = sum(samples[i].weight for i in indices)
    rate = sum(sample_rates[i] for i in indices) / max(total_weight, 1.0e-300)
    return VRCResult(float(temperature_k), float(rate), float(bottleneck), float(fluxes[bottleneck]), tuple(sample_rates))


def vrc_vtst_rate_si(reaction: Any, temperature_k: float) -> VRCResult:
    """Evaluate VRC samples stored on ``reaction.vrc``."""

    model = getattr(reaction, "vrc", None)
    if model is None:
        raise ValueError("VRC-VTST requires reaction.vrc")
    samples = [
        VRCWeightedSample(
            coordinate=float(sample.coordinate),
            energy_kj_mol=float(getattr(sample, "energy_kj_mol", getattr(sample, "activation_gibbs_kj_mol", 0.0))),
            weight=float(sample.weight),
            frequencies_cm1=tuple(getattr(sample, "frequencies_cm1", ())),
            degeneracy=float(getattr(sample, "degeneracy", 1.0)),
        )
        for sample in model.samples
    ]
    result = vrc_vtst_rate(samples, temperature_k)
    symmetry = float(getattr(model, "symmetry_factor", 1.0))
    if symmetry <= 0.0:
        raise ValueError("VRC symmetry_factor must be positive")
    return replace(result, rate_si=result.rate_si / symmetry, weighted_flux=result.weighted_flux / symmetry)


# ---------------------------------------------------------------------------
# RRKM / SS-QRRK and pressure falloff


def rrkm_microcanonical_rate(
    energies_cm1: Sequence[float],
    sum_of_states: Sequence[float],
    density_of_states: Sequence[float],
    *,
    grain_cm1: float | None = None,
) -> np.ndarray:
    """Return ``k(E) = N‡(E)/(h ρ(E))`` on a common energy grid.

    ``density_of_states`` is per cm⁻¹ when ``grain_cm1`` is omitted, or per
    grain when it is supplied.  Invalid/zero densities yield zero rate rather
    than NaN, which is convenient at the low-energy end of a Beyer--Swinehart
    grid.
    """

    energies = np.asarray(energies_cm1, dtype=float)
    n = np.asarray(sum_of_states, dtype=float)
    rho = np.asarray(density_of_states, dtype=float)
    if energies.ndim != 1 or n.shape != energies.shape or rho.shape != energies.shape:
        raise ValueError("energies, sum_of_states, and density_of_states must have equal 1-D shape")
    if np.any(~np.isfinite(energies)) or np.any(energies < 0.0):
        raise ValueError("energies must be finite and non-negative")
    if grain_cm1 is None:
        d_e_j = WAVENUMBER_TO_J
    else:
        if grain_cm1 <= 0.0:
            raise ValueError("grain_cm1 must be positive")
        d_e_j = WAVENUMBER_TO_J * float(grain_cm1)
    rates = np.zeros_like(energies)
    valid = rho > 0.0
    rates[valid] = np.maximum(0.0, n[valid]) / (PLANCK_J_S * rho[valid] * d_e_j)
    return rates


def lindemann_falloff(k_zero: float, k_infinity: float, concentration_molecule_m3: float, *, broadening: float = 1.0) -> float:
    """Lindemann/Troe-form pressure falloff from low/high-pressure limits."""

    if min(k_zero, k_infinity, concentration_molecule_m3) < 0.0:
        raise ValueError("rate constants and concentration must be non-negative")
    if broadening <= 0.0:
        raise ValueError("broadening must be positive")
    k0m = float(k_zero) * float(concentration_molecule_m3)
    reduced = k0m / max(float(k_infinity), 1.0e-300)
    return float(k_infinity) * reduced / (1.0 + reduced) * float(broadening)


def troe_falloff_factor(reduced_pressure: float, f_cent: float = 0.6) -> float:
    """Return the standard Troe broadening factor ``F``."""

    if reduced_pressure < 0.0:
        raise ValueError("reduced_pressure must be non-negative")
    if not 0.0 < f_cent <= 1.0:
        raise ValueError("f_cent must be in (0, 1]")
    if reduced_pressure == 0.0:
        return 1.0
    log_pr = math.log10(reduced_pressure)
    log_fc = math.log10(f_cent)
    c = -0.4 - 0.67 * log_fc
    n = 0.75 - 1.27 * log_fc
    d = 0.14
    x = log_pr + c
    denominator = n - d * x
    log_f = log_fc / (1.0 + (x / max(abs(denominator), 1.0e-300)) ** 2)
    return float(10.0 ** log_f)


def pressure_dependent_rate(
    pressures_pa: Sequence[float],
    k_zero_m3_molecule_s: float,
    k_infinity_s: float,
    temperature_k: float,
    *,
    f_cent: float = 1.0,
    collision_efficiency: float = 1.0,
) -> np.ndarray:
    """Evaluate a pressure grid using a Lindemann/Troe falloff model."""

    pressures = np.asarray(pressures_pa, dtype=float)
    if pressures.ndim != 1 or np.any(~np.isfinite(pressures)) or np.any(pressures < 0.0):
        raise ValueError("pressures_pa must be a finite non-negative 1-D array")
    if temperature_k <= 0.0 or k_zero_m3_molecule_s < 0.0 or k_infinity_s < 0.0:
        raise ValueError("temperature and rate constants must be positive/non-negative")
    if collision_efficiency <= 0.0:
        raise ValueError("collision_efficiency must be positive")
    concentration = pressures / (BOLTZMANN_J_K * temperature_k) * collision_efficiency
    reduced = k_zero_m3_molecule_s * concentration / max(k_infinity_s, 1.0e-300)
    rates = np.empty_like(pressures)
    for i, pr in enumerate(reduced):
        factor = 1.0 if f_cent == 1.0 else troe_falloff_factor(float(pr), f_cent)
        rates[i] = lindemann_falloff(k_zero_m3_molecule_s, k_infinity_s, float(concentration[i]), broadening=factor)
    return rates


@dataclass(frozen=True, slots=True)
class SSQRRKModel:
    """System-specific RRKM plus pressure-falloff model."""

    energies_cm1: tuple[float, ...]
    transition_sum_of_states: tuple[float, ...]
    reactant_density_of_states: tuple[float, ...]
    grain_cm1: float
    high_pressure_rate_s: float
    low_pressure_rate_m3_molecule_s: float
    f_cent: float = 1.0
    collision_efficiency: float = 1.0

    def __post_init__(self) -> None:
        if len(self.energies_cm1) < 2 or len(self.energies_cm1) != len(self.transition_sum_of_states) or len(self.energies_cm1) != len(self.reactant_density_of_states):
            raise ValueError("SS-QRRK arrays must have equal length >= 2")
        if self.grain_cm1 <= 0.0 or self.high_pressure_rate_s < 0.0 or self.low_pressure_rate_m3_molecule_s < 0.0:
            raise ValueError("SS-QRRK grid and rate constants must be non-negative")

    @property
    def microcanonical_rates_s(self) -> np.ndarray:
        return rrkm_microcanonical_rate(self.energies_cm1, self.transition_sum_of_states, self.reactant_density_of_states, grain_cm1=self.grain_cm1)

    def rates_at_pressure(self, pressures_pa: Sequence[float], temperature_k: float) -> np.ndarray:
        return pressure_dependent_rate(pressures_pa, self.low_pressure_rate_m3_molecule_s, self.high_pressure_rate_s, temperature_k, f_cent=self.f_cent, collision_efficiency=self.collision_efficiency)

    def rate_at_pressure(self, pressure_pa: float, temperature_k: float) -> float:
        return float(self.rates_at_pressure((pressure_pa,), temperature_k)[0])


ss_qrrk_rate = pressure_dependent_rate
master_equation_rate = pressure_dependent_rate


# ---------------------------------------------------------------------------
# Solvent and surface corrections


def born_solvation_free_energy_kj_mol(charge: float, radius_angstrom: float, dielectric: float) -> float:
    """Born continuum solvation free energy for a spherical ion."""

    if radius_angstrom <= 0.0 or dielectric <= 0.0:
        raise ValueError("radius and dielectric must be positive")
    # epsilon_0 and elementary charge are exact enough for rate corrections;
    # writing them here avoids a dependency on a chemistry package.
    epsilon_0 = 8.854_187_8128e-12
    elementary_charge = 1.602_176_634e-19
    energy_j_per_molecule = -(charge * elementary_charge) ** 2 / (8.0 * math.pi * epsilon_0 * radius_angstrom * ANGSTROM_M) * (1.0 - 1.0 / dielectric)
    return float(energy_j_per_molecule * 6.022_140_76e23 / 1000.0)


def _species_environment_energy(species: Species, energies: Mapping[str, float], *, dielectric: float | None = None) -> float:
    metadata = species.metadata
    for key in ("solvation_free_energy_kj_mol", "solvation_kj_mol", "surface_binding_energy_kj_mol"):
        if key in metadata:
            return float(metadata[key])
    if species.name in energies:
        return float(energies[species.name])
    if dielectric is not None and metadata.get("charge") is not None and metadata.get("radius_angstrom") is not None:
        return born_solvation_free_energy_kj_mol(float(metadata["charge"]), float(metadata["radius_angstrom"]), dielectric)
    return 0.0


@dataclass(frozen=True, slots=True)
class SolventCorrection:
    """Continuum/explicit solvation free-energy correction."""

    solvation_free_energies_kj_mol: Mapping[str, float] = ()
    dielectric: float | None = None
    standard_state_molar: float = 1.0
    temperature_k: float = 298.15

    def correction_kj_mol(self, transition_state: Species, reactants: Sequence[Species]) -> float:
        if self.standard_state_molar <= 0.0 or self.temperature_k <= 0.0:
            raise ValueError("solvent standard state and temperature must be positive")
        mapping = dict(self.solvation_free_energies_kj_mol or {})
        return _species_environment_energy(transition_state, mapping, dielectric=self.dielectric) - sum(_species_environment_energy(species, mapping, dielectric=self.dielectric) for species in reactants)

    def corrected_barrier_kj_mol(self, barrier_kj_mol: float, transition_state: Species, reactants: Sequence[Species]) -> float:
        return float(barrier_kj_mol) + self.correction_kj_mol(transition_state, reactants)


@dataclass(frozen=True, slots=True)
class SurfaceCorrection:
    """Simple mean-field adsorbate/site correction for gas--surface reactions."""

    site_density_m2: float
    coverage: float = 0.0
    binding_energies_kj_mol: Mapping[str, float] = ()
    lateral_interaction_kj_mol: float = 0.0
    site_occupancy_ts: float = 1.0

    def __post_init__(self) -> None:
        if self.site_density_m2 <= 0.0:
            raise ValueError("site_density_m2 must be positive")
        if not 0.0 <= self.coverage < 1.0:
            raise ValueError("coverage must be in [0, 1)")
        if self.site_occupancy_ts <= 0.0:
            raise ValueError("site_occupancy_ts must be positive")

    def correction_kj_mol(self, transition_state: Species, reactants: Sequence[Species]) -> float:
        mapping = dict(self.binding_energies_kj_mol or {})
        ts_binding = _species_environment_energy(transition_state, mapping)
        reactant_binding = sum(_species_environment_energy(species, mapping) for species in reactants)
        # A positive lateral interaction raises the free energy as coverage
        # grows; the configurational site term penalizes a multi-site TS.
        lateral = self.lateral_interaction_kj_mol * self.coverage * self.site_occupancy_ts
        configurational = -GAS_CONSTANT_J_MOL_K * 298.15 / 1000.0 * self.site_occupancy_ts * math.log1p(-self.coverage)
        return float(ts_binding - reactant_binding + lateral + configurational)

    def rate_multiplier(self, temperature_k: float, transition_state: Species, reactants: Sequence[Species]) -> float:
        if temperature_k <= 0.0:
            raise ValueError("temperature must be positive")
        correction = self.correction_kj_mol(transition_state, reactants)
        return float(math.exp(max(-745.0, min(700.0, -correction * 1000.0 / (GAS_CONSTANT_J_MOL_K * temperature_k)))))


def environment_corrected_barrier_kj_mol(
    barrier_kj_mol: float,
    transition_state: Species,
    reactants: Sequence[Species],
    *,
    solvent: SolventCorrection | None = None,
    surface: SurfaceCorrection | None = None,
) -> float:
    """Apply solvent and surface corrections in sequence."""

    corrected = float(barrier_kj_mol)
    if solvent is not None:
        corrected += solvent.correction_kj_mol(transition_state, reactants)
    if surface is not None:
        corrected += surface.correction_kj_mol(transition_state, reactants)
    return corrected
