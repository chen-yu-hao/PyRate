"""Stable Python interface to optional C++ numerical kernels.

The package remains usable from a source checkout when the extension has not
been built.  Production installations compile :mod:`pyrate._kernels`; the
fallbacks below deliberately mirror the native algorithms for reproducibility.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .constants import BOLTZMANN_J_K, HBAR_J_S, LIGHT_SPEED_M_S, PLANCK_J_S

_DISABLE_NATIVE = os.environ.get("PYRATE_DISABLE_NATIVE", "").lower() in {"1", "true", "yes"}

try:
    if _DISABLE_NATIVE:
        raise ImportError("native kernels disabled by PYRATE_DISABLE_NATIVE")
    from . import _kernels as _cpp
except ImportError:  # pragma: no cover - exercised explicitly through fallback tests
    _cpp = None

HAS_NATIVE = _cpp is not None


@dataclass(frozen=True, slots=True)
class RefinedMaximum:
    coordinate: float
    value: float
    sample_index: int
    fit_points: int


def backend_info() -> dict[str, str | bool]:
    return {
        "native": HAS_NATIVE,
        "backend": "cpp" if HAS_NATIVE else "python",
        "version": _cpp.native_version() if HAS_NATIVE else "python-fallback-0.2",
    }


def harmonic_log_partition(frequencies_cm1: Iterable[float], temperature_k: float) -> float:
    frequencies = tuple(float(value) for value in frequencies_cm1)
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if HAS_NATIVE:
        return float(_cpp.harmonic_log_partition(frequencies, float(temperature_k)))
    total = 0.0
    for frequency in frequencies:
        if frequency <= 0.0:
            continue
        ratio = (
            PLANCK_J_S
            * LIGHT_SPEED_M_S
            * 100.0
            * frequency
            / (BOLTZMANN_J_K * temperature_k)
        )
        total -= math.log1p(-math.exp(-ratio))
    return total


def wigner_kappa(imaginary_frequency_cm1: float, temperature_k: float) -> float:
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if HAS_NATIVE:
        return float(_cpp.wigner_kappa(float(imaginary_frequency_cm1), float(temperature_k)))
    ratio = (
        PLANCK_J_S
        * LIGHT_SPEED_M_S
        * 100.0
        * abs(imaginary_frequency_cm1)
        / (BOLTZMANN_J_K * temperature_k)
    )
    return 1.0 + ratio * ratio / 24.0


def refine_maximum(x: Sequence[float], y: Sequence[float]) -> RefinedMaximum:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if x_values.size == 0:
        raise ValueError("x and y cannot be empty")
    if np.any(np.diff(x_values) <= 0.0):
        raise ValueError("x must be strictly increasing")
    if HAS_NATIVE:
        coordinate, value, sample_index, fit_points = _cpp.refine_maximum(
            x_values.tolist(), y_values.tolist()
        )
        return RefinedMaximum(
            float(coordinate), float(value), int(sample_index), int(fit_points)
        )

    maximum_index = int(np.argmax(y_values))
    if x_values.size < 3 or maximum_index in {0, x_values.size - 1}:
        return RefinedMaximum(
            float(x_values[maximum_index]), float(y_values[maximum_index]), maximum_index, 1
        )
    can_use_five = bool(x_values.size >= 5 and 1 < maximum_index < x_values.size - 2)
    count = 5 if can_use_five else 3
    start = maximum_index - count // 2
    local_x = x_values[start : start + count]
    local_y = y_values[start : start + count]
    center = float(x_values[maximum_index])
    scale = max(abs(float(local_x[0]) - center), abs(float(local_x[-1]) - center))
    if scale <= 0.0:
        return RefinedMaximum(center, float(y_values[maximum_index]), maximum_index, 1)
    t = (local_x - center) / scale
    coefficients = np.polynomial.polynomial.polyfit(t, local_y, count - 1)
    derivative = np.polynomial.polynomial.polyder(coefficients)
    roots = np.polynomial.polynomial.polyroots(derivative)
    candidates = [float(t[0]), float(t[-1]), 0.0]
    candidates.extend(
        float(root.real)
        for root in roots
        if abs(float(root.imag)) < 1.0e-10 and t[0] <= root.real <= t[-1]
    )
    values = [float(np.polynomial.polynomial.polyval(candidate, coefficients)) for candidate in candidates]
    best = int(np.argmax(values))
    best_value = values[best]
    local_span = float(np.ptp(local_y))
    if not math.isfinite(best_value) or best_value > y_values[maximum_index] + 2.0 * max(
        1.0, local_span
    ):
        return RefinedMaximum(center, float(y_values[maximum_index]), maximum_index, 1)
    return RefinedMaximum(
        center + scale * candidates[best], best_value, maximum_index, count
    )


def refine_minimum(x: Sequence[float], y: Sequence[float]) -> RefinedMaximum:
    """Refine a sampled minimum using a local 3-point or 5-point fit (by negating y)."""

    neg_y = [-float(value) for value in y]
    refined = refine_maximum(x, neg_y)
    return RefinedMaximum(
        coordinate=refined.coordinate,
        value=-refined.value,
        sample_index=refined.sample_index,
        fit_points=refined.fit_points,
    )


def _linear_sqrt_integral(u0: float, u1: float) -> float:
    if u0 <= 0.0 and u1 <= 0.0:
        return 0.0
    delta = u1 - u0
    scale = max(1.0, abs(u0), abs(u1))
    if abs(delta) <= np.finfo(float).eps * scale:
        return math.sqrt(max(0.0, u0))
    if u0 > 0.0 and u1 > 0.0:
        return (2.0 / 3.0) * (u1**1.5 - u0**1.5) / delta
    if u0 > 0.0:
        return (2.0 / 3.0) * u0**1.5 / (u0 - u1)
    return (2.0 / 3.0) * u1**1.5 / (u1 - u0)


def wkb_exponent(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    energy_j_per_molecule: float,
    reduced_mass_kg: float,
) -> float:
    coordinates = tuple(float(value) for value in coordinates_m)
    potentials = tuple(float(value) for value in potentials_j_per_molecule)
    if len(coordinates) != len(potentials) or len(coordinates) < 2:
        raise ValueError("coordinates and potentials require matching lengths of at least 2")
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("coordinates must be strictly increasing")
    if reduced_mass_kg <= 0.0:
        raise ValueError("reduced mass must be positive")
    if HAS_NATIVE:
        return float(
            _cpp.wkb_exponent(
                coordinates, potentials, float(energy_j_per_molecule), float(reduced_mass_kg)
            )
        )
    integral = 0.0
    for index in range(1, len(coordinates)):
        dx = coordinates[index] - coordinates[index - 1]
        u0 = potentials[index - 1] - energy_j_per_molecule
        u1 = potentials[index] - energy_j_per_molecule
        integral += dx * _linear_sqrt_integral(u0, u1)
    return 2.0 * math.sqrt(2.0 * reduced_mass_kg) * integral / HBAR_J_S


def wkb_exponent_profile(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    energy_j_per_molecule: float,
    effective_masses_kg: Sequence[float],
) -> float:
    coordinates = tuple(float(value) for value in coordinates_m)
    potentials = tuple(float(value) for value in potentials_j_per_molecule)
    masses = tuple(float(value) for value in effective_masses_kg)
    if len(coordinates) != len(potentials) or len(coordinates) != len(masses) or len(coordinates) < 2:
        raise ValueError(
            "coordinates, potentials, and effective_masses require matching lengths >= 2"
        )
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("coordinates must be strictly increasing")
    if any(m <= 0.0 for m in masses):
        raise ValueError("effective masses must be positive")
    if HAS_NATIVE:
        return float(
            _cpp.wkb_exponent_profile(
                coordinates, potentials, float(energy_j_per_molecule), masses
            )
        )
    integral = 0.0
    for index in range(1, len(coordinates)):
        dx = coordinates[index] - coordinates[index - 1]
        u0 = 2.0 * masses[index - 1] * (potentials[index - 1] - energy_j_per_molecule)
        u1 = 2.0 * masses[index] * (potentials[index] - energy_j_per_molecule)
        integral += dx * _linear_sqrt_integral(u0, u1)
    return 2.0 * integral / HBAR_J_S


def _safe_exp(exponent: float) -> float:
    if exponent < -745.0:
        return 0.0
    if exponent > 709.0:
        return math.inf
    return math.exp(exponent)


def _thermal_sag_integral_fallback(
    potentials: tuple[float, ...],
    temperature_k: float,
    energy_grid_points: int,
    exponent_fn,
) -> float:
    threshold = max(potentials[0], potentials[-1])
    vmax = max(potentials)
    vm = vmax - threshold
    if vm <= 0.0:
        return 1.0
    count = energy_grid_points + energy_grid_points % 2
    beta = 1.0 / (BOLTZMANN_J_K * temperature_k)
    step = vm / count

    def integrand(energy: float) -> float:
        gamma = exponent_fn(energy)
        delta = max(0.0, beta * (vmax - energy))
        if delta == 0.0:
            return 0.0
        exp_num = _safe_exp(delta - gamma)
        if exp_num == 0.0:
            return 0.0
        reflection_factor = -math.expm1(-2.0 * delta)
        kemble_denom = 1.0 + _safe_exp(-gamma)
        return beta * exp_num * (reflection_factor / kemble_denom)

    total = integrand(threshold) + integrand(vmax)
    for index in range(1, count):
        total += (2.0 if index % 2 == 0 else 4.0) * integrand(threshold + index * step)
    return 1.0 + step * total / 3.0


def thermal_wkb_kappa(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    temperature_k: float,
    reduced_mass_kg: float,
    energy_grid_points: int = 400,
) -> float:
    coordinates = tuple(float(value) for value in coordinates_m)
    potentials = tuple(float(value) for value in potentials_j_per_molecule)
    if len(coordinates) != len(potentials) or len(coordinates) < 3:
        raise ValueError("coordinates and potentials require matching lengths of at least 3")
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("coordinates must be strictly increasing")
    if temperature_k <= 0.0 or reduced_mass_kg <= 0.0:
        raise ValueError("temperature and reduced mass must be positive")
    if energy_grid_points < 20:
        raise ValueError("energy_grid_points must be at least 20")
    if HAS_NATIVE:
        return float(
            _cpp.thermal_wkb_kappa(
                coordinates,
                potentials,
                float(temperature_k),
                float(reduced_mass_kg),
                int(energy_grid_points),
            )
        )
    return _thermal_sag_integral_fallback(
        potentials,
        temperature_k,
        energy_grid_points,
        lambda energy: wkb_exponent(coordinates, potentials, energy, reduced_mass_kg),
    )


def thermal_wkb_profile_kappa(
    coordinates_m: Sequence[float],
    potentials_j_per_molecule: Sequence[float],
    temperature_k: float,
    effective_masses_kg: Sequence[float],
    energy_grid_points: int = 400,
) -> float:
    coordinates = tuple(float(value) for value in coordinates_m)
    potentials = tuple(float(value) for value in potentials_j_per_molecule)
    masses = tuple(float(value) for value in effective_masses_kg)
    if len(coordinates) != len(potentials) or len(coordinates) != len(masses) or len(coordinates) < 3:
        raise ValueError(
            "coordinates, potentials, and effective_masses require matching lengths >= 3"
        )
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("coordinates must be strictly increasing")
    if temperature_k <= 0.0 or any(m <= 0.0 for m in masses):
        raise ValueError("temperature and effective masses must be positive")
    if energy_grid_points < 20:
        raise ValueError("energy_grid_points must be at least 20")
    if HAS_NATIVE:
        return float(
            _cpp.thermal_wkb_profile_kappa(
                coordinates,
                potentials,
                float(temperature_k),
                masses,
                int(energy_grid_points),
            )
        )
    return _thermal_sag_integral_fallback(
        potentials,
        temperature_k,
        energy_grid_points,
        lambda energy: wkb_exponent_profile(coordinates, potentials, energy, masses),
    )


def eckart_transmission(
    energy_j_per_molecule: float,
    barrier_forward_j_per_molecule: float,
    barrier_reverse_j_per_molecule: float,
    imaginary_frequency_cm1: float,
) -> float:
    if HAS_NATIVE:
        return float(
            _cpp.eckart_transmission(
                float(energy_j_per_molecule),
                float(barrier_forward_j_per_molecule),
                float(barrier_reverse_j_per_molecule),
                float(imaginary_frequency_cm1),
            )
        )
    if barrier_forward_j_per_molecule <= 0.0 or barrier_reverse_j_per_molecule <= 0.0:
        threshold = max(0.0, barrier_forward_j_per_molecule - barrier_reverse_j_per_molecule)
        return 1.0 if energy_j_per_molecule >= threshold else 0.0
    threshold = max(0.0, barrier_forward_j_per_molecule - barrier_reverse_j_per_molecule)
    if energy_j_per_molecule <= threshold:
        return 0.0
    u_star = PLANCK_J_S * LIGHT_SPEED_M_S * 100.0 * abs(imaginary_frequency_cm1)
    if u_star <= 0.0:
        return 1.0 if energy_j_per_molecule >= barrier_forward_j_per_molecule else 0.0
    inv_sqrt_sum = (1.0 / math.sqrt(barrier_forward_j_per_molecule)) + (
        1.0 / math.sqrt(barrier_reverse_j_per_molecule)
    )
    # Johnston & Heicklen (1962): 2*pi*a = 2*sqrt(alpha_1*xi) / (alpha_1^-1/2 + alpha_2^-1/2)
    # with alpha_i = 2*pi*V_i / (h*nu*), i.e. a prefactor of 4*pi / (h*nu*).
    two_pi_alpha = (4.0 * math.pi / u_star) * math.sqrt(energy_j_per_molecule) / inv_sqrt_sum
    two_pi_beta = (
        (4.0 * math.pi / u_star)
        * math.sqrt(
            max(
                0.0,
                energy_j_per_molecule
                - barrier_forward_j_per_molecule
                + barrier_reverse_j_per_molecule,
            )
        )
        / inv_sqrt_sum
    )
    a_plus = two_pi_alpha + two_pi_beta
    a_minus = abs(two_pi_alpha - two_pi_beta)
    d_arg = (
        16.0
        * barrier_forward_j_per_molecule
        * barrier_reverse_j_per_molecule
        / (u_star * u_star)
        - 1.0
    )
    if d_arg >= 0.0:
        two_pi_delta = math.pi * math.sqrt(d_arg)
        scale = max(a_plus, two_pi_delta)
        num = (_safe_exp(a_plus - scale) + _safe_exp(-a_plus - scale)) - (
            _safe_exp(a_minus - scale) + _safe_exp(-a_minus - scale)
        )
        den = (_safe_exp(a_plus - scale) + _safe_exp(-a_plus - scale)) + (
            _safe_exp(two_pi_delta - scale) + _safe_exp(-two_pi_delta - scale)
        )
        return min(1.0, max(0.0, num / den))
    cos_term = 2.0 * math.cos(math.pi * math.sqrt(-d_arg))
    scale = a_plus
    num = (1.0 + _safe_exp(-2.0 * a_plus)) - (
        _safe_exp(a_minus - scale) + _safe_exp(-a_minus - scale)
    )
    den = (1.0 + _safe_exp(-2.0 * a_plus)) + cos_term * _safe_exp(-scale)
    return min(1.0, max(0.0, num / den))


def thermal_eckart_kappa(
    barrier_forward_j_per_molecule: float,
    barrier_reverse_j_per_molecule: float,
    imaginary_frequency_cm1: float,
    temperature_k: float,
    energy_grid_points: int = 400,
) -> float:
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if energy_grid_points < 20:
        raise ValueError("energy_grid_points must be at least 20")
    if HAS_NATIVE:
        return float(
            _cpp.thermal_eckart_kappa(
                float(barrier_forward_j_per_molecule),
                float(barrier_reverse_j_per_molecule),
                float(imaginary_frequency_cm1),
                float(temperature_k),
                int(energy_grid_points),
            )
        )
    threshold = max(0.0, barrier_forward_j_per_molecule - barrier_reverse_j_per_molecule)
    vm = barrier_forward_j_per_molecule - threshold
    if vm <= 0.0 or abs(imaginary_frequency_cm1) <= 0.0:
        return 1.0
    count = energy_grid_points + energy_grid_points % 2
    beta = 1.0 / (BOLTZMANN_J_K * temperature_k)
    e_upper = barrier_forward_j_per_molecule + 16.0 * BOLTZMANN_J_K * temperature_k
    step = (e_upper - threshold) / count

    def integrand(energy: float) -> float:
        prob = eckart_transmission(
            energy,
            barrier_forward_j_per_molecule,
            barrier_reverse_j_per_molecule,
            imaginary_frequency_cm1,
        )
        if prob <= 0.0:
            return 0.0
        return beta * prob * _safe_exp(beta * (barrier_forward_j_per_molecule - energy))

    total = integrand(threshold) + integrand(e_upper)
    for index in range(1, count):
        total += (2.0 if index % 2 == 0 else 4.0) * integrand(threshold + index * step)
    return step * total / 3.0 + _safe_exp(beta * (barrier_forward_j_per_molecule - e_upper))


def beyer_swinehart(
    frequencies_cm1: Iterable[float], grain_cm1: float, grain_count: int
) -> np.ndarray:
    frequencies = tuple(float(value) for value in frequencies_cm1)
    if grain_cm1 <= 0.0 or grain_count <= 0:
        raise ValueError("grain_cm1 and grain_count must be positive")
    if HAS_NATIVE:
        return np.asarray(_cpp.beyer_swinehart(frequencies, grain_cm1, grain_count), dtype=float)
    density = np.zeros(grain_count, dtype=float)
    density[0] = 1.0
    for frequency in frequencies:
        if frequency <= 0.0:
            continue
        step = max(1, int(round(frequency / grain_cm1)))
        for index in range(step, grain_count):
            density[index] += density[index - step]
    return density


def _gauss_legendre_fallback(
    lower: float, upper: float, n_points: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    m = (n_points + 1) // 2
    xm = 0.5 * (upper + lower)
    xl = 0.5 * (upper - lower)
    nodes = [0.0] * n_points
    weights = [0.0] * n_points
    for i in range(m):
        z = math.cos(math.pi * (i + 0.75) / (n_points + 0.5))
        pp = 0.0
        for _ in range(64):
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n_points + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            pp = n_points * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) <= 1.0e-15:
                break
        nodes[i] = xm - xl * z
        nodes[n_points - 1 - i] = xm + xl * z
        w = 2.0 * abs(xl) / ((1.0 - z * z) * pp * pp)
        weights[i] = w
        weights[n_points - 1 - i] = w
    if lower > upper:
        nodes.reverse()
        weights.reverse()
    return tuple(nodes), tuple(weights)


def gauss_legendre_quadrature(
    lower: float, upper: float, n_points: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if HAS_NATIVE:
        nodes, weights = _cpp.gauss_legendre_quadrature(
            float(lower), float(upper), int(n_points)
        )
        return tuple(float(v) for v in nodes), tuple(float(v) for v in weights)
    return _gauss_legendre_fallback(float(lower), float(upper), int(n_points))


def _neville_fallback(
    xa: Sequence[float], ya: Sequence[float], x: float
) -> float:
    n = len(xa)
    if n == 0:
        return 0.0
    if n == 1:
        return float(ya[0])
    c = [float(v) for v in ya]
    d = [float(v) for v in ya]
    ns = 0
    dif = abs(x - xa[0])
    for i in range(n):
        dift = abs(x - xa[i])
        if dift < dif:
            ns = i
            dif = dift
    y = float(ya[ns])
    for m in range(1, n):
        for i in range(n - m):
            ho = xa[i] - x
            hp = xa[i + m] - x
            w = c[i + 1] - d[i]
            den = ho - hp
            if abs(den) <= 1.0e-30:
                d[i] = 0.0
                c[i] = 0.0
            else:
                ratio = w / den
                d[i] = hp * ratio
                c[i] = ho * ratio
        if 2 * ns < (n - m):
            dy = c[ns]
        else:
            ns -= 1
            dy = d[ns]
        y += dy
    return y


def neville_interpolate(
    x_nodes: Sequence[float], y_nodes: Sequence[float], target_x: float
) -> float:
    xa = tuple(float(v) for v in x_nodes)
    ya = tuple(float(v) for v in y_nodes)
    if len(xa) != len(ya) or not xa:
        raise ValueError("x_nodes and y_nodes must have matching non-zero length")
    if HAS_NATIVE:
        return float(_cpp.neville_interpolate(xa, ya, float(target_x)))
    return _neville_fallback(xa, ya, float(target_x))


def lct_integrate_amplitudes(
    st: Sequence[float],
    energies: Sequence[float],
    sin_chi: Sequence[float],
    theta: Sequence[float],
    tau: Sequence[float],
    reduced_mass: float,
    ie0_index: int,
    n_quad: int = 120,
    poly_order: int = 4,
) -> tuple[float, ...]:
    st_vals = tuple(float(v) for v in st)
    esv_vals = tuple(float(v) for v in energies)
    sin_vals = tuple(float(v) for v in sin_chi)
    th_vals = tuple(float(v) for v in theta)
    tau_vals = tuple(float(v) for v in tau)
    nemx = len(esv_vals)
    if (
        nemx < 2
        or len(st_vals) != nemx
        or len(sin_vals) != nemx
        or len(th_vals) != nemx
        or len(tau_vals) != nemx
    ):
        raise ValueError(
            "st, energies, sin_chi, theta, and tau must have matching length >= 2"
        )
    if reduced_mass <= 0.0 or n_quad <= 0 or poly_order <= 0:
        raise ValueError("reduced_mass, n_quad, and poly_order must be positive")
    if HAS_NATIVE:
        return tuple(
            float(v)
            for v in _cpp.lct_integrate_amplitudes(
                st_vals,
                esv_vals,
                sin_vals,
                th_vals,
                tau_vals,
                float(reduced_mass),
                int(ie0_index),
                int(n_quad),
                int(poly_order),
            )
        )

    amplitudes = [0.0] * nemx
    is0 = max(0, int(ie0_index))
    if is0 + 1 < nemx:
        avail = nemx - is0
        nn = min(int(poly_order), avail)
        for is_idx in range(is0 + 1, nemx - 1):
            s1 = st_vals[is_idx - 1]
            s2 = st_vals[is_idx]
            if abs(s2 - s1) <= 1.0e-15:
                continue
            low, high = (s1, s2) if s1 < s2 else (s2, s1)
            sx, wtx = _gauss_legendre_fallback(low, high, int(n_quad))
            jj = is_idx - 1
            nm = nn // 2
            if jj == is0 or nm == 0:
                nm = 1
            if jj + 1 == nemx and nn >= 3:
                nm = nn - 1
            jnl = (jj + 1 - nm) if (jj + 1 >= nm) else is0
            if jnl < is0:
                jnl = is0
            if jnl + nn > nemx:
                jnl = nemx - nn
            st_sub = st_vals[jnl : jnl + nn]
            th_sub = th_vals[jnl : jnl + nn]
            tau_sub = tau_vals[jnl : jnl + nn]
            esv_sub = esv_vals[jnl : jnl + nn]
            sin_sub = sin_vals[jnl : jnl + nn]
            for iq, sxx in enumerate(sx):
                yth = _neville_fallback(st_sub, th_sub, sxx)
                yt = _neville_fallback(st_sub, tau_sub, sxx)
                yv = _neville_fallback(st_sub, esv_sub, sxx)
                ys = _neville_fallback(st_sub, sin_sub, sxx)
                if yt <= 0.0 or ys == 0.0:
                    continue
                exp_th = _safe_exp(-yth)
                if exp_th <= 0.0 or not math.isfinite(exp_th):
                    continue
                term = wtx[iq] * (ys / yt) * exp_th
                for ie in range(is_idx, nemx - 1):
                    vr_sq = 2.0 * (esv_vals[ie] - yv) / reduced_mass
                    if vr_sq > 0.0:
                        amplitudes[ie] += term / math.sqrt(vr_sq)
    return tuple(amplitudes)


def lct_uniformize_probabilities(
    cumulative_primitives: Sequence[Sequence[float]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    rows = tuple(tuple(float(v) for v in row) for row in cumulative_primitives)
    if not rows:
        raise ValueError("cumulative_primitives must contain at least one state")
    nemx = len(rows[0])
    if nemx < 2:
        raise ValueError("each state row must contain at least 2 entries")
    if any(len(row) != nemx for row in rows):
        raise ValueError("all state rows must have equal length")
    if HAS_NATIVE:
        unif, sorted_p = _cpp.lct_uniformize_probabilities(rows)
        return (
            tuple(tuple(float(v) for v in row) for row in unif),
            tuple(float(v) for v in sorted_p),
        )

    nemax = nemx - 1
    unif_rows: list[tuple[float, ...]] = []
    sorted_probs = [0.0] * nemax
    for s_idx, row in enumerate(rows):
        pvag = row[nemax]
        out_row = [0.0] * nemx
        out_row[nemax] = 0.5
        for j in range(nemax):
            pixx = row[j]
            if pixx > 0.0:
                term = pixx / (1.0 + pixx)
                if pvag > 1.0e-30:
                    plc = (1.0 + (1.0 / pvag - 1.0) * pixx / (2.0 * pvag)) * term
                else:
                    plc = term
                out_row[j] = min(1.0, max(0.0, plc))
            else:
                out_row[j] = 0.0
            if s_idx == 0 or out_row[j] > sorted_probs[j]:
                sorted_probs[j] = out_row[j]
        unif_rows.append(tuple(out_row))
    return tuple(unif_rows), tuple(sorted_probs)


def thermal_boltz_probabilities_kappa(
    energies_j: Sequence[float],
    weights_j: Sequence[float],
    probabilities: Sequence[float],
    barrier_j: float,
    temperature_k: float,
) -> float:
    e_vals = tuple(float(v) for v in energies_j)
    w_vals = tuple(float(v) for v in weights_j)
    p_vals = tuple(float(v) for v in probabilities)
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if len(e_vals) != len(w_vals) or len(e_vals) != len(p_vals):
        raise ValueError(
            "energies_j, weights_j, and probabilities must have equal length"
        )
    if HAS_NATIVE:
        return float(
            _cpp.thermal_boltz_probabilities_kappa(
                e_vals, w_vals, p_vals, float(barrier_j), float(temperature_k)
            )
        )
    beta = 1.0 / (BOLTZMANN_J_K * temperature_k)
    total = 0.0
    for e_val, w_val, p_val in zip(e_vals, w_vals, p_vals):
        if p_val <= 0.0:
            continue
        delta = max(0.0, beta * (barrier_j - e_val))
        if delta <= 0.0:
            continue
        exp_num = _safe_exp(delta + math.log(p_val))
        if exp_num == 0.0:
            continue
        reflection_factor = -math.expm1(-2.0 * delta)
        total += beta * w_val * exp_num * reflection_factor
    return 1.0 + total


