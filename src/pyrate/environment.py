"""Environment corrections and pressure-dependent kinetics helpers.

The equilibrium/rate models in :mod:`pyrate` are deliberately gas-phase and
pressure independent.  This module contains small, composable adapters for
workflows that need an environment while keeping those core models unchanged:

* continuum-solvent free-energy corrections (Born and empirical terms);
* Langmuir surface adsorption/coverage corrections;
* collision-frequency estimates and a conservative SS-QRRK/master-equation
  discretisation.

All energies are in kJ mol⁻¹, distances in Å, concentrations in mol L⁻¹ and
pressures in Pa unless a function says otherwise.  The routines are designed to
be useful as corrections around a :class:`~pyrate.models.Reaction`; callers can
add their returned free-energy differences to a gas-phase activation barrier.
They intentionally do not mutate the existing model or calculation classes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .constants import (
    AMU_KG,
    ANGSTROM_M,
    AVOGADRO_MOL,
    BOLTZMANN_J_K,
    GAS_CONSTANT_J_MOL_K,
)

# Vacuum permittivity and the common Born prefactor.  The latter is kept as a
# named constant so the unit conversion is visible and testable.
_EPSILON_0_F_M = 8.854_187_8128e-12
_EPSILON_R = 8.987_551_7923e9
_BORN_KJ_MOL_ANGSTROM_E2 = (
    0.5
    * _EPSILON_R
    * (1.602_176_634e-19**2)
    / (ANGSTROM_M * 1000.0)
    * AVOGADRO_MOL
)


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class SolvationCorrection:
    """Continuum-solvent correction for one species or transition state.

    ``charge_e`` and ``radius_angstrom`` activate the Born electrostatic term
    ``-166.0118 q²/r (1 - 1/ε)``.  The optional ``nonpolar_surface_angstrom2``
    and ``surface_tension_kj_mol_angstrom2`` terms provide a compact cavity/
    dispersion model.  ``reference_offset_kj_mol`` can hold a fitted residual
    from an explicit-solvent calculation.  The object is immutable and can be
    shared by reactant, product and transition-state corrections.
    """

    dielectric: float = 78.4
    charge_e: float = 0.0
    radius_angstrom: float | None = None
    nonpolar_surface_angstrom2: float = 0.0
    surface_tension_kj_mol_angstrom2: float = 0.0
    reference_offset_kj_mol: float = 0.0

    def __post_init__(self) -> None:
        dielectric = _positive(self.dielectric, "dielectric")
        if dielectric < 1.0:
            raise ValueError("dielectric must be at least one")
        object.__setattr__(self, "dielectric", dielectric)
        object.__setattr__(self, "charge_e", _finite(self.charge_e, "charge_e"))
        if self.radius_angstrom is not None:
            object.__setattr__(self, "radius_angstrom", _positive(self.radius_angstrom, "radius_angstrom"))
        area = _finite(self.nonpolar_surface_angstrom2, "nonpolar_surface_angstrom2")
        if area < 0.0:
            raise ValueError("nonpolar_surface_angstrom2 cannot be negative")
        object.__setattr__(self, "nonpolar_surface_angstrom2", area)
        object.__setattr__(
            self,
            "surface_tension_kj_mol_angstrom2",
            _finite(self.surface_tension_kj_mol_angstrom2, "surface_tension_kj_mol_angstrom2"),
        )
        object.__setattr__(
            self,
            "reference_offset_kj_mol",
            _finite(self.reference_offset_kj_mol, "reference_offset_kj_mol"),
        )
        if self.charge_e and self.radius_angstrom is None:
            raise ValueError("radius_angstrom is required for a charged Born correction")

    def free_energy_kj_mol(self) -> float:
        """Return the solvation free energy relative to vacuum."""

        born = 0.0
        if self.charge_e:
            born = -_BORN_KJ_MOL_ANGSTROM_E2 * self.charge_e**2 / self.radius_angstrom * (
                1.0 - 1.0 / self.dielectric
            )
        nonpolar = self.surface_tension_kj_mol_angstrom2 * self.nonpolar_surface_angstrom2
        return born + nonpolar + self.reference_offset_kj_mol

    def correction(self, *, vacuum_dielectric: float = 1.0) -> float:
        """Return a dielectric-only correction relative to another continuum.

        This is useful when a gas-phase free-energy value is used as the
        reference.  For a true vacuum reference use the default ``1``.
        """

        vacuum_dielectric = _positive(vacuum_dielectric, "vacuum_dielectric")
        if vacuum_dielectric < 1.0:
            raise ValueError("vacuum_dielectric must be at least one")
        if not self.charge_e or self.radius_angstrom is None:
            return self.free_energy_kj_mol()
        born = -_BORN_KJ_MOL_ANGSTROM_E2 * self.charge_e**2 / self.radius_angstrom * (
            1.0 / vacuum_dielectric - 1.0 / self.dielectric
        )
        return born + self.nonpolar_surface_angstrom2 * self.surface_tension_kj_mol_angstrom2 + self.reference_offset_kj_mol


def solvation_free_energy_kj_mol(
    *,
    charge_e: float = 0.0,
    radius_angstrom: float | None = None,
    dielectric: float = 78.4,
    nonpolar_surface_angstrom2: float = 0.0,
    surface_tension_kj_mol_angstrom2: float = 0.0,
    reference_offset_kj_mol: float = 0.0,
) -> float:
    """Convenience wrapper around :class:`SolvationCorrection`."""

    return SolvationCorrection(
        dielectric=dielectric,
        charge_e=charge_e,
        radius_angstrom=radius_angstrom,
        nonpolar_surface_angstrom2=nonpolar_surface_angstrom2,
        surface_tension_kj_mol_angstrom2=surface_tension_kj_mol_angstrom2,
        reference_offset_kj_mol=reference_offset_kj_mol,
    ).free_energy_kj_mol()


def reaction_solvation_delta_g_kj_mol(
    transition_state: SolvationCorrection,
    reactants: Iterable[SolvationCorrection] = (),
    products: Iterable[SolvationCorrection] = (),
) -> float:
    """Return ``G_TS - ΣG_reactants`` for a set of continuum corrections.

    ``products`` is accepted for symmetry and validation, but activation
    barriers depend only on the transition state and reactants.  It lets a
    caller pass the same species/environment lists to forward and reverse
    workflows without branching.
    """

    del products  # deliberately not part of the activation barrier
    reactant_sum = sum(item.free_energy_kj_mol() for item in reactants)
    return transition_state.free_energy_kj_mol() - reactant_sum


@dataclass(frozen=True, slots=True)
class SurfaceModel:
    """Mean-field Langmuir model for adsorption on an ideal surface.

    ``adsorption_free_energy_kj_mol`` is the standard free energy for an empty
    site and ``site_density_mol_m2`` is the total site density.  A species with
    ``adsorbate_concentration_m`` competes for sites through the dimensionless
    activity ``K p``/``K c``.  ``lateral_interaction_kj_mol`` is a mean-field
    Frumkin interaction: positive values make adsorption less favourable as
    coverage grows, negative values make it more favourable.
    """

    adsorption_free_energy_kj_mol: float
    site_density_mol_m2: float = 1.0e-5
    standard_concentration_m: float = 1000.0
    lateral_interaction_kj_mol: float = 0.0
    sticking_coefficient: float = 1.0
    degeneracy: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "adsorption_free_energy_kj_mol", _finite(self.adsorption_free_energy_kj_mol, "adsorption_free_energy_kj_mol"))
        object.__setattr__(self, "site_density_mol_m2", _positive(self.site_density_mol_m2, "site_density_mol_m2"))
        object.__setattr__(self, "standard_concentration_m", _positive(self.standard_concentration_m, "standard_concentration_m"))
        object.__setattr__(self, "lateral_interaction_kj_mol", _finite(self.lateral_interaction_kj_mol, "lateral_interaction_kj_mol"))
        sticking = _positive(self.sticking_coefficient, "sticking_coefficient")
        if sticking > 1.0:
            raise ValueError("sticking_coefficient must be in (0, 1]")
        object.__setattr__(self, "sticking_coefficient", sticking)
        if self.degeneracy <= 0:
            raise ValueError("degeneracy must be positive")

    def equilibrium_constant(self, temperature_k: float) -> float:
        """Return dimensionless adsorption equilibrium constant ``exp(-ΔG/RT)``."""

        temperature_k = _positive(temperature_k, "temperature_k")
        rt = GAS_CONSTANT_J_MOL_K * temperature_k / 1000.0
        return float(self.degeneracy) * math.exp(-self.adsorption_free_energy_kj_mol / rt)

    def coverage(
        self,
        temperature_k: float,
        *,
        concentration_m: float | None = None,
        pressure_pa: float | None = None,
        pressure_standard_pa: float = 100_000.0,
    ) -> float:
        """Return Langmuir coverage ``θ`` in ``[0, 1]``.

        Exactly one of ``concentration_m`` and ``pressure_pa`` may be supplied;
        when neither is supplied the standard concentration is used.
        """

        if concentration_m is not None and pressure_pa is not None:
            raise ValueError("provide concentration_m or pressure_pa, not both")
        if pressure_pa is not None:
            pressure_pa = _positive(pressure_pa, "pressure_pa")
            pressure_standard_pa = _positive(pressure_standard_pa, "pressure_standard_pa")
            activity = pressure_pa / pressure_standard_pa
        else:
            concentration = self.standard_concentration_m if concentration_m is None else _positive(concentration_m, "concentration_m")
            activity = concentration / self.standard_concentration_m
        rt = GAS_CONSTANT_J_MOL_K * _positive(temperature_k, "temperature_k") / 1000.0
        # Fixed-point iteration is stable for ordinary Frumkin interactions;
        # clipping protects strongly attractive fits from numerical overflow.
        theta = 0.0
        for _ in range(200):
            exponent = -(self.adsorption_free_energy_kj_mol + self.lateral_interaction_kj_mol * theta) / rt
            exponent = min(700.0, max(-700.0, exponent))
            target = (float(self.degeneracy) * activity * math.exp(exponent)) / (
                1.0 + float(self.degeneracy) * activity * math.exp(exponent)
            )
            updated = 0.5 * theta + 0.5 * target
            if abs(updated - theta) < 1.0e-13:
                theta = updated
                break
            theta = updated
        return min(1.0, max(0.0, theta))

    def adsorption_correction_kj_mol(self, temperature_k: float, coverage: float = 0.0) -> float:
        """Return the free-energy correction for a covered site."""

        temperature_k = _positive(temperature_k, "temperature_k")
        coverage = float(coverage)
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("coverage must lie in [0, 1]")
        rt = GAS_CONSTANT_J_MOL_K * temperature_k / 1000.0
        entropy = rt * math.log(max(1.0e-300, 1.0 - coverage))
        return self.adsorption_free_energy_kj_mol + self.lateral_interaction_kj_mol * coverage - entropy

    def rate_correction(
        self,
        temperature_k: float,
        *,
        concentration_m: float | None = None,
        pressure_pa: float | None = None,
        coverage: float | None = None,
    ) -> float:
        """Return a dimensionless multiplicative surface rate correction."""

        theta = self.coverage(temperature_k, concentration_m=concentration_m, pressure_pa=pressure_pa) if coverage is None else float(coverage)
        if not 0.0 <= theta <= 1.0:
            raise ValueError("coverage must lie in [0, 1]")
        # Adsorbed reactants contribute θ; an activated complex contributes the
        # same empty-site blocking factor.  The sticking coefficient accounts
        # for the fraction of collisions that form an adsorbate.
        return max(0.0, self.sticking_coefficient * theta * (1.0 - theta))


def surface_coverage(
    adsorption_free_energy_kj_mol: float,
    temperature_k: float,
    *,
    pressure_pa: float | None = None,
    concentration_m: float | None = None,
    **kwargs: float,
) -> float:
    """Convenience wrapper returning Langmuir coverage."""

    return SurfaceModel(adsorption_free_energy_kj_mol, **kwargs).coverage(
        temperature_k, pressure_pa=pressure_pa, concentration_m=concentration_m
    )


@dataclass(frozen=True, slots=True)
class CollisionModel:
    """Hard-sphere collision model used by SS-QRRK/master-equation adapters."""

    bath_gas_mass_amu: float = 28.0134
    collision_diameter_angstrom: float = 3.5
    energy_transfer_kj_mol: float = 10.0
    efficiency: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bath_gas_mass_amu", _positive(self.bath_gas_mass_amu, "bath_gas_mass_amu"))
        object.__setattr__(self, "collision_diameter_angstrom", _positive(self.collision_diameter_angstrom, "collision_diameter_angstrom"))
        object.__setattr__(self, "energy_transfer_kj_mol", _positive(self.energy_transfer_kj_mol, "energy_transfer_kj_mol"))
        efficiency = _positive(self.efficiency, "efficiency")
        object.__setattr__(self, "efficiency", efficiency)

    def frequency_s_inv(self, temperature_k: float, pressure_pa: float, *, target_mass_amu: float | None = None, target_diameter_angstrom: float | None = None) -> float:
        """Estimate binary hard-sphere collision frequency in s⁻¹."""

        temperature_k = _positive(temperature_k, "temperature_k")
        pressure_pa = _positive(pressure_pa, "pressure_pa")
        target_mass = self.bath_gas_mass_amu if target_mass_amu is None else _positive(target_mass_amu, "target_mass_amu")
        target_diameter = self.collision_diameter_angstrom if target_diameter_angstrom is None else _positive(target_diameter_angstrom, "target_diameter_angstrom")
        reduced_mass = AMU_KG * (target_mass * self.bath_gas_mass_amu) / (target_mass + self.bath_gas_mass_amu)
        mean_speed = math.sqrt(8.0 * BOLTZMANN_J_K * temperature_k / (math.pi * reduced_mass))
        number_density = pressure_pa / (BOLTZMANN_J_K * temperature_k)
        cross_section = math.pi * ((target_diameter + self.collision_diameter_angstrom) * ANGSTROM_M / 2.0) ** 2
        return number_density * cross_section * mean_speed * self.efficiency


@dataclass(frozen=True, slots=True)
class SSQRRKModel:
    """Discretised strong-collision SS-QRRK model.

    ``energy_grid_kj_mol`` and ``density_of_states`` describe the activated
    molecule. ``microcanonical_rates_s_inv`` are the reactive rates at each
    grain. The collision operator redistributes population toward the thermal
    Boltzmann distribution with a one-sided exponential energy-transfer kernel.
    """

    energy_grid_kj_mol: tuple[float, ...]
    density_of_states: tuple[float, ...]
    microcanonical_rates_s_inv: tuple[float, ...]
    collision_frequency_s_inv: float
    temperature_k: float
    energy_transfer_kj_mol: float = 10.0

    def __post_init__(self) -> None:
        energies = tuple(float(x) for x in self.energy_grid_kj_mol)
        dos = tuple(float(x) for x in self.density_of_states)
        rates = tuple(float(x) for x in self.microcanonical_rates_s_inv)
        if len(energies) < 2 or len(energies) != len(dos) or len(energies) != len(rates):
            raise ValueError("SS-QRRK arrays must have equal length >= 2")
        if any(not math.isfinite(x) for x in energies + dos + rates):
            raise ValueError("SS-QRRK arrays must be finite")
        if any(b <= a for a, b in zip(energies, energies[1:])):
            raise ValueError("energy_grid_kj_mol must be strictly increasing")
        if any(x < 0.0 for x in dos + rates):
            raise ValueError("density of states and rates cannot be negative")
        object.__setattr__(self, "energy_grid_kj_mol", energies)
        object.__setattr__(self, "density_of_states", dos)
        object.__setattr__(self, "microcanonical_rates_s_inv", rates)
        object.__setattr__(self, "collision_frequency_s_inv", _positive(self.collision_frequency_s_inv, "collision_frequency_s_inv"))
        object.__setattr__(self, "temperature_k", _positive(self.temperature_k, "temperature_k"))
        object.__setattr__(self, "energy_transfer_kj_mol", _positive(self.energy_transfer_kj_mol, "energy_transfer_kj_mol"))

    def thermal_distribution(self) -> np.ndarray:
        """Return the normalized DOS-weighted Boltzmann distribution."""

        rt = GAS_CONSTANT_J_MOL_K * self.temperature_k / 1000.0
        weights = np.asarray(self.density_of_states, dtype=float) * np.exp(-np.asarray(self.energy_grid_kj_mol) / rt)
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("density of states has zero thermal population")
        return weights / total

    def rate_matrix(self) -> np.ndarray:
        """Return a continuous-time generator with reaction sink."""

        energies = np.asarray(self.energy_grid_kj_mol, dtype=float)
        n = energies.size
        thermal = self.thermal_distribution()
        step = np.abs(energies[:, None] - energies[None, :])
        # Exponential energy-transfer kernel, with diagonal removed.  Each row
        # is renormalized, so the collision term conserves population.
        kernel = np.exp(-step / self.energy_transfer_kj_mol)
        np.fill_diagonal(kernel, 0.0)
        row_sums = kernel.sum(axis=1)
        kernel[row_sums > 0.0] /= row_sums[row_sums > 0.0, None]
        collision = self.collision_frequency_s_inv * (thermal[None, :] - np.eye(n))
        # Blend toward the DOS-weighted thermal target while retaining the
        # explicitly requested energy-transfer bandwidth.
        collision = 0.5 * collision + 0.5 * self.collision_frequency_s_inv * (kernel - np.eye(n))
        generator = collision - np.diag(np.asarray(self.microcanonical_rates_s_inv, dtype=float))
        # Numerical roundoff can perturb the diagonal; enforce a generator sign.
        for i in range(n):
            generator[i, i] = -float(np.sum(generator[i, np.arange(n) != i])) - self.microcanonical_rates_s_inv[i]
        return generator

    def effective_rate_s_inv(self) -> float:
        """Return the thermal SS-QRRK rate before collisional depletion."""

        return float(np.dot(self.thermal_distribution(), np.asarray(self.microcanonical_rates_s_inv)))


@dataclass(frozen=True, slots=True)
class MasterEquationResult:
    """Result of propagating a pressure-dependent master equation."""

    time_s: np.ndarray
    populations: np.ndarray
    effective_rate_s_inv: float
    eigenvalues_s_inv: np.ndarray

    @property
    def survival_probability(self) -> np.ndarray:
        return np.sum(self.populations, axis=1)


def solve_master_equation(
    model: SSQRRKModel,
    *,
    time_s: Sequence[float] | None = None,
    initial_population: Sequence[float] | None = None,
) -> MasterEquationResult:
    """Propagate a linear SS-QRRK/master equation with matrix exponentials.

    The implementation uses an eigen-decomposition, which is deterministic for
    the small grids normally used in pressure-dependent tests and examples.
    """

    matrix = model.rate_matrix()
    n = matrix.shape[0]
    if time_s is None:
        collision = model.collision_frequency_s_inv
        time = np.linspace(0.0, 8.0 / collision, 101)
    else:
        time = np.asarray(tuple(float(x) for x in time_s), dtype=float)
        if time.ndim != 1 or time.size == 0 or np.any(~np.isfinite(time)) or np.any(time < 0.0) or np.any(np.diff(time) < 0.0):
            raise ValueError("time_s must be finite, nonnegative and nondecreasing")
    if initial_population is None:
        initial = model.thermal_distribution()
    else:
        initial = np.asarray(tuple(float(x) for x in initial_population), dtype=float)
        if initial.shape != (n,) or np.any(initial < 0.0) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_population must be a finite nonnegative vector")
        total = float(np.sum(initial))
        if total <= 0.0:
            raise ValueError("initial_population must have positive sum")
        initial = initial / total
    # ``rate_matrix`` uses a row-generator convention (rows sum to the
    # negative reactive sink), so propagate the transpose for column vectors.
    generator = matrix.T
    eigenvalues, eigenvectors = np.linalg.eig(generator)
    inverse = np.linalg.inv(eigenvectors)
    coefficients = inverse @ initial
    populations = np.empty((time.size, n), dtype=float)
    for index, t in enumerate(time):
        populations[index] = np.real(eigenvectors @ (np.exp(eigenvalues * t) * coefficients))
    # Eigen decompositions can leave ~1e-15 negative noise.  Keep the physical
    # nonnegative domain without clipping meaningful reaction loss.
    populations[populations < 0.0] = 0.0
    effective = model.effective_rate_s_inv()
    return MasterEquationResult(time, populations, effective, np.real(eigenvalues))


def pressure_dependent_rate(
    model: SSQRRKModel,
    pressure_pa: float,
    *,
    bath_pressure_pa: float = 100_000.0,
) -> float:
    """Return the thermal effective rate scaled by collision frequency.

    This compact SS-QRRK closure is useful when a full master-equation solve is
    unnecessary: in the low-pressure limit the rate is collision controlled,
    while at high pressure it approaches the thermal microcanonical average.
    """

    pressure_pa = _positive(pressure_pa, "pressure_pa")
    bath_pressure_pa = _positive(bath_pressure_pa, "bath_pressure_pa")
    thermal_rate = model.effective_rate_s_inv()
    collision = model.collision_frequency_s_inv * pressure_pa / bath_pressure_pa
    return thermal_rate * collision / (thermal_rate + collision)


# Backwards-compatible short aliases used by examples and notebooks.
solvation_correction_kj_mol = solvation_free_energy_kj_mol
collision_frequency_s_inv = CollisionModel().frequency_s_inv

# Public aliases keep the terminology used in Polyrate documentation while the
# descriptive class names remain available for new code.
SolventModel = SolvationCorrection
MasterEquation = SSQRRKModel


def surface_correction_kj_mol(
    model_or_energy: SurfaceModel | float,
    temperature_k: float,
    coverage: float = 0.0,
    **kwargs: float,
) -> float:
    """Return a surface adsorption free-energy correction.

    ``model_or_energy`` may be an existing :class:`SurfaceModel` or the
    adsorption free energy in kJ mol⁻¹, in which case ``kwargs`` are passed to
    the model constructor.
    """

    model = (
        model_or_energy
        if isinstance(model_or_energy, SurfaceModel)
        else SurfaceModel(float(model_or_energy), **kwargs)
    )
    return model.adsorption_correction_kj_mol(temperature_k, coverage)


pressure_dependent_rate_si = pressure_dependent_rate
collision_frequency = CollisionModel().frequency_s_inv
# Terminology aliases used by solvent/surface input adapters.
solvent_correction_kj_mol = solvation_free_energy_kj_mol
surface_rate_correction = SurfaceModel.rate_correction
