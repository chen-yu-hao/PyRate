"""Modern TOML input loading, validation, and Polyrate ``.dat``/``.fu40``/PES execution."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .constants import (
    AMU_KG,
    ANGSTROM_M,
    AVOGADRO_MOL,
    BAR_PA,
    BOHR_ANGSTROM,
    BOHR_M,
    HARTREE_J,
    LIGHT_SPEED_M_S,
    PLANCK_J_S,
)
from .elements import atomic_number
from .legacy import (
    LegacyInput,
    LegacySection,
    LegacySemanticError,
    load_legacy,
    parse_legacy_text,
    validate_legacy_semantics,
)
from .mep import MepResult, curvilinear_corrected_hessian, integrate_mep
from .models import (
    Atom,
    CalculationOptions,
    ElectronicState,
    MepOptions,
    PathPoint,
    Reaction,
    ReactionPath,
    Species,
    TunnelingProfile,
)
from .normal_modes import _rigid_basis, normal_modes
from .pes import (
    EckartValleySurface,
    ExtendedLepsSurface,
    MullerBrownSurface,
    PesEvaluation,
    PotentialEnergySurface,
    SupermoleculeFragmentPes,
    build_pes_from_mapping,
    build_species_from_pes,
    get_builtin_pes,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LoadedCalculation:
    name: str
    reaction: Reaction
    options: CalculationOptions
    raw: Mapping[str, Any]
    source: str | None = None
    mep_result: MepResult | None = field(default=None, compare=False)
    reactant_evaluations: tuple[PesEvaluation | None, ...] = field(
        default=(), repr=False, compare=False
    )
    product_evaluations: tuple[PesEvaluation | None, ...] = field(
        default=(), repr=False, compare=False
    )

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": "pyrate-toml",
            "source": self.source,
            "name": self.name,
            "reactants": [species.name for species in self.reaction.reactants],
            "transition_state": self.reaction.transition_state.name,
            "products": [species.name for species in self.reaction.products],
            "molecularity": self.reaction.molecularity,
            "temperatures_k": list(self.options.temperatures_k),
            "method": self.options.method,
            "tunneling": self.options.tunneling,
            "path_points": len(self.reaction.path.points) if self.reaction.path else 0,
            "standard_pressure_pa": self.reaction.standard_pressure_pa,
        }
        if self.mep_result is not None:
            payload["mep_method"] = self.mep_result.method
            payload["mep_hessian_evaluations"] = self.mep_result.hessian_evaluations
        return payload

    def to_fu40_text(
        self,
        *,
        hessform: str = "packed",
        freqsource: str = "hessian",
    ) -> str:
        """Serialize the integrated MEP and stationary points to a Polyrate ``.fu40`` file."""

        if self.mep_result is None:
            raise ValueError("LoadedCalculation has no integrated MEP result to export")
        r_evals = self.reactant_evaluations if self.reactant_evaluations else None
        p_evals = self.product_evaluations if self.product_evaluations else None
        return self.mep_result.to_fu40_text(
            reactants=self.reaction.reactants,
            products=self.reaction.products,
            hessform=hessform,
            freqsource=freqsource,
            reactant_evals=r_evals,
            product_evals=p_evals,
        )

    def save_fu40(
        self,
        path: str | Path,
        *,
        hessform: str = "packed",
        freqsource: str = "hessian",
    ) -> Path:
        """Write the integrated MEP and stationary points to a ``.fu40`` file."""

        target = Path(path)
        text = self.to_fu40_text(hessform=hessform, freqsource=freqsource)
        target.write_text(text, encoding="utf-8")
        return target


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _with_calculation_defaults(
    species_table: Mapping[str, Any], calculation: Mapping[str, Any]
) -> dict[str, Any]:
    merged = dict(species_table)
    for key in ("mass_convention", "frequency_scale", "frequency_floor_cm1"):
        if key not in merged and calculation.get(key) is not None:
            merged[key] = calculation[key]
    return merged


def _match_atom_indices_by_symbol(
    fragment_symbols: Sequence[str],
    full_symbols: Sequence[str],
    used: set[int],
) -> tuple[int, ...] | None:
    full_z = [atomic_number(sym) for sym in full_symbols]
    assigned: list[int] = []
    for sym in fragment_symbols:
        z_target = atomic_number(sym)
        found: int | None = None
        for idx, z_val in enumerate(full_z):
            if idx not in used and z_val == z_target:
                found = idx
                break
        if found is None:
            return None
        used.add(found)
        assigned.append(found)
    return tuple(assigned)


def _species_atoms_from_mapping(
    table: Mapping[str, Any], *, default_mass_convention: str
) -> tuple[Atom, ...]:
    raw_atoms = table.get("atoms", ())
    return tuple(
        Atom.from_mapping(item, default_mass_convention=default_mass_convention)
        for item in raw_atoms
    )


def _species_elec_states_from_mapping(table: Mapping[str, Any]) -> tuple[ElectronicState, ...]:
    raw = table.get("electronic_states")
    if raw:
        return tuple(ElectronicState.from_mapping(item) for item in raw)
    return (ElectronicState(),)


def _is_supermolecule_pes(
    pes: PotentialEnergySurface,
    pot_table: Mapping[str, Any] | None = None,
    *,
    explicit_flag: bool | None = None,
) -> bool:
    if explicit_flag is not None:
        return explicit_flag
    if pot_table is not None and "supermolecule" in pot_table:
        return bool(pot_table["supermolecule"])
    return isinstance(pes, (ExtendedLepsSurface, MullerBrownSurface, EckartValleySurface))


def _assemble_supermolecule_group(
    pes: PotentialEnergySurface,
    full_symbols: tuple[str, ...],
    ts_coords_angstrom: FloatArray,
    group_tables: Sequence[Mapping[str, Any]],
    default_prefix: str,
    *,
    calculation: Mapping[str, Any],
    mass_convention: str,
    use_supermolecule: bool,
    global_optimize: bool,
) -> tuple[tuple[Species, ...], tuple[PesEvaluation | None, ...], list[tuple[int, ...]]]:
    used_indices: set[int] = set()
    parsed_specs: list[tuple[str, tuple[Atom, ...], dict[str, Any]]] = []
    index_groups: list[tuple[int, ...]] = []

    for idx, raw_tbl in enumerate(group_tables):
        merged = _with_calculation_defaults(raw_tbl, calculation)
        sp_name = str(merged.get("name", f"{default_prefix}-{idx + 1}"))
        atoms = _species_atoms_from_mapping(merged, default_mass_convention=mass_convention)
        parsed_specs.append((sp_name, atoms, merged))
        explicit_idx = merged.get("fragment_indices")
        if explicit_idx is not None:
            grp = tuple(int(i) for i in explicit_idx)
            used_indices.update(grp)
        else:
            frag_syms = [a.symbol for a in atoms]
            matched = _match_atom_indices_by_symbol(frag_syms, full_symbols, used_indices)
            grp = matched if matched is not None else tuple(range(len(atoms)))
        index_groups.append(grp)

    species_out: list[Species] = []
    evals_out: list[PesEvaluation | None] = []
    frag_pess: list[SupermoleculeFragmentPes | None] = []

    for (sp_name, atoms, merged), grp_idx in zip(parsed_specs, index_groups, strict=True):
        needs_pes = bool(merged.get("optimize", global_optimize)) or (
            len(atoms) > 1 and "frequencies_cm1" not in merged
        ) or ("electronic_energy_kj_mol" not in merged)

        if not atoms or not needs_pes:
            species_out.append(Species.from_mapping(merged, sp_name))
            evals_out.append(None)
            frag_pess.append(None)
            continue

        coords = np.asarray([a.coordinates_angstrom for a in atoms], dtype=float)
        frag_sym_tuple = tuple(a.symbol for a in atoms)
        frag_masses = [float(a.mass_amu or 1.0) for a in atoms]
        if use_supermolecule and len(grp_idx) < len(full_symbols):
            frag_pes = SupermoleculeFragmentPes(
                pes, full_symbols, grp_idx, ts_coords_angstrom
            )
            frag_pess.append(frag_pes)
            sp_pes: PotentialEnergySurface = frag_pes
        else:
            sp_pes = pes
            frag_pess.append(None)

        do_opt = bool(merged.get("optimize", global_optimize)) and len(atoms) > 1
        sp_built, ev_built = build_species_from_pes(
            sp_pes,
            sp_name,
            frag_sym_tuple,
            coords,
            symmetry_number=int(merged.get("symmetry_number", 1)),
            electronic_states=_species_elec_states_from_mapping(merged),
            charge=int(merged.get("charge", 0)),
            multiplicity=int(merged.get("multiplicity", 1)),
            mass_convention=mass_convention,
            masses_amu=frag_masses,
            is_transition_state=False,
            rotor_type=str(merged.get("rotor_type", "auto")),
            frequency_scale=float(merged.get("frequency_scale", 1.0)),
            frequency_floor_cm1=(
                float(merged["frequency_floor_cm1"])
                if merged.get("frequency_floor_cm1") is not None
                else None
            ),
            optimize=do_opt,
            gtol_au=float(merged.get("gtol_au", 1.0e-6)),
        )
        species_out.append(sp_built)
        evals_out.append(ev_built)

    # For supermolecule fragment evaluations (A + B evaluated together on one full PES),
    # avoid double-counting the asymptotic energy across two fragments in the same pair.
    if (
        len(species_out) == 2
        and use_supermolecule
        and any(fp is not None for fp in frag_pess)
        and evals_out[0] is not None
        and evals_out[1] is not None
    ):
        sp0, sp1 = species_out
        full_pair = np.array(ts_coords_angstrom, dtype=float, copy=True)
        for local_i, full_i in enumerate(index_groups[0]):
            full_pair[full_i] = np.asarray(sp0.atoms[local_i].coordinates_angstrom, dtype=float)
        frag0_center = np.mean([a.coordinates_angstrom for a in sp0.atoms], axis=0)
        spec1_raw = np.asarray([a.coordinates_angstrom for a in sp1.atoms], dtype=float)
        spec1_center = np.mean(spec1_raw, axis=0)
        shifted1 = spec1_raw - spec1_center + (frag0_center + np.array([0.0, 0.0, 200.0]))
        for local_i, full_i in enumerate(index_groups[1]):
            full_pair[full_i] = shifted1[local_i]
        pair_eval = pes.evaluate(full_symbols, full_pair)
        evals_out[0] = replace(evals_out[0], energy_kj_mol=pair_eval.energy_kj_mol)
        evals_out[1] = replace(evals_out[1], energy_kj_mol=0.0)
        species_out[0] = replace(sp0, electronic_energy_kj_mol=pair_eval.energy_kj_mol)
        species_out[1] = replace(sp1, electronic_energy_kj_mol=0.0)

    return tuple(species_out), tuple(evals_out), index_groups


def parse_toml_data(
    data: Mapping[str, Any],
    *,
    source: str | None = None,
    pes: PotentialEnergySurface | None = None,
) -> LoadedCalculation:
    calculation = _mapping(data.get("calculation", {}), "calculation")
    name = str(calculation.get("name", "PyRate calculation"))
    default_mass_conv = str(calculation.get("mass_convention", "iupac")).lower()

    potential_raw = data.get("potential")
    pot_table = _mapping(potential_raw, "potential") if potential_raw is not None else None
    active_pes = pes
    if active_pes is None and pot_table is not None:
        base_dir = str(Path(source).parent) if source is not None else None
        active_pes = build_pes_from_mapping(pot_table, base_dir=base_dir)

    path_raw = data.get("path")
    path_mapping = _mapping(path_raw, "path") if path_raw is not None else None
    mep_raw = data.get("mep")
    path_int_raw = path_mapping.get("integration") if path_mapping is not None else None
    if mep_raw is not None and path_int_raw is not None:
        raise ValueError("specify either [mep] or [path.integration], not both")
    mep_table_raw = mep_raw if mep_raw is not None else path_int_raw
    mep_table = _mapping(mep_table_raw, "mep") if mep_table_raw is not None else None

    mep_opts: MepOptions | None = None
    if mep_table is not None:
        mep_cfg = dict(mep_table)
        if "mass_convention" not in mep_cfg and calculation.get("mass_convention") is not None:
            mep_cfg["mass_convention"] = calculation["mass_convention"]
        mep_opts = MepOptions.from_mapping(mep_cfg)

    global_opt = mep_opts.optimize_stationary_points if mep_opts is not None else False

    reactant_tables = data.get("reactants")
    if not isinstance(reactant_tables, list) or not reactant_tables:
        raise ValueError("input requires one or two [[reactants]] tables")
    ts_table = _mapping(data.get("transition_state"), "transition_state")
    product_tables = data.get("products", [])
    if not isinstance(product_tables, list):
        raise ValueError("products must use [[products]] array tables")

    ts_merged = _with_calculation_defaults(ts_table, calculation)
    mep_result: MepResult | None = None
    r_evals: tuple[PesEvaluation | None, ...] = ()
    p_evals: tuple[PesEvaluation | None, ...] = ()

    if active_pes is not None:
        ts_atoms = _species_atoms_from_mapping(ts_merged, default_mass_convention=default_mass_conv)
        if not ts_atoms:
            raise ValueError("[transition_state] requires 'atoms' when using a [potential] surface")
        ts_symbols = tuple(a.symbol for a in ts_atoms)
        ts_coords = np.asarray([a.coordinates_angstrom for a in ts_atoms], dtype=float)
        ts_masses = [float(a.mass_amu or 1.0) for a in ts_atoms]

        ts_needs_pes = (
            bool(ts_merged.get("optimize", global_opt))
            or ("frequencies_cm1" not in ts_merged)
            or ("electronic_energy_kj_mol" not in ts_merged)
        )
        ts_eval: PesEvaluation | None = None
        if ts_needs_pes:
            transition_state, ts_eval = build_species_from_pes(
                active_pes,
                str(ts_merged.get("name", "transition-state")),
                ts_symbols,
                ts_coords,
                symmetry_number=int(ts_merged.get("symmetry_number", 1)),
                electronic_states=_species_elec_states_from_mapping(ts_merged),
                charge=int(ts_merged.get("charge", 0)),
                multiplicity=int(ts_merged.get("multiplicity", 1)),
                mass_convention=default_mass_conv,
                masses_amu=ts_masses,
                is_transition_state=True,
                rotor_type=str(ts_merged.get("rotor_type", "auto")),
                frequency_scale=float(ts_merged.get("frequency_scale", 1.0)),
                frequency_floor_cm1=(
                    float(ts_merged["frequency_floor_cm1"])
                    if ts_merged.get("frequency_floor_cm1") is not None
                    else None
                ),
                optimize=bool(ts_merged.get("optimize", global_opt)),
                gtol_au=float(ts_merged.get("gtol_au", 1.0e-6)),
            )
            ts_coords = np.asarray(
                [a.coordinates_angstrom for a in transition_state.atoms], dtype=float
            )
        else:
            transition_state = Species.from_mapping(ts_merged, "transition-state")

        use_supermol = _is_supermolecule_pes(active_pes, pot_table)
        reactant_maps = [_mapping(t, f"reactants[{i}]") for i, t in enumerate(reactant_tables)]
        reactants, r_evals, r_indices = _assemble_supermolecule_group(
            active_pes,
            ts_symbols,
            ts_coords,
            reactant_maps,
            "reactant",
            calculation=calculation,
            mass_convention=default_mass_conv,
            use_supermolecule=use_supermol,
            global_optimize=global_opt,
        )
        product_maps = [_mapping(t, f"products[{i}]") for i, t in enumerate(product_tables)]
        products, p_evals, p_indices = _assemble_supermolecule_group(
            active_pes,
            ts_symbols,
            ts_coords,
            product_maps,
            "product",
            calculation=calculation,
            mass_convention=default_mass_conv,
            use_supermolecule=use_supermol,
            global_optimize=global_opt,
        )

        # Shift energies relative to reactant total when energies were evaluated from the PES
        relative_energies = bool(calculation.get("relative_energies", True))
        ref_kj_mol = 0.0
        if relative_energies and any(ev is not None for ev in (*r_evals, ts_eval, *p_evals)):
            ref_kj_mol = sum(sp.electronic_energy_kj_mol for sp in reactants)
            if all(ev is not None for ev in r_evals):
                reactants = tuple(
                    replace(sp, electronic_energy_kj_mol=0.0) for sp in reactants
                )
            elif any(ev is not None for ev in r_evals):
                shift_r = ref_kj_mol / len(reactants)
                reactants = tuple(
                    replace(
                        sp, electronic_energy_kj_mol=sp.electronic_energy_kj_mol - shift_r
                    )
                    for sp in reactants
                )
            if ts_eval is not None or any(ev is not None for ev in r_evals):
                transition_state = replace(
                    transition_state,
                    electronic_energy_kj_mol=transition_state.electronic_energy_kj_mol - ref_kj_mol,
                )
            if products:
                if len(products) == 2 and use_supermol and all(ev is not None for ev in p_evals):
                    products = (
                        replace(
                            products[0],
                            electronic_energy_kj_mol=products[0].electronic_energy_kj_mol
                            - ref_kj_mol,
                        ),
                        replace(products[1], electronic_energy_kj_mol=0.0),
                    )
                elif any(ev is not None for ev in p_evals):
                    shift_p = ref_kj_mol / len(products)
                    products = tuple(
                        replace(
                            sp,
                            electronic_energy_kj_mol=sp.electronic_energy_kj_mol - shift_p,
                        )
                        for sp in products
                    )
    else:
        reactants = tuple(
            Species.from_mapping(
                _with_calculation_defaults(_mapping(table, f"reactants[{index}]"), calculation),
                f"reactant-{index + 1}",
            )
            for index, table in enumerate(reactant_tables)
        )
        transition_state = Species.from_mapping(ts_merged, "transition-state")
        products = tuple(
            Species.from_mapping(
                _with_calculation_defaults(_mapping(table, f"products[{index}]"), calculation),
                f"product-{index + 1}",
            )
            for index, table in enumerate(product_tables)
        )
        ref_kj_mol = 0.0
        r_indices = []
        p_indices = []

    path_model: ReactionPath | None = None
    if path_mapping is not None and "points" in path_mapping:
        point_tables = path_mapping.get("points")
        if not isinstance(point_tables, list):
            raise ValueError("[path] requires one or more [[path.points]] tables")
        points = tuple(
            PathPoint.from_mapping(
                _with_calculation_defaults(
                    _mapping(table, f"path.points[{index}]"), calculation
                ),
                index,
            )
            for index, table in enumerate(point_tables)
        )
        path_model = ReactionPath(
            points=points,
            coordinate_unit=str(path_mapping.get("coordinate_unit", "angstrom")).lower(),
            reduced_mass_amu=float(path_mapping.get("reduced_mass_amu", 1.0)),
        )
    elif mep_opts is not None:
        if active_pes is None:
            raise ValueError("[mep] / [path.integration] requires a [potential] surface or pes=")
        if not transition_state.atoms:
            raise ValueError("[transition_state] requires 'atoms' for MEP integration")
        ts_syms = tuple(a.symbol for a in transition_state.atoms)
        ts_xyz = np.asarray(
            [a.coordinates_angstrom for a in transition_state.atoms], dtype=float
        )
        ts_masses_opt = [float(a.mass_amu or 1.0) for a in transition_state.atoms]
        dir_vec = None
        if mep_table is not None and mep_table.get("direction_vector_angstrom") is not None:
            dir_vec = np.asarray(mep_table["direction_vector_angstrom"], dtype=float)
        mep_result = integrate_mep(
            active_pes,
            ts_syms,
            ts_xyz,
            options=mep_opts,
            masses_amu=ts_masses_opt,
            electronic_states=transition_state.electronic_states,
            symmetry_number=transition_state.symmetry_number,
            frequency_scale=transition_state.frequency_scale,
            frequency_floor_cm1=transition_state.frequency_floor_cm1,
            charge=int(ts_merged.get("charge", 0)),
            multiplicity=int(ts_merged.get("multiplicity", 1)),
            reference_energy_kj_mol=ref_kj_mol,
            reactants=reactants,
            products=products,
            reactant_atom_indices=r_indices or None,
            product_atom_indices=p_indices or None,
            direction_vector_angstrom=dir_vec,
        )
        path_model = mep_result.reaction_path

    pressure_pa: float
    if calculation.get("standard_pressure_pa") is not None:
        pressure_pa = float(calculation["standard_pressure_pa"])
    else:
        pressure_pa = float(calculation.get("standard_pressure_bar", 1.0)) * BAR_PA

    reaction = Reaction(
        reactants=reactants,
        transition_state=transition_state,
        products=products,
        path=path_model,
        standard_pressure_pa=pressure_pa,
    )

    temperatures = calculation.get("temperatures_k")
    if not isinstance(temperatures, list):
        raise ValueError("[calculation] requires temperatures_k = [...] ")

    tunneling_table = _mapping(data.get("tunneling", {}), "tunneling")
    tunneling_method = str(
        calculation.get("tunneling", tunneling_table.get("method", "none"))
    ).lower()
    tunneling_profile: TunnelingProfile | None = None
    profile_data = tunneling_table.get("profile", tunneling_table)
    has_explicit_profile = (
        isinstance(profile_data, Mapping)
        and "coordinates_angstrom" in profile_data
        and "energies_kj_mol" in profile_data
    )
    if tunneling_method == "wkb" or has_explicit_profile:
        tunneling_profile = TunnelingProfile.from_mapping(
            _mapping(profile_data, "tunneling.profile")
        )

    options = CalculationOptions(
        temperatures_k=tuple(float(value) for value in temperatures),
        method=str(calculation.get("method", "tst")),
        tunneling=tunneling_method,
        output_rate_unit=str(calculation.get("rate_unit", "auto")),
        tunneling_profile=tunneling_profile,
    )
    return LoadedCalculation(
        name=name,
        reaction=reaction,
        options=options,
        raw=data,
        source=source,
        mep_result=mep_result,
        reactant_evaluations=r_evals,
        product_evaluations=p_evals,
    )


def load_toml(path: str | Path, *, pes: PotentialEnergySurface | None = None) -> LoadedCalculation:
    source = Path(path)
    with source.open("rb") as handle:
        data = tomllib.load(handle)
    return parse_toml_data(data, source=str(source), pes=pes)


def load_input(
    path: str | Path, *, pes: PotentialEnergySurface | None = None
) -> LoadedCalculation | LegacyInput:
    source = Path(path)
    if source.suffix.lower() == ".toml":
        return load_toml(source, pes=pes)
    return load_legacy(source)


def _fortran_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _unpack_fu40_hessian(section: LegacySection, dimension: int, *, hessform: str) -> FloatArray:
    entry = section.get("HESSIAN")
    if entry is None:
        raise ValueError(f"missing HESSIAN block in *{section.name}")
    tokens: list[str] = list(entry.arguments)
    for line in entry.body:
        tokens.extend(line.split())
    values = [_fortran_float(tok) for tok in tokens]
    matrix: FloatArray = np.zeros((dimension, dimension), dtype=float)
    if hessform == "PACKED":
        expected = dimension * (dimension + 1) // 2
        if len(values) < expected:
            raise ValueError(
                f"packed Hessian in *{section.name} requires {expected} values, got {len(values)}"
            )
        idx = 0
        for row in range(dimension):
            for col in range(row + 1):
                matrix[row, col] = matrix[col, row] = values[idx]
                idx += 1
    else:
        expected = dimension * dimension
        if len(values) < expected:
            raise ValueError(
                f"full Hessian in *{section.name} requires {expected} values, got {len(values)}"
            )
        matrix = np.asarray(values[:expected], dtype=float).reshape((dimension, dimension))
        matrix = np.asarray(0.5 * (matrix + matrix.T), dtype=float)
    return matrix


def _legacy_rate_and_tunnel(
    document: LegacyInput, *, has_path: bool
) -> tuple[str, str, float]:
    requested = set(document.requested_methods())
    rate_sec = document.section("RATE")
    if rate_sec is not None:
        if rate_sec.has("US"):
            method = "us"
        elif rate_sec.has("MUVT") or rate_sec.has("UVT"):
            method = "muvt"
        elif rate_sec.has("CUS"):
            method = "cus"
        elif rate_sec.has("CVT") and not rate_sec.has("NOCVT") and has_path:
            method = "cvt"
        else:
            method = "tst"
    elif "US" in requested:
        method = "us"
    elif "MUVT" in requested:
        method = "muvt"
    elif "CUS" in requested:
        method = "cus"
    elif "CVT" in requested and has_path:
        method = "cvt"
    else:
        method = "tst"

    tunnel_sec = document.section("TUNNEL")
    tunneling = "none"
    if tunnel_sec is not None:
        if tunnel_sec.has("NONE") or tunnel_sec.has("NOTUNNEL"):
            tunneling = "none"
        elif tunnel_sec.has("COMT") and has_path:
            tunneling = "comt"
        elif (tunnel_sec.has("MUOMT") or tunnel_sec.has("MOMT")) and has_path:
            tunneling = "muomt"
        elif tunnel_sec.has("OMT") and has_path:
            tunneling = "omt"
        elif (
            tunnel_sec.has("SCT") or tunnel_sec.has("CD-SCSAG") or tunnel_sec.has("CDSCSAG")
        ) and has_path:
            tunneling = "sct"
        elif (tunnel_sec.has("ZCT") or tunnel_sec.has("MEPSAG")) and has_path:
            tunneling = "zct"
        elif tunnel_sec.has("ECKART"):
            tunneling = "eckart"
        elif tunnel_sec.has("WIGNER"):
            tunneling = "wigner"
    elif "SCT" in requested and has_path:
        tunneling = "sct"
    elif "ZCT" in requested and has_path:
        tunneling = "zct"

    pressure_pa = 1.0 * BAR_PA
    if rate_sec is not None:
        p_entry = rate_sec.get("PRESSURE") or rate_sec.get("PSTD")
        if p_entry and p_entry.arguments:
            pressure_pa = _fortran_float(p_entry.arguments[0]) * BAR_PA
    return method, tunneling, pressure_pa


def _legacy_coord_system(document: LegacyInput, *, default: str) -> str:
    path_sec = document.section("PATH")
    sec_sec = document.section("SECOND")
    for sec in (path_sec, sec_sec):
        if sec is None:
            continue
        if sec.has("RECT") or sec.has("RECTILINEAR"):
            return "rectilinear"
        if sec.has("CURV") or sec.has("CURVILINEAR"):
            return "curvilinear"
        rpm = sec.get("RPMCOORD")
        if rpm and rpm.arguments:
            val = rpm.arguments[0].upper()
            if val in {"RECT", "RECTILINEAR", "CART", "CARTESIAN"}:
                return "rectilinear"
            if val in {"CURV", "CURVILINEAR"}:
                return "curvilinear"
    return default


def _legacy_freq_scale(sec: LegacySection | None) -> float:
    if sec is None:
        return 1.0
    entry = sec.get("FREQSCALE") or sec.get("SCALEFREQ")
    if entry and entry.arguments:
        return _fortran_float(entry.arguments[0])
    return 1.0


def _legacy_species_atoms(
    dsec: LegacySection,
    atom_map: Mapping[int, str],
    mass_overrides: Mapping[int, float],
    geom_scale: float,
) -> tuple[list[int], tuple[Atom, ...]]:
    geom_entry = dsec.get("GEOM")
    if geom_entry is None:
        raise ValueError(f"missing GEOM block in *{dsec.name}")
    poly_indices: list[int] = []
    atoms: list[Atom] = []
    for line in geom_entry.body:
        tokens = line.split()
        if len(tokens) >= 4:
            atom_idx = int(tokens[0])
            if atom_idx not in atom_map:
                raise LegacySemanticError(
                    f"atom index {atom_idx} missing from *GENERAL ATOMS",
                    section=dsec.name,
                    keyword="GEOM",
                    line_number=geom_entry.line_number,
                )
            coords = tuple(_fortran_float(tok) * geom_scale for tok in tokens[1:4])
            poly_indices.append(atom_idx)
            atoms.append(
                Atom(
                    symbol=atom_map[atom_idx],
                    x_angstrom=coords[0],
                    y_angstrom=coords[1],
                    z_angstrom=coords[2],
                    mass_amu=mass_overrides.get(atom_idx),
                    mass_convention="polyrate",
                )
            )
    return poly_indices, tuple(atoms)


def _legacy_elec_states(
    dsec: LegacySection, hartree_to_kj_mol: float
) -> tuple[ElectronicState, ...]:
    elec_states: list[ElectronicState] = []
    elec_entry = dsec.get("ELEC")
    if elec_entry is not None:
        for line in elec_entry.body:
            tokens = line.split()
            if len(tokens) >= 2:
                elec_states.append(
                    ElectronicState(
                        degeneracy=int(tokens[0]),
                        energy_kj_mol=_fortran_float(tokens[1]) * hartree_to_kj_mol,
                    )
                )
    if not elec_states:
        elec_states.append(ElectronicState())
    return tuple(elec_states)


def _legacy_mep_options(document: LegacyInput) -> MepOptions:
    path_sec = document.section("PATH")
    sec_sec = document.section("SECOND")

    method = "pm"
    step_size_bohr = 0.01
    hessian_stride = 1
    s_min_bohr = -1.5
    s_max_bohr = 1.5
    scale_mass_amu = 1.0
    start_from_saddle = True
    direction_sign = 1
    project_rigid_modes = True

    if path_sec is not None:
        if path_sec.has("ES1"):
            method = "es1"
        elif path_sec.has("ESD") or path_sec.has("EULER"):
            method = "euler"
        elif path_sec.has("PM") or path_sec.has("RPM") or path_sec.has("LQA"):
            method = "pm"

        sm_entry = path_sec.get("SCALEMASS")
        if sm_entry and sm_entry.arguments:
            scale_mass_amu = _fortran_float(sm_entry.arguments[0])

        ss_entry = path_sec.get("SSTEP")
        if ss_entry and ss_entry.arguments:
            step_size_bohr = _fortran_float(ss_entry.arguments[0])

        inh_entry = path_sec.get("INH") or path_sec.get("INP")
        if inh_entry and inh_entry.arguments:
            hessian_stride = int(_fortran_float(inh_entry.arguments[0]))

        sr_entry = path_sec.get("SRANGE")
        if sr_entry is not None:
            sr_toks = list(sr_entry.arguments)
            for line in sr_entry.body:
                sr_toks.extend(line.split())
            if len(sr_toks) >= 2:
                vals = [_fortran_float(t) for t in sr_toks[:2]]
                s_min_bohr, s_max_bohr = min(vals), max(vals)
        if path_sec.has("SMIN") and path_sec.get("SMIN").arguments:  # type: ignore[union-attr]
            s_min_bohr = _fortran_float(path_sec.get("SMIN").arguments[0])  # type: ignore[union-attr]
        if path_sec.has("SMAX") and path_sec.get("SMAX").arguments:  # type: ignore[union-attr]
            s_max_bohr = _fortran_float(path_sec.get("SMAX").arguments[0])  # type: ignore[union-attr]

        if path_sec.has("NOSADDLE"):
            start_from_saddle = False
        elif path_sec.has("SADDLE"):
            start_from_saddle = True

        sign_entry = path_sec.get("SIGN")
        if sign_entry and sign_entry.arguments:
            sval = sign_entry.arguments[0].upper()
            if sval.startswith("-"):
                direction_sign = -1
            elif sval.startswith("+") or sval == "1":
                direction_sign = 1

        if path_sec.has("NOPROJECT"):
            project_rigid_modes = False

    dx_bohr = 1.0e-4
    hdiff = "central"
    if sec_sec is not None:
        dx_entry = sec_sec.get("DX")
        if dx_entry and dx_entry.arguments:
            dx_bohr = _fortran_float(dx_entry.arguments[0])
        if sec_sec.has("FIRST") or sec_sec.has("FORWARD"):
            hdiff = "forward"
        elif sec_sec.has("CENTRAL"):
            hdiff = "central"
        if sec_sec.has("NOPROJECT"):
            project_rigid_modes = False

    coord_sys = _legacy_coord_system(document, default="curvilinear")
    return MepOptions(
        method=method,
        step_size_bohr=step_size_bohr,
        hessian_stride=hessian_stride,
        save_stride=1,
        s_min_bohr=s_min_bohr,
        s_max_bohr=s_max_bohr,
        scale_mass_amu=scale_mass_amu,
        coordinate_system=coord_sys,
        start_from_saddle=start_from_saddle,
        direction_sign=direction_sign,
        project_rigid_modes=project_rigid_modes,
        hessian_step_bohr=dx_bohr,
        hessian_difference=hdiff,
        mass_convention="polyrate",
    )


def _legacy_with_pes(
    document: LegacyInput,
    pes: PotentialEnergySurface,
    *,
    run_mep: bool = True,
) -> LoadedCalculation:
    atom_map = dict(document.atom_table)
    if not atom_map:
        raise ValueError("legacy Polyrate input is missing *GENERAL / ATOMS")
    mass_overrides = document.mass_overrides

    general = document.section("GENERAL")
    inpunit_entry = general.get("INPUNIT") if general else None
    inpunit = (
        inpunit_entry.arguments[0].upper()
        if inpunit_entry and inpunit_entry.arguments
        else "AU"
    )
    geom_scale = BOHR_ANGSTROM if inpunit in {"AU", "BOHR"} else 1.0
    hartree_to_kj_mol = HARTREE_J * AVOGADRO_MOL / 1000.0

    rate_sec = document.section("RATE")
    sigma_f = 1
    sigma_r = 1
    if rate_sec is not None:
        sigmaf_entry = rate_sec.get("SIGMAF")
        if sigmaf_entry and sigmaf_entry.arguments:
            sigma_f = max(1, int(round(_fortran_float(sigmaf_entry.arguments[0]))))
        sigmar_entry = rate_sec.get("SIGMAR")
        if sigmar_entry and sigmar_entry.arguments:
            sigma_r = max(1, int(round(_fortran_float(sigmar_entry.arguments[0]))))

    opt_sec = document.section("OPTIMIZATION")
    opt_min = bool(opt_sec and (opt_sec.has("OPTMIN") or opt_sec.has("BFGS")))
    opt_ts = bool(opt_sec and (opt_sec.has("OPTTS") or opt_sec.has("PRFO") or opt_sec.has("EF")))
    gtol_au = 1.0e-6
    if opt_sec is not None and opt_sec.has("GTOL"):
        gtol_e = opt_sec.get("GTOL")
        if gtol_e and gtol_e.arguments:
            gtol_au = _fortran_float(gtol_e.arguments[0])

    start_sec = document.section("START")
    if start_sec is None:
        raise ValueError("legacy Polyrate input requires *START section for PES execution")
    ts_poly_indices, ts_atoms = _legacy_species_atoms(
        start_sec, atom_map, mass_overrides, geom_scale
    )
    ts_symbols = tuple(a.symbol for a in ts_atoms)
    ts_coords = np.asarray([a.coordinates_angstrom for a in ts_atoms], dtype=float)
    ts_masses = [float(a.mass_amu or 1.0) for a in ts_atoms]
    ts_pos_map = {poly_i: local_i for local_i, poly_i in enumerate(ts_poly_indices)}

    mep_opts = _legacy_mep_options(document)
    ts_scale = _legacy_freq_scale(start_sec)
    ts_sp_entry = start_sec.get("SPECIES")
    ts_sp_type = (
        ts_sp_entry.arguments[0].upper()
        if ts_sp_entry and ts_sp_entry.arguments
        else "NONLINTS"
    )
    ts_rotor = "linear" if ts_sp_type == "LINTS" else "auto"

    ts, _ = build_species_from_pes(
        pes,
        "transition-state",
        ts_symbols,
        ts_coords,
        symmetry_number=1,
        electronic_states=_legacy_elec_states(start_sec, hartree_to_kj_mol),
        mass_convention="polyrate",
        masses_amu=ts_masses,
        is_transition_state=mep_opts.start_from_saddle,
        rotor_type=ts_rotor,
        frequency_scale=ts_scale,
        optimize=opt_ts,
        gtol_au=gtol_au,
        hessian_step_bohr=mep_opts.hessian_step_bohr,
        hessian_difference=mep_opts.hessian_difference,
    )
    ts_coords = np.asarray([a.coordinates_angstrom for a in ts.atoms], dtype=float)

    force_supermol = bool(
        (general is not None and general.has("SUPERMOL"))
        or isinstance(pes, (ExtendedLepsSurface, MullerBrownSurface, EckartValleySurface))
    )

    def _build_group(
        names: Sequence[str], default_sym: int
    ) -> tuple[tuple[Species, ...], tuple[PesEvaluation, ...], list[tuple[int, ...]]]:
        built_sps: list[Species] = []
        built_evals: list[PesEvaluation] = []
        built_idxs: list[tuple[int, ...]] = []
        used_frag_pes: list[bool] = []
        for idx_g, sec_name in enumerate(names):
            dsec = document.section(sec_name)
            if dsec is None:
                continue
            p_indices, atoms = _legacy_species_atoms(
                dsec, atom_map, mass_overrides, geom_scale
            )
            local_grp = tuple(ts_pos_map[pi] for pi in p_indices)
            built_idxs.append(local_grp)
            coords = np.asarray([a.coordinates_angstrom for a in atoms], dtype=float)
            syms = tuple(a.symbol for a in atoms)
            masses = [float(a.mass_amu or 1.0) for a in atoms]
            sp_entry = dsec.get("SPECIES")
            sp_type = sp_entry.arguments[0].upper() if sp_entry and sp_entry.arguments else ""
            rotor = (
                "atomic"
                if (sp_type == "ATOMIC" or len(atoms) == 1)
                else ("linear" if sp_type == "LINRP" else "auto")
            )
            use_frag = force_supermol and len(local_grp) < len(ts_symbols)
            used_frag_pes.append(use_frag)
            eval_pes: PotentialEnergySurface = (
                SupermoleculeFragmentPes(pes, ts_symbols, local_grp, ts_coords)
                if use_frag
                else pes
            )
            sp_built, ev_built = build_species_from_pes(
                eval_pes,
                sec_name.lower(),
                syms,
                coords,
                symmetry_number=default_sym if idx_g == 0 else 1,
                electronic_states=_legacy_elec_states(dsec, hartree_to_kj_mol),
                mass_convention="polyrate",
                masses_amu=masses,
                is_transition_state=False,
                rotor_type=rotor,
                frequency_scale=_legacy_freq_scale(dsec),
                optimize=opt_min and len(atoms) > 1,
                gtol_au=gtol_au,
                hessian_step_bohr=mep_opts.hessian_step_bohr,
                hessian_difference=mep_opts.hessian_difference,
            )
            built_sps.append(sp_built)
            built_evals.append(ev_built)

        if len(built_sps) == 2 and any(used_frag_pes):
            sp0, sp1 = built_sps
            full_pair = np.array(ts_coords, dtype=float, copy=True)
            for li, fi in enumerate(built_idxs[0]):
                full_pair[fi] = np.asarray(sp0.atoms[li].coordinates_angstrom, dtype=float)
            f0_ctr = np.mean([a.coordinates_angstrom for a in sp0.atoms], axis=0)
            s1_raw = np.asarray([a.coordinates_angstrom for a in sp1.atoms], dtype=float)
            s1_ctr = np.mean(s1_raw, axis=0)
            s1_shift = s1_raw - s1_ctr + (f0_ctr + np.array([0.0, 0.0, 200.0]))
            for li, fi in enumerate(built_idxs[1]):
                full_pair[fi] = s1_shift[li]
            pair_ev = pes.evaluate(ts_symbols, full_pair)
            built_evals[0] = replace(built_evals[0], energy_kj_mol=pair_ev.energy_kj_mol)
            built_evals[1] = replace(built_evals[1], energy_kj_mol=0.0)
            built_sps[0] = replace(sp0, electronic_energy_kj_mol=pair_ev.energy_kj_mol)
            built_sps[1] = replace(sp1, electronic_energy_kj_mol=0.0)
        return tuple(built_sps), tuple(built_evals), built_idxs

    reactants, r_evals, r_idxs = _build_group(("REACT1", "REACT2"), sigma_f)
    if not reactants:
        raise ValueError("legacy Polyrate input requires at least *REACT1")
    products, p_evals, p_idxs = _build_group(("PROD1", "PROD2"), sigma_r)

    ref_e = sum(sp.electronic_energy_kj_mol for sp in reactants)
    reactants = tuple(replace(sp, electronic_energy_kj_mol=0.0) for sp in reactants)
    ts = replace(ts, electronic_energy_kj_mol=ts.electronic_energy_kj_mol - ref_e)
    if products:
        if len(products) == 2 and force_supermol:
            products = (
                replace(
                    products[0],
                    electronic_energy_kj_mol=products[0].electronic_energy_kj_mol - ref_e,
                ),
                replace(products[1], electronic_energy_kj_mol=0.0),
            )
        else:
            shift_p = ref_e / len(products)
            products = tuple(
                replace(sp, electronic_energy_kj_mol=sp.electronic_energy_kj_mol - shift_p)
                for sp in products
            )

    want_path = run_mep and (
        document.section("PATH") is not None
        or (document.section("RATE") is not None and document.section("RATE").has("CVT"))  # type: ignore[union-attr]
        or (document.section("TUNNEL") is not None)
    )
    mep_result: MepResult | None = None
    path_model: ReactionPath | None = None
    if want_path:
        mep_result = integrate_mep(
            pes,
            ts_symbols,
            ts_coords,
            options=mep_opts,
            masses_amu=ts_masses,
            electronic_states=ts.electronic_states,
            symmetry_number=1,
            frequency_scale=ts.frequency_scale,
            reference_energy_kj_mol=ref_e,
            reactants=reactants,
            products=products,
            reactant_atom_indices=r_idxs or None,
            product_atom_indices=p_idxs or None,
        )
        path_model = mep_result.reaction_path

    method, tunneling, pressure_pa = _legacy_rate_and_tunnel(
        document, has_path=path_model is not None
    )
    rate_unit = "cm^3 molecule^-1 s^-1" if len(reactants) == 2 else "s^-1"
    reaction = Reaction(
        reactants=reactants,
        transition_state=ts,
        products=products,
        path=path_model,
        standard_pressure_pa=pressure_pa,
    )
    options = CalculationOptions(
        temperatures_k=document.temperatures_k,
        method=method,
        tunneling=tunneling,
        output_rate_unit=rate_unit,
    )
    return LoadedCalculation(
        name=document.title or "Polyrate PES calculation",
        reaction=reaction,
        options=options,
        raw=document.as_dict(),
        source=document.source,
        mep_result=mep_result,
        reactant_evaluations=r_evals,
        product_evaluations=p_evals,
    )


def legacy_to_calculation(
    document: LegacyInput,
    *,
    fu40_document: LegacyInput | None = None,
    fu40_path: str | Path | None = None,
    fu40_text: str | None = None,
    pes: PotentialEnergySurface | None = None,
    run_mep: bool = True,
) -> LoadedCalculation:
    """Translate a legacy Polyrate ``.dat`` file and ``.fu40`` or ``pes`` to PyRate models."""

    validate_legacy_semantics(document)

    active_pes = pes
    if active_pes is None:
        ener = document.section("ENERGETICS")
        pot_e = ener.get("POTENTIAL") if ener else None
        if pot_e and pot_e.arguments:
            pot_val = pot_e.arguments[0].upper()
            if pot_val.startswith("BUILTIN:"):
                active_pes = get_builtin_pes(pot_e.arguments[0].split(":", 1)[1])

    if active_pes is not None and fu40_document is None and fu40_path is None and fu40_text is None:
        return _legacy_with_pes(document, active_pes, run_mep=run_mep)

    if fu40_document is None:
        if fu40_text is not None:
            fu40_document = parse_legacy_text(fu40_text, source="inline-fu40")
        elif fu40_path is not None:
            fu40_document = load_legacy(fu40_path)
        elif document.source is not None:
            candidate = Path(document.source).with_suffix(".fu40")
            if candidate.is_file():
                fu40_document = load_legacy(candidate)
    if fu40_document is None:
        if active_pes is not None:
            return _legacy_with_pes(document, active_pes, run_mep=run_mep)
        raise ValueError(
            "legacy Polyrate input requires a companion .fu40 electronic-structure file "
            "or a PES / [potential] backend"
        )
    validate_legacy_semantics(fu40_document)

    atom_map = dict(document.atom_table)
    if not atom_map:
        raise ValueError("legacy Polyrate input is missing *GENERAL / ATOMS")
    mass_overrides = document.mass_overrides
    ordered_atom_indices = sorted(atom_map)
    ordered_raw_symbols = [atom_map[idx] for idx in ordered_atom_indices]
    ordered_masses = [mass_overrides.get(idx) for idx in ordered_atom_indices]

    general = document.section("GENERAL")
    inpunit_entry = general.get("INPUNIT") if general else None
    inpunit = (
        inpunit_entry.arguments[0].upper()
        if inpunit_entry and inpunit_entry.arguments
        else "AU"
    )
    geom_scale = BOHR_ANGSTROM if inpunit in {"AU", "BOHR"} else 1.0
    hartree_to_kj_mol = HARTREE_J * AVOGADRO_MOL / 1000.0
    au_to_cm1 = HARTREE_J / (PLANCK_J_S * LIGHT_SPEED_M_S * 100.0)

    gen40 = fu40_document.section("GEN40")
    freqsource_entry = gen40.get("FREQSOURCE") if gen40 else None
    freqsource = (
        freqsource_entry.arguments[0].upper()
        if freqsource_entry and freqsource_entry.arguments
        else "READ"
    )
    frequnit_entry = gen40.get("FREQUNIT") if gen40 else None
    frequnit = (
        frequnit_entry.arguments[0].upper()
        if frequnit_entry and frequnit_entry.arguments
        else "WAVEN"
    )
    hessform_entry = gen40.get("HESSFORM") if gen40 else None
    hessform = (
        hessform_entry.arguments[0].upper()
        if hessform_entry and hessform_entry.arguments
        else "PACKED"
    )
    geomunit40_entry = gen40.get("GEOMUNIT") if gen40 else None
    geomunit40 = (
        geomunit40_entry.arguments[0].upper()
        if geomunit40_entry and geomunit40_entry.arguments
        else "BOHR"
    )
    geom40_scale = BOHR_ANGSTROM if geomunit40 == "BOHR" else 1.0
    freq_unit_scale = au_to_cm1 if frequnit == "AU" else 1.0

    rate_sec = document.section("RATE")
    sigma_f = 1
    sigma_r = 1
    if rate_sec is not None:
        sigmaf_entry = rate_sec.get("SIGMAF")
        if sigmaf_entry and sigmaf_entry.arguments:
            sigma_f = max(1, int(round(_fortran_float(sigmaf_entry.arguments[0]))))
        sigmar_entry = rate_sec.get("SIGMAR")
        if sigmar_entry and sigmar_entry.arguments:
            sigma_r = max(1, int(round(_fortran_float(sigmar_entry.arguments[0]))))

    def _build_stationary_species(
        dat_name: str,
        fu40_name: str,
        *,
        symmetry_number: int = 1,
        is_ts: bool = False,
    ) -> Species | None:
        dsec = document.section(dat_name)
        fsec = fu40_document.section(fu40_name)
        if dsec is None or fsec is None:
            return None
        _, atoms = _legacy_species_atoms(dsec, atom_map, mass_overrides, geom_scale)
        f_geom = fsec.get("GEOM")
        if f_geom is not None and f_geom.body:
            rows = [line.split() for line in f_geom.body if line.split()]
            if len(rows) == len(atoms):
                updated_atoms: list[Atom] = []
                for atom, toks in zip(atoms, rows, strict=True):
                    if len(toks) >= 4 and toks[0].lstrip("+-").isdigit():
                        coord_toks = toks[1:4]
                    else:
                        coord_toks = toks[:3]
                    if len(coord_toks) >= 3:
                        updated_atoms.append(
                            Atom(
                                symbol=atom.symbol,
                                x_angstrom=_fortran_float(coord_toks[0]) * geom40_scale,
                                y_angstrom=_fortran_float(coord_toks[1]) * geom40_scale,
                                z_angstrom=_fortran_float(coord_toks[2]) * geom40_scale,
                                mass_amu=atom.mass_amu,
                                label=atom.label,
                                mass_convention=atom.mass_convention,
                            )
                        )
                if len(updated_atoms) == len(atoms):
                    atoms = tuple(updated_atoms)
        v_entry = fsec.get("VVALUE") or fsec.get("VMEP")
        v_kj_mol = (
            _fortran_float(v_entry.arguments[0]) * hartree_to_kj_mol
            if v_entry and v_entry.arguments
            else 0.0
        )
        sp_entry = dsec.get("SPECIES")
        sp_type = (
            sp_entry.arguments[0].upper()
            if sp_entry and sp_entry.arguments
            else ("ATOMIC" if len(atoms) == 1 else "NONLINRP")
        )
        if sp_type == "ATOMIC" or len(atoms) == 1:
            rotor_type = "atomic"
        elif sp_type in {"LINRP", "LINTS"}:
            rotor_type = "linear"
        else:
            rotor_type = "nonlinear"

        freqs_cm1: tuple[float, ...]
        if len(atoms) == 1:
            freqs_cm1 = ()
        elif freqsource == "READ" or fsec.get("FREQ") is not None:
            freq_entry = fsec.get("FREQ")
            raw_tokens: list[str] = []
            if freq_entry is not None:
                raw_tokens.extend(freq_entry.arguments)
                for line in freq_entry.body:
                    raw_tokens.extend(line.split())
            real_freqs = [_fortran_float(tok) * freq_unit_scale for tok in raw_tokens]
            wstar_entry = fsec.get("WSTAR")
            if is_ts and wstar_entry and wstar_entry.arguments:
                real_freqs.append(-abs(_fortran_float(wstar_entry.arguments[0]) * freq_unit_scale))
            freqs_cm1 = tuple(real_freqs)
        else:
            dim = 3 * len(atoms)
            hessian = _unpack_fu40_hessian(fsec, dim, hessform=hessform)
            masses = [float(atom.mass_amu or 1.0) for atom in atoms]
            nm = normal_modes(
                hessian, masses, remove_rigid_modes=False, zero_threshold_cm1=0.0
            )
            n_vib = (
                3 * len(atoms) - 5 if rotor_type == "linear" else 3 * len(atoms) - 6
            ) - (1 if is_ts else 0)
            all_freqs = nm.frequencies_cm1
            if is_ts:
                min_idx = int(np.argmin(all_freqs))
                imag_freq = float(all_freqs[min_idx])
                rem_freqs = np.delete(all_freqs, min_idx)
            else:
                imag_freq = None
                rem_freqs = all_freqs
            order = np.argsort(np.abs(rem_freqs))
            vib_freqs = np.sort(rem_freqs[order][-n_vib:])[::-1]
            freqs_list = [float(val) for val in vib_freqs]
            if imag_freq is not None:
                freqs_list.append(imag_freq)
            freqs_cm1 = tuple(freqs_list)

        elec_states = _legacy_elec_states(dsec, hartree_to_kj_mol)
        return Species(
            name=dat_name.lower(),
            atoms=atoms,
            electronic_energy_kj_mol=v_kj_mol,
            frequencies_cm1=freqs_cm1,
            symmetry_number=symmetry_number,
            rotor_type=rotor_type,
            electronic_states=elec_states,
            frequency_scale=_legacy_freq_scale(dsec),
        )

    r1 = _build_stationary_species("REACT1", "R140", symmetry_number=sigma_f)
    r2 = _build_stationary_species("REACT2", "R240", symmetry_number=1)
    reactants = tuple(sp for sp in (r1, r2) if sp is not None)
    if not reactants:
        raise ValueError("legacy Polyrate input requires at least *REACT1 / *R140")

    p1 = _build_stationary_species("PROD1", "P140", symmetry_number=sigma_r)
    p2 = _build_stationary_species("PROD2", "P240", symmetry_number=1)
    products = tuple(sp for sp in (p1, p2) if sp is not None)

    ts = _build_stationary_species("START", "SADDLE40", symmetry_number=1, is_ts=True)
    if ts is None:
        raise ValueError("legacy Polyrate input requires *START / *SADDLE40")

    path_model: ReactionPath | None = None
    point_sections = fu40_document.section_all("POINT40")
    if point_sections:
        path_sec = document.section("PATH")
        scalemass_entry = path_sec.get("SCALEMASS") if path_sec else None
        reduced_mass_amu = (
            _fortran_float(scalemass_entry.arguments[0])
            if scalemass_entry and scalemass_entry.arguments
            else 1.0
        )
        # Preserve rectilinear projection by default for standalone .fu40 compatibility
        # (matching Polyrate ch5fu40 benchmarks), unless CURV/CURVILINEAR is requested.
        fu40_coord_sys = _legacy_coord_system(document, default="rectilinear")
        n_total_atoms = len(ordered_raw_symbols)
        dim_total = 3 * n_total_atoms
        n_vib_path = (
            3 * n_total_atoms - 6 if ts.rotor_type == "linear" else 3 * n_total_atoms - 7
        )
        raw_points: list[
            tuple[float, Species, FloatArray | None, FloatArray | None, tuple[float, ...]]
        ] = []
        for idx, pt_sec in enumerate(point_sections):
            s_entry = pt_sec.get("SMEP") or pt_sec.get("SVALUE")
            v_entry = pt_sec.get("VMEP") or pt_sec.get("VVALUE")
            if s_entry is None or v_entry is None:
                continue
            s_bohr = _fortran_float(s_entry.arguments[0])
            v_kj = _fortran_float(v_entry.arguments[0]) * hartree_to_kj_mol
            geom_entry = pt_sec.get("GEOM")
            if geom_entry is not None:
                coord_rows = [
                    [_fortran_float(tok) * geom40_scale for tok in line.split()[:3]]
                    for line in geom_entry.body
                    if line.split()
                ]
                pt_atoms = tuple(
                    Atom(
                        symbol=ordered_raw_symbols[i],
                        x_angstrom=coord_rows[i][0],
                        y_angstrom=coord_rows[i][1],
                        z_angstrom=coord_rows[i][2],
                        mass_amu=ordered_masses[i],
                        mass_convention="polyrate",
                    )
                    for i in range(n_total_atoms)
                )
            else:
                pt_atoms = ts.atoms

            bfs_entry = pt_sec.get("BFS")
            explicit_bfs: tuple[float, ...] = ()
            if bfs_entry is not None:
                bfs_tokens = list(bfs_entry.arguments)
                for line in bfs_entry.body:
                    bfs_tokens.extend(line.split())
                explicit_bfs = tuple(_fortran_float(tok) for tok in bfs_tokens)

            if freqsource == "READ" or pt_sec.get("FREQ") is not None:
                freq_entry = pt_sec.get("FREQ")
                ftoks: list[str] = []
                if freq_entry is not None:
                    ftoks.extend(freq_entry.arguments)
                    for line in freq_entry.body:
                        ftoks.extend(line.split())
                pt_freqs = tuple(_fortran_float(tok) * freq_unit_scale for tok in ftoks)
                y_vec: FloatArray | None = None
                cof_mat: FloatArray | None = None
            else:
                hessian = _unpack_fu40_hessian(pt_sec, dim_total, hessform=hessform)
                masses_amu = np.asarray(
                    [float(atom.mass_amu or 1.0) for atom in pt_atoms], dtype=float
                )
                grad_entry = pt_sec.get("GRADS") or pt_sec.get("FORCES")
                grads_au: FloatArray | None = None
                if grad_entry is not None:
                    gtoks = list(grad_entry.arguments)
                    for line in grad_entry.body:
                        gtoks.extend(line.split())
                    grad_sign = -1.0 if grad_entry.keyword == "FORCES" else 1.0
                    grads_au = grad_sign * np.asarray(
                        [_fortran_float(tok) for tok in gtoks[:dim_total]], dtype=float
                    )
                coords_ang = np.asarray(
                    [atom.coordinates_angstrom for atom in pt_atoms], dtype=float
                )
                if fu40_coord_sys == "curvilinear" and grads_au is not None:
                    hessian = curvilinear_corrected_hessian(
                        coords_ang / BOHR_ANGSTROM,
                        grads_au,
                        hessian,
                        masses_amu,
                    )

                masses_kg = masses_amu * AMU_KG
                coord_masses_amu = np.repeat(masses_amu, 3)
                coord_masses_kg = np.repeat(masses_kg, 3)
                mw_hessian = (hessian * HARTREE_J / BOHR_M**2) / np.sqrt(
                    coord_masses_kg[:, None] * coord_masses_kg[None, :]
                )
                coords_m = coords_ang * ANGSTROM_M
                basis, _ = _rigid_basis(masses_kg, coords_m)
                y_vec = None
                if grads_au is not None:
                    dx = grads_au / np.sqrt(coord_masses_amu / reduced_mass_amu)
                    dx_norm = float(np.linalg.norm(dx))
                    if dx_norm > 0.0:
                        dx_unit = dx / dx_norm
                        y_vec = dx_unit if s_bohr <= 0.0 else -dx_unit
                        mw_grad = dx_unit - basis @ (basis.T @ dx_unit)
                        mw_grad_norm = float(np.linalg.norm(mw_grad))
                        if mw_grad_norm > 1.0e-12:
                            basis = np.column_stack([basis, mw_grad / mw_grad_norm])
                projector = np.eye(dim_total) - basis @ basis.T
                raw_proj = projector.T @ mw_hessian @ projector
                proj_h = 0.5 * (raw_proj + raw_proj.T)
                evals, evecs = np.linalg.eigh(proj_h)
                all_f = (
                    np.sign(evals)
                    * np.sqrt(np.abs(evals))
                    / (2.0 * np.pi * LIGHT_SPEED_M_S * 100.0)
                )
                order = np.argsort(np.abs(all_f))
                top_idx = order[-n_vib_path:]
                desc_idx = top_idx[np.argsort(all_f[top_idx])[::-1]]
                pt_freqs = tuple(float(val) for val in all_f[desc_idx])
                cof_mat = evecs[:, desc_idx]

            pt_species = Species(
                name=f"point40-{idx + 1}",
                atoms=pt_atoms,
                electronic_energy_kj_mol=v_kj,
                frequencies_cm1=pt_freqs,
                symmetry_number=1,
                rotor_type=ts.rotor_type,
                electronic_states=ts.electronic_states,
                frequency_scale=ts.frequency_scale,
            )
            raw_points.append((s_bohr, pt_species, y_vec, cof_mat, explicit_bfs))

        raw_points.sort(key=lambda item: item[0])
        have_rphb = all(item[2] is not None and item[3] is not None for item in raw_points)
        dyds: FloatArray | None = None
        if have_rphb and len(raw_points) >= 3:
            s_arr = np.asarray([item[0] for item in raw_points], dtype=float)
            y_arr = np.asarray([item[2] for item in raw_points], dtype=float)
            dyds = np.gradient(y_arr, s_arr, axis=0)

        path_points: list[PathPoint] = []
        for idx, (s_bohr, pt_species, _, cof_mat, explicit_bfs) in enumerate(raw_points):
            if explicit_bfs:
                curvatures = explicit_bfs
            elif dyds is not None and cof_mat is not None:
                der = dyds[idx]
                curvatures = tuple(
                    float(-np.dot(der, cof_mat[:, mode_idx]))
                    for mode_idx in range(cof_mat.shape[1])
                )
            else:
                curvatures = ()
            path_points.append(
                PathPoint(
                    coordinate=s_bohr,
                    dividing_surface=pt_species,
                    curvatures_bohr_inv=curvatures,
                )
            )

        if not any(abs(pt.coordinate) <= 1.0e-12 for pt in path_points):
            path_points.append(
                PathPoint(
                    coordinate=0.0,
                    dividing_surface=ts,
                    curvatures_bohr_inv=tuple(0.0 for _ in range(n_vib_path)),
                )
            )
            path_points.sort(key=lambda pt: pt.coordinate)
            idx0 = next(i for i, pt in enumerate(path_points) if pt.coordinate == 0.0)
            if 0 < idx0 < len(path_points) - 1:
                left_b = path_points[idx0 - 1].curvatures_bohr_inv
                right_b = path_points[idx0 + 1].curvatures_bohr_inv
                if left_b and len(left_b) == len(right_b):
                    interp_b = tuple(
                        0.5 * (abs(left_b[m]) + abs(right_b[m]))
                        for m in range(len(left_b))
                    )
                    path_points[idx0] = PathPoint(
                        coordinate=0.0,
                        dividing_surface=ts,
                        curvatures_bohr_inv=interp_b,
                    )

        path_model = ReactionPath(
            points=tuple(path_points),
            coordinate_unit="bohr",
            reduced_mass_amu=reduced_mass_amu,
        )

    method, tunneling, pressure_pa = _legacy_rate_and_tunnel(
        document, has_path=path_model is not None
    )
    rate_unit = "cm^3 molecule^-1 s^-1" if len(reactants) == 2 else "s^-1"
    reaction = Reaction(
        reactants=reactants,
        transition_state=ts,
        products=products,
        path=path_model,
        standard_pressure_pa=pressure_pa,
    )
    options = CalculationOptions(
        temperatures_k=document.temperatures_k,
        method=method,
        tunneling=tunneling,
        output_rate_unit=rate_unit,
    )
    return LoadedCalculation(
        name=document.title or "Polyrate unit40 calculation",
        reaction=reaction,
        options=options,
        raw=document.as_dict(),
        source=document.source,
    )
