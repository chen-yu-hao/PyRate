"""Modern TOML input loading and validation."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

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
from .legacy import LegacyInput, LegacySection, load_legacy, parse_legacy_text
from .models import (
    Atom,
    CalculationOptions,
    ElectronicState,
    PathPoint,
    Reaction,
    ReactionPath,
    Species,
    TunnelingProfile,
)
from .normal_modes import _rigid_basis, normal_modes


@dataclass(frozen=True, slots=True)
class LoadedCalculation:
    name: str
    reaction: Reaction
    options: CalculationOptions
    raw: Mapping[str, Any]
    source: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
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


def parse_toml_data(
    data: Mapping[str, Any], *, source: str | None = None
) -> LoadedCalculation:
    calculation = _mapping(data.get("calculation", {}), "calculation")
    name = str(calculation.get("name", "PyRate calculation"))

    reactant_tables = data.get("reactants")
    if not isinstance(reactant_tables, list) or not reactant_tables:
        raise ValueError("input requires one or two [[reactants]] tables")
    reactants = tuple(
        Species.from_mapping(
            _with_calculation_defaults(_mapping(table, f"reactants[{index}]"), calculation),
            f"reactant-{index + 1}",
        )
        for index, table in enumerate(reactant_tables)
    )

    transition_table = _mapping(data.get("transition_state"), "transition_state")
    transition_state = Species.from_mapping(
        _with_calculation_defaults(transition_table, calculation), "transition-state"
    )

    product_tables = data.get("products", [])
    if not isinstance(product_tables, list):
        raise ValueError("products must use [[products]] array tables")
    products = tuple(
        Species.from_mapping(
            _with_calculation_defaults(_mapping(table, f"products[{index}]"), calculation),
            f"product-{index + 1}",
        )
        for index, table in enumerate(product_tables)
    )

    path_model: ReactionPath | None = None
    path_table = data.get("path")
    if path_table is not None:
        path_mapping = _mapping(path_table, "path")
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
        name=name, reaction=reaction, options=options, raw=data, source=source
    )


def load_toml(path: str | Path) -> LoadedCalculation:
    source = Path(path)
    with source.open("rb") as handle:
        data = tomllib.load(handle)
    return parse_toml_data(data, source=str(source))


def load_input(path: str | Path) -> LoadedCalculation | LegacyInput:
    source = Path(path)
    if source.suffix.lower() == ".toml":
        return load_toml(source)
    return load_legacy(source)


def _fortran_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _unpack_fu40_hessian(section: LegacySection, dimension: int, *, hessform: str) -> np.ndarray:
    entry = section.get("HESSIAN")
    if entry is None:
        raise ValueError(f"missing HESSIAN block in *{section.name}")
    tokens: list[str] = list(entry.arguments)
    for line in entry.body:
        tokens.extend(line.split())
    values = [_fortran_float(tok) for tok in tokens]
    matrix = np.zeros((dimension, dimension), dtype=float)
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
        matrix = 0.5 * (matrix + matrix.T)
    return matrix


def legacy_to_calculation(
    document: LegacyInput,
    *,
    fu40_document: LegacyInput | None = None,
    fu40_path: str | Path | None = None,
    fu40_text: str | None = None,
) -> LoadedCalculation:
    """Translate a Polyrate ``.dat`` document plus companion ``.fu40`` data into a ``LoadedCalculation``."""

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
        raise ValueError(
            "legacy Polyrate input requires a companion .fu40 electronic-structure file "
            "or conversion to a PyRate TOML file"
        )

    atom_map = dict(document.atom_table)
    if not atom_map:
        raise ValueError("legacy Polyrate input is missing *GENERAL / ATOMS")
    ordered_atom_indices = sorted(atom_map)
    ordered_elements = [atom_map[idx] for idx in ordered_atom_indices]

    general = document.section("GENERAL")
    inpunit_entry = general.get("INPUNIT") if general else None
    inpunit = (
        inpunit_entry.arguments[0].upper()
        if inpunit_entry and inpunit_entry.arguments
        else "AU"
    )
    geom_scale = BOHR_ANGSTROM if inpunit == "AU" else 1.0
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
        geom_entry = dsec.get("GEOM")
        if geom_entry is None:
            raise ValueError(f"missing GEOM block in *{dat_name}")
        atoms: list[Atom] = []
        for line in geom_entry.body:
            tokens = line.split()
            if len(tokens) >= 4:
                atom_idx = int(tokens[0])
                coords = tuple(_fortran_float(tok) * geom_scale for tok in tokens[1:4])
                atoms.append(
                    Atom(
                        symbol=atom_map[atom_idx],
                        x_angstrom=coords[0],
                        y_angstrom=coords[1],
                        z_angstrom=coords[2],
                        mass_convention="polyrate",
                    )
                )
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
            masses = [atom.mass_amu for atom in atoms]
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

        return Species(
            name=dat_name.lower(),
            atoms=tuple(atoms),
            electronic_energy_kj_mol=v_kj_mol,
            frequencies_cm1=freqs_cm1,
            symmetry_number=symmetry_number,
            rotor_type=rotor_type,
            electronic_states=tuple(elec_states),
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
        n_total_atoms = len(ordered_elements)
        dim_total = 3 * n_total_atoms
        n_vib_path = (
            3 * n_total_atoms - 6 if ts.rotor_type == "linear" else 3 * n_total_atoms - 7
        )
        raw_points: list[tuple[float, Species, np.ndarray | None, np.ndarray | None, tuple[float, ...]]] = []
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
                        symbol=ordered_elements[i],
                        x_angstrom=coord_rows[i][0],
                        y_angstrom=coord_rows[i][1],
                        z_angstrom=coord_rows[i][2],
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
                y_vec = None
                cof_mat = None
            else:
                hessian = _unpack_fu40_hessian(pt_sec, dim_total, hessform=hessform)
                masses_amu = np.asarray([atom.mass_amu for atom in pt_atoms], dtype=float)
                masses_kg = masses_amu * AMU_KG
                coord_masses_amu = np.repeat(masses_amu, 3)
                coord_masses_kg = np.repeat(masses_kg, 3)
                mw_hessian = (hessian * HARTREE_J / BOHR_M**2) / np.sqrt(
                    coord_masses_kg[:, None] * coord_masses_kg[None, :]
                )
                coords_m = (
                    np.asarray(
                        [
                            (atom.x_angstrom, atom.y_angstrom, atom.z_angstrom)
                            for atom in pt_atoms
                        ],
                        dtype=float,
                    )
                    * ANGSTROM_M
                )
                basis, _ = _rigid_basis(masses_kg, coords_m)
                grad_entry = pt_sec.get("GRADS") or pt_sec.get("FORCES")
                y_vec = None
                if grad_entry is not None:
                    gtoks = list(grad_entry.arguments)
                    for line in grad_entry.body:
                        gtoks.extend(line.split())
                    grad_sign = -1.0 if grad_entry.keyword == "FORCES" else 1.0
                    grads = grad_sign * np.asarray(
                        [_fortran_float(tok) for tok in gtoks[:dim_total]], dtype=float
                    )
                    dx = grads / np.sqrt(coord_masses_amu / reduced_mass_amu)
                    dx_norm = float(np.linalg.norm(dx))
                    if dx_norm > 0.0:
                        dx_unit = dx / dx_norm
                        y_vec = dx_unit if s_bohr <= 0.0 else -dx_unit
                        mw_grad = dx_unit - basis @ (basis.T @ dx_unit)
                        mw_grad_norm = float(np.linalg.norm(mw_grad))
                        if mw_grad_norm > 1.0e-12:
                            basis = np.column_stack([basis, mw_grad / mw_grad_norm])
                projector = np.eye(dim_total) - basis @ basis.T
                proj_h = 0.5 * (projector.T @ mw_hessian @ projector + (projector.T @ mw_hessian @ projector).T)
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
            )
            raw_points.append((s_bohr, pt_species, y_vec, cof_mat, explicit_bfs))

        raw_points.sort(key=lambda item: item[0])
        have_rphb = all(item[2] is not None and item[3] is not None for item in raw_points)
        dyds: np.ndarray | None = None
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

    requested = set(document.requested_methods())
    if "US" in requested:
        method = "us"
    elif "MUVT" in requested:
        method = "muvt"
    elif "CUS" in requested:
        method = "cus"
    elif "CVT" in requested and path_model is not None:
        method = "cvt"
    else:
        method = "tst"

    if "SCT" in requested and path_model is not None:
        tunneling = "sct"
    elif "ZCT" in requested and path_model is not None:
        tunneling = "zct"
    else:
        tunneling = "none"

    rate_unit = (
        "cm^3 molecule^-1 s^-1" if len(reactants) == 2 else "s^-1"
    )
    reaction = Reaction(
        reactants=reactants,
        transition_state=ts,
        products=products,
        path=path_model,
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

