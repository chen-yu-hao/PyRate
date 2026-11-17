from __future__ import annotations

import json
import math
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

from pyrate.calculation import calculate_rates
from pyrate.constants import BOHR_ANGSTROM, HARTREE_KJ_MOL
from pyrate.direct_dynamics import (
    DirectDynamicsError,
    ExternalCommandPes,
    GamessProvider,
    GaussianProvider,
    RestartCache,
    direct_dynamics_from_mapping,
    parse_gamess_output,
    parse_gaussian_fchk,
    parse_gaussian_log,
)
from pyrate.mep import integrate_mep
from pyrate.models import Atom, CalculationOptions, MepOptions, Species
from pyrate.pes import EckartValleySurface, optimize_saddle


def _write_external_driver(script_path: Path, count_path: Path) -> None:
    code = f"""
import json
import os
import sys
from pathlib import Path
import numpy as np
from pyrate.pes import EckartValleySurface

count_file = Path({str(count_path)!r})
prev = int(count_file.read_text()) if count_file.exists() else 0
count_file.write_text(str(prev + 1))

if "--fail" in sys.argv:
    sys.stderr.write("intentional QM failure\\n")
    sys.exit(7)
if "--sleep" in sys.argv:
    import time
    time.sleep(2.0)

mode = "stdout" if "--stdout" in sys.argv else "file"
req_arg = (
    sys.argv[1]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
    else os.environ["PYRATE_DD_INPUT"]
)
req_file = Path(req_arg)
req = json.loads(req_file.read_text(encoding="utf-8"))
pes = EckartValleySurface(v1_kj_mol=40.0, v2_kj_mol=55.0, coupling_angstrom=0.06)
syms = tuple(req["symbols"])
coords = np.asarray(req["coords_angstrom"], dtype=float)
ev = pes.evaluate(
    syms,
    coords,
    need_gradient=bool(req["need_gradient"] or req["need_hessian"]),
    need_hessian=bool(req["need_hessian"]),
)
payload = {{"energy_hartree": ev.energy_hartree}}
if req["need_gradient"] and ev.gradient_hartree_bohr is not None:
    payload["gradient_hartree_bohr"] = ev.gradient_hartree_bohr.tolist()
if req["need_hessian"] and ev.hessian_hartree_bohr2 is not None:
    # Emit packed lower-triangle Hessian when --packed is passed
    h = ev.hessian_hartree_bohr2
    if "--packed" in sys.argv:
        dim = h.shape[0]
        payload["hessian_hartree_bohr2"] = [
            float(h[r, c]) for r in range(dim) for c in range(r + 1)
        ]
    else:
        payload["hessian_hartree_bohr2"] = h.tolist()
if mode == "stdout":
    sys.stdout.write(json.dumps(payload))
else:
    out_arg = (
        sys.argv[2]
        if len(sys.argv) > 2 and not sys.argv[2].startswith("--")
        else os.environ["PYRATE_DD_OUTPUT"]
    )
    out_file = Path(out_arg)
    out_file.write_text(json.dumps(payload), encoding="utf-8")
"""
    script_path.write_text(code.strip() + "\n", encoding="utf-8")


def test_external_command_pes_end_to_end_mep_and_restart_cache(tmp_path: Path) -> None:
    driver = tmp_path / "qm_driver.py"
    counter = tmp_path / "invocations.txt"
    restart = tmp_path / "dd_restart.json"
    _write_external_driver(driver, counter)

    ref_pes = EckartValleySurface(v1_kj_mol=40.0, v2_kj_mol=55.0, coupling_angstrom=0.06)
    guess_coords = np.array([[0.0, 0.0, -1.2], [0.0, 0.0, 0.0], [0.0, 0.0, 1.2]], dtype=float)
    ts_coords = optimize_saddle(ref_pes, ("H", "C", "O"), guess_coords).coords_angstrom

    pes_dd = ExternalCommandPes(
        command=[sys.executable, str(driver), "{input}", "{output}", "--packed"],
        restart_file=restart,
        rst_tol_bohr=1.0e-8,
    )
    r_sp = Species(
        name="R",
        atoms=(Atom("H", 0.0, 0.0, -1.2), Atom("C", 0.0, 0.0, 0.0), Atom("O", 0.0, 0.0, 1.2)),
        electronic_energy_kj_mol=0.0,
        frequencies_cm1=(1000.0, 500.0, 500.0),
    )
    opts = MepOptions(method="pm", step_size_bohr=0.08, s_min_bohr=-0.4, s_max_bohr=0.4)

    mep_dd = integrate_mep(pes_dd, ("H", "C", "O"), ts_coords, options=opts, reactants=(r_sp,))
    rxn_dd = mep_dd.build_reaction(reactants=(r_sp,))
    res_dd = calculate_rates(
        rxn_dd, CalculationOptions(temperatures_k=(300.0,), method="cvt", tunneling="sct")
    )

    mep_ref = integrate_mep(ref_pes, ("H", "C", "O"), ts_coords, options=opts, reactants=(r_sp,))
    res_ref = calculate_rates(
        mep_ref.build_reaction(reactants=(r_sp,)),
        CalculationOptions(temperatures_k=(300.0,), method="cvt", tunneling="sct"),
    )
    assert math.isclose(res_dd.rows[0].k_final, res_ref.rows[0].k_final, rel_tol=1.0e-5)

    first_calls = int(counter.read_text())
    assert first_calls > 5
    assert restart.is_file()

    # Re-run with fresh ExternalCommandPes reading same restart file -> 0 external calls!
    pes_reloaded = ExternalCommandPes(
        command=[sys.executable, str(driver), "{input}", "{output}", "--packed"],
        restart_file=restart,
        rst_tol_bohr=1.0e-8,
    )
    _ = integrate_mep(pes_reloaded, ("H", "C", "O"), ts_coords, options=opts, reactants=(r_sp,))
    assert int(counter.read_text()) == first_calls
    assert pes_reloaded.cache.misses == 0
    assert pes_reloaded.cache.hits >= len(mep_dd.points)

    # Test JSON-on-stdout fallback
    pes_stdout = ExternalCommandPes(command=[sys.executable, str(driver), "--stdout"])
    ev_out = pes_stdout.evaluate(("H", "C", "O"), ts_coords, need_gradient=True)
    assert ev_out.gradient_kj_mol_angstrom is not None

    # Test non-zero exit & timeout errors
    bad_pes = ExternalCommandPes(command=[sys.executable, str(driver), "--fail"])
    with pytest.raises(DirectDynamicsError, match="exited with status 7") as exc_info:
        bad_pes.evaluate(("H", "C", "O"), ts_coords)
    assert exc_info.value.returncode == 7

    slow_pes = ExternalCommandPes(command=[sys.executable, str(driver), "--sleep"], timeout_s=0.1)
    with pytest.raises(DirectDynamicsError, match="timed out"):
        slow_pes.evaluate(("H", "C", "O"), ts_coords)


def test_restart_cache_tolerance_upgrade_and_scale_mass_conflict(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    cache = RestartCache(cache_path, rst_tol_bohr=1.0e-4, scale_mass_amu=1.5)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    ev_e = ExternalCommandPes(command=[sys.executable, "-c", "pass"])  # dummy type check avoidance
    del ev_e
    from pyrate.pes import PesEvaluation

    cache.store(
        ("H", "H"),
        coords,
        PesEvaluation(energy_kj_mol=12.5, gradient_kj_mol_angstrom=np.ones((2, 3))),
        charge=0,
        multiplicity=1,
    )
    # Within tolerance -> hit for gradient, miss for Hessian
    near_coords = coords.copy()
    near_coords[1, 2] += 1.0e-5 * BOHR_ANGSTROM
    hit = cache.lookup(("H", "H"), near_coords, need_gradient=True, need_hessian=False)
    assert hit is not None and math.isclose(hit.energy_kj_mol, 12.5)
    assert cache.lookup(("H", "H"), near_coords, need_gradient=True, need_hessian=True) is None
    # Different charge/multiplicity misses
    assert (
        cache.lookup(("H", "H"), near_coords, need_gradient=False, need_hessian=False, charge=1)
        is None
    )
    # Outside tolerance misses
    far_coords = coords.copy()
    far_coords[1, 2] += 1.0e-3 * BOHR_ANGSTROM
    assert (
        cache.lookup(("H", "H"), far_coords, need_gradient=False, need_hessian=False) is None
    )

    # Reload with matching scale_mass preserves entries, mismatched raises ValueError
    reloaded = RestartCache(cache_path, rst_tol_bohr=1.0e-4, scale_mass_amu=1.5)
    assert reloaded.lookup(("H", "H"), coords, need_gradient=True, need_hessian=False) is not None
    with pytest.raises(ValueError, match="scale_mass_amu"):
        RestartCache(cache_path, scale_mass_amu=2.0)

    # Upgrade existing entry with Hessian while preserving stored gradient
    reloaded.store(
        ("H", "H"),
        coords,
        PesEvaluation(energy_kj_mol=12.5, hessian_kj_mol_angstrom2=np.eye(6)),
    )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert math.isclose(payload["scale_mass_amu"], 1.5)
    assert len(payload["entries"]) == 1

    upgraded = RestartCache(cache_path).lookup(
        ("H", "H"), coords, need_gradient=True, need_hessian=True
    )
    assert upgraded is not None
    assert upgraded.gradient_kj_mol_angstrom is not None
    assert upgraded.hessian_kj_mol_angstrom2 is not None


def test_gaussian_provider_parsers_and_formchk_workflow(tmp_path: Path) -> None:
    fake_g16 = tmp_path / "fake_g16.py"
    fake_formchk = tmp_path / "fake_formchk.py"

    # Build a synthetic .fchk text for 2-atom system (dim=6, packed length=21)
    grad_au = np.array([[0.0, 0.0, -0.02], [0.0, 0.0, 0.02]], dtype=float)
    hess_au = np.diag([0.1, 0.1, 0.4, 0.1, 0.1, 0.4])
    packed = [float(hess_au[r, c]) for r in range(6) for c in range(r + 1)]
    fchk_body = [
        "Number of atoms                            I               2",
        "Total Energy                               R     -1.170000000000000E+00",
        "Cartesian Gradient                         R   N=           6",
        " ".join(f"{v:16.8E}" for v in grad_au.reshape(-1)),
        f"Cartesian Force Constants                  R   N=          {len(packed)}",
    ]
    for i in range(0, len(packed), 5):
        fchk_body.append(" ".join(f"{v:16.8E}" for v in packed[i : i + 5]))
    fchk_text = "\n".join(fchk_body) + "\n"

    parsed = parse_gaussian_fchk(fchk_text)
    assert math.isclose(parsed.energy_hartree, -1.17, rel_tol=1.0e-12)
    assert parsed.gradient_hartree_bohr is not None
    assert np.allclose(parsed.gradient_hartree_bohr, grad_au, atol=1.0e-10)
    assert parsed.hessian_hartree_bohr2 is not None
    assert np.allclose(parsed.hessian_hartree_bohr2, hess_au, atol=1.0e-10)

    log_ok = " SCF Done:  E(RHF) =  -1.17000000000     A.U. after 5 cycles\n Normal termination\n"
    assert math.isclose(parse_gaussian_log(log_ok).energy_hartree, -1.17)
    log_archive = (
        " 1\\1\\HOST\\SP\\RHF\\STO-3G\\\\Version=1\\\\\n"
        " HF=-1.170000D+00\\\\RMSD=1.0e-09\\\\\\@\n Normal termination\n"
    )
    assert math.isclose(parse_gaussian_log(log_archive).energy_hartree, -1.17)
    with pytest.raises(DirectDynamicsError, match="error termination"):
        parse_gaussian_log(" Error termination via Lnk1e\n")

    fake_g16.write_text(
        """
from pathlib import Path
Path("gaussian.log").write_text(" Normal termination\\n")
Path("pyrate_dd.chk").write_text("dummy-chk")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    fake_formchk.write_text(
        f"""
import sys
from pathlib import Path
Path(sys.argv[2]).write_text({fchk_text!r})
""".strip()
        + "\n",
        encoding="utf-8",
    )

    gp = GaussianProvider(
        command=[sys.executable, str(fake_g16)],
        fchk_command=[sys.executable, str(fake_formchk)],
        method="MP2",
        basis="cc-pVTZ",
        mem="2GB",
        nprocshared=4,
    )
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    gjf_preview = gp.build_gjf(("H", "H"), coords, need_gradient=False, need_hessian=True)
    assert "#p MP2/cc-pVTZ Freq=NoRaman Units=AU NoSymm FChk" in gjf_preview
    assert "%mem=2GB" in gjf_preview and "%nprocshared=4" in gjf_preview

    ev = gp.evaluate(("H", "H"), coords, need_gradient=True, need_hessian=True)
    assert math.isclose(ev.energy_hartree, -1.17)
    assert ev.hessian_hartree_bohr2 is not None and ev.hessian_hartree_bohr2.shape == (6, 6)

    # Log-only fallback when no .chk / .fchk is emitted
    log_only_wrapper = tmp_path / "g16_logonly.py"
    log_only_wrapper.write_text(
        "from pathlib import Path\n"
        "Path('gaussian.log').write_text(\n"
        "    ' SCF Done: E(RHF) = -1.17 A.U.\\n Normal termination\\n'\n"
        ")\n",
        encoding="utf-8",
    )
    ev_log = GaussianProvider(command=[sys.executable, str(log_only_wrapper)]).evaluate(
        ("H", "H"), coords
    )
    assert math.isclose(ev_log.energy_kj_mol, -1.17 * HARTREE_KJ_MOL, rel_tol=1.0e-12)

    err_wrapper = tmp_path / "g16_err.py"
    err_wrapper.write_text(
        "from pathlib import Path; Path('gaussian.log').write_text(' Error termination\\n')\n",
        encoding="utf-8",
    )
    with pytest.raises(DirectDynamicsError, match="terminated with an error"):
        GaussianProvider(command=[sys.executable, str(err_wrapper)]).evaluate(("H", "H"), coords)


def test_gamess_provider_fixed_and_free_hessian_parsing(tmp_path: Path) -> None:
    # 2 atoms -> dim=6, spans 2 GAMESS column blocks (cols 0..4 then col 5)
    hess_ref = np.diag([0.2, 0.3, 0.5, 0.25, 0.35, 0.55])
    hess_ref[0, 5] = hess_ref[5, 0] = -0.04
    e_au = -76.12345678
    grad_au = np.array([[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]], dtype=float)

    punch_lines = [
        " $GRAD",
        f"E=   {e_au:17.10f}",
        f" H   1.0 {grad_au[0, 0]:15.8E}{grad_au[0, 1]:15.8E}{grad_au[0, 2]:15.8E}",
        f" H   1.0 {grad_au[1, 0]:15.8E}{grad_au[1, 1]:15.8E}{grad_au[1, 2]:15.8E}",
        " $END",
        " $HESS",
        f"ENERGY IS   {e_au:17.10f} E(NUC) IS   1.0",
    ]
    # Fixed-width E15.8 blocks: first 5 columns, then 6th column
    for r in range(6):
        row_vals = "".join(f"{hess_ref[r, c]:15.8E}" for c in range(5))
        punch_lines.append(f"{r + 1:2d}{1:3d}{row_vals}")
    for r in range(6):
        punch_lines.append(f"{r + 1:2d}{2:3d}{hess_ref[r, 5]:15.8E}")
    punch_lines.append(" $END")
    dat_text = "\n".join(punch_lines) + "\n"

    parsed = parse_gamess_output(dat_text, n_atoms=2, log_text=" EXECUTION TERMINATED NORMALLY\n")
    assert math.isclose(parsed.energy_hartree, e_au, rel_tol=1.0e-10)
    assert parsed.gradient_hartree_bohr is not None
    assert np.allclose(parsed.gradient_hartree_bohr, grad_au, atol=1.0e-8)
    assert parsed.hessian_hartree_bohr2 is not None
    assert np.allclose(parsed.hessian_hartree_bohr2, hess_ref, atol=1.0e-8)

    # Free-format space-separated $HESS + energy parsed from .log fallback
    free_lines = [" $HESS"]
    for r in range(6):
        vals = " ".join(f"{hess_ref[r, c]:.8f}" for c in range(6))
        free_lines.append(vals)
    free_lines.append(" $END")
    log_only_e = f" FINAL RHF ENERGY IS {e_au:.10f}\n EXECUTION TERMINATED NORMALLY\n"
    parsed_free = parse_gamess_output("\n".join(free_lines), n_atoms=2, log_text=log_only_e)
    assert math.isclose(parsed_free.energy_hartree, e_au, rel_tol=1.0e-10)
    assert parsed_free.hessian_hartree_bohr2 is not None
    assert np.allclose(parsed_free.hessian_hartree_bohr2, hess_ref, atol=1.0e-8)

    with pytest.raises(DirectDynamicsError, match="abnormally"):
        parse_gamess_output(dat_text, n_atoms=2, log_text=" EXECUTION TERMINATED ABNORMALLY\n")

    fake_gms = tmp_path / "fake_rungms.py"
    fake_gms.write_text(
        f"""
from pathlib import Path
Path("gamess.dat").write_text({dat_text!r})
Path("gamess.log").write_text(" EXECUTION OF GAMESS TERMINATED NORMALLY\\n")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    gms = GamessProvider(command=[sys.executable, str(fake_gms)], scftyp="UHF")
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    inp_txt = gms.build_inp(
        ("H", "H"), coords, need_gradient=True, need_hessian=True, multiplicity=3
    )
    assert "SCFTYP=UHF RUNTYP=HESSIAN UNITS=BOHR ICHARG=0 MULT=3" in inp_txt

    ev = gms.evaluate(("H", "H"), coords, need_gradient=True, need_hessian=True, multiplicity=3)
    assert ev.hessian_hartree_bohr2 is not None
    assert np.allclose(ev.hessian_hartree_bohr2, hess_ref, atol=1.0e-8)

    # Factory mapping builder
    built = direct_dynamics_from_mapping(
        {
            "type": "command",
            "command": f"{shlex.quote(sys.executable)} {shlex.quote(str(fake_gms))}",
            "restart_file": "sub/r.json",
            "rst_tol_bohr": 1.0e-6,
        },
        base_dir=str(tmp_path),
    )
    assert built.cache.path == tmp_path / "sub" / "r.json"
    assert built.command == (sys.executable, str(fake_gms))
