from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace

from pyrate.calculation import calculate_rates
from pyrate.input import load_legacy_calculation
from pyrate.legacy import load_legacy
from pyrate.legacy_execution import analyze_legacy_execution
from pyrate.rates import conventional_tst_rate_si


LEGACY_INPUT = """
*GENERAL
  TITLE
    H2 + H synthetic direct-frequency legacy case
  END
  ATOMS
    1 H
    2 H
    3 H
  END
  INPUNIT AU
*ENERGETICS
  POTENTIAL UNIT40
*REACT1
  GEOM
    1 0.0 0.0 0.0
    2 1.4 0.0 0.0
  END
  SPECIES LINRP
*REACT2
  GEOM
    3 20.0 0.0 0.0
  END
  SPECIES ATOMIC
  ELEC
    2 0.0
  END
*START
  GEOM
    1 0.0 0.0 0.0
    2 1.3 0.0 0.0
    3 2.2 0.8 0.0
  END
  SPECIES NONLINTS
  ELEC
    2 0.0
  END
*RATE
  FORWARDK
  SIGMAF 2
  TST
  NOCVT
  TEMP
    300.0
  END
"""


FIXED_UNIT_40 = """
*gen40
  title
    synthetic direct-frequency data
  end
  frequnit au
*r140
  vvalue 0.0
  freq
    0.020
  end
  ifreq
    1
  end
*r240
  vvalue 0.0
*saddle40
  vvalue 0.010
  freq
    0.015 0.010
  end
  ifreq
    1 2
  end
  wstar 0.005
"""


def _write_case(tmp_path, fixed_text: str = FIXED_UNIT_40):
    input_path = tmp_path / "case.dat"
    input_path.write_text(LEGACY_INPUT, encoding="utf-8")
    input_path.with_suffix(".fu40").write_text(fixed_text, encoding="utf-8")
    return input_path


def test_direct_frequency_legacy_case_is_executable(tmp_path) -> None:
    input_path = _write_case(tmp_path)
    document = load_legacy(input_path)
    analysis = analyze_legacy_execution(document, input_path)

    assert analysis.semantic_status == "mapped"
    assert analysis.execution_status == "ready"
    assert analysis.executable
    assert analysis.fixed_unit == 40
    assert not analysis.blockers

    loaded = load_legacy_calculation(input_path)
    assert loaded.options.method == "tst"
    assert loaded.options.output_rate_unit == "cm^3 molecule^-1 s^-1"
    assert loaded.reaction.transition_state.metadata["reaction_path_degeneracy"] == 2.0
    assert len(loaded.reaction.reactants) == 2
    assert len(loaded.reaction.transition_state.imaginary_frequencies_cm1) == 1

    result = calculate_rates(loaded.reaction, loaded.options, name=loaded.name)
    assert len(result.rows) == 1
    assert result.rows[0].rate_unit == "cm^3 molecule^-1 s^-1"
    assert result.rows[0].k_tst > 0.0
    assert math.isfinite(result.rows[0].k_tst)


def test_sigmaf_is_a_rate_factor_not_a_molecular_symmetry_number(tmp_path) -> None:
    loaded = load_legacy_calculation(_write_case(tmp_path))
    temperature = loaded.options.temperatures_k[0]
    _, legacy_rate = conventional_tst_rate_si(loaded.reaction, temperature)

    metadata = dict(loaded.reaction.transition_state.metadata)
    metadata.pop("reaction_path_degeneracy")
    unit_degeneracy_ts = replace(
        loaded.reaction.transition_state,
        metadata=metadata,
    )
    unit_degeneracy_reaction = replace(
        loaded.reaction,
        transition_state=unit_degeneracy_ts,
    )
    _, unit_rate = conventional_tst_rate_si(unit_degeneracy_reaction, temperature)

    assert math.isclose(legacy_rate / unit_rate, 2.0, rel_tol=1.0e-14)
    assert all(species.symmetry_number == 1 for species in loaded.reaction.reactants)
    assert loaded.reaction.transition_state.symmetry_number == 1


def test_hessian_only_fixed_unit_is_semantically_blocked(tmp_path) -> None:
    fixed_text = FIXED_UNIT_40.replace(
        "  freq\n    0.015 0.010\n  end\n  ifreq\n    1 2\n  end\n",
        "  hessform packed\n  hessian\n    0.1 0.0 0.2\n  end\n",
    )
    input_path = _write_case(tmp_path, fixed_text)
    analysis = analyze_legacy_execution(load_legacy(input_path), input_path)

    assert analysis.semantic_status == "partial"
    assert analysis.execution_status == "blocked"
    assert any("Hessian" in blocker for blocker in analysis.blockers)


def test_cli_runs_executable_legacy_case(tmp_path) -> None:
    input_path = _write_case(tmp_path)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrate",
            "run",
            str(input_path),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(process.stdout)
    assert payload["method"] == "tst"
    assert payload["molecularity"] == 2
    assert payload["rows"][0]["rate_unit"] == "cm^3 molecule^-1 s^-1"
    assert payload["rows"][0]["k_tst"] > 0.0


def test_cli_validate_distinguishes_syntax_and_execution(tmp_path) -> None:
    input_path = _write_case(tmp_path)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrate",
            "validate",
            str(input_path),
            "--require-executable",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "syntax_status: pass" in process.stdout
    assert "semantic_status: mapped" in process.stdout
    assert "execution_status: ready" in process.stdout
