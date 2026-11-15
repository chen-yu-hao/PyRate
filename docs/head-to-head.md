# Polyrate 2023 vs PyRate: reproducible head-to-head checks

## Scope and provenance

Reference: Polyrate 2023, DOI `10.17632/vpf3y9cbhf.1` (program, Apache-2.0). The published distribution includes **112 test runs** (46 direct-dynamics tests, 85 single-level tests and 27 dual-level tests). These classifications overlap; do not sum them. Reference outputs are not bundled in PyRate; users can point the comparator or test suite at a local copy of the Polyrate 2023 distribution.

PyRate 0.2 implements single-level harmonic/anharmonic/hindered-rotor thermodynamics, CTST/CVT/CUS/μVT/US variational methods, Wigner/CAG/ZCT/SCT/LCT(LCG3 & LCG4)/COMT/μOMT/Eckart tunneling, and direct execution of `.dat + .fu40` reaction-path inputs.

## Basic run

Run a Polyrate `.dat + .fu40` input (or compatible PyRate TOML input) and compare it directly against the Polyrate `fu15` reference output:

```bash
pyrate run /path/to/polyrate2023/testrun/ch5/ch5fu40tr5.dat \
  --fu40 /path/to/polyrate2023/testrun/ch5/ch5fu40tr5.fu40 \
  --format json -o /tmp/pyrate.json
python tools/compare_polyrate_rates.py \
  /path/to/polyrate2023/testo/ch5fu40tr5.fu15 /tmp/pyrate.json \
  --tunneling --direction forward --rtol 0.05 --output /tmp/comparison.json
```

The tool parses all three Polyrate 2023 `.fu15` rate table header styles and merges multi-subtable continuation blocks (`CVT/COMT`, `CVT/mOMT`, etc.), matches temperatures within 0.001 K, converts supported first-order or second-order units to common SI values, compares mapped columns, and records any unmapped columns (such as historical `ICVT`).

Built-in column mappings (`DEFAULT_COLUMN_MAP`, `EXTENDED_COLUMN_MAP`, and `TUNNELING_COLUMN_MAP` via `--tunneling`) resolve:
- Base variational methods: `TST` → `k_tst`, `CVT` → `k_cvt`, `CUS` → `k_cus`, `MUVT`/`MCVT` → `k_muvt`, `US` → `k_us`.
- Tunneling-corrected CVT columns: `CVT/ZCT` → `k_cvt_zct`, `CVT/SCT` → `k_cvt_sct`, `CVT/LCT` / `CVT/LCG3` / `CVT/LCG4` → `k_cvt_lct`, `CVT/COMT` → `k_cvt_comt`, `CVT/mOMT` / `CVT/MUOMT` → `k_cvt_muomt` (plus arbitrary `--column <REF>=<FIELD>` overrides).

Exit code: 0 = comparisons present and all required comparisons passed; 1 = quantitative mismatch, missing values/temperatures, or strict-mode unmapped columns; 2 = invalid input, ambiguous reference tables or unsupported units.

## Acceptance gates per feature

1. Verify that both programs use the same physical model, normalization and reference temperature grid; extract exact intermediate quantities where possible.
2. Validate parsing and execution of legacy keywords, including every default and unit convention; parsing alone is not enough.
3. Compare primitive partition functions, ZPE, free-energy profiles and reaction-path saddle maxima before final rate constants.
4. Compare transmission actions and microcanonical/canonical tunneling coefficients independently of final rates.
5. Require native C++ and Python fallback agreement, then compare Polyrate output (allowing for printed precision).
6. Add each verified official test case to a manifest with archive SHA-256 and output provenance; count cases *only after* numerical comparison passes.

## Current evidence

1. **Parser & `.fu15` coverage**: All 112/112 `.dat` inputs and all 101/101 non-empty `.fu15` files in the official Polyrate 2023 distribution parse cleanly without errors.
2. **End-to-end `.dat + .fu40` numerical verification (`ch5fu40tr1`–`ch5fu40tr6`)**:
   - `ch5fu40tr1` (`freqsource read`, pre-diagonalized frequencies + explicit curvatures) and `ch5fu40tr2`–`ch5fu40tr6` (`freqsource hessian`, packed/full Cartesian force-constant matrices projected with `PROJCT` and curvature vectors computed via `RPHB`):
   - **Forward `TST` and `CVT` rates**: `0.08%` relative difference across all temperatures (`298–1550 K`), matching Polyrate's 3-significant-digit `.fu15` output.
   - **Forward `CVT/ZCT` and `CVT/SCT` rates**: Across a $1.85 \times 10^5$ low-temperature tunneling enhancement at 298 K up to 1550 K, PyRate agrees with Polyrate within `2.1%–8.4%`.
3. **Large-curvature (`LCG3` & `LCG4`), `COMT`, and true microcanonical OMT (`mOMT`) verification (`clhbrtr1` & `clhbrtr3`)**:
   - On the heavy-light-heavy collinear Extended LEPS benchmark `Cl + HBr -> ClH + Br` (`200–2400 K`, 7 temperatures × 7 columns = 49 points per case, strict mode with zero unmapped columns):
   - **`clhbrtr1` (`LCG3` to all accessible excited states)**: All 49/49 points pass within `4.42%` (`TST` `0.26%`, `CVT` `0.06%`, `CVT/SCT` `1.33%`, `CVT/LCG3` `4.42%`).
   - **`clhbrtr3` (`LCG4` to all accessible excited states + `COMT` + `mOMT`)**: All 49/49 points pass within `4.46%` (`CVT/LCG4` `4.46%`, `CVT/COMT` `1.33%`, `CVT/mOMT` `1.61%` at 200 K), reproducing both the factor-of-3.5 reduction from `LCG3` to `LCG4` via continuous nonadiabatic boundary extension and `CVT/mOMT > CVT/COMT` where LCT dominates at low $E$ while SCT dominates near the barrier top.

