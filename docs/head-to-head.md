# Polyrate 2023 vs PyRate: reproducible head-to-head checks

## Scope and provenance

Reference: Polyrate 2023, DOI `10.17632/vpf3y9cbhf.1` (program, Apache-2.0). The published distribution includes **112 test runs** (46 direct-dynamics tests, 85 single-level tests and 27 dual-level tests). These classifications overlap; do not sum them. Reference outputs are not bundled in PyRate; users can point the comparator or test suite at a local copy of the Polyrate 2023 distribution.

PyRate 0.2 implements single-level harmonic/anharmonic/hindered-rotor thermodynamics, CTST/CVT/CUS/μVT/US variational methods, Wigner/CAG/ZCT/SCT/Eckart tunneling, direct execution of `.dat + .fu40` reaction-path inputs, Page–McIver/`ES1`/`Euler` mass-scaled MEP integration with Jackels–Gu–Truhlar curvilinear projection (`CURV`), analytical PES models (`ExtendedLepsSurface` for `Cl+HBr` and `H+H2`, `MullerBrownSurface`, `EckartValleySurface`), and isolated subprocess direct-dynamics providers (`ExternalCommandPes`, `GaussianProvider`, `GamessProvider`) with `RestartCache`.

## Basic run

Run a Polyrate `.dat + .fu40` input (or compatible PyRate TOML input, or `.dat` paired with `--pes`) and compare it directly against the Polyrate `fu15` reference output:

```bash
pyrate run /path/to/polyrate2023/testrun/ch5/ch5fu40tr5.dat \
  --fu40 /path/to/polyrate2023/testrun/ch5/ch5fu40tr5.fu40 \
  --format json -o /tmp/pyrate.json
python tools/compare_polyrate_rates.py \
  /path/to/polyrate2023/testrun/ch5/ch5fu40tr5.fu15 /tmp/pyrate.json \
  --direction forward --rtol 0.01 --output /tmp/comparison.json
```

The tool parses all three Polyrate 2023 `.fu15` rate table header styles and merges multi-subtable continuation blocks (`CVT/COMT`, `CVT/mOMT`, etc.), matches temperatures within 0.001 K, converts supported first-order or second-order units to common SI values, compares mapped columns, and records any unmapped columns (such as `LCT` or `ICVT`).

Built-in column mappings automatically resolve:
- Base variational methods: `TST` → `k_tst`, `CVT` → `k_cvt`, `CUS` → `k_cus`, `muVT`/`uVT` → `k_muvt`, `US` → `k_us`.
- Tunneling-corrected columns (`<METHOD>/<TUNNEL>`): `/W`, `/ZCT`, `/MEPSAG`, `/SCT`, `/CD-SCSAG`, `/OMT`, `/COMT`, `/mOMT`, `/muOMT` when the PyRate run uses the matching tunneling method (or override via `--column <REF>=<FIELD>`). Note that PyRate's `omt`/`comt`/`muomt` options only optimize over ZCT and SCT (LCT is not implemented), so they reduce to SCT and are not expected to reproduce Polyrate's `/COMT` or `/mOMT` columns when large-curvature tunneling matters.

Exit code: 0 = comparisons present and all required comparisons passed; 1 = quantitative mismatch, missing values/temperatures, or strict-mode unmapped columns; 2 = invalid input, ambiguous reference tables or unsupported units.

## Acceptance gates per feature

1. Verify that both programs use the same physical model, normalization and reference temperature grid; extract exact intermediate quantities where possible.
2. Validate parsing and execution of legacy keywords, including every default and unit convention; parsing alone is not enough.
3. Compare primitive partition functions, ZPE, free-energy profiles and reaction-path saddle maxima before final rate constants.
4. Compare transmission actions and microcanonical/canonical tunneling coefficients independently of final rates.
5. Require native C++ and Python fallback agreement, then compare Polyrate output (allowing for printed precision).
6. Add each verified official test case to a manifest with archive SHA-256 and output provenance; count cases *only after* numerical comparison passes.

## Current evidence

1. **Parser & `.fu15` coverage**: All 112/112 `.dat` inputs and all 101/101 non-empty `.fu15` files in the official Polyrate 2023 distribution parse cleanly without errors; `pyrate validate --strict` rejects unsupported legacy options (`LCG3`/`LCG4`, `ISPE`, `ICVT`, `VRCTST`, `SOLV`, `RODS`, invalid units/scales) with structured section/keyword/line diagnostics.
2. **End-to-end `.dat + .fu40` numerical verification (`ch5fu40tr1`–`ch5fu40tr6`)**:
   - `ch5fu40tr1` (`freqsource read`, pre-diagonalized frequencies + explicit curvatures) and `ch5fu40tr2`–`ch5fu40tr6` (`freqsource hessian`, packed/full Cartesian force-constant matrices projected with `PROJCT` and curvature vectors computed via `RPHB`):
   - **Forward `TST` and `CVT` rates**: `0.08%` relative difference across all temperatures (`298–1550 K`), matching Polyrate's 3-significant-digit `.fu15` output.
   - **Forward `CVT/ZCT` and `CVT/SCT` rates**: Across a $1.85 \times 10^5$ low-temperature tunneling enhancement at 298 K up to 1550 K, PyRate's piecewise-linear Kemble WKB + `TPCDSC` small-curvature implementation agrees with Polyrate's quintic-spline integration within `2.1%–8.4%` (`ratio = 0.916–0.979`).
3. **Analytical PES + Curvilinear Page–McIver MEP + reversible `.fu40` round-trip (`Cl + HBr -> HCl + Br` extended LEPS)**:
   - Stationary-point optimization reproduces the extended LEPS barrier (`10.9993 kcal mol⁻¹`), reaction energy (`-16.160 kcal mol⁻¹`), HBr/HCl stretches (`2706.94` and `2988.15 cm⁻¹`), and linear saddle frequencies (`452.31, 452.31, 305.89, 1563.29i cm⁻¹`);
   - Jackels–Gu–Truhlar curvilinear projection (`CURV`) yields `0` spurious negative transverse frequencies along the Page–McIver MEP (versus spurious negative bending modes under rectilinear `RECT`), locates `s_CVT < -0.1 bohr` (`k_CVT < k_TST`), and recovers `κ_SCT > κ_ZCT > 1`;
   - Exporting `.dat` (`POTENTIAL BUILTIN:cl_hbr_leps`) to `.fu40` (`--export-fu40` / `save_fu40`, both `freqsource hessian` packed/full and `freqsource read`) and re-running under `POTENTIAL UNIT40` agrees within `rel_tol < 1e-6` on `k_TST`, `k_CVT`, and `k_CVT/SCT`, and passes strict 3-significant-digit `.fu15` comparator validation across 200–1000 K (15/15 comparisons within `0.5%`).
4. **Collinear `H + H2` / `D + H2` / `H + D2` LEPS isotopic substitution & native/fallback parity**:
   - Symmetric `H + H2` Page–McIver MEP locates `|s_CVT| < 1e-4 bohr` with `k_CVT == k_TST` (`rel_tol < 1e-4`), reproduces identical vibrational frequencies and rates between `"2H"` and `"D"` isotope symbols (`rel_tol < 1e-12`), captures the primary H/D kinetic isotope effect (`> 10` at 300 K between `H+H2` and `H+D2`), and matches end-to-end between the compiled `pyrate-cpp-0.3` backend and `PYRATE_DISABLE_NATIVE=1` within `rel_tol < 1e-8`.

