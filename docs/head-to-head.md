# Polyrate 2023 vs PyRate: reproducible head-to-head checks

## Scope and provenance

Reference: Polyrate 2023, DOI `10.17632/vpf3y9cbhf.1` (program, Apache-2.0). The published distribution includes **112 test runs** (46 direct-dynamics tests, 85 single-level tests and 27 dual-level tests). These classifications overlap; do not sum them. Reference outputs are not bundled in PyRate and no numerical agreement with them is currently claimed.

PyRate 0.1 includes a subset of the chemistry algorithms. This comparator is an **evidence tool**, not proof of feature parity. Legacy feature support, direct-dynamics inputs, exact file-unit behavior, and C++/Python parity must be verified separately.

## Basic run

Use a compatible PyRate TOML approximation of the *same physical model* and compare it to the Polyrate `fu15` reference output:

```bash
pyrate run examples/harmonic_cvt.toml --format json -o /tmp/pyrate.json
python tools/compare_polyrate_rates.py \
  /path/to/polyrate/reference/example.fu15 /tmp/pyrate.json \
  --direction forward --rtol 0.005 --output /tmp/comparison.json
```

The tool matches temperatures within 0.001 K, converts supported first-order or second-order units to common SI values, compares the explicitly mapped columns, and records all unmatched columns. Default mappings are **TST → `k_tst`** and **CVT → `k_cvt`**. Other methods are not silently equated.

For controlled, validated algorithms only, add an explicit mapping such as `--column CVT/ZCT=k_final`. **Do not interpret this as evidence of Polyrate's full ZCT implementation:** the current PyRate tunneling correction remains approximate. Use `--strict` to require that every listed reference column is evaluated.

Exit code: 0 = comparisons present and all required comparisons passed; 1 = quantitative mismatch, missing values/temperatures, or strict-mode unmapped columns; 2 = invalid input, ambiguous reference tables or unsupported units. The JSON report includes per-point pass/fail, temperature, reference value, PyRate value and relative error. Zero reference rates are only considered equal to exactly zero; non-finite values fail and are serializable.

## Acceptance gates per feature

1. Verify that both programs use the same physical model, normalization and reference temperature grid; extract exact intermediate quantities where possible.
2. Validate parsing and execution of legacy keywords, including every default and unit convention; parsing alone is not enough.
3. Compare primitive partition functions, ZPE, free-energy profiles and reaction-path saddle maxima before final rate constants.
4. Compare transmission actions and microcanonical/canonical tunneling coefficients independently of final rates.
5. Require native C++ and Python fallback agreement, then compare Polyrate output (allowing for printed precision).
6. Add each verified official test case to a manifest with archive SHA-256 and output provenance; count cases *only after* numerical comparison passes.

Polyrate 2023 removed ICVT from its available options. It is therefore a legacy/historical compatibility consideration, not a required **Polyrate 2023** feature. Primary missing modules are microcanonical VTST, small/large-curvature tunneling, reaction-path integration, direct dynamics, VRC, anharmonic modes, IVTST/ISPE/IOC and SS-QRRK.

## Current evidence

The comparator has synthetic regression tests covering forward and reverse rate tables, D exponents, unit conversion, missing values, non-finite outputs, and strict coverage failure. This validates comparison *infrastructure* only. No official Polyrate 2023 reference file has been numerically reproduced in these tests.
