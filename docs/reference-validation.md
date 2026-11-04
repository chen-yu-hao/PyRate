# Polyrate reference validation

## Reference source

The intended reference is Polyrate 2023 revision B from the official Mendeley/Zenodo distribution.  It contains the program, manual, potential routines, input files, and reference outputs.  The archive is not committed to this repository.

After extracting a local copy, create a machine-readable inventory:

```bash
python tools/import_polyrate_reference.py /path/to/polyrate2023 \
  --output reference/manifest.json
```

`reference/` is ignored by Git so licensed reference artifacts and large outputs remain local.

## Comparison levels

1. **Parser** — every section and keyword is retained; no unknown input disappears.
2. **Primitive physics** — compare masses, moments, partition functions, ZPE and standard-state concentration.
3. **Path thermodynamics** — compare `V_MEP`, `V_a^G`, `ΔG(s,T)`, fitted maximum and `s_CVT`.
4. **Tunneling** — compare action and canonical `κ(T)` before multiplying rates.
5. **Final rates** — compare CTST/CVT/tunneling-corrected columns in `fu15`.

The `pyrate.reference` module parses summary tables and supplies relative-error helpers.  Reference tables printed with only three significant digits require looser tolerances than full-precision intermediate files.

## Recommended test tagging

- ordinary unit tests: always run in CI;
- `reference_core`: small distributable or generated fixtures;
- `reference_polyrate`: requires `POLYRATE_REFERENCE_ROOT` and is skipped in public CI when absent;
- `slow`: direct dynamics or dense energy/path grids.

## Provenance

A reference manifest should record archive version, DOI, file SHA-256, platform/compiler used to generate outputs, and any local changes to potential routines.  Never compare against an output whose build provenance is unknown.
