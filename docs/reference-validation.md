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

## Verified source locations (not yet benchmarked)

- Official 2023 revision B record: https://zenodo.org/records/8309384
- Archive: `polyrate2023.0831.xz` (reported MD5 `fa1423cc3f4a1279631b05f49228a2d2`)
- Manual: `Polyrate2023_manual_230831.pdf` (reported MD5 `853e51ff3d50b07736a5d92993ab089c`)
- Mendeley program DOI: https://doi.org/10.17632/vpf3y9cbhf.1
- CPC announcement: https://doi.org/10.1016/j.cpc.2023.108933

PR #6's successful GitHub Actions reference-audit run downloaded the archive, checked its published MD5 and extracted 112 reference cases. The PDF manual has not been downloaded and hashed by this branch; full staged Polyrate numeric equivalence remains unverified. The release article confirms ICVT removal and 1 bar standard-state free energies, not algorithm-by-algorithm PyRate equivalence. Record SHA-256 only after fetching the bytes.
