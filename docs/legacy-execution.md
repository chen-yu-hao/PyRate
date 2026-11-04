# Executable Polyrate legacy subset

PyRate keeps three independent compatibility states for a legacy deck:

1. **syntax** — the section/keyword structure can be retained;
2. **semantics** — every value required by a supported physical model has an explicit mapping;
3. **execution** — the typed `Reaction` and `CalculationOptions` models can be constructed without guessed data.

A syntactically valid deck is therefore not automatically executable.

## Currently executable

The first executable subset is conventional harmonic TST with stationary-point data read from Polyrate fixed unit 30, 31, or 40 files. Execution currently requires:

- `*RATE/TST` and a `TEMP` grid;
- `*ENERGETICS/POTENTIAL UNIT30`, `UNIT31`, or `UNIT40`;
- legacy geometries and an atom table;
- explicit stationary-point potentials and direct frequencies for each non-atomic reactant and the saddle point;
- zero electronic excitation energies in legacy `ELEC` blocks;
- no CVT, multidimensional tunnelling, anharmonic, or torsional request.

`SIGMAF` is represented as a reaction-path degeneracy multiplying the rate. It is deliberately not folded into a molecular rotational symmetry number or an electronic degeneracy. `SIGMAR` is retained in metadata for later reverse-rate execution.

For a supported deck:

```bash
pyrate validate case.dat --require-executable
pyrate run case.dat --format json
pyrate trace case.dat -o case-trace.json
```

The fixed-unit file is resolved next to the primary input, for example `case.dat` plus `case.fu40`.

## Explicitly blocked

The converter refuses execution rather than inventing values when it encounters:

- Hessian-only stationary-point data before legacy Hessian-to-mode semantics are available;
- nonzero legacy electronic excitation energies whose unit convention is not yet mapped;
- CVT/ICVT, ZCT/SCT/LCT, μVT/US/CUS, anharmonic, or torsional models;
- missing potentials, frequency units, geometries, atoms, or temperature grids.

`pyrate inspect` and `pyrate validate` report the blockers separately from syntax status.

## Official reference gate

The CI reference workflow materializes the 16 core-method candidates from the Polyrate 2023 revision B distribution. The initial executable gate requires four direct-frequency CH3 + H2 conventional-TST cases to run and match their printed forward TST tables. Comparison uses the declared relative tolerance plus half of one unit in the last printed decimal digit, because compact `fu15` tables retain only a few significant digits.

This gate demonstrates a real legacy-to-solver path for a narrow subset. It is not a claim that all 112 official inputs, reverse rates, or unprinted intermediate quantities are reproduced.
