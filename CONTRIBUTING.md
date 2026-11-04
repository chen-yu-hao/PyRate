# Contributing

1. Keep physical units explicit at public boundaries.
2. Put orchestration and validation in Python; move measured hotspots to C++.
3. Every native function must have a `py_*` Python fallback and an in-process parity test
   (`tests/test_parity_<area>.py`); see `docs/architecture.md` §2.
4. Do not mark a Polyrate feature implemented without a reference regression.
5. Preserve unknown legacy keywords and report unsupported semantics explicitly.

Development setup:

```bash
python -m pip install -e ".[test,dev]"
ruff check src tests tools
mypy
pytest -q
PYRATE_DISABLE_NATIVE=1 pytest -q
```

For a new compatibility module, update `docs/compatibility.md` in the same pull request and include the reference case, source provenance, intermediate quantities and final tolerance.
