# io — libqcx-io

The run-schema module: TOML run-input parsing into a validated `RunInput`
(molecule / basis / aux-family / method / accuracy preset / SCF options / guess
plus the `[properties]` block and the `[grid]` block), schema
validation with actionable errors, and the result-JSON serialization. It does
**not** execute anything (that is `driver/`) and does not run analyses (that is
`properties/` — io owns its own `EspFitScheme` enum because the DAG forbids an
io→properties edge).

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

12 of 14 — depends on: core, molecule, basisset, integrals, scf, grid (grid is
implementation-only: parse_input.cpp reads `AngularGrid::kAvailableSizes` so
the `[grid] angular_points` refusal names the shipped Lebedev sizes instead of
copying the list) — consumed by: driver. Machine-checked by
`tools/check_dag.py`; this line is a summary, not the source of truth.

## Public headers / key types

- `io/include/qcx/io/`: `run_input.hpp`, `parse_input.hpp`,
  `validate_input.hpp`, `result_json.hpp`
- `RunInput` — the validated run schema.
- `ParseRunInputFile` — the TOML entry point.
- `ValidationReport` — schema-violation reporting.
- `RunResult` + `SerializeRunResultJson` — the result block (null when unset,
  never placeholder numbers).
