# driver — libqcx-driver

The `qcx run` orchestrator — the last link in the DAG and the one module
allowed to compose everything: input parsing (io) → SCF through the
`FockBuilderFn` seam (scf/integrals) → property analyses (properties) → JSON
result (io), with exit codes 0 success / 1 parse-or-run error / 2 usage. It
does **not** implement engines, analyses, or serialization itself.

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

14 of 14 — depends on: molecule, basisset, integrals, scf, io, properties —
consumed by: (none — terminal). Machine-checked by `tools/check_dag.py`; this
line is a summary, not the source of truth.

## Public headers / key types

- `driver/include/qcx/driver/run_driver.hpp`
- `RunDriver` — the run entry point (input file → result JSON, exit code).
