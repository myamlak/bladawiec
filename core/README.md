# core — libqcx-core

The qcx error-handling and logging foundation, shared by every module in the
dependency DAG: the `Result`/`Error`/`ErrorCode` error model (no exceptions —
fallible operations return `std::expected`-based results) and the spdlog-based
logging wrapper. It does **not** contain module logic, does not allocate
(allocation is `memory/`'s job), and does not pick execution backends.

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

1 of 14 — depends on: (none) — consumed by: backend, memory, linalg, molecule,
grid, integrals, scf, storage, io, properties. Machine-checked by
`tools/check_dag.py`; this line is a summary, not the source of truth.

## Public headers / key types

- `core/include/qcx/error.hpp`, `core/include/qcx/logging.hpp`
- `Result<T>` — the fallible-operation return type; `std::expected<T, Error>`.
- `ErrorCode` — the error categories (`kInvalidArgument`, `kConvergenceFailure`, ...).
- `Info` — the logging entry point (spdlog wrapper, compile-time-literal format).
