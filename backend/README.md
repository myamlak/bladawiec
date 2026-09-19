# backend — libqcx-backend

The execution-abstraction module: dispatch tags, the `ExecutionBackend` concept,
the OpenMP CPU backend (thread pool, topology-aware team-size policy), the CUDA
backend (gated by `QCX_ENABLE_CUDA`), and the machine probes (`DetectCpuTopology`,
`DetectHostMemory`). It does **not**
own data or allocate buffers (that is `memory/`), does not implement numerical
kernels (that is `integrals/`), and carries no integral or SCF logic.

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

2 of 14 — depends on: core — consumed by: memory, integrals. Machine-checked by
`tools/check_dag.py`; this line is a summary, not the source of truth.

## Public headers / key types

- `backend/include/qcx/backend/`: `backend.hpp`, `concepts.hpp`, `tags.hpp`,
  `cpu_backend.hpp`, `cpu_scheduler.hpp`, `cpu_topology.hpp`,
  `memory_topology.hpp`, `cuda_backend.hpp`
- `CpuTag` / `CudaTag` — the execution-backend dispatch tags.
- `ExecutionBackend` — the concept every backend satisfies.
- `CpuTopology` + `DetectCpuTopology` — P/E-core and hyperthread detection.
- `DefaultTeamSize` — the team-size policy.
- `DetectHostMemory` — the host-RAM probe that sizes the ERI cache.
