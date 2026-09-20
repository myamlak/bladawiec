# memory — libqcx-memory

The exclusive allocation site and the data-container module: device-abstracted
buffers, tensors, and sparsity patterns, plus the policy types that describe a
tensor's symmetry and sparsity (`Dense` / `SparsityPatternBacked` x
`NoSymmetry`). It does **not** do arithmetic (that is `linalg/`), does not pick
execution backends (that is `backend/`), and contains no numerical engines.

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

3 of 14 — depends on: core, backend — consumed by: molecule, integrals, scf.
Machine-checked by `tools/check_dag.py`; this line is a summary, not the source
of truth.

## Public headers / key types

- `memory/include/qcx/memory/`: `device_buffer.hpp`, `device_ptr.hpp`,
  `allocator_traits.hpp`, `cuda_allocator_traits.hpp`, `tensor.hpp`,
  `tensor_policies.hpp`, `sparsity_pattern.hpp`
- `DeviceBuffer<T, Backend>` — the exclusive allocation site, with explicit
  residency/sync control (`Residency`, `MarkHostDirty`/`MarkDeviceDirty`).
- `Tensor<T, Rank, Backend, SymmetryPolicy, SparsityPolicy>` — the row-major
  workhorse container.
- `SparsityPattern<Backend>` — the CSR-adjacency primitive with validating
  `Create`, `BuildSparsityFromAdjacency` (LinK shape) and
  `BuildSparsityFromTree` (QFMM shape).
- `tensor_policies.hpp` — the `ContractionPolicy` pairs whose dispatch drives
  `contract()`.
