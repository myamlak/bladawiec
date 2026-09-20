# linalg — libqcx-linalg

Dense and blocked linear algebra over the vendor seam: `DenseMultiply` (Eigen
for small blocks, optional vendor BLAS via `QCX_BLAS_VENDOR` for large ones),
the batched-GEMM seam for the MD-engine contractions, the Spectra-backed
iterative eigensolver, and the amgcl-backed sparse solver. It does **not**
define the tensor container (that is `memory/`), does not implement integral
engines, and carries no SCF logic.

Vendor BLAS is pinned to one thread per process: parallelism is the backend's
OpenMP loops, and the BLAS kernels are meant to run serial per thread. MKL gets
the sequential threading layer at link time; the fallback vendors thread by
default and are pinned to one thread through the process environment instead.

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

4 of 14 — depends on: core — consumed by: symmetry, integrals, scf.
Machine-checked by `tools/check_dag.py`; this line is a summary, not the source
of truth.

## Public headers / key types

- `linalg/include/qcx/linalg/`: `dense_ops.hpp`, `batched_ops.hpp`,
  `iterative_eigensolver.hpp`, `sparse_solver.hpp`
- `DenseMultiply` — the dense multiply entry point (Eigen/BLAS dispatch).
- `MultiplyBatched` + `GemmKernel` — the batched-GEMM seam.
- `ComputeLowestEigenpairs` — the Spectra-backed eigensolver.
- `SolveSparseSystem` — the amgcl-backed sparse solver.
