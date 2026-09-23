# integrals — libqcx-integrals

The integral-engines module: the Boys kernel (double/float/SIMD/CUDA lanes), the
one- and two-electron builders, the matrix-form MD engine, the Schwarz and
density-based screening, the direct J/K and RI-J and QFMM Fock builders, the GPU
Fock path, the certified fp32 lane, the sparsity `contract()` dispatch,
the symmetry reduction, and the in-memory ERI cache tier. It does **not**
run SCF (that is `scf/`, which drives these builders through the `FockBuilderFn`
seam) and does not persist integrals to disk (that is `storage/`).

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

9 of 14 — depends on: core, backend, memory, linalg, molecule, basisset —
consumed by: scf, storage, io, properties, driver. Machine-checked by
`tools/check_dag.py`; this line is a summary, not the source of truth.

## Public headers / key types

- `integrals/include/qcx/integrals/`: `boys.hpp`, `f16.hpp`,
  `eri_batch.hpp`, `eri_dense.hpp`, `eri_cuda.hpp`, `eri_cache.hpp`,
  `one_electron.hpp`, `two_electron.hpp`, `shell_pairs.hpp`, `screening.hpp`,
  `fock_build.hpp`, `gpu_fock_build.hpp`, `incremental_fock.hpp`,
  `ri_engine.hpp`, `qfmm_fock_build.hpp`, `aux_basis.hpp`, `accuracy.hpp`,
  `limits.hpp`, `symmetry_reduction.hpp`, `engine_version.hpp`
- `BoysAllOrders` / `BoysSingle` — the Boys kernel.
- `ComputeEriBatch` / `ComputeEriBatchCertified` — the batched ERI API and its
  certified fp32 lane.
- `DirectJkFockBuilder` — the density-screened direct J/K builder behind the
  `FockBuilderFn` seam.
- `RiJkFockBuilder` — the 3-center RI-J builder.
- `QfmmJBuilder` — the QFMM composed-J builder.
- `EriBatchCache` — the in-memory value cache tier.

## Rationale pointers

The design rationale for each builder is carried in the Doxygen comments on its
public header, next to the value or the choice it explains.

## Generation / regeneration notes

- The generator-owned headers `integrals/src/internal/md_tables_gen.hpp`,
  `md_dispatch_gen.hpp`, and `qfmm_tables_gen.hpp` are byte-identity-checked by
  `tools/gen_md_tables.py --check` and `tools/gen_qfmm_translation_tables.py
  --check`; reference data by `tools/gen_md_reference.py --check` and
  `tools/gen_integrals_reference.py --check` — all under
  `QCX_REFERENCE_REGENERATION_CHECK` (integrals/CMakeLists.txt). Fix a
  generator, never a generated file.
- The Boys kernel is an exception: it is owned by the boys submodule
  (upstream-first). Its generated files
  (`external/boys/src/boys_coefficients.hpp`,
  `external/boys/tests/data/boys_reference.csv`) are checked by the
  same `boys-reference-regeneration` test, which drives the submodule's
  generator (`external/boys/tools/gen_boys_coefficients.py --check`)
  with explicit paths. The monorepo never regenerates or edits submodule
  files; fix upstream.
