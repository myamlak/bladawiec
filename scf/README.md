# scf — libqcx-scf

The self-consistent-field module: RHF and UHF drivers (`RunRhfScf`,
`RunUhfScf`), density/energy evaluation, CDIIS acceleration (two-phase per-spin
DIIS and the robustness gates), starting guesses (SAD, GWH),
checkpoint round-trip via `scf_state`, and symmetry-blocked diagonalization.
The module does **not** build integrals itself — the `FockBuilderFn`
seam lets the integrals-side builders (`DirectJkFockBuilder`,
`RiJkFockBuilder`) drive the loop.

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

10 of 14 — depends on: core, memory, linalg, molecule, symmetry, basisset,
integrals — consumed by: storage, io, properties, driver. Machine-checked by
`tools/check_dag.py`; this line is a summary, not the source of truth.

## Public headers / key types

- `scf/include/qcx/scf/`: `rhf.hpp`, `uhf.hpp`, `diis.hpp`,
  `density_energy.hpp`, `scf_state.hpp`, `symmetry_reduction.hpp`
- `RunRhfScf` / `RunUhfScf` — the SCF entry points.
- `FockBuilderFn` / `UhfFockBuilderFn` — the builder-callback seam.
- `DiisExtrapolator` — the DIIS accelerator.
- `RhfDensityEnergy` / `UhfDensityEnergy` — density/energy evaluation.
- `HfResult` / `UhfResult` — the converged-state results.
