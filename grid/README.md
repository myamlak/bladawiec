# grid — libqcx-grid

The molecular integration-grid module: Lebedev angular grids (all
sixteen orders), Euler–Maclaurin radial grids, per-atom product grids, the
Becke/SSF atomic partition, the molecular grid that combines them, and the
AO-at-point evaluator (`N_l(a)` normalization convention). Its consumers (the
density-grid charges, ESP fits) live in `properties/`. The quadratures are
cross-validated against pyscf. It does **not** evaluate integrals (that is
`integrals/`) and does not run SCF.

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

8 of 14 — depends on: core, molecule, basisset — consumed by: properties.
Machine-checked by `tools/check_dag.py`; this line is a summary, not the source
of truth.

## Public headers / key types

- `grid/include/qcx/grid/`: `angular_grid.hpp`, `radial_grid.hpp`,
  `atomic_grid.hpp`, `molecular_grid.hpp`, `partition_function.hpp`,
  `ao_evaluator.hpp` (+ `internal/lebedev_tables.hpp`, `internal/solid_harmonics.hpp`)
- `AngularGrid` — the Lebedev quadrature.
- `RadialGrid` — the Euler–Maclaurin quadrature.
- `MolecularGrid` — the combined per-atom product grid.
- `AoEvaluator` — AO values at arbitrary points (the `N_l(a)` convention).

## Generation / regeneration notes

- `grid/include/qcx/grid/internal/lebedev_tables.hpp` and
  `internal/solid_harmonics.hpp` are generated from first principles by
  `tools/grid/gen_lebedev.py` and `tools/grid/gen_solid_harmonics.py`
  (`--check` mode verifies them) — never hand-edit the generated tables.
