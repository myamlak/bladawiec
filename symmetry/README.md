# symmetry — libqcx-symmetry

Point-group detection and the SALC machinery: the permutation-based point-group
detector, character tables, and general SALC construction. The module sat
consumed-by-nobody until symmetry-blocked SCF diagonalization was wired through
`scf/`; the integral-side symmetry reduction
(petite lists, pair classes) lives in `integrals/` and `scf/`, not here. This
module does **not** reduce integrals and does not run SCF.

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

6 of 14 — depends on: linalg, molecule — consumed by: scf. Machine-checked by
`tools/check_dag.py`; this line is a summary, not the source of truth.

## Public headers / key types

- `symmetry/include/qcx/symmetry/`: `detection.hpp`, `point_group.hpp`,
  `point_group_name.hpp`, `character_tables.hpp`, `salc.hpp`
- `DetectPointGroup` + `SymmetryAnalysis` — the permutation-based detector.
- `PointGroup` — the detected-group enum.
- `GenerateSalcs` / `SalcSet` — the general SALC construction.
- `LargestAbelianSubgroup` — reduces the detected group to the computational
  group reported in `SymmetryAnalysis`.
