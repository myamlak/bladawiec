# basisset — libqcx-basisset

The Gaussian basis-set module: the vendored BSE corpus (NWChem text, the single
supported input format), `BasisSet` with shells and ECPs, and basis I/O.
BSE-JSON and Gaussian94 parsers were deliberately dropped. It does **not**
select aux-basis families at runtime — that lives in
`integrals/include/qcx/integrals/aux_basis.hpp` — and does not compute
integrals.

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

7 of 14 — depends on: molecule — consumed by: grid, integrals, scf, io,
properties, driver. Machine-checked by `tools/check_dag.py`; this line is a
summary, not the source of truth.

## Public headers / key types

- `basisset/include/qcx/basisset/basis_set.hpp`
- `BasisSet` — the parsed basis set (normalized per primitive).
- `Shell` — one contraction row of one angular-momentum class.
- `ElementBasis` — per-element shell/ECP collections.
- `EcpDefinition` — effective-core-potential entries.

## Generation / regeneration notes

- The corpus lives in `data/basis` (with `data/basis/manifest.json`) and is
  verified by `tools/fetch_basis_data.py --check` (`basis-corpus-integrity`
  test, basisset/CMakeLists.txt). Regeneration is a network operation — never
  edit vendored basis files by hand.
