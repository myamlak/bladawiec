# basisset — libqcx-basisset

The Gaussian basis-set module: the vendored BSE corpus (NWChem text, the single
supported input format), `BasisSet` with shells and ECPs, and basis I/O.
BSE-JSON and Gaussian94 parsers were deliberately dropped, leaving NWChem text
as the single supported format. It does **not** select aux-basis families at
runtime — that lives in
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

## Rationale pointers

- The module owns data, not algorithms: it parses and serves the vendored
  corpus and computes nothing.
- Exactly one parser family is supported. Anything that would add a second
  input format adds a second source of truth for the same numbers.
- Aux-basis family auto-selection is a consumer concern, not a basisset one,
  so it lives beside the integrals that need it.

## Generation / regeneration notes

- The corpus lives in `data/basis` (with `data/basis/manifest.json`) and is
  verified by `tools/fetch_basis_data.py --check` (`basis-corpus-integrity`
  test, basisset/CMakeLists.txt). Regeneration is a network operation owned by
  the `qcx-basis-refresh` skill — never edit vendored basis files by hand.
