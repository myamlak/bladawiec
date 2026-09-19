# molecule — libqcx-molecule

The molecular-structure module: the validated move-only `Molecule` with
canonical renumbering, the full 118-element generated table (isotopes, Cordero
covalent + Alvarez vdW radii), connectivity (Boost.Graph plus the CSR primitive
consumed by the sparsity patterns), parallel mass properties, and nuclear
repulsion. It does **not** handle basis sets (that is `basisset/`) and does not
compute integrals.

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

5 of 14 — depends on: core, memory — consumed by: symmetry, basisset, grid,
integrals, scf, io, properties, driver. Machine-checked by
`tools/check_dag.py`; this line is a summary, not the source of truth.

## Public headers / key types

- `molecule/include/qcx/molecule/`: `molecule.hpp`, `elements.hpp`,
  `connectivity.hpp`, `mass_properties.hpp`
- `Molecule` — the validated, move-only molecular container.
- `ElementData` — the generated per-element table (`elements.hpp`).
- `Connectivity` / `ConnectivityCsr` — the bond-graph heuristic in both forms.
- `ComputeMassProperties` — parallel COM/inertia evaluation.

## Generation / regeneration notes

- `molecule/include/qcx/molecule/elements.hpp` is generated from
  `tools/periodic/elements_data.json` by `tools/periodic/generate_element_table.py`;
  the `elements-regeneration-deterministic` test (molecule/CMakeLists.txt) keeps
  it byte-identical. Regeneration is offline — never hand-edit the header.
