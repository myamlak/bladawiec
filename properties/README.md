# properties — libqcx-properties

The analysis module: population analyses (Mulliken, Löwdin,
Mayer bond orders, Gopinathan–Jug), electric moments (dipole + traceless
quadrupole), density-grid charges (Hirshfeld/Voronoi), CHELPG/MK ESP fits,
EDDB, Nalewajski–Mrozek bond orders, the Fukui response, and the ETS-NOCV
energy decomposition. It does **not** define the grids (that is `grid/`), does
not run SCF (that is `scf/`), and does not serialize results (that is `io/`).

Doxygen (the `qcx-docs` target) documents the API; this file is a map, not a
re-documentation.

## DAG position

13 of 14 — depends on: core, molecule, basisset, grid, integrals, scf —
consumed by: driver. Machine-checked by `tools/check_dag.py`; this line is a
summary, not the source of truth.

## Public headers / key types

- `properties/include/qcx/properties/`: `populations.hpp`, `multipoles.hpp`,
  `charges.hpp`, `esp.hpp`, `eddb.hpp`, `nalewajski.hpp`, `fukui.hpp`,
  `nocv.hpp`
- `AnalyzePopulations` — the four population analyses (S and S^1/2 shared).
- `AnalyzeMultipoles` — dipole + traceless quadrupole.
- `AnalyzeHirshfeld` / `AnalyzeVoronoi` — density-grid charges.
- `AnalyzeEspCharges` — the CHELPG/MK fits.
- `AnalyzeNocvEts` — the ETS-NOCV decomposition.

## Rationale pointers

- Scope stops at the analyses themselves: `grid/` defines the grids, `scf/`
  runs the SCF, `io/` serializes the results.
- Every analysis is implemented from its published formulation. A referee
  that reproduces another code's numbers validates; it does not license
  transcribing that code's expressions.
- The `[properties]` input section wires these analyses into a driver run.

## Generation / regeneration notes

- `tools/properties/*.py` are the numpy/pyscf cross-validation referees
  (e.g. `nocv_ref.py`) — self-documented manual workflows, not
  committed-data generators.
