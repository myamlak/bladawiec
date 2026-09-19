# Changelog

All notable changes to this project are documented in this file. The format
follows Keep a Changelog, and the project adheres to Semantic Versioning.

There is one entry per released version, and the history carries one
commit per version — so this file and the history are the same record read
two ways.

## [0.1.0] - 2026-09-18

The first release: the framework as a whole, under the BSD-3-Clause
license.

### Added

- **The module DAG.** `core → backend → memory → linalg → molecule →
  symmetry → basisset → grid → integrals → scf → storage → io → properties
  → driver`, with the dependency direction enforced by a check that runs in
  CI and in the commit hook rather than by convention.
- **core** — the `Result`/`std::expected` error type every fallible surface
  returns instead of throwing, and the logging front end over spdlog.
- **backend** — CPU and CUDA backend tags, CPU feature and topology
  detection, the compute-profile and memory-topology descriptions, and the
  scheduler seam built on the P2300 `std::execution` model.
- **memory** — the `Tensor` type and its policies, device buffers and
  pointers, sparsity patterns, and the workspace-budget accounting that
  makes allocation growth explicit.
- **linalg** — dense operations over Eigen with an optional vendor-BLAS
  path, batched operations, an iterative eigensolver seam (Spectra), and a
  sparse solver seam (AMGCL).
- **molecule** — geometry and element model, connectivity and mass
  properties.
- **symmetry** — point-group detection from geometry (Kabsch fit), the
  character and full-group tables, and symmetry-adapted linear combinations.
- **basisset** — the Gaussian basis-set model over the vendored Basis Set
  Exchange corpus (25 families, orbital and auxiliary), with the matching
  rules that select a fitting set for a given orbital set.
- **grid** — radial grids (Murray-Handy-Laming), Lebedev angular grids,
  atomic and molecular grids, Becke partition functions, the AO evaluator
  and the exchange-correlation grid engine over the excgrid library.
- **integrals** — the Boys-function kernel with its certified
  double/float/half-precision and SIMD lanes, the one-electron engines
  (overlap, kinetic, nuclear attraction), the two-electron ERI engines
  (dense, batched, cached, CUDA), the general-angular-momentum builders,
  Schwarz screening and shell-pair handling, the Fock builders (direct,
  incremental, lean, QFMM, GPU) and the resolution-of-the-identity
  engines.
- **scf** — closed-shell RHF and unrestricted HF with their DIIS
  convergence acceleration, density and energy assembly, and symmetry
  reduction of the Fock and density matrices.
- **storage** — screened and cached ERI stores, resolution-of-the-identity
  tensor stores and chunked disk-backed Fock builds, and SCF
  checkpoint/restart with a versioned schema.
- **io** — the TOML run-input schema with validation, the JSON result
  block, Molden export, and the selection-coefficient output.
- **properties** — population and charge analyses, multipole moments,
  electrostatic potential, density at the nuclei, QTAIM, Fukui functions,
  NOCV and the electron-density-of-delocalized-bonds analyses.
- **driver** — the `qcx run` command line: parse, validate, solve, report.
- **Build and dependencies.** CMake presets for the Windows/MSVC build (and
  its AddressSanitizer and CUDA variants), precompiled headers on every
  target, and dependencies pinned as git submodules — see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Tests and benchmarks.** Per-module test suites with their reference
  values and boundary cases, the end-to-end fixtures, and the scaling
  benchmark set.
- **Documentation.** The Doxygen API reference, built with `EXTRACT_ALL`
  off and gating on zero warnings; the front page is the README.
- **License and notices.** BSD-3-Clause ([LICENSE](LICENSE)), the
  third-party obligations in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
  the citation file ([CITATION.bib](CITATION.bib)), this changelog, and the
  contribution rules ([CONTRIBUTING.md](CONTRIBUTING.md)).
