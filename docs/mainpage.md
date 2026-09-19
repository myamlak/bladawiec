# qcx API reference

qcx is a C++23 quantum-chemistry framework. It reads Gaussian basis sets in NWChem
format, evaluates the one- and two-electron Gaussian integrals those basis sets define,
detects a molecule's point-group symmetry, builds DFT integration grids, solves the
self-consistent field equations, persists integrals and restart state, and derives
post-SCF properties from a converged density.

The modules form a dependency-enforced DAG and each builds as its own library
(`qcx-core`, `qcx-integrals`, ...), so a program that needs only basis sets and
integrals links those and nothing else.

Two conventions hold across the whole public API.

- **Errors are values.** Every fallible operation returns \ref qcx::Result — a
  `std::expected` holding either the value or a \ref qcx::Error. Nothing in the
  framework throws. \ref qcx::Error pairs a \ref qcx::ErrorCode category with a
  message, and \ref qcx::ToString names the category.
- **Construction is fallible.** A type that cannot exist without validating its inputs
  has no public constructor; it exposes a static `Create(...)` returning
  `qcx::Result<T>`. Every container, engine and builder below is created that way.

## Entry points

| Entry point | What it is |
| --- | --- |
| \ref qcx::io::ParseRunInputFile, \ref qcx::io::ValidateInput, \ref qcx::driver::RunDriver | the file-driven path: parse a TOML run description, check it, run it end to end |
| \ref qcx::io::RunInput, \ref qcx::io::RunResult, \ref qcx::io::SerializeRunResultJson | the input schema, the result it produces, and the JSON writer |
| \ref qcx::molecule::Molecule, \ref qcx::molecule::Atom | a molecule: validated atoms carrying Bohr coordinates |
| \ref qcx::molecule::Connectivity, \ref qcx::molecule::FindElement | bond connectivity, and the periodic-table lookup |
| \ref qcx::basisset::BasisSet, \ref qcx::basisset::Shell | a parsed basis set and the contracted shells it holds |
| \ref qcx::symmetry::SymmetryAnalysis, \ref qcx::symmetry::PointGroup, \ref qcx::symmetry::SalcSet | point-group detection, the Abelian group used for computation, and symmetry-adapted combinations |
| \ref qcx::integrals::BuildOverlapMatrix, \ref qcx::integrals::BuildKineticMatrix, \ref qcx::integrals::BuildNuclearAttractionMatrix, \ref qcx::integrals::BuildDipoleMatrix | the one-electron matrices |
| \ref qcx::integrals::ComputeEriBatch, \ref qcx::integrals::BuildEriTensor, \ref qcx::integrals::ComputeEriBatchCertified | two-electron repulsion integrals: batched, dense, or through the certified mixed-precision pipeline |
| \ref qcx::integrals::DirectJkFockBuilder, \ref qcx::integrals::RiFullFockBuilder, \ref qcx::integrals::QfmmJBuilder, \ref qcx::integrals::IncrementalFockBuilder, \ref qcx::integrals::LeanDirectFockBuilder | Fock-matrix builders |
| \ref qcx::scf::RunRhfScf, \ref qcx::scf::RunUhfScf | the closed-shell and unrestricted SCF drivers |
| \ref qcx::scf::DiisExtrapolator, \ref qcx::scf::BuildSadGuess, \ref qcx::scf::BuildGwhGuess | convergence acceleration and the initial guesses |
| \ref qcx::grid::MolecularGrid, \ref qcx::grid::AoEvaluator, \ref qcx::grid::XcGridEngine | the DFT integration grid, point-wise AO evaluation, and the exchange-correlation engine |
| \ref qcx::storage::EriStore, \ref qcx::storage::SaveScfCheckpoint, \ref qcx::storage::LoadScfCheckpoint | on-disk integrals and SCF restart |
| \ref qcx::properties::AnalyzePopulations, \ref qcx::properties::AnalyzeQtaim, \ref qcx::properties::AnalyzeNocvEts, \ref qcx::properties::AnalyzeEspCharges | populations, QTAIM, ETS-NOCV, and ESP charges |
| \ref qcx::linalg::DenseMultiply, \ref qcx::linalg::MultiplyBatched, \ref qcx::linalg::Eigenpair | the linear-algebra seam: dense products, batched products, eigenpairs |
| \ref qcx::memory::Tensor, \ref qcx::memory::DeviceBuffer, \ref qcx::memory::WorkspaceBudget | dense storage and workspace accounting |
| \ref qcx::backend::CpuTag, \ref qcx::backend::CudaTag, \ref qcx::backend::DetectTopologyProfile | the execution backends and the machine probe |

## Accuracy

The framework has one user-facing accuracy word — \ref qcx::integrals::AccuracyPreset —
and derives every threshold the engines use from it, so accuracy is chosen once rather
than per subsystem:

| Preset | J/K energy budget | Schwarz | Density | Certified fp32 gate | QFMM theta | QFMM order |
| --- | --- | --- | --- | --- | --- | --- |
| `kLoose` | 1e-6 Eh | 1e-8 | 1e-8 | 1e-8 | 0.45 | 5 |
| `kNormal` (default) | 1e-10 Eh | 1e-10 | 1e-10 | 1e-10 | 0.3 | 5 |
| `kTight` | 1e-12 Eh | 1e-12 | 1e-12 | 0 (off) | 0.0 | 0 |

\ref qcx::integrals::SchwarzThreshold, \ref qcx::integrals::DensityThreshold and
\ref qcx::integrals::MixedPrecisionThreshold are the mappers behind those columns, and
\ref qcx::integrals::ThetaForPreset the QFMM column. A gate of zero disables the
single-precision path rather than tightening it, which is why `kTight` reproduces the
strict fp64 path.
The two QFMM columns are the near/far classification and multipole order; a theta of
zero degenerates to the near-field-only path.

## Module DAG

A module may use the modules to its left and none to its right. Upward and sideways
dependencies fail the build, so the arrows below are the whole of the allowed
dependency surface.

```
core -> backend -> memory -> linalg -> molecule -> symmetry -> basisset
     -> grid -> integrals -> scf -> storage -> io -> properties -> driver
```

| Module | API group | What it provides |
| --- | --- | --- |
| `qcx-core` | \ref qcx-core, \ref qcx-log | errors as values, result types, logging |
| `qcx-backend` | \ref qcx-backend | execution backends and machine probing |
| `qcx-memory` | \ref qcx-memory | dense tensors, device buffers, workspace accounting |
| `qcx-linalg` | \ref qcx-linalg | dense and batched products, iterative eigenproblems, sparse solves |
| `qcx-molecule` | \ref qcx-molecule | atoms, geometry, elements, connectivity |
| `qcx-symmetry` | \ref qcx-symmetry, \ref qcx-symmetry-salc | point-group detection, character tables, SALCs |
| `qcx-basisset` | \ref qcx-basisset | NWChem-format basis parsing |
| `qcx-grid` | \ref qcx-grid | radial and angular quadrature, Becke partitioning, XC integration |
| `qcx-integrals` | \ref qcx-integrals | one- and two-electron integrals, screening, Fock builders |
| `qcx-scf` | \ref qcx-scf | RHF and UHF drivers, DIIS, initial guesses, symmetry labelling |
| `qcx-storage` | \ref qcx-storage | on-disk integral stores and checkpoints |
| `qcx-io` | \ref qcx-io | the TOML input schema, validation, JSON results, Molden export |
| `qcx-properties` | \ref qcx-properties | population analyses, moments, QTAIM, ETS-NOCV, EDDB, ESP |
| `qcx-driver` | \ref qcx-driver | end-to-end run orchestration and builder selection |

## References

The algorithms, kernels and tabulated data in this framework derive from the
following published work. Bibliographic details are kept in `CITATION.bib`.

- S. F. Boys, *Electronic Wave Functions. I. A General Method of Calculation for the
  Stationary States of any Molecular System*, Proceedings of the Royal Society of
  London. Series A 200 (1950) 542–554.
- I. Shavitt, *The Gaussian Function in Calculations of Statistical Mechanics and
  Quantum Mechanics*, Methods in Computational Physics 2 (1963) 1–45.
- L. E. McMurchie and E. R. Davidson, *One- and two-electron integrals over Cartesian
  Gaussian functions*, Journal of Computational Physics 26 (1978) 218–231.
- T. Helgaker, P. Jørgensen and J. Olsen, *Molecular Electronic-Structure Theory*,
  John Wiley & Sons, Chichester (2000).
- C. C. J. Roothaan, *New Developments in Molecular Orbital Theory*, Reviews of Modern
  Physics 23 (1951) 69–89.
- P.-O. Löwdin, *On the Non-Orthogonality Problem Connected with the Use of Atomic
  Wave Functions in the Theory of Molecules and Crystals*, The Journal of Chemical
  Physics 18 (1950) 365–375.
- P. Pulay, *Convergence acceleration of iterative sequences. The case of SCF
  iteration*, Chemical Physics Letters 73 (1980) 393–398.
- V. R. Saunders and I. H. Hillier, *A "Level-Shifting" Method for Converging Closed
  Shell Hartree–Fock Wave Functions*, International Journal of Quantum Chemistry 7
  (1973) 699–705.
- M. Häser and R. Ahlrichs, *Improvements on the Direct SCF Method*, Journal of
  Computational Chemistry 10 (1989) 104–111.
- J. Almlöf, K. Fægri and K. Korsell, *Principles for a Direct SCF Approach to
  LCAO–MO Ab Initio Calculations*, Journal of Computational Chemistry 3 (1982)
  385–399.
- M. Wolfsberg and L. Helmholz, *The spectra and electronic structure of the
  tetrahedral ions MnO4-, CrO4--, and ClO4-*, The Journal of Chemical Physics 20
  (1952) 837–843.
- N. Luehr, I. S. Ufimtsev and T. J. Martínez, *Dynamic Precision for Electron
  Repulsion Integral Evaluation on Graphical Processing Units (GPUs)*, Journal of
  Chemical Theory and Computation 7 (2011) 949–954.
- W. Kabsch, *A solution for the best rotation to relate two sets of vectors*, Acta
  Crystallographica Section A 32 (1976) 922–923.
- F. A. Cotton, *Chemical Applications of Group Theory*, 3rd edition, Wiley (1990).
- V. I. Lebedev and D. N. Laikov, *A Quadrature Formula for the Sphere of the 131st
  Algebraic Order of Accuracy*, Doklady Mathematics 59 (1999) 477–481.
- A. D. Becke, J. Chem. Phys. 88 (1988) 2547; R. Stratmann, G. Scuseria and M. Frisch,
  Chem. Phys. Lett. 257 (1996) 213; J. C. Slater, J. Chem. Phys. 41 (1964) 3199 —
  the fuzzy-cell partition and the Bragg-Slater radii it is built from.
- R. S. Mulliken, *Electronic Population Analysis on LCAO–MO Molecular Wave
  Functions. I*, The Journal of Chemical Physics 23 (1955) 1833–1840.
- I. Mayer, *Charge, bond order and valence in the ab initio SCF theory*, Chemical
  Physics Letters 97 (1983) 270–274.
- F. L. Hirshfeld, *Bonded-atom fragments for describing molecular charge densities*,
  Theoretica Chimica Acta 44 (1977) 129–138.
- C. M. Breneman and K. B. Wiberg, *Determining atom-centered monopoles from molecular
  electrostatic potentials. The need for high sampling density in formamide
  conformational analysis*, Journal of Computational Chemistry 11 (1990) 361–373.
- R. F. Nalewajski and J. Mrozek, *Modified valence indices from the two-particle
  density matrix*, Canadian Journal of Chemistry 74 (1996) 1121–1130.
- M. P. Mitoraj, A. Michalak and T. Ziegler, *A combined charge and energy
  decomposition scheme for bond analysis*, Journal of Chemical Theory and Computation
  5 (2009) 962–975.
- D. W. Szczepanik, *On the three-center orbital projection formalism within the
  electron density of delocalized bonds method*, Computational and Theoretical
  Chemistry 1100 (2017) 13–17.
- D. Demidov, *AMGCL: An Efficient, Flexible, and Extensible Algebraic Multigrid
  Implementation*, Lobachevskii Journal of Mathematics 40 (2019) 535–546.

## Indexes

- [Class index](annotated.html) — every documented class and struct.
- [File index](files.html) — the headers, by module.

Two companion pages ship alongside this reference: **the API guide**, which states what a
caller may rely on and how to choose between the interchangeable builders, and **the
maintainer guide**, which covers building, testing and extending the framework.
