# API guide

This page is the boundary contract between qcx and the code that calls it: the error
rule, the construction rule, the units, the accuracy ladder, and how to choose between
the interchangeable builders. The generated reference beside it gives the signatures;
this page gives the rules that are not visible in a signature.

## 1. The error contract

Every fallible operation returns `qcx::Result<T>`, and nothing in the framework throws.

```cpp
namespace qcx {

enum class ErrorCode : std::uint8_t {
    kInvalidArgument,    // An argument does not satisfy the operation's contract.
    kConvergenceFailure, // An iterative procedure failed to converge within its budget.
    kIOError,            // File or device I/O failed.
    kUnimplemented,      // The requested feature is not implemented (yet).
    kDeviceError,        // A device (GPU or accelerator) operation failed.
    kOutOfMemory,        // An allocation failed (host or device memory exhausted).
    kInternalError,      // An unexpected internal failure (e.g. a third-party library threw).
};

struct Error {
    ErrorCode code;      // Category of the error.
    std::string message; // Human-readable description (English, no trailing newline).
};

template <typename T> using Result = std::expected<T, Error>;

std::string_view ToString(ErrorCode code);
}
```

Four rules follow from this and they hold everywhere:

- **Failure is a return value, not a control-flow exception.** There is no try block to
  write around a qcx call, and no path on which an error escapes as a thrown object.
  The result is examined, not caught.
- **`Result<T>` is `std::expected<T, Error>`** — the happy path carries no overhead and
  the error path needs no exceptions. Public signatures spell the type `qcx::Result<T>`;
  `std::expected` appears only in that alias.
- **`ToString` names a category** and is the right thing to put in a log line.
  A code outside the known range — never produced by qcx itself — yields `"UnknownError"`.
- **`message` is for humans.** Do not parse it. Branch on `code`.

The category is what a caller switches on. `kInvalidArgument` means the caller's input
was rejected and re-running will not help; `kConvergenceFailure` means the numerics ran
out of budget and a different setting might; `kOutOfMemory` means an allocation or an
admission check refused, so a smaller problem or a different route is the answer;
`kInternalError` means an invariant was broken that the caller cannot influence.

## 2. Fallible construction

A type that cannot exist without validating its inputs has no public constructor. It
exposes a static factory instead:

```cpp
static Result<Thing> Thing::Create(/* validated inputs */);
```

Validation happens once, at the boundary, and the type's invariants hold from then on —
a `Thing` in hand is a `Thing` whose inputs were checked. The pattern is uniform across
the framework:

| Factory | File | Creates |
| --- | --- | --- |
| `qcx::molecule::Molecule::Create` | `molecule/include/qcx/molecule/molecule.hpp` | a molecule from atoms, Bohr coordinates, charge and multiplicity |
| `qcx::memory::Tensor::Create` | `memory/include/qcx/memory/tensor.hpp` | a dense tensor of a given shape, allocating now |
| `qcx::linalg::GemmKernel::Create` | `linalg/include/qcx/linalg/batched_ops.hpp` | a compiled (or prepared) `C = alpha A B + beta C` kernel |
| `qcx::grid::MolecularGrid::Create` | `grid/include/qcx/grid/molecular_grid.hpp` | the per-atom radial x angular integration grid |
| `qcx::integrals::DirectJkFockBuilder::Create` | `integrals/include/qcx/integrals/fock_build.hpp` | a prepared direct J/K Fock builder |
| `qcx::storage::EriStore::Create` | `storage/include/qcx/storage/eri_store.hpp` | a fresh on-disk integral store |

Two consequences are worth knowing before you call one. First, a factory that allocates
reports `kOutOfMemory` from the factory rather than failing later, so the error surfaces
where the cost is decided. Second, a factory is the only way in: if a type has no public
constructor, there is no half-valid instance to reason about.

`EriStore` is the one place with two entry points, because the two cases differ:
`Create` makes a fresh store file, and `Open` attaches to an existing one for append.

## 3. Units and conventions

| Quantity | Unit | Where it is stated |
| --- | --- | --- |
| Molecular coordinates | Bohr | `Molecule` carries Bohr coordinates |
| Coordinates read from input | the file's own unit | the input records the unit it read under, so the parser converts rather than assumes |
| Energies | Hartree | `HfResult::totalEnergy` is electronic plus nuclear repulsion; `electronicEnergy` is ½ Tr[D (H + F[D])], both Hartree |
| Accuracy budgets | Hartree | the preset targets are Eh |
| Schwarz and density thresholds | dimensionless | products of integral bounds, compared against them |
| QFMM theta | dimensionless | the near/far classification parameter |

Two type conventions are used in place of raw arrays, and a caller passing geometry
should follow them: fixed-size 3-vectors (coordinates, centers) are
`std::array<double, 3>`, and pointer-plus-count APIs take a `std::span`.

## 4. The accuracy rungs

One word selects the accuracy. `qcx::integrals::AccuracyPreset` is the single
user-facing knob, and every threshold the engines use is derived from it, so accuracy is
chosen once rather than per subsystem.

| Preset | J/K energy budget | Schwarz `SchwarzThreshold` | Density `DensityThreshold` | Certified fp32 gate `MixedPrecisionThreshold` | QFMM theta `ThetaForPreset` | QFMM order `LMultForPreset` |
| --- | --- | --- | --- | --- | --- | --- |
| `kLoose` | 1e-6 Eh | 1e-8 | 1e-8 | 1e-8 | 0.45 | 5 |
| `kNormal` (default) | 1e-10 Eh | 1e-10 | 1e-10 | 1e-10 | 0.3 | 5 |
| `kTight` | 1e-12 Eh | 1e-12 | 1e-12 | 0 (off) | 0.0 | 0 |

What each rung gates:

- **Schwarz** drops a shell quartet whose bound product `Q_ij * Q_kl` falls below the
  threshold. It bounds what the integral engine evaluates at all.
- **Density** additionally weights by the density-matrix element, so it drops quartets
  that are large but carry no density. It is the rung the direct Fock builders use.
- **The certified fp32 gate** is the budget for letting a quartet take the certified
  single-precision path; the engine admits a quartet to that path only while its a-priori
  density-weighted bound stays at or below the gate.
- **QFMM theta** is the well-separatedness parameter: smaller means a stricter near/far
  classification, more pairs computed exactly, more cost. A theta of zero degenerates to
  the near-field-only path.
- **QFMM order** is the multipole expansion order of the far field. Higher is more
  accurate and more costly per interaction.

Two behaviours are deliberate and worth stating plainly, because they are easy to
misread as bugs. A gate of **zero disables an evaluation path rather than tightening
it** — that is why `kTight` sets the fp32 gate to zero, so the strict preset reproduces
the fp64 path exactly. And the **budgets are targets, not delivered errors**: a preset
is a promise that the engine's thresholds are set to meet that budget, not a measurement
of the error your particular system will show.

The mappers are exhaustive by construction — each has no `default:` arm — so a build
that adds an enumerator is stopped by the compiler rather than silently answering with
the default preset's number.

## 5. The precision ladder

Orthogonal to the accuracy preset is *which* floating-point band an integral may be
computed in. `qcx::integrals::PrecisionBand` is the ladder, loosest first:

| Band | Arithmetic | Accumulation |
| --- | --- | --- |
| `kFp16` | fp16 (tensor cores, GPU only) | fp32 |
| `kFp32Certified` | fp32 through the certified pipeline | fp32 |
| `kFp32EvalFp64Accumulate` | fp32 evaluation | fp64 |
| `kFp64` | fp64 | fp64 — the reference arithmetic, and the one a cross-check uses |

`qcx::integrals::PrecisionCap` is the caller's clamp on that ladder:
`kAuto` (the policy decides), `kFp32Certified` (the fp16 and mixed bands clamp up to the
certified fp32 band), `kFp64` (everything runs in fp64).

The rule that matters: **the cap constrains the ladder, it never replaces it.** Setting
`kFp64` does not disable the accuracy policy — the thresholds, screening and convergence
behaviour of the chosen preset all still apply; only the arithmetic band is pinned. The
certified band's a-priori bound is
`eps * cClass * (1 + rounding) * Q_bra * Q_ket * |D|` with `eps = kCertifiedBandEpsilon`,
and the fp16 band carries a correspondingly larger epsilon, so it is harder to enter.

## 6. Choosing a Fock builder

If the input names no builder, the framework does not guess from a model — it follows a
size ladder:

| Basis functions | Tier |
| --- | --- |
| `<= kLeanTierMaxBasisFunctions` (1000) | the direct family's lean (Schwarz-only) member |
| 1000 < n `<= kRiJTierMaxBasisFunctions` (2000) | `ri_j_link` |
| n > 2000 | `qfmm` |

There is one exception, and it is keyed on the functional's character rather than on the
method: a **non-hybrid** (pure) Kohn-Sham run promotes `ri_j_link` above the lean member,
because a non-hybrid functional needs J and no K, so the RI-J path is the cheap one there
while a screened direct build would do work the method does not need. Hybrid Kohn-Sham
keeps the shared order, because it needs K. The caller states the fact — it is not
inferred from a functional name.

The builders themselves are interchangeable behind that choice, and each forms a
specific Fock matrix:

| Builder | Forms |
| --- | --- |
| `qcx::integrals::DirectJkFockBuilder` | `F(D) = H + 2J(rho) - K(rho)` for closed-shell RHF, J and K from the batched engine |
| `qcx::integrals::LeanDirectFockBuilder` | the same, built by streaming — the lean, Schwarz-only member |
| `qcx::integrals::RiFullFockBuilder` | RI-J and RI-K from a single traversal — the composed full-RI builder |
| `qcx::integrals::RiJkFockBuilder` | `F(D) = H + 2J_RI(rho) - K(rho)`: RI-J with direct exchange |
| `qcx::integrals::QfmmJBuilder` | `F(D) = H + 2J(rho)` from the QFMM skeleton — Coulomb only |
| `qcx::integrals::QfmmHfFockBuilder` | `F(D) = H + 2J(rho) - K(rho)`, composed from the QFMM Coulomb path |
| `qcx::integrals::IncrementalFockBuilder` | wraps a `DirectJkFockBuilder` and accumulates only the density-difference contribution |
| `qcx::integrals::GpuJkFockBuilder` | `F(D) = H + 2J(rho) - K(rho)` on the CUDA device |
| `qcx::storage::DiskRiFockBuilder` | streamed RI-J over chunked on-disk three-center integrals |

The budgeted `direct` machinery and both disk routes are explicit opt-ins rather than
ladder entries. Two named crossovers exist as constants —
`qcx::integrals::RecommendQfmmOverRiJ` / `kQfmmCrossoverBasisFunctionCount` for QFMM
against RI-J, and the two tier ceilings above. Both carry their own provenance in their
headers and are documented there as chosen boundaries rather than measured crossovers, so
read the constant that applies to your run instead of assuming a validated threshold.
`qcx::driver::ResolveBuilderSelection` resolves the run's builder and reports the
reasoning; `qcx::driver::EstimateCandidates` estimates and ranks every candidate, which is
the entry point to use if you want the estimate table rather than the ladder's answer.

If your builder needs an auxiliary basis, `qcx::integrals::SelectAuxBasis` picks the
bundled auxiliary set for an orbital basis name, and `qcx::integrals::FockBuilderKind`
says which builder that set will feed.

## 7. Choosing a storage tier

| Entry point | Use it when |
| --- | --- |
| `qcx::storage::EriStore` | you want one HDF5 store for one system: one-electron matrices, batch chunks, and a lookup by what it contains |
| `qcx::storage::CachedEriBatchEngine` | you want a drop-in decorator over the batch API that caches instead of recomputing, with read-vs-recompute counters |
| `qcx::storage::SaveRiTensor` / `LoadRiTensor` | you want the whole `n^2 x nAux` RI tensor persisted |
| `qcx::storage::DiskRiFockBuilder` | the RI-J build itself should stream from disk rather than hold the tensor |
| `qcx::storage::SaveScfCheckpoint` / `LoadScfCheckpoint` | you want to resume an SCF run rather than restart it |

`qcx::storage::ComputeFingerprint` is the deterministic system fingerprint — the store's
identity — and `qcx::storage::kStoreSchemaVersion` is the schema version the store
records, so a file written by an older schema is detectable rather than misread.

## 8. What a caller may rely on

- **The entry points named in this guide and in the reference are the public surface.**
  Anything in a `detail` or `internal` namespace is implementation detail and may change
  without notice.
- **Fallible operations return `qcx::Result<T>`.** A caller never needs a try block, and
  never gets a thrown exception out of the framework.
- **Types are created through their factory.** If a type has no public constructor, no
  other route produces a valid instance.
- **A converged run is gated by its own flag, not inferred.** `HfResult::converged` says
  whether a convergence criterion fired, and `iterations` says how many iterations were
  spent (the budget, when it did not converge). On a non-converged result the returned
  state holds the **final** iterate rather than an empty matrix — so a property computed
  from a non-converged run describes the last density, and `converged` remains the thing
  to check.
- **The accuracy preset is the contract.** Choosing `kNormal` states the budget the
  thresholds were set to meet; it is not a measurement of your system's error.
- **The version is recorded in three places** — `CMakeLists.txt`, `Doxyfile` and
  `CITATION.cff` — and the three are expected to agree. The project follows semantic
  versioning, so a change that breaks the surface described here is a major-version
  change.
