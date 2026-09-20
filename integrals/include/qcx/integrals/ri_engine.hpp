#pragma once

/// \file
/// The 3-center RI-J engine: the (uv|P) integrals over an auxiliary basis
/// through the matrix-form MD
/// kernels (md_vrr_3c.hpp - the VRR and transform machinery are shared
/// with the 2e engine; only the single-shell ket pair data and the
/// canonicalization-free assembly are 3c-specific), the (P|Q) metric, and
/// the RI-J Fock builder: J_uv = sum_P (uv|P) w_P with M w = v,
/// v_P = sum_uv (uv|P) D_uv - combined with the direct exchange of
/// fock_build.hpp (buildExchangeOnly). [arXiv:2210.03192]

#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace qcx::integrals {

/// RI engine settings.
/// \ingroup qcx-integrals
struct RiEngineOptions {
    AccuracyPreset accuracy = AccuracyPreset::kNormal; ///< Passed to the direct-exchange side.
    std::size_t maxBatchBytes = 512 * 1024 * 1024; ///< Per-class batch cap (512 MB).
    /// Route the two n2 x nAux RI contractions (v = I^T d, J = I w) through
    /// cuBLAS instead of the CPU BLAS seam. The metric's
    /// floored eigen-inverse stays Eigen on the CPU in both cases;
    /// only the plain products move. Meaningless in a CUDA-less build -
    /// the CPU seam runs regardless.
    bool useCudaGemm = false;
    /// Relative floor for the (P|Q) metric inverse: eigenvalues below
    /// metricFloorEpsilon * lambdaMax (lambdaMax = largest metric
    /// eigenvalue) are zeroed in the inverse - the kMetricFloorEpsilon
    /// constant's semantics, now per-engine configurable.
    /// The default equals kMetricFloorEpsilon = 1e-10, so the default
    /// behavior is bit-identical to the constant's own value. Zero disables
    /// the floor; the Create-time degenerate guard (lambdaMax <= 0 or no
    /// positive inverse eigenvalue) still rejects a fully degenerate
    /// metric.
    double metricFloorEpsilon = 1e-10;
    /// The adaptive-memory workspace budget (fock_build.hpp
    /// FockBuildOptions::workspaceBudget - same contract): when set, Create
    /// decides the build mode from the Create-time footprint estimate
    /// (tensor, metric, pair stores, task list and the direct-exchange
    /// half's own estimate) against the budget's remaining bytes. The
    /// budget is shared by pointer with the nested direct-exchange builder
    /// (its Create-time reservation orders after this one - nesting order =
    /// reservation order); the pointer is MUTABLE (Reserve charges the
    /// cumulative counter). Null (the default) keeps the legacy behavior
    /// exactly. The pointer must outlive the Create call.
    qcx::memory::WorkspaceBudget* workspaceBudget = nullptr;
    /// Engage the light rung directly (the per-iteration 3c recompute mode:
    /// no (uv|P) tensor is materialized, BuildFock recomputes the two RI
    /// contractions batch-at-a-time against a maxBatchBytes-class buffer).
    /// The budget seam selects the light rung on its own at nt84-class
    /// scales (the tensor term must dominate the residual quadratic terms -
    /// small systems never reach the window through the real estimate), so
    /// the flag is the unit-scale validation seam: the L1 tests force it to
    /// compare the recomputed Fock matrix against the materialized fast
    /// path within the per-preset accuracy budget. With a budget set, the light
    /// rung's own estimate and reservation still run (the fast-path attempt
    /// is skipped); RiMatrix() returns the empty matrix in this mode.
    bool forceLightRung = false;
    /// The point-group reduction over the ORBITAL basis (the machinery's
    /// classification seam: qcx::scf::BuildSymmetryReduction).
    /// Null (the default) = no symmetry. The reduction is the caller's: the
    /// builder retains no reference into it past the call.
    const SymmetryReduction* symmetryReduction = nullptr;
    /// The point-group reduction over the AUXILIARY basis - the SECOND
    /// instance the 3c task grid needs, because aux shells and aux
    /// functions live in their own index space with their own signed
    /// permutation (symmetry_reduction.hpp exposes no atom map, so the aux
    /// reduction cannot be derived from the orbital one). Built the same
    /// way (qcx::scf::BuildSymmetryReduction over the aux basis; the group
    /// order must match the orbital reduction's). Null (the default) = not
    /// supplied, which leaves the orbit expansion inert; supplying it
    /// WITHOUT symmetryReduction (or the other way round) is the
    /// kInvalidArgument Create refuses, because a joint orbit is one group
    /// acting on two bases and half of it is not a group action.
    const SymmetryReduction* auxSymmetryReduction = nullptr;
    /// Engage the joint task-grid ORBIT EXPANSION on the fast path
    /// (ruled 2026-09-13): the screened
    /// (braPair, auxShell) grid is walked to one representative per joint
    /// group orbit and the remaining members' blocks are expanded from the
    /// representative's - 3.41x fewer ERI blocks on the grid's own
    /// fixture, a BLOCK-COUNT ratio of a ONE-TIME Create-time pass (see
    /// ri_orbit_action.hpp; never a wall ratio, never a per-iteration
    /// saving on this rung).
    ///
    /// A PERMISSION, ON BY DEFAULT, and the default is set from the COUNTS.
    ///
    /// **The accounting, as identities, and what is MEASUREMENT in them.** The
    /// expansion removes EVALUATIONS and removes nothing from the WRITE: every
    /// surviving cell is scattered whether it was evaluated or expanded, so
    /// `copies == survivors - reps` and `scatters(engaged) ==
    /// scatters(disengaged)` exactly. Both identities are pinned by
    /// `RiOrbitExpansionTest.ExpansionTradesEvaluationsForCopiesAndAddsNoScatter`
    /// on a fixture this module's suite validates by independent walk.
    ///
    /// **The magnitudes below are the symmetry audit's, not a measurement this
    /// tree reproduces.** On c8h18/def2-SVP + def2-universal-jfit (C2h, order
    /// 4) that audit's probe measured a screened grid of 900,698 cells over
    /// 264,280 orbits, so **636,418 kernel evaluations removed** (-70.7%: each
    /// a VRR plus two Hermite-contraction transforms over the block's
    /// nI x nJ x nP elements) against 636,418 block copies of the SAME element
    /// counts (a member's block has its representative's element count - the
    /// shell closure, asserted in RiExpandOrbitMemberBlock and verified
    /// against the tensor by reconstruction), an O(order) representative test
    /// per surviving cell, and 108,240 B (~105.7 KiB) of retained tables. The
    /// probe was untracked and **no test in this tree
    /// reproduces those four magnitudes** - they are the one citation in this
    /// option's rationale without a fixture behind it, and reproducing them
    /// needs a c8h18/def2-SVP C2h fixture this module cannot build without
    /// hand-transcribing 202-entry reduction rows. Cite them as the audit's.
    ///
    /// **What the fix of 2026-09-15 changed in that picture.** The walk
    /// formerly expanded one member cell once per group element that landed on
    /// it, so a cell its own stabilizer carried two elements onto was copied
    /// and scattered twice and only the LAST write survived: the audit's
    /// 714,629 copies over 636,418 removed evaluations, and **78,211 duplicate
    /// scatters (+8.7%)**. The walk now skips an element a later one
    /// supersedes, which keeps the surviving write (the largest g that lands
    /// on the cell and clears the screen) byte for byte and drops every dead
    /// store. So the retired pair of numbers is `714,629 copies / 78,211
    /// duplicate scatters`, and the current ones DERIVE from the same audit
    /// probe by subtraction: **636,418 copies and 0 duplicate scatters**, with
    /// the scatter count now the plain walk's exactly. Every added term is
    /// Theta(the same sum of block elements) or O(1) per run.
    ///
    /// Engaged exactly when both reductions are supplied, both group orders
    /// exceed 1 and both reductions are non-trivial; with no reductions it is
    /// inert (every caller that ignores symmetry), and supplying exactly one
    /// is the kInvalidArgument Create refuses.
    ///
    /// The wall samples this default does NOT rest on (recorded as CLI
    /// samples on a machine with thirteen lanes building and testing): three
    /// paired runs had the expanded build 0.65 / 1.18 / 0.99 s SLOWER. They
    /// are not load-bearing - the added work is a memory-bound permutation
    /// copy while the removed work is a compute-bound Hermite/transform
    /// pipeline, so bandwidth contention penalises the added side by
    /// construction, which is the wrong instrument for choosing a default.
    /// What CAN overturn this default is a quiet-machine or min-of-N
    /// interleaved measurement of the per-unit cost of the two kinds of
    /// work, and only that.
    ///
    /// What it changes beyond the block count: the TENSOR-PASS occurrence's
    /// term counters (RiTermCounters x/p3/g3) count the blocks the
    /// engine EVALUATED, so they fall to the representative set - the
    /// honest count of the work done, and the observable the tests pin the
    /// reduction on. The tensor itself is filled at every surviving cell
    /// either way (the members by expansion), so nothing downstream of the
    /// tensor changes. Set false for the plain walk, byte for byte the
    /// pre-existing path.
    bool symmetryOrbitExpansion = true;
};

/// The RI-J run's term counters: the engine-emitted subset of
/// the cost-table backbone's ri_j term vector — exactly the five ids x, p3,
/// g3, qx, gx of the record contract (the run-level JSON's term_counters
/// block; s/p and qx_occ are model terms, never engine-emitted). Each
/// counter is a non-negative INTEGER per-run total over the whole SCF run
/// (not per iteration); a term whose work never occurred reads 0 — never
/// omission, never null.
///
/// The count moments are defined by this instrument (per attack note 3), by
/// occurrence class:
/// - The METRIC occurrence — once per builder at Create on every rung (the
///   full unscreened (P|Q) triangle, stripped or not): x += the distinct
///   phantom-pair store entries engaged (every aux shell — x == nAuxShells),
///   g3 += the |primPairs(bra)| x |primPairs(ket)| kernel weight summed over
///   the evaluated blocks. p3 stays untouched: the metric engages no
///   orbital pair.
/// - The TENSOR-PASS occurrence — one kernel evaluation of the screened 3c
///   task list: x += the distinct aux entries with a surviving task, p3 +=
///   the distinct orbital bra-pair entries with a surviving task, g3 += the
///   |primPairs(bra)| x |primPairs(aux entry)| kernel weight over the
///   surviving tasks. The fast/legacy path evaluates the list once, at
///   Create (inside BuildRiTensor); the light rung re-evaluates the
///   retained list twice per BuildFock call (the recompute's v pass and j
///   pass). Under the orbit expansion
///   (RiEngineOptions::symmetryOrbitExpansion) the TENSOR-PASS
///   occurrence counts the blocks the engine EVALUATED — the orbit
///   representatives — not the surviving cells they cover, so these three
///   counters fall with the block count rather than with the grid.
///   A caller that partitions the pass's task list into chunks
///   (BuildRiTensorChunk) must carry the orbital-pair dedup ACROSS the
///   chunks (RiScreenedPassState): counted on the standalone per-call
///   stamps, a partition reports each orbital pair once per chunk it
///   survives in, which is not the moment.
/// - qx/gx — the nested direct exchange's per-call counts summed over the
///   run's instrumented BuildFock calls (a statsOut was given): qx +=
///   fp64QuartetCount + fp32QuartetCount, gx += primitiveProductSum.
///
/// The 3-center occurrences count unconditionally; qx/gx count only on
/// instrumented calls. A snapshot is complete exactly when the driver's
/// run-level JSON gate held (an ri_j run with a trace file — every call
/// instrumented by construction); the RiJkFockBuilder::TermCounters
/// accessor on an uninstrumented run reads the complete 3c counts and
/// qx == gx == 0.
/// \ingroup qcx-integrals
struct RiTermCounters {
    std::size_t x = 0; ///< The aux pair-class engagements (the x term).
    std::size_t p3 = 0; ///< The screened 3c orbital pair-class engagements (the p3 term).
    std::size_t g3 = 0; ///< The 3c ERI kernel weight (the g3 term).
    std::size_t qx = 0; ///< The exchange's screened quartet count (the qx term).
    std::size_t gx = 0; ///< The exchange's ERI kernel weight (the gx term).
};

/// Builds the auxiliary-basis metric (P|Q) = (PP|QQ) through the 2e engine
/// over the diagonal-pair quartets. Symmetric positive definite by
/// construction; the RiJkFockBuilder applies its floored eigen-inverse
/// (the jfit-style metrics are extremely ill-conditioned; the near-null
/// eigenvalues are zeroed at the kMetricFloorEpsilon * lambdaMax floor).
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param auxBasisSet Auxiliary basis; every shell must satisfy
/// l <= kMaxEngineL (kUnimplemented otherwise) and every atom must have an
/// entry (kInvalidArgument otherwise).
/// \param options Batch settings; maxBatchBytes must be positive.
/// \param termCountersOut Optional term-counter sink: when given,
/// the metric occurrence's x/g3 contributions are added to it (p3 stays
/// untouched — the metric engages no orbital pair; qx/gx are per-call
/// exchange counters, never filled here). Null (the default) keeps the
/// zero-cost path.
/// \returns The metric as a rank-2 tensor with shape {nAux, nAux} in aux
/// function order, or an Error.
/// \ingroup qcx-integrals
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildAuxMetric(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& auxBasisSet,
    const RiEngineOptions& options = {},
    RiTermCounters* termCountersOut = nullptr);

/// Builds the same (P|Q) metric as BuildAuxMetric through the
/// blocked-metric rung's strip-wise build: the phantom-pair quartets are
/// processed in byte-capped strips over the metric rows (each strip closed
/// before the
/// next row's byte mass - the quartets' assembly lists plus the values -
/// would push it over options.maxBatchBytes; a row the cap alone cannot
/// hold becomes a forced single-row strip), and the scatter writes the
/// symmetric copy DIRECTLY into the returned Eigen matrix - the metric
/// memory Tensor, the TensorToEigen copy and the full values buffer of
/// the unblocked build never exist, at the cost of one values buffer per
/// strip. The result is bit-identical to BuildAuxMetric's on the same
/// inputs: every quartet is evaluated once by the same kernels against
/// the same pair store, and each element is written once per triangle
/// (the batch and strip partitions are value-neutral by contract).
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param auxBasisSet Auxiliary basis; every shell must satisfy
/// l <= kMaxEngineL (kUnimplemented otherwise) and every atom must have an
/// entry (kInvalidArgument otherwise).
/// \param options Batch settings; maxBatchBytes must be positive. The
/// workspace budget is NOT consulted here - the caller owns the
/// memory-fit decision (RiJkFockBuilder::Create's rung selection reserves
/// the estimate before the build).
/// \param termCountersOut Optional term-counter sink (the same
/// contract as BuildAuxMetric's): the metric occurrence's x/g3
/// contributions, deduplicated across the strip partition.
/// \returns The metric as an nAux x nAux Eigen matrix in aux function
/// order, or an Error.
/// \ingroup qcx-integrals
qcx::Result<Eigen::MatrixXd> BuildBlockedAuxMetric(const qcx::molecule::Molecule& molecule,
                                                   const qcx::basisset::BasisSet& auxBasisSet,
                                                   const RiEngineOptions& options = {},
                                                   RiTermCounters* termCountersOut = nullptr);

/// Builds the dense 3-center tensor (uv|P), shape {n, n, nAux} in orbital
/// and aux function order (shell_pairs.hpp). The tensor is materialized in
/// full; orbital-pair x aux-shell blocks whose Schwarz product Q_uv * Q_P
/// falls below the preset's cutoff are skipped and stay ZERO (the
/// density-free Create-time screen - the only screening on this
/// path, applied once at construction; BuildFock's density-weighted
/// screens are a direct-builder concept).
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Orbital basis; every shell must satisfy l <= kMaxEngineL
/// (kUnimplemented otherwise).
/// \param auxBasisSet Auxiliary basis (same contract as BuildAuxMetric).
/// \param options Batch settings; maxBatchBytes must be positive.
/// \param termCountersOut Optional term-counter sink: when given,
/// the tensor-pass occurrence's x/p3/g3 contributions are added to it
/// (one kernel evaluation of the screened task list). Null (the default)
/// keeps the zero-cost path.
/// \returns The 3c tensor, host-canonical, or an Error.
/// \ingroup qcx-integrals
qcx::Result<qcx::memory::Tensor<double, 3, qcx::backend::CpuTag>> BuildRiTensor(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::basisset::BasisSet& auxBasisSet,
    const RiEngineOptions& options = {},
    RiTermCounters* termCountersOut = nullptr);

/// The cross-chunk dedup state of a chunked 3-center build's term-counter
/// TENSOR-PASS occurrence (BuildRiTensorChunk's counter sink).
///
/// A chunked build partitions ONE kernel evaluation of the screened task
/// list: the chunk ranges partition the auxiliary shells, so an aux entry
/// (x) and a task (g3) belong to exactly one chunk, but an ORBITAL shell
/// pair recurs in every chunk whose aux range leaves it a surviving task.
/// Counted on the standalone per-call stamps
/// (internal/md_vrr_3c.hpp AccumulateScreenedRiPass, "one epoch per
/// evaluation") each chunk counts that pair afresh, so a chunked caller's
/// sum reads it once per chunk and the documented summation contract holds
/// for x and g3 only. This state carries the orbital pair stamps across the
/// calls instead - the caller-held pattern the blocked metric build already
/// uses for its strip partition (ri_engine.cpp AccumulateMetricBatches'
/// pairStamps).
///
/// Declare ONE state per chunked build, pass it to every chunk call of that
/// build, and drop it with the build: it is meaningful for one build only,
/// and one carried into a second would suppress that build's counts (clear
/// `braStamps` to restart the dedup). A null state (the default) keeps the
/// standalone per-call stamps - the monolithic build, the disk-backed route
/// and the crossover benchmark pass none and are unchanged.
/// \ingroup qcx-integrals
struct RiScreenedPassState {
    /// One stamp per orbital shell pair (0 = not yet engaged by this build's
    /// pass); grown to the pass's orbital pair count on first use.
    std::vector<std::size_t> braStamps;
};

/// Builds the (n^2 x k_c) contraction-layout slice of the dense 3-center
/// tensor (uv|P) for one contiguous range of aux SHELLS
/// [auxShellStart, auxShellEnd) - the chunked-build entry point of the
/// disk-backed rung. The shell indices are the aux pair-list shell order
/// (molecule-scoped, BuildShellPairs); the slice's columns are the range's
/// aux FUNCTIONS in global order: column c is the global aux function
/// auxPairList.shells[auxShellStart].functionOffset + c.
///
/// The contract for this entry point: the screened task list is
/// constructed PER CHUNK - the screen runs over the chunk's aux shells
/// only, with a chunk-scoped reserve (nOrbitalPairs x chunkShells x 16 B),
/// and the full nPairs x nAuxShells task list of the monolithic build is
/// never materialized. The chunk's values are bit-identical to the
/// monolithic BuildRiTensor's for the same tasks (per-block kernel outputs
/// do not depend on batch grouping - assembly only partitions work, and the
/// chunked-build tests pin it). The Create-time screen applies
/// unchanged; screened-out blocks stay ZERO (dense slice).
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Orbital basis (same contract as BuildRiTensor).
/// \param auxBasisSet Auxiliary basis (same contract as BuildAuxMetric).
/// \param auxShellStart First aux shell of the chunk (inclusive).
/// \param auxShellEnd One past the last aux shell of the chunk (must not
/// exceed the molecule-scoped aux shell count).
/// \param options Batch settings; maxBatchBytes must be positive. A NON-NULL
/// symmetryReduction or auxSymmetryReduction is REFUSED by name
/// (kUnimplemented) — this entry point reads neither, so accepting one would
/// be a silent substitution: a caller-supplied cell that changed
/// nothing. Both fields are refused together, so a half-supplied pair is
/// covered by the same sentence. RiFullFockBuilder::Create carries the
/// matching refusal for its own callers.
/// \param termCountersOut Optional term-counter sink, the same
/// contract as BuildRiTensor's: the chunk's TENSOR-PASS occurrence's x/p3/g3
/// contributions, accumulated by the shared rule
/// (internal/md_vrr_3c.hpp AccumulateScreenedRiPass) over the tasks the
/// chunk actually evaluates. A full-range call reports the monolithic
/// build's totals, and a chunked caller sums its chunks' - for x and g3 by
/// construction (the chunk ranges partition the aux shells and the tasks),
/// for p3 only when the caller threads \p passState across the calls (the
/// orbital pairs are not partitioned; RiScreenedPassState states why). Null
/// (the default) keeps the zero-cost path. The RI-K path
/// (ri_full_fock.hpp RiFullFockBuilder) is the sink's consumer: without it
/// that builder could not populate a single term.
/// \param passState Optional cross-chunk dedup state (RiScreenedPassState):
/// the caller's orbital pair stamps, shared by every chunk call of ONE
/// chunked build, which is what makes a chunked caller's p3 sum the
/// monolithic total. Null (the default) keeps the standalone per-call
/// stamps, and a chunked caller that passes none reads each pair once per
/// chunk it survives in.
/// \returns The n^2 x k_c contraction-layout slice (rows u*n + v, columns
/// the chunk's aux functions), or an Error (kInvalidArgument for an empty
/// or out-of-range aux shell range or a zero maxBatchBytes).
/// \ingroup qcx-integrals
qcx::Result<Eigen::MatrixXd> BuildRiTensorChunk(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const qcx::basisset::BasisSet& auxBasisSet,
                                                std::size_t auxShellStart,
                                                std::size_t auxShellEnd,
                                                const RiEngineOptions& options = {},
                                                RiTermCounters* termCountersOut = nullptr,
                                                RiScreenedPassState* passState = nullptr);

/// Builds F(D) = H + 2J_RI(rho) - K(rho) for closed-shell RHF: the RI
/// Coulomb part from the precomputed (uv|P) tensor and the metric's floored
/// eigen-inverse, the exchange part from the direct builder (fock_build.hpp)
/// in its exchange-only mode. The input density is the SPATIAL closed-shell
/// density rho = D/2 (spin-summed density over 2) - the direct/RI builder
/// convention; the scf FockBuilderFn seam hands over the spin-summed D, so
/// adapters must scale by 1/2 before calling BuildFock (rhf.hpp documents
/// the seam contract).
///
/// Create precomputes the 3c tensor and the metric eigen-inverse once; each
/// BuildFock call runs the direct exchange build plus the two RI
/// contractions (v = I^T D, J = I w) - no integrals are evaluated per
/// iteration.
/// \ingroup qcx-integrals
class RiJkFockBuilder {
public:
    /// Prepares the builder: the 3c tensor, the metric's floored
    /// eigen-inverse, and the direct-exchange builder.
    /// \param molecule Molecule providing the atom coordinates (Bohr).
    /// \param basisSet Orbital basis (same contract as BuildRiTensor).
    /// \param auxBasisSet Auxiliary basis (same contract as BuildAuxMetric).
    /// \param coreHamiltonian H = T + V, host-canonical rank-2 tensor with
    /// shape {n, n}.
    /// \param options Build settings; maxBatchBytes must be positive.
    /// \returns The builder, or an Error.
    static qcx::Result<RiJkFockBuilder> Create(
        const qcx::molecule::Molecule& molecule,
        const qcx::basisset::BasisSet& basisSet,
        const qcx::basisset::BasisSet& auxBasisSet,
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
        const RiEngineOptions& options = {});

    /// Builds F(D) = H + 2J_RI(rho) - K(rho) for the given closed-shell
    /// density.
    /// \param density The SPATIAL closed-shell density rho = D/2 (spin-summed
    /// density over 2), host-canonical rank-2 tensor with shape {n, n}.
    /// \param statsOut Optional per-call diagnostics sink (fock_build.hpp
    /// FockBuildStats): the nested direct-exchange build's quartet counts
    /// are forwarded when a pointer is given; null keeps the zero-cost
    /// path. The RI contractions contribute no quartet counts (they are
    /// dense n^2 x nAux products, not integral evaluations).
    /// \returns The Fock matrix as a rank-2 tensor, or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildFock(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        FockBuildStats* statsOut = nullptr) const;

    /// Builds the RI-J half on its own: the Coulomb term 2 J_RI(rho) the
    /// fused BuildFock adds to the exchange half, with nothing else in it -
    /// no core Hamiltonian and no exchange. This is the half an unrestricted
    /// run's per-spin assembly contracts ONCE on the spin-summed density
    /// (the J(P_alpha + P_beta) of the UHF Fock), the same way the direct
    /// family's `buildCoulombOnly` half serves
    /// MakeDirectUhfFockBuilder (fock_build.hpp).
    ///
    /// **The core Hamiltonian is deliberately absent, and that is the
    /// accounting difference from the direct family's split halves**: there
    /// BOTH halves carry a full H and the per-spin assembly subtracts one
    /// copy per channel (the double-H trap). On this path the exchange-only
    /// call is the sole H carrier (the note in BuildFock's implementation:
    /// "the RI path contributes no H, the exchange-only call is the only H
    /// carrier"), so the assembly adds the two halves with no subtraction -
    /// exactly the RIJCOSX combination's accounting, and the reason the two
    /// compositions are not unified into one helper.
    /// \param density The density to contract, host-canonical rank-2 tensor
    /// with shape {n, n}; the caller halves it (0.5 (P_alpha + P_beta)) when
    /// it is assembling an unrestricted Fock, because the RI-J term is the
    /// 2 J(rho) of this convention, not a bare J.
    /// \param statsOut Optional per-call diagnostics sink (fock_build.hpp
    /// FockBuildStats). The RI contractions evaluate NO quartets, so this
    /// call WRITES a default (all-zero) struct rather than leaving a stale
    /// one behind - the measured zero is the structural record of the absent
    /// direct-exchange half, the same rule RiFullFockBuilder::BuildFock
    /// follows. Null (the default) skips the write.
    /// \returns The n x n Coulomb half 2 J_RI(rho), or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildCoulombOnly(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        FockBuildStats* statsOut = nullptr) const;

    /// Builds the exchange half on its own: the nested direct-exchange
    /// builder's result verbatim - H - K(rho), the exchange-only mode of
    /// fock_build.hpp's DirectJkFockBuilder that Create configured
    /// from this engine's own options - and nothing else in it. The
    /// unrestricted per-spin assembly contracts it ONCE PER SPIN on that
    /// spin's RAW density P_sigma (no halving: the exchange-only builder's
    /// -K is linear in its input, so P_sigma is the K(P_sigma) of the UHF
    /// channel, not the K(rho) of the closed shell).
    ///
    /// It is the SAME builder instance the fused BuildFock runs, so the K
    /// half of an unrestricted ri_j_link run is the K half of the restricted
    /// one - one accuracy preset, one batch cap, one certified-lane default,
    /// one shared workspace budget - rather than a second exchange wiring
    /// that could drift from it.
    /// \param density The spin density P_sigma, host-canonical rank-2
    /// tensor with shape {n, n}.
    /// \param statsOut Optional per-call diagnostics sink: forwarded to the
    /// nested exchange build, which fills the quartet counts; null keeps the
    /// zero-cost path.
    /// \returns The n x n exchange half H - K(rho), or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildExchangeOnly(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        FockBuildStats* statsOut = nullptr) const;

    /// The precomputed 3c tensor in the contraction layout: rows
    /// u*n + v, columns P, column-major (the layout of the two RI
    /// contractions). Exposed for persistence: the tensor
    /// is built once at Create and is reused by every correlated iteration.
    /// \returns The n^2 x nAux matrix; the empty 0 x 0 matrix when the
    /// light rung engaged (the tensor is never materialized there - the
    /// per-iteration recompute holds its inputs instead).
    const Eigen::MatrixXd& RiMatrix() const noexcept;

    /// The Create-time mode record (fock_build.hpp FockModeInfo); nullopt
    /// when no workspace budget was given (the legacy path - no decision).
    /// \returns The mode record, or nullopt on the legacy path.
    const std::optional<FockModeInfo>& ModeInfo() const noexcept;

    /// The NESTED direct-exchange half's own Create-time mode record
    /// (fock_build.hpp FockModeInfo): the rung that half selected at its own
    /// Create, from the band this engine's reservation left it. This
    /// engine's ModeInfo() reports the OUTER rung and the exchange charge it
    /// made, never the nested's own decision - and since the two are
    /// independent (the outer's tensor rung and the nested's pattern rung
    /// are decided on different objects), the nested's record is the only
    /// place its rung is named.
    /// \returns The nested builder's mode record, or nullopt when it ran the
    /// legacy path (no workspace budget).
    const std::optional<FockModeInfo>& ExchangeModeInfo() const noexcept;

    /// Whether Create ENGAGED the joint task-grid orbit expansion
    /// (RiEngineOptions::symmetryOrbitExpansion) - the disclosure that option
    /// owes: a switch no record can name is a switch nobody can check. True
    /// only under the exact condition the tables are built with the flag set,
    /// BOTH reductions supplied, and both group orders above 1 (a trivial
    /// group leaves the mechanism the identity, so it is not engaged); false
    /// on every other path, which includes the light rung - there the flag is
    /// REFUSED at Create by name, so no builder exists to ask.
    /// \returns True when the expansion ran on this builder's tensor pass.
    bool OrbitExpansionEngaged() const noexcept;

    /// The run's accumulated term counters (ri_engine.hpp
    /// RiTermCounters): the per-run totals over everything this builder's
    /// BuildFock calls (and its Create-time 3c build) counted so far. Read
    /// after the SCF run. The 3c occurrences count unconditionally; the
    /// qx/gx exchange occurrences count only on instrumented calls (a
    /// statsOut was given - see RiTermCounters for the completeness rule).
    /// \returns The run counters (builder copies share the state, so the
    /// counters accumulate across calls made through any copy).
    const RiTermCounters& TermCounters() const noexcept;

private:
    // The implementation state lives in the .cpp (Eigen stays
    // implementation-only).
    struct State;
    explicit RiJkFockBuilder(std::shared_ptr<const State> state);

    /// The RI-J contraction itself, in the fused call's own flat layout (the
    /// column-major u*n + v vector): v = I^T d, w = V diag(invLambda)
    /// V^T v through the floored metric inverse, then the 2 I w the Fock
    /// assembly adds. One code path serves the fused BuildFock and the
    /// split BuildCoulombOnly, so the two cannot drift about the rung they
    /// took, the counters they charged or the convention's factor of two;
    /// a successful call also adds the light rung's per-call pass partial
    /// (the v and j passes each walked the retained screened task list once)
    /// - the occurrence this half owns, counted here because here is where
    /// it happened.
    /// \param densityVector The flat density, rows u*n + v, column-major.
    /// \param n The matrix side length (the same n the tensor shape carries).
    /// \returns The flat 2 J_RI(rho) vector, or an Error.
    qcx::Result<Eigen::VectorXd> BuildRiCoulombVector(const Eigen::VectorXd& densityVector,
                                                      std::size_t n) const;

    std::shared_ptr<const State> _state;
};

} // namespace qcx::integrals
