#pragma once

/// \file
/// The lean direct Fock builder (the 2026-09-08 lean-direct
/// reframe): a Schwarz-only recompute loop over ONE immutable 24-byte
/// record per canonical shell pair (lean_pair_record.hpp) - no per-pair
/// task records, no retained neighbor CSR, no admission record, no
/// workspace budget. Each BuildFock call walks the records in bra-pair
/// (row) order, screens each bra row's full ket prefix IN THE LOOP
/// (Q_MN * Q_LS against SchwarzThreshold(accuracy) x kNeighborListSlack -
/// the existing machinery's neighbor-list cutoff verbatim), flushes the
/// surviving shell quartets through the existing class-batch machinery
/// (md_batch.hpp) into bounded per-thread tiles, and contracts every
/// quartet with the existing kernels' verbatim symmetry arithmetic
/// (fock_contract_kernel.hpp). No nested parallelism: the parallel
/// dimension is the row windows (round-robin rows, one window per chunk);
/// every flush runs single-threaded. Symmetry: the row walk enumerates
/// each unordered pair-pair exactly ONCE as its canonical cell, so every
/// unique shell quartet is computed exactly once and its 8-fold
/// symmetry-related Fock contributions accumulate in that one pass. When
/// the caller supplies an Abelian point-group reduction (the options'
/// symmetryReduction: the machinery's classification seam) each
/// pair record ADDITIONALLY carries the pair's point-group classification,
/// built at the same Create-time moment as the record and packed into it,
/// and the row walk drops the canonical cells that classification proves to
/// be exact-zero ERI blocks - cells a group element fixes with opposite
/// signs on the two pairs. That is a throughput-only saving (the dropped
/// contributions are exact zeros, so the bytes do not move); without a
/// reduction every record's mask bytes stay zero and no cell is dropped. The
/// mask's own measured reach is on the option's note below: it drops nothing
/// on any bench fixture.
/// With symmetryOrbitExpansion the walk goes further along the same
/// reduction: it visits ONE canonical cell per symmetry ORBIT (the
/// petite-list increment) and the contraction expands that
/// representative's ERI block over the orbit's members, so the ERI work
/// drops by the orbit multiplicity (3.70x fewer blocks on c8h18/def2-SVP,
/// C2h) while every surviving cell still contributes exactly once.
/// That mechanism REPLACES a member's own evaluation with a
/// symmetry-related one rather than dropping an exact zero, so it is a
/// numerics change - the one difference the two mechanisms have; see the
/// symmetryOrbitExpansion option's own note below.
/// Determinism: the k = 1 path (maxParallelChunks = 1) runs the whole
/// build on one thread with no OpenMP involvement - the path whose bytes
/// the lean tests pin bit-identically across calls and builders.
///
/// No Eigen in this header (integrals public headers keep Eigen
/// implementation-only - the stated CMake policy).

#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace qcx::integrals {

struct FockBuildStats; // The per-call stats carrier (fock_build.hpp); pointer-only here.

/// The lean builder's build settings. Deliberately a small subset of
/// FockBuildOptions: the lean path is recompute-only fp64 - no density
/// screening, no per-element screening, no certified mixed-precision lane,
/// no cache, no budget/admission objects (the builder holds nothing the
/// loop can charge; the one Create-time memory ceiling is a last-resort
/// refusal, not a budget).
struct LeanFockBuildOptions {
    /// Screening + cutoff preset: the row walk's in-loop survivor test is
    /// Q_MN * Q_LS against SchwarzThreshold(accuracy) x kNeighborListSlack
    /// - the existing machinery's neighbor-list cutoff verbatim.
    AccuracyPreset accuracy = AccuracyPreset::kNormal;
    /// The per-flush contraction tile cap (bytes of ERI values): each
    /// flush's survivors are assembled into class batches bounded by this
    /// cap and computed into one bounded per-thread tile before the
    /// contraction. Must be positive (Create refuses otherwise); the
    /// machinery default (512 MB) is the operating setting.
    std::size_t maxBatchBytes = 512 * 1024 * 1024;
    /// The per-live-window pair-data retention cap (bytes; 0 = unbounded,
    /// the historic retention). Each live window's store keeps the
    /// contracted pair data - the bra/ket transforms, the contraction
    /// weights and the primitive pairs with their per-axis E tables - of the
    /// pairs its flush built, and one window's rows reach most of the pair
    /// space, so the unbounded retention is the lean path's dominant
    /// allocation (5 live windows x ~9.7 GiB at n = 4,974: the 48.4 GiB term
    /// of EstimatePeakBytes against 0.99 GB for the five O(N^2) matrices).
    /// With the cap set, the store is a BOUNDED BAND over the walk: the
    /// flush's built pairs join the band and it releases oldest-first once
    /// it exceeds the cap, so a pair the band dropped is simply rebuilt when
    /// a later flush lists it. That rebuild is exact, not approximate - the
    /// pair builder's only inputs are the pair's own shells and centers, so
    /// the rebuilt bits are the released copy's, and the empty braTransform
    /// IS the not-built marker the rebuild tests. What the band costs is
    /// pair builds: with the contiguous row blocks each window walks, a
    /// pair's users are its own row block, so the band's reuse distance is
    /// the band itself (the retention cap is a memory-for-recompute trade,
    /// and 0 restores the unbounded retention). The default holds one live
    /// window's band in the SCF matrix class (5 x 192 MiB = 0.94 GiB at
    /// n = 4,974 against the 0.99 GB the five matrices take); the band never
    /// drops below the flush it is building.
    std::size_t windowStoreBytes = 192 * 1024 * 1024;
    /// The last-resort memory ceiling (GiB): 0 (the default) = no ceiling,
    /// no check. When positive, Create() REFUSES (kInvalidArgument) if the
    /// simple-actuals estimate of the whole SCF's matrix state, the pair
    /// state and the per-window scratch EXCEEDS the ceiling by more than a
    /// generous 1.5x slack - headroom for mechanisms that can use extra
    /// free memory. No razor margins anywhere on the lean path.
    double memoryCapGiB = 0.0;
    /// The parallel-chunk (row-window) count override; 0 (the default, the
    /// driver's auto spelling) = kWindowOversubscription x the backend's
    /// OpenMP team size, clamped to the row count - the row space is cut
    /// into many more windows than threads and the window loop is scheduled
    /// DYNAMICALLY at a chunk of one window, so the row-to-row cost
    /// variance averages out over the run instead of landing in one
    /// thread's static share (PlanWindows in
    /// lean_fock_build.cpp). An explicit N >= 2 pins exactly N windows on N
    /// threads (the window-count experiment seam); 1 is the deterministic
    /// bit-pin path - one window, run directly on the calling thread with
    /// no OpenMP involvement. The windows are round-robin rows, so the
    /// decomposition parallelizes the row space directly with no nested
    /// parallelism, and the per-window accumulators reduce in window order.
    std::size_t maxParallelChunks = 0;
    /// The row-window WAVE WIDTH - the accumulator-residency bound of the
    /// generation-grouped reduction. The windows run in waves of this many
    /// and each wave's n x n accumulators fold into the running Fock in
    /// window order as the wave joins, so at most `waveWidth` partials are
    /// ever live instead of one per window: the auto plan's 8 x team windows
    /// cost 64 n^2 bytes of accumulator (7.37 GiB at n = 4,974 - the largest
    /// term of every lean run) against the 16 n^2 the auto width's 2 x team
    /// windows cost. 0 (the default, the driver's spelling) = auto =
    /// kWindowWaveFactor x the region's thread count, clamped to the window
    /// count (lean_fock_build.cpp PlanWaveWidth); an explicit N pins exactly
    /// N windows per wave, N >= the window count being the single-wave form
    /// (the historic residency, the seam the bit-identity pin uses).
    ///
    /// The width moves NO value: the fold is window-ordered at every width,
    /// so the Fock - and with it the whole SCF trajectory - is bit-identical
    /// across widths (LeanDirectFockBuilderTest's wave pin). What it moves is
    /// how many accumulators are live, and how many windows the dynamic
    /// schedule can rebalance over at a time: the wave is the schedule's
    /// rebalancing granularity, so a width smaller than the region's thread
    /// count would leave threads idle. The two exact plans - the k = 1 pin
    /// and an explicit maxParallelChunks = N - take a one-wave width by
    /// construction and are byte for byte the calls they were.
    std::size_t windowWaveCount = 0;
    /// Force the scalar contract kernel (the k = 1 options pin). The
    /// scalar and AVX2 copies are bit-identical by contract, so this
    /// changes nothing numerically - it pins the exercise path in tests.
    bool forceScalarContract = false;
    /// The Abelian point-group reduction (the machinery's classification
    /// seam): when non-null and non-trivial, Create() derives one
    /// symmetry classification per canonical pair from it and BuildFock
    /// drops the canonical cells whose ERI block the classification proves
    /// EXACTLY zero. TRIVIAL here is the reduction's own isTrivial flag -
    /// the group's action on the CANONICAL SHELL PAIRS, not its order (the
    /// mechanism-value audit, 2026-09-13) - so a non-C1 sign-only group
    /// (every atom on the symmetry element: a planar molecule's
    /// out-of-plane mirror) is trivial and engages nothing, which is the
    /// measured shape of a mechanism that costs and returns 1.0000x.
    /// The dropped cells contribute exact zeros to every
    /// contraction (the invariance relation forces every element of the
    /// block to its own negation), so the Fock matrix is byte-identical to
    /// the unfiltered build - this is a throughput-only mechanism, not an
    /// approximation. Its measured reach, so the rule is not read as a source
    /// of savings: it drops 0 cells on every bench fixture (water/STO-3G, C2v,
    /// 120 -> 120; c8h18/def2-SVP, C2h, 8,811,481 -> 8,811,481), while its
    /// positive control - linear H-O-H/STO-3G with the O on the inversion
    /// centre - drops exactly 8 of 120 cells, byte-identically both ways
    /// (LeanPointGroupTest.InversionCentreFixtureDropsItsProvablyZeroCells,
    /// lean_point_group_test.cpp:546).
    /// Null (the default) and a trivial reduction are the
    /// same code path and the same bytes: nothing is classified, nothing is
    /// dropped. The reduction is consulted at CREATE only (the classification
    /// is derived once and retained); the caller keeps ownership and no
    /// reference outlives the Create call. What the classification can prove
    /// is deliberately narrow - a cell is dropped only when one group element
    /// fixes all four of its shells function-for-function and the four
    /// per-shell signs multiply to -1 at every position - and a quartet the
    /// symmetry merely RELATES to another (a non-trivial atom permutation)
    /// is never dropped: the invariance identity then links two non-zero
    /// integrals instead of forcing a zero. See lean_point_group_test.cpp
    /// for the measured counterexample that pins this boundary.
    const SymmetryReduction* symmetryReduction = nullptr;
    /// The ORBIT-EXPANSION engagement (the petite-list increment): with this
    /// true AND a non-trivial
    /// symmetryReduction (trivial = the reduction's isTrivial flag, the
    /// group's action on the canonical shell pairs - see the field above;
    /// a pair-trivial group leaves the action unbuilt and the call is the
    /// C1 call byte for byte), Create() derives the reduction's per-PAIR
    /// action -
    /// the canonical image pair of every pair under every element, the shell
    /// images and the function-level signed permutation, O(nPairs x |group|)
    /// bytes (on c8h18/def2-SVP: 84 KB of pair images, ~90 KB for the whole
    /// action), which is the whole retained state of
    /// the mechanism - and BuildFock computes ONE ERI block per symmetry
    /// orbit: the row walk visits only each orbit's surviving minimum cell,
    /// and every other member of that orbit contracts from the
    /// representative's block, expanded into the member's own canonical-role
    /// frame by the generator's function action (its signs are +-1, so the
    /// expansion itself introduces no rounding). The work therefore drops in
    /// proportion to the orbit multiplicity - measured 2,379,972
    /// blocks over 8,811,481 screened cells on c8h18/def2-SVP (0.2701, 3.70x
    /// fewer ERI blocks, mass-weighted 0.2873) - which is a BLOCK-COUNT ratio
    /// and never a wall-clock ratio: the per-member contraction over all
    /// 8,811,481 members still runs, and the expansion adds a per-member copy
    /// the plain walk has no equivalent of.
    ///
    /// This is NOT a through-only saving like the classification above, and
    /// that is the one way the two mechanisms differ: the mask drops a term
    /// that is exactly zero, while the expansion REPLACES a member's own
    /// evaluation with the representative's - legitimate because the ERI is
    /// invariant under the group action, exact in exact arithmetic, and NOT
    /// exact in IEEE double, because the engine's summation order for the
    /// member's axis order is not the representative's. The Fock matrix
    /// therefore moves in the last bits against the plain walk (the
    /// reduction's own machinery pins its class path against the plain path
    /// at 1e-12, never byte-for-byte - integrals/tests/fock_build_test.cpp),
    /// and the lean tests pin the deltas this path delivers. With the option
    /// false - the default - nothing of this mechanism runs: no action is
    /// built, no cell is skipped as a non-representative, and the call is the
    /// plain path's byte for byte. True without a reduction is refused
    /// (kInvalidArgument) at Create: the reduction is what defines the orbits.
    bool symmetryOrbitExpansion = false;
    /// Split-mode flags (the direct-UHF adapter's halves): with
    /// buildCoulombOnly the builder contracts J(rho) only
    /// (F = H + 2J(rho)); with buildExchangeOnly, K(rho) only
    /// (F = H - K(rho)). At most one of the two should be set.
    bool buildCoulombOnly = false;
    /// See buildCoulombOnly. At most one of the two should be set.
    bool buildExchangeOnly = false;
};

/// The lean builder's Create-time memory envelope WITHOUT a builder, for
/// callers that must decide before the builder can exist: the Create-time
/// last-resort ceiling is a function of (molecule, basis set, options)
/// alone, and the core Hamiltonian Create consumes is not one of its
/// inputs. A driver that pays a setup ramp before the core Hamiltonian
/// exists can therefore consult the admission before that ramp - the
/// allocation the ceiling sizes - instead of after it (the above-ceiling
/// crash class: a cap below the ramp's own footprint killed the process
/// inside the ramp, with no admission decision ever reached).
struct LeanEnvelope {
    /// The envelope's total, in bytes: the same simple-actuals sum the
    /// builder's Create-time check consumes and EstimatedPeakBytes
    /// reports. Always computed.
    double totalBytes = 0.0;
    /// The last-resort refusal text when the envelope exceeds the options'
    /// ceiling by more than the generous slack; empty when the ceiling
    /// admits it (or when memoryCapGiB is 0 - no ceiling, no check). The
    /// text is the builder's own, byte for byte, so an early consult and
    /// the builder's Create-time check can never drift apart.
    std::string refusal;
};

/// Computes the lean builder's Create-time envelope and its ceiling
/// verdict without constructing a builder and without a core Hamiltonian:
/// the canonical shell pairs, the row-window plan a call will run, and the
/// same envelope arithmetic Create consumes (EstimatePeakBytes). The
/// numbers agree with the builder's own by construction - the driver's
/// memory audit, an early admission consult and the builder's refusal all
/// read one formula.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set (the primitive-count authority).
/// \param options The lean options; the ceiling verdict is read from
/// options.memoryCapGiB (the slack rule is Create's).
/// \returns The envelope, or an Error from the canonical shell-pair build
/// (the same error Create reports).
/// \ingroup qcx-integrals
qcx::Result<LeanEnvelope> EstimateLeanEnvelope(const qcx::molecule::Molecule& molecule,
                                               const qcx::basisset::BasisSet& basisSet,
                                               const LeanFockBuildOptions& options);

/// Builds F(D) = H + 2J(rho) - K(rho) for closed-shell RHF by
/// recomputing the surviving ERIs on every call - the lean-direct path,
/// named "lean" because its per-build cost is the
/// screened quartet recompute plus the O(pair) row walk: no neighbor CSR
/// retained from the previous call, no admission record, no workspace
/// budget and no cache - every byte of the state is built once at Create
/// and then immutable. The input density is the SPATIAL closed-shell
/// density rho = D/2 (the direct/RI builder convention of the
/// DirectJkFockBuilder seam; rhf.hpp documents the seam contract).
///
/// J and K are accumulated with the 8-fold symmetry folded in, with the
/// SAME arithmetic as the direct machinery's serial kernels: each
/// canonical quartet block contributes to the bra and ket pair blocks of J
/// and to the four exchange targets of K (the density is permuted per
/// block, integrals are never transposed). Every unique shell quartet is
/// computed exactly once.
/// \ingroup qcx-integrals
class LeanDirectFockBuilder {
public:
    /// Prepares the builder: builds the canonical shell pairs, computes
    /// the Schwarz bounds, records one immutable 24-byte record per pair,
    /// flattens the per-shell contraction inputs, builds the
    /// geometry-only pair-store template and copies the core Hamiltonian.
    /// \param molecule Molecule providing the atom coordinates (Bohr).
    /// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
    /// (kUnimplemented otherwise).
    /// \param coreHamiltonian H = T + V, host-canonical rank-2 tensor with
    /// shape {n, n}.
    /// \param options Build settings; maxBatchBytes must be positive,
    /// memoryCapGiB must not be negative.
    /// \returns The builder, or an Error (kInvalidArgument for the option
    /// validation failures and for the last-resort memory ceiling refusal,
    /// which reports required_peak_bytes, available_bytes and the largest
    /// contributor).
    static qcx::Result<LeanDirectFockBuilder> Create(
        const qcx::molecule::Molecule& molecule,
        const qcx::basisset::BasisSet& basisSet,
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
        const LeanFockBuildOptions& options = {});

    /// Builds F(D) = H + 2J(rho) - K(rho) for the given closed-shell
    /// density.
    /// \param density The SPATIAL closed-shell density rho = D/2 (spin-summed
    /// density over 2), host-canonical rank-2 tensor with shape {n, n}. Must
    /// be SYMMETRIC: the canonical-pair contractions read both orientations
    /// of every unordered pair block (d(c,d) + d(d,c)) and the K
    /// transpose-writes assume it (a Debug assert checks it in Debug builds).
    /// \returns The Fock matrix as a rank-2 tensor, or an Error. The call
    /// is deterministic: with maxParallelChunks = 1 the result is
    /// byte-identical across calls and across fresh builders (the lean
    /// pin contract); with more chunks the per-window accumulators reduce
    /// in window order, so the result is reproducible but its per-element
    /// summation order differs from k = 1. The wave width does NOT move the
    /// summation order: the fold stays window-ordered at every
    /// windowWaveCount, so the k > 1 result is bit-identical across wave
    /// widths (the wave pin), and only the window COUNT may move it.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildFock(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density) const;

    /// Builds F(D) with the per-call stats seam:
    /// when \p statsOut is non-null the call fills its fp64QuartetCount
    /// (the call's screened-in shell-quartet count - every canonical cell
    /// whose Q_MN * Q_LS product clears the cutoff, computed exactly once
    /// per call; with symmetryOrbitExpansion engaged the count is the
    /// number of ERI BLOCKS the call evaluates, one per orbit, which is how
    /// the mechanism's work reduction is read off a live build) and its
    /// totalWallTime (one steady_clock span over the
    /// whole call on the calling thread - the wall-comparable number at
    /// every window count). All other FockBuildStats fields stay at their
    /// caller-provided values (a fresh zeroed struct per call): the lean
    /// path is fp64-only recompute with no cache, no screening tiers and
    /// no merge chain, so the fp32 lane, the per-phase split times and
    /// the calibration terms are not lean concepts and are never
    /// written. The result and bytes are identical to the
    /// single-argument call.
    /// \param density The same input and contract as the single-argument
    /// call.
    /// \param statsOut Optional per-call stats collector (fock_build.hpp);
    /// never dereferenced when null (the zero-cost path).
    /// \returns The Fock matrix as a rank-2 tensor, or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildFock(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        FockBuildStats* statsOut) const;

    /// The core Hamiltonian this builder was constructed with (H in
    /// F = H + 2J(rho) - K(rho)). Exposed for callers that need to combine
    /// multiple BuildFock results and must account for H being included in
    /// each one.
    /// \returns The construction-time H, host-canonical rank-2 tensor with
    /// shape {n, n}; a reference into this builder's state (valid for the
    /// builder's lifetime, stable across calls).
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& CoreHamiltonian() const;

    /// The Create-time memory envelope, in bytes: the simple-actuals sum
    /// the last-resort ceiling check consumes (SCF matrices + DIIS
    /// history + the retained pair state + the live wave's accumulators
    /// and the live windows' scratch + the
    /// retained pair-store payload of the live row windows + the run's own
    /// pair-store materialization) - the same number the refusal message's
    /// required_peak_bytes reports. Exposed for the driver's memory audit,
    /// so the audit and the refusal can never drift apart. It is a
    /// live-bytes model of this builder's run context, not a whole-process
    /// model: the process image's own floor and the allocator's committed
    /// high-water above the live set are outside its reach
    /// (EstimatePeakBytes' note, lean_fock_build.cpp).
    /// \returns The estimated peak bytes; always the same value for a
    /// given builder (computed once at Create, whether or not a ceiling
    /// was set).
    double EstimatedPeakBytes() const noexcept;

private:
    // The implementation state lives in the .cpp (Eigen stays
    // implementation-only).
    struct State;
    explicit LeanDirectFockBuilder(std::shared_ptr<const State> state);
    std::shared_ptr<const State> _state;
};

} // namespace qcx::integrals
