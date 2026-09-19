#pragma once

/// \file
/// The direct J/K Fock builder: batched ERIs contracted with the density on
/// the fly - no dense n^4 tensor. Schwarz screening, density-weighted
/// screening, and the certified-mixed-precision routing per iteration. The
/// engine performs no
/// screening internally; this builder is the screening caller, the dense
/// driver of eri_dense.hpp the other one.
///
/// No Eigen in this header (integrals public headers keep Eigen
/// implementation-only - the stated CMake policy).

#include "qcx/backend/gpu_compute_profile.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_cache.hpp"
#include "qcx/integrals/precision_policy.hpp"
#include "qcx/memory/device_workspace_budget.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/memory/workspace_budget.hpp"
#include "qcx/molecule/molecule.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace qcx::integrals {

struct SymmetryReduction; // The class-aware contraction seam (fock_build.hpp).

/// Per-call diagnostics of one direct Fock build (the measurement
/// seam): filled by BuildFock only when a statsOut pointer is passed;
/// null keeps the zero-cost path. Purely observational - no numerical
/// decision reads these values. The wall-time fields are
/// std::chrono::steady_clock durations of the phases a caching tier would
/// want to account for; on the class-aware path the counts are
/// filled and the per-phase split time fields stay zero (the class pass
/// is a separate contraction path and is not time-split; totalWallTime
/// still covers the whole call). The lean builder is its own
/// filler: fp64QuartetCount, batchCount, the eri split (with its prep
/// sub-span, the prep's own three-part split and the kernel-span
/// probe's three sub-spans and four counters), the pair-build count,
/// the optional per-window times, contract and totalWallTime are backed on
/// the lean path (its
/// certified fp32 lane is never engaged, so fp32QuartetCount reads its
/// true zero), while the
/// machinery-only fields (np, g, merge, the cache hits, elementDrops)
/// stay at their defaults there.
///
/// THE k > 1 REGIME (the concurrent-slot batch loop;
/// concurrentSlots > 1): eriWallTime and contractWallTime are then SUMS
/// of the per-slot spans - each slot times its own claimed batches into a
/// per-slot partial, and the partials are added after the join. The
/// concurrent slots' spans overlap, so the sums are slot-accumulated
/// (per-thread CPU-style) totals, NOT whole-call wall spans: they can
/// exceed totalWallTime by up to the slot count, and the k = 1
/// accounting identity (split sum <= total; total minus the splits is the
/// screen+infra residual) does not hold there. totalWallTime (one clock
/// around the whole call on the calling thread) and mergeWallTime (the
/// merges are strictly serialized by the gate, each timed by exactly one
/// slot, so the intervals never overlap) stay wall-comparable at every k
/// - the reported fields that distinguish the merge-chain cost from the
/// compute without double-counting the per-thread spans into the wall.
/// \ingroup qcx-integrals
struct FockBuildStats {
    std::size_t fp64QuartetCount = 0; ///< Quartets screened into the fp64 lane this call.
    std::size_t fp32QuartetCount = 0; ///< Quartets routed through the certified fp32 lane.
    /// The call's assembled class-batch count (the split probe, the
    /// lean path only): the number of MdClassBatch units the call's flushes
    /// ran through RunBatches. One unit is one kernel dispatch set on the
    /// lean path (its batches run single-threaded, regionThreads = 1), so
    /// this pairs with eriWallTime as the dispatch-granularity denominator.
    /// Zero on the machinery paths, whose batches stream from the cursor and
    /// are not counted here (the field is lean-backed by contract).
    std::size_t batchCount = 0;
    /// The call's pair-data BUILDS (the prep probe, the lean path
    /// only): the number of pairs the call's BuildChunkPairData passes
    /// actually built - one contracted pair transform (one
    /// BuildContractedPairTransform) per built pair, so the
    /// pair-proportional part of eriPrepPairDataWallTime reads as this
    /// count's denominator. A pass builds exactly the listed pairs whose
    /// not-built marker is still off, and the lean flush pass no longer
    /// tears the pair data down, so a pair a LATER flush lists again is
    /// skipped rather than rebuilt: the count is a build count, not the
    /// flush re-cutting's listed volume, and the builder reads the same
    /// marker (outside both prep clocks) to form it. With the retention
    /// holding it stays at or below the call's reachable pair count - one
    /// build per pair per window - instead of growing with the flush
    /// count. Zero on the machinery paths, whose pair data is built once
    /// and not counted here (the field is lean-backed by contract).
    std::size_t pairBuildCount = 0;
    /// The call's CONTRACT-BLOCK count: how many times the contraction phase
    /// ran one stored ERI block into the Fock - one per assembled class task,
    /// plus (with the orbit expansion engaged) one per surviving member cell
    /// (`lean_fock_build.cpp`, `contractOrbitMembers`). The mechanism-value
    /// audit (2026-09-13) found the ERI side carrying eight denominators and
    /// this phase carrying none, which is why "does the symmetry actually
    /// save work" could not be asked from the counters: on c8h18/def2-SVP the
    /// orbit expansion takes the ERI blocks from 8,811,481 to 2,379,972
    /// (-72.99%) while the accumulation is INVARIANT at 8,811,481, because
    /// `ContractBlock` runs once per member regardless. Read against
    /// `fp64QuartetCount`, this is the evaluation-to-accumulation ratio the
    /// audit had to reconstruct by hand. One per call per task/member cell,
    /// summed over the row windows (the same race-free partials as
    /// `pairBuildCount`); zero on the machinery paths (the field is
    /// lean-backed by contract, like its neighbours).
    std::size_t contractBlockCalls = 0;
    std::chrono::nanoseconds eriWallTime{}; ///< The ERI-phase span: at k = 1 the batch-computation
                                            ///< wall time (the part a value cache would eliminate);
                                            ///< at k > 1 the slot-accumulated sum over the
                                            ///< concurrent slots' own spans (see the struct note -
                                            ///< not a wall span there).
    /// The pre-kernel part of the lean path's ERI span (the split
    /// probe): one flush's BuildChunkPairData (the flush pairs' contracted
    /// transforms), AssembleClassBatches and tile setup - everything the ERI
    /// span covers BEFORE the RunBatches kernel dispatch, so
    /// eriWallTime - eriPrepWallTime is the kernel time proper. A sub-span of
    /// eriWallTime, never added to it by a reader. Zero on the machinery
    /// paths (the field is lean-backed by contract).
    std::chrono::nanoseconds eriPrepWallTime{};
    /// The pair-data sub-span of eriPrepWallTime (the prep probe, the
    /// lean path only): the flush's BuildChunkPairData pass - the
    /// not-built-marker walk over every pair the flush lists plus one
    /// contracted transform per pair the pass builds (pairBuildCount's
    /// unit: the count itself is the build part alone, so the span's
    /// pair-proportional share is the built pairs' transforms and its
    /// remaining part grows with the re-cutting). THE FIRST OF THE PREP SPLIT:
    /// with the two fields below it accounts for eriPrepWallTime's three
    /// named parts, so the three sum to at most the prep span (the
    /// residual between the parts' clocks belongs to no part). Zero on the
    /// machinery paths.
    std::chrono::nanoseconds eriPrepPairDataWallTime{};
    /// The assembly sub-span of eriPrepWallTime (the prep probe, the
    /// lean path only): the flush's AssembleClassBatches call - the
    /// quartets' canonicalization, their class grouping and the per-class
    /// byte-cap cuts. Quartet-proportional in its scan, per-batch in its
    /// emission. Zero on the machinery paths.
    std::chrono::nanoseconds eriPrepAssembleWallTime{};
    /// The tile-setup sub-span of eriPrepWallTime (the prep probe,
    /// the lean path only): everything between the assembly and the
    /// kernel dispatch - the flush's tile sizing walks (the per-task
    /// element sums), values.resize, the per-batch output-pointer rebase
    /// and the copy of the assembled batches into the dispatch list. The
    /// per-batch allocation part of the prep span. Zero on the machinery
    /// paths.
    std::chrono::nanoseconds eriPrepTileSetupWallTime{};
    std::chrono::nanoseconds
        contractWallTime{}; ///< The contraction-phase span: at k = 1 the
                            ///< density-contraction wall time - NOT eliminated
                            ///< by a value cache (the cached blocks still
                            ///< contract with the new density); at k > 1 the
                            ///< slot-accumulated sum over the concurrent slots'
                            ///< own spans, each including its batch's gate wait
                            ///< for the canonical turn (see the struct note).
    /// The kernel-span probe (the lean path only): the
    /// VRR/Boys phase of the class kernels - one pass-1 group's one ket
    /// primitive pair, from the top of the primitive-pair body over the pq
    /// and acc block zeroing and the RunVrrQuadruple recurrence loop of
    /// every task of that group. A sub-span of the kernel time
    /// (eriWallTime - eriPrepWallTime), measured INSIDE the dispatch, at a
    /// granularity of one primitive pair per group - never one per quartet
    /// and never one per primitive quadruple, whose clock cost would exceed
    /// the span it reports. Like eriWallTime it is a true wall span at
    /// k = 1 and a window-accumulated total above k = 1. Zero on the
    /// machinery paths.
    std::chrono::nanoseconds kernelVrrWallTime{};
    /// The ket-transform sub-span of the kernel time (the kernel-span probe,
    /// the lean path only): the group's one KetTransformBatched call, from the
    /// close of kernelVrrWallTime to its return. Zero on the machinery
    /// paths. Same k = 1 / k > 1 convention as kernelVrrWallTime.
    std::chrono::nanoseconds kernelKetWallTime{};
    /// The bra-transform sub-span of the kernel time (the kernel-span probe,
    /// the lean path only): the whole pass-2 task loop - the per-task bra
    /// transform and its packed-block write, plus the loop's own per-task
    /// bookkeeping. The three kernel sub-spans are disjoint and sum to AT
    /// MOST the kernel time: the remainder (the per-batch scratch layout
    /// walk, scratch.assign, the group splitting, the loop increments and
    /// the pass-2 clock opens) belongs to no phase. Zero on the machinery
    /// paths. Same k = 1 / k > 1 convention as kernelVrrWallTime.
    std::chrono::nanoseconds kernelBraWallTime{};
    /// The VRR phase's work denominator (the kernel-span probe, the lean path
    /// only): the RunVrrQuadruple call count the call's class batches ran -
    /// summed over every task, ket primitive pair and bra primitive pair.
    /// kernelVrrWallTime over this count is the recurrence's per-primitive-
    /// quadruple cost, the unit a vectorization lever moves. Zero on the
    /// machinery paths and on a call that ran no class batch.
    std::size_t kernelVrrQuadruples = 0;
    /// The transform calls that missed the micro-gate shape test (the
    /// kernel-span probe, the lean path only): the fp64-lane transform shapes above
    /// kMicroGemmShape in some dimension, which reach the linalg batched
    /// seam instead of the in-tree micro kernel (md_transform.hpp
    /// MicroGemmEligible). KetTransformBatched and BraTransformSingle
    /// together - the ket call total is kernelPrimPasses and the bra call
    /// total the row's fp64QuartetCount, so the eligible share is derivable
    /// from this count alone. The fp32 lane's shapes are not gated and never
    /// counted here. Zero on the machinery paths.
    std::size_t kernelGateSeamCalls = 0;
    /// The pass-1 group census (the kernel-span probe, the lean path only):
    /// how
    /// many ket-primitive-pair groups the call's class batches walked. A
    /// group is a maximal run of tasks sharing one ket pair and one bra
    /// row-pair count (md_vrr.hpp). Zero on the machinery paths.
    std::size_t kernelGroupCount = 0;
    /// The (group, ket primitive pair) iteration count (the kernel-span
    /// probe, the lean path only) - exactly the KetTransformBatched call count,
    /// and the census the kernel-span instrument's own granularity sits on.
    /// Zero on the machinery paths.
    std::size_t kernelPrimPasses = 0;
    /// The merge-chain measurement: time in the
    /// canonical-order batch-delta merge chain - the per-batch shared-Fock
    /// accumulation (_fock += delta.fock) of the chunked contraction, at
    /// the batch's canonical turn (the BatchMergeGate order at
    /// k > 1; the k = 1 batch loop merges in the same canonical order).
    /// Each merge is timed by exactly one slot with the gate's lock held
    /// (or, at k = 1, in the sole slot), so the intervals never overlap
    /// and the per-call sum is the merge chain's busy time - a true wall
    /// contribution at every k, unlike the eri/contract spans. Batches
    /// that fold per index (numChunks <= 1, the serial fold) accumulate
    /// inside the contraction and are not separately timed; the gate
    /// waits (the chain idle for the next ready delta) and the merges
    /// themselves stay inside the owning slot's contract span - this
    /// field is a split of the contract phase, never a separate span.
    /// Filled on the plain per-quartet path only (the class-aware path
    /// keeps the split time fields zero by contract).
    std::chrono::nanoseconds mergeWallTime{};
    /// The call's observed concurrency (the k > 1 regime marker of the
    /// struct note): the maximum over the call's passes of the batch-slot
    /// count the merge gate actually ran (kEff = min(slots authorized at
    /// Create, the pass's batch count)); 1 on every k = 1 path and on the
    /// class-aware path (whose split time fields stay zero anyway). A
    /// call that read > 1 ran overlapping slots, so its eri/contract
    /// spans are slot-accumulated; total and merge stay wall-comparable.
    std::size_t concurrentSlots = 1;
    std::chrono::nanoseconds totalWallTime{}; ///< Whole-call wall time: one steady_clock span
                                              ///< around the whole call on the calling thread -
                                              ///< the wall-comparable number at every k.
    /// Optional per-window wall times (the per-window probe, the lean path
    /// only): when non-null, cleared and refilled with one entry per row
    /// window, in window order - the elapsed time of that window's whole
    /// body (its accumulator and scratch setup plus its round-robin row
    /// walk). The auto decomposition oversubscribes the team
    /// (min(8 x team, nPairs) windows, PlanWindows in lean_fock_build.cpp)
    /// and the window loop is scheduled DYNAMICALLY at a chunk of one
    /// window, so an entry is the cost of its own window's rows, not a
    /// thread's static share of the rows: with the schedule handing windows
    /// out one at a time, the makespan's tail above sum/team cannot exceed
    /// the LARGEST entry, so max/mean and max/min read the row-cost
    /// variance and the residual schedule slack is bounded by max rather
    /// than measured by the spread. An explicit maxParallelChunks = N pins
    /// exactly N windows; one entry on the k = 1 path (each entry is a true
    /// wall span of the window's own thread, so they may overlap at k > 1;
    /// every entry stays inside totalWallTime). Null (its default) leaves
    /// the caller's record untouched - the builder writes no per-window
    /// storage of its own.
    std::vector<std::chrono::nanoseconds>* windowWallTimesOut = nullptr;
    /// Optional per-call capture of every screened-in quartet as the
    /// canonical pair key bra * nPairs + ket (nPairs = the pair-list size
    /// of the builder's Create; the recurrence analysis of the per-call
    /// measurement). Cleared and refilled when non-null; order is
    /// unspecified (the fp64 lane first, then the fp32 lane).
    std::vector<std::size_t>* quartetKeysOut = nullptr;
    /// Quartets served from the in-memory ERI cache this call
    /// (fp64 lane). Zero when the cache is disabled (maxCacheBytes 0, or
    /// the class-aware path, where the cache stays disengaged) or cold.
    std::size_t cacheHitFp64QuartetCount = 0;
    /// Quartets served from the in-memory ERI cache this call
    /// (the certified fp32 lane). Same zero conditions as
    /// cacheHitFp64QuartetCount.
    std::size_t cacheHitFp32QuartetCount = 0;
    /// Quartets' per-element contributions skipped by the
    /// per-element density re-filter this call: each kernel-section element with |g| < tau_T (the
    /// per-target threshold tau / max|D_T|, halved on the J sections)
    /// increments the counter. Zero when the per-element screening flag is
    /// off (the flag contract pins elementDrops == 0 on the legacy path) and
    /// on the GPU path (the device kernel does not count; the host-side
    /// vector still gates, so the count would underreport - the parity tests
    /// run the flag off for this reason).
    std::size_t elementDrops = 0;
    /// Calibration term P (per call): the number
    /// of DISTINCT canonical shell pairs touched by the call's screened-in
    /// quartets (bra or ket side, counted once per pair). The direct
    /// screened backbone's per-iteration pair volume; a pair re-appearing as
    /// the ket of several LightPath chunks counts once. Zero when stats are
    /// not requested (statsOut null keeps the whole Fock build allocation
    /// and counter free).
    std::size_t significantPairCount = 0;
    /// Calibration term G (per call): the sum
    /// over the call's screened-in quartets of the quartet's primitive-pair
    /// product weight (bra pair's primitive-pair count times the ket pair's
    /// - each pair's count is the product of its two shells' primitive
    /// counts). The per-iteration primitive-product volume, summed over the
    /// fp64 and fp32 lanes. Zero under the same conditions as
    /// significantPairCount.
    std::size_t primitiveProductSum = 0;
};

/// The QFMM near-field restriction: when engaged, BuildFock only screens and
/// contracts quartets
/// whose pair-pair (bra, ket) is in the restriction. Stored as a symmetric
/// bitset over leaf-node pairs - nLeaves^2 bits, bit
/// (leafOfPair[bra] * nLeaves + leafOfPair[ket]) set for every near-field
/// leaf pair in both orientations - instead of the former pair-pair key set
/// (an unordered_set keyed bra * nPairs + ket: up to 2*nPairs^2 nodes,
/// ~8.5 GB for c60_sto3g as measured on 2026-08-29). The bitset is
/// < 1 MB for the same input. leafOfPair must cover every pair index
/// 0..nPairs-1 (the octree assigns every pair a leaf); an index outside it
/// is a caller bug, not validated - the screening hot path stays
/// bounds-check-free, matching the old set semantics.
struct PairPairRestriction {
    std::vector<std::uint64_t> words; ///< nLeaves^2 bits, both orientations set.
    std::size_t nLeaves = 0; ///< Leaf-node count (the index stride).
    std::vector<std::size_t> leafOfPair; ///< Pair index -> leaf node index.

    /// True when the restriction is engaged (no bits set = unrestricted).
    /// \returns True when no restriction is in force.
    bool Empty() const noexcept {
        return words.empty();
    }

    /// True when the (bra, ket) pair-pair may be contracted. Both
    /// orientations of every near-field leaf pair are set, so the canonical
    /// (bra, ket) iteration of the Schwarz neighbor list matches either way.
    /// \param bra The bra pair index (canonical pair order).
    /// \param ket The ket pair index.
    /// \returns True when the pair-pair is in the restriction.
    bool Contains(std::size_t bra, std::size_t ket) const noexcept {
        const std::size_t bit = leafOfPair[bra] * nLeaves + leafOfPair[ket];
        return ((words[bit >> 6] >> (bit & 63)) & 1ULL) != 0;
    }

    /// Sets the (leafA, leafB) restriction bit in both orientations.
    /// \param leafA The bra side's leaf-node index.
    /// \param leafB The ket side's leaf-node index.
    void InsertBoth(std::size_t leafA, std::size_t leafB) noexcept {
        Set(leafA * nLeaves + leafB);
        Set(leafB * nLeaves + leafA);
    }

private:
    void Set(std::size_t bit) noexcept {
        words[bit >> 6] |= 1ULL << (bit & 63);
    }
};

/// The QFMM leaf-driven near-field domain: when engaged,
/// DirectJkFockBuilder::Create builds its cached Schwarz neighbor list by
/// enumerating the near-field pair-pairs directly from the octree's
/// near-field leaf pairs - the nested loop over A.pairIndices x
/// B.pairIndices per near-field leaf pair (A, B), the diagonal leaf A == A
/// included, canonical (bra, ket) = (max, min) pair index per unordered
/// pair-pair - instead of sweeping every bra row of the full pair space and
/// filtering per candidate by the PairPairRestriction bitset (which stays
/// for the test harness as the gated-sweep reference). The construction
/// applies the same per-pair Schwarz cutoff as BuildNeighborList (the
/// slack-tolerant pair cutoff, unchanged), so the leaf-driven candidate
/// rows are exactly the gated sweep's rows: near-field membership per
/// pair-pair is the near-field leaf pair of the two pairs' leaves, and the
/// interaction lists carry each unordered leaf pair exactly once (a <= b,
/// diagonal included - qfmm_tree.cpp), so every unordered pair-pair is
/// generated exactly once. nearFieldLeafPairs and leafOfPair must cover
/// every pair index 0..nPairs-1 and every leaf 0..nLeaves-1 - the octree
/// contract; an index outside them is a caller bug, not validated here.
struct LeafNearFieldDomain {
    std::size_t nLeaves = 0; ///< Leaf-node count.
    std::vector<std::size_t> leafOfPair; ///< Pair index -> leaf node index.
    std::vector<std::pair<std::size_t, std::size_t>> nearFieldLeafPairs; ///< Unordered
                                                                         ///< near-field leaf
                                                                         ///< pairs (a <= b,
                                                                         ///< diagonal
                                                                         ///< included, each
                                                                         ///< exactly once).

    /// True when the domain is engaged (no near-field leaf pairs =
    /// unrestricted).
    /// \returns True when no domain is in force.
    bool Empty() const noexcept {
        return nearFieldLeafPairs.empty();
    }
};

/// Direct-Fock build settings.
/// \ingroup qcx-integrals
struct FockBuildOptions {
    AccuracyPreset accuracy = AccuracyPreset::kNormal; ///< Screening + certified-lane gate preset.
    std::size_t maxBatchBytes = 512 * 1024 * 1024; ///< Per-class batch cap (512 MB).
    bool useDensityScreening = true; ///< Density-weighted pair screening per iteration.
    /// The two-level density screen - the quartet-level
    /// product gate AND the per-element re-filter. On (the default): the
    /// screening gate forms each quartet's
    /// keep test from the per-call per-pair max-density vector
    /// (shellPairMaxDensity: P_ab = max|D_ab| * Q_ab products, mode-aware
    /// J/K/full), and the contraction kernel re-filters every element of
    /// every kept quartet against tau_T = tau / max|D_T| per target (the
    /// niedoida current_threshold shape; the J sections halve the threshold
    /// for their two-orientation, 2.0*j accumulation). Off: EXACTLY the
    /// legacy single-level behavior - the six-block max-density gate inside
    /// ScreenOne, no kernel filter, and elementDrops stays 0 (the flag
    /// contract: the two levels are inseparable; there is no
    /// product-gate-with-filter-disabled mode). The off path exists for the
    /// GPU-parity pins and the bit-identity pins (the GPU device kernel
    /// predates the re-filter) and for any caller that must reproduce
    /// earlier builds bit-for-bit.
    bool usePerElementScreening = true;
    /// The certified fp32 lane's REQUEST. A SET value is an explicit
    /// caller decision and is honoured as written (an explicit
    /// request resolves at one point) - `true` routes quartets through the
    /// lane when their a-priori bound fits the preset budget, `false`
    /// never engages it (kTight keeps the lane disabled regardless - the
    /// strict pins depend on the fp64 path).
    ///
    /// UNSET (the default) is the DEVICE's: the lane's default follows the
    /// device probe's measured fp32/fp64 throughput ratio, read from
    /// `deviceComputeProfile` (precision_policy.hpp
    /// CertifiedLaneDefaultForRatio) - ON where fp32's throughput premium
    /// pays for the lane's routing and conversion overhead, OFF where it
    /// does not and where no probe result is available (the conservative
    /// interim). An explicit request still wins
    /// over the device, and the preset gate still wins over both.
    std::optional<bool> useCertifiedMixedPrecision;
    /// The run's measured device compute profile
    /// (backend/gpu_compute_profile.hpp). The default is that header's
    /// documented "unknown" profile of a device-less host - which is also
    /// the no-probe-applies case, so an unseeded run's certified-lane
    /// default is the conservative interim. The read belongs to the layer
    /// that owns the run boundary and its memory-cap probe gate: the probe
    /// initializes the CUDA runtime once per run, never per build
    /// (gpu_compute_profile.hpp's own caller contract).
    qcx::backend::GpuComputeProfile deviceComputeProfile;
    /// The certified-bound global budget ENFORCEMENT (opt-in, off by default): when
    /// set, the build's routed certified bound sum is accumulated on the
    /// production path and compared against CertifiedBoundBudget(accuracy,
    /// B_screen) - the preset's J/K target less the screening co-term and
    /// the ladder's slack - and the comparison DECIDES the routing: a
    /// build whose routed sum does not fit the budget falls back to the
    /// fp64 lane for its whole quartet set (the same arithmetic the
    /// ladder applies when its per-batch budget is not positive), so
    /// BuildFock then runs the same quartet set on the same code path as
    /// useCertifiedMixedPrecision = false - the two agree to the builder's
    /// own run-to-run reproducibility, with no partly-fp32 mixture in
    /// between. A build that fits keeps the ordinary routing untouched.
    /// The per-quartet gate is unchanged either way - this is a global
    /// check over the gate's own admissions, never a second gate.
    /// Off by default: the switch is a behaviour change (the fp32 lane's
    /// quartets move to fp64, so energies and wall times move with it),
    /// and an explicit builder request must keep computing what it
    /// computed. The budget is derived from the accuracy preset, never
    /// passed in - a caller cannot widen it to keep the lane alive.
    /// Composition: the FastPath only. The LightPath (the chunked
    /// low-memory route) and an engaged precision ladder both refuse it
    /// (kUnimplemented) rather than ignore it silently; the GPU builder
    /// does not implement it either, so a GPU-family request is refused by
    /// name at resolution (run_driver.cpp) - the one exception is the
    /// retargeted device-less fallback, which lands on this member and can
    /// honour the key.
    bool enforceCertifiedBoundBudget = false;
    bool buildExchangeOnly = false; ///< Skip the J accumulation (the four K targets
                                    ///< only): the RiJkFockBuilder mode - RI provides J,
                                    ///< this builder the exchange part.
    bool buildCoulombOnly = false; ///< Skip K (the UHF J-only mode, the companion
                                   ///< of buildExchangeOnly). At most one of the two should
                                   ///< be true; both true is nonsensical (returns bare H)
                                   ///< and both false is the existing fused RHF mode.
    /// Force the scalar contraction kernel even when the CPU
    /// reports AVX2 (the runtime dispatch's fallback). The AVX2 path is the
    /// same kernel body compiled with /arch:AVX2 (fock_contract_simd.cpp);
    /// both copies are bit-identical by contract, so this is a test pin (the
    /// same role as maxParallelChunks): it exists so the fallback is
    /// exercisable on AVX2-capable machines - a bug in the scalar copy
    /// would otherwise sit unnoticed on any machine whose CPU takes the AVX2
    /// path. Default false = the cpuid-dispatched path.
    bool forceScalarContract = false;
    std::size_t maxParallelChunks = 0; ///< Parallel-build chunk override (the
                                       ///< schedule-independence test pin): 0 = auto
                                       ///< (min(tasks, hardware_concurrency())); 1 forces
                                       ///< the serial fallback; >= 2 a fixed split.
    /// The in-memory ERI-value cache's payload byte budget;
    /// 0 (the default) disables the cache. When positive, Create() probes
    /// DetectHostMemory() FRESH and clamps the budget to half of the probe's
    /// availableBytes - the conservative fraction (when the probe reports
    /// zero available bytes, the caller's cap applies unclamped) - and the
    /// cache's maxCacheBytes reports the clamped budget actually in force
    /// (EriCacheStats::maxCacheBytes, via CacheStats()). On the
    /// adaptive-memory budget path (workspaceBudget set) Create()
    /// additionally clamps the cap until its 3x live footprint - the two
    /// payload lanes plus the entry map (DirectFootprint's cacheBytes
    /// term) - fits the budget's remaining bytes, else zeroes it: the
    /// cache survives only when it provably fits (the sure-fit gate of
    /// the user's O(N^2)-default selection bound). The probe's
    /// availableBytes are a point-in-time snapshot, not a reservation
    /// (memory_topology.hpp - it changes as the process allocates), so this
    /// is a sizing hint at Create() time: size for the union of one SCF
    /// run's computed quartets (one molecule, one basis, fixed for the run -
    /// the cache has NO eviction), and the snapshot semantics are the
    /// stated assumption. The budget is PER LANE: each lane's payload is
    /// capped at it independently (fp64 as doubles, fp32 as floats), so the
    /// two lanes together can reach 2x the budget - size for the union of
    /// both lanes' steady-state blocks. The budget covers the block payload
    /// only; the key table and arena overhead are extra. The cache is
    /// engaged on the plain path only - on the class-aware path
    /// (symmetryReduction with groupOrder > 1) it stays disengaged regardless
    /// (the reduced-element-set interplay is the documented follow-on).
    std::size_t maxCacheBytes = 0;
    /// The engine-decorator seam (EngineDecoratorFactory,
    /// eri_cache.hpp): what stands between the builder and the basis.
    /// Create() calls the factory ONCE with the two RAW engines it would
    /// otherwise use directly, and every request after that goes through the
    /// RETURNED pair - so the tier is chosen from outside this module and
    /// `integrals` never names it (storage, which owns the disk tier, is
    /// downstream of integrals, so the driver - the one module allowed to link
    /// both sides - constructs the decorator, owns it and reads its stats).
    ///
    /// Empty (the default) is the in-memory tier and today's behaviour
    /// exactly: the raw engines reach the RAM tier, or nothing does.
    ///
    /// ENGAGEMENT is the decorator's own presence, never a cap-derived
    /// quantity. A set factory engages the ordinary engine path even when
    /// maxCacheBytes is 0 - the RAM tier is then granted the smallest budget
    /// EriBatchCache accepts, which admits no block, so it is a PASSTHROUGH
    /// and the decorator serves every request while no cache RAM is charged
    /// (the Create-time estimate reads maxCacheBytes, not that grant). That
    /// independence is the point: the retired cap-derived grant was
    /// policy, an explicit request bypasses a default, and a request that is
    /// accepted and then dropped is precisely the run record that can differ
    /// from what ran.
    ///
    /// Disengaged, like maxCacheBytes, by the two paths that cannot serve it:
    /// the class-aware path (the reduced-element-set interplay is the
    /// documented follow-on) and the LightPath (whose miss-assembly reads the
    /// full pair store, which the light mode never materializes). Both are
    /// disengagements of the whole engine tier, not of the decorator alone.
    ///
    /// The RAM the decorator commits is the DECORATOR's to charge: this
    /// builder's Create-time estimate covers the cacheBytes term only, so a
    /// factory whose store needs memory must refuse (or size itself) at the
    /// point it is built - the caller holds the store and knows its shape.
    std::optional<EngineDecoratorFactory> engineDecorator;
    /// The class-aware contraction seam. When set to a valid
    /// reduction with groupOrder > 1, Create() builds the pair-class table
    /// and BuildFock contracts through the petite list: one ERI block per
    /// orbit of symmetry-equivalent member quartets, expanded per member
    /// (symmetry_reduction.hpp). The reduction's data is captured at
    /// Create() time (the builder owns a copy), so the pointer need not
    /// outlive the Create call. Null (the default) keeps the plain path.
    /// A trivial reduction (groupOrder 1) also keeps the plain path - the
    /// class enumeration would be the plain enumeration, evaluated with
    /// expansion overhead. The certified fp32 lane engages on the class
    /// path exactly as on the plain path: each screened
    /// member quartet routes by its own density-weighted bound, and the
    /// certified bound sum accumulates weight(member) * bound(orbit rep) -
    /// the rep block's per-quartet bound covers every expanded member (the
    /// expansion is an exact signed permutation of the rep block, so the
    /// member error is the rep error). The ERI cache stays disengaged on
    /// the class path regardless of maxCacheBytes (the
    /// reduced-element-set interplay with the cache is the documented
    /// follow-on). On the budget path the class path is ADMISSION-GATED
    /// (in place since 2026-08-31): the table's Create-time
    /// never-under estimate (footprint.hpp ClassTableBytes - the worst
    /// case over the canonical member-quartet space) must fit the budget's
    /// remaining bytes, else the class path disengages and the plain
    /// screened path runs (FockModeInfo::classPathDisengaged). The root
    /// restructure (the same change) makes the estimate a ceiling rather
    /// than the materialized shape: the table stores only the pair classes
    /// and the Schwarz class-bound-screened class pairs, and the orbit
    /// expansions are generated on demand per reached class pair - the
    /// live table is the screened subset, never the full eager table. On
    /// the legacy null-budget path (workspaceBudget null) the admission
    /// gate does not run; Create() instead REFUSES with kOutOfMemory when
    /// the same never-under estimate exceeds the node's total physical RAM
    /// (the escape-hatch host-RAM guard, 2026-09-01) - a clean refusal,
    /// never a raw allocation death.
    const SymmetryReduction* symmetryReduction = nullptr;
    PairPairRestriction restrictToPairPairs; ///< QFMM near-field restriction: when
                                             ///< engaged, BuildFock only screens and contracts
                                             ///< quartets whose pair-pair (bra, ket) is in the
                                             ///< restriction - the octree near-field list, a
                                             ///< symmetric leaf-pair bitset (PairPairRestriction;
                                             ///< the pair-pair key set it replaced was a quadratic
                                             ///< memory blowup). Empty = no restriction.
                                             ///< The QFMM builder contracts its octree near-field
                                             ///< list this way and adds the far-field multipole
                                             ///< accumulation on top; at theta -> 0 the restricted
                                             ///< set is everything, so the restricted build is
                                             ///< bit-identical to the unrestricted one (the
                                             ///< acceptance gate).
    /// The QFMM leaf-driven near-field domain: when engaged,
    /// DirectJkFockBuilder::Create enumerates its cached Schwarz neighbor
    /// list from the near-field leaf pairs directly (LeafNearFieldDomain)
    /// instead of the global sweep - the production QFMM near-field
    /// enumeration (the quadratic-in-N sweep of the bitset-gated path does
    /// not scale to the 5000+ function target while the near-field
    /// pair-pair set is linear in N). Empty = the plain global-sweep build.
    /// restrictToPairPairs and this domain are mutually exclusive carriers
    /// of the same near-field set: the builder's restrictToPairPairs gate
    /// (the gated sweep, the test-harness reference) and this domain's
    /// enumerated rows deliver identical candidate sets under the same
    /// cutoff, so the two builds are bit-identical at the serial pin.
    LeafNearFieldDomain leafNearFieldDomain;
    /// The adaptive-memory workspace budget: when set,
    /// Create() decides the build mode from the Create-time
    /// footprint estimate against the budget's remaining bytes (FastPath
    /// when the estimate fits; LightPath - the geometry-only light store
    /// with the per-chunk pattern and transforms - when the fast rung
    /// cannot fit and the light rung can, or when an a-priori exclusion
    /// fires; refused only when the light rung also cannot fit). Null (the
    /// default) keeps the legacy behavior: no estimate, no mode decision,
    /// no reservation - save the class-path host-RAM guard (in place since
    /// 2026-09-01): with a symmetryReduction, Create() refuses (kOutOfMemory) when the
    /// table's never-under estimate exceeds the node's total physical RAM
    /// (the escape hatch stays a clean refusal, never a raw allocation
    /// death). The pointer must outlive the Create call
    /// and is MUTABLE (Reserve charges the cumulative counter - the
    /// nested-builder reservations), shared by the nested builders
    /// (exchange/near-field) at their own Creates; the builder captures the
    /// budget's state, not the pointer - the options copy stored in the
    /// state carries null.
    qcx::memory::WorkspaceBudget* workspaceBudget = nullptr;
    /// The enclosing builder's counted surviving-pair sweep (the QFMM
    /// outer): when non-zero,
    /// Create() skips the a-priori exclusion (i) all-survive shortcut. The
    /// enclosing builder already proved the stack feasible with the counted
    /// pattern (nesting order = reservation order - the outer's estimate
    /// covers the nested half, which charges its own parts at its own
    /// Create), and the all-survive bound can exceed the post-reservation
    /// remaining even when the counted pattern fits - the exclusion would
    /// refuse a stack the outer proved feasible. The nested still runs its
    /// own counting pass and estimate, so the decision is not weakened.
    /// Zero (the default) keeps the plain a-priori exclusion.
    ///
    /// The RI-J's nested exchange half does NOT carry it (it did until
    /// 2026-09-17): that cell's pre-gate is
    /// the counted form now, so it can no longer refuse a stack the outer
    /// proved feasible - and it is the route by which the nested reaches
    /// its own LightPath, so skipping it forced the fast rung's
    /// whole-pattern CSR (183.74 GiB at the 4,974 manifest case, 16.8x the
    /// grant) to be materialized before the light decision could be made.
    /// The RI-J prices the nested at the rung its own rule selects
    /// (ri_engine.cpp priceExchange) instead.
    std::size_t validatedSweepCount = 0;
    /// The LightPath chunk knob: the chunk's bra-row count. Zero (the
    /// default) auto-sizes the
    /// chunk from the budget's remaining bytes at Create (the chunk
    /// formula's closed form - the loose bounds' conservative floor).
    /// Non-zero FORCES the LightPath mode with that chunk size - the
    /// mode-forcing test surface: the budget-driven
    /// LightPath is unreachable at small scales, where the pattern saving
    /// is zero and the light rung's floor sits above the fast path's, so
    /// the bit-identity pins force the mode through the knob. Only
    /// meaningful together with a workspaceBudget (ignored on the legacy
    /// path, whose bit-identity must not be perturbed).
    std::size_t lightPathChunkPairs = 0;
    /// The DEVICE workspace budget mirror (memory/
    /// device_workspace_budget.hpp) the GPU builder's Create-time mode
    /// decision runs against - the FastPath/LightPath surface
    /// generalized to the device context (the fast rung keeps the device
    /// tables resident; the light rung uploads them per call and releases
    /// them at the call end). Ignored by the CPU builders (they read
    /// workspaceBudget); the GPU builder forwards it into EriCudaOptions.
    /// Null (the default) keeps the legacy behavior: the GPU engine always
    /// takes the fast rung unless the knob below forces the light rung.
    qcx::memory::DeviceWorkspaceBudget* deviceWorkspaceBudget = nullptr;
    /// The forced-LightPath knob of the GPU builder -
    /// non-zero FORCES the per-call-statics rung (the mode-forcing test
    /// surface, the same role as lightPathChunkPairs on the CPU: the
    /// budget-driven LightPath is unreachable at small scales). Forwarded
    /// into EriCudaOptions by the GPU builder; ignored by the CPU builders.
    bool forceGpuLightPath = false;
};

/// Resolves the certified fp32 lane's REQUEST - the one point every
/// Fock builder reads it from (an explicit request resolves at one
/// point, and the resolution IS the request, never a second opinion).
///
/// An explicit `useCertifiedMixedPrecision` is returned as written. An
/// unset one is the DEVICE's: the device probe's verdict on the run's
/// measured fp32/fp64 throughput ratio (CertifiedLaneDefaultForRatio).
/// The unseeded default carries the probe's own "unknown" profile, whose
/// ratio of 1.0 is below the threshold by construction, so a run that
/// supplies no probe result gets the conservative interim - the lane off.
/// The PRESET gate (MixedPrecisionThreshold) is a separate, later
/// condition and is not part of this resolution.
/// \param options The build options.
/// \returns True when the lane is requested for this build.
/// \ingroup qcx-integrals
inline bool ResolveCertifiedLane(const FockBuildOptions& options) noexcept {
    return options.useCertifiedMixedPrecision.value_or(
        CertifiedLaneDefaultForRatio(options.deviceComputeProfile.fp32ToFp64Ratio));
}

/// The Create-time build mode:
/// decided once per builder, deterministically, from the footprint estimate
/// against the budget's remaining bytes at Create (the monotone
/// accounting - a failed Reserve charges nothing and the budget is
/// shared by pointer through the nested builders, whose reservations order
/// by construction = nesting order).
enum class FockBuildMode {
    kFastPath, ///< The full store: the estimate fits the budget, the reservation
               ///< succeeded, the batch cap was clamped if needed.
    kLightPath, ///< The per-chunk light rung: the geometry-only light store
                ///< with the chunked pattern and transforms. The
                ///< estimate cannot fit the fast rung (or an a-priori
                ///< exclusion fired) and the light rung's clamped estimate
                ///< fits the remaining budget without exceeding the fast
                ///< path's estimate; otherwise Create refuses with the
                ///< ladder diagnostics.
    kDisk ///< The disk-backed rung: the (uv|P) tensor's chunks live in the
          ///< storage-module chunked store (storage/disk_ri_fock_build.hpp)
          ///< and BuildFock streams two passes over them. NO engine sets
          ///< this value - integrals cannot link storage, so
          ///< the engine's Create-time ladder stops at the refusal that
          ///< names the disk store; the DRIVER's disk route (the explicit
          ///< opt-in method.ri_tensor_mode = "disk" knob)
          ///< constructs the storage builder as the ladder's LAST rung
          ///< after that refusal and records this mode. The mode record is
          ///< the only consumer (purely observational, like every mode).
};

/// The Create-time mode record (FockModeInfo; every byte field is the
/// fired estimate's term - the fields not applicable to the builder's
/// family stay zero). Purely observational - no later code reads the mode
/// or the terms; the decision is consumed at Create.
/// \ingroup qcx-integrals
struct FockModeInfo {
    FockBuildMode mode = FockBuildMode::kFastPath; ///< The decision.
    std::size_t predictedBytes = 0; ///< The full Create-time footprint estimate.
    std::size_t reservedBytes = 0; ///< Bytes actually reserved (FastPath; the
                                   ///< nested-builder subtraction, RI/QFMM only).
    std::size_t budgetBytes = 0; ///< The budget's capacity at Create.
    std::size_t remainingAtDecision = 0; ///< Remaining() read at the decision.
    std::size_t maxBatchBytes = 0; ///< The batch cap in force (clamped when needed).
    /// The Create-time authorized concurrent batch slots (the
    /// bounded-concurrency batch loop): k = min(max(1, floor(remaining /
    /// cap)), DefaultOmpTeamSize()) at the fired estimate on the budgeted
    /// fast path with the exchange engaged and the cache off - a
    /// free-standing run and a
    /// pre-reserved nested half (validatedSweepCount != 0, the RI-J
    /// stack's exchange) deciding from its post-reservation remaining; 1
    /// everywhere else. A later re-measurement reads this field to
    /// know the actual k a fired run carried.
    std::size_t concurrentSlots = 1;
    /// The decision's clamp-origin team read (the k = 5
    /// team-size read): DefaultOmpTeamSize() - the cached topology team
    /// bounded by the process-wide OmpThreadCeiling - read at Create on
    /// every budgeted path (the fast and light decisions both size their
    /// per-thread arenas with it, and the fast decision clamps
    /// k = min(...) against it when the exchange's slots are authorized).
    /// A run whose concurrentSlots equals this field was team-clamped (the
    /// ceiling bit), one below it was budget-clamped. Read-only semantics
    /// of the backend seam - the field records the read, it does not size
    /// anything; 0 on the legacy no-budget path, which never consults the
    /// team at Create (mirroring the absent mode record).
    std::size_t defaultTeamSize = 0;
    std::size_t pairStoreBytes = 0; ///< The MD pair data (E tables, transforms, weights).
    std::size_t patternBytes = 0; ///< The neighbor CSR indices (8 bytes per surviving pair).
    std::size_t scratchBytes = 0; ///< The per-thread batch arena (batch x threads).
    std::size_t structuralBytes = 0; ///< Pair list, Schwarz vector, CSR offsets, core-H copies.
    std::size_t cacheBytes = 0; ///< The ERI cache (two lanes x maxCacheBytes, plain path only).
    std::size_t lightStoreBytes = 0; ///< The LightPath light store: nPairs geometry-only
                                     ///< MdPairData entries (the per-pair payload the fast
                                     ///< store's transforms would add is materialized per
                                     ///< chunk instead).
    std::size_t chunkArenaBytes = 0; ///< The LightPath peak-chunk arena: the chunk's bra
                                     ///< rows plus its distinct kets (plus the class-path
                                     ///< orbit representatives), at the max per-pair
                                     ///< payload - the peak-chunk rule: O(batch), never
                                     ///< the union over chunks.
    std::size_t chunkPatternBytes = 0; ///< The LightPath peak-chunk pattern: the largest
                                       ///< chunk's counted surviving rows (the per-chunk CSR,
                                       ///< capacity-preserved across chunks).
    std::size_t chunkIndexBytes = 0; ///< The LightPath per-call chunk-pair bookkeeping: the
                                     ///< stamp vector and the chunk pair set (nPairs index
                                     ///< entries each, live for the whole build - the peak
                                     ///< is the union, never a sum over chunks).
    std::size_t lightShellsBytes = 0; ///< The LightPath retained flattened shells (one
                                      ///< MdShellInput per shell, built at Create for the
                                      ///< per-chunk transform fills).
    std::size_t chunkPairs = 0; ///< The LightPath chunk size in bra rows (the knob's value
                                ///< when forced, the auto-sized C otherwise).
    std::size_t diskBytes = 0; ///< The disk rung's modeled on-disk (uv|P) payload, the
                               ///< dense 8 n^2 nAux across the chunked store: chunk stores
                               ///< are dense - screened-out blocks land as zeros, so the
                               ///< sum over chunks equals the dense model exactly. Named
                               ///< in the model but NOT reserved from RAM (the RAM
                               ///< reservation list is the composition:
                               ///< chunk arena, orbital-store rebuild, chunk-scoped task
                               ///< list, metric, exchange-side terms); the driver's disk
                               ///< route synthesizes the record (no engine sets the term -
                               ///< engine records stay zero), and the reconciliation
                               ///< compares it against the store's measured file bytes.
    std::size_t tensorBytes = 0; ///< The RI-J (uv|P) values buffer (8 n^2 nAux) - the
                                 ///< exclusion (ii) term.
    std::size_t riMatrixBytes = 0; ///< The RI-J retained n^2 x nAux Eigen copy of the tensor.
    std::size_t taskListBytes =
        0; ///< The RI-J screened task-list charge (16 B per surviving task).
    std::size_t metricBytes = 0; ///< The RI-J metric store (phantom holes), values and
                                 ///< eigendecomposition.
    std::size_t orbitalAuxBytes = 0; ///< The RI-J orbital and aux pair stores.
    std::size_t outerStoreBytes = 0; ///< The QFMM outer store: the octree node vector (the
                                     ///< leaf model - the moments and the restriction bitset
                                     ///< ride the QFMM lane's own record fields).
    std::size_t exchangeBytes = 0; ///< The nested direct-exchange half's own estimate
                                   ///< (RI-J: its pair store, pattern, scratch, structural
                                   ///< terms; charged at ITS Create).
    std::size_t exchangeScratchBytes = 0; ///< The nested half's batch arena (clamped).
    bool patternExcluded = false; ///< Exclusion (i): even the counted
                                  ///< Schwarz-screened pattern cannot fit - LightPath.
    bool tensorExcluded = false; ///< Exclusion (ii, RI-J): the tensor term alone
                                 ///< cannot fit - the light rung is mandatory.
    bool blockedMetricRung = false; ///< RI-J: the blocked-metric rung engaged:
                                    ///< the light rung's own estimate
                                    ///< could not fit, and the metric build blocked instead
                                    ///< (the strip-wise BuildBlockedAuxMetric - no full
                                    ///< values buffer, no Tensor/TensorToEigen copies, a
                                    ///< 3 x 8 nAuxFuncs^2 eigen-class peak against the
                                    ///< unblocked 6 x). mode stays kLightPath (the
                                    ///< per-iteration recompute is the light payload); the
                                    ///< marker is read only at Create for the metric-build
                                    ///< dispatch and is purely observational after.
    std::size_t classTableBytes = 0; ///< The pair-class table's Create-time structural
                                     ///< charge (footprint.hpp ClassTableBytes - the never-under
                                     ///< worst case over the canonical member-quartet space;
                                     ///< the live table is the on-demand screened subset, so
                                     ///< the charge is a ceiling), charged on the class path; 0
                                     ///< when the path disengaged or was never requested.
    bool classPathDisengaged = false; ///< The class-path admission gate fired (in place
                                      ///< since 2026-08-31): the table's never-under
                                      ///< estimate could not fit the budget's remaining bytes,
                                      ///< so the plain screened path ran instead - the
                                      ///< reduction was dropped, the results are bit-identical.
};

/// The certified-bound global budget enforcement's per-call outcome
/// (FockBuildOptions::enforceCertifiedBoundBudget): what the build's routed
/// bound sum was, what budget the accuracy preset derived for it, and which
/// way the comparison went. Written only by a build that ran the
/// enforcement; every member keeps its zero default otherwise, so a caller
/// must read Enforced first - an unenforced call's zeros are not
/// measurements.
/// \ingroup qcx-integrals
struct CertifiedBudgetOutcome {
    /// The enforcement ran on this call (the option was set and the route
    /// admitted it), so every member below is a measurement.
    bool enforced = false;
    /// The derived budget in Eh: CertifiedBoundBudget(accuracy, screenedHa).
    /// 0.0 means the preset's own screening co-term consumed the target.
    double budgetHa = 0.0;
    /// The build's screened-OUT certified bound sum in Eh (the budget's
    /// B_screen co-term) - the quantity the budget was derived with.
    double screenedHa = 0.0;
    /// The routed certified bound sum in Eh (the routing gate's own bound
    /// units) - the quantity COMPARED against the budget. Only an enforced
    /// call fills it.
    double routedHa = 0.0;
    /// The candidates the routing gate admitted before the comparison
    /// (the fp32 task-list size the routing pass produced).
    std::size_t routedQuartets = 0;
    /// The comparison's verdict: the routed sum did not fit the budget and
    /// the build ran its whole quartet set on the fp64 lane - the same
    /// quartet set on the same code path as useCertifiedMixedPrecision =
    /// false, so the two agree to the builder's own run-to-run
    /// reproducibility, which is BIT-IDENTITY since the fixed-order
    /// join (internal/fixed_order_reduce.hpp): two builds of one density
    /// agree exactly, so the fall-back's agreement is no longer read
    /// against a last-bit floor. False keeps the ordinary certified
    /// routing.
    bool fellBackToFp64 = false;
};

/// Builds F(D) = H + 2J(rho) - K(rho) for closed-shell RHF from the batched
/// MD engines, contracting each computed quartet with the density on the
/// fly; with \p FockBuildOptions::buildCoulombOnly / buildExchangeOnly it
/// builds the split H + 2J(rho) and H - K(rho) forms the direct-UHF adapter
/// composes per spin. The input density is the SPATIAL closed-shell
/// density rho = D/2 (spin-summed density over 2) - the direct/RI builder
/// convention; the scf FockBuilderFn seam hands over the spin-summed D, so
/// adapters must scale by 1/2 before calling BuildFock (rhf.hpp documents
/// the seam contract).
///
/// J and K are accumulated with the 8-fold symmetry folded in: every
/// canonical quartet block contributes to the bra and ket pair blocks of J
/// and to the four exchange targets of K (the papers' rule - the density is
/// permuted per batch, integrals are never transposed).
/// \ingroup qcx-integrals
class DirectJkFockBuilder {
public:
    /// Prepares the builder: flattens the pairs, builds the pair data and
    /// the Schwarz bounds, and copies the core Hamiltonian.
    /// \param molecule Molecule providing the atom coordinates (Bohr).
    /// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
    /// (kUnimplemented otherwise).
    /// \param coreHamiltonian H = T + V, host-canonical rank-2 tensor with
    /// shape {n, n}.
    /// \param options Build settings; maxBatchBytes must be positive.
    /// \returns The builder, or an Error.
    static qcx::Result<DirectJkFockBuilder> Create(
        const qcx::molecule::Molecule& molecule,
        const qcx::basisset::BasisSet& basisSet,
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
        const FockBuildOptions& options = {});

    /// Builds F(D) = H + 2J(rho) - K(rho) for the given closed-shell
    /// density.
    /// \param density The SPATIAL closed-shell density rho = D/2 (spin-summed
    /// density over 2), host-canonical rank-2 tensor with shape {n, n}. Must
    /// be SYMMETRIC: the canonical-pair contractions read both orientations
    /// of every unordered pair block (d(c,d) + d(d,c)) and the K
    /// transpose-writes assume it (a Debug assert checks it in Debug builds).
    /// \param certifiedBoundSumOut Optional out-parameter receiving the sum
    /// of the DENSITY-WEIGHTED a-priori certified bounds of every quartet
    /// the routing gate sent through the fp32 lane (0.0 when the lane is
    /// disabled): each quartet's kernel bound times its max-|D| block weight
    /// - the same weight the routing gate uses. The weighted sum bounds
    /// every Fock-element error from the fp32 lane (an upper bound on the
    /// delivered Fock error, consumable as certified); the sweep
    /// checks the per-quartet bounds against the preset's J/K budgets.
    /// \param statsOut Optional per-call diagnostics sink (FockBuildStats);
    /// null (the default) keeps the zero-cost path.
    /// \param ladderInputs The per-call precision-ladder inputs
    /// (PrecisionLadderInputs: the SCF's density-convergence error, the
    /// composed RI path's measured error, and the run's escalation
    /// schedule). Null (the default) keeps the pre-ladder two-pass build
    /// bit-identical. The ladder runs the v1 surface: the plain path
    /// (no class reduction) FastPath only - the LightPath and the class
    /// path return kUnimplemented when the ladder is engaged (their
    /// ladder composition is the documented follow-on).
    /// \param budgetOut Optional out-parameter receiving the global
    /// budget enforcement's outcome (CertifiedBudgetOutcome): the routed
    /// bound sum, the preset-derived budget, and the comparison's verdict.
    /// Filled only when FockBuildOptions::enforceCertifiedBoundBudget is set
    /// and the route admits the enforcement; the LightPath and an engaged
    /// precision ladder return kUnimplemented instead, so the option is
    /// never silently dropped. Null (the default) keeps the zero-cost path.
    /// \returns The Fock matrix as a rank-2 tensor, or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildFock(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        double* certifiedBoundSumOut = nullptr,
        FockBuildStats* statsOut = nullptr,
        const PrecisionLadderInputs* ladderInputs = nullptr,
        CertifiedBudgetOutcome* budgetOut = nullptr) const;

    /// The core Hamiltonian this builder was constructed with (H in
    /// F = H + 2J(rho) - K(rho)). Exposed for callers that need to combine
    /// multiple BuildFock results and must account for H being included in
    /// each one - e.g. IncrementalFockBuilder's delta-accumulation.
    /// \returns The construction-time H, host-canonical rank-2 tensor with
    /// shape {n, n}; a reference into this builder's state (valid for the
    /// builder's lifetime, stable across calls).
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& CoreHamiltonian() const;

    /// The in-memory ERI cache's cumulative stats, or nullptr
    /// when no engine tier was built at all: neither a positive
    /// maxCacheBytes nor a set engineDecorator, or the class-aware path. A
    /// zero maxCacheBytes beside a set decorator yields a PASSTHROUGH tier -
    /// a live cache that admits nothing and serves every request through the
    /// decorator - so its stats are real and this is non-null; read the
    /// decorator's own stats for the store's numbers (the driver holds it).
    /// The stats accumulate over the builder's BuildFock calls
    /// (EriCacheStats); per-call cache hits are the FockBuildStats
    /// cacheHit*QuartetCount deltas.
    /// \returns The cache stats, or nullptr.
    const EriCacheStats* CacheStats() const noexcept;

    /// The Create-time mode record (adaptive memory): the mode decision, the
    /// fired estimate terms
    /// and the reservation outcome; nullopt when no workspace budget was
    /// given (the legacy path - no decision was made).
    /// \returns The mode record, or nullopt on the legacy path.
    const std::optional<FockModeInfo>& ModeInfo() const noexcept;

private:
    // The implementation state lives in the .cpp (Eigen stays
    // implementation-only).
    struct State;
    explicit DirectJkFockBuilder(std::shared_ptr<const State> state);
    std::shared_ptr<const State> _state;
};

} // namespace qcx::integrals
