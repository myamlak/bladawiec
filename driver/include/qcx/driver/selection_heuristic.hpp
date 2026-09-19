#pragma once

/// \file
/// The algorithm-selection heuristic's cost table: one
/// order-of-magnitude cost estimate per Fock-builder candidate - direct,
/// ri_j_link, ri_jk, qfmm, gpu, gpu_split - as a function of (nBasis,
/// nPairs, effectiveThreads, cpuBudget, deviceBudget). An order-of-
/// magnitude heuristic, not a performance model: "usually good enough" is
/// the bar, refined by crossover measurements when they
/// land (the constants are one named struct, not
/// scattered literals, so the measured values replace the seeds in one
/// place).
///
/// The priority drivers of the design (GPU presence -> GPU memory
/// available -> CPU memory cap -> thread count) are expressed as
/// admission floors: a candidate is selectable only when its modeled
/// footprint fits its budget (the GPU family against the device budget,
/// the CPU family against the memory cap) and the run could actually wire
/// it today. ri_jk stays OUT of the ranking DELIBERATELY: its builder
/// exists and runs on both legs since the unrestricted wiring, but it is
/// an APPROXIMATED-exchange path
/// whose error is disclosed rather than clean (3 of 6 cells met, and the
/// miss is preset-invariant because the aux fit sets it) - so it is opted into
/// BY NAME and never chosen for the user. The nullopt is deliberate, not a
/// leftover; while
/// gpu_split IS wireable: its io kind
/// exists (BuilderKindOf maps it) and its default state - the zero device
/// share - executes the CPU-only fallback, the k = 1 path unchanged,
/// so the row is selectable at exactly the direct cost and never
/// outranks direct. The co-execution partition itself (the device suffix
/// behind the gpuSplitDeviceFraction) is not executed yet.
/// The non-single-node columns are stubs: nodeCount > 1 reports
/// every candidate "not selectable" (MPI is not wired yet).
///
/// All seeds are recorded or provisional as marked on the constant;
/// nothing here is a performance claim. The estimator is a pure function
/// of its input - no probing, no state - so the selection is testable and
/// the driver reports it verbatim in resources_resolved.

#include "qcx/io/run_input.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace qcx::driver {

/// The cost-table candidate set: the five runnable-in-schema builder
/// kinds plus gpu_split. gpu_split (the intra-build CPU+GPU batch
/// partition candidate) deliberately has
/// no io [method] schema word - it is a heuristic candidate, not a
/// builder the user can request - and its split shape must NOT be read as
/// the inter-iteration CPU/GPU
/// pipelining shape (SCF-dependency-capped:
/// lagging the density would change the physics), while gpu_split is the
/// INTRA-build batch partition of one Fock build. The two overlap shapes
/// are distinct; this candidate's design home is the batch partition only.
/// \ingroup qcx-driver
enum class CostCandidate {
    kDirect, ///< The batched-MD direct builder.
    kRiJLink, ///< RI-J with direct exchange.
    kRiJk, ///< Full RI exchange - IMPLEMENTED and running on both legs;
           ///< deliberately kept out of the RANKING (see BuilderKindOf), not out of the build.
    kQfmm, ///< The octree multipole builder.
    kGpu, ///< The CUDA device builder.
    /// Intra-build CPU+GPU batch partition. Selectable as wired:
    /// the io kind exists and the default zero device
    /// share wires the CPU-only fallback (the k = 1 path); the
    /// partition execution itself is not wired yet.
    kGpuSplit,
};

/// The builder kind of a candidate, for the driver to wire a pick through
/// the io vocabulary. Only kRiJk maps to nullopt, and that is deliberate rather
/// than a state of the builder: ri_jk exists (`RiFullFockBuilder`)
/// and runs on both legs, but it is an approximated-exchange
/// path and stays out of the automatic ranking so it is never chosen for a user
/// who did not name it. kGpuSplit maps to io::BuilderKind::kGpuSplit -
/// the pick vocabulary names it today (its kind exists and reports), even though no execution path
/// consumes it yet.
/// \param candidate The candidate.
/// \returns The builder kind, or nullopt when the candidate has no
/// builder in v1.
/// \ingroup qcx-driver
std::optional<qcx::io::BuilderKind> BuilderKindOf(CostCandidate candidate) noexcept;

/// The seeded cost-table constants (measured
/// crossover constants replace these in one place when they land). Every
/// value carries its recorded source or the provisional tag in its
/// comment; the calibration step re-measures the timed ones on an
/// otherwise-idle machine.
///
/// The LightPath pricing pair (the former directLightPathCapFactor and
/// directLightPathPenalty) was REMOVED with the predictive memory model:
/// its trigger was the model's direct-family floor,
/// so with the model gone it had no trigger, and no
/// model-derived floor is to be reconstructed here. Nothing prices the
/// direct path up for a tight cap any more - the estimator ranks on cost
/// alone and the job-object cap is the admission.
/// \ingroup qcx-driver
struct SelectionCostConstants {
    /// The direct builder's per-Fock-build constant, in seconds x threads
    /// per n^3. ANCHOR DISQUALIFIED, VALUE RETAINED UNTIL RE-ANCHORED
    /// (re-anchoring is a measurement, tracked separately): the cited capture
    /// (2026-08-29) was NEVER COMMITTED - no git
    /// object, not gitignored, a lost scratch capture rather than a
    /// fabricated citation - and its 5717 ms is the PRE-correction alkane
    /// two-level row, already disowned in-repo as superseded history,
    /// i.e. a timing of the defective J-mode gate, since corrected. The corrected
    /// same-fixture, same-day row is 2478 ms, a 2.31x
    /// gap. The seed's team factor (12.0) was uncorroborated, and it goes
    /// with the superseded value it scaled.
    /// The value below is the 2026-09-13 clean-machine re-anchor: a 14697.3 ms
    /// median (MAD 309.9 ms over 7 independent process repetitions) for one
    /// two-level BuildFock on the committed C80H162/STO-3G alkane vehicle at
    /// n = 562 - the 14.6973 s the expression below is built from. THE TEAM IS
    /// PART OF THE VALUE, so it is stated in the same breath: the run reports
    /// the team it fired on its own mode line, "k 5 | team 5 (team-clamped)",
    /// so 5 is read back rather than assumed - and the same measurement read
    /// at the machine's 12 logical CPUs is 2.4x this number (9.936e-07 in the
    /// same units), which is the size of the mis-read a team-less constant
    /// invites. Its 7.1% agreement with the disqualified seed is ARITHMETIC
    /// COINCIDENCE, NOT CORROBORATION: the measured wall is 2.571x the seed's
    /// while the team factor folded in here is 5/12 of the seed's, and two
    /// independent movements that very nearly cancel validate neither number.
    /// One laptop (i7-9850H, 12 logical / 6 physical), one data point - this
    /// project forbids hardware-general conclusions from it. Regime: the
    /// budgeted fast path at k = 5, not the k = 1 row the seed cited, which
    /// the committed vehicle cannot reproduce (its alkane fixture always
    /// carries a workspace budget). A pre-HEAD binary of the same benchmark
    /// is not this number either: its certified fp32 lane resolved on, the
    /// committed one resolves it off, and that flip is confounded with
    /// machine load - no attribution between the two was made.
    /// The model folds the screened-quartet work into an n^3 law (the
    /// density-screened scaling measures ~n^2.7 over the recorded range - n^3
    /// is the conservative ceiling) and anchors the constant there.
    double directPerNCubedSecondsThreads = 14.6973 * 5.0 / (562.0 * 562.0 * 562.0);
    /// The ri_j_link per-iteration share of the fused direct cost: its
    /// exchange half is the same direct K work, its Coulomb half the
    /// cheaper RI-J tensor build. Provisional - a crossover measurement sets
    /// the real ratio.
    double riJkShareOfDirect = 0.65;
    /// The ri_j_link tensor build + contraction constant, in seconds x
    /// threads per (n^2 x nAux). The tensor work is anchored on the
    /// 2026-08-29 record: the c24h50_def2svp-class tensor (4.416 GiB at
    /// n = 586, nAux ~ 1600) writes and contracts in the ~0.5 s/iteration
    /// class. The 12.0 factor that carries it to seconds x threads is
    /// UNCORROBORATED, for the reason directPerNCubedSecondsThreads and
    /// gpuDirectAnchorThreads name: the anchor run's team size is stated
    /// nowhere in the tree (only the corrected run's "12 cores" machine
    /// is), so the factor is unverified rather than recorded - not a value
    /// the model can point at a measurement for.
    double riJTensorPerNAuxN2SecondsThreads = 0.5 * 12.0 / (586.0 * 586.0 * 1608.0);
    /// The ri_j_link builder-setup cost in seconds, flat (the metric
    /// inverse is single-threaded - it does not scale with the team).
    /// Recorded 2026-08-29: ~40 s per create on the small
    /// fixtures, minutes on the large ones.
    double riJCreateSeconds = 40.0;
    /// The ri_jk (full RI exchange) tensor-work multiple of ri_j_link's:
    /// J and K both run tensor-driven, roughly three times the ri_j
    /// tensor work. Provisional - the seed awaits a measured re-anchor; the
    /// PATH itself is no longer pending (the builder landed, both legs run it).
    double riJkTensorMultiple = 3.0;
    /// The qfmm builder's fixed per-build cost in seconds (octree build +
    /// near-field list). Provisional - no clean-machine qfmm timing is
    /// recorded yet; a crossover measurement calibrates it.
    double qfmmFixedSeconds = 1.0;
    /// The qfmm per-iteration share of the fused direct cost: the
    /// far-field compression beats direct at scale, at parity-plus-
    /// overhead below the crossover. Provisional - a crossover measurement
    /// sets the real curve.
    double qfmmVsDirect = 0.7;
    /// The GPU candidate's speedup over the CPU team (the fp32-certified
    /// device lane vs the fp64 team). Provisional - a device-profile
    /// probe measures the throughput ratio and the
    /// crossover; the design's 2-8x fp32-vs-fp64 hardware-unit band is
    /// the sanity range.
    double gpuSpeedup = 4.0;
    /// The GPU candidate's fixed per-build cost in seconds (kernel launch
    /// + the per-iteration density/Fock transfer floor). Provisional.
    double gpuFixedSeconds = 0.05;
    /// The team size the GPU candidate's direct-cost anchor assumes - the
    /// 12.0 factor folded into directPerNCubedSecondsThreads above, and
    /// UNCORROBORATED for the reason that constant names: the anchor run's
    /// team size is stated nowhere in the tree (only the corrected run's
    /// "12 cores" machine is), so this factor is unverified rather than
    /// recorded - not a value the model can point at a measurement for.
    /// The device's wall time does not depend on the CPU team, so the GPU
    /// estimate never scales with effectiveThreads - only the CPU candidates
    /// and the gpu_split CPU share do.
    double gpuDirectAnchorThreads = 12.0;
    /// The GPU residency footprint's base term in GiB (CUDA context,
    /// statics, screening tables - the "statics once"
    /// store). Provisional; a device-budget measurement sets it.
    double gpuFootprintBaseGiB = 0.25;
    /// The GPU residency footprint's matrix term, GiB per n^2 (the
    /// density and Fock matrices live on device, 8 bytes each). Exact by
    /// construction.
    double gpuFootprintMatrixPerN2GiB = 2.0 * 8.0 / 1073741824.0;
    /// The GPU residency footprint's pair-store term, bytes per shell
    /// pair (the pair store + Schwarz neighbor CSR; a measurement recorded
    /// the direct stores at 7-45 MB across the fixtures,
    /// ~225 B/pair at the top - 256 is the conservative seed). Provisional.
    double gpuFootprintPerPairBytes = 256.0;
    /// The CUDA build's host-footprint term in GiB, folded into the GPU
    /// tier's admission floor: a CUDA-linked process holds this much host
    /// commit before any CUDA call - the runtime DLLs (cuBLAS is a
    /// load-time import), committed at process start and counted against
    /// the job-object cap once the driver applies it. Measured 2026-08-30
    /// (the reference T1000 CUDA gate, cap-child bisect): a
    /// process-memory cap at or below 0.25 GiB fails the run with no
    /// marker - a hard access violation when the topology probe's init
    /// makes the first failing commit, an allocation fast-fail
    /// (0xC0000409) once the probe is gated - while a 0.3 GiB cap
    /// completes. The probe's own marginal commit is at most ~0.05 GiB;
    /// the footprint is dominated by the load-time runtime. 0.35 = the
    /// upper bound plus headroom. The run's cap must hold this footprint
    /// before the device probe may run (the driver's probe gate, see
    /// GpuProbeFloorGiB): below the floor the tier cannot fit the cap
    /// anyway, and the probe's init under a sub-footprint cap is a hard
    /// access violation, never a clean CUDA error. A device-budget
    /// measurement sets it properly. Provisional.
    ///
    /// The memory model's base term was once added to this floor; with
    /// the model deleted the floor is this
    /// commit alone, which is a real loosening, recorded rather than
    /// hidden.
    double gpuProbeCommitGiB = 0.35;
    /// The gpu_split candidate's batch-boundary coordination overhead as
    /// a share of the co-execution cost (the rendezvous + the fixed-order
    /// CPU + device combine once both sides ran). Applied only when the
    /// device share is nonzero - the zero-share CPU-only fallback prices
    /// exactly the direct cost. Provisional; a measured median
    /// replaces it (and gpuSpeedup).
    double gpuSplitOverhead = 0.05;
    /// The gpu_split candidate's DEVICE-side share of the cumulative
    /// estimated batch cost: the GPU suffix takes this share of the
    /// canonical batch stream at the nearest equal-ket group boundary, the
    /// CPU prefix the rest (the batch-partition contract: cost-based
    /// and group-boundary-aware, never a raw batch count, never inside a
    /// group). The default ZERO is the gate: with the share at
    /// zero the suffix is empty and the candidate executes the CPU-only
    /// fallback - the k = 1 path unchanged, and the row prices exactly
    /// the direct cost, so the fixed candidate order keeps direct ahead on
    /// the exact tie. A measured median replaces the gate with
    /// the balanced concurrent-run value gpuSpeedup/(1+gpuSpeedup), whose
    /// CPU-side complement 1/(1+gpuSpeedup) is the pre-existing balanced
    /// share (selection_heuristic.cpp). In [0, 1]; provisional.
    double gpuSplitDeviceFraction = 0.0;
    /// The display/transient headroom the device budget subtracts from
    /// the probed free VRAM. Provisional.
    double deviceHeadroomGiB = 0.5;
};

/// The estimator's pure input: everything the cost table needs, probed
/// once at driver start by the caller.
/// \ingroup qcx-driver
struct SelectionInput {
    std::size_t nBasis = 0; ///< Orbital basis-function count.
    std::size_t nAux = 0; ///< Auxiliary basis-function count; 0 for the direct family.
    /// Why ri_j_link cannot be wired, when it cannot (the run has no
    /// [basis].aux and no auto-selectable aux exists, or the aux parse
    /// failed): the wiring would refuse at the seam, so the estimator
    /// marks the candidate not-selectable instead of pricing it at the
    /// phantom zero-aux cost and floor. Empty = ri_j is selectable per
    /// the fit.
    std::optional<std::string> riJUnavailableReason;
    std::size_t nPairs = 0; ///< Shell-pair count (the pair-store sizing input).
    /// The effective team size (threadCap-clamped); scales the CPU costs.
    int effectiveThreads = 1;
    /// The CPU memory cap in GiB (the [resources] memory_cap_gib; the
    /// admission floor is the memory model's modeled peak).
    double cpuBudgetGiB = 16.0;
    /// The device budget in GiB (free VRAM minus the headroom constant);
    /// 0 = no device tier. The GPU family's admission floor.
    double deviceBudgetGiB = 0.0;
    /// The node count (1 today; MPI is not wired - every candidate
    /// reports "not selectable" beyond one node, the stub columns).
    std::size_t nodeCount = 1;
    /// The constants in force; default-constructed = the seeded values.
    SelectionCostConstants constants;
};

/// One candidate's estimate.
/// \ingroup qcx-driver
struct CandidateEstimate {
    CostCandidate candidate = CostCandidate::kDirect; ///< The candidate this row estimates.
    /// The order-of-magnitude per-Fock-build wall time in seconds
    /// (selectable candidates only; not-selectable ones carry 0.0).
    double costSeconds = 0.0;
    /// True when the run could wire this candidate today under the given
    /// budgets (admission floors fit AND the builder exists in v1).
    bool selectable = false;
    /// Why: the admission verdict for selectable candidates, the reason
    /// otherwise ("no CUDA device", "device footprint X GiB does not fit
    /// Y GiB free", "models X GiB but the cap is Y GiB", "not wired in
    /// v1", "distributed execution is not wired").
    std::string reason;
};

/// The estimator's output: the candidates ranked into a total order
/// (selectable first, ascending cost; not-selectable after, ascending
/// cost) plus the pick and the reasoning string for resources_resolved.
/// \ingroup qcx-driver
struct SelectionRanking {
    /// All six candidates, rank order (see the struct note).
    std::vector<CandidateEstimate> candidates;
    /// The best selectable candidate; nullopt when none fits (the driver
    /// then refuses through the admission gate with its full ladder
    /// diagnostics - the estimator only reports).
    std::optional<qcx::io::BuilderKind> bestSelectable;
    /// The human reasoning string for resources_resolved, e.g.
    /// "gpu: device present, footprint 1.2 GiB fits 3.1 GiB free;
    /// threads 8". Total - always filled, whatever the outcome.
    std::string reasoning;
};

/// Estimates every candidate and ranks them (see SelectionRanking).
/// Pure and total: no probing, no I/O, no failure - nBasis = 0 ranks
/// everything "not selectable" ("no basis functions") rather than error.
/// \param input The probed size, budgets and constants.
/// \returns The ranked candidates with the pick and reasoning string.
/// \ingroup qcx-driver
SelectionRanking EstimateCandidates(const SelectionInput& input) noexcept;

/// The GPU tier's minimum host-side admission floor in GiB: the CUDA
/// probe's host commit (gpuProbeCommitGiB), the runtime DLLs' load-time
/// footprint. A cap below this floor cannot hold a GPU run - the tier is
/// unselectable - so the driver's probe gate (run_driver.cpp) must not
/// even start the device sweep below it: the sweep's init would be a hard
/// access violation under the cap, never a clean error, and its result
/// could not change the selection.
///
/// The memory model's base term was once part of this sum; with the model
/// deleted the floor is consequently lower by that
/// term, which is recorded rather than hidden.
/// \param constants The constants in force.
/// \returns The floor in GiB.
/// \ingroup qcx-driver
double GpuProbeFloorGiB(const SelectionCostConstants& constants) noexcept;

} // namespace qcx::driver
