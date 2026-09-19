// The algorithm-selection cost table (selection_heuristic.hpp): one
// order-of-magnitude estimate per Fock-builder candidate against the
// probed size, budgets and threads. Pure and total - the driver probes
// once and reports the reasoning string verbatim.

#include "qcx/driver/selection_heuristic.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <utility>

namespace qcx::driver {
namespace {

// The GiB conversion (2^30 bytes).
constexpr double kGiB = 1073741824.0;

// The stub reason of the non-single-node columns: MPI
// is not wired and nothing distributes a Fock build today.
constexpr std::string_view kDistributedStub =
    "distributed execution is not wired; single-node runs only";

// The candidate's short name for the reasoning strings.
std::string_view NameOf(CostCandidate candidate) {
    switch (candidate)
    {
    case CostCandidate::kDirect:
        return "direct";
    case CostCandidate::kRiJLink:
        return "ri_j_link";
    case CostCandidate::kRiJk:
        return "ri_jk";
    case CostCandidate::kQfmm:
        return "qfmm";
    case CostCandidate::kGpu:
        return "gpu";
    case CostCandidate::kGpuSplit:
        return "gpu_split";
    }

    return "unknown";
}

// Renders a value at one decimal, trailing zeros stripped ("48.2", "1.0"
// -> "1"), for the reasoning strings.
std::string FormatTrimmed(double value) {
    std::string s = std::to_string(std::round(value * 10.0) / 10.0);
    const std::size_t lastNonZero = s.find_last_not_of('0');

    if (lastNonZero == std::string::npos)
    {
        return "0";
    }

    s.erase(lastNonZero + 1);

    if (s.back() == '.')
    {
        s.pop_back();
    }

    return s;
}

// Renders a GiB value at one decimal, same trimming.
std::string FormatGiB(double gib) {
    return FormatTrimmed(gib);
}

// Renders a seconds value at one decimal (milliseconds below one second,
// so sub-second estimates do not read as "0 s").
std::string FormatSeconds(double seconds) {
    if (seconds < 1.0)
    {
        return FormatTrimmed(seconds * 1000.0) + " ms";
    }

    return FormatTrimmed(seconds) + " s";
}

// The GPU family's device-residency footprint:
// the statics base, the on-device density/Fock matrices, and the pair
// store. The seeds are the SelectionCostConstants fields; a
// device-budget measurement sets the real terms.
double GpuFootprintGiB(const SelectionInput& input) {
    const SelectionCostConstants& k = input.constants;
    const double n = static_cast<double>(input.nBasis);
    const double pairs = static_cast<double>(input.nPairs);
    return k.gpuFootprintBaseGiB + k.gpuFootprintMatrixPerN2GiB * n * n +
           k.gpuFootprintPerPairBytes * pairs / kGiB;
}

// The direct builder's per-Fock-build estimate at the team size; the
// anchor every other CPU candidate scales from.
double DirectCostSeconds(const SelectionInput& input) {
    const SelectionCostConstants& k = input.constants;
    const double n = static_cast<double>(input.nBasis);
    const double threads = static_cast<double>(std::max(input.effectiveThreads, 1));
    return k.directPerNCubedSecondsThreads * n * n * n / threads;
}

// (The LightPath pricing hook that stood here - a cap below the memory
// model's direct floor priced the direct path up - was deleted with the
// model: its trigger was the model's floor, so it had none.)

// One candidate's estimated cost and the verdict from the budgets. The
// wiring check comes last: an unwired candidate reports its cost too, so
// the table shows exactly what blocks once the path lands. The memory
// ADMISSION the model used to apply here (a floor against the cap) was
// deleted with the model: the job-object cap is the
// admission, and the engine's own Create-time estimates decide the rung.
CandidateEstimate Evaluate(const SelectionInput& input,
                           CostCandidate candidate,
                           bool wired,
                           double costSeconds,
                           bool needsDevice) {
    CandidateEstimate estimate;
    estimate.candidate = candidate;
    estimate.costSeconds = costSeconds;

    if (input.nBasis == 0)
    {
        estimate.reason = "no basis functions (nBasis = 0)";
        return estimate;
    }

    if (input.nodeCount > 1)
    {
        estimate.reason = std::string(kDistributedStub);
        return estimate;
    }

    if (!wired)
    {
        // Only kRiJk reaches here - every other candidate wires in v1
        // (gpu_split included: its io kind exists and its zero device
        // share is the CPU-only fallback).
        estimate.reason = "deliberately excluded from the ranking: ri_jk is an "
                          "approximated-exchange path, so it is opted into by name "
                          "rather than proposed here";
        return estimate;
    }

    if (needsDevice && input.deviceBudgetGiB <= 0.0)
    {
        estimate.reason =
            "no CUDA device (device budget " + FormatGiB(input.deviceBudgetGiB) + " GiB)";
        return estimate;
    }

    if (needsDevice && GpuFootprintGiB(input) > input.deviceBudgetGiB)
    {
        estimate.reason = "device footprint " + FormatGiB(GpuFootprintGiB(input)) +
                          " GiB does not fit " + FormatGiB(input.deviceBudgetGiB) + " GiB free";
        return estimate;
    }

    estimate.selectable = true;
    estimate.reason = "fits the budgets (the CPU cap is " + FormatGiB(input.cpuBudgetGiB) +
                      " GiB; the job-object cap is the admission)";
    return estimate;
}

// The exhaustiveness guard around this translation unit's two candidate
// classifiers. The diagnostics have to be promoted here to be worth
// anything: MSVC emits C4062/C4061 for an unhandled enumerator at neither
// /W3 nor /W4 (both are off-by-default), GCC emits nothing without -Wall,
// and Clang's -Wswitch is a warning nobody reads. C4061 and -Wswitch-enum
// are the variants that also catch a `default:` arm added to silence the
// check. Each region spans exactly its own function; every other switch in
// the translation unit keeps the project's default diagnostic settings.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(error : 4061)
#pragma warning(error : 4062)
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#pragma clang diagnostic error "-Wswitch-enum"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif

// The fixed candidate order exact ties break by (the sort is stable, so
// the total order stays deterministic when two costs are exactly equal).
//
// Exhaustive BY CONSTRUCTION - no `default:` arm, and the region above makes
// an unhandled enumerator a BUILD FAILURE - so a candidate added to
// CostCandidate cannot leave the tie-break order to a fall-through. The
// equality chain a sort key invites would have given every new candidate the
// same implicit slot, silently reordering which of two exactly-tied costs
// wins - a decision nobody wrote.
int CandidateOrderIndex(CostCandidate candidate) {
    switch (candidate)
    {
    case CostCandidate::kDirect:
        return 0;
    case CostCandidate::kRiJLink:
        return 1;
    case CostCandidate::kRiJk:
        return 2;
    case CostCandidate::kQfmm:
        return 3;
    case CostCandidate::kGpu:
        return 4;
    case CostCandidate::kGpuSplit:
        return 5;
    }

    // Reached only by a value no enumerator names (a programmatic caller's
    // out-of-range cast): the last slot, so no named candidate's order moves.
    return 6;
}

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__clang__)
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

} // namespace

// The candidate's builder kind, or nullopt when the candidate has no wired
// builder. The nullable builder-kind result is the shape to copy for a
// selector that must be able to answer "there is nothing to run here" - and,
// as a standing obligation, a resolver whose switch lacks
// the promoted diagnostics is non-conformant even if its arms are currently
// complete, which is what the region below supplies.
//
// The switch is exhaustive BY CONSTRUCTION - no `default:` arm, and the
// region makes an unhandled enumerator a BUILD FAILURE - so a candidate
// added to CostCandidate cannot leave the question unanswered. The equality
// test this shape replaces answers "nothing wired" for every candidate
// nobody had thought about yet, which is the safe answer only by accident: a
// candidate that DID have a builder would be dropped from the ranking's
// selectable set and the record's reasoning would read as though the
// heuristic had priced it and passed.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(error : 4061)
#pragma warning(error : 4062)
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#pragma clang diagnostic error "-Wswitch-enum"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif

std::optional<qcx::io::BuilderKind> BuilderKindOf(CostCandidate candidate) noexcept {
    switch (candidate)
    {
    case CostCandidate::kDirect:
        return qcx::io::BuilderKind::kDirect;
    case CostCandidate::kRiJLink:
        return qcx::io::BuilderKind::kRiJLink;
    // NOT because the builder is missing - it exists and runs on both
    // legs - but because ri_jk is an APPROXIMATED-exchange
    // path: it is opted into by name and never proposed by the ranking.
    // Removing this nullopt would let the heuristic route
    // a run onto a fitted exchange nobody asked for.
    case CostCandidate::kRiJk:
        return std::nullopt;
    case CostCandidate::kQfmm:
        return qcx::io::BuilderKind::kQfmm;
    case CostCandidate::kGpu:
        return qcx::io::BuilderKind::kGpu;
    case CostCandidate::kGpuSplit:
        return qcx::io::BuilderKind::kGpuSplit;
    }

    // Reached only by a value no enumerator names (a programmatic caller's
    // out-of-range cast): "nothing wired" is the only answer this build can
    // substantiate, and no named candidate's answer moves.
    return std::nullopt;
}

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__clang__)
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

SelectionRanking EstimateCandidates(const SelectionInput& input) noexcept {
    const SelectionCostConstants& k = input.constants;
    const double n = static_cast<double>(input.nBasis);
    const double m = static_cast<double>(input.nAux);
    const double threads = static_cast<double>(std::max(input.effectiveThreads, 1));

    SelectionRanking ranking;

    // The GPU candidate's direct-cost anchor at the constant's recorded
    // team: the device's wall time does not depend on the CPU team, so the
    // GPU estimate never scales with effectiveThreads.
    const double gpuAnchorDirect =
        k.directPerNCubedSecondsThreads * n * n * n / k.gpuDirectAnchorThreads;

    // (The five admission floors that stood here - one modeled peak per
    // candidate, fed to Evaluate against the cap - were deleted with the
    // memory model. A candidate is no longer excluded from
    // this ranking for its modeled size: the job-object cap is the
    // admission and the engine's own Create-time estimates decide each
    // builder's rung. The GPU candidate keeps its DEVICE-side footprint
    // check in Evaluate, which is a device budget, not the CPU memory
    // model.)

    const double directCost = DirectCostSeconds(input);
    const double riJTensorCost = k.riJTensorPerNAuxN2SecondsThreads * n * n * m / threads;
    // The gpu_split row (the batch-partition contract): the batch
    // stream splits on cumulative estimated batch cost at the nearest
    // equal-ket group boundary - the CPU prefix takes the (1 - g) share of
    // the direct row's work, the device suffix the g share at the speedup
    // rate, and the estimate is the shared completion time (the max of the
    // two sides) plus the coordination overhead. The DEVICE share g comes
    // from gpuSplitDeviceFraction; the pre-existing balanced CPU-side
    // reading 1/(1+gpuSpeedup) is its complement at the balanced value g =
    // gpuSpeedup/(1+gpuSpeedup) - a measured value replaces
    // the default zero gate. At the default g = 0 the
    // suffix is empty and the row IS the CPU-only fallback - the k = 1
    // path - so it prices exactly the direct row (no overhead: nothing
    // rendezvouses) and can never outrank
    // direct, the exact tie breaking by the fixed candidate order.
    const double directRow = directCost;
    const double gpuShare = std::clamp(k.gpuSplitDeviceFraction, 0.0, 1.0);
    const bool gpuSplitEngaged = gpuShare > 0.0;
    const double gpuSplitCost = gpuSplitEngaged ? std::max((1.0 - gpuShare) * directRow,
                                                           gpuShare * directRow / k.gpuSpeedup) *
                                                      (1.0 + k.gpuSplitOverhead)
                                                : directRow;

    ranking.candidates.push_back(Evaluate(input,
                                          CostCandidate::kDirect,
                                          /* wired */ true,
                                          directRow,
                                          /* needsDevice */ false));
    CandidateEstimate riJ =
        Evaluate(input,
                 CostCandidate::kRiJLink,
                 /* wired */ true,
                 k.riJkShareOfDirect * directCost + riJTensorCost + k.riJCreateSeconds,
                 /* needsDevice */ false);

    // The unwireable ri_j (no aux the wiring could auto-select): never
    // price it at the phantom zero-aux cost - the run would be refused at
    // the wiring seam while a selectable direct path ran.
    if (input.riJUnavailableReason.has_value())
    {
        riJ.selectable = false;
        riJ.reason = *input.riJUnavailableReason;
    }

    ranking.candidates.push_back(std::move(riJ));
    ranking.candidates.push_back(Evaluate(input,
                                          CostCandidate::kRiJk,
                                          /* wired */ false,
                                          k.riJkTensorMultiple * riJTensorCost,
                                          /* needsDevice */ false));
    ranking.candidates.push_back(Evaluate(input,
                                          CostCandidate::kQfmm,
                                          /* wired */ true,
                                          k.qfmmFixedSeconds + k.qfmmVsDirect * directCost,
                                          /* needsDevice */ false));
    ranking.candidates.push_back(Evaluate(input,
                                          CostCandidate::kGpu,
                                          /* wired */ true,
                                          gpuAnchorDirect / k.gpuSpeedup + k.gpuFixedSeconds,
                                          /* needsDevice */ true));
    ranking.candidates.push_back(Evaluate(input,
                                          CostCandidate::kGpuSplit,
                                          /* wired */ true,
                                          gpuSplitCost,
                                          /* needsDevice */ gpuSplitEngaged));

    // The total order: selectable first ascending by cost, not-selectable
    // after ascending by cost; exact ties break by the fixed candidate
    // order (the sort is stable, so the tie-break needs no extra key).
    std::stable_sort(ranking.candidates.begin(),
                     ranking.candidates.end(),
                     [](const CandidateEstimate& a, const CandidateEstimate& b) {
                         if (a.selectable != b.selectable)
                         {
                             return a.selectable;
                         }

                         if (a.costSeconds != b.costSeconds)
                         {
                             return a.costSeconds < b.costSeconds;
                         }

                         return CandidateOrderIndex(a.candidate) < CandidateOrderIndex(b.candidate);
                     });

    if (!ranking.candidates.empty() && ranking.candidates.front().selectable)
    {
        ranking.bestSelectable = BuilderKindOf(ranking.candidates.front().candidate);
    }

    // The reasoning string: names the pick and why.
    // The driver copies it into resources_resolved verbatim.
    if (input.nBasis == 0)
    {
        ranking.reasoning = "no basis functions; nothing to select";
    } else if (input.nodeCount > 1)
    {
        ranking.reasoning = std::string(kDistributedStub);
    } else if (ranking.bestSelectable.has_value())
    {
        const CandidateEstimate& pick = ranking.candidates.front();

        if (pick.candidate == CostCandidate::kGpu)
        {
            ranking.reasoning = "gpu: device present, footprint " +
                                FormatGiB(GpuFootprintGiB(input)) + " GiB fits " +
                                FormatGiB(input.deviceBudgetGiB) + " GiB free; threads " +
                                std::to_string(std::max(input.effectiveThreads, 1));
        } else
        {
            ranking.reasoning = std::string(NameOf(pick.candidate)) +
                                ": cheapest selectable path at " +
                                std::to_string(std::max(input.effectiveThreads, 1)) +
                                " threads (cap " + FormatGiB(input.cpuBudgetGiB) + " GiB; ~" +
                                FormatSeconds(pick.costSeconds) + "/iteration)";
        }
    } else
    {
        ranking.reasoning = "no candidate fits the budgets: raise memory_cap_gib or free "
                            "device memory";
    }

    return ranking;
}

double GpuProbeFloorGiB(const SelectionCostConstants& constants) noexcept {
    // The CUDA build's host footprint alone (gpuProbeCommitGiB): the
    // runtime DLLs' load-time commit, measured 2026-08-30.
    //
    // The memory model's base term was once added to this sum; with
    // the model deleted the term is gone with it. The gate is
    // therefore LOOSER by the model's former base term (0.056 GiB at
    // n = 0, where the model was flat) - a real, small change to a safety
    // gate, recorded here rather than hidden. What remains is not a
    // prediction: it is the runtime's own measured commit, the quantity
    // the probe's init actually needs under the cap.
    return constants.gpuProbeCommitGiB;
}

} // namespace qcx::driver
