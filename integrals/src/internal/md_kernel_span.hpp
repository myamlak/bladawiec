#pragma once

// The kernel-span probe: the phase accumulators that
// subdivide the lean path's kernel span (eriWallTime - eriPrepWallTime) into
// the three phases the profiling reads. Observation only -
// no numerical path reads a value, and with the accumulator disarmed the
// kernels pay one thread-local read and one predictable branch per class
// batch and nothing else.
//
// Why a thread_local accumulator rather than a parameter: the class kernels
// are reached through the generated function-pointer table
// (md_dispatch_gen.hpp, MdClassFnF64), whose signature is one MdClassBatch -
// threading a stats pointer down would change that signature and force the
// table (+ its fp32 twin) to be regenerated and re-instantiated, and it would
// put the machinery paths' kernel signature at risk for a lean-only probe.
// The accumulator carries the observation out of band: the lean builder arms
// the CALLING thread's instance around its RunBatches call (the lean path
// runs its batches at regionThreads = 1, so the kernel body executes on
// exactly that thread) and reads it back after the join, while every other
// caller leaves it disarmed and pays nothing.
//
// The phase spans are taken at the granularity of ONE pass-1 group's one ket
// primitive pair (the VRR/ket boundary inside ComputeEriClassImpl) and of the
// whole pass-2 bra loop - never per quartet, never per primitive quadruple.
// The clock cost is the reason: steady_clock::now() measures 15 ns per read
// on the reference machine (a dedicated probe),
// against a per-quartet kernel cost of ~3.3 us, so a per-quartet instrument
// would inflate the very span it reports by percent levels.
//
// What the chosen granularity costs, MEASURED rather than assumed: on the
// 586-BF c24h50/def2-SVP leg the counters report 4,652,554 class batches,
// 104,715,933 pass-1 groups and 255,134,765 primitive passes per call, so the
// instrument takes 3 x 255,134,765 + 2 x (flushes) clock reads - about 166
// reads per batch, against a 97.9 us batch = 2.5 us of read time, i.e. an
// analytic bound of ~2.5% of the kernel span (11.5 s of CPU against the
// leg's 551 s of pooled thread-work). The census is what makes that number
// what it is: the lean path's pass-1 groups average 1.17 tasks, so the
// primitive-pass count is O(quartets) rather than the O(quartets / group)
// a multi-task grouping would give, and the ket transform is called with
// batch = 1.17 rather than batched. The bound is stated rather than removed
// because it is an even, additive offset that does not move the ranking it
// exists to decide (the measured shares are 69.4% VRR / 22.6% ket / 4.5%
// bra) - and the honest alternative, a sampled instrument, would give up the
// exact sum the decomposition is read against.

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace qcx::integrals::internal {

/// The kernel-span phase accumulators of one thread.
///
/// Every span is a SUM of sub-spans of the kernel phase (eriWallTime minus
/// eriPrepWallTime), so at k = 1 the three spans sum to at most the kernel
/// span, and above k = 1 they follow the same window-accumulated convention
/// as the rest of FockBuildStats (see fock_build.hpp): the per-window
/// partials are summed after the join, so the columns are per-thread
/// accumulated totals whose intervals overlap, never whole-call wall spans.
/// The reader of a trace row must apply that convention - it is the same one
/// that makes eri_wall_ms un-wall-comparable above k = 1.
struct KernelSpanAccum {
    /// Armed by the profiling caller; the kernels hoist it once per class
    /// batch and skip every read and every counter when it is false.
    bool enabled = false;
    /// The VRR phase: one pass-1 group's one ket primitive pair, from the top
    /// of the primitive-pair body over the pq/acc block zeroing and the
    /// RunVrrQuadruple recurrence loop of every task of that group.
    std::int64_t vrrNanos = 0;
    /// The ket-transform phase: the group's one KetTransformBatched call
    /// (the batched strided GEMM over the group's tasks), from the close of
    /// the VRR phase to its return.
    std::int64_t ketNanos = 0;
    /// The bra-transform phase: the whole pass-2 loop, from the first task's
    /// cost up to the last task's packed-block write.
    std::int64_t braNanos = 0;
    /// The VRR phase's work denominator: the RunVrrQuadruple call count
    /// (summed over every task, ket primitive pair and bra primitive pair the
    /// class batches ran). One add per (task, primitive pair).
    std::size_t vrrQuadruples = 0;
    /// The transform calls that MISSED the micro-gate shape test and so
    /// reached the linalg batched seam instead of the in-tree micro kernel
    /// (KetTransformBatched + BraTransformSingle; the bra-call total is the
    /// row's nq64 and the ket-call total this struct's primPasses, so the
    /// eligible share is derivable without a second counter).
    std::size_t gateSeamCalls = 0;
    /// The pass-1 group census: how many ket-primitive-pair groups the class
    /// batches walked.
    std::size_t groupCount = 0;
    /// The (group, ket primitive pair) iteration count - the census the
    /// instrument's own granularity sits on, and exactly the
    /// KetTransformBatched call count.
    std::size_t primPasses = 0;

    /// Zeroes every counter and arms the accumulator. Call on the thread
    /// whose kernels are to be observed, immediately before the dispatch.
    void Begin() noexcept {
        *this = KernelSpanAccum{};
        enabled = true;
    }

    /// Disarms the accumulator, leaving the counters readable. Always call
    /// it, including on the error paths, or the thread's kernels keep
    /// accumulating into a result nobody reads.
    void End() noexcept {
        enabled = false;
    }
};

/// The calling thread's kernel-span accumulator (one instance per thread
/// across the whole program - the inline function's static thread_local).
/// \returns A mutable reference to the calling thread's accumulator.
inline KernelSpanAccum& KernelSpan() noexcept {
    static thread_local KernelSpanAccum accum;

    return accum;
}

/// Whether the calling thread's kernel spans are being recorded. The kernels
/// hoist this once per class batch; it is the only cost a disarmed
/// accumulator adds to the hot path.
/// \returns True when the calling thread's accumulator is armed.
inline bool KernelSpanEnabled() noexcept {
    return KernelSpan().enabled;
}

} // namespace qcx::integrals::internal
