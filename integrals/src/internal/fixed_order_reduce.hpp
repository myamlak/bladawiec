#pragma once

// The fixed-order block reduction (the reproducibility fix for the direct
// builder's two block reductions).
//
// qcx::backend::ParallelReduce documents its own merge order as
// unspecified: "each thread accumulates a thread-local partial via
// accumulate, then the partials are merged under a critical section in
// unspecified thread-completion order ... combine must be associative and
// zero its identity, because the merge order across threads is
// unspecified" (qcx/backend/cpu_backend.hpp). The Fock accumulation's
// combine is the fp64 matrix sum, which is NOT associative, so that
// precondition does not hold at the call sites here: the unspecified order
// made two builds of the same density differ in the last bits.
//
// MEASURED before the fix (Release, C4H10/def2-SVP,
// 106 basis functions, the four_cell attribution fixture): the same
// builder's two consecutive BuildFock calls differed by 1.78e-15 to
// 2.66e-15 max-abs-element and a fresh builder's build of the same cell by
// 1.78e-15 to 5.33e-15, and the deviation MOVED RUN TO RUN, so the value
// the tree documented as a floor (5.3e-15) is a sample of a distribution
// rather than a bound. Every measured deviation was an exact multiple of
// 2^-49 - one ulp of the ~10 a.u. Fock elements - so the whole variation
// is the last 1-3 bits of the reduction's grouping.
//
// The fix joins the partials in BLOCK INDEX order instead of thread
// completion order. That is the module's own proven pattern, used twice
// already: the QFMM far field's "fixed-order join: partial 0 + partial 1 +
// ... in chunk order" (internal/qfmm_multipole.cpp), which makes that
// pass bit-reproducible run to run at every chunk count, and the lean
// builder's window-ordered fold. This header applies the same join to the
// direct builder's two block reductions.
//
// VALUE IDENTITY. The partials themselves are unchanged - block i is still
// accumulated over the same contiguous task range by the same lambda - so
// only the grouping of the join moves, and the join is what the primitive
// left unspecified. At n <= 1 (the serial fallback, FockBuildOptions::
// maxParallelChunks = 1) the result is bit-identical to the previous
// implementation by construction: one partial is combined once, as
// `zero + p_0`, exactly as `total = combine(total, partial)` computed it
// with total still at the identity.

#include "qcx/backend/cpu_backend.hpp"

#include <cstddef>
#include <vector>

namespace qcx::integrals::internal {

/// Reduces blocks [0, \p n) in FIXED block-index order.
///
/// \tparam T The partial type; default-constructible and copy-assignable.
/// \tparam F Callable T(const T&, std::size_t) - folds block i into a fresh
/// identity-seeded partial, exactly ParallelReduce's accumulate contract.
/// \tparam G Callable T(const T&, const T&) - combines two partials.
/// \param n Number of blocks.
/// \param zero Identity value for \p combine.
/// \param accumulate Folds block i into a partial.
/// \param combine Combines two partials.
/// \param numThreads The region's thread count, or 0 for DefaultOmpTeamSize().
/// \returns The combined total, joined in block-index order.
template <typename T, typename F, typename G>
T FixedOrderReduce(std::size_t n, const T& zero, F&& accumulate, G&& combine, int numThreads) {
    // One slot per block, not per thread: the blocks are the reduction's
    // own unit, and the slot count is bounded by the chunk count the caller
    // already chose (ChunkCountFor clamps it to the runtime's default team
    // size), so the live
    // partial mass is the same as the primitive's per-thread accumulators.
    std::vector<T> partials(n);

    qcx::backend::Backend<qcx::backend::CpuTag> backend;
    backend.ParallelFor(
        n, [&](std::size_t block) { partials[block] = accumulate(zero, block); }, numThreads);

    // The fixed-order join, on the calling thread, outside any lock: the
    // same adds the primitive performed inside its critical section, in an
    // order the schedule cannot move.
    T total = zero;

    for (std::size_t block = 0; block < n; ++block)
    {
        total = combine(total, partials[block]);
    }

    return total;
}

} // namespace qcx::integrals::internal
