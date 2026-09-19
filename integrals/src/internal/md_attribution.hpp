#pragma once

// The per-term attribution plumbing of the MD-engine sites: the
// tagged-allocator vector alias of the model families and the per-thread
// kernel-arena accessor. The kernels' batch arenas (footprint's scratch
// term - batch x threads on the direct and 3c paths) are function-local
// thread_local vectors inside the per-class template, so their allocators
// capture the active tag at each thread's first construction, which happens
// inside an OpenMP batch loop where the worker threads carry no scope; the
// accessor pins that capture to the scratch family regardless of the
// caller's scope.

#include "qcx/memory/allocation_instrument.hpp"

#include <vector>

namespace qcx::integrals::internal {

/// The std::vector alias over the tagged allocator: the allocation
/// sites of the model families declare their containers through this alias
/// inside the family's scope, so the allocator captures the family tag at
/// the container's construction (the instrument's capture-at-construction
/// contract).
template <typename T> using TaggedVector = std::vector<T, qcx::memory::TaggedAllocator<T>>;

/// The kernel-arena slot of one ComputeEriClassImpl call: the call's
/// arenas are independent buffers with interleaved writes, so every slot
/// must name DISTINCT thread_local storage - the accessor keys the instance
/// by (element type, slot), never by element type alone, which would alias
/// the separate scratch regions of one call onto a shared buffer (the
/// batch-layout corruption the retagging once caused, fixed here).
enum class ScratchArenaSlot {
    kBatch, ///< The group-contiguous pq/acc batch buffer.
    kTaskBase, ///< The per-task pq block offsets.
    kTaskAcc, ///< The per-task acc block offsets.
    kConvert, ///< The per-transform scratch (the ket GEMM staging and the
              ///< bra pair's converted transform rows).
    kTaskBound, ///< The per-task certified-bound accumulation.
};

/// The per-thread batch arena accessor of one kernel slot (the
/// ComputeEriClassImpl thread_locals): the thread_local is constructed
/// under the scratch family's scope on every thread, so the arena's
/// allocator captures kScratch deterministically at first touch.
/// \tparam T The element type.
/// \tparam kSlot The arena's role within the call (ScratchArenaSlot) - the
/// instance is per (T, kSlot), so the five arenas of one call never share
/// storage.
template <typename T, ScratchArenaSlot kSlot> TaggedVector<T>& ThreadScratchVector() noexcept {
    qcx::memory::AllocationTagScope scope(qcx::memory::AllocationTag::kScratch);
    static thread_local TaggedVector<T> arena;

    return arena;
}

} // namespace qcx::integrals::internal
