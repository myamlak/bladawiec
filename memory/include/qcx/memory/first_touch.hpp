#pragma once

#include "qcx/memory/allocator_traits.hpp"

#include <cstddef>

namespace qcx::memory {

/// First-touch pass over the whole block [\p p, \p p + \p bytes), striped
/// across the cached OpenMP team (one contiguous 64-B-aligned line range per
/// thread).
///
/// The allocation-site convention this pass implements: a large host block
/// whose consuming team is the OpenMP team is allocated uninitialized, run
/// through this pass, and only then filled by the (possibly serial) fill.
/// Anonymous pages are placed on the NUMA node of the thread that first
/// accesses them and never migrate, so the pages end up distributed over the
/// consuming team's nodes and later team-wide reads hit local memory. Over a
/// single-socket machine the pass is a content-preserving no-op (the
/// write-back restores every byte it reads), so it is safe unconditionally;
/// the placement effect only exists on multi-socket hardware. The team is
/// the cached size (backend::DefaultOmpTeamSize), which is the same team
/// every CPU parallel primitive in qcx uses; with no thread affinity in the
/// process the placement is best-effort, but it is strictly no worse than
/// the alternative (every page on the allocating thread's node). Call from
/// a serial context: inside an existing parallel region the inner pass
/// serializes and the placement falls back to one thread.
/// \ingroup qcx-memory
/// \param p Block start; nullptr is a no-op.
/// \param bytes Block size in bytes; 0 is a no-op.
void TouchPagesAcrossTeam(void* p, std::size_t bytes) noexcept;

} // namespace qcx::memory
