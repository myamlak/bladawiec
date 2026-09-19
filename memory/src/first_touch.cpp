#include "qcx/memory/first_touch.hpp"

#include "qcx/backend/cpu_backend.hpp"

#include <algorithm>
#include <cstddef>

namespace qcx::memory {

void TouchPagesAcrossTeam(void* p, std::size_t bytes) noexcept {
    if (p == nullptr || bytes == 0)
    {
        return;
    }

    const std::size_t team = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
    const std::size_t lineCount = (bytes + kCacheLineBytes - 1) / kCacheLineBytes;
    // One stripe per team thread up to one stripe per line: tiny blocks
    // collapse to a single-threaded pass (or to the line count), and the
    // integer division guarantees every stripe owns at least one line.
    const std::size_t stripeCount = std::min(team, lineCount);

    if (stripeCount <= 1)
    {
        AllocatorTraits<qcx::backend::CpuTag>::TouchPages(p, bytes);
        return;
    }

    qcx::backend::Backend<qcx::backend::CpuTag> cpu;

    cpu.ParallelFor(stripeCount, [p, bytes, lineCount, stripeCount](std::size_t stripe) {
        const std::size_t beginLine = lineCount * stripe / stripeCount;
        const std::size_t endLine = lineCount * (stripe + 1) / stripeCount;
        const std::size_t begin = beginLine * kCacheLineBytes;
        // The last stripe takes the ragged tail verbatim; every other
        // boundary is a 64-B multiple, so adjacent stripes never share a
        // cache line during the pass.
        const std::size_t end = stripe + 1 == stripeCount ? bytes : endLine * kCacheLineBytes;

        AllocatorTraits<qcx::backend::CpuTag>::TouchPages(static_cast<std::byte*>(p) + begin,
                                                          end - begin);
    });
}

} // namespace qcx::memory
