// The allocation census - see memory_probe_census.hpp for what it does and
// which hook produces the numbers in which configuration.
//
// THE HOOK MUST NOT ALLOCATE. Both hooks below run inside the allocator, so
// every piece of their state is fixed static storage and the size histogram is
// a plain array with a linear probe. A std::map here would recurse.

#include "memory_probe_census.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
// psapi.h defines no types of its own and must follow windows.h: on this SDK
// GetProcessMemoryInfo maps to K32GetProcessMemoryInfo, which kernel32.lib
// already imports, so no extra link is needed.
#include <psapi.h>
#endif

#ifdef _DEBUG
#include <crtdbg.h>
#endif

namespace memprobe {
namespace {

constexpr std::size_t kMaxClasses = 512;

struct ClassSlot {
    std::size_t bytes = 0;
    std::uint64_t count = 0;
    std::uint64_t outstanding = 0;
};

/// The census state. Static storage only: the hooks may run on any thread and
/// must never touch the heap.
struct CensusState {
    std::atomic<bool> on{false};
    std::atomic<std::uint64_t> allocations{0};
    std::atomic<std::uint64_t> frees{0};
    std::atomic<std::uint64_t> reallocations{0};
    std::atomic<std::uint64_t> totalRequestBytes{0};
    std::atomic<std::uint64_t> liveBytes{0};
    std::atomic<std::uint64_t> peakLiveBytes{0};
    std::atomic<std::uint64_t> liveBytesAtEnd{0};
    std::atomic<std::size_t> classCount{0};
    std::atomic<std::uint64_t> uncountedClasses{0};
    // The CRT's free hook does not always deliver the block's size, and a
    // free whose size is unknown cannot be subtracted from the live total.
    // Counted, so the report can say the live/peak figures are unavailable
    // rather than print a number that is silently the sum of all requests.
    std::atomic<std::uint64_t> freesWithoutSize{0};
    std::array<ClassSlot, kMaxClasses> classes{};
};

CensusState gCensus;

/// Records one allocation. Called from the hook: no heap traffic, no locks
/// that can re-enter the allocator.
void RecordAllocate(std::size_t bytes) noexcept {
    gCensus.allocations.fetch_add(1, std::memory_order_relaxed);
    gCensus.totalRequestBytes.fetch_add(bytes, std::memory_order_relaxed);

    const std::uint64_t live =
        gCensus.liveBytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    std::uint64_t peak = gCensus.peakLiveBytes.load(std::memory_order_relaxed);

    while (live > peak &&
           !gCensus.peakLiveBytes.compare_exchange_weak(peak, live, std::memory_order_relaxed))
    {
    }

    const std::size_t count = gCensus.classCount.load(std::memory_order_relaxed);

    for (std::size_t i = 0; i < count; ++i)
    {
        if (gCensus.classes[i].bytes == bytes)
        {
            gCensus.classes[i].count += 1;
            gCensus.classes[i].outstanding += 1;
            return;
        }
    }

    if (count >= kMaxClasses)
    {
        gCensus.uncountedClasses.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    gCensus.classes[count].bytes = bytes;
    gCensus.classes[count].count = 1;
    gCensus.classes[count].outstanding = 1;
    gCensus.classCount.store(count + 1, std::memory_order_relaxed);
}

/// Records one release of \p bytes.
void RecordFree(std::size_t bytes) noexcept {
    gCensus.frees.fetch_add(1, std::memory_order_relaxed);

    if (bytes == 0)
    {
        // The size is unknown: the block stays in the live total. See
        // freesWithoutSize - the snapshot reports the live accounting as
        // unavailable whenever this fires.
        gCensus.freesWithoutSize.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    gCensus.liveBytes.fetch_sub(bytes, std::memory_order_relaxed);

    const std::size_t count = gCensus.classCount.load(std::memory_order_relaxed);

    for (std::size_t i = 0; i < count; ++i)
    {
        if (gCensus.classes[i].bytes == bytes)
        {
            if (gCensus.classes[i].outstanding > 0)
            {
                gCensus.classes[i].outstanding -= 1;
            }

            return;
        }
    }
}

#ifdef _DEBUG

int __cdecl CensusAllocHook(int allocType,
                            void* userData,
                            std::size_t size,
                            int /*blockType*/,
                            long /*requestNumber*/,
                            const unsigned char* /*filename*/,
                            int /*lineNumber*/) {
    (void)userData;

    if (!gCensus.on.load(std::memory_order_relaxed))
    {
        return 1;
    }

    switch (allocType)
    {
    case _HOOK_ALLOC:
        RecordAllocate(size);
        break;
    case _HOOK_FREE:
        RecordFree(size);
        break;
    case _HOOK_REALLOC:
        // The CRT hook passes only the NEW size, so the old block cannot be
        // subtracted: reallocations are counted and excluded from the live
        // accounting, and the containers this harness measures do not use it.
        gCensus.reallocations.fetch_add(1, std::memory_order_relaxed);
        break;
    default:
        break;
    }

    return 1;
}

constexpr const char* kCensusMode = "debug-CRT hook (_CrtSetAllocHook): every CRT allocation";
constexpr bool kCensusAvailable = true;

#else

// A Release build has no CRT allocation hook, so the census has no honest way
// to see a library's own malloc (Eigen's aligned path above all). Rather than
// report a partial count as if it were the whole one, it reports none and
// names itself unavailable - the caller must not read a zero as "no
// allocations".
constexpr const char* kCensusMode = "unavailable: Release build has no CRT allocation hook";
constexpr bool kCensusAvailable = false;

#endif

} // namespace

void CensusBegin() noexcept {
    gCensus.allocations.store(0, std::memory_order_relaxed);
    gCensus.frees.store(0, std::memory_order_relaxed);
    gCensus.reallocations.store(0, std::memory_order_relaxed);
    gCensus.totalRequestBytes.store(0, std::memory_order_relaxed);
    gCensus.liveBytes.store(0, std::memory_order_relaxed);
    gCensus.peakLiveBytes.store(0, std::memory_order_relaxed);
    gCensus.liveBytesAtEnd.store(0, std::memory_order_relaxed);
    gCensus.classCount.store(0, std::memory_order_relaxed);
    gCensus.uncountedClasses.store(0, std::memory_order_relaxed);
    gCensus.freesWithoutSize.store(0, std::memory_order_relaxed);

#ifdef _DEBUG
    _CrtSetAllocHook(&CensusAllocHook);
    gCensus.on.store(true, std::memory_order_relaxed);
#endif
}

CensusSnapshot CensusEnd() noexcept {
    gCensus.on.store(false, std::memory_order_relaxed);

    CensusSnapshot snapshot;
    snapshot.allocations = gCensus.allocations.load(std::memory_order_relaxed);
    snapshot.frees = gCensus.frees.load(std::memory_order_relaxed);
    snapshot.reallocations = gCensus.reallocations.load(std::memory_order_relaxed);
    snapshot.totalRequestBytes = gCensus.totalRequestBytes.load(std::memory_order_relaxed);
    snapshot.peakLiveBytes = gCensus.peakLiveBytes.load(std::memory_order_relaxed);
    snapshot.liveBytesAtEnd = gCensus.liveBytes.load(std::memory_order_relaxed);

    // A live accounting with unknown-size releases in it is NOT a high-water:
    // it can only over-count, and it reads as the sum of every request. Say so
    // in the mode string rather than let the number be read as a peak.
    snapshot.liveAccountingValid = gCensus.freesWithoutSize.load(std::memory_order_relaxed) == 0;
    snapshot.freesWithoutSize = gCensus.freesWithoutSize.load(std::memory_order_relaxed);
    snapshot.available = kCensusAvailable;
    snapshot.mode = kCensusMode;

    return snapshot;
}

std::size_t CensusClassCount() noexcept {
    return gCensus.classCount.load(std::memory_order_relaxed);
}

CensusSizeClass CensusClassAt(std::size_t index) noexcept {
    // Descending by requested size: the largest allocations are the ones a
    // layout decision turns on, and a stable order makes two runs comparable.
    const std::size_t count = gCensus.classCount.load(std::memory_order_relaxed);
    std::size_t rank = 0;

    for (std::size_t i = 0; i < count; ++i)
    {
        std::size_t better = 0;

        for (std::size_t j = 0; j < count; ++j)
        {
            if (gCensus.classes[j].bytes > gCensus.classes[i].bytes)
            {
                ++better;
            }
        }

        if (better == index)
        {
            rank = i;
            break;
        }
    }

    if (index >= count)
    {
        return CensusSizeClass{};
    }

    CensusSizeClass out;
    out.bytes = gCensus.classes[rank].bytes;
    out.count = gCensus.classes[rank].count;
    out.outstanding = gCensus.classes[rank].outstanding;

    return out;
}

std::uint64_t ProcessPeakWorkingSetBytes() noexcept {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};

    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) == 0)
    {
        return 0;
    }

    return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
#else
    return 0;
#endif
}

std::uint64_t ProcessWorkingSetBytes() noexcept {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};

    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) == 0)
    {
        return 0;
    }

    return static_cast<std::uint64_t>(counters.WorkingSetSize);
#else
    return 0;
#endif
}

} // namespace memprobe
