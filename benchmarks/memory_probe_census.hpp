#pragma once

// The one-block allocation census: every heap allocation made while a
// measurement window is open, recorded by requested size.
//
// WHERE THE NUMBERS COME FROM. A Debug build links the debug CRT, whose
// allocation hook (_CrtSetAllocHook) sees EVERY allocation that reaches the
// CRT heap - std::vector, std::string, Eigen's aligned path (which bottoms out
// in malloc), and the module code's own operator new. That is wider than the
// repo's own qcx::memory instrument, which is call-site only by its documented
// contract and cannot see Eigen's allocator at all. A Release build has no
// such hook, so there the census falls back to replacing global operator new
// and delete, which sees every C++ allocation but NOT a library's own malloc.
// The mode is reported with the numbers so the two are never confused.

#include <cstddef>
#include <cstdint>

namespace memprobe {

/// One requested size class of the census.
struct CensusSizeClass {
    std::size_t bytes = 0; ///< The requested size.
    std::uint64_t count = 0; ///< Allocations of that size, as requests.
    std::uint64_t outstanding = 0; ///< Live blocks of that size at the snapshot.
};

/// The census's recorded totals at one snapshot.
struct CensusSnapshot {
    std::uint64_t allocations = 0; ///< Allocation requests seen in the window.
    std::uint64_t frees = 0; ///< Frees seen in the window.
    std::uint64_t reallocations = 0; ///< Reallocations seen in the window.
    std::uint64_t totalRequestBytes = 0; ///< Sum of the requested sizes.
    std::uint64_t peakLiveBytes = 0; ///< High-water live request bytes; see liveAccountingValid.
    std::uint64_t liveBytesAtEnd = 0; ///< Live request bytes when the window closed.
    /// False when any release reached the hook without its size, in which case
    /// the two live figures above are upper bounds and NOT high-waters: they
    /// can only over-count. The count and the size classes stay exact.
    bool liveAccountingValid = true;
    std::uint64_t freesWithoutSize = 0; ///< Releases that carried no size.
    /// False when this build has no allocation hook, in which case every count
    /// in this snapshot is zero because nothing was SEEN, not because nothing
    /// happened. A caller reports such a census as unavailable instead of
    /// printing those zeros as measurements.
    bool available = false;
    const char* mode = ""; ///< Which hook produced these numbers.
};

/// Opens a measurement window: the census starts recording (and its counters
/// restart). Allocations made before this call are not counted.
void CensusBegin() noexcept;

/// Closes the window and returns the totals.
/// \returns The snapshot.
CensusSnapshot CensusEnd() noexcept;

/// The number of distinct size classes recorded in the last window.
/// \returns The class count.
std::size_t CensusClassCount() noexcept;

/// One recorded size class of the last window, largest request first.
/// \param index 0-based index into the descending-order list.
/// \returns The class; a zeroed class when \p index is out of range.
CensusSizeClass CensusClassAt(std::size_t index) noexcept;

/// The peak working set of this process, in bytes (the OS-side check on the
/// request-scale accounting).
/// \returns The peak working set, or 0 when the query fails.
std::uint64_t ProcessPeakWorkingSetBytes() noexcept;

/// The current working set of this process, in bytes.
/// \returns The working set, or 0 when the query fails.
std::uint64_t ProcessWorkingSetBytes() noexcept;

} // namespace memprobe
