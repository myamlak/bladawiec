// Unit tests for the per-term attribution instrument: the high-water
// math, the tag attribution, the snapshot logic, the write-through flush
// (rows are on disk BEFORE disable - the cap-kill survival property at the
// unit level), the realloc/calloc/aligned paths, the instrumented-baseline
// subtraction helpers, and the unclassified scope-hygiene assertion (NOT a
// coverage check - the contract in the instrument header).

#include "qcx/memory/allocation_instrument.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
// NOMINMAX: the windows min/max macros are not wanted in a test translation
// unit that includes <windows.h> only for the process id below.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using qcx::memory::AllocateTagged;
using qcx::memory::AllocationInstrumentBindCapJob;
using qcx::memory::AllocationInstrumentDisable;
using qcx::memory::AllocationInstrumentEnable;
using qcx::memory::AllocationInstrumentEnabled;
using qcx::memory::AllocationInstrumentOptions;
using qcx::memory::AllocationInstrumentRecordRefusal;
using qcx::memory::AllocationInstrumentSetJobPeakQuery;
using qcx::memory::AllocationTag;
using qcx::memory::AllocationTagName;
using qcx::memory::AllocationTagScope;
using qcx::memory::AttributionDeltaRefuted;
using qcx::memory::AttributionRealizedDeltaBytes;
using qcx::memory::AttributionTermRefuted;
using qcx::memory::GlobalCurrentRequestBytes;
using qcx::memory::GlobalPeakRequestBytes;
using qcx::memory::kAllocationTagCount;
using qcx::memory::kTermRefutationToleranceBytes;
using qcx::memory::ReallocateTagged;
using qcx::memory::TagAllocationCount;
using qcx::memory::TagCurrentBytes;
using qcx::memory::TaggedAlignedAllocate;
using qcx::memory::TaggedAlignedDeallocate;
using qcx::memory::TaggedAllocate;
using qcx::memory::TaggedAllocator;
using qcx::memory::TaggedCalloc;
using qcx::memory::TaggedDeallocate;
using qcx::memory::TaggedReallocate;
using qcx::memory::TagPeakBytes;

namespace {

// A session that enables the instrument for one test and always disables it
// again (the global instrument must never leak across tests).
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class InstrumentSession {
public:
    explicit InstrumentSession(const AllocationInstrumentOptions& options = {}) :
        _ok(AllocationInstrumentEnable(options).has_value()) {}

    ~InstrumentSession() {
        AllocationInstrumentSetJobPeakQuery(nullptr);
        AllocationInstrumentBindCapJob(nullptr);
        auto result = AllocationInstrumentDisable();
        EXPECT_TRUE(result.has_value());
    }

    bool Ok() const noexcept {
        return _ok;
    }

private:
    bool _ok;
};

// The trace path for one session. The process id belongs in the name: each
// gtest case is its own ctest test, hence its own process, so the counter
// below restarts at 0 in every case while ctest runs cases with -j
// parallelism. The counter alone therefore had two cases running at once open
// the SAME file - and the instrument opens the trace with "w", so the later
// session truncated the earlier session's rows.
std::string MakeTracePath() {
    static std::atomic<int> counter{0};
#ifdef _WIN32
    const unsigned long processId = static_cast<unsigned long>(GetCurrentProcessId());
#else
    const unsigned long processId = static_cast<unsigned long>(getpid());
#endif
    std::filesystem::path path = std::filesystem::temp_directory_path() /
                                 ("qcx_attribution_test_" + std::to_string(processId) + "_" +
                                  std::to_string(counter.fetch_add(1)) + ".csv");
    return path.string();
}

// The parsed columns of one snapshot row (see the trace header docs).
struct TraceRow {
    std::string source;
    std::uint64_t jobPeakCommit = 0;
    std::uint64_t globalCurrent = 0;
    std::uint64_t globalPeak = 0;
    std::vector<std::uint64_t> current;
    std::vector<std::uint64_t> peak;
    std::vector<std::uint64_t> count;
};

// Parses the complete rows of the trace file (comment lines skipped; the
// trailing partial row is ignored - the watchdog flushes per row, so a read
// can catch a row mid-write).
std::vector<TraceRow> ReadTraceRows(const std::string& path) {
    std::vector<TraceRow> rows;
    std::ifstream file(path);

    if (!file)
    {
        return rows;
    }

    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::vector<std::string> fields;
        std::size_t start = 0;

        while (start <= line.size())
        {
            const std::size_t comma = line.find(',', start);

            if (comma == std::string::npos)
            {
                fields.push_back(line.substr(start));
                break;
            }

            fields.push_back(line.substr(start, comma - start));
            start = comma + 1;
        }

        if (fields.size() != 6 + 3 * kAllocationTagCount)
        {
            continue;
        }

        TraceRow row;
        row.source = fields[2];
        row.jobPeakCommit = std::stoull(fields[3]);
        row.globalCurrent = std::stoull(fields[4]);
        row.globalPeak = std::stoull(fields[5]);
        row.current.resize(kAllocationTagCount);
        row.peak.resize(kAllocationTagCount);
        row.count.resize(kAllocationTagCount);

        for (std::size_t tag = 0; tag < kAllocationTagCount; ++tag)
        {
            row.current[tag] = std::stoull(fields[6 + 3 * tag]);
            row.peak[tag] = std::stoull(fields[7 + 3 * tag]);
            row.count[tag] = std::stoull(fields[8 + 3 * tag]);
        }

        rows.push_back(std::move(row));
    }

    return rows;
}

// The raw trace text (every line, comments included) - the reader for the
// non-row records, which ReadTraceRows drops by construction.
std::string ReadTraceText(const std::string& path) {
    std::ifstream file(path);

    if (!file)
    {
        return {};
    }

    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

// Polls the trace file until it holds at least \p minimum rows or the
// timeout elapses (the watchdog rows arrive on its interval or on the global
// request high-water events - both asynchronous to the test).
bool WaitForRows(const std::string& path, std::size_t minimum, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;

    while (std::chrono::steady_clock::now() < deadline)
    {
        if (ReadTraceRows(path).size() >= minimum)
        {
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return ReadTraceRows(path).size() >= minimum;
}

// The first row whose per-tag current of \p tag equals \p bytes, or nullptr.
const TraceRow* FindRowWithTagCurrent(const std::vector<TraceRow>& rows,
                                      AllocationTag tag,
                                      std::uint64_t bytes) {
    for (const TraceRow& row : rows)
    {
        if (row.current[static_cast<std::size_t>(tag)] == bytes)
        {
            return &row;
        }
    }

    return nullptr;
}

constexpr std::size_t TagIndex(AllocationTag tag) {
    return static_cast<std::size_t>(tag);
}

// A scripted job-peak source for the snapshot tests (the default reads the
// bound cap job on Windows; the tests never create a real job).
std::atomic<std::uint64_t> gScriptedJobPeak{0};

std::uint64_t ScriptedJobPeak() noexcept {
    return gScriptedJobPeak.load(std::memory_order_relaxed);
}

} // namespace

TEST(AllocationInstrumentTest, OffByDefaultAndPassThrough) {
    EXPECT_FALSE(AllocationInstrumentEnabled());
    EXPECT_EQ(GlobalCurrentRequestBytes(), 0u);
    EXPECT_EQ(GlobalPeakRequestBytes(), 0u);

    // With the instrument off the wrappers are plain allocator calls: the
    // block is usable and nothing is counted. The analyzer cannot see the
    // instrumented deallocation (the wrappers' operator delete is in another
    // TU), so each raw-allocation check in this file reads as a leak on its
    // ASSERT_NE's failure path - the per-site NOLINTs suppress that false
    // positive only.
    AllocationTagScope scope(AllocationTag::kPairStore);
    void* p = TaggedAllocate(4096);
    ASSERT_NE(p, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
    static_cast<volatile char*>(p)[0] = 1;
    EXPECT_EQ(TagCurrentBytes(AllocationTag::kPairStore), 0u);
    EXPECT_EQ(TagAllocationCount(AllocationTag::kPairStore), 0u);
    TaggedDeallocate(p, 4096);
}

TEST(AllocationInstrumentTest, HighWaterMath) {
    InstrumentSession session;
    ASSERT_TRUE(session.Ok());

    {
        AllocationTagScope scope(AllocationTag::kPairStore);
        void* a = TaggedAllocate(100);
        ASSERT_NE(a, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        void* b = TaggedAllocate(300);
        ASSERT_NE(b, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        void* c = TaggedAllocate(200);
        ASSERT_NE(c, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kPairStore), 600u);
        EXPECT_EQ(TagPeakBytes(AllocationTag::kPairStore), 600u);
        EXPECT_EQ(TagAllocationCount(AllocationTag::kPairStore), 3u);
        EXPECT_EQ(GlobalCurrentRequestBytes(), 600u);
        EXPECT_EQ(GlobalPeakRequestBytes(), 600u);
        TaggedDeallocate(b, 300);
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kPairStore), 300u);
        EXPECT_EQ(TagPeakBytes(AllocationTag::kPairStore), 600u);
        TaggedDeallocate(a, 100);
        TaggedDeallocate(c, 200);
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kPairStore), 0u);
    }

    // The peak is a high-water: it never comes back down.
    EXPECT_EQ(TagPeakBytes(AllocationTag::kPairStore), 600u);
}

TEST(AllocationInstrumentTest, TagAttributionAndNestedScopes) {
    InstrumentSession session;
    ASSERT_TRUE(session.Ok());

    {
        AllocationTagScope outer(AllocationTag::kPairStore);
        void* pair = TaggedAllocate(1000);
        ASSERT_NE(pair, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kPairStore), 1000u);
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kScratch), 0u);

        {
            AllocationTagScope inner(AllocationTag::kScratch);
            void* scratch = TaggedAllocate(500);
            ASSERT_NE(scratch, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
            EXPECT_EQ(TagCurrentBytes(AllocationTag::kScratch), 500u);
            // The outer tag's traffic is untouched by the nested scope.
            EXPECT_EQ(TagCurrentBytes(AllocationTag::kPairStore), 1000u);
            TaggedDeallocate(scratch, 500);
        }

        // The inner scope exited: the thread's tag is the outer one again.
        void* pair2 = TaggedAllocate(2000);
        ASSERT_NE(pair2, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kPairStore), 3000u);
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kScratch), 0u);
        TaggedDeallocate(pair2, 2000);
        TaggedDeallocate(pair, 1000);
    }

    EXPECT_EQ(TagPeakBytes(AllocationTag::kPairStore), 3000u);
    EXPECT_EQ(TagPeakBytes(AllocationTag::kScratch), 500u);
}

TEST(AllocationInstrumentTest, UnclassifiedLeakBucket) {
    InstrumentSession session;
    ASSERT_TRUE(session.Ok());

    // No scope active: the traffic lands in unclassified.
    void* p = TaggedAllocate(700);
    ASSERT_NE(p, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_EQ(TagCurrentBytes(AllocationTag::kUnclassified), 700u);
    EXPECT_EQ(TagAllocationCount(AllocationTag::kUnclassified), 1u);
    TaggedDeallocate(p, 700);
    EXPECT_EQ(TagCurrentBytes(AllocationTag::kUnclassified), 0u);
}

TEST(AllocationInstrumentTest, ScopedTrafficLeavesUnclassifiedEmpty) {
    // What this pins: SCOPE HYGIENE, and only that - between this test's
    // enable and disable every allocation it makes rides a tag scope, so no
    // wrapper traffic without a scope reaches the instrument in that window.
    // It is NOT a coverage check, and an empty bucket here is not evidence
    // that the window's memory was accounted for: this assertion counts only
    // what reached a WRAPPER, so it reads zero however large the header's
    // blind spots are (memory that never reached a wrapper is invisible to
    // it either way).
    auto result = AllocationInstrumentEnable(AllocationInstrumentOptions{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(TagAllocationCount(AllocationTag::kUnclassified), 0u);

    {
        AllocationTagScope scope(AllocationTag::kStructural);
        void* p = TaggedAllocate(4096);
        ASSERT_NE(p, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        TaggedDeallocate(p, 4096);
    }

    result = AllocationInstrumentDisable();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(TagAllocationCount(AllocationTag::kUnclassified), 0u);
    EXPECT_EQ(TagPeakBytes(AllocationTag::kUnclassified), 0u);
}

TEST(AllocationInstrumentTest, SumOfPeaksNeverEqualsThePeak) {
    InstrumentSession session;
    ASSERT_TRUE(session.Ok());

    // The reconciliation trap regression: the per-tag peaks are not
    // simultaneous, so the global peak is the peak of the SUM over time,
    // never the sum of the per-tag peaks.
    {
        AllocationTagScope scopeA(AllocationTag::kPairStore);
        void* a = TaggedAllocate(1000);
        ASSERT_NE(a, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        TaggedDeallocate(a, 1000);
    }

    {
        AllocationTagScope scopeB(AllocationTag::kScratch);
        void* b = TaggedAllocate(1000);
        ASSERT_NE(b, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        TaggedDeallocate(b, 1000);
    }

    EXPECT_EQ(TagPeakBytes(AllocationTag::kPairStore), 1000u);
    EXPECT_EQ(TagPeakBytes(AllocationTag::kScratch), 1000u);
    // The NEVER-add rule: the sum is 2000, the global peak is 1000.
    EXPECT_EQ(TagPeakBytes(AllocationTag::kPairStore) + TagPeakBytes(AllocationTag::kScratch),
              2000u);
    EXPECT_EQ(GlobalPeakRequestBytes(), 1000u);
}

TEST(AllocationInstrumentTest, ReallocPathCoverage) {
    InstrumentSession session;
    ASSERT_TRUE(session.Ok());

    {
        AllocationTagScope scope(AllocationTag::kRijFastTensor);
        // Realloc from null allocates.
        void* p = TaggedReallocate(nullptr, 0, 100);
        ASSERT_NE(p, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        static_cast<char*>(p)[0] = 0x5a;
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kRijFastTensor), 100u);
        EXPECT_EQ(TagAllocationCount(AllocationTag::kRijFastTensor), 1u);

        // Grow: the copy preserves the payload; the stats move by the delta,
        // and the transient of both live blocks mirrors the real copy moment
        // (the peak sees 400 while the old block is still live).
        void* grown = TaggedReallocate(p, 100, 300);
        ASSERT_NE(grown, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        EXPECT_EQ(static_cast<char*>(grown)[0], 0x5a);
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kRijFastTensor), 300u);
        EXPECT_EQ(TagPeakBytes(AllocationTag::kRijFastTensor), 400u);
        EXPECT_EQ(TagAllocationCount(AllocationTag::kRijFastTensor), 2u);

        // Shrink.
        void* shrunk = TaggedReallocate(grown, 300, 200);
        ASSERT_NE(shrunk, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kRijFastTensor), 200u);

        // Realloc to zero frees.
        void* freed = TaggedReallocate(shrunk, 200, 0);
        EXPECT_EQ(freed, nullptr);
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kRijFastTensor), 0u);
    }
}

TEST(AllocationInstrumentTest, CallocPathCoverage) {
    InstrumentSession session;
    ASSERT_TRUE(session.Ok());

    {
        AllocationTagScope scope(AllocationTag::kClassTable);
        void* p = TaggedCalloc(64, 32);
        ASSERT_NE(p, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        const auto* bytes = static_cast<const unsigned char*>(p);

        for (int i = 0; i < 64 * 32; ++i)
        {
            EXPECT_EQ(bytes[i], 0u);
        }

        EXPECT_EQ(TagCurrentBytes(AllocationTag::kClassTable), 2048u);
        EXPECT_EQ(TagAllocationCount(AllocationTag::kClassTable), 1u);
        TaggedDeallocate(p, 2048);
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kClassTable), 0u);
    }
}

TEST(AllocationInstrumentTest, AlignedPathCoverage) {
    InstrumentSession session;
    ASSERT_TRUE(session.Ok());

    {
        AllocationTagScope scope(AllocationTag::kScreenedQuartet);
        constexpr std::size_t alignment = 4096;
        void* p = TaggedAlignedAllocate(10000, alignment);
        ASSERT_NE(p, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % alignment, 0u);
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kScreenedQuartet), 10000u);
        TaggedAlignedDeallocate(p, 10000, alignment);
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kScreenedQuartet), 0u);
    }
}

TEST(AllocationInstrumentTest, TaggedAllocatorVectorCoverage) {
    InstrumentSession session;
    ASSERT_TRUE(session.Ok());

    // The vector object itself is plain-new (invisible to the instrument);
    // only its buffer goes through the tagged allocator. It is heap-held so
    // the buffer can be freed after the scope closes: the allocator captured
    // the kCache tag at the vector's construction (the call site), so the
    // out-of-scope deallocation still lands on the same tag.
    std::vector<std::uint64_t, TaggedAllocator<std::uint64_t>>* values = nullptr;

    {
        AllocationTagScope scope(AllocationTag::kCache);
        values = new std::vector<std::uint64_t, TaggedAllocator<std::uint64_t>>();

        for (int i = 0; i < 1000; ++i)
        {
            values->push_back(static_cast<std::uint64_t>(i));
        }

        EXPECT_EQ(values->size(), 1000u);
        // The vector's live capacity is counted exactly at the request scale
        // (the growth allocations happened inside the scope under the tag).
        // The MSVC Debug STL requests two extra bookkeeping elements from the
        // allocator for the iterator-tracking machinery (measured: the charge
        // reads 8544 B against capacity() * 8 = 8528 B on 14.51), so the
        // exact-charge pin holds in Release only; Debug pins the request scale
        // including the extras.
#ifdef _DEBUG
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kCache), (values->capacity() + 2) * 8u);
#else
        EXPECT_EQ(TagCurrentBytes(AllocationTag::kCache), values->capacity() * 8u);
#endif
        EXPECT_EQ(TagAllocationCount(AllocationTag::kCache) >= 1, true);
    }

    // Outside any scope now: the captured tag still routes the destruction.
    delete values;
    EXPECT_EQ(TagCurrentBytes(AllocationTag::kCache), 0u);
}

TEST(AllocationInstrumentTest, SnapshotRowCarriesJobPeakAndPerTagCurrents) {
    const std::string path = MakeTracePath();
    ASSERT_FALSE(path.empty());
    AllocationInstrumentSetJobPeakQuery(ScriptedJobPeak);
    gScriptedJobPeak.store(123456789u);

    AllocationInstrumentOptions options;
    options.traceFilePath = path;
    options.snapshotIntervalMs = std::chrono::milliseconds(40);
    InstrumentSession session(options);
    ASSERT_TRUE(session.Ok());

    {
        AllocationTagScope scope(AllocationTag::kPairStore);
        void* p = TaggedAllocate(1000);
        ASSERT_NE(p, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        EXPECT_TRUE(WaitForRows(path, 1, std::chrono::seconds(5)));
        TaggedDeallocate(p, 1000);
    }

    const std::vector<TraceRow> rows = ReadTraceRows(path);
    const TraceRow* live = FindRowWithTagCurrent(rows, AllocationTag::kPairStore, 1000);
    ASSERT_NE(live, nullptr);
    EXPECT_EQ(live->jobPeakCommit, 123456789u);
    EXPECT_EQ(live->peak[TagIndex(AllocationTag::kPairStore)], 1000u);
    EXPECT_EQ(live->count[TagIndex(AllocationTag::kPairStore)], 1u);
    EXPECT_EQ(live->globalCurrent, 1000u);
    EXPECT_EQ(live->globalPeak, 1000u);
}

TEST(AllocationInstrumentTest, WriteThroughRowsSurviveBeforeDisable) {
    // The cap-kill survival property at the unit level: the snapshot rows
    // are flushed write-through by the watchdog, so a row is readable from
    // the file WHILE the instrument is still enabled - the process could be
    // killed by the cap at any moment and the record so far is already on
    // disk. No disable is needed to land it.
    const std::string path = MakeTracePath();
    ASSERT_FALSE(path.empty());
    AllocationInstrumentSetJobPeakQuery(ScriptedJobPeak);

    AllocationInstrumentOptions options;
    options.traceFilePath = path;
    options.snapshotIntervalMs = std::chrono::milliseconds(40);
    auto result = AllocationInstrumentEnable(options);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(AllocationInstrumentEnabled());

    {
        AllocationTagScope scope(AllocationTag::kBase);
        void* p = TaggedAllocate(4096);
        ASSERT_NE(p, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        EXPECT_TRUE(WaitForRows(path, 1, std::chrono::seconds(5)));
        TaggedDeallocate(p, 4096);
    }

    // Still enabled here; the row must already be on disk.
    const std::vector<TraceRow> rows = ReadTraceRows(path);
    EXPECT_NE(FindRowWithTagCurrent(rows, AllocationTag::kBase, 4096), nullptr);

    result = AllocationInstrumentDisable();
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(AllocationInstrumentEnabled());
}

TEST(AllocationInstrumentTest, HighWaterEventFiresAnImmediateRow) {
    // A new global request high-water fires a snapshot row immediately (the
    // watchdog polls the event flag at its slice rate) - it does not wait
    // for the next interval.
    const std::string path = MakeTracePath();
    ASSERT_FALSE(path.empty());
    AllocationInstrumentSetJobPeakQuery(ScriptedJobPeak);

    AllocationInstrumentOptions options;
    options.traceFilePath = path;
    options.snapshotIntervalMs = std::chrono::hours(1);
    InstrumentSession session(options);
    ASSERT_TRUE(session.Ok());

    {
        AllocationTagScope scope(AllocationTag::kScratch);
        void* p = TaggedAllocate(8192);
        ASSERT_NE(p, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        EXPECT_TRUE(WaitForRows(path, 1, std::chrono::seconds(5)));

        const std::vector<TraceRow> rows = ReadTraceRows(path);
        bool sawHighWater = false;

        for (const TraceRow& row : rows)
        {
            if (row.source == "highwater")
            {
                sawHighWater = true;
                EXPECT_EQ(row.globalPeak, 8192u);
                EXPECT_EQ(row.current[TagIndex(AllocationTag::kScratch)], 8192u);
            }
        }

        EXPECT_TRUE(sawHighWater);
        TaggedDeallocate(p, 8192);
    }
}

TEST(AllocationInstrumentTest, DisableWritesFinalRowAndStopsTheWatchdog) {
    const std::string path = MakeTracePath();
    ASSERT_FALSE(path.empty());
    AllocationInstrumentSetJobPeakQuery(ScriptedJobPeak);

    AllocationInstrumentOptions options;
    options.traceFilePath = path;
    options.snapshotIntervalMs = std::chrono::milliseconds(20);

    // The session lives in an inner scope: its destructor disables the
    // instrument, which must land one last "final" row before the watchdog
    // stops and the file closes.
    std::size_t rowsAfterDisable = 0;

    {
        InstrumentSession session(options);
        ASSERT_TRUE(session.Ok());
        ASSERT_TRUE(WaitForRows(path, 1, std::chrono::seconds(5)));
        const std::size_t rowsWhileEnabled = ReadTraceRows(path).size();
        ASSERT_GE(rowsWhileEnabled, 1u);
        rowsAfterDisable = rowsWhileEnabled + 1;
    }

    EXPECT_GE(ReadTraceRows(path).size(), rowsAfterDisable);

    const std::vector<TraceRow> rows = ReadTraceRows(path);
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows.back().source, "final");

    // No further rows are written after the disable (the destructor stopped
    // the watchdog and closed the file): the file stops growing.
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    EXPECT_EQ(ReadTraceRows(path).size(), rows.size());
}

TEST(AllocationInstrumentTest, RefusalRecordNamesTheRefusalAndKeepsTheRowGrammar) {
    // A run that reaches no tagged
    // allocation site records a refusal instead of ending as a zeroed trace
    // that reads like a measurement. The record is one `#refused ` line, so
    // the snapshot-row grammar is untouched - the rows before it still parse.
    const std::string path = MakeTracePath();
    ASSERT_FALSE(path.empty());
    AllocationInstrumentSetJobPeakQuery(ScriptedJobPeak);

    AllocationInstrumentOptions options;
    options.traceFilePath = path;
    options.snapshotIntervalMs = std::chrono::milliseconds(40);
    auto result = AllocationInstrumentEnable(options);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(WaitForRows(path, 1, std::chrono::seconds(5)));

    // The reason is the driver's own refusal text, newlines and all: the
    // record must stay ONE line whatever the message contains.
    AllocationInstrumentRecordRefusal("the ri_j_link builder needs a workspace budget\nof at "
                                      "least the modeled base term (26.4 GiB)\rbut the cap is 16");
    result = AllocationInstrumentDisable();
    ASSERT_TRUE(result.has_value());

    const std::string text = ReadTraceText(path);
    ASSERT_FALSE(text.empty());
    const std::string marker = "\n#refused ";
    const std::size_t at = text.find(marker);
    ASSERT_NE(at, std::string::npos) << "trace tail: " << text.substr(text.size() - 200);
    const std::size_t lineEnd = text.find('\n', at + 1);
    ASSERT_NE(lineEnd, std::string::npos);
    std::string recorded = text.substr(at + marker.size(), lineEnd - at - marker.size());

    // The trace file is opened in text mode (CRLF on disk), so the line's
    // terminator is the '\n' the search found plus a preceding '\r'.
    while (!recorded.empty() && (recorded.back() == '\r' || recorded.back() == '\n'))
    {
        recorded.pop_back();
    }

    EXPECT_EQ(recorded,
              "the ri_j_link builder needs a workspace budget of at least the modeled base term "
              "(26.4 GiB) but the cap is 16");

    // The row grammar survived: the rows written before the refusal still
    // parse, and the refusal line is not one of them.
    const std::vector<TraceRow> rows = ReadTraceRows(path);
    EXPECT_GE(rows.size(), 1u);
    EXPECT_EQ(rows.back().source, "final");
}

TEST(AllocationInstrumentTest, RefusalRecordIsASilentNoOpWithNoTraceFile) {
    // The recorder must not fail the refusal path it reports on: with no
    // open trace file (stats-only mode, or a window already closed) the call
    // is a no-op. The check that matters is that it leaves nothing behind -
    // a later window must open clean, with no refusal line inherited from
    // the no-op calls.
    AllocationInstrumentRecordRefusal("recorded with no instrument enabled at all");
    EXPECT_FALSE(AllocationInstrumentEnabled());

    AllocationInstrumentOptions statsOnly;
    auto result = AllocationInstrumentEnable(statsOnly);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(statsOnly.traceFilePath.empty());
    AllocationInstrumentRecordRefusal("recorded in stats-only mode");
    result = AllocationInstrumentDisable();
    ASSERT_TRUE(result.has_value());

    const std::string path = MakeTracePath();
    ASSERT_FALSE(path.empty());
    AllocationInstrumentSetJobPeakQuery(ScriptedJobPeak);
    AllocationInstrumentOptions options;
    options.traceFilePath = path;
    options.snapshotIntervalMs = std::chrono::milliseconds(40);
    result = AllocationInstrumentEnable(options);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(WaitForRows(path, 1, std::chrono::seconds(5)));
    result = AllocationInstrumentDisable();
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(ReadTraceText(path).find("#refused"), std::string::npos);
    EXPECT_GE(ReadTraceRows(path).size(), 1u);
}

TEST(AllocationInstrumentTest, ReenableResetsAndDisableIsIdempotent) {
    auto result = AllocationInstrumentDisable();
    ASSERT_TRUE(result.has_value());

    {
        InstrumentSession session;
        ASSERT_TRUE(session.Ok());
        AllocationTagScope scope(AllocationTag::kBase);
        void* p = TaggedAllocate(500);
        ASSERT_NE(p, nullptr); // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
        EXPECT_EQ(TagPeakBytes(AllocationTag::kBase), 500u);
        TaggedDeallocate(p, 500);
    }

    // The next enable zeroes the stats block: the earlier peak is gone.
    InstrumentSession session;
    ASSERT_TRUE(session.Ok());
    EXPECT_EQ(TagPeakBytes(AllocationTag::kBase), 0u);
    EXPECT_EQ(GlobalPeakRequestBytes(), 0u);
}

TEST(AllocationInstrumentTest, EnableTwiceFailsCleanly) {
    InstrumentSession session;
    ASSERT_TRUE(session.Ok());
    auto result = AllocationInstrumentEnable(AllocationInstrumentOptions{});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(AllocationInstrumentTest, AttributionAcrossThreads) {
    InstrumentSession session;
    ASSERT_TRUE(session.Ok());
    constexpr int kThreadCount = 4;
    constexpr std::uint64_t kBlocksPerThread = 8;
    constexpr std::uint64_t kBlockBytes = 1024;
    std::atomic<int> arrived{0};
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(kThreadCount));

    for (int t = 0; t < kThreadCount; ++t)
    {
        threads.emplace_back([&arrived] {
            AllocationTagScope scope(AllocationTag::kScratch);
            std::vector<void*> blocks;
            blocks.reserve(static_cast<std::size_t>(kBlocksPerThread));

            for (std::uint64_t i = 0; i < kBlocksPerThread; ++i)
            {
                blocks.push_back(TaggedAllocate(kBlockBytes));
            }

            // All threads hold their blocks before any frees: the tag's peak
            // is then deterministic (the concurrent-live sum).
            arrived.fetch_add(1, std::memory_order_release);

            while (arrived.load(std::memory_order_acquire) < kThreadCount)
            {
                std::this_thread::yield();
            }

            for (void* p : blocks)
            {
                TaggedDeallocate(p, kBlockBytes);
            }
        });
    }

    for (std::thread& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(TagAllocationCount(AllocationTag::kScratch), kThreadCount * kBlocksPerThread);
    EXPECT_EQ(TagCurrentBytes(AllocationTag::kScratch), 0u);
    EXPECT_EQ(TagPeakBytes(AllocationTag::kScratch), kThreadCount * kBlocksPerThread * kBlockBytes);
}

TEST(AllocationInstrumentTest, TagNamesAreTheSpecTokens) {
    EXPECT_STREQ(AllocationTagName(AllocationTag::kBase), "base");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kPairStore), "pair_store");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kPattern), "pattern");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kScratch), "scratch");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kStructural), "structural");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kCache), "cache");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kExchangeF64BoundF32Live),
                 "exchange_f64_bound_f32_live");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kClassTable), "class_table");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kScreenedQuartet), "screened_quartet");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kRijFastTensor), "rij_fast_tensor");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kRijTranspose), "rij_transpose");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kLightRungSlice), "light_rung_slice");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kLightRungValues), "light_rung_values");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kQfmmOuter), "qfmm_outer");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kBlockedMetricStrip), "blocked_metric_strip");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kFarFieldPairVector), "far_field_pair_vector");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kStartsTable), "starts_table");
    EXPECT_STREQ(AllocationTagName(AllocationTag::kUnclassified), "unclassified");
}

TEST(AllocationInstrumentTest, TermRefutationBoundaryUsesTheNamedTolerance) {
    // measured > charged + tolerance = refuted; at or below the boundary the
    // term stands.
    constexpr std::uint64_t charged = 1000;

    EXPECT_FALSE(AttributionTermRefuted(charged, charged));
    EXPECT_FALSE(AttributionTermRefuted(charged + kTermRefutationToleranceBytes, charged));
    EXPECT_TRUE(AttributionTermRefuted(charged + kTermRefutationToleranceBytes + 1, charged));
}

TEST(AllocationInstrumentTest, BaselineSubtractionHelpers) {
    // The realized delta = the contract's RESIDUAL = job peak commit minus
    // the instrumented global peak request minus the measured instrumentation
    // overhead (saturating), and the aggregate falsification: realized > the
    // named margin + tolerance. The residual is a whole-run quantity, never a
    // family's share - see the instrument header.
    constexpr std::uint64_t cap = 16ull * 1024 * 1024 * 1024;
    constexpr std::uint64_t requestPeak = 10ull * 1024 * 1024 * 1024;
    constexpr std::uint64_t overhead = 128ull * 1024 * 1024;
    constexpr std::uint64_t margin = 256ull * 1024 * 1024;

    EXPECT_EQ(AttributionRealizedDeltaBytes(cap, requestPeak, overhead),
              cap - requestPeak - overhead);
    // The subtraction saturates at zero (an over-counted overhead cannot
    // wrap).
    EXPECT_EQ(AttributionRealizedDeltaBytes(requestPeak, cap, overhead), 0u);
    EXPECT_EQ(AttributionRealizedDeltaBytes(cap, requestPeak, 0u), cap - requestPeak);

    // The margin boundary: realized == margin + tolerance stands; one byte
    // more is refuted.
    const std::uint64_t realizedAtBoundary = margin + kTermRefutationToleranceBytes;
    EXPECT_FALSE(AttributionDeltaRefuted(
        requestPeak + overhead + realizedAtBoundary, requestPeak, overhead, margin));
    EXPECT_TRUE(AttributionDeltaRefuted(
        requestPeak + overhead + realizedAtBoundary + 1, requestPeak, overhead, margin));
}
