// The per-term allocation-attribution instrument implementation: the fixed
// preallocated stats block lives in the header (static storage, never heap);
// this file owns the enable/disable lifecycle, the tag-name table, the one
// watchdog thread, the job-object peak commit sampling (JOB_OBJECT_LIMIT_
// PROCESS_MEMORY - QueryInformationJobObject, JobObjectExtendedLimit
// Information class 9, PeakProcessMemoryUsed: the metric IDENTICAL to the
// cap metric, the same structure the benchmark harness gate and the driver's
// ApplyProcessCaps use) and the write-through trace rows. The watchdog never
// allocates: it formats each row into a fixed stack buffer and writes it
// through the C stream with a flush after every row, so the snapshots
// survive a cap kill. The one further writer is the terminal refusal record
// (AllocationInstrumentRecordRefusal): a refused run
// reaches no tagged allocation site, so without it the trace is a run's worth
// of snapshot rows with every tag column zero - indistinguishable from a run
// that allocated nothing tagged. It writes one `#refused ` line, also
// without allocating (a memory-refusal path must not need memory to report
// itself), and shares the row writers' mutex so the two can never interleave
// inside a line.

#include "qcx/memory/allocation_instrument.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

namespace qcx::memory {
namespace {

// The watchdog poll slice: the stop latency and the high-water-event row
// latency (the interval bounds only the commit sampling staleness).
constexpr std::chrono::milliseconds kWatchdogPollSliceMs{10};

// The watchdog's runtime state (never heap traffic on its own paths).
struct WatchdogState {
    std::thread thread;
    std::FILE* traceFile = nullptr;
    std::atomic<bool> stop{false};
    std::chrono::milliseconds intervalMs{250};
    std::chrono::steady_clock::time_point start{};
    AttributionTraceMetadata metadata;
    std::uint64_t sequence = 0;
};

WatchdogState gWatchdog;

// The trace writers' mutual exclusion: the watchdog thread appends snapshot
// rows while the run thread may append the terminal refusal record
// (AllocationInstrumentRecordRefusal), and an interleaved pair would corrupt
// whichever row lost the race. Taken only on the row/refusal paths - the
// watchdog's 10 ms poll slice and the hot allocator hooks never touch it, so
// the instrument's off path and its stats block stay lock-free.
std::mutex gTraceMutex;

// The bound cap job handle (HANDLE on Windows, carried as void*).
std::atomic<void*> gCapJobHandle{nullptr};

// The job-peak query override (nullptr = the bound-handle default).
std::atomic<JobPeakCommitQuery> gJobPeakQuery{nullptr};

std::uint64_t DefaultJobPeakCommit() noexcept {
#ifdef _WIN32
    const HANDLE job = static_cast<HANDLE>(gCapJobHandle.load(std::memory_order_acquire));

    if (job == nullptr)
    {
        return 0;
    }

    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};

    if (!QueryInformationJobObject(
            job, JobObjectExtendedLimitInformation, &info, sizeof(info), nullptr))
    {
        return 0;
    }

    return info.PeakProcessMemoryUsed;
#else
    return 0;
#endif
}

std::uint64_t JobPeakCommitNow() noexcept {
    const JobPeakCommitQuery overrideQuery = gJobPeakQuery.load(std::memory_order_acquire);

    if (overrideQuery != nullptr)
    {
        return overrideQuery();
    }

    return DefaultJobPeakCommit();
}

// Writes one snapshot row: seq, time, source, the job peak commit, the
// global request current/peak, and the per-tag currents/peaks/counts at that
// instant - then a write-through flush. Called only from the watchdog
// thread; stack-buffer formatted, never heap traffic.
void WriteSnapshotRow(const char* source) {
    std::lock_guard<std::mutex> guard(gTraceMutex);
    std::FILE* file = gWatchdog.traceFile;

    if (file == nullptr)
    {
        return;
    }

    const std::uint64_t timeMs =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::steady_clock::now() - gWatchdog.start)
                                       .count());

    char buffer[4096];
    char* cursor = buffer;
    const char* const end = buffer + sizeof(buffer);

    cursor += std::snprintf(cursor,
                            static_cast<std::size_t>(end - cursor),
                            "%llu,%llu,%s,%llu,%llu,%llu",
                            static_cast<unsigned long long>(gWatchdog.sequence++),
                            static_cast<unsigned long long>(timeMs),
                            source,
                            static_cast<unsigned long long>(JobPeakCommitNow()),
                            static_cast<unsigned long long>(
                                gAttributionGlobalCurrentBytes.load(std::memory_order_relaxed)),
                            static_cast<unsigned long long>(
                                gAttributionGlobalPeakBytes.load(std::memory_order_relaxed)));

    for (std::size_t tag = 0; tag < kAllocationTagCount; ++tag)
    {
        const TagAttributionCounters& counters = gAttributionStats[tag];
        cursor += std::snprintf(
            cursor,
            static_cast<std::size_t>(end - cursor),
            ",%llu,%llu,%llu",
            static_cast<unsigned long long>(counters.currentBytes.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(counters.peakBytes.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(counters.count.load(std::memory_order_relaxed)));
    }

    cursor += std::snprintf(cursor, static_cast<std::size_t>(end - cursor), "\n");
    std::fwrite(buffer, 1, static_cast<std::size_t>(cursor - buffer), file);
    std::fflush(file);
}

void WatchdogMain() {
    auto lastRow = gWatchdog.start;

    while (true)
    {
        if (gWatchdog.stop.load(std::memory_order_relaxed))
        {
            // The final row: the process's normal exit still lands a last
            // snapshot on disk (the cap-kill path relies on the per-row
            // write-through flushes instead).
            WriteSnapshotRow("final");
            return;
        }

        const bool highWater =
            gAttributionHighWaterEvent.exchange(false, std::memory_order_relaxed);
        const auto now = std::chrono::steady_clock::now();

        if (highWater || now - lastRow >= gWatchdog.intervalMs)
        {
            WriteSnapshotRow(highWater ? "highwater" : "tick");
            lastRow = now;
        }

        std::this_thread::sleep_for(kWatchdogPollSliceMs);
    }
}

void WriteMetaLine(const char* key, const std::string& value) {
    std::fprintf(gWatchdog.traceFile, "#meta %s = %s\n", key, value.c_str());
}

void WriteMetaLine(const char* key, std::uint64_t value) {
    std::fprintf(
        gWatchdog.traceFile, "#meta %s = %llu\n", key, static_cast<unsigned long long>(value));
}

void WriteTraceHeader() {
    std::fprintf(
        gWatchdog.traceFile, "# qcx allocation attribution trace v%s\n", kAttributionTraceVersion);
    const AttributionTraceMetadata& metadata = gWatchdog.metadata;
    WriteMetaLine("run_id", metadata.runId);
    WriteMetaLine("fixture", metadata.fixture);
    WriteMetaLine("basis_function_count", metadata.basisFunctionCount);
    WriteMetaLine("path_flags", metadata.pathFlags);
    WriteMetaLine("threads", metadata.threadCount);
    WriteMetaLine("allocator_version", metadata.allocatorVersion);
    WriteMetaLine("cap_bytes", metadata.memoryCapBytes);

    for (std::size_t tag = 0; tag < kAllocationTagCount; ++tag)
    {
        const std::string key = std::string("predicted_high_water_") +
                                AllocationTagName(static_cast<AllocationTag>(tag));
        WriteMetaLine(key.c_str(), metadata.predictedHighWaterBytes[tag]);
    }

    WriteMetaLine("predicted_total_request_bytes", metadata.predictedTotalRequestBytes);
    WriteMetaLine("predicted_commit_bytes", metadata.predictedCommitBytes);
    WriteMetaLine("delta_margin_bytes", metadata.deltaMarginBytes);
    WriteMetaLine("base_term_bytes", metadata.baseTermBytes);
    WriteMetaLine("base_anchor_1_bytes", metadata.baseAnchor1Bytes);
    WriteMetaLine("base_anchor_2_bytes", metadata.baseAnchor2Bytes);
    WriteMetaLine("admission", metadata.admission);
    WriteMetaLine("binary_git_hash", metadata.binaryGitHash);
    WriteMetaLine("check_label", metadata.checkLabel);
    WriteMetaLine("run_timestamp", metadata.runTimestamp);

    std::fprintf(gWatchdog.traceFile, "#columns seq,time_ms,source,job_peak_commit,");
    std::fprintf(gWatchdog.traceFile, "global_current_request,global_peak_request");

    for (std::size_t tag = 0; tag < kAllocationTagCount; ++tag)
    {
        const char* token = AllocationTagName(static_cast<AllocationTag>(tag));
        std::fprintf(gWatchdog.traceFile, ",cur_%s,peak_%s,cnt_%s", token, token, token);
    }

    std::fprintf(gWatchdog.traceFile, "\n");
    std::fflush(gWatchdog.traceFile);
}

std::FILE* OpenTraceFile(const std::string& path) {
#ifdef _WIN32
    const int wideLength = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);

    if (wideLength <= 0)
    {
        return nullptr;
    }

    std::wstring widePath(static_cast<std::size_t>(wideLength), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &widePath[0], wideLength);
    return _wfopen(widePath.c_str(), L"w");
#else
    return std::fopen(path.c_str(), "w");
#endif
}

void ZeroStatsBlock() {
    for (TagAttributionCounters& counters : gAttributionStats)
    {
        counters.currentBytes.store(0, std::memory_order_relaxed);
        counters.peakBytes.store(0, std::memory_order_relaxed);
        counters.count.store(0, std::memory_order_relaxed);
    }

    gAttributionGlobalCurrentBytes.store(0, std::memory_order_relaxed);
    gAttributionGlobalPeakBytes.store(0, std::memory_order_relaxed);
    gAttributionHighWaterEvent.store(false, std::memory_order_relaxed);
}

qcx::Error EnableError(std::string message) {
    return qcx::Error{qcx::ErrorCode::kInternalError, std::move(message)};
}

} // namespace

const char* AllocationTagName(AllocationTag tag) noexcept {
    switch (tag)
    {
    case AllocationTag::kBase:
        return "base";
    case AllocationTag::kPairStore:
        return "pair_store";
    case AllocationTag::kPattern:
        return "pattern";
    case AllocationTag::kScratch:
        return "scratch";
    case AllocationTag::kStructural:
        return "structural";
    case AllocationTag::kCache:
        return "cache";
    case AllocationTag::kExchangeF64BoundF32Live:
        return "exchange_f64_bound_f32_live";
    case AllocationTag::kClassTable:
        return "class_table";
    case AllocationTag::kScreenedQuartet:
        return "screened_quartet";
    case AllocationTag::kRijFastTensor:
        return "rij_fast_tensor";
    case AllocationTag::kRijTranspose:
        return "rij_transpose";
    case AllocationTag::kLightRungSlice:
        return "light_rung_slice";
    case AllocationTag::kLightRungValues:
        return "light_rung_values";
    case AllocationTag::kQfmmOuter:
        return "qfmm_outer";
    case AllocationTag::kBlockedMetricStrip:
        return "blocked_metric_strip";
    case AllocationTag::kFarFieldPairVector:
        return "far_field_pair_vector";
    case AllocationTag::kStartsTable:
        return "starts_table";
    case AllocationTag::kUnclassified:
        return "unclassified";
    }

    return "unclassified";
}

qcx::Result<void> AllocationInstrumentEnable(const AllocationInstrumentOptions& options) {
    if (gAttributionEnabled.load(std::memory_order_acquire))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the allocation attribution instrument is already enabled"});
    }

    ZeroStatsBlock();
    gAttributionEnabled.store(true, std::memory_order_release);

    if (options.traceFilePath.empty())
    {
        return {};
    }

    gWatchdog.intervalMs = options.snapshotIntervalMs;
    gWatchdog.metadata = options.metadata;
    gWatchdog.sequence = 0;
    gWatchdog.traceFile = OpenTraceFile(options.traceFilePath);

    if (gWatchdog.traceFile == nullptr)
    {
        gAttributionEnabled.store(false, std::memory_order_release);
        return std::unexpected(EnableError(
            "the allocation attribution trace file could not be opened: " + options.traceFilePath));
    }

    WriteTraceHeader();
    gWatchdog.start = std::chrono::steady_clock::now();
    gWatchdog.stop.store(false, std::memory_order_relaxed);

    try
    { gWatchdog.thread = std::thread(WatchdogMain); } catch (const std::system_error&)
    {
        gWatchdog.stop.store(true, std::memory_order_relaxed);
        std::fclose(gWatchdog.traceFile);
        gWatchdog.traceFile = nullptr;
        gAttributionEnabled.store(false, std::memory_order_release);
        return std::unexpected(
            EnableError("the allocation attribution watchdog thread could not start"));
    }

    return {};
}

void AllocationInstrumentRecordRefusal(std::string_view reason) noexcept {
    std::lock_guard<std::mutex> guard(gTraceMutex);
    std::FILE* file = gWatchdog.traceFile;

    if (file == nullptr)
    {
        return;
    }

    // One `#`-prefixed line, so it never parses as a snapshot row and the
    // readers' comment-skip keeps the row grammar untouched. Written a
    // character at a time: no allocation (this runs on a refusal path that
    // may itself be a memory-refusal), no fixed buffer to truncate a long
    // refusal text, and the newline flattening is exact.
    std::fputs("#refused ", file);

    for (const char character : reason)
    {
        std::fputc(character == '\n' || character == '\r' ? ' ' : character, file);
    }

    std::fputc('\n', file);
    std::fflush(file);
}

qcx::Result<void> AllocationInstrumentDisable() {
    gAttributionEnabled.store(false, std::memory_order_release);
    gWatchdog.stop.store(true, std::memory_order_relaxed);

    if (gWatchdog.thread.joinable())
    {
        gWatchdog.thread.join();
    }

    // Only after the join: the watchdog's own final row takes this lock, so
    // holding it across the join would deadlock. The window between the join
    // and this lock is closed by the recorder being the run thread's own
    // call, in front of this one.
    std::lock_guard<std::mutex> guard(gTraceMutex);

    if (gWatchdog.traceFile != nullptr)
    {
        std::fclose(gWatchdog.traceFile);
        gWatchdog.traceFile = nullptr;
    }

    return {};
}

void AllocationInstrumentBindCapJob(void* jobObjectHandle) noexcept {
    gCapJobHandle.store(jobObjectHandle, std::memory_order_release);
}

void AllocationInstrumentSetJobPeakQuery(JobPeakCommitQuery query) noexcept {
    gJobPeakQuery.store(query, std::memory_order_release);
}

} // namespace qcx::memory
