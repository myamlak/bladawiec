// The allocation-delta probe: a no-chemistry synthetic allocation stream over the model's
// size classes plus a thread-stack probe plus a no-workload base run, all
// under the per-term attribution instrument. The probe's job is to measure,
// on the run's own machine, the isolated components behind each real run's
// realized aggregate delta:
//
//     realizedDelta = job peak commit - instrumented global request peak
//                     - the measured instrumentation overhead
//
// with the components isolated as (each leg's job peak commit, read back from
// the cap job - the metric identical to the cap metric):
//
//     plain      - the uninstrumented base (libs, static data):        B0
//     base       - the instrumented base (+ stats block + watchdog):   P_base
//     threads    - base + thread stacks committed:                     P_threads
//     synthetic  - threads + the size-class stream (all live, page     P_synth
//                  touched, held at the peak)
//
//     baseContribution      = P_base - B0            (the instrumented
//                                                      baseline, measured)
//     stackContribution     = P_threads - P_base
//     heapCommitContribution = P_synth - P_threads
//     realizedDelta(synthetic leg) = heapCommitContribution
//                                    - the leg's global request peak
//                                    (the rounding/metadata curve over the
//                                    model's size classes)
//
// The trace rows carry the per-instant job peak + global request + per-tag
// currents; the authoritative job peak is the cap job's PeakProcessMemoryUsed
// read back after exit. Nothing here is a timed performance pass: the stream
// is deterministic and the numbers are request/commit bytes only.
//
// The size-class table is a documented stand-in at the 5000-BF scale of the
// admission runs (the per-family request-scale tables of the model; the
// runner may replace the magnitudes once the ladder calibration lands -
// the probe's purpose is the request-to-commit curve, which depends on the
// allocator behavior over these sizes, not on chemistry).
//
// Plain main() with legs (the preset_sweep precedent):
//
//   qcx-bench-delta-probe <leg> [--trace PATH] [--threads N]
//       [--hold-ms N] [--run-id ID] [--cap-bytes N] [--margin-bytes N]
//
// Legs: plain, base, threads, synthetic, selftest. The self-test executes a
// small scripted stream and verifies the accounting against the stats block
// (global and per-tag peaks/counts) and the named tolerance constant, then
// exits - runnable without any job object.

#include "qcx/memory/allocation_instrument.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using qcx::memory::AllocateTagged;
using qcx::memory::AllocationInstrumentBindCapJob;
using qcx::memory::AllocationInstrumentDisable;
using qcx::memory::AllocationInstrumentEnable;
using qcx::memory::AllocationInstrumentOptions;
using qcx::memory::AllocationTag;
using qcx::memory::AllocationTagName;
using qcx::memory::AttributionTraceMetadata;
using qcx::memory::DeallocateTagged;
using qcx::memory::GlobalCurrentRequestBytes;
using qcx::memory::GlobalPeakRequestBytes;
using qcx::memory::TagAllocationCount;
using qcx::memory::TagCurrentBytes;
using qcx::memory::TagPeakBytes;

namespace {

constexpr std::size_t kKiB = 1024u;
constexpr std::size_t kMiB = 1024u * kKiB;
constexpr std::size_t kPageSize = 4096u;

struct ScheduleRow {
    AllocationTag tag;
    std::size_t blockBytes;
    int blockCount;
};

// One block-size row per modeled family, at the 5000-BF request scale. The
// block shapes are the families' real granularities (many small pair blocks,
// few large scratch/RI blocks); the magnitudes are stand-ins until the ladder
// calibration lands. Grand total ~5.06 GiB - a request-scale stand-in for the
// full-path peak, well under the 16 GiB cap when the leg runs under it.
const std::array<ScheduleRow, 17> kSyntheticSchedule{{
    {AllocationTag::kBase, 4 * kKiB, 64},
    {AllocationTag::kPairStore, 1 * kMiB, 3072},
    {AllocationTag::kPattern, 4 * kMiB, 32},
    {AllocationTag::kScratch, 64 * kMiB, 12},
    {AllocationTag::kStructural, 4 * kMiB, 4},
    {AllocationTag::kCache, 8 * kMiB, 8},
    {AllocationTag::kExchangeF64BoundF32Live, 32 * kMiB, 16},
    {AllocationTag::kClassTable, 1 * kMiB, 8},
    {AllocationTag::kScreenedQuartet, 4 * kKiB, 4096},
    {AllocationTag::kRijFastTensor, 64 * kMiB, 2},
    {AllocationTag::kRijTranspose, 64 * kMiB, 2},
    {AllocationTag::kLightRungSlice, 1 * kMiB, 16},
    {AllocationTag::kLightRungValues, 1 * kMiB, 16},
    {AllocationTag::kQfmmOuter, 32 * kMiB, 4},
    {AllocationTag::kBlockedMetricStrip, 16 * kMiB, 8},
    {AllocationTag::kFarFieldPairVector, 4 * kMiB, 4},
    {AllocationTag::kStartsTable, 16 * kKiB, 2048},
}};

struct ScheduleTotals {
    std::array<std::uint64_t, static_cast<std::size_t>(AllocationTag::kUnclassified) + 1u>
        perTagBytes{};
    std::uint64_t totalBytes = 0;
    int blockCount = 0;
};

ScheduleTotals ComputeScheduleTotals(const std::array<ScheduleRow, 17>& rows) {
    ScheduleTotals totals;

    for (const ScheduleRow& row : rows)
    {
        const std::size_t tagIndex = static_cast<std::size_t>(row.tag);
        totals.perTagBytes[tagIndex] += row.blockBytes * static_cast<std::uint64_t>(row.blockCount);
        totals.totalBytes += row.blockBytes * static_cast<std::uint64_t>(row.blockCount);
        totals.blockCount += row.blockCount;
    }

    return totals;
}

// Writes one byte per page: commit follows the first write on Windows (the
// cap metric is commit, not reservation - an untouched buffer never reaches
// the job peak, so every block of the synthetic stream is swept before the
// hold).
void TouchPages(void* block, std::size_t bytes) {
    auto* cursor = static_cast<volatile char*>(block);
    const auto* end = cursor + bytes;

    for (; cursor < end; cursor += kPageSize)
    {
        *cursor = 1;
    }
}

// Recursively commits the calling thread's stack: each frame zero-initializes
// an 8 KiB array (two pages, both written) at its own depth. Depth 96 commits
// ~768 KiB of the default 1 MiB stack, safely below the guard page.
void CommitStackDepth(int depth) {
    volatile char frame[8192];
    frame[0] = static_cast<char>(depth);
    frame[8191] = static_cast<char>(depth + 1);

    if (depth > 0)
    {
        CommitStackDepth(depth - 1);
    }

    (void)frame[0];
}

// Self-assigns the probe to its own unnamed job object so the instrument's
// bound-handle default job-peak query has a live handle even when no outer
// cap job exists (the python driver applies the real cap job; the two nest -
// the driver's read-out stays authoritative). Fails open: nullptr on any
// error (the trace's job-peak column is then 0 and the run still measures
// everything else).
void* CreateOwnJobHandle() {
#ifdef _WIN32
    const HANDLE job = CreateJobObjectW(nullptr, nullptr);

    if (job == nullptr)
    {
        return nullptr;
    }

    if (!AssignProcessToJobObject(job, GetCurrentProcess()))
    {
        CloseHandle(job);
        return nullptr;
    }

    return job;
#else
    return nullptr;
#endif
}

bool ParseUint64(const char* text, std::uint64_t& out) {
    if (text == nullptr || *text == '\0')
    {
        return false;
    }

    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);

    if (end == text || *end != '\0')
    {
        return false;
    }

    out = static_cast<std::uint64_t>(value);
    return true;
}

bool ParseUint32(const char* text, std::uint32_t& out) {
    std::uint64_t value = 0;

    if (!ParseUint64(text, value) || value > 0xFFFFFFFFull)
    {
        return false;
    }

    out = static_cast<std::uint32_t>(value);
    return true;
}

struct ProbeArgs {
    std::string leg;
    std::string tracePath;
    std::string runId;
    std::uint32_t threads = 32;
    std::uint64_t holdMs = 4000;
    std::uint64_t capBytes = 0;
    std::uint64_t marginBytes = 256u * kMiB;
};

int PrintUsageAndFail() {
    std::printf("usage: delta_probe <plain|base|threads|synthetic|selftest> "
                "[--trace PATH] [--threads N] [--hold-ms N] [--run-id ID] "
                "[--cap-bytes N] [--margin-bytes N]\n");
    return 2;
}

// The shared arg parse for the four measured legs (selftest takes none).
bool ParseArgs(int argc, char** argv, ProbeArgs& args) {
    if (argc < 2)
    {
        return false;
    }

    args.leg = argv[1];

    for (int i = 2; i + 1 < argc + 1; i += 2)
    {
        if (i + 1 >= argc)
        {
            return false;
        }

        const std::string flag = argv[i];
        const char* value = argv[i + 1];

        if (flag == "--trace")
        {
            args.tracePath = value;
        } else if (flag == "--threads")
        {
            if (!ParseUint32(value, args.threads))
            {
                return false;
            }
        } else if (flag == "--hold-ms")
        {
            if (!ParseUint64(value, args.holdMs))
            {
                return false;
            }
        } else if (flag == "--run-id")
        {
            args.runId = value;
        } else if (flag == "--cap-bytes")
        {
            if (!ParseUint64(value, args.capBytes))
            {
                return false;
            }
        } else if (flag == "--margin-bytes")
        {
            if (!ParseUint64(value, args.marginBytes))
            {
                return false;
            }
        } else
        {
            return false;
        }
    }

    return true;
}

// Enables the instrument with the probe's metadata (the schedule's totals are
// the "predicted" per-tag high waters: the probe's own planned stream; the
// measured rows verify the accounting against them).
int EnableForLeg(const ProbeArgs& args, bool synthetic) {
    AllocationInstrumentOptions options;
    options.traceFilePath = args.tracePath;
    options.metadata.runId = args.runId;
    options.metadata.fixture = "delta-probe:" + args.leg;
    options.metadata.threadCount = args.threads;
    options.metadata.allocatorVersion = "qcx-delta-probe-v1";
    options.metadata.memoryCapBytes = args.capBytes;
    options.metadata.deltaMarginBytes = args.marginBytes;
    options.metadata.runTimestamp = "";

    if (synthetic)
    {
        const ScheduleTotals totals = ComputeScheduleTotals(kSyntheticSchedule);
        options.metadata.predictedHighWaterBytes = totals.perTagBytes;
        options.metadata.predictedTotalRequestBytes = totals.totalBytes;
    }

    const auto result = AllocationInstrumentEnable(options);

    if (!result.has_value())
    {
        std::printf("enable_failed=%s\n", result.error().message.c_str());
        return 1;
    }

    return 0;
}

void PrintStatsSummary() {
    std::printf("global_peak_request_bytes=%llu\n",
                static_cast<unsigned long long>(GlobalPeakRequestBytes()));
    std::printf("global_current_request_bytes=%llu\n",
                static_cast<unsigned long long>(GlobalCurrentRequestBytes()));

    for (int tag = 0; tag < static_cast<int>(AllocationTag::kUnclassified); ++tag)
    {
        const AllocationTag tagValue = static_cast<AllocationTag>(tag);
        std::printf("tag_%s_peak=%llu tag_%s_current=%llu tag_%s_count=%llu\n",
                    AllocationTagName(tagValue),
                    static_cast<unsigned long long>(TagPeakBytes(tagValue)),
                    AllocationTagName(tagValue),
                    static_cast<unsigned long long>(TagCurrentBytes(tagValue)),
                    AllocationTagName(tagValue),
                    static_cast<unsigned long long>(TagAllocationCount(tagValue)));
    }
}

int RunPlain(const ProbeArgs& args) {
    // The uninstrumented base (B0): identical argv and hold to the base leg
    // except the instrument is never enabled - the difference between the two
    // job peaks is the measured instrumented baseline.
    std::this_thread::sleep_for(std::chrono::milliseconds(args.holdMs));
    std::printf("leg=plain\n");
    return 0;
}

// The own-job handle for the measured legs: the instrument samples it for the
// trace's job-peak column (the cap metric on the process itself).
struct OwnJobGuard {
    void* handle;

    OwnJobGuard() : handle(CreateOwnJobHandle()) {
        AllocationInstrumentBindCapJob(handle);
    }

    ~OwnJobGuard() {
        AllocationInstrumentBindCapJob(nullptr);

#ifdef _WIN32
        if (handle != nullptr)
        {
            CloseHandle(handle);
        }
#endif
    }
};

int RunBase(const ProbeArgs& args) {
    OwnJobGuard jobGuard;
    (void)jobGuard;
    int status = EnableForLeg(args, false);

    if (status != 0)
    {
        return status;
    }

    const std::chrono::milliseconds hold(args.holdMs);
    std::this_thread::sleep_for(hold);
    PrintStatsSummary();

    const auto result = AllocationInstrumentDisable();

    if (!result.has_value())
    {
        std::printf("disable_failed=%s\n", result.error().message.c_str());
        return 1;
    }

    std::printf("leg=base\n");
    return 0;
}

void ThreadProbeMain(std::chrono::milliseconds hold, std::atomic<int>& arrived) {
    CommitStackDepth(96);
    arrived.fetch_add(1, std::memory_order_relaxed);
    std::this_thread::sleep_for(hold);
}

int RunThreads(const ProbeArgs& args) {
    OwnJobGuard jobGuard;
    (void)jobGuard;
    int status = EnableForLeg(args, false);

    if (status != 0)
    {
        return status;
    }

    const std::chrono::milliseconds hold(args.holdMs);
    std::atomic<int> arrived{0};
    std::vector<std::thread> workers;

    for (std::uint32_t i = 0; i < args.threads; ++i)
    {
        workers.emplace_back(ThreadProbeMain, hold, std::ref(arrived));
    }

    // All stacks are committed only when every worker has touched its own.
    while (arrived.load(std::memory_order_relaxed) < static_cast<int>(args.threads))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    for (std::thread& worker : workers)
    {
        worker.join();
    }

    PrintStatsSummary();

    const auto result = AllocationInstrumentDisable();

    if (!result.has_value())
    {
        std::printf("disable_failed=%s\n", result.error().message.c_str());
        return 1;
    }

    std::printf("leg=threads threads=%u\n", args.threads);
    return 0;
}

// The probe's own held-block records: the deallocating call site must state
// the size again (the instrument keeps no per-allocation metadata), so the
// stream keeps (tag, bytes, pointer) triples - plain heap of the std::vector,
// invisible to the instrument (only the tagged entries are counted).
struct HeldBlock {
    AllocationTag tag;
    std::size_t bytes;
    void* p;
};

int RunSynthetic(const ProbeArgs& args) {
    OwnJobGuard jobGuard;
    (void)jobGuard;
    int status = EnableForLeg(args, true);

    if (status != 0)
    {
        return status;
    }

    const ScheduleTotals totals = ComputeScheduleTotals(kSyntheticSchedule);
    std::vector<HeldBlock> held;
    held.reserve(static_cast<std::size_t>(totals.blockCount));

    // The allocation ramp: every row's blocks go live, so the global request
    // high-water fires at every step and the last allocation lands the
    // schedule's total as the global request peak.
    for (const ScheduleRow& row : kSyntheticSchedule)
    {
        for (int i = 0; i < row.blockCount; ++i)
        {
            void* block = AllocateTagged(row.tag, row.blockBytes);

            if (block == nullptr)
            {
                std::printf("allocation_failed tag=%s\n", AllocationTagName(row.tag));
                return 1;
            }

            held.push_back(HeldBlock{row.tag, row.blockBytes, block});
        }
    }

    // The commit ramp: every block is page-touched, so the job peak commit
    // rises to the stream's true commit (reservation alone never reaches the
    // cap metric).
    for (const HeldBlock& block : held)
    {
        TouchPages(block.p, block.bytes);
    }

    // The peak plateau: everything is live and committed; the watchdog's
    // interval rows sample the job peak at its maximum while the per-tag
    // currents read the schedule's totals.
    std::this_thread::sleep_for(std::chrono::milliseconds(args.holdMs));

    // The mirror-image release (explicit tag + size on the free path).
    for (auto it = held.rbegin(); it != held.rend(); ++it)
    {
        DeallocateTagged(it->tag, it->p, it->bytes);
    }

    PrintStatsSummary();

    const auto result = AllocationInstrumentDisable();

    if (!result.has_value())
    {
        std::printf("disable_failed=%s\n", result.error().message.c_str());
        return 1;
    }

    // The leg doubles as an accounting self-check at the stream's own scale:
    // the measured request peaks must equal the schedule the trace metadata
    // predicted, and the counters must return to zero.
    bool accountingOk =
        GlobalPeakRequestBytes() == totals.totalBytes && GlobalCurrentRequestBytes() == 0u;

    for (std::size_t tag = 0; tag < totals.perTagBytes.size(); ++tag)
    {
        const AllocationTag tagValue = static_cast<AllocationTag>(tag);

        if (TagPeakBytes(tagValue) != totals.perTagBytes[tag] || TagCurrentBytes(tagValue) != 0u)
        {
            accountingOk = false;
        }
    }

    std::printf("leg=synthetic blocks=%d expected_total_request=%llu "
                "accounting_ok=%d\n",
                totals.blockCount,
                static_cast<unsigned long long>(totals.totalBytes),
                accountingOk ? 1 : 0);
    return accountingOk ? 0 : 1;
}

int RunSelfTest() {
    // A small scripted stream over three tags plus the aligned path: the
    // accounting must be exact (global and per-tag peaks/counts) and the
    // unclassified bucket must stay empty - the SCOPE-HYGIENE assertion at the
    // binary level, runnable without a job object. It is not a coverage check:
    // the bucket counts only wrapper traffic with no scope, so it says nothing
    // about memory that never reached a wrapper (the instrument header's
    // blind-spot list).
    AllocationInstrumentOptions options;
    const auto result = AllocationInstrumentEnable(options);

    if (!result.has_value())
    {
        std::printf("enable_failed=%s\n", result.error().message.c_str());
        return 1;
    }

    constexpr std::size_t kBlockA = 1000;
    constexpr std::size_t kBlockB = 3000;
    constexpr std::size_t kBlockC = 2000;

    void* a = AllocateTagged(AllocationTag::kPairStore, kBlockA);
    void* b = AllocateTagged(AllocationTag::kScratch, kBlockB);
    void* c = AllocateTagged(AllocationTag::kPairStore, kBlockC);

    if (a == nullptr || b == nullptr || c == nullptr)
    {
        std::printf("selftest allocation failed\n");
        return 1;
    }

    const bool peakCorrect = GlobalPeakRequestBytes() == kBlockA + kBlockB + kBlockC &&
                             TagPeakBytes(AllocationTag::kPairStore) == kBlockA + kBlockC &&
                             TagPeakBytes(AllocationTag::kScratch) == kBlockB;
    DeallocateTagged(AllocationTag::kPairStore, a, kBlockA);
    DeallocateTagged(AllocationTag::kScratch, b, kBlockB);
    DeallocateTagged(AllocationTag::kPairStore, c, kBlockC);

    const bool emptyAfterFree = GlobalCurrentRequestBytes() == 0u &&
                                TagCurrentBytes(AllocationTag::kPairStore) == 0u &&
                                TagCurrentBytes(AllocationTag::kScratch) == 0u &&
                                TagAllocationCount(AllocationTag::kPairStore) == 2u &&
                                TagAllocationCount(AllocationTag::kScratch) == 1u &&
                                TagCurrentBytes(AllocationTag::kUnclassified) == 0u &&
                                TagAllocationCount(AllocationTag::kUnclassified) == 0u;

    const auto disableResult = AllocationInstrumentDisable();
    const bool disabled = disableResult.has_value();

    std::printf("leg=selftest peak_correct=%d empty_after_free=%d disabled=%d\n",
                peakCorrect ? 1 : 0,
                emptyAfterFree ? 1 : 0,
                disabled ? 1 : 0);
    return peakCorrect && emptyAfterFree && disabled ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    ProbeArgs args;

    if (!ParseArgs(argc, argv, args))
    {
        return PrintUsageAndFail();
    }

    if (args.leg == "plain")
    {
        return RunPlain(args);
    }

    if (args.leg == "base")
    {
        return RunBase(args);
    }

    if (args.leg == "threads")
    {
        return RunThreads(args);
    }

    if (args.leg == "synthetic")
    {
        return RunSynthetic(args);
    }

    if (args.leg == "selftest")
    {
        return RunSelfTest();
    }

    std::printf("unknown leg=%s\n", args.leg.c_str());
    return PrintUsageAndFail();
}
