#pragma once

/// \file
/// The per-term allocation-attribution instrument. Tagged size-class
/// high-water at the allocator-request scale: source-level tagged malloc/free
/// wrappers carry a thread-local tag to the real allocator with the SAME size
/// and alignment and update a fixed preallocated stats block in static storage
/// (per-tag current/peak/count; global current/peak). No per-allocation
/// metadata (the tag passes through the call site, never stored in the block);
/// no heap traffic from the hooks (the stats block is the only state the hot
/// paths touch). One watchdog thread samples the job-object peak commit
/// (JOB_OBJECT_LIMIT_PROCESS_MEMORY - the metric IDENTICAL to the cap metric:
/// QueryInformationJobObject with JobObjectExtendedLimitInformation class 9,
/// PeakProcessMemoryUsed, the same structure the benchmark harness gate and
/// ApplyProcessCaps read) on an interval and on every global request
/// high-water event, and writes snapshots WRITE-THROUGH (a snapshot is the
/// job peak commit + the global request + the per-tag currents at that
/// instant; flushed to the trace file row by row, so the record survives a
/// cap kill).
///
/// The instrument is OFF by default (AllocationInstrumentEnabled() is false
/// until AllocationInstrumentEnable is called) and its off path is the plain
/// real-allocator call: two relaxed loads and a branch, no stats traffic.
///
/// WHAT THIS INSTRUMENT REPORTS, AND WHAT IT CANNOT. This header is the single
/// home of the instrument's contract; a document that needs it POINTS here
/// rather than restating it. Three numbers, stated and used SEPARATELY - never
/// one in place of another:
///
/// 1. THE JOB-OBJECT PEAK COMMIT - the ground truth: the maximum of the
/// instantaneous committed footprint over the run, from the same job-object
/// metric the cap reads. The primary metric.
/// 2. THE SUM OF THE TAGGED PEAKS - a LOWER BOUND on the part of that peak
/// reached through a tagged wrapper, and explicitly NOT a decomposition of it.
/// The per-tag peaks are NOT simultaneous: a family whose peak does not
/// coincide with the global peak contributes ZERO to it, so the sum must NEVER
/// be treated as the peak commit and never presented as one (the
/// reconciliation rule above). The trace rows carry both the
/// per-tag peaks over the run and the per-tag currents AT the job-peak
/// snapshot (the row with the highest job peak commit column).
/// 3. THE RESIDUAL - the job peak commit minus the sum of the tagged peaks
/// minus the measured instrumentation overhead (AttributionRealizedDeltaBytes).
/// Report it as it stands; do not distribute it over the families.
///
/// There is no coverage percentage and no coverage test here: a ratio of 2 to 1
/// would assert an additivity the second number does not have.
///
/// BLIND SPOTS - memory this instrument cannot see AT ALL, whatever tags the
/// sites carry. Interception is CALL-SITE ONLY: every entry bottoms out in
/// ::operator new, and memory/ carries no global operator new replacement and
/// no malloc hook. Invisible: (a) allocations through a library's own allocator
/// - Eigen::MatrixXd's aligned path above all, which holds this fixture's
/// largest modelled terms; (b) any other third-party allocator reached without
/// a qcx wrapper (BLAS/MKL workspaces included); (c) thread stacks; (d) CRT
/// startup and static-init memory; (e) page-fault-driven commits of
/// already-reserved regions - allocation is not the only way memory arrives,
/// and that path has no allocator call to intercept at all; (f) page-table and
/// kernel structures charged to the job; (g) memory-mapped sections.
///
/// THE BASELINE: a trivial program under the same job object is the floor of
/// the residual, and it cannot be attributed to families. Measure it; never
/// assume it negligible.
///
/// DEFERRED - the tagging campaign: the documented contract is correct now,
/// the tagging is deferred. Ten of the eighteen tags (below)
/// name real memory and have no production site today, and the fix is not "add
/// a scope": eight of the ten are plain std::vector sites, so the cost is the
/// CONTAINER TYPE at its declaration (pair_store alone is passed by reference
/// across 30+ files and carries four inner vectors as struct members; pattern
/// and structural are public headers), and the Eigen group (base's SCF matrix
/// set, structural's core-H copies, rij_fast_tensor's retained _riMatrix)
/// cannot be tagged at all - no qcx scope or container change reaches
/// Eigen's allocator.
///
/// The 18 family tags (the verbatim tokens of the trace):
/// base, pair_store, pattern, scratch, structural, cache,
/// exchange_f64_bound_f32_live, class_table, screened_quartet,
/// rij_fast_tensor, rij_transpose, light_rung_slice, light_rung_values,
/// qfmm_outer, blocked_metric_strip, far_field_pair_vector, starts_table,
/// unclassified. Their allocation-site map (integrals/src/internal/
/// footprint.hpp is the ground truth for the model families) lives in the
/// AllocationTag enum docs below; unattributed traffic (wrapper calls with no
/// tag scope active) lands in unclassified.
///
/// The verified-vs-refuted tolerance is the single named constant
/// kTermRefutationToleranceBytes: a term is refuted when
/// measured > charged + kTermRefutationToleranceBytes.
/// \ingroup qcx-memory

#include "qcx/error.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <string_view>

namespace qcx::memory {

/// The attribution family tags.
///
/// The site map (footprint.hpp is the ground truth; each family is charged
/// by the term named in its docs): \li base - the driver/SCF-side per-
/// iteration n^2 working set (the model's base term, anchored at the commit
/// scale); \li pair_store - the BuildContractedPairTransform pair stores of
/// md_batch.cpp (MdPairData payloads), the BuildRiTensor orbital store, the
/// BuildAuxMetric metric kets (ri_engine.cpp); \li pattern - the neighbor
/// CSR (BuildNeighborList, fock_screen.hpp); \li scratch - the per-thread
/// batch arenas (batch x threads, the direct and 3c arenas); \li structural
/// - the retained ShellPairList copies, Schwarz vector, CSR offsets, the
/// core-H copies; \li cache - the ERI cache payload lanes and entry map
/// (eri_cache.cpp); \li exchange_f64_bound_f32_live - the per-batch
/// exchange buffers of both lanes (fock_build.cpp RunPass: the packed fp64
/// values + workspace, the fp32 valuesF32 and per-quartet bounds);
/// \li class_table - the pair-class table and the on-demand orbit
/// generation (BuildPairClasses / GenerateClassPairOrbits); \li
/// screened_quartet - the whole-call sorted task masters (MdQuartetTask
/// arrays, the S2 survivor); \li rij_fast_tensor - the (uv|P) values buffer
/// and the retained riMatrix (8 n^2 nAux); \li rij_transpose - the fast
/// rung's per-iteration transpose copy; \li light_rung_slice - the light
/// rung's per-iteration recompute slice; \li light_rung_values - the light
/// rung's per-iteration values buffer; \li qfmm_outer - the QFMM state's
/// retained allocations (octree, leafOfPair copies, restriction bitset,
/// moment table, per-pair geometries, its pair-store and pair-list copies,
/// qfmm_tree.cpp); \li blocked_metric_strip - the blocked-metric rung's
/// per-strip arena (BuildBlockedAuxMetric); \li far_field_pair_vector - the
/// retained far-field pair vector (16 B per well-separated node pair, the
/// documented unmodeled family); \li starts_table - the k > 1 rung's
/// per-half batchStarts tables; \li unclassified -
/// wrapper traffic with no scope active. It is NOT a coverage check and never
/// evidence that the run's memory was accounted for: it counts only what
/// REACHED A WRAPPER without a scope, so it reads zero however large the blind
/// spots above are, and it says nothing at all about memory that never reached
/// a wrapper. Read it as "no badly-scoped wrapper traffic", and read nothing
/// else into it; the instrumented test runs assert it stays empty, which pins
/// the call sites' scope hygiene and not coverage.
enum class AllocationTag : std::uint8_t {
    kBase = 0,
    kPairStore = 1,
    kPattern = 2,
    kScratch = 3,
    kStructural = 4,
    kCache = 5,
    kExchangeF64BoundF32Live = 6,
    kClassTable = 7,
    kScreenedQuartet = 8,
    kRijFastTensor = 9,
    kRijTranspose = 10,
    kLightRungSlice = 11,
    kLightRungValues = 12,
    kQfmmOuter = 13,
    kBlockedMetricStrip = 14,
    kFarFieldPairVector = 15,
    kStartsTable = 16,
    kUnclassified = 17,
};

/// The number of allocation tags (kUnclassified included).
inline constexpr std::size_t kAllocationTagCount = 18;

/// The default alignment of the plain tagged paths: the alignment of the
/// real allocator the sites call today (::operator new on the CRT), which
/// the hooks must call with the SAME size and alignment.
inline constexpr std::size_t kAttributionDefaultAlignment = alignof(std::max_align_t);

/// The verified-vs-refuted tolerance: a term is refuted when its measured
/// peak request exceeds its charged bytes by more than this. One MiB: the
/// smallest whole-run report unit, sized above the request-scale accounting
/// noise (a single block crossing a scope boundary at a term edge). Change it
/// only deliberately; never as a silent tweak.
inline constexpr std::uint64_t kTermRefutationToleranceBytes = 1024u * 1024u;

/// The trace-file format version written by the watchdog.
inline constexpr const char* kAttributionTraceVersion = "1";

/// The tag's trace token ("base", "pair_store", ...).
/// \param tag The tag.
/// \returns The token; never null.
const char* AllocationTagName(AllocationTag tag) noexcept;

/// True once AllocationInstrumentEnable has been called and not yet disabled.
/// \returns true while the instrument is enabled; false otherwise.
bool AllocationInstrumentEnabled() noexcept;

/// The per-tag counters of the fixed preallocated stats block (one cache
/// line per tag so tag-hot lines never share).
struct alignas(64) TagAttributionCounters {
    /// Live request bytes under the tag (fetch-add/sub paired at the sites).
    std::atomic<std::uint64_t> currentBytes{0};
    /// The tag's high-water live request bytes over the run.
    std::atomic<std::uint64_t> peakBytes{0};
    /// The tag's allocation count over the run.
    std::atomic<std::uint64_t> count{0};
};

/// The fixed preallocated stats block: static storage, never heap - the
/// hooks' only state. Reset (zeroed) by every AllocationInstrumentEnable.
inline std::array<TagAttributionCounters, kAllocationTagCount> gAttributionStats{};

/// The global live request bytes over all tags (wrapper traffic only).
inline std::atomic<std::uint64_t> gAttributionGlobalCurrentBytes{0};

/// The global request high-water over all tags.
inline std::atomic<std::uint64_t> gAttributionGlobalPeakBytes{0};

/// The instrument's enabled flag (relaxed; the wrapper fast path reads it).
inline std::atomic<bool> gAttributionEnabled{false};

/// Set by the allocation fast path whenever the global request high-water
/// rises; cleared by the watchdog after the snapshot row. A lost set (a
/// snapshot already in flight) only delays the row by at most one interval.
inline std::atomic<bool> gAttributionHighWaterEvent{false};

/// The tag the calling thread's allocations attribute to right now
/// (kUnclassified with no scope active - wrapper traffic with no scope, which
/// is not a coverage reading; see the tag's own note).
/// \returns The active tag; kUnclassified when no scope is live.
AllocationTag ActiveAllocationTag() noexcept;

/// RAII tag scope: every tagged allocation on this thread while the scope is
/// live attributes to \p tag. Nested scopes restore the enclosing tag on
/// exit. Constructing and destroying a scope is never heap traffic.
class AllocationTagScope {
public:
    /// Opens a scope: the calling thread's active tag becomes \p tag.
    /// \param tag The tag allocations on this thread attribute to while the
    /// scope is live.
    explicit AllocationTagScope(AllocationTag tag) noexcept;
    /// Closes the scope: the calling thread's active tag is restored.
    ~AllocationTagScope() noexcept;

    AllocationTagScope(const AllocationTagScope&) = delete;
    AllocationTagScope& operator=(const AllocationTagScope&) = delete;

    /// The tag this scope attributes to.
    /// \returns The scope's tag.
    AllocationTag Tag() const noexcept {
        return _tag;
    }

private:
    AllocationTagScope* _previous;
    AllocationTag _tag;
};

/// The thread-local scope chain (inline variable so the header-inline hot
/// paths and the scope RAII share one pointer).
inline thread_local AllocationTagScope* gAttributionActiveScope = nullptr;

/// The run metadata carried into the trace header. Filled by whoever enables
/// the instrument for a run; the measured fields are the rows'.
struct AttributionTraceMetadata {
    /// The run identifier.
    std::string runId;
    /// The fixture name (e.g. C42H86/def2-QZVP).
    std::string fixture;
    /// The basis function count.
    std::uint64_t basisFunctionCount = 0;
    /// The path/preset flags.
    std::string pathFlags;
    /// The thread count of the run.
    std::uint32_t threadCount = 0;
    /// The allocator version string.
    std::string allocatorVersion;
    /// The run's memory cap in bytes.
    std::uint64_t memoryCapBytes = 0;
    /// The predicted per-term high-water requests (per tag, bytes).
    std::array<std::uint64_t, kAllocationTagCount> predictedHighWaterBytes{};
    /// The total predicted request bytes.
    std::uint64_t predictedTotalRequestBytes = 0;
    /// The predicted commit bytes.
    std::uint64_t predictedCommitBytes = 0;
    /// The named delta margin (bytes; the model record's margin_gib wins).
    std::uint64_t deltaMarginBytes = 0;
    /// The base term (bytes).
    std::uint64_t baseTermBytes = 0;
    /// The first of the base term's two anchor commits (bytes).
    std::uint64_t baseAnchor1Bytes = 0;
    /// The second of the base term's two anchor commits (bytes).
    std::uint64_t baseAnchor2Bytes = 0;
    /// The admission text ("admit" / "refuse").
    std::string admission;
    /// The binary git hash (40 hex characters).
    std::string binaryGitHash;
    /// The check label (names the check this trace belongs to).
    std::string checkLabel;
    /// The run timestamp (filled by the enabler).
    std::string runTimestamp;
};

/// The watchdog options (AllocationInstrumentEnable).
struct AllocationInstrumentOptions {
    /// The snapshot interval (rows also fire on every global request
    /// high-water event, so the interval bounds only the commit sampling
    /// staleness).
    std::chrono::milliseconds snapshotIntervalMs{250};
    /// The write-through trace file path; empty disables the watchdog (the
    /// stats block still tracks; nothing is written).
    std::string traceFilePath;
    /// The run metadata emitted as the trace header.
    AttributionTraceMetadata metadata;
};

/// The job-peak commit query override (tests and embedded runners inject
/// their own source; the default queries the bound cap job on Windows and
/// returns 0 elsewhere).
using JobPeakCommitQuery = std::uint64_t (*)() noexcept;

/// Enables the instrument: zeroes the stats block, turns the wrapper fast
/// path on, and (with a traceFilePath) starts the one watchdog thread, which
/// samples the job-object peak commit on the interval and on every global
/// request high-water event and appends snapshot rows to the trace file with
/// a write-through flush after every row.
///
/// Enabling is the only place the stats block is touched wholesale - safe
/// because the enabled flag is stored strictly after the zeroing and cleared
/// strictly before the next enable.
/// \param options The watchdog and trace options.
/// \returns Success, or an Error (kInvalidArgument when already enabled;
/// kInternalError when the trace file cannot be opened or the watchdog
/// thread cannot start).
qcx::Result<void> AllocationInstrumentEnable(const AllocationInstrumentOptions& options);

/// Disables the instrument: the wrapper fast path turns off, the watchdog
/// (if any) writes one final snapshot row and exits, and the trace file is
/// closed. The stats block stays readable until the next enable zeroes it.
/// Idempotent (disabling a disabled instrument succeeds).
/// \returns Success.
qcx::Result<void> AllocationInstrumentDisable();

/// Records a run-level refusal in the open attribution trace. A refused run
/// reaches no tagged allocation site, so its
/// trace ends indistinguishable from a run that allocated nothing tagged:
/// every per-tag counter is zero in every snapshot row and every
/// `predicted_high_water_*` meta value is the placeholder. The trace is NOT
/// byte-empty in that case - the watchdog ticks for the whole run - which is
/// exactly what makes the zeroed record look like a measurement instead of a
/// non-event. This appends ONE terminal line naming the refusal:
///
///     #refused REASON
///
/// A `#`-prefixed line, so it never parses as a snapshot row (the readers
/// skip `#` lines, and the row grammar is unchanged); the token a consumer
/// greps for is a line opening with `#refused ` (with the trailing space),
/// and REASON is the caller's own refusal text, verbatim except that every
/// newline and carriage return is flattened to a space, so the record stays
/// one line whatever the message contains.
///
/// Call it BEFORE AllocationInstrumentDisable: the trace file is closed by
/// the disable, and a call with no open trace file is a silent no-op (the
/// stats-only mode has nowhere to write it, and a disabled instrument is
/// never an error here - the refusal path must not fail closed on its own
/// record).
/// \param reason The refusal text (the driver's own error message).
void AllocationInstrumentRecordRefusal(std::string_view reason) noexcept;

/// Binds the cap job object whose ProcessMemoryLimit is the run's cap; the
/// watchdog's default peak query reads its PeakProcessMemoryUsed (class 9
/// extended-limit information - the metric identical to the cap metric).
/// \param jobObjectHandle The Windows job handle (a HANDLE, carried as
/// void* so the header stays portable); nullptr unbinds.
void AllocationInstrumentBindCapJob(void* jobObjectHandle) noexcept;

/// Overrides the watchdog's job-peak commit query (the default reads the
/// bound cap job on Windows; 0 elsewhere). Tests inject a scripted source;
/// embedded non-Windows runners inject their external-job query.
/// \param query The query; nullptr restores the default.
void AllocationInstrumentSetJobPeakQuery(JobPeakCommitQuery query) noexcept;

/// The per-tag live request bytes.
/// \param tag The tag.
/// \returns The live request bytes currently charged to \p tag.
std::uint64_t TagCurrentBytes(AllocationTag tag) noexcept;

/// The per-tag high-water live request bytes.
/// \param tag The tag.
/// \returns The high-water live request bytes charged to \p tag.
std::uint64_t TagPeakBytes(AllocationTag tag) noexcept;

/// The per-tag allocation count.
/// \param tag The tag.
/// \returns The number of allocations charged to \p tag.
std::uint64_t TagAllocationCount(AllocationTag tag) noexcept;

/// The global live request bytes over all wrapper traffic.
/// \returns The live request bytes currently charged across all tags.
std::uint64_t GlobalCurrentRequestBytes() noexcept;

/// The global request high-water over all wrapper traffic.
/// \returns The high-water live request bytes across all tags.
std::uint64_t GlobalPeakRequestBytes() noexcept;

/// The per-term falsification: a term is refuted when its measured peak request
/// exceeds its charged bytes by more than kTermRefutationToleranceBytes
/// (measured > charged + tolerance). Measured at or below the charge needs
/// no tolerance - the charged formulas are never-under by construction.
/// \param measuredPeakBytes The tag's measured peak request over the run.
/// \param chargedBytes The term's charged request bytes.
/// \returns true when the term is refuted; false otherwise.
constexpr bool AttributionTermRefuted(std::uint64_t measuredPeakBytes,
                                      std::uint64_t chargedBytes) noexcept {
    return measuredPeakBytes > chargedBytes + kTermRefutationToleranceBytes;
}

/// The realized aggregate delta (saturating) - the third number of the
/// contract above, THE RESIDUAL: job peak commit minus the instrumented global
/// peak request minus the measured instrumentation overhead. The delta is
/// path-specific, measured per real run, never assumed global; it is not a
/// family quantity, and the same job object's trivial-program baseline is its
/// floor.
/// \param jobPeakCommitBytes The job-object peak commit of the run.
/// \param globalPeakRequestBytes The instrumented global peak request.
/// \param instrumentationOverheadBytes The measured instrumentation
/// overhead (the delta probe's commit-scale components).
/// \returns The realized aggregate delta in bytes, saturating at zero.
constexpr std::uint64_t AttributionRealizedDeltaBytes(std::uint64_t jobPeakCommitBytes,
                                                      std::uint64_t globalPeakRequestBytes,
                                                      std::uint64_t instrumentationOverheadBytes) {
    return jobPeakCommitBytes > globalPeakRequestBytes + instrumentationOverheadBytes
               ? jobPeakCommitBytes - globalPeakRequestBytes - instrumentationOverheadBytes
               : 0;
}

/// The aggregate falsification: the realized delta exceeding the named
/// margin by more than the tolerance is refuted.
/// \param jobPeakCommitBytes The job-object peak commit of the run.
/// \param globalPeakRequestBytes The instrumented global peak request.
/// \param instrumentationOverheadBytes The measured instrumentation
/// overhead.
/// \param deltaMarginBytes The named delta margin (the model record's
/// margin_gib, bytes).
/// \returns true when the realized delta exceeds the margin by more than
/// the tolerance; false otherwise.
constexpr bool AttributionDeltaRefuted(std::uint64_t jobPeakCommitBytes,
                                       std::uint64_t globalPeakRequestBytes,
                                       std::uint64_t instrumentationOverheadBytes,
                                       std::uint64_t deltaMarginBytes) {
    return AttributionRealizedDeltaBytes(
               jobPeakCommitBytes, globalPeakRequestBytes, instrumentationOverheadBytes) >
           deltaMarginBytes + kTermRefutationToleranceBytes;
}

/// The explicit-tag allocation entry: the real allocator (::operator new,
/// nothrow) with the SAME size, then the fixed stats block under \p tag when
/// the instrument is enabled. Never heap traffic of its own.
/// \param tag The attribution tag.
/// \param bytes The requested bytes (never zero).
/// \returns The block, or nullptr when the allocator cannot satisfy the
/// request.
void* AllocateTagged(AllocationTag tag, std::size_t bytes) noexcept;

/// The explicit-tag deallocation entry, paired with AllocateTagged: the tag
/// and byte count pass through the call site (never stored in the block).
/// \param tag The allocation's tag (the same tag the AllocateTagged call
/// used - the per-tag current is exact only under symmetric pairing).
/// \param p The block to release; nullptr is a no-op.
/// \param bytes The allocation's requested bytes.
void DeallocateTagged(AllocationTag tag, void* p, std::size_t bytes) noexcept;

/// The explicit-tag aligned allocation entry: the real allocator (aligned
/// ::operator new, nothrow) with the SAME size and alignment.
/// \param tag The attribution tag.
/// \param bytes The requested bytes (never zero).
/// \param alignment The alignment (a power of two; when it does not exceed
/// kAttributionDefaultAlignment the plain path is used).
/// \returns The block, or nullptr when the allocator cannot satisfy the
/// request.
void* AlignedAllocateTagged(AllocationTag tag, std::size_t bytes, std::size_t alignment) noexcept;

/// The explicit-tag aligned deallocation entry, paired with
/// AlignedAllocateTagged (same tag, size and alignment).
/// \param tag The allocation's tag (the same tag the AlignedAllocateTagged
/// call used - the per-tag current is exact only under symmetric pairing).
/// \param p The block to release; nullptr is a no-op.
/// \param bytes The allocation's requested bytes.
/// \param alignment The allocation's alignment (the same alignment the
/// AlignedAllocateTagged call used).
void AlignedDeallocateTagged(AllocationTag tag,
                             void* p,
                             std::size_t bytes,
                             std::size_t alignment) noexcept;

/// The explicit-tag calloc entry: a zero-initialized block of
/// count x elementSize bytes, counted as one allocation under \p tag.
/// \param tag The attribution tag.
/// \param count The element count (never zero).
/// \param elementSize The element size in bytes (never zero).
/// \returns The block, or nullptr when the allocator cannot satisfy the
/// request.
void* CallocTagged(AllocationTag tag, std::size_t count, std::size_t elementSize) noexcept;

/// The explicit-tag realloc entry. The block is always moved (the stats need
/// both byte counts, which only the call site knows): the new block is
/// allocated, min(oldBytes, newBytes) copied, the old block released.
/// p == nullptr allocates newBytes; newBytes == 0 releases and returns
/// nullptr.
/// \param tag The attribution tag.
/// \param p The block to resize; nullptr allocates \p newBytes.
/// \param oldBytes The current block's byte count.
/// \param newBytes The requested byte count; zero releases \p p and returns
/// nullptr.
/// \returns The new block, or nullptr when the allocator cannot satisfy the
/// request (the original block stays allocated).
void* ReallocateTagged(AllocationTag tag,
                       void* p,
                       std::size_t oldBytes,
                       std::size_t newBytes) noexcept;

/// The scope-based tagged allocation: AllocateTagged(ActiveAllocationTag(),
/// bytes) - the tag flows from the calling site's scope.
/// \param bytes The requested bytes (never zero).
/// \returns The block, or nullptr when the allocator cannot satisfy the
/// request.
void* TaggedAllocate(std::size_t bytes) noexcept;

/// The scope-based tagged deallocation:
/// DeallocateTagged(ActiveAllocationTag(), p, bytes).
/// \param p The block to release; nullptr is a no-op.
/// \param bytes The allocation's requested bytes.
void TaggedDeallocate(void* p, std::size_t bytes) noexcept;

/// The scope-based aligned tagged allocation.
/// \param bytes The requested bytes (never zero).
/// \param alignment The alignment (a power of two; when it does not exceed
/// kAttributionDefaultAlignment the plain path is used).
/// \returns The block, or nullptr when the allocator cannot satisfy the
/// request.
void* TaggedAlignedAllocate(std::size_t bytes, std::size_t alignment) noexcept;

/// The scope-based aligned tagged deallocation.
/// \param p The block to release; nullptr is a no-op.
/// \param bytes The allocation's requested bytes.
/// \param alignment The allocation's alignment.
void TaggedAlignedDeallocate(void* p, std::size_t bytes, std::size_t alignment) noexcept;

/// The scope-based calloc.
/// \param count The element count (never zero).
/// \param elementSize The element size in bytes (never zero).
/// \returns The zero-initialized block, or nullptr when the allocator cannot
/// satisfy the request.
void* TaggedCalloc(std::size_t count, std::size_t elementSize) noexcept;

/// The scope-based realloc.
/// \param p The block to resize; nullptr allocates \p newBytes.
/// \param oldBytes The current block's byte count.
/// \param newBytes The requested byte count; zero releases \p p and returns
/// nullptr.
/// \returns The new block, or nullptr when the allocator cannot satisfy the
/// request (the original block stays allocated).
void* TaggedReallocate(void* p, std::size_t oldBytes, std::size_t newBytes) noexcept;

/// The std::allocator-compatible tagged allocator for the std::vector family
/// sites of the model families (the pair stores, patterns, task lists,
/// arenas): the tag is captured at the allocator's construction - the call
/// site - and every allocate/deallocate of the container (internal growth
/// included) attributes through it, exactly (deallocate receives the count).
/// The tag is never stored in the block.
/// \tparam T The element type.
template <typename T> class TaggedAllocator {
public:
    /// The element type (std::allocator conformance).
    using value_type = T;

    /// Captures the constructing thread's active tag (kUnclassified with no
    /// scope - the vector sites must construct the allocator inside the
    /// family's scope, which is how the site map tags them).
    TaggedAllocator() noexcept : _tag(ActiveAllocationTag()) {}

    /// The rebinding copy keeps the captured tag.
    /// \tparam U The other element type.
    /// \param other The allocator whose captured tag is copied.
    template <typename U>
    explicit TaggedAllocator(const TaggedAllocator<U>& other) noexcept : _tag(other.Tag()) {}

    /// Allocates storage for \p n elements through the tagged entry.
    /// \param n The element count.
    /// \returns The block; throws std::bad_alloc on failure (the
    /// std::allocator contract - the same failure mode the default
    /// allocator's vector growth has today).
    T* allocate(std::size_t n) {
        const std::size_t bytes = n * sizeof(T);

        void* p = sizeof(T) > kAttributionDefaultAlignment
                      ? static_cast<void*>(AlignedAllocateTagged(_tag, bytes, alignof(T)))
                      : AllocateTagged(_tag, bytes);

        if (p == nullptr)
        {
            throw std::bad_alloc();
        }

        return static_cast<T*>(p);
    }

    /// Releases storage for \p n elements (the byte count the stats need).
    /// \param p The block from allocate.
    /// \param n The element count allocated.
    void deallocate(T* p, std::size_t n) noexcept {
        const std::size_t bytes = n * sizeof(T);

        if (sizeof(T) > kAttributionDefaultAlignment)
        {
            AlignedDeallocateTagged(_tag, p, bytes, alignof(T));
        } else
        {
            DeallocateTagged(_tag, p, bytes);
        }
    }

    /// The captured tag (the rebinding source).
    /// \returns The tag.
    AllocationTag Tag() const noexcept {
        return _tag;
    }

private:
    AllocationTag _tag;

    template <typename U> friend class TaggedAllocator;
};

/// Equality: two tagged allocators are interchangeable when the captured
/// tags match (required for vector reallocation hand-offs).
/// \tparam T The element type of the left allocator.
/// \tparam U The element type of the right allocator.
/// \param lhs The left allocator.
/// \param rhs The right allocator.
/// \returns true when the captured tags match; false otherwise.
template <typename T, typename U>
bool operator==(const TaggedAllocator<T>& lhs, const TaggedAllocator<U>& rhs) noexcept {
    return lhs.Tag() == rhs.Tag();
}

/// Inequality: the negation of operator==.
/// \tparam T The element type of the left allocator.
/// \tparam U The element type of the right allocator.
/// \param lhs The left allocator.
/// \param rhs The right allocator.
/// \returns true when the captured tags differ; false otherwise.
template <typename T, typename U>
bool operator!=(const TaggedAllocator<T>& lhs, const TaggedAllocator<U>& rhs) noexcept {
    return !(lhs == rhs);
}

/// The fixed per-tag counter helper of the header-inline fast paths.
/// \param tag The tag; an out-of-range tag is clamped to kUnclassified.
/// \returns The tag's counter block.
inline TagAttributionCounters& TagCounters(AllocationTag tag) noexcept {
    const std::size_t index = static_cast<std::size_t>(tag);

    return gAttributionStats[index < kAllocationTagCount
                                 ? index
                                 : static_cast<std::size_t>(AllocationTag::kUnclassified)];
}

/// The tag's live request bytes (the header-inline read of the stats block).
/// \returns The live request bytes currently charged to \p tag.
inline std::uint64_t TagCurrentBytes(AllocationTag tag) noexcept {
    return TagCounters(tag).currentBytes.load(std::memory_order_relaxed);
}

/// The tag's high-water request bytes.
/// \returns The high-water live request bytes charged to \p tag.
inline std::uint64_t TagPeakBytes(AllocationTag tag) noexcept {
    return TagCounters(tag).peakBytes.load(std::memory_order_relaxed);
}

/// The tag's allocation count.
/// \returns The number of allocations charged to \p tag.
inline std::uint64_t TagAllocationCount(AllocationTag tag) noexcept {
    return TagCounters(tag).count.load(std::memory_order_relaxed);
}

/// The global live request bytes.
/// \returns The live request bytes currently charged across all tags.
inline std::uint64_t GlobalCurrentRequestBytes() noexcept {
    return gAttributionGlobalCurrentBytes.load(std::memory_order_relaxed);
}

/// The global request high-water.
/// \returns The high-water live request bytes across all tags.
inline std::uint64_t GlobalPeakRequestBytes() noexcept {
    return gAttributionGlobalPeakBytes.load(std::memory_order_relaxed);
}

/// The instrument's enabled flag.
/// \returns true while the instrument is enabled; false otherwise.
inline bool AllocationInstrumentEnabled() noexcept {
    return gAttributionEnabled.load(std::memory_order_relaxed);
}

/// The tag the calling thread's allocations attribute to right now.
/// \returns The active tag; kUnclassified when no scope is live.
inline AllocationTag ActiveAllocationTag() noexcept {
    const AllocationTagScope* scope = gAttributionActiveScope;

    return scope != nullptr ? scope->Tag() : AllocationTag::kUnclassified;
}

inline AllocationTagScope::AllocationTagScope(AllocationTag tag) noexcept :
    _previous(gAttributionActiveScope), _tag(tag) {
    gAttributionActiveScope = this;
}

inline AllocationTagScope::~AllocationTagScope() noexcept {
    gAttributionActiveScope = _previous;
}

/// The header-inline accounting of one allocation under \p tag: current and
/// count first, then the peak (a CAS loop - a new peak is rare), then the
/// global current and peak; a rising global high-water arms the watchdog
/// event flag. Atomic-only: never heap traffic.
/// \param tag The attribution tag.
/// \param bytes The allocated byte count.
inline void AccountAllocate(AllocationTag tag, std::size_t bytes) noexcept {
    if (!gAttributionEnabled.load(std::memory_order_relaxed))
    {
        return;
    }

    TagAttributionCounters& counters = TagCounters(tag);
    const std::uint64_t after =
        counters.currentBytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    counters.count.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t peak = counters.peakBytes.load(std::memory_order_relaxed);

    while (after > peak &&
           !counters.peakBytes.compare_exchange_weak(peak, after, std::memory_order_relaxed))
    {
    }

    const std::uint64_t globalAfter =
        gAttributionGlobalCurrentBytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    std::uint64_t globalPeak = gAttributionGlobalPeakBytes.load(std::memory_order_relaxed);

    while (globalAfter > globalPeak && !gAttributionGlobalPeakBytes.compare_exchange_weak(
                                           globalPeak, globalAfter, std::memory_order_relaxed))
    {
    }

    if (globalAfter > globalPeak)
    {
        gAttributionHighWaterEvent.store(true, std::memory_order_relaxed);
    }
}

/// The header-inline accounting of one deallocation under \p tag (the
/// mirror of AccountAllocate; the pairing is the call sites' contract).
/// \param tag The attribution tag.
/// \param bytes The released byte count.
inline void AccountDeallocate(AllocationTag tag, std::size_t bytes) noexcept {
    if (!gAttributionEnabled.load(std::memory_order_relaxed))
    {
        return;
    }

    TagCounters(tag).currentBytes.fetch_sub(bytes, std::memory_order_relaxed);
    gAttributionGlobalCurrentBytes.fetch_sub(bytes, std::memory_order_relaxed);
}

inline void* AllocateTagged(AllocationTag tag, std::size_t bytes) noexcept {
    void* p = ::operator new(bytes, std::nothrow);

    if (p != nullptr)
    {
        AccountAllocate(tag, bytes);
    }

    return p;
}

inline void DeallocateTagged(AllocationTag tag, void* p, std::size_t bytes) noexcept {
    if (p == nullptr)
    {
        return;
    }

    AccountDeallocate(tag, bytes);
    ::operator delete(p);
}

inline void* AlignedAllocateTagged(AllocationTag tag,
                                   std::size_t bytes,
                                   std::size_t alignment) noexcept {
    void* p;

    if (alignment <= kAttributionDefaultAlignment)
    {
        p = ::operator new(bytes, std::nothrow);
    } else
    {
        p = ::operator new(bytes, std::align_val_t(alignment), std::nothrow);
    }

    if (p != nullptr)
    {
        AccountAllocate(tag, bytes);
    }

    return p;
}

inline void AlignedDeallocateTagged(AllocationTag tag,
                                    void* p,
                                    std::size_t bytes,
                                    std::size_t alignment) noexcept {
    if (p == nullptr)
    {
        return;
    }

    AccountDeallocate(tag, bytes);

    if (alignment <= kAttributionDefaultAlignment)
    {
        ::operator delete(p);
    } else
    {
        ::operator delete(p, std::align_val_t(alignment));
    }
}

inline void* CallocTagged(AllocationTag tag, std::size_t count, std::size_t elementSize) noexcept {
    const std::size_t bytes = count * elementSize;
    void* p = ::operator new(bytes, std::nothrow);

    if (p != nullptr)
    {
        std::memset(p, 0, bytes);
        AccountAllocate(tag, bytes);
    }

    return p;
}

inline void* ReallocateTagged(AllocationTag tag,
                              void* p,
                              std::size_t oldBytes,
                              std::size_t newBytes) noexcept {
    if (p == nullptr)
    {
        return AllocateTagged(tag, newBytes);
    }

    if (newBytes == 0)
    {
        DeallocateTagged(tag, p, oldBytes);
        return nullptr;
    }

    void* next = ::operator new(newBytes, std::nothrow);

    if (next == nullptr)
    {
        return nullptr;
    }

    AccountAllocate(tag, newBytes);
    std::memcpy(next, p, oldBytes < newBytes ? oldBytes : newBytes);
    AccountDeallocate(tag, oldBytes);
    ::operator delete(p);
    return next;
}

inline void* TaggedAllocate(std::size_t bytes) noexcept {
    return AllocateTagged(ActiveAllocationTag(), bytes);
}

inline void TaggedDeallocate(void* p, std::size_t bytes) noexcept {
    DeallocateTagged(ActiveAllocationTag(), p, bytes);
}

inline void* TaggedAlignedAllocate(std::size_t bytes, std::size_t alignment) noexcept {
    return AlignedAllocateTagged(ActiveAllocationTag(), bytes, alignment);
}

inline void TaggedAlignedDeallocate(void* p, std::size_t bytes, std::size_t alignment) noexcept {
    AlignedDeallocateTagged(ActiveAllocationTag(), p, bytes, alignment);
}

inline void* TaggedCalloc(std::size_t count, std::size_t elementSize) noexcept {
    return CallocTagged(ActiveAllocationTag(), count, elementSize);
}

inline void* TaggedReallocate(void* p, std::size_t oldBytes, std::size_t newBytes) noexcept {
    return ReallocateTagged(ActiveAllocationTag(), p, oldBytes, newBytes);
}

} // namespace qcx::memory
