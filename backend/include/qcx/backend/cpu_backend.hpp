#pragma once

#include "qcx/backend/backend.hpp"
#include "qcx/backend/concepts.hpp"
#include "qcx/backend/cpu_topology.hpp"
#include "qcx/backend/tags.hpp"

#include <cstddef>

namespace qcx::backend {

/// The process-wide OpenMP team-size ceiling (the run input's thread_cap;
/// 0 = no ceiling). Internal state - write via SetOmpThreadCeiling, read
/// via DefaultOmpTeamSize. Set once at run start, before any parallel
/// work: every parallel primitive reads the value at its own start, so a
/// late write would race with in-flight regions.
inline int& OmpThreadCeiling() {
    static int ceiling = 0;

    return ceiling;
}

/// The process-wide OpenMP team size, computed once from DefaultTeamSize /
/// DetectCpuTopology and cached. All CPU parallel primitives use this value
/// via an explicit num_threads() clause rather than the OpenMP runtime
/// default, which would otherwise schedule onto every logical processor
/// (hyperthreaded siblings on uniform hardware, P + E + HT siblings on
/// hybrid parts). The E-core exclusion is exact only where the probe can
/// distinguish the classes (the POSIX sysfs-capacity path); the Windows
/// probe deliberately keeps every physical core because the OS efficiency
/// class is vendor-convention-dependent (Intel P = 0, AMD P = 1). The
/// ceiling clamps the result after the cache (min with the ceiling when
/// set).
/// \ingroup qcx-backend
/// \returns The team size (>= 1).
inline int DefaultOmpTeamSize() {
    static const int team = static_cast<int>(DefaultTeamSize(DetectCpuTopology()));

    const int ceiling = OmpThreadCeiling();
    return ceiling > 0 ? std::min(team, ceiling) : team;
}

/// Pins the OpenMP runtime's OWN thread maximum to \p team (the threads the
/// engine will actually use) with omp_set_num_threads. The runtime sizes its
/// thread pool by that maximum, NOT by the num_threads() clause a region
/// requests: with an identical 6-worker team and identical clauses, the
/// ri_jk fixture peaks at 72.4 MiB committed when the runtime maximum is the
/// 12 logical processors and 52.8 MiB when it is the 6-thread team - and
/// under a 0.06 GiB job-object cap the first configuration cannot allocate
/// its pool, retries inside the runtime and never terminates, while the
/// second completes. Pinning the maximum to the team removes only threads
/// above the team - every engine region already passes an explicit
/// num_threads() at or below it - so no region's participation changes.
/// A no-op for \p team <= 0. Call it (via SetOmpThreadCeiling) at run start,
/// before any parallel work: the maximum governs the pool the first region
/// creates, so a later call cannot shrink an already-allocated pool.
///
/// The environment variable route is NOT equivalent: OMP_NUM_THREADS is read
/// when the runtime loads, so setting it from inside the process is too late
/// (measured: _putenv_s in main leaves omp_get_max_threads() at its old
/// value), and the OpenMP API call is the only in-process control.
/// \ingroup qcx-backend
/// \param team The runtime's thread maximum; <= 0 leaves it untouched.
void BoundOmpRuntimeThreads(int team) noexcept;

/// The OpenMP runtime's own thread maximum (omp_get_max_threads()) - the
/// value BoundOmpRuntimeThreads pins, exposed so a test can assert the pin
/// rather than infer it. It is the runtime's, not the engine's: the engine's
/// team is DefaultOmpTeamSize().
/// \ingroup qcx-backend
/// \returns The runtime's current thread maximum (>= 1).
int OmpRuntimeMaxThreads() noexcept;

/// Sets the process-wide OpenMP team-size ceiling (the run input's
/// thread_cap). \p ceiling <= 0 removes the ceiling; a positive value
/// clamps every CPU parallel primitive - ParallelFor, ParallelReduce, the
/// thread pool, and the Fock-build chunk split (which reads
/// DefaultOmpTeamSize) - to at most \p ceiling threads. A ceiling of 1
/// degrades the whole run to the single-chunk serial path (numChunks <=
/// 1), whose bit-identity is pinned. The ceiling is process-wide and
/// sticky like the cached team; it applies after the cache, so setting it
/// late still clamps (a cached team is never bypassed).
///
/// It also pins the OpenMP runtime's own thread maximum to the resolved
/// team (BoundOmpRuntimeThreads), which is why the run-start call site
/// matters: this is the one point where the team is known and the runtime
/// has not yet created its pool. The pin is what keeps a run under a hard
/// memory cap that cannot hold the runtime's default pool from hanging in
/// the runtime's own allocation - the run keeps its full team and finishes
/// with the pool sized to it. On a machine whose logical processors equal
/// the team there is nothing to trim and the call changes nothing.
/// \ingroup qcx-backend
/// \param ceiling The team-size ceiling; <= 0 removes the ceiling.
inline void SetOmpThreadCeiling(int ceiling) {
    OmpThreadCeiling() = ceiling;
    BoundOmpRuntimeThreads(DefaultOmpTeamSize());
}

/// CPU backend: parallel loops over OpenMP.
///
/// The CPU has no real stream concept, so StreamType is a placeholder; the
/// std::execution analogue lives in cpu_scheduler.hpp.
/// \ingroup qcx-backend
template <> class Backend<CpuTag> {
public:
    using StreamType = int; ///< Placeholder - no real stream concept on the CPU.
    using DeviceIdType = int; ///< Device identifier (always 0 on the CPU).

    /// Id of the device this backend runs on; always 0 on the CPU.
    /// \returns The device id (0).
    DeviceIdType DeviceId() const {
        return 0;
    }

    /// Calls \p f(i) for every i in [0, \p n), parallelized with OpenMP.
    /// \tparam F Callable taking a std::size_t index.
    /// \param n Number of iterations.
    /// \param f Per-iteration callable.
    /// \param numThreads The region's thread count, or 0 (the default) for
    /// DefaultOmpTeamSize() - the bounded-concurrency seam: concurrent
    /// per-slot regions each pass their slot's share of the team, keeping the
    /// k x q thread product at or under the policy team.
    template <typename F> void ParallelFor(std::size_t n, F&& f, int numThreads = 0) const {
        const int team = numThreads > 0 ? numThreads : DefaultOmpTeamSize();

#pragma omp parallel for num_threads(team)
        // MSVC OpenMP requires a signed loop index; the public contract still
        // hands f a std::size_t.

        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        {
            f(static_cast<std::size_t>(i));
        }
    }

    /// Calls \p f(i) for every i in [0, \p n) with dynamic OpenMP scheduling.
    ///
    /// Prefer over ParallelFor when per-iteration cost varies wildly (e.g.
    /// symmetry-candidate verification); dynamic scheduling balances the load.
    /// The chunk size is one iteration - the finest grab granularity - so the
    /// caller owns the balancing by choosing \p n: oversubscribing the
    /// iteration space (many more iterations than threads) lets a single
    /// expensive iteration cost at most one thread's tail, while \p n equal to
    /// the team degenerates to one iteration per thread, the same split as the
    /// static schedule. The iterations must be independent (the same contract
    /// as ParallelFor, only the assignment order differs).
    /// \tparam F Callable taking a std::size_t index.
    /// \param n Number of iterations.
    /// \param f Per-iteration callable.
    /// \param numThreads The region's thread count, or 0 (the default) for
    /// DefaultOmpTeamSize() - the same bounded-concurrency seam as
    /// ParallelFor.
    template <typename F> void ParallelForDynamic(std::size_t n, F&& f, int numThreads = 0) const {
        const int team = numThreads > 0 ? numThreads : DefaultOmpTeamSize();

#pragma omp parallel for schedule(dynamic, 1) num_threads(team)

        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        {
            f(static_cast<std::size_t>(i));
        }
    }
};

/// Parallel blocked reduction over [0, \p n): each thread accumulates a
/// thread-local partial via \p accumulate, then the partials are merged
/// under a critical section in unspecified thread-completion order.
///
/// \p combine must be associative and \p zero its identity, because the
/// merge order across threads is unspecified. The result is deterministic
/// for associative combines only (e.g. floating-point sums may differ from
/// the serial order in the last bits - acceptable per the architecture
/// doc's cross-backend tolerance note).
///
/// The default static chunk schedule over a triangular index range (the
/// row loop of NuclearRepulsionEnergy, where iteration i does n - i - 1
/// units of work) leaves the tail threads idle - pass
/// \p dynamicSchedule = true for such irregular per-iteration loads
/// (the same shape connectivity.cpp schedules dynamically). The
/// OpenMP schedule clause is a literal, so the two schedules are two
/// parallel regions behind the flag.
/// \ingroup qcx-backend
/// \tparam T Result type; copyable and default-assignable.
/// \tparam F Callable T(const T&, std::size_t) - folds element i into a partial.
/// \tparam G Callable T(const T&, const T&) - combines two partials.
/// \param n Number of elements.
/// \param zero Identity value for \p combine.
/// \param accumulate Folds element i into a partial.
/// \param combine Combines two partials.
/// \param dynamicSchedule Use schedule(dynamic) instead of the static
/// default; prefer for triangular or otherwise irregular index ranges.
/// \param numThreads The region's thread count, or 0 (the default) for
/// DefaultOmpTeamSize() - the bounded-concurrency seam (see
/// ParallelFor).
/// \returns The combined total.
template <typename T, typename F, typename G>
T ParallelReduce(std::size_t n,
                 const T& zero,
                 F&& accumulate,
                 G&& combine,
                 bool dynamicSchedule = false,
                 int numThreads = 0) {
    T total = zero;
    const int team = numThreads > 0 ? numThreads : DefaultOmpTeamSize();

    if (dynamicSchedule)
    {
#pragma omp parallel num_threads(team)
        {
            T partial = zero;
#pragma omp for nowait schedule(dynamic)

            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
            {
                partial = accumulate(partial, static_cast<std::size_t>(i));
            }

#pragma omp critical
            total = combine(total, partial);
        }

        return total;
    }

#pragma omp parallel num_threads(team)
    {
        T partial = zero;
#pragma omp for nowait

        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        {
            partial = accumulate(partial, static_cast<std::size_t>(i));
        }

#pragma omp critical
        total = combine(total, partial);
    }

    return total;
}

static_assert(ExecutionBackend<Backend<CpuTag>>);

} // namespace qcx::backend
