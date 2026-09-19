#pragma once
#include "qcx/backend/cpu_backend.hpp"

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#include <utility>

namespace qcx::backend {

/// The process-wide CPU thread pool backing std::execution-style scheduling.
///
/// A single static instance sized to the same team-size policy as the OpenMP
/// primitives (DefaultOmpTeamSize, cpu_backend.hpp) - not to every logical
/// processor; the scheduler it returns is the stdexec counterpart of the
/// OpenMP ParallelFor.
/// \ingroup qcx-backend
/// \returns The pool (single process-wide instance).
inline exec::static_thread_pool& CpuThreadPool() {
    // The same process-wide team-size policy as the OpenMP primitives
    // (DefaultOmpTeamSize, cpu_backend.hpp) - hardware_concurrency() alone
    // would size the pool to every logical processor (P + E + HT siblings)
    // on hybrid hardware.
    static exec::static_thread_pool pool(static_cast<std::size_t>(DefaultOmpTeamSize()));
    return pool;
}

/// Schedules \p f once on the CPU thread pool and blocks until it completes.
/// \ingroup qcx-backend
/// \tparam F Callable taking no arguments.
/// \param f Work to run on a pool thread.
template <typename F> void ScheduleAndRun(F&& f) {
    auto sched = CpuThreadPool().get_scheduler();
    stdexec::sync_wait(stdexec::schedule(sched) | stdexec::then(std::forward<F>(f)));
}

} // namespace qcx::backend
