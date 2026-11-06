#pragma once

/// \file
/// The process execution context: the communicator every collective and every
/// work distribution runs over.
///
/// A build with MPI support (QCX_ENABLE_MPI) owns the MPI world; a build without
/// it compiles the same interface as a serial stub (rank 0, size 1, a no-op
/// barrier and an all-reduce that copies its input), so a consumer is written
/// once against both. The public header never includes <mpi.h>: the MPI types
/// live behind the implementation pointer.
///
/// The context is a process-wide singleton created by Create(). MPI is
/// initialized with MPI_THREAD_FUNNELED on the first call and finalized once,
/// when the process's context is destroyed at exit; only the thread that called
/// Create may call the collective operations.

#include "qcx/error.hpp"

#include <memory>

namespace qcx::backend {

/// The process's execution context.
///
/// One instance per process: Create() returns the same object on every call.
/// Rank, size, barrier and the integer/double sum all-reduce are the whole
/// interface; a work distribution builds on them (StaticDispatcher).
/// \ingroup qcx-backend
class ExecutionContext {
public:
    /// Creates the process's context, or returns the existing one.
    ///
    /// The first call initializes MPI with MPI_THREAD_FUNNELED when the build
    /// has MPI; later calls return the same instance. A failure is permanent:
    /// the MPI runtime cannot be re-initialized after a failed or completed
    /// initialization, so every later call reports the first failure.
    /// \returns The process's context, or an Error (kInternalError) when the
    /// MPI runtime fails to initialize or provides a threading level below
    /// MPI_THREAD_FUNNELED.
    static qcx::Result<std::shared_ptr<ExecutionContext>> Create();

    /// This process's rank in the context's communicator.
    /// \returns The rank, in [0, Size()).
    int Rank() const noexcept;

    /// The number of ranks in the context's communicator.
    /// \returns The rank count (1 in a serial build).
    int Size() const noexcept;

    /// Blocks until every rank reaches the barrier.
    /// \returns An Error (kInternalError) when the MPI call fails.
    qcx::Result<void> Barrier() const;

    /// Sums one integer across every rank.
    /// \param value This rank's contribution.
    /// \returns The sum over all ranks (a copy of \p value in a serial build),
    /// or an Error (kInternalError) when the MPI call fails.
    qcx::Result<int> AllReduceSum(int value) const;

    /// Sums one double across every rank.
    /// \param value This rank's contribution.
    /// \returns The sum over all ranks (a copy of \p value in a serial build),
    /// or an Error (kInternalError) when the MPI call fails.
    qcx::Result<double> AllReduceSum(double value) const;

    ExecutionContext(const ExecutionContext&) = delete;
    ExecutionContext& operator=(const ExecutionContext&) = delete;
    ExecutionContext(ExecutionContext&&) = delete;
    ExecutionContext& operator=(ExecutionContext&&) = delete;

    /// Destroys the context. The process's MPI runtime is finalized here when
    /// the build has MPI; the context lives until process exit, so this runs
    /// once.
    ~ExecutionContext();

private:
    struct Impl;

    explicit ExecutionContext(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> _impl;
};

} // namespace qcx::backend
