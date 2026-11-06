// The execution context (execution_context.hpp): the MPI world behind a
// header that never includes <mpi.h>, and the serial stub the non-MPI build
// compiles instead. Every MPI call in the project's context lives in this
// translation unit.

#include "qcx/backend/execution_context.hpp"

#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#if defined(QcxHasMpi)
#include <mpi.h>
#endif

namespace qcx::backend {
namespace {

#if defined(QcxHasMpi)

// The MPI runtime's own text for a failure, so the error names what went
// wrong rather than only the numeric code.
qcx::Error MpiFailure(const char* operation, int code) {
    std::string message(operation);
    message += " failed";

    char text[MPI_MAX_ERROR_STRING] = {};
    int length = 0;

    if (MPI_Error_string(code, text, &length) == MPI_SUCCESS)
    {
        message += ": ";
        message += text;
    }

    return qcx::Error{qcx::ErrorCode::kInternalError, std::move(message)};
}

#endif

// The process's single context. Function-local so the mutex and the context
// are constructed on first use; the context member is declared last so it is
// destroyed first, before the mutex that guarded it.
struct Registry {
    std::mutex mutex;
    std::optional<qcx::Error> failure;
    std::shared_ptr<ExecutionContext> context;
};

Registry& ContextRegistry() {
    static Registry registry;
    return registry;
}

} // namespace

struct ExecutionContext::Impl {
    int rank = 0;
    int size = 1;

#if defined(QcxHasMpi)
    MPI_Comm communicator = MPI_COMM_NULL;
#endif
};

ExecutionContext::ExecutionContext(std::unique_ptr<Impl> impl) noexcept : _impl(std::move(impl)) {}

ExecutionContext::~ExecutionContext() {
#if defined(QcxHasMpi)
    if (_impl->communicator != MPI_COMM_NULL)
    {
        MPI_Finalize();
    }
#endif
}

qcx::Result<std::shared_ptr<ExecutionContext>> ExecutionContext::Create() {
    Registry& registry = ContextRegistry();
    const std::lock_guard<std::mutex> lock(registry.mutex);

    if (registry.failure)
    {
        return std::unexpected(*registry.failure);
    }

    if (registry.context)
    {
        return registry.context;
    }

    auto impl = std::make_unique<Impl>();

#if defined(QcxHasMpi)
    int provided = 0;

    if (const int code = MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
        code != MPI_SUCCESS)
    {
        registry.failure = MpiFailure("MPI_Init_thread", code);
        return std::unexpected(*registry.failure);
    }

    if (provided < MPI_THREAD_FUNNELED)
    {
        MPI_Finalize();
        registry.failure =
            qcx::Error{qcx::ErrorCode::kInternalError,
                       "MPI_Init_thread provided threading level " + std::to_string(provided) +
                           ", below the requested MPI_THREAD_FUNNELED"};
        return std::unexpected(*registry.failure);
    }

    int rank = 0;

    if (const int code = MPI_Comm_rank(MPI_COMM_WORLD, &rank); code != MPI_SUCCESS)
    {
        registry.failure = MpiFailure("MPI_Comm_rank", code);
        MPI_Finalize();
        return std::unexpected(*registry.failure);
    }

    int size = 0;

    if (const int code = MPI_Comm_size(MPI_COMM_WORLD, &size); code != MPI_SUCCESS)
    {
        registry.failure = MpiFailure("MPI_Comm_size", code);
        MPI_Finalize();
        return std::unexpected(*registry.failure);
    }

    impl->rank = rank;
    impl->size = size;
    impl->communicator = MPI_COMM_WORLD;
#endif

    registry.context = std::shared_ptr<ExecutionContext>(new ExecutionContext(std::move(impl)));
    return registry.context;
}

int ExecutionContext::Rank() const noexcept {
    return _impl->rank;
}

int ExecutionContext::Size() const noexcept {
    return _impl->size;
}

qcx::Result<void> ExecutionContext::Barrier() const {
#if defined(QcxHasMpi)
    if (const int code = MPI_Barrier(_impl->communicator); code != MPI_SUCCESS)
    {
        return std::unexpected(MpiFailure("MPI_Barrier", code));
    }
#endif

    return {};
}

qcx::Result<int> ExecutionContext::AllReduceSum(int value) const {
#if defined(QcxHasMpi)
    int sum = 0;

    if (const int code = MPI_Allreduce(&value, &sum, 1, MPI_INT, MPI_SUM, _impl->communicator);
        code != MPI_SUCCESS)
    {
        return std::unexpected(MpiFailure("MPI_Allreduce(MPI_INT)", code));
    }

    return sum;
#else
    return value;
#endif
}

qcx::Result<double> ExecutionContext::AllReduceSum(double value) const {
#if defined(QcxHasMpi)
    double sum = 0.0;

    if (const int code = MPI_Allreduce(&value, &sum, 1, MPI_DOUBLE, MPI_SUM, _impl->communicator);
        code != MPI_SUCCESS)
    {
        return std::unexpected(MpiFailure("MPI_Allreduce(MPI_DOUBLE)", code));
    }

    return sum;
#else
    return value;
#endif
}

qcx::Result<void> ExecutionContext::AllReduceSumInPlace(std::span<double> values) const {
#if defined(QcxHasMpi)
    // The MPI count is an int; refuse a buffer one call cannot carry instead
    // of truncating the count and summing a prefix.
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "AllReduceSumInPlace: the buffer holds " +
                                              std::to_string(values.size()) +
                                              " elements, more than one MPI call can carry"});
    }

    if (!values.empty())
    {
        const int count = static_cast<int>(values.size());

        if (const int code = MPI_Allreduce(
                MPI_IN_PLACE, values.data(), count, MPI_DOUBLE, MPI_SUM, _impl->communicator);
            code != MPI_SUCCESS)
        {
            return std::unexpected(MpiFailure("MPI_Allreduce(MPI_DOUBLE, dense)", code));
        }
    }
#else
    (void)values;
#endif

    return {};
}

} // namespace qcx::backend
