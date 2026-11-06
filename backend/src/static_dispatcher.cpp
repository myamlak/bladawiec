// The static work partition (static_dispatcher.hpp): the arithmetic rule in one
// place, and the thin class that binds it to a rank/size pair. No collective is
// called here - the refusal path must produce the same message on every rank
// before any communication.

#include "qcx/backend/static_dispatcher.hpp"

#include <string>

namespace qcx::backend {

// (count, rank, size) is the published reading of the partition; the two
// rank numbers are validated against each other below.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
qcx::Result<IndexRange> PartitionRange(std::size_t count, int rank, int size) {
    if (size <= 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "StaticDispatcher: the rank count must be positive (got " +
                           std::to_string(size) + ")"});
    }

    if (rank < 0 || rank >= size)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "StaticDispatcher: rank " + std::to_string(rank) +
                                              " is outside [0, " + std::to_string(size) + ")"});
    }

    if (count < static_cast<std::size_t>(size))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "StaticDispatcher: " + std::to_string(count) +
                                              " items cannot fill " + std::to_string(size) +
                                              " ranks; every rank must receive at least one item"});
    }

    const std::size_t rankCount = static_cast<std::size_t>(size);
    const std::size_t base = count / rankCount;
    const std::size_t remainder = count % rankCount;
    const std::size_t begin = static_cast<std::size_t>(rank) * base;
    const std::size_t end =
        static_cast<std::size_t>(rank + 1) * base + (rank == size - 1 ? remainder : 0);

    return IndexRange{begin, end};
}

// The pair is the published (rank, size) reading, validated by PartitionRange.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
StaticDispatcher::StaticDispatcher(int rank, int size) noexcept : _rank(rank), _size(size) {}

int StaticDispatcher::Rank() const noexcept {
    return _rank;
}

int StaticDispatcher::Size() const noexcept {
    return _size;
}

qcx::Result<IndexRange> StaticDispatcher::Partition(std::size_t count) const {
    return PartitionRange(count, _rank, _size);
}

} // namespace qcx::backend
