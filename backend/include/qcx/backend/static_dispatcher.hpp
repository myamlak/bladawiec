#pragma once

/// \file
/// The static work partition: one contiguous block of items per rank, and
/// nothing else.
///
/// The partition is a pure function of the item count and the rank/size pair,
/// so every rank computes the same bounds without a collective. There is no
/// load balancing by design: the split is fixed arithmetic, and a caller that
/// wants a different one asks for a different count.
///
/// Every rank must receive at least one item. When the item count is below the
/// rank count, PartitionRange returns kInvalidArgument with a message computed
/// from the count and the rank count alone - the same on every rank - so a
/// caller can refuse before the first collective.

#include "qcx/error.hpp"

#include <cstddef>

namespace qcx::backend {

/// A half-open item range [begin, end).
/// \ingroup qcx-backend
struct IndexRange {
    std::size_t begin = 0; ///< Index of the first item of this rank's block.
    std::size_t end = 0; ///< One past the last item of this rank's block.
};

/// Partitions \p count contiguous items over \p size ranks.
///
/// Each rank takes base = count / size items; the last rank additionally takes
/// the remainder = count % size, so the blocks are contiguous, cover every item
/// exactly once and end at \p count.
/// \param count The number of items to partition.
/// \param rank This rank's index, in [0, size).
/// \param size The number of ranks.
/// \returns This rank's [begin, end) block, or an Error (kInvalidArgument) when
/// \p size is not positive, \p rank is outside [0, size), or \p count is below
/// \p size (at least one rank would receive no work).
/// \ingroup qcx-backend
qcx::Result<IndexRange> PartitionRange(std::size_t count, int rank, int size);

/// A partitioner bound to one rank/size pair.
///
/// The pair comes from the execution context or from a test; the class makes no
/// MPI call and holds no state beyond the two numbers.
/// \ingroup qcx-backend
class StaticDispatcher {
public:
    /// Binds the partition to a rank and a rank count.
    /// \param rank This rank's index.
    /// \param size The number of ranks.
    StaticDispatcher(int rank, int size) noexcept;

    /// This dispatcher's rank.
    /// \returns The rank it was constructed with.
    int Rank() const noexcept;

    /// This dispatcher's rank count.
    /// \returns The rank count it was constructed with.
    int Size() const noexcept;

    /// Partitions \p count items with the bound rank and rank count; the rule
    /// and the refusals are PartitionRange's.
    /// \param count The number of items to partition.
    /// \returns This rank's [begin, end) block, or an Error (kInvalidArgument)
    /// when the rank pair is invalid or a rank would receive no work.
    qcx::Result<IndexRange> Partition(std::size_t count) const;

private:
    int _rank;
    int _size;
};

} // namespace qcx::backend
