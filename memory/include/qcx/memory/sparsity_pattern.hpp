#pragma once

#include "qcx/error.hpp"
#include "qcx/memory/device_buffer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace qcx::memory {

/// Sparse index structure in CSR form, resident on \p Backend.
///
/// The single "index arrays in a DeviceBuffer" primitive: row i holds the
/// entries indices[rowOffsets[i] .. rowOffsets[i+1]).
/// One type serves every sparse consumer - LinK-style shell neighbor lists,
/// QFMM octree child lists, and domain/pair lists - so no sparse scheme
/// ever breaks the no-naked-pointers device abstraction. The pattern
/// is structural only: the numerical payload lives alongside it.
/// \ingroup qcx-memory
/// \tparam Backend Execution backend owning the buffers (CpuTag/CudaTag).
template <typename Backend> class SparsityPattern {
public:
    /// Creates a pattern from validated CSR arrays.
    /// \param rowOffsets CSR offsets, size rowCount + 1, offsets[0] == 0,
    /// monotonic, offsets.back() == indices.size(). Must hold at least one
    /// row.
    /// \param indices Packed indices, size rowOffsets.back(); may be empty
    /// (an all-empty pattern, e.g. a tree whose nodes are all leaves). Every
    /// entry must name an existing row (indices[k] < rowCount): consumers
    /// index payload arrays by these entries, so an out-of-range index would
    /// silently read or write out of bounds.
    /// \returns The pattern, or an Error (kInvalidArgument when the CSR
    /// invariants fail; allocation errors propagate from DeviceBuffer).
    static qcx::Result<SparsityPattern> Create(const std::vector<std::size_t>& rowOffsets,
                                               const std::vector<std::size_t>& indices) {
        if (rowOffsets.size() < 2)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "a sparsity pattern needs at least one row"});
        }

        if (rowOffsets.front() != 0 || rowOffsets.back() != indices.size() ||
            !std::is_sorted(rowOffsets.begin(), rowOffsets.end()))
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "CSR offsets must start at 0, end at the packed "
                                              "size, and be monotonic"});
        }

        const auto rowCount = rowOffsets.size() - 1;
        const auto outOfRange =
            std::find_if(indices.begin(), indices.end(), [rowCount](std::size_t index) {
                return index >= rowCount;
            });

        if (outOfRange != indices.end())
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "packed index " + std::to_string(*outOfRange) +
                                                  " is out of range for a " +
                                                  std::to_string(rowCount) + "-row pattern"});
        }

        auto offsetsBuffer = DeviceBuffer<std::size_t, Backend>::Create(rowOffsets.size());

        if (!offsetsBuffer.has_value())
        {
            return std::unexpected(offsetsBuffer.error());
        }

        // DeviceBuffer rejects zero-element allocations, so an all-empty
        // pattern pads the indices buffer with one unused slot; IndexCount()
        // reports the logical packed size and consumers never read the pad.
        auto indicesBuffer =
            DeviceBuffer<std::size_t, Backend>::Create(std::max<std::size_t>(1, indices.size()));

        if (!indicesBuffer.has_value())
        {
            return std::unexpected(indicesBuffer.error());
        }

        offsetsBuffer->HostView() = rowOffsets;

        if (!indices.empty())
        {
            indicesBuffer->HostView() = indices;
        }

        return SparsityPattern{
            std::move(*offsetsBuffer), std::move(*indicesBuffer), indices.size()};
    }

    SparsityPattern(const SparsityPattern&) = delete;
    SparsityPattern& operator=(const SparsityPattern&) = delete;
    /// Move construction: transfers both buffers (move-only type).
    SparsityPattern(SparsityPattern&&) noexcept = default;
    /// Move assignment: transfers both buffers (move-only type).
    /// \returns This pattern.
    SparsityPattern& operator=(SparsityPattern&&) noexcept = default;

    /// Number of rows (shells / octree nodes / atoms).
    /// \returns rowOffsets.size() - 1.
    std::size_t RowCount() const noexcept {
        return _rowOffsets.Size() - 1;
    }

    /// Logical number of packed indices. May be zero even though the backing
    /// buffer holds one padding slot (see Create).
    /// \returns The packed size.
    std::size_t IndexCount() const noexcept {
        return _indexCount;
    }

    /// The CSR row offsets.
    /// \returns The offsets buffer.
    DeviceBuffer<std::size_t, Backend>& RowOffsets() noexcept {
        return _rowOffsets;
    }

    /// The CSR row offsets, read-only.
    /// \returns The offsets buffer.
    const DeviceBuffer<std::size_t, Backend>& RowOffsets() const noexcept {
        return _rowOffsets;
    }

    /// The packed indices.
    /// \returns The indices buffer.
    DeviceBuffer<std::size_t, Backend>& Indices() noexcept {
        return _indices;
    }

    /// The packed indices, read-only.
    /// \returns The indices buffer.
    const DeviceBuffer<std::size_t, Backend>& Indices() const noexcept {
        return _indices;
    }

    /// Contracts the pattern: folds \p acc over every (row, entry) pair in
    /// CSR order. One primitive serves every consumer - LinK-style shell
    /// neighbor lists and QFMM octree lists both materialize as this same
    /// CSR shape, so a single access-pattern definition replaces per-shape
    /// specializations of the same loop. The packed
    /// position is passed alongside the entry so payload arrays aligned
    /// with the pattern (one value per packed index) can be looked up
    /// directly.
    ///
    /// The pattern must be host-canonical (as Create and the builders
    /// produce it): the fold reads the host copies of the buffers, so a
    /// pattern transferred device-side must be downloaded first.
    /// \tparam T The accumulator type.
    /// \tparam F fold(acc, row, packedIndex, entry) -> acc.
    /// \param acc The initial accumulator (e.g. 0.0, or an all-zeros matrix).
    /// \param fold The per-pair fold, called once per covered pair.
    /// \returns The folded accumulator.
    template <typename T, typename F> T Contract(T acc, F fold) const {
        const auto& offsets = _rowOffsets.HostView();
        const auto& indices = _indices.HostView();

        for (std::size_t row = 0; row < RowCount(); ++row)
        {
            for (std::size_t k = offsets[row]; k < offsets[row + 1]; ++k)
            {
                const auto entry = indices[k];
                acc = fold(std::move(acc), row, k, entry);
            }
        }

        return acc;
    }

private:
    /// Constructs from freshly allocated buffers (host copies canonical).
    SparsityPattern(DeviceBuffer<std::size_t, Backend> rowOffsets,
                    DeviceBuffer<std::size_t, Backend> indices,
                    std::size_t indexCount) :
        _rowOffsets(std::move(rowOffsets)), _indices(std::move(indices)), _indexCount(indexCount) {}

    DeviceBuffer<std::size_t, Backend> _rowOffsets;
    DeviceBuffer<std::size_t, Backend> _indices;
    std::size_t _indexCount;
};

/// Builds a pattern from a CSR neighbor list - the LinK-style shell
/// neighbor-list shape (molecule's ConnectivityCsr feeds this
/// builder directly).
/// \ingroup qcx-memory
/// \tparam Backend Execution backend owning the buffers.
/// \param rowOffsets CSR offsets (see SparsityPattern::Create).
/// \param neighbors Packed neighbor indices.
/// \returns The pattern, or an Error.
template <typename Backend>
qcx::Result<SparsityPattern<Backend>> BuildSparsityFromAdjacency(
    const std::vector<std::size_t>& rowOffsets, const std::vector<std::size_t>& neighbors) {
    return SparsityPattern<Backend>::Create(rowOffsets, neighbors);
}

/// Builds a pattern from a QFMM-style octree given as per-node child lists.
///
/// Validates forest structure: every node has at most one parent, no node is
/// its own child, and the graph has no cycle (each node reaches exactly one
/// root, checked by traversal - not by counting edges).
/// \ingroup qcx-memory
/// \tparam Backend Execution backend owning the buffers.
/// \param children children[i] = the child node indices of node i.
/// \returns The pattern, or an Error (kInvalidArgument when the lists do
/// not form a forest).
template <typename Backend>
qcx::Result<SparsityPattern<Backend>> BuildSparsityFromTree(
    const std::vector<std::vector<std::size_t>>& children) {
    if (children.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "an octree needs at least one node"});
    }

    const auto nodeCount = children.size();
    // The single parent of each node; the nodeCount sentinel marks roots.
    std::vector<std::size_t> parent(nodeCount, nodeCount);
    std::size_t totalEdges = 0;

    for (std::size_t node = 0; node < nodeCount; ++node)
    {
        for (const auto child : children[node])
        {
            if (child >= nodeCount)
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                                  "tree child index is out of range"});
            }

            if (child == node)
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                                  "a tree node cannot be its own child"});
            }

            if (parent[child] != nodeCount)
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                                  "a tree node has more than one parent"});
            }

            parent[child] = node;
            totalEdges += 1;
        }
    }

    // Cycle detection by parent-chain walking. Edge counting cannot catch
    // cycles: a cycle component contributes exactly edges == nodes - roots,
    // indistinguishable from a forest. Every node has at most one parent
    // here, so following parents from each node either terminates at a root
    // or revisits a node of the current walk.
    // State: 0 = unseen, 1 = in flight on the current walk, 2 = validated.
    std::vector<std::uint8_t> state(nodeCount, 0);

    for (std::size_t node = 0; node < nodeCount; ++node)
    {
        std::size_t cursor = node;

        while (cursor != nodeCount && state[cursor] == 0)
        {
            state[cursor] = 1;
            cursor = parent[cursor];
        }

        if (cursor != nodeCount && state[cursor] == 1)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "the child lists contain a cycle"});
        }

        cursor = node;

        while (cursor != nodeCount && state[cursor] == 1)
        {
            state[cursor] = 2;
            cursor = parent[cursor];
        }
    }

    std::vector<std::size_t> offsets{0};
    std::vector<std::size_t> flatIndices;
    flatIndices.reserve(totalEdges);

    for (const auto& row : children)
    {
        flatIndices.insert(flatIndices.end(), row.begin(), row.end());
        offsets.push_back(flatIndices.size());
    }

    return SparsityPattern<Backend>::Create(offsets, flatIndices);
}

} // namespace qcx::memory
