#pragma once

#include "qcx/backend/tags.hpp"
#include "qcx/error.hpp"
#include "qcx/memory/device_buffer.hpp"
#include "qcx/memory/tensor_policies.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace qcx::memory {

/// Multi-dimensional dense array backed by a DeviceBuffer.
///
/// The workhorse storage type: the shape is row-major, element access
/// compiles only for the CPU backend (device-resident tensors migrate
/// through ToDevice/ToHost/WithDevice), and synchronization and error
/// handling follow the DeviceBuffer rules.
/// \ingroup qcx-memory
/// \tparam T Element type (trivially copyable math types).
/// \tparam Rank Number of dimensions.
/// \tparam Backend Execution backend owning the device copy.
/// \tparam SymmetryPolicy Symmetry tag (NoSymmetry today).
/// \tparam SparsityPolicy Sparsity tag (Dense today).
template <typename T,
          std::size_t Rank,
          typename Backend,
          typename SymmetryPolicy = NoSymmetry,
          typename SparsityPolicy = Dense>
class Tensor {
public:
    using ShapeType = std::array<std::size_t, Rank>; ///< Per-dimension extent array.
    using BackendType = Backend; ///< The backend owning the device copy.

    /// Creates a tensor with the given shape, allocating storage now.
    /// \param shape Per-dimension extents; every dimension must be non-zero.
    /// \returns The tensor, or an Error (kInvalidArgument when any extent is
    /// zero or the element count overflows size_t, kDeviceError when the
    /// device allocation fails).
    static qcx::Result<Tensor> Create(const ShapeType& shape) {
        auto count = ElementCount(shape);

        if (!count.has_value())
        {
            return std::unexpected(count.error());
        }

        auto buffer = DeviceBuffer<T, Backend>::Create(*count);

        if (!buffer.has_value())
        {
            return std::unexpected(buffer.error());
        }

        return Tensor{shape, std::move(*buffer)};
    }

    /// Deep-copies this tensor (const: the source is only synchronized).
    ///
    /// Synchronizes the host copy first, then allocates a fresh buffer and
    /// copies the contents; the clone starts host-canonical with the same
    /// logical values as the source.
    /// \returns The copy, or an Error when synchronization or allocation fails.
    qcx::Result<Tensor> Clone() const {
        auto sync = _buffer.SyncToHost();

        if (!sync.has_value())
        {
            return std::unexpected(sync.error());
        }

        auto copy = DeviceBuffer<T, Backend>::Create(_buffer.Size());

        if (!copy.has_value())
        {
            return std::unexpected(copy.error());
        }

        copy->HostView() = _buffer.HostView();
        return Tensor{_shape, std::move(*copy)};
    }

    /// Number of dimensions of this tensor type.
    /// \returns Rank (the template parameter).
    static constexpr std::size_t TensorRank() {
        return Rank;
    }

    /// Per-dimension extents.
    /// \returns The shape array.
    const ShapeType& Shape() const noexcept {
        return _shape;
    }

    /// Total number of elements (product of the extents).
    /// \returns The element count.
    std::size_t Size() const noexcept {
        return _buffer.Size();
    }

    /// Element access, row-major. Only compiles for the CPU backend.
    /// \param indices Rank indices, one per dimension.
    /// \returns A mutable reference to the element at \p indices.
    /// \pre The host copy is canonical (see ToHost).
    T& operator()(auto... indices)
        requires std::same_as<Backend, qcx::backend::CpuTag>
    {
        static_assert(sizeof...(indices) == Rank, "index count must match tensor rank");
        ShapeType idx{static_cast<std::size_t>(indices)...};
        return _buffer.HostView()[Flatten(idx)];
    }

    /// Element access, row-major, read-only. Only compiles for the CPU backend.
    /// \param indices Rank indices, one per dimension.
    /// \returns A const reference to the element at \p indices.
    /// \pre The host copy is canonical (see ToHost).
    const T& operator()(auto... indices) const
        requires std::same_as<Backend, qcx::backend::CpuTag>
    {
        static_assert(sizeof...(indices) == Rank, "index count must match tensor rank");
        ShapeType idx{static_cast<std::size_t>(indices)...};
        return _buffer.HostView()[Flatten(idx)];
    }

    /// Synchronizes the device copy from the host and returns its handle.
    /// \returns The device handle, or an Error when the copy fails.
    qcx::Result<DevicePtr<T, Backend>> ToDevice() {
        auto sync = _buffer.SyncToDevice();

        if (!sync.has_value())
        {
            return std::unexpected(sync.error());
        }

        return _buffer.DeviceHandle();
    }

    /// Synchronizes the host copy from the device.
    /// \returns An Error when the copy fails.
    qcx::Result<void> ToHost() {
        return _buffer.SyncToHost();
    }

    /// The host copy. Precondition: the host side is canonical (see ToHost).
    /// \returns A mutable reference to the host data.
    std::vector<T>& HostView() noexcept {
        return _buffer.HostView();
    }

    /// Declares the device copy dirty (a kernel wrote through the handle).
    void MarkDeviceDirty() noexcept {
        _buffer.MarkDeviceDirty();
    }

    /// Declares the host copy dirty (the host vector was modified).
    void MarkHostDirty() noexcept {
        _buffer.MarkHostDirty();
    }

    /// Current residency state of the two copies.
    /// \returns The residency marker.
    Residency CurrentResidency() const noexcept {
        return _buffer.CurrentResidency();
    }

private:
    /// Constructs from a fresh buffer; the host copy starts canonical.
    explicit Tensor(ShapeType shape, DeviceBuffer<T, Backend>&& buffer) :
        _shape(shape), _buffer(std::move(buffer)) {}

    /// Validates the shape and computes the total element count.
    /// \returns The product of the extents, or an Error (kInvalidArgument for
    /// a zero extent or a product that overflows size_t).
    static qcx::Result<std::size_t> ElementCount(const ShapeType& s) {
        std::size_t n = 1;

        for (std::size_t d : s)
        {
            if (d == 0)
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                                  "Tensor shape requires non-zero extents"});
            }

            if (n > std::numeric_limits<std::size_t>::max() / d)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument, "Tensor count overflows size_t"});
            }

            n *= d;
        }

        return n;
    }

    std::size_t Flatten(const ShapeType& idx) const {
        std::size_t offset = 0;
        std::size_t stride = 1;

        for (std::size_t d = Rank; d-- > 0;)
        {
            offset += idx[d] * stride;
            stride *= _shape[d];
        }

        return offset;
    }

    ShapeType _shape;
    DeviceBuffer<T, Backend> _buffer;
};

/// Runs \p f on the tensor's device copy, migrating lazily.
///
/// Synchronizes host-to-device on entry and marks the device copy dirty on
/// exit; \p Backend must match the tensor's own backend type.
/// \ingroup qcx-memory
/// \tparam Backend Backend the tensor is migrated to.
/// \tparam TensorT Tensor type (deduced).
/// \tparam F Callable taking a DevicePtr handle.
/// \param tensor Tensor to migrate.
/// \param f Work to run on the device copy.
/// \returns An Error when the host-to-device copy fails.
template <typename Backend, typename TensorT, typename F>
qcx::Result<void> WithDevice(TensorT& tensor, F&& f) {
    static_assert(std::same_as<Backend, typename TensorT::BackendType>,
                  "WithDevice<Backend> must match the tensor's own backend type");
    auto dptr = tensor.ToDevice();

    if (!dptr.has_value())
    {
        return std::unexpected(dptr.error());
    }

    std::forward<F>(f)(*dptr);
    tensor.MarkDeviceDirty();
    return {};
}

} // namespace qcx::memory
