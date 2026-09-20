#pragma once

#include "qcx/error.hpp"
#include "qcx/memory/allocator_traits.hpp"
#include "qcx/memory/device_ptr.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace qcx::memory {

/// Which side of the buffer holds the canonical copy.
/// \ingroup qcx-memory
enum class Residency : std::uint8_t {
    kHost, ///< Only the host copy is valid; the device side is stale.
    kDevice, ///< Only the device copy is valid; the host side is stale.
    kBothValid, ///< Both copies agree.
    kDeviceDirty, ///< The device side was written; the host copy is stale.
    kHostDirty, ///< The host side was written; the device copy is stale.
};

/// RAII buffer owning one host copy and one device copy of \p count elements.
///
/// The exclusive site of host and device allocation in qcx: everything
/// above memory in the DAG allocates through this class, never through
/// new/cudaMalloc directly. Copying is deleted; moves transfer ownership.
/// Synchronization between the two copies is explicit (SyncToHost /
/// SyncToDevice) and tracked by the residency state machine. Syncing is
/// const: it refreshes caches (the host copy and the residency marker are
/// mutable bookkeeping) without changing the logical value, so const
/// buffers - and const Tensors - can still synchronize.
/// \ingroup qcx-memory
/// \tparam T Element type (trivially copyable math types - memcpy semantics).
/// \tparam Backend Execution backend owning the device copy.
template <typename T, typename Backend> class DeviceBuffer {
public:
    /// Creates a buffer of \p count elements, allocating device memory now.
    /// \param count Number of elements; must be non-zero.
    /// \returns The buffer, or an Error (kInvalidArgument for count == 0,
    /// kDeviceError when the device allocation fails).
    static qcx::Result<DeviceBuffer> Create(std::size_t count) {
        if (count == 0)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "DeviceBuffer requires a non-zero element count"});
        }

        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "DeviceBuffer count overflows size_t"});
        }

        auto raw = AllocatorTraits<Backend>::Allocate(count * sizeof(T));

        if (!raw.has_value())
        {
            return std::unexpected(raw.error());
        }

        return DeviceBuffer{count, *raw};
    }

    /// Releases the device allocation (the host copy is released with it).
    ~DeviceBuffer() {
        if (_deviceRaw != nullptr)
        {
            AllocatorTraits<Backend>::Deallocate(_deviceRaw);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    /// Copying is deleted: buffers are movable-only, one owner per allocation.
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /// Move construction: transfers the allocation and the host copy.
    /// \param other Source buffer; left empty afterwards.
    DeviceBuffer(DeviceBuffer&& other) noexcept :
        _count(other._count), _deviceRaw(other._deviceRaw), _host(std::move(other._host)),
        _residency(other._residency) {
        other._deviceRaw = nullptr;
        other._count = 0;
        other._residency = Residency::kHost;
    }

    /// Move assignment: releases the current allocation, then transfers.
    /// \param other Source buffer; left empty afterwards.
    /// \returns This buffer.
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other)
        {
            if (_deviceRaw != nullptr)
            {
                AllocatorTraits<Backend>::Deallocate(_deviceRaw);
            }

            _count = other._count;
            _deviceRaw = other._deviceRaw;
            _host = std::move(other._host);
            _residency = other._residency;
            other._deviceRaw = nullptr;
            other._count = 0;
            other._residency = Residency::kHost;
        }

        return *this;
    }

    /// Copies the device side into the host copy when the device side is
    /// dirty; no-op otherwise.
    /// \returns An Error when the device-to-host copy fails.
    qcx::Result<void> SyncToHost() const {
        if (_residency == Residency::kDevice || _residency == Residency::kDeviceDirty)
        {
            auto copy =
                AllocatorTraits<Backend>::CopyToHost(_host.data(), _deviceRaw, _count * sizeof(T));

            if (!copy.has_value())
            {
                return std::unexpected(copy.error());
            }

            _residency = Residency::kBothValid;
        }

        return {};
    }

    /// Copies the host side into the device copy when the host side is
    /// dirty; no-op otherwise.
    /// \returns An Error when the host-to-device copy fails.
    qcx::Result<void> SyncToDevice() const {
        if (_residency == Residency::kHost || _residency == Residency::kHostDirty)
        {
            auto copy = AllocatorTraits<Backend>::CopyFromHost(
                _deviceRaw, _host.data(), _count * sizeof(T));

            if (!copy.has_value())
            {
                return std::unexpected(copy.error());
            }

            _residency = Residency::kBothValid;
        }

        return {};
    }

    /// The host copy.
    ///
    /// Precondition: the host side is canonical - call SyncToHost() after
    /// any device-side writes (see MarkDeviceDirty) before touching it.
    /// \returns A mutable reference to the host data.
    std::vector<T>& HostView() noexcept {
        return _host;
    }

    /// The host copy, read-only.
    /// \pre The host side is canonical (see SyncToHost).
    /// \returns A const reference to the host data.
    const std::vector<T>& HostView() const noexcept {
        return _host;
    }

    /// An opaque handle to the device allocation.
    ///
    /// The handle always points at the device memory; whether its contents
    /// are current depends on the residency state - call SyncToDevice()
    /// after any host-side writes (see MarkHostDirty). Named DeviceHandle
    /// rather than DevicePtr: a member named after the DevicePtr type
    /// changes the type's meaning in class scope (GCC -Wchanges-meaning).
    /// \returns The device handle.
    DevicePtr<T, Backend> DeviceHandle() noexcept {
        return qcx::memory::DevicePtr<T, Backend>{_deviceRaw};
    }

    /// Declares the device copy dirty (a kernel wrote through the handle).
    void MarkDeviceDirty() noexcept {
        _residency = Residency::kDeviceDirty;
    }

    /// Declares the host copy dirty (the host vector was modified).
    void MarkHostDirty() noexcept {
        _residency = Residency::kHostDirty;
    }

    /// Number of elements in the buffer.
    /// \returns The element count.
    std::size_t Size() const noexcept {
        return _count;
    }

    /// Current residency state of the two copies.
    /// \returns The residency marker.
    Residency CurrentResidency() const noexcept {
        return _residency;
    }

private:
    /// Constructs from a fresh device allocation; the host copy starts canonical.
    /// \param count Number of elements.
    /// \param deviceRaw Fresh device allocation owned from here on.
    explicit DeviceBuffer(std::size_t count, void* deviceRaw) :
        _count(count), _deviceRaw(deviceRaw), _host(count), _residency(Residency::kHost) {}

    std::size_t _count;
    void* _deviceRaw;
    // mutable: refreshed by the const syncs - the host copy and the
    // residency marker are caches of the logical value, not part of it.
    mutable std::vector<T> _host;
    mutable Residency _residency;
};

} // namespace qcx::memory
