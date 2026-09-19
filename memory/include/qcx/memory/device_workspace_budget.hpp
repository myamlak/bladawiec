#pragma once

#include "qcx/error.hpp"

#include <atomic>
#include <cstddef>
#include <utility>

namespace qcx::memory {

/// Pure accounting device workspace budget: the device-side mirror of
/// WorkspaceBudget - a fixed byte capacity against which GPU builders reserve
/// their device footprint.
///
/// The capacity is the free-VRAM probe minus the display headroom, the
/// design's "free-VRAM probe (minus display headroom) at Create": the
/// caller probes the device (backend::DetectDevices or a direct
/// cudaMemGetInfo) and passes the snapshot in. The probe is a point-in-time
/// value - "hint, not guarantee" (the DeviceProfile caveat) - and the
/// headroom keeps the display's own working set out of the budget's claim.
/// No allocation and no device knowledge beyond the snapshot: the budget is
/// pure accounting, the deterministic input to per-builder device mode
/// selection.
///
/// Retention-aware like the host seam: Release() does not restore capacity,
/// because device allocations are retained by the context until they are
/// freed (the driver's mapped region does not shrink eagerly), so the
/// committed counter is monotone non-decreasing for the budget's lifetime
/// and Remaining() never re-opens consumed capacity. Only Reserve moves the
/// counter, and only when the full amount fits. Thread-safe: concurrent
/// reservations charge exactly once each. Not copyable - the counter member
/// is a std::atomic - and the seam shares one budget by pointer, never by
/// value.
/// \ingroup qcx-memory
class DeviceWorkspaceBudget {
public:
    /// Creates a budget for one device from the free-VRAM probe minus the
    /// display headroom; the capacity is freeBytes - headroomBytes.
    /// \param deviceId The CUDA device ordinal (diagnostics; the budget
    /// never touches the device).
    /// \param freeBytes The probed free VRAM (a point-in-time snapshot).
    /// \param headroomBytes The display working set kept out of the claim.
    /// \returns The budget, or an Error (kInvalidArgument when the probe
    /// minus the headroom is zero - the device has no usable budget).
    static qcx::Result<DeviceWorkspaceBudget> Create(int deviceId,
                                                     std::size_t freeBytes,
                                                     std::size_t headroomBytes);

    /// Reserves \p bytes against the cumulative commit.
    ///
    /// On success the bytes are charged permanently; a later Release does
    /// not give them back. A zero reservation always succeeds and charges
    /// nothing.
    /// \param bytes Number of bytes to reserve.
    /// \returns true when the full amount fits and is charged; false when
    /// the reservation would exceed the capacity, leaving state unchanged.
    bool Reserve(std::size_t bytes) noexcept;

    /// Marks \p bytes as released, for diagnostics only.
    ///
    /// The cumulative counter is intentionally unchanged: released device
    /// memory is not returned to the budget, so a later Reserve still
    /// charges the context's committed-so-far. Provided so RAII reservation
    /// scopes and audit trails can mirror their Reserve calls; it cannot
    /// fail and never re-opens capacity.
    /// \param bytes Number of bytes being released; has no effect.
    void Release(std::size_t bytes) noexcept;

    /// Bytes not yet committed.
    /// \returns The capacity minus the cumulative commit; monotone
    /// non-increasing for the budget's lifetime.
    std::size_t Remaining() const noexcept;

    /// The fixed capacity the budget was created with (freeBytes minus
    /// headroomBytes).
    /// \returns The capacity in bytes.
    std::size_t CapacityBytes() const noexcept;

    /// The cumulative high-water commit.
    /// \returns The bytes charged by successful Reservations so far.
    std::size_t CommittedBytes() const noexcept;

    /// The device the budget was created for.
    /// \returns The CUDA device ordinal passed to Create.
    int DeviceId() const noexcept;

    /// The probe the capacity was derived from.
    /// \returns The free-VRAM snapshot passed to Create.
    std::size_t FreeBytesAtCreate() const noexcept;

    /// The display headroom kept out of the capacity.
    /// \returns The headroom passed to Create.
    std::size_t HeadroomBytes() const noexcept;

    /// Move construction: copies the committed value.
    ///
    /// There is nothing to transfer from a counter, so the source keeps its
    /// state. Present for the Result-creating house pattern (Create returns
    /// by value); budgets are shared by pointer, so moves are never
    /// part of the seam.
    /// \param other The budget whose values are copied.
    DeviceWorkspaceBudget(DeviceWorkspaceBudget&& other) noexcept :
        _deviceId(other._deviceId), _capacityBytes(other._capacityBytes),
        _freeBytesAtCreate(other._freeBytesAtCreate), _headroomBytes(other._headroomBytes),
        _committedBytes(other._committedBytes.load(std::memory_order_relaxed)) {}

    DeviceWorkspaceBudget(const DeviceWorkspaceBudget&) = delete;
    /// Copying is deleted: budgets are shared by pointer, never by value.
    DeviceWorkspaceBudget& operator=(const DeviceWorkspaceBudget&) = delete;
    /// Move assignment is deleted like copying: budgets are shared by pointer, never by value.
    DeviceWorkspaceBudget& operator=(DeviceWorkspaceBudget&&) = delete;

private:
    DeviceWorkspaceBudget(int deviceId,
                          std::size_t capacityBytes,
                          std::size_t freeBytes,
                          std::size_t headroomBytes) :
        _deviceId(deviceId), _capacityBytes(capacityBytes), _freeBytesAtCreate(freeBytes),
        _headroomBytes(headroomBytes) {}

    int _deviceId = 0;
    std::size_t _capacityBytes = 0;
    std::size_t _freeBytesAtCreate = 0;
    std::size_t _headroomBytes = 0;
    // Monotone: only Reserve writes it, and only upwards; no ordering
    // requirements with other memory, so relaxed is sufficient.
    std::atomic<std::size_t> _committedBytes{0};
};

inline qcx::Result<DeviceWorkspaceBudget> DeviceWorkspaceBudget::Create(int deviceId,
                                                                        std::size_t freeBytes,
                                                                        std::size_t headroomBytes) {
    if (freeBytes <= headroomBytes)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "device " + std::to_string(deviceId) +
                                              " has no usable workspace budget: the free-VRAM "
                                              "probe (" +
                                              std::to_string(freeBytes) +
                                              " bytes) does not clear the display headroom (" +
                                              std::to_string(headroomBytes) + " bytes)"});
    }

    return DeviceWorkspaceBudget{deviceId, freeBytes - headroomBytes, freeBytes, headroomBytes};
}

inline bool DeviceWorkspaceBudget::Reserve(std::size_t bytes) noexcept {
    if (bytes == 0)
    {
        return true;
    }

    // Guard before the fit check: capacity - bytes underflows when bytes
    // exceeds the capacity, which must read as a rejection, not a wrapped
    // acceptance.
    if (bytes > _capacityBytes)
    {
        return false;
    }

    std::size_t current = _committedBytes.load(std::memory_order_relaxed);

    for (;;)
    {
        if (current > _capacityBytes - bytes)
        {
            return false;
        }

        if (_committedBytes.compare_exchange_weak(
                current, current + bytes, std::memory_order_relaxed, std::memory_order_relaxed))
        {
            return true;
        }

        // CAS failed: `current` was refreshed with the observed value.
    }
}

inline void DeviceWorkspaceBudget::Release(std::size_t bytes) noexcept {
    // Intentional no-op on the counter: released device memory is not
    // returned to the budget (the context retains its mapped region until
    // the allocations are freed), so the cumulative commit is unchanged.
    // The parameter is accepted so callers can mirror their Reserve
    // arguments in RAII scopes and diagnostics.
    (void)bytes;
}

inline std::size_t DeviceWorkspaceBudget::Remaining() const noexcept {
    return _capacityBytes - _committedBytes.load(std::memory_order_relaxed);
}

inline std::size_t DeviceWorkspaceBudget::CapacityBytes() const noexcept {
    return _capacityBytes;
}

inline std::size_t DeviceWorkspaceBudget::CommittedBytes() const noexcept {
    return _committedBytes.load(std::memory_order_relaxed);
}

inline int DeviceWorkspaceBudget::DeviceId() const noexcept {
    return _deviceId;
}

inline std::size_t DeviceWorkspaceBudget::FreeBytesAtCreate() const noexcept {
    return _freeBytesAtCreate;
}

inline std::size_t DeviceWorkspaceBudget::HeadroomBytes() const noexcept {
    return _headroomBytes;
}

} // namespace qcx::memory
