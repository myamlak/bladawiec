#pragma once

#include "qcx/error.hpp"

#include <atomic>
#include <cstddef>
#include <utility>

namespace qcx::memory {

/// Pure accounting workspace budget: a fixed byte capacity against which
/// builders reserve their create-time footprint.
///
/// No allocation and no device knowledge - the deterministic input to
/// per-builder mode selection. Retention-aware: Release() does not restore
/// capacity, because the system heap does not decommit freed pages, so the
/// committed counter is monotone non-decreasing for the process lifetime
/// and Remaining() never re-opens consumed capacity. Only Reserve moves the
/// counter, and only when the full amount fits. Thread-safe: concurrent
/// reservations charge exactly once each (nested-builder creates are
/// sequential today, but the invariant is load-bearing). Not copyable -
/// the counter member is a std::atomic - and the seam shares one budget
/// by pointer, never by value.
/// \ingroup qcx-memory
class WorkspaceBudget {
public:
    /// Creates a budget with a fixed byte capacity.
    /// \param capacityBytes Total capacity in bytes; must be non-zero.
    /// \returns The budget, or an Error (kInvalidArgument when
    /// capacityBytes is zero).
    static qcx::Result<WorkspaceBudget> Create(std::size_t capacityBytes);

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
    /// The cumulative counter is intentionally unchanged: released memory
    /// is not returned to the budget, so a later Reserve still charges the
    /// process's committed-so-far. Provided so RAII reservation scopes and
    /// audit trails can mirror their Reserve calls; it cannot fail and
    /// never re-opens capacity.
    /// \param bytes Number of bytes being released; has no effect.
    void Release(std::size_t bytes) noexcept;

    /// Bytes not yet committed.
    /// \returns The capacity minus the cumulative commit; monotone
    /// non-increasing for the process lifetime.
    std::size_t Remaining() const noexcept;

    /// The fixed capacity the budget was created with.
    /// \returns The capacity in bytes.
    std::size_t CapacityBytes() const noexcept;

    /// The cumulative high-water commit.
    /// \returns The bytes charged by successful Reservations so far.
    std::size_t CommittedBytes() const noexcept;

    /// Move construction: copies the committed value.
    ///
    /// There is nothing to transfer from a counter, so the source keeps its
    /// state. Present for the Result-creating house pattern (Create returns
    /// by value); budgets are shared by pointer, so moves are never
    /// part of the seam.
    /// \param other The budget whose values are copied.
    WorkspaceBudget(WorkspaceBudget&& other) noexcept :
        _capacityBytes(other._capacityBytes),
        _committedBytes(other._committedBytes.load(std::memory_order_relaxed)) {}

    WorkspaceBudget(const WorkspaceBudget&) = delete;
    /// Copying is deleted: budgets are shared by pointer, never by value.
    WorkspaceBudget& operator=(const WorkspaceBudget&) = delete;
    /// Move assignment is deleted like copying: budgets are shared by pointer, never by value.
    WorkspaceBudget& operator=(WorkspaceBudget&&) = delete;

private:
    explicit WorkspaceBudget(std::size_t capacityBytes) : _capacityBytes(capacityBytes) {}

    std::size_t _capacityBytes = 0;
    // Monotone: only Reserve writes it, and only upwards; no ordering
    // requirements with other memory, so relaxed is sufficient.
    std::atomic<std::size_t> _committedBytes{0};
};

inline qcx::Result<WorkspaceBudget> WorkspaceBudget::Create(std::size_t capacityBytes) {
    if (capacityBytes == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "WorkspaceBudget requires a non-zero capacity"});
    }

    return WorkspaceBudget{capacityBytes};
}

inline bool WorkspaceBudget::Reserve(std::size_t bytes) noexcept {
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

inline void WorkspaceBudget::Release(std::size_t bytes) noexcept {
    // Intentional no-op on the counter: released memory is not returned to
    // the budget (the system heap does not decommit freed pages), so the
    // cumulative commit is unchanged. The parameter is accepted so callers
    // can mirror their Reserve arguments in RAII scopes and diagnostics.
    (void)bytes;
}

inline std::size_t WorkspaceBudget::Remaining() const noexcept {
    return _capacityBytes - _committedBytes.load(std::memory_order_relaxed);
}

inline std::size_t WorkspaceBudget::CapacityBytes() const noexcept {
    return _capacityBytes;
}

inline std::size_t WorkspaceBudget::CommittedBytes() const noexcept {
    return _committedBytes.load(std::memory_order_relaxed);
}

} // namespace qcx::memory
