#pragma once

namespace qcx::memory {

/// \defgroup qcx-memory Memory module
/// Device-aware memory management: per-backend allocation traits and the
/// RAII DeviceBuffer, the exclusive site of all host and device allocation.
/// \{

/// Opaque, backend-typed handle to device memory.
///
/// Deliberately not a T*: host code cannot accidentally dereference device
/// memory through it. Pass the handle (or its Raw() pointer) to kernels.
/// \tparam T Element type of the buffer the handle points into.
/// \tparam Backend Execution backend owning the memory.
template <typename T, typename Backend> class DevicePtr {
public:
    /// Constructs a handle from a raw allocation (nullptr = empty handle).
    /// \param raw Raw allocation; nullptr yields an empty handle.
    explicit DevicePtr(void* raw = nullptr) : _raw(raw) {}

    /// The raw pointer underlying the handle.
    /// \returns The raw device pointer (nullptr for an empty handle).
    void* Raw() const {
        return _raw;
    }

    /// True when the handle refers to memory.
    explicit operator bool() const {
        return _raw != nullptr;
    }

private:
    void* _raw;
};

/// \}
} // namespace qcx::memory
