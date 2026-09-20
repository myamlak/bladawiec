#pragma once
#include <concepts>

namespace qcx::backend {

/// Concept satisfied by every execution backend.
///
/// A backend exposes the types its stream and device ids take, and reports
/// the id of the device it runs on.
/// \ingroup qcx-backend
template <typename B>
concept ExecutionBackend = requires(const B b) {
    typename B::StreamType;
    typename B::DeviceIdType;
    { b.DeviceId() } -> std::convertible_to<typename B::DeviceIdType>;
};

} // namespace qcx::backend
