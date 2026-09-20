#include "qcx/error.hpp"

#include <array>

namespace qcx {

std::string_view ToString(const ErrorCode code) {
    static constexpr std::array kNames = {
        "InvalidArgument",
        "ConvergenceFailure",
        "IOError",
        "Unimplemented",
        "DeviceError",
        "OutOfMemory",
        "InternalError",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < kNames.size() ? kNames[index] : "UnknownError";
}

} // namespace qcx
