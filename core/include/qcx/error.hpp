#pragma once
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

namespace qcx {

/// \defgroup qcx-core Core module
/// Fundamental types, error handling, and logging shared by every module in
/// the qcx dependency DAG.
/// \{

/// Error categories returned by qcx operations.
enum class ErrorCode : std::uint8_t {
    kInvalidArgument, ///< An argument does not satisfy the operation's contract.
    kConvergenceFailure, ///< An iterative procedure failed to converge within its budget.
    kIOError, ///< File or device I/O failed.
    kUnimplemented, ///< The requested feature is not implemented (yet).
    kDeviceError, ///< A device (GPU or accelerator) operation failed.
    kOutOfMemory, ///< An allocation failed (host or device memory exhausted).
    kInternalError, ///< An unexpected internal failure (e.g. a third-party library threw).
};

/// An error: a category plus a human-readable message.
struct Error {
    ErrorCode code; ///< Category of the error.
    std::string message; ///< Human-readable description (English, no trailing newline).
};

/// Result of a fallible operation - either a value of type \p T or an Error.
///
/// Implemented as std::expected, so the happy path carries no overhead and
/// the error path needs no exceptions.
template <typename T> using Result = std::expected<T, Error>;

/// Returns the human-readable name of an ErrorCode.
///
/// Used by logging and diagnostics. A code outside the known range (never
/// produced by qcx itself) yields "UnknownError".
/// \param code Error code to name.
/// \returns The code's name ("UnknownError" outside the known range).
std::string_view ToString(ErrorCode code);

/// \}
} // namespace qcx
