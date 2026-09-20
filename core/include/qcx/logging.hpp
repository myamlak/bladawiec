#pragma once
#include <spdlog/spdlog.h>
#include <utility>

namespace qcx::log {

/// \defgroup qcx-log Logging
/// Thin wrapper over spdlog so the rest of the codebase depends on qcx::log,
/// not on spdlog directly - keeps the logging backend swappable later.
/// \{

/// Initializes the default logger: stdout with the pattern
/// `[HH:MM:SS] [level] message`. Called once at startup.
void Init();

/// Logs an informational message with spdlog-style formatting.
/// \param fmt Format string; must be a compile-time literal (see the spdlog documentation).
/// \param args Format arguments.
template <typename... Args> void Info(spdlog::format_string_t<Args...> fmt, Args&&... args) {
    spdlog::info(fmt, std::forward<Args>(args)...);
}

/// Logs an error message with spdlog-style formatting.
/// \param fmt Format string; must be a compile-time literal (see the spdlog documentation).
/// \param args Format arguments.
template <typename... Args> void Error(spdlog::format_string_t<Args...> fmt, Args&&... args) {
    spdlog::error(fmt, std::forward<Args>(args)...);
}

/// \}
} // namespace qcx::log
