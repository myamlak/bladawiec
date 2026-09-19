#pragma once

namespace qcx::backend {

/// \defgroup qcx-backend Backend module
/// Execution abstraction: backend dispatch tags, the ExecutionBackend concept,
/// and concrete backends (CPU, CUDA) selected by tag.
/// \{

/// Tag selecting the CPU (OpenMP) backend.
struct CpuTag {};

/// Tag selecting the CUDA backend.
struct CudaTag {};

/// \}
} // namespace qcx::backend
