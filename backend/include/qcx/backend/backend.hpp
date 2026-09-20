#pragma once
#include "qcx/backend/tags.hpp"

namespace qcx::backend {

/// Policy object holding the capabilities of a backend selected by \p Tag.
///
/// Deliberately undefined as a primary template: every tag must have an
/// explicit specialization (cpu_backend.hpp, cuda_backend.hpp),
/// so using an unimplemented backend fails at compile time.
/// \ingroup qcx-backend
/// \tparam Tag One of CpuTag or CudaTag.
template <typename Tag> class Backend;

} // namespace qcx::backend
