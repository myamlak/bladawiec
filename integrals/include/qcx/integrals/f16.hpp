#pragma once

// f16.hpp forwarding shim:
// the fp16/bf16 I/O types are owned by the boys submodule
// (external/boys, pinned gitlink - v1.0.0). This header forwards the
// public surface into namespace qcx::integrals so include sites
// compile unchanged; the submodule's include/boys/f16.hpp is the
// implementation. The submodule's f16.hpp is unconditional (the F16/Bf16
// I/O types ship on every toolchain), so the shim is unconditional too.

#include "boys/f16.hpp"

namespace qcx::integrals {

using boys::Bf16;
using boys::F16;
using boys::NextUp;

// The earlier in-tree copies declared their internals in
// qcx::integrals::detail; the submodule keeps the same helpers under
// boys::detail. One alias keeps every retained detail:: reference
// (tests, benchmarks, eri_cuda.cpp) resolving without text edits.
namespace detail = ::boys::detail;

} // namespace qcx::integrals
