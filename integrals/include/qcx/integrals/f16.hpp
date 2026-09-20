#pragma once

// f16.hpp forwarding shim (the boys submodule switch):
// the fp16/bf16 I/O types are now owned by the boys submodule
// (external/boys, pinned gitlink - v1.0.0 at this flip), whose
// public repo is the slice's source of truth. This header forwards the
// public surface into namespace qcx::integrals so monorepo include sites
// compile unchanged; the submodule's include/boys/f16.hpp is the
// implementation. The submodule's f16.hpp is unconditional (the F16/Bf16
// I/O types ship on every toolchain), so the shim is unconditional too.

#include "boys/f16.hpp"

namespace qcx::integrals {

using boys::Bf16;
using boys::F16;
using boys::NextUp;

// The retired monorepo copies declared their internals in
// qcx::integrals::detail; the submodule keeps the same helpers under
// boys::detail. One alias keeps every retained detail:: reference
// (tests, benchmarks, eri_cuda.cpp) resolving without text edits.
namespace detail = ::boys::detail;

} // namespace qcx::integrals
