#pragma once

/// \defgroup qcx-integrals Integrals module
///
/// One- and two-electron molecular integrals over Gaussian-type orbitals.
/// Everything here is measured against the provisional accuracy targets:
/// absolute error 5e-14 in double precision and 1e-7 in single
/// precision for the Boys kernel, which is the accuracy bottleneck of all
/// subsequent integral recursions.

// boys.hpp forwarding shim (the boys submodule switch):
// the Boys-function kernel is now built from the boys submodule's
// sources (external/boys, pinned gitlink - v1.0.0 at this flip;
// the public repo is the slice's source of truth, the monorepo bumps the
// gitlink only). This header forwards the submodule's public API into
// namespace qcx::integrals so every monorepo include site compiles
// unchanged; integrals/CMakeLists.txt compiles the kernel from
// external/boys/src/boys.cpp + boys_simd.cpp into qcx-integrals.
//
// The fp16/bf16 entries sit behind the submodule's BoysFp16 seam, which the
// build mirrors onto QCX_INTEGRALS_FP16 (a PUBLIC target definition on
// qcx-integrals), and the F16/Bf16 types plus the qcx::integrals::detail
// alias come from the f16.hpp shim included below.
//
// The CUDA lane (boys/boys_cuda.hpp, boys::BoysStatus) is
// NOT forwarded - CUDA consumers switched directly to the upstream
// namespace when boys_cuda.hpp retired.

#include "boys/boys.hpp"
#include "qcx/integrals/f16.hpp"

namespace qcx::integrals {

using boys::BoysAvx2Available;
using boys::BoysBatch;
using boys::BoysBatchF32;
using boys::BoysRegionASimd;
using boys::BoysRegionBSimd;
using boys::BoysRegionCSimd;
using boys::BoysSingle;
using boys::BoysSingleF32;
using boys::kBoysFullAccuracyMultiplier;
using boys::kMaxBoysOrder;

#if BoysFp16
using boys::BoysBatchBf16;
using boys::BoysBatchF16;
using boys::BoysRegionASimdBf16;
using boys::BoysRegionASimdF16;
using boys::BoysRegionBSimdBf16;
using boys::BoysRegionBSimdF16;
using boys::BoysRegionCSimdBf16;
using boys::BoysRegionCSimdF16;
using boys::BoysSingleBf16;
using boys::BoysSingleF16;
#endif // BoysFp16

} // namespace qcx::integrals
