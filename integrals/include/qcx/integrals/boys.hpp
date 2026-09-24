#pragma once

/// \defgroup qcx-integrals Integrals module
///
/// One- and two-electron molecular integrals over Gaussian-type orbitals.
/// Everything here is measured against the provisional accuracy targets:
/// absolute error 5e-14 in double precision and 1e-7 in single
/// precision for the Boys kernel, which is the accuracy bottleneck of all
/// subsequent integral recursions.

// The Boys kernel is built from the boys submodule's sources into
// qcx-integrals; this header forwards the submodule's public API into
// namespace qcx::integrals.
//
// The per-region SIMD entries stay internal: a caller cannot name the
// library's regions, so the many-argument entry (BoysAllN) is the public shape
// that reaches the vector lanes.
//
// The fp16/bf16 entries sit behind the submodule's BoysFp16 seam, which the
// build mirrors onto QCX_INTEGRALS_FP16 (a PUBLIC target definition on
// qcx-integrals), and the F16/Bf16 types plus the qcx::integrals::detail
// alias come from the f16.hpp shim included below.
//
// The CUDA lane is not forwarded: CUDA consumers use the upstream namespace
// directly.

#include "boys/boys.hpp"
#include "qcx/integrals/f16.hpp"

namespace qcx::integrals {

using boys::BoysAllN;
using boys::BoysAllNWorkspaceSize;
using boys::BoysAllOrders;
using boys::BoysAllOrdersF32;
using boys::BoysAvx2Available;
using boys::BoysSingle;
using boys::BoysSingleF32;
using boys::BoysSortedArgs;
using boys::kBoysFullAccuracyMultiplier;
using boys::kMaxBoysOrder;

// The upstream all-orders entries are named BoysAllOrders*; these keep the
// local forwarding names so every include site in this tree compiles
// unchanged. Same instantiation the submodule exports, so the numerics are
// those of the kernel itself.
template <double kAccuracyMultiplier = boys::kBoysFullAccuracyMultiplier>
inline void BoysBatch(int nmax, double x, double* out) noexcept {
    boys::BoysAllOrders<kAccuracyMultiplier>(nmax, x, out);
}

template <double kAccuracyMultiplier = boys::kBoysFullAccuracyMultiplier>
inline void BoysBatchF32(int nmax, float x, float* out) noexcept {
    boys::BoysAllOrdersF32<kAccuracyMultiplier>(nmax, x, out);
}

// The run-time tier, forwarded so a caller in this tree selects the accuracy
// of one call without reaching into the submodule's namespace. The tier is a
// property of the call and carries nothing between calls. BoysBatch above is
// the reference tier, so a caller that never names a tier gets the certified
// lane - the same code it reaches today.
using boys::AccuracyComponent;
using boys::AccuracyMultiplier;
using boys::AccuracyRegion;
using boys::AccuracyTier;
using boys::BoysAllOrdersAtTier;
using boys::QueryTier;
using boys::TierCoverage;

#if BoysFp16
using boys::BoysAllOrdersBf16;
using boys::BoysAllOrdersF16;
using boys::BoysSingleBf16;
using boys::BoysSingleF16;
#endif // BoysFp16

} // namespace qcx::integrals
