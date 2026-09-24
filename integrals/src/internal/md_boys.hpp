#pragma once

// The Boys seed adapter of the MD engine: the fp64 pipeline consumes
// BoysAllOrders directly, the certified fp32 pipeline consumes
// BoysAllOrdersF32 - whose region-A seeds are computed in double internally,
// so the amplification-sensitive seeds are safe. The fp32-vs-fp64
// choice is a per-call precision policy, never a per-quartet branch inside
// the kernel: the caller's gate decides which pipeline each quartet goes
// through.

#include "qcx/integrals/boys.hpp"

#include <cstddef>
#include <type_traits>

namespace qcx::integrals::internal {

/// F_0(x)..F_nmax(x) into out, in the working scalar type T.
/// \tparam T double (5e-14 per value) or float (1e-7 per value).
template <typename T> inline void MdBoysBatch(int nmax, double x, T* out) noexcept {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);

    if constexpr (std::is_same_v<T, double>)
    {
        BoysAllOrders(nmax, x, out);
    } else
    {
        float seeds[1 + qcx::integrals::kMaxBoysOrder];
        BoysAllOrdersF32(nmax, static_cast<float>(x), seeds);

        for (int m = 0; m <= nmax; ++m)
        {
            out[m] = seeds[m];
        }
    }
}

/// F_0(x)..F_nmax(x) into out, at a run-time-selected tier.
///
/// Double only: the relaxed rungs exist for the fp64 pipeline. The fp32
/// pipeline keeps its own 1e-7 contract with no tier to select, so it has no
/// entry here. The reference tier is the compile-time default MdBoysBatch
/// above reaches, so an engine that never selects a tier runs the code it
/// runs today.
/// \param tier the tier, a property of this call only
template <typename T>
inline void MdBoysBatchAtTier(AccuracyTier tier, int nmax, double x, T* out) noexcept {
    static_assert(std::is_same_v<T, double>,
                  "the relaxed tiers are instantiated for the fp64 pipeline only");

    BoysAllOrdersAtTier(tier, nmax, x, out);
}

} // namespace qcx::integrals::internal
