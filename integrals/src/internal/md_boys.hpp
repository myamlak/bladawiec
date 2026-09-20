#pragma once

// The Boys seed adapter of the MD engine: the fp64 pipeline consumes
// BoysBatch directly, the certified fp32 pipeline consumes
// BoysBatchF32 - whose region-A seeds are computed in double internally,
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
        BoysBatch(nmax, x, out);
    } else
    {
        float seeds[1 + qcx::integrals::kMaxBoysOrder];
        BoysBatchF32(nmax, static_cast<float>(x), seeds);

        for (int m = 0; m <= nmax; ++m)
        {
            out[m] = seeds[m];
        }
    }
}

} // namespace qcx::integrals::internal
