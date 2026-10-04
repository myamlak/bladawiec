#pragma once

// The Boys seed adapter of the MD engine: the fp64 pipeline consumes
// BoysAllOrders directly, the certified fp32 pipeline consumes
// BoysAllOrdersF32 - whose region-A seeds are computed in double internally,
// so the amplification-sensitive seeds are safe. The fp32-vs-fp64
// choice is a per-call precision policy, never a per-quartet branch inside
// the kernel: the caller's gate decides which pipeline each quartet goes
// through.
//
// The accuracy rung below is the other axis and it is the weaker one: it
// selects a looser fit of the same arithmetic rather than a different
// arithmetic, so it is a value a batch carries and not a dispatch dimension
// the engine branches on.

#include "qcx/integrals/boys.hpp"

#include <cstddef>
#include <type_traits>

namespace qcx::integrals::internal {

/// F_0(x)..F_nmax(x) into out, in the working scalar type T.
///
/// The shape is per-argument: one argument, the whole ladder up to \p nmax.
/// The engine forms its argument inside the primitive-quadruple loop and
/// consumes the ladder on the next line, and the library's many-argument
/// entries buy their grouping only from a gather the caller has to perform
/// first - see the seed-entry note in qcx/integrals/boys.hpp for the
/// measurement behind the choice and for what a batched shape was measured to
/// be worth on this engine's own argument stream.
/// \tparam T double (5e-14 per value) or float (1e-7 per value).
template <typename T> inline void MdBoysBatch(int nmax, double x, T* out) noexcept {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);

    if constexpr (std::is_same_v<T, double>)
    {
        BoysAllOrders(nmax, x, out);
    } else
    {
        // The ladder is written straight into the caller's array: the fp32
        // entry takes and returns float, so an intermediate buffer would be a
        // float-to-float copy and nothing else. The argument is the one
        // conversion, and it is the call's own (the region-A seeds are
        // computed in double inside the entry, so the
        // amplification-sensitive seeds are safe).
        BoysAllOrdersF32(nmax, static_cast<float>(x), out);
    }
}

/// F_0(x)..F_nmax(x) into out, at a run-time-selected rung.
///
/// Double only: the relaxed rungs exist for the fp64 pipeline. The fp32
/// pipeline keeps its own 1e-7 contract with no rung to select, so it has no
/// entry here. The reference rung is the compile-time default that MdBoysBatch
/// above reaches, so an engine that never selects a rung runs the code it
/// runs today.
///
/// A rung is not a tolerance the caller may ignore: the library's released
/// rungs multiply the lane's own Boys bound by m, so a rung selected here
/// loosens 5e-14 to m x 5.5e-14 for every value the seeded ladder feeds. The
/// call sites that select one state what they do with that.
/// \param tier the rung, a property of this call only
template <typename T>
inline void MdBoysBatchAtTier(AccuracyTier tier, int nmax, double x, T* out) noexcept {
    static_assert(std::is_same_v<T, double>,
                  "the relaxed tiers are instantiated for the fp64 pipeline only");

    BoysAllOrdersAtTier(tier, nmax, x, out);
}

} // namespace qcx::integrals::internal
