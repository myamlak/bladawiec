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

// The direct-timing instrument of the Boys-share probe
// (benchmarks/boys_share_probe.cpp). It is defined only where
// QCX_BOYS_SHARE_INSTRUMENT is, which the build sets on qcx-integrals and on
// that one benchmark and on nothing else, so every other build compiles this
// header to the text it always had.
//
// Two instruments run at once, because they fail differently. The stamp
// counter reads the calls' own cost in place, and the elision flag beside it
// takes the library call out altogether so the same spans can be differenced
// with no per-call model at all. The first is what the second checks.
#if defined(QCX_BOYS_SHARE_INSTRUMENT)

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace qcx::integrals::internal {

/// The time-stamp counter, read at the instrument's two ends.
inline std::uint64_t BoysShareStamp() noexcept {
    return static_cast<std::uint64_t>(__rdtsc());
}

/// Cycles spent inside the instrumented calls, and how many of them, on the
/// calling thread. Valid only while \c gBoysShareSkip is false: the elided
/// arm times its own zero-fill instead.
inline thread_local std::uint64_t gBoysShareCycles = 0;
inline thread_local std::uint64_t gBoysShareCalls = 0;

/// True while the probe runs the Boys-elided arm. The call is then skipped and
/// the ladder left zeroed, so the arm executes the same instruction stream
/// with the library's arithmetic taken out. It is one predictable branch per
/// call and it is present in both arms, so it cancels in the difference.
inline bool gBoysShareSkip = false;

/// The ladder an elided call leaves: zeros, which no later step can turn into
/// a denormal, so the rest of the kernel runs at its own speed.
template <typename T> inline void BoysShareZero(int nmax, T* out) noexcept {
    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = T{};
    }
}

} // namespace qcx::integrals::internal

#endif

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

#if defined(QCX_BOYS_SHARE_INSTRUMENT)
    const std::uint64_t boysShareStart = BoysShareStamp();

    if (gBoysShareSkip)
    {
        BoysShareZero(nmax, out);
        gBoysShareCycles += BoysShareStamp() - boysShareStart;
        ++gBoysShareCalls;

        return;
    }
#endif

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

#if defined(QCX_BOYS_SHARE_INSTRUMENT)
    gBoysShareCycles += BoysShareStamp() - boysShareStart;
    ++gBoysShareCalls;
#endif
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
