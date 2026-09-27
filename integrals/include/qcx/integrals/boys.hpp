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

/// F_0(x)..F_nmax(x) in double precision: the entry this framework's Boys
/// kernel calls, and the measurement behind that choice.
///
/// The library offers three call shapes at the certified accuracy, all three
/// forwarded here: this one, the many-argument entry `BoysAllN` called once per
/// order run, and that same call with the arguments declared already sorted
/// (`BoysAllN` with `BoysSortedArgs`). This framework calls this one - per
/// argument, one seed then the recursion up to that argument's order - in the
/// MD engine's `MdBoysBatch` and in the derivative kernels.
///
/// **The choice was measured, on the development machine, and it is that
/// machine's.** 12 logical processors with the AVX2+FMA tier present, MSVC 14.51
/// release build of the submodule at the revision this tree pins, the library's
/// option probe (`boys::RunOptionProbe`), 16384 arguments log-uniform on
/// [1e-3, 40] with each argument's
/// highest order drawn as the sum of two shell angular momenta over 0..16. Over
/// eleven runs in which all three shapes produced a figure, this entry was the
/// fastest in ten, by 1.0% to 6.9% over the next.
///
/// **The probe did not separate them, and the default rests on that refusal
/// rather than on a ranking.** Each run's resolution - the wider of its
/// fixed-work canary's own disagreement inside an admitted pass and the leading
/// option's disagreement across its admitted passes - came out between 5.2% and
/// 56.2%, never tighter than the margin it was asked to order. Every run
/// therefore reported that it could not determine a winner: nothing here places
/// this entry ahead of the other two. It is the entry the probe never placed
/// behind either of them, and so the one not worse than they are within the
/// resolution this machine could measure. It is also the shape already in force,
/// so the measurement confirms the default rather than moving it.
///
/// None of this is a claim about another machine. The probe is run where its
/// numbers are used, and a host with a steadier clock may separate the three.
/// The resolution moved with how steady the machine was rather than with how
/// busy it was, and on this laptop in both directions: the same protocol gave
/// 5.2% to 32.7% while a second job held the machine and 35.9% to 56.2% once it
/// had ended, because the term that dominated was the leading option's own
/// spread across its admitted passes, and an idle machine's clock state differs
/// from pass to pass.
///
/// Several options the library exposes are not defaulted here. The relaxed
/// accuracy tiers are the run-time selection below and trade accuracy for speed.
/// The fit route, the evaluation scheme, the engine budget and the packing axis
/// are the library's compile-time option space: this tree compiles and links
/// every unit that serves them, and expresses no default among them, so the
/// calls above take the library's own. The multiply-add route is a
/// property of the build rather than of a call, and this build's arithmetic does
/// not contract a bare product-plus-add. The CUDA lane the same probe measured
/// fastest is not consumed here at all: the CUDA engine evaluates Boys from the
/// submodule's coefficient tables in its own device kernels rather than calling
/// the library's CUDA lane.
///
/// \ingroup qcx-integrals
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
