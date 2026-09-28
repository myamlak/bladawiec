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

// The many-argument entries: one call for a whole array of arguments, in the
// order the caller holds them, and the same call with the arguments declared
// already sorted (BoysSortedArgs, the precondition-checked overload that skips
// the sort).
//
// **This framework calls neither, and the choice is a structural one rather
// than a preference.** Its hot path forms one argument per primitive quadruple
// inside the recursion that consumes it - `x = p q / (p + q) * |P - Q|^2`, then
// the ladder, then the transform - and never holds a second argument at that
// point. A batched shape therefore costs the caller a gather of every
// (order, argument) pair before the loop, an arena of `count * (order + 1)`
// seeds, and a second pass to read them back, which is engine work this tree
// has not done rather than a substitution at the call site.
//
// **What that work would buy is claimed by two measurements taken in this tree,
// and they do not agree.** Neither supports the factor of three once recorded
// for the batched shape, a figure from two runs on a uniform argument stream at
// one order whose fast arm was a pre-partitioned region kernel rather than an
// entry a caller takes.
//
// The first is a replay harness, timed, 2026-09-22, MSVC release: it enumerates
// the screened-in quartets of real alkane builds at def2-SVP in the engine's own
// emission order, counting-sorts them by region and order, and runs the
// library's per-region lanes once per bucket, the sort inside the timed span
// because a sorted path that wins only when the sort is free is not a win a
// batch entry delivers. c12h26 replayed at 57.1 ns per call the per-argument way
// against 25.9 ns sorted (2.20x), c24h50 at 40.0 ns against 20.5 ns (1.95x),
// both arms agreeing to 2.2e-16 - against rounds that carried 34% to 53% of
// spread.
//
// The second counts instead of timing, 2026-09-25, over the same workload and
// the same three fp64 call shapes: the per-argument entry did the least work at
// orders 16 and 32, the batched entry at orders 4 and 8, and the spread across
// rounds ran from 12.6% to 97%. A count is the load-immune instrument and a time
// is not, so this is not one instrument being noisier than the other: the two
// ran different groupings - the replay's fast arm buckets by region *and* order,
// the counted one groups by order alone - on different days on the same laptop.
//
// The honest state is therefore that this tree does not know which shape wins on
// its own workload, and the per-argument call stands until a measurement
// separates them. It is also the reason adopting the batched shape is not a
// call-site substitution here: it is a two-pass restructuring of the VRR, and it
// would be made against a contested number.
using boys::BoysAllN;
using boys::BoysAllNWorkspaceSize;

/// F_0(x)..F_nmax(x) in double precision, one argument, every order up to
/// `nmax`: the entry this framework's Boys kernel calls.
///
/// The shape is the per-argument one described above the many-argument entries,
/// and the note there carries the measurement of what the batched alternative
/// is worth on this engine's own quartets. This entry is the other side of that
/// decision: it is what the engine calls today, once per primitive quadruple,
/// in the MD engine's `MdBoysBatch`, in the derivative kernels and in the
/// one-electron recursion.
///
/// **The choice was measured on the development machine, and it is that
/// machine's.** 12 logical processors with the AVX2+FMA tier present, MSVC 14.51
/// release build of the submodule at the revision this tree pins, the library's
/// option probe (`boys::RunOptionProbe`, the CPU-lane probe), 16384 arguments
/// log-uniform on [1e-3, 40] with each argument's highest order drawn as the
/// sum of two shell angular momenta over 0..16. Over eleven runs in which all
/// three shapes produced a figure, this entry was the fastest in ten, by 1.0%
/// to 6.9% over the next.
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
/// accuracy rungs are the run-time selection below and trade accuracy for
/// speed. The fit route, the evaluation scheme, the interval granularity, the
/// engine budget and the packing axis are the library's compile-time option
/// space: this tree compiles and links every unit that serves them, and
/// expresses no default among them, so the calls above take the library's own.
/// Those defaults moved between the revision this tree pins and the revision it
/// moves to, so a caller that names none of the five axes compiles a different
/// kernel than it did before - the accuracy contract each entry publishes is
/// unchanged, and the arithmetic behind it is not. The multiply-add route is a
/// property of the build rather than of a call, and this build's arithmetic
/// does not contract a bare product-plus-add.
///
/// The CUDA engine does not call the library's CUDA lane. It evaluates Boys in
/// its own device kernels, from tables this tree builds out of the library's
/// generated coefficient header at device creation, and the library's launched
/// and device-callable entries are reached by the CUDA tests and benchmarks
/// only. That duplicate is about 405 lines and a device table struct whose
/// declared arrays sum to 43 KiB, and replacing it is the one change to this
/// surface whose cost is unmeasured on the machine it would run on: nothing in
/// this tree times the fused kernel's Boys evaluation on either side.
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

// The run-time accuracy rung, forwarded so a caller in this tree selects the
// accuracy of one call without reaching into the submodule's namespace. The
// rung is a property of the call and carries nothing between calls.
//
// `BoysAllOrdersAtTier` is the reference rung's own entry until a caller names
// another one, so a caller that never selects a rung gets the certified lane -
// the same code it reaches through `BoysAllOrders` above.
using boys::AccuracyComponent;
using boys::AccuracyMultiplier;
using boys::AccuracyRegion;
using boys::AccuracyTier;
using boys::BoysAllOrdersAtTier;
using boys::BoysSingleAtTier;
using boys::QueryTier;
using boys::TierCoverage;

// The float lane's rung entry arrives with the library's next major revision
// and is forwarded with it. The fp32 pipeline carries its own 1e-7 contract
// with no rung to select, so that surface exists for callers that need a looser
// rung than that contract, not for the engine: the engine's fp32 path calls the
// reference entry above.

// The five structural axes. This tree names them so a caller can pin a
// combination where the library's default would move under it; nothing in the
// engine does, and the entries above take the library's own.
using boys::BoysBudget;
using boys::EvalPolicy;
using boys::EvalScheme;
using boys::FitGranularity;
using boys::FitRoute;
using boys::PackAxis;

#if BoysFp16
using boys::BoysAllOrdersBf16;
using boys::BoysAllOrdersF16;
using boys::BoysSingleBf16;
using boys::BoysSingleF16;
#endif // BoysFp16

} // namespace qcx::integrals
