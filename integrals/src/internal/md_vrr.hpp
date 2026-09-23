#pragma once

// The 4-step matrix-form McMurchie-Davidson pipeline per {L_bra, L_ket}
// class, templated on the working scalar type (double/float - the certified
// mixed precision). [McMurchie1978] [LibintXMatrixForm]
//
// Algebra (derived here; validated against an unfolded reference path in
// eri_batch_test.cpp and the independent mpmath grids of
// tools/gen_md_reference.py):
//
//   [t]^(m) := (d/dP)^t [0]^(m),  [0]^(m) = K F_m(a R_PQ^2),
//   K = 2 pi^(5/2) / (p q sqrt(p+q)) E_ab E_cd,  a = p q / (p+q),
//   (ab|cd) = sum_t E_t^(ab) sum_u E_u^(cd) [t+u]^(0)   (t, u 3D-Hermite).
//
// With (d/dP_x) F_m(a R_PQ^2) = -2a (P-Q)_x F_{m+1}(a R_PQ^2):
//
//   VRR:  [t+e]^(m) = -2a (P-Q)_dir [t]^(m+1) - 2a t_dir [t-e]^(m+1)
//
// computed per "slice" T = |t| + m, tier n = |t| in ascending order (the
// first source lives on the same slice one tier below, the second one tier
// below that - md_defs.hpp lays the tiers out; the entry (t, m) of slice T
// sits at the flat 3D-Hermite index Hermite3DIndex(t)). The engine keeps
// the whole slice triangle up to L = LBra + LKet; the targets are [t]^(0)
// (tier |t| of slice |t|), gathered into [p~|q~] = [p~+q~]^(0).
//
// The 4 steps per class batch (contraction-with-recurrence, [LibintXMatrixForm]):
//   1. VRR: BoysAllOrders(L, a R_PQ^2) seeds -> the slice triangle -> [p~|q~],
//      contracted over the bra primitive pairs on the fly (CWR part 1:
//      d_a d_b weights, row-pair-aware - general contractions ride along
//      the p~ index).
//   2. Ket transform: acc[(rowAB,p~),(fc,fd)] += sum_q~ [p~|q~] T_ket(q~,
//      (fc,fd)) - one strided-batched GEMM per ket primitive pair, shared
//      B per (g,d) and shared shapes per task group (the ket contraction
//      d_g d_d is folded into T_ket; it cannot move earlier: the Boys
//      argument depends on (g,d)).
//   3. Bra transform: out = T_bra^ctd x acc per task (shared A; T_ab^ctd =
//      sum_ab d_a d_b T_ab^(ab), CWR part 2).
//   4. The packed block layout of eri_batch.hpp (i innermost).
//
// Certified mixed precision: the fp32 lane writes, per quartet, the
// a-priori error bound
//   bound = eps * C_class * (1 + G_class) * EabSum * EcdSum
//           * sum_(abgd) f * sum_m (|seed_m| + eps)
// with eps = 1e-7 (the BoysAllOrdersF32 seed tolerance and the fp32 roundoff
// bound of the GEMM/representation terms), C_class/G_class the
// generator-derived class constants (md_tables_gen.hpp: kappa_m =
// sup sqrt(x) F_{m+1}/F_m bounds each recurrence step, the path sums bound
// the amplification; factor 2 safety margin), f = max(1, (2 sqrt(a))^L)
// the geometric growth per primitive quadruple, and EabSum/EcdSum the
// contracted transform row/column sums of |.| (MdPairData). Every term is
// derived, never tuned; conservative by construction, validated by the
// committed accuracy-preset sweep.

#include "md_attribution.hpp"
#include "md_batch.hpp"
#include "md_boys.hpp"
#include "md_defs.hpp"
#include "md_kernel_span.hpp"
#include "md_scatter_table.hpp"
#include "md_transform.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

namespace qcx::integrals::internal {

/// The certified seed/roundoff epsilon: BoysAllOrdersF32's documented absolute
/// tolerance, also the fp32 roundoff bound of the transform/representation
/// terms.
inline constexpr double kCertifiedEpsilon = 1e-7;

namespace detail {

// One primitive quadruple of one quartet: the Boys seeds, the slice
// triangle, and the scatter into the task's pq block. The pq rows are per
// (rowAB, bra prim pair b, p~) and UNWEIGHTED - the bra contraction d_a d_b
// rides the bra transform T_ab^ctd (the contraction happens exactly once,
// in the bra GEMM). For the certified fp32 lane, accumulates
// f * sum_m (|seed_m| + eps) into \p bound (the rest of the per-quartet
// bound is assembled by the caller).
template <int L, int LBra, int LKet, typename T>
void RunVrrQuadruple(const MdPrimPair& braPrim,
                     const MdPrimPair& ketPrim,
                     std::size_t rowPairs,
                     std::size_t primIdx,
                     std::size_t nPrimBra,
                     int kHermBra,
                     int kHermKet,
                     T* pq,
                     double* bound) {
    const double alpha = braPrim.p * ketPrim.p / (braPrim.p + ketPrim.p);
    const double dqx = braPrim.px - ketPrim.px;
    const double dqy = braPrim.py - ketPrim.py;
    const double dqz = braPrim.pz - ketPrim.pz;
    const double x = alpha * (dqx * dqx + dqy * dqy + dqz * dqz);
    // K = 2 pi^(5/2) / (p q sqrt(p+q)) * E_ab * E_cd.
    const double prefactor = 2.0 * std::pow(std::numbers::pi, 2.5) /
                             (braPrim.p * ketPrim.p * std::sqrt(braPrim.p + ketPrim.p)) *
                             braPrim.prefactor * ketPrim.prefactor;

    T seeds[L + 1];
    MdBoysBatch<T>(L, x, seeds);
    double seedSum = 0.0;

    for (int m = 0; m <= L; ++m)
    {
        seeds[m] *= static_cast<T>(prefactor);
        seedSum += std::abs(static_cast<double>(seeds[m])) + kCertifiedEpsilon;
    }

    if (bound != nullptr)
    {
        double growth = 1.0;

        if constexpr (L > 0)
        {
            growth = std::pow(2.0 * std::sqrt(alpha), L);
            growth = growth > 1.0 ? growth : 1.0;
        }

        *bound += growth * seedSum;
    }

    const T c2 = static_cast<T>(-2.0 * alpha);
    const T c1x = static_cast<T>(-2.0 * alpha * dqx);
    const T c1y = static_cast<T>(-2.0 * alpha * dqy);
    const T c1z = static_cast<T>(-2.0 * alpha * dqz);

    // Slice T holds [t]^(T - |t|): the first VRR source lives one tier below
    // in the same slice, the second source [t-2e]^(m+1) lives in slice T-1
    // (its |t-2e| + m + 1 = T - 1) - hence the two-array ping-pong. Reading
    // the same-slice tier n-2 instead uses the wrong seed F_{T} (the
    // p-shell regression found against the unfolded path and pyscf).
    // std::array: zero-initialized (the pad entries of the slice triangle
    // beyond tier |t| are never written but must stay zero for the gather)
    // and bounds-checked in MSVC Debug builds (the off-by-one here already
    // fired once as the p-shell regression).
    std::array<T, Hermite3DCount(L)> slice{};
    std::array<T, Hermite3DCount(L)> prevSlice{};
    // The pq entry |r~| = 0 is the seed [0]^(0) itself - written at every
    // (rowAB, prim pair) row; tier 0 of every slice T >= 1 is the seed
    // [0]^(T).
    for (std::size_t rowAB = 0; rowAB < rowPairs; ++rowAB)
    {
        pq[((rowAB * nPrimBra + primIdx) * static_cast<std::size_t>(kHermBra)) *
           static_cast<std::size_t>(kHermKet)] = seeds[0];
    }

    for (int sliceT = 1; sliceT <= L; ++sliceT)
    {
        slice[0] = seeds[sliceT];

        for (int n = 1; n <= sliceT; ++n)
        {
            const int off = kH2Prefix[n];
            const int off1 = kH2Prefix[n - 1];
            const int off2 = n >= 2 ? kH2Prefix[n - 2] : 0;

            for (int ty = 0; ty <= n; ++ty)
            {
                for (int tz = 0; tz <= n - ty; ++tz)
                {
                    const int tx = n - ty - tz;
                    const int sub = SubIndex3(ty, tz, n);
                    T value = T{};

                    if (tx >= 1)
                    {
                        value = c1x * slice[off1 + SubIndex3(ty, tz, n - 1)];

                        if (tx >= 2)
                        {
                            value += c2 * static_cast<T>(tx - 1) *
                                     prevSlice[off2 + SubIndex3(ty, tz, n - 2)];
                        }
                    } else if (ty >= 1)
                    {
                        value = c1y * slice[off1 + SubIndex3(ty - 1, tz, n - 1)];

                        if (ty >= 2)
                        {
                            value += c2 * static_cast<T>(ty - 1) *
                                     prevSlice[off2 + SubIndex3(ty - 2, tz, n - 2)];
                        }
                    } else
                    {
                        value = c1z * slice[off1 + SubIndex3(ty, tz - 1, n - 1)];

                        if (tz >= 2)
                        {
                            value += c2 * static_cast<T>(tz - 1) *
                                     prevSlice[off2 + SubIndex3(ty, tz - 2, n - 2)];
                        }
                    }

                    slice[off + sub] = value;
                }
            }
        }

        // Scatter tier sliceT into pq: [p~|q~] = [r~]^(0) for every split
        // p~ + q~ = r~ with |p~| <= LBra and |q~| <= LKet, weighted per
        // contraction row pair (CWR part 1).
        //
        // The table path (md_scatter_table.hpp) replaces the
        // (ty,tz)/(ps,px,py) enumeration with a walk of this slice T's
        // precomputed cells: the enumeration visited every (p~, t) combination
        // and discarded the q~ < 0 ones - 61% dead iterations at (2, 2) - and
        // paid two Hermite3DIndex evaluations per visit. The table adds the
        // same value to the same cell exactly once, in the same `+=` against
        // the zeroed block, so the pq block is bit-identical (the k = 1
        // pin), and it measured 40-65% cheaper per quadruple on the VRR body
        // across the classes a def2-SVP lean build runs. A class whose pq block
        // is past kMaxVrrScatterCells - the two-large-index corners of the
        // dispatch, l >= 6 in both indices, which the lean path never reaches -
        // keeps the enumeration below, correct and unchanged.
        if constexpr (kVrrScatterTabled<LBra, LKet>)
        {
            const std::size_t groupBegin =
                static_cast<std::size_t>(kVrrScatterBegin<LBra, LKet>[sliceT]);
            const std::size_t groupEnd =
                static_cast<std::size_t>(kVrrScatterBegin<LBra, LKet>[sliceT + 1]);

            for (std::size_t cell = groupBegin; cell < groupEnd; ++cell)
            {
                const VrrScatterCell& entry = kVrrScatterCells<LBra, LKet>[cell];
                const T value = slice[static_cast<std::size_t>(entry.src)];

                for (std::size_t rowAB = 0; rowAB < rowPairs; ++rowAB)
                {
                    pq[(rowAB * nPrimBra + primIdx) *
                           static_cast<std::size_t>(kHermBra * kHermKet) +
                       static_cast<std::size_t>(entry.dst)] += value;
                }
            }
        } else
        {
            const int off = kH2Prefix[sliceT];
            const int pLo = sliceT - LKet > 0 ? sliceT - LKet : 0;
            const int pHi = LBra < sliceT ? LBra : sliceT;

            for (int ty = 0; ty <= sliceT; ++ty)
            {
                for (int tz = 0; tz <= sliceT - ty; ++tz)
                {
                    const int tx = sliceT - ty - tz;
                    const T value = slice[off + SubIndex3(ty, tz, sliceT)];

                    for (int ps = pLo; ps <= pHi; ++ps)
                    {
                        for (int px = 0; px <= ps; ++px)
                        {
                            for (int py = 0; py <= ps - px; ++py)
                            {
                                const int pz = ps - px - py;
                                const int qx = tx - px;
                                const int qy = ty - py;
                                const int qz = tz - pz;

                                if (qx < 0 || qy < 0 || qz < 0)
                                {
                                    continue;
                                }

                                for (std::size_t rowAB = 0; rowAB < rowPairs; ++rowAB)
                                {
                                    pq[(((rowAB * nPrimBra + primIdx) *
                                             static_cast<std::size_t>(kHermBra) +
                                         Hermite3DIndex(px, py, pz)) *
                                            static_cast<std::size_t>(kHermKet) +
                                        Hermite3DIndex(qx, qy, qz))] += static_cast<T>(value);
                                }
                            }
                        }
                    }
                }
            }
        }

        // The completed slice becomes the previous slice for sliceT + 1.
        std::swap(slice, prevSlice);
    }
}

} // namespace detail

/// The fp64/fp32 pipeline of one class (definition; instantiated in the
/// per-class object libraries, dispatched through md_engine.hpp).
/// \returns An Error from the transform GEMM seam (excluded by
/// construction in normal operation).
template <int LBra, int LKet, typename T>
qcx::Result<void> ComputeEriClassImpl(const MdClassBatch& batch) {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
    constexpr int L = LBra + LKet;
    constexpr int kHermBra = Hermite3DCount(LBra);
    constexpr int kHermKet = Hermite3DCount(LKet);
    constexpr bool kCertified = std::is_same_v<T, float>;

    const std::vector<MdPairData>& pairs = *batch.pairStore;
    const std::size_t nTasks = batch.tasks.size();
    T* out = nullptr;

    if constexpr (kCertified)
    {
        out = batch.outF32;
    } else
    {
        out = batch.outF64;
    }

    // The kernel-span probe (md_kernel_span.hpp): the
    // calling thread's phase accumulators, hoisted once per class batch so
    // the hot path pays one thread-local read and one predictable branch
    // when the profiling caller has not armed them. Every accumulation below
    // sits behind `timed`, and no numerical path reads a counter - the
    // batches, the tile layout and the arithmetic are untouched.
    KernelSpanAccum& span = KernelSpan();
    const bool timed = span.enabled;
    std::chrono::steady_clock::time_point vrrStarted{};
    std::chrono::steady_clock::time_point vrrMid{};
    std::chrono::steady_clock::time_point braStarted{};

    // Whole-batch scratch: per task (in batch order) the pq block
    // (rowPairs*kHermBra x kHermKet) followed by the acc block
    // (rowPairs*kHermBra x nFuncsKet). Within one ket group (same ket pair
    // and bra row-pair count - hence equal shapes) the layout is
    // group-contiguous: the group's pq blocks first, then its acc blocks,
    // so the batched ket GEMM strides both A and C contiguously. The
    // assembler's maxBatchBytes cap accounts for the same total.
    // The scratch family (footprint's batch x threads arena term): the
    // accessors construct the thread_locals under the scratch scope on
    // every thread - the allocator's tag capture must not depend on the
    // caller's scope (md_attribution.hpp). The per-call arena slots are
    // distinct thread_local instances (the accessor keys by slot and
    // element type): the five buffers of one call have interleaved writes
    // and independent sizes - a shared instance per element type would
    // alias them onto one buffer and corrupt the batch layout.
    auto& scratch = ThreadScratchVector<T, ScratchArenaSlot::kBatch>();
    auto& taskBase = ThreadScratchVector<std::size_t, ScratchArenaSlot::kTaskBase>();
    auto& taskAcc = ThreadScratchVector<std::size_t, ScratchArenaSlot::kTaskAcc>();
    auto& convert = ThreadScratchVector<T, ScratchArenaSlot::kConvert>();
    // The per-task certified-bound accumulation, summed across every ket
    // primitive pair (the header formula requires sum over all g; the fp64
    // lane never touches it).
    auto& taskBound = ThreadScratchVector<double, ScratchArenaSlot::kTaskBound>();

    if (taskBase.size() < nTasks + 1)
    {
        taskBase.resize(nTasks + 1);
        taskAcc.resize(nTasks + 1);
    }

    if constexpr (kCertified)
    {
        if (taskBound.size() < nTasks + 1)
        {
            taskBound.resize(nTasks + 1);
        }
    }

    std::size_t total = 0;
    std::size_t layoutStart = 0;

    while (layoutStart < nTasks)
    {
        std::size_t layoutEnd = layoutStart + 1;

        while (layoutEnd < nTasks &&
               batch.tasks[layoutEnd].ketPair == batch.tasks[layoutStart].ketPair &&
               pairs[batch.tasks[layoutEnd].braPair].rowPairs ==
                   pairs[batch.tasks[layoutStart].braPair].rowPairs &&
               pairs[batch.tasks[layoutEnd].braPair].primPairs.size() ==
                   pairs[batch.tasks[layoutStart].braPair].primPairs.size())
        {
            ++layoutEnd;
        }

        const std::size_t nGroup = layoutEnd - layoutStart;
        const MdPairData& ket = pairs[batch.tasks[layoutStart].ketPair];
        const std::size_t mPq = pairs[batch.tasks[layoutStart].braPair].rowPairs *
                                pairs[batch.tasks[layoutStart].braPair].primPairs.size() *
                                static_cast<std::size_t>(kHermBra);
        const std::size_t pqSize = mPq * static_cast<std::size_t>(kHermKet);
        const std::size_t accSize = mPq * ket.nFuncs;

        for (std::size_t t = layoutStart; t < layoutEnd; ++t)
        {
            taskBase[t] = total + (t - layoutStart) * pqSize;
            taskAcc[t] = total + nGroup * pqSize + (t - layoutStart) * accSize;

            // The group-layout invariant promised in md_defs.hpp: every
            // task block lies inside its group's span (Debug builds).
            assert(taskBase[t] + pqSize <= total + nGroup * (pqSize + accSize));
            assert(taskAcc[t] + accSize <= total + nGroup * (pqSize + accSize));
        }

        total += nGroup * (pqSize + accSize);
        layoutStart = layoutEnd;
    }

    taskBase[nTasks] = total;

    scratch.assign(total, T{});

    // Pass 1 - the ket groups: the assembler sorts tasks by ket pair, so
    // equal-ket runs are contiguous and share the GEMM shapes and, per
    // primitive pair, the transform matrix B. The pq row dimension is the
    // BRA pair's rowPairs x kHermBra, so the group boundary also splits
    // where the bra row-pair count changes.
    std::size_t groupStart = 0;

    while (groupStart < nTasks)
    {
        std::size_t groupEnd = groupStart + 1;

        while (groupEnd < nTasks &&
               batch.tasks[groupEnd].ketPair == batch.tasks[groupStart].ketPair &&
               pairs[batch.tasks[groupEnd].braPair].rowPairs ==
                   pairs[batch.tasks[groupStart].braPair].rowPairs &&
               pairs[batch.tasks[groupEnd].braPair].primPairs.size() ==
                   pairs[batch.tasks[groupStart].braPair].primPairs.size())
        {
            ++groupEnd;
        }

        const std::size_t nGroup = groupEnd - groupStart;
        const MdPairData& ket = pairs[batch.tasks[groupStart].ketPair];
        const std::size_t mPq = pairs[batch.tasks[groupStart].braPair].rowPairs *
                                pairs[batch.tasks[groupStart].braPair].primPairs.size() *
                                static_cast<std::size_t>(kHermBra);
        const std::size_t nFuncsKet = ket.nFuncs;
        // The kernel-span probe: this group's ket transform shape is fixed
        // over g, so the micro-gate verdict is hoisted out of the primitive
        // loop. Short-circuited on `timed`, so a disarmed accumulator
        // evaluates nothing.
        const bool ketSeam =
            timed && !MicroGemmEligible(mPq, static_cast<std::size_t>(kHermKet), nFuncsKet);

        if (timed)
        {
            span.groupCount += 1;
        }

        // The certified bound must sum the VRR seed sums over EVERY bra and
        // ket primitive pair (the header formula). The per-g write used to
        // overwrite the accumulation with the last ket pair's contribution,
        // silently dropping the other (typically 8 of 9 STO-3G) pairs from
        // the a-priori bound (2026-08-22).
        if constexpr (kCertified)
        {
            for (std::size_t t = groupStart; t < groupEnd; ++t)
            {
                taskBound[t] = 0.0;
            }
        }

        for (std::size_t g = 0; g < ket.primPairs.size(); ++g)
        {
            // The kernel-span probe: one VRR/ket clock pair per (group, ket
            // primitive pair). The VRR span opens here, at the top of the
            // primitive-pair body, so it covers the block zeroing below as
            // well as the recurrence loop; the ket span opens where the VRR
            // span closes.
            if (timed)
            {
                span.primPasses += 1;
                span.gateSeamCalls += ketSeam ? 1u : 0u;
                vrrStarted = std::chrono::steady_clock::now();
            }

            // Zero the group's pq blocks, then run the VRR per task. The
            // acc blocks are zeroed once per group on the first primitive
            // pair (g == 0): the ket GEMM accumulates over g, and the
            // thread_local scratch persists across kernel calls (the
            // stale-accumulator fix, 2026-08-18).
            for (std::size_t t = groupStart; t < groupEnd; ++t)
            {
                T* pq = scratch.data() + taskBase[t];
                std::fill(pq, pq + mPq * kHermKet, T{});

                if (g == 0)
                {
                    std::fill(scratch.data() + taskAcc[t],
                              scratch.data() + taskAcc[t] + mPq * nFuncsKet,
                              T{});
                }

                const MdQuartetTask& task = batch.tasks[t];
                const MdPairData& bra = pairs[task.braPair];
                double boundAcc = 0.0;

                if (timed)
                {
                    // The recurrence's work denominator, read off the bra
                    // pair the loop below walks - one add per (task,
                    // primitive pair), never one per primitive quadruple.
                    span.vrrQuadruples += bra.primPairs.size();
                }

                for (std::size_t b = 0; b < bra.primPairs.size(); ++b)
                {
                    detail::RunVrrQuadruple<L, LBra, LKet, T>(bra.primPairs[b],
                                                              ket.primPairs[g],
                                                              bra.rowPairs,
                                                              b,
                                                              bra.primPairs.size(),
                                                              kHermBra,
                                                              kHermKet,
                                                              pq,
                                                              kCertified ? &boundAcc : nullptr);
                }

                if constexpr (kCertified)
                {
                    taskBound[t] += boundAcc;
                }
            }

            if (convert.size() < kHermKet * nFuncsKet)
            {
                convert.resize(kHermKet * nFuncsKet);
            }

            T* const pqBase = scratch.data() + taskBase[groupStart];
            T* const accBase = scratch.data() + taskAcc[groupStart];

            if (timed)
            {
                vrrMid = std::chrono::steady_clock::now();
            }

            auto ketTransform = KetTransformBatched<T>(nGroup,
                                                       mPq,
                                                       static_cast<std::size_t>(kHermKet),
                                                       nFuncsKet,
                                                       pqBase,
                                                       ket.ketTransforms[g].data(),
                                                       accBase,
                                                       convert.data());

            if (!ketTransform.has_value())
            {
                return std::unexpected(ketTransform.error());
            }

            if (timed)
            {
                // The pair's spans close together: the VRR span is the body
                // above, the ket span is the one batched GEMM (plus the
                // convert resize that guards it), and the loop's own
                // increment and back-edge belong to neither - the reader
                // finds them in the residual the phase sum leaves against
                // the kernel span.
                const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                span.vrrNanos +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(vrrMid - vrrStarted)
                        .count();
                span.ketNanos +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now - vrrMid).count();
            }
        }

        // The bound for every task of the group, summed over all g. The
        // errorBounds array is zeroed once per batch by the caller
        // (eri_batch.cpp), so this write is the only one per task.
        if constexpr (kCertified)
        {
            for (std::size_t t = groupStart; t < groupEnd; ++t)
            {
                const MdQuartetTask& task = batch.tasks[t];
                const MdPairData& bra = pairs[task.braPair];
                const double cClass = kClassAmplification[LBra][LKet];
                const double rounding = 1.0 + kClassRounding[LBra][LKet];
                batch.errorBounds[batch.boundsBase + t] = kCertifiedEpsilon * cClass * rounding *
                                                          bra.braRowSum * ket.ketColSum *
                                                          taskBound[t];
            }
        }

        groupStart = groupEnd;
    }

    // Pass 2 - the bra transform, one task at a time (shared A per bra
    // pair; the fp32 lane converts T_bra^ctd once per pair per kernel
    // call). Batching across tasks is an open tuning item.
    //
    // The kernel-span probe's bra phase opens here: this one clock pair
    // around the whole task loop, never one per task (a per-task pair would
    // cost more than the loop's own per-task setup). The span therefore
    // covers the loop's per-task bookkeeping - the task/pair reads, the
    // convert resize guard - beside the transform itself.
    if (timed)
    {
        braStarted = std::chrono::steady_clock::now();
    }

    for (std::size_t t = 0; t < nTasks; ++t)
    {
        const MdQuartetTask& task = batch.tasks[t];
        const MdPairData& bra = pairs[task.braPair];
        const MdPairData& ket = pairs[task.ketPair];
        const std::size_t mBra = bra.nFuncs;
        const std::size_t kBra =
            bra.rowPairs * bra.primPairs.size() * static_cast<std::size_t>(kHermBra);
        T* const acc = scratch.data() + taskAcc[t];
        T* const blockOut = out + task.outputOffset;

        if (timed && !MicroGemmEligible(mBra, kBra, ket.nFuncs))
        {
            span.gateSeamCalls += 1;
        }

        if (convert.size() < mBra * kBra)
        {
            convert.resize(mBra * kBra);
        }

        auto braTransform = BraTransformSingle<T>(
            mBra, kBra, ket.nFuncs, bra.braTransform.data(), acc, blockOut, convert.data());

        if (!braTransform.has_value())
        {
            return std::unexpected(braTransform.error());
        }
    }

    if (timed)
    {
        span.braNanos += std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - braStarted)
                             .count();
    }

    return {};
}

/// The fp64 class kernel (the dispatch-table entry).
template <int LBra, int LKet> qcx::Result<void> ComputeEriClass(const MdClassBatch& batch) {
    return ComputeEriClassImpl<LBra, LKet, double>(batch);
}

/// The certified fp32 class kernel (the dispatch-table entry).
template <int LBra, int LKet> qcx::Result<void> ComputeEriClassF32(const MdClassBatch& batch) {
    return ComputeEriClassImpl<LBra, LKet, float>(batch);
}

} // namespace qcx::integrals::internal
