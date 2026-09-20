#pragma once

// The VRR scatter table: the pq scatter of
// `RunVrrQuadruple` (md_vrr.hpp), precomputed at compile time.
//
// The scatter writes [p~|q~] = [t]^(0) for every split p~ + q~ = t with
// |p~| <= LBra, |q~| <= LKet. The kernel used to enumerate that split with a
// (ty,tz) sweep and a (ps,px,py) sweep, discarding the combinations with a
// negative q~ component - which is MOST of them: at (LBra, LKet) = (2, 2) the
// enumeration visits 252 (p~, t) combinations to produce the 99 cells of the
// block, i.e. 61% of its iterations are dead. Each visit also paid two
// Hermite3DIndex evaluations (a tier search plus SubIndex3's multiply and
// halving) before it could even test the sign.
//
// The table replaces the enumeration with one entry per cell of the
// (kHermBra x kHermKet) block, grouped by slice T and ordered by ascending
// destination inside a group, so the kernel walks the group of the slice it has
// just built with no branch and no index arithmetic. Two properties make it
// safe rather than merely fast:
//
//  1. Every cell of the block is written exactly once. The sliceT = 0 corner
//     (p~ = q~ = 0) is the seeds[0] write the recurrence already does, so the
//     table holds kHermBra * kHermKet - 1 entries - and the build REFUSES to
//     produce a table that misses or duplicates a cell. A table that missed one
//     would leave a stale value in pq, which the ket transform would read as
//     data: the build is where that cannot survive.
//  2. The arithmetic is untouched. The table carries (source index,
//     destination offset) pairs only; the kernel still does `pq[..] += value`
//     against the zeroed block, one add per cell, so the pq block comes out
//     bit-identical to the enumeration's - the k = 1 pin is preserved by
//     construction, not by inspection. Verified bit-for-bit against the
//     enumeration across five classes and three row-pair counts, sign of zero
//     included.
//
// Measured on the VRR body after the seeds (same probe, 6 bra primitive pairs):
// -40% to -65% per quadruple at (LBra, LKet) from (0,0) to (2,2), the classes a
// def2-SVP lean build runs. The end-to-end verdict is the kernel-span phase
// split of the driver rows.
//
// A NOTE ON THE ONE VARIANT THAT DID NOT WIN, because it is the obvious first
// idea: keeping the enumeration but tightening its loop bounds to the closed
// forms (px <= tx, py <= ty, pz <= tz) removes the dead iterations too, and it
// is bit-identical as well - but it measured 6-36% SLOWER than the current
// kernel, because the per-cell index arithmetic it leaves in place costs more
// than the branches it removes. The win is the arithmetic, not the branches.
//
// WHY THE TABLE IS BOUNDED: the table costs one entry per pq cell, i.e. the
// size of one pq block of read-only data per class, and the dispatch
// instantiates 154 classes up to l = 12 (a (12, 12) table would be 207,024
// entries). kMaxVrrScatterCells bounds it; a class above the bound keeps the
// enumeration, which is correct, just not faster. The bound covers every class
// up to l = 5 in both indices.

#include "md_defs.hpp"
#include "md_tables_gen.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace qcx::integrals::internal {

/// The largest pq block, in cells, for which the scatter table is
/// materialized. A class above it keeps the enumeration.
///
/// The bound is a build-cost bound, not a numerical one: the table is
/// kHermBra * kHermKet entries of read-only data and is built by constant
/// evaluation in every class's translation unit. At 4096 cells the table is
/// 32 KiB and the build is O(cells); the dispatch's largest classes (l = 12,
/// 207,024 cells) are two orders of magnitude past it and gain nothing from a
/// table they could not afford to materialize. Every class up to l = 5 in both
/// indices is inside the bound.
inline constexpr int kMaxVrrScatterCells = 4096;

/// One cell of the VRR pq scatter: where the value comes from and where it
/// goes.
struct VrrScatterCell {
    /// Index into the slice buffer the recurrence has just built for this
    /// slice T.
    std::int32_t src;
    /// Offset of the destination cell inside one (kHermBra x kHermKet) pq
    /// block, i.e. Hermite3DIndex(p~) * kHermKet + Hermite3DIndex(q~).
    std::int32_t dst;
};

/// The number of VRR scatter cells of one slice T: the (p~, q~) Hermite
/// triples with |p~| <= LBra, |q~| <= LKet and |p~| + |q~| = sliceT.
/// \tparam LBra Bra angular momentum of the class.
/// \tparam LKet Ket angular momentum of the class.
/// \param sliceT The slice index, 1..LBra + LKet.
/// \returns The number of cells the slice contributes to the block.
///
/// constexpr rather than consteval: the table build uses it as the group
/// boundary, and the tests use it at run time to check that a group holds the
/// count the closed form predicts.
template <int LBra, int LKet> constexpr int VrrSliceCells(int sliceT) {
    const int pLo = sliceT - LKet > 0 ? sliceT - LKet : 0;
    const int pHi = LBra < sliceT ? LBra : sliceT;
    int total = 0;

    for (int ps = pLo; ps <= pHi; ++ps)
    {
        total += CartesianCount(ps) * CartesianCount(sliceT - ps);
    }

    return total;
}

/// The group boundaries of the scatter table: the cells of slice T occupy
/// [begin[sliceT], begin[sliceT + 1]).
///
/// The groups are what make the table correct and not merely ordered: the
/// recurrence holds one slice at a time and ping-pongs it with the previous
/// one, so scattering a slice's cells anywhere but between the recurrence that
/// built it and the swap that retires it would read a stale tier.
/// \tparam LBra Bra angular momentum of the class.
/// \tparam LKet Ket angular momentum of the class.
/// \returns An array of LBra + LKet + 2 boundaries; entry 0 and 1 are zero
/// because slice T = 0 is the seeds write and contributes no cell.
template <int LBra, int LKet> consteval auto BuildVrrScatterBegin() {
    constexpr int kL = LBra + LKet;
    std::array<std::int32_t, static_cast<std::size_t>(kL) + 2> begin{};

    for (int sliceT = 1; sliceT <= kL; ++sliceT)
    {
        begin[static_cast<std::size_t>(sliceT) + 1] =
            begin[static_cast<std::size_t>(sliceT)] + VrrSliceCells<LBra, LKet>(sliceT);
    }

    return begin;
}

/// The VRR scatter table itself: one entry per cell of the (kHermBra x
/// kHermKet) pq block other than the sliceT = 0 corner, grouped by slice T and
/// ordered by ascending destination inside each group.
///
/// The build walks the DESTINATION cells in ascending order, recovers the
/// (p~, q~) split each one belongs to - the scatter map is a bijection, so this
/// is a decode and not a search - and appends the entry to its slice T group
/// bucket. It then checks, and refuses to produce a table otherwise, that each
/// group holds exactly the count the closed form
/// (VrrSliceCells) predicts and that no destination was written twice. A
/// duplicate or a gap would leave a wrong value in pq that the ket transform
/// would read as data, so the build is where both are caught.
/// \tparam LBra Bra angular momentum of the class.
/// \tparam LKet Ket angular momentum of the class.
/// \returns The table, kHermBra * kHermKet - 1 entries long.
template <int LBra, int LKet> consteval auto BuildVrrScatterCells() {
    constexpr int kHermBra = Hermite3DCount(LBra);
    constexpr int kHermKet = Hermite3DCount(LKet);
    constexpr int kL = LBra + LKet;
    constexpr auto begin = BuildVrrScatterBegin<LBra, LKet>();
    std::array<VrrScatterCell, static_cast<std::size_t>(kHermBra * kHermKet) - 1> cells{};
    // The write cursor of each group, seeded from the closed-form boundaries.
    std::array<std::int32_t, static_cast<std::size_t>(kL) + 2> cursor = begin;

    for (int dst = 1; dst < kHermBra * kHermKet; ++dst)
    {
        // The destination offset encodes (p~, q~): split the row and column
        // halves, then invert Hermite3DIndex on each - the tier is the largest
        // n whose prefix does not exceed the index.
        const int hp = dst / kHermKet;
        const int hq = dst % kHermKet;
        int ps = 0;

        while (ps + 1 <= LBra && kH2Prefix[ps + 1] <= hp)
        {
            ++ps;
        }

        int qs = 0;

        while (qs + 1 <= LKet && kH2Prefix[qs + 1] <= hq)
        {
            ++qs;
        }

        const int si = hp - kH2Prefix[ps];
        const int sj = hq - kH2Prefix[qs];
        // Within the tier the sub-index is SubIndex3(ty, tz, n) =
        // ty * (n + 1) - ty * (ty - 1) / 2 + tz: recover ty as the last one
        // whose tz = 0 sub-index does not exceed it, then read tz off.
        int ty = 0;

        while (ty + 1 <= ps && SubIndex3(ty + 1, 0, ps) <= si)
        {
            ++ty;
        }

        int uy = 0;

        while (uy + 1 <= qs && SubIndex3(uy + 1, 0, qs) <= sj)
        {
            ++uy;
        }

        const int tz = si - SubIndex3(ty, 0, ps);
        const int uz = sj - SubIndex3(uy, 0, qs);
        const int sliceT = ps + qs;

        if (sliceT < 1 || sliceT > kL)
        {
            throw "the VRR scatter table reached a cell outside every slice";
        }

        // The value lives at the tier of t = p~ + q~ inside slice T.
        const int slot = cursor[static_cast<std::size_t>(sliceT)]++;

        cells[static_cast<std::size_t>(slot)] = VrrScatterCell{
            static_cast<std::int32_t>(kH2Prefix[sliceT] + SubIndex3(ty + uy, tz + uz, sliceT)),
            dst};
    }

    for (int sliceT = 1; sliceT <= kL; ++sliceT)
    {
        if (cursor[static_cast<std::size_t>(sliceT)] != begin[static_cast<std::size_t>(sliceT) + 1])
        {
            throw "a VRR scatter slice group does not hold the cells its closed form predicts";
        }
    }

    return cells;
}

/// The VRR scatter table of one class, grouped by slice T.
template <int LBra, int LKet>
inline constexpr auto kVrrScatterCells = BuildVrrScatterCells<LBra, LKet>();

/// The group boundaries of kVrrScatterCells.
template <int LBra, int LKet>
inline constexpr auto kVrrScatterBegin = BuildVrrScatterBegin<LBra, LKet>();

/// Whether this class's scatter is table-driven rather than enumerated.
template <int LBra, int LKet>
inline constexpr bool kVrrScatterTabled =
    Hermite3DCount(LBra) * Hermite3DCount(LKet) <= kMaxVrrScatterCells;

} // namespace qcx::integrals::internal
