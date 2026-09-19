// The VRR scatter table (md_scatter_table.hpp): the
// producer/consumer contract that lets the kernel replace its (p~, t)
// enumeration with a table walk.
//
// The kernel's substitution is only safe if the table says EXACTLY what the
// enumeration it replaced said - same cells, exactly once each, each fed from
// the same slice slot - because a table that missed a cell would leave a stale
// value in pq and the ket transform would read it as data. The build refuses to
// produce such a table, and these tests re-derive the enumeration here, in the
// test, and compare the two as sets: a build that checked itself against its own
// closed form would not have caught a wrong src mapping.
//
// The classes covered are the ones the lean path runs (l <= 2 in both indices)
// plus the l = 3 corner, which is inside kMaxVrrScatterCells.

#include "internal/md_scatter_table.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <set>
#include <utility>
#include <vector>

namespace {

namespace md = qcx::integrals::internal;

/// The (src, dst) pairs the KERNEL's enumeration produces for one class,
/// written out independently of the table - the reference the table must
/// reproduce. Mirrors RunVrrQuadruple's loops verbatim.
template <int LBra, int LKet> std::set<std::pair<std::int32_t, std::int32_t>> EnumerateCells() {
    constexpr int kHermBra = md::Hermite3DCount(LBra);
    constexpr int kHermKet = md::Hermite3DCount(LKet);
    constexpr int kL = LBra + LKet;
    std::set<std::pair<std::int32_t, std::int32_t>> cells;

    for (int sliceT = 1; sliceT <= kL; ++sliceT)
    {
        const int off = md::kH2Prefix[sliceT];
        const int pLo = sliceT - LKet > 0 ? sliceT - LKet : 0;
        const int pHi = LBra < sliceT ? LBra : sliceT;

        for (int ty = 0; ty <= sliceT; ++ty)
        {
            for (int tz = 0; tz <= sliceT - ty; ++tz)
            {
                const int tx = sliceT - ty - tz;
                const int src = off + md::SubIndex3(ty, tz, sliceT);

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

                            cells.insert({src,
                                          md::Hermite3DIndex(px, py, pz) * kHermKet +
                                              md::Hermite3DIndex(qx, qy, qz)});
                        }
                    }
                }
            }
        }
    }

    return cells;
}

/// The slice T a destination offset belongs to, recovered from the tier
/// prefix: the kernel never needs this, the test does, to check that a group
/// holds the cells of its OWN slice and no others.
// (lBraMax, lKetMax) is the bra-then-ket angular-momentum order.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int SliceOfDst(std::int32_t dst, int kHermKet, int lBraMax, int lKetMax) {
    const int hp = dst / kHermKet;
    const int hq = dst % kHermKet;
    int ps = 0;

    while (ps + 1 <= lBraMax && md::kH2Prefix[ps + 1] <= hp)
    {
        ++ps;
    }

    int qs = 0;

    while (qs + 1 <= lKetMax && md::kH2Prefix[qs + 1] <= hq)
    {
        ++qs;
    }

    return ps + qs;
}

template <int LBra, int LKet> void CheckClass() {
    constexpr int kHermBra = md::Hermite3DCount(LBra);
    constexpr int kHermKet = md::Hermite3DCount(LKet);
    constexpr int kL = LBra + LKet;
    constexpr bool kTabled = md::kVrrScatterTabled<LBra, LKet>;
    // Every class these tests cover is inside kMaxVrrScatterCells.
    ASSERT_TRUE(kTabled) << "class (" << LBra << ", " << LKet << ") is not tabled";

    const auto& table = md::kVrrScatterCells<LBra, LKet>;
    const auto& begin = md::kVrrScatterBegin<LBra, LKet>;
    const auto expected = EnumerateCells<LBra, LKet>();

    // The table has one entry per cell of the block but the sliceT = 0 corner,
    // which the kernel writes from the seeds.
    EXPECT_EQ(table.size(), static_cast<std::size_t>(kHermBra * kHermKet) - 1);
    EXPECT_EQ(expected.size(), table.size());
    EXPECT_EQ(begin[0], 0);
    EXPECT_EQ(begin[1], 0);
    // The groups tile the whole block but its corner. VrrSliceCells is defined
    // for slice T >= 1: slice 0 is the seeds write the kernel does itself, and
    // contributes no entry.
    EXPECT_EQ(begin[static_cast<std::size_t>(kL) + 1], kHermBra * kHermKet - 1);

    for (int sliceT = 1; sliceT <= kL; ++sliceT)
    {
        // The group of slice T holds exactly the cells of slice T.
        EXPECT_EQ(begin[static_cast<std::size_t>(sliceT) + 1] -
                      begin[static_cast<std::size_t>(sliceT)],
                  (md::VrrSliceCells<LBra, LKet>(sliceT)))
            << "slice " << sliceT;

        std::int32_t previous = -1;

        for (std::int32_t e = begin[static_cast<std::size_t>(sliceT)];
             e < begin[static_cast<std::size_t>(sliceT) + 1];
             ++e)
        {
            const md::VrrScatterCell& cell = table[static_cast<std::size_t>(e)];
            // The source lives in this slice's tier block...
            EXPECT_GE(cell.src, md::kH2Prefix[sliceT]);
            EXPECT_LT(cell.src, md::kH2Prefix[sliceT + 1]);
            // ...the destination belongs to this slice...
            EXPECT_EQ(SliceOfDst(cell.dst, kHermKet, LBra, LKet), sliceT);
            // ...and the group walks destinations in ascending order, which is
            // what gives the kernel a monotone store stream.
            EXPECT_GT(cell.dst, previous);
            previous = cell.dst;
        }
    }

    // The whole table reproduces the enumeration exactly: same pairs, no more,
    // no fewer. This is the producer/consumer check - the table build checks
    // its counts against a closed form, which would not catch a wrong src.
    std::set<std::pair<std::int32_t, std::int32_t>> actual;

    for (const md::VrrScatterCell& cell : table)
    {
        actual.insert({cell.src, cell.dst});
    }

    EXPECT_EQ(actual.size(), table.size()) << "the table repeats a (src, dst) pair";
    EXPECT_EQ(actual, expected);
}

} // namespace

TEST(MdScatterTableTest, ReproducesTheEnumerationForTheLeanClasses) {
    CheckClass<0, 0>();
    CheckClass<0, 1>();
    CheckClass<1, 0>();
    CheckClass<0, 2>();
    CheckClass<1, 1>();
    CheckClass<2, 0>();
    CheckClass<1, 2>();
    CheckClass<2, 1>();
    CheckClass<2, 2>();
}

TEST(MdScatterTableTest, ReproducesTheEnumerationForTheHigherCorners) {
    CheckClass<0, 3>();
    CheckClass<3, 0>();
    CheckClass<1, 3>();
    CheckClass<3, 1>();
    CheckClass<2, 3>();
    CheckClass<3, 2>();
    CheckClass<3, 3>();
}

TEST(MdScatterTableTest, TheBoundCoversTheLeanClassesAndRejectsTheCorners) {
    EXPECT_TRUE((md::kVrrScatterTabled<0, 0>));
    EXPECT_TRUE((md::kVrrScatterTabled<2, 2>));
    EXPECT_TRUE((md::kVrrScatterTabled<3, 3>));
    // l = 6 (Hermite3DCount = 84) is 7056 cells, past the bound: the kernel
    // keeps its enumeration there, so the table must not be materialized.
    EXPECT_FALSE((md::kVrrScatterTabled<6, 6>));
    EXPECT_FALSE((md::kVrrScatterTabled<12, 12>));
}
