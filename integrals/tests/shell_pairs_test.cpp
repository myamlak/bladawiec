// Shell-pair list construction: the flattened shells (function offsets,
// counts, angular momenta), the canonical pair list, the pair-index
// formula, and the error paths.

#include "h2_sto3g.hpp"
#include "hf_sto3g.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <gtest/gtest.h>

namespace {

std::size_t FunctionCount(const qcx::integrals::ShellInfo& shell) {
    const std::size_t angular = shell.isSpherical
                                    ? static_cast<std::size_t>(2 * shell.angularMomentum + 1)
                                    : static_cast<std::size_t>((shell.angularMomentum + 1) *
                                                               (shell.angularMomentum + 2) / 2);
    return shell.contractionCount * angular;
}

TEST(ShellPairsTest, H2Sto3g) {
    auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    EXPECT_EQ(pairList->shells.size(), 2u);
    EXPECT_EQ(pairList->functionCount, 2u);
    EXPECT_EQ(pairList->pairs.size(), 3u);
    EXPECT_EQ(pairList->shells[0].angularMomentum, 0);
    EXPECT_EQ(pairList->shells[0].functionOffset, 0u);
    EXPECT_EQ(pairList->shells[1].functionOffset, 1u);
    EXPECT_EQ(pairList->pairs[0].i, 0u);
    EXPECT_EQ(pairList->pairs[0].j, 0u);
    EXPECT_EQ(pairList->pairs[1].i, 0u);
    EXPECT_EQ(pairList->pairs[1].j, 1u);
    EXPECT_EQ(pairList->pairs[2].i, 1u);
    EXPECT_EQ(pairList->pairs[2].j, 1u);
    EXPECT_EQ(qcx::integrals::PairIndexOf(0, 1, *pairList), 1u);
}

TEST(ShellPairsTest, HfSto3g) {
    auto basis = qcx::testing::MakeHfSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeHfSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    // The (Z, x, y, z) renumbering puts H first: [H-s, F-1s, F-2s,
    // F-2p] = 6 functions over 4 shells.
    EXPECT_EQ(pairList->shells.size(), 4u);
    EXPECT_EQ(pairList->functionCount, 6u);
    EXPECT_EQ(pairList->pairs.size(), 10u);
    EXPECT_EQ(pairList->shells[0].atomIndex, 0u);
    EXPECT_EQ(pairList->shells[0].angularMomentum, 0);
    EXPECT_EQ(pairList->shells[2].angularMomentum, 0);
    EXPECT_EQ(pairList->shells[3].angularMomentum, 1);
    EXPECT_TRUE(pairList->shells[3].isSpherical);
    EXPECT_EQ(FunctionCount(pairList->shells[3]), 3u);

    // The canonical pairs cover the upper triangle in pair-index order.
    std::size_t expected = 0;

    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = i; j < 4; ++j)
        {
            ASSERT_LT(expected, pairList->pairs.size());
            EXPECT_EQ(pairList->pairs[expected].i, i);
            EXPECT_EQ(pairList->pairs[expected].j, j);
            EXPECT_EQ(qcx::integrals::PairIndexOf(i, j, *pairList), expected);
            ++expected;
        }
    }

    EXPECT_EQ(expected, pairList->pairs.size());
}

TEST(ShellPairsTest, MissingElementIsInvalidArgument) {
    auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeHeAtom();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_FALSE(pairList.has_value());
    EXPECT_EQ(pairList.error().code, qcx::ErrorCode::kInvalidArgument);
}

} // namespace
