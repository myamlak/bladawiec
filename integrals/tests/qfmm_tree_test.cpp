// Octree construction of the QFMM octree: every shell pair lands in
// exactly one leaf, every node's
// box contains its children's boxes, the children adjacency is a genuine
// forest (BuildSparsityFromTree accepts it), and a real molecule's tree is
// structurally sane.

#include "h2o_sto3g.hpp"
#include "internal/md_batch.hpp"
#include "internal/qfmm_geometry.hpp"
#include "internal/qfmm_tree.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/sparsity_pattern.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <set>
#include <utility>
#include <vector>

namespace {

using qcx::integrals::AccuracyPreset;
using qcx::integrals::QfmmExtentForPreset;
using qcx::integrals::internal::BuildQfmmTree;
using qcx::integrals::internal::ComputePairGeometries;
using qcx::integrals::internal::kQfmmMaxLeafSize;
using qcx::integrals::internal::kQfmmMaxTreeDepth;
using qcx::integrals::internal::kQfmmMinBoxHalfWidth;
using qcx::integrals::internal::QfmmPairGeometry;
using qcx::integrals::internal::QfmmTreeBuildResult;
using qcx::integrals::internal::QfmmTreeNode;

/// A 4 x 4 x 4 grid of pair geometries at spacing 2.0, zero extent: 64
/// points on [0, 6]^3 - with the default leaf cap of 8 the root splits
/// into its 8 octants, each holding one 2 x 2 x 2 block of points.
std::vector<QfmmPairGeometry> MakeGridGeometries() {
    std::vector<QfmmPairGeometry> geometries;

    for (int iz = 0; iz < 4; ++iz)
    {
        for (int iy = 0; iy < 4; ++iy)
        {
            for (int ix = 0; ix < 4; ++ix)
            {
                geometries.push_back(QfmmPairGeometry{2.0 * ix, 2.0 * iy, 2.0 * iz, 0.0});
            }
        }
    }

    return geometries;
}

/// Sums the leaf pair counts and collects every pair index, to verify the
/// exactly-one-leaf invariant.
std::pair<std::size_t, std::set<std::size_t>> LeafCoverage(const QfmmTreeBuildResult& tree) {
    std::size_t total = 0;
    std::set<std::size_t> seen;

    for (const QfmmTreeNode& node : tree.nodes)
    {
        if (!node.isLeaf)
        {
            continue;
        }

        total += node.pairIndices.size();
        seen.insert(node.pairIndices.begin(), node.pairIndices.end());
    }

    return {total, seen};
}

TEST(QfmmTreeTest, EveryPairInExactlyOneLeaf) {
    const auto result = BuildQfmmTree(MakeGridGeometries());
    ASSERT_TRUE(result.has_value());
    const auto [total, seen] = LeafCoverage(*result);

    EXPECT_EQ(total, 64u);
    EXPECT_EQ(seen.size(), 64u);

    // leafOfPair is consistent with the leaf contents.
    for (std::size_t pairIndex = 0; pairIndex < 64; ++pairIndex)
    {
        const std::size_t leaf = result->leafOfPair[pairIndex];
        ASSERT_LT(leaf, result->nodes.size());
        EXPECT_TRUE(result->nodes[leaf].isLeaf);

        const auto& indices = result->nodes[leaf].pairIndices;
        EXPECT_NE(std::find(indices.begin(), indices.end(), pairIndex), indices.end());
    }
}

TEST(QfmmTreeTest, EveryChildBoxIsContainedInItsParent) {
    const auto result = BuildQfmmTree(MakeGridGeometries());
    ASSERT_TRUE(result.has_value());

    // The child cube [ccx - ccw, ccx + ccw]^3 lies inside the parent cube
    // [cx - w, cx + w]^3 exactly when |cc - c| + ccw <= w per axis; the
    // child AABB (NodeBox) is a subset of the parent's, so the cubes are
    // contained; the tolerance absorbs only the square-root rounding.
    constexpr double kTolerance = 1e-9;

    for (std::size_t parent = 0; parent < result->nodes.size(); ++parent)
    {
        for (const std::size_t child : result->children[parent])
        {
            const QfmmTreeNode& p = result->nodes[parent];
            const QfmmTreeNode& c = result->nodes[child];
            EXPECT_LE(std::abs(c.centerX - p.centerX) + c.halfWidth, p.halfWidth + kTolerance);
            EXPECT_LE(std::abs(c.centerY - p.centerY) + c.halfWidth, p.halfWidth + kTolerance);
            EXPECT_LE(std::abs(c.centerZ - p.centerZ) + c.halfWidth, p.halfWidth + kTolerance);
        }
    }
}

TEST(QfmmTreeTest, ChildrenListIsAGenuineForest) {
    const auto result = BuildQfmmTree(MakeGridGeometries());
    ASSERT_TRUE(result.has_value());

    // BuildSparsityFromTree validates the adjacency (no cycles, no
    // two-parent node, no self-child) - a structural proof, not just a
    // by-eye inspection.
    const auto pattern = qcx::memory::BuildSparsityFromTree<qcx::backend::CpuTag>(result->children);
    ASSERT_TRUE(pattern.has_value());
    EXPECT_EQ(pattern->RowCount(), result->nodes.size());
}

TEST(QfmmTreeTest, RootBoxContainsEveryPaddedPair) {
    const auto result = BuildQfmmTree(MakeGridGeometries());
    ASSERT_TRUE(result.has_value());
    const QfmmTreeNode& root = result->nodes.front();

    // Every geometry center plus its extent stays inside the root cube.
    for (const QfmmPairGeometry& geometry : MakeGridGeometries())
    {
        EXPECT_LE(std::abs(geometry.centerX - root.centerX) + geometry.extent,
                  root.halfWidth + 1e-9);
        EXPECT_LE(std::abs(geometry.centerY - root.centerY) + geometry.extent,
                  root.halfWidth + 1e-9);
        EXPECT_LE(std::abs(geometry.centerZ - root.centerZ) + geometry.extent,
                  root.halfWidth + 1e-9);
    }
}

TEST(QfmmTreeTest, DegenerateInputsAreRejected) {
    EXPECT_FALSE(BuildQfmmTree({}).has_value());
    EXPECT_FALSE(BuildQfmmTree(MakeGridGeometries(), 0, kQfmmMaxLeafSize).has_value());
    EXPECT_FALSE(BuildQfmmTree(MakeGridGeometries(), kQfmmMaxLeafSize, 0).has_value());
}

TEST(QfmmTreeTest, CoincidentPointsChainToTheDepthCap) {
    // 16 geometries at the SAME center, extent 0.5: the root box is the
    // padded AABB [0.5, 1.5]^3 (half-width 0.5), every point lands in the
    // same octant each level, and each child's box is ITS OWN padded AABB
    // - which never shrinks (the coincident centers stay at the same
    // point), so the recursion would be infinite without the depth cap,
    // which stops it with one overfull leaf at depth kQfmmMaxTreeDepth
    // (the documented degenerate-input safety valve). The min-width floor
    // never binds: the extent-padded box stays at 0.5.
    std::vector<QfmmPairGeometry> coincident(16, QfmmPairGeometry{1.0, 1.0, 1.0, 0.5});
    const auto result = BuildQfmmTree(coincident);
    ASSERT_TRUE(result.has_value());

    // One node per level: the chain from the root to the depth cap.
    EXPECT_EQ(result->nodes.size(), kQfmmMaxTreeDepth + 1);

    const QfmmTreeNode& leaf = result->nodes.back();
    EXPECT_TRUE(leaf.isLeaf);
    EXPECT_EQ(leaf.pairIndices.size(), 16u);
    // The chain hit the DEPTH cap (20 splits): the leaf half-width is
    // still the full padded extent (the box never shrinks).
    EXPECT_NEAR(leaf.halfWidth, 0.5, 1e-15);

    // Every internal node of the chain split into exactly one child.
    for (std::size_t node = 0; node < result->nodes.size() - 1; ++node)
    {
        EXPECT_EQ(result->children[node].size(), 1u);
    }
}

TEST(QfmmTreeTest, WaterSto3gTreeIsStructurallySane) {
    // The "build it for a real molecule and inspect by hand" check:
    // H2O/STO-3G - 7 shells, 28 canonical shell pairs. Tiny molecule, but
    // with the padded extents (the O 2p pair extents are ~13 Bohr) the
    // root still splits; every pair must end up in exactly one leaf.
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value());

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto pairStore = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);
    ASSERT_TRUE(pairStore.has_value());

    const auto geometries =
        ComputePairGeometries(*pairStore, QfmmExtentForPreset(AccuracyPreset::kTight));
    const auto result = BuildQfmmTree(geometries);
    ASSERT_TRUE(result.has_value());

    const auto [total, seen] = LeafCoverage(*result);
    EXPECT_EQ(total, pairList->pairs.size());
    EXPECT_EQ(seen.size(), pairList->pairs.size());

    for (std::size_t pairIndex = 0; pairIndex < pairList->pairs.size(); ++pairIndex)
    {
        EXPECT_LT(result->leafOfPair[pairIndex], result->nodes.size());
    }

    const auto pattern = qcx::memory::BuildSparsityFromTree<qcx::backend::CpuTag>(result->children);
    ASSERT_TRUE(pattern.has_value());
}

} // namespace
