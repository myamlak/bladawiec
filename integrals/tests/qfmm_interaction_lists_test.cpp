// The θ well-separatedness test and the interaction lists of the QFMM
// octree: completeness (every unordered pair of shell pairs appears in
// exactly one list) and θ monotonicity (a stricter θ never grows the
// far-field list).

#include "internal/qfmm_geometry.hpp"
#include "internal/qfmm_tree.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <set>
#include <utility>
#include <vector>

namespace {

using qcx::integrals::internal::BuildInteractionLists;
using qcx::integrals::internal::BuildQfmmTree;
using qcx::integrals::internal::IsWellSeparated;
using qcx::integrals::internal::QfmmPairGeometry;
using qcx::integrals::internal::QfmmTreeBuildResult;
using qcx::integrals::internal::QfmmTreeNode;

/// A 4 x 4 x 4 grid of pair geometries at spacing 2.0, zero extent (see
/// qfmm_tree_test.cpp): 64 points on [0, 6]^3 - with the default leaf cap
/// the root's box is the full [0, 6]^3 AABB, and each octant child's box
/// is ITS points' AABB, the 2 x 2 x 2 corner block [0, 2]^3, so the tree
/// is the root plus 8 leaves at the centers {1, 5}^3 with half-width 1.
/// Leaf centers sit 4.0 apart along an axis, 4 sqrt 2 face-diagonal and
/// 4 sqrt 3 body-diagonal; two leaves are well-separated when distance *
/// theta >= 2, so at theta = 0.3 only the 4 body-diagonal pairs are far,
/// at 0.4 the 12 face-diagonal pairs join them, and at 0.5 the 12 edge
/// pairs join too - clean hand-checkable far-field counts.
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

/// The leaf nodes under \p nodeIndex (itself included when a leaf).
std::set<std::size_t> LeafDescendants(const QfmmTreeBuildResult& tree, std::size_t nodeIndex) {
    if (tree.nodes[nodeIndex].isLeaf)
    {
        return {nodeIndex};
    }

    std::set<std::size_t> leaves;

    for (const std::size_t child : tree.children[nodeIndex])
    {
        const auto childLeaves = LeafDescendants(tree, child);
        leaves.insert(childLeaves.begin(), childLeaves.end());
    }

    return leaves;
}

/// Expands the interaction lists into the covered unordered shell-pair
/// pairs (P, Q), P <= Q: a far-field node pair (A, B) covers every pair
/// across A's and B's leaf descendants; a near-field leaf pair covers its
/// own function-pair pairs (the leaf with itself included - shell-pair
/// pairs INSIDE one leaf are always near field).
std::set<std::pair<std::size_t, std::size_t>> CoveredPairPairs(
    const QfmmTreeBuildResult& tree,
    // (farFieldPairs, nearFieldLeafPairs) are the two list kinds - the
    // far field and the near field, expanded identically.
    //
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
    const std::vector<std::pair<std::size_t, std::size_t>>& nearFieldLeafPairs) {
    std::set<std::pair<std::size_t, std::size_t>> covered;

    for (const auto& [a, b] : farFieldPairs)
    {
        const auto leavesA = LeafDescendants(tree, a);
        const auto leavesB = LeafDescendants(tree, b);

        for (const std::size_t leafA : leavesA)
        {
            for (const std::size_t leafB : leavesB)
            {
                for (const std::size_t p : tree.nodes[leafA].pairIndices)
                {
                    for (const std::size_t q : tree.nodes[leafB].pairIndices)
                    {
                        covered.emplace(std::min(p, q), std::max(p, q));
                    }
                }
            }
        }
    }

    for (const auto& [leafA, leafB] : nearFieldLeafPairs)
    {
        const std::vector<std::size_t>& pairsA = tree.nodes[leafA].pairIndices;
        const std::vector<std::size_t>& pairsB = tree.nodes[leafB].pairIndices;

        for (const std::size_t p : pairsA)
        {
            for (const std::size_t q : pairsB)
            {
                covered.emplace(std::min(p, q), std::max(p, q));
            }
        }
    }

    return covered;
}

TEST(QfmmInteractionListsTest, GridCompletenessAtEveryTheta) {
    const auto tree = BuildQfmmTree(MakeGridGeometries());
    ASSERT_TRUE(tree.has_value());
    // 64 shell pairs: 64 * 65 / 2 = 2080 unordered pairs, self pairs
    // included.
    constexpr std::size_t kPairCount = 2080;

    // Hand-computed far-field structure of the grid tree (leaf cap 8,
    // zero extents): the root's box is the full [0, 6]^3 AABB, and each
    // octant's child box is ITS points' AABB - the 2 x 2 x 2 block in the
    // corner, [0, 2]^3 - so the tree is the root plus 8 leaves at the
    // centers {1, 5}^3 with half-width 1. Two leaves are well-separated
    // when distance * theta >= 2: the 12 edge pairs (d = 4), the 12
    // face-diagonal pairs (d = 4 sqrt 2), and the 4 body-diagonal pairs
    // (d = 4 sqrt 3). The far pair-pair coverage is 64 per node pair
    // (8 x 8 points); the self pairs are never far (the traversal's
    // (node, node) descent makes them near by construction).
    struct Expectation {
        double theta;
        std::size_t farNodePairs;
        std::size_t farCoveredPairs;
    };

    const Expectation expectations[] = {
        {0.2, 0, 0},
        {0.3, 4, std::size_t{4} * 64},
        {0.4, 12 + 4, std::size_t{16} * 64},
        {0.5, 12 + 12 + 4, std::size_t{28} * 64},
        {0.8, 12 + 12 + 4, std::size_t{28} * 64},
    };

    for (const Expectation& expectation : expectations)
    {
        std::vector<std::pair<std::size_t, std::size_t>> farFieldPairs;
        std::vector<std::pair<std::size_t, std::size_t>> nearFieldLeafPairs;
        BuildInteractionLists(*tree, expectation.theta, farFieldPairs, nearFieldLeafPairs);

        EXPECT_EQ(farFieldPairs.size(), expectation.farNodePairs)
            << "theta = " << expectation.theta;

        const auto covered = CoveredPairPairs(*tree, farFieldPairs, nearFieldLeafPairs);
        EXPECT_EQ(covered.size(), kPairCount) << "theta = " << expectation.theta;
    }
}

TEST(QfmmInteractionListsTest, ThetaZeroIsEverythingNearField) {
    const auto tree = BuildQfmmTree(MakeGridGeometries());
    ASSERT_TRUE(tree.has_value());

    std::vector<std::pair<std::size_t, std::size_t>> farFieldPairs;
    std::vector<std::pair<std::size_t, std::size_t>> nearFieldLeafPairs;
    BuildInteractionLists(*tree, 0.0, farFieldPairs, nearFieldLeafPairs);

    EXPECT_TRUE(farFieldPairs.empty());
    // The θ→0 exact-recovery gate: everything near field, nothing missing.
    EXPECT_EQ(CoveredPairPairs(*tree, farFieldPairs, nearFieldLeafPairs).size(), 2080u);
}

TEST(QfmmInteractionListsTest, FarFieldShrinksMonotonicallyWithStricterTheta) {
    const auto tree = BuildQfmmTree(MakeGridGeometries());
    ASSERT_TRUE(tree.has_value());

    // The far-field LEAF-PAIR partition at one θ must be a subset of the
    // partition at every LOOSER θ (a pair that is far field at a stricter
    // θ is certainly far field at a looser one - the ancestor pair that
    // separated it is still well-separated, and the looser traversal finds
    // it at the same or a higher level). Check the full ordering from
    // LOOSE to TIGHT, comparing each partition against the looser one.
    const double thetas[] = {0.9, 0.8, 0.7, 0.5, 0.3};
    std::set<std::pair<std::size_t, std::size_t>> previous;

    for (const double theta : thetas)
    {
        std::vector<std::pair<std::size_t, std::size_t>> farFieldPairs;
        std::vector<std::pair<std::size_t, std::size_t>> nearFieldLeafPairs;
        BuildInteractionLists(*tree, theta, farFieldPairs, nearFieldLeafPairs);

        // The covered far-field pairs ARE the far-field leaf-pair
        // partition: a far-field node pair is never covered by another
        // entry (the traversal stops at the first well-separated pair).
        const auto farPartition = CoveredPairPairs(*tree, farFieldPairs, {});

        for (const auto& pair : farPartition)
        {
            EXPECT_TRUE(previous.empty() || previous.count(pair) != 0u)
                << "pair (" << pair.first << ", " << pair.second
                << ") far field at theta = " << theta << " but not at a looser theta";
        }

        previous = farPartition;
    }
}

TEST(QfmmInteractionListsTest, NodeIsNeverWellSeparatedFromItself) {
    const auto tree = BuildQfmmTree(MakeGridGeometries());
    ASSERT_TRUE(tree.has_value());

    for (const QfmmTreeNode& node : tree->nodes)
    {
        EXPECT_FALSE(IsWellSeparated(node, node, 0.9));
        EXPECT_FALSE(IsWellSeparated(node, node, 0.0));
    }
}

TEST(QfmmInteractionListsTest, TouchingBoxesAreNeverFarFieldBelowThetaOne) {
    const auto tree = BuildQfmmTree(MakeGridGeometries());
    ASSERT_TRUE(tree.has_value());

    // The sanity invariant: boxes that touch or overlap are never
    // far field. In the grid tree the root's box [0, 6]^3 contains the
    // leaves' [0, 2]^3 / [4, 6]^3 boxes, so every (root, leaf) pair has a
    // center distance of 2 sqrt 3 <= wA + wB = 4 - and distance * theta <=
    // distance < wA + wB for theta <= 1, so no touching pair can pass the
    // well-separatedness test. (The edge-adjacent leaves, d = 4 vs wA +
    // wB = 2, are NOT touching and ARE far field at theta >= 0.5.)
    for (std::size_t a = 0; a < tree->nodes.size(); ++a)
    {
        for (std::size_t b = a; b < tree->nodes.size(); ++b)
        {
            if (a == b)
            {
                continue;
            }

            const QfmmTreeNode& nodeA = tree->nodes[a];
            const QfmmTreeNode& nodeB = tree->nodes[b];
            const double dx = nodeA.centerX - nodeB.centerX;
            const double dy = nodeA.centerY - nodeB.centerY;
            const double dz = nodeA.centerZ - nodeB.centerZ;
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (distance <= nodeA.halfWidth + nodeB.halfWidth + 1e-12)
            {
                EXPECT_FALSE(IsWellSeparated(nodeA, nodeB, 0.9));
            }
        }
    }
}

} // namespace
