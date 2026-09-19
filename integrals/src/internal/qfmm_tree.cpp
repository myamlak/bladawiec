// The QFMM octree and its interaction lists: top-down
// bounding-box bisection, the θ well-separatedness test, and the dual-tree
// traversal. Standard computational geometry - no multipole math in this
// file.

#include "internal/qfmm_tree.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

namespace qcx::integrals::internal {

namespace {

/// The recursion state of one BuildQfmmTree call: the geometry input, the
/// nodes/children under construction, and the leaf assignment.
struct BuildState {
    const std::vector<QfmmPairGeometry>* geometries = nullptr;
    std::vector<QfmmTreeNode> nodes;
    std::vector<std::vector<std::size_t>> children;
    std::vector<std::size_t> leafOfPair;
};

/// The cubed axis-aligned bounding box of the shell pairs' centers padded
/// by their extents - the computation applied at EVERY
/// level, so each node's box contains the padded extent of every pair in
/// it and the θ test's box geometry alone is honest (the
/// premise). A raw octant would violate it: a pair's cloud straddles its
/// box, and coincident same-shell pairs (every (i, i) pair of an atom
/// shares its center) would chain to the depth cap with vanishingly small
/// boxes - "well-separated" from adjacent atoms at any θ despite 7-13 Bohr
/// clouds 2.9 Bohr apart, silently destroying the accuracy of the largest
/// J blocks. Returns {centerX, centerY, centerZ, halfWidth}.
std::array<double, 4> NodeBox(const std::vector<QfmmPairGeometry>& geometries,
                              const std::vector<std::size_t>& pairIndices) {
    double minX = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    double minZ = std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();

    for (const std::size_t pairIndex : pairIndices)
    {
        const QfmmPairGeometry& geometry = geometries[pairIndex];
        minX = std::min(minX, geometry.centerX - geometry.extent);
        maxX = std::max(maxX, geometry.centerX + geometry.extent);
        minY = std::min(minY, geometry.centerY - geometry.extent);
        maxY = std::max(maxY, geometry.centerY + geometry.extent);
        minZ = std::min(minZ, geometry.centerZ - geometry.extent);
        maxZ = std::max(maxZ, geometry.centerZ + geometry.extent);
    }

    return {0.5 * (minX + maxX),
            0.5 * (minY + maxY),
            0.5 * (minZ + maxZ),
            0.5 * std::max({maxX - minX, maxY - minY, maxZ - minZ})};
}

/// Builds one node: keeps \p pairIndices as the leaf contents when the
/// split conditions fail, otherwise distributes them over the 8 octants of
/// the bisected box (centers on the bisection planes go to the lower
/// octant, the <= convention), computes each child's box from ITS OWN
/// pairs via NodeBox (the children are not the raw octants), and
/// recurses. Returns the node index.
std::size_t BuildNode(BuildState& state,
                      std::vector<std::size_t> pairIndices,
                      double centerX,
                      double centerY,
                      // (centerZ, halfWidth) are distinct quantities; the single call
                      // site passes named members in fixed order.
                      //
                      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                      double centerZ,
                      double halfWidth,
                      std::size_t depth,
                      std::size_t maxLeafSize,
                      std::size_t maxDepth) {
    const std::size_t nodeIndex = state.nodes.size();
    state.nodes.push_back(QfmmTreeNode{});
    state.children.emplace_back();
    QfmmTreeNode& node = state.nodes.back();
    node.centerX = centerX;
    node.centerY = centerY;
    node.centerZ = centerZ;
    node.halfWidth = halfWidth;

    const bool keepAsLeaf =
        pairIndices.size() <= maxLeafSize || depth >= maxDepth || halfWidth <= kQfmmMinBoxHalfWidth;

    if (keepAsLeaf)
    {
        node.isLeaf = true;
        node.pairIndices = std::move(pairIndices);
        node.radius = 0.0;

        for (const std::size_t pairIndex : node.pairIndices)
        {
            state.leafOfPair[pairIndex] = nodeIndex;
            const QfmmPairGeometry& geometry = (*state.geometries)[pairIndex];
            const double dx = geometry.centerX - node.centerX;
            const double dy = geometry.centerY - node.centerY;
            const double dz = geometry.centerZ - node.centerZ;
            node.radius =
                std::max(node.radius, std::sqrt(dx * dx + dy * dy + dz * dz) + geometry.extent);
        }

        return nodeIndex;
    }

    node.isLeaf = false;

    // Octant layout: bit 2 = +x, bit 1 = +y, bit 0 = +z (octant 0 = the
    // -x,-y,-z corner). The bisection planes sit at the box center.
    std::array<std::vector<std::size_t>, 8> octants;

    for (const std::size_t pairIndex : pairIndices)
    {
        const QfmmPairGeometry& geometry = (*state.geometries)[pairIndex];
        const std::size_t octant = (geometry.centerX <= centerX ? 0u : 4u) |
                                   (geometry.centerY <= centerY ? 0u : 2u) |
                                   (geometry.centerZ <= centerZ ? 0u : 1u);
        octants[octant].push_back(pairIndex);
    }

    for (std::size_t octant = 0; octant < 8; ++octant)
    {
        if (octants[octant].empty())
        {
            continue;
        }

        // The child's box comes from ITS pairs (NodeBox), not the octant
        // geometry: the child AABB lies inside the parent's (a subset of
        // the pairs), so the containment and the honest-box properties
        // hold at every level.
        const auto [childCenterX, childCenterY, childCenterZ, childHalf] =
            NodeBox(*state.geometries, octants[octant]);
        // The child index FIRST, then the push: the recursive call grows
        // state.nodes/state.children, so subscripting state.children before
        // it runs would hand push_back a reference invalidated by the
        // reallocation (the object expression is sequenced before the
        // arguments - a dangling-write crash caught by the grid tree test).
        const std::size_t childIndex = BuildNode(state,
                                                 std::move(octants[octant]),
                                                 childCenterX,
                                                 childCenterY,
                                                 childCenterZ,
                                                 childHalf,
                                                 depth + 1,
                                                 maxLeafSize,
                                                 maxDepth);
        state.children[nodeIndex].push_back(childIndex);
    }

    // The node's charge radius follows its children (they already carry
    // theirs): the covering radius about this node's centre is the furthest
    // point any child's ball reaches. Computed after the children exist -
    // BuildNode sizes state.nodes past this reference, so the radius is
    // written through the index, not the dangling reference.
    double radius = 0.0;

    for (const std::size_t childIndex : state.children[nodeIndex])
    {
        const QfmmTreeNode& child = state.nodes[childIndex];
        const double dx = child.centerX - centerX;
        const double dy = child.centerY - centerY;
        const double dz = child.centerZ - centerZ;
        radius = std::max(radius, std::sqrt(dx * dx + dy * dy + dz * dz) + child.radius);
    }

    state.nodes[nodeIndex].radius = radius;
    return nodeIndex;
}

/// The dual-tree recursion of BuildInteractionLists (documented there).
void RecurseInteraction(const QfmmTreeBuildResult& tree,
                        std::size_t a,
                        std::size_t b,
                        const QfmmSeparation& separation,
                        std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
                        std::vector<std::pair<std::size_t, std::size_t>>& nearFieldLeafPairs) {
    if (a == b)
    {
        const QfmmTreeNode& node = tree.nodes[a];

        if (node.isLeaf)
        {
            nearFieldLeafPairs.emplace_back(a, a);
            return;
        }

        // All unordered pairs of the node's own children, a child with
        // itself included (a node is never well-separated from itself).
        const std::vector<std::size_t>& children = tree.children[a];

        for (std::size_t i = 0; i < children.size(); ++i)
        {
            for (std::size_t j = i; j < children.size(); ++j)
            {
                RecurseInteraction(
                    tree, children[i], children[j], separation, farFieldPairs, nearFieldLeafPairs);
            }
        }

        return;
    }

    if (IsWellSeparated(tree.nodes[a], tree.nodes[b], separation))
    {
        farFieldPairs.emplace_back(a, b);
        return;
    }

    const bool leafA = tree.nodes[a].isLeaf;
    const bool leafB = tree.nodes[b].isLeaf;

    if (leafA && leafB)
    {
        nearFieldLeafPairs.emplace_back(a, b);
        return;
    }

    // Descend into the LARGER node's children (ties: A); a leaf side is
    // never split.
    const bool canSplitA = !leafA;
    const bool splitA = canSplitA && (leafB || tree.nodes[a].halfWidth >= tree.nodes[b].halfWidth);

    if (splitA)
    {
        for (const std::size_t child : tree.children[a])
        {
            RecurseInteraction(tree, child, b, separation, farFieldPairs, nearFieldLeafPairs);
        }
    } else
    {
        for (const std::size_t child : tree.children[b])
        {
            RecurseInteraction(tree, a, child, separation, farFieldPairs, nearFieldLeafPairs);
        }
    }
}

} // namespace

qcx::Result<QfmmTreeBuildResult> BuildQfmmTree(const std::vector<QfmmPairGeometry>& pairGeometries,
                                               std::size_t maxLeafSize,
                                               std::size_t maxDepth) {
    if (pairGeometries.empty())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "a QFMM octree needs at least one shell pair"});
    }

    if (maxLeafSize == 0 || maxDepth == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the QFMM octree leaf-size and depth caps must be >= 1"});
    }

    BuildState state;
    state.geometries = &pairGeometries;
    state.leafOfPair.assign(pairGeometries.size(), 0);

    std::vector<std::size_t> allIndices;
    allIndices.reserve(pairGeometries.size());

    for (std::size_t i = 0; i < pairGeometries.size(); ++i)
    {
        allIndices.push_back(i);
    }

    // The root box: the same NodeBox computation every level uses.
    const auto [centerX, centerY, centerZ, halfWidth] = NodeBox(pairGeometries, allIndices);

    BuildNode(state,
              std::move(allIndices),
              centerX,
              centerY,
              centerZ,
              halfWidth,
              0,
              maxLeafSize,
              maxDepth);

    return QfmmTreeBuildResult{
        std::move(state.nodes), std::move(state.children), std::move(state.leafOfPair)};
}

void BuildInteractionLists(const QfmmTreeBuildResult& tree,
                           const QfmmSeparation& separation,
                           std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
                           std::vector<std::pair<std::size_t, std::size_t>>& nearFieldLeafPairs) {
    // Both lists are OVERWRITTEN: a second call on an already-built tree
    // drops the previous lists (the caller decides whether to rebuild).
    farFieldPairs.clear();
    nearFieldLeafPairs.clear();

    if (tree.nodes.empty())
    {
        return;
    }

    RecurseInteraction(tree, 0, 0, separation, farFieldPairs, nearFieldLeafPairs);
}

void BuildInteractionLists(const QfmmTreeBuildResult& tree,
                           double theta,
                           std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
                           std::vector<std::pair<std::size_t, std::size_t>>& nearFieldLeafPairs) {
    QfmmSeparation separation;
    separation.test = QfmmSeparationTest::kWidthTheta;
    separation.theta = theta;
    BuildInteractionLists(tree, separation, farFieldPairs, nearFieldLeafPairs);
}

bool IsWellSeparated(const QfmmTreeNode& a,
                     const QfmmTreeNode& b,
                     const QfmmSeparation& separation) noexcept {
    if (separation.test == QfmmSeparationTest::kWidthTheta)
    {
        return IsWellSeparated(a, b, separation.theta);
    }

    // The surface-to-surface form: the distance must clear both radii plus a
    // buffer proportional to the larger one (the multipole error bound's
    // demand). k < 0 is the degenerate gate - nothing is well separated,
    // the theta -> 0 analogue.
    if (separation.k < 0.0)
    {
        return false;
    }

    const double dx = a.centerX - b.centerX;
    const double dy = a.centerY - b.centerY;
    const double dz = a.centerZ - b.centerZ;
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    return distance > a.radius + b.radius + separation.k * std::max(a.radius, b.radius);
}

} // namespace qcx::integrals::internal
