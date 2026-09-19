#pragma once

// The QFMM octree: top-down bounding-box
// bisection over the shell-pair expansion centers (padded by each pair's
// bounding extent so nothing pokes outside the root box), plus the
// classical FMM well-separatedness test and the dual-tree traversal that
// builds the near-field/far-field interaction lists. Pure computational
// geometry - no multipole math yet; the θ test and interaction lists are
// the well/far classification that the multipole machinery
// consumes.
//
// The `children` adjacency is the single source of truth for tree
// structure: it is what BuildSparsityFromTree (memory/sparsity_pattern.hpp)
// consumes and validates, and every parent-child loop reads it. (The
// original sketch's per-node childOffset member is deliberately not kept - a
// second copy of the same structure can only drift from the validated
// adjacency.)

#include "qcx/error.hpp"
#include "qfmm_geometry.hpp"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace qcx::integrals::internal {

/// The octree leaf-size cap: a node with more shell pairs splits. Small
/// and easy to debug (a starting value; a performance
/// tune, if ever wanted, belongs to a later stage).
inline constexpr std::size_t kQfmmMaxLeafSize = 8;

/// The octree depth safety valve: a node at this depth stops splitting
/// even when overfull. Never actually hit for realistic molecules - a hit
/// signals a degenerate input (many pairs at nearly the same point), not
/// a real leaf.
inline constexpr std::size_t kQfmmMaxTreeDepth = 20;

/// The minimum box half-width (Bohr) that still splits: below it all
/// contained centers coincide numerically, so bisection would recurse into
/// identical child boxes forever. An overfull leaf at this size is
/// physically meaningful (one density point), not a bug.
inline constexpr double kQfmmMinBoxHalfWidth = 1e-9;

/// One octree node: a cube (all three axes share the half-width), located
/// by its center. The root box is the padded axis-aligned bounding box of
/// every shell pair (center + extent per axis), so the box geometry alone
/// is sufficient for the node-to-node well-separatedness test - the box
/// already contains every shell pair's padded extent by construction.
struct QfmmTreeNode {
    double centerX, centerY, centerZ; ///< Box center (Bohr).
    double halfWidth; ///< Box half-width (cube, same on all 3 axes; Bohr).
    /// The covering radius of this node's charge about its centre (Bohr):
    /// the max over the node's pairs of (|pair centre - node centre| + the
    /// pair's extent). The box half-width is NOT this number - the box is a
    /// cube whose diagonal reaches past it - so the surface-to-surface test
    /// (QfmmSeparationTest::kSurfaceBall) reads this field and the recorded
    /// centre-to-width test reads halfWidth.
    double radius = 0.0;
    /// Shell-pair indices IN THIS NODE ONLY when the node is a leaf;
    /// empty for internal nodes.
    std::vector<std::size_t> pairIndices;
    bool isLeaf = true;
};

/// The octree build result: the flat node list (node 0 = the root), the
/// children[node] adjacency BuildSparsityFromTree expects (only non-empty
/// octants listed, in octant order - a node with fewer than 8 children is
/// completely normal), and for each ORIGINAL shell pair index the leaf
/// node it landed in (consumed by the interaction-list construction and
/// the per-leaf moment aggregation).
struct QfmmTreeBuildResult {
    std::vector<QfmmTreeNode> nodes;
    std::vector<std::vector<std::size_t>> children;
    std::vector<std::size_t> leafOfPair; ///< leafOfPair[pairIndex] = node index.
};

/// Builds the octree: computes the padded bounding box of every shell
/// pair's geometry, then splits nodes with more than \p maxLeafSize pairs
/// (and a box above kQfmmMinBoxHalfWidth) into their 8 octants by
/// bisecting each axis at the box midpoint, until leaves, the depth cap,
/// or the minimum-size floor is reached. A center exactly ON a bisection
/// plane goes to the lower octant (the <= convention, documented for the
/// containment tests). Empty octants are skipped entirely.
/// \param pairGeometries One geometry per shell pair (qfmm_geometry.hpp);
/// the center decides the octant, the extent pads the root box.
/// \param maxLeafSize Split cap, must be >= 1.
/// \param maxDepth Depth cap, must be >= 1.
/// \returns The tree, or an Error (kInvalidArgument for an empty input or
/// a zero cap).
qcx::Result<QfmmTreeBuildResult> BuildQfmmTree(const std::vector<QfmmPairGeometry>& pairGeometries,
                                               std::size_t maxLeafSize = kQfmmMaxLeafSize,
                                               std::size_t maxDepth = kQfmmMaxTreeDepth);

/// The classical FMM well-separatedness test: nodes A and B are
/// well-separated (far field - the multipole interaction is valid) when
/// distance(cA, cB) >= (wA + wB) / theta, equivalently
/// distance(cA, cB) * theta >= wA + wB. Smaller theta means a stricter
/// separation requirement - more pairs near-field, higher accuracy, more
/// cost; theta -> 0 degenerates to "everything is near field", which is
/// the acceptance criterion ("θ→0 recovers the exact direct result").
/// The half-widths alone are used, NOT the shell pairs' individual
/// extents - the box already contains every padded extent by construction.
/// \param theta Well-separatedness parameter; theta <= 0 is "never
/// well-separated" (the degenerate near-field-everything gate).
inline bool IsWellSeparated(const QfmmTreeNode& a, const QfmmTreeNode& b, double theta) noexcept {
    if (theta <= 0.0)
    {
        return false;
    }

    const double dx = a.centerX - b.centerX;
    const double dy = a.centerY - b.centerY;
    const double dz = a.centerZ - b.centerZ;
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    return distance * theta >= a.halfWidth + b.halfWidth;
}

/// The node-pair separation test (option-gated).
enum class QfmmSeparationTest {
    /// The recorded form: distance x theta >= wA + wB, the node box
    /// half-widths - a centre-to-width test whose width is a cube's
    /// half-width, not the charge's reach.
    kWidthTheta,
    /// The corrected form: the distributions are the objects, so
    /// the test is surface-to-surface - distance > rA + rB + k * max(rA, rB)
    /// with the nodes' charge radii. k is the buffer the multipole error
    /// bound demands; k = 0 is the bare touching test, and k < 0 is the
    /// degenerate gate (nothing well-separated: the theta -> 0 analogue, so
    /// the whole problem lands in the near field exactly as the recorded
    /// gate does).
    kSurfaceBall,
};

/// One separation classification: which test, and its parameter.
struct QfmmSeparation {
    QfmmSeparationTest test = QfmmSeparationTest::kWidthTheta;
    double theta = 0.0; ///< kWidthTheta's well-separatedness parameter.
    double k = 0.0; ///< kSurfaceBall's buffer (see QfmmSeparationTest); < 0 = the gate.
};

/// The separation test of one node pair (the QfmmSeparation form above; the
/// recorded IsWellSeparated is this with test == kWidthTheta).
/// \param a First node.
/// \param b Second node.
/// \param separation The test and its parameter.
/// \returns True when the pair is well separated.
bool IsWellSeparated(const QfmmTreeNode& a,
                     const QfmmTreeNode& b,
                     const QfmmSeparation& separation) noexcept;

/// BuildInteractionLists under an explicit separation form (the recorded
/// overload is this with the kWidthTheta test). Same traversal, same
/// completeness invariant - only the classification predicate differs.
/// \param tree The octree (BuildQfmmTree).
/// \param separation The test and its parameter.
/// \param farFieldPairs Receives the well-separated node pairs (unordered).
/// \param nearFieldLeafPairs Receives the non-separated LEAF pairs.
void BuildInteractionLists(const QfmmTreeBuildResult& tree,
                           const QfmmSeparation& separation,
                           std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
                           std::vector<std::pair<std::size_t, std::size_t>>& nearFieldLeafPairs);

/// Builds the interaction lists by the standard dual-tree traversal,
/// starting from (root, root): for every pair of nodes (A, B) whose
/// parents were NOT already well-separated (i.e. the finest-level pair
/// that first becomes well-separated - this avoids double-counting an
/// interaction at multiple tree levels) and that ARE well-separated
/// themselves, the unordered pair (A, B) is appended to \p farFieldPairs.
/// A pair that is not well-separated and has both sides leaves lands in
/// \p nearFieldLeafPairs (handled by the direct builder, not the multipole
/// code); otherwise the recursion descends into the LARGER node's children
/// (ties: A). A node with itself is never well-separated - always near
/// field when a leaf, and the recursion into all child pairs (including a
/// child with itself) when internal. Every unordered pair of shell pairs
/// ends up in exactly one of the two lists (the completeness invariant the
/// tests check exhaustively).
/// \param tree The octree (BuildQfmmTree).
/// \param theta Well-separatedness parameter (IsWellSeparated).
/// \param farFieldPairs Receives the well-separated node pairs (unordered).
/// \param nearFieldLeafPairs Receives the non-separated LEAF pairs
/// (unordered; a leaf with itself included).
void BuildInteractionLists(const QfmmTreeBuildResult& tree,
                           double theta,
                           std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
                           std::vector<std::pair<std::size_t, std::size_t>>& nearFieldLeafPairs);

} // namespace qcx::integrals::internal
