// The QFMM separation-counts instrument (2026-09-14). Counts ONLY - no wall
// clock, no SCF, no timed pass: the tree
// shape (node count, leaf-size and depth distributions), the interaction
// lists (near leaf pairs, far node pairs, per-leaf list lengths) and the
// near-field share of the direct screened CSR, each swept over the
// well-separatedness parameter theta and the leaf-size cap, at a ladder of
// fixtures of increasing size.
//
// Why it exists: the recorded QFMM near-field share (the 4,974-basis
// footprint point: the near field covers 24,123,496,017 of the
// full-screened 24,660,876,272 = 97.8% of the direct CSR) says the tree's
// separation criterion removes almost nothing at the operating theta
// (ThetaForPreset(kNormal) = 0.3). Whether that is the criterion, the
// calibration, or the fixtures is a question for counts, not timings, and
// the tree's viability at this scale is gated behind exactly this number.
//
// The screened near-field count is this instrument's own sorted-walk
// counter (the CountSchwarzSurvivingPairs argument: the cutoff test is a
// monotone product, so a per-row boundary walk over each leaf's descending
// Schwarz values counts the same entries the literal double loop does, in
// O(|A| log |B|) instead of O(|A| |B|) per leaf pair - the literal walk is
// quadratic in the pair space and cannot run at the larger fixtures). The
// two agree exactly by construction and are cross-checked in-run on every
// fixture small enough for the literal form (the verify= column of each
// fixture block); the repo's own CountNearFieldPatternEntries is the
// reference.
//
// Local-only instrument (never CI), like the other benchmarks. Usage:
//   qcx-bench-qfmm-separation-counts [--no-big] [--out FILE]
// Every block is printed and flushed as it is produced: a fixture that dies
// of allocation still leaves the earlier fixtures' counts on stdout.

#include "alkane_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "internal/fock_screen.hpp"
#include "internal/md_batch.hpp"
#include "internal/qfmm_geometry.hpp"
#include "internal/qfmm_tree.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/qfmm_fock_build.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using qcx::integrals::AccuracyPreset;
using qcx::integrals::LeafNearFieldDomain;
using qcx::integrals::ShellPairList;
using qcx::integrals::internal::QfmmPairGeometry;
using qcx::integrals::internal::QfmmTreeBuildResult;

/// The screening preset of every cell (the default kNormal: the same preset
/// the direct family's footprint uses, so the share is comparable to the
/// recorded 97.8%).
constexpr AccuracyPreset kPreset = AccuracyPreset::kNormal;

/// The theta rungs: the two operating values the preset table resolves
/// (kNormal 0.3, kLoose 0.45), the recalibration sweep's upper anchors
/// (0.7, 1.05) and the classical FMM regime (1.5 / 2.0 = the one-box buffer
/// rule / 3.0).
constexpr std::array<double, 7> kThetas = {0.3, 0.45, 0.7, 1.05, 1.5, 2.0, 3.0};

/// The leaf-size caps: the QFMM default (8, kQfmmMaxLeafSize) and coarser
/// rungs - the leaf size sets how much of the near field one multipole-free
/// direct block has to carry.
constexpr std::array<std::size_t, 4> kLeafCaps = {4, 8, 16, 32};

/// One ladder fixture: the H2O control (carbonCount 0) or a linear alkane in
/// the vendored STO-3G fixture basis (empty basisDir) or a vendored corpus
/// directory.
struct FixtureSpec {
    std::string_view label;
    std::size_t carbonCount;
    std::string_view basisDir;
};

constexpr std::array<FixtureSpec, 6> kFixtures = {{
    {"H2O/STO-3G", 0, ""},
    {"C12H26/STO-3G", 12, ""},
    {"C24H50/STO-3G", 24, ""},
    {"C12H26/def2-SVP", 12, "def2-svp"},
    {"C24H50/def2-SVP", 24, "def2-svp"},
    {"C42H86/def2-SVP", 42, "def2-svp"},
}};

/// The literal-form verification limit: the fixture's first cell is
/// cross-checked against CountNearFieldPatternEntries (the quadratic walk)
/// only when the pair count is at or below this - above it the literal form
/// is hundreds of millions to billions of iterations per cell.
constexpr std::size_t kVerifyPairLimit = 60000;

/// min / lower-median / mean / max of a value list (empty list: all zero).
struct Stats {
    std::size_t minimum = 0;
    std::size_t median = 0;
    double mean = 0.0;
    std::size_t maximum = 0;
};

Stats Summarize(std::vector<std::size_t> values) {
    Stats stats;

    if (values.empty())
    {
        return stats;
    }

    std::sort(values.begin(), values.end());
    stats.minimum = values.front();
    stats.median = values[values.size() / 2];
    stats.maximum = values.back();

    double sum = 0.0;

    for (const std::size_t value : values)
    {
        sum += static_cast<double>(value);
    }

    stats.mean = sum / static_cast<double>(values.size());
    return stats;
}

/// min / lower-median / mean / max of a double-valued list (the extent
/// table's form; empty list: all zero).
struct ValueStats {
    double minimum = 0.0;
    double median = 0.0;
    double mean = 0.0;
    double maximum = 0.0;
};

ValueStats SummarizeValues(std::vector<double> values) {
    ValueStats stats;

    if (values.empty())
    {
        return stats;
    }

    std::sort(values.begin(), values.end());
    stats.minimum = values.front();
    stats.median = values[values.size() / 2];
    stats.maximum = values.back();

    double sum = 0.0;

    for (const double value : values)
    {
        sum += value;
    }

    stats.mean = sum / static_cast<double>(values.size());
    return stats;
}

/// The per-leaf pair-index lists (the leaf-driven enumeration's own
/// grouping, reproduced once here) plus each leaf's Schwarz values in
/// DESCENDING order - the sorted-walk counter's input.
struct LeafIndex {
    std::vector<std::size_t> offsets; ///< leaf -> [offsets[l], offsets[l + 1]).
    std::vector<std::size_t> indices; ///< Pair indices grouped by leaf, ascending.
    /// Sorted-descending Schwarz values, one vector per leaf (the count-only
    /// walk's monotone input).
    std::vector<std::vector<double>> descending;
    /// The function-count prefix sums in the SAME descending order (the
    /// weighted walk's range-sum carrier), one vector of size |values| + 1
    /// per leaf.
    std::vector<std::vector<double>> descendingWeightPrefix;
};

/// Builds the leaf grouping of the pair indices (leafOfPair is the octree's
/// contract carrier) and the per-leaf descending Schwarz values (with the
/// matching weight prefix sums).
LeafIndex BuildLeafIndex(const std::vector<std::size_t>& leafOfPair,
                         const std::vector<double>& schwarz,
                         const std::vector<double>& weights,
                         std::size_t nLeaves) {
    LeafIndex index;
    index.offsets.assign(nLeaves + 1, 0);

    for (const std::size_t leaf : leafOfPair)
    {
        ++index.offsets[leaf + 1];
    }

    for (std::size_t leaf = 0; leaf < nLeaves; ++leaf)
    {
        index.offsets[leaf + 1] += index.offsets[leaf];
    }

    index.indices.resize(leafOfPair.size());
    {
        std::vector<std::size_t> cursors = index.offsets;

        for (std::size_t pairIndex = 0; pairIndex < leafOfPair.size(); ++pairIndex)
        {
            const std::size_t leaf = leafOfPair[pairIndex];
            index.indices[cursors[leaf]++] = pairIndex;
        }
    }

    index.descending.resize(nLeaves);
    index.descendingWeightPrefix.resize(nLeaves);

    for (std::size_t leaf = 0; leaf < nLeaves; ++leaf)
    {
        std::vector<std::pair<double, double>> sorted; // (Schwarz value, function count).
        sorted.reserve(index.offsets[leaf + 1] - index.offsets[leaf]);

        for (std::size_t at = index.offsets[leaf]; at < index.offsets[leaf + 1]; ++at)
        {
            const std::size_t pairIndex = index.indices[at];
            sorted.emplace_back(schwarz[pairIndex], weights[pairIndex]);
        }

        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        std::vector<double>& values = index.descending[leaf];
        std::vector<double>& prefix = index.descendingWeightPrefix[leaf];
        values.reserve(sorted.size());
        prefix.assign(sorted.size() + 1, 0.0);

        for (std::size_t at = 0; at < sorted.size(); ++at)
        {
            values.push_back(sorted[at].first);
            prefix[at + 1] = prefix[at] + sorted[at].second;
        }
    }

    return index;
}

/// The first index in [lo, hi) of a DESCENDING sequence whose product with
/// \p value falls below \p threshold (hi when every one of them clears it).
/// The literal predicate is used, so the boundary is exact: the product is
/// monotone in the second operand for non-negative bounds (the
/// CountSchwarzSurvivingPairs argument), which makes the binary search land
/// on the same split point the double loop would.
std::size_t FirstFailing(const std::vector<double>& descending,
                         std::size_t lo,
                         std::size_t hi,
                         double value,
                         double threshold) {
    while (lo < hi)
    {
        const std::size_t mid = lo + (hi - lo) / 2;

        if (value * descending[mid] >= threshold)
        {
            lo = mid + 1;
        } else
        {
            hi = mid;
        }
    }

    return lo;
}

/// The near-field screened count by the sorted walk: the exact
/// CountNearFieldPatternEntries result without the quadratic per-leaf-pair
/// loop. The diagonal (A, A) leaf pair counts each unordered in-leaf pair
/// once (j >= i in the leaf's own order, the repo form); an off-diagonal
/// (A, B) pair counts every |A| x |B| product (the repo form - the product
/// is symmetric, so max/min canonicalization does not change it).
std::size_t CountNearScreenedSorted(const LeafIndex& index,
                                    const std::vector<std::pair<std::size_t, std::size_t>>& pairs,
                                    double threshold) {
    std::size_t count = 0;

    for (const auto& [leafA, leafB] : pairs)
    {
        const std::vector<double>& valuesA = index.descending[leafA];
        const std::vector<double>& valuesB = index.descending[leafB];

        if (leafA == leafB)
        {
            for (std::size_t i = 0; i < valuesA.size(); ++i)
            {
                const std::size_t boundary =
                    FirstFailing(valuesA, i, valuesA.size(), valuesA[i], threshold);
                count += boundary - i;
            }
        } else
        {
            for (const double value : valuesA)
            {
                count += FirstFailing(valuesB, 0, valuesB.size(), value, threshold);
            }
        }
    }

    return count;
}

/// The distance ratio of one node pair in units of the separation criterion:
/// distance x theta / (wA + wB), so a value below 1 is near field at this
/// theta and 1 is the boundary. The distance uses the node centers and the
/// box half-widths - IsWellSeparated's own quantities.
double CriterionRatio(const QfmmTreeBuildResult& tree, std::size_t a, std::size_t b, double theta) {
    const double dx = tree.nodes[a].centerX - tree.nodes[b].centerX;
    const double dy = tree.nodes[a].centerY - tree.nodes[b].centerY;
    const double dz = tree.nodes[a].centerZ - tree.nodes[b].centerZ;
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double width = tree.nodes[a].halfWidth + tree.nodes[b].halfWidth;

    if (width <= 0.0)
    {
        return 0.0;
    }

    return distance * theta / width;
}

/// The leaves under every node (a post-order pass over the children
/// adjacency): the per-leaf far-list expansion's carrier.
std::vector<std::vector<std::size_t>> LeavesUnder(const QfmmTreeBuildResult& tree,
                                                  std::size_t nodeIndex) {
    std::vector<std::vector<std::size_t>> result(tree.nodes.size());

    // An explicit post-order walk: the tree can be deeper than a comfortable
    // recursion, and the children adjacency is the single source of truth.
    std::vector<std::size_t> stack{nodeIndex};
    std::vector<std::size_t> order;

    while (!stack.empty())
    {
        const std::size_t node = stack.back();
        stack.pop_back();
        order.push_back(node);

        for (const std::size_t child : tree.children[node])
        {
            stack.push_back(child);
        }
    }

    for (auto it = order.rbegin(); it != order.rend(); ++it)
    {
        const std::size_t node = *it;

        if (tree.nodes[node].isLeaf)
        {
            result[node].push_back(node);
            continue;
        }

        for (const std::size_t child : tree.children[node])
        {
            result[node].insert(result[node].end(), result[child].begin(), result[child].end());
        }
    }

    return result;
}

/// The node depths (the root is 0). The node list is pre-order (BuildNode
/// pushes the parent before recursing), so one forward sweep through the
/// children adjacency is enough.
std::vector<std::size_t> NodeDepths(const QfmmTreeBuildResult& tree) {
    std::vector<std::size_t> depth(tree.nodes.size(), 0);

    for (std::size_t node = 0; node < tree.nodes.size(); ++node)
    {
        for (const std::size_t child : tree.children[node])
        {
            depth[child] = depth[node] + 1;
        }
    }

    return depth;
}

/// The largest span of a leaf's pair centers (max over the three axes), the
/// depth-cap carrier test: a leaf that stopped at the depth cap with a
/// NEAR-ZERO span holds pairs at one point (the same-atom cluster the
/// bisection can never separate), a large span means the box inflation, not
/// the point cluster, is what the cap caught.
double CenterSpan(const std::vector<QfmmPairGeometry>& geometries,
                  const std::vector<std::size_t>& pairIndices) {
    if (pairIndices.empty())
    {
        return 0.0;
    }

    std::array<double, 3> minimum{geometries[pairIndices[0]].centerX,
                                  geometries[pairIndices[0]].centerY,
                                  geometries[pairIndices[0]].centerZ};
    std::array<double, 3> maximum = minimum;

    for (const std::size_t pairIndex : pairIndices)
    {
        const std::array<double, 3> center{geometries[pairIndex].centerX,
                                           geometries[pairIndex].centerY,
                                           geometries[pairIndex].centerZ};

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], center[axis]);
            maximum[axis] = std::max(maximum[axis], center[axis]);
        }
    }

    return std::max({maximum[0] - minimum[0], maximum[1] - minimum[1], maximum[2] - minimum[2]});
}

class Output {
public:
    explicit Output(std::FILE* file) : _file(file) {}

    void Line(const std::string& text) {
        std::fputs(text.c_str(), _file);
        std::fputc('\n', _file);
        std::fflush(_file);
    }

private:
    std::FILE* _file;
};

/// The per-cell counts of one (fixture, cap, theta) cell.
struct CellCounts {
    std::size_t nearLeafPairs = 0;
    std::size_t diagonalLeafPairs = 0;
    std::size_t farNodePairs = 0;
    std::size_t nearGeometric = 0;
    std::size_t nearScreened = 0;
    std::size_t screenedReference = 0; ///< CountNearFieldPatternEntries (0 = not run).
    double flipAt2x = 0.0; ///< Class share of the near set that a 2x theta flips to far.
    double flipAt4x = 0.0; ///< The same at 4x.
    std::array<std::size_t, 6> ratioBins{}; ///< Criterion-ratio histogram (class weighted).
    Stats nearListPerLeaf;
    Stats farListPerLeaf;
};

/// Fills the per-leaf near list lengths and the class-weighted criterion-ratio
/// histogram (the far list is filled by FarListPerLeaf below - its expansion
/// is the one super-linear step and is skipped when it cannot fit the budget).
CellCounts CountCell(const QfmmTreeBuildResult& tree,
                     const std::vector<std::pair<std::size_t, std::size_t>>& farPairs,
                     const std::vector<std::pair<std::size_t, std::size_t>>& nearPairs,
                     double theta,
                     std::size_t leafCount) {
    CellCounts counts;
    counts.nearLeafPairs = nearPairs.size();
    counts.farNodePairs = farPairs.size();

    std::vector<std::size_t> nearList(leafCount, 0);
    std::size_t nonDiagonalClasses = 0;

    for (const auto& [leafA, leafB] : nearPairs)
    {
        const std::size_t sizeA = tree.nodes[leafA].pairIndices.size();
        const std::size_t sizeB = tree.nodes[leafB].pairIndices.size();

        ++nearList[leafA];

        if (leafA == leafB)
        {
            ++counts.diagonalLeafPairs;
            counts.nearGeometric += sizeA * (sizeA + 1) / 2;
            continue;
        }

        ++nearList[leafB];
        const std::size_t classes = sizeA * sizeB;
        counts.nearGeometric += classes;
        nonDiagonalClasses += classes;

        const double ratio = CriterionRatio(tree, leafA, leafB, theta);

        if (ratio >= 0.5)
        {
            counts.flipAt2x += static_cast<double>(classes);
        }

        if (ratio >= 0.25)
        {
            counts.flipAt4x += static_cast<double>(classes);
        }

        const std::size_t bin = ratio < 0.25   ? 0
                                : ratio < 0.5  ? 1
                                : ratio < 0.75 ? 2
                                : ratio < 0.9  ? 3
                                : ratio < 0.99 ? 4
                                               : 5;
        counts.ratioBins[bin] += classes;
    }

    if (nonDiagonalClasses > 0)
    {
        counts.flipAt2x /= static_cast<double>(nonDiagonalClasses);
        counts.flipAt4x /= static_cast<double>(nonDiagonalClasses);
    }

    counts.nearListPerLeaf = Summarize(std::move(nearList));
    return counts;
}

/// A class count and its function-quartet weight (the weighted
/// operation count: nFuncs(bra) x nFuncs(ket) summed over the surviving
/// classes - the direct path's contract size, in the same units the
/// multipole translation blocks are compared against).
struct WorkCount {
    std::size_t classes = 0;
    double quartets = 0.0;
};

/// The weighted near-field count by the sorted walk (the class count is the
/// same CountNearScreenedSorted total; the quartet weight rides a prefix sum
/// of the per-pair function counts over each leaf's descending Schwarz
/// order, so a qualifying run of kets is one range sum).
WorkCount CountNearWeighted(const LeafIndex& index,
                            const std::vector<double>& weights,
                            const std::vector<std::pair<std::size_t, std::size_t>>& pairs,
                            double threshold) {
    WorkCount count;

    for (const auto& [leafA, leafB] : pairs)
    {
        const std::vector<double>& valuesA = index.descending[leafA];
        const std::vector<double>& valuesB = index.descending[leafB];
        const std::vector<double>& prefixB = index.descendingWeightPrefix[leafB];

        if (leafA == leafB)
        {
            for (std::size_t i = 0; i < valuesA.size(); ++i)
            {
                const std::size_t boundary =
                    FirstFailing(valuesA, i, valuesA.size(), valuesA[i], threshold);
                count.classes += boundary - i;
                count.quartets += weights[index.indices[index.offsets[leafA] + i]] *
                                  (prefixB[boundary] - prefixB[i]);
            }
        } else
        {
            for (std::size_t i = 0; i < valuesA.size(); ++i)
            {
                const std::size_t boundary =
                    FirstFailing(valuesB, 0, valuesB.size(), valuesA[i], threshold);
                count.classes += boundary;
                count.quartets +=
                    weights[index.indices[index.offsets[leafA] + i]] * prefixB[boundary];
            }
        }
    }

    return count;
}

/// The full-screened weighted count (the direct builder's own pattern, the
/// CountSchwarzSurvivingPairs walk with the function-quartet weight): the
/// same descending-boundary argument, with the upper-triangle correction the
/// repo's counter uses.
WorkCount CountFullWeighted(const std::vector<double>& schwarz,
                            const std::vector<double>& weights,
                            double threshold) {
    const std::size_t n = schwarz.size();
    std::vector<std::size_t> order(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        order[i] = i;
    }

    std::sort(order.begin(), order.end(), [&schwarz](std::size_t a, std::size_t b) {
        return schwarz[a] > schwarz[b];
    });

    std::vector<double> prefix(n + 1, 0.0);

    for (std::size_t i = 0; i < n; ++i)
    {
        prefix[i + 1] = prefix[i] + weights[order[i]];
    }

    WorkCount count;
    std::size_t boundary = n;
    double rowTotal = 0.0;
    double diagonalTotal = 0.0;

    for (std::size_t i = 0; i < n; ++i)
    {
        while (boundary > 0 && schwarz[order[i]] * schwarz[order[boundary - 1]] < threshold)
        {
            --boundary;
        }

        rowTotal += weights[order[i]] * prefix[boundary];

        if (boundary > i)
        {
            diagonalTotal += weights[order[i]] * weights[order[i]];
        }
    }

    // The class total is CountSchwarzSurvivingPairs' own number (the
    // instrument prints both); only the weight rides this walk.
    count.quartets = diagonalTotal + 0.5 * (rowTotal - diagonalTotal);
    return count;
}

/// The per-leaf far list (the FMM interaction-list reading): every far node
/// pair (a, b) contributes every leaf under b to every leaf under a (and the
/// mirror), so a leaf's far list is the number of leaf-level far
/// interactions it owns. The expansion is skipped (all-zero stats) when its
/// work estimate exceeds the cap - the coarse-level far pairs cover large
/// subtrees and the expansion is the instrument's only super-linear step.
Stats FarListPerLeaf(const std::vector<std::vector<std::size_t>>& leavesUnder,
                     const std::vector<std::pair<std::size_t, std::size_t>>& farPairs,
                     std::size_t leafCount) {
    constexpr std::size_t kExpansionBudget = 200000000;
    std::size_t work = 0;

    for (const auto& [a, b] : farPairs)
    {
        work += leavesUnder[a].size() + leavesUnder[b].size();

        if (work > kExpansionBudget)
        {
            return Summarize({});
        }
    }

    std::vector<std::size_t> farList(leafCount, 0);

    for (const auto& [a, b] : farPairs)
    {
        const std::size_t sizeB = leavesUnder[b].size();
        const std::size_t sizeA = leavesUnder[a].size();

        for (const std::size_t leaf : leavesUnder[a])
        {
            farList[leaf] += sizeB;
        }

        for (const std::size_t leaf : leavesUnder[b])
        {
            farList[leaf] += sizeA;
        }
    }

    return Summarize(std::move(farList));
}

/// The corrected-geometry theta rungs (the named numbers of the correction)
/// and the surface test's buffers (k in units of max(rA, rB); 0 = touching).
constexpr std::array<double, 3> kCorrectedThetas = {0.5, 1.0, 2.0};
constexpr std::array<double, 5> kSurfaceKs = {0.0, 0.5, 1.0, 2.0, 4.0};

/// The leaf count of a built tree.
std::size_t CountLeaves(const QfmmTreeBuildResult& tree) {
    std::size_t leaves = 0;

    for (const qcx::integrals::internal::QfmmTreeNode& node : tree.nodes)
    {
        if (node.isLeaf)
        {
            ++leaves;
        }
    }

    return leaves;
}

/// Prints one tree-shape block (a model/cap cell): the node and leaf counts,
/// the leaf-size distribution by depth, the depth-capped share and the
/// largest leaves' center spans (the depth-cap carrier test). Returns the
/// node count.
std::size_t PrintTreeBlock(Output& output,
                           const std::string& label,
                           std::string_view model,
                           std::size_t cap,
                           const std::vector<QfmmPairGeometry>& geometries,
                           const qcx::integrals::internal::QfmmTreeBuildResult& tree,
                           std::size_t nPairs) {
    const std::size_t nodeCount = tree.nodes.size();
    std::size_t leafCount = 0;
    std::vector<std::size_t> leafSizes;
    std::size_t singletons = 0;

    for (const qcx::integrals::internal::QfmmTreeNode& node : tree.nodes)
    {
        if (!node.isLeaf)
        {
            continue;
        }

        ++leafCount;
        leafSizes.push_back(node.pairIndices.size());

        if (node.pairIndices.size() == 1)
        {
            ++singletons;
        }
    }

    const std::vector<std::size_t> depths = NodeDepths(tree);
    std::size_t treeDepth = 0;

    for (const std::size_t depth : depths)
    {
        treeDepth = std::max(treeDepth, depth);
    }

    const Stats leafStats = Summarize(std::move(leafSizes));
    char block[512];
    std::snprintf(block,
                  sizeof(block),
                  "TREE fixture=%s model=%s cap=%zu nodes=%zu internal=%zu leaves=%zu depth=%zu "
                  "leafMin=%zu leafMed=%zu leafMean=%.2f leafMax=%zu singletons=%zu",
                  label.c_str(),
                  std::string(model).c_str(),
                  cap,
                  nodeCount,
                  nodeCount - leafCount,
                  leafCount,
                  treeDepth,
                  leafStats.minimum,
                  leafStats.median,
                  leafStats.mean,
                  leafStats.maximum,
                  singletons);
    output.Line(block);

    std::size_t depthCappedLeaves = 0;
    std::size_t depthCappedPairTotal = 0;

    for (std::size_t depth = 0; depth <= treeDepth; ++depth)
    {
        std::vector<std::size_t> sizes;

        for (std::size_t node = 0; node < nodeCount; ++node)
        {
            if (tree.nodes[node].isLeaf && depths[node] == depth)
            {
                sizes.push_back(tree.nodes[node].pairIndices.size());
            }
        }

        if (sizes.empty())
        {
            continue;
        }

        const Stats stats = Summarize(sizes);
        std::size_t pairTotal = 0;

        for (const std::size_t size : sizes)
        {
            pairTotal += size;
        }

        if (depth == treeDepth)
        {
            depthCappedLeaves = sizes.size();
            depthCappedPairTotal = pairTotal;
        }

        std::snprintf(block,
                      sizeof(block),
                      "LEAFDEPTH fixture=%s model=%s cap=%zu depth=%zu leaves=%zu pairs=%zu "
                      "min=%zu med=%zu max=%zu",
                      label.c_str(),
                      std::string(model).c_str(),
                      cap,
                      depth,
                      sizes.size(),
                      pairTotal,
                      stats.minimum,
                      stats.median,
                      stats.maximum);
        output.Line(block);
    }

    std::snprintf(block,
                  sizeof(block),
                  "DEPTHCAP fixture=%s model=%s cap=%zu depth=%zu leaves=%zu pairs=%zu "
                  "pairSharePct=%.2f",
                  label.c_str(),
                  std::string(model).c_str(),
                  cap,
                  treeDepth,
                  depthCappedLeaves,
                  depthCappedPairTotal,
                  100.0 * static_cast<double>(depthCappedPairTotal) / static_cast<double>(nPairs));
    output.Line(block);

    std::vector<std::size_t> leavesBySize;

    for (std::size_t node = 0; node < nodeCount; ++node)
    {
        if (tree.nodes[node].isLeaf)
        {
            leavesBySize.push_back(node);
        }
    }

    std::sort(leavesBySize.begin(), leavesBySize.end(), [&tree](std::size_t a, std::size_t b) {
        return tree.nodes[a].pairIndices.size() > tree.nodes[b].pairIndices.size();
    });

    for (std::size_t rank = 0; rank < std::min<std::size_t>(3, leavesBySize.size()); ++rank)
    {
        const std::size_t node = leavesBySize[rank];
        double maxExtent = 0.0;

        for (const std::size_t pairIndex : tree.nodes[node].pairIndices)
        {
            maxExtent = std::max(maxExtent, geometries[pairIndex].extent);
        }

        std::snprintf(block,
                      sizeof(block),
                      "BIGLEAF fixture=%s model=%s cap=%zu rank=%zu size=%zu depth=%zu span=%.3f "
                      "maxExtent=%.3f nodeRadius=%.3f",
                      label.c_str(),
                      std::string(model).c_str(),
                      cap,
                      rank,
                      tree.nodes[node].pairIndices.size(),
                      depths[node],
                      CenterSpan(geometries, tree.nodes[node].pairIndices),
                      maxExtent,
                      tree.nodes[node].radius);
        output.Line(block);
    }

    return nodeCount;
}

/// The pair-extent decomposition table over the fixture's
/// pairs, all products vs the significant ones, plus the product-ball radius
/// and the centre shift it allows. Printed once per fixture.
void PrintExtentTable(Output& output,
                      const std::string& label,
                      const std::vector<qcx::integrals::internal::MdPairData>& pairs,
                      double tau) {
    std::vector<double> offsets;
    std::vector<double> falloffs;
    std::vector<double> significantOffsets;
    std::vector<double> significantFalloffs;
    std::vector<double> productRadii;
    std::vector<double> productOffsets;
    offsets.reserve(pairs.size());
    falloffs.reserve(pairs.size());
    significantOffsets.reserve(pairs.size());
    significantFalloffs.reserve(pairs.size());
    productRadii.reserve(pairs.size());
    productOffsets.reserve(pairs.size());

    for (const qcx::integrals::internal::MdPairData& pair : pairs)
    {
        const qcx::integrals::internal::QfmmExtentSplit split =
            qcx::integrals::internal::SplitPairExtent(pair, tau);
        offsets.push_back(split.maxOffset);
        falloffs.push_back(split.maxFalloff);
        significantOffsets.push_back(split.significantOffset);
        significantFalloffs.push_back(split.significantFalloff);
        productRadii.push_back(split.productRadius);
        productOffsets.push_back(split.productOffset);
    }

    const auto printRow = [&output, &label](const char* name, const ValueStats& stats) {
        char row[256];
        std::snprintf(row,
                      sizeof(row),
                      "EXTENT fixture=%s term=%s min=%.3f med=%.3f mean=%.3f max=%.3f",
                      label.c_str(),
                      name,
                      stats.minimum,
                      stats.median,
                      stats.mean,
                      stats.maximum);
        output.Line(row);
    };

    // The radius tail's carriers: the three pairs with the largest
    // product-ball radius, with the numbers that explain them (the shell
    // separation, the angular momenta, the product count and the pair's own
    // largest product amplitude).
    std::vector<std::size_t> byRadius(pairs.size());

    for (std::size_t i = 0; i < pairs.size(); ++i)
    {
        byRadius[i] = i;
    }

    std::sort(byRadius.begin(), byRadius.end(), [&productRadii](std::size_t a, std::size_t b) {
        return productRadii[a] > productRadii[b];
    });

    for (std::size_t rank = 0; rank < std::min<std::size_t>(3, byRadius.size()); ++rank)
    {
        const std::size_t index = byRadius[rank];
        const qcx::integrals::internal::MdPairData& pair = pairs[index];
        const double dx = pair.ax - pair.bx;
        const double dy = pair.ay - pair.by;
        const double dz = pair.az - pair.bz;
        const double separation = std::sqrt(dx * dx + dy * dy + dz * dz);
        double maxWeight = 0.0;

        for (std::size_t prim = 0; prim < pair.primPairs.size(); ++prim)
        {
            maxWeight =
                std::max(maxWeight, qcx::integrals::internal::PrimitiveProductWeight(pair, prim));
        }

        char row[320];
        std::snprintf(row,
                      sizeof(row),
                      "RADIUSTAIL fixture=%s rank=%zu la=%d lb=%d prims=%zu shellSep=%.2f "
                      "centreShift=%.3f radius=%.3f maxWeight=%.3e",
                      label.c_str(),
                      rank,
                      pair.la,
                      pair.lb,
                      pair.primPairs.size(),
                      separation,
                      productOffsets[index],
                      productRadii[index],
                      maxWeight);
        output.Line(row);
    }

    printRow("allOffset", SummarizeValues(offsets));
    printRow("allFalloff", SummarizeValues(falloffs));
    printRow("significantOffset", SummarizeValues(significantOffsets));
    printRow("significantFalloff", SummarizeValues(significantFalloffs));
    printRow("productRadius", SummarizeValues(productRadii));
    printRow("productCentreShift", SummarizeValues(productOffsets));
}

/// Runs one ladder fixture: the geometry chain once, then the (model, cap,
/// test) sweep. Returns false when the fixture could not be prepared.
bool RunFixture(const FixtureSpec& spec, bool verifyLiteral, Output& output) {
    auto molecule = spec.carbonCount == 0 ? qcx::testing::MakeH2oSto3g()
                                          : qcx::testing::MakeAlkaneSto3g(spec.carbonCount);

    if (!molecule.has_value())
    {
        output.Line(std::string(spec.label) + ": the molecule failed: " + molecule.error().message);
        return false;
    }

    qcx::Result<qcx::basisset::BasisSet> basis =
        spec.carbonCount == 0
            ? qcx::testing::MakeH2oSto3gBasis()
            : (spec.basisDir.empty()
                   ? qcx::testing::MakeAlkaneSto3gBasis()
                   : qcx::basisset::ParseNwchemDirectoryFiltered(
                         std::string(QcxBasisDataDir) + "/" + std::string(spec.basisDir),
                         std::array<int, 2>{6, 1}));

    if (!basis.has_value())
    {
        output.Line(std::string(spec.label) + ": the basis failed: " + basis.error().message);
        return false;
    }

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        output.Line(std::string(spec.label) +
                    ": the pair list failed: " + pairList.error().message);
        return false;
    }

    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);

    if (!schwarz.has_value())
    {
        output.Line(std::string(spec.label) +
                    ": the Schwarz bounds failed: " + schwarz.error().message);
        return false;
    }

    auto pairStore = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);

    if (!pairStore.has_value())
    {
        output.Line(std::string(spec.label) +
                    ": the pair store failed: " + pairStore.error().message);
        return false;
    }

    const std::size_t n = pairList->functionCount;
    const std::size_t nPairs = pairList->pairs.size();
    const double tau = qcx::integrals::QfmmExtentForPreset(kPreset);
    std::vector<QfmmPairGeometry> geometries =
        qcx::integrals::internal::ComputePairGeometries(*pairStore, tau);
    std::vector<QfmmPairGeometry> productGeometries =
        qcx::integrals::internal::ComputePairGeometries(
            *pairStore, tau, qcx::integrals::internal::QfmmExtentModel::kProductBall);
    std::vector<double> weights(nPairs, 0.0);

    for (std::size_t pairIndex = 0; pairIndex < nPairs; ++pairIndex)
    {
        weights[pairIndex] = static_cast<double>((*pairStore)[pairIndex].nFuncs);
    }

    char block[512];
    std::snprintf(block, sizeof(block), "\n===== %s =====", std::string(spec.label).c_str());
    output.Line(block);
    std::snprintf(block,
                  sizeof(block),
                  "n=%zu shells=%zu pairs=%zu preset=%d tau=%.3e fullScreened=%zu "
                  "totalClasses=%zu",
                  n,
                  pairList->shells.size(),
                  nPairs,
                  static_cast<int>(kPreset),
                  tau,
                  qcx::integrals::internal::CountSchwarzSurvivingPairs(*schwarz, kPreset),
                  nPairs * (nPairs + 1) / 2);
    output.Line(block);
    PrintExtentTable(output, std::string(spec.label), *pairStore, tau);
    std::fflush(stdout);

    // The pair store is the ladder's dominant allocation and nothing below
    // reads it again - the geometry passes above are its last consumers.
    std::vector<qcx::integrals::internal::MdPairData> released;
    released.swap(*pairStore);

    const std::size_t fullScreened =
        qcx::integrals::internal::CountSchwarzSurvivingPairs(*schwarz, kPreset);
    const double threshold =
        qcx::integrals::SchwarzThreshold(kPreset) * qcx::integrals::internal::kNeighborListSlack;
    const WorkCount fullWork = CountFullWeighted(*schwarz, weights, threshold);
    const std::size_t totalClasses = nPairs * (nPairs + 1) / 2;
    const auto lMult = static_cast<std::size_t>(qcx::integrals::LMultForPreset(kPreset));
    const double blockUnit = static_cast<double>((lMult + 1) * (lMult + 1));
    bool verified = false;

    for (const std::size_t cap : kLeafCaps)
    {
        auto tree = qcx::integrals::internal::BuildQfmmTree(geometries, cap);
        auto productTree = qcx::integrals::internal::BuildQfmmTree(productGeometries, cap);

        if (!tree.has_value() || !productTree.has_value())
        {
            output.Line(std::string("cap=") + std::to_string(cap) + ": the tree failed");
            continue;
        }

        const std::size_t nodeCount = PrintTreeBlock(
            output, std::string(spec.label), "recorded", cap, geometries, *tree, nPairs);
        const std::size_t productNodeCount = PrintTreeBlock(output,
                                                            std::string(spec.label),
                                                            "product",
                                                            cap,
                                                            productGeometries,
                                                            *productTree,
                                                            nPairs);
        std::fflush(stdout);

        const LeafIndex leafIndex = BuildLeafIndex(tree->leafOfPair, *schwarz, weights, nodeCount);
        const LeafIndex productLeafIndex =
            BuildLeafIndex(productTree->leafOfPair, *schwarz, weights, productNodeCount);
        const std::vector<std::vector<std::size_t>> leavesUnder = LeavesUnder(*tree, 0);

        // The recorded cell, now with the weighted work.
        for (const double theta : kThetas)
        {
            std::vector<std::pair<std::size_t, std::size_t>> farPairs;
            std::vector<std::pair<std::size_t, std::size_t>> nearPairs;
            qcx::integrals::internal::BuildInteractionLists(*tree, theta, farPairs, nearPairs);
            CellCounts counts = CountCell(*tree, farPairs, nearPairs, theta, nodeCount);
            const WorkCount nearWork = CountNearWeighted(leafIndex, weights, nearPairs, threshold);
            counts.nearScreened = nearWork.classes;

            if (verifyLiteral && !verified && nPairs <= kVerifyPairLimit)
            {
                LeafNearFieldDomain domain;
                domain.nLeaves = nodeCount;
                domain.leafOfPair = tree->leafOfPair;
                domain.nearFieldLeafPairs = nearPairs;
                counts.screenedReference = qcx::integrals::internal::CountNearFieldPatternEntries(
                    *schwarz, kPreset, domain);
                verified = true;
            }

            counts.farListPerLeaf = FarListPerLeaf(leavesUnder, farPairs, nodeCount);
            const double nearGeomPercent = 100.0 * static_cast<double>(counts.nearGeometric) /
                                           static_cast<double>(totalClasses);
            const double nearScreenedPercent =
                fullScreened == 0 ? 0.0
                                  : 100.0 * static_cast<double>(counts.nearScreened) /
                                        static_cast<double>(fullScreened);
            const double farBlocks = 2.0 * static_cast<double>(counts.farNodePairs) * blockUnit +
                                     static_cast<double>(nodeCount) * blockUnit;
            std::snprintf(block,
                          sizeof(block),
                          "CELL fixture=%s model=recorded test=width n=%zu pairs=%zu theta=%.2f "
                          "cap=%zu nodes=%zu leaves=%zu farPairs=%zu nearLeafPairs=%zu "
                          "diagLeafPairs=%zu nearGeomPct=%.3f nearScr=%zu fullScr=%zu "
                          "nearScrPct=%.3f flip2xPct=%.2f flip4xPct=%.2f nearListMed=%zu "
                          "nearListMax=%zu farListMed=%zu farListMax=%zu nearQuartets=%.0f "
                          "farQuartets=%.0f farBlocks=%.0f verify=%s",
                          std::string(spec.label).c_str(),
                          n,
                          nPairs,
                          theta,
                          cap,
                          nodeCount,
                          nodeCount - CountLeaves(*tree),
                          counts.farNodePairs,
                          counts.nearLeafPairs,
                          counts.diagonalLeafPairs,
                          nearGeomPercent,
                          counts.nearScreened,
                          fullScreened,
                          nearScreenedPercent,
                          100.0 * counts.flipAt2x,
                          100.0 * counts.flipAt4x,
                          counts.nearListPerLeaf.median,
                          counts.nearListPerLeaf.maximum,
                          counts.farListPerLeaf.median,
                          counts.farListPerLeaf.maximum,
                          nearWork.quartets,
                          fullWork.quartets - nearWork.quartets,
                          farBlocks,
                          counts.screenedReference == 0
                              ? "-"
                              : (counts.screenedReference == counts.nearScreened ? "OK" : "FAIL"));
            output.Line(block);

            if (counts.screenedReference != 0 && counts.screenedReference != counts.nearScreened)
            {
                std::snprintf(block,
                              sizeof(block),
                              "  VERIFY MISMATCH theta=%.2f sorted=%zu literal=%zu",
                              theta,
                              counts.nearScreened,
                              counts.screenedReference);
                output.Line(block);
            }
        }

        // The corrected geometry under the RECORDED test: isolates the extent
        // fix alone. Then the surface-to-surface test on the same tree: the
        // two fixes, separately and together.
        for (const double theta : kCorrectedThetas)
        {
            std::vector<std::pair<std::size_t, std::size_t>> farPairs;
            std::vector<std::pair<std::size_t, std::size_t>> nearPairs;
            qcx::integrals::internal::BuildInteractionLists(
                *productTree, theta, farPairs, nearPairs);
            const CellCounts counts =
                CountCell(*productTree, farPairs, nearPairs, theta, productNodeCount);
            const WorkCount nearWork =
                CountNearWeighted(productLeafIndex, weights, nearPairs, threshold);
            const double farBlocks = 2.0 * static_cast<double>(counts.farNodePairs) * blockUnit +
                                     static_cast<double>(productNodeCount) * blockUnit;
            std::snprintf(block,
                          sizeof(block),
                          "CELL fixture=%s model=product test=width n=%zu pairs=%zu theta=%.2f "
                          "cap=%zu nodes=%zu leaves=%zu farPairs=%zu nearLeafPairs=%zu "
                          "nearGeomPct=%.3f nearScr=%zu fullScr=%zu nearScrPct=%.3f "
                          "removedPct=%.4f nearQuartets=%.0f farQuartets=%.0f farBlocks=%.0f",
                          std::string(spec.label).c_str(),
                          n,
                          nPairs,
                          theta,
                          cap,
                          productNodeCount,
                          productNodeCount - CountLeaves(*productTree),
                          counts.farNodePairs,
                          counts.nearLeafPairs,
                          100.0 * static_cast<double>(counts.nearGeometric) /
                              static_cast<double>(totalClasses),
                          nearWork.classes,
                          fullScreened,
                          100.0 * static_cast<double>(nearWork.classes) /
                              static_cast<double>(fullScreened),
                          100.0 * static_cast<double>(fullScreened - nearWork.classes) /
                              static_cast<double>(fullScreened),
                          nearWork.quartets,
                          fullWork.quartets - nearWork.quartets,
                          farBlocks);
            output.Line(block);
        }

        for (const double k : kSurfaceKs)
        {
            qcx::integrals::internal::QfmmSeparation separation;
            separation.test = qcx::integrals::internal::QfmmSeparationTest::kSurfaceBall;
            separation.k = k;
            std::vector<std::pair<std::size_t, std::size_t>> farPairs;
            std::vector<std::pair<std::size_t, std::size_t>> nearPairs;
            qcx::integrals::internal::BuildInteractionLists(
                *productTree, separation, farPairs, nearPairs);
            const CellCounts counts =
                CountCell(*productTree, farPairs, nearPairs, k, productNodeCount);
            const WorkCount nearWork =
                CountNearWeighted(productLeafIndex, weights, nearPairs, threshold);
            const double farBlocks = 2.0 * static_cast<double>(counts.farNodePairs) * blockUnit +
                                     static_cast<double>(productNodeCount) * blockUnit;
            std::snprintf(block,
                          sizeof(block),
                          "CELL fixture=%s model=product test=surface n=%zu pairs=%zu k=%.2f "
                          "cap=%zu nodes=%zu leaves=%zu farPairs=%zu nearLeafPairs=%zu "
                          "nearGeomPct=%.3f nearScr=%zu fullScr=%zu nearScrPct=%.3f "
                          "removedPct=%.4f nearQuartets=%.0f farQuartets=%.0f farBlocks=%.0f "
                          "nearListMed=%zu nearListMax=%zu",
                          std::string(spec.label).c_str(),
                          n,
                          nPairs,
                          k,
                          cap,
                          productNodeCount,
                          productNodeCount - CountLeaves(*productTree),
                          counts.farNodePairs,
                          counts.nearLeafPairs,
                          100.0 * static_cast<double>(counts.nearGeometric) /
                              static_cast<double>(totalClasses),
                          nearWork.classes,
                          fullScreened,
                          100.0 * static_cast<double>(nearWork.classes) /
                              static_cast<double>(fullScreened),
                          100.0 * static_cast<double>(fullScreened - nearWork.classes) /
                              static_cast<double>(fullScreened),
                          nearWork.quartets,
                          fullWork.quartets - nearWork.quartets,
                          farBlocks,
                          counts.nearListPerLeaf.median,
                          counts.nearListPerLeaf.maximum);
            output.Line(block);
            std::fflush(stdout);
        }
    }

    return true;
}

/// The error limb: the composed QFMM Fock at the recorded and at the
/// corrected settings against the EXACT near-field-only build (the
/// degenerate separation gate - every class lands in the near field, so the
/// far field contributes nothing and the result is the restricted direct
/// build at the same preset). Also the bit-identity gate: the corrected
/// geometry at ITS degenerate gate must reproduce the recorded gate build
/// bit for bit, because nothing in the far field may move a value there.
/// \param spec The fixture (small ones only - the near-only build is a full
/// direct build).
/// \returns True when the limb ran.
bool RunErrorLimb(const FixtureSpec& spec, Output& output) {
    auto molecule = spec.carbonCount == 0 ? qcx::testing::MakeH2oSto3g()
                                          : qcx::testing::MakeAlkaneSto3g(spec.carbonCount);

    if (!molecule.has_value())
    {
        return false;
    }

    qcx::Result<qcx::basisset::BasisSet> basis =
        spec.carbonCount == 0
            ? qcx::testing::MakeH2oSto3gBasis()
            : (spec.basisDir.empty()
                   ? qcx::testing::MakeAlkaneSto3gBasis()
                   : qcx::basisset::ParseNwchemDirectoryFiltered(
                         std::string(QcxBasisDataDir) + "/" + std::string(spec.basisDir),
                         std::array<int, 2>{6, 1}));

    if (!basis.has_value())
    {
        return false;
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);

    if (!kinetic.has_value() || !nuclear.has_value())
    {
        return false;
    }

    const std::size_t n = kinetic->Shape()[0];
    Eigen::MatrixXd core =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            core(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                (*kinetic)(i, j) + (*nuclear)(i, j);
        }
    }

    auto coreTensor = qcx::testing::ToTensor(core);

    if (!coreTensor.has_value())
    {
        return false;
    }

    // The fixture's physical-shaped probe density (a diagonal-only density
    // zeroes whole screened classes).
    std::mt19937_64 rng(20260817);
    std::uniform_real_distribution<double> dist(-0.25, 0.25);
    Eigen::MatrixXd density(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = (i == j) ? 0.5 + dist(rng) : dist(rng);
            density(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
            density(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = value;
        }
    }

    auto rho = qcx::testing::ToTensor(0.5 * density);

    if (!rho.has_value())
    {
        return false;
    }

    const auto buildFock =
        [&](const qcx::integrals::QfmmOptions& options) -> std::optional<Eigen::MatrixXd> {
        auto builder =
            qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *coreTensor, options);

        if (!builder.has_value())
        {
            return std::nullopt;
        }

        auto fock = builder->BuildFock(*rho);

        if (!fock.has_value())
        {
            return std::nullopt;
        }

        return qcx::testing::ToMatrix(*fock);
    };

    qcx::integrals::QfmmOptions base;
    base.accuracy = kPreset;
    base.useCertifiedMixedPrecision = false;
    base.maxParallelChunks = 1;
    base.theta = -1.0; // The recorded degenerate gate: everything near field.

    // The RETIRED model is pinned EXPLICITLY: since the 2026-09-15 owner
    // ruling the defaults are the corrected ones, so the "recorded" cells of
    // this limb must name the retired form to keep comparing the same thing.
    const qcx::integrals::QfmmExtentMode retiredExtent =
        qcx::integrals::QfmmExtentMode::kMidpointBound;
    const qcx::integrals::QfmmSeparationMode retiredSeparation =
        qcx::integrals::QfmmSeparationMode::kWidthTheta;
    base.extentModel = retiredExtent;
    base.separationMode = retiredSeparation;
    base.separationK = 0.0;

    const auto nearOnly = buildFock(base);

    if (!nearOnly.has_value())
    {
        output.Line(std::string("ERRORLIMB ") + std::string(spec.label) +
                    ": the near-only build failed");
        return false;
    }

    // The corrected geometry at ITS gate (k < 0): a different tree, the same
    // near-field-everything domain - the bit-identity the gate demands.
    qcx::integrals::QfmmOptions correctedGate = base;
    correctedGate.theta = 0.0;
    correctedGate.extentModel = qcx::integrals::QfmmExtentMode::kProductBall;
    correctedGate.separationMode = qcx::integrals::QfmmSeparationMode::kSurfaceBall;
    correctedGate.separationK = -1.0;
    const auto correctedGateFock = buildFock(correctedGate);
    bool bitIdentical = false;

    if (correctedGateFock.has_value())
    {
        bitIdentical = true;

        for (std::size_t i = 0; i < n && bitIdentical; ++i)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                if ((*nearOnly)(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) !=
                    (*correctedGateFock)(static_cast<Eigen::Index>(i),
                                         static_cast<Eigen::Index>(j)))
                {
                    bitIdentical = false;
                    break;
                }
            }
        }
    }

    const auto report = [&](const char* label, const Eigen::MatrixXd& fock) {
        const double maxAbs = (fock - *nearOnly).cwiseAbs().maxCoeff();
        const double energy = 0.5 * std::abs((density.array() * (fock - *nearOnly).array()).sum());
        char row[256];
        std::snprintf(row,
                      sizeof(row),
                      "ERRORLIMB fixture=%s cell=%s maxAbsF=%.3e deltaE=%.3e",
                      std::string(spec.label).c_str(),
                      label,
                      maxAbs,
                      energy);
        output.Line(row);
        std::fflush(stdout);
    };

    char row[320];
    std::snprintf(row,
                  sizeof(row),
                  "ERRORLIMB fixture=%s cell=gate productGeometryBitIdentical=%s",
                  std::string(spec.label).c_str(),
                  bitIdentical ? "OK" : "FAIL");
    output.Line(row);
    std::fflush(stdout);

    for (const double theta : {0.3, 2.0})
    {
        qcx::integrals::QfmmOptions options = base; // The retired model, explicit.
        options.theta = theta;
        const auto fock = buildFock(options);

        if (fock.has_value())
        {
            char label[64];
            std::snprintf(label, sizeof(label), "recorded_theta%.2f", theta);
            report(label, *fock);
        }
    }

    for (const double k : {0.0, 1.0, 2.0})
    {
        qcx::integrals::QfmmOptions options = base;
        options.theta = 0.0; // The corrected model below overrides both keys.
        options.extentModel = qcx::integrals::QfmmExtentMode::kProductBall;
        options.separationMode = qcx::integrals::QfmmSeparationMode::kSurfaceBall;
        options.separationK = k;
        const auto fock = buildFock(options);

        if (fock.has_value())
        {
            char label[64];
            std::snprintf(label, sizeof(label), "product_surface_k%.1f", k);
            report(label, *fock);
        }
    }

    return true;
}

} // namespace

int main(int argc, char** argv) {
    bool includeBig = true;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--no-big") == 0)
        {
            includeBig = false;
        }
    }

    std::printf(
        "QFMM separation counts (the separation-gate instrument) - counts only, no timings\n");
    std::printf("theta rungs:");

    for (const double theta : kThetas)
    {
        std::printf(" %.2f", theta);
    }

    std::printf(" | leaf caps:");

    for (const std::size_t cap : kLeafCaps)
    {
        std::printf(" %zu", cap);
    }

    std::printf("\n\n");
    std::fflush(stdout);

    Output output(stdout);

    for (const FixtureSpec& spec : kFixtures)
    {
        if (!includeBig && spec.basisDir == "def2-svp" && spec.carbonCount == 42)
        {
            output.Line("SKIP C42H86/def2-SVP (--no-big)");
            continue;
        }

        RunFixture(spec, true, output);

        if (spec.carbonCount <= 12 && spec.basisDir.empty())
        {
            RunErrorLimb(spec, output);
        }
    }

    std::printf("\nthe counts run completed\n");
    return 0;
}
