#include "large_molecules.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/memory/sparsity_pattern.hpp"
#include "qcx/memory/tensor_policies.hpp"
#include "qcx/molecule/connectivity.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace {

using CpuPattern = qcx::memory::SparsityPattern<qcx::backend::CpuTag>;
// The two specializations with real bodies today: the missing-combination
// case is a deliberate compile error, not tested here.
using DenseContraction =
    qcx::memory::ContractionPolicy<qcx::memory::NoSymmetry, qcx::memory::Dense>;
using SparseContraction =
    qcx::memory::ContractionPolicy<qcx::memory::NoSymmetry, qcx::memory::SparsityPatternBacked>;

constexpr std::size_t kC60AtomCount = 60;

/// Deterministic synthetic octree over a 4 x 4 point grid: node 0 is the
/// root, nodes 1..4 the quadrants, nodes 5..20 the sixteen leaves.
std::vector<std::vector<std::size_t>> MakeSyntheticOctree() {
    return {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
        {17, 18, 19, 20},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    };
}

} // namespace

TEST(SparsityPatternTest, RepresentsLinkStyleNeighborListFromRealConnectivity) {
    const auto molecule = qcx::molecule::testing::MakeBuckminsterfullerene();
    ASSERT_TRUE(molecule.has_value());
    const auto connectivity = qcx::molecule::BuildConnectivity(*molecule);

    const auto result = qcx::memory::BuildSparsityFromAdjacency<qcx::backend::CpuTag>(
        connectivity.csr.offsets, connectivity.csr.neighbors);
    ASSERT_TRUE(result.has_value());
    const auto& pattern = *result;

    EXPECT_EQ(pattern.RowCount(), kC60AtomCount);
    EXPECT_EQ(pattern.IndexCount(), connectivity.csr.neighbors.size());
    EXPECT_EQ(pattern.RowOffsets().HostView(), connectivity.csr.offsets);
    EXPECT_EQ(pattern.Indices().HostView(), connectivity.csr.neighbors);

    for (const auto neighbor : pattern.Indices().HostView())
    {
        EXPECT_LT(neighbor, kC60AtomCount);
    }
}

TEST(SparsityPatternTest, RepresentsQfmmStyleTreeParentChildPattern) {
    const auto result =
        qcx::memory::BuildSparsityFromTree<qcx::backend::CpuTag>(MakeSyntheticOctree());
    ASSERT_TRUE(result.has_value());
    const auto& pattern = *result;

    EXPECT_EQ(pattern.RowCount(), 21u);
    EXPECT_EQ(pattern.IndexCount(), 20u);

    const auto& offsets = pattern.RowOffsets().HostView();
    const auto& indices = pattern.Indices().HostView();
    EXPECT_EQ(offsets[1] - offsets[0], 4u); // root has 4 children
    EXPECT_EQ(offsets[5] - offsets[4], 4u); // each quadrant has 4 leaves
    EXPECT_EQ(offsets[6] - offsets[5], 0u); // leaves have no children
    EXPECT_EQ(indices[0], 1u);
}

TEST(SparsityPatternTest, AllowsAllEmptyRowsWithPaddedIndicesBuffer) {
    const auto result =
        qcx::memory::SparsityPattern<qcx::backend::CpuTag>::Create({0, 0, 0, 0}, {});
    ASSERT_TRUE(result.has_value());
    const auto& pattern = *result;

    EXPECT_EQ(pattern.RowCount(), 3u);
    EXPECT_EQ(pattern.IndexCount(), 0u);
    EXPECT_EQ(pattern.RowOffsets().HostView(), std::vector<std::size_t>({0, 0, 0, 0}));
    EXPECT_EQ(pattern.Indices().HostView().size(), 1u); // the padding slot
}

TEST(SparsityPatternTest, RejectsMalformedCsrArrays) {
    const auto mismatched = CpuPattern::Create({0, 2}, {1});
    ASSERT_FALSE(mismatched.has_value());
    EXPECT_EQ(mismatched.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto nonMonotonic = CpuPattern::Create({0, 3, 1}, {0, 1, 2});
    ASSERT_FALSE(nonMonotonic.has_value());
    EXPECT_EQ(nonMonotonic.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto noRows = CpuPattern::Create({0}, {});
    ASSERT_FALSE(noRows.has_value());
    EXPECT_EQ(noRows.error().code, qcx::ErrorCode::kInvalidArgument);
}

// Coverage gap in the adjacency builder: BuildSparsityFromAdjacency passed
// neighbor indices through without column validation (only the tree builder
// checked its range), so an out-of-range packed index reached the
// payload-scatter loops and read or wrote past the payload arrays. Create
// now rejects any index >= rowCount up front.
TEST(SparsityPatternTest, RejectsOutOfRangePackedIndices) {
    const auto adjacency =
        qcx::memory::BuildSparsityFromAdjacency<qcx::backend::CpuTag>({0, 1, 2}, {0, 5});
    ASSERT_FALSE(adjacency.has_value());
    EXPECT_EQ(adjacency.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto direct = CpuPattern::Create({0, 2}, {1, 2});
    ASSERT_FALSE(direct.has_value());
    EXPECT_EQ(direct.error().code, qcx::ErrorCode::kInvalidArgument);

    // An index equal to the row count is just as out of range as a larger
    // one (rows are 0..rowCount - 1).
    const auto boundary = CpuPattern::Create({0, 1}, {1});
    ASSERT_FALSE(boundary.has_value());
    EXPECT_EQ(boundary.error().code, qcx::ErrorCode::kInvalidArgument);

    // The valid in-range case must keep working: the last row's neighbor is
    // rowCount - 1, not an error.
    const auto valid = CpuPattern::Create({0, 1}, {0});
    ASSERT_TRUE(valid.has_value());
    EXPECT_EQ(valid->RowCount(), 1u);
}

TEST(SparsityPatternTest, RejectsNonForestTreeShapes) {
    const auto empty = qcx::memory::BuildSparsityFromTree<qcx::backend::CpuTag>({});
    ASSERT_FALSE(empty.has_value());
    EXPECT_EQ(empty.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto multiParent = qcx::memory::BuildSparsityFromTree<qcx::backend::CpuTag>({{1}, {1}});
    ASSERT_FALSE(multiParent.has_value());
    EXPECT_EQ(multiParent.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto cycle = qcx::memory::BuildSparsityFromTree<qcx::backend::CpuTag>({{1}, {0}});
    ASSERT_FALSE(cycle.has_value());
    EXPECT_EQ(cycle.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto outOfRange = qcx::memory::BuildSparsityFromTree<qcx::backend::CpuTag>({{5}});
    ASSERT_FALSE(outOfRange.has_value());
    EXPECT_EQ(outOfRange.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto selfLoop = qcx::memory::BuildSparsityFromTree<qcx::backend::CpuTag>({{0}});
    ASSERT_FALSE(selfLoop.has_value());
    EXPECT_EQ(selfLoop.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(SparsityPatternTest, ContractEnumeratesEveryLinkStylePairExactlyOnce) {
    const auto molecule = qcx::molecule::testing::MakeBuckminsterfullerene();
    ASSERT_TRUE(molecule.has_value());
    const auto connectivity = qcx::molecule::BuildConnectivity(*molecule);
    const auto result = qcx::memory::BuildSparsityFromAdjacency<qcx::backend::CpuTag>(
        connectivity.csr.offsets, connectivity.csr.neighbors);
    ASSERT_TRUE(result.has_value());

    // A pure count fold over the pattern must visit exactly the packed
    // index space: one visit per packed index catches double-counting and
    // off-by-one offsets without needing payload machinery.
    const std::size_t pairCount = result->Contract<std::size_t>(
        0, [](std::size_t acc, std::size_t, std::size_t, std::size_t) { return acc + 1; });

    EXPECT_EQ(pairCount, result->IndexCount());
    EXPECT_EQ(pairCount, connectivity.csr.neighbors.size());
}

TEST(SparsityPatternTest, ContractMatchesDenseScatterOnLinkStylePattern) {
    const auto molecule = qcx::molecule::testing::MakeBuckminsterfullerene();
    ASSERT_TRUE(molecule.has_value());
    const auto connectivity = qcx::molecule::BuildConnectivity(*molecule);
    const auto result = qcx::memory::BuildSparsityFromAdjacency<qcx::backend::CpuTag>(
        connectivity.csr.offsets, connectivity.csr.neighbors);
    ASSERT_TRUE(result.has_value());
    const auto& pattern = *result;

    // A payload aligned with the packed indices. Positive values keep every
    // partial sum away from -0.0, so the zeros the dense sum interleaves
    // cannot flip a sign bit (x + 0.0 == x exactly for x != -0.0).
    std::vector<double> payload(pattern.IndexCount());

    for (std::size_t k = 0; k < payload.size(); ++k)
    {
        payload[k] = 1.0 + static_cast<double>(k);
    }

    // Dense side: scatter the payload at exactly the covered entries, then
    // sum the whole matrix (the uncovered entries stay zero).
    const std::size_t n = pattern.RowCount();
    std::vector<double> dense(n * n, 0.0);
    const auto& offsets = pattern.RowOffsets().HostView();
    const auto& indices = pattern.Indices().HostView();

    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t k = offsets[row]; k < offsets[row + 1]; ++k)
        {
            dense[row * n + indices[k]] = payload[k];
        }
    }

    double denseSum = 0.0;

    for (double value : dense)
    {
        denseSum += value;
    }

    // Sparse side: the same sum through Contract, which must enumerate the
    // identical (row, k) pairs in the identical CSR order.
    const double sparseSum = pattern.Contract<double>(
        0.0, [&payload](double acc, std::size_t, std::size_t k, std::size_t) {
            return acc + payload[k];
        });

    EXPECT_EQ(denseSum, sparseSum);
}

TEST(SparsityPatternTest, ContractMatchesDenseScatterOnTreePattern) {
    const auto result =
        qcx::memory::BuildSparsityFromTree<qcx::backend::CpuTag>(MakeSyntheticOctree());
    ASSERT_TRUE(result.has_value());
    const auto& pattern = *result;

    std::vector<double> payload(pattern.IndexCount());

    for (std::size_t k = 0; k < payload.size(); ++k)
    {
        payload[k] = 1.0 + static_cast<double>(k);
    }

    const std::size_t n = pattern.RowCount();
    std::vector<double> dense(n * n, 0.0);
    const auto& offsets = pattern.RowOffsets().HostView();
    const auto& indices = pattern.Indices().HostView();

    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t k = offsets[row]; k < offsets[row + 1]; ++k)
        {
            dense[row * n + indices[k]] = payload[k];
        }
    }

    double denseSum = 0.0;

    for (double value : dense)
    {
        denseSum += value;
    }

    const double sparseSum = pattern.Contract<double>(
        0.0, [&payload](double acc, std::size_t, std::size_t k, std::size_t) {
            return acc + payload[k];
        });

    EXPECT_EQ(denseSum, sparseSum);
}

TEST(SparsityPatternTest, ContractOverEmptyPatternReturnsAccumulatorUnchanged) {
    const auto result = CpuPattern::Create({0, 0, 0, 0}, {});
    ASSERT_TRUE(result.has_value());

    const double sum = result->Contract<double>(
        7.0, [](double acc, std::size_t, std::size_t, std::size_t) { return acc + 1.0; });

    EXPECT_EQ(sum, 7.0);
}

TEST(SparsityPatternTest, ContractionPolicyDenseIsPlainProduct) {
    // The Dense x NoSymmetry specialization is a thin pass-through: exactly
    // a * b - what every existing Fock/RI call site already computes
    // directly - nothing more.
    const Eigen::MatrixXd a = (Eigen::MatrixXd(2, 3) << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0).finished();
    const Eigen::MatrixXd b =
        (Eigen::MatrixXd(3, 4) << 2.0, 0.0, 1.0, 3.0, 1.0, 3.0, 4.0, 0.0, 0.0, 2.0, 1.0, 1.0)
            .finished();
    const Eigen::MatrixXd expected = a * b;
    const Eigen::MatrixXd actual = DenseContraction::Contract(a, b);

    EXPECT_EQ(actual.rows(), expected.rows());
    EXPECT_EQ(actual.cols(), expected.cols());
    EXPECT_TRUE(expected.isApprox(actual));
}

TEST(SparsityPatternTest, ContractionPolicySparseMatchesDenseScatterOnLinkStylePattern) {
    const auto molecule = qcx::molecule::testing::MakeBuckminsterfullerene();
    ASSERT_TRUE(molecule.has_value());
    const auto connectivity = qcx::molecule::BuildConnectivity(*molecule);
    const auto result = qcx::memory::BuildSparsityFromAdjacency<qcx::backend::CpuTag>(
        connectivity.csr.offsets, connectivity.csr.neighbors);
    ASSERT_TRUE(result.has_value());
    const auto& pattern = *result;

    // A payload aligned with the packed indices. Positive values keep every
    // partial sum away from -0.0, so the zeros the dense sum interleaves
    // cannot flip a sign bit (x + 0.0 == x exactly for x != -0.0).
    std::vector<double> payload(pattern.IndexCount());

    for (std::size_t k = 0; k < payload.size(); ++k)
    {
        payload[k] = 1.0 + static_cast<double>(k);
    }

    // Dense reference: scatter the payload at exactly the covered entries
    // (the uncovered entries stay zero).
    const std::size_t n = pattern.RowCount();
    Eigen::MatrixXd reference =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    const auto& offsets = pattern.RowOffsets().HostView();
    const auto& indices = pattern.Indices().HostView();

    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t k = offsets[row]; k < offsets[row + 1]; ++k)
        {
            reference(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(indices[k])) +=
                payload[k];
        }
    }

    // Policy side: the same masked accumulation through the dispatch -
    // touching only the (row, col) pairs the pattern lists.
    const Eigen::MatrixXd zero =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    const Eigen::MatrixXd actual = SparseContraction::Contract(zero, pattern, payload);

    EXPECT_TRUE((reference.array() == actual.array()).all());

    // The fold form enumerates the identical pairs with payload lookup: the
    // same sum through the pattern's own Contract, reached via the policy.
    const double denseSum = reference.sum();
    const double sparseSum = SparseContraction::Contract(
        pattern, 0.0, [&payload](double acc, std::size_t, std::size_t k, std::size_t) {
            return acc + payload[k];
        });

    EXPECT_EQ(denseSum, sparseSum);
}

TEST(SparsityPatternTest, ContractionPolicySparseMatchesDenseScatterOnTreePattern) {
    const auto result =
        qcx::memory::BuildSparsityFromTree<qcx::backend::CpuTag>(MakeSyntheticOctree());
    ASSERT_TRUE(result.has_value());
    const auto& pattern = *result;

    std::vector<double> payload(pattern.IndexCount());

    for (std::size_t k = 0; k < payload.size(); ++k)
    {
        payload[k] = 1.0 + static_cast<double>(k);
    }

    const std::size_t n = pattern.RowCount();
    Eigen::MatrixXd reference =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    const auto& offsets = pattern.RowOffsets().HostView();
    const auto& indices = pattern.Indices().HostView();

    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t k = offsets[row]; k < offsets[row + 1]; ++k)
        {
            reference(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(indices[k])) +=
                payload[k];
        }
    }

    const Eigen::MatrixXd zero =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    const Eigen::MatrixXd actual = SparseContraction::Contract(zero, pattern, payload);

    EXPECT_TRUE((reference.array() == actual.array()).all());

    const double denseSum = reference.sum();
    const double sparseSum = SparseContraction::Contract(
        pattern, 0.0, [&payload](double acc, std::size_t, std::size_t k, std::size_t) {
            return acc + payload[k];
        });

    EXPECT_EQ(denseSum, sparseSum);
}

TEST(SparsityPatternTest, ContractionPolicySparseOnEmptyPatternReturnsDenseUnchanged) {
    // The all-empty pattern (a tree whose nodes are all leaves) pads the
    // indices buffer with one unused slot; the masked accumulation must
    // touch nothing and leave the dense operand bit-identical - including
    // the empty payload (no packed index, no read).
    const auto result = CpuPattern::Create({0, 0, 0, 0}, {});
    ASSERT_TRUE(result.has_value());

    const Eigen::MatrixXd dense = Eigen::MatrixXd::Identity(3, 3);
    const Eigen::MatrixXd actual = SparseContraction::Contract(dense, *result, {});

    EXPECT_TRUE(dense.isApprox(actual));
}
