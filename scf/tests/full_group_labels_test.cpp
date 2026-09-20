// The full-group labeling stage (a posteriori). The tests pin:
//   - the lambda-path operator's exact spectrum (the z-axis component-squared
//     operator on spherical and Cartesian shells),
//   - the labels of synthetic states built from the symmetry-blocked basis U
//     (F = U diag(eps) U^T, C = U R): the expected full-group label
//     multiset per molecule, the per-block label uniformity (exact for
//     these fixtures - the correlation multiplicity is 1 per block),
//   - the degenerate-partner canonicalization: F-restricted eigensolve
//     recovers U's columns from arbitrarily rotated inputs (near-degenerate
//     energy pairs), and the D-invariance of the canonicalized density,
//   - the aufbau-straddle skip: a subspace with occupied AND virtual MOs is
//     recorded, NOT rotated,
//   - the symmetrized density: commutes with every realized AO action,
//     trace-preserving, symmetric,
//   - the linear-molecule lambda path: CO2 labels vs an independent
//     parity-vector (C2z / inversion / sigma-v) check,
//   - the C1 trivial case.
//
// The synthetic states are pure classification inputs: nothing here runs an
// SCF, and no energy is computed by the stage (labels and the symmetrized
// density only; the density of the canonicalized state is invariant by the
// straddle rule, which the tests assert directly).

#include "corpus.hpp"
#include "large_molecules.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/symmetry_labels.hpp"
#include "qcx/scf/uhf.hpp"
#include "qcx/symmetry/detection.hpp"
#include "qcx/symmetry/full_group_tables.hpp"
#include "qcx/symmetry/point_group.hpp"
#include "qcx/symmetry/point_group_name.hpp"
#include "scf_common.hpp"
#include "symmetry_blocks.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using qcx::scf::DegenerateSubspaceRecord;
using qcx::scf::kUnknownLabel;
using qcx::scf::LabelOutcome;
using qcx::scf::LabelVector;
using qcx::scf::LabelVectors;
using qcx::scf::SymmetryLabels;
using qcx::scf::VectorLabel;
using qcx::scf::internal::AngularMomentumSquaredAboutZ;
using qcx::scf::internal::BuildBlockedDiagonalizeData;
using qcx::scf::internal::BuildFullGroupRealization;
using qcx::scf::internal::BuildSymmetryBlocks;
using qcx::scf::internal::DiagonalizeFock;
using qcx::scf::internal::DiagonalizeFockBlocked;
using qcx::scf::internal::FullGroupClass;
using qcx::scf::internal::FullGroupElement;
using qcx::scf::internal::FullGroupRealization;
using qcx::scf::internal::OrthogonalizeOverlap;
using qcx::scf::internal::SymmetryBlocks;
using qcx::scf::internal::SymmetryLabelAndSymmetrize;
using qcx::symmetry::DetectPointGroup;
using qcx::symmetry::FullGroupTableFor;
using qcx::symmetry::PointGroupName;

// STO-3G merged from the vendored corpus files (identical literals to the
// fixtures, but single-source for C, O, H, F, Cl alike). Repeated symbols
// (H, H, H in ammonia) merge once each - BasisSet::Merge rejects duplicates.
qcx::Result<qcx::basisset::BasisSet> MakeSto3g(const std::vector<std::string_view>& symbols) {
    qcx::basisset::BasisSet merged;
    std::vector<std::string_view> seen;

    for (const std::string_view symbol : symbols)
    {
        if (std::find(seen.begin(), seen.end(), symbol) != seen.end())
        {
            continue;
        }

        seen.push_back(symbol);
        const std::string path =
            std::string(QcxBasisDataDir) + "/sto-3g/" + std::string(symbol) + ".nwchem";
        auto parsed = qcx::basisset::ParseNwchemFile(path);

        if (!parsed.has_value())
        {
            return std::unexpected(parsed.error());
        }

        auto mergeResult = merged.Merge(*parsed);

        if (!mergeResult.has_value())
        {
            return std::unexpected(mergeResult.error());
        }
    }

    return merged;
}

// The synthetic SCF state on the symmetry-blocked basis U: F = U diag(eps)
// U^T (so U's columns are the exact F eigenvectors), C = U R (R the
// within-block partner rotations the arbitrary inputs carry), D = 2 C_occ
// C_occ^T (the RHF spin-summed density).
struct SyntheticState {
    Eigen::MatrixXd fock;
    Eigen::MatrixXd coefficients;
    Eigen::MatrixXd density;
};

SyntheticState MakeSynthetic(const Eigen::MatrixXd& u,
                             const Eigen::VectorXd& eps,
                             const Eigen::MatrixXd& rotation,
                             int occupiedCount) {
    Eigen::MatrixXd coefficients = u * rotation;
    SyntheticState state;
    state.fock = u * eps.asDiagonal() * u.transpose();
    state.coefficients = coefficients;
    state.density = 2.0 * coefficients.leftCols(occupiedCount) *
                    coefficients.leftCols(occupiedCount).transpose();
    return state;
}

// Column-wise sign-agnostic closeness: the eigensolver's column signs are
// arbitrary, so two columns may agree up to a global sign.
void ExpectColumnsClose(const Eigen::MatrixXd& a,
                        const Eigen::MatrixXd& b,
                        double tolerance,
                        const std::vector<int>& columns) {
    for (const int j : columns)
    {
        const double d1 = (a.col(j) - b.col(j)).norm();
        const double d2 = (a.col(j) + b.col(j)).norm();
        EXPECT_LT(std::min(d1, d2), tolerance) << "column " << j << " differs";
    }
}

// The pure per-irrep columns of the synthetic states: the orthonormal
// ranges of the isotypic operators sum_g chi_i(class g) A(g) (the scalar
// normalization is irrelevant for the range, so the nu factor is skipped).
// The blocked basis mixes full-group irreps whenever a computational row
// subducts from several (NH3/Cs: A' = A1 + E), so the raw U columns are
// NOT per-column label-pure; a converged symmetric Fock's MOs are (the
// arbitrary mixing of a degenerate pair stays INSIDE the irrep). The
// synthetic states therefore use these pure columns.
void PureIrrepBasis(const Eigen::MatrixXd& u,
                    const FullGroupRealization& realization,
                    const qcx::symmetry::FullGroupTable& table,
                    Eigen::MatrixXd& out) {
    const Eigen::Index n = u.cols();
    out.resize(n, n);
    Eigen::Index offset = 0;

    for (int i = 0; i < table.irrepCount; ++i)
    {
        // Runtime class -> table class by the (kind, order, power, slot)
        // keys (the stage's matching; the scalar normalization is irrelevant
        // for the RANGE, so the nu factor is skipped here).
        std::vector<int> runtimeToTable(realization.classes.size(), -1);

        for (std::size_t c = 0; c < realization.classes.size(); ++c)
        {
            const FullGroupClass& cls = realization.classes[c];

            for (int t = 0; t < table.classCount; ++t)
            {
                const qcx::symmetry::OperationClass& candidate = table.classes[t];

                if (candidate.kind == cls.kind && candidate.order == cls.order &&
                    candidate.power == cls.power && candidate.axis == cls.slot)
                {
                    runtimeToTable[c] = t;
                    break;
                }
            }
        }

        Eigen::MatrixXd isotypic = Eigen::MatrixXd::Zero(n, n);

        for (int g = 0; g < realization.groupOrder; ++g)
        {
            const int runtimeClass = realization.classOfElement[static_cast<std::size_t>(g)];
            isotypic += table.characters[i][static_cast<std::size_t>(
                            runtimeToTable[static_cast<std::size_t>(runtimeClass)])] *
                        realization.actions[static_cast<std::size_t>(g)];
        }

        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(isotypic);
        const Eigen::Index blockSize =
            static_cast<Eigen::Index>((solver.eigenvalues().array().abs() > 1e-8).count());
        out.middleCols(offset, blockSize) = solver.eigenvectors().rightCols(blockSize);
        offset += blockSize;
    }
}

// Every column of every symmetry block carries the same full-group label (exact
// for water and methane, where each Abelian row subduced from exactly one
// full irrep; NOT for ammonia - its Cs rows mix A1 and E, see
// PureIrrepBasis).
void ExpectBlockUniformLabels(const qcx::scf::internal::SymmetryBlocks& blocks,
                              const std::vector<std::string>& labels) {
    Eigen::Index offset = 0;

    for (const Eigen::Index size : blocks.blockSizes)
    {
        SCOPED_TRACE(offset);

        for (Eigen::Index j = 1; j < size; ++j)
        {
            EXPECT_EQ(labels[static_cast<std::size_t>(offset + j)],
                      labels[static_cast<std::size_t>(offset)]);
        }

        offset += size;
    }
}

// The density symmetrization's defining property: D_sym commutes with every
// realized AO action (A D_sym A^T = D_sym for all g in the group), the trace
// is preserved, and D_sym is symmetric.
void ExpectSymmetrizedDensity(const Eigen::MatrixXd& density,
                              const Eigen::MatrixXd& inputDensity,
                              const FullGroupRealization& realization,
                              double tolerance) {
    EXPECT_NEAR((density - density.transpose()).norm(), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(density.trace() - inputDensity.trace()), 0.0, 1e-9);

    for (const auto& action : realization.actions)
    {
        const Eigen::MatrixXd transformed = action * density * action.transpose();
        EXPECT_NEAR((transformed - density).norm(), 0.0, tolerance)
            << "D_sym does not commute with an action";
    }
}

// The aufbau-straddle rule, checked against the labels and the occupancy
// window alone: every dim >= 2 MANIFOLD - a maximal set of same-label columns
// whose consecutive energies are within kManifoldEnergyTolerance - is
// canonicalized exactly when its columns are entirely occupied or entirely
// virtual, and recorded (not rotated) otherwise.
//
// The energy cut is the rule, not a detail: without it two spatially separate
// manifolds that share one irrep label form one group, and the boundary test is
// then applied to their union.
void ExpectRecordsMatchOccupancy(const SymmetryLabels& result,
                                 const Eigen::MatrixXd& fock,
                                 int occupiedCount) {
    const int columnCount = static_cast<int>(result.labels.size());
    std::vector<std::pair<double, int>> ordered;

    for (int j = 0; j < columnCount; ++j)
    {
        const Eigen::VectorXd column = result.coefficients.col(j);
        ordered.emplace_back(column.dot(fock * column), j);
    }

    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
    });

    std::map<std::string, int> dimensions;

    for (int j = 0; j < columnCount; ++j)
    {
        const int irrep = result.irrepIndices[j];

        if (irrep >= 0)
        {
            dimensions[result.labels[j]] = FullGroupTableFor(result.fullGroup)->dimensions[irrep];
        } else if (result.labels[j] == std::string(kUnknownLabel))
        {
            // UNKNOWN carries no irrep and no dimension: it is never a
            // manifold member.
            dimensions[result.labels[j]] = 0;
        } else
        {
            // The lambda path: the Sigma sectors are non-degenerate, every
            // higher sector is a 2-dim partner pair.
            dimensions[result.labels[j]] = result.labels[j].starts_with("S") ? 1 : 2;
        }
    }

    std::vector<DegenerateSubspaceRecord> expectedCanonicalized;
    std::vector<DegenerateSubspaceRecord> expectedStraddled;

    std::size_t start = 0;

    while (start < ordered.size())
    {
        std::size_t end = start + 1;

        while (end < ordered.size() && ordered[end].first - ordered[end - 1].first <=
                                           qcx::scf::internal::kManifoldEnergyTolerance)
        {
            ++end;
        }

        std::map<std::string, std::vector<int>> byLabel;

        for (std::size_t k = start; k < end; ++k)
        {
            byLabel[result.labels[static_cast<std::size_t>(ordered[k].second)]].push_back(
                ordered[k].second);
        }

        for (auto& [label, indices] : byLabel)
        {
            if (dimensions[label] < 2 || indices.size() < 2)
            {
                continue;
            }

            std::sort(indices.begin(), indices.end());
            const int occupiedIn = static_cast<int>(
                std::count_if(indices.begin(), indices.end(), [occupiedCount](int j) {
                    return j < occupiedCount;
                }));
            const bool entirelyOneSide =
                occupiedIn == 0 || occupiedIn == static_cast<int>(indices.size());
            DegenerateSubspaceRecord record{label, indices};

            if (entirelyOneSide)
            {
                expectedCanonicalized.push_back(std::move(record));
            } else
            {
                expectedStraddled.push_back(std::move(record));
            }
        }

        start = end;
    }

    // Sorted by their lowest column: two records of one label make a
    // label-only sort no longer a total order.
    const auto sortByFirstColumn = [](std::vector<DegenerateSubspaceRecord>& records) {
        std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
            return a.moIndices.front() < b.moIndices.front();
        });
    };
    std::vector<DegenerateSubspaceRecord> canonicalized = result.canonicalized;
    std::vector<DegenerateSubspaceRecord> straddled = result.straddled;
    sortByFirstColumn(canonicalized);
    sortByFirstColumn(straddled);
    sortByFirstColumn(expectedCanonicalized);
    sortByFirstColumn(expectedStraddled);

    ASSERT_EQ(canonicalized.size(), expectedCanonicalized.size());
    ASSERT_EQ(straddled.size(), expectedStraddled.size());

    for (std::size_t k = 0; k < canonicalized.size(); ++k)
    {
        EXPECT_EQ(canonicalized[k].irrepLabel, expectedCanonicalized[k].irrepLabel);
        EXPECT_EQ(canonicalized[k].moIndices, expectedCanonicalized[k].moIndices);
    }

    for (std::size_t k = 0; k < straddled.size(); ++k)
    {
        EXPECT_EQ(straddled[k].irrepLabel, expectedStraddled[k].irrepLabel);
        EXPECT_EQ(straddled[k].moIndices, expectedStraddled[k].moIndices);
    }
}

// ---------------------------------------------------------------------------
// The lambda-path operator
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, LambdaOperatorSpectrum) {
    // Spherical shells: the operator is exactly diag((index - l)^2), so the
    // eigenvalues are the squares of m = -l..+l. SelfAdjointEigenSolver
    // returns them sorted ascending, hence the reordered expectations.
    const auto spherical2 = AngularMomentumSquaredAboutZ(2, true);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver2(spherical2);
    const Eigen::VectorXd expected2((Eigen::VectorXd(5) << 0.0, 1.0, 1.0, 4.0, 4.0).finished());
    EXPECT_NEAR((solver2.eigenvalues() - expected2).cwiseAbs().maxCoeff(), 0.0, 1e-9);

    const auto spherical3 = AngularMomentumSquaredAboutZ(3, true);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver3(spherical3);
    const Eigen::VectorXd expected3(
        (Eigen::VectorXd(7) << 0.0, 1.0, 1.0, 4.0, 4.0, 9.0, 9.0).finished());
    EXPECT_NEAR((solver3.eigenvalues() - expected3).cwiseAbs().maxCoeff(), 0.0, 1e-9);

    // Cartesian shells: the closed-form monomial map symmetrized by the Gram
    // congruence (S = G^(1/2) M^T G^(-1/2), similar to the operator). The
    // eigenvalues are the m^2 values with the z^2-direction twice degenerate
    // (m = 0), the xz/yz pair (m = +-1) and the x^2-y^2/xy pair (m = +-2).
    const auto cartesian2 = AngularMomentumSquaredAboutZ(2, false);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solverC2(cartesian2);
    const Eigen::VectorXd expectedC2(
        (Eigen::VectorXd(6) << 0.0, 0.0, 1.0, 1.0, 4.0, 4.0).finished());
    EXPECT_NEAR((solverC2.eigenvalues() - expectedC2).cwiseAbs().maxCoeff(), 0.0, 1e-9);

    const auto cartesian3 = AngularMomentumSquaredAboutZ(3, false);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solverC3(cartesian3);
    const Eigen::VectorXd expectedC3(
        (Eigen::VectorXd(10) << 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 9.0, 9.0).finished());
    EXPECT_NEAR((solverC3.eigenvalues() - expectedC3).cwiseAbs().maxCoeff(), 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// C1: the trivial case
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, C1Trivial) {
    const auto molecule = qcx::symmetry::testing::MakeGenericC1();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"C", "H", "F", "Cl"});
    ASSERT_TRUE(basis.has_value());

    const auto analysis = DetectPointGroup(*molecule);
    EXPECT_EQ(analysis.group, PointGroupName::kC1);

    const auto blocks = BuildSymmetryBlocks(*molecule, *basis, analysis);
    ASSERT_TRUE(blocks.has_value());
    // C [2s,1p] 5 + H 1 + F 5 + Cl [3s,2p] 9 = 20.
    EXPECT_EQ(blocks->u.cols(), 20);

    const Eigen::VectorXd eps = Eigen::VectorXd::LinSpaced(20, 0.5, 2.0);
    const SyntheticState state =
        MakeSynthetic(blocks->u, eps, Eigen::MatrixXd::Identity(20, 20), 10);
    const auto result = SymmetryLabelAndSymmetrize(
        state.fock, state.coefficients, state.density, 10, *molecule, *basis, analysis);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->fullGroup, PointGroupName::kC1);
    EXPECT_EQ(result->labels, std::vector<std::string>(20, "A"));
    EXPECT_EQ(result->irrepIndices, std::vector<int>(20, 0));
    EXPECT_EQ(result->averagedElementCount, 1);
    EXPECT_TRUE(result->canonicalized.empty());
    EXPECT_TRUE(result->straddled.empty());
    EXPECT_NEAR((result->coefficients - state.coefficients).norm(), 0.0, 1e-12);
    EXPECT_NEAR((result->density - state.density).norm(), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// H2O: C2v, all-irreps-1-dim
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, WaterC2vLabels) {
    const auto molecule = qcx::symmetry::testing::MakeWater();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"O", "H", "H"});
    ASSERT_TRUE(basis.has_value());

    const auto analysis = DetectPointGroup(*molecule);
    ASSERT_EQ(analysis.group, PointGroupName::kC2v);

    const auto blocks = BuildSymmetryBlocks(*molecule, *basis, analysis);
    ASSERT_TRUE(blocks.has_value());

    const int n = static_cast<int>(blocks->u.cols());
    const Eigen::VectorXd eps = Eigen::VectorXd::LinSpaced(n, 1.0, 2.0);
    const SyntheticState state = MakeSynthetic(blocks->u, eps, Eigen::MatrixXd::Identity(n, n), 5);
    const auto result = SymmetryLabelAndSymmetrize(
        state.fock, state.coefficients, state.density, 5, *molecule, *basis, analysis);
    ASSERT_TRUE(result.has_value());

    // STO-3G water: 1s/2s/2pz + the symmetric H combination are A1, 2py +
    // the antisymmetric H combination B2, 2px B1, A2 empty.
    const std::map<std::string, int> counts = [&result]() {
        std::map<std::string, int> out;

        for (const auto& label : result->labels)
        {
            ++out[label];
        }

        return out;
    }();
    const std::map<std::string, int> expected{{"A1", 4}, {"B2", 2}, {"B1", 1}};
    EXPECT_EQ(counts, expected);

    ExpectBlockUniformLabels(*blocks, result->labels);
    ExpectRecordsMatchOccupancy(*result, state.fock, 5);
    EXPECT_TRUE(result->canonicalized.empty());
    EXPECT_TRUE(result->straddled.empty());
    EXPECT_NEAR((result->coefficients - state.coefficients).norm(), 0.0, 1e-12);

    const auto realization = BuildFullGroupRealization(*molecule, *basis, analysis);
    ASSERT_TRUE(realization.has_value());
    EXPECT_EQ(realization->elements.size(), 4u);
    EXPECT_EQ(result->averagedElementCount, 4);
    ExpectSymmetrizedDensity(result->density, state.density, *realization, 1e-8);
}

// ---------------------------------------------------------------------------
// NH3: C3v, E-pair canonicalization recovery
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, AmmoniaC3vCanonicalization) {
    const auto molecule = qcx::symmetry::testing::MakeAmmonia();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"N", "H", "H", "H"});
    ASSERT_TRUE(basis.has_value());

    const auto analysis = DetectPointGroup(*molecule);
    ASSERT_EQ(analysis.group, PointGroupName::kC3v);

    const auto blocks = BuildSymmetryBlocks(*molecule, *basis, analysis);
    ASSERT_TRUE(blocks.has_value());

    // The C3v reduction is Cs: the A' block mixes A1 and E columns (a
    // computational row subducing from two full irreps), so the synthetic
    // state is built on the PURE per-irrep columns (columns 0..3 = A1,
    // 4..7 = E - what a converged symmetric Fock's eigenbasis looks like).
    const auto realization = BuildFullGroupRealization(*molecule, *basis, analysis);
    ASSERT_TRUE(realization.has_value());
    const qcx::symmetry::FullGroupTable* table = FullGroupTableFor(analysis.group);
    ASSERT_NE(table, nullptr);
    Eigen::MatrixXd pure;
    PureIrrepBasis(blocks->u, *realization, *table, pure);

    // The E energies form two near-degenerate pairs - the synthetic
    // "arbitrary partner rotations" the canonicalization must repair.
    const Eigen::VectorXd eps(
        (Eigen::VectorXd(8) << 0.5, 0.8, 1.2, 1.5, 0.3, 0.3 + 1e-8, 1.0, 1.0 + 1e-8).finished());
    Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(8, 8);

    // The degenerate pairs are rotated by 0.7 rad and 1.3 rad; the A1 block
    // stays put.
    {
        const double theta1 = 0.7;
        const double theta2 = 1.3;
        Eigen::Matrix2d r1;
        r1 << std::cos(theta1), -std::sin(theta1), std::sin(theta1), std::cos(theta1);
        Eigen::Matrix2d r2;
        r2 << std::cos(theta2), -std::sin(theta2), std::sin(theta2), std::cos(theta2);
        rotation.block<2, 2>(4, 4) = r1;
        rotation.block<2, 2>(6, 6) = r2;
    }

    const SyntheticState state = MakeSynthetic(pure, eps, rotation, 8);
    const auto result = SymmetryLabelAndSymmetrize(
        state.fock, state.coefficients, state.density, 8, *molecule, *basis, analysis);
    ASSERT_TRUE(result.has_value());

    const std::map<std::string, int> counts = [&result]() {
        std::map<std::string, int> out;

        for (const auto& label : result->labels)
        {
            ++out[label];
        }

        return out;
    }();
    const std::map<std::string, int> expected{{"A1", 4}, {"E", 4}};
    EXPECT_EQ(counts, expected);

    ExpectRecordsMatchOccupancy(*result, state.fock, 8);
    // TWO E manifolds, TWO records (the energy-then-label key). Grouped by
    // the label alone they collapse into one four-column record, and the
    // aufbau-straddle test is then applied to their UNION - see
    // AmmoniaC3vTwoEManifolds for the reproduction on today's code.
    ASSERT_EQ(result->canonicalized.size(), 2u);
    EXPECT_EQ(result->canonicalized[0].irrepLabel, "E");
    EXPECT_EQ(result->canonicalized[0].moIndices, (std::vector<int>{4, 5}));
    EXPECT_EQ(result->canonicalized[1].irrepLabel, "E");
    EXPECT_EQ(result->canonicalized[1].moIndices, (std::vector<int>{6, 7}));
    EXPECT_TRUE(result->straddled.empty());

    // The canonicalization recovers the exact F eigenvectors (the pure
    // columns) from the arbitrarily rotated inputs.
    ExpectColumnsClose(result->coefficients, pure, 1e-6, {4, 5, 6, 7});

    // The density of the canonicalized state is invariant (the subspace was
    // entirely occupied).
    const Eigen::MatrixXd canonicalizedDensity =
        2.0 * result->coefficients.leftCols(8) * result->coefficients.leftCols(8).transpose();
    EXPECT_NEAR((canonicalizedDensity - state.density).norm() / state.density.norm(), 0.0, 1e-9);

    EXPECT_EQ(realization->elements.size(), 6u);
    EXPECT_EQ(result->averagedElementCount, 6);
    ExpectSymmetrizedDensity(result->density, state.density, *realization, 1e-8);
}

// ---------------------------------------------------------------------------
// NH3: the aufbau-straddle skip (half the E pair occupied)
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, AmmoniaC3vStraddleSkip) {
    const auto molecule = qcx::symmetry::testing::MakeAmmonia();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"N", "H", "H", "H"});
    ASSERT_TRUE(basis.has_value());

    const auto analysis = DetectPointGroup(*molecule);
    const auto blocks = BuildSymmetryBlocks(*molecule, *basis, analysis);
    ASSERT_TRUE(blocks.has_value());

    // Pure columns again (see AmmoniaC3vCanonicalization): A1 at 0..3, the
    // two E pairs at 4..7.
    const auto realization = BuildFullGroupRealization(*molecule, *basis, analysis);
    ASSERT_TRUE(realization.has_value());
    const qcx::symmetry::FullGroupTable* table = FullGroupTableFor(analysis.group);
    ASSERT_NE(table, nullptr);
    Eigen::MatrixXd pure;
    PureIrrepBasis(blocks->u, *realization, *table, pure);

    // Occupancy cuts THROUGH one manifold: of the E pair {4, 5} at ~1.0,
    // column 4 is occupied and column 5 is virtual (occupiedCount = 5). The
    // other E columns 6 (1.2) and 7 (1.5) are separate single-column clusters
    // and are not dim >= 2 manifolds at all, so they are neither rotated nor
    // recorded.
    const Eigen::VectorXd eps(
        (Eigen::VectorXd(8) << 0.3, 0.3 + 1e-8, 0.5, 0.8, 1.0, 1.0 + 1e-8, 1.2, 1.5).finished());
    Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(8, 8);

    {
        const double theta = 0.7;
        Eigen::Matrix2d r1;
        r1 << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);
        rotation.block<2, 2>(4, 4) = r1;
    }

    const SyntheticState state = MakeSynthetic(pure, eps, rotation, 5);
    const auto result = SymmetryLabelAndSymmetrize(
        state.fock, state.coefficients, state.density, 5, *molecule, *basis, analysis);
    ASSERT_TRUE(result.has_value());

    ExpectRecordsMatchOccupancy(*result, state.fock, 5);
    // The record names the PAIR the boundary actually cuts, not every column
    // that shares its label (the key is the energy cluster, then the
    // label; before it this record read {4, 5, 6, 7}).
    ASSERT_EQ(result->straddled.size(), 1u);
    EXPECT_EQ(result->straddled[0].irrepLabel, "E");
    EXPECT_EQ(result->straddled[0].moIndices, (std::vector<int>{4, 5}));
    EXPECT_TRUE(result->canonicalized.empty());

    // The straddled subspace is NOT rotated: the arbitrary input partners
    // survive verbatim (the honest-output rule).
    EXPECT_NEAR((result->coefficients - state.coefficients).norm(), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// NH3: the E irrep's TWO manifolds are two records, not one
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, AmmoniaC3vTwoEManifolds) {
    const auto molecule = qcx::symmetry::testing::MakeAmmonia();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"N", "H", "H", "H"});
    ASSERT_TRUE(basis.has_value());

    const auto analysis = DetectPointGroup(*molecule);
    ASSERT_EQ(analysis.group, PointGroupName::kC3v);

    const auto blocks = BuildSymmetryBlocks(*molecule, *basis, analysis);
    ASSERT_TRUE(blocks.has_value());

    const auto realization = BuildFullGroupRealization(*molecule, *basis, analysis);
    ASSERT_TRUE(realization.has_value());
    const qcx::symmetry::FullGroupTable* table = FullGroupTableFor(analysis.group);
    ASSERT_NE(table, nullptr);
    Eigen::MatrixXd pure;
    PureIrrepBasis(blocks->u, *realization, *table, pure);

    // The E irrep carries TWO spatially separate partner pairs - {4, 5} at
    // ~0.3 and {6, 7} at ~1.0 - and both are entirely occupied (occupiedCount
    // = 8). Grouped by the LABEL alone they are ONE four-column subspace;
    // grouped by ENERGY and then by label they are two two-column manifolds,
    // and that is the record a reader needs: two separate degenerate states,
    // not one. This is the measured ammonia case.
    const Eigen::VectorXd eps(
        (Eigen::VectorXd(8) << 0.5, 0.8, 1.2, 1.5, 0.3, 0.3 + 1e-8, 1.0, 1.0 + 1e-8).finished());
    Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(8, 8);

    {
        const double theta1 = 0.7;
        const double theta2 = 1.3;
        Eigen::Matrix2d r1;
        r1 << std::cos(theta1), -std::sin(theta1), std::sin(theta1), std::cos(theta1);
        Eigen::Matrix2d r2;
        r2 << std::cos(theta2), -std::sin(theta2), std::sin(theta2), std::cos(theta2);
        rotation.block<2, 2>(4, 4) = r1;
        rotation.block<2, 2>(6, 6) = r2;
    }

    const SyntheticState state = MakeSynthetic(pure, eps, rotation, 8);
    const auto result = SymmetryLabelAndSymmetrize(
        state.fock, state.coefficients, state.density, 8, *molecule, *basis, analysis);
    ASSERT_TRUE(result.has_value());

    // TWO two-index records, one per manifold - not one four-index record.
    ASSERT_EQ(result->canonicalized.size(), 2u);
    EXPECT_EQ(result->canonicalized[0].irrepLabel, "E");
    EXPECT_EQ(result->canonicalized[0].moIndices, (std::vector<int>{4, 5}));
    EXPECT_EQ(result->canonicalized[1].irrepLabel, "E");
    EXPECT_EQ(result->canonicalized[1].moIndices, (std::vector<int>{6, 7}));
    EXPECT_TRUE(result->straddled.empty());

    // Every column here is DETERMINED: the collapse was a record defect, not a
    // labelling one, and no column is pushed into UNKNOWN by the fix.
    for (const auto outcome : result->labelOutcomes)
    {
        EXPECT_EQ(outcome, LabelOutcome::kDetermined);
    }

    // Each manifold recovers its own Fock eigenbasis from its own arbitrary
    // rotation: the rotation never mixed the two pairs, and the record now
    // says which columns belong together.
    ExpectColumnsClose(result->coefficients, pure, 1e-6, {4, 5, 6, 7});

    // The density is invariant: both manifolds are entirely occupied.
    const Eigen::MatrixXd canonicalizedDensity =
        2.0 * result->coefficients.leftCols(8) * result->coefficients.leftCols(8).transpose();
    EXPECT_NEAR((canonicalizedDensity - state.density).norm() / state.density.norm(), 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// The UNKNOWN outcome: a vector with no dominant irrep
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, AmmoniaC3vEqualMixtureIsUnknown) {
    const auto molecule = qcx::symmetry::testing::MakeAmmonia();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"N", "H", "H", "H"});
    ASSERT_TRUE(basis.has_value());

    const auto analysis = DetectPointGroup(*molecule);
    const auto blocks = BuildSymmetryBlocks(*molecule, *basis, analysis);
    ASSERT_TRUE(blocks.has_value());

    const auto realization = BuildFullGroupRealization(*molecule, *basis, analysis);
    ASSERT_TRUE(realization.has_value());
    const qcx::symmetry::FullGroupTable* table = FullGroupTableFor(analysis.group);
    ASSERT_NE(table, nullptr);
    Eigen::MatrixXd pure;
    PureIrrepBasis(blocks->u, *realization, *table, pure);

    // Column 7 is an EQUAL mixture of one pure A1 column (0) and one pure E
    // column (4): each isotypic component then carries half the weight. No
    // single irrep dominates, so the honest answer is UNKNOWN - an argmax over
    // this vector names whichever component happens to be ahead by roundoff.
    Eigen::MatrixXd mixture = pure;
    mixture.col(7) = (pure.col(0) + pure.col(4)).normalized();

    const Eigen::VectorXd eps(
        (Eigen::VectorXd(8) << 0.5, 0.8, 1.2, 1.5, 0.3, 0.3 + 1e-8, 1.0, 1.0 + 1e-8).finished());
    const SyntheticState state = MakeSynthetic(mixture, eps, Eigen::MatrixXd::Identity(8, 8), 8);

    const auto result = SymmetryLabelAndSymmetrize(
        state.fock, state.coefficients, state.density, 8, *molecule, *basis, analysis);
    ASSERT_TRUE(result.has_value());

    // The balanced mixture is UNKNOWN, and the un-mixed columns around it are
    // NOT: the rule separates them rather than degrading a whole state.
    EXPECT_EQ(result->labels[7], "UNKNOWN");
    EXPECT_EQ(result->labelOutcomes[7], LabelOutcome::kUnknown);
    EXPECT_EQ(result->irrepIndices[7], -1);
    EXPECT_EQ(result->labels[0], "A1");
    EXPECT_EQ(result->labels[4], "E");
    EXPECT_EQ(result->labelOutcomes[0], LabelOutcome::kDetermined);
    EXPECT_EQ(result->labelOutcomes[4], LabelOutcome::kDetermined);
}

// ---------------------------------------------------------------------------
// Labelling an arbitrary caller-supplied vector, off the SCF path
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, LabelVectorOffTheScfPath) {
    const auto molecule = qcx::symmetry::testing::MakeAmmonia();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"N", "H", "H", "H"});
    ASSERT_TRUE(basis.has_value());

    const auto analysis = DetectPointGroup(*molecule);
    const auto blocks = BuildSymmetryBlocks(*molecule, *basis, analysis);
    ASSERT_TRUE(blocks.has_value());

    const auto realization = BuildFullGroupRealization(*molecule, *basis, analysis);
    ASSERT_TRUE(realization.has_value());
    const qcx::symmetry::FullGroupTable* table = FullGroupTableFor(analysis.group);
    ASSERT_NE(table, nullptr);
    Eigen::MatrixXd pure;
    PureIrrepBasis(blocks->u, *realization, *table, pure);

    // A vector that is NOT an MO column of any state: the caller supplies it,
    // and there is no SCF state and no Fock in sight.
    const auto labeled = LabelVector(pure.col(0), *molecule, *basis, analysis);
    ASSERT_TRUE(labeled.has_value());
    EXPECT_EQ(labeled->outcome, LabelOutcome::kDetermined);
    EXPECT_EQ(labeled->label, "A1");
    EXPECT_GE(labeled->irrepIndex, 0);
    EXPECT_NEAR(labeled->normSquared, 1.0, 1e-12);
    EXPECT_NEAR(labeled->weightFraction, 1.0, 1e-9);
    // The projections resolve the identity on this vector: the numbers are
    // carried so a reader can check the outcome rather than trust it.
    EXPECT_NEAR(labeled->projectorResidual, 0.0, 1e-9);

    // The degenerate irrep is named too, and the E partner pair gets the SAME
    // answer - the label is the irrep, not a partner index.
    const auto eFirst = LabelVector(pure.col(4), *molecule, *basis, analysis);
    ASSERT_TRUE(eFirst.has_value());
    EXPECT_EQ(eFirst->outcome, LabelOutcome::kDetermined);
    EXPECT_EQ(eFirst->label, "E");
    EXPECT_NE(eFirst->irrepIndex, labeled->irrepIndex);
    // The index is a row into the table the label came from, not a private
    // numbering: a consumer can read the label back off the table.
    ASSERT_GE(eFirst->irrepIndex, 0);
    EXPECT_EQ(std::string(table->irrepLabels[static_cast<std::size_t>(eFirst->irrepIndex)]),
              eFirst->label);

    // A balanced A1/E mixture: no single irrep dominates, so the answer is
    // UNKNOWN, NO index is named, and the weight fraction shows why (it is the
    // 0.5 the rule refuses to resolve into a winner).
    const Eigen::VectorXd mixed = (pure.col(0) + pure.col(4)).normalized();
    const auto mixedLabeled = LabelVector(mixed, *molecule, *basis, analysis);
    ASSERT_TRUE(mixedLabeled.has_value());
    EXPECT_EQ(mixedLabeled->outcome, LabelOutcome::kUnknown);
    EXPECT_EQ(mixedLabeled->label, std::string(kUnknownLabel));
    EXPECT_EQ(mixedLabeled->irrepIndex, -1);
    EXPECT_NEAR(mixedLabeled->weightFraction, 0.5, 1e-6);

    // The batch form builds the realization ONCE and labels identically.
    Eigen::MatrixXd batch(blocks->u.cols(), 3);
    batch.col(0) = pure.col(0);
    batch.col(1) = pure.col(4);
    batch.col(2) = mixed;
    const auto batchLabeled = LabelVectors(batch, *molecule, *basis, analysis);
    ASSERT_TRUE(batchLabeled.has_value());
    ASSERT_EQ(batchLabeled->size(), 3u);
    EXPECT_EQ((*batchLabeled)[0].label, labeled->label);
    EXPECT_EQ((*batchLabeled)[0].outcome, labeled->outcome);
    EXPECT_EQ((*batchLabeled)[1].label, eFirst->label);
    EXPECT_EQ((*batchLabeled)[2].outcome, LabelOutcome::kUnknown);

    // A vector of the wrong length is REFUSED, not read past or padded: the
    // entry point does not guess what the caller meant.
    const auto refused = LabelVector(Eigen::VectorXd::Zero(3), *molecule, *basis, analysis);
    EXPECT_FALSE(refused.has_value());
    EXPECT_NE(refused.error().message.find("rows"), std::string::npos);

    // A linear group is refused BY NAME: its label is read off the |lambda|
    // operator, not scored from isotypic projectors, so the rule this entry
    // point documents is not the rule in force there.
    const auto linearMolecule = qcx::symmetry::testing::MakeCarbonDioxide();
    ASSERT_TRUE(linearMolecule.has_value());
    auto linearBasis = MakeSto3g({"C", "O", "O"});
    ASSERT_TRUE(linearBasis.has_value());
    const auto linearAnalysis = DetectPointGroup(*linearMolecule);
    ASSERT_EQ(linearAnalysis.group, PointGroupName::kDInfH);
    // The length is deliberately wrong here: the refusal is by GROUP and it
    // comes first, so a caller cannot learn the vector's shape by being told
    // its length is wrong.
    const auto linearRefused =
        LabelVector(Eigen::VectorXd::Zero(1), *linearMolecule, *linearBasis, linearAnalysis);
    EXPECT_FALSE(linearRefused.has_value());
    EXPECT_NE(linearRefused.error().message.find("linear group"), std::string::npos);
}

// ---------------------------------------------------------------------------
// CH4: Td, distinct energies carry NO manifold (a 3-dim irrep)
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, MethaneTdDistinctEnergiesCarryNoManifold) {
    const auto molecule = qcx::symmetry::testing::MakeMethane();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"C", "H", "H", "H", "H"});
    ASSERT_TRUE(basis.has_value());

    const auto analysis = DetectPointGroup(*molecule);
    ASSERT_EQ(analysis.group, PointGroupName::kTd);

    const auto blocks = BuildSymmetryBlocks(*molecule, *basis, analysis);
    ASSERT_TRUE(blocks.has_value());

    // The D2 reduction of Td: A1(Td) subduced to A1, T2(Td) to the three B
    // rows (the correlation-multiplicity exception the fixtures avoid).
    const int n = static_cast<int>(blocks->u.cols());
    // ONE energy per column, so no two columns are degenerate. The six T2
    // columns therefore form no manifold at all: the energy cut is what makes
    // a manifold degenerate, and these columns are already the Fock's
    // own eigenvectors (F = u eps u^T is diagonal in u), so there is no
    // arbitrary rotation to canonicalize and nothing to record. Grouped by the
    // LABEL alone they were reported as ONE six-column straddled subspace -
    // the same collapse AmmoniaC3vTwoEManifolds reproduces with two pairs.
    Eigen::VectorXd eps(n);

    for (int j = 0; j < n; ++j)
    {
        eps(j) = 0.3 + 0.15 * static_cast<double>(j);
    }

    // Five occupied, out of a state with no manifold to split: the A1 sector
    // (dim 1) is never canonicalized either.
    const SyntheticState state = MakeSynthetic(blocks->u, eps, Eigen::MatrixXd::Identity(n, n), 5);
    const auto result = SymmetryLabelAndSymmetrize(
        state.fock, state.coefficients, state.density, 5, *molecule, *basis, analysis);
    ASSERT_TRUE(result.has_value());

    const std::map<std::string, int> counts = [&result]() {
        std::map<std::string, int> out;

        for (const auto& label : result->labels)
        {
            ++out[label];
        }

        return out;
    }();
    const std::map<std::string, int> expected{{"A1", 3}, {"T2", 6}};
    EXPECT_EQ(counts, expected);

    ExpectBlockUniformLabels(*blocks, result->labels);
    ExpectRecordsMatchOccupancy(*result, state.fock, 5);
    EXPECT_TRUE(result->straddled.empty());
    EXPECT_TRUE(result->canonicalized.empty());
    EXPECT_NEAR((result->coefficients - state.coefficients).norm(), 0.0, 1e-12);

    const auto realization = BuildFullGroupRealization(*molecule, *basis, analysis);
    ASSERT_TRUE(realization.has_value());
    EXPECT_EQ(realization->elements.size(), 24u);
    EXPECT_EQ(result->averagedElementCount, 24);
    ExpectSymmetrizedDensity(result->density, state.density, *realization, 1e-8);
}

// ---------------------------------------------------------------------------
// C60: Ih, 300 functions (5 per carbon atom)
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, FullereneIhSynthetic) {
    const auto molecule = qcx::molecule::testing::MakeBuckminsterfullerene();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g(std::vector<std::string_view>(60, "C"));
    ASSERT_TRUE(basis.has_value());

    const auto analysis = DetectPointGroup(*molecule);
    ASSERT_EQ(analysis.group, PointGroupName::kIh);

    const auto blocks = BuildSymmetryBlocks(*molecule, *basis, analysis);
    ASSERT_TRUE(blocks.has_value());

    const int n = static_cast<int>(blocks->u.cols());
    // STO-3G carbon is 5 functions per atom (2s + 2p).
    EXPECT_EQ(n, 300);

    // Distinct energies everywhere; the Fock's eigenbasis is the blocked basis
    // itself, so the canonicalization recovery is directly checkable.
    const Eigen::VectorXd eps = Eigen::VectorXd::LinSpaced(n, 0.1, 24.0);
    const SyntheticState state =
        MakeSynthetic(blocks->u, eps, Eigen::MatrixXd::Identity(n, n), n / 2);
    const auto result = SymmetryLabelAndSymmetrize(
        state.fock, state.coefficients, state.density, n / 2, *molecule, *basis, analysis);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Every label is an Ih irrep label; the label set is small (the STO-3G
    // basis reaches the fully symmetric, T1u, T2u, G, H, ... sectors).
    const auto* table = FullGroupTableFor(result->fullGroup);
    ASSERT_NE(table, nullptr);
    EXPECT_EQ(result->fullGroup, PointGroupName::kIh);

    for (const auto& label : result->labels)
    {
        // A column of a MIXED block has no single full-group irrep of its
        // own, and the analysis REPORTS UNKNOWN for it rather than naming
        // the argmax - the guess this loop used to accept. Any label that IS
        // named must still be a real table irrep.
        if (label == std::string(kUnknownLabel))
        {
            continue;
        }

        bool found = false;

        for (int i = 0; i < table->irrepCount; ++i)
        {
            found = found || table->irrepLabels[i] == label;
        }

        EXPECT_TRUE(found) << "label " << label << " not in the Ih table";
    }

    // And the UNKNOWNs are reported CONSISTENTLY: no irrep index, the outcome
    // saying so, and no dimension borrowed from a table row.
    for (int j = 0; j < static_cast<int>(result->labels.size()); ++j)
    {
        if (result->labels[static_cast<std::size_t>(j)] == std::string(kUnknownLabel))
        {
            EXPECT_EQ(result->labelOutcomes[static_cast<std::size_t>(j)], LabelOutcome::kUnknown);
            EXPECT_EQ(result->irrepIndices[static_cast<std::size_t>(j)], -1);
        } else
        {
            EXPECT_EQ(result->labelOutcomes[static_cast<std::size_t>(j)],
                      LabelOutcome::kDetermined);
        }
    }

    ExpectRecordsMatchOccupancy(*result, state.fock, n / 2);

    // Canonicalized subspaces: the Fock becomes diagonal within them, and
    // the eigenbasis recovers the blocked columns; straddled subspaces are left
    // verbatim.
    std::vector<int> rotated;

    for (const auto& record : result->canonicalized)
    {
        rotated.insert(rotated.end(), record.moIndices.begin(), record.moIndices.end());

        for (const int i : record.moIndices)
        {
            for (const int j : record.moIndices)
            {
                if (i != j)
                {
                    EXPECT_LT(std::abs(result->coefficients.col(i).dot(
                                  state.fock * result->coefficients.col(j))),
                              1e-8);
                }
            }
        }
    }

    ExpectColumnsClose(result->coefficients, blocks->u, 1e-6, rotated);

    for (const auto& record : result->straddled)
    {
        for (const int j : record.moIndices)
        {
            EXPECT_NEAR(
                (result->coefficients.col(j) - state.coefficients.col(j)).norm(), 0.0, 1e-12);
        }
    }

    // The canonicalized density is invariant.
    const Eigen::MatrixXd canonicalizedDensity = 2.0 * result->coefficients.leftCols(n / 2) *
                                                 result->coefficients.leftCols(n / 2).transpose();
    EXPECT_NEAR((canonicalizedDensity - state.density).norm() / state.density.norm(), 0.0, 1e-9);

    const auto realization = BuildFullGroupRealization(*molecule, *basis, analysis);
    ASSERT_TRUE(realization.has_value());
    EXPECT_EQ(realization->elements.size(), 120u);
    EXPECT_EQ(result->averagedElementCount, 120);
    ExpectSymmetrizedDensity(result->density, state.density, *realization, 1e-8);
}

// ---------------------------------------------------------------------------
// CO2: the linear-molecule lambda path
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, CarbonDioxideLinearLabels) {
    const auto molecule = qcx::symmetry::testing::MakeCarbonDioxide();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"C", "O", "O"});
    ASSERT_TRUE(basis.has_value());

    const auto analysis = DetectPointGroup(*molecule);
    ASSERT_EQ(analysis.group, PointGroupName::kDInfH);

    const auto blocks = BuildSymmetryBlocks(*molecule, *basis, analysis);
    ASSERT_TRUE(blocks.has_value());

    const int n = static_cast<int>(blocks->u.cols());
    const Eigen::VectorXd eps = Eigen::VectorXd::LinSpaced(n, 0.4, 2.8);
    const SyntheticState state = MakeSynthetic(blocks->u, eps, Eigen::MatrixXd::Identity(n, n), 6);
    const auto result = SymmetryLabelAndSymmetrize(
        state.fock, state.coefficients, state.density, 6, *molecule, *basis, analysis);
    ASSERT_TRUE(result.has_value());

    // The independent parity-vector check: the C2z, inversion and sigma-v
    // eigenvalues of each column determine its label - C2z = +1 means an
    // even lambda (S), -1 an odd one (Pi); inversion the g/u; the sigma-v
    // parity splits Sigma+ from Sigma-.
    const auto realization = BuildFullGroupRealization(*molecule, *basis, analysis);
    ASSERT_TRUE(realization.has_value());
    EXPECT_EQ(realization->elements.size(), 8u);

    const auto findElement =
        [&realization](qcx::symmetry::OperationKind kind,
                       int order,
                       const Eigen::Vector3d& axis) -> const FullGroupElement* {
        for (const auto& element : realization->elements)
        {
            if (element.kind != kind || element.order != order)
            {
                continue;
            }

            if ((element.axis - axis).norm() < 1e-4)
            {
                return &element;
            }
        }

        return nullptr;
    };

    const Eigen::Vector3d zAxis(0.0, 0.0, 1.0);
    const Eigen::Vector3d yAxis(0.0, 1.0, 0.0);
    const FullGroupElement* c2z = findElement(qcx::symmetry::OperationKind::kRotation, 2, zAxis);
    const FullGroupElement* inversion =
        findElement(qcx::symmetry::OperationKind::kInversion, 1, Eigen::Vector3d::Zero());
    const FullGroupElement* sigmaV = findElement(qcx::symmetry::OperationKind::kSigma, 1, yAxis);
    ASSERT_NE(c2z, nullptr);
    ASSERT_NE(inversion, nullptr);
    ASSERT_NE(sigmaV, nullptr);

    const int c2zIndex = static_cast<int>(c2z - realization->elements.data());
    const int inversionIndex = static_cast<int>(inversion - realization->elements.data());
    const int sigmaVIndex = static_cast<int>(sigmaV - realization->elements.data());
    const Eigen::MatrixXd& c2zAction = realization->actions[static_cast<std::size_t>(c2zIndex)];
    const Eigen::MatrixXd& inversionAction =
        realization->actions[static_cast<std::size_t>(inversionIndex)];
    const Eigen::MatrixXd& sigmaVAction =
        realization->actions[static_cast<std::size_t>(sigmaVIndex)];

    for (int j = 0; j < n; ++j)
    {
        const Eigen::VectorXd column = result->coefficients.col(j);
        const double c2zParity = column.dot(c2zAction * column);
        const double inversionParity = column.dot(inversionAction * column);
        const double sigmaVParity = column.dot(sigmaVAction * column);
        ASSERT_NEAR(std::abs(c2zParity), 1.0, 1e-6);
        ASSERT_NEAR(std::abs(inversionParity), 1.0, 1e-6);
        ASSERT_NEAR(std::abs(sigmaVParity), 1.0, 1e-6);

        std::string expected;

        if (c2zParity > 0.0)
        {
            // The sigma label carries the sigma-v parity (+/-) and the
            // INVERSION parity (g/u) - a Sigma_u+ orbital is even under
            // sigma-v and odd under inversion.
            expected = std::string("S") + (sigmaVParity > 0.0 ? "+" : "-") +
                       (inversionParity > 0.0 ? "g" : "u");
        } else
        {
            expected = inversionParity > 0.0 ? "Pig" : "Piu";
        }

        EXPECT_EQ(result->labels[static_cast<std::size_t>(j)], expected)
            << "column " << j << " c2z=" << c2zParity << " i=" << inversionParity
            << " sv=" << sigmaVParity;
        EXPECT_EQ(result->irrepIndices[static_cast<std::size_t>(j)], -1);
    }

    ExpectRecordsMatchOccupancy(*result, state.fock, 6);
    EXPECT_EQ(result->averagedElementCount, 8);
    EXPECT_EQ(result->symmetrizationSubset.size(), 8u);
    ExpectSymmetrizedDensity(result->density, state.density, *realization, 1e-8);
}

// ---------------------------------------------------------------------------
// Ethene: the D2h slot convention is physically correct and orientation-
// invariant (the D2-family principal rule)
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, EtheneD2hLabelsStandardAndTilted) {
    using qcx::scf::internal::BuildSymmetryBlocks;
    using qcx::symmetry::ClassAxis;
    using qcx::symmetry::OperationKind;

    const auto molecule = qcx::symmetry::testing::MakeEthene();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"C", "C", "H", "H", "H", "H"});
    ASSERT_TRUE(basis.has_value());

    // The tilted copy: a 45-degree rotation about x - a "bad" rotation that
    // moves the raw-lex-min axis off its standard direction (the D2h class
    // sizes are all 1, so the runtime-vs-table size gate cannot catch a
    // silent slot permutation: the fixture pins the assignment itself).
    const Eigen::Matrix3d rotation =
        (Eigen::AngleAxisd(std::numbers::pi / 4.0, Eigen::Vector3d::UnitX())).toRotationMatrix();
    const auto tilt =
        [&rotation](const qcx::molecule::Molecule& source) -> qcx::Result<qcx::molecule::Molecule> {
        const auto& coords = source.CoordinatesBohr();
        const std::size_t atomCount = coords.Shape()[0];
        auto tensor = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({atomCount, 3});

        if (!tensor.has_value())
        {
            return std::unexpected(tensor.error());
        }

        for (std::size_t a = 0; a < atomCount; ++a)
        {
            const Eigen::Vector3d rotated =
                rotation * Eigen::Vector3d(coords(a, 0), coords(a, 1), coords(a, 2));

            for (std::size_t i = 0; i < 3; ++i)
            {
                (*tensor)(a, i) = rotated(static_cast<Eigen::Index>(i));
            }
        }

        tensor->MarkHostDirty();
        return qcx::molecule::Molecule::Create(source.Atoms(), std::move(*tensor), 0, 1);
    };
    auto tilted = tilt(*molecule);
    ASSERT_TRUE(tilted.has_value());

    // The synthetic-state stage run for one orientation: labels plus the
    // realization the class-key assertions read.
    const auto runStage = [&basis](const qcx::molecule::Molecule& mol)
        -> qcx::Result<std::pair<std::vector<std::string>, FullGroupRealization>> {
        const auto analysis = DetectPointGroup(mol);

        if (analysis.group != PointGroupName::kD2h)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "the fixture must detect as D2h"});
        }

        const auto blocks = BuildSymmetryBlocks(mol, *basis, analysis);

        if (!blocks.has_value())
        {
            return std::unexpected(blocks.error());
        }

        const int n = static_cast<int>(blocks->u.cols());
        const Eigen::VectorXd eps = Eigen::VectorXd::LinSpaced(n, 0.4, 2.8);
        const SyntheticState state =
            MakeSynthetic(blocks->u, eps, Eigen::MatrixXd::Identity(n, n), 7);
        auto result = SymmetryLabelAndSymmetrize(
            state.fock, state.coefficients, state.density, 7, mol, *basis, analysis);

        if (!result.has_value())
        {
            return std::unexpected(result.error());
        }

        auto realization = BuildFullGroupRealization(mol, *basis, analysis);

        if (!realization.has_value())
        {
            return std::unexpected(realization.error());
        }

        return std::pair{result->labels, std::move(*realization)};
    };
    const auto standard = runStage(*molecule);
    ASSERT_TRUE(standard.has_value()) << standard.error().message;
    const auto tiltedRun = runStage(*tilted);
    ASSERT_TRUE(tiltedRun.has_value()) << tiltedRun.error().message;
    ASSERT_EQ(standard->second.classes.size(), 8u);

    // The per-MO labels are orientation-invariant.
    ASSERT_EQ(standard->first.size(), tiltedRun->first.size());

    for (std::size_t j = 0; j < standard->first.size(); ++j)
    {
        EXPECT_EQ(standard->first[j], tiltedRun->first[j]) << "column " << j;
    }

    // The standard orientation: the physically-correct assignment, matched
    // to the table's axis letters - C2z and the molecular-plane mirror
    // principal (the D2h principal rule: the C2 whose same-axis mirror
    // fixes the most atoms), C2y/sigma(xz) orbit-0, C2x/sigma(yz) orbit-1.
    const auto slotOf = [](const FullGroupRealization& realization,
                           OperationKind kind,
                           const Eigen::Vector3d& axis) -> std::optional<ClassAxis> {
        for (const FullGroupClass& cls : realization.classes)
        {
            if (cls.kind == kind && (cls.minAxis - axis).norm() < 1e-4)
            {
                return cls.slot;
            }
        }

        return std::nullopt;
    };
    EXPECT_EQ(slotOf(standard->second, OperationKind::kRotation, {0.0, 0.0, 1.0}),
              ClassAxis::kPrincipal);
    EXPECT_EQ(slotOf(standard->second, OperationKind::kRotation, {0.0, 1.0, 0.0}),
              ClassAxis::kOrbit0);
    EXPECT_EQ(slotOf(standard->second, OperationKind::kRotation, {1.0, 0.0, 0.0}),
              ClassAxis::kOrbit1);
    EXPECT_EQ(slotOf(standard->second, OperationKind::kSigma, {0.0, 0.0, 1.0}),
              ClassAxis::kPrincipal);
    EXPECT_EQ(slotOf(standard->second, OperationKind::kSigma, {0.0, 1.0, 0.0}), ClassAxis::kOrbit0);
    EXPECT_EQ(slotOf(standard->second, OperationKind::kSigma, {1.0, 0.0, 0.0}), ClassAxis::kOrbit1);

    // The tilted orientation reproduces the same (kind, order, power, slot)
    // key per class, with the axes the rotated standard ones - the class
    // assignment is rotation-covariant, not merely size-consistent.
    const auto keyOf = [](const FullGroupClass& cls) {
        return std::tuple{cls.kind, cls.order, cls.power, cls.slot};
    };
    std::map<std::tuple<OperationKind, int, int, ClassAxis>, Eigen::Vector3d> tiltedKeys;

    for (const FullGroupClass& cls : tiltedRun->second.classes)
    {
        tiltedKeys[keyOf(cls)] = cls.minAxis;
    }

    EXPECT_EQ(tiltedKeys.size(), standard->second.classes.size());

    for (const FullGroupClass& cls : standard->second.classes)
    {
        const auto it = tiltedKeys.find(keyOf(cls));
        ASSERT_NE(it, tiltedKeys.end()) << "the tilted run lost a class key";

        if (cls.minAxis.norm() > 0.0)
        {
            // Sign-agnostic: the runtime stores the canonicalized axis (first
            // nonzero component positive), which may flip the raw rotated
            // image's sign.
            EXPECT_LT(std::min((it->second - rotation * cls.minAxis).norm(),
                               (it->second + rotation * cls.minAxis).norm()),
                      1e-4)
                << "the tilted class axis is not the rotated standard axis";
        }
    }
}

// ---------------------------------------------------------------------------
// Allene: D2d, the S4-axis branch of the D2-family principal rule (the
// axis carrying an improper order-4 class wins the tie)
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, AlleneD2dS4AxisPrincipalStandardAndTilted) {
    using qcx::scf::internal::BuildSymmetryBlocks;
    using qcx::symmetry::ClassAxis;
    using qcx::symmetry::OperationKind;

    const auto molecule = qcx::symmetry::testing::MakeAllene();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"C", "C", "C", "H", "H", "H", "H"});
    ASSERT_TRUE(basis.has_value());

    // The tilted copy: a 45-degree rotation about x (the D2h fixture's
    // rotation), moving the S4 axis off the raw frame axis.
    const Eigen::Matrix3d rotation =
        (Eigen::AngleAxisd(std::numbers::pi / 4.0, Eigen::Vector3d::UnitX())).toRotationMatrix();
    const auto tilt =
        [&rotation](const qcx::molecule::Molecule& source) -> qcx::Result<qcx::molecule::Molecule> {
        const auto& coords = source.CoordinatesBohr();
        const std::size_t atomCount = coords.Shape()[0];
        auto tensor = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({atomCount, 3});

        if (!tensor.has_value())
        {
            return std::unexpected(tensor.error());
        }

        for (std::size_t a = 0; a < atomCount; ++a)
        {
            const Eigen::Vector3d rotated =
                rotation * Eigen::Vector3d(coords(a, 0), coords(a, 1), coords(a, 2));

            for (std::size_t i = 0; i < 3; ++i)
            {
                (*tensor)(a, i) = rotated(static_cast<Eigen::Index>(i));
            }
        }

        tensor->MarkHostDirty();
        return qcx::molecule::Molecule::Create(source.Atoms(), std::move(*tensor), 0, 1);
    };
    auto tilted = tilt(*molecule);
    ASSERT_TRUE(tilted.has_value());

    // The synthetic-state stage run for one orientation: labels plus the
    // realization the class-key assertions read. Running the stage IS the
    // assignment pin: a principal mis-pick flips the (kind, order, power,
    // slot) keys and the stage's runtime-vs-table class-size gate fires
    // (the D2d principal C2 class has size 1, the C2' class size 2). The
    // pin covers assignment flips; a regression confined to the S4-axis
    // rule alone does not flip allene's assignment (the atom-fixed-count
    // fallback and the residual lex ordering pick the same winner) - the
    // rule's discrimination lives in D2dSyntheticS4AxisPrincipalStandardAndTilted.
    const auto runStage = [&basis](const qcx::molecule::Molecule& mol)
        -> qcx::Result<std::pair<std::vector<std::string>, FullGroupRealization>> {
        const auto analysis = DetectPointGroup(mol);

        if (analysis.group != PointGroupName::kD2d)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "the fixture must detect as D2d"});
        }

        const auto blocks = BuildSymmetryBlocks(mol, *basis, analysis);

        if (!blocks.has_value())
        {
            return std::unexpected(blocks.error());
        }

        const int n = static_cast<int>(blocks->u.cols());
        const Eigen::VectorXd eps = Eigen::VectorXd::LinSpaced(n, 0.4, 2.8);
        const SyntheticState state =
            MakeSynthetic(blocks->u, eps, Eigen::MatrixXd::Identity(n, n), 11);
        auto result = SymmetryLabelAndSymmetrize(
            state.fock, state.coefficients, state.density, 11, mol, *basis, analysis);

        if (!result.has_value())
        {
            return std::unexpected(result.error());
        }

        auto realization = BuildFullGroupRealization(mol, *basis, analysis);

        if (!realization.has_value())
        {
            return std::unexpected(realization.error());
        }

        return std::pair{result->labels, std::move(*realization)};
    };
    const auto standard = runStage(*molecule);
    ASSERT_TRUE(standard.has_value()) << standard.error().message;
    const auto tiltedRun = runStage(*tilted);
    ASSERT_TRUE(tiltedRun.has_value()) << tiltedRun.error().message;
    ASSERT_EQ(standard->second.classes.size(), 5u);
    EXPECT_EQ(standard->second.elements.size(), 8u);

    // Every label is a D2d irrep label.
    const auto* table = FullGroupTableFor(PointGroupName::kD2d);
    ASSERT_NE(table, nullptr);

    for (const auto& label : standard->first)
    {
        bool found = false;

        for (int i = 0; i < table->irrepCount; ++i)
        {
            found = found || table->irrepLabels[i] == label;
        }

        EXPECT_TRUE(found) << "label " << label << " not in the D2d table";
    }

    // The per-MO labels are orientation-invariant.
    ASSERT_EQ(standard->first.size(), tiltedRun->first.size());

    for (std::size_t j = 0; j < standard->first.size(); ++j)
    {
        EXPECT_EQ(standard->first[j], tiltedRun->first[j]) << "column " << j;
    }

    // The standard orientation: the physically-correct assignment, matched
    // to the table's axis letters. Allene's S4 axis is z, so the
    // D2-family rule's improper-order-4 branch makes the C2 ON THE S4 AXIS
    // principal (not one of the two perpendicular C2' axes - the size gate
    // catches a mis-pick, but the pin asserts the positive assignment); the
    // S4 class itself is principal too, and the C2' axes / sigma_d planes
    // are the orbit class(es). The two C2' axes (1,1,0)/sqrt(2) and
    // (1,-1,0)/sqrt(2) are conjugate under the S4 (one class; its lex-min
    // canonical member axis is (1,-1,0)/sqrt(2)); the sigma_d normals y and
    // x likewise (lex-min (0,1,0)).
    const auto slotOf = [](const FullGroupRealization& realization,
                           OperationKind kind,
                           const Eigen::Vector3d& axis) -> std::optional<ClassAxis> {
        for (const FullGroupClass& cls : realization.classes)
        {
            if (cls.kind == kind && (cls.minAxis - axis).norm() < 1e-4)
            {
                return cls.slot;
            }
        }

        return std::nullopt;
    };
    EXPECT_EQ(slotOf(standard->second, OperationKind::kRotation, {0.0, 0.0, 1.0}),
              ClassAxis::kPrincipal);
    EXPECT_EQ(slotOf(standard->second, OperationKind::kImproper, {0.0, 0.0, 1.0}),
              ClassAxis::kPrincipal);
    EXPECT_EQ(slotOf(standard->second,
                     OperationKind::kRotation,
                     Eigen::Vector3d(1.0, -1.0, 0.0).normalized()),
              ClassAxis::kOrbit0);
    EXPECT_EQ(slotOf(standard->second, OperationKind::kSigma, {0.0, 1.0, 0.0}), ClassAxis::kOrbit0);

    // The tilted orientation reproduces the same (kind, order, power, slot)
    // key per class, with the member axes the rotated standard ones - the
    // class assignment is rotation-covariant, not merely size-consistent.
    // The comparison is over the member-axis LINES, not the lex-min
    // representative: the min axis of a multi-member class is a
    // noise-sensitive pick (a rotated molecule's eigh-path axes carry solver
    // rounding that an exact-component lexicographic compare keys on - the
    // allene C2' class's lex-min flips members between the standard and the
    // tilted frame at the 1e-16 level), while the lines themselves are
    // exact to solver precision.
    const auto keyOf = [](const FullGroupClass& cls) {
        return std::tuple{cls.kind, cls.order, cls.power, cls.slot};
    };
    const auto memberAxesOf = [&keyOf](const FullGroupRealization& realization,
                                       const std::tuple<OperationKind, int, int, ClassAxis>& key) {
        std::vector<Eigen::Vector3d> axes;

        for (std::size_t i = 0; i < realization.elements.size(); ++i)
        {
            const FullGroupElement& element = realization.elements[i];

            if (element.axis.norm() == 0.0 ||
                keyOf(
                    realization.classes[static_cast<std::size_t>(realization.classOfElement[i])]) !=
                    key)
            {
                continue;
            }

            bool duplicate = false;

            for (const Eigen::Vector3d& have : axes)
            {
                if (std::abs(have.dot(element.axis)) > 1.0 - 1e-4)
                {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate)
            {
                axes.push_back(element.axis);
            }
        }

        return axes;
    };
    std::set<std::tuple<OperationKind, int, int, ClassAxis>> tiltedKeys;

    for (const FullGroupClass& cls : tiltedRun->second.classes)
    {
        tiltedKeys.insert(keyOf(cls));
    }

    EXPECT_EQ(tiltedKeys.size(), standard->second.classes.size());

    for (const FullGroupClass& cls : standard->second.classes)
    {
        ASSERT_NE(tiltedKeys.find(keyOf(cls)), tiltedKeys.end())
            << "the tilted run lost a class key";

        if (cls.minAxis.norm() == 0.0)
        {
            continue;
        }

        // Sign-agnostic: the runtime stores the canonicalized axes (first
        // nonzero component positive), which may flip the rotated image's
        // sign per axis; the dot test is a line test.
        const auto standardMembers = memberAxesOf(standard->second, keyOf(cls));
        const auto tiltedMembers = memberAxesOf(tiltedRun->second, keyOf(cls));
        ASSERT_EQ(standardMembers.size(), tiltedMembers.size());

        for (const Eigen::Vector3d& standardAxis : standardMembers)
        {
            const Eigen::Vector3d rotated = rotation * standardAxis;
            bool matched = false;

            for (const Eigen::Vector3d& tiltedAxis : tiltedMembers)
            {
                if (std::abs(rotated.dot(tiltedAxis)) > 1.0 - 1e-4)
                {
                    matched = true;
                    break;
                }
            }

            EXPECT_TRUE(matched) << "key = (" << static_cast<int>(cls.kind) << ", " << cls.order
                                 << ", " << cls.power << ", " << static_cast<int>(cls.slot)
                                 << "): no tilted member axis on the rotated standard line ("
                                 << rotated.x() << ", " << rotated.y() << ", " << rotated.z()
                                 << ")";
        }
    }
}

// ---------------------------------------------------------------------------
// D2d synthetic: the geometry that discriminates the S4-axis branch of the
// D2-family principal rule. The allene fixture takes the S4-axis return
// but would survive its removal: allene's S4-axis C2 fixes all three
// carbons, so the atom-fixed-count fallback elects the same principal.
// Here the S4 axis (z) carries NO atom - the C2 on it fixes nothing -
// while each C2' (the xy diagonals) passes through two Cl atoms
// and fixes them, so the fallback's counts elect a C2' as principal. Only
// the S4-axis rule (an improper order-4 class on the axis wins the tie)
// yields the physically-correct principal, the C2 on the S4 axis; a
// regression confined to that rule makes the class-size gate fire (the
// principal C2 class has size 1, the C2' class size 2).
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, D2dSyntheticS4AxisPrincipalStandardAndTilted) {
    using qcx::scf::internal::BuildSymmetryBlocks;
    using qcx::symmetry::ClassAxis;
    using qcx::symmetry::OperationKind;

    const auto molecule = qcx::symmetry::testing::MakeD2dSynthetic();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"Cl", "H"});
    ASSERT_TRUE(basis.has_value());

    // The tilted copy: a 45-degree rotation about x (the D2h fixture's
    // rotation), moving the S4 axis off the raw frame axis.
    const Eigen::Matrix3d rotation =
        (Eigen::AngleAxisd(std::numbers::pi / 4.0, Eigen::Vector3d::UnitX())).toRotationMatrix();
    const auto tilt =
        [&rotation](const qcx::molecule::Molecule& source) -> qcx::Result<qcx::molecule::Molecule> {
        const auto& coords = source.CoordinatesBohr();
        const std::size_t atomCount = coords.Shape()[0];
        auto tensor = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({atomCount, 3});

        if (!tensor.has_value())
        {
            return std::unexpected(tensor.error());
        }

        for (std::size_t a = 0; a < atomCount; ++a)
        {
            const Eigen::Vector3d rotated =
                rotation * Eigen::Vector3d(coords(a, 0), coords(a, 1), coords(a, 2));

            for (std::size_t i = 0; i < 3; ++i)
            {
                (*tensor)(a, i) = rotated(static_cast<Eigen::Index>(i));
            }
        }

        tensor->MarkHostDirty();
        return qcx::molecule::Molecule::Create(source.Atoms(), std::move(*tensor), 0, 1);
    };
    auto tilted = tilt(*molecule);
    ASSERT_TRUE(tilted.has_value());

    // The synthetic-state stage run for one orientation: labels plus the
    // realization the class-key assertions read. Running the stage IS the
    // rule pin HERE: a stage-0-only regression (the S4-wins return deleted)
    // elects a C2' principal on the atom-fixed counts and the class-size
    // gate fires (the principal C2 class has size 1, the C2' class size 2).
    const auto runStage = [&basis](const qcx::molecule::Molecule& mol)
        -> qcx::Result<std::pair<std::vector<std::string>, FullGroupRealization>> {
        const auto analysis = DetectPointGroup(mol);

        if (analysis.group != PointGroupName::kD2d)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "the fixture must detect as D2d"});
        }

        const auto blocks = BuildSymmetryBlocks(mol, *basis, analysis);

        if (!blocks.has_value())
        {
            return std::unexpected(blocks.error());
        }

        const int n = static_cast<int>(blocks->u.cols());
        const Eigen::VectorXd eps = Eigen::VectorXd::LinSpaced(n, 0.4, 2.8);
        const SyntheticState state =
            MakeSynthetic(blocks->u, eps, Eigen::MatrixXd::Identity(n, n), 36);
        auto result = SymmetryLabelAndSymmetrize(
            state.fock, state.coefficients, state.density, 36, mol, *basis, analysis);

        if (!result.has_value())
        {
            return std::unexpected(result.error());
        }

        auto realization = BuildFullGroupRealization(mol, *basis, analysis);

        if (!realization.has_value())
        {
            return std::unexpected(realization.error());
        }

        return std::pair{result->labels, std::move(*realization)};
    };
    const auto standard = runStage(*molecule);
    ASSERT_TRUE(standard.has_value()) << standard.error().message;
    const auto tiltedRun = runStage(*tilted);
    ASSERT_TRUE(tiltedRun.has_value()) << tiltedRun.error().message;
    ASSERT_EQ(standard->second.classes.size(), 5u);
    EXPECT_EQ(standard->second.elements.size(), 8u);

    // Every label of either run is a D2d irrep label.
    const auto* table = FullGroupTableFor(PointGroupName::kD2d);
    ASSERT_NE(table, nullptr);

    const auto expectLabelsFromTable = [&table](const std::vector<std::string>& labels) {
        for (const auto& label : labels)
        {
            // The blocked basis spans A1+B1 and A2+B2 (see below), so a MIXED
            // block's column has no single full-group irrep of its own: the
            // analysis reports UNKNOWN for it. Any label that IS named must be
            // a table irrep.
            if (label == std::string(kUnknownLabel))
            {
                continue;
            }

            bool found = false;

            for (int i = 0; i < table->irrepCount; ++i)
            {
                found = found || table->irrepLabels[i] == label;
            }

            EXPECT_TRUE(found) << "label " << label << " not in the D2d table";
        }
    };
    expectLabelsFromTable(standard->first);
    expectLabelsFromTable(tiltedRun->first);

    // The per-column labels of the D2 blocks that mix two full-group irreps
    // are NOT orientation-invariant: the blocked basis spans A1+B1 (the 11-column
    // block) and A2+B2 (the 9-column block), and the within-block basis is
    // the projector eigensolver's choice. Within a mixed block the two
    // irreps share every character but the S4/sigma_d ones (S4 = sigma_d on
    // the A1+B1 block, S4 = -sigma_d on the A2+B2 block), so the per-column
    // argmax reduces to the sign of the S4 expectation of the solver-chosen
    // column - a basis-dependent quantity that flips under the tilt when a
    // column sits near the block's equator (column 24 sits exactly there:
    // A2 = B2 = 0.5 in both orientations, so its argmax is a last-ulp coin
    // flip, not a stable label). The allene fixture's exact per-column
    // invariance reflects its pure blocks. What IS guaranteed: the D2 block
    // membership of every column (the isotypic subspaces are exact), and
    // the exact labels of the pure E1 blocks (argmax score 1.0 against a
    // next-best ~1e-16 on the C2(z)-antisymmetric D2 blocks).
    const auto labelClass = [](const std::string& label) {
        if (label == "A1" || label == "B1")
        {
            return 0; // the D2-A block
        }

        if (label == "A2" || label == "B2")
        {
            return 1; // the antisym-under-both-C2' block
        }

        if (label == std::string(kUnknownLabel))
        {
            return 3; // reported, not guessed
        }

        return 2; // the E1 blocks
    };
    ASSERT_EQ(standard->first.size(), tiltedRun->first.size());

    for (std::size_t j = 0; j < standard->first.size(); ++j)
    {
        const int left = labelClass(standard->first[j]);
        const int right = labelClass(tiltedRun->first[j]);

        if (left == 3 || right == 3)
        {
            // UNKNOWN is the equator case the note above describes, and a
            // mixed block's within-block basis is the solver's choice, so the
            // label legitimately flips with the tilt. What the tilt CANNOT
            // move is the BLOCK: an UNKNOWN column lies in a MIXED block, so
            // the other orientation must not call it a pure E1 one (whose
            // argmax score is 1 against ~1e-16).
            EXPECT_NE(left == 3 ? right : left, 2)
                << "column " << j << " is E1 in one orientation and UNKNOWN in the other";
            continue;
        }

        EXPECT_EQ(left, right) << "column " << j << " crosses a D2 block between orientations ("
                               << standard->first[j] << " vs " << tiltedRun->first[j] << ")";
    }

    for (std::size_t j = 0; j < standard->first.size(); ++j)
    {
        if (standard->first[j] == "E1" || tiltedRun->first[j] == "E1")
        {
            EXPECT_EQ(standard->first[j], "E1") << "column " << j;
            EXPECT_EQ(tiltedRun->first[j], "E1") << "column " << j;
        }
    }

    // The standard orientation: the physically-correct assignment, matched
    // to the table's axis letters. The C2 ON THE S4 AXIS (z) is principal
    // (the D2-family rule's improper-order-4 branch - the fallback would
    // pick a C2' here), the S4 class itself principal too, the C2' axes /
    // sigma_d planes the orbit class(es). The C2' class's lex-min canonical
    // member axis is (1,-1,0)/sqrt(2) and the sigma_d class's (0,1,0),
    // exactly as in the allene fixture (the same axes).
    const auto slotOf = [](const FullGroupRealization& realization,
                           OperationKind kind,
                           const Eigen::Vector3d& axis) -> std::optional<ClassAxis> {
        for (const FullGroupClass& cls : realization.classes)
        {
            if (cls.kind == kind && (cls.minAxis - axis).norm() < 1e-4)
            {
                return cls.slot;
            }
        }

        return std::nullopt;
    };
    EXPECT_EQ(slotOf(standard->second, OperationKind::kRotation, {0.0, 0.0, 1.0}),
              ClassAxis::kPrincipal);
    EXPECT_EQ(slotOf(standard->second, OperationKind::kImproper, {0.0, 0.0, 1.0}),
              ClassAxis::kPrincipal);
    EXPECT_EQ(slotOf(standard->second,
                     OperationKind::kRotation,
                     Eigen::Vector3d(1.0, -1.0, 0.0).normalized()),
              ClassAxis::kOrbit0);
    EXPECT_EQ(slotOf(standard->second, OperationKind::kSigma, {0.0, 1.0, 0.0}), ClassAxis::kOrbit0);

    // The tilted orientation reproduces the same (kind, order, power, slot)
    // key per class, with the member axes the rotated standard ones - the
    // class assignment is rotation-covariant, not merely size-consistent.
    // The comparison is over the member-axis LINES, not the lex-min
    // representative (the min axis of a multi-member class is a
    // noise-sensitive pick; the lines themselves are exact to solver
    // precision).
    const auto keyOf = [](const FullGroupClass& cls) {
        return std::tuple{cls.kind, cls.order, cls.power, cls.slot};
    };
    const auto memberAxesOf = [&keyOf](const FullGroupRealization& realization,
                                       const std::tuple<OperationKind, int, int, ClassAxis>& key) {
        std::vector<Eigen::Vector3d> axes;

        for (std::size_t i = 0; i < realization.elements.size(); ++i)
        {
            const FullGroupElement& element = realization.elements[i];

            if (element.axis.norm() == 0.0 ||
                keyOf(
                    realization.classes[static_cast<std::size_t>(realization.classOfElement[i])]) !=
                    key)
            {
                continue;
            }

            bool duplicate = false;

            for (const Eigen::Vector3d& have : axes)
            {
                if (std::abs(have.dot(element.axis)) > 1.0 - 1e-4)
                {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate)
            {
                axes.push_back(element.axis);
            }
        }

        return axes;
    };
    std::set<std::tuple<OperationKind, int, int, ClassAxis>> tiltedKeys;

    for (const FullGroupClass& cls : tiltedRun->second.classes)
    {
        tiltedKeys.insert(keyOf(cls));
    }

    EXPECT_EQ(tiltedKeys.size(), standard->second.classes.size());

    for (const FullGroupClass& cls : standard->second.classes)
    {
        ASSERT_NE(tiltedKeys.find(keyOf(cls)), tiltedKeys.end())
            << "the tilted run lost a class key";

        if (cls.minAxis.norm() == 0.0)
        {
            continue;
        }

        // Sign-agnostic: the runtime stores the canonicalized axes (first
        // nonzero component positive), which may flip the rotated image's
        // sign per axis; the dot test is a line test.
        const auto standardMembers = memberAxesOf(standard->second, keyOf(cls));
        const auto tiltedMembers = memberAxesOf(tiltedRun->second, keyOf(cls));
        ASSERT_EQ(standardMembers.size(), tiltedMembers.size());

        for (const Eigen::Vector3d& standardAxis : standardMembers)
        {
            const Eigen::Vector3d rotated = rotation * standardAxis;
            bool matched = false;

            for (const Eigen::Vector3d& tiltedAxis : tiltedMembers)
            {
                if (std::abs(rotated.dot(tiltedAxis)) > 1.0 - 1e-4)
                {
                    matched = true;
                    break;
                }
            }

            EXPECT_TRUE(matched) << "key = (" << static_cast<int>(cls.kind) << ", " << cls.order
                                 << ", " << cls.power << ", " << static_cast<int>(cls.slot)
                                 << "): no tilted member axis on the rotated standard line ("
                                 << rotated.x() << ", " << rotated.y() << ", " << rotated.z()
                                 << ")";
        }
    }
}

// ---------------------------------------------------------------------------
// The wiring gates: the toggle is bit-identical off, and the stage is a
// pure add-on (energy invariance) on the wired-in path.
// ---------------------------------------------------------------------------

TEST(FullGroupLabelsTest, RhfLabelingToggleOffIsBitIdenticalAndEnergyInvariant) {
    using qcx::testing::ToMatrix;

    const auto molecule = qcx::symmetry::testing::MakeWater();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"O", "H", "H"});
    ASSERT_TRUE(basis.has_value());

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value());
    ASSERT_TRUE(kinetic.has_value());
    ASSERT_TRUE(nuclear.has_value());

    // The fixed-Fock builder: the loop diagonalizes H_core = T + V once and
    // converges immediately - a full trajectory through the blocked
    // path, the convergence gate, and the post-SCF stage, with no ERI
    // tensor. The stage is a pure add-on, so the two runs must agree
    // bit-for-bit on every field the stage does not touch.
    const Eigen::MatrixXd fixedFock = ToMatrix(*kinetic) + ToMatrix(*nuclear);
    const qcx::scf::FockBuilderFn builder =
        [&fixedFock](const Eigen::MatrixXd&) -> qcx::Result<Eigen::MatrixXd> { return fixedFock; };

    qcx::scf::RhfOptions onOptions; // fullGroupLabeling defaults to true.
    qcx::scf::RhfOptions offOptions;
    offOptions.fullGroupLabeling = false;

    const auto runOn =
        qcx::scf::RunRhfScf(*molecule, ToMatrix(*overlap), fixedFock, onOptions, builder, &*basis);
    const auto runOff =
        qcx::scf::RunRhfScf(*molecule, ToMatrix(*overlap), fixedFock, offOptions, builder, &*basis);
    ASSERT_TRUE(runOn.has_value()) << runOn.error().message;
    ASSERT_TRUE(runOff.has_value()) << runOff.error().message;

    EXPECT_EQ(runOn->converged, runOff->converged);
    EXPECT_EQ(runOn->iterations, runOff->iterations);
    EXPECT_EQ(runOn->totalEnergy, runOff->totalEnergy);
    EXPECT_EQ(runOn->electronicEnergy, runOff->electronicEnergy);
    EXPECT_EQ((runOn->density - runOff->density).norm(), 0.0);
    EXPECT_EQ((runOn->coefficients - runOff->coefficients).norm(), 0.0);
    EXPECT_EQ((runOn->orbitalEnergies - runOff->orbitalEnergies).norm(), 0.0);

    // The restart path is pinned too (the checkpoint the driver reads).
    EXPECT_EQ((runOn->restart.density - runOff->restart.density).norm(), 0.0);
    EXPECT_EQ(runOn->restart.previousTotalEnergy, runOff->restart.previousTotalEnergy);
    EXPECT_EQ(runOn->restart.previousElectronicEnergy, runOff->restart.previousElectronicEnergy);
    EXPECT_EQ(runOn->restart.diis.fockHistory.size(), runOff->restart.diis.fockHistory.size());
    EXPECT_EQ(runOn->restart.diis.errorHistory.size(), runOff->restart.diis.errorHistory.size());

    for (std::size_t i = 0; i < runOn->restart.diis.fockHistory.size(); ++i)
    {
        EXPECT_EQ((runOn->restart.diis.fockHistory[i] - runOff->restart.diis.fockHistory[i]).norm(),
                  0.0);
        EXPECT_EQ(
            (runOn->restart.diis.errorHistory[i] - runOff->restart.diis.errorHistory[i]).norm(),
            0.0);
    }

    // The stage present on the ON run only; the energy invariance is
    // implied by the bit-identity above (the stage never recomputes it).
    ASSERT_TRUE(runOn->symmetryLabels.has_value());
    EXPECT_FALSE(runOff->symmetryLabels.has_value());

    // The wired-in stage labels the converged state exactly like the direct
    // synthetic test (the stage receives the trajectory's real matrices).
    const std::map<std::string, int> counts = [&runOn]() {
        std::map<std::string, int> out;

        for (const auto& label : runOn->symmetryLabels->labels)
        {
            ++out[label];
        }

        return out;
    }();
    const std::map<std::string, int> expected{{"A1", 4}, {"B1", 1}, {"B2", 2}};
    EXPECT_EQ(counts, expected);
}

TEST(FullGroupLabelsTest, UhfLabelingToggleOffIsBitIdenticalAndEnergyInvariant) {
    using qcx::testing::ToMatrix;

    const auto molecule = qcx::symmetry::testing::MakeWater();
    ASSERT_TRUE(molecule.has_value());
    auto basis = MakeSto3g({"O", "H", "H"});
    ASSERT_TRUE(basis.has_value());

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value());
    ASSERT_TRUE(kinetic.has_value());
    ASSERT_TRUE(nuclear.has_value());

    // The fixed two-Fock builder: both channels diagonalize H_core = T + V
    // once and converge immediately - a full trajectory through the blocked
    // path, the convergence gate, and the per-spin post-SCF stage, with no
    // ERI tensor. The stage is a pure add-on, so the two runs must agree
    // bit-for-bit on every field the stage does not touch.
    const Eigen::MatrixXd fixedFock = ToMatrix(*kinetic) + ToMatrix(*nuclear);
    const qcx::scf::UhfFockBuilderFn builder =
        [&fixedFock](
            const Eigen::MatrixXd&,
            const Eigen::MatrixXd&) -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{fixedFock, fixedFock};
    };

    qcx::scf::UhfOptions onOptions; // fullGroupLabeling defaults to true.
    qcx::scf::UhfOptions offOptions;
    offOptions.fullGroupLabeling = false;

    const auto runOn =
        qcx::scf::RunUhfScf(*molecule, ToMatrix(*overlap), fixedFock, onOptions, builder, &*basis);
    const auto runOff =
        qcx::scf::RunUhfScf(*molecule, ToMatrix(*overlap), fixedFock, offOptions, builder, &*basis);
    ASSERT_TRUE(runOn.has_value()) << runOn.error().message;
    ASSERT_TRUE(runOff.has_value()) << runOff.error().message;

    EXPECT_EQ(runOn->converged, runOff->converged);
    EXPECT_EQ(runOn->iterations, runOff->iterations);
    EXPECT_EQ(runOn->totalEnergy, runOff->totalEnergy);
    EXPECT_EQ(runOn->electronicEnergy, runOff->electronicEnergy);
    EXPECT_EQ(runOn->spinSquared, runOff->spinSquared);
    EXPECT_EQ((runOn->densityAlpha - runOff->densityAlpha).norm(), 0.0);
    EXPECT_EQ((runOn->densityBeta - runOff->densityBeta).norm(), 0.0);
    EXPECT_EQ((runOn->coefficientsAlpha - runOff->coefficientsAlpha).norm(), 0.0);
    EXPECT_EQ((runOn->coefficientsBeta - runOff->coefficientsBeta).norm(), 0.0);
    EXPECT_EQ((runOn->orbitalEnergiesAlpha - runOff->orbitalEnergiesAlpha).norm(), 0.0);
    EXPECT_EQ((runOn->orbitalEnergiesBeta - runOff->orbitalEnergiesBeta).norm(), 0.0);

    // The restart path is pinned too (both spins and their DIIS histories).
    EXPECT_EQ((runOn->restart.densityAlpha - runOff->restart.densityAlpha).norm(), 0.0);
    EXPECT_EQ((runOn->restart.densityBeta - runOff->restart.densityBeta).norm(), 0.0);
    EXPECT_EQ(runOn->restart.previousTotalEnergy, runOff->restart.previousTotalEnergy);
    EXPECT_EQ(runOn->restart.previousElectronicEnergy, runOff->restart.previousElectronicEnergy);
    EXPECT_EQ(runOn->restart.diisAlpha.fockHistory.size(),
              runOff->restart.diisAlpha.fockHistory.size());
    EXPECT_EQ(runOn->restart.diisAlpha.errorHistory.size(),
              runOff->restart.diisAlpha.errorHistory.size());
    EXPECT_EQ(runOn->restart.diisBeta.fockHistory.size(),
              runOff->restart.diisBeta.fockHistory.size());
    EXPECT_EQ(runOn->restart.diisBeta.errorHistory.size(),
              runOff->restart.diisBeta.errorHistory.size());

    for (std::size_t i = 0; i < runOn->restart.diisAlpha.fockHistory.size(); ++i)
    {
        EXPECT_EQ(
            (runOn->restart.diisAlpha.fockHistory[i] - runOff->restart.diisAlpha.fockHistory[i])
                .norm(),
            0.0);
        EXPECT_EQ(
            (runOn->restart.diisAlpha.errorHistory[i] - runOff->restart.diisAlpha.errorHistory[i])
                .norm(),
            0.0);
    }

    for (std::size_t i = 0; i < runOn->restart.diisBeta.fockHistory.size(); ++i)
    {
        EXPECT_EQ((runOn->restart.diisBeta.fockHistory[i] - runOff->restart.diisBeta.fockHistory[i])
                      .norm(),
                  0.0);
        EXPECT_EQ(
            (runOn->restart.diisBeta.errorHistory[i] - runOff->restart.diisBeta.errorHistory[i])
                .norm(),
            0.0);
    }

    // The stage present on the ON run only, per spin.
    ASSERT_TRUE(runOn->symmetryLabelsAlpha.has_value());
    ASSERT_TRUE(runOn->symmetryLabelsBeta.has_value());
    EXPECT_FALSE(runOff->symmetryLabelsAlpha.has_value());
    EXPECT_FALSE(runOff->symmetryLabelsBeta.has_value());

    // Both spins carry the water C2v label counts (the two channels are
    // identical here, so the per-spin labels match the RHF gate's).
    const auto countsOf = [](const qcx::scf::SymmetryLabels& labels) {
        std::map<std::string, int> out;

        for (const auto& label : labels.labels)
        {
            ++out[label];
        }

        return out;
    };
    const std::map<std::string, int> expected{{"A1", 4}, {"B1", 1}, {"B2", 2}};
    EXPECT_EQ(countsOf(*runOn->symmetryLabelsAlpha), expected);
    EXPECT_EQ(countsOf(*runOn->symmetryLabelsBeta), expected);
}

// ---------------------------------------------------------------------------
// The linear-dependence removal vs the symmetry stages.
//
// The square X = S^{-1/2} commutes with the point group and the
// per-irrep blocked diagonalization is built on that property; the
// rectangular form does NOT have it, "so it is used only when directions were
// actually removed, where the blocked path falls back to the plain
// n-dimensional solve". The full-group labeling stage has no reduced-space
// contract at all - it labels and symmetrizes the n-dimensional MO set - so it
// stands down with the blocked path instead of refusing the run.
//
// The fixture is the measured linear-dependence deck: two H
// centres 1e-4 bohr apart (5.29177210903e-5 Angstrom) in aug-cc-pVTZ, a real
// diffuse basis whose two centres' shells nearly coincide, so the overlap
// ratio falls far below kOverlapEigenvalueFloorTolerance and the removal
// fires. The Fock is the fixed H_core = T + V (the toggle test's idiom): no
// ERI tensor, and the trajectory still runs the real seam.
// ---------------------------------------------------------------------------

// The two coalesced H centres, in ANGSTROM (the [molecule] atoms convention).
constexpr double kCoalescedSeparation = 5.29177210903e-5; // 1e-4 bohr.

qcx::Result<qcx::molecule::Molecule> MakeCoalescedH2() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 0.0;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = kCoalescedSeparation;
    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

qcx::Result<qcx::basisset::BasisSet> MakeAugCcPvtzH() {
    auto parsed =
        qcx::basisset::ParseNwchemFile(std::string(QcxBasisDataDir) + "/aug-cc-pvtz/H.nwchem");

    if (!parsed.has_value())
    {
        return std::unexpected(parsed.error());
    }

    return *parsed;
}

// H_core = T + V, held FIXED as the Fock: the loop converges on the one-shot
// state and every remaining stage - the seam, the convergence gate, the
// removal, the labeling - still runs for real.
qcx::Result<Eigen::MatrixXd> MakeCoreHamiltonian(const qcx::molecule::Molecule& molecule,
                                                 const qcx::basisset::BasisSet& basisSet) {
    using qcx::testing::ToMatrix;

    const auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    return ToMatrix(*kinetic) + ToMatrix(*nuclear);
}

qcx::Result<Eigen::MatrixXd> MakeOverlap(const qcx::molecule::Molecule& molecule,
                                         const qcx::basisset::BasisSet& basisSet) {
    const auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    return qcx::testing::ToMatrix(*overlap);
}

// ---------------------------------------------------------------------------
// 1. The design's own sentence, on the seam: a rectangular X has no square
//    per-irrep block, so the blocked solver takes the plain solve.
// ---------------------------------------------------------------------------

TEST(SymmetryDisengageTest, BlockedSolveFallsBackOnARectangularX) {
    const auto molecule = MakeCoalescedH2();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAugCcPvtzH();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const auto overlap = MakeOverlap(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    const auto orthogonalized = OrthogonalizeOverlap(*overlap);
    ASSERT_TRUE(orthogonalized.has_value()) << orthogonalized.error().message;

    // The fixture really is the ill-conditioned case - without this the test
    // would silently stop covering the branch it exists for.
    ASSERT_GT(orthogonalized->numRemoved, 0u);

    const Eigen::MatrixXd x = orthogonalized->x;
    EXPECT_NE(x.rows(), x.cols());

    // A non-trivial block decomposition with a rectangular X: the data must
    // come back trivial so the solver below takes the plain path.
    SymmetryBlocks blocks;
    blocks.isTrivial = false;
    blocks.u = Eigen::MatrixXd::Identity(x.rows(), x.rows());
    blocks.blockSizes = {static_cast<Eigen::Index>(x.rows())};

    const auto data = BuildBlockedDiagonalizeData(x, blocks);
    ASSERT_TRUE(data.has_value()) << data.error().message;
    EXPECT_TRUE(data->isTrivial) << "a rectangular X must mark the blocked data trivial";

    const auto core = MakeCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const auto blocked = DiagonalizeFockBlocked(*core, x, blocks, *data);
    ASSERT_TRUE(blocked.has_value()) << blocked.error().message;

    // EXACT, not approximate: the fallback IS the plain solve.
    EXPECT_EQ((*blocked - DiagonalizeFock(*core, x)).norm(), 0.0);
}

// ---------------------------------------------------------------------------
// 2. The run: a non-C1 deck that trips the floor completes, and the symmetry
//    stages stand down by name.
// ---------------------------------------------------------------------------

TEST(SymmetryDisengageTest, RhfRemovalDisengagesTheLabelingStageInsteadOfRefusing) {
    const auto molecule = MakeCoalescedH2();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAugCcPvtzH();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    // The guard this test exists for is only reached on a non-C1 group.
    EXPECT_NE(DetectPointGroup(*molecule).group, PointGroupName::kC1);

    const auto overlap = MakeOverlap(*molecule, *basis);
    const auto core = MakeCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const auto& fixedFock = *core;
    const qcx::scf::FockBuilderFn builder =
        [&fixedFock](const Eigen::MatrixXd&) -> qcx::Result<Eigen::MatrixXd> { return fixedFock; };

    qcx::scf::RhfOptions onOptions; // fullGroupLabeling defaults to true.
    qcx::scf::RhfOptions offOptions;
    offOptions.fullGroupLabeling = false;

    const auto runOn = qcx::scf::RunRhfScf(*molecule, *overlap, *core, onOptions, builder, &*basis);
    const auto runOff =
        qcx::scf::RunRhfScf(*molecule, *overlap, *core, offOptions, builder, &*basis);

    // THE ACCEPTANCE: the labeling-on run completes rather than refusing the
    // run for non-square coefficients (the fallback is the design).
    ASSERT_TRUE(runOn.has_value()) << runOn.error().message;
    ASSERT_TRUE(runOff.has_value()) << runOff.error().message;

    // The removal is what both runs are about, and it is disclosed.
    EXPECT_GT(runOn->numRemovedOverlapDirections, 0u);
    EXPECT_EQ(runOn->numRemovedOverlapDirections, runOff->numRemovedOverlapDirections);

    // The full-space answer, unchanged by the disengagement: the two runs
    // differ only in the stage that now stands down on both.
    EXPECT_EQ(runOn->converged, runOff->converged);
    EXPECT_EQ(runOn->iterations, runOff->iterations);
    EXPECT_EQ(runOn->totalEnergy, runOff->totalEnergy);
    EXPECT_EQ(runOn->electronicEnergy, runOff->electronicEnergy);
    EXPECT_EQ((runOn->density - runOff->density).norm(), 0.0);
    EXPECT_EQ((runOn->coefficients - runOff->coefficients).norm(), 0.0);
    EXPECT_EQ((runOn->orbitalEnergies - runOff->orbitalEnergies).norm(), 0.0);
    EXPECT_EQ((runOn->restart.density - runOff->restart.density).norm(), 0.0);

    // ...and it IS the full-space answer: the back-transformed density is
    // n x n and carries the electron count (Tr[D S] = N = 2), which a
    // smaller-system answer could not.
    EXPECT_EQ(runOn->density.rows(), overlap->rows());
    EXPECT_EQ(runOn->density.cols(), overlap->rows());
    EXPECT_NEAR((runOn->density * (*overlap)).trace(), 2.0, 1e-10);

    // The stage stood down on both: absent, never a fabricated C1 record.
    EXPECT_FALSE(runOn->symmetryLabels.has_value());
    EXPECT_FALSE(runOff->symmetryLabels.has_value());
}

TEST(SymmetryDisengageTest, UhfRemovalDisengagesTheLabelingStageAndSizesTheOccupations) {
    const auto molecule = MakeCoalescedH2();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAugCcPvtzH();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    EXPECT_NE(DetectPointGroup(*molecule).group, PointGroupName::kC1);

    const auto overlap = MakeOverlap(*molecule, *basis);
    const auto core = MakeCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(core.has_value()) << core.error().message;

    // The occupation vector is sized n = 46 by the caller (the AO function
    // count, all the caller can know); the removal leaves the loop with 23
    // kept directions, so the loop must truncate it - before this, the run
    // died on the dimension mismatch between the mask and the coefficients.
    EXPECT_EQ(overlap->rows(), 46);

    const auto& fixedFock = *core;
    const qcx::scf::UhfFockBuilderFn builder =
        [&fixedFock](
            const Eigen::MatrixXd&,
            const Eigen::MatrixXd&) -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{fixedFock, fixedFock};
    };

    qcx::scf::UhfOptions onOptions; // fullGroupLabeling defaults to true.
    qcx::scf::UhfOptions offOptions;
    offOptions.fullGroupLabeling = false;

    const auto runOn = qcx::scf::RunUhfScf(*molecule, *overlap, *core, onOptions, builder, &*basis);
    const auto runOff =
        qcx::scf::RunUhfScf(*molecule, *overlap, *core, offOptions, builder, &*basis);

    ASSERT_TRUE(runOn.has_value()) << runOn.error().message;
    ASSERT_TRUE(runOff.has_value()) << runOff.error().message;

    EXPECT_GT(runOn->numRemovedOverlapDirections, 0u);
    EXPECT_EQ(runOn->numRemovedOverlapDirections, runOff->numRemovedOverlapDirections);

    EXPECT_EQ(runOn->converged, runOff->converged);
    EXPECT_EQ(runOn->totalEnergy, runOff->totalEnergy);
    EXPECT_EQ((runOn->densityAlpha - runOff->densityAlpha).norm(), 0.0);
    EXPECT_EQ((runOn->densityBeta - runOff->densityBeta).norm(), 0.0);
    EXPECT_EQ((runOn->coefficientsAlpha - runOff->coefficientsAlpha).norm(), 0.0);
    EXPECT_EQ((runOn->coefficientsBeta - runOff->coefficientsBeta).norm(), 0.0);

    EXPECT_EQ(runOn->densityAlpha.rows(), overlap->rows());
    EXPECT_NEAR((runOn->densityAlpha * (*overlap)).trace(), 1.0, 1e-10);
    EXPECT_NEAR((runOn->densityBeta * (*overlap)).trace(), 1.0, 1e-10);

    EXPECT_FALSE(runOn->symmetryLabelsAlpha.has_value());
    EXPECT_FALSE(runOn->symmetryLabelsBeta.has_value());
    EXPECT_FALSE(runOff->symmetryLabelsAlpha.has_value());
}
} // namespace
