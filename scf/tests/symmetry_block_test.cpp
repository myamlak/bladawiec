// Point-group symmetry-blocked diagonalization (first
// pass, diagonalization-only scope). For molecules with a non-trivial
// Abelian computational point group the tests pin: the orthonormality of the
// symmetrizing unitary, the off-block-near-zero invariant, and the
// PROVABLE equivalence of the blocked path with the plain n x n path -
// identical sorted spectra, eigenvector residuals, and eigenvector spaces
// that coincide up to rotations within degenerate eigenvalue blocks. The C1
// case must delegate to the plain path exactly.

#include "corpus.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/symmetry/detection.hpp"
#include "qcx/symmetry/point_group.hpp"
#include "scf_common.hpp"
#include "symmetry_blocks.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

using qcx::testing::ToMatrix;

// STO-3G merged from the vendored corpus files (identical literals to the
// fixtures, but single-source for C, O, H, F, Cl alike).
qcx::Result<qcx::basisset::BasisSet> MakeSto3g(const std::vector<std::string_view>& symbols) {
    qcx::basisset::BasisSet merged;

    for (const std::string_view symbol : symbols)
    {
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

struct BlockedPathResult {
    qcx::scf::internal::SymmetryBlocks blocks;
    Eigen::VectorXd eigenvaluesPlain;
    Eigen::VectorXd eigenvaluesBlocked;
    Eigen::MatrixXd coefficientsPlain;
    Eigen::MatrixXd coefficientsBlocked;
    double offBlockNormFock = 0.0;
    double residualPlain = 0.0;
    double residualBlocked = 0.0;
    double spaceOverlapDeviation = 0.0;
};

// Runs both diagonalization paths on H_core = T + V (a symmetric matrix
// commuting with the group's action) and returns everything the equivalence
// asserts need.
qcx::Result<BlockedPathResult> RunBlockedPath(const qcx::molecule::Molecule& molecule,
                                              const qcx::basisset::BasisSet& basis) {
    const auto analysis = qcx::symmetry::DetectPointGroup(molecule);
    const auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basis);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    const Eigen::MatrixXd overlapMatrix = ToMatrix(*overlap);
    const Eigen::MatrixXd fock = ToMatrix(*kinetic) + ToMatrix(*nuclear);
    auto orthogonalized = qcx::scf::internal::OrthogonalizeOverlap(overlapMatrix);

    if (!orthogonalized.has_value())
    {
        return std::unexpected(orthogonalized.error());
    }

    const Eigen::MatrixXd x = *orthogonalized;

    const auto blocks = qcx::scf::internal::BuildSymmetryBlocks(molecule, basis, analysis);

    if (!blocks.has_value())
    {
        return std::unexpected(blocks.error());
    }

    const auto coefficientsBlocked =
        qcx::scf::internal::DiagonalizeFockBlocked(fock, x, molecule, basis, analysis);

    if (!coefficientsBlocked.has_value())
    {
        return std::unexpected(coefficientsBlocked.error());
    }

    const Eigen::MatrixXd coefficientsPlain = qcx::scf::internal::DiagonalizeFock(fock, x);

    // Eigenvalues of both paths: C = X V with V orthonormal makes
    // C^T F C = diag(lambda).
    const Eigen::VectorXd eigenvaluesPlain =
        (coefficientsPlain.transpose() * fock * coefficientsPlain).diagonal();
    const Eigen::VectorXd eigenvaluesBlocked =
        (coefficientsBlocked->transpose() * fock * *coefficientsBlocked).diagonal();

    // Eigenvector residuals in the orthogonal basis: V = X^{-1} C with
    // X = U s^{-1/2} U^T (U and s from the overlap's own eigensolve).
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> overlapSolver(overlapMatrix);
    const Eigen::MatrixXd xInverse = overlapSolver.eigenvectors() *
                                     overlapSolver.eigenvalues().cwiseSqrt().asDiagonal() *
                                     overlapSolver.eigenvectors().transpose();
    const Eigen::MatrixXd fOrth = x.transpose() * fock * x;
    const Eigen::MatrixXd vPlain = xInverse * coefficientsPlain;
    const Eigen::MatrixXd vBlocked = xInverse * *coefficientsBlocked;
    const double residualPlain =
        (fOrth * vPlain - vPlain * eigenvaluesPlain.asDiagonal()).cwiseAbs().maxCoeff();
    const double residualBlocked =
        (fOrth * vBlocked - vBlocked * eigenvaluesBlocked.asDiagonal()).cwiseAbs().maxCoeff();

    // Both paths span the same eigenspaces. Degenerate eigenvalues (benzene's
    // E irreps, CO2's Pi pairs) make |V_plain^T V_blocked| an arbitrary
    // rotation WITHIN each degenerate block rather than a signed permutation,
    // so the invariant is: every column of V_blocked has its full squared
    // norm concentrated on the V_plain columns with the same eigenvalue
    // (equal spectra then force the per-eigenvalue subspaces to coincide).
    const Eigen::MatrixXd spaceOverlap = (vPlain.transpose() * vBlocked).cwiseAbs();
    constexpr double kDegeneracyTolerance = 1e-8;
    double spaceOverlapDeviation = 0.0;

    for (Eigen::Index col = 0; col < spaceOverlap.cols(); ++col)
    {
        double sameEigenvalueNormSq = 0.0;

        for (Eigen::Index row = 0; row < spaceOverlap.rows(); ++row)
        {
            const double entry = spaceOverlap(row, col);

            if (std::abs(eigenvaluesPlain(row) - eigenvaluesBlocked(col)) < kDegeneracyTolerance)
            {
                sameEigenvalueNormSq += entry * entry;
            }
        }

        spaceOverlapDeviation =
            std::max(spaceOverlapDeviation, std::abs(1.0 - sameEigenvalueNormSq));
    }

    return BlockedPathResult{*blocks,
                             eigenvaluesPlain,
                             eigenvaluesBlocked,
                             coefficientsPlain,
                             *coefficientsBlocked,
                             qcx::scf::internal::OffBlockNorm(fock, *blocks),
                             residualPlain,
                             residualBlocked,
                             spaceOverlapDeviation};
}

// The equivalence asserts shared by every symmetric case.
void CheckBlockedPath(const std::string& name,
                      const qcx::Result<qcx::molecule::Molecule>& moleculeResult,
                      const std::vector<std::string_view>& basisSymbols,
                      qcx::symmetry::PointGroup expectedComputational) {
    ASSERT_TRUE(moleculeResult.has_value()) << name;
    const auto basis = MakeSto3g(basisSymbols);
    ASSERT_TRUE(basis.has_value()) << name;
    const auto analysis = qcx::symmetry::DetectPointGroup(*moleculeResult);
    ASSERT_EQ(analysis.computational, expectedComputational) << name;
    const auto result = RunBlockedPath(*moleculeResult, *basis);
    ASSERT_TRUE(result.has_value()) << name << ": " << result.error().message;

    const Eigen::Index n = result->blocks.u.rows();

    // The symmetrizing unitary is orthonormal.
    const Eigen::MatrixXd uTu = result->blocks.u.transpose() * result->blocks.u;
    EXPECT_LT((uTu - Eigen::MatrixXd::Identity(n, n)).cwiseAbs().maxCoeff(), 1e-12) << name;

    // The block sizes partition the basis and the orbit partition covers it
    // disjointly. Zero-size blocks are legitimate: an irrep with no AO
    // functions (e.g. water's A2) has a one-dimensional projector range of
    // dimension zero.
    Eigen::Index sizeSum = 0;

    for (const Eigen::Index size : result->blocks.blockSizes)
    {
        EXPECT_GE(size, 0) << name;
        sizeSum += size;
    }

    EXPECT_EQ(sizeSum, n) << name;
    std::vector<std::size_t> orbitFlat;

    for (const auto& orbit : result->blocks.aoOrbits)
    {
        orbitFlat.insert(orbitFlat.end(), orbit.begin(), orbit.end());
    }

    std::sort(orbitFlat.begin(), orbitFlat.end());

    for (std::size_t i = 0; i < orbitFlat.size(); ++i)
    {
        EXPECT_EQ(orbitFlat[i], i) << name;
    }

    // U^T F U is block-diagonal up to noise.
    EXPECT_LT(result->offBlockNormFock, 1e-8) << name;

    // Blocked == plain: identical sorted spectra.
    // Not a convergence gate: two eigensolves of the SAME matrix (the
    // blocked and the plain path) must return the same spectrum to
    // roundoff, which is a linear-algebra property of the pair.
    const double eigenvalueDeviation =
        (result->eigenvaluesPlain - result->eigenvaluesBlocked).cwiseAbs().maxCoeff();
    EXPECT_LT(eigenvalueDeviation, 1e-10) << name;

    // Both paths are genuine eigensolves.
    EXPECT_LT(result->residualPlain, 1e-8) << name;
    EXPECT_LT(result->residualBlocked, 1e-8) << name;

    // And their eigenspaces coincide (signed-permutation overlap).
    EXPECT_LT(result->spaceOverlapDeviation, 1e-6) << name;
}

} // namespace

TEST(SymmetryBlockTest, WaterC2vBlockedEqualsPlain) {
    CheckBlockedPath(
        "water", qcx::symmetry::testing::MakeWater(), {"O", "H"}, qcx::symmetry::PointGroup::kC2v);
}

TEST(SymmetryBlockTest, BenzeneD2hBlockedEqualsPlain) {
    // D6h detected, reduced to the computational D2h.
    CheckBlockedPath("benzene",
                     qcx::symmetry::testing::MakeBenzene(),
                     {"C", "H"},
                     qcx::symmetry::PointGroup::kD2h);
}

TEST(SymmetryBlockTest, CarbonDioxideD2hBlockedEqualsPlain) {
    // Linear branch: D infinity h detected, reduced to D2h.
    CheckBlockedPath("carbon dioxide",
                     qcx::symmetry::testing::MakeCarbonDioxide(),
                     {"C", "O"},
                     qcx::symmetry::PointGroup::kD2h);
}

TEST(SymmetryBlockTest, MethaneD2hBlockedEqualsPlain) {
    // Td detected, nominally reduced to D2h - but Td contains no inversion,
    // so the realizable blocking group is the D2 of the three perpendicular
    // C2 axes (a real subgroup of Td, equally valid: F commutes with the
    // whole group). The axes sit on diagonal directions, exercising the
    // general-axis angular action.
    CheckBlockedPath("methane",
                     qcx::symmetry::testing::MakeMethane(),
                     {"C", "H"},
                     qcx::symmetry::PointGroup::kD2h);
}

TEST(SymmetryBlockTest, OrthogonalizedXIsBlockDiagonalInSymmetryBasis) {
    // The per-irrep transform design assumes X = S^{-1/2}
    // commutes with the group action - S is group-invariant, so U^T X U is
    // block-diagonal up to noise and the per-irrep sandwich
    // g_b = X_bb^T (U_b^T F U_b) X_bb replaces the full n x n transforms.
    // Pin the foundation on both fixture families.
    const auto check = [](const qcx::Result<qcx::molecule::Molecule>& moleculeResult,
                          const std::vector<std::string_view>& basisSymbols,
                          qcx::symmetry::PointGroup expectedComputational) {
        ASSERT_TRUE(moleculeResult.has_value());
        const auto basis = MakeSto3g(basisSymbols);
        ASSERT_TRUE(basis.has_value());
        const auto analysis = qcx::symmetry::DetectPointGroup(*moleculeResult);
        ASSERT_EQ(analysis.computational, expectedComputational);
        const auto overlap = qcx::integrals::BuildOverlapMatrix(*moleculeResult, *basis);
        ASSERT_TRUE(overlap.has_value());
        auto orthogonalized = qcx::scf::internal::OrthogonalizeOverlap(ToMatrix(*overlap));
        ASSERT_TRUE(orthogonalized.has_value()) << orthogonalized.error().message;
        const auto blocks =
            qcx::scf::internal::BuildSymmetryBlocks(*moleculeResult, *basis, analysis);
        ASSERT_TRUE(blocks.has_value());
        EXPECT_LT(qcx::scf::internal::OffBlockNorm(*orthogonalized, *blocks), 1e-8);
    };

    check(qcx::symmetry::testing::MakeWater(), {"O", "H"}, qcx::symmetry::PointGroup::kC2v);
    check(qcx::symmetry::testing::MakeBenzene(), {"C", "H"}, qcx::symmetry::PointGroup::kD2h);
}

TEST(SymmetryBlockTest, BlockedDiagonalizeIsBitIdenticalAcrossTeamSizes) {
    // The threaded eigensolve exception to the BLAS-sequential
    // rule: the per-irrep transforms and solves run on the OpenMP team,
    // and every block's work is a pure function of its inputs (disjoint
    // outputs, no reductions), so the result is bit-identical across team
    // sizes. Pin it via the thread-ceiling API, which clamps every
    // parallel primitive to at most the given team size.
    const auto check = [](const qcx::Result<qcx::molecule::Molecule>& moleculeResult,
                          const std::vector<std::string_view>& basisSymbols,
                          qcx::symmetry::PointGroup expectedComputational) {
        ASSERT_TRUE(moleculeResult.has_value());
        const auto basis = MakeSto3g(basisSymbols);
        ASSERT_TRUE(basis.has_value());
        const auto analysis = qcx::symmetry::DetectPointGroup(*moleculeResult);
        ASSERT_EQ(analysis.computational, expectedComputational);
        const auto overlap = qcx::integrals::BuildOverlapMatrix(*moleculeResult, *basis);
        const auto kinetic = qcx::integrals::BuildKineticMatrix(*moleculeResult, *basis);
        const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*moleculeResult, *basis);
        ASSERT_TRUE(overlap.has_value());
        ASSERT_TRUE(kinetic.has_value());
        ASSERT_TRUE(nuclear.has_value());
        const Eigen::MatrixXd fock = ToMatrix(*kinetic) + ToMatrix(*nuclear);
        auto orthogonalized = qcx::scf::internal::OrthogonalizeOverlap(ToMatrix(*overlap));
        ASSERT_TRUE(orthogonalized.has_value()) << orthogonalized.error().message;
        const auto blocks =
            qcx::scf::internal::BuildSymmetryBlocks(*moleculeResult, *basis, analysis);
        ASSERT_TRUE(blocks.has_value());
        const auto data = qcx::scf::internal::BuildBlockedDiagonalizeData(*orthogonalized, *blocks);
        ASSERT_TRUE(data.has_value());

        qcx::backend::SetOmpThreadCeiling(1);
        const auto serialResult =
            qcx::scf::internal::DiagonalizeFockBlocked(fock, *orthogonalized, *blocks, *data);
        qcx::backend::SetOmpThreadCeiling(4);
        const auto threadedResult =
            qcx::scf::internal::DiagonalizeFockBlocked(fock, *orthogonalized, *blocks, *data);
        // Restore the process-wide ceiling before any ASSERT: the ceiling
        // is sticky and would otherwise clamp the rest of the suite.
        qcx::backend::SetOmpThreadCeiling(0);

        ASSERT_TRUE(serialResult.has_value());
        ASSERT_TRUE(threadedResult.has_value());
        EXPECT_DOUBLE_EQ((*serialResult - *threadedResult).cwiseAbs().maxCoeff(), 0.0);
    };

    check(qcx::symmetry::testing::MakeWater(), {"O", "H"}, qcx::symmetry::PointGroup::kC2v);
    check(qcx::symmetry::testing::MakeBenzene(), {"C", "H"}, qcx::symmetry::PointGroup::kD2h);
}

TEST(SymmetryBlockTest, C1FallsBackToPlainPath) {
    const auto molecule = qcx::symmetry::testing::MakeGenericC1();
    ASSERT_TRUE(molecule.has_value());
    const auto basis = MakeSto3g({"C", "H", "F", "Cl"});
    ASSERT_TRUE(basis.has_value());
    const auto analysis = qcx::symmetry::DetectPointGroup(*molecule);
    ASSERT_EQ(analysis.computational, qcx::symmetry::PointGroup::kC1);

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value());
    ASSERT_TRUE(kinetic.has_value());
    ASSERT_TRUE(nuclear.has_value());
    const Eigen::MatrixXd fock = ToMatrix(*kinetic) + ToMatrix(*nuclear);
    auto orthogonalized = qcx::scf::internal::OrthogonalizeOverlap(ToMatrix(*overlap));
    ASSERT_TRUE(orthogonalized.has_value()) << orthogonalized.error().message;
    const Eigen::MatrixXd& x = *orthogonalized;

    // The decomposition is trivial (single block, identity unitary).
    const auto blocks = qcx::scf::internal::BuildSymmetryBlocks(*molecule, *basis, analysis);
    ASSERT_TRUE(blocks.has_value());
    EXPECT_TRUE(blocks->isTrivial);
    ASSERT_EQ(blocks->blockSizes.size(), 1u);
    EXPECT_EQ(blocks->blockSizes[0], static_cast<Eigen::Index>(fock.rows()));
    EXPECT_LT(
        (blocks->u - Eigen::MatrixXd::Identity(fock.rows(), fock.rows())).cwiseAbs().maxCoeff(),
        1e-15);

    // The blocked path delegates and equals the plain path exactly.
    const auto blocked =
        qcx::scf::internal::DiagonalizeFockBlocked(fock, x, *molecule, *basis, analysis);
    ASSERT_TRUE(blocked.has_value());
    const Eigen::MatrixXd plain = qcx::scf::internal::DiagonalizeFock(fock, x);
    EXPECT_DOUBLE_EQ((*blocked - plain).cwiseAbs().maxCoeff(), 0.0);
}
