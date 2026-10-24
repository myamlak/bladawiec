// The gradient reference's acceptance: a converged H2/STO-3G run extracts
// into the validated state the total-gradient consumer starts from, and the
// extraction refuses every HfResult a gradient cannot use - a non-converged
// result, shapes that do not describe one state, and a density that is not
// 2 C_occ C_occ^T for the stated occupied count.
//
// The refusal fixtures after the first test are synthetic HfResult values:
// the point is the gate's own arithmetic on malformed input, which a real SCF
// run cannot cheaply produce. The first test carries the seam claim - the
// extractor reads a real run's fields and changes none of them.

#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/scf/gradient_reference.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <string>

namespace {

qcx::Result<qcx::scf::HfResult> RunH2Scf() {
    auto basis = qcx::testing::MakeSto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto molecule = qcx::testing::MakeH2Sto3g();

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto overlapTensor = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);

    if (!overlapTensor.has_value())
    {
        return std::unexpected(overlapTensor.error());
    }

    auto kineticTensor = qcx::integrals::BuildKineticMatrix(*molecule, *basis);

    if (!kineticTensor.has_value())
    {
        return std::unexpected(kineticTensor.error());
    }

    auto nuclearTensor = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);

    if (!nuclearTensor.has_value())
    {
        return std::unexpected(nuclearTensor.error());
    }

    auto eri = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    const Eigen::MatrixXd overlap = qcx::testing::ToMatrix(*overlapTensor);
    const Eigen::MatrixXd coreHamiltonian =
        qcx::testing::ToMatrix(*kineticTensor) + qcx::testing::ToMatrix(*nuclearTensor);
    return qcx::scf::RunRhfScf(*molecule, overlap, coreHamiltonian, *eri);
}

// A state that passes the extraction's shape checks by construction: identity
// coefficients, `occupied` columns' worth of density, and the arrays one
// result carries. Each refusal test mutates exactly the field it refuses.
qcx::scf::HfResult SyntheticResult(Eigen::Index size, Eigen::Index occupied) {
    qcx::scf::HfResult result;
    result.converged = true;
    result.iterations = 7;
    result.totalEnergy = -1.0;
    result.electronicEnergy = -1.5;
    result.coefficients = Eigen::MatrixXd::Identity(size, size);
    result.orbitalEnergies = Eigen::VectorXd::LinSpaced(size, -1.0, 1.0);
    result.density = Eigen::MatrixXd::Zero(size, size);

    for (Eigen::Index orbital = 0; orbital < occupied; ++orbital)
    {
        result.density +=
            2.0 * result.coefficients.col(orbital) * result.coefficients.col(orbital).transpose();
    }

    return result;
}

} // namespace

TEST(GradientReferenceTest, AConvergedRunExtractsEveryFieldUnchanged) {
    auto result = RunH2Scf();
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->converged);

    auto reference = qcx::scf::ExtractGradientReference(*result, 1);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    EXPECT_EQ(reference->numBasisFunctions, 2u);
    EXPECT_EQ(reference->numOccupied, 1u);
    EXPECT_EQ(reference->numVirtual, 1u);
    EXPECT_DOUBLE_EQ(reference->totalEnergy, result->totalEnergy);
    EXPECT_DOUBLE_EQ(reference->electronicEnergy, result->electronicEnergy);
    EXPECT_EQ(reference->iterations, result->iterations);
    EXPECT_DOUBLE_EQ(reference->achievedEnergyDelta, result->achievedEnergyDelta);
    EXPECT_DOUBLE_EQ(reference->achievedRmsDensityChange, result->achievedRmsDensityChange);
    EXPECT_TRUE(reference->coefficients == result->coefficients);
    EXPECT_TRUE(reference->orbitalEnergies == result->orbitalEnergies);
    EXPECT_TRUE(reference->density == result->density);

    // The extraction's own consistency check, restated: the reference's
    // density is the occupied block's projector, not merely a matrix that
    // rode along.
    const Eigen::MatrixXd occupied = reference->coefficients.leftCols(1);
    const Eigen::MatrixXd projected = 2.0 * occupied * occupied.transpose();
    EXPECT_LT((projected - reference->density).norm(), 1e-12);

    // The extraction owns copies: a later change to the result does not move
    // the reference a consumer is differentiating.
    result->density(0, 0) += 0.5;
    EXPECT_NE(reference->density(0, 0), result->density(0, 0));
}

TEST(GradientReferenceTest, ANonConvergedResultIsRefusedByName) {
    auto result = SyntheticResult(3, 1);
    result.converged = false;
    result.iterations = 100;

    auto reference = qcx::scf::ExtractGradientReference(result, 1);
    ASSERT_FALSE(reference.has_value());
    EXPECT_EQ(reference.error().code, qcx::ErrorCode::kInvalidArgument);
    const std::string message = reference.error().message;
    EXPECT_NE(message.find("converged = false"), std::string::npos);
    EXPECT_NE(message.find("100 iterations"), std::string::npos);
}

TEST(GradientReferenceTest, AWrongOccupiedCountIsRefusedByTheDensityCheck) {
    // Three orbitals, two occupied by construction; the request says one.
    // Both counts are legal bounds, so only the state check can refuse it.
    auto result = SyntheticResult(3, 2);

    auto reference = qcx::scf::ExtractGradientReference(result, 1);
    ASSERT_FALSE(reference.has_value());
    EXPECT_EQ(reference.error().code, qcx::ErrorCode::kInvalidArgument);
    const std::string message = reference.error().message;
    EXPECT_NE(message.find("2 C_occ C_occ^T"), std::string::npos);
    EXPECT_NE(message.find("1 occupied"), std::string::npos);
}

TEST(GradientReferenceTest, AnOccupiedCountOutsideTheManifoldsIsRefused) {
    auto result = SyntheticResult(3, 1);

    auto none = qcx::scf::ExtractGradientReference(result, 0);
    ASSERT_FALSE(none.has_value());
    EXPECT_EQ(none.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(none.error().message.find("occupied count"), std::string::npos);

    auto all = qcx::scf::ExtractGradientReference(result, 3);
    ASSERT_FALSE(all.has_value());
    EXPECT_EQ(all.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(all.error().message.find("occupied count"), std::string::npos);
}

TEST(GradientReferenceTest, ANonSquareCoefficientMatrixIsRefused) {
    auto result = SyntheticResult(3, 1);
    result.coefficients.resize(3, 2);

    auto reference = qcx::scf::ExtractGradientReference(result, 1);
    ASSERT_FALSE(reference.has_value());
    EXPECT_EQ(reference.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(reference.error().message.find("square"), std::string::npos);
}

TEST(GradientReferenceTest, ADensityOfTheWrongShapeIsRefused) {
    auto result = SyntheticResult(3, 1);
    result.density.resize(2, 3);

    auto reference = qcx::scf::ExtractGradientReference(result, 1);
    ASSERT_FALSE(reference.has_value());
    EXPECT_EQ(reference.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(reference.error().message.find("density"), std::string::npos);
}

TEST(GradientReferenceTest, AnOrbitalEnergyVectorOfTheWrongShapeIsRefused) {
    auto result = SyntheticResult(3, 1);
    result.orbitalEnergies.resize(2);

    auto reference = qcx::scf::ExtractGradientReference(result, 1);
    ASSERT_FALSE(reference.has_value());
    EXPECT_EQ(reference.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(reference.error().message.find("orbital energies"), std::string::npos);
}
