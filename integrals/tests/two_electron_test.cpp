// Two-electron integral tests: the committed mpmath reference grid at 1e-10,
// the 8-fold permutational symmetry of (uv|ws), and both rejection paths.
#include "h2_sto3g.hpp"
#include "h2_sto3g_fixture.hpp"
#include "qcx/integrals/two_electron.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;
using qcx::integrals::test::LoadReferenceValues;
using qcx::integrals::test::ReferenceValue;
using qcx::testing::kPOrbitalBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeHeAtom;
using qcx::testing::MakeSto3gBasis;

constexpr double kReferenceTolerance = 1e-10;

TEST(TwoElectronTest, MatchesMpmathReference) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto tensor = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(tensor.has_value()) << tensor.error().message;
    const std::vector<ReferenceValue> reference = LoadReferenceValues();
    ASSERT_GT(reference.size(), 0u);

    int checked = 0;

    for (const ReferenceValue& row : reference)
    {
        if (row.kind != "ERI")
        {
            continue;
        }

        EXPECT_NEAR((*tensor)(row.i, row.j, row.k, row.l), row.value, kReferenceTolerance)
            << "ERI(" << row.i << "," << row.j << "," << row.k << "," << row.l << ")";
        ++checked;
    }

    EXPECT_EQ(checked, 16);
}

TEST(TwoElectronTest, EightFoldPermutationalSymmetry) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto tensor = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(tensor.has_value()) << tensor.error().message;

    // (ij|kl) is invariant under i<->j, k<->l, and (ij)<->(kl). The distinct
    // classes - e.g. the exchange integral (01|01) and the Coulomb integral
    // (00|11) - differ, so only same-orbit permutations are compared.
    EXPECT_NEAR((*tensor)(0, 1, 0, 1), (*tensor)(1, 0, 0, 1), kReferenceTolerance);
    EXPECT_NEAR((*tensor)(0, 1, 0, 1), (*tensor)(0, 1, 1, 0), kReferenceTolerance);
    EXPECT_NEAR((*tensor)(0, 1, 1, 0), (*tensor)(1, 0, 1, 0), kReferenceTolerance);
    EXPECT_NEAR((*tensor)(0, 0, 0, 1), (*tensor)(0, 0, 1, 0), kReferenceTolerance);
    EXPECT_NEAR((*tensor)(0, 0, 0, 1), (*tensor)(0, 1, 0, 0), kReferenceTolerance);
    EXPECT_NEAR((*tensor)(0, 0, 1, 1), (*tensor)(1, 1, 0, 0), kReferenceTolerance);
}

TEST(TwoElectronTest, FirstQuartetPositive) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto tensor = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(tensor.has_value()) << tensor.error().message;

    EXPECT_GT((*tensor)(0, 0, 0, 0), 0.0);
}

TEST(TwoElectronTest, NonSShellIsUnimplemented) {
    auto basis = qcx::basisset::ParseNwchemText(kPOrbitalBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto tensor = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_FALSE(tensor.has_value());
    EXPECT_EQ(tensor.error().code, qcx::ErrorCode::kUnimplemented);
}

TEST(TwoElectronTest, MissingElementIsInvalidArgument) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeHeAtom();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto tensor = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_FALSE(tensor.has_value());
    EXPECT_EQ(tensor.error().code, qcx::ErrorCode::kInvalidArgument);
}

// The s-only admission gate, exact-formula pin: H2/STO-3G has 2 s
// functions, so the tensor estimate is 8 * 2^4 = 128 B - the cap admits at
// 128 and refuses one byte below with the graceful kInvalidArgument (never
// an allocation death). The default cap (2 GiB, shared with the general-l
// dense builder) admits the build.
TEST(TwoElectronTest, AdmissionGatePinsTheEstimate) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto admitted = qcx::integrals::BuildEriTensor(*molecule, *basis, 128);
    ASSERT_TRUE(admitted.has_value()) << admitted.error().message;

    auto refused = qcx::integrals::BuildEriTensor(*molecule, *basis, 127);
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(refused.error().message.find("maxTensorBytes"), std::string::npos);

    auto defaulted = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(defaulted.has_value()) << defaulted.error().message;
}

} // namespace
