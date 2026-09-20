// The RI 3-center tensor persistence: the n^2 x nAux
// contraction layout round-trips bit-identically through Save/Load, and
// the system fingerprint (which carries the aux basis name) rejects a
// mismatched auxiliary.
#include "fast_test_mode.hpp"
#include "h2o_ccpvdz.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/storage/ri_tensor_store.hpp"
#include "temp_store.hpp"

#include <cstring>
#include <gtest/gtest.h>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::integrals::RiEngineOptions;
using qcx::integrals::RiJkFockBuilder;
using qcx::storage::LoadRiTensor;
using qcx::storage::SaveRiTensor;
using qcx::testing::MakeH2oCcpvdz;
using qcx::testing::MakeH2oCcpvdzBasis;
using qcx::testing::MakeH2oCcpvdzRifitBasis;
using qcx::testing::ScopedTempStoreFile;

// The one-electron core Hamiltonian H = T + V (the integrals test suite's
// shared fixture pattern, ri_engine_test.cpp).
qcx::Result<CpuTensor2> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basisSet) {
    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    const std::size_t n = kinetic->Shape()[0];

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*kinetic)(i, j) += (*nuclear)(i, j);
        }
    }

    kinetic->MarkHostDirty();
    return std::move(*kinetic);
}

TEST(RiTensorTest, RoundTripIsBitIdentical) {
    if (qcx::testing::IsFastOnlyMode() || !qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "the RI build is a full integrals sweep; the rifit f shells "
                        "exceed this build's kMaxEngineL (CI lmax=2)";
    }

    const auto molecule = MakeH2oCcpvdz();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oCcpvdzBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto aux = MakeH2oCcpvdzRifitBasis();
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    const auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto ri = RiJkFockBuilder::Create(*molecule, *basis, *aux, *core);
    ASSERT_TRUE(ri.has_value()) << ri.error().message;
    const Eigen::MatrixXd& expected = ri->RiMatrix();
    ASSERT_GT(expected.rows(), 0);
    ASSERT_GT(expected.cols(), 0);

    ScopedTempStoreFile tempFile("qcx_ri_tensor");
    auto saved = SaveRiTensor(tempFile.Path(), *molecule, "cc-pvdz", "cc-pvdz-rifit", expected);
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    auto loaded = LoadRiTensor(tempFile.Path(), *molecule, "cc-pvdz", "cc-pvdz-rifit");
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
    ASSERT_EQ(loaded->rows(), expected.rows());
    ASSERT_EQ(loaded->cols(), expected.cols());
    // Element-wise bit identity (the element-wise column-major <-> row-major
    // copies are lossless).
    EXPECT_EQ(std::memcmp(loaded->data(),
                          expected.data(),
                          expected.rows() * expected.cols() * sizeof(double)),
              0);

    // Append-only: a second save refuses.
    const auto second =
        SaveRiTensor(tempFile.Path(), *molecule, "cc-pvdz", "cc-pvdz-rifit", expected);
    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(RiTensorTest, LoadRejectsWrongAux) {
    if (qcx::testing::IsFastOnlyMode() || !qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "the RI build is a full integrals sweep; the rifit f shells "
                        "exceed this build's kMaxEngineL (CI lmax=2)";
    }

    const auto molecule = MakeH2oCcpvdz();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oCcpvdzBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto aux = MakeH2oCcpvdzRifitBasis();
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    const auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto ri = RiJkFockBuilder::Create(*molecule, *basis, *aux, *core);
    ASSERT_TRUE(ri.has_value()) << ri.error().message;

    ScopedTempStoreFile tempFile("qcx_ri_wrong_aux");
    auto saved =
        SaveRiTensor(tempFile.Path(), *molecule, "cc-pvdz", "cc-pvdz-rifit", ri->RiMatrix());
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    // The aux basis name is part of the system fingerprint.
    const auto loaded = LoadRiTensor(tempFile.Path(), *molecule, "cc-pvdz", "def2-universal-jfit");
    ASSERT_FALSE(loaded.has_value());
    EXPECT_EQ(loaded.error().code, qcx::ErrorCode::kInvalidArgument);
}

} // namespace
