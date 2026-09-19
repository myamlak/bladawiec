// The RI-J SCF pin (the RI-J SCF pin): the
// RiJkFockBuilder behind the FockBuilderFn seam (the builder itself and the
// 3c/metric engines are tested in integrals/tests/ri_engine_test.cpp; scf
// is the only place that can close the loop - the module DAG). The pinned
// energy is the dense_rhf_test.cpp record; the run uses kTight so the
// direct-exchange side reproduces the fp64 path and the deviation is the
// RI approximation alone.

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <filesystem>
#include <gtest/gtest.h>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

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

TEST(RiRhfTest, H2oRiJScfMatchesPin) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jfit g shells";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const std::filesystem::path root(QcxBasisDataDir);
    // The per-element aux parse filter: the pin must hold unchanged through
    // the molecule-scoped parse - the filter is value-neutral, so a pin move
    // here is a filter bug, not a tolerance.
    const std::array<int, 2> auxElements{1, 8};
    auto aux = qcx::basisset::ParseNwchemDirectoryFiltered((root / "def2-universal-jfit").string(),
                                                           auxElements);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::RiEngineOptions riOptions;
    riOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto builder =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, riOptions);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const auto result = qcx::scf::RunRhfScf(
        *molecule,
        ToMatrix(*overlap),
        ToMatrix(*core),
        qcx::scf::RhfOptions{},
        [builder = *builder](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
            // The seam hands over the spin-summed density D = 2 C_occ C_occ^T;
            // the builder contracts the spatial density rho = D/2 (BUG-3,
            // 2026-08-21; rhf.hpp documents the seam convention).
            auto densityTensor = ToTensor(0.5 * density);

            if (!densityTensor.has_value())
            {
                return std::unexpected(densityTensor.error());
            }

            auto fock = builder.BuildFock(*densityTensor);

            if (!fock.has_value())
            {
                return std::unexpected(fock.error());
            }

            return ToMatrix(*fock);
        });
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->converged) << "iterations: " << result->iterations;
    // Re-measured 2026-08-22 after the 3c function-scheme fix (the phantom
    // (P, s_0) aux pair: E-table fold + ket sign + the symmetric I write-
    // back); the pre-fix pin -74.96302314 was 6.7e-4 off. The RI-J
    // approximation leaves the energy 7.6e-4 above the direct pin.
    EXPECT_NEAR(result->totalEnergy, -74.96369193, 1e-5);
}

} // namespace
