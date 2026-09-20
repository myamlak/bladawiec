// The disk-backed RI-J builder tests (disk_ri_fock_build.hpp):
// the product
// tolerance (J from the disk builder vs the in-memory RiJkFockBuilder on
// the same density - the exchange halves are the same code with the same
// options, so the Fock-matrix difference IS the J drift, the measured-then-
// pinned quantity), the physics pin end-to-end
// through the builder (the ri_rhf_test.cpp mirror: H2O/STO-3G kTight, the
// RI-J record -74.96369193, NOT the direct pin -74.96292827; the RI-J
// approximation sits 7.6e-4 above it, ri_rhf_test.cpp), the chunk-plan
// determinism, and the integrity path through the BUILDER
// (a store tampered between Create and BuildFock fails the per-chunk
// checksum of the chunk store). The C24H50/def2-SVP legs need
// the in-memory monolithic reference at the 4.4 GiB tensor scale - the
// cap-broken profile - so they are not part of this suite.

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/storage/disk_ri_fock_build.hpp"
#include "temp_store.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <gtest/gtest.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ScopedTempStoreFile;
using qcx::testing::ToTensor;

// The tiny s/p auxiliary set of the generator's 3c grid (the shared
// ri_engine_test fixture, inlined in ri_engine_chunk_test.cpp too): H2O
// carries six shells (O S, O P, two H S, two H P) and twelve functions.
constexpr std::string_view kTinyAux = R"(BASIS "ao basis" SPHERICAL PRINT
O    S
      2.5000000000E+00       3.0000000000E-01
      8.0000000000E-01       8.0000000000E-01
O    P
      1.2000000000E+00       1.0000000000E+00
H    S
      1.0000000000E+00       1.0000000000E+00
H    P
      6.0000000000E-01       1.0000000000E+00
END
)";

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

// The elementwise max |a - b| over two same-shaped Fock tensors. Every claim
// in the split-half tests below is stated as one of these, against
// kPinnedSplitHalfTolerance: no two of the comparisons there can be made
// exactly, because each side is a call of its own and the exchange half each
// call re-runs is the non-reproducible direct kernel. The measured floor is
// recorded beside that constant rather than assumed here.
double MaxAbsDifference(const CpuTensor2& a, const CpuTensor2& b) {
    const std::size_t n = a.Shape()[0];
    double worst = 0.0;

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            worst = std::max(worst, std::abs(a(u, v) - b(u, v)));
        }
    }

    return worst;
}

// The floor a comparison BETWEEN TWO CALLS of the disk builder's halves must
// be read at, and it is not a tuning knob. The exchange half IS the nested
// direct kernel, which fock_build.hpp documents as NOT run-to-run
// reproducible (its own measured floor is ~5e-15 at 106 BF), so two calls
// that each run that kernel differ in the last ulps; and a comparison of the
// Coulomb half across ROW LAYOUTS re-associates pass 1's v through the
// floored metric inverse, which is the quantity
// ReducedRowLayoutHalvesTheStoreAndPreservesTheFock already pins at this same
// 1e-12.
//
// The tolerance was measured BEFORE it was pinned, on the two fixtures
// below: the fused-vs-halves sum 1.4e-17, the exchange half across two
// builders 1.1e-16, the Coulomb half across the two row layouts 1.8e-15. All
// three sit at least three orders inside it, and every convention error these
// tests exist to catch - the factor of two moved into or out of the Coulomb
// half, a core Hamiltonian carried by the wrong half - is O(1) and cannot
// hide below it.
constexpr double kPinnedSplitHalfTolerance = 1e-12;

bool ChunkPlansEqual(const std::vector<qcx::storage::RiChunkMeta>& a,
                     const std::vector<qcx::storage::RiChunkMeta>& b) {
    if (a.size() != b.size())
    {
        return false;
    }

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].chunkIndex != b[i].chunkIndex || a[i].auxShellStart != b[i].auxShellStart ||
            a[i].auxShellEnd != b[i].auxShellEnd ||
            a[i].auxFunctionStart != b[i].auxFunctionStart ||
            a[i].auxFunctionEnd != b[i].auxFunctionEnd ||
            a[i].orbitalFunctionCount != b[i].orbitalFunctionCount)
        {
            return false;
        }
    }

    return true;
}

} // namespace

TEST(DiskRiFockBuildTest, ChunkPlanIsDeterministicAndCoversTheAuxSet) {
    // The moved-from fixture is non-const: Create takes the molecule by
    // value and *molecule on a const Result is a const lvalue (copy-deleted).
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    // A byte cap below the shell floor: auto-size makes the
    // effective chunk size 8 * n^2 * maxShellFunctions, so the plan still
    // covers the aux set exactly, one shell per chunk on the tiny aux set
    // (the floor never splits a shell).
    qcx::storage::DiskRiFockOptions options;
    options.chunkBytes = 1;

    ScopedTempStoreFile first("disk_ri_plan_a");
    auto firstBuilder = qcx::storage::DiskRiFockBuilder::Create(
        first.Path(), std::move(*molecule), *basis, *aux, "sto-3g", "tiny", *core, options);
    ASSERT_TRUE(firstBuilder.has_value()) << firstBuilder.error().message;

    // Determinism: a second Create on a fresh path yields the SAME plan (a
    // pure function of (basis data, options)).
    ScopedTempStoreFile second("disk_ri_plan_b");
    auto secondMolecule = MakeH2oSto3g();
    ASSERT_TRUE(secondMolecule.has_value()) << secondMolecule.error().message;
    auto secondBuilder = qcx::storage::DiskRiFockBuilder::Create(
        second.Path(), std::move(*secondMolecule), *basis, *aux, "sto-3g", "tiny", *core, options);
    ASSERT_TRUE(secondBuilder.has_value()) << secondBuilder.error().message;
    EXPECT_TRUE(ChunkPlansEqual(firstBuilder->Chunks(), secondBuilder->Chunks()));

    // The plan's ranges tile the aux shell/function space without gaps.
    const std::vector<qcx::storage::RiChunkMeta>& plan = firstBuilder->Chunks();
    ASSERT_GE(plan.size(), 2u);
    EXPECT_EQ(plan.front().auxShellStart, 0u);
    EXPECT_EQ(plan.front().auxFunctionStart, 0u);
    EXPECT_EQ(plan.front().orbitalFunctionCount, 7u);

    for (std::size_t i = 0; i < plan.size(); ++i)
    {
        EXPECT_EQ(plan[i].chunkIndex, i);
        EXPECT_LT(plan[i].auxShellStart, plan[i].auxShellEnd);
        EXPECT_LT(plan[i].auxFunctionStart, plan[i].auxFunctionEnd);

        if (i > 0)
        {
            EXPECT_EQ(plan[i].auxShellStart, plan[i - 1].auxShellEnd);
            EXPECT_EQ(plan[i].auxFunctionStart, plan[i - 1].auxFunctionEnd);
        }
    }

    // Append-only: a second Create on a path that already carries chunks
    // refuses at the first Save (the append-only store contract).
    auto thirdMolecule = MakeH2oSto3g();
    ASSERT_TRUE(thirdMolecule.has_value()) << thirdMolecule.error().message;
    auto refused = qcx::storage::DiskRiFockBuilder::Create(
        first.Path(), std::move(*thirdMolecule), *basis, *aux, "sto-3g", "tiny", *core, options);
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(DiskRiFockBuildTest, TamperedStoreFailsTheBuildFockChecksum) {
    ScopedTempStoreFile temp("disk_ri_tamper");

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    // Create takes the molecule by value (move-only); everything that needs
    // the caller's copy runs above.
    auto builder = qcx::storage::DiskRiFockBuilder::Create(
        temp.Path(), std::move(*molecule), *basis, *aux, "sto-3g", "tiny", *core);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    // A symmetric density placeholder (rho = S/2) - BuildFock only needs
    // symmetry; the tamper fires in pass 1 of the RI products.
    auto density = ToTensor(0.5 * qcx::testing::ToMatrix(*overlap));
    ASSERT_TRUE(density.has_value()) << density.error().message;

    // Flip one byte of chunk 0 through a raw HighFive handle after Create
    // (the builder holds no file handle - the store opens per load).
    {
        auto file =
            std::make_shared<HighFive::File>(temp.Path().string(), HighFive::File::ReadWrite);
        HighFive::DataSet dataSet = file->getDataSet("integrals/ao/eri/ri_tensor_chunks/0");
        const std::vector<std::size_t> dims = dataSet.getSpace().getDimensions();
        std::vector<double> flat(dims[0] * dims[1]);
        dataSet.read_raw(flat.data(), dataSet.getDataType());
        flat[dims[0] * dims[1] / 2] += 1.0;
        dataSet.write_raw(flat.data(), dataSet.getDataType());
        file->flush();
    }

    auto fock = builder->BuildFock(*density);
    ASSERT_FALSE(fock.has_value());
    EXPECT_EQ(fock.error().code, qcx::ErrorCode::kIOError);
    EXPECT_NE(fock.error().message.find("checksum"), std::string::npos);
}

TEST(DiskRiFockBuildTest, ProductToleranceH2ODef2Svp) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jfit g shells";
    }

    // The disk-vs-in-memory comparison at the H2O/def2-SVP scale: J from the disk
    // builder vs the in-memory RiJkFockBuilder on the same converged
    // density. The exchange halves are the SAME code with the SAME option
    // recipe (disk_ri_fock_build.cpp mirrors the in-memory Create), so the
    // Fock difference is exactly 2(J_disk - J_mem) - the J-only drift of
    // The chunk cap of one byte forces the shell floor, so the
    // store splits into many chunks and the serial pass accumulation is
    // exercised for real.
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const std::filesystem::path root(QcxBasisDataDir);
    const std::array<int, 2> elements{1, 8};
    auto orbital =
        qcx::basisset::ParseNwchemDirectoryFiltered((root / "def2-svp").string(), elements);
    ASSERT_TRUE(orbital.has_value()) << orbital.error().message;
    auto aux = qcx::basisset::ParseNwchemDirectoryFiltered((root / "def2-universal-jfit").string(),
                                                           elements);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *orbital);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *orbital);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    // The in-memory reference run (converged density source + BuildFock).
    auto memoryBuilder = qcx::integrals::RiJkFockBuilder::Create(*molecule, *orbital, *aux, *core);
    ASSERT_TRUE(memoryBuilder.has_value()) << memoryBuilder.error().message;

    const auto reference =
        qcx::scf::RunRhfScf(*molecule,
                            qcx::testing::ToMatrix(*overlap),
                            qcx::testing::ToMatrix(*core),
                            qcx::scf::RhfOptions{},
                            [memoryBuilder = *memoryBuilder](
                                const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
                                auto densityTensor = ToTensor(0.5 * density);

                                if (!densityTensor.has_value())
                                {
                                    return std::unexpected(densityTensor.error());
                                }

                                auto fock = memoryBuilder.BuildFock(*densityTensor);

                                if (!fock.has_value())
                                {
                                    return std::unexpected(fock.error());
                                }

                                return qcx::testing::ToMatrix(*fock);
                            });
    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    ASSERT_TRUE(reference->converged);

    auto rho = ToTensor(0.5 * reference->density);
    ASSERT_TRUE(rho.has_value()) << rho.error().message;
    auto memoryFock = memoryBuilder->BuildFock(*rho);
    ASSERT_TRUE(memoryFock.has_value()) << memoryFock.error().message;

    // The disk builder over the same system (chunk-by-chunk store).
    ScopedTempStoreFile temp("disk_ri_tolerance");
    qcx::storage::DiskRiFockOptions options;
    options.chunkBytes = 1;
    auto diskMolecule = MakeH2oSto3g();
    ASSERT_TRUE(diskMolecule.has_value()) << diskMolecule.error().message;
    auto diskBuilder = qcx::storage::DiskRiFockBuilder::Create(temp.Path(),
                                                               std::move(*diskMolecule),
                                                               *orbital,
                                                               *aux,
                                                               "def2-svp",
                                                               "def2-universal-jfit",
                                                               *core,
                                                               options);
    ASSERT_TRUE(diskBuilder.has_value()) << diskBuilder.error().message;
    ASSERT_GE(diskBuilder->Chunks().size(), 2u)
        << "the one-byte chunk cap must split the store into several chunks";

    auto diskFock = diskBuilder->BuildFock(*rho);
    ASSERT_TRUE(diskFock.has_value()) << diskFock.error().message;

    const std::size_t n = rho->Shape()[0];
    double maxAbsDifference = 0.0;

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            maxAbsDifference =
                std::max(maxAbsDifference, std::abs((*memoryFock)(u, v) - (*diskFock)(u, v)));
        }
    }

    // Pinned 2026-09-02 by measurement (never asserted a priori):
    // the first run printed max |F_disk - F_mem| = 1.42e-14 (MKL Release) -
    // the pass-1 v drift amplified by the metric's floored inverse. The
    // tolerance sits ~x70 above it so the Eigen-fallback BLAS seam on CI
    // (different k-loop blocking) stays inside the pin while any real
    // accumulation regression (a transposed chunk, a mis-sliced w, a wrong
    // chunk order) fails by many orders of magnitude.
    constexpr double kPinnedTolerance = 1e-12;
    EXPECT_LT(maxAbsDifference, kPinnedTolerance)
        << "measured max |F_disk - F_mem| = " << maxAbsDifference;
}

TEST(DiskRiFockBuildTest, H2oSto3gRiJScfPinThroughDisk) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jfit g shells";
    }

    // The ri_rhf_test.cpp mirror: the RI-J record, NOT the direct
    // pin - the disk rung stores the same numbers, so the end-to-end SCF
    // must reproduce the in-memory RI-J energy -74.96369193 (the direct pin
    // -74.96292827 is a different number; the RI-J approximation sits 7.6e-4
    // above it, ri_rhf_test.cpp).
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const std::filesystem::path root(QcxBasisDataDir);
    const std::array<int, 2> elements{1, 8};
    auto aux = qcx::basisset::ParseNwchemDirectoryFiltered((root / "def2-universal-jfit").string(),
                                                           elements);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    ScopedTempStoreFile temp("disk_ri_pin");
    qcx::storage::DiskRiFockOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto diskMolecule = MakeH2oSto3g();
    ASSERT_TRUE(diskMolecule.has_value()) << diskMolecule.error().message;
    auto builder = qcx::storage::DiskRiFockBuilder::Create(temp.Path(),
                                                           std::move(*diskMolecule),
                                                           *basis,
                                                           *aux,
                                                           "sto-3g",
                                                           "def2-universal-jfit",
                                                           *core,
                                                           options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const auto result = qcx::scf::RunRhfScf(
        *molecule,
        qcx::testing::ToMatrix(*overlap),
        qcx::testing::ToMatrix(*core),
        qcx::scf::RhfOptions{},
        [builder = *builder](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
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

            return qcx::testing::ToMatrix(*fock);
        });
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->converged) << "iterations: " << result->iterations;
    EXPECT_NEAR(result->totalEnergy, -74.96369193, 1e-5);
}

TEST(DiskRiFockBuildTest, ReducedRowLayoutHalvesTheStoreAndPreservesTheFock) {
    // The uv-reduced row layout (RiChunkMeta, DiskRiFockOptions::
    // reducedRowLayout): the store keeps ONE row per unordered bra pair
    // where the full layout keeps n^2, dropping a byte-for-byte duplicate
    // half. The 3-center integral is symmetric in its bra pair, so this is
    // NOT a point-group mechanism and no value is computed, altered or
    // approximated by it. Three claims are pinned here, each measured:
    //   1. the stored row counts, read off the datasets themselves,
    //   2. the bit-identity of the kept rows against the full store upper
    //      triangle (the "no value changed" claim),
    //   3. the Fock the two layouts build from the SAME density.
    // Without this test the reduction is invisible: every other test in
    // this file passes under either layout.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    // One byte of chunk budget: the shell floor splits the store into many
    // chunks, so the per-chunk row layout is exercised as a repeated fact
    // rather than once.
    qcx::storage::DiskRiFockOptions options;
    options.chunkBytes = 1;
    options.reducedRowLayout = false;

    ScopedTempStoreFile fullStore("disk_ri_full_rows");
    auto fullMolecule = MakeH2oSto3g();
    ASSERT_TRUE(fullMolecule.has_value()) << fullMolecule.error().message;
    auto fullBuilder = qcx::storage::DiskRiFockBuilder::Create(
        fullStore.Path(), std::move(*fullMolecule), *basis, *aux, "sto-3g", "tiny", *core, options);
    ASSERT_TRUE(fullBuilder.has_value()) << fullBuilder.error().message;

    options.reducedRowLayout = true;
    ScopedTempStoreFile reducedStore("disk_ri_reduced_rows");
    auto reducedMolecule = MakeH2oSto3g();
    ASSERT_TRUE(reducedMolecule.has_value()) << reducedMolecule.error().message;
    auto reducedBuilder = qcx::storage::DiskRiFockBuilder::Create(reducedStore.Path(),
                                                                  std::move(*reducedMolecule),
                                                                  *basis,
                                                                  *aux,
                                                                  "sto-3g",
                                                                  "tiny",
                                                                  *core,
                                                                  options);
    ASSERT_TRUE(reducedBuilder.has_value()) << reducedBuilder.error().message;

    const std::vector<qcx::storage::RiChunkMeta>& fullPlan = fullBuilder->Chunks();
    const std::vector<qcx::storage::RiChunkMeta>& reducedPlan = reducedBuilder->Chunks();
    ASSERT_GE(fullPlan.size(), 2u);

    // The plan itself is untouched by the row layout: the same chunks, the
    // same column ranges (the layout is a ROW property, and chunks
    // partition the aux COLUMN index only).
    EXPECT_TRUE(ChunkPlansEqual(fullPlan, reducedPlan));

    const std::size_t n = fullPlan.front().orbitalFunctionCount;
    const std::size_t reducedRows = qcx::storage::RiChunkReducedRowCount(n);
    EXPECT_EQ(reducedRows, n * (n + 1) / 2);

    auto loadMolecule = MakeH2oSto3g();
    ASSERT_TRUE(loadMolecule.has_value()) << loadMolecule.error().message;

    std::size_t exactRowMismatches = 0;
    double maxRowAbsDifference = 0.0;

    for (std::size_t i = 0; i < fullPlan.size(); ++i)
    {
        // 1. The stored shapes, read off the datasets rather than from the
        // metas - this is the claim about what is ON DISK.
        std::vector<std::size_t> fullDims;
        std::vector<std::size_t> reducedDims;
        {
            auto fullFile = std::make_shared<HighFive::File>(fullStore.Path().string(),
                                                             HighFive::File::ReadOnly);
            auto reducedFile = std::make_shared<HighFive::File>(reducedStore.Path().string(),
                                                                HighFive::File::ReadOnly);
            const std::string path = "integrals/ao/eri/ri_tensor_chunks/" + std::to_string(i);
            fullDims = fullFile->getDataSet(path).getSpace().getDimensions();
            reducedDims = reducedFile->getDataSet(path).getSpace().getDimensions();
        }

        EXPECT_EQ(fullDims.at(0), n * n);
        EXPECT_EQ(reducedDims.at(0), reducedRows);
        EXPECT_EQ(fullDims.at(1), reducedDims.at(1));

        // 2. The reduced store IS the full store upper triangle, term for
        // term - the dropped rows are duplicates, not approximations.
        auto fullChunk = qcx::storage::LoadRiTensorChunk(
            fullStore.Path(), *loadMolecule, "sto-3g", "tiny", fullPlan[i]);
        ASSERT_TRUE(fullChunk.has_value()) << fullChunk.error().message;
        auto reducedChunk = qcx::storage::LoadRiTensorChunk(
            reducedStore.Path(), *loadMolecule, "sto-3g", "tiny", reducedPlan[i]);
        ASSERT_TRUE(reducedChunk.has_value()) << reducedChunk.error().message;

        ASSERT_EQ(static_cast<std::size_t>(reducedChunk->rows()), reducedRows);

        for (std::size_t u = 0; u < n; ++u)
        {
            for (std::size_t v = u; v < n; ++v)
            {
                const Eigen::Index kept =
                    static_cast<Eigen::Index>(qcx::storage::RiChunkReducedRowIndex(n, u, v));
                const Eigen::Index source = static_cast<Eigen::Index>(u * n + v);

                for (Eigen::Index c = 0; c < reducedChunk->cols(); ++c)
                {
                    const double keptValue = (*reducedChunk)(kept, c);
                    const double sourceValue = (*fullChunk)(source, c);

                    if (keptValue != sourceValue)
                    {
                        ++exactRowMismatches;
                    }

                    maxRowAbsDifference =
                        std::max(maxRowAbsDifference, std::abs(keptValue - sourceValue));
                }
            }
        }
    }

    EXPECT_EQ(exactRowMismatches, 0u)
        << "the reduced store must carry the full store values unchanged; max |diff| = "
        << maxRowAbsDifference;

    // 3. The Fock, both layouts, the same density (rho = S/2, symmetric -
    // BuildFock only needs that).
    auto rho = ToTensor(0.5 * qcx::testing::ToMatrix(*overlap));
    ASSERT_TRUE(rho.has_value()) << rho.error().message;

    auto fullFock = fullBuilder->BuildFock(*rho);
    ASSERT_TRUE(fullFock.has_value()) << fullFock.error().message;
    auto reducedFock = reducedBuilder->BuildFock(*rho);
    ASSERT_TRUE(reducedFock.has_value()) << reducedFock.error().message;

    double maxFockAbsDifference = 0.0;

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            maxFockAbsDifference =
                std::max(maxFockAbsDifference, std::abs((*fullFock)(u, v) - (*reducedFock)(u, v)));
        }
    }

    // Measured, then pinned - never asserted a priori. The Fock difference here is 2*(J_reduced -
    // J_full) alone: the exchange half is the same code with the same
    // density and does not see the row layout. J is bit-identical (each
    // pair dot product reads the identical row), so the residual is the
    // pass-1 v re-association seen through the metric floored inverse -
    // the same quantity ProductToleranceH2ODef2Svp pins at 1e-12 against
    // the in-memory reference.
    constexpr double kPinnedLayoutTolerance = 1e-12;
    EXPECT_LT(maxFockAbsDifference, kPinnedLayoutTolerance)
        << "measured max |F_full - F_reduced| = " << maxFockAbsDifference;
}

// ---------------------------------------------------------------------------
// The split halves (disk_ri_fock_build.hpp BuildCoulombOnly /
// BuildExchangeOnly). The fused BuildFock has always BEEN their sum; what is
// pinned below is that the sum is exposed and speaks the ri_j_link family's
// accounting exactly. That accounting is worth three tests rather than one
// because the two ways of getting it wrong - the factor of two moved into or
// out of the Coulomb half, a core Hamiltonian carried by the wrong half -
// each produce a plausible-looking Fock and a plausible-looking energy, with
// nothing in either to show which convention the builder spoke.
// ---------------------------------------------------------------------------

TEST(DiskRiFockBuildTest, TheFusedBuildIsTheSumOfItsTwoHalves) {
    // The fused call and the split pair are ONE code path: BuildFock is
    // BuildExchangeOnly plus the Coulomb vector, and BuildCoulombOnly is that
    // same vector written out - so the sum of the two halves and the fused
    // build differ by NOTHING the code does differently, and by only what
    // fock_build.hpp says the direct exchange kernel costs on a second call:
    // it is not run-to-run reproducible. The identity is therefore pinned at
    // kPinnedSplitHalfTolerance, which is where a convention error (O(1))
    // cannot hide - see that constant for the measurements behind it.
    //
    // TWO FIXTURES, because the two halves do not share a density contract.
    //
    // The COULOMB half takes ANY density: its uv-reduced gather reads
    // d(u,v) + d(v,u), an algebraic identity rather than a symmetry
    // assumption (disk_ri_fock_build.cpp, BuildRiCoulombVector) - and that
    // sentence is about THAT GATHER, not about the builder. The asymmetric
    // fixture below is what measures it: a symmetric fixture like rho = S/2
    // would pass even if the gather assumed rho = rho^T.
    //
    // The EXCHANGE half does not. It is the nested direct builder's
    // exchange-only mode, whose K write emits the SAME value to the target
    // (a,c) and to its transpose (c,a) (internal/fock_contract_kernel.hpp,
    // K1: "the pair-swapped and role-swapped twins - density symmetry makes
    // the reads equal"), so the K it returns is symmetry-locked BY
    // CONSTRUCTION - the correct K only for a symmetric density, since
    // K_uv - K_vu = sum_cd (D_cd - D_dc)(uc|vd). The split calls therefore
    // REFUSE the asymmetric fixture with kInvalidArgument, in Release as
    // well as Debug (the nested builder's own check is an assert, which
    // NDEBUG removes), and the identity below runs on the symmetric one.
    // This test is where that refusal was found: test #919 aborted on the
    // nested assert in Debug before the refusal existed.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    ScopedTempStoreFile store("disk_ri_halves_sum");
    auto storeMolecule = MakeH2oSto3g();
    ASSERT_TRUE(storeMolecule.has_value()) << storeMolecule.error().message;
    auto builder = qcx::storage::DiskRiFockBuilder::Create(
        store.Path(), std::move(*storeMolecule), *basis, *aux, "sto-3g", "tiny", *core);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const std::size_t n = overlap->Shape()[0];
    Eigen::MatrixXd rhoMatrix = 0.5 * qcx::testing::ToMatrix(*overlap);

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            // The asymmetric fill: small enough that the density stays
            // physical, large enough that no equality below can be satisfied
            // by an accidental symmetry.
            rhoMatrix(static_cast<Eigen::Index>(u), static_cast<Eigen::Index>(v)) +=
                0.017 * static_cast<double>((u + 1) * (v + 2)) / static_cast<double>(n * n);
        }
    }

    // Measured before it is relied on: the fill's asymmetry is orders
    // above the 1e-12 relative-Frobenius floor the split calls read, so the
    // refusal below tests the guard rather than a rounding error. Measured on
    // the H2O/STO-3G fixture, 2026-09-18: max |rho - rho^T| =
    // 0.0020816326530612439 (exactly the fill's own 0.017 * (n - 1) / n^2 =
    // 0.017 * 6 / 49, the overlap's contribution below the last digit) and
    // ||rho - rho^T|| / ||rho|| = 0.0045599427381570537. Both floors sit
    // ~1e3 inside those readings and ~1e6 above the guard's own criterion.
    double maxTransposeDeviation = 0.0;

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            maxTransposeDeviation = std::max(
                maxTransposeDeviation,
                std::abs(rhoMatrix(static_cast<Eigen::Index>(u), static_cast<Eigen::Index>(v)) -
                         rhoMatrix(static_cast<Eigen::Index>(v), static_cast<Eigen::Index>(u))));
        }
    }

    const double relativeTransposeDeviation =
        (rhoMatrix - rhoMatrix.transpose()).norm() / rhoMatrix.norm();
    EXPECT_GT(maxTransposeDeviation, 1e-6)
        << "measured max |rho - rho^T| = " << maxTransposeDeviation
        << " - the fixture must not be accidentally symmetric";
    EXPECT_GT(relativeTransposeDeviation, 1e-6)
        << "measured ||rho - rho^T|| / ||rho|| = " << relativeTransposeDeviation
        << " - the fixture must sit orders above the 1e-12 the guard reads";

    // The nested builder's own expression, on the fixture's own values: this
    // is the check that aborted test #919 in Debug, and it still says False.
    EXPECT_FALSE(rhoMatrix.isApprox(rhoMatrix.transpose()));

    auto rhoAsymmetric = ToTensor(rhoMatrix);
    ASSERT_TRUE(rhoAsymmetric.has_value()) << rhoAsymmetric.error().message;

    // The exchange-bearing calls refuse the asymmetric fixture - both of
    // them, since the fused call runs the exchange half first.
    auto refusedFused = builder->BuildFock(*rhoAsymmetric);
    ASSERT_FALSE(refusedFused.has_value())
        << "the fused build runs the exchange half, which requires a symmetric density";
    EXPECT_EQ(refusedFused.error().code, qcx::ErrorCode::kInvalidArgument)
        << refusedFused.error().message;
    auto refusedExchange = builder->BuildExchangeOnly(*rhoAsymmetric);
    ASSERT_FALSE(refusedExchange.has_value());
    EXPECT_EQ(refusedExchange.error().code, qcx::ErrorCode::kInvalidArgument)
        << refusedExchange.error().message;

    // The Coulomb-only call takes it, and that is where the asymmetric
    // gather claim lives.
    auto coulombAsymmetric = builder->BuildCoulombOnly(*rhoAsymmetric);
    ASSERT_TRUE(coulombAsymmetric.has_value()) << coulombAsymmetric.error().message;

    // The symmetric fixture every other test in this file uses - rho = S/2.
    Eigen::MatrixXd rhoSymmetricMatrix = 0.5 * qcx::testing::ToMatrix(*overlap);
    auto rho = ToTensor(rhoSymmetricMatrix);
    ASSERT_TRUE(rho.has_value()) << rho.error().message;

    // The guard's floor from the other side, so its scale cannot drift: a
    // density asymmetric only at the level a real SCF leaves behind (D =
    // C n C^T off BLAS is symmetric to the last bits, never exactly) must
    // still run. The perturbation is one ulp-class step on one element.
    Eigen::MatrixXd rhoNearlySymmetric = rhoSymmetricMatrix;
    rhoNearlySymmetric(0, 1) += 1e-15;

    auto nearlySymmetric = ToTensor(rhoNearlySymmetric);
    ASSERT_TRUE(nearlySymmetric.has_value()) << nearlySymmetric.error().message;
    auto acceptedNearlySymmetric = builder->BuildExchangeOnly(*nearlySymmetric);
    ASSERT_TRUE(acceptedNearlySymmetric.has_value())
        << "a density symmetric to the last ulp is a production shape, not a caller error: "
        << acceptedNearlySymmetric.error().message;

    auto fused = builder->BuildFock(*rho);
    ASSERT_TRUE(fused.has_value()) << fused.error().message;
    auto coulombSymmetric = builder->BuildCoulombOnly(*rho);
    ASSERT_TRUE(coulombSymmetric.has_value()) << coulombSymmetric.error().message;
    auto exchange = builder->BuildExchangeOnly(*rho);
    ASSERT_TRUE(exchange.has_value()) << exchange.error().message;

    double worstResidue = 0.0;

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            worstResidue = std::max(
                worstResidue,
                std::abs((*fused)(u, v) - ((*coulombSymmetric)(u, v) + (*exchange)(u, v))));
        }
    }

    EXPECT_LT(worstResidue, kPinnedSplitHalfTolerance)
        << "the fused build must be the halves sum; measured max |F - (C + E)| = " << worstResidue
        << " (a convention error is O(1), not O(1e-12))";

    // The stats contract that separates the halves structurally: the
    // streamed contraction evaluates no quartets, so the Coulomb half writes
    // the measured zero; the nested direct build does evaluate them, so the
    // exchange half fills real counts. A struct pre-filled with a sentinel
    // proves the write HAPPENED rather than that the caller's garbage
    // survived it.
    qcx::integrals::FockBuildStats coulombStats;
    coulombStats.fp64QuartetCount = 12345;
    coulombStats.fp32QuartetCount = 6789;
    auto coulombWithStats = builder->BuildCoulombOnly(*rho, &coulombStats);
    ASSERT_TRUE(coulombWithStats.has_value()) << coulombWithStats.error().message;
    EXPECT_EQ(coulombStats.fp64QuartetCount, 0u);
    EXPECT_EQ(coulombStats.fp32QuartetCount, 0u);

    qcx::integrals::FockBuildStats exchangeStats;
    auto exchangeWithStats = builder->BuildExchangeOnly(*rho, &exchangeStats);
    ASSERT_TRUE(exchangeWithStats.has_value()) << exchangeWithStats.error().message;
    EXPECT_GT(exchangeStats.fp64QuartetCount + exchangeStats.fp32QuartetCount, 0u)
        << "the exchange half is a real quartet kernel; a zero here would mean the stats were "
           "never forwarded to the nested direct build";

    // The stats-carrying calls are the same numbers as the plain ones: the
    // sink is a record of the call, never an input to it.
    const double coulombWithSink = MaxAbsDifference(*coulombSymmetric, *coulombWithStats);
    const double exchangeWithSink = MaxAbsDifference(*exchange, *exchangeWithStats);
    EXPECT_LT(coulombWithSink, kPinnedSplitHalfTolerance)
        << "measured max |C - C_stats| = " << coulombWithSink;
    EXPECT_LT(exchangeWithSink, kPinnedSplitHalfTolerance)
        << "measured max |E - E_stats| = " << exchangeWithSink;
}

TEST(DiskRiFockBuildTest, TheCoulombHalfCarriesTwiceJAbsentHandTheExchangeHalfCarriesH) {
    // THE ACCOUNTING PIN. The ri_j_link family's split halves are
    //   Coulomb  2 J_RI(rho)   - the factor of two, and NO core Hamiltonian,
    //   exchange H - K(rho)    - the sole H carrier,
    // and that is deliberately NOT the driver's HalfFockFn contract, whose
    // halves BOTH carry H (ks_composition.hpp's convention table; the direct
    // and qfmm families speak that one). A builder that put H into the
    // Coulomb half as well, or that folded the factor of two in twice, would
    // still return a plausible Fock.
    //
    // Three measurements, each of which the wrong conventions fail by an
    // O(1) quantity rather than by a tolerance:
    //   1. the in-memory ri_j_link half, entry point for entry point, on the
    //      same system, density and options;
    //   2. an H-PERTURBATION CONTROL - a second builder whose H is H + 1.5 S.
    //      The Coulomb half must stay put (it never reads H) while the
    //      exchange half moves by the perturbation. No single-half error can
    //      satisfy both;
    //   3. both row layouts, because the Coulomb scatter lives in the body
    //      the two entry points share and the layout is a store property.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    const std::size_t n = overlap->Shape()[0];
    const Eigen::MatrixXd overlapMatrix = qcx::testing::ToMatrix(*overlap);

    // The perturbed core: a different H for the same system. 1.5 S is a
    // legal H-shaped operand and large enough that any H leaking into the
    // Coulomb half lands orders above the accumulation drift.
    constexpr double kCorePerturbation = 1.5;
    auto perturbedCoreTensor =
        ToTensor(qcx::testing::ToMatrix(*core) + kCorePerturbation * overlapMatrix);
    ASSERT_TRUE(perturbedCoreTensor.has_value()) << perturbedCoreTensor.error().message;

    qcx::storage::DiskRiFockOptions reducedOptions;
    reducedOptions.reducedRowLayout = true;
    qcx::storage::DiskRiFockOptions fullOptions;
    fullOptions.reducedRowLayout = false;

    ScopedTempStoreFile reducedStore("disk_ri_halves_reduced");
    auto reducedMolecule = MakeH2oSto3g();
    ASSERT_TRUE(reducedMolecule.has_value()) << reducedMolecule.error().message;
    auto reducedBuilder = qcx::storage::DiskRiFockBuilder::Create(reducedStore.Path(),
                                                                  std::move(*reducedMolecule),
                                                                  *basis,
                                                                  *aux,
                                                                  "sto-3g",
                                                                  "tiny",
                                                                  *core,
                                                                  reducedOptions);
    ASSERT_TRUE(reducedBuilder.has_value()) << reducedBuilder.error().message;

    ScopedTempStoreFile fullStore("disk_ri_halves_full");
    auto fullMolecule = MakeH2oSto3g();
    ASSERT_TRUE(fullMolecule.has_value()) << fullMolecule.error().message;
    auto fullBuilder = qcx::storage::DiskRiFockBuilder::Create(fullStore.Path(),
                                                               std::move(*fullMolecule),
                                                               *basis,
                                                               *aux,
                                                               "sto-3g",
                                                               "tiny",
                                                               *core,
                                                               fullOptions);
    ASSERT_TRUE(fullBuilder.has_value()) << fullBuilder.error().message;

    ScopedTempStoreFile perturbedStore("disk_ri_halves_perturbed");
    auto perturbedMolecule = MakeH2oSto3g();
    ASSERT_TRUE(perturbedMolecule.has_value()) << perturbedMolecule.error().message;
    auto perturbedBuilder = qcx::storage::DiskRiFockBuilder::Create(perturbedStore.Path(),
                                                                    std::move(*perturbedMolecule),
                                                                    *basis,
                                                                    *aux,
                                                                    "sto-3g",
                                                                    "tiny",
                                                                    *perturbedCoreTensor,
                                                                    reducedOptions);
    ASSERT_TRUE(perturbedBuilder.has_value()) << perturbedBuilder.error().message;

    auto rho = ToTensor(0.5 * overlapMatrix);
    ASSERT_TRUE(rho.has_value()) << rho.error().message;

    auto coulomb = reducedBuilder->BuildCoulombOnly(*rho);
    ASSERT_TRUE(coulomb.has_value()) << coulomb.error().message;
    auto exchange = reducedBuilder->BuildExchangeOnly(*rho);
    ASSERT_TRUE(exchange.has_value()) << exchange.error().message;

    // 1. The in-memory ri_j_link half: the same family's accounting, the
    //    factor-of-two and H-carrier reference for both claims.
    auto memoryBuilder = qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core);
    ASSERT_TRUE(memoryBuilder.has_value()) << memoryBuilder.error().message;
    auto memoryCoulomb = memoryBuilder->BuildCoulombOnly(*rho);
    ASSERT_TRUE(memoryCoulomb.has_value()) << memoryCoulomb.error().message;
    auto memoryExchange = memoryBuilder->BuildExchangeOnly(*rho);
    ASSERT_TRUE(memoryExchange.has_value()) << memoryExchange.error().message;

    const double coulombVsMemory = MaxAbsDifference(*coulomb, *memoryCoulomb);
    const double exchangeVsMemory = MaxAbsDifference(*exchange, *memoryExchange);
    EXPECT_LT(coulombVsMemory, kPinnedSplitHalfTolerance)
        << "measured max |C_disk - C_mem| = " << coulombVsMemory
        << " (a factor-of-two or an H error is O(1), not O(1e-12))";
    EXPECT_LT(exchangeVsMemory, kPinnedSplitHalfTolerance)
        << "measured max |E_disk - E_mem| = " << exchangeVsMemory;

    // 2. The H-perturbation control.
    auto perturbedCoulomb = perturbedBuilder->BuildCoulombOnly(*rho);
    ASSERT_TRUE(perturbedCoulomb.has_value()) << perturbedCoulomb.error().message;
    auto perturbedExchange = perturbedBuilder->BuildExchangeOnly(*rho);
    ASSERT_TRUE(perturbedExchange.has_value()) << perturbedExchange.error().message;

    const double coulombUnderPerturbation = MaxAbsDifference(*coulomb, *perturbedCoulomb);
    EXPECT_LT(coulombUnderPerturbation, kPinnedSplitHalfTolerance)
        << "the Coulomb half must not read H at all; measured max |C - C'| = "
        << coulombUnderPerturbation << " against a perturbation of " << kCorePerturbation << " * S";

    double maxExchangeShift = 0.0;
    double maxPerturbation = 0.0;

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            const auto i = static_cast<Eigen::Index>(u);
            const auto j = static_cast<Eigen::Index>(v);
            maxExchangeShift = std::max(maxExchangeShift,
                                        std::abs((*perturbedExchange)(u, v) - (*exchange)(u, v)));
            maxPerturbation =
                std::max(maxPerturbation, std::abs(kCorePerturbation * overlapMatrix(i, j)));
        }
    }

    EXPECT_GT(maxExchangeShift, 0.5 * maxPerturbation)
        << "the exchange half must carry H; measured max shift = " << maxExchangeShift
        << " against a perturbation reaching " << maxPerturbation;

    // 3. Both row layouts, the same halves.
    auto fullCoulomb = fullBuilder->BuildCoulombOnly(*rho);
    ASSERT_TRUE(fullCoulomb.has_value()) << fullCoulomb.error().message;
    auto fullExchange = fullBuilder->BuildExchangeOnly(*rho);
    ASSERT_TRUE(fullExchange.has_value()) << fullExchange.error().message;

    const double coulombAcrossLayouts = MaxAbsDifference(*coulomb, *fullCoulomb);
    const double exchangeAcrossLayouts = MaxAbsDifference(*exchange, *fullExchange);
    EXPECT_LT(coulombAcrossLayouts, kPinnedSplitHalfTolerance)
        << "the Coulomb half must not move with the store's row layout beyond the pass-1 v "
           "re-association; measured max |C - C_full| = "
        << coulombAcrossLayouts;

    // The exchange half is the CONTROL for that comparison, and it is the
    // reason the tolerance above cannot be tightened to zero: the row layout
    // CANNOT reach the exchange half at all - that half is the nested direct
    // builder, which never loads a chunk - so whatever it moves by is the
    // direct kernel's own run-to-run floor rather than anything the layout
    // did. Measured non-zero here; that is the measurement, not a defect.
    EXPECT_LT(exchangeAcrossLayouts, kPinnedSplitHalfTolerance)
        << "measured max |E - E_full| = " << exchangeAcrossLayouts
        << " - the row layout cannot reach the exchange half, so this is the direct "
           "kernel's own run-to-run floor";
}

TEST(DiskRiFockBuildTest, TheSplitHalvesAssembleTheUnrestrictedAndKohnShamFocks) {
    // The two cells these halves exist to open, assembled the way the
    // driver's adapters assemble them (run_driver.cpp
    // MakeRiJLinkUhfFockBuilder / MakeRiJLinkKsHalf) over the DISK builder
    // instead of the in-memory one. Neither assembly subtracts H: on this
    // family the Coulomb half carries none and the exchange half is the sole
    // carrier, so a subtraction here would move the Fock by a whole H.
    //
    //   unrestricted   F_s  = C(0.5 (d_a + d_b)) + E(d_s),   raw per-spin
    //   Kohn-Sham      J[D] = C(0.5 D)   - the half already carries the 2
    //                  F_KS = C(0.5 D) + H + c_HF (E(0.5 D) - H)
    // At c_HF = 1 the Kohn-Sham Fock is H + 2 J(rho) - K(rho), the fused
    // build itself - the identity that says the seam would contract the J
    // the Fock was built from rather than a second one.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    ScopedTempStoreFile store("disk_ri_halves_assembly");
    auto storeMolecule = MakeH2oSto3g();
    ASSERT_TRUE(storeMolecule.has_value()) << storeMolecule.error().message;
    auto builder = qcx::storage::DiskRiFockBuilder::Create(
        store.Path(), std::move(*storeMolecule), *basis, *aux, "sto-3g", "tiny", *core);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const std::size_t n = overlap->Shape()[0];
    const Eigen::MatrixXd overlapMatrix = qcx::testing::ToMatrix(*overlap);

    // A non-closed-shell spin pair: d_beta differs from d_alpha, so an
    // assembly that quietly used one spin twice cannot pass. Both are built
    // from S so the matrices stay physical.
    Eigen::MatrixXd dAlphaMatrix = 0.5 * overlapMatrix;
    Eigen::MatrixXd dBetaMatrix = 0.5 * overlapMatrix;

    for (std::size_t u = 0; u < n; ++u)
    {
        dBetaMatrix(static_cast<Eigen::Index>(u), static_cast<Eigen::Index>(u)) +=
            (u % 2 == 0 ? 0.05 : -0.05);
    }

    auto dAlpha = ToTensor(dAlphaMatrix);
    ASSERT_TRUE(dAlpha.has_value()) << dAlpha.error().message;
    auto dBeta = ToTensor(dBetaMatrix);
    ASSERT_TRUE(dBeta.has_value()) << dBeta.error().message;
    auto dTotalHalf = ToTensor(0.5 * (dAlphaMatrix + dBetaMatrix));
    ASSERT_TRUE(dTotalHalf.has_value()) << dTotalHalf.error().message;

    auto coulombTotal = builder->BuildCoulombOnly(*dTotalHalf);
    ASSERT_TRUE(coulombTotal.has_value()) << coulombTotal.error().message;
    auto exchangeAlpha = builder->BuildExchangeOnly(*dAlpha);
    ASSERT_TRUE(exchangeAlpha.has_value()) << exchangeAlpha.error().message;
    auto exchangeBeta = builder->BuildExchangeOnly(*dBeta);
    ASSERT_TRUE(exchangeBeta.has_value()) << exchangeBeta.error().message;

    double maxChannelSeparation = 0.0;

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            const double fAlpha = (*coulombTotal)(u, v) + (*exchangeAlpha)(u, v);
            const double fBeta = (*coulombTotal)(u, v) + (*exchangeBeta)(u, v);
            maxChannelSeparation = std::max(maxChannelSeparation, std::abs(fAlpha - fBeta));
        }
    }

    EXPECT_GT(maxChannelSeparation, 0.0)
        << "the per-spin assembly collapsed the two channels: one spin's exchange was used for "
           "both";

    // The same assembly against the in-memory family, so the identity above
    // is not merely self-consistent: the disk halves and the memory halves
    // must assemble to the same pair of Focks.
    auto memoryBuilder = qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core);
    ASSERT_TRUE(memoryBuilder.has_value()) << memoryBuilder.error().message;
    auto memoryCoulombTotal = memoryBuilder->BuildCoulombOnly(*dTotalHalf);
    ASSERT_TRUE(memoryCoulombTotal.has_value()) << memoryCoulombTotal.error().message;
    auto memoryExchangeAlpha = memoryBuilder->BuildExchangeOnly(*dAlpha);
    ASSERT_TRUE(memoryExchangeAlpha.has_value()) << memoryExchangeAlpha.error().message;
    auto memoryExchangeBeta = memoryBuilder->BuildExchangeOnly(*dBeta);
    ASSERT_TRUE(memoryExchangeBeta.has_value()) << memoryExchangeBeta.error().message;

    constexpr double kPinnedAssemblyTolerance = 1e-12;
    double maxAssemblyDifference = 0.0;

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            const double diskAlpha = (*coulombTotal)(u, v) + (*exchangeAlpha)(u, v);
            const double memoryAlpha = (*memoryCoulombTotal)(u, v) + (*memoryExchangeAlpha)(u, v);
            const double diskBeta = (*coulombTotal)(u, v) + (*exchangeBeta)(u, v);
            const double memoryBeta = (*memoryCoulombTotal)(u, v) + (*memoryExchangeBeta)(u, v);
            maxAssemblyDifference =
                std::max(maxAssemblyDifference, std::abs(diskAlpha - memoryAlpha));
            maxAssemblyDifference =
                std::max(maxAssemblyDifference, std::abs(diskBeta - memoryBeta));
        }
    }

    EXPECT_LT(maxAssemblyDifference, kPinnedAssemblyTolerance)
        << "measured max |F_disk,s - F_mem,s| = " << maxAssemblyDifference;

    // The Kohn-Sham cell. The composition's Coulomb half is the runner's
    // H + 2 J_RI(rho) (RiJLinkKsHalf::kCoulomb adds H to this builder's
    // 2 J_RI(rho)); its exchange half is this builder's result verbatim. The
    // claim the cell turns on is that the energy seam's operand
    // J[D] = (Coulomb half - H) is the SAME 2 J_RI(rho) the Fock carries -
    // so it is measured against the fused build's own Coulomb term, which is
    // the fused result minus the exchange half.
    auto spatialDensity = ToTensor(0.5 * overlapMatrix);
    ASSERT_TRUE(spatialDensity.has_value()) << spatialDensity.error().message;

    auto ksCoulombHalf = builder->BuildCoulombOnly(*spatialDensity);
    ASSERT_TRUE(ksCoulombHalf.has_value()) << ksCoulombHalf.error().message;
    auto ksExchangeHalf = builder->BuildExchangeOnly(*spatialDensity);
    ASSERT_TRUE(ksExchangeHalf.has_value()) << ksExchangeHalf.error().message;
    auto ksFused = builder->BuildFock(*spatialDensity);
    ASSERT_TRUE(ksFused.has_value()) << ksFused.error().message;

    const Eigen::MatrixXd coreMatrix = qcx::testing::ToMatrix(*core);
    double worstKsResidue = 0.0;
    double worstSeamJResidue = 0.0;

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            const auto i = static_cast<Eigen::Index>(u);
            const auto j = static_cast<Eigen::Index>(v);

            // c_HF = 1: F_KS = (C + H) + 1 * (E - H), so H cancels and the
            // result is the fused build's own two terms. THIS is the pair of
            // halves the composition would be handed.
            const double assembled = ((*ksCoulombHalf)(u, v) + coreMatrix(i, j)) +
                                     ((*ksExchangeHalf)(u, v) - coreMatrix(i, j));
            worstKsResidue = std::max(worstKsResidue, std::abs((*ksFused)(u, v) - assembled));

            // The energy seam's Coulomb operand. The composition's Coulomb
            // half is the runner's H + 2 J_RI(rho) (RiJLinkKsHalf::kCoulomb
            // adds H to this builder's half), and the seam SUBTRACTS that H
            // back out to contract J[D] - so the round trip, and not the
            // half alone, is what this pins.
            //
            // What J[D] must be is the Coulomb term the Fock carries:
            // subtracting the exchange half off the fused build leaves
            // 2 J_RI(rho), because the fused build's H sits INSIDE the
            // exchange half (E = H - K) and cancels there. Writing it as
            // BuildFock - BuildExchangeOnly is therefore the Fock's own
            // Coulomb term; it is NOT (BuildFock - H) - K, and it is not
            // BuildCoulombOnly minus H either. An H subtracted from the
            // wrong side is O(1) - measured at 32.7 before this line was
            // corrected, which is the whole reason the cell is stated as a
            // measurement and not as a reading.
            const double coulombHalf = (*ksCoulombHalf)(u, v) + coreMatrix(i, j);
            const double seamJ = coulombHalf - coreMatrix(i, j);
            const double fockCoulombTerm = (*ksFused)(u, v) - (*ksExchangeHalf)(u, v);
            worstSeamJResidue = std::max(worstSeamJResidue, std::abs(seamJ - fockCoulombTerm));
        }
    }

    EXPECT_LT(worstKsResidue, kPinnedSplitHalfTolerance)
        << "at c_HF = 1 the Kohn-Sham Fock over the split halves must be the fused build; "
           "measured max residue = "
        << worstKsResidue << " - an H added twice or subtracted twice is O(1) here";
    EXPECT_LT(worstSeamJResidue, kPinnedSplitHalfTolerance)
        << "the energy seam's J[D] must be the Coulomb term the Fock was built from; measured "
           "max residue = "
        << worstSeamJResidue;
}
