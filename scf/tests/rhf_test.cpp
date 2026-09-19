// RHF tests: the H2/STO-3G total energy pinned against the committed mpmath
// reference grid (integrals/tests/data/h2_sto3g_reference.csv - 30-digit
// integrals from tools/gen_integrals_reference.py), the electronic/nuclear
// decomposition, and the rejection paths.
#include "h2_sto3g.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/scf/rhf.hpp"
#include "scf_common.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

TEST(RhfTest, H2Sto3gConvergesToPinnedEnergy) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    // Plain Roothaan (no DIIS): the un-accelerated reference path.
    qcx::scf::RhfOptions options;
    options.useDiis = false;
    const auto result = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri, options);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // R = 1.4 bohr. Three independent reference chains agree to 1e-9: the
    // committed 30-digit mpmath grid (tools/gen_integrals_reference.py)
    // through an independent SCF gives -1.116714325176, and pyscf
    // (RHF/STO-3G, conv_tol 1e-12, WSL 2026-08-17) gives
    // -1.116714325062551. The tolerance 1e-8 covers both.
    EXPECT_NEAR(result->totalEnergy, -1.1167143252, 1e-8);
    EXPECT_NEAR(result->electronicEnergy, -1.8310000395, 1e-8);
    EXPECT_NEAR(result->totalEnergy - result->electronicEnergy, 1.0 / 1.4, 1e-12);
    EXPECT_TRUE(result->converged);
    EXPECT_LT(result->iterations, 10);
}

TEST(RhfTest, TraceFileReceivesPerIterationLines) {
    // The trace side-channel: with traceFile set, the loop appends one
    // flushed line per iteration after its convergence decision. H2/STO-3G
    // converges in a couple of iterations, so the file must carry at least
    // the iter=1 line and a converged one.
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    const std::filesystem::path tracePath =
        std::filesystem::temp_directory_path() / "qcx_scf_trace_test.log";
    std::filesystem::remove(tracePath);

    qcx::scf::RhfOptions options;
    options.traceFile = tracePath.string();
    const auto result = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri, options);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->converged);

    ASSERT_TRUE(std::filesystem::exists(tracePath));
    std::ifstream trace(tracePath);
    std::vector<std::string> lines;
    std::string line;

    while (std::getline(trace, line))
    {
        if (!line.empty())
        {
            lines.push_back(line);
        }
    }

    ASSERT_GE(lines.size(), 2u);
    EXPECT_TRUE(lines.front().starts_with("iter=1 "));
    const bool sawConverged = std::any_of(lines.begin(), lines.end(), [](const std::string& l) {
        return l.find("conv=1") != std::string::npos;
    });
    EXPECT_TRUE(sawConverged);
}

TEST(RhfTest, DensityDumpFileReceivesBinaryStream) {
    // The C12H26 discriminating-experiment side-channel: with
    // densityDumpFile set, the loop writes the magic-prefixed binary stream
    // (ScfDensityDumpWriter format): magic, dimension, the overlap and core
    // records once, then alternating (density, physical-Fock) records per
    // iteration. The empty default keeps the zero-cost path (no file, no
    // allocation) - asserted by DensityDumpDefaultStaysOff below.
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    const std::filesystem::path dumpPath =
        std::filesystem::temp_directory_path() / "qcx_scf_density_dump_test.bin";
    std::filesystem::remove(dumpPath);

    qcx::scf::RhfOptions options;
    options.densityDumpFile = dumpPath.string();
    const auto result = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri, options);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->converged);

    ASSERT_TRUE(std::filesystem::exists(dumpPath));
    std::ifstream dump(dumpPath, std::ios::binary);
    ASSERT_TRUE(dump.good());

    char magic[8] = {};
    dump.read(magic, 8);
    EXPECT_EQ(std::string(magic, 8), "QXCDFDMP");

    std::uint64_t n = 0;
    dump.read(reinterpret_cast<char*>(&n), 8);
    EXPECT_EQ(n, 2u); // H2/STO-3G: one s function per hydrogen.

    // Walk the records: the overlap and core records lead, then alternating
    // density/physical-Fock pairs - one per iteration, with the pair order
    // preserved (the density the Fock was built from precedes it).
    std::vector<std::uint8_t> tags;
    constexpr std::uint8_t kOverlapTag = 0;
    constexpr std::uint8_t kCoreHamiltonianTag = 1;
    constexpr std::uint8_t kDensityTag = 2;
    constexpr std::uint8_t kFockTag = 3;

    while (true)
    {
        char tagByte = 0;
        dump.read(&tagByte, 1);

        if (!dump.good())
        {
            break; // Clean EOF; a truncated record trips the reads below.
        }

        const auto tag = static_cast<std::uint8_t>(tagByte);
        std::uint64_t iteration = 0;
        std::uint64_t recordN = 0;
        dump.read(reinterpret_cast<char*>(&iteration), 8);
        dump.read(reinterpret_cast<char*>(&recordN), 8);
        EXPECT_EQ(recordN, n) << "record dimension mismatch at tag " << static_cast<int>(tag);
        dump.seekg(static_cast<std::streamoff>(recordN * recordN) * 8, std::ios::cur);
        tags.push_back(tag);
    }

    ASSERT_GE(tags.size(), 4u);
    EXPECT_EQ(tags[0], kOverlapTag);
    EXPECT_EQ(tags[1], kCoreHamiltonianTag);
    EXPECT_EQ(tags[2], kDensityTag);
    EXPECT_EQ(tags[3], kFockTag);

    // At least one full (density, fock) pair beyond the lead-in, and the
    // pair structure must hold for every subsequent record.
    for (std::size_t i = 2; i + 1 < tags.size(); i += 2)
    {
        EXPECT_EQ(tags[i], kDensityTag);
        EXPECT_EQ(tags[i + 1], kFockTag);
    }

    EXPECT_EQ(tags.size() % 2, 0u);

    // Clean up so the sibling zero-cost test can assert the default leaves
    // no file behind.
    dump.close();
    std::filesystem::remove(dumpPath);
}

TEST(RhfTest, DensityDumpDefaultStaysOff) {
    // The zero-cost default: no file, no stream - the bit-parity pins
    // are absolute, so this side-channel must not exist unless opened.
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    qcx::scf::RhfOptions options;
    const auto result = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri, options);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->converged);

    // Nothing was configured, so nothing may exist: the temp path that the
    // binary-stream test uses must not appear here.
    EXPECT_FALSE(std::filesystem::exists(std::filesystem::temp_directory_path() /
                                         "qcx_scf_density_dump_test.bin"));
}

TEST(RhfTest, OddElectronCountIsRejected) {
    // H2 with charge -1: three electrons, outside closed-shell RHF.
    auto coordinates = CpuTensor2::Create({2, 3});
    ASSERT_TRUE(coordinates.has_value()) << coordinates.error().message;
    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.4;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();
    const auto molecule = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        -1,
        2);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    const auto result = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kUnimplemented);
}

TEST(RhfTest, MismatchedFunctionCountIsRejected) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    // The ERI tensor covers two functions; 1x1 matrices cannot match it.
    const auto result = qcx::scf::RunRhfScf(
        *molecule, Eigen::MatrixXd::Identity(1, 1), Eigen::MatrixXd::Identity(1, 1), *eri);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(RhfTest, NonPositiveOptionsAreRejected) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    // A zero iteration budget is the empty run: reject it like the linalg
    // seams reject non-positive budgets.
    qcx::scf::RhfOptions options;
    options.maxIterations = 0;
    const auto result = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(RhfTest, NearSingularOverlapIsRejected) {
    // Coverage: no test fed a near-singular
    // overlap to either SCF loop. REVISED: the old response was to abort
    // the run. A linearly
    // dependent direction is expected in a large diffuse basis, so the
    // response is now to REMOVE it - the run proceeds in the reduced
    // orthonormal space - and to DISCLOSE the count. Aborting survives only
    // where it is a genuine error: not enough directions left for the
    // occupied orbitals.
    const Eigen::MatrixXd degenerate =
        Eigen::MatrixXd::Ones(3, 3); // Rank 1: eigenvalues {3, 0, 0}.
    auto removed = qcx::scf::internal::OrthogonalizeOverlap(degenerate);
    ASSERT_TRUE(removed.has_value()) << removed.error().message;
    EXPECT_EQ(removed->numRemoved, 2u); // The two null directions.
    EXPECT_EQ(removed->KeptDimension(), 1u);
    EXPECT_EQ(removed->x.rows(), 3); // n x nKept: rectangular, not square.
    EXPECT_EQ(removed->x.cols(), 1);

    // A near-singular (but non-degenerate) matrix crosses the same floor: the
    // all-ones rank-1 block plus 1e-12 shifts the two null eigenvalues to
    // 1e-12, ratio 3.3e-13 < 1e-8.
    Eigen::MatrixXd nearSingular = Eigen::MatrixXd::Ones(3, 3);
    nearSingular.diagonal().array() += 1e-12;
    auto alsoRemoved = qcx::scf::internal::OrthogonalizeOverlap(nearSingular);
    ASSERT_TRUE(alsoRemoved.has_value()) << alsoRemoved.error().message;
    EXPECT_EQ(alsoRemoved->numRemoved, 2u);

    // A well-conditioned matrix removes nothing and still orthogonalizes:
    // X^T S X = I, with X the classic square n x n.
    Eigen::MatrixXd wellConditioned = Eigen::MatrixXd::Ones(3, 3);
    wellConditioned.diagonal().array() += 1e-4;
    auto accepted = qcx::scf::internal::OrthogonalizeOverlap(wellConditioned);
    ASSERT_TRUE(accepted.has_value()) << accepted.error().message;
    EXPECT_EQ(accepted->numRemoved, 0u);
    const Eigen::MatrixXd& acceptedX = accepted->x;
    const Eigen::MatrixXd product = acceptedX.transpose() * wellConditioned * acceptedX;
    EXPECT_LT((product - Eigen::MatrixXd::Identity(3, 3)).norm(), 1e-8);

    // The removal path's whole point: for the KEPT directions the reduced
    // transform is an identity too - X^T S X = I_numKept is what makes the
    // SCF valid in the reduced space, and it is what the square case above
    // checks in the no-removal limit.
    const Eigen::MatrixXd& removedX = removed->x;
    const Eigen::MatrixXd removedProduct = removedX.transpose() * degenerate * removedX;
    EXPECT_LT((removedProduct - Eigen::MatrixXd::Identity(1, 1)).norm(), 1e-8);

    // The RHF loop now RUNS this end-to-end instead of refusing: the real
    // H2/STO-3G overlap with the first two rows (and columns, keeping it
    // symmetric) made identical is rank-deficient.
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    Eigen::MatrixXd corrupted = ToMatrix(*overlap);
    corrupted.row(1) = corrupted.row(0);
    corrupted.col(1) = corrupted.col(0);
    // H2/STO-3G is n = 2 with one occupied orbital, so the one removed
    // direction leaves exactly enough room: the run must PROCEED, report the
    // removal, and come back with a density in the full AO dimension (the
    // back-transform) rather than a truncated one.
    const auto result =
        qcx::scf::RunRhfScf(*molecule, corrupted, ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(result->numRemovedOverlapDirections, 1u);
    EXPECT_EQ(result->density.rows(), 2);
    EXPECT_EQ(result->density.cols(), 2);
    EXPECT_EQ(result->coefficients.rows(), 2);
    EXPECT_EQ(result->coefficients.cols(), 1); // n x numKept.

    // Not enough room is the genuine error: the same overlap with a system
    // that needs TWO occupied orbitals cannot be run in one dimension.
    Eigen::MatrixXd twoOccupied = corrupted;
    const auto refused = qcx::scf::internal::OrthogonalizeOverlap(twoOccupied);
    ASSERT_TRUE(refused.has_value());
    EXPECT_EQ(refused->KeptDimension(), 1u);
    // The loop's own guard is what refuses (KeptDimension < numOccupied); a
    // 2-electron system in a 2-function basis that keeps only one direction
    // is the boundary this check exists for.
}

TEST(RhfTest, RemovedDirectionsRecoverTheFullSpaceAnswer) {
    // THE ACCEPTANCE COMPARISON for the linear-dependence removal.
    // The rule it enforces: "a run that merely proceeds is not the
    // deliverable; a run that proceeds AND gets the same answer is".
    //
    // THE REFERENCE, NAMED: the ordinary H2/STO-3G run at n = 2, on the
    // integrals as built - no direction falls below the floor and the removal
    // is inert (asserted below, so the reference cannot silently become a
    // second removal run).
    //
    // THE FIXTURE: the SAME problem in a 3-function basis built by DUPLICATING
    // function 0 - S, the core Hamiltonian and the ERI block are the
    // reference's, mapped through {0, 1, 0}. The duplicate spans nothing new,
    // so S has rank 2 with n = 3, exactly one direction falls below the
    // relative floor, and the SCF runs in the kept 2-dimensional orthonormal
    // space and maps back.
    //
    // THE CLAIM, and it is the one this change can get wrong: the two runs
    // must land on the SAME energy and the SAME density, with the extended
    // run's density the reference's embedded by the same map. A wrong
    // back-transform - iterating D in the reduced space and reporting that, or
    // dropping the X ... X^T - is an O(1) error here, five orders above the
    // tolerance asserted.
    //
    // TOLERANCE, and why this one: each run stops on its own gate (energy
    // 1e-8 Ha, density RMS 1e-6), so agreement is bounded by those gates and
    // NOT by round-off. 1e-7 Ha and 1e-6 per density element are therefore the
    // honest claims; asserting tighter would be asserting the gates, and
    // tightening the gates for a test is not allowed on this project.
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    const Eigen::MatrixXd s = ToMatrix(*overlap);
    const Eigen::MatrixXd core = ToMatrix(*kinetic) + ToMatrix(*nuclear);

    const auto reference = qcx::scf::RunRhfScf(*molecule, s, core, *eri);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    EXPECT_EQ(reference->numRemovedOverlapDirections, 0u);

    const std::array<std::size_t, 3> map = {0, 1, 0};
    Eigen::MatrixXd sExtended(3, 3);
    Eigen::MatrixXd coreExtended(3, 3);

    for (Eigen::Index i = 0; i < 3; ++i)
    {
        for (Eigen::Index j = 0; j < 3; ++j)
        {
            sExtended(i, j) = s(static_cast<Eigen::Index>(map[static_cast<std::size_t>(i)]),
                                static_cast<Eigen::Index>(map[static_cast<std::size_t>(j)]));
            coreExtended(i, j) = core(static_cast<Eigen::Index>(map[static_cast<std::size_t>(i)]),
                                      static_cast<Eigen::Index>(map[static_cast<std::size_t>(j)]));
        }
    }

    auto eriExtended = CpuTensor4::Create({3, 3, 3, 3});
    ASSERT_TRUE(eriExtended.has_value()) << eriExtended.error().message;

    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            for (std::size_t k = 0; k < 3; ++k)
            {
                for (std::size_t l = 0; l < 3; ++l)
                {
                    (*eriExtended)(i, j, k, l) = (*eri)(map[i], map[j], map[k], map[l]);
                }
            }
        }
    }

    const auto extended = qcx::scf::RunRhfScf(*molecule, sExtended, coreExtended, *eriExtended);
    ASSERT_TRUE(extended.has_value()) << extended.error().message;
    EXPECT_EQ(extended->numRemovedOverlapDirections, 1u);
    EXPECT_EQ(extended->density.rows(), 3); // Back-transformed into the FULL AO dimension.
    EXPECT_EQ(extended->density.cols(), 3);

    EXPECT_NEAR(extended->totalEnergy, reference->totalEnergy, 1e-7);

    // THE DENSITY, compared as an OPERATOR and not element by element - and
    // that distinction is a measured lesson from this test's own first run,
    // not a style preference. The first version asserted
    // D_ext(i,j) == D_ref(map(i), map(j)) and FAILED, correctly: with a
    // duplicated function the density MATRIX is NOT unique (any split of the
    // weight between the duplicate pair represents the same operator), so
    // element-wise comparison asserts a property the mathematics does not
    // give. The SCF returns the symmetric split - with d = D_ref(0,0), the
    // block is [[d/4, d/2, d/4], [d/2, d, d/2], [d/4, d/2, d/4]] - and that
    // IS a valid representation of the same operator.
    //
    // The invariant that does hold, and the one this pins: the operator's own
    // matrix elements. With P the density matrix and S the overlap,
    // <phi_i|D|phi_j> = (S P S)_ij, which is independent of how the operator
    // is represented. The reference's two functions ARE the extended basis's
    // first two, so those four elements must agree with the reference's.
    // A wrong back-transform - the reduced-space density reported instead of
    // X P_orth X^T, or the S conjugation dropped - moves them by order 1,
    // five orders above the tolerance.
    const Eigen::MatrixXd operatorRef = s * reference->density * s;
    const Eigen::MatrixXd operatorExt = sExtended * extended->density * sExtended;

    for (Eigen::Index i = 0; i < 2; ++i)
    {
        for (Eigen::Index j = 0; j < 2; ++j)
        {
            EXPECT_NEAR(operatorExt(i, j), operatorRef(i, j), 1e-6)
                << "<phi_" << i << "|D|phi_" << j << ">";
        }
    }

    // The electron count is carried through the removal and back-transform:
    // Tr(P S) = N = 2 on both sides.
    EXPECT_NEAR((extended->density * sExtended).trace(), 2.0, 1e-6);
    EXPECT_NEAR((reference->density * s).trace(), 2.0, 1e-6);
}

TEST(RhfTest, H2Sto3gCarriesTheConvergedState) {
    // The converged-state handoff (HfResult trailing fields): the result
    // must carry the converged density/coefficients/orbital energies, in the
    // documented conventions, with pyscf cross-checks (pyscf 2.14.0,
    // RHF/STO-3G, conv_tol 1e-12, WSL 2026-08-25).
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;
    const Eigen::MatrixXd s = ToMatrix(*overlap);
    const Eigen::MatrixXd h = ToMatrix(*kinetic) + ToMatrix(*nuclear);

    const auto result = qcx::scf::RunRhfScf(*molecule, s, h, *eri);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->converged);

    // D = 2 C_occ C_occ^T with one doubly occupied orbital, and the
    // idempotency-adjacent trace Tr[D S] = N_e = 2 (electron count).
    const Eigen::MatrixXd& density = result->density;
    const Eigen::MatrixXd& coefficients = result->coefficients;
    EXPECT_EQ(density.rows(), 2);
    EXPECT_EQ(coefficients.rows(), 2);
    EXPECT_EQ(coefficients.cols(), 2);
    // Not a convergence gate: the returned density must BE this build of
    // the returned coefficients (a construction identity), so the bound is
    // the roundoff of C C^T, not a stopping rule.
    EXPECT_NEAR(
        (density - 2.0 * coefficients.leftCols(1) * coefficients.leftCols(1).transpose()).norm(),
        0.0,
        1e-10);
    EXPECT_NEAR((density * s).trace(), 2.0, 1e-9);

    // Orbital energies in ascending order, pinned against pyscf's
    // mo_energy [-0.5782029775, 0.6702677683]; the ascending-by-energy
    // column order follows from the pinned values themselves.
    const Eigen::VectorXd& eps = result->orbitalEnergies;
    ASSERT_EQ(eps.size(), 2);
    EXPECT_NEAR(eps(0), -0.5782029775, 1e-7);
    EXPECT_NEAR(eps(1), 0.6702677683, 1e-7);
    EXPECT_LT(eps(0), eps(1));
}

TEST(RhfTest, BudgetExhaustionCarriesTheFinalIterate) {
    // A non-converged result must still carry the last iterate (the
    // converged flag stays the gate): with a one-iteration budget the
    // matrices are non-empty and the energy matches the first-iterate
    // total, not the zeros the default-initialized fields would give.
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    qcx::scf::RhfOptions options;
    options.maxIterations = 1;
    const auto result = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri, options);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_FALSE(result->converged);
    EXPECT_EQ(result->iterations, 1);
    EXPECT_EQ(result->density.rows(), 2);
    EXPECT_EQ(result->coefficients.rows(), 2);
    EXPECT_EQ(result->coefficients.cols(), 2);
    EXPECT_EQ(result->orbitalEnergies.size(), 2);
    // The density of the first iterate is non-trivial: the default GWH
    // start is non-zero to begin with and one iteration moves it
    // further (the result must never fall back to the zero matrix).
    EXPECT_GT(result->density.norm(), 1e-6);
}

TEST(ScfCheckpointTest, HfResultRestartCarriesTheLastIterate) {
    // The restart handoff: a budget-capped run's trailing restart
    // state must carry the last density and the DIIS history, so a resume
    // (initialScfState) continues exactly where the capped run stopped.
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    // The budget is 1: H2/sto-3g converges within 2 iterations in this
    // worktree - DIIS never extrapolates for H2, because the commutator
    // error sits at the kDiisFloorThreshold floor from the first iteration
    // (the symmetric-orbital density makes F D S symmetric to rounding,
    // see rhf.cpp). Under the floor guard the RHF path never appends
    // such pairs, so the capped run leaves an EMPTY history - the shape
    // consistency check below is the discipline's observable, and the
    // density handoff is what the restart state must carry.
    qcx::scf::RhfOptions options;
    options.maxIterations = 1;
    const auto result = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri, options);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_FALSE(result->converged);
    EXPECT_EQ(result->iterations, 1);

    // The restart state is the last iterate: an n x n density that is not
    // the zero default (the GWH start), and a DIIS history kept
    // degenerate-pair-free by the floor guard.
    EXPECT_EQ(result->restart.density.rows(), 2);
    EXPECT_EQ(result->restart.density.cols(), 2);
    EXPECT_GT(result->restart.density.norm(), 1e-6);
    EXPECT_TRUE(result->restart.diis.fockHistory.empty());
    EXPECT_EQ(result->restart.diis.fockHistory.size(), result->restart.diis.errorHistory.size());
}

TEST(ScfCheckpointTest, HfResultRestartRejectsShapeMismatch) {
    // A restart seed whose density does not match the function count must
    // be rejected up front - an unchecked 3x3 seed against the 2-function
    // H2/sto-3g basis would corrupt the iterate loop, not error.
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    qcx::scf::RhfOptions options;
    options.initialScfState.density = Eigen::MatrixXd::Zero(3, 3);
    const auto result = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri, options);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}

} // namespace
