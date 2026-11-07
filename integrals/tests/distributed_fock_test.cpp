// The distributed Fock build (fock_build.hpp's executionContext parameter):
//   - a size-one context runs the serial path bit for bit (the identity a
//     serial build and a one-rank launch depend on),
//   - two and four ranks agree with the single-process builder: the fp64
//     build at 1e-12, the certified fp32 lane within the composition of the
//     two runs' certified bound sums, and the C2v strong form at 1e-12 with
//     its closure record intact and the contraction census split across the
//     ranks,
//   - a call whose screened work cannot fill the ranks refuses by name with
//     the identical message on every rank,
//   - the LightPath and the precision ladder refuse under distribution by
//     name (the two documented composition decisions).
//
// ctest runs this binary as one process, where the distributed cases skip.
// The CMake target registers mpiexec -n 2 and -n 4 runs of the
// DistributedFockTest.* filter, which is where they execute.
//
// The molecule and helper definitions mirror fock_build_test.cpp (the shared
// proof shapes; the fixtures header cannot carry them because they need the
// integrals engines and the reductions).

#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/backend/execution_context.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/precision_policy.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>
#include <string>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
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

// A symmetric density with |D| <= 0.75 and a diagonal near 0.5 - the
// magnitudes the screening and certified-lane gates expect.
Eigen::MatrixXd PhysicalDensity(std::size_t n) {
    std::mt19937_64 rng(20260817);
    std::uniform_real_distribution<double> dist(-0.25, 0.25);
    Eigen::MatrixXd d(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = (i == j) ? 0.5 + dist(rng) : dist(rng);
            d(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
            d(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = value;
        }
    }

    return d;
}

// D_sym = (1/|G|) sum_g A(g) D A(g)^T - the input the strong form's
// precondition requires (the fock_build_test.cpp definition).
Eigen::MatrixXd SymmetrizeUnderReduction(const Eigen::MatrixXd& density,
                                         const qcx::integrals::SymmetryReduction& reduction) {
    const Eigen::Index n = density.rows();
    const std::size_t order = reduction.groupOrder;
    Eigen::MatrixXd averaged = Eigen::MatrixXd::Zero(n, n);

    for (std::size_t g = 0; g < order; ++g)
    {
        for (Eigen::Index i = 0; i < n; ++i)
        {
            const auto indexI = static_cast<std::size_t>(i);
            const auto imageI = static_cast<Eigen::Index>(reduction.permutation[g][indexI]);
            const double signI = static_cast<double>(reduction.sign[g][indexI]);

            for (Eigen::Index j = 0; j < n; ++j)
            {
                const auto indexJ = static_cast<std::size_t>(j);
                const auto imageJ = static_cast<Eigen::Index>(reduction.permutation[g][indexJ]);
                const double signJ = static_cast<double>(reduction.sign[g][indexJ]);
                averaged(i, j) += signI * signJ * density(imageI, imageJ);
            }
        }
    }

    averaged /= static_cast<double>(order);
    return averaged;
}

// max |a - b| with the NaN-propagating form the class-path pins use.
double MaxAbsoluteDelta(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    double maxDelta = 0.0;

    for (Eigen::Index i = 0; i < a.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < a.cols(); ++j)
        {
            const double delta = std::abs(a(i, j) - b(i, j));

            if (!(delta <= maxDelta))
            {
                maxDelta = delta;
            }
        }
    }

    return maxDelta;
}

// True when two same-shaped matrices agree bit for bit.
bool BitIdenticalMatrices(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        return false;
    }

    // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison)
    return std::memcmp(a.data(), b.data(), static_cast<std::size_t>(a.size()) * sizeof(double)) ==
           0;
}

// The water C2v reduction over the real pair-list function order (the
// fock_build_test.cpp fixture: identity, sigma_z, sigma_x, C2).
qcx::integrals::SymmetryReduction MakeWaterC2vReductionHFirst() {
    qcx::integrals::SymmetryReduction reduction;
    reduction.groupOrder = 4;
    reduction.isTrivial = false;
    reduction.permutation = {
        {0, 1, 2, 3, 4, 5, 6}, // I.
        {0, 1, 2, 3, 4, 5, 6}, // sigma_z: identity permutation.
        {1, 0, 2, 3, 4, 5, 6}, // sigma_x: swaps the two H functions.
        {1, 0, 2, 3, 4, 5, 6}, // C2: swaps the two H functions.
    };
    reduction.sign = {
        {1, 1, 1, 1, 1, 1, 1}, // I.
        {1, 1, 1, 1, 1, -1, 1}, // sigma_z: p_z -> -p_z.
        {1, 1, 1, 1, 1, 1, -1}, // sigma_x: p_x -> -p_x.
        {1, 1, 1, 1, 1, -1, -1}, // C2: p_z and p_x flip.
    };
    return reduction;
}

// The process's context, or nullptr after a failed expectation (the test then
// fails; the pointer stays usable in the ASSERT-free tail).
std::shared_ptr<qcx::backend::ExecutionContext> RequireContext() {
    auto context = qcx::backend::ExecutionContext::Create();

    if (!context.has_value())
    {
        ADD_FAILURE() << context.error().message;
        return nullptr;
    }

    return *context;
}

// The distributed cases need at least two ranks; a plain ctest run skips them
// (all ranks see the same size, so the skip is collective).
bool HaveDistribution(const qcx::backend::ExecutionContext& context) {
    return context.Size() >= 2;
}

} // namespace

// A size-one context is the serial path, bit for bit - including the certified
// bound sum and the strong form's record. This is the no-regression identity
// every serial build and one-rank launch relies on.
TEST(DistributedFockTest, SizeOneContextIsTheSerialPathBitForBit) {
    const std::shared_ptr<qcx::backend::ExecutionContext> context = RequireContext();
    ASSERT_NE(context, nullptr);

    if (context->Size() != 1)
    {
        GTEST_SKIP() << "the identity contract is a size-one property; this launch has "
                     << context->Size() << " ranks";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    const qcx::integrals::SymmetryReduction reduction = MakeWaterC2vReductionHFirst();
    const Eigen::MatrixXd symmetrized = SymmetrizeUnderReduction(density, reduction);
    auto symmetrizedTensor = ToTensor(symmetrized);
    ASSERT_TRUE(symmetrizedTensor.has_value()) << symmetrizedTensor.error().message;

    qcx::integrals::FockBuildOptions mixedOptions;
    mixedOptions.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    mixedOptions.useCertifiedMixedPrecision = true;
    auto mixedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, mixedOptions);
    ASSERT_TRUE(mixedBuilder.has_value()) << mixedBuilder.error().message;

    qcx::integrals::FockBuildOptions strongOptions;
    strongOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    strongOptions.useCertifiedMixedPrecision = false;
    strongOptions.symmetryReduction = &reduction;
    strongOptions.symmetryStrongForm = true;
    auto strongBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, strongOptions);
    ASSERT_TRUE(strongBuilder.has_value()) << strongBuilder.error().message;

    // The plain certified-lane build: the same Fock and the same bound sum.
    double serialBound = -1.0;
    auto serialMixed = mixedBuilder->BuildFock(*densityTensor, &serialBound);
    ASSERT_TRUE(serialMixed.has_value()) << serialMixed.error().message;
    double contextualBound = -1.0;
    auto contextualMixed = mixedBuilder->BuildFock(
        *densityTensor, &contextualBound, nullptr, nullptr, nullptr, context.get());
    ASSERT_TRUE(contextualMixed.has_value()) << contextualMixed.error().message;

    EXPECT_TRUE(BitIdenticalMatrices(ToMatrix(*serialMixed), ToMatrix(*contextualMixed)));
    EXPECT_EQ(contextualBound, serialBound);

    // The strong-form class build: the same Fock and the same record.
    qcx::integrals::FockBuildStats serialStats;
    qcx::integrals::FockBuildStats contextualStats;
    auto serialStrong = strongBuilder->BuildFock(*symmetrizedTensor, nullptr, &serialStats);
    ASSERT_TRUE(serialStrong.has_value()) << serialStrong.error().message;
    auto contextualStrong = strongBuilder->BuildFock(
        *symmetrizedTensor, nullptr, &contextualStats, nullptr, nullptr, context.get());
    ASSERT_TRUE(contextualStrong.has_value()) << contextualStrong.error().message;

    EXPECT_TRUE(BitIdenticalMatrices(ToMatrix(*serialStrong), ToMatrix(*contextualStrong)));
    EXPECT_EQ(contextualStats.classBlockContractCalls, serialStats.classBlockContractCalls);
    EXPECT_EQ(contextualStats.classOrbitTaskCount, serialStats.classOrbitTaskCount);
    EXPECT_EQ(contextualStats.strongFormFallbacks, serialStats.strongFormFallbacks);
    EXPECT_EQ(contextualStats.strongFormClosureMisses, serialStats.strongFormClosureMisses);

    // The strong form actually ran (the fixture is eligible and closed), so
    // the identity above is a statement about the strong path.
    EXPECT_EQ(serialStats.strongFormFallbacks, 0u);
    EXPECT_EQ(serialStats.strongFormClosureMisses, 0u);
    EXPECT_GT(serialStats.classBlockContractCalls, 0u);
}

// The plain fp64 build: the distributed matrix equals the single-process one
// at the class-path pins' 1e-12 (the partition changes summation order only).
TEST(DistributedFockTest, DistributedFp64MatchesTheSerialBuilder) {
    const std::shared_ptr<qcx::backend::ExecutionContext> context = RequireContext();
    ASSERT_NE(context, nullptr);

    if (!HaveDistribution(*context))
    {
        GTEST_SKIP() << "a plain ctest run is one process; the mpiexec entries distribute";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    options.useCertifiedMixedPrecision = false;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    auto serial = builder->BuildFock(*densityTensor);
    ASSERT_TRUE(serial.has_value()) << serial.error().message;
    auto distributed =
        builder->BuildFock(*densityTensor, nullptr, nullptr, nullptr, nullptr, context.get());
    ASSERT_TRUE(distributed.has_value()) << distributed.error().message;

    const double delta = MaxAbsoluteDelta(ToMatrix(*distributed), ToMatrix(*serial));
    // The figures this run's report quotes.
    std::cout << "rank " << context->Rank() << "/" << context->Size()
              << ": H2O/STO-3G fp64 distributed-vs-serial max|dF| = " << delta
              << " (tolerance 1e-12)" << std::endl;
    EXPECT_LE(delta, 1e-12);
}

// The certified fp32 lane: the distributed build routes the same quartets
// (the screening is replicated), so its bound sum equals the serial one, and
// its Fock sits inside the composition of the two runs' certified bounds of
// the fp64 result - the acceptance tolerance.
TEST(DistributedFockTest, DistributedCertifiedLaneAgreesWithinTheCertifiedBound) {
    const std::shared_ptr<qcx::backend::ExecutionContext> context = RequireContext();
    ASSERT_NE(context, nullptr);

    if (!HaveDistribution(*context))
    {
        GTEST_SKIP() << "a plain ctest run is one process; the mpiexec entries distribute";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    qcx::integrals::FockBuildOptions mixedOptions;
    mixedOptions.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    mixedOptions.useCertifiedMixedPrecision = true;
    auto mixedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, mixedOptions);
    ASSERT_TRUE(mixedBuilder.has_value()) << mixedBuilder.error().message;

    qcx::integrals::FockBuildOptions fp64Options;
    fp64Options.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    fp64Options.useCertifiedMixedPrecision = false;
    auto fp64Builder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, fp64Options);
    ASSERT_TRUE(fp64Builder.has_value()) << fp64Builder.error().message;

    double serialBound = -1.0;
    auto serial = mixedBuilder->BuildFock(*densityTensor, &serialBound);
    ASSERT_TRUE(serial.has_value()) << serial.error().message;
    double distributedBound = -1.0;
    auto distributed = mixedBuilder->BuildFock(
        *densityTensor, &distributedBound, nullptr, nullptr, nullptr, context.get());
    ASSERT_TRUE(distributed.has_value()) << distributed.error().message;
    auto fp64 = fp64Builder->BuildFock(*densityTensor);
    ASSERT_TRUE(fp64.has_value()) << fp64.error().message;

    // The lane must actually route on this fixture (otherwise the tolerance
    // below is vacuous).
    EXPECT_GT(serialBound, 0.0);
    EXPECT_GT(distributedBound, 0.0);

    // The same member-weighted sum, reduced in a different order.
    EXPECT_NEAR(distributedBound, serialBound, 1e-12);

    const Eigen::MatrixXd serialMatrix = ToMatrix(*serial);
    const Eigen::MatrixXd distributedMatrix = ToMatrix(*distributed);
    const Eigen::MatrixXd fp64Matrix = ToMatrix(*fp64);

    // Distributed vs serial: the composition of the two runs' certified
    // bounds bounds the deviation (1.001 absorbs the composition's own
    // rounding - the fock_build_test.cpp convention).
    const double crossTolerance = (serialBound + distributedBound) * 1.001 + 1e-14;
    const double crossDelta = MaxAbsoluteDelta(distributedMatrix, serialMatrix);
    const double boundDelta = MaxAbsoluteDelta(distributedMatrix, fp64Matrix);
    // The figures this run's report quotes: the bound sums, the bound-sum
    // difference, and both matrix deviations against their tolerances.
    std::cout << "rank " << context->Rank() << "/" << context->Size()
              << ": H2O/STO-3G certified lane serialBound = " << serialBound
              << ", distributedBound = " << distributedBound
              << ", |bound difference| = " << std::abs(distributedBound - serialBound)
              << ", distributed-vs-serial max|dF| = " << crossDelta << " (tolerance "
              << crossTolerance << "), distributed-vs-fp64 max|dF| = " << boundDelta
              << " (tolerance " << distributedBound * 1.001 + 1e-14 << ")" << std::endl;

    EXPECT_LE(crossDelta, crossTolerance);

    // And each lane build sits inside its own certified bound of the fp64
    // reference.
    EXPECT_LE(boundDelta, distributedBound * 1.001 + 1e-14);
}

// The strong form through the class path: the partition is a per-rank slice of
// class tasks (built replicated), the orbit closure stays whole on every rank,
// and the reduced Fock equals the serial one at 1e-12. The contraction census
// proves the share was split: the replicated orbit-task count is the serial
// one, the rank's contract count is a proper share of it.
TEST(DistributedFockTest, DistributedStrongFormMatchesTheSerialBuilder) {
    const std::shared_ptr<qcx::backend::ExecutionContext> context = RequireContext();
    ASSERT_NE(context, nullptr);

    if (!HaveDistribution(*context))
    {
        GTEST_SKIP() << "a plain ctest run is one process; the mpiexec entries distribute";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const qcx::integrals::SymmetryReduction reduction = MakeWaterC2vReductionHFirst();
    const Eigen::MatrixXd density = SymmetrizeUnderReduction(PhysicalDensity(7), reduction);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    options.useCertifiedMixedPrecision = false;
    options.symmetryReduction = &reduction;
    options.symmetryStrongForm = true;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    qcx::integrals::FockBuildStats serialStats;
    auto serial = builder->BuildFock(*densityTensor, nullptr, &serialStats);
    ASSERT_TRUE(serial.has_value()) << serial.error().message;

    // The distributed partition cuts the class-task list, so the ranks can
    // only be filled when the fixture has at least one task per rank.
    ASSERT_GE(serialStats.classOrbitTaskCount, static_cast<std::size_t>(context->Size()))
        << "the fixture has fewer class tasks than ranks; the dispatcher refuses by name";

    qcx::integrals::FockBuildStats distributedStats;
    auto distributed = builder->BuildFock(
        *densityTensor, nullptr, &distributedStats, nullptr, nullptr, context.get());
    ASSERT_TRUE(distributed.has_value()) << distributed.error().message;

    // The strong form ran and stayed closed under distribution.
    EXPECT_EQ(distributedStats.strongFormFallbacks, 0u);
    EXPECT_EQ(distributedStats.strongFormClosureMisses, 0u);

    // The replicated counters carry the full-call values on every rank; the
    // contraction census is this rank's share of the orbit tasks.
    EXPECT_EQ(distributedStats.classOrbitTaskCount, serialStats.classOrbitTaskCount);
    EXPECT_EQ(distributedStats.fp64QuartetCount, serialStats.fp64QuartetCount);
    EXPECT_GT(distributedStats.classBlockContractCalls, 0u);
    EXPECT_LE(distributedStats.classBlockContractCalls, serialStats.classBlockContractCalls);

    const double strongDelta = MaxAbsoluteDelta(ToMatrix(*distributed), ToMatrix(*serial));
    // The figures this run's report quotes: the replicated task count, the
    // rank's contraction share and the matrix deviation.
    std::cout << "rank " << context->Rank() << "/" << context->Size()
              << ": H2O/STO-3G C2v strong form classOrbitTaskCount = "
              << serialStats.classOrbitTaskCount << " (serial) / "
              << distributedStats.classOrbitTaskCount
              << " (distributed), classBlockContractCalls = " << serialStats.classBlockContractCalls
              << " (serial) / " << distributedStats.classBlockContractCalls
              << " (this rank), distributed-vs-serial max|dF| = " << strongDelta
              << " (tolerance 1e-12)" << std::endl;
    EXPECT_LE(strongDelta, 1e-12);
}

// The refusal: a density that screens out every quartet leaves zero work, so
// the dispatcher refuses by name before the call's first collective, with the
// identical message on every rank (every rank checks the full string).
TEST(DistributedFockTest, RefusesByNameWhenNoQuartetFillsTheRanks) {
    const std::shared_ptr<qcx::backend::ExecutionContext> context = RequireContext();
    ASSERT_NE(context, nullptr);

    if (!HaveDistribution(*context))
    {
        GTEST_SKIP() << "a plain ctest run is one process; the mpiexec entries distribute";
    }

    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    // The near-zero density of the screening pin: both gates drop every
    // quartet (F = H to 1e-15 serially).
    Eigen::MatrixXd tiny(2, 2);
    tiny << 1.1e-11, 1e-12, 1e-12, 1.1e-11;
    auto densityTensor = ToTensor(tiny);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    options.useCertifiedMixedPrecision = false;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    auto serial = builder->BuildFock(*densityTensor);
    ASSERT_TRUE(serial.has_value()) << serial.error().message;
    EXPECT_LE(MaxAbsoluteDelta(ToMatrix(*serial), ToMatrix(*core)), 1e-15);

    auto distributed =
        builder->BuildFock(*densityTensor, nullptr, nullptr, nullptr, nullptr, context.get());
    ASSERT_FALSE(distributed.has_value());

    EXPECT_EQ(distributed.error().code, qcx::ErrorCode::kInvalidArgument);
    // The full expected text: the count is replicated, so every rank must
    // produce exactly this string.
    const std::string expected = "StaticDispatcher: 0 items cannot fill " +
                                 std::to_string(context->Size()) +
                                 " ranks; every rank must receive at least one item";
    EXPECT_EQ(distributed.error().message, expected);
}

// The LightPath refuses under distribution by name (the documented
// composition decision): its chunk pair data is built per rank in full, so
// distributing the contraction would not lower the memory peak that gives the
// path its purpose.
TEST(DistributedFockTest, LightPathRefusesByNameUnderDistribution) {
    const std::shared_ptr<qcx::backend::ExecutionContext> context = RequireContext();
    ASSERT_NE(context, nullptr);

    if (!HaveDistribution(*context))
    {
        GTEST_SKIP() << "a plain ctest run is one process; the mpiexec entries distribute";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    // The knob needs a workspace budget to reach the mode decision (the
    // decision block runs only with a budget); this is the same 64 MiB band
    // the serial LightPath pins use on this fixture.
    auto budget = qcx::memory::WorkspaceBudget::Create(std::size_t{64} * 1024 * 1024);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    options.useCertifiedMixedPrecision = false;
    options.maxParallelChunks = 1;
    options.workspaceBudget = &*budget;
    options.lightPathChunkPairs = 2; // Forced LightPath mode.
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    // The knob really engaged the chunk loop; a demoted mode would make the
    // refusal assertion below pass for the wrong reason.
    ASSERT_TRUE(builder->ModeInfo().has_value());
    EXPECT_EQ(builder->ModeInfo()->mode, qcx::integrals::FockBuildMode::kLightPath);

    auto distributed =
        builder->BuildFock(*densityTensor, nullptr, nullptr, nullptr, nullptr, context.get());
    ASSERT_FALSE(distributed.has_value());
    EXPECT_EQ(distributed.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(distributed.error().message.find("LightPath"), std::string::npos);
}

// The precision ladder refuses under distribution by name (the second
// documented composition decision): its per-band certified sums and committed
// coarse sum are a per-call contract.
TEST(DistributedFockTest, PrecisionLadderRefusesByNameUnderDistribution) {
    const std::shared_ptr<qcx::backend::ExecutionContext> context = RequireContext();
    ASSERT_NE(context, nullptr);

    if (!HaveDistribution(*context))
    {
        GTEST_SKIP() << "a plain ctest run is one process; the mpiexec entries distribute";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    options.useCertifiedMixedPrecision = false;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    qcx::integrals::PrecisionBandProfile profile;
    qcx::integrals::EscalationSchedule schedule(
        qcx::integrals::AccuracyPreset::kNormal, profile, qcx::integrals::PrecisionCap::kAuto);
    qcx::integrals::PrecisionLadderInputs ladder;
    ladder.schedule = &schedule;

    auto distributed =
        builder->BuildFock(*densityTensor, nullptr, nullptr, &ladder, nullptr, context.get());
    ASSERT_FALSE(distributed.has_value());
    EXPECT_EQ(distributed.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(distributed.error().message.find("precision ladder"), std::string::npos);
}
