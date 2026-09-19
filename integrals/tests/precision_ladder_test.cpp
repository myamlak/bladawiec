// The precision-ladder builder tests (Package B steps 3-6 gates): the C12
// classification covers every screened batch with the fp64 band non-empty
// and the fp16 band empty at kTight; the certified-lane pins hold with the
// dispatch dimension disabled (the fp64 cap reproduces the pre-ladder fp64
// build bit-for-bit); the scripted SCF runs the escalation schedule -
// monotone committed sums, the referee firing on the prescribed
// iterations and at the final, and the referee checks passing. The
// partition-level tests drive internal/precision_ladder.hpp directly (the
// screening_test.cpp convention); the schedule-level tests drive the real
// DirectJkFockBuilder.

#include "alkane_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "internal/fock_screen.hpp"
#include "internal/md_batch.hpp"
#include "internal/precision_ladder.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/sparsity_pattern.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::integrals::AccuracyPreset;
using qcx::integrals::EscalationSchedule;
using qcx::integrals::FockBuildOptions;
using qcx::integrals::PrecisionBand;
using qcx::integrals::PrecisionBandProfile;
using qcx::integrals::PrecisionCap;
using qcx::integrals::PrecisionLadderInputs;
using qcx::integrals::RefereeErrorCheck;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::MakeAlkaneSto3gBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;

// A physical-density-shaped symmetric matrix (the fock_build_test.cpp
// helper, duplicated test-locally): 0.5 on the diagonal (D ~ N/2 in the
// spatial density), U(-0.25, 0.25) deviations everywhere.
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

// H = T + V from the one-electron engines (the direct_rhf_test.cpp
// helper, test-local).
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

TEST(PrecisionLadderTest, C12ClassificationCoversAllBatches) {
    // The step-3 gate on the C12 alkane fixture: the classification
    // covers every screened batch, the fp64 band is non-empty at kTight
    // and the fp16 band is empty there (the preset gate forces it), and
    // the ratchet moves batches only up the ladder.
    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto pairStore = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);
    ASSERT_TRUE(pairStore.has_value()) << pairStore.error().message;
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;

    FockBuildOptions options;
    options.accuracy = AccuracyPreset::kNormal;
    options.maxParallelChunks = 1; // Serial: canonical order, exact decisions.

    std::vector<std::size_t> rowOffsets;
    std::vector<std::size_t> neighborIndices;
    qcx::integrals::internal::BuildNeighborList(
        *pairList, *schwarz, options.accuracy, rowOffsets, neighborIndices);

    auto pattern =
        qcx::memory::BuildSparsityFromAdjacency<qcx::backend::CpuTag>(rowOffsets, neighborIndices);
    ASSERT_TRUE(pattern.has_value()) << pattern.error().message;

    const std::size_t n = pairList->functionCount;
    const Eigen::MatrixXd density = PhysicalDensity(n);
    const std::vector<double> pairMaxDensity =
        qcx::integrals::internal::BuildShellPairMaxDensity(density, *pairList);

    const qcx::integrals::internal::ScreeningContext context{
        *pairList, *schwarz, *pairStore, options, &pairMaxDensity};
    const double densityThreshold = qcx::integrals::DensityThreshold(options.accuracy);
    const double mixedThreshold = qcx::integrals::MixedPrecisionThreshold(options.accuracy);

    // The ladder's screening shape: the certified routing off, every kept
    // quartet in one list for the partition to classify.
    std::vector<qcx::integrals::internal::MdQuartetTask> kept;
    std::vector<qcx::integrals::internal::MdQuartetTask> unused32;
    std::vector<double> unusedWeights;
    qcx::integrals::internal::ScreenAll(context,
                                        density,
                                        *pattern,
                                        densityThreshold,
                                        mixedThreshold,
                                        false,
                                        kept,
                                        unused32,
                                        unusedWeights);
    ASSERT_FALSE(kept.empty());

    const double bScreen =
        qcx::integrals::internal::ScreenBoundSum(context, density, *pattern, densityThreshold);

    PrecisionBandProfile profile;
    profile.fp16TensorCores = true;

    // kTight: the preset gate keeps the coarse bands off, so everything
    // routes fp64 - the fp64 band non-empty, the fp16 (and every coarse)
    // band empty (bounds force it: the kTight strict-pins contract).
    EscalationSchedule tight(AccuracyPreset::kTight, profile, PrecisionCap::kAuto);
    const double tightBudget = tight.ClassificationBudget({bScreen, 0.0, 0.0});
    qcx::integrals::internal::LadderPartition tightPartition =
        qcx::integrals::internal::PartitionBatches(context, density, kept, tightBudget, tight);
    EXPECT_DOUBLE_EQ(tightBudget, 0.0);
    EXPECT_EQ(tightPartition.fp64.size(), kept.size());
    EXPECT_GT(tightPartition.fp64.size(), 0u);
    EXPECT_TRUE(tightPartition.fp16.empty());
    EXPECT_TRUE(tightPartition.fp32Certified.empty());
    EXPECT_TRUE(tightPartition.fp32Mixed.empty());

    // kNormal with the coarse bands available: the classification covers
    // every kept quartet, and the ratchet across a shrinking budget only
    // moves batches up the ladder (the coarse-quartet count never grows).
    EscalationSchedule schedule(AccuracyPreset::kNormal, profile, PrecisionCap::kAuto);
    const double largeBudget = schedule.ClassificationBudget({bScreen, 0.0, 1e-2});
    qcx::integrals::internal::LadderPartition large =
        qcx::integrals::internal::PartitionBatches(context, density, kept, largeBudget, schedule);
    const std::size_t largeCoverage =
        large.fp64.size() + large.fp16.size() + large.fp32Certified.size() + large.fp32Mixed.size();
    EXPECT_EQ(largeCoverage, kept.size());

    const double smallBudget = schedule.ClassificationBudget({bScreen, 0.0, 1e-8});
    ASSERT_LE(smallBudget, largeBudget);
    qcx::integrals::internal::LadderPartition small =
        qcx::integrals::internal::PartitionBatches(context, density, kept, smallBudget, schedule);
    const std::size_t smallCoverage =
        small.fp64.size() + small.fp16.size() + small.fp32Certified.size() + small.fp32Mixed.size();
    EXPECT_EQ(smallCoverage, kept.size());

    const std::size_t largeCoarse =
        large.fp16.size() + large.fp32Certified.size() + large.fp32Mixed.size();
    const std::size_t smallCoarse =
        small.fp16.size() + small.fp32Certified.size() + small.fp32Mixed.size();
    EXPECT_LE(smallCoarse, largeCoarse);
}

TEST(PrecisionLadderTest, Fp64CapMatchesPlainFp64Build) {
    // The step-6 gate (CPU half): the ladder with the fp64 cap reproduces
    // the pre-ladder fp64 build bit-for-bit - the certified-lane pins
    // hold with the dispatch dimension engaged and clamped.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    // Both builds share the screening shape (the per-element flag on, the
    // certified lane off - the ladder's own screening shape), so the kept
    // lists and the gate are identical.
    FockBuildOptions options;
    options.accuracy = AccuracyPreset::kNormal;
    options.useCertifiedMixedPrecision = false;
    options.usePerElementScreening = true;
    options.maxParallelChunks = 1; // Serial: bit-identity.

    auto plain = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    auto ladder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(ladder.has_value()) << ladder.error().message;

    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd density = PhysicalDensity(n);
    const auto densityTensor = qcx::testing::ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    double plainSum = -1.0;
    auto plainFock = plain->BuildFock(*densityTensor, &plainSum);
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;

    PrecisionBandProfile profile; // No tensor cores: the CPU profile.
    EscalationSchedule schedule(AccuracyPreset::kNormal, profile, PrecisionCap::kFp64);
    const PrecisionLadderInputs inputs{0.0, 0.0, &schedule};

    double ladderSum = -1.0;
    auto ladderFock = ladder->BuildFock(*densityTensor, &ladderSum, nullptr, &inputs);
    ASSERT_TRUE(ladderFock.has_value()) << ladderFock.error().message;

    const Eigen::MatrixXd plainMatrix = ToMatrix(*plainFock);
    const Eigen::MatrixXd ladderMatrix = ToMatrix(*ladderFock);
    EXPECT_TRUE(plainMatrix == ladderMatrix)
        << "the fp64 cap must reproduce the pre-ladder fp64 build bit-for-bit";
    EXPECT_DOUBLE_EQ(plainSum, 0.0); // The certified lane off: no bound sum.
    EXPECT_DOUBLE_EQ(ladderSum, 0.0);
}

TEST(PrecisionLadderTest, ScriptedScfRunsTheSchedule) {
    // The step-4 gate: a scripted SCF runs the escalation schedule - the
    // committed sums shrink monotonically as the density error shrinks,
    // the referee fires on the prescribed iterations (every 5 at kNormal)
    // and at the final, the referee's delivered-error check holds against
    // the fp64 reference build, and the final delivery contract holds.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    FockBuildOptions options;
    options.accuracy = AccuracyPreset::kNormal;
    options.usePerElementScreening = true;
    options.maxParallelChunks = 1; // Serial: exact decisions.

    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    FockBuildOptions refereeOptions = options;
    refereeOptions.useCertifiedMixedPrecision = false;
    auto refereeBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, refereeOptions);
    ASSERT_TRUE(refereeBuilder.has_value()) << refereeBuilder.error().message;

    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd density = PhysicalDensity(n);
    const auto densityTensor = qcx::testing::ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    PrecisionBandProfile profile; // CPU profile: no fp16 band.
    EscalationSchedule schedule(AccuracyPreset::kNormal, profile, PrecisionCap::kAuto);

    double previousCommitted = std::numeric_limits<double>::infinity();
    double finalKernelSum = 0.0;
    Eigen::MatrixXd finalLadderFock;

    constexpr int kIterations = 10;

    for (int iteration = 1; iteration <= kIterations; ++iteration)
    {
        // The scripted density-convergence error: monotone downward (the
        // SCF's rms density change; the density itself stays the fixture
        // density - the escalation consumes the error, not the wiggle).
        const double densityError = 1e-2 / static_cast<double>(1 << iteration);
        const PrecisionLadderInputs inputs{densityError, 0.0, &schedule};

        double kernelSum = 0.0;
        auto fock = builder->BuildFock(*densityTensor, &kernelSum, nullptr, &inputs);
        ASSERT_TRUE(fock.has_value()) << fock.error().message;
        const Eigen::MatrixXd fockMatrix = ToMatrix(*fock);

        // Monotone precision switches: the committed coarse-work sum never
        // grows across iterations (the budget shrinks, the ratchet only
        // moves batches up the ladder).
        const double committed = schedule.LastCommittedSum();
        EXPECT_LE(committed, previousCommitted * (1.0 + 1e-12));
        previousCommitted = committed;

        // The referee fires every 5 iterations plus the final one.
        EXPECT_EQ(schedule.RefereeDue(iteration, iteration == kIterations),
                  iteration % 5 == 0 || iteration == kIterations);

        if (schedule.RefereeDue(iteration, iteration == kIterations))
        {
            // The fp64 reference build over the same density and the
            // delivered-error check: the ladder Fock's deviation from the
            // reference must stay inside the committed kernel-bound sum.
            double refereeSum = 0.0;
            auto refereeFock = refereeBuilder->BuildFock(*densityTensor, &refereeSum);
            ASSERT_TRUE(refereeFock.has_value()) << refereeFock.error().message;
            const Eigen::MatrixXd delta = fockMatrix - ToMatrix(*refereeFock);
            const double maxAbsDelta = delta.cwiseAbs().maxCoeff();
            auto errorCheck = RefereeErrorCheck(maxAbsDelta, kernelSum);

            if (!errorCheck.has_value())
            {
                FAIL() << "referee error check failed at iteration " << iteration << ": "
                       << errorCheck.error().message << " (delta " << maxAbsDelta << " vs "
                       << kernelSum << ")";
            }

            if (iteration == kIterations)
            {
                // The final delivery contract: the committed a-priori sum
                // fits the density-free delivery remainder.
                const qcx::integrals::EriBudgetInputs finalInputs{0.0, 0.0, densityError};
                auto delivery = schedule.RefereeDelivery(schedule.LastCommittedSum(), finalInputs);

                if (!delivery.has_value())
                {
                    FAIL() << "final delivery check failed: " << delivery.error().message;
                }

                finalKernelSum = kernelSum;
                finalLadderFock = fockMatrix;
            }
        }
    }

    // The final iteration's committed sum sits inside the delivery
    // remainder with the density error gone.
    EXPECT_LE(schedule.LastCommittedSum(), 9.0e-11 * (1.0 + 1e-12));
    EXPECT_GE(finalKernelSum, 0.0);
    EXPECT_GT(finalLadderFock.size(), 0);
}

TEST(PrecisionLadderTest, NullScheduleKeepsThePreLadderBuild) {
    // A null schedule is the ladder-off path: no error, the two-pass
    // build, the certified-lane pins untouched (the v1 engagement rule).
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    FockBuildOptions options;
    options.maxParallelChunks = 1;

    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const std::size_t n = core->Shape()[0];
    const auto densityTensor = qcx::testing::ToTensor(PhysicalDensity(n));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    const PrecisionLadderInputs noSchedule{1e-4, 0.0, nullptr};
    auto pass = builder->BuildFock(*densityTensor, nullptr, nullptr, &noSchedule);
    ASSERT_TRUE(pass.has_value()) << pass.error().message;
}

} // namespace
