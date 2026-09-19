// Direct J/K Fock builder tests (fock_build.hpp):
//   - the contraction identity F = H + 2J - K against the dense supermatrix
//     (both fp64, unscreened) at 1e-12 on the pinned molecules,
//   - the density-weighted screening gate: a near-zero density drops every
//     quartet (F = H exactly) while the physical density drops none on the
//     STO-3G fixtures (the pair-list correctness check),
//   - the certified fp32 lane: the delivered bound sum dominates the
//     actual element-wise fp64-vs-fp32 deviation,
//   - the preset mapping and the error paths.
//
// The molecules are tiny (2-7 functions) so the dense tensors are trivial;
// the HF/H2O walks are fast-mode-gated like the other engine tests.

#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "hf_sto3g.hpp"
#include "internal/fock_screen.hpp"
#include "internal/footprint.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "qcx/memory/allocation_instrument.hpp"
#include "qcx/memory/workspace_budget.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <string>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeHfSto3g;
using qcx::testing::MakeHfSto3gBasis;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// H = T + V from the one-electron engines.
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

// F = H + 2J - K contracted serially from the dense (uv|ws) tensor - the
// independent reference. K uses (i k | l j) = eri(i, k, l, j).
Eigen::MatrixXd ReferenceFock(const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
                              // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (h, d) order.
                              const Eigen::MatrixXd& h,
                              const Eigen::MatrixXd& d) {
    const Eigen::Index n = h.rows();
    Eigen::MatrixXd fock = h;

    for (Eigen::Index i = 0; i < n; ++i)
    {
        for (Eigen::Index j = 0; j < n; ++j)
        {
            double jContraction = 0.0;
            double kContraction = 0.0;

            for (Eigen::Index k = 0; k < n; ++k)
            {
                for (Eigen::Index l = 0; l < n; ++l)
                {
                    jContraction += eri(i, j, k, l) * d(k, l);
                    kContraction += eri(i, k, l, j) * d(k, l);
                }
            }

            fock(i, j) += 2.0 * jContraction - kContraction;
        }
    }

    return fock;
}

// A symmetric density with |D| <= 0.75 and a diagonal near 0.5 - the
// magnitudes the screening and certified-bound gates expect.
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

void CheckDirectMatchesDense(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& density,
    double tolerance,
    qcx::integrals::AccuracyPreset preset = qcx::integrals::AccuracyPreset::kNormal) {
    auto core = BuildCoreHamiltonian(molecule, basisSet);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::EriDenseOptions eriOptions;
    eriOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    eriOptions.screen = false;
    auto eri = qcx::integrals::BuildEriTensorGeneral(molecule, basisSet, eriOptions);
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    const Eigen::MatrixXd reference = ReferenceFock(*eri, ToMatrix(*core), density);

    qcx::integrals::FockBuildOptions options;
    options.accuracy = preset;
    options.useDensityScreening = false;
    options.useCertifiedMixedPrecision = false;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    auto fock = builder->BuildFock(*densityTensor);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;

    const Eigen::MatrixXd direct = ToMatrix(*fock);

    for (std::size_t i = 0; i < static_cast<std::size_t>(density.rows()); ++i)
    {
        for (std::size_t j = 0; j < static_cast<std::size_t>(density.cols()); ++j)
        {
            EXPECT_NEAR(direct(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        reference(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        tolerance)
                << "element (" << i << "," << j << ")";
        }
    }
}

TEST(FockBuildTest, DirectMatchesDenseH2) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    CheckDirectMatchesDense(*molecule, *basis, PhysicalDensity(2), 1e-12);
}

TEST(FockBuildTest, DirectMatchesDenseHf) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeHfSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeHfSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    CheckDirectMatchesDense(*molecule, *basis, PhysicalDensity(6), 1e-12);
}

TEST(DirectJkFockBuilderTest, NeighborListMatchesBruteForceSchwarzScan) {
    // The cached neighbor list is the ONLY driver of
    // BuildFock's screening loop, so an unscreened build must match the
    // dense contraction exactly at every preset: a wrong list (an off-by-
    // one in the row offsets, a wrong cutoff) would drop a significant
    // quartet and show up as a deviation. The list's internals are private
    // State, so the observable contract - every pair above the cutoff is
    // evaluated - is what this pins; the existing FockBuildTest pins
    // complement it at kNormal. The dense reference is the brute-force
    // scan: no screening at all.
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(7);

    CheckDirectMatchesDense(
        *molecule, *basis, density, 1e-12, qcx::integrals::AccuracyPreset::kLoose);
    CheckDirectMatchesDense(
        *molecule, *basis, density, 1e-12, qcx::integrals::AccuracyPreset::kTight);
}

TEST(FockBuildTest, DirectMatchesDenseH2o) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    CheckDirectMatchesDense(*molecule, *basis, PhysicalDensity(7), 1e-12);
}

TEST(FockBuildTest, DensityScreeningDropsNegligibleQuartets) {
    // A near-zero density: m = 1.1e-11 < 1, so BOTH gates drop every quartet
    // here - the six-block test dMax * Q * Q ~ 3e-11 < the kLoose threshold
    // 1e-10, and the product gate's values are smaller still (the
    // J-mode product m^2 * Q * Q ~ 1e-22 * Q^2, the K-mode cross products
    // likewise; the product gate is provably tighter at m < 1:
    // expectations re-derived from the new gate's definition, not tuned).
    // F = H exactly (the positive exercise of the drop path - H2 is
    // instant).
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    Eigen::MatrixXd tiny(2, 2);
    tiny << 1.1e-11, 1e-12, 1e-12, 1.1e-11;

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    options.useDensityScreening = true;
    options.useCertifiedMixedPrecision = false;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    auto densityTensor = ToTensor(tiny);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    auto fock = builder->BuildFock(*densityTensor);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            EXPECT_NEAR((*fock)(i, j), (*core)(i, j), 1e-15) << "element (" << i << "," << j << ")";
        }
    }
}

TEST(FockBuildTest, DensityScreeningKeepsThePhysicalPairs) {
    // The physical density is far above the kLoose density threshold on the
    // H2O/STO-3G fixture: nothing is dropped, so the screened and
    // unscreened fp64 paths must agree to machine precision (a dropped
    // significant quartet shows up as a large deviation). This now runs
    // with the per-element screening on (the default): the
    // product gate keeps the same quartets here (all six block maxima are
    // O(0.25)-scale, so the mode products ~ d^2 * Q * Q stay far above
    // 1e-10), and the per-element filter drops nothing (tau_T =
    // 1e-10/max|D_T| is far below every STO-3G integral element) - the
    // TRAP-7 re-derivation holds both gates to the same keep partition on
    // this fixture.
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
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

    qcx::integrals::FockBuildOptions screened;
    screened.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    screened.useDensityScreening = true;
    screened.useCertifiedMixedPrecision = false;
    auto screenedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, screened);
    ASSERT_TRUE(screenedBuilder.has_value()) << screenedBuilder.error().message;
    auto screenedFock = screenedBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(screenedFock.has_value()) << screenedFock.error().message;

    qcx::integrals::FockBuildOptions unscreened;
    unscreened.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    unscreened.useDensityScreening = false;
    unscreened.useCertifiedMixedPrecision = false;
    auto unscreenedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, unscreened);
    ASSERT_TRUE(unscreenedBuilder.has_value()) << unscreenedBuilder.error().message;
    auto unscreenedFock = unscreenedBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(unscreenedFock.has_value()) << unscreenedFock.error().message;

    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j < 7; ++j)
        {
            EXPECT_NEAR((*screenedFock)(i, j), (*unscreenedFock)(i, j), 1e-12)
                << "element (" << i << "," << j << ")";
        }
    }
}

TEST(FockBuildTest, CertifiedFp32LaneBoundDominates) {
    // With the kLoose preset the certified fp32 lane is active; the
    // delivered bound sum must be positive and dominate the actual
    // element-wise deviation from the fp64-only Fock matrix.
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
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

    qcx::integrals::FockBuildOptions mixed;
    mixed.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    mixed.useDensityScreening = true;
    mixed.useCertifiedMixedPrecision = true;
    auto mixedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, mixed);
    ASSERT_TRUE(mixedBuilder.has_value()) << mixedBuilder.error().message;
    double boundSum = -1.0;
    auto mixedFock = mixedBuilder->BuildFock(*densityTensor, &boundSum);
    ASSERT_TRUE(mixedFock.has_value()) << mixedFock.error().message;
    EXPECT_GT(boundSum, 0.0); // the s-pair quartets must route through the lane

    qcx::integrals::FockBuildOptions fp64Only;
    fp64Only.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    fp64Only.useDensityScreening = true;
    fp64Only.useCertifiedMixedPrecision = false;
    auto fp64Builder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, fp64Only);
    ASSERT_TRUE(fp64Builder.has_value()) << fp64Builder.error().message;
    auto fp64Fock = fp64Builder->BuildFock(*densityTensor);
    ASSERT_TRUE(fp64Fock.has_value()) << fp64Fock.error().message;

    double maxDeviation = 0.0;

    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j < 7; ++j)
        {
            maxDeviation = std::max(maxDeviation, std::abs((*mixedFock)(i, j) - (*fp64Fock)(i, j)));
        }
    }

    // No slack: each element's error is bounded by the sum of its
    // density-weighted quartet bounds exactly (the sweep proves the
    // per-quartet bound itself with no slack), so maxDeviation <=
    // boundSum up to fp noise.
    EXPECT_LE(maxDeviation, boundSum + 1e-15)
        << "max deviation " << maxDeviation << " exceeds the certified sum " << boundSum;
}

TEST(FockBuildTest, PresetThresholdsAreMonotone) {
    using qcx::integrals::AccuracyPreset;
    using qcx::integrals::DensityThreshold;
    using qcx::integrals::MixedPrecisionThreshold;
    using qcx::integrals::SchwarzThreshold;

    // Looser presets drop more: Schwarz and density thresholds fall as the
    // preset tightens.
    EXPECT_GT(SchwarzThreshold(AccuracyPreset::kLoose), SchwarzThreshold(AccuracyPreset::kNormal));
    EXPECT_GT(SchwarzThreshold(AccuracyPreset::kNormal), SchwarzThreshold(AccuracyPreset::kTight));
    EXPECT_GE(DensityThreshold(AccuracyPreset::kLoose), DensityThreshold(AccuracyPreset::kNormal));
    EXPECT_GE(DensityThreshold(AccuracyPreset::kNormal), DensityThreshold(AccuracyPreset::kTight));

    // The fp32 gate admits the most on the loosest preset and disables the
    // lane entirely on the tightest.
    EXPECT_GT(MixedPrecisionThreshold(AccuracyPreset::kLoose),
              MixedPrecisionThreshold(AccuracyPreset::kNormal));
    EXPECT_EQ(MixedPrecisionThreshold(AccuracyPreset::kTight), 0.0);
}

TEST(DirectJkFockBuilderTest, CoulombOnlyMatchesFullMinusExchangeOnly) {
    // The split modes (buildCoulombOnly / buildExchangeOnly) must
    // partition the fused result exactly: coulombOnly + exchangeOnly - H =
    // H + 2J - K (each split result carries one full H copy). H2 is instant,
    // so the identity runs ungated.
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(2);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    qcx::integrals::FockBuildOptions fusedOptions;
    fusedOptions.useDensityScreening = false;
    fusedOptions.useCertifiedMixedPrecision = false;
    auto fusedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, fusedOptions);
    ASSERT_TRUE(fusedBuilder.has_value()) << fusedBuilder.error().message;
    auto fusedFock = fusedBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(fusedFock.has_value()) << fusedFock.error().message;

    qcx::integrals::FockBuildOptions coulombOptions = fusedOptions;
    coulombOptions.buildCoulombOnly = true;
    auto coulombBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, coulombOptions);
    ASSERT_TRUE(coulombBuilder.has_value()) << coulombBuilder.error().message;
    auto coulombFock = coulombBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(coulombFock.has_value()) << coulombFock.error().message;

    qcx::integrals::FockBuildOptions exchangeOptions = fusedOptions;
    exchangeOptions.buildExchangeOnly = true;
    auto exchangeBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, exchangeOptions);
    ASSERT_TRUE(exchangeBuilder.has_value()) << exchangeBuilder.error().message;
    auto exchangeFock = exchangeBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(exchangeFock.has_value()) << exchangeFock.error().message;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            const double splitSum = (*coulombFock)(i, j) + (*exchangeFock)(i, j) - (*core)(i, j);
            EXPECT_NEAR(splitSum, (*fusedFock)(i, j), 1e-12) << "element (" << i << "," << j << ")";
        }
    }
}

TEST(DirectJkFockBuilderTest, BothSplitFlagsReturnBareH) {
    // The documented degenerate contract (fock_build.hpp): both split flags
    // together skip BOTH accumulations, so BuildFock returns the core
    // Hamiltonian exactly - no J, no K, no screening influence. Run it with
    // the full stack on (kLoose: the density screen AND the certified fp32
    // lane both active) so the contract holds at its strongest - the lane
    // must not touch the bare-H path (nothing to certify).
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto densityTensor = ToTensor(PhysicalDensity(2));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    options.buildCoulombOnly = true;
    options.buildExchangeOnly = true;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;
    auto fock = builder->BuildFock(*densityTensor);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            EXPECT_DOUBLE_EQ((*fock)(i, j), (*core)(i, j)) << "element (" << i << "," << j << ")";
        }
    }
}

TEST(DirectJkFockBuilderTest, SplitModeAtLoosePresetStaysInsideCertifiedBound) {
    // The fp32-lane interplay the preset sweep records but never asserts: the
    // J-only split mode at kLoose (the density screen AND the certified lane
    // both active) must stay inside its certified bound sum - nothing
    // approximate unless the bound covers it - plus a physicality budget.
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const std::size_t n = core->Shape()[0];
    auto densityTensor = ToTensor(PhysicalDensity(n));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    const Eigen::MatrixXd density = ToMatrix(*densityTensor);

    // The exact reference J: the dense unscreened supermatrix contracted
    // with the full density (screen = false is the retained reference path).
    qcx::integrals::EriDenseOptions eriOptions;
    eriOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    eriOptions.screen = false;
    auto eri = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis, eriOptions);
    ASSERT_TRUE(eri.has_value()) << eri.error().message;
    Eigen::MatrixXd jExact =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t mu = 0; mu < n; ++mu)
    {
        for (std::size_t nu = 0; nu < n; ++nu)
        {
            for (std::size_t l = 0; l < n; ++l)
            {
                for (std::size_t s = 0; s < n; ++s)
                {
                    jExact(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) +=
                        (*eri)(mu, nu, l, s) *
                        density(static_cast<Eigen::Index>(l), static_cast<Eigen::Index>(s));
                }
            }
        }
    }

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    options.buildCoulombOnly = true;
    // The lane is this pin's SUBJECT, so it is requested explicitly: the
    // lane's default is hardware-dependent, and a test that pins what the
    // lane delivers must not ride a default the host decides.
    options.useCertifiedMixedPrecision = true;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;
    double certifiedBoundSum = 0.0;
    auto fock = builder->BuildFock(*densityTensor, &certifiedBoundSum);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;

    const Eigen::MatrixXd jPreset = (ToMatrix(*fock) - ToMatrix(*core)) / 2.0;
    const double maxAbsDJ = (jPreset - jExact).cwiseAbs().maxCoeff();
    EXPECT_GE(certifiedBoundSum, 0.0);
    EXPECT_LE(maxAbsDJ, certifiedBoundSum)
        << "the certified bound sum must dominate the delivered J error";
    EXPECT_LE(maxAbsDJ, 1e-5)
        << "physicality pin (the kLoose bound on this fixture family is 1e-6-scale)";
}

TEST(DirectJkFockBuilderTest, CoreHamiltonianAccessorIsStable) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::FockBuildOptions options;
    options.useDensityScreening = false;
    options.useCertifiedMixedPrecision = false;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    // The accessor returns the construction-time H, bit-identical to the
    // input tensor (0.0 is an exact isApprox tolerance).
    EXPECT_TRUE(ToMatrix(builder->CoreHamiltonian()).isApprox(ToMatrix(*core), 0.0));

    // ...and the reference stays valid and unchanged across BuildFock
    // calls - IncrementalFockBuilder strips H out of every incremental
    // contract, so a mutated or re-pointed accessor would corrupt the
    // delta-accumulation.
    auto densityTensor = ToTensor(PhysicalDensity(2));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    auto fock = builder->BuildFock(*densityTensor);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;
    EXPECT_TRUE(ToMatrix(builder->CoreHamiltonian()).isApprox(ToMatrix(*core), 0.0));
}

TEST(DirectJkFockBuilderTest, ParallelChunkScheduleIsOrderIndependent) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "parallel-schedule pin - not run in the fast smoke subset";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    // Both builds use the DEFAULT preset (kNormal - screening and the
    // certified lane on): the serial-vs-parallel agreement must hold on the
    // full delivery path, not on a lane-off special case. maxParallelChunks
    // alone selects the schedule: 1 forces the serial fallback, 2 a fixed
    // two-chunk split (auto would pick hardware_concurrency()). OMP_NUM_THREADS
    // does NOT select the serial path - the option is the pin.
    qcx::integrals::FockBuildOptions serialOptions;
    serialOptions.maxParallelChunks = 1;
    auto serialBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, serialOptions);
    ASSERT_TRUE(serialBuilder.has_value()) << serialBuilder.error().message;

    qcx::integrals::FockBuildOptions parallelOptions;
    parallelOptions.maxParallelChunks = 2;
    auto parallelBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, parallelOptions);
    ASSERT_TRUE(parallelBuilder.has_value()) << parallelBuilder.error().message;

    auto densityTensor = ToTensor(PhysicalDensity(7));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    // The per-element filter is active on both schedules (the
    // default flag). The Fock sum is order-sensitive in the parallel
    // reduction (the existing last-bit tolerance), but the drop count is an
    // exact addition - the serial and parallel counts must match EXACTLY
    // (the filter is a pure per-element predicate; the count pins
    // are exact, the energy pins keep their last-bit tolerance).
    double serialBound = -1.0;
    qcx::integrals::FockBuildStats serialStats;
    auto serialFock = serialBuilder->BuildFock(*densityTensor, &serialBound, &serialStats);
    ASSERT_TRUE(serialFock.has_value()) << serialFock.error().message;

    double parallelBound = -1.0;
    qcx::integrals::FockBuildStats parallelStats;
    auto parallelFock = parallelBuilder->BuildFock(*densityTensor, &parallelBound, &parallelStats);
    ASSERT_TRUE(parallelFock.has_value()) << parallelFock.error().message;

    const Eigen::MatrixXd serial = ToMatrix(*serialFock);
    const Eigen::MatrixXd parallel = ToMatrix(*parallelFock);

    for (std::size_t i = 0; i < static_cast<std::size_t>(serial.rows()); ++i)
    {
        for (std::size_t j = 0; j < static_cast<std::size_t>(serial.cols()); ++j)
        {
            EXPECT_NEAR(parallel(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        serial(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        1e-12)
                << "element (" << i << "," << j << ")";
        }
    }

    // The certified bound sum is a plain fp64 accumulation in chunk order,
    // so its last bits are schedule-dependent like every other sum here.
    EXPECT_NEAR(parallelBound, serialBound, 1e-12);

    // The exact drop-count pin: per-chunk partials
    // combine by exact addition, so the schedule must not move the count.
    EXPECT_EQ(parallelStats.elementDrops, serialStats.elementDrops);

    // The calibration-term pins: the P and G
    // counters accumulate over the same screened task lists as the quartet
    // counts - exact per-task additions, so the schedule must not move
    // them either (the distinct-pair stamping dedups across the chunk
    // passes, which both schedules run).
    EXPECT_EQ(parallelStats.significantPairCount, serialStats.significantPairCount);
    EXPECT_EQ(parallelStats.primitiveProductSum, serialStats.primitiveProductSum);

    // Semantic bounds on this fixture (7 shells, kNormal, a physical
    // density): the screened set is non-empty; every significant quartet
    // touches at most two pairs (so P <= 2Q); every quartet's primitive-
    // pair product weight is >= 1 and the weight of the pairs it touches
    // makes G >= P (each distinct pair appears in at least one quartet).
    const std::size_t serialQuartets = serialStats.fp64QuartetCount + serialStats.fp32QuartetCount;
    EXPECT_GT(serialStats.significantPairCount, 0U);
    EXPECT_GT(serialStats.primitiveProductSum, 0U);
    EXPECT_LE(serialStats.significantPairCount, 2U * serialQuartets);
    EXPECT_GE(serialStats.primitiveProductSum, serialQuartets);
    EXPECT_GE(serialStats.primitiveProductSum, serialStats.significantPairCount);
}

// The fp32-routing weight lookup miss (the read-back rejection in
// fock_build.cpp) is deliberately not covered by a direct test: it fires
// only on a State-internal bookkeeping bug and is unreachable through the
// public API (the keys that populate the routing list always match the
// screen loop that reads it back).
TEST(FockBuildTest, ErrorPaths) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    // A zero batch cap is rejected at Create.
    qcx::integrals::FockBuildOptions zeroCap;
    zeroCap.maxBatchBytes = 0;
    auto zeroCapBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, zeroCap);
    EXPECT_FALSE(zeroCapBuilder.has_value());
    EXPECT_EQ(zeroCapBuilder.error().code, qcx::ErrorCode::kInvalidArgument);

    // A wrong-shaped density is rejected at BuildFock.
    qcx::integrals::FockBuildOptions options;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    auto wrongShape = CpuTensor2::Create({1, 1});
    ASSERT_TRUE(wrongShape.has_value()) << wrongShape.error().message;
    auto wrongFock = builder->BuildFock(*wrongShape);
    EXPECT_FALSE(wrongFock.has_value());
    EXPECT_EQ(wrongFock.error().code, qcx::ErrorCode::kInvalidArgument);

    // With the lane disabled the certified sum stays zero.
    qcx::integrals::FockBuildOptions noLane;
    noLane.useCertifiedMixedPrecision = false;
    auto noLaneBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, noLane);
    ASSERT_TRUE(noLaneBuilder.has_value()) << noLaneBuilder.error().message;
    auto densityTensor = ToTensor(PhysicalDensity(2));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    double boundSum = -1.0;
    auto noLaneFock = noLaneBuilder->BuildFock(*densityTensor, &boundSum);
    ASSERT_TRUE(noLaneFock.has_value()) << noLaneFock.error().message;
    EXPECT_EQ(boundSum, 0.0);
}

// The water C2v reduction over the real pair-list function order. The
// canonicalized water molecule (atoms renumbered by Z, so H first) yields
// the H-first shell order [H1s, H2s, O1s, O2s, Op] with function order
// [H1s, H2s, O1s, O2s, Opy, Opz, Opx] (m ascending -l..+l: m=-1 ~ y,
// m=0 ~ z, m=+1 ~ x); with the molecule in the xy plane the C2v elements
// are {I, sigma_z (z-flip), sigma_x (x-flip), C2 (pi about y)} - sigma_x
// and C2 swap the two H functions. The duplicate of the scf fixture (the
// integrals module cannot include scf headers).
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

    // The class-aware contraction must equal the plain contraction
// at machine precision - the class path computes each orbit rep block once
// and expands it per screened member, and every expansion must reproduce
// the plain path's per-quartet contraction exactly (the same screened
// quartets, the same member-level kMult).
TEST(DirectJkFockBuilderTest, WaterC2vClassPathMatchesPlain) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
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

    qcx::integrals::FockBuildOptions plainOptions;
    plainOptions.useCertifiedMixedPrecision = false;
    auto plainBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, plainOptions);
    ASSERT_TRUE(plainBuilder.has_value()) << plainBuilder.error().message;

    qcx::integrals::FockBuildOptions classOptions = plainOptions;
    classOptions.symmetryReduction = &reduction;
    auto classBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, classOptions);
    ASSERT_TRUE(classBuilder.has_value()) << classBuilder.error().message;

    double plainBoundSum = -1.0;
    auto plainFock = plainBuilder->BuildFock(*densityTensor, &plainBoundSum);
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;
    double classBoundSum = -1.0;
    auto classFock = classBuilder->BuildFock(*densityTensor, &classBoundSum);
    ASSERT_TRUE(classFock.has_value()) << classFock.error().message;

    const Eigen::MatrixXd plain = ToMatrix(*plainFock);
    const Eigen::MatrixXd contracted = ToMatrix(*classFock);

    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j < 7; ++j)
        {
            EXPECT_NEAR(contracted(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        plain(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        1e-12)
                << "element (" << i << "," << j << ")";
        }
    }

    EXPECT_EQ(plainBoundSum, 0.0);
    EXPECT_EQ(classBoundSum, 0.0);

    // kTight keeps the certified lane off (MixedPrecisionThreshold(kTight) = 0)
    // on BOTH paths - the class path with the lane requested at kTight
    // runs fp64 and the certified sum stays 0 (the same gate the plain
    // path applies; the strict preset's pins depend on the fp64 path).
    qcx::integrals::FockBuildOptions tightOptions;
    tightOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    tightOptions.useCertifiedMixedPrecision = true;
    tightOptions.symmetryReduction = &reduction;
    auto tightClassBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, tightOptions);
    ASSERT_TRUE(tightClassBuilder.has_value()) << tightClassBuilder.error().message;
    double tightBoundSum = -1.0;
    auto tightClassFock = tightClassBuilder->BuildFock(*densityTensor, &tightBoundSum);
    ASSERT_TRUE(tightClassFock.has_value()) << tightClassFock.error().message;
    EXPECT_EQ(tightBoundSum, 0.0);

    const Eigen::MatrixXd tightClass = ToMatrix(*tightClassFock);

    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j < 7; ++j)
        {
            EXPECT_NEAR(tightClass(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        plain(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        1e-12)
                << "kTight lane-off element (" << i << "," << j << ")";
        }
    }

    // A random-symmetric density with the same fixture: the same
    // equivalence (the class table is density-independent; the screened
    // member sets change per iteration, so the contraction must track the
    // screening decisions).
    Eigen::MatrixXd randomSymmetric(7, 7);
    std::mt19937_64 rng(20260828);
    std::uniform_real_distribution<double> dist(-0.25, 0.25);

    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = (i == j) ? 0.5 + dist(rng) : dist(rng);
            randomSymmetric(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
            randomSymmetric(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = value;
        }
    }

    auto randomTensor = ToTensor(randomSymmetric);
    ASSERT_TRUE(randomTensor.has_value()) << randomTensor.error().message;
    auto randomPlainFock = plainBuilder->BuildFock(*randomTensor);
    ASSERT_TRUE(randomPlainFock.has_value()) << randomPlainFock.error().message;
    auto randomClassFock = classBuilder->BuildFock(*randomTensor);
    ASSERT_TRUE(randomClassFock.has_value()) << randomClassFock.error().message;

    const Eigen::MatrixXd randomPlain = ToMatrix(*randomPlainFock);
    const Eigen::MatrixXd randomContracted = ToMatrix(*randomClassFock);

    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j < 7; ++j)
        {
            EXPECT_NEAR(
                randomContracted(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                randomPlain(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                1e-12)
                << "random element (" << i << "," << j << ")";
        }
    }
}

// The certified fp32 lane engages on the class-aware path
// exactly as on the plain path. Each screened member quartet routes by its
// own density-weighted bound (the same ScreenAll partition), the orbit rep
// is assembled once PER precision (an orbit whose members split across
// precisions has its rep computed in both), and the certified sum
// accumulates weight(member) * bound(orbit rep) - the rep block's
// per-quartet bound covers every expanded member (the expansion is an
// exact signed permutation of the rep block, so the member error is the
// rep error), and equals the member's own bound (the engine's bound
// derives from the quartet's isometry-invariant data), so the class sum
// equals the plain sum up to the parallel reduction's summation order.
TEST(DirectJkFockBuilderTest, WaterC2vClassPathMixedPrecisionMatchesPlain) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
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

    // kLoose: the gate routes the low-weight quartets through the lane.
    // The lane is requested explicitly: its default is hardware-dependent,
    // and this pin's subject is the lane's arithmetic on both the plain and
    // the class path.
    qcx::integrals::FockBuildOptions mixedOptions;
    mixedOptions.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    mixedOptions.useCertifiedMixedPrecision = true;
    auto plainMixedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, mixedOptions);
    ASSERT_TRUE(plainMixedBuilder.has_value()) << plainMixedBuilder.error().message;
    qcx::integrals::FockBuildOptions mixedClassOptions = mixedOptions;
    mixedClassOptions.symmetryReduction = &reduction;
    auto mixedClassBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, mixedClassOptions);
    ASSERT_TRUE(mixedClassBuilder.has_value()) << mixedClassBuilder.error().message;

    double plainMixedBoundSum = -1.0;
    auto plainMixedFock = plainMixedBuilder->BuildFock(*densityTensor, &plainMixedBoundSum);
    ASSERT_TRUE(plainMixedFock.has_value()) << plainMixedFock.error().message;
    double mixedClassBoundSum = -1.0;
    auto mixedClassFock = mixedClassBuilder->BuildFock(*densityTensor, &mixedClassBoundSum);
    ASSERT_TRUE(mixedClassFock.has_value()) << mixedClassFock.error().message;

    // The lane must actually route on this fixture with the physical
    // density (the s-pair quartets sit inside the kLoose gate).
    EXPECT_GT(plainMixedBoundSum, 0.0);
    EXPECT_GT(mixedClassBoundSum, 0.0);

    // The class sum accumulates the same per-member contributions as the
    // plain sum (weight(member) * bound(rep) == weight(member) *
    // bound(member)); only the summation order differs (the parallel
    // reduction's last bits).
    EXPECT_NEAR(mixedClassBoundSum, plainMixedBoundSum, 1e-12);

    const Eigen::MatrixXd plainMixed = ToMatrix(*plainMixedFock);
    const Eigen::MatrixXd classMixed = ToMatrix(*mixedClassFock);

    // Each path's per-element Fock error is at most its own certified
    // bound sum (the composition of the per-quartet bounds), so the
    // class-vs-plain deviation is bounded by the sum of the two runs'
    // sums; 1.001 absorbs the composition's own rounding.
    const double mixedTolerance = (plainMixedBoundSum + mixedClassBoundSum) * 1.001 + 1e-14;
    EXPECT_LE((classMixed - plainMixed).cwiseAbs().maxCoeff(), mixedTolerance);

    // And each path sits inside its own certified bound of the fp64
    // reference: the certified-bound dominance through the class path (the expansion
    // preserves the rep's error element-wise, so the rep bound covers the
    // members).
    qcx::integrals::FockBuildOptions fp64Options;
    fp64Options.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    fp64Options.useCertifiedMixedPrecision = false;
    auto fp64Builder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, fp64Options);
    ASSERT_TRUE(fp64Builder.has_value()) << fp64Builder.error().message;
    auto fp64Fock = fp64Builder->BuildFock(*densityTensor);
    ASSERT_TRUE(fp64Fock.has_value()) << fp64Fock.error().message;
    const Eigen::MatrixXd plainFp64 = ToMatrix(*fp64Fock);

    EXPECT_LE((classMixed - plainFp64).cwiseAbs().maxCoeff(), mixedClassBoundSum * 1.001 + 1e-14);
    EXPECT_LE((plainMixed - plainFp64).cwiseAbs().maxCoeff(), plainMixedBoundSum * 1.001 + 1e-14);

    // The split builder shapes the direct-UHF adapter uses
    // (buildCoulombOnly / buildExchangeOnly): the same class-vs-plain
    // contract holds in each mode (the lane engages on the class path in
    // both - the member expansion and the certified accumulation are
    // mode-independent).
    for (const bool coulombOnly : {true, false})
    {
        qcx::integrals::FockBuildOptions splitOptions;
        splitOptions.accuracy = qcx::integrals::AccuracyPreset::kLoose;

        if (coulombOnly)
        {
            splitOptions.buildCoulombOnly = true;
        } else
        {
            splitOptions.buildExchangeOnly = true;
        }

        auto plainSplitBuilder =
            qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, splitOptions);
        ASSERT_TRUE(plainSplitBuilder.has_value()) << plainSplitBuilder.error().message;
        qcx::integrals::FockBuildOptions splitClassOptions = splitOptions;
        splitClassOptions.symmetryReduction = &reduction;
        auto splitClassBuilder = qcx::integrals::DirectJkFockBuilder::Create(
            *molecule, *basis, *core, splitClassOptions);
        ASSERT_TRUE(splitClassBuilder.has_value()) << splitClassBuilder.error().message;

        double plainSplitBoundSum = -1.0;
        auto plainSplitFock = plainSplitBuilder->BuildFock(*densityTensor, &plainSplitBoundSum);
        ASSERT_TRUE(plainSplitFock.has_value()) << plainSplitFock.error().message;
        double splitClassBoundSum = -1.0;
        auto splitClassFock = splitClassBuilder->BuildFock(*densityTensor, &splitClassBoundSum);
        ASSERT_TRUE(splitClassFock.has_value()) << splitClassFock.error().message;

        EXPECT_NEAR(splitClassBoundSum, plainSplitBoundSum, 1e-12);
        EXPECT_LE((ToMatrix(*splitClassFock) - ToMatrix(*plainSplitFock)).cwiseAbs().maxCoeff(),
                  (plainSplitBoundSum + splitClassBoundSum) * 1.001 + 1e-14);
    }

    // A random-symmetric density with the same fixture: the routing
    // partition changes with the density (different quartets enter the
    // lane), so the member-level routing must track the screening
    // decisions - the same equivalence at the same contract.
    Eigen::MatrixXd randomSymmetric(7, 7);
    std::mt19937_64 rng(20260828);
    std::uniform_real_distribution<double> dist(-0.25, 0.25);

    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = (i == j) ? 0.5 + dist(rng) : dist(rng);
            randomSymmetric(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
            randomSymmetric(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = value;
        }
    }

    auto randomTensor = ToTensor(randomSymmetric);
    ASSERT_TRUE(randomTensor.has_value()) << randomTensor.error().message;
    double randomPlainBoundSum = -1.0;
    auto randomPlainFock = plainMixedBuilder->BuildFock(*randomTensor, &randomPlainBoundSum);
    ASSERT_TRUE(randomPlainFock.has_value()) << randomPlainFock.error().message;
    double randomClassBoundSum = -1.0;
    auto randomClassFock = mixedClassBuilder->BuildFock(*randomTensor, &randomClassBoundSum);
    ASSERT_TRUE(randomClassFock.has_value()) << randomClassFock.error().message;

    EXPECT_NEAR(randomClassBoundSum, randomPlainBoundSum, 1e-12);
    EXPECT_LE((ToMatrix(*randomClassFock) - ToMatrix(*randomPlainFock)).cwiseAbs().maxCoeff(),
              (randomPlainBoundSum + randomClassBoundSum) * 1.001 + 1e-14);
}

// A trivial reduction (groupOrder 1) keeps the plain path: Create() never
// engages the class table, so the build takes the null reduction's exact
// code path - including the certified lane routing (the reduction pointer
// must not disable it).
TEST(DirectJkFockBuilderTest, TrivialReductionKeepsPlainPath) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto densityTensor = ToTensor(PhysicalDensity(7));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    qcx::integrals::FockBuildOptions plainOptions;
    plainOptions.useCertifiedMixedPrecision = true;
    auto plainBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, plainOptions);
    ASSERT_TRUE(plainBuilder.has_value()) << plainBuilder.error().message;

    // The default-constructed reduction: groupOrder 1, isTrivial - the C1
    // case (also what the scf extraction returns for C1 molecules).
    const qcx::integrals::SymmetryReduction trivial;
    qcx::integrals::FockBuildOptions trivialOptions = plainOptions;
    trivialOptions.symmetryReduction = &trivial;
    auto trivialBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, trivialOptions);
    ASSERT_TRUE(trivialBuilder.has_value()) << trivialBuilder.error().message;

    double plainBoundSum = -1.0;
    auto plainFock = plainBuilder->BuildFock(*densityTensor, &plainBoundSum);
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;
    double trivialBoundSum = -1.0;
    auto trivialFock = trivialBuilder->BuildFock(*densityTensor, &trivialBoundSum);
    ASSERT_TRUE(trivialFock.has_value()) << trivialFock.error().message;

    // Both take the plain path - the same code, the same certified-lane
    // routing (the trivial reduction must not disable it). The results are
    // now BIT-IDENTICAL: the parallel contraction's summation order used
    // to be unspecified (the scatter, observed ~1e-14 element-wise on
    // water), and the fixed-order join pinned it to the block index
    // (internal/fixed_order_reduce.hpp). The tolerance is 0.0 so this
    // stays a pin on the path identity AND on the reduction order - a
    // regression to the completion-order merge fails here.
    EXPECT_TRUE(ToMatrix(*trivialFock).isApprox(ToMatrix(*plainFock), 0.0));
    EXPECT_EQ(trivialBoundSum, plainBoundSum);
}

// The dispatch A/B pin. The contraction kernel is compiled
// twice - the scalar copy inside fock_build.cpp (no /arch flag), an
// /arch:AVX2 copy in fock_contract_simd.cpp - and dispatched at runtime on
// cpuid (FockAvx2Available, the boys_simd.cpp shape).
// FockBuildOptions::forceScalarContract pins the scalar copy even on an
// AVX2-capable machine, so BOTH copies are exercised on every machine: any
// difference beyond the band below is a dispatch bug (the fallback's
// contract - a bug in the scalar copy would otherwise
// The comparison is bit-exact on MSVC Release (/fp:precise never contracts)
// and on the Debug legs (nothing vectorizes at -O0). On GNU Release the
// -mavx2;-mfma copy fuses mul+add into FMA (-ffp-contract=fast is the GNU
// default) where the SSE2 scalar copy cannot, so the copies diverge at the
// last ulp elementwise (H2o's memcmp returned 1, SplitModes' exchange-only
// leg 2). The WSL gcc
// 15.2 reproduce at OMP_NUM_THREADS=1 measured the H2o max element
// divergence at 1.11e-16 (one ulp at 1.0); the byte evidence bounds the
// divergence at a few ulps of the largest element (~20 a.u. -> ~9e-15), and
// the band is that bound times ~3.4 (the measured geometry; the local
// actuals ride this build's own record). The comparison stays serial-only
// (maxParallelChunks 1): the parallel path's critical-section reduction
// order is not deterministic (observed last-ulp scatter on water), which is
// orthogonal to the dispatch - the parallel agreement is the separate
// Avx2DispatchPathsAgreeAtDefaultParallelism test.
constexpr double kDispatchCopyMaxAbsDiff = 5e-14;

void CheckDispatchPathsBitIdentical(const qcx::molecule::Molecule& molecule,
                                    const qcx::basisset::BasisSet& basisSet,
                                    const qcx::integrals::FockBuildOptions& options,
                                    std::size_t nFunctions) {
    auto core = BuildCoreHamiltonian(molecule, basisSet);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    auto defaultBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, *core, options);
    ASSERT_TRUE(defaultBuilder.has_value()) << defaultBuilder.error().message;

    auto scalarOptions = options;
    scalarOptions.forceScalarContract = true;
    auto scalarBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, *core, scalarOptions);
    ASSERT_TRUE(scalarBuilder.has_value()) << scalarBuilder.error().message;

    auto densityTensor = ToTensor(PhysicalDensity(nFunctions));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    auto defaultFock = defaultBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(defaultFock.has_value()) << defaultFock.error().message;
    auto scalarFock = scalarBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(scalarFock.has_value()) << scalarFock.error().message;

    const Eigen::MatrixXd defaultMatrix = ToMatrix(*defaultFock);
    const Eigen::MatrixXd scalarMatrix = ToMatrix(*scalarFock);
    const double maxAbsDiff = (defaultMatrix - scalarMatrix).cwiseAbs().maxCoeff();
    EXPECT_LE(maxAbsDiff, kDispatchCopyMaxAbsDiff)
        << "the AVX2 and scalar kernel copies diverge beyond the cross-platform band";
}

TEST(FockBuildTest, Avx2DispatchPathsAreBitIdenticalH2) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::FockBuildOptions options;
    options.maxParallelChunks = 1;
    CheckDispatchPathsBitIdentical(*molecule, *basis, options, 2);
}

TEST(FockBuildTest, Avx2DispatchPathsAreBitIdenticalH2o) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::FockBuildOptions options;
    options.maxParallelChunks = 1;
    CheckDispatchPathsBitIdentical(*molecule, *basis, options, 7);
}

TEST(FockBuildTest, Avx2DispatchPathsAreBitIdenticalSplitModes) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::FockBuildOptions exchangeOnly;
    exchangeOnly.buildExchangeOnly = true;
    exchangeOnly.maxParallelChunks = 1;
    CheckDispatchPathsBitIdentical(*molecule, *basis, exchangeOnly, 7);

    qcx::integrals::FockBuildOptions coulombOnly;
    coulombOnly.buildCoulombOnly = true;
    coulombOnly.maxParallelChunks = 1;
    CheckDispatchPathsBitIdentical(*molecule, *basis, coulombOnly, 7);
}

// The parallel-path agreement: at the default chunk schedule
// the two dispatch copies must agree to the same tolerance two builds of
// the SAME copy agree to (the reduction's last-ulp scatter, not a kernel
// difference). Forces the scalar copy through an AVX2-capable CPU's default
// path and compares against the dispatched (AVX2) build.
TEST(FockBuildTest, Avx2DispatchPathsAgreeAtDefaultParallelism) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::FockBuildOptions defaultOptions;
    auto defaultBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, defaultOptions);
    ASSERT_TRUE(defaultBuilder.has_value()) << defaultBuilder.error().message;

    qcx::integrals::FockBuildOptions scalarOptions = defaultOptions;
    scalarOptions.forceScalarContract = true;
    auto scalarBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, scalarOptions);
    ASSERT_TRUE(scalarBuilder.has_value()) << scalarBuilder.error().message;

    auto densityTensor = ToTensor(PhysicalDensity(7));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    auto defaultFock = defaultBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(defaultFock.has_value()) << defaultFock.error().message;
    auto scalarFock = scalarBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(scalarFock.has_value()) << scalarFock.error().message;

    // The same 1e-13 budget the path-identity test above pins; the
    // reduction scatter is ~1e-14 element-wise on water.
    EXPECT_TRUE(ToMatrix(*defaultFock).isApprox(ToMatrix(*scalarFock), 1e-13));
}

// The per-element re-filter's accounting. At a
// density SCALED UP by 100 both gates keep every quartet provably (six-block:
// 100*dMax*Q_bra*Q_ket ~ 1e-2-100; product: 1e4*maxD_bra*maxD_ket*Q_bra*Q_ket
// ~ 1-1e4 - both far above the 1e-8 tau), so the keep sets are equal, the
// filter is the only difference between the two builds, and the accumulation
// is bit-identical iff the filter only ever drops elements whose
// contributions are exactly zero. The water STO-3G molecule lies in the z=0
// plane, so every integral with an odd count of O2p_z functions is exactly
// zero (~496 elements): the filter drops them at ANY density. Leg A pins
// drops > 0 with the ON builder's counter, the flag contract pins the OFF
// builder's counter at exactly 0, and the builds stay bit-identical.
//
// The sparse legs zero one shell-pair block of the density to 1e-6: the
// (H1s,H2s) block at elements (1,2)/(2,1) - the far H-H pair, whose
// integrals are the small ones (g(1,2|1,1) = 0.0296, g(1,2|1,2) = 0.00731,
// g(1,2|2,4) = 8.8e-4, g(1,2|3,4) = 1.65e-3; Q(1,2) = sqrt(g(1,2|1,2)) =
// 0.0855). In FULL mode the product gate keeps every quartet at this
// density: the tightest J branch is (1,2|1,1) at
// 1e-6 * 0.408 * 0.0855 * 0.880 = 3.07e-8 = 3.07*tau, and every other
// (1,2|x,y) quartet keeps via a dense K cross product (>= 1e-3, dominated
// by pairMax products 0.14-0.25 against Schwarz bounds 0.1+). The sparse
// full-mode leg is therefore a pure-filter comparison too: the per-target
// thresholds tau/(2*1e-6) = 5e-3 drop real elements (g(1,2|2,4) and
// g(1,2|3,4) are both < 5e-3), each dropped J element-pair carries <= 2*tau
// to a touched Fock element (both orientation multiplies under the halved J
// threshold and the 2.0*j accumulation) and each dropped K element <= tau
// (single-orientation multiply), so the conservative elementwise assertion
// 2*tau*(total dropped elements) holds - the per-element attribution of
// drops is not tracked.
//
// The K-only legs: at the scaled dense density both gates keep everything
// (Leg C-a pins the K bound tau*drops elementwise - single-orientation
// multiplies). At the sparse density the K gate differs from the six-block
// gate: a quartet whose BOTH K products carry the tiny pair (e.g.
// (0,1|2,2): cross pairs (0,2),(1,2),(0,2),(1,2)) is dropped whole even
// though its K3/K4 sections contract the DENSE density pair (0,2) - the
// gate difference carries that quartet's full K contribution (~0.08 at
// tau = 1e-8), so Leg C-b pins drops > 0 and the gate difference, with NO
// elementwise bound (the K filter bound itself is pinned in Leg C-a).
//
// The J-only leg pins the gate/filter interplay instead: the J gate drops
// whole quartets whose bra or ket pair carries the tiny block (the product
// gate's intended tightening - (1,2|x,y) drops whenever the density pair
// (x,y) is small enough that d_xy*Q(x,y) < tau/(d12*Q(1,2)) = 0.118; the
// dense-diagonal pairs (0,0),(1,1),(2,2),(3,3),(4,4) keep at 2.6-5.9x
// margin). The gate-dropped quartets carry their FULL J contributions, so
// the deviation (~1e-4-1e-3) is the gate pin, not the filter's - no
// elementwise bound is asserted in J-only mode: gate agreement AND filter
// engagement cannot coexist there (the (1,2|1,2) diagonal needs d12 >=
// sqrt(tau)/Q(1,2) = 3.3e-3 for its J product to pass the gate, while the
// filter stays engaged only below d12 ~ 1e-5). The filter still fires
// inside the J-kept quartets (the (1,2|4,4) block drops its six
// exact-zero O2p components at threshold 5e-3), so the J-only drop counter
// is nonzero too.
TEST(DirectJkFockBuilderTest, PerElementFilterDropsBoundedContributions) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const double tau = qcx::integrals::DensityThreshold(qcx::integrals::AccuracyPreset::kLoose);

    qcx::integrals::FockBuildOptions offOptions;
    offOptions.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    offOptions.useCertifiedMixedPrecision = false;
    offOptions.maxParallelChunks = 1;
    offOptions.usePerElementScreening = false;
    auto offBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, offOptions);
    ASSERT_TRUE(offBuilder.has_value()) << offBuilder.error().message;

    qcx::integrals::FockBuildOptions onOptions = offOptions;
    onOptions.usePerElementScreening = true;
    auto onBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, onOptions);
    ASSERT_TRUE(onBuilder.has_value()) << onBuilder.error().message;

    // Leg A: the scaled dense density - only the exact-zero z-plane
    // symmetry elements drop (their contributions are exactly zero), so the
    // filtered build is BIT-identical to the unfiltered one (both serial).
    // The counters pin the flag contract: exactly 0 when the flag is off
    // (the legacy gate path never counts), strictly positive when it is on.
    auto denseTensor = ToTensor(100.0 * PhysicalDensity(7));
    ASSERT_TRUE(denseTensor.has_value()) << denseTensor.error().message;
    qcx::integrals::FockBuildStats denseOffStats;
    auto denseOff = offBuilder->BuildFock(*denseTensor, nullptr, &denseOffStats);
    ASSERT_TRUE(denseOff.has_value()) << denseOff.error().message;
    qcx::integrals::FockBuildStats denseOnStats;
    auto denseOn = onBuilder->BuildFock(*denseTensor, nullptr, &denseOnStats);
    ASSERT_TRUE(denseOn.has_value()) << denseOn.error().message;
    EXPECT_EQ(denseOffStats.elementDrops, 0u);
    EXPECT_GT(denseOnStats.elementDrops, 0u);
    EXPECT_EQ(
        // Bit-identity is the tested property (see the leg comment above);
        // memcmp is its oracle - doubles do not have unique object
        // representations, but the contract here is exact bits.
        //
        // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison)
        std::memcmp(ToMatrix(*denseOff).data(),
                    ToMatrix(*denseOn).data(),
                    std::size_t{7} * 7 * sizeof(double)),
        0);

    // Leg B: the (H1s,H2s) shell-pair block zeroed to 1e-6 - the filter
    // engages on REAL elements and the deviation respects the recorded
    // bound 2*tau*drops elementwise (the gate agreement argument is in the
    // header comment).
    Eigen::MatrixXd sparse = PhysicalDensity(7);
    sparse(1, 2) = 1e-6;
    sparse(2, 1) = 1e-6;
    auto sparseTensor = ToTensor(sparse);
    ASSERT_TRUE(sparseTensor.has_value()) << sparseTensor.error().message;
    qcx::integrals::FockBuildStats sparseOnStats;
    auto sparseOff = offBuilder->BuildFock(*sparseTensor);
    ASSERT_TRUE(sparseOff.has_value()) << sparseOff.error().message;
    auto sparseOn = onBuilder->BuildFock(*sparseTensor, nullptr, &sparseOnStats);
    ASSERT_TRUE(sparseOn.has_value()) << sparseOn.error().message;
    EXPECT_GT(sparseOnStats.elementDrops, 0u);
    const double maxDeviation = (ToMatrix(*sparseOff) - ToMatrix(*sparseOn)).cwiseAbs().maxCoeff();
    EXPECT_GT(maxDeviation, 0.0)
        << "the filter must drop real nonzero elements, not only the symmetry zeros";
    EXPECT_LE(maxDeviation, 2.0 * tau * static_cast<double>(sparseOnStats.elementDrops))
        << "max deviation " << maxDeviation << " exceeds 2*tau*drops (tau " << tau << ", drops "
        << sparseOnStats.elementDrops << ")";

    // Leg C-a: the K-only mode at the scaled dense density - both gates
    // keep every quartet (the argument of Leg A applies to the K products
    // unchanged), so the comparison isolates the filter and the dropped
    // elements are single-orientation K multiplies: the K bound
    // tau*drops holds elementwise.
    qcx::integrals::FockBuildOptions exchangeOff = offOptions;
    exchangeOff.buildExchangeOnly = true;
    auto exchangeOffBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, exchangeOff);
    ASSERT_TRUE(exchangeOffBuilder.has_value()) << exchangeOffBuilder.error().message;
    qcx::integrals::FockBuildOptions exchangeOn = exchangeOff;
    exchangeOn.usePerElementScreening = true;
    auto exchangeOnBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, exchangeOn);
    ASSERT_TRUE(exchangeOnBuilder.has_value()) << exchangeOnBuilder.error().message;
    qcx::integrals::FockBuildStats exchangeOnStats;
    auto exchangeOffFock = exchangeOffBuilder->BuildFock(*denseTensor);
    ASSERT_TRUE(exchangeOffFock.has_value()) << exchangeOffFock.error().message;
    auto exchangeOnFock = exchangeOnBuilder->BuildFock(*denseTensor, nullptr, &exchangeOnStats);
    ASSERT_TRUE(exchangeOnFock.has_value()) << exchangeOnFock.error().message;
    EXPECT_GT(exchangeOnStats.elementDrops, 0u);
    const double exchangeDeviation =
        (ToMatrix(*exchangeOffFock) - ToMatrix(*exchangeOnFock)).cwiseAbs().maxCoeff();
    EXPECT_LE(exchangeDeviation, tau * static_cast<double>(exchangeOnStats.elementDrops))
        << "K-mode deviation " << exchangeDeviation << " exceeds tau*drops (drops "
        << exchangeOnStats.elementDrops << ")";

    // Leg C-b: the K-only mode at the sparse density - the K gate engages
    // ((1,2|1,1) keeps via its own K product at 3.07e-8 = 3.07*tau, so the
    // filter fires), and the K gate DIFFERS from the six-block gate here:
    // a quartet like (0,1|2,2) has BOTH K products carrying the tiny pair
    // (its cross pairs are (0,2),(1,2),(0,2),(1,2)), so the K gate drops it
    // whole while the six-block gate kept it (dMax*Q_bra*Q_ket = 0.26) -
    // and its K3/K4 sections contract the DENSE density pair (0,2) = 0.24,
    // so the gate difference carries the quartet's FULL K contribution
    // (~0.08, measured 0.0848 - no per-drop bound can cover it). This is
    // the mode-aware K gate's documented tightening: on a non-uniform
    // density the K products are NOT comparable to Q_bra*Q_ket, so the
    // deviation includes gate-difference contributions the elementDrops
    // counter never sees. The K filter bound itself is pinned in Leg C-a;
    // this leg pins drops > 0 and the gate difference (deviation > 0).
    qcx::integrals::FockBuildStats exchangeSparseOnStats;
    auto exchangeSparseOnFock =
        exchangeOnBuilder->BuildFock(*sparseTensor, nullptr, &exchangeSparseOnStats);
    ASSERT_TRUE(exchangeSparseOnFock.has_value()) << exchangeSparseOnFock.error().message;
    EXPECT_GT(exchangeSparseOnStats.elementDrops, 0u);
    auto exchangeSparseOffFock = exchangeOffBuilder->BuildFock(*sparseTensor);
    ASSERT_TRUE(exchangeSparseOffFock.has_value()) << exchangeSparseOffFock.error().message;
    EXPECT_GT(
        (ToMatrix(*exchangeSparseOffFock) - ToMatrix(*exchangeSparseOnFock)).cwiseAbs().maxCoeff(),
        0.0)
        << "the K-only gate must differ from the six-block gate on a non-uniform density";

    // Leg D: the J-only mode - the gate/filter interplay (header comment):
    // the product J gate drops the tiny-pair quartets whole, so the
    // deviation (~1e-4-1e-3) is the gate pin and no filter bound applies,
    // while the filter still fires inside the J-kept quartets (the
    // (1,2|4,4) block's six exact-zero O2p components at threshold 5e-3).
    qcx::integrals::FockBuildOptions coulombOff = offOptions;
    coulombOff.buildCoulombOnly = true;
    auto coulombOffBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, coulombOff);
    ASSERT_TRUE(coulombOffBuilder.has_value()) << coulombOffBuilder.error().message;
    qcx::integrals::FockBuildOptions coulombOn = coulombOff;
    coulombOn.usePerElementScreening = true;
    auto coulombOnBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, coulombOn);
    ASSERT_TRUE(coulombOnBuilder.has_value()) << coulombOnBuilder.error().message;
    qcx::integrals::FockBuildStats coulombOnStats;
    auto coulombOffFock = coulombOffBuilder->BuildFock(*sparseTensor);
    ASSERT_TRUE(coulombOffFock.has_value()) << coulombOffFock.error().message;
    auto coulombOnFock = coulombOnBuilder->BuildFock(*sparseTensor, nullptr, &coulombOnStats);
    ASSERT_TRUE(coulombOnFock.has_value()) << coulombOnFock.error().message;
    EXPECT_GT(coulombOnStats.elementDrops, 0u);
    EXPECT_GT((ToMatrix(*coulombOffFock) - ToMatrix(*coulombOnFock)).cwiseAbs().maxCoeff(), 0.0)
        << "the J-mode product gate must drop the tiny-pair quartets the legacy gate kept";
    // ...and at the dense density the J-only builds stay bit-identical
    // (only the exact-zero elements drop on the filtered side).
    auto denseCoulombOff = coulombOffBuilder->BuildFock(*denseTensor);
    ASSERT_TRUE(denseCoulombOff.has_value()) << denseCoulombOff.error().message;
    auto denseCoulombOn = coulombOnBuilder->BuildFock(*denseTensor);
    ASSERT_TRUE(denseCoulombOn.has_value()) << denseCoulombOn.error().message;
    EXPECT_EQ(
        // Bit-identity is the tested property (see the comment above);
        // memcmp is its oracle - doubles do not have unique object
        // representations, but the contract here is exact bits.
        //
        // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison)
        std::memcmp(ToMatrix(*denseCoulombOff).data(),
                    ToMatrix(*denseCoulombOn).data(),
                    std::size_t{7} * 7 * sizeof(double)),
        0);
}

// The class-aware path sees the same per-element
// thresholds as the plain path - the class contraction expands orbit reps
// into member quartets with the SAME shells, so the six per-target maxima
// and the drop decisions are identical, and the exact drop count must match
// (both paths accumulate into their contractor's counter).
TEST(DirectJkFockBuilderTest, PerElementFilterPreservesClassPathEquivalence) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
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

    qcx::integrals::FockBuildOptions plainOptions;
    plainOptions.useCertifiedMixedPrecision = false;
    plainOptions.usePerElementScreening = true;
    auto plainBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, plainOptions);
    ASSERT_TRUE(plainBuilder.has_value()) << plainBuilder.error().message;

    qcx::integrals::FockBuildOptions classOptions = plainOptions;
    classOptions.symmetryReduction = &reduction;
    auto classBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, classOptions);
    ASSERT_TRUE(classBuilder.has_value()) << classBuilder.error().message;

    qcx::integrals::FockBuildStats plainStats;
    auto plainFock = plainBuilder->BuildFock(*densityTensor, nullptr, &plainStats);
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;
    qcx::integrals::FockBuildStats classStats;
    auto classFock = classBuilder->BuildFock(*densityTensor, nullptr, &classStats);
    ASSERT_TRUE(classFock.has_value()) << classFock.error().message;

    const Eigen::MatrixXd plain = ToMatrix(*plainFock);
    const Eigen::MatrixXd contracted = ToMatrix(*classFock);

    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j < 7; ++j)
        {
            EXPECT_NEAR(contracted(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        plain(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        1e-12)
                << "element (" << i << "," << j << ")";
        }
    }

    // The exact drop-count pin: the class path must drop the same elements
    // (same thresholds, same shells - the expansion changes nothing).
    EXPECT_EQ(classStats.elementDrops, plainStats.elementDrops);
}

// The certified-bound dominance re-run with
// the per-element filter active (the default - the flag is set explicitly to
// pin the configuration): the filter only REMOVES terms, so the certified
// sum still dominates the delivered fp32-lane deviation. BOTH sides of the
// comparison filter (same gate, same per-target thresholds - the filter is
// precision-independent), so the residual is the lane's delivered error
// exactly, and the kLoose leg asserts the certified sum dominates it. The
// kNormal leg pins the composition at the preset where the lane is
// provably silent on this fixture (every routing bound cClass*eps*Q*max|D|
// ~ 5e-8 exceeds the 1e-10 gate, so boundSum == 0 and the two builds are
// bit-identical).
TEST(DirectJkFockBuilderTest, PerElementFilterKeepsCertifiedBoundDominant) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
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

    qcx::integrals::FockBuildOptions mixed;
    mixed.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    mixed.useDensityScreening = true;
    mixed.useCertifiedMixedPrecision = true;
    mixed.usePerElementScreening = true;
    auto mixedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, mixed);
    ASSERT_TRUE(mixedBuilder.has_value()) << mixedBuilder.error().message;
    double boundSum = -1.0;
    auto mixedFock = mixedBuilder->BuildFock(*densityTensor, &boundSum);
    ASSERT_TRUE(mixedFock.has_value()) << mixedFock.error().message;
    EXPECT_GT(boundSum, 0.0); // the s-pair quartets must route through the lane

    qcx::integrals::FockBuildOptions fp64Only = mixed;
    fp64Only.useCertifiedMixedPrecision = false;
    auto fp64Builder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, fp64Only);
    ASSERT_TRUE(fp64Builder.has_value()) << fp64Builder.error().message;
    auto fp64Fock = fp64Builder->BuildFock(*densityTensor);
    ASSERT_TRUE(fp64Fock.has_value()) << fp64Fock.error().message;

    double maxDeviation = 0.0;

    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j < 7; ++j)
        {
            maxDeviation = std::max(maxDeviation, std::abs((*mixedFock)(i, j) - (*fp64Fock)(i, j)));
        }
    }

    // No slack: each element's error is bounded by the sum of its
    // density-weighted quartet bounds exactly (the sweep proves the
    // per-quartet bound itself with no slack), so maxDeviation <=
    // boundSum up to fp noise.
    EXPECT_LE(maxDeviation, boundSum + 1e-15)
        << "max deviation " << maxDeviation << " exceeds the certified sum " << boundSum;

    // The kNormal leg: the lane is silent (boundSum == 0) and the filtered
    // builds are bit-identical - the filter composes with the certified routing
    // at every preset.
    qcx::integrals::FockBuildOptions normalMixed = mixed;
    normalMixed.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    normalMixed.maxParallelChunks = 1;
    auto normalMixedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, normalMixed);
    ASSERT_TRUE(normalMixedBuilder.has_value()) << normalMixedBuilder.error().message;
    double normalBoundSum = -1.0;
    auto normalMixedFock = normalMixedBuilder->BuildFock(*densityTensor, &normalBoundSum);
    ASSERT_TRUE(normalMixedFock.has_value()) << normalMixedFock.error().message;
    EXPECT_EQ(normalBoundSum, 0.0);

    qcx::integrals::FockBuildOptions normalFp64Only = normalMixed;
    normalFp64Only.useCertifiedMixedPrecision = false;
    auto normalFp64Builder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, normalFp64Only);
    ASSERT_TRUE(normalFp64Builder.has_value()) << normalFp64Builder.error().message;
    auto normalFp64Fock = normalFp64Builder->BuildFock(*densityTensor);
    ASSERT_TRUE(normalFp64Fock.has_value()) << normalFp64Fock.error().message;

    EXPECT_EQ(
        // Bit-identity is the tested property (see the test comment above);
        // memcmp is its oracle - doubles do not have unique object
        // representations, but the contract here is exact bits.
        //
        // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison)
        std::memcmp(ToMatrix(*normalMixedFock).data(),
                    ToMatrix(*normalFp64Fock).data(),
                    std::size_t{7} * 7 * sizeof(double)),
        0);
}

// The split-mode certified-bound test re-run
// with the per-element filter active (the default; explicit here).
TEST(DirectJkFockBuilderTest, SplitModeStaysInsideCertifiedBoundWithFilterOn) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const std::size_t n = core->Shape()[0];
    auto densityTensor = ToTensor(PhysicalDensity(n));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    const Eigen::MatrixXd density = ToMatrix(*densityTensor);

    // The exact reference J: the dense unscreened supermatrix contracted
    // with the full density (screen = false is the retained reference path).
    qcx::integrals::EriDenseOptions eriOptions;
    eriOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    eriOptions.screen = false;
    auto eri = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis, eriOptions);
    ASSERT_TRUE(eri.has_value()) << eri.error().message;
    Eigen::MatrixXd jExact =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t mu = 0; mu < n; ++mu)
    {
        for (std::size_t nu = 0; nu < n; ++nu)
        {
            for (std::size_t l = 0; l < n; ++l)
            {
                for (std::size_t s = 0; s < n; ++s)
                {
                    jExact(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) +=
                        (*eri)(mu, nu, l, s) *
                        density(static_cast<Eigen::Index>(l), static_cast<Eigen::Index>(s));
                }
            }
        }
    }

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    options.buildCoulombOnly = true;
    options.usePerElementScreening = true;
    // Explicit, as in the sibling pin above: the lane's default is
    // hardware-dependent and this pin is about what the lane delivers.
    options.useCertifiedMixedPrecision = true;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;
    double certifiedBoundSum = 0.0;
    auto fock = builder->BuildFock(*densityTensor, &certifiedBoundSum);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;

    const Eigen::MatrixXd jPreset = (ToMatrix(*fock) - ToMatrix(*core)) / 2.0;
    const double maxAbsDJ = (jPreset - jExact).cwiseAbs().maxCoeff();
    EXPECT_GE(certifiedBoundSum, 0.0);
    EXPECT_LE(maxAbsDJ, certifiedBoundSum)
        << "the certified bound sum must dominate the delivered J error";
    EXPECT_LE(maxAbsDJ, 1e-5)
        << "physicality pin (the kLoose bound on this fixture family is 1e-6-scale)";
}

// The split-mode linear identity re-run with the
// filter active. Both split builds and the fused build drop the same terms
// from the same per-target thresholds (the J-only drops match the fused
// build's J sections, the K-only drops its K sections), so the identity
// coulombOnly + exchangeOnly - H = fused holds bit-identically in the serial
// configuration.
TEST(DirectJkFockBuilderTest, CoulombOnlyMinusExchangeOnlyIdentityWithFilterOn) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(2);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    qcx::integrals::FockBuildOptions fusedOptions;
    fusedOptions.useDensityScreening = true;
    fusedOptions.useCertifiedMixedPrecision = false;
    fusedOptions.usePerElementScreening = true;
    fusedOptions.maxParallelChunks = 1;
    auto fusedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, fusedOptions);
    ASSERT_TRUE(fusedBuilder.has_value()) << fusedBuilder.error().message;
    auto fusedFock = fusedBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(fusedFock.has_value()) << fusedFock.error().message;

    qcx::integrals::FockBuildOptions coulombOptions = fusedOptions;
    coulombOptions.buildCoulombOnly = true;
    auto coulombBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, coulombOptions);
    ASSERT_TRUE(coulombBuilder.has_value()) << coulombBuilder.error().message;
    auto coulombFock = coulombBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(coulombFock.has_value()) << coulombFock.error().message;

    qcx::integrals::FockBuildOptions exchangeOptions = fusedOptions;
    exchangeOptions.buildExchangeOnly = true;
    auto exchangeBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, exchangeOptions);
    ASSERT_TRUE(exchangeBuilder.has_value()) << exchangeBuilder.error().message;
    auto exchangeFock = exchangeBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(exchangeFock.has_value()) << exchangeFock.error().message;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            const double splitSum = (*coulombFock)(i, j) + (*exchangeFock)(i, j) - (*core)(i, j);
            EXPECT_NEAR(splitSum, (*fusedFock)(i, j), 1e-12) << "element (" << i << "," << j << ")";
        }
    }
}

// The batch-loop bit-identity pins below were authored for a team size of
// four (the slot-count arithmetic and the k = team charge both assume it),
// but a host can expose only ONE physical core, and a one-core host caches
// team = 1 on the first team-size read. Team 1 collapses the k-factor
// formula k = min(max(1, floor(remaining / cap)), team) to k = 1 always,
// which neither exercises the k > 1 merge path nor reproduces the authored
// geometry. Seeding OMP_NUM_THREADS to four before the first team-size read
// pins the Create-time team at four on every host (per-test ctest
// processes make the seed the first call; a whole-driver run on a >= 4-core
// host no-ops it - env seeding is the team_size_test.cpp precedent, and the
// footprint tests below adapt to the actually cached team, so the seed
// cannot poison them). The guard restores the previous value on exit
// (RAII, the cpu_topology_test.cpp shape).
// A scoped RAII guard restored in its destructor; it is never copied or moved.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
struct OmpNumThreadsSeedGuard {
    bool _seeded = false;
    std::string _previous;

    OmpNumThreadsSeedGuard() {
        const char* previous = std::getenv("OMP_NUM_THREADS");

        if (previous != nullptr)
        {
            _previous = previous;
        }
#ifdef _WIN32
        if (_putenv_s("OMP_NUM_THREADS", "4") == 0)
        {
            _seeded = true;
        }
#else
        if (setenv("OMP_NUM_THREADS", "4", 1) == 0)
        {
            _seeded = true;
        }
#endif
    }

    ~OmpNumThreadsSeedGuard() {
        if (_seeded)
        {
#ifdef _WIN32
            if (!_previous.empty())
            {
                _putenv_s("OMP_NUM_THREADS", _previous.c_str());
            } else
            {
                _putenv_s("OMP_NUM_THREADS", "");
            }
#else
            if (!_previous.empty())
            {
                setenv("OMP_NUM_THREADS", _previous.c_str(), 1);
            } else
            {
                unsetenv("OMP_NUM_THREADS");
            }
#endif
        }
    }

    // RAII guard: a copy or move would restore the env twice.
    OmpNumThreadsSeedGuard(const OmpNumThreadsSeedGuard&) = delete;
    OmpNumThreadsSeedGuard& operator=(const OmpNumThreadsSeedGuard&) = delete;
};

// The k = 1 vs
// k > 1 bit-identity pin. A fitting budgeted exchange run ALWAYS fires
// k = team (k = 1 is unreachable for a fitting budgeted exchange run - the
// k-factor charge authorizes a single slot only in a degenerate band), so
// the pair is a budgeted run at k = team against a same-cap no-budget
// legacy run at k = 1, both at a FIXED team size (distinct from the
// team-size pin, which holds the slots at 1). maxParallelChunks = 1 forces
// the per-batch serial fold on both rungs, so the only structural
// difference is the slot overlap; the tiny batch cap spans many batches
// (one s-shell quartet each) so the slots actually pull. The k > 1 merge
// chain serializes the shared-Fock accumulation in batch-major order, so
// every element's summation sequence - and the certified bound's - is
// identical to the k = 1 run's: bitwise pins, not tolerances.
TEST(DirectJkFockBuilderTest, BatchLoopBitIdenticalAcrossConcurrentSlots) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "parallel-schedule pin - not run in the fast smoke subset";
    }

    const OmpNumThreadsSeedGuard seedGuard;

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto densityTensor = ToTensor(PhysicalDensity(7));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    // The process-wide team ceiling is sticky - restore the pre-test value
    // on every exit path (a failed pin must not leak a capped team into the
    // rest of the suite). The ceiling alone cannot make the Create-time team
    // size deterministic on any host: a one-physical-core host caches
    // team = 1 at the first team-size read, so the seed guard above pins the
    // env-seeded first call at four (the authored geometry for the slot
    // arithmetic and the per-slot thread shares below).
    struct CeilingGuard {
        int restoreValue;

        explicit CeilingGuard(int restore) : restoreValue(restore) {}

        // RAII guard: a copy or move would restore the ceiling twice.
        CeilingGuard(const CeilingGuard&) = delete;
        CeilingGuard& operator=(const CeilingGuard&) = delete;
        CeilingGuard(CeilingGuard&&) = delete;
        CeilingGuard& operator=(CeilingGuard&&) = delete;

        ~CeilingGuard() {
            qcx::backend::SetOmpThreadCeiling(restoreValue);
        }
    };

    const CeilingGuard guard(qcx::backend::OmpThreadCeiling());
    qcx::backend::SetOmpThreadCeiling(4);

    // The shared option recipe: the tiny cap spans one s-shell quartet per
    // batch (the per-item overhead ~150 B plus the ~50-double payload puts
    // the second quartet past 1024 B), and maxParallelChunks = 1 forces the
    // per-batch serial fold (ChunkCountFor at one quartet per batch and one
    // authorized chunk) - the exact-summation precondition of the pin.
    qcx::integrals::FockBuildOptions options;
    options.maxBatchBytes = std::size_t{1024};
    options.maxParallelChunks = 1;

    // The k = 1 rung: the legacy no-budget builder (no fast-path decision,
    // the State default of one slot).
    auto legacyBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(legacyBuilder.has_value()) << legacyBuilder.error().message;
    EXPECT_FALSE(legacyBuilder->ModeInfo().has_value());
    double legacyBound = -1.0;
    auto legacyFock = legacyBuilder->BuildFock(*densityTensor, &legacyBound);
    ASSERT_TRUE(legacyFock.has_value()) << legacyFock.error().message;

    // The k = team rung: the budgeted fast path at the SAME cap. The budget
    // (64 MiB) fits the k = 4 charge at 1024 B unclamped (the fixed terms
    // are a few tens of KB), so the fired cap equals the legacy cap and
    // both runs partition the batches identically.
    auto budget = qcx::memory::WorkspaceBudget::Create(std::size_t{64} * 1024 * 1024);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;
    qcx::integrals::FockBuildOptions budgetedOptions = options;
    budgetedOptions.workspaceBudget = &*budget;
    auto budgetedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, budgetedOptions);
    ASSERT_TRUE(budgetedBuilder.has_value()) << budgetedBuilder.error().message;

    const std::optional<qcx::integrals::FockModeInfo>& info = budgetedBuilder->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, qcx::integrals::FockBuildMode::kFastPath);
    // k = min(max(1, floor(remaining / cap)), team) at the fired estimate:
    // 64 MiB / 1024 B saturates the fixed ceiling of 4.
    EXPECT_EQ(info->concurrentSlots, 4u);
    EXPECT_EQ(info->maxBatchBytes, options.maxBatchBytes);
    double budgetedBound = -1.0;
    auto budgetedFock = budgetedBuilder->BuildFock(*densityTensor, &budgetedBound);
    ASSERT_TRUE(budgetedFock.has_value()) << budgetedFock.error().message;

    // Bitwise pins: the batch-major merge order at k = 4 reproduces the
    // k = 1 sequential accumulation exactly (per-batch serial folds, the
    // gate's batch-ordered shared-Fock writes), so the Fock matrix and the
    // certified bound sum must agree in every bit.
    for (Eigen::Index i = 0; i < 7; ++i)
    {
        for (Eigen::Index j = 0; j < 7; ++j)
        {
            EXPECT_EQ(std::bit_cast<std::uint64_t>((*budgetedFock)(i, j)),
                      std::bit_cast<std::uint64_t>((*legacyFock)(i, j)))
                << "element (" << i << "," << j << ")";
        }
    }

    EXPECT_EQ(std::bit_cast<std::uint64_t>(budgetedBound),
              std::bit_cast<std::uint64_t>(legacyBound));
}

// The team-ceiling pin helpers: the process-wide team
// ceiling is sticky, so the guard restores the pre-test value on every
// exit path (a failed pin must not leak a capped team into the rest of the
// suite). The ceiling alone cannot make the Create-time team size
// deterministic on any host: a one-physical-core host caches team = 1
// at the first team-size read, so the OmpNumThreadsSeedGuard in the tests
// below pins the env-seeded first call at four (the authored geometry for
// the slot counts these guards authorize).
struct Slice1CeilingGuard {
    int restoreValue;

    explicit Slice1CeilingGuard(int restore) : restoreValue(restore) {}

    // RAII guard: a copy or move would restore the ceiling twice.
    Slice1CeilingGuard(const Slice1CeilingGuard&) = delete;
    Slice1CeilingGuard& operator=(const Slice1CeilingGuard&) = delete;
    Slice1CeilingGuard(Slice1CeilingGuard&&) = delete;
    Slice1CeilingGuard& operator=(Slice1CeilingGuard&&) = delete;

    ~Slice1CeilingGuard() {
        qcx::backend::SetOmpThreadCeiling(restoreValue);
    }
};

// The k-selection pin on the nested
// exchange half. The RI-J stack's exchange runs with a pre-reserved budget
// (the outer charged its own terms first - nesting order = reservation
// order, ri_engine.cpp attempt()), validatedSweepCount != 0 and
// buildExchangeOnly; the lift lets the nested's own decision fit its k-fold
// charge into the POST-reservation remaining exactly like a free-standing
// run. Here the remaining band (1 MiB) fits the k = team charge at the
// 1024-B cap unclamped, so the fired slot count is 4 - the pre-lift nested
// run forced k = 1 even with this slack. The bitwise pins repeat the same
// argument at the nested seam: same cap, same
// batch partition, per-batch serial folds at k = 4 merging batch-major -
// identical summation sequences in every bit.
TEST(DirectJkFockBuilderTest, NestedExchangeHalfFiresBudgetConditionalSlots) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "parallel-schedule pin - not run in the fast smoke subset";
    }

    const OmpNumThreadsSeedGuard seedGuard;

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto densityTensor = ToTensor(PhysicalDensity(7));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    const Slice1CeilingGuard guard(qcx::backend::OmpThreadCeiling());
    qcx::backend::SetOmpThreadCeiling(4);

    // The shared option recipe: the tiny cap spans one s-shell quartet per
    // batch and maxParallelChunks = 1 forces the per-batch serial fold - the
    // exact-summation precondition of the bitwise pins. buildExchangeOnly
    // is the nested half's pass shape (the RI-J stack's exchange).
    qcx::integrals::FockBuildOptions options;
    options.maxBatchBytes = std::size_t{1024};
    options.maxParallelChunks = 1;
    options.buildExchangeOnly = true;

    // The k = 1 rung: the legacy no-budget builder (no fast-path decision,
    // the State default of one slot).
    auto legacyBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(legacyBuilder.has_value()) << legacyBuilder.error().message;
    EXPECT_FALSE(legacyBuilder->ModeInfo().has_value());
    double legacyBound = -1.0;
    auto legacyFock = legacyBuilder->BuildFock(*densityTensor, &legacyBound);
    ASSERT_TRUE(legacyFock.has_value()) << legacyFock.error().message;

    // The nested half: the outer charged 63 MiB of the 64 MiB budget before
    // the exchange's Create (the reservation order), leaving the 1 MiB
    // exchange band. validatedSweepCount != 0 marks the outer's counted
    // sweep - the exclusion (i) all-survive refusal is skipped for the
    // stack the outer proved feasible.
    auto budget = qcx::memory::WorkspaceBudget::Create(std::size_t{64} * 1024 * 1024);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;
    ASSERT_TRUE(budget->Reserve(std::size_t{63} * 1024 * 1024));
    qcx::integrals::FockBuildOptions nestedOptions = options;
    nestedOptions.validatedSweepCount = 1;
    nestedOptions.workspaceBudget = &*budget;
    auto nestedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, nestedOptions);
    ASSERT_TRUE(nestedBuilder.has_value()) << nestedBuilder.error().message;

    const std::optional<qcx::integrals::FockModeInfo>& info = nestedBuilder->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, qcx::integrals::FockBuildMode::kFastPath);
    // k = min(max(1, floor(remaining / cap)), team) at the fired estimate:
    // the 1 MiB band over the 1024-B cap saturates the fixed ceiling of 4.
    EXPECT_EQ(info->concurrentSlots, 4u);
    // The clamp-origin team read: the k = 4 above is the
    // WHOLE ceiling-bounded team (team-clamped, not budget-clamped) - the
    // record of the read the decision clamped against.
    EXPECT_EQ(info->defaultTeamSize, 4u);
    EXPECT_EQ(info->maxBatchBytes, options.maxBatchBytes);
    // The charge fits its band exactly (the CWA: the stack's commits - the
    // outer's 63 MiB plus the exchange's predicted - sit at or below the
    // budget's capacity, never past it).
    EXPECT_EQ(budget->CommittedBytes(), std::size_t{63} * 1024 * 1024 + info->predictedBytes);
    EXPECT_LE(budget->CommittedBytes(), budget->CapacityBytes());
    double nestedBound = -1.0;
    // The capture carries the observed-concurrency marker too: the stats
    // out is observational, the Fock and the
    // bound bitwise pins below are unchanged by it.
    qcx::integrals::FockBuildStats nestedStats;
    auto nestedFock = nestedBuilder->BuildFock(*densityTensor, &nestedBound, &nestedStats);
    ASSERT_TRUE(nestedFock.has_value()) << nestedFock.error().message;
    // The merge gate actually ran 4 slots on this call (kEff saturates at
    // the authorized k - the batch count exceeds it), so the call's
    // eri/contract spans are slot-accumulated and the marker says so.
    EXPECT_EQ(nestedStats.concurrentSlots, 4u);

    // Bitwise pins: the batch-major merge order at k = 4 reproduces the
    // k = 1 sequential accumulation exactly (per-batch serial folds, the
    // gate's batch-ordered shared-Fock writes), so the Fock matrix and the
    // certified bound sum must agree in every bit.
    for (Eigen::Index i = 0; i < 7; ++i)
    {
        for (Eigen::Index j = 0; j < 7; ++j)
        {
            EXPECT_EQ(std::bit_cast<std::uint64_t>((*nestedFock)(i, j)),
                      std::bit_cast<std::uint64_t>((*legacyFock)(i, j)))
                << "element (" << i << "," << j << ")";
        }
    }

    EXPECT_EQ(std::bit_cast<std::uint64_t>(nestedBound), std::bit_cast<std::uint64_t>(legacyBound));
}

// The k = 1 floor at the nested seam. A one-thread team cannot
// host a second slot (k = min(max(1, floor(remaining / cap)), team) caps at
// team = 1), so even the slack band fires a single slot at the unclamped
// cap - the pre-lift behavior on this team, kept by the lift's min. Same
// bitwise pin against the same-cap k = 1 legacy run.
TEST(DirectJkFockBuilderTest, NestedExchangeHalfOnSingleThreadTeamKeepsSingleSlot) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "parallel-schedule pin - not run in the fast smoke subset";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto densityTensor = ToTensor(PhysicalDensity(7));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    const Slice1CeilingGuard guard(qcx::backend::OmpThreadCeiling());
    qcx::backend::SetOmpThreadCeiling(1);

    qcx::integrals::FockBuildOptions options;
    options.maxBatchBytes = std::size_t{1024};
    options.maxParallelChunks = 1;
    options.buildExchangeOnly = true;

    auto legacyBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(legacyBuilder.has_value()) << legacyBuilder.error().message;
    double legacyBound = -1.0;
    auto legacyFock = legacyBuilder->BuildFock(*densityTensor, &legacyBound);
    ASSERT_TRUE(legacyFock.has_value()) << legacyFock.error().message;

    auto budget = qcx::memory::WorkspaceBudget::Create(std::size_t{64} * 1024 * 1024);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;
    ASSERT_TRUE(budget->Reserve(std::size_t{63} * 1024 * 1024));
    qcx::integrals::FockBuildOptions nestedOptions = options;
    nestedOptions.validatedSweepCount = 1;
    nestedOptions.workspaceBudget = &*budget;
    auto nestedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, nestedOptions);
    ASSERT_TRUE(nestedBuilder.has_value()) << nestedBuilder.error().message;

    const std::optional<qcx::integrals::FockModeInfo>& info = nestedBuilder->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, qcx::integrals::FockBuildMode::kFastPath);
    EXPECT_EQ(info->concurrentSlots, 1u);
    // The clamp-origin team read records the ceiling of
    // 1 - k = min(..., team) could not rise above it.
    EXPECT_EQ(info->defaultTeamSize, 1u);
    EXPECT_EQ(info->maxBatchBytes, options.maxBatchBytes);
    EXPECT_LE(budget->CommittedBytes(), budget->CapacityBytes());
    double nestedBound = -1.0;
    qcx::integrals::FockBuildStats nestedStats;
    auto nestedFock = nestedBuilder->BuildFock(*densityTensor, &nestedBound, &nestedStats);
    ASSERT_TRUE(nestedFock.has_value()) << nestedFock.error().message;
    EXPECT_EQ(nestedStats.concurrentSlots, 1u);

    for (Eigen::Index i = 0; i < 7; ++i)
    {
        for (Eigen::Index j = 0; j < 7; ++j)
        {
            EXPECT_EQ(std::bit_cast<std::uint64_t>((*nestedFock)(i, j)),
                      std::bit_cast<std::uint64_t>((*legacyFock)(i, j)))
                << "element (" << i << "," << j << ")";
        }
    }

    EXPECT_EQ(std::bit_cast<std::uint64_t>(nestedBound), std::bit_cast<std::uint64_t>(legacyBound));
}

// The lift's self-gating when the saturated k = team charge
// does not fit the post-reservation band at the options' cap. The nested
// decision clamps ITS OWN cap down (never the outer's - the outer's terms
// were reserved before the exchange's decision read the remaining). The
// probe run establishes the unclamped k = 4 charge at the 1024-B cap
// (predicted = fixed + 16 x 1024; the 16-B-per-B slope = 4 threads + 4
// slots x 3 for the certified exchange band). The band is then the probe's
// predicted minus exactly 8 KiB. The descent lands in two steps: the
// 8-KiB deficit shrinks the cap by ceil(8192 / 16) = 512, and the batch
// starts-tables rows (1d2d880 - 8 B per start row per half, batch-count
// quantized) double the per-half batch count across the 1024 -> 512 step
// (+16 B), overshooting the band by that 16 B and forcing a second
// ceil(16 / 16) = 1 step to 511 - where the batch count is unchanged, so
// the re-estimated charge lands exactly on the band - fired, not refused,
// never past it. The output stays bit-identical to the unclamped k = 1
// run (the schedule-independence contract of the batch machinery: the cap
// change is value-neutral).
TEST(DirectJkFockBuilderTest, NestedExchangeHalfClampsItsOwnCapWithinItsBand) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "parallel-schedule pin - not run in the fast smoke subset";
    }

    const OmpNumThreadsSeedGuard seedGuard;

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto densityTensor = ToTensor(PhysicalDensity(7));
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    const Slice1CeilingGuard guard(qcx::backend::OmpThreadCeiling());
    qcx::backend::SetOmpThreadCeiling(4);

    qcx::integrals::FockBuildOptions options;
    options.maxBatchBytes = std::size_t{1024};
    options.maxParallelChunks = 1;
    options.buildExchangeOnly = true;

    // The k = 1 rung at the unclamped 1024-B cap: the bitwise reference.
    auto legacyBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(legacyBuilder.has_value()) << legacyBuilder.error().message;
    double legacyBound = -1.0;
    auto legacyFock = legacyBuilder->BuildFock(*densityTensor, &legacyBound);
    ASSERT_TRUE(legacyFock.has_value()) << legacyFock.error().message;

    // The probe: a slack 1 MiB band fires k = 4 at the unclamped cap - its
    // predictedBytes is the exact fixed-plus-16-KiB charge this test's
    // clamp math starts from.
    auto probeBudget = qcx::memory::WorkspaceBudget::Create(std::size_t{64} * 1024 * 1024);
    ASSERT_TRUE(probeBudget.has_value()) << probeBudget.error().message;
    ASSERT_TRUE(probeBudget->Reserve(std::size_t{63} * 1024 * 1024));
    qcx::integrals::FockBuildOptions probeOptions = options;
    probeOptions.validatedSweepCount = 1;
    probeOptions.workspaceBudget = &*probeBudget;
    auto probeBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, probeOptions);
    ASSERT_TRUE(probeBuilder.has_value()) << probeBuilder.error().message;
    ASSERT_TRUE(probeBuilder->ModeInfo().has_value());
    EXPECT_EQ(probeBuilder->ModeInfo()->concurrentSlots, 4u);
    EXPECT_EQ(probeBuilder->ModeInfo()->maxBatchBytes, options.maxBatchBytes);

    // The tight band: exactly 8 KiB less than the probe's unclamped charge.
    // The first estimate exceeds the band by that 8 KiB exactly; one clamp
    // step by the true slope (4 + 4 x 3 = 16 B per B of cap) shrinks the
    // cap by ceil(8192 / 16) = 512. The starts-tables rows (the 1d2d880
    // charge) are batch-count quantized: 1024 -> 512 doubles the per-half
    // batch count (1 -> 2 start rows, +16 B), so the re-estimate at 512
    // still overshoots the band by that 16 B and a second ceil(16 / 16)
    // step lands the cap at 511 - where the batch count is unchanged and
    // the estimate equals the band exactly.
    const std::size_t band = probeBuilder->ModeInfo()->predictedBytes - std::size_t{8} * 1024;
    auto budget = qcx::memory::WorkspaceBudget::Create(std::size_t{64} * 1024 * 1024);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;
    ASSERT_TRUE(budget->Reserve(budget->CapacityBytes() - band));
    qcx::integrals::FockBuildOptions nestedOptions = options;
    nestedOptions.validatedSweepCount = 1;
    nestedOptions.workspaceBudget = &*budget;
    auto nestedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, nestedOptions);
    ASSERT_TRUE(nestedBuilder.has_value()) << nestedBuilder.error().message;

    const std::optional<qcx::integrals::FockModeInfo>& info = nestedBuilder->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, qcx::integrals::FockBuildMode::kFastPath);
    EXPECT_EQ(info->concurrentSlots, 4u);
    EXPECT_EQ(info->maxBatchBytes, std::size_t{511});
    EXPECT_EQ(info->predictedBytes, band);
    // The CWA is exact at the fired estimate: the exchange's charge fills
    // the band, and the stack's total commit sits at the capacity.
    EXPECT_EQ(budget->CommittedBytes(), budget->CapacityBytes());
    double nestedBound = -1.0;
    auto nestedFock = nestedBuilder->BuildFock(*densityTensor, &nestedBound);
    ASSERT_TRUE(nestedFock.has_value()) << nestedFock.error().message;

    // Bitwise pins: the clamped-cap run's partitions differ from the legacy
    // run's (cap 512 vs cap 1024) yet the batch-major merge order still
    // reproduces the k = 1 sequential accumulation exactly - cross-cap and
    // cross-slot identical in every bit.
    for (Eigen::Index i = 0; i < 7; ++i)
    {
        for (Eigen::Index j = 0; j < 7; ++j)
        {
            EXPECT_EQ(std::bit_cast<std::uint64_t>((*nestedFock)(i, j)),
                      std::bit_cast<std::uint64_t>((*legacyFock)(i, j)))
                << "element (" << i << "," << j << ")";
        }
    }

    EXPECT_EQ(std::bit_cast<std::uint64_t>(nestedBound), std::bit_cast<std::uint64_t>(legacyBound));
}

// ---------------------------------------------------------------------------
// The LIVE class-table measurement.
// ---------------------------------------------------------------------------
//
// The admission gate charges internal::ClassTableBytes - a never-under
// CEILING computed before any table exists - against the budget's remaining
// bytes and DISENGAGES the class path when the charge does not fit. The
// hypothesis: since the root restructure (2026-08-31) BuildPairClasses
// stores the Schwarz-screened class pairs with EMPTY orbit vectors and the
// members generate on demand per REACHED class pair, so the ceiling no
// longer describes the shape the path materializes - and the gate would drop
// the symmetry of every system big enough to need it.
//
// The instrument is qcx::memory::TagCurrentBytes(kClassTable): the
// class_table family's live request bytes, which every container of the
// table attributes to (the tagged allocator). It is the production family;
// no new code, no timing.

// The N-water STO-3G cluster at 20 Bohr monomer spacing, in the canonical
// atom order (Molecule::Create renumbers by Z then position): the 2N H
// functions first (H of water k at -x in function k, at +x in function
// N + k), then per water the O s1 (2N + 5k), O s2 (2N + 5k + 1) and the O p
// shell (2N + 5k + 2 py, + 3 pz, + 4 px) - the shape FootprintTest's cluster
// fixture and its gate test pin.
qcx::Result<qcx::molecule::Molecule> MakeWaterCluster(std::size_t count) {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3 * count, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    std::vector<qcx::molecule::Atom> atoms;

    for (std::size_t k = 0; k < count; ++k)
    {
        const double y = 20.0 * static_cast<double>(k);
        (*coordinates)(3 * k + 0, 0) = 0.0;
        (*coordinates)(3 * k + 0, 1) = y;
        (*coordinates)(3 * k + 0, 2) = 0.0;
        (*coordinates)(3 * k + 1, 0) = 1.430428808474167;
        (*coordinates)(3 * k + 1, 1) = y + 1.107157044080814;
        (*coordinates)(3 * k + 1, 2) = 0.0;
        (*coordinates)(3 * k + 2, 0) = -1.430428808474167;
        (*coordinates)(3 * k + 2, 1) = y + 1.107157044080814;
        (*coordinates)(3 * k + 2, 2) = 0.0;
        atoms.push_back(qcx::molecule::Atom{"O", 8, 0.0});
        atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});
        atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});
    }

    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

// The cluster's x-mirror reduction (identity plus the (x -> -x) mirror):
// it swaps each water's two H functions and flips the O p_x sign. Built by
// brute force over the layout the fixture documents (7 functions per water,
// H first) - and the class-vs-plain 1e-12 comparison below is what
// validates it: a permutation that is not a true symmetry of the cluster
// cannot reproduce the plain Fock.
qcx::integrals::SymmetryReduction MakeClusterMirrorReduction(std::size_t functionCount) {
    const std::size_t monomerCount = functionCount / 7;
    std::vector<std::size_t> identity(functionCount);
    std::vector<std::size_t> swap(functionCount);
    std::vector<int> signsIdentity(functionCount, 1);
    std::vector<int> signsMirror(functionCount, 1);

    for (std::size_t i = 0; i < functionCount; ++i)
    {
        identity[i] = i;
        swap[i] = i;
    }

    for (std::size_t k = 0; k < monomerCount; ++k)
    {
        swap[k] = k + monomerCount;
        swap[k + monomerCount] = k;
        signsMirror[2 * monomerCount + 5 * k + 4] = -1;
    }

    qcx::integrals::SymmetryReduction reduction;
    reduction.permutation = {std::move(identity), std::move(swap)};
    reduction.sign = {std::move(signsIdentity), std::move(signsMirror)};
    reduction.groupOrder = 2;
    reduction.isTrivial = false;

    return reduction;
}

// The canonical member quartets the table's KEPT class pairs cover: the
// exact count of ClassOrbitMember entries the on-demand generation can ever
// materialize (a reached pair fills its whole member-quartet set), summed
// over the kept pairs. The identity sum over ALL class pairs is the full
// canonical quartet count nPairs(nPairs + 1)/2 - the model check below.
std::size_t KeptMemberQuartets(const qcx::integrals::PairClassTable& table) {
    std::size_t total = 0;

    for (const qcx::integrals::ClassPair& classPair : table.classPairs)
    {
        const std::size_t membersP = table.classes[classPair.p].members.size();
        const std::size_t membersQ = table.classes[classPair.q].members.size();
        total += (classPair.p == classPair.q) ? membersP * (membersP + 1) / 2 : membersP * membersQ;
    }

    return total;
}

// Enables the attribution instrument for one measurement and restores the
// prior state on every exit path (the instrument is process-global, and a
// failed ASSERT_ must not leave it on for the rest of the binary's tests).
// A scoped RAII guard; copy is already deleted and it is never moved.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class ScopedInstrumentEnable {
public:
    ScopedInstrumentEnable() : _wasEnabled(qcx::memory::AllocationInstrumentEnabled()) {
        if (!_wasEnabled)
        {
            const auto enabled =
                qcx::memory::AllocationInstrumentEnable(qcx::memory::AllocationInstrumentOptions{});
            _enabled = enabled.has_value();
        }
    }

    ~ScopedInstrumentEnable() {
        if (_enabled && !_wasEnabled)
        {
            // The Result is deliberately discarded: a destructor cannot report it.
            // NOLINTNEXTLINE(bugprone-unused-return-value)
            (void)qcx::memory::AllocationInstrumentDisable();
        }
    }

    ScopedInstrumentEnable(const ScopedInstrumentEnable&) = delete;
    ScopedInstrumentEnable& operator=(const ScopedInstrumentEnable&) = delete;

    bool Ok() const noexcept {
        return _wasEnabled || _enabled;
    }

private:
    bool _wasEnabled = false;
    bool _enabled = true;
};

TEST(DirectJkFockBuilderTest, LiveClassTableBytesAgainstTheNeverUnderCeiling) {
    const ScopedInstrumentEnable instrument;
    ASSERT_TRUE(instrument.Ok()) << "the allocation attribution instrument did not enable";

    auto molecule = MakeWaterCluster(24);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;

    const std::size_t nPairs = pairList->pairs.size();
    const std::size_t nFunctions = pairList->functionCount;
    const std::size_t ceilingBytes = qcx::integrals::internal::ClassTableBytes(nPairs, nFunctions);
    const std::size_t memberQuartets = nPairs * (nPairs + 1) / 2;
    const qcx::integrals::SymmetryReduction reduction = MakeClusterMirrorReduction(nFunctions);

    std::printf("\n  CLASS-TABLE GATE: the live table vs the never-under ceiling\n");
    std::printf("    nPairs %zu, nFunctions %zu, canonical member quartets %zu\n",
                nPairs,
                nFunctions,
                memberQuartets);
    std::printf("    ClassTableBytes ceiling      %15zu B  %9.4f GiB\n",
                ceilingBytes,
                static_cast<double>(ceilingBytes) / (1024.0 * 1024.0 * 1024.0));

    // The Create-time table alone: BuildPairClasses on the real reduction
    // and the production Schwarz bounds, at the neighbor-list threshold the
    // builder screens with.
    const double threshold =
        qcx::integrals::SchwarzThreshold(qcx::integrals::AccuracyPreset::kNormal) *
        qcx::integrals::internal::kNeighborListSlack;
    const std::uint64_t beforeCreate =
        qcx::memory::TagCurrentBytes(qcx::memory::AllocationTag::kClassTable);
    std::uint64_t createLive = 0;
    std::size_t keptQuartets = 0;
    std::size_t keptClassPairs = 0;

    {
        auto table = qcx::integrals::BuildPairClasses(reduction, *pairList, *schwarz, threshold);
        ASSERT_TRUE(table.has_value()) << table.error().message;
        createLive =
            qcx::memory::TagCurrentBytes(qcx::memory::AllocationTag::kClassTable) - beforeCreate;
        keptQuartets = KeptMemberQuartets(*table);
        keptClassPairs = table->classPairs.size();
        std::printf(
            "    classes %zu, kept class pairs %zu, kept member quartets %zu (%.3f%% of all)\n",
            table->classes.size(),
            table->classPairs.size(),
            keptQuartets,
            100.0 * static_cast<double>(keptQuartets) / static_cast<double>(memberQuartets));
        std::printf("    LIVE at Create               %15llu B  %9.4f GiB\n",
                    static_cast<unsigned long long>(createLive),
                    static_cast<double>(createLive) / (1024.0 * 1024.0 * 1024.0));
    }

    // The gate's own charge, from the same decomposition the table build
    // uses: the never-under materialization bound the fix gates on.
    const auto counts =
        qcx::integrals::CountClassMaterialization(reduction, *pairList, *schwarz, threshold);
    ASSERT_TRUE(counts.has_value()) << counts.error().message;
    const std::size_t materializationBytes =
        qcx::integrals::internal::ClassMaterializationBytes(*counts, nPairs, nFunctions);
    std::printf("    the materialization bound    %15zu B  %9.4f GiB\n",
                materializationBytes,
                static_cast<double>(materializationBytes) / (1024.0 * 1024.0 * 1024.0));
    std::printf("    counted: %zu class pairs, %zu member quartets, %zu classes\n",
                counts->classPairs,
                counts->memberQuartets,
                counts->classes);

    // The never-under property, element by element: every counted quantity
    // bounds the table it describes, and the byte charge bounds the live
    // allocation the family reports.
    EXPECT_GE(counts->classPairs, keptClassPairs);
    EXPECT_GE(counts->memberQuartets, keptQuartets);
    EXPECT_GE(counts->classes, 1u);

    const std::uint64_t afterCreateFree =
        qcx::memory::TagCurrentBytes(qcx::memory::AllocationTag::kClassTable);
    std::printf("    LIVE after the table frees  %15llu B\n",
                static_cast<unsigned long long>(afterCreateFree));

    ASSERT_GT(createLive, 0ull)
        << "the class_table family reported nothing - the instrument is not measuring";
    EXPECT_LT(createLive, ceilingBytes)
        << "the live Create-time table is NOT below the never-under ceiling";

    // The plain-path reference: the measurement is only meaningful if the
    // reduction is a true symmetry (a wrong hand-built permutation would
    // silently measure a bogus table), so the engaged leg's Fock must
    // reproduce this one element-wise.
    const Eigen::MatrixXd density = PhysicalDensity(nFunctions);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value());

    qcx::integrals::FockBuildOptions plainOptions;
    plainOptions.useCertifiedMixedPrecision = false;
    plainOptions.maxBatchBytes = std::size_t{2} * 1024 * 1024;
    plainOptions.maxParallelChunks = 1;
    auto plain =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, plainOptions);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    auto plainFock = plain->BuildFock(*densityTensor);
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;

    // The production path: the engaged leg at a budget that fits the
    // ceiling, measured after Create (the stored table) and after BuildFock
    // (the accumulated on-demand orbit fills - the generations are cached,
    // so the live table is the whole run's materialization, never just the
    // current pass).
    std::uint64_t afterBuilderCreate = 0;
    std::uint64_t afterBuild = 0;
    std::uint64_t peak = 0;

    {
        auto budget = qcx::memory::WorkspaceBudget::Create(16ull * 1024 * 1024 * 1024);
        ASSERT_TRUE(budget.has_value());
        qcx::integrals::FockBuildOptions options;
        options.workspaceBudget = &*budget;
        options.symmetryReduction = &reduction;
        options.useCertifiedMixedPrecision = false;
        options.maxBatchBytes = std::size_t{2} * 1024 * 1024;
        options.maxParallelChunks = 1;
        auto builder =
            qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
        ASSERT_TRUE(builder.has_value()) << builder.error().message;
        const std::optional<qcx::integrals::FockModeInfo>& info = builder->ModeInfo();
        ASSERT_TRUE(info.has_value());
        EXPECT_FALSE(info->classPathDisengaged);
        std::printf("    the gate CHARGED               %15zu B  (ceiling %zu)\n",
                    info->classTableBytes,
                    ceilingBytes);
        EXPECT_LE(info->classTableBytes, ceilingBytes)
            << "the gate charges more than the never-under ceiling";

        afterBuilderCreate = qcx::memory::TagCurrentBytes(qcx::memory::AllocationTag::kClassTable);

        auto fock = builder->BuildFock(*densityTensor);
        ASSERT_TRUE(fock.has_value()) << fock.error().message;

        // The class path's own correctness at this scale: same screened
        // quartets, same Fock.
        EXPECT_NEAR((ToMatrix(*fock) - ToMatrix(*plainFock)).norm(), 0.0, 1e-12);

        afterBuild = qcx::memory::TagCurrentBytes(qcx::memory::AllocationTag::kClassTable);
        peak = qcx::memory::TagPeakBytes(qcx::memory::AllocationTag::kClassTable);
    }

    std::printf("    LIVE after Create (builder)  %15llu B  %9.4f GiB\n",
                static_cast<unsigned long long>(afterBuilderCreate),
                static_cast<double>(afterBuilderCreate) / (1024.0 * 1024.0 * 1024.0));
    std::printf("    LIVE after BuildFock         %15llu B  %9.4f GiB\n",
                static_cast<unsigned long long>(afterBuild),
                static_cast<double>(afterBuild) / (1024.0 * 1024.0 * 1024.0));
    std::printf("    PEAK of the family           %15llu B  %9.4f GiB\n",
                static_cast<unsigned long long>(peak),
                static_cast<double>(peak) / (1024.0 * 1024.0 * 1024.0));
    std::printf("    the ceiling is %.1fx the live-after-BuildFock table\n",
                static_cast<double>(ceilingBytes) / static_cast<double>(afterBuild));
    std::printf("    the materialization bound is %.2fx the live-after-BuildFock table\n",
                static_cast<double>(materializationBytes) / static_cast<double>(afterBuild));
    std::fflush(stdout);

    // The never-under property of the charge the fix gates on: the bound
    // covers what the path actually held at its peak, not just its
    // end state.
    EXPECT_GE(materializationBytes, afterBuild)
        << "the materialization bound under-charges the live table";
    EXPECT_GE(materializationBytes, peak) << "the materialization bound under-charges the peak";
    EXPECT_LT(afterBuild, ceilingBytes) << "the whole run's live table exceeds the ceiling";
}

// The same measurement's second scale: the class-path materialization bound
// against the eager ceiling at the c60 STO-3G SHELL shape - 60 waters (300
// shells, 45,150 canonical pairs, 420 functions), the scale the eager table
// quotes as ~228 GiB of eager class table. No Fock is built and no orbit is
// generated: the counter never materializes the table, which is the point -
// the admission question is answered at a scale whose eager table could not
// be built at all.
TEST(DirectJkFockBuilderTest, ClassMaterializationBoundAtTheC60ShellShape) {
    auto molecule = MakeWaterCluster(60);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;

    const std::size_t nPairs = pairList->pairs.size();
    const std::size_t nFunctions = pairList->functionCount;
    const std::size_t ceilingBytes = qcx::integrals::internal::ClassTableBytes(nPairs, nFunctions);
    const qcx::integrals::SymmetryReduction reduction = MakeClusterMirrorReduction(nFunctions);
    const double threshold =
        qcx::integrals::SchwarzThreshold(qcx::integrals::AccuracyPreset::kNormal) *
        qcx::integrals::internal::kNeighborListSlack;
    const auto counts =
        qcx::integrals::CountClassMaterialization(reduction, *pairList, *schwarz, threshold);
    ASSERT_TRUE(counts.has_value()) << counts.error().message;
    const std::size_t materializationBytes =
        qcx::integrals::internal::ClassMaterializationBytes(*counts, nPairs, nFunctions);
    const double gib = 1024.0 * 1024.0 * 1024.0;

    std::printf("\n  CLASS-TABLE GATE at the c60 shell shape\n");
    std::printf("    nPairs %zu, nFunctions %zu, canonical member quartets %zu\n",
                nPairs,
                nFunctions,
                nPairs * (nPairs + 1) / 2);
    std::printf("    ClassTableBytes ceiling       %15zu B  %9.4f GiB\n",
                ceilingBytes,
                static_cast<double>(ceilingBytes) / gib);
    std::printf("    the materialization bound     %15zu B  %9.4f GiB\n",
                materializationBytes,
                static_cast<double>(materializationBytes) / gib);
    std::printf("    counted: %zu class pairs, %zu member quartets, %zu classes\n",
                counts->classPairs,
                counts->memberQuartets,
                counts->classes);
    std::printf("    the ceiling is %.1fx the materialization bound\n",
                static_cast<double>(ceilingBytes) / static_cast<double>(materializationBytes));
    std::fflush(stdout);

    EXPECT_LT(materializationBytes, ceilingBytes)
        << "the materialization bound is not below the eager ceiling at this scale";
}

} // namespace
