// The direct-path restricted open-shell pins: RunRohfScf wired to the split
// DirectJkFockBuilder modes through the UhfFockBuilderFn seam (the same
// adapter the UHF direct pins use - the seam is unchanged), against the
// acceptance table of the ROHF/ROKS spike:
//
//   1. OH/STO-3G doublet converges at the external reference
//      (-74.36153805173, pyscf 2.14.0) with <S^2> = 0.75 exactly, and its
//      returned state satisfies the three Roothaan stationarity conditions;
//   2. triplet H2 (nBeta = 0) equals UHF for the same determinant;
//   3. singlet H2 ROHF equals RHF;
//   4. singlet H2 ROKS (Slater) equals RKS and UKS and their shared pin;
//   5. OH doublet ROKS sits at or above UKS (the restricted determinant is
//      inside the unrestricted variational space).
//
// The ROKS seam is the per-spin one (the UKS composition's own shape): the
// Coulomb half is built from rho = (D_alpha + D_beta)/2, the XC engine's
// per-spin potentials enter each spin Fock, and the contribution is Exc -
// exactly the composition the driver wires for uks, so a restricted run
// through it is ROKS with no new seam. Slater is a pure functional
// (c_HF = 0), so no exchange half is built.

#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/rohf.hpp"
#include "qcx/scf/uhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// The external references (pyscf 2.14.0, conv_tol 1e-12, minao guess), quoted
// exactly as driver/tests/run_driver_test.cpp records them: OH at
// R = 1.8326 Bohr, and the H2 Slater RKS/UKS pin of the driver's own cells.
constexpr double kOhRohfReferenceEnergy = -74.36153805173;
constexpr double kOhUhfReferenceEnergy = -74.36264494805;
constexpr double kH2SlaterReferenceEnergy = -1.025009958088;

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

// OH in STO-3G: O at the origin, H at z = 1.8326 Bohr - the driver's OH
// fixture (0.9697701567008378 Angstrom through kAngstromToBohr).
qcx::Result<qcx::molecule::Molecule> MakeOhSto3g(int multiplicity) {
    auto coordinates = CpuTensor2::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 0.0;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 1.8326;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        multiplicity);
}

// H2 in STO-3G with a chosen multiplicity (the triplet is the nBeta = 0
// limit's fixture).
qcx::Result<qcx::molecule::Molecule> MakeH2Sto3gWithMultiplicity(int multiplicity) {
    auto coordinates = CpuTensor2::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.4;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        multiplicity);
}

// The split-builder pair the three runners below share.
struct DirectBuilders {
    qcx::integrals::DirectJkFockBuilder coulomb;
    qcx::integrals::DirectJkFockBuilder exchange;
};

qcx::Result<DirectBuilders> MakeDirectBuilders(const qcx::molecule::Molecule& molecule,
                                               const qcx::basisset::BasisSet& basisSet,
                                               const qcx::integrals::FockBuildOptions& fockOptions,
                                               const Eigen::MatrixXd& core) {
    auto coreTensor = ToTensor(core);

    if (!coreTensor.has_value())
    {
        return std::unexpected(coreTensor.error());
    }

    qcx::integrals::FockBuildOptions coulombOptions = fockOptions;
    coulombOptions.buildCoulombOnly = true;
    auto coulombBuilder = qcx::integrals::DirectJkFockBuilder::Create(
        molecule, basisSet, *coreTensor, coulombOptions);

    if (!coulombBuilder.has_value())
    {
        return std::unexpected(coulombBuilder.error());
    }

    qcx::integrals::FockBuildOptions exchangeOptions = fockOptions;
    exchangeOptions.buildExchangeOnly = true;
    auto exchangeBuilder = qcx::integrals::DirectJkFockBuilder::Create(
        molecule, basisSet, *coreTensor, exchangeOptions);

    if (!exchangeBuilder.has_value())
    {
        return std::unexpected(exchangeBuilder.error());
    }

    return DirectBuilders{std::move(*coulombBuilder), std::move(*exchangeBuilder)};
}

// The UhfFockBuilderFn adapter (the direct_uhf_test assembly): the Coulomb
// half takes the half-summed density (its J loop bakes in the 2.0 factor and
// delivers H + J[D]), the exchange half takes each raw spin density and
// delivers H - K[D_sigma]; the double-H is subtracted back per channel.
qcx::scf::UhfFockBuilderFn MakeDirectUhfFockBuilder(const DirectBuilders& builders,
                                                    const Eigen::MatrixXd& core) {
    return [coulomb = builders.coulomb, exchange = builders.exchange, core](
               const Eigen::MatrixXd& dAlpha, const Eigen::MatrixXd& dBeta)
               -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        auto halfSum = ToTensor(0.5 * (dAlpha + dBeta));

        if (!halfSum.has_value())
        {
            return std::unexpected(halfSum.error());
        }

        auto coulombHalf = coulomb.BuildFock(*halfSum);

        if (!coulombHalf.has_value())
        {
            return std::unexpected(coulombHalf.error());
        }

        auto alphaTensor = ToTensor(dAlpha);

        if (!alphaTensor.has_value())
        {
            return std::unexpected(alphaTensor.error());
        }

        auto exchangeAlpha = exchange.BuildFock(*alphaTensor);

        if (!exchangeAlpha.has_value())
        {
            return std::unexpected(exchangeAlpha.error());
        }

        auto betaTensor = ToTensor(dBeta);

        if (!betaTensor.has_value())
        {
            return std::unexpected(betaTensor.error());
        }

        auto exchangeBeta = exchange.BuildFock(*betaTensor);

        if (!exchangeBeta.has_value())
        {
            return std::unexpected(exchangeBeta.error());
        }

        const Eigen::MatrixXd coulombMatrix = ToMatrix(*coulombHalf);
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{
            coulombMatrix + ToMatrix(*exchangeAlpha) - core,
            coulombMatrix + ToMatrix(*exchangeBeta) - core};
    };
}

// The closed-shell FockBuilderFn from the same split builders:
// F = H + J[D] - K[D]/2 = coulombHalf(D/2) + (exchangeHalf(D) - H)/2.
qcx::scf::FockBuilderFn MakeDirectRhfFockBuilder(const DirectBuilders& builders,
                                                 const Eigen::MatrixXd& core) {
    return [coulomb = builders.coulomb, exchange = builders.exchange, core](
               const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        auto halfDensity = ToTensor(0.5 * density);

        if (!halfDensity.has_value())
        {
            return std::unexpected(halfDensity.error());
        }

        auto fullDensity = ToTensor(density);

        if (!fullDensity.has_value())
        {
            return std::unexpected(fullDensity.error());
        }

        auto coulombHalf = coulomb.BuildFock(*halfDensity);

        if (!coulombHalf.has_value())
        {
            return std::unexpected(coulombHalf.error());
        }

        auto exchangeHalf = exchange.BuildFock(*fullDensity);

        if (!exchangeHalf.has_value())
        {
            return std::unexpected(exchangeHalf.error());
        }

        return ToMatrix(*coulombHalf) + 0.5 * (ToMatrix(*exchangeHalf) - core);
    };
}

// The per-spin Kohn-Sham seam of one pure functional - the composition the
// driver wires for uks (MakeUksSeam with c_HF = 0): every callback goes
// through one cache, keyed on the total for the Coulomb half and on the exact
// pair for the XC evaluation, so the three halves are built once per density
// no matter which asks first.
struct KohnShamSeam {
    qcx::scf::UhfFockBuilderFn fock;
    qcx::scf::CoulombFn coulomb;
    qcx::scf::UhfEnergyContributionFn contribution;
};

qcx::Result<KohnShamSeam> MakeSlaterUksSeam(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& core,
    const qcx::integrals::DirectJkFockBuilder& coulombBuilder) {
    auto engine = qcx::grid::XcGridEngine::Create(molecule, basisSet, "slater");

    if (!engine.has_value())
    {
        return std::unexpected(engine.error());
    }

    auto shared = std::make_shared<qcx::grid::XcGridEngine>(std::move(*engine));

    struct Cache {
        Eigen::MatrixXd total;
        Eigen::MatrixXd coulombHalf;
        bool coulombValid = false;
        Eigen::MatrixXd densityAlpha;
        Eigen::MatrixXd densityBeta;
        qcx::grid::XcEvaluation xc;
        bool pairValid = false;
    };

    auto cache = std::make_shared<Cache>();

    const auto ensureCoulomb = [cache,
                                coulombBuilder](const Eigen::MatrixXd& total) -> qcx::Result<void> {
        if (cache->coulombValid && cache->total.rows() == total.rows() && cache->total == total)
        {
            return {};
        }

        auto rho = ToTensor(0.5 * total);

        if (!rho.has_value())
        {
            return std::unexpected(rho.error());
        }

        auto half = coulombBuilder.BuildFock(*rho);

        if (!half.has_value())
        {
            return std::unexpected(half.error());
        }

        cache->total = total;
        cache->coulombHalf = ToMatrix(*half);
        cache->coulombValid = true;
        return {};
    };

    const auto ensurePair = [cache,
                             shared](const Eigen::MatrixXd& densityAlpha,
                                     const Eigen::MatrixXd& densityBeta) -> qcx::Result<void> {
        if (cache->pairValid && cache->densityAlpha.rows() == densityAlpha.rows() &&
            cache->densityAlpha == densityAlpha && cache->densityBeta == densityBeta)
        {
            return {};
        }

        auto evaluation = shared->EvaluateScreened(densityAlpha, densityBeta, 0.0);

        if (!evaluation.has_value())
        {
            return std::unexpected(evaluation.error());
        }

        cache->densityAlpha = densityAlpha;
        cache->densityBeta = densityBeta;
        cache->xc = std::move(*evaluation);
        cache->pairValid = true;
        return {};
    };

    KohnShamSeam seam;

    seam.fock = [ensureCoulomb, ensurePair, cache](const Eigen::MatrixXd& densityAlpha,
                                                   const Eigen::MatrixXd& densityBeta)
        -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        const auto coulombFilled = ensureCoulomb(densityAlpha + densityBeta);

        if (!coulombFilled.has_value())
        {
            return std::unexpected(coulombFilled.error());
        }

        const auto pairFilled = ensurePair(densityAlpha, densityBeta);

        if (!pairFilled.has_value())
        {
            return std::unexpected(pairFilled.error());
        }

        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{
            cache->coulombHalf + cache->xc.potentialAlpha,
            cache->coulombHalf + cache->xc.potentialBeta};
    };

    seam.coulomb =
        [ensureCoulomb, cache, core](const Eigen::MatrixXd& total) -> qcx::Result<Eigen::MatrixXd> {
        const auto filled = ensureCoulomb(total);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        return cache->coulombHalf - core;
    };

    seam.contribution = [ensurePair,
                         cache](const Eigen::MatrixXd& densityAlpha,
                                const Eigen::MatrixXd& densityBeta) -> qcx::Result<double> {
        const auto filled = ensurePair(densityAlpha, densityBeta);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        return cache->xc.energy;
    };

    return seam;
}

// The closed-shell Kohn-Sham seam (the RKS counterpart of the driver's
// MakeRksSeam with c_HF = 0): the Fock is the Coulomb half plus HALF the sum
// of the engine's per-spin potentials, the closed-shell chain rule.
struct ClosedShellSeam {
    qcx::scf::FockBuilderFn fock;
    qcx::scf::CoulombFn coulomb;
    qcx::scf::EnergyContributionFn contribution;
};

qcx::Result<ClosedShellSeam> MakeSlaterRksSeam(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& core,
    const qcx::integrals::DirectJkFockBuilder& coulombBuilder) {
    auto engine = qcx::grid::XcGridEngine::Create(molecule, basisSet, "slater");

    if (!engine.has_value())
    {
        return std::unexpected(engine.error());
    }

    auto shared = std::make_shared<qcx::grid::XcGridEngine>(std::move(*engine));

    struct Cache {
        Eigen::MatrixXd density;
        Eigen::MatrixXd coulombHalf;
        qcx::grid::XcEvaluation xc;
        bool valid = false;
    };

    auto cache = std::make_shared<Cache>();

    const auto ensure =
        [cache, shared, coulombBuilder](const Eigen::MatrixXd& density) -> qcx::Result<void> {
        if (cache->valid && cache->density.rows() == density.rows() && cache->density == density)
        {
            return {};
        }

        const Eigen::MatrixXd rho = 0.5 * density;
        auto rhoTensor = ToTensor(rho);

        if (!rhoTensor.has_value())
        {
            return std::unexpected(rhoTensor.error());
        }

        auto half = coulombBuilder.BuildFock(*rhoTensor);

        if (!half.has_value())
        {
            return std::unexpected(half.error());
        }

        auto evaluation = shared->EvaluateScreened(rho, rho, 0.0);

        if (!evaluation.has_value())
        {
            return std::unexpected(evaluation.error());
        }

        cache->density = density;
        cache->coulombHalf = ToMatrix(*half);
        cache->xc = std::move(*evaluation);
        cache->valid = true;
        return {};
    };

    ClosedShellSeam seam;

    seam.fock = [ensure, cache](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        const auto filled = ensure(density);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        return cache->coulombHalf + 0.5 * (cache->xc.potentialAlpha + cache->xc.potentialBeta);
    };

    seam.coulomb =
        [ensure, cache, core](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        const auto filled = ensure(density);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        return cache->coulombHalf - core;
    };

    seam.contribution = [ensure, cache](const Eigen::MatrixXd& density) -> qcx::Result<double> {
        const auto filled = ensure(density);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        return cache->xc.energy;
    };

    return seam;
}

// The three runners: same integrals, same builders, one loop each.
qcx::Result<qcx::scf::RohfResult> RunDirectRohf(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const qcx::integrals::FockBuildOptions& fockOptions,
                                                const qcx::scf::RohfOptions& scfOptions) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto core = BuildCoreHamiltonian(molecule, basisSet);

    if (!core.has_value())
    {
        return std::unexpected(core.error());
    }

    auto builders = MakeDirectBuilders(molecule, basisSet, fockOptions, ToMatrix(*core));

    if (!builders.has_value())
    {
        return std::unexpected(builders.error());
    }

    return qcx::scf::RunRohfScf(molecule,
                                ToMatrix(*overlap),
                                ToMatrix(*core),
                                scfOptions,
                                MakeDirectUhfFockBuilder(*builders, ToMatrix(*core)));
}

qcx::Result<qcx::scf::UhfResult> RunDirectUhf(const qcx::molecule::Molecule& molecule,
                                              const qcx::basisset::BasisSet& basisSet,
                                              const qcx::integrals::FockBuildOptions& fockOptions,
                                              const qcx::scf::UhfOptions& scfOptions) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto core = BuildCoreHamiltonian(molecule, basisSet);

    if (!core.has_value())
    {
        return std::unexpected(core.error());
    }

    auto builders = MakeDirectBuilders(molecule, basisSet, fockOptions, ToMatrix(*core));

    if (!builders.has_value())
    {
        return std::unexpected(builders.error());
    }

    return qcx::scf::RunUhfScf(molecule,
                               ToMatrix(*overlap),
                               ToMatrix(*core),
                               scfOptions,
                               MakeDirectUhfFockBuilder(*builders, ToMatrix(*core)));
}

qcx::Result<qcx::scf::HfResult> RunDirectRhf(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basisSet,
                                             const qcx::integrals::FockBuildOptions& fockOptions,
                                             const qcx::scf::RhfOptions& scfOptions) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto core = BuildCoreHamiltonian(molecule, basisSet);

    if (!core.has_value())
    {
        return std::unexpected(core.error());
    }

    auto builders = MakeDirectBuilders(molecule, basisSet, fockOptions, ToMatrix(*core));

    if (!builders.has_value())
    {
        return std::unexpected(builders.error());
    }

    return qcx::scf::RunRhfScf(molecule,
                               ToMatrix(*overlap),
                               ToMatrix(*core),
                               scfOptions,
                               MakeDirectRhfFockBuilder(*builders, ToMatrix(*core)));
}

} // namespace

TEST(DirectRohfTest, OhDoubletMatchesTheExternalReference) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeOhSto3g(2);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // kTight: the certified fp32 lane is disabled, so the direct path
    // reproduces the fp64 dense reference (the recipe the driver's OH UHF
    // cell pins).
    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;

    qcx::scf::RohfOptions scfOptions;
    const auto result = RunDirectRohf(*molecule, *basis, fockOptions, scfOptions);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->converged) << "iterations: " << result->iterations;

    // The external reference, and the variational ordering against the UHF
    // pin: a restricted determinant is a determinant, so the unrestricted
    // minimum can never sit above it.
    EXPECT_NEAR(result->totalEnergy, kOhRohfReferenceEnergy, 1e-7);
    EXPECT_GT(result->totalEnergy, kOhUhfReferenceEnergy);
    // A restricted doublet has S = 1/2 exactly: <S^2> = 0.75, read from the
    // returned orbitals rather than asserted from the occupation counts.
    EXPECT_NEAR(result->spinSquared, 0.75, 1e-8);

    // The three Roothaan stationarity conditions at the returned state:
    // the beta Fock on the closed-open block, the spin average on the
    // closed-virtual block, the alpha Fock on the open-virtual block. The
    // loop's gate watches the density; these are the equations themselves,
    // so a run that stopped with a stationary density but a non-stationary
    // Fock would show up here and nowhere else.
    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto builders = MakeDirectBuilders(*molecule, *basis, fockOptions, ToMatrix(*core));
    ASSERT_TRUE(builders.has_value()) << builders.error().message;
    const auto focks = MakeDirectUhfFockBuilder(*builders, ToMatrix(*core))(result->densityAlpha,
                                                                            result->densityBeta);
    ASSERT_TRUE(focks.has_value()) << focks.error().message;

    const Eigen::MatrixXd s = ToMatrix(*overlap);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> overlapSolver(s);
    ASSERT_EQ(overlapSolver.info(), Eigen::Success);
    const Eigen::MatrixXd x = overlapSolver.eigenvectors() *
                              overlapSolver.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() *
                              overlapSolver.eigenvectors().transpose();
    // The solver-space coefficients: X^T S C (equal to X^{-1} C because
    // X^T S X = I).
    const Eigen::MatrixXd solverCoefficientsAlpha = x.transpose() * s * result->coefficientsAlpha;
    const Eigen::Index numClosed = 4;
    const Eigen::Index numOpen = 1;
    const Eigen::Index numVirtual = solverCoefficientsAlpha.cols() - numClosed - numOpen;
    const Eigen::MatrixXd closed = solverCoefficientsAlpha.leftCols(numClosed);
    const Eigen::MatrixXd open = solverCoefficientsAlpha.middleCols(numClosed, numOpen);
    const Eigen::MatrixXd virtualSpace = solverCoefficientsAlpha.rightCols(numVirtual);
    const Eigen::MatrixXd fockAlphaOrth = x.transpose() * focks->first * x;
    const Eigen::MatrixXd fockBetaOrth = x.transpose() * focks->second * x;
    const Eigen::MatrixXd averageOrth = 0.5 * (fockAlphaOrth + fockBetaOrth);

    const double closedOpen = (closed.transpose() * fockBetaOrth * open).norm();
    const double closedVirtual = (closed.transpose() * averageOrth * virtualSpace).norm();
    const double openVirtual = (open.transpose() * fockAlphaOrth * virtualSpace).norm();
    // Measured at this cell's own gate; the band is the record's, with the
    // same order of headroom the UHF pins carry.
    EXPECT_LT(closedOpen, 1e-6);
    EXPECT_LT(closedVirtual, 1e-6);
    EXPECT_LT(openVirtual, 1e-6);

    // The returned pair is the restricted determinant: one coefficient set
    // in both channels, and the densities rebuild from it.
    EXPECT_LE((result->coefficientsAlpha - result->coefficientsBeta).cwiseAbs().maxCoeff(), 1e-14);
    EXPECT_NEAR((result->densityAlpha - result->densityBeta).trace(), 1.0, 1e-8);
}

// The level shift in the LOOP (the matrix-level test lives in rohf_test.cpp):
// a shifted run must reach the same unshifted pin, because the shift is
// applied to the diagonalization input only and the equations it is solved
// against are unchanged.
TEST(DirectRohfTest, OhDoubletWithTheLevelShiftConvergesToTheSamePin) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeOhSto3g(2);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;

    qcx::scf::RohfOptions shifted;
    shifted.useLevelShift = true;
    shifted.levelShiftOpen = 0.25;
    shifted.levelShiftVirtual = 0.75;

    const auto result = RunDirectRohf(*molecule, *basis, fockOptions, shifted);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->converged) << "iterations: " << result->iterations;

    EXPECT_NEAR(result->totalEnergy, kOhRohfReferenceEnergy, 1e-7);
    EXPECT_NEAR(result->spinSquared, 0.75, 1e-8);
    // The returned spectrum is the UNshifted effective Fock's: a shifted
    // diagonalization must not leak into it.
    EXPECT_TRUE(result->orbitalEnergiesAlpha.allFinite());
}

TEST(DirectRohfTest, TripletH2EqualsUhfForTheSameDeterminant) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    // H2 with multiplicity 3: the same two electrons, both alpha.
    auto tripletMolecule = MakeH2Sto3gWithMultiplicity(3);
    ASSERT_TRUE(tripletMolecule.has_value()) << tripletMolecule.error().message;

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;

    qcx::scf::UhfOptions uhfOptions;
    const auto uhf = RunDirectUhf(*tripletMolecule, *basis, fockOptions, uhfOptions);
    ASSERT_TRUE(uhf.has_value()) << uhf.error().message;
    ASSERT_TRUE(uhf->converged) << "iterations: " << uhf->iterations;

    const qcx::scf::RohfOptions rohfOptions;
    const auto rohf = RunDirectRohf(*tripletMolecule, *basis, fockOptions, rohfOptions);
    ASSERT_TRUE(rohf.has_value()) << rohf.error().message;
    ASSERT_TRUE(rohf->converged) << "iterations: " << rohf->iterations;

    // The nBeta = 0 limit: the restricted open-shell loop with every orbital
    // open is the alpha equation of UHF, so the two loops find the same
    // determinant and the same energy.
    EXPECT_NEAR(rohf->totalEnergy, uhf->totalEnergy, 1e-9);
    EXPECT_NEAR(rohf->spinSquared, 2.0, 1e-8);
    EXPECT_NEAR(uhf->spinSquared, 2.0, 1e-8);
}

TEST(DirectRohfTest, SingletH2EqualsRhf) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;

    const qcx::scf::RhfOptions rhfOptions;
    const auto rhf = RunDirectRhf(*molecule, *basis, fockOptions, rhfOptions);
    ASSERT_TRUE(rhf.has_value()) << rhf.error().message;
    ASSERT_TRUE(rhf->converged) << "iterations: " << rhf->iterations;

    const qcx::scf::RohfOptions rohfOptions;
    const auto rohf = RunDirectRohf(*molecule, *basis, fockOptions, rohfOptions);
    ASSERT_TRUE(rohf.has_value()) << rohf.error().message;
    ASSERT_TRUE(rohf->converged) << "iterations: " << rohf->iterations;

    // The nO = 0 limit: the effective Fock is the spin average, which for a
    // closed pair is the Fock both loops build, so the ROHF run IS the RHF
    // run - the same energy, and the same spin-summed density.
    EXPECT_NEAR(rohf->totalEnergy, rhf->totalEnergy, 1e-9);
    EXPECT_NEAR(
        (rohf->densityAlpha + rohf->densityBeta - rhf->density).cwiseAbs().maxCoeff(), 0.0, 1e-8);
    EXPECT_NEAR(rohf->spinSquared, 0.0, 1e-10);
}

TEST(DirectRohfTest, SingletH2RoksEqualsRksAndUks) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto builders = MakeDirectBuilders(*molecule, *basis, fockOptions, ToMatrix(*core));
    ASSERT_TRUE(builders.has_value()) << builders.error().message;

    auto restricted = MakeSlaterRksSeam(*molecule, *basis, ToMatrix(*core), builders->coulomb);
    ASSERT_TRUE(restricted.has_value()) << restricted.error().message;
    auto unrestricted = MakeSlaterUksSeam(*molecule, *basis, ToMatrix(*core), builders->coulomb);
    ASSERT_TRUE(unrestricted.has_value()) << unrestricted.error().message;

    const qcx::scf::RhfOptions rksOptions;
    const auto rks = qcx::scf::RunRhfScf(*molecule,
                                         ToMatrix(*overlap),
                                         ToMatrix(*core),
                                         rksOptions,
                                         restricted->fock,
                                         restricted->coulomb,
                                         restricted->contribution);
    ASSERT_TRUE(rks.has_value()) << rks.error().message;
    ASSERT_TRUE(rks->converged) << "iterations: " << rks->iterations;

    const qcx::scf::UhfOptions uksOptions;
    const auto uks = qcx::scf::RunUhfScf(*molecule,
                                         ToMatrix(*overlap),
                                         ToMatrix(*core),
                                         uksOptions,
                                         unrestricted->fock,
                                         unrestricted->coulomb,
                                         unrestricted->contribution);
    ASSERT_TRUE(uks.has_value()) << uks.error().message;
    ASSERT_TRUE(uks->converged) << "iterations: " << uks->iterations;

    const qcx::scf::RohfOptions roksOptions;
    const auto roks = qcx::scf::RunRohfScf(*molecule,
                                           ToMatrix(*overlap),
                                           ToMatrix(*core),
                                           roksOptions,
                                           unrestricted->fock,
                                           unrestricted->coulomb,
                                           unrestricted->contribution);
    ASSERT_TRUE(roks.has_value()) << roks.error().message;
    ASSERT_TRUE(roks->converged) << "iterations: " << roks->iterations;

    // The funded acceptance: on a closed shell the spin density vanishes, so
    // the per-spin ROKS seam and the closed-shell RKS seam must reach the
    // same Kohn-Sham fixed point, and it is the pinned number the driver's
    // rks and uks cells both carry.
    EXPECT_NEAR(roks->totalEnergy, kH2SlaterReferenceEnergy, 1e-9);
    EXPECT_NEAR(roks->totalEnergy, rks->totalEnergy, 1e-9);
    EXPECT_NEAR(roks->totalEnergy, uks->totalEnergy, 1e-9);
    EXPECT_NEAR(roks->spinSquared, 0.0, 1e-10);

    // A Kohn-Sham number, not Hartree-Fock: Slater exchange moves this
    // fixture by ~9e-2 Ha, far above the tolerance above.
    EXPECT_GT(std::abs(roks->totalEnergy - (-1.1167143252)), 1e-3);
}

TEST(DirectRohfTest, OhDoubletRoksSitsAboveUks) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeOhSto3g(2);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto builders = MakeDirectBuilders(*molecule, *basis, fockOptions, ToMatrix(*core));
    ASSERT_TRUE(builders.has_value()) << builders.error().message;

    auto seam = MakeSlaterUksSeam(*molecule, *basis, ToMatrix(*core), builders->coulomb);
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    qcx::scf::UhfOptions uksOptions;
    const auto uks = qcx::scf::RunUhfScf(*molecule,
                                         ToMatrix(*overlap),
                                         ToMatrix(*core),
                                         uksOptions,
                                         seam->fock,
                                         seam->coulomb,
                                         seam->contribution);
    ASSERT_TRUE(uks.has_value()) << uks.error().message;
    ASSERT_TRUE(uks->converged) << "iterations: " << uks->iterations;

    // The ROKS run starts from the loop's own zero start on purpose: this
    // fixture's closed/open boundary passes through OH's degenerate pi pair
    // (linear geometry), so the closed and open descriptions alternate
    // between two states of one energy. The loop's gate reads the alpha
    // density, which is the occupied-space projector and does not move under
    // that alternation; a run that read the per-spin pair here never
    // terminates, and this one must.
    const qcx::scf::RohfOptions roksOptions;
    const auto roks = qcx::scf::RunRohfScf(*molecule,
                                           ToMatrix(*overlap),
                                           ToMatrix(*core),
                                           roksOptions,
                                           seam->fock,
                                           seam->coulomb,
                                           seam->contribution);
    ASSERT_TRUE(roks.has_value()) << roks.error().message;
    ASSERT_TRUE(roks->converged) << "iterations: " << roks->iterations;

    // A restricted determinant with the same spin counts is a determinant in
    // the unrestricted variational space, so the restricted minimum can
    // never sit below the unrestricted one, and for a weakly spin-polarized
    // radical the gap is small rather than a different state.
    EXPECT_GE(roks->totalEnergy, uks->totalEnergy - 1e-8);
    EXPECT_NEAR(roks->spinSquared, 0.75, 1e-8);
    // The unrestricted run is free to break spin symmetry; if it did, the
    // gap would be real and larger. Both are one number each, so the band is
    // the reading and the ordering is the assertion.
    EXPECT_LT(roks->totalEnergy - uks->totalEnergy, 0.05) << "uks = " << uks->totalEnergy;
}
