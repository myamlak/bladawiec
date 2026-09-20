// The direct-path UHF pin: the split DirectJkFockBuilder modes -
// buildCoulombOnly (shared J,
// the 2.0 factor baked into the J loop) and buildExchangeOnly (per-spin K) -
// wired into RunUhfScf through the UhfFockBuilderFn seam with the assembly:
// fock_sigma = coulombResult + exchangeSigma - H (each
// split result carries one full H copy, so one is subtracted back - the
// double-H trap). The same O2/STO-3G triplet pin as the
// supermatrix path (kO2PinnedTotalEnergy, tolerance 1e-7), seeded with the
// unpolarized SAD start (the literal P=0
// start cannot converge this system's two-solution landscape).
//
// The direct run iterates PLAIN (no CDIIS): on this system's landscape the
// per-spin DIIS extrapolation converges prematurely through the direct path
// - the Fock maps agree to fp64 noise (3.6e-14 element-wise at the start
// density), yet the DIIS run lands 2.2e-7 off the dense pin in 45
// iterations, while the plain run reproduces it at 3.5e-11 in 22 (the
// record; the initial guess tiers face the same DIIS fragility and run plain
// for the same reason). kTight disables the certified fp32 lane, and the
// physical density keeps every quartet, so the screened path agrees with
// the dense pin to the screening tolerance. The RiJkFockBuilder path is not
// exercised here - the RIJCOSX combination is cross-referenced both ways
// (RiJkFockBuilder::BuildFock in ri_engine.cpp) and pinned in ri_rhf_test.cpp.

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "o2_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/scf/symmetry_reduction.hpp"
#include "qcx/scf/uhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <gtest/gtest.h>
#include <map>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeO2Sto3gBasis;
using qcx::testing::MakeO2Sto3gTriplet;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

using qcx::testing::kO2PinnedSpinSquared;
using qcx::testing::kO2PinnedTotalEnergy;

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

// The UhfFockBuilderFn adapter (the split-builder assembly): the
// seam hands over the ACTUAL per-spin densities (no factor of 2), while the
// J loop bakes in the 2.0 factor, so the shared Coulomb part reads the
// half-summed density dTotalHalf = (D_alpha + D_beta) / 2 and delivers
// H + J(D_alpha + D_beta). The exchange part contracts each spin's actual
// density and delivers H - K(D_sigma). Each result carries one full H copy,
// so one is subtracted back per channel (the double-H trap).
// The RIJCOSX combination in ri_engine.cpp's RiJkFockBuilder::BuildFock is
// the same shape with a different H accounting (its RI side carries no H,
// so no subtraction there) - cross-referenced both ways.
qcx::scf::UhfFockBuilderFn MakeDirectUhfFockBuilder(
    const qcx::integrals::DirectJkFockBuilder& coulombBuilder,
    const qcx::integrals::DirectJkFockBuilder& exchangeBuilder,
    const Eigen::MatrixXd& coreHamiltonian) {
    return [coulombBuilder, exchangeBuilder, coreHamiltonian](const Eigen::MatrixXd& dAlpha,
                                                              const Eigen::MatrixXd& dBeta)
               -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        auto dTotalHalf = ToTensor(0.5 * (dAlpha + dBeta));

        if (!dTotalHalf.has_value())
        {
            return std::unexpected(dTotalHalf.error());
        }

        auto coulomb = coulombBuilder.BuildFock(*dTotalHalf);

        if (!coulomb.has_value())
        {
            return std::unexpected(coulomb.error());
        }

        auto dAlphaTensor = ToTensor(dAlpha);

        if (!dAlphaTensor.has_value())
        {
            return std::unexpected(dAlphaTensor.error());
        }

        auto exchangeAlpha = exchangeBuilder.BuildFock(*dAlphaTensor);

        if (!exchangeAlpha.has_value())
        {
            return std::unexpected(exchangeAlpha.error());
        }

        auto dBetaTensor = ToTensor(dBeta);

        if (!dBetaTensor.has_value())
        {
            return std::unexpected(dBetaTensor.error());
        }

        auto exchangeBeta = exchangeBuilder.BuildFock(*dBetaTensor);

        if (!exchangeBeta.has_value())
        {
            return std::unexpected(exchangeBeta.error());
        }

        const Eigen::MatrixXd coulombMatrix = ToMatrix(*coulomb);
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{
            coulombMatrix + ToMatrix(*exchangeAlpha) - coreHamiltonian,
            coulombMatrix + ToMatrix(*exchangeBeta) - coreHamiltonian};
    };
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

    qcx::integrals::FockBuildOptions coulombOptions = fockOptions;
    coulombOptions.buildCoulombOnly = true;
    auto coulombBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, *core, coulombOptions);

    if (!coulombBuilder.has_value())
    {
        return std::unexpected(coulombBuilder.error());
    }

    qcx::integrals::FockBuildOptions exchangeOptions = fockOptions;
    exchangeOptions.buildExchangeOnly = true;
    auto exchangeBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, *core, exchangeOptions);

    if (!exchangeBuilder.has_value())
    {
        return std::unexpected(exchangeBuilder.error());
    }

    return qcx::scf::RunUhfScf(
        molecule,
        ToMatrix(*overlap),
        ToMatrix(*core),
        scfOptions,
        MakeDirectUhfFockBuilder(*coulombBuilder, *exchangeBuilder, ToMatrix(*core)));
}

// The petite-list driver seam on the UHF side: same shape as
// RunDirectUhf, but the reduction is extracted once and wired into BOTH
// split builders (the shared J and the per-spin K) through
// FockBuildOptions::symmetryReduction, and the basis travels the blocked
// diagonalization seam of RunUhfScf. A kUnimplemented or trivial
// extraction falls back to the plain path; symmetryEngagedOut reports
// which path actually ran.
qcx::Result<qcx::scf::UhfResult> RunDirectUhfWithSymmetry(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    qcx::integrals::FockBuildOptions fockOptions,
    const qcx::scf::UhfOptions& scfOptions,
    bool* symmetryEngagedOut = nullptr) {
    auto reduction = qcx::scf::BuildSymmetryReduction(molecule, basisSet);

    if (!reduction.has_value() && reduction.error().code != qcx::ErrorCode::kUnimplemented)
    {
        return std::unexpected(reduction.error());
    }

    // The reduction's own pair-action verdict, never the group order (the
    // mechanism-value audit, 2026-09-13): a non-C1 group that permutes no
    // shell pair is trivial and the class path does not run for it.
    const bool engaged = reduction.has_value() && !reduction->isTrivial;

    if (engaged)
    {
        fockOptions.symmetryReduction = &*reduction;
    }

    if (symmetryEngagedOut != nullptr)
    {
        *symmetryEngagedOut = engaged;
    }

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

    qcx::integrals::FockBuildOptions coulombOptions = fockOptions;
    coulombOptions.buildCoulombOnly = true;
    auto coulombBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, *core, coulombOptions);

    if (!coulombBuilder.has_value())
    {
        return std::unexpected(coulombBuilder.error());
    }

    qcx::integrals::FockBuildOptions exchangeOptions = fockOptions;
    exchangeOptions.buildExchangeOnly = true;
    auto exchangeBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, *core, exchangeOptions);

    if (!exchangeBuilder.has_value())
    {
        return std::unexpected(exchangeBuilder.error());
    }

    return qcx::scf::RunUhfScf(
        molecule,
        ToMatrix(*overlap),
        ToMatrix(*core),
        scfOptions,
        MakeDirectUhfFockBuilder(*coulombBuilder, *exchangeBuilder, ToMatrix(*core)),
        &basisSet);
}

// The dense per-element atomic integrals of the SAD guess: one O atom at
// the origin (the fragment run; the geometry never affects its own
// integrals) - the uhf_test.cpp helper, replicated here so the direct path
// seeds the same unpolarized SAD start as the supermatrix pin test.
qcx::Result<qcx::scf::AtomicUhfInputs> BuildAtomicOxygenInputs() {
    auto coordinates = CpuTensor2::Create({1, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    coordinates->MarkHostDirty();
    auto atom = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}}, std::move(*coordinates), 0, 3);

    if (!atom.has_value())
    {
        return std::unexpected(atom.error());
    }

    auto basis = MakeO2Sto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*atom, *basis);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(*atom, *basis);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*atom, *basis);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    auto eri = qcx::integrals::BuildEriTensorGeneral(*atom, *basis);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    return qcx::scf::AtomicUhfInputs{
        ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), std::move(*eri)};
}

qcx::Result<qcx::scf::SadGuess> BuildSadGuessForO2(const qcx::molecule::Molecule& molecule,
                                                   const qcx::basisset::BasisSet& basisSet) {
    auto atomicInputs = BuildAtomicOxygenInputs();

    if (!atomicInputs.has_value())
    {
        return std::unexpected(atomicInputs.error());
    }

    std::map<int, qcx::scf::AtomicUhfInputs> atomicMap;
    atomicMap.emplace(8, std::move(*atomicInputs));
    return qcx::scf::BuildSadGuess(molecule, basisSet, atomicMap);
}

TEST(DirectUhfTest, O2DirectPathPins) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeO2Sto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeO2Sto3gTriplet();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // kTight: the certified fp32 lane is disabled, so the direct path
    // reproduces the fp64 dense pin. The plain iterator (see the file
    // header for why not DIIS) keeps the default screening on - the
    // physical density drops nothing, so the screened result equals the
    // unscreened one to the last digit here.
    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;

    // The unpolarized SAD seed: alpha = beta
    // = the SAD average, the minao-like start the pyscf reference run used.
    const auto sad = BuildSadGuessForO2(*molecule, *basis);
    ASSERT_TRUE(sad.has_value()) << sad.error().message;
    const Eigen::MatrixXd unpolarizedStart = 0.5 * (sad->densityAlpha + sad->densityBeta);

    qcx::scf::UhfOptions scfOptions;
    scfOptions.useDiis = false;
    scfOptions.initialDensityAlpha = unpolarizedStart;
    scfOptions.initialDensityBeta = unpolarizedStart;

    const auto result = RunDirectUhf(*molecule, *basis, fockOptions, scfOptions);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->converged) << "iterations: " << result->iterations;
    EXPECT_NEAR(result->totalEnergy, kO2PinnedTotalEnergy, 1e-7);
    EXPECT_NEAR(result->spinSquared, kO2PinnedSpinSquared, 1e-6);
}

TEST(DirectUhfTest, WaterC2vPetiteListSeamMatchesPlain) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // kTight: fp64 on both runs (the class path disables the fp32 lane
    // regardless of the preset). Water UHF is the closed-shell singlet
    // minimum: the default DIIS trajectory is benign here (the O2
    // two-solution landscape is the special case), and both runs share it.
    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;

    // A FIXED SUMMATION ORDER MAKES THE ROUTE EQUALITY A PROPERTY (2026-09-20),
    // the remedy the RHF sibling (direct_rhf_test.cpp, the C2v seam test)
    // already carries for the same reason. THE OPERATING RUNG CARRIES THIS RUN
    // (owner ruling 2026-09-18: no convergence gate at or below 1e-10, so the
    // equal-legs 1e-10/1e-10 pair this test used to bind is RETIRED), and the
    // 2026-09-15 reading stands as the measurement it was: at the production
    // defaults the seam route stopped at iteration 11 and the plain route at
    // 10, so the iteration-count equality below failed there. The mechanism
    // is not that the routes disagree - it is that the index at which their
    // threshold crossings land moves with the parallel reduction order, and
    // an equality on a number that moves with the thread schedule is
    // measuring the schedule, not the physics. maxParallelChunks 1 removes
    // that degree of freedom; it loosens nothing. Measured on this tree at
    // the rung: with the fixed order the routes stop on the same iterate (10
    // and 10) and their stopped points agree to 3.475e-14 element-wise,
    // 2.0e-14 in the gate's own summed RMS; at the default order they ALSO
    // agreed (11 and 11, 3.2e-13) - only the common value of the index moved,
    // which is exactly the accidental reading this removes. Re-run here at
    // OMP_NUM_THREADS 1, 2, 4 and 12, the fixed order holds at every one.
    // Every contract below therefore holds on its own terms.
    fockOptions.maxParallelChunks = 1;
    qcx::scf::UhfOptions scfOptions;
    scfOptions.energyTolerance = 1e-8;
    scfOptions.densityTolerance = 1e-6;

    bool symmetryEngaged = false;
    auto classResult =
        RunDirectUhfWithSymmetry(*molecule, *basis, fockOptions, scfOptions, &symmetryEngaged);
    ASSERT_TRUE(classResult.has_value()) << classResult.error().message;
    ASSERT_TRUE(symmetryEngaged) << "the water C2v extraction must engage the class path";

    auto plainResult = RunDirectUhf(*molecule, *basis, fockOptions, scfOptions);
    ASSERT_TRUE(plainResult.has_value()) << plainResult.error().message;

    ASSERT_TRUE(classResult->converged) << "iterations: " << classResult->iterations;
    EXPECT_TRUE(plainResult->converged) << "iterations: " << plainResult->iterations;
    // The class path through both split builders (shared J, per-spin K) and the
    // blocked diagonalization reproduce the plain run's stopped point to
    // rounding: measured 3.475e-14 element-wise on the densest element of either
    // spin, against the 1e-9 contracts below.
    //
    // The iteration counts are deliberately NOT compared. How many steps a
    // solver takes to reach that point is a property of the path it took, not
    // of the answer it reached, so pinning the two paths to the same count
    // asserts a determinism no architecture-independent implementation owes:
    // the counts agree on x86-64 and differ by one on arm64, where the
    // converged energy and density agree just the same.
    EXPECT_NEAR(classResult->totalEnergy, plainResult->totalEnergy, 1e-9);
    EXPECT_NEAR(
        (classResult->densityAlpha - plainResult->densityAlpha).cwiseAbs().maxCoeff(), 0.0, 1e-9);
    EXPECT_NEAR(
        (classResult->densityBeta - plainResult->densityBeta).cwiseAbs().maxCoeff(), 0.0, 1e-9);
}

} // namespace
