// The direct-path SCF pins: the
// DirectJkFockBuilder wired into RunRhfScf through the FockBuilderFn seam
// (seam_test.cpp validates the seam itself; here the direct builder is the
// Fock source). The pinned energies are the dense_rhf_test.cpp records:
// H2 -1.1167143252 always, HF -98.570757591618 and H2O -74.962928246436
// fast-gated (the pyscf anchors of dense_rhf_test.cpp).
// The HF/H2O pin runs use kTight (fp32 lane disabled) so the direct path
// reproduces the fp64 dense path; the certified mixed lane gets its own
// bound-dominance test on the plain (no-DIIS) loop, where the extrapolation
// weights cannot amplify the delivered bounds.
//
// THE SCF GATE (owner ruling 2026-09-15: "This is ridiculous to use such
// tight tolerance unless you have very good reasons for it"). All but one
// walk below drives the PRODUCTION defaults (energy 1e-8, density
// 1e-6); the explicit 1e-10/1e-10 gate they used to carry was the
// fixed-point records' convention - history, not a property of the contracts
// asserted here, since a pin recorded at a non-default gate asserts the gate
// as much as the physics. The one site that still sets a gate states its
// measured reason in-line (FullyRotatedWaterSymmetryFallsBackToPlainPath).

#include "benzene_sto3g.hpp"
#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "hf_sto3g.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/symmetry_reduction.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeBenzeneSto3gBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeHfSto3g;
using qcx::testing::MakeHfSto3gBasis;
using qcx::testing::MakeRotatedBenzeneSto3g;
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

// The FockBuilderFn adapter: density tensor in, direct J/K Fock out. The
// builder is prepared once (pair data, Schwarz bounds, the H copy); each
// call is one screened pass. The seam hands over the spin-summed density
// D = 2 C_occ C_occ^T, while the builder contracts the spatial density
// rho = D/2 (BUG-3, 2026-08-21; rhf.hpp documents the seam convention) -
// the adapter scales before handing over. The certified bound sum of every
// call is accumulated into boundSumOut, and the certified energy bound -
// the delivered sum S times sum |rho|: every Fock element's fp32 error is
// at most S, so |dE| <= 1/2 sum|D| S = S sum|rho| - into energyBoundOut
// when requested.
qcx::scf::FockBuilderFn MakeDirectFockBuilder(const qcx::integrals::DirectJkFockBuilder& builder,
                                              double* boundSumOut,
                                              double* energyBoundOut = nullptr) {
    return [builder, boundSumOut, energyBoundOut](
               const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        auto densityTensor = ToTensor(0.5 * density);

        if (!densityTensor.has_value())
        {
            return std::unexpected(densityTensor.error());
        }

        double callBoundSum = 0.0;
        auto fock =
            builder.BuildFock(*densityTensor, boundSumOut != nullptr ? &callBoundSum : nullptr);

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        if (boundSumOut != nullptr)
        {
            *boundSumOut += callBoundSum;

            if (energyBoundOut != nullptr)
            {
                const double densityL1 = density.cwiseAbs().sum() * 0.5; // sum |rho|.
                *energyBoundOut += callBoundSum * densityL1;
            }
        }

        return ToMatrix(*fock);
    };
}

qcx::Result<qcx::scf::HfResult> RunDirectRhf(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basisSet,
                                             const qcx::integrals::FockBuildOptions& fockOptions,
                                             const qcx::scf::RhfOptions& scfOptions,
                                             double* totalBoundSumOut = nullptr,
                                             double* totalEnergyBoundOut = nullptr) {
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

    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, *core, fockOptions);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    return qcx::scf::RunRhfScf(
        molecule,
        ToMatrix(*overlap),
        ToMatrix(*core),
        scfOptions,
        MakeDirectFockBuilder(*builder, totalBoundSumOut, totalEnergyBoundOut));
}

// The petite-list driver seam: the FockBuilderFn-path wiring of the
// petite list, mirroring the blocked-diagonalization seam. The driver
// extracts the reduction once (scf::BuildSymmetryReduction - the same
// point-group detection the blocked diagonalization uses), hands it to the
// builder through FockBuildOptions::symmetryReduction (the builder copies it
// at Create, so the pointer need not outlive the call), and passes the basis
// to RunRhfScf so every iteration diagonalizes the irrep blocks separately.
// An extraction that falls back to an axis-aligned subgroup (e.g. a
// rotated molecule keeping its C2) still engages the class path on that
// subgroup; a kUnimplemented extraction (a point group no non-identity
// element expresses as a signed coordinate permutation, e.g. water turned
// about a tilted axis) falls back to the plain path; a trivial (C1)
// extraction keeps the plain path too. symmetryEngagedOut reports which
// path actually ran.
qcx::Result<qcx::scf::HfResult> RunDirectRhfWithSymmetry(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    qcx::integrals::FockBuildOptions fockOptions,
    const qcx::scf::RhfOptions& scfOptions,
    bool* symmetryEngagedOut = nullptr,
    double* totalBoundSumOut = nullptr,
    double* totalEnergyBoundOut = nullptr) {
    auto reduction = qcx::scf::BuildSymmetryReduction(molecule, basisSet);

    if (!reduction.has_value() && reduction.error().code != qcx::ErrorCode::kUnimplemented)
    {
        return std::unexpected(reduction.error());
    }

    // The engagement flag is the reduction's own pair-action verdict, never
    // the group order (the mechanism-value audit, 2026-09-13): a non-C1
    // group that permutes no shell pair is trivial and the class path does
    // not run for it, so an order test would report an engagement that did
    // not happen. A trivial extraction is the plain enumeration and must
    // NOT be wired (the builder's Create ignores it anyway - this keeps the
    // engaged flag honest).
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

    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, *core, fockOptions);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    // The basis always travels the seam: blocked diagonalization engages
    // whenever the group is realizable, and a kUnimplemented detection falls
    // back inside RunScfLoop to the very same plain diagonalization the
    // basis-less call uses.
    return qcx::scf::RunRhfScf(
        molecule,
        ToMatrix(*overlap),
        ToMatrix(*core),
        scfOptions,
        MakeDirectFockBuilder(*builder, totalBoundSumOut, totalEnergyBoundOut),
        &basisSet);
}

TEST(DirectRhfTest, H2DirectPathPins) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // kNormal: the H2 quartets sit above the fp32 gate (C_class(0,0) * 1e-7
    // * Q^2 * dMax ~ 4e-8 > 1e-8), so the direct path is all-fp64 here.
    // Screening stays on: the physical density keeps every quartet, so the
    // screened and unscreened paths agree to machine precision.
    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    // The production defaults, per the file header's gate note (the tight
    // gate this test carried was the record's convention).
    qcx::scf::RhfOptions scfOptions;
    auto result = RunDirectRhf(*molecule, *basis, fockOptions, scfOptions);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->converged) << "iterations: " << result->iterations;
    EXPECT_NEAR(result->totalEnergy, -1.1167143252, 1e-8);
}

TEST(DirectRhfTest, HfDirectPathPins) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeHfSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeHfSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // kTight: the fp32 lane is disabled, so the direct path reproduces the
    // fp64 dense pin.
    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto result = RunDirectRhf(*molecule, *basis, fockOptions, qcx::scf::RhfOptions{});
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->converged) << "iterations: " << result->iterations;
    // pyscf-anchored 2026-08-22: pyscf RHF/STO-3G at this geometry gives
    // -98.570757591618; agrees with the dense pin at 1e-11.
    EXPECT_NEAR(result->totalEnergy, -98.57075766, 1e-5);
}

TEST(DirectRhfTest, H2oDirectPathPins) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto result = RunDirectRhf(*molecule, *basis, fockOptions, qcx::scf::RhfOptions{});
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->converged) << "iterations: " << result->iterations;
    // pyscf-anchored 2026-08-22: pyscf RHF/STO-3G at this geometry gives
    // -74.962928246436; agrees with the dense pin at 1e-13.
    EXPECT_NEAR(result->totalEnergy, -74.96292827, 1e-5);
}

// The rotated water (the symmetry_reduction_test.cpp fixture): the C2v
// geometry turned 30 degrees about z. The C2 axis - the H-O-H bisector, the
// global y - leaves the global frame, so it does NOT survive as a signed
// coordinate permutation; the element that survives is the molecular
// plane's z-flip, which maps every atom to itself. BuildSymmetryReduction
// therefore falls back to the {E, sigma_z} subgroup and reports it TRIVIAL
// (it permutes no shell pair), so the seam keeps the plain path - see
// RotatedWaterPlaneMirrorSubgroupStaysOnThePlainPath.
qcx::Result<qcx::molecule::Molecule> MakeRotatedWater() {
    constexpr double kCos30 = 0.8660254037844387;
    constexpr double kSin30 = 0.5;
    constexpr double kX = 1.430428808474167;
    constexpr double kY = 1.107157044080814;

    Eigen::MatrixXd coordinates(3, 3);
    coordinates.row(0) << 0.0, 0.0, 0.0;

    for (const double sign : {-1.0, 1.0})
    {
        const double x = sign * kX;
        const Eigen::Index row = sign < 0 ? 1 : 2;
        coordinates.row(row) << x * kCos30 - kY * kSin30, x * kSin30 + kY * kCos30, 0.0;
    }

    auto tensorResult = ToTensor(coordinates);

    if (!tensorResult.has_value())
    {
        return std::unexpected(tensorResult.error());
    }

    return qcx::molecule::Molecule::Create({qcx::molecule::Atom{"O", 8, 0.0},
                                            qcx::molecule::Atom{"H", 1, 0.0},
                                            qcx::molecule::Atom{"H", 1, 0.0}},
                                           std::move(*tensorResult),
                                           0,
                                           1);
}

// Water turned 45 degrees about the (1,1,1)/sqrt(3) axis on top of the
// base geometry (the H atoms at +-(kX, kY, 0)): the C2 axis and both mirror
// normals leave the global frame, so no non-identity element is a signed
// coordinate permutation - BuildSymmetryReduction must report kUnimplemented
// and the seam must fall back to the plain path. (A rotation about a global
// axis would keep that axis' element aligned - e.g. 45 degrees about x
// keeps the yz-plane mirror whose normal is the rotation axis.)
qcx::Result<qcx::molecule::Molecule> MakeRotatedWater3d() {
    constexpr double kX = 1.430428808474167;
    constexpr double kY = 1.107157044080814;
    constexpr double kCos45 = 0.7071067811865476;
    constexpr double kInvSqrt3 = 0.5773502691896258;

    // The 45-degree rotation about n = (1,1,1)/sqrt(3):
    // R = cos(theta) I + (1 - cos(theta)) n n^T + sin(theta) [n]_x.
    const Eigen::Matrix3d skew =
        kInvSqrt3 *
        (Eigen::Matrix3d() << 0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0).finished();
    const Eigen::Matrix3d rotation = kCos45 * Eigen::Matrix3d::Identity() +
                                     (1.0 - kCos45) / 3.0 * Eigen::Matrix3d::Ones() + kCos45 * skew;

    Eigen::MatrixXd coordinates(3, 3);
    coordinates.row(0) << 0.0, 0.0, 0.0;

    for (const double sign : {-1.0, 1.0})
    {
        const Eigen::Vector3d hydrogen{kX * sign, kY, 0.0};
        const Eigen::Index row = sign < 0 ? 1 : 2;
        coordinates.row(row) = (rotation * hydrogen).transpose();
    }

    auto tensorResult = ToTensor(coordinates);

    if (!tensorResult.has_value())
    {
        return std::unexpected(tensorResult.error());
    }

    return qcx::molecule::Molecule::Create({qcx::molecule::Atom{"O", 8, 0.0},
                                            qcx::molecule::Atom{"H", 1, 0.0},
                                            qcx::molecule::Atom{"H", 1, 0.0}},
                                           std::move(*tensorResult),
                                           0,
                                           1);
}

TEST(DirectRhfTest, WaterC2vPetiteListSeamMatchesPlain) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // kTight: fp64 on both runs (the class path disables the fp32 lane
    // regardless of the preset; the plain comparison run must be fp64 too
    // for the equivalence pin to be meaningful).
    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    // The iteration-count equality below compares the two routes' stop
    // INDEXES, i.e. two threshold crossings of the gate. Both gate legs are
    // live, so the stop is the first iterate where BOTH hold. The
    // thread measurements in this note are TIGHT-GATE readings (1e-10/1e-10),
    // where near the tail |dE| moves in a ~1e-10 band, the same scale at
    // which the two routes' arithmetic differs; at the production defaults
    // this fixture's routes stop on the same iterate (measured 2026-09-15).
    // At team > 1 the crossing index is then decided by the critical-section
    // reduction order:
    // measured 0/20 equal at 2 threads (class 8, plain 10; at the shared
    // density crossing the class route's |dE| reads 6.35e-11 against the
    // plain route's 1.86e-10, one side of the gate each) and 10/20 at the
    // 12-thread default. A fixed summation order is what makes the equality
    // a property: maxParallelChunks 1 (the pin Avx2DispatchPathsAreBit-
    // Identical applies below for the same reason) reproduces it 20/20 at
    // 1, 2, 4 and 12 threads, with the routes' converged energies 9.948e-14
    // apart on every walk. The energy 1e-9, density 1e-9 and anchor
    // contracts below are unchanged and hold at every thread count.
    fockOptions.maxParallelChunks = 1;

    // PRODUCTION DEFAULTS, per the file header's gate note (owner ruling
    // 2026-09-15). Measured at the defaults on this tree (2026-09-15): the
    // class and plain routes stop on the SAME iterate and their stopped
    // points sit inside the contracts below - this fixture's route equality
    // survives the looser gate.
    qcx::scf::RhfOptions scfOptions;
    bool symmetryEngaged = false;
    auto classResult =
        RunDirectRhfWithSymmetry(*molecule, *basis, fockOptions, scfOptions, &symmetryEngaged);
    ASSERT_TRUE(classResult.has_value()) << classResult.error().message;
    ASSERT_TRUE(symmetryEngaged) << "the water C2v extraction must engage the class path";

    auto plainResult = RunDirectRhf(*molecule, *basis, fockOptions, scfOptions);
    ASSERT_TRUE(plainResult.has_value()) << plainResult.error().message;

    ASSERT_TRUE(classResult->converged) << "iterations: " << classResult->iterations;
    EXPECT_TRUE(plainResult->converged) << "iterations: " << plainResult->iterations;
    EXPECT_EQ(classResult->iterations, plainResult->iterations);
    // At the fixed summation order above, the class path and the blocked
    // diagonalization take the same discrete DIIS decisions as the plain
    // run; the routes' own arithmetic differs at the 1e-13 level, which the
    // parallel reduction order scatters up to the gate's own scale (the
    // maxParallelChunks note at the options).
    EXPECT_NEAR(classResult->totalEnergy, plainResult->totalEnergy, 1e-9);
    EXPECT_NEAR((classResult->density - plainResult->density).cwiseAbs().maxCoeff(), 0.0, 1e-9);
    // The pinned pyscf anchor (the dense-path record) holds through the
    // class path: the reduction changes the contraction, not the result.
    EXPECT_NEAR(classResult->totalEnergy, -74.96292827, 1e-5);
}

// The certified fp32 lane engages on the class-aware path through the
// driver seam. kLoose so the gate routes quartets down the fp32 lane, no
// DIIS so the delivered Fock matrices are exactly the diagonalized ones -
// the energy-bound composition. The class-path
// run must stay inside the accumulated certified energy bounds of the
// plain mixed-precision run (the per-call bound sums bound each call's
// delivered Fock error on the class path too; the trajectory term is
// second-order with no DIIS - the same argument as
// MixedLaneEnergyBoundDominates).
TEST(DirectRhfTest, WaterC2vPetiteListSeamMixedPrecisionMatchesPlain) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::scf::RhfOptions scfOptions;
    scfOptions.useDiis = false;

    qcx::integrals::FockBuildOptions mixed;
    mixed.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    mixed.useDensityScreening = true;
    mixed.useCertifiedMixedPrecision = true;

    bool symmetryEngaged = false;
    double classBoundSum = 0.0;
    double classEnergyBound = 0.0;
    auto classResult = RunDirectRhfWithSymmetry(
        *molecule, *basis, mixed, scfOptions, &symmetryEngaged, &classBoundSum, &classEnergyBound);
    ASSERT_TRUE(classResult.has_value()) << classResult.error().message;
    ASSERT_TRUE(symmetryEngaged) << "the water C2v extraction must engage the class path";
    ASSERT_TRUE(classResult->converged) << "iterations: " << classResult->iterations;
    // The fp32 lane must route on the class path (the s-pair quartets sit
    // inside the kLoose gate).
    EXPECT_GT(classBoundSum, 0.0);
    EXPECT_GT(classEnergyBound, 0.0);

    double plainBoundSum = 0.0;
    double plainEnergyBound = 0.0;
    auto plainResult =
        RunDirectRhf(*molecule, *basis, mixed, scfOptions, &plainBoundSum, &plainEnergyBound);
    ASSERT_TRUE(plainResult.has_value()) << plainResult.error().message;
    ASSERT_TRUE(plainResult->converged) << "iterations: " << plainResult->iterations;

    // The bound dominance through the petite list: the class-path run's
    // deviation from the plain mixed run is bounded by the sum of the two
    // runs' accumulated certified energy bounds.
    EXPECT_LE(std::abs(classResult->totalEnergy - plainResult->totalEnergy),
              (classEnergyBound + plainEnergyBound) * 1.001 + 1e-12)
        << "class-path energy deviation exceeds the certified bounds";
    // The densities agree at the certified-error level (the fp32 lane's
    // delivered accuracy, orders of magnitude above the fp64 trajectory
    // noise the kTight pin checks at 1e-9).
    EXPECT_NEAR((classResult->density - plainResult->density).cwiseAbs().maxCoeff(), 0.0, 1e-6);
}

TEST(DirectRhfTest, RotatedWaterPlaneMirrorSubgroupStaysOnThePlainPath) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeRotatedWater();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;

    // PRODUCTION DEFAULTS, per the file header's gate note (owner ruling
    // 2026-09-15). No iteration-count equality is asserted here, and the
    // fallback seam keeps the plain path's fixed point at the defaults
    // (measured 2026-09-15: the 1e-9 energy and 1e-8 density contracts hold).
    qcx::scf::RhfOptions scfOptions;
    bool symmetryEngaged = false;
    auto seamResult =
        RunDirectRhfWithSymmetry(*molecule, *basis, fockOptions, scfOptions, &symmetryEngaged);
    ASSERT_TRUE(seamResult.has_value()) << seamResult.error().message;
    // WHAT SURVIVES, and why the seam must NOT engage on it (the
    // mechanism-value audit, 2026-09-13; this test pinned the opposite
    // verdict before, on an explanation the fixture's own hand-built action
    // table contradicts). The 30-degree turn takes the C2v molecule's C2
    // axis - the H-O-H bisector, the global y - OFF the global frame, so
    // the C2 does not survive as a signed coordinate permutation. What does
    // is the molecular PLANE (the molecule stays in z = 0): the z-flip maps
    // every atom to itself, so its action is the identity permutation with
    // p_z negated - and a group that permutes no shell pair has every orbit
    // a singleton. The reduction would cost the classification, the class
    // tables and the orbit tables and remove nothing (hocl/Cs is the audit's
    // measured case, exactly 1.0000x), so the reduction reports itself
    // trivial and the seam keeps the plain path. The blocked
    // diagonalization still uses the full C2v.
    EXPECT_FALSE(symmetryEngaged)
        << "a subgroup that permutes no shell pair must not engage the class path";

    auto plainResult = RunDirectRhf(*molecule, *basis, fockOptions, scfOptions);
    ASSERT_TRUE(plainResult.has_value()) << plainResult.error().message;

    ASSERT_TRUE(seamResult->converged) << "iterations: " << seamResult->iterations;
    EXPECT_TRUE(plainResult->converged) << "iterations: " << plainResult->iterations;
    // The C2 class path and the blocked diagonalization both reproduce the
    // plain run's FIXED POINT, not its route. The DIIS minimum-norm
    // extrapolation is discontinuous under rounding near rank transitions
    // of the error Gram, so two builds of the same matrix can take
    // different discrete DIIS decisions and reach the same fixed point at
    // different iteration counts (observed 8 vs 15 after the GWH
    // default start; the former P = 0 start's exactly-zero error pair
    // pinned the singular B-matrix and aligned the routes as a bug
    // artifact). Two converged iterates that reach the same fixed point via
    // different routes sit at gate-scale x route noise of each other: the
    // energy contract is 1e-9, the density contract 1e-8 - the 2.8e-9 spread
    // and its 3.5x margin are TIGHT-GATE readings (1e-10-RMS), and at the
    // production defaults this fixture's routes hold both contracts
    // (measured 2026-09-15; 100x below the certified 1e-6 level).
    EXPECT_NEAR(seamResult->totalEnergy, plainResult->totalEnergy, 1e-9);
    EXPECT_NEAR((seamResult->density - plainResult->density).cwiseAbs().maxCoeff(), 0.0, 1e-8);
    // The pinned pyscf anchor (the dense-path record) holds through the
    // C2 class path: the reduction changes the contraction, not the result.
    EXPECT_NEAR(seamResult->totalEnergy, -74.96292827, 1e-5);
}

TEST(DirectRhfTest, FullyRotatedWaterSymmetryFallsBackToPlainPath) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeRotatedWater3d();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;

    // THE OPERATING RUNG CARRIES THIS RUN (owner ruling 2026-09-18: no
    // convergence gate at or below 1e-10, so the equal-legs 1e-10/1e-10
    // pair this test used to bind is RETIRED). This is the site that ruling
    // turned into a finding, and the finding has now been acted on rather
    // than reported again (2026-09-20): the two routes stop on DIFFERENT
    // iterates at the rung - measured on this tree, seam 8 / plain 9, in
    // both the default and the maxParallelChunks 1 summation order, so no
    // fixed order aligns this fixture the way it aligns the C2v sibling
    // above. The density contract below was an absolute 1e-8 written for
    // the retired gate; the difference of two stopped points is a gate
    // quantity, and the contract is now stated as one. Its reasoning and
    // its measurement sit at the assertion.
    qcx::scf::RhfOptions scfOptions;
    scfOptions.energyTolerance = 1e-8;
    scfOptions.densityTolerance = 1e-6;
    bool symmetryEngaged = true;
    auto seamResult =
        RunDirectRhfWithSymmetry(*molecule, *basis, fockOptions, scfOptions, &symmetryEngaged);
    ASSERT_TRUE(seamResult.has_value()) << seamResult.error().message;
    EXPECT_FALSE(symmetryEngaged) << "the fully rotated molecule must fall back to the plain path";

    auto plainResult = RunDirectRhf(*molecule, *basis, fockOptions, scfOptions);
    ASSERT_TRUE(plainResult.has_value()) << plainResult.error().message;

    ASSERT_TRUE(seamResult->converged) << "iterations: " << seamResult->iterations;
    EXPECT_TRUE(plainResult->converged) << "iterations: " << plainResult->iterations;
    // The kUnimplemented fallback takes the SAME eigenspaces as the plain
    // run - no reduction wired, and RunScfLoop's block build succeeds on
    // the dense action but preserves the eigenspaces - so the FIXED POINT
    // agrees. The routes need not agree: the DIIS minimum-norm
    // extrapolation is discontinuous under rounding near rank transitions
    // of the error Gram, and after the GWH default start the paths
    // take different discrete DIIS decisions (observed 8 vs 16 iterations
    // at the retired gate; the former P = 0 start's exactly-zero error
    // pair pinned the singular B-matrix and aligned the routes as a bug
    // artifact).
    //
    // WHAT THE DENSITY CONTRACT MAY BE, exactly. A stopped point is
    // determined by the gate that stopped it and by nothing finer, so a
    // contract on the difference of two stopped points is a statement about
    // the gate, not about the physics. Each stopped point sits within one
    // density leg of the shared fixed point (that is the gate's own leg,
    // |D_stop - D*| <= densityTolerance plus the last step's contraction),
    // so their separation is bounded by twice that leg. The assertion below
    // is written in the gate's own norm - (new - old).norm() / n, the
    // Frobenius norm over the n x n density at the loop's divisor, the
    // quantity IsConverged itself tests - at 2 x densityTolerance. The
    // bound is derived from the gate, not fitted to a reading.
    //
    // The max-element form it replaces was calibrated at the retired gate
    // (1e-10-RMS), where this fixture's routes spread 2.8e-9 - a 3.5x
    // margin. At the rung the same quantity reads 4.62e-9 on this tree and
    // read 1.0977e-8 on 2026-09-15, i.e. it sits on both sides of its own
    // 1e-8 bound: a bound that straddles its measurement is not a bound,
    // it is a coin. What the contract still earns is that the two routes
    // reach one fixed point at the rung to a gate leg - the physics
    // statement - which the energy contract below asserts independently
    // and more sharply: the energy is stationary at the fixed point, so it
    // is second-order in the density error and survives the rung unweakened
    // (measured 1.4e-14 against its 1e-9, a 7e4x margin; it is NOT
    // restated).
    EXPECT_NEAR(seamResult->totalEnergy, plainResult->totalEnergy, 1e-9);
    const double routeDensityRms = (seamResult->density - plainResult->density).norm() /
                                   static_cast<double>(seamResult->density.rows());
    EXPECT_LE(routeDensityRms, 2.0 * scfOptions.densityTolerance);
}

TEST(DirectRhfTest, RotatedBenzeneC2hClassesMatchPlain) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeBenzeneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeRotatedBenzeneSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;

    // PRODUCTION DEFAULTS, per the file header's gate note (owner ruling
    // 2026-09-15). Measured at the defaults on this tree (2026-09-15): the
    // two routes still stop on the same iterate and hold the 1e-9 energy and
    // density contracts below.
    qcx::scf::RhfOptions scfOptions;
    bool symmetryEngaged = false;
    auto seamResult =
        RunDirectRhfWithSymmetry(*molecule, *basis, fockOptions, scfOptions, &symmetryEngaged);
    ASSERT_TRUE(seamResult.has_value()) << seamResult.error().message;
    // The rotated benzene keeps the C2h subgroup (the C6-axis C2, the
    // inversion, the molecular-plane mirror): the fallback selects it for
    // the class contraction, while the blocked diagonalization keeps the
    // full D2h of the unrotated fixture's realization frame.
    ASSERT_TRUE(symmetryEngaged) << "the C2h fallback must engage the class path";

    auto plainResult = RunDirectRhf(*molecule, *basis, fockOptions, scfOptions);
    ASSERT_TRUE(plainResult.has_value()) << plainResult.error().message;

    ASSERT_TRUE(seamResult->converged) << "iterations: " << seamResult->iterations;
    EXPECT_TRUE(plainResult->converged) << "iterations: " << plainResult->iterations;
    // The same iteration-count equality as the H2O seam test above, but not
    // the same margin: at the TIGHT gate this fixture's density leg crosses
    // at iteration 11 and its energy leg at 19 (both routes, 1/2/12
    // threads), so the stop index was not decided inside the |dE| ~ 1e-10
    // band - the exit's energy operand reads 4.5e-12, 22x inside the gate,
    // and the two routes agree on it to ~10 percent. The equality holds at
    // the production defaults too (measured 2026-09-15: equal counts).
    // Apply maxParallelChunks 1 here as well if a fixture or tolerance
    // change ever brings the two crossings together.
    EXPECT_EQ(seamResult->iterations, plainResult->iterations);
    EXPECT_NEAR(seamResult->totalEnergy, plainResult->totalEnergy, 1e-9);
    EXPECT_NEAR((seamResult->density - plainResult->density).cwiseAbs().maxCoeff(), 0.0, 1e-9);
}

TEST(DirectRhfTest, MixedLaneEnergyBoundDominates) {
    // The certified fp32 bound carried to the SCF energy: on the plain
    // (no-DIIS) loop the Fock matrices used for diagonalization are exactly
    // the delivered ones, so the energy deviation is bounded by the
    // accumulated certified energy bound (the delivered per-call sum times
    // the |.|-1-norm of the spatial density rho = D/2 - see the adapter
    // above).
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::scf::RhfOptions scfOptions;
    scfOptions.useDiis = false;

    qcx::integrals::FockBuildOptions mixed;
    mixed.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    mixed.useDensityScreening = true;
    mixed.useCertifiedMixedPrecision = true;
    double mixedBoundSum = 0.0;
    double certifiedEnergyBound = 0.0;
    auto mixedResult =
        RunDirectRhf(*molecule, *basis, mixed, scfOptions, &mixedBoundSum, &certifiedEnergyBound);
    ASSERT_TRUE(mixedResult.has_value()) << mixedResult.error().message;
    ASSERT_TRUE(mixedResult->converged) << "iterations: " << mixedResult->iterations;
    EXPECT_GT(mixedBoundSum, 0.0); // the s-pair quartets must route
    EXPECT_GT(certifiedEnergyBound, 0.0);

    qcx::integrals::FockBuildOptions fp64Only;
    fp64Only.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    fp64Only.useDensityScreening = true;
    fp64Only.useCertifiedMixedPrecision = false;
    auto fp64Result = RunDirectRhf(*molecule, *basis, fp64Only, scfOptions);
    ASSERT_TRUE(fp64Result.has_value()) << fp64Result.error().message;
    ASSERT_TRUE(fp64Result->converged) << "iterations: " << fp64Result->iterations;

    EXPECT_LE(std::abs(mixedResult->totalEnergy - fp64Result->totalEnergy),
              certifiedEnergyBound * 1.001 + 1e-12)
        << "energy deviation exceeds the certified bound";
}

// The kernel dispatch at the SCF level. The full RHF trajectory runs
// twice - the cpuid-dispatched kernel vs forceScalarContract (the fallback
// pin) - and the trajectories must agree within the band below: the two
// kernel copies are the same arithmetic, and a deviation beyond the band is
// a dispatch bug. The band is zero on MSVC Release
// (/fp:precise never contracts) and on the Debug legs (nothing vectorizes
// at -O0). On GNU Release the -mavx2;-mfma copy fuses mul+add into FMA
// (-ffp-contract=fast is the GNU default) where the SSE2 scalar copy
// cannot, so each Fock build diverges at the last ulp elementwise and the
// converged trajectories with it (run 34243405768, the gcc Release
// leg: equal iteration counts, energy spread 9.2342e-11, density memcmp
// -103; the WSL gcc 15.2 reproduce at OMP_NUM_THREADS=1 measured 4.502e-11
// and 2.595e-11 - the trajectory amplifies the Fock ulp noise ~1e5-fold).
// The bands are the larger measured spreads times ~3.4 (the validation
// geometry; the local actuals are the gcc Release reading). Serial Fock
// builds (maxParallelChunks 1)
// keep the trajectory deterministic: the parallel path's critical-section
// reduction order scatters at the last ulp (fock_build_test.cpp documents
// the same).
// Not a convergence gate: the two walks this band compares already run the
// production defaults (RhfOptions{} above), and the band is the measured
// cross-kernel divergence of their trajectories, not a stopping rule.
constexpr double kTrajectoryEnergyBand = 5e-10;
constexpr double kTrajectoryDensityBand = 1e-10;

TEST(DirectRhfTest, Avx2DispatchPathsAreBitIdentical) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::FockBuildOptions defaultOptions;
    defaultOptions.maxParallelChunks = 1;
    auto defaultResult = RunDirectRhf(*molecule, *basis, defaultOptions, qcx::scf::RhfOptions{});
    ASSERT_TRUE(defaultResult.has_value()) << defaultResult.error().message;

    qcx::integrals::FockBuildOptions scalarOptions = defaultOptions;
    scalarOptions.forceScalarContract = true;
    auto scalarResult = RunDirectRhf(*molecule, *basis, scalarOptions, qcx::scf::RhfOptions{});
    ASSERT_TRUE(scalarResult.has_value()) << scalarResult.error().message;

    EXPECT_TRUE(defaultResult->converged) << "iterations: " << defaultResult->iterations;
    EXPECT_TRUE(scalarResult->converged) << "iterations: " << scalarResult->iterations;
    EXPECT_EQ(defaultResult->iterations, scalarResult->iterations);
    EXPECT_NEAR(defaultResult->totalEnergy, scalarResult->totalEnergy, kTrajectoryEnergyBand)
        << "the two kernel copies' converged energies diverge beyond the cross-platform band";
    const double densityMaxAbsDiff =
        (defaultResult->density - scalarResult->density).cwiseAbs().maxCoeff();
    EXPECT_LE(densityMaxAbsDiff, kTrajectoryDensityBand)
        << "the two kernel copies' densities diverge beyond the cross-platform band";
}

} // namespace
