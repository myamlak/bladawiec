// LeanDirectFockBuilder tests (lean_fock_build.hpp):
//   - the k=1 serial self pin: the builder's own bytes, bit-identical across
//     calls and across fresh builders (and with the scalar kernel forced -
//     the scalar/AVX2 copies are bit-identical by contract),
//   - the same serial pin at a fixture ABOVE the 96-pair flush cap, where a
//     pair is re-listed by later flushes instead of built once (the retention
//     coverage the H2O-sized pins cannot reach): the repeat call and a fresh
//     builder bit-identical, plus a re-cut flush decomposition held to 1e-12
//     (see the test - the cut is NOT value-free at the last bits, the reuse
//     is), with the pair-build count held at the call's reachable pair set as
//     the in-test retention evidence,
//   - the threaded path matching the serial path in fp64 accumulation order
//     (elementwise, NOT bit - the reduction order differs by construction),
//   - the survivor-set identity: the lean row walk over the canonical
//     pair-pair cells counts exactly the machinery's canonical
//     enumeration (CountSchwarzSurvivingPairs - one raw cell per
//     unordered pair-pair, the 8-fold orbit rep set),
//   - the parity of the Fock output with the direct machinery family at
//     equal preset (same survivor set, same per-quartet kernels, verbatim
//     symmetry arithmetic - only the quartet visitation order differs),
//     full and half (buildCoulombOnly / buildExchangeOnly) modes,
//   - a converged RHF walk on H2O/STO-3G landing on the pinned total energy
//     -74.96292827 (the in-tree anchor, +-1e-5), byte-identical across two
//     independent walks,
//   - the Create-time validation and the last-resort ceiling refusal (the
//     generous-slack shape, required/available/largest contributor),
//   - the pre-ramp envelope's identity with the builder's own number and
//     refusal text (EstimateLeanEnvelope - the admission a caller consults
//     before the core Hamiltonian exists),
//   - the per-call stats-out seam: fp64QuartetCount
//     == the independent survivor-walk count, serial AND threaded (the
//     per-window partials sum without double counting), totalWallTime
//     live, every other FockBuildStats field untouched at its default,
//     and the stats-carrying call byte-identical to the plain call,
//   - the stats fields added by the measurement probes: the prep split (pair
//     data / assembly / tile setup as disjoint spans inside eriPrepWallTime),
//     the pair-build count, and the per-window record (one entry per window,
//     each inside the call's span).
//   - the kernel-span probe (md_kernel_span.hpp): the three sub-spans of the
//     kernel time (VRR / ket transform / bra transform) and their
//     four counters - each observed, summing to at most the kernel span and
//     covering a majority of it, with the counters against counts the call
//     already publishes, and the instrument bit-invisible,
//   - the wave-grouping pin (the accumulator-residency bound): the SAME
//     window count and chunk plan at four wave widths - one wave, one window
//     per wave, waves of four with a tail, and the width the build's own rule
//     picks - lands on byte-identical Focks and a byte-identical converged
//     energy, so the grouping moves residency and never a value,
//   - the wave-residency instrument (env-gated, QCX_LEAN_WAVE_MEASURE): one
//     lean build at an alkane fixture with the retention band isolated,
//     printing the row-window plan and the envelope for an external peak-working-set
//     sampler.
//
// The tiny fixtures run everywhere; the alkane builds and the SCF walk are
// fast-mode-gated like the other engine tests. No scf link exists:
// the walk is this file's own damped fixed-point loop.

#include "alkane_sto3g.hpp"
#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "internal/fock_screen.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/lean_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/molecule/mass_properties.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::MakeAlkaneSto3gBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// H = T + V from the one-electron engines (the same helper as
// fock_build_test.cpp - the builders' shared test convention).
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

// A symmetric density with |D| <= 0.75 and a diagonal near 0.5 - the same
// magnitudes fock_build_test.cpp's PhysicalDensity uses (0.5 on the
// diagonal = the closed-shell spatial density's scale).
// (n, seed) is the size-then-seed order of the random-density helper.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::MatrixXd PhysicalDensity(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
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

// Bit equality of two matrices: same shape and identical bytes (the k=1
// pin comparison - the lean path's determinism is a byte contract).
bool BitIdentical(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        return false;
    }

    if (a.size() == 0)
    {
        return true;
    }

    return std::memcmp(a.data(), b.data(), static_cast<std::size_t>(a.size()) * sizeof(double)) ==
           0;
}

// The lean kTight k=1 options every bit-identity assertion uses.
qcx::integrals::LeanFockBuildOptions LeanSerialOptions() {
    qcx::integrals::LeanFockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kTight;
    options.maxParallelChunks = 1;
    return options;
}

// The machinery comparison options: fp64 only, no density or per-element
// screening, serial (k = 1 - deterministic, the equal-terms comparison).
qcx::integrals::FockBuildOptions MachinerySerialOptions() {
    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kTight;
    options.useDensityScreening = false;
    options.usePerElementScreening = false;
    options.useCertifiedMixedPrecision = false;
    options.maxParallelChunks = 1;
    return options;
}

// Asserts lean == machinery elementwise at 1e-11 on one (molecule, basis,
// density) - the parity legs' single comparison.
void ExpectLeanMatchesMachinery(const qcx::molecule::Molecule& molecule,
                                const qcx::basisset::BasisSet& basisSet,
                                // (core, densityTensor) names the Hamiltonian-then-density pair.
                                // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                const CpuTensor2& core,
                                const CpuTensor2& densityTensor,
                                bool coulombOnly = false,
                                bool exchangeOnly = false) {
    qcx::integrals::LeanFockBuildOptions leanOptions = LeanSerialOptions();
    leanOptions.buildCoulombOnly = coulombOnly;
    leanOptions.buildExchangeOnly = exchangeOnly;
    qcx::integrals::FockBuildOptions machineryOptions = MachinerySerialOptions();
    machineryOptions.buildCoulombOnly = coulombOnly;
    machineryOptions.buildExchangeOnly = exchangeOnly;
    auto lean =
        qcx::integrals::LeanDirectFockBuilder::Create(molecule, basisSet, core, leanOptions);

    ASSERT_TRUE(lean.has_value()) << lean.error().message;
    auto machinery =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, core, machineryOptions);

    ASSERT_TRUE(machinery.has_value()) << machinery.error().message;
    auto leanFock = lean->BuildFock(densityTensor);

    ASSERT_TRUE(leanFock.has_value()) << leanFock.error().message;
    auto machineryFock = machinery->BuildFock(densityTensor);

    ASSERT_TRUE(machineryFock.has_value()) << machineryFock.error().message;
    const Eigen::MatrixXd leanMatrix = ToMatrix(*leanFock);
    const Eigen::MatrixXd machineryMatrix = ToMatrix(*machineryFock);

    for (Eigen::Index i = 0; i < leanMatrix.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < leanMatrix.cols(); ++j)
        {
            EXPECT_NEAR(leanMatrix(i, j), machineryMatrix(i, j), 1e-11)
                << "element (" << i << "," << j << ") in the "
                << (coulombOnly    ? "J-only"
                    : exchangeOnly ? "K-only"
                                   : "full")
                << " mode";
        }
    }
}

// The independent canonical-cell walk of the survivor count: every
// unordered pair-pair (rows r, kets q <= r - the lean row walk's exact
// cells) whose bound product clears the machinery's neighbor-list cutoff.
std::size_t WalkSurvivorCount(const std::vector<double>& bounds,
                              qcx::integrals::AccuracyPreset accuracy) {
    const double cutoff =
        qcx::integrals::SchwarzThreshold(accuracy) * qcx::integrals::internal::kNeighborListSlack;
    const std::size_t n = bounds.size();
    std::size_t count = 0;

    for (std::size_t r = 0; r < n; ++r)
    {
        for (std::size_t q = 0; q <= r; ++q)
        {
            if (bounds[r] * bounds[q] >= cutoff)
            {
                ++count;
            }
        }
    }

    return count;
}

// The converged closed-shell walk result of one damped RHF fixed-point
// loop (no scf link - this test's own loop; the density fed to BuildFock
// is the SPATIAL density rho = C_occ C_occ^T - the driver's
// MakeRhfFockBuilder convention, where the builder returns
// H + 2J(rho) - K(rho) with the J 2.0 baked in).
struct RhfWalkResult {
    double totalEnergy;
    Eigen::MatrixXd density;
};

// The damped RHF walk over one builder: canonical orthogonalization
// (S^-1/2), core-H guess, alpha = 0.5 damping, converged when the
// undamped density step's Frobenius norm drops below 1e-8.
qcx::Result<RhfWalkResult> WalkH2oToConvergence(const qcx::integrals::LeanDirectFockBuilder& lean,
                                                const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const Eigen::MatrixXd& coreH) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    const Eigen::MatrixXd s = ToMatrix(*overlap);
    const Eigen::Index n = s.rows();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sSolver(s);

    if (sSolver.eigenvalues()(0) <= 0.0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "non-positive overlap eigenvalue"});
    }

    // X = S^-1/2 via the overlap's eigenelements.
    const Eigen::MatrixXd x = sSolver.eigenvectors() *
                              sSolver.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() *
                              sSolver.eigenvectors().transpose();
    constexpr Eigen::Index kOccupied = 5; // H2O: 10 electrons, closed shell.
    constexpr std::size_t kMaxIterations = 300;
    constexpr double kConvergedStep = 1e-8;
    constexpr double kDamping = 0.5;

    Eigen::MatrixXd density = Eigen::MatrixXd::Zero(n, n);
    Eigen::MatrixXd fock = coreH;
    bool converged = false;

    for (std::size_t iteration = 0; iteration < kMaxIterations; ++iteration)
    {
        auto densityTensor = ToTensor(density);

        if (!densityTensor.has_value())
        {
            return std::unexpected(densityTensor.error());
        }

        auto built = lean.BuildFock(*densityTensor);

        if (!built.has_value())
        {
            return std::unexpected(built.error());
        }

        fock = ToMatrix(*built);

        // The generalized eigenproblem in the orthogonalized basis.
        const Eigen::MatrixXd transformed = x.transpose() * fock * x;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> fSolver(transformed);
        const Eigen::MatrixXd coefficients = x * fSolver.eigenvectors();
        const Eigen::MatrixXd occupied = coefficients.leftCols(kOccupied);
        const Eigen::MatrixXd nextDensity = occupied * occupied.transpose();
        const double step = (nextDensity - density).norm();

        if (step < kConvergedStep)
        {
            converged = true;
        }

        density = (1.0 - kDamping) * density + kDamping * nextDensity;

        if (converged)
        {
            break;
        }
    }

    if (!converged)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kConvergenceFailure,
                                          "the lean H2O walk did not converge in 300 iterations"});
    }

    // The final Fock at the converged (damped) density.
    auto finalDensity = ToTensor(density);

    if (!finalDensity.has_value())
    {
        return std::unexpected(finalDensity.error());
    }

    auto finalFock = lean.BuildFock(*finalDensity);

    if (!finalFock.has_value())
    {
        return std::unexpected(finalFock.error());
    }

    // E = Tr(rho (H + F)) + V_nn (the closed-shell form of Tr(D(H + F))/2
    // with D = 2 rho).
    const double electronic = (density.cwiseProduct(coreH + ToMatrix(*finalFock))).sum();
    const double total = electronic + qcx::molecule::NuclearRepulsionEnergy(molecule);

    return RhfWalkResult{total, std::move(density)};
}

} // namespace

TEST(LeanDirectFockBuilderTest, SerialPathPinsItsOwnBytesAcrossCallsAndBuilders) {
    // The k=1 self pin (the lean path pins ITS OWN bytes): the
    // same density through the same builder twice and through a fresh
    // builder must produce bit-identical Fock matrices; forcing the scalar
    // contract kernel changes nothing (the scalar/AVX2 copies are
    // bit-identical by contract).
    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;
    qcx::integrals::LeanFockBuildOptions scalarOptions = LeanSerialOptions();
    scalarOptions.forceScalarContract = true;
    auto builderA = qcx::integrals::LeanDirectFockBuilder::Create(
        *molecule, *basis, *core, LeanSerialOptions());

    ASSERT_TRUE(builderA.has_value()) << builderA.error().message;
    auto builderB = qcx::integrals::LeanDirectFockBuilder::Create(
        *molecule, *basis, *core, LeanSerialOptions());

    ASSERT_TRUE(builderB.has_value()) << builderB.error().message;
    auto builderScalar =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, scalarOptions);

    ASSERT_TRUE(builderScalar.has_value()) << builderScalar.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(7, 20260908);
    auto densityTensor = ToTensor(density);

    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    auto first = builderA->BuildFock(*densityTensor);

    ASSERT_TRUE(first.has_value()) << first.error().message;
    auto second = builderA->BuildFock(*densityTensor);

    ASSERT_TRUE(second.has_value()) << second.error().message;
    auto fresh = builderB->BuildFock(*densityTensor);

    ASSERT_TRUE(fresh.has_value()) << fresh.error().message;
    auto scalar = builderScalar->BuildFock(*densityTensor);

    ASSERT_TRUE(scalar.has_value()) << scalar.error().message;
    const Eigen::MatrixXd firstMatrix = ToMatrix(*first);

    EXPECT_TRUE(BitIdentical(firstMatrix, ToMatrix(*second))) << "repeat call drifted";
    EXPECT_TRUE(BitIdentical(firstMatrix, ToMatrix(*fresh))) << "fresh builder drifted";
    EXPECT_TRUE(BitIdentical(firstMatrix, ToMatrix(*scalar))) << "scalar kernel drifted";
    EXPECT_TRUE(firstMatrix.isApprox(firstMatrix.transpose())) << "Fock not symmetric";

    // The Create-time envelope (the driver's memory-audit number): always
    // reported, identical across fresh builders, positive on every real
    // system, and monotone in the system size (C8H18 > H2O at the same
    // options).
    EXPECT_GT(builderA->EstimatedPeakBytes(), 0.0);
    EXPECT_DOUBLE_EQ(builderA->EstimatedPeakBytes(), builderB->EstimatedPeakBytes());
    EXPECT_DOUBLE_EQ(builderA->EstimatedPeakBytes(), builderScalar->EstimatedPeakBytes());
}

TEST(LeanDirectFockBuilderTest, ReusedPairArenaPinsItsOwnBytesAboveTheFlushCap) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    // The retention pin. Every other
    // bit-identity pin in this file is H2O-sized, and H2O/STO-3G's 15 shell
    // pairs never reach the lean flush's distinct-pair cap
    // (kLeanFlushPairCap = 96, lean_fock_build.cpp), so those pins never
    // re-list a pair across flushes and the store retention is outside their
    // reach. C8H18/STO-3G crosses the cap by an order of magnitude: 42 shells
    // (the parser splits the carbon SP block into two, basisset
    // ParseShellType) give 903 canonical pairs, 851 of them reachable at
    // kTight, so the k = 1 call runs ~3,400 flushes under the 96-pair cap and
    // re-lists the same pairs thousands of times - measured 320,269 pair
    // slots listed over the call.
    //
    // Since the flush pair-data teardown was dropped the build-once guard
    // fires instead: the store survives every flush, so a re-listed pair's
    // not-built marker is still set and BuildChunkPairData skips it. The call
    // BUILDS 851 pairs - one per reachable pair, 0.94x the canonical count
    // and 1/376 of the slots it lists - where the teardown-era call built one
    // per slot (320,269). The reuse this pin protects is that skip: the arena
    // a pair was built into is still there, so the build does not run again.
    // (The S1/E-slice in-place re-fit is no longer exercised HERE for the
    // same reason - no pair is built twice in a call - and stays covered by
    // the machinery's chunk pass, which does tear down and re-fit.) The bytes
    // must still be exactly a fresh builder's: retention is capacity-only,
    // never value.
    auto basis = MakeAlkaneSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeAlkaneSto3g(8);

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;
    const std::size_t n = core->Shape()[0];

    ASSERT_EQ(n, 58u) << "C8H18/STO-3G basis functions";
    const Eigen::MatrixXd density = PhysicalDensity(n, 20260908);
    auto densityTensor = ToTensor(density);

    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    auto builderA = qcx::integrals::LeanDirectFockBuilder::Create(
        *molecule, *basis, *core, LeanSerialOptions());

    ASSERT_TRUE(builderA.has_value()) << builderA.error().message;
    auto builderB = qcx::integrals::LeanDirectFockBuilder::Create(
        *molecule, *basis, *core, LeanSerialOptions());

    ASSERT_TRUE(builderB.has_value()) << builderB.error().message;
    qcx::integrals::FockBuildStats stats;
    auto first = builderA->BuildFock(*densityTensor, &stats);

    ASSERT_TRUE(first.has_value()) << first.error().message;
    auto second = builderA->BuildFock(*densityTensor);

    ASSERT_TRUE(second.has_value()) << second.error().message;
    auto fresh = builderB->BuildFock(*densityTensor);

    ASSERT_TRUE(fresh.has_value()) << fresh.error().message;
    // The re-cut leg, the one that gives this pin its teeth. Two runs at the
    // same options reuse the same arenas in the same order, so a stale-carry
    // bug in the re-fit sits in both runs identically and the repeat/fresh
    // comparisons below still agree - verified, not argued: with the
    // ketTransform re-fit mutated to skip its zero-fill on a retained vector
    // (a temporary edit, reverted) those two legs stayed green. The flush
    // boundary is a memory-management choice, so a re-cut decomposition must
    // land on the same physical result: a tiny maxBatchBytes makes the
    // element cap bind (the 512 MB default leaves the 96-pair cap in charge
    // of every boundary), re-cutting every flush and so re-fitting a
    // different set of retained arenas.
    //
    // This leg alone is a 1e-12 comparison, never bit equality, because the
    // CUT is not value-free at the last bits: the measured max |delta|
    // against the default decomposition is 8.9e-15, and it is unchanged when
    // the retention is switched off (primPairs cleared per flush - a
    // temporary edit too), so the residual is the cut and not the reuse. The
    // tolerance leaves that floor ~100x of headroom while a retention error
    // of this pin's class moves the same comparison by 2.3e2, so the leg
    // fails on the bug it exists for and passes on the kernel's own
    // batch-order noise. The same-options legs stay bitwise: that equality is
    // achievable, this one is not, and loosening them would be a different
    // (and a wrong) pin.
    qcx::integrals::LeanFockBuildOptions recutOptions = LeanSerialOptions();
    recutOptions.maxBatchBytes = 4096;
    auto builderRecut =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, recutOptions);

    ASSERT_TRUE(builderRecut.has_value()) << builderRecut.error().message;
    qcx::integrals::FockBuildStats recutStats;
    auto recut = builderRecut->BuildFock(*densityTensor, &recutStats);

    ASSERT_TRUE(recut.has_value()) << recut.error().message;
    const Eigen::MatrixXd firstMatrix = ToMatrix(*first);
    const Eigen::MatrixXd recutMatrix = ToMatrix(*recut);
    double recutDelta = 0.0;

    for (Eigen::Index i = 0; i < firstMatrix.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < firstMatrix.cols(); ++j)
        {
            recutDelta = std::max(recutDelta, std::abs(firstMatrix(i, j) - recutMatrix(i, j)));
        }
    }

    EXPECT_TRUE(BitIdentical(firstMatrix, ToMatrix(*second))) << "repeat call drifted";
    EXPECT_TRUE(BitIdentical(firstMatrix, ToMatrix(*fresh))) << "fresh builder drifted";
    EXPECT_LT(recutDelta, 1e-12) << "flush re-cut drifted: max |delta| " << recutDelta
                                 << " (the measured decomposition floor is 8.9e-15)";
    EXPECT_TRUE(firstMatrix.isApprox(firstMatrix.transpose())) << "Fock not symmetric";

    // The retention evidence, re-pointed at the property this pin exists for
    // (the pair-rebuild instrument lane): pairBuildCount is a BUILD count, so
    // the count states the retention instead of standing in for it. The store
    // survives every flush of the call (k = 1 runs one window) and
    // BuildChunkPairData skips a pair whose not-built marker is still set, so
    // each of the fixture's 851 reachable pairs is built exactly once: 851
    // builds against 903 canonical pairs and the 320,269 pair slots the same
    // call's flushes list. The listing volume is what this counter read before
    // the teardown was dropped, and what it reads again if the drop is
    // reverted.
    //
    // The upper bound IS the retention property: a pair is built at most once
    // per window, so a k = 1 call cannot build more pairs than the fixture has
    // canonical pairs. Revert the teardown drop, or stop the not-built guard
    // from firing, and every re-listed pair is rebuilt - the count returns to
    // the listed volume and this assertion fails. The floor keeps the pin from
    // passing vacuously: the walk must still reach the fixture's pair space
    // (measured 851 of 903), so an over-tight flush cap or a coarser screen
    // reports itself here instead of leaving the pin with nothing to retain.
    const std::size_t pairCount = 903;
    EXPECT_LE(stats.pairBuildCount, pairCount)
        << "pair builds " << stats.pairBuildCount << " > the " << pairCount
        << " canonical shell pairs: a pair was built twice in a k = 1 call, so the "
           "flush retention the reuse rests on is not holding (measured 851 builds "
           "against the 320,269 pair slots the call lists)";
    EXPECT_GT(stats.pairBuildCount, pairCount / 2)
        << "pair builds " << stats.pairBuildCount << " <= half the " << pairCount
        << " canonical shell pairs: the call listed too few pairs for this pin to "
           "stand in for the retention path (measured 851 of 903)";
    // The re-cut leg: a 4096-byte batch cap re-cuts the same quartets into a
    // different decomposition - the batches by 13.4x (measured 7,566 ->
    // 101,200) and the flush list by a hair (+0.5% of its slots) - and the
    // built set must follow neither cut. It is the call's reachable pair set,
    // so both calls build the same 851 pairs; an inequality means a build
    // followed the re-cutting rather than the pair, which is exactly what a
    // reverted teardown produces (the re-cut then lists more, so it builds
    // more too). The batch leg comes first so the equality cannot pass on a
    // re-cut that changed nothing.
    EXPECT_GT(recutStats.batchCount, stats.batchCount)
        << "the re-cut assembled " << recutStats.batchCount << " batches against the default's "
        << stats.batchCount
        << ": the two calls are not two decompositions, so the equality below proves nothing";
    EXPECT_EQ(recutStats.pairBuildCount, stats.pairBuildCount)
        << "the re-cut built " << recutStats.pairBuildCount << " pairs against the default's "
        << stats.pairBuildCount
        << ": the built set is the call's reachable pair set, so a flush re-cutting "
           "cannot move it";
}

TEST(LeanDirectFockBuilderTest, ThreadedPathMatchesTheSerialPathInAccumulationOrder) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    // The k > 1 path reduces per-window accumulators in window order - the
    // per-element summation order differs from k = 1, so the match is
    // elementwise fp64 accumulation order, never bit equality.
    auto basis = MakeAlkaneSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeAlkaneSto3g(8);

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;
    const std::size_t n = core->Shape()[0];

    ASSERT_EQ(n, 58u) << "C8H18/STO-3G basis functions";
    const Eigen::MatrixXd density = PhysicalDensity(n, 20260908);
    auto densityTensor = ToTensor(density);

    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    qcx::integrals::LeanFockBuildOptions threadedOptions = LeanSerialOptions();
    threadedOptions.maxParallelChunks = 6;
    auto serialBuilder = qcx::integrals::LeanDirectFockBuilder::Create(
        *molecule, *basis, *core, LeanSerialOptions());

    ASSERT_TRUE(serialBuilder.has_value()) << serialBuilder.error().message;
    auto threadedBuilder =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, threadedOptions);

    ASSERT_TRUE(threadedBuilder.has_value()) << threadedBuilder.error().message;
    auto serialFock = serialBuilder->BuildFock(*densityTensor);

    ASSERT_TRUE(serialFock.has_value()) << serialFock.error().message;
    auto threadedFock = threadedBuilder->BuildFock(*densityTensor);

    ASSERT_TRUE(threadedFock.has_value()) << threadedFock.error().message;
    const Eigen::MatrixXd serial = ToMatrix(*serialFock);
    const Eigen::MatrixXd threaded = ToMatrix(*threadedFock);

    for (Eigen::Index i = 0; i < serial.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < serial.cols(); ++j)
        {
            EXPECT_NEAR(serial(i, j), threaded(i, j), 1e-11) << "element (" << i << "," << j << ")";
        }
    }
}

TEST(LeanDirectFockBuilderTest, SurvivorSetMatchesTheCanonicalEnumerationCount) {
    // The symmetry structural item: the lean row walk's cells are the
    // canonical pair-pair cells - one raw cell per unordered pair-pair
    // (the 8-fold orbit rep set) - so its independent survivor count must
    // equal the machinery's canonical enumeration
    // (CountSchwarzSurvivingPairs, the BuildNeighborList candidate count
    // identity) at every preset.
    auto h2oBasis = MakeH2oSto3gBasis();

    ASSERT_TRUE(h2oBasis.has_value()) << h2oBasis.error().message;
    auto h2oMolecule = MakeH2oSto3g();

    ASSERT_TRUE(h2oMolecule.has_value()) << h2oMolecule.error().message;
    auto h2oBounds = qcx::integrals::ComputeSchwarzBounds(*h2oMolecule, *h2oBasis);

    ASSERT_TRUE(h2oBounds.has_value()) << h2oBounds.error().message;
    auto alkaneBasis = MakeAlkaneSto3gBasis();

    ASSERT_TRUE(alkaneBasis.has_value()) << alkaneBasis.error().message;
    auto alkaneMolecule = MakeAlkaneSto3g(8);

    ASSERT_TRUE(alkaneMolecule.has_value()) << alkaneMolecule.error().message;
    auto alkaneBounds = qcx::integrals::ComputeSchwarzBounds(*alkaneMolecule, *alkaneBasis);

    ASSERT_TRUE(alkaneBounds.has_value()) << alkaneBounds.error().message;

    for (const qcx::integrals::AccuracyPreset preset : {qcx::integrals::AccuracyPreset::kLoose,
                                                        qcx::integrals::AccuracyPreset::kNormal,
                                                        qcx::integrals::AccuracyPreset::kTight})
    {
        EXPECT_EQ(WalkSurvivorCount(*h2oBounds, preset),
                  qcx::integrals::internal::CountSchwarzSurvivingPairs(*h2oBounds, preset))
            << "H2O preset mismatch";
        EXPECT_EQ(WalkSurvivorCount(*alkaneBounds, preset),
                  qcx::integrals::internal::CountSchwarzSurvivingPairs(*alkaneBounds, preset))
            << "C8H18 preset mismatch";
    }
}

TEST(LeanDirectFockBuilderTest, StatsOutFillsTheSurvivorCountAndTheWallTime) {
    // The per-call stats seam: a stats-carrying call on
    // the k=1 path fills fp64QuartetCount with exactly the call's
    // screened-in survivors (the independent survivor-walk count), a live
    // totalWallTime, the split probe (batchCount plus the
    // eri/contract phase spans), the prep probe (the prep split's
    // three parts and the pair-build count) and the window probe (the
    // caller's per-window record) - NOTHING else: the remaining FockBuildStats
    // fields are not lean concepts (no fp32 lane, no cache, no screening
    // tiers, no merge chain, no calibration terms) and must stay at
    // their defaults. The stats-carrying call's Fock bytes equal the
    // plain call's (the instrument never perturbs the pinned path).
    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto bounds = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);

    ASSERT_TRUE(bounds.has_value()) << bounds.error().message;
    auto builder = qcx::integrals::LeanDirectFockBuilder::Create(
        *molecule, *basis, *core, LeanSerialOptions());

    ASSERT_TRUE(builder.has_value()) << builder.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(7, 20260908);
    auto densityTensor = ToTensor(density);

    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    auto plainFock = builder->BuildFock(*densityTensor);

    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;
    qcx::integrals::FockBuildStats firstStats;
    std::vector<std::chrono::nanoseconds> serialWindowTimes;
    firstStats.windowWallTimesOut = &serialWindowTimes;
    auto firstFock = builder->BuildFock(*densityTensor, &firstStats);

    ASSERT_TRUE(firstFock.has_value()) << firstFock.error().message;
    qcx::integrals::FockBuildStats secondStats;
    auto secondFock = builder->BuildFock(*densityTensor, &secondStats);

    ASSERT_TRUE(secondFock.has_value()) << secondFock.error().message;
    const std::size_t expected = WalkSurvivorCount(*bounds, qcx::integrals::AccuracyPreset::kTight);
    const Eigen::MatrixXd plain = ToMatrix(*plainFock);

    EXPECT_TRUE(BitIdentical(plain, ToMatrix(*firstFock))) << "stats call drifted";
    EXPECT_TRUE(BitIdentical(plain, ToMatrix(*secondFock))) << "repeat stats call drifted";
    EXPECT_EQ(firstStats.fp64QuartetCount, expected);
    EXPECT_EQ(secondStats.fp64QuartetCount, expected)
        << "the per-call count must not leak across calls";
    EXPECT_GT(firstStats.totalWallTime.count(), 0);
    // The lean-backed phase accounting (the split probe): the call
    // assembled at least one class batch and both phase spans were observed,
    // so their sum stays inside the whole-call span.
    EXPECT_GT(firstStats.batchCount, 0u);
    EXPECT_GT(firstStats.eriWallTime.count(), 0);
    EXPECT_GT(firstStats.contractWallTime.count(), 0);
    // The prep sub-boundary: a sub-span of the ERI span, never past it.
    EXPECT_GE(firstStats.eriPrepWallTime.count(), 0);
    EXPECT_LE(firstStats.eriPrepWallTime.count(), firstStats.eriWallTime.count());
    EXPECT_LE((firstStats.eriWallTime + firstStats.contractWallTime).count(),
              firstStats.totalWallTime.count());
    // The prep split (the prep probe): the pair-data build and the
    // batch assembly were both observed (the flush builds pairs and
    // assembles batches), the three parts are disjoint spans inside the
    // prep span, and the pair-build count is the pair-proportional
    // denominator the pair-part span is read against.
    EXPECT_GT(firstStats.pairBuildCount, 0u);
    EXPECT_GT(firstStats.eriPrepPairDataWallTime.count(), 0);
    EXPECT_GT(firstStats.eriPrepAssembleWallTime.count(), 0);
    EXPECT_GE(firstStats.eriPrepTileSetupWallTime.count(), 0);
    EXPECT_LE((firstStats.eriPrepPairDataWallTime + firstStats.eriPrepAssembleWallTime +
               firstStats.eriPrepTileSetupWallTime)
                  .count(),
              firstStats.eriPrepWallTime.count());
    // The per-window record (the window probe): the k = 1 call ran one
    // window, whose elapsed span stays inside the whole call's span.
    ASSERT_EQ(serialWindowTimes.size(), 1u) << "the k=1 call runs exactly one window";
    EXPECT_GT(serialWindowTimes[0].count(), 0);
    EXPECT_LE(serialWindowTimes[0].count(), firstStats.totalWallTime.count());
    // The fields the lean path never fills (see the overload's contract):
    // defaults only.
    EXPECT_EQ(firstStats.fp32QuartetCount, 0u);
    EXPECT_EQ(firstStats.mergeWallTime.count(), 0);
    EXPECT_EQ(firstStats.concurrentSlots, 1u);
    EXPECT_EQ(firstStats.quartetKeysOut, nullptr);
    EXPECT_EQ(secondStats.windowWallTimesOut, nullptr)
        << "the builder only fills the per-window record the caller hands it";
    EXPECT_EQ(firstStats.cacheHitFp64QuartetCount, 0u);
    EXPECT_EQ(firstStats.cacheHitFp32QuartetCount, 0u);
    EXPECT_EQ(firstStats.elementDrops, 0u);
    EXPECT_EQ(firstStats.significantPairCount, 0u);
    EXPECT_EQ(firstStats.primitiveProductSum, 0u);
    // The k > 1 path counts the same survivors (per-window partials).
    qcx::integrals::LeanFockBuildOptions threadedOptions = LeanSerialOptions();
    threadedOptions.maxParallelChunks = 4;
    auto threaded =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, threadedOptions);

    ASSERT_TRUE(threaded.has_value()) << threaded.error().message;
    qcx::integrals::FockBuildStats threadedStats;
    std::vector<std::chrono::nanoseconds> threadedWindowTimes;
    threadedStats.windowWallTimesOut = &threadedWindowTimes;
    auto threadedFock = threaded->BuildFock(*densityTensor, &threadedStats);

    ASSERT_TRUE(threadedFock.has_value()) << threadedFock.error().message;
    EXPECT_EQ(threadedStats.fp64QuartetCount, expected)
        << "the per-window partials must sum exactly once per survivor";
    // The per-window record on the k > 1 path: one entry per window, in
    // window order, each a live span of its own window (the windows overlap
    // by construction, so each entry only has to stay inside the call).
    ASSERT_EQ(threadedWindowTimes.size(), 4u) << "one entry per configured window";

    for (const std::chrono::nanoseconds entry : threadedWindowTimes)
    {
        EXPECT_GT(entry.count(), 0);
        EXPECT_LE(entry.count(), threadedStats.totalWallTime.count());
    }
}

TEST(LeanDirectFockBuilderTest, StatsOutCountsTheSameSurvivorsOnTheAlkanePath) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    // The threaded partial-sum pin on a larger fixture: the k > 1 stats
    // call counts exactly the same screened-in quartets as the serial
    // walk of the same cutoff (windowCount = 6, the alkane path).
    auto basis = MakeAlkaneSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeAlkaneSto3g(8);

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto bounds = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);

    ASSERT_TRUE(bounds.has_value()) << bounds.error().message;
    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd density = PhysicalDensity(n, 20260908);
    auto densityTensor = ToTensor(density);

    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    qcx::integrals::LeanFockBuildOptions serialOptions = LeanSerialOptions();
    qcx::integrals::LeanFockBuildOptions threadedOptions = LeanSerialOptions();
    threadedOptions.maxParallelChunks = 6;
    auto serialBuilder =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, serialOptions);

    ASSERT_TRUE(serialBuilder.has_value()) << serialBuilder.error().message;
    auto threadedBuilder =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, threadedOptions);

    ASSERT_TRUE(threadedBuilder.has_value()) << threadedBuilder.error().message;
    qcx::integrals::FockBuildStats serialStats;
    qcx::integrals::FockBuildStats threadedStats;
    std::vector<std::chrono::nanoseconds> serialWindowTimes;
    std::vector<std::chrono::nanoseconds> alkaneWindowTimes;
    serialStats.windowWallTimesOut = &serialWindowTimes;
    threadedStats.windowWallTimesOut = &alkaneWindowTimes;
    auto serialFock = serialBuilder->BuildFock(*densityTensor, &serialStats);

    ASSERT_TRUE(serialFock.has_value()) << serialFock.error().message;
    auto threadedFock = threadedBuilder->BuildFock(*densityTensor, &threadedStats);

    ASSERT_TRUE(threadedFock.has_value()) << threadedFock.error().message;
    const std::size_t expected = WalkSurvivorCount(*bounds, qcx::integrals::AccuracyPreset::kTight);

    EXPECT_EQ(serialStats.fp64QuartetCount, expected);
    EXPECT_EQ(threadedStats.fp64QuartetCount, expected)
        << "the per-window partials must sum exactly once per survivor";
    // The per-window record on the larger fixture (the window probe):
    // the serial walk ran one window, the six-window walk six - each entry
    // a live span of its own window, inside the call's own span.
    EXPECT_EQ(serialWindowTimes.size(), 1u);
    EXPECT_EQ(alkaneWindowTimes.size(), 6u) << "one entry per configured window";

    for (const std::chrono::nanoseconds entry : alkaneWindowTimes)
    {
        EXPECT_GT(entry.count(), 0);
        EXPECT_LE(entry.count(), threadedStats.totalWallTime.count());
    }
}

TEST(LeanDirectFockBuilderTest, KernelSpanProbeSubdividesTheKernelAndStaysInsideIt) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    // The kernel-span probe: the kernel time (eriWallTime -
    // eriPrepWallTime) now reports three disjoint sub-spans - the VRR/Boys
    // phase, the ket transform and the bra transform - beside the four
    // counters that denominate them. The pin is the decomposition's own
    // arithmetic, so it cannot silently stop adding up:
    //
    //  * each phase was observed on a call that ran class batches;
    //  * the three sum to AT MOST the kernel span they subdivide, and cover
    //    a substantial share of it (the residual - the per-batch layout walk,
    //    scratch.assign, the group splitting and the loop overheads - belongs
    //    to no phase, so the sum is an inequality by contract, not an
    //    identity);
    //  * the counters agree with counts the call already publishes: the
    //    recurrence runs at least one primitive quadruple per surviving
    //    quartet, the pass-1 group census never exceeds the primitive-pass
    //    count, and the seam calls never exceed the two transforms' call
    //    totals (the ket transform is called once per primitive pass and the
    //    bra transform once per surviving quartet).
    //
    // The instrument must also be invisible: the stats-carrying calls stay
    // bit-identical to a stats-free one, and a second stats call reports the
    // same counters (the per-call zeroing, no leak across calls).
    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto builder = qcx::integrals::LeanDirectFockBuilder::Create(
        *molecule, *basis, *core, LeanSerialOptions());

    ASSERT_TRUE(builder.has_value()) << builder.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(7, 20260912);
    auto densityTensor = ToTensor(density);

    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    auto plainFock = builder->BuildFock(*densityTensor);

    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;
    qcx::integrals::FockBuildStats firstStats;
    std::vector<std::chrono::nanoseconds> windowTimes;
    firstStats.windowWallTimesOut = &windowTimes;
    auto firstFock = builder->BuildFock(*densityTensor, &firstStats);

    ASSERT_TRUE(firstFock.has_value()) << firstFock.error().message;
    qcx::integrals::FockBuildStats secondStats;
    auto secondFock = builder->BuildFock(*densityTensor, &secondStats);

    ASSERT_TRUE(secondFock.has_value()) << secondFock.error().message;
    const Eigen::MatrixXd plain = ToMatrix(*plainFock);

    EXPECT_TRUE(BitIdentical(plain, ToMatrix(*firstFock))) << "stats call drifted";
    EXPECT_TRUE(BitIdentical(plain, ToMatrix(*secondFock))) << "repeat stats call drifted";
    // The kernel span the probe subdivides: the ERI span less its prep part
    // (the same derivation the analyzer reads off the row).
    const long long kernel = firstStats.eriWallTime.count() - firstStats.eriPrepWallTime.count();
    const long long phases = firstStats.kernelVrrWallTime.count() +
                             firstStats.kernelKetWallTime.count() +
                             firstStats.kernelBraWallTime.count();

    ASSERT_GT(kernel, 0) << "the k = 1 call has no kernel span to subdivide";
    EXPECT_GT(firstStats.kernelVrrWallTime.count(), 0);
    EXPECT_GT(firstStats.kernelKetWallTime.count(), 0);
    EXPECT_GT(firstStats.kernelBraWallTime.count(), 0);
    EXPECT_LE(phases, kernel) << "the kernel phases exceeded the kernel span: vrr="
                              << firstStats.kernelVrrWallTime.count()
                              << " ket=" << firstStats.kernelKetWallTime.count()
                              << " bra=" << firstStats.kernelBraWallTime.count()
                              << " kernel=" << kernel;
    EXPECT_GE(2 * phases, kernel)
        << "the kernel phases covered less than half the kernel span - the "
           "decomposition has stopped adding up: vrr="
        << firstStats.kernelVrrWallTime.count() << " ket=" << firstStats.kernelKetWallTime.count()
        << " bra=" << firstStats.kernelBraWallTime.count() << " kernel=" << kernel;
    // The denominators, against counts the call already publishes.
    EXPECT_GT(firstStats.kernelPrimPasses, 0u);
    EXPECT_GT(firstStats.kernelGroupCount, 0u);
    EXPECT_GE(firstStats.kernelPrimPasses, firstStats.kernelGroupCount)
        << "every pass-1 group runs at least one ket primitive pair";
    EXPECT_GE(firstStats.kernelVrrQuadruples, firstStats.fp64QuartetCount)
        << "every surviving quartet exists at one ket primitive pair and one bra "
           "primitive pair at the least";
    EXPECT_LE(firstStats.kernelGateSeamCalls,
              firstStats.kernelPrimPasses + firstStats.fp64QuartetCount)
        << "the seam count exceeds the ket + bra transform call totals";
    // No leak across calls: the second stats call reports the same counts
    // (the spans differ by measurement noise, the counts do not).
    EXPECT_EQ(secondStats.kernelVrrQuadruples, firstStats.kernelVrrQuadruples);
    EXPECT_EQ(secondStats.kernelPrimPasses, firstStats.kernelPrimPasses);
    EXPECT_EQ(secondStats.kernelGroupCount, firstStats.kernelGroupCount);
    EXPECT_EQ(secondStats.kernelGateSeamCalls, firstStats.kernelGateSeamCalls);
    // The k > 1 leg: the same inequality against the window-accumulated
    // spans (the phases are sub-spans of the eri span at every k, so the
    // accumulated sum stays inside the accumulated parent).
    qcx::integrals::LeanFockBuildOptions threadedOptions = LeanSerialOptions();
    threadedOptions.maxParallelChunks = 4;
    auto threaded =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, threadedOptions);

    ASSERT_TRUE(threaded.has_value()) << threaded.error().message;
    qcx::integrals::FockBuildStats threadedStats;
    std::vector<std::chrono::nanoseconds> threadedWindowTimes;
    threadedStats.windowWallTimesOut = &threadedWindowTimes;
    auto threadedFock = threaded->BuildFock(*densityTensor, &threadedStats);

    ASSERT_TRUE(threadedFock.has_value()) << threadedFock.error().message;
    // The work denominator is decomposition-invariant: it sums the same
    // per-task primitive-pair lengths over the same survivor set, so the
    // window split cannot move it. The STRUCTURAL counters cannot make that
    // claim - kernelGroupCount and kernelPrimPasses count the pass-1 group
    // runs and the batch cut that carries them, and a different window split
    // composes different flushes, hence different batches, hence a different
    // group census for the same work. Only the work figure is pinned here;
    // the census is reported, not asserted.
    EXPECT_EQ(threadedStats.kernelVrrQuadruples, firstStats.kernelVrrQuadruples)
        << "the primitive-quadruple count is per-window partials summed once, "
           "over a window-split-invariant population";
    EXPECT_GT(threadedStats.kernelPrimPasses, 0u);
    EXPECT_GE(threadedStats.kernelPrimPasses, threadedStats.kernelGroupCount);
    const long long threadedKernel =
        threadedStats.eriWallTime.count() - threadedStats.eriPrepWallTime.count();
    const long long threadedPhases = threadedStats.kernelVrrWallTime.count() +
                                     threadedStats.kernelKetWallTime.count() +
                                     threadedStats.kernelBraWallTime.count();

    EXPECT_GT(threadedKernel, 0);
    EXPECT_LE(threadedPhases, threadedKernel);
    EXPECT_GE(2 * threadedPhases, threadedKernel);
}

TEST(LeanDirectFockBuilderTest, MatchesTheDirectFamilyAtEqualPreset) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    // The parity leg: same survivor set (the identical canonical cells
    // under the identical cutoff), the same ERI kernels, the verbatim
    // symmetry arithmetic - only the quartet visitation order differs, so
    // the deviation is fp64 accumulation order. H2O and C8H18 at kTight.
    auto h2oBasis = MakeH2oSto3gBasis();

    ASSERT_TRUE(h2oBasis.has_value()) << h2oBasis.error().message;
    auto h2oMolecule = MakeH2oSto3g();

    ASSERT_TRUE(h2oMolecule.has_value()) << h2oMolecule.error().message;
    auto coreH2o = BuildCoreHamiltonian(*h2oMolecule, *h2oBasis);

    ASSERT_TRUE(coreH2o.has_value()) << coreH2o.error().message;
    const Eigen::MatrixXd densityH2o = PhysicalDensity(7, 20260909);
    auto densityTensorH2o = ToTensor(densityH2o);

    ASSERT_TRUE(densityTensorH2o.has_value()) << densityTensorH2o.error().message;
    auto alkaneBasis = MakeAlkaneSto3gBasis();

    ASSERT_TRUE(alkaneBasis.has_value()) << alkaneBasis.error().message;
    auto alkaneMolecule = MakeAlkaneSto3g(8);

    ASSERT_TRUE(alkaneMolecule.has_value()) << alkaneMolecule.error().message;
    auto coreAlkane = BuildCoreHamiltonian(*alkaneMolecule, *alkaneBasis);

    ASSERT_TRUE(coreAlkane.has_value()) << coreAlkane.error().message;
    const std::size_t alkaneFunctions = coreAlkane->Shape()[0];
    const Eigen::MatrixXd densityAlkane = PhysicalDensity(alkaneFunctions, 20260909);
    auto densityTensorAlkane = ToTensor(densityAlkane);

    ASSERT_TRUE(densityTensorAlkane.has_value()) << densityTensorAlkane.error().message;

    ExpectLeanMatchesMachinery(*h2oMolecule, *h2oBasis, *coreH2o, *densityTensorH2o);
    ExpectLeanMatchesMachinery(*alkaneMolecule, *alkaneBasis, *coreAlkane, *densityTensorAlkane);
}

TEST(LeanDirectFockBuilderTest, HalfModesMatchTheDirectFamily) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    // The UHF halves (buildCoulombOnly / buildExchangeOnly - the
    // driver's split-mode arms) contract the same terms as the machinery's
    // half-mode builders.
    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(7, 20260910);
    auto densityTensor = ToTensor(density);

    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    ExpectLeanMatchesMachinery(*molecule, *basis, *core, *densityTensor, true, false);
    ExpectLeanMatchesMachinery(*molecule, *basis, *core, *densityTensor, false, true);
}

TEST(LeanDirectFockBuilderTest, DampedWalkLandsOnThePinnedH2oEnergyBitIdentically) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    // The converged-leg pin: two INDEPENDENT k=1 walks (fresh builders)
    // converge to bit-identical energies, and that energy matches the
    // H2O/STO-3G kTight pin -74.96292827 (the direct-family
    // converged reference value -74.962928246436 sits 2.4e-8 off the
    // anchor; the +-1e-5 tolerance is the documented band of both).
    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;

    const auto walkEnergy = [&]() -> qcx::Result<double> {
        auto builder = qcx::integrals::LeanDirectFockBuilder::Create(
            *molecule, *basis, *core, LeanSerialOptions());

        if (!builder.has_value())
        {
            return std::unexpected(builder.error());
        }

        auto walk = WalkH2oToConvergence(*builder, *molecule, *basis, ToMatrix(*core));

        if (!walk.has_value())
        {
            return std::unexpected(walk.error());
        }

        return walk->totalEnergy;
    };

    auto first = walkEnergy();

    ASSERT_TRUE(first.has_value()) << first.error().message;
    auto second = walkEnergy();

    ASSERT_TRUE(second.has_value()) << second.error().message;
    // Byte identity is the assertion - the failure message says 'drifted in bytes'.
    // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison)
    EXPECT_TRUE(std::memcmp(&*first, &*second, sizeof(double)) == 0)
        << "two independent walks drifted in bytes";
    EXPECT_NEAR(*first, -74.96292827, 1e-5) << "the kTight H2O energy pin";
}

TEST(LeanDirectFockBuilderTest, ValidationAndLastResortCeilingRefusal) {
    // The Create-time validation and the last-resort refusal shape: refuse
    // only when the simple-actuals estimate exceeds the ceiling
    // by more than the generous slack, reporting the estimated peak, the
    // ceiling and the largest contributor. A ceiling above the estimate
    // never refuses; 0 = no ceiling, no check.
    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::LeanFockBuildOptions badBytes;
    badBytes.maxBatchBytes = 0;
    auto rejected =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, badBytes);

    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(rejected.error().message.find("maxBatchBytes must be positive"), std::string::npos);

    qcx::integrals::LeanFockBuildOptions badCap;
    badCap.memoryCapGiB = -1.0;
    auto rejectedCap =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, badCap);

    ASSERT_FALSE(rejectedCap.has_value());
    EXPECT_EQ(rejectedCap.error().code, qcx::ErrorCode::kInvalidArgument);

    // A ceiling of one millionth of a GiB (~1 KiB) is far under the
    // estimate: refuse, with the report's required/available/message.
    qcx::integrals::LeanFockBuildOptions tinyCap;
    tinyCap.memoryCapGiB = 1e-6;
    auto refused = qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, tinyCap);

    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(refused.error().message.find("required_peak_bytes"), std::string::npos);
    EXPECT_NE(refused.error().message.find("available_bytes"), std::string::npos);
    EXPECT_NE(refused.error().message.find("largest contributor"), std::string::npos);

    // A generous ceiling passes, and the default 0 (no check) passes.
    qcx::integrals::LeanFockBuildOptions generousCap;
    generousCap.memoryCapGiB = 1.0;
    auto accepted =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, generousCap);

    ASSERT_TRUE(accepted.has_value()) << accepted.error().message;
}

TEST(LeanDirectFockBuilderTest, PreRampEnvelopeMatchesTheBuildersOwnNumber) {
    // EstimateLeanEnvelope is the same admission without a builder (the
    // above-ceiling crash class): a caller that must decide BEFORE the core
    // Hamiltonian exists - the driver's setup ramp materializes the pair
    // store, so a gate that can only run after it is unreachable below its
    // own footprint - reads the envelope, not a second estimate. This is
    // the drift guard: the envelope's total must equal the builder's own
    // EstimatedPeakBytes for the same options (the row-window plan included), and its
    // refusal text must be the builder's Create-time text, byte for byte.
    // A second estimate that agreed only approximately would decide
    // differently from Create on some input, which is exactly the failure
    // the shared formula exists to prevent.
    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;

    // The default options (the auto window plan) and the pinned one-window
    // plan: the row-window plan the envelope rides must be the row-window plan a call runs.
    const qcx::integrals::LeanFockBuildOptions plans[] = {
        {}, [] {
            qcx::integrals::LeanFockBuildOptions pinned;
            pinned.maxParallelChunks = 1;
            return pinned;
        }()};

    for (const qcx::integrals::LeanFockBuildOptions& options : plans)
    {
        auto envelope = qcx::integrals::EstimateLeanEnvelope(*molecule, *basis, options);

        ASSERT_TRUE(envelope.has_value()) << envelope.error().message;
        // No ceiling: the total is computed, the verdict admits.
        EXPECT_GT(envelope->totalBytes, 0.0);
        EXPECT_TRUE(envelope->refusal.empty());

        auto builder =
            qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, options);

        ASSERT_TRUE(builder.has_value()) << builder.error().message;
        EXPECT_DOUBLE_EQ(envelope->totalBytes, builder->EstimatedPeakBytes());

        // The ceiling verdict itself: the same options with a ceiling far
        // under the estimate must give the driver the builder's own text
        // (the driver composes the refusal straight out of the envelope).
        // The ceiling narrows the plan before it refuses, so what refuses
        // here is the SMALLEST plan the builder can run - the one-window
        // plan (the k = 1 pin's own decomposition) - while the text still
        // names the host plan's requirement beside it.
        qcx::integrals::LeanFockBuildOptions tinyCap = options;
        tinyCap.memoryCapGiB = 1e-6;
        auto refusedEnvelope = qcx::integrals::EstimateLeanEnvelope(*molecule, *basis, tinyCap);

        ASSERT_TRUE(refusedEnvelope.has_value()) << refusedEnvelope.error().message;
        // The number the refusal names is the SMALLEST plan's requirement -
        // the ceiling narrowed the plan to the floor of its search before
        // refusing - so the anchor is the pinned one-window builder, the
        // decomposition the k = 1 pin names.
        qcx::integrals::LeanFockBuildOptions pinned = options;
        pinned.maxParallelChunks = 1;
        auto pinnedBuilder =
            qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, pinned);

        ASSERT_TRUE(pinnedBuilder.has_value()) << pinnedBuilder.error().message;
        EXPECT_DOUBLE_EQ(refusedEnvelope->totalBytes, pinnedBuilder->EstimatedPeakBytes());
        EXPECT_FALSE(refusedEnvelope->refusal.empty());
        EXPECT_NE(refusedEnvelope->refusal.find("host plan"), std::string::npos)
            << refusedEnvelope->refusal;
        // The host plan's number is the uncapped options' own total, which
        // is what names this leg's second requirement.
        EXPECT_NE(refusedEnvelope->refusal.find(
                      std::to_string(static_cast<long long>(envelope->totalBytes))),
                  std::string::npos)
            << refusedEnvelope->refusal;
        auto refusedBuilder =
            qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, tinyCap);

        ASSERT_FALSE(refusedBuilder.has_value());
        EXPECT_EQ(refusedBuilder.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_EQ(refusedEnvelope->refusal, refusedBuilder.error().message);

        // No ceiling (0) is no check: the text stays empty even though the
        // same envelope would refuse a tiny positive ceiling.
        qcx::integrals::LeanFockBuildOptions noCeiling = options;
        noCeiling.memoryCapGiB = 0.0;
        auto unchecked = qcx::integrals::EstimateLeanEnvelope(*molecule, *basis, noCeiling);

        ASSERT_TRUE(unchecked.has_value()) << unchecked.error().message;
        EXPECT_DOUBLE_EQ(unchecked->totalBytes, builder->EstimatedPeakBytes());
        EXPECT_TRUE(unchecked->refusal.empty());

        // A ceiling that covers the one-window plan's requirement ADMITS,
        // and the plan it selects sits between that floor and the host's own
        // plan: the ceiling buys the widest decomposition it can afford
        // instead of refusing, and the builder's own number is the selected
        // plan's - the ceiling the run was admitted under is the ceiling its
        // Create-time check consumed. The bounds are one-sided in both
        // directions so the leg holds at any team size (a host whose own
        // plan already is the one-window plan narrows nothing).
        qcx::integrals::LeanFockBuildOptions floorCap = options;
        floorCap.memoryCapGiB =
            pinnedBuilder->EstimatedPeakBytes() / static_cast<double>(1ULL << 30);
        auto admittedEnvelope = qcx::integrals::EstimateLeanEnvelope(*molecule, *basis, floorCap);

        ASSERT_TRUE(admittedEnvelope.has_value()) << admittedEnvelope.error().message;
        EXPECT_TRUE(admittedEnvelope->refusal.empty()) << admittedEnvelope->refusal;
        EXPECT_GE(admittedEnvelope->totalBytes, pinnedBuilder->EstimatedPeakBytes());
        EXPECT_LE(admittedEnvelope->totalBytes, builder->EstimatedPeakBytes());

        auto admittedBuilder =
            qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, floorCap);

        ASSERT_TRUE(admittedBuilder.has_value()) << admittedBuilder.error().message;
        EXPECT_DOUBLE_EQ(admittedEnvelope->totalBytes, admittedBuilder->EstimatedPeakBytes());
    }
}

TEST(LeanDirectFockBuilderTest, TheRetentionBandIsValueExactAndItsCostIsPairBuilds) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    // The retention band's own pin. The window store is a BOUNDED BAND
    // (options.windowStoreBytes): the flush's built pairs join it and the
    // band releases the oldest entry's contracted payload once the band
    // exceeds the cap, so a pair the band dropped is rebuilt when a later
    // flush lists it. Two properties decide whether that is a trade or a
    // break, and both are counts, never wall time (the instrument rule):
    //  1. the VALUES do not move. The release empties exactly the containers
    //     the pair builder refills from the pair's own shells and centers, so
    //     the rebuilt bits are the released bits' and the Fock matrix is
    //     identical element for element - not close, identical;
    //  2. the cost is pair BUILDS (FockBuildStats::pairBuildCount). A cap far
    //     below the store's own mass makes every flush re-list pairs the band
    //     has already dropped, so the build count rises while the peak
    //     estimate falls - the two numbers the band trades against each
    //     other.
    auto basis = MakeAlkaneSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeAlkaneSto3g(8);

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;
    const std::size_t n = core->Shape()[0];

    ASSERT_EQ(n, 58u) << "C8H18/STO-3G basis functions";
    const Eigen::MatrixXd density = PhysicalDensity(n, 20260908);
    auto densityTensor = ToTensor(density);

    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    qcx::integrals::LeanFockBuildOptions unboundedOptions = LeanSerialOptions();
    unboundedOptions.windowStoreBytes = 0; // The legacy retention: everything the window built.
    qcx::integrals::LeanFockBuildOptions bandedOptions = LeanSerialOptions();
    bandedOptions.windowStoreBytes = 4096; // Far below one flush's own pair payload.
    auto unboundedBuilder =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, unboundedOptions);

    ASSERT_TRUE(unboundedBuilder.has_value()) << unboundedBuilder.error().message;
    auto bandedBuilder =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, bandedOptions);

    ASSERT_TRUE(bandedBuilder.has_value()) << bandedBuilder.error().message;
    qcx::integrals::FockBuildStats unboundedStats;
    auto unboundedFock = unboundedBuilder->BuildFock(*densityTensor, &unboundedStats);

    ASSERT_TRUE(unboundedFock.has_value()) << unboundedFock.error().message;
    qcx::integrals::FockBuildStats bandedStats;
    auto bandedFock = bandedBuilder->BuildFock(*densityTensor, &bandedStats);

    ASSERT_TRUE(bandedFock.has_value()) << bandedFock.error().message;
    // 1. Value-exactness: the band moves no bit.
    EXPECT_DOUBLE_EQ((ToMatrix(*bandedFock) - ToMatrix(*unboundedFock)).norm(), 0.0)
        << "a released-and-rebuilt pair wrote different bits than the retained copy";
    // 2. The band's two numbers: the peak estimate falls and the pair builds
    //    rise, and the rise is the band's whole cost - so it is measured, not
    //    assumed. The unbounded build is the build-once leg the retention pin
    //    above pins (851 of the 903 canonical pairs); the banded one re-lists
    //    pairs the band dropped and builds them again.
    EXPECT_LT(bandedBuilder->EstimatedPeakBytes(), unboundedBuilder->EstimatedPeakBytes())
        << "the band's peak estimate must be below the unbounded retention's";
    EXPECT_GT(bandedStats.pairBuildCount, unboundedStats.pairBuildCount)
        << "the band built " << bandedStats.pairBuildCount << " pairs against the unbounded "
        << unboundedStats.pairBuildCount
        << ": a cap this far below the store's mass must re-list pairs the band dropped";
    std::printf("BAND bandedBuilds=%zu unboundedBuilds=%zu bandedPeak=%.0f unboundedPeak=%.0f\n",
                bandedStats.pairBuildCount,
                unboundedStats.pairBuildCount,
                bandedBuilder->EstimatedPeakBytes(),
                unboundedBuilder->EstimatedPeakBytes());
}

TEST(LeanDirectFockBuilderTest, WaveGroupingIsBitIdenticalToTheOneWaveReduction) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    // The generation-grouped reduction's pin (the accumulator-residency
    // bound): each wave's accumulators fold into the running Fock in window
    // order as the wave joins, so the fold's addition SEQUENCE is the
    // window-order one at EVERY wave width - the grouping moves only WHEN a
    // partial is added, never its position in the sum. The four legs below
    // share one window count (every canonical pair its own window - the
    // explicit experiment seam) AND one chunk plan, so the WIDTH is the only
    // difference between them: one wave (the historic residency, the width at
    // the window count), one window per wave (the finest grouping), waves of
    // four with a tail wave of three, and the width the build's own rule picks
    // (twice the region's team). That last width is this machine's value,
    // which is exactly why it belongs in the pin: the property must hold at
    // EVERY width, and a width nobody chose would not test it. Every leg must
    // land on byte-identical bytes for both the Fock at a fixed density and
    // the converged energy of the damped walk - a value that moved would be a
    // silent summation regrouping, so the comparison is memcmp, not a
    // tolerance.
    //
    // The FULLY auto plan is deliberately NOT a leg here (it was one until
    // 2026-09-19): with maxParallelChunks = 0 the build picks the WINDOW and
    // chunk counts too, so its summation order is its own - the plan moves the
    // bytes, not the width. It failed on windows-msvc Release (CI run
    // 35451596080) carrying its own estimate, 1,893,736 B against this plan's
    // 27,273,608 B: a different plan, not a different order over the same
    // terms. No bit-identity contract spans two plans, and the three explicit
    // legs passed on every leg of that run - so the mechanism was never a
    // contracting compiler, which is what a "MSVC only" failure of this cell
    // otherwise looks like.
    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd coreH = ToMatrix(*core);
    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd density = PhysicalDensity(n, 20260914);
    auto densityTensor = ToTensor(density);

    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    // The shared window count of the three explicit legs: 15 = every
    // canonical H2O/STO-3G pair its own window. Asserted below against the
    // realized per-window record, so a fixture whose pair count moved fails
    // loudly instead of quietly weakening the pin.
    constexpr std::size_t kWindows = 15;

    struct WaveLeg {
        const char* name;
        std::size_t windowWaveCount; // 0 = auto (2 x the region's team).
        std::size_t maxParallelChunks; // 0 = the auto plan.
        bool explicitWindows;
    };

    const std::array<WaveLeg, 4> legs{{
        {"one wave (the width at the window count)", kWindows, kWindows, true},
        {"one window per wave", 1, kWindows, true},
        {"waves of four (a tail wave of three)", 4, kWindows, true},
        {"the width the build picks (twice the region's team)", 0, kWindows, true},
    }};
    std::array<Eigen::MatrixXd, 4> focks;
    std::array<double, 4> energies{};
    std::array<double, 4> estimates{};

    for (std::size_t leg = 0; leg < legs.size(); ++leg)
    {
        qcx::integrals::LeanFockBuildOptions options;
        options.accuracy = qcx::integrals::AccuracyPreset::kTight;
        options.maxParallelChunks = legs[leg].maxParallelChunks;
        options.windowWaveCount = legs[leg].windowWaveCount;
        auto builder =
            qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, options);

        ASSERT_TRUE(builder.has_value()) << legs[leg].name << ": " << builder.error().message;
        estimates[leg] = builder->EstimatedPeakBytes();
        qcx::integrals::FockBuildStats stats;
        std::vector<std::chrono::nanoseconds> windowTimes;
        stats.windowWallTimesOut = &windowTimes;
        auto fock = builder->BuildFock(*densityTensor, &stats);

        ASSERT_TRUE(fock.has_value()) << legs[leg].name << ": " << fock.error().message;
        focks[leg] = ToMatrix(*fock);

        if (legs[leg].explicitWindows)
        {
            ASSERT_EQ(windowTimes.size(), kWindows)
                << legs[leg].name << ": the explicit leg must carry one window per canonical pair";
        }

        auto walk = WalkH2oToConvergence(*builder, *molecule, *basis, coreH);

        ASSERT_TRUE(walk.has_value()) << legs[leg].name << ": " << walk.error().message;
        energies[leg] = walk->totalEnergy;
    }

    for (std::size_t leg = 1; leg < legs.size(); ++leg)
    {
        EXPECT_TRUE(BitIdentical(focks[0], focks[leg]))
            << legs[leg].name << " moved the Fock bytes against the one-wave leg";
        // Byte identity is the assertion - the leg must not move the converged energy.
        // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison)
        EXPECT_TRUE(std::memcmp(&energies[0], &energies[leg], sizeof(double)) == 0)
            << legs[leg].name << " moved the converged energy against the one-wave leg";
    }

    // The charge follows the bound: at a shared plan the accumulators are the
    // only estimated term the width moves, so the estimate is strictly
    // decreasing in the width, and the move is exactly (window count - width)
    // x 8 n^2 bytes. A charge that did not move with the width would leave the
    // last-resort refusal gated on a never-under number again.
    EXPECT_LT(estimates[1], estimates[2]) << "one-window waves must charge below waves of four";
    EXPECT_LT(estimates[2], estimates[0]) << "waves of four must charge below the one-wave form";
    EXPECT_NEAR(estimates[0] - estimates[1],
                static_cast<double>(kWindows - 1) * 8.0 * static_cast<double>(n * n),
                1.0)
        << "the accumulator term must move by (window count - width) x 8n^2";
    std::printf(
        "WAVE n=%zu windows=%zu widths={%zu,%zu,%zu,auto} estimates={%.0f,%.0f,%.0f,%.0f}\n",
        n,
        kWindows,
        legs[0].windowWaveCount,
        legs[1].windowWaveCount,
        legs[2].windowWaveCount,
        estimates[0],
        estimates[1],
        estimates[2],
        estimates[3]);
}

TEST(LeanDirectFockBuilderTest, PrintsTheWaveResidencyLadder) {
    if (std::getenv("QCX_LEAN_WAVE_MEASURE") == nullptr)
    {
        GTEST_SKIP() << "set QCX_LEAN_WAVE_MEASURE=1 to run the wave-residency instrument "
                        "(the measured sizes walk hundreds of millions of cells)";
    }

    // The accumulator-residency bound's MEASURED delivery: one lean build at
    // an alkane fixture whose peak is read from the OUTSIDE (the process's
    // peak working set - PSAPI GetProcessMemoryInfo / PeakWorkingSetSize),
    // so this test prints the run's own plan and
    // envelope and the binary stays platform-neutral.
    //
    // The configuration isolates the term under test: the retention band is
    // pinned to 8 MiB so the store term cannot mask the accumulators, and the
    // walk runs at kLoose - the preset moves the screening, never the
    // accumulator a window allocates, so the residency is preset-independent
    // and the fixture can be large enough for the term to be visible. Run the
    // binary TWICE, once with the auto width and once with a width at or above
    // the window count (the one-wave form), and compare the two peaks:
    //   QCX_LEAN_WAVE_MEASURE=1 QCX_LEAN_WAVE_CARBONS=96
    //   QCX_LEAN_WAVE_MEASURE=1 QCX_LEAN_WAVE_CARBONS=96 QCX_LEAN_WAVE_WIDTH=1024
    const auto positiveOf = [](const char* text, std::size_t fallback) {
        if (text == nullptr || *text == '\0')
        {
            return fallback;
        }

        const long parsed = std::strtol(text, nullptr, 10);
        return parsed > 0 ? static_cast<std::size_t>(parsed) : fallback;
    };

    const std::size_t carbons = positiveOf(std::getenv("QCX_LEAN_WAVE_CARBONS"), 96);
    const std::size_t waveWidth = positiveOf(std::getenv("QCX_LEAN_WAVE_WIDTH"), 0);
    auto basis = MakeAlkaneSto3gBasis();

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeAlkaneSto3g(carbons);

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value()) << core.error().message;
    const std::size_t n = core->Shape()[0];
    qcx::integrals::LeanFockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    // The int product is 8 MiB: in range, the widening is the assignment's.
    // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result)
    options.windowStoreBytes = 8 * 1024 * 1024;
    options.maxParallelChunks = 0; // The driver's auto spelling.
    options.windowWaveCount = waveWidth;
    auto builder = qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, options);

    ASSERT_TRUE(builder.has_value()) << builder.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(n, 20260914);
    auto densityTensor = ToTensor(density);

    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    qcx::integrals::FockBuildStats stats;
    std::vector<std::chrono::nanoseconds> windowTimes;
    stats.windowWallTimesOut = &windowTimes;
    const auto started = std::chrono::steady_clock::now();
    auto fock = builder->BuildFock(*densityTensor, &stats);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    ASSERT_TRUE(fock.has_value()) << fock.error().message;
    const Eigen::MatrixXd matrix = ToMatrix(*fock);
    double diagonalSum = 0.0;

    for (Eigen::Index i = 0; i < matrix.rows() && i < matrix.cols(); ++i)
    {
        diagonalSum += matrix(i, i);
    }

    std::printf("WAVE_RESIDENCY carbons=%zu n=%zu windows=%zu requested_width=%zu "
                "estimate=%.0f pair_builds=%zu wall_s=%.2f diag_sum=%.17g\n",
                carbons,
                n,
                windowTimes.size(),
                waveWidth,
                builder->EstimatedPeakBytes(),
                stats.pairBuildCount,
                seconds,
                diagonalSum);
    std::fflush(stdout);
}
