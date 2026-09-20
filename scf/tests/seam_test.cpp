// FockBuilderFn seam tests: the
// callback overload of RunRhfScf must reproduce the supermatrix path exactly,
// reject an empty callback, run a degenerate constant Fock to a finite result
// within the iteration budget, and propagate a builder failure as the run's
// error (the direct/RI builders of qcx-integrals report through the same
// path). The supermatrix builders below are independent reimplementations of
// the J/K contraction of rhf.cpp and uhf.cpp - for the fixtures used here (H2
// and H3 in STO-3G, n = 2 and 3) the plain Eigen products sit below the
// kSmallBlockThreshold, so they are bit-identical to the linalg seam path.
//
// The energy-seam tests (the second half of the same
// callback family) pin the formula the seam exists for:
//     E = Tr[D H] + 1/2 Tr[D J[D]] + energyContribution(D),
// the two-spin form being Tr[D_total H] + 1/2 Tr[D_total J[D_total]] plus a
// contribution of BOTH spin densities. Each of them recomputes the expected
// number here from the density the run returned, so a seam that drops the
// contribution, scales it, or forms it at the wrong density fails by value
// rather than by convergence drift.
#include "h2_sto3g.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/uhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace {

using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

// The J and K supermatrices of one (uv|ws) tensor, in the flat row-major order
// the Fock builds contract against (rhf.cpp's BuildJkSupermatrices layout):
// J = E d and K = E_x d for the flat density d.
struct JkSuper {
    Eigen::MatrixXd coulomb;
    Eigen::MatrixXd exchange;
};

JkSuper BuildJkSuper(const CpuTensor4& eri, std::size_t n) {
    const Eigen::Index eigenN2 = static_cast<Eigen::Index>(n * n);
    JkSuper super{Eigen::MatrixXd::Zero(eigenN2, eigenN2), Eigen::MatrixXd::Zero(eigenN2, eigenN2)};

    for (std::size_t mu = 0; mu < n; ++mu)
    {
        for (std::size_t nu = 0; nu < n; ++nu)
        {
            const Eigen::Index row = static_cast<Eigen::Index>(mu * n + nu);

            for (std::size_t l = 0; l < n; ++l)
            {
                for (std::size_t s = 0; s < n; ++s)
                {
                    const Eigen::Index col = static_cast<Eigen::Index>(l * n + s);
                    super.coulomb(row, col) = eri(mu, nu, l, s);
                    super.exchange(row, col) = eri(mu, l, s, nu);
                }
            }
        }
    }

    return super;
}

// The flat row-major vector of a density matrix (rhf.cpp's dVec order).
Eigen::MatrixXd FlattenRowMajor(const Eigen::MatrixXd& matrix) {
    const Eigen::Index n = matrix.rows();
    Eigen::MatrixXd flat(n * n, 1);

    for (Eigen::Index mu = 0; mu < n; ++mu)
    {
        for (Eigen::Index nu = 0; nu < n; ++nu)
        {
            flat(mu * n + nu, 0) = matrix(mu, nu);
        }
    }

    return flat;
}

// The inverse of FlattenRowMajor, applied to a J/K product vector.
Eigen::MatrixXd UnflattenRowMajor(const Eigen::MatrixXd& flat, Eigen::Index n) {
    Eigen::MatrixXd matrix(n, n);

    for (Eigen::Index mu = 0; mu < n; ++mu)
    {
        for (Eigen::Index nu = 0; nu < n; ++nu)
        {
            matrix(mu, nu) = flat(mu * n + nu, 0);
        }
    }

    return matrix;
}

// J[D] = E d - the Coulomb half of the supermatrix Fock build.
Eigen::MatrixXd CoulombOf(const JkSuper& super, const Eigen::MatrixXd& density) {
    return UnflattenRowMajor(super.coulomb * FlattenRowMajor(density), density.rows());
}

// K[D] = E_x d - the exchange half.
Eigen::MatrixXd ExchangeOf(const JkSuper& super, const Eigen::MatrixXd& density) {
    return UnflattenRowMajor(super.exchange * FlattenRowMajor(density), density.rows());
}

// The supermatrix Fock build of rhf.cpp as a FockBuilderFn: the J and K
// supermatrices are density-independent and built once, then each call
// contracts the current density into F = H + J - K/2 (the per-element
// accumulation order of rhf.cpp's BuildFock, kept so the two agree bit for
// bit below the small-block threshold).
qcx::scf::FockBuilderFn MakeSupermatrixFockBuilder(const JkSuper& super,
                                                   const Eigen::MatrixXd& coreHamiltonian) {
    return
        [super, coreHamiltonian](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
            const Eigen::MatrixXd j = CoulombOf(super, density);
            const Eigen::MatrixXd k = ExchangeOf(super, density);
            Eigen::MatrixXd fock = coreHamiltonian;

            for (Eigen::Index mu = 0; mu < fock.rows(); ++mu)
            {
                for (Eigen::Index nu = 0; nu < fock.cols(); ++nu)
                {
                    fock(mu, nu) += j(mu, nu) - 0.5 * k(mu, nu);
                }
            }

            return fock;
        };
}

// The same builder from the raw tensor (the four tests below build their own
// supermatrices through this overload).
qcx::scf::FockBuilderFn MakeSupermatrixFockBuilder(const CpuTensor4& eri,
                                                   const Eigen::MatrixXd& coreHamiltonian) {
    return MakeSupermatrixFockBuilder(
        BuildJkSuper(eri, static_cast<std::size_t>(coreHamiltonian.rows())), coreHamiltonian);
}

// The UHF supermatrix Fock build of uhf.cpp as a UhfFockBuilderFn:
// F_sigma = H + J(D_alpha + D_beta) - K(D_sigma), no scaling prefactor - the
// per-spin densities are the REAL ones, unlike RHF's spin-summed D.
qcx::scf::UhfFockBuilderFn MakeUhfSupermatrixFockBuilder(const JkSuper& super,
                                                         const Eigen::MatrixXd& coreHamiltonian) {
    return [super, coreHamiltonian](const Eigen::MatrixXd& dAlpha, const Eigen::MatrixXd& dBeta)
               -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        const Eigen::MatrixXd j = CoulombOf(super, dAlpha + dBeta);
        const Eigen::MatrixXd kAlpha = ExchangeOf(super, dAlpha);
        const Eigen::MatrixXd kBeta = ExchangeOf(super, dBeta);
        Eigen::MatrixXd fockAlpha = coreHamiltonian;
        Eigen::MatrixXd fockBeta = coreHamiltonian;

        for (Eigen::Index mu = 0; mu < fockAlpha.rows(); ++mu)
        {
            for (Eigen::Index nu = 0; nu < fockAlpha.cols(); ++nu)
            {
                fockAlpha(mu, nu) += j(mu, nu) - kAlpha(mu, nu);
                fockBeta(mu, nu) += j(mu, nu) - kBeta(mu, nu);
            }
        }

        return std::make_pair(std::move(fockAlpha), std::move(fockBeta));
    };
}

// J[D] as the energy seam's Coulomb provider. ONE instance serves the RHF and
// the UHF overload: the callback's density is the spin-summed object in both
// (RHF's D = 2 C_occ C_occ^T, UHF's D_alpha + D_beta), which is what J is
// built from - the reason uhf.hpp reuses rhf.hpp's CoulombFn rather than
// declaring a twin.
qcx::scf::CoulombFn MakeCoulombProvider(const JkSuper& super) {
    return [super](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        return CoulombOf(super, density);
    };
}

// The trace half of the two-spin seam, at one FIXED pair of densities:
// Tr[D_total H] + 1/2 Tr[D_total J[D_total]].
double TraceOnlyAt(const JkSuper& super,
                   // (coreHamiltonian, dAlpha, dBeta) is the core-then-alpha-then-beta order of the
                   // core, alpha, beta order
                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                   const Eigen::MatrixXd& coreHamiltonian,
                   const Eigen::MatrixXd& dAlpha,
                   const Eigen::MatrixXd& dBeta) {
    const Eigen::MatrixXd totalDensity = dAlpha + dBeta;

    return (totalDensity.cwiseProduct(coreHamiltonian)).sum() +
           0.5 * (totalDensity.cwiseProduct(CoulombOf(super, totalDensity))).sum();
}

// The exchange energy at one fixed pair of densities:
// -1/2 sum_sigma Tr[D_sigma K[D_sigma]] - the whole contribution at fraction 1,
// because for Hartree-Fock the exchange IS the energy outside the trace.
double ExchangeEnergyAt(const JkSuper& super,
                        const Eigen::MatrixXd& dAlpha,
                        const Eigen::MatrixXd& dBeta) {
    return -0.5 * ((dAlpha.cwiseProduct(ExchangeOf(super, dAlpha))).sum() +
                   (dBeta.cwiseProduct(ExchangeOf(super, dBeta))).sum());
}

// The Hartree-Fock trace identity at one fixed pair of densities, with the Fock
// rebuilt FROM THAT DENSITY: 1/2 sum_sigma Tr[D_sigma (H + F(D)_sigma)].
// Deliberately density-consistent - see the regression test below for why the
// reported energy of an SCF run is not this number.
double TraceIdentityAt(const JkSuper& super,
                       const Eigen::MatrixXd& coreHamiltonian,
                       const Eigen::MatrixXd& dAlpha,
                       const Eigen::MatrixXd& dBeta) {
    const Eigen::MatrixXd coulomb = CoulombOf(super, dAlpha + dBeta);
    const Eigen::MatrixXd fockAlpha = coreHamiltonian + coulomb - ExchangeOf(super, dAlpha);
    const Eigen::MatrixXd fockBeta = coreHamiltonian + coulomb - ExchangeOf(super, dBeta);

    return 0.5 * ((dAlpha.cwiseProduct(coreHamiltonian + fockAlpha)).sum() +
                  (dBeta.cwiseProduct(coreHamiltonian + fockBeta)).sum());
}

// The exchange piece of the energy at a hybrid's exact-exchange fraction:
// fraction * (-1/4) Tr[D K[D]] in the closed-shell convention. This is the term
// the loop cannot form - K is a bilinear form the caller computes, not a trace
// of the Fock matrix the loop holds - so it is exactly what
// EnergyContributionFn exists to carry, and fraction = 1 makes the total the
// Hartree-Fock energy.
qcx::scf::EnergyContributionFn MakeExchangeContribution(const JkSuper& super, double fraction) {
    return [super, fraction](const Eigen::MatrixXd& density) -> qcx::Result<double> {
        return fraction * (-0.25) * (density.cwiseProduct(ExchangeOf(super, density))).sum();
    };
}

// The two-spin twin: fraction * (-1/2) sum_sigma Tr[D_sigma K[D_sigma]].
qcx::scf::UhfEnergyContributionFn MakeUhfExchangeContribution(const JkSuper& super,
                                                              double fraction) {
    return [super, fraction](const Eigen::MatrixXd& dAlpha,
                             const Eigen::MatrixXd& dBeta) -> qcx::Result<double> {
        return fraction * ExchangeEnergyAt(super, dAlpha, dBeta);
    };
}

// H3 in STO-3G: three hydrogens on the z axis at an ASYMMETRIC spacing, charge
// 0, multiplicity 2 (nAlpha = 2, nBeta = 1), so the two spin channels carry
// genuinely different densities. Asymmetric on purpose - a symmetric linear H3
// has a multi-solution landscape of its own, and this fixture exists to be free
// of one.
qcx::Result<qcx::molecule::Molecule> MakeH3Sto3gDoublet() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 0.0;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 1.6;
    (*coordinates)(2, 0) = 0.0;
    (*coordinates)(2, 1) = 0.0;
    (*coordinates)(2, 2) = 3.4;
    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        2);
}

// The H3/STO-3G doublet fixture the UHF energy-seam tests share: the molecule,
// the one-electron matrices, the dense ERI tensor and the J/K supermatrices,
// built once per test. Returned as a Result so a fixture failure is reported by
// the calling test rather than swallowed here (ASSERT_* cannot be used in a
// function that returns a value).
//
// Deliberately SMALL. At n = 3 the J/K contraction is 9 x 9, below the
// kSmallBlockThreshold, so the builders below are bit-identical to uhf.cpp's own
// and the HF-shaped regression measures the seam's ALGEBRA rather than a
// rounding difference between two contraction paths.
//
// NOT O2/STO-3G, the repo's other open-shell fixture, which this test tried
// first: the default P=0 start lands that system on the documented saddle
// (-147.37855918 - uhf_test.cpp's O2TripletConvergesToPinnedEnergy note), whose
// trajectory is chaotic. Measured here: the two paths reached the same saddle
// but 3e-7 apart, in 20 and 37 iterations, so an agreement pin on it would be
// pinning the trajectory's chaos, not the seam. The published O2 pin needs the
// SAD unpolarized start, and belongs to the start-tier tests in uhf_test.cpp -
// which run at the production defaults since the 2026-09-15 gate conversion
// (the pin's cluster-scale band covers the default gate's landing).
struct H3Inputs {
    qcx::molecule::Molecule molecule;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd coreHamiltonian;
    CpuTensor4 eri;
    JkSuper super;
};

qcx::Result<H3Inputs> BuildH3Inputs() {
    const auto basis = MakeSto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto molecule = MakeH3Sto3gDoublet();

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    const std::size_t n = static_cast<std::size_t>(overlap->Shape()[0]);

    // The supermatrices are built from the tensor BEFORE it is moved into the
    // fixture: a braced-init-list evaluates left to right, so reading *eri in
    // the same list after the move would read a moved-from tensor.
    const JkSuper super = BuildJkSuper(*eri, n);
    CpuTensor4 tensor = std::move(*eri);

    return H3Inputs{std::move(*molecule),
                    ToMatrix(*overlap),
                    ToMatrix(*kinetic) + ToMatrix(*nuclear),
                    std::move(tensor),
                    super};
}

TEST(ScfSeamTest, FockBuilderCallbackReproducesSupermatrixPath) {
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

    const Eigen::MatrixXd overlapMatrix = ToMatrix(*overlap);
    const Eigen::MatrixXd coreHamiltonian = ToMatrix(*kinetic) + ToMatrix(*nuclear);

    const auto supermatrixResult =
        qcx::scf::RunRhfScf(*molecule, overlapMatrix, coreHamiltonian, *eri);
    ASSERT_TRUE(supermatrixResult.has_value()) << supermatrixResult.error().message;

    const qcx::scf::RhfOptions options;
    const auto seamResult = qcx::scf::RunRhfScf(*molecule,
                                                overlapMatrix,
                                                coreHamiltonian,
                                                options,
                                                MakeSupermatrixFockBuilder(*eri, coreHamiltonian));
    ASSERT_TRUE(seamResult.has_value()) << seamResult.error().message;

    // The same loop with bit-identical Fock matrices must land on the pinned
    // energy of the 5-arg path (rhf_test.cpp) and match that run exactly.
    EXPECT_NEAR(seamResult->totalEnergy, -1.1167143252, 1e-8);
    EXPECT_NEAR(seamResult->totalEnergy, supermatrixResult->totalEnergy, 1e-12);
    EXPECT_NEAR(seamResult->electronicEnergy, supermatrixResult->electronicEnergy, 1e-12);
    EXPECT_EQ(seamResult->converged, supermatrixResult->converged);
    EXPECT_EQ(seamResult->iterations, supermatrixResult->iterations);
}

TEST(ScfSeamTest, EmptyFockBuilderIsRejected) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

    const qcx::scf::FockBuilderFn emptyBuilder;
    const auto result = qcx::scf::RunRhfScf(*molecule,
                                            ToMatrix(*overlap),
                                            ToMatrix(*kinetic) + ToMatrix(*nuclear),
                                            qcx::scf::RhfOptions{},
                                            emptyBuilder);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(ScfSeamTest, FockBuilderErrorsPropagate) {
    // The loop must not swallow the builder's failure: a FockBuilderFn that
    // returns an error on the first call (the density starts at zero, so the
    // failure lands on iteration 0) stops the run with exactly that error.
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

    int calls = 0;
    const auto result = qcx::scf::RunRhfScf(
        *molecule,
        ToMatrix(*overlap),
        ToMatrix(*kinetic) + ToMatrix(*nuclear),
        qcx::scf::RhfOptions{},
        [&calls](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
            ++calls;
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError, "probe failure"});
        });
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInternalError);
    EXPECT_EQ(calls, 1);
}

TEST(ScfSeamTest, ConstantFockRunsToFiniteResult) {
    // A density-independent Fock matrix is degenerate but well-formed input:
    // the loop must run through diagonalization, the DIIS window, and the
    // convergence checks to a finite result within the iteration budget.
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

    const qcx::scf::RhfOptions options;
    const auto result =
        qcx::scf::RunRhfScf(*molecule,
                            ToMatrix(*overlap),
                            ToMatrix(*kinetic) + ToMatrix(*nuclear),
                            options,
                            [](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
                                return Eigen::MatrixXd::Identity(density.rows(), density.cols());
                            });
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(std::isfinite(result->totalEnergy));
    EXPECT_TRUE(std::isfinite(result->electronicEnergy));
    EXPECT_LE(result->iterations, options.maxIterations);
}

TEST(ScfSeamTest, ConstantContributionEntersTheTraceOnce) {
    // The seam's shape, with no functional in sight: a contribution that
    // ignores the density must land in the total exactly once, unscaled, on top
    // of the Coulomb trace Tr[D H] + 1/2 Tr[D J[D]].
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

    const Eigen::MatrixXd overlapMatrix = ToMatrix(*overlap);
    const Eigen::MatrixXd coreHamiltonian = ToMatrix(*kinetic) + ToMatrix(*nuclear);
    const JkSuper super = BuildJkSuper(*eri, 2);
    const qcx::scf::FockBuilderFn fockBuilder = MakeSupermatrixFockBuilder(super, coreHamiltonian);
    const qcx::scf::CoulombFn coulomb = MakeCoulombProvider(super);
    const qcx::scf::RhfOptions options;
    const double constant = 0.125;

    const auto contributed = qcx::scf::RunRhfScf(
        *molecule,
        overlapMatrix,
        coreHamiltonian,
        options,
        fockBuilder,
        coulomb,
        [constant](const Eigen::MatrixXd&) -> qcx::Result<double> { return constant; });
    ASSERT_TRUE(contributed.has_value()) << contributed.error().message;
    ASSERT_TRUE(contributed->converged);

    // The expected number, recomputed here from the density the run returned:
    // the formula above, with the run's own converged D. Not EXPECT_DOUBLE_EQ -
    // the two expressions reach the same value through different summations
    // (the loop forms each trace as its own scalar product and then adds the
    // contribution; this recomputation does the same, but the totals are also
    // rounded through the nuclear-repulsion addition, so "exactly" is not a
    // claim the arithmetic supports).
    const Eigen::MatrixXd& density = contributed->density;
    const double traceOnly = (density.cwiseProduct(coreHamiltonian)).sum() +
                             0.5 * (density.cwiseProduct(CoulombOf(super, density))).sum();
    EXPECT_NEAR(contributed->electronicEnergy, traceOnly + constant, 1e-12);

    // And the contribution does not disturb the trajectory: a constant shift of
    // the total energy cannot move the Fock matrix, so the zero-contribution
    // run stops at the same iterate and differs by exactly the constant.
    const auto plain =
        qcx::scf::RunRhfScf(*molecule,
                            overlapMatrix,
                            coreHamiltonian,
                            options,
                            fockBuilder,
                            coulomb,
                            [](const Eigen::MatrixXd&) -> qcx::Result<double> { return 0.0; });
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    EXPECT_EQ(contributed->iterations, plain->iterations);
    EXPECT_NEAR(contributed->electronicEnergy, plain->electronicEnergy + constant, 1e-12);
    EXPECT_NEAR(contributed->totalEnergy, plain->totalEnergy + constant, 1e-12);
}

TEST(ScfSeamTest, HybridShapedContributionMatchesTheHandValue) {
    // The test that fails if someone later "simplifies" the contribution away:
    // a hybrid-shaped term (a fraction of the exact exchange, the bilinear form
    // the loop cannot recover from D and an opaque Fock builder) must reach the
    // total at its own value and with its own density.
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

    const Eigen::MatrixXd coreHamiltonian = ToMatrix(*kinetic) + ToMatrix(*nuclear);
    const JkSuper super = BuildJkSuper(*eri, 2);
    const double exchangeFraction = 0.2;

    const auto result = qcx::scf::RunRhfScf(*molecule,
                                            ToMatrix(*overlap),
                                            coreHamiltonian,
                                            qcx::scf::RhfOptions{},
                                            MakeSupermatrixFockBuilder(super, coreHamiltonian),
                                            MakeCoulombProvider(super),
                                            MakeExchangeContribution(super, exchangeFraction));
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->converged);

    // The hand value, from the returned density: Tr[D H] + 1/2 Tr[D J]
    // + 0.2 * (-1/4) Tr[D K].
    const Eigen::MatrixXd& density = result->density;
    const double traceOnly = (density.cwiseProduct(coreHamiltonian)).sum() +
                             0.5 * (density.cwiseProduct(CoulombOf(super, density))).sum();
    const double exchangePiece =
        exchangeFraction * (-0.25) * (density.cwiseProduct(ExchangeOf(super, density))).sum();
    EXPECT_GT(std::fabs(exchangePiece), 1e-3);
    EXPECT_NEAR(result->electronicEnergy, traceOnly + exchangePiece, 1e-12);
}

TEST(ScfSeamTest, HfShapedSeamReproducesTheSupermatrixPath) {
    // The regression that makes the seam's algebra falsifiable: with the
    // Coulomb provider and an HF-shaped contribution (-1/4 Tr[D K[D]], i.e. the
    // exact exchange at fraction 1) the three-callback overload must reproduce
    // the existing supermatrix run.
    //
    // The match is ALGEBRAIC, not bitwise: Tr[D H] + 1/2 Tr[D J] - 1/4 Tr[D K]
    // and 1/2 Tr[D (H + F)] sum the same terms in different orders, so the two
    // runs differ at the last bits and the pin is a tight RELATIVE tolerance
    // rather than an equality. The baseline is the post-#22 tree (the RHF
    // energy leg live, zero re-pins): against the pre-#22 tree this pin would
    // have frozen the inert gate's behaviour and then read as a regression.
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

    const Eigen::MatrixXd overlapMatrix = ToMatrix(*overlap);
    const Eigen::MatrixXd coreHamiltonian = ToMatrix(*kinetic) + ToMatrix(*nuclear);
    const JkSuper super = BuildJkSuper(*eri, 2);

    const auto supermatrixResult =
        qcx::scf::RunRhfScf(*molecule, overlapMatrix, coreHamiltonian, *eri);
    ASSERT_TRUE(supermatrixResult.has_value()) << supermatrixResult.error().message;

    const auto seamResult = qcx::scf::RunRhfScf(*molecule,
                                                overlapMatrix,
                                                coreHamiltonian,
                                                qcx::scf::RhfOptions{},
                                                MakeSupermatrixFockBuilder(super, coreHamiltonian),
                                                MakeCoulombProvider(super),
                                                MakeExchangeContribution(super, 1.0));
    ASSERT_TRUE(seamResult.has_value()) << seamResult.error().message;

    const double totalRelative =
        std::fabs(seamResult->totalEnergy - supermatrixResult->totalEnergy) /
        std::fabs(supermatrixResult->totalEnergy);
    const double electronicRelative =
        std::fabs(seamResult->electronicEnergy - supermatrixResult->electronicEnergy) /
        std::fabs(supermatrixResult->electronicEnergy);
    EXPECT_LT(totalRelative, 1e-12);
    EXPECT_LT(electronicRelative, 1e-12);
    EXPECT_EQ(seamResult->converged, supermatrixResult->converged);
    EXPECT_EQ(seamResult->iterations, supermatrixResult->iterations);

    // The same closed-shell pin the neighbouring test uses: the seam must land
    // on the published number, not merely near the other path.
    EXPECT_NEAR(seamResult->totalEnergy, -1.1167143252, 1e-8);
}

TEST(ScfSeamTest, UhfConstantContributionEntersTheTraceOnce) {
    // The two-spin twin of the constant-contribution test: the formula is
    // Tr[D_total H] + 1/2 Tr[D_total J[D_total]] + contribution, and the
    // contribution is what a UKS run's Exc(dAlpha, dBeta) rides in on.
    const auto inputs = BuildH3Inputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const qcx::scf::UhfOptions options;
    const double constant = 0.125;
    const auto result = qcx::scf::RunUhfScf(
        inputs->molecule,
        inputs->overlap,
        inputs->coreHamiltonian,
        options,
        MakeUhfSupermatrixFockBuilder(inputs->super, inputs->coreHamiltonian),
        MakeCoulombProvider(inputs->super),
        [constant](const Eigen::MatrixXd&, const Eigen::MatrixXd&) -> qcx::Result<double> {
            return constant;
        });
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->converged);

    const double traceOnly = TraceOnlyAt(
        inputs->super, inputs->coreHamiltonian, result->densityAlpha, result->densityBeta);
    EXPECT_NEAR(result->electronicEnergy, traceOnly + constant, 1e-12);
}

TEST(ScfSeamTest, UhfHybridShapedContributionMatchesTheHandValue) {
    // The two-spin exchange form the twin's signature implies: the contribution
    // takes BOTH spin densities and carries -1/2 sum_sigma Tr[D_sigma K[D_sigma]]
    // at fraction 1. A seam that passed only the total density to the
    // contribution could not form this term at all, so this test is what pins
    // the two-argument signature rather than the one-argument convenience.
    const auto inputs = BuildH3Inputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const double exchangeFraction = 0.2;
    const auto result =
        qcx::scf::RunUhfScf(inputs->molecule,
                            inputs->overlap,
                            inputs->coreHamiltonian,
                            qcx::scf::UhfOptions{},
                            MakeUhfSupermatrixFockBuilder(inputs->super, inputs->coreHamiltonian),
                            MakeCoulombProvider(inputs->super),
                            MakeUhfExchangeContribution(inputs->super, exchangeFraction));
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->converged);

    const Eigen::MatrixXd& dAlpha = result->densityAlpha;
    const Eigen::MatrixXd& dBeta = result->densityBeta;
    const double traceOnly = TraceOnlyAt(inputs->super, inputs->coreHamiltonian, dAlpha, dBeta);
    const double exchangePiece = exchangeFraction * ExchangeEnergyAt(inputs->super, dAlpha, dBeta);
    EXPECT_GT(std::fabs(exchangePiece), 1e-3);

    // Both spin channels must be polarized, or the two-argument signature this
    // test exists to pin would be exercised on a closed-shell-shaped density.
    EXPECT_GT((dAlpha - dBeta).cwiseAbs().maxCoeff(), 1e-3);
    EXPECT_NEAR(result->electronicEnergy, traceOnly + exchangePiece, 1e-12);
}

TEST(ScfSeamTest, UhfHfShapedSeamReportsTheHartreeFockEnergyOfItsDensity) {
    // The two-spin regression, and the one that pins the twin's algebra: at
    // fraction 1 the contribution is -1/2 sum_sigma Tr[D_sigma K[D_sigma]] and
    // the seam's total must be the Hartree-Fock energy. Copying RHF's expression
    // without the derivation - or halving the H term along with it, which is the
    // plausible slip, since the UHF identity IS written with a leading 1/2 -
    // fails here by value.
    //
    // Asserted as an IDENTITY AT ONE DENSITY rather than as a comparison of two
    // SCF runs' reported energies, and that choice is measured, not stylistic.
    // The two paths do NOT agree along the trajectory: the existing path reports
    // 1/2 sum_sigma Tr[D_new (H + F)] with the DIIS-EXTRAPOLATED Fock, not the
    // Fock of the density it returns. Measured on this fixture, the supermatrix
    // run's reported electronic energy is -3.02263 while the density-consistent
    // 1/2 sum_sigma Tr[D (H + F(D))] at its OWN returned density is -3.02261.
    // That ~2e-5 gap is the SCF's own residual; it moves the gate's energy leg,
    // so the two runs legitimately stop at different iterates (27 against 33
    // here), and comparing their totals would pin the stopping point rather than
    // the seam. At a FIXED density the two expressions are the same number:
    // 2.9e-16 relative on this fixture.
    //
    // Baseline: the post-#22 tree, as the RHF twin says.
    const auto inputs = BuildH3Inputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const auto supermatrixResult = qcx::scf::RunUhfScf(
        inputs->molecule, inputs->overlap, inputs->coreHamiltonian, inputs->eri);
    ASSERT_TRUE(supermatrixResult.has_value()) << supermatrixResult.error().message;
    ASSERT_TRUE(supermatrixResult->converged);

    const auto seamResult =
        qcx::scf::RunUhfScf(inputs->molecule,
                            inputs->overlap,
                            inputs->coreHamiltonian,
                            qcx::scf::UhfOptions{},
                            MakeUhfSupermatrixFockBuilder(inputs->super, inputs->coreHamiltonian),
                            MakeCoulombProvider(inputs->super),
                            MakeUhfExchangeContribution(inputs->super, 1.0));
    ASSERT_TRUE(seamResult.has_value()) << seamResult.error().message;
    ASSERT_TRUE(seamResult->converged);

    // (1) The seam's reported energy is its own formula at the density it
    // RETURNED - the check that fails if the energy is formed at the wrong
    // iterate (the previous density, or the one the Fock was built from).
    const double seamAtItsOwnDensity =
        TraceOnlyAt(inputs->super,
                    inputs->coreHamiltonian,
                    seamResult->densityAlpha,
                    seamResult->densityBeta) +
        ExchangeEnergyAt(inputs->super, seamResult->densityAlpha, seamResult->densityBeta);
    EXPECT_NEAR(seamResult->electronicEnergy, seamAtItsOwnDensity, 1e-12);

    // (2) And that formula IS the Hartree-Fock energy: at one density, with the
    // Fock rebuilt from that same density, the two expressions agree.
    const double traceIdentity = TraceIdentityAt(inputs->super,
                                                 inputs->coreHamiltonian,
                                                 supermatrixResult->densityAlpha,
                                                 supermatrixResult->densityBeta);
    const double seamIdentity = TraceOnlyAt(inputs->super,
                                            inputs->coreHamiltonian,
                                            supermatrixResult->densityAlpha,
                                            supermatrixResult->densityBeta) +
                                ExchangeEnergyAt(inputs->super,
                                                 supermatrixResult->densityAlpha,
                                                 supermatrixResult->densityBeta);
    EXPECT_NEAR(traceIdentity, seamIdentity, 1e-12);

    // (3) Both paths also reach the same chemistry, coarsely: see the header
    // comment for why the stopping points differ. An algebra error in the seam
    // shifts this by >= 1e-3 relative, so the bound still catches one.
    const double relative = std::fabs(seamResult->totalEnergy - supermatrixResult->totalEnergy) /
                            std::fabs(supermatrixResult->totalEnergy);
    EXPECT_LT(relative, 1e-4);

    // Both spin channels must be polarized, or the twin's whole subject would be
    // exercised on a closed-shell-shaped density.
    EXPECT_GT((seamResult->densityAlpha - seamResult->densityBeta).cwiseAbs().maxCoeff(), 1e-3);
}

} // namespace
