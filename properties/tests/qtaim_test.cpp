// Bader QTAIM tests: the (3,-1) bond
// critical points of the electron density and their bond paths.
//
// The defining pins:
//   (a) single H (no pairs): the empty-result contract - an empty BCP list
//       is legitimate output, not a failure
//   (b) H2/STO-3G: exactly one BCP at the midpoint (0.7, 0, 0) with rho
//       matching BOTH the in-test closed form (evaluator convention
//       reimplemented) and the pyscf 2.14.0 cross-check row, ellipticity
//       exactly 0 (cylindrical symmetry: lambda1 = lambda2)
//   (c) the H2 bond path: both rays terminate at the two nuclei, the path
//       lies on the bond axis, its polyline length is exactly 2.1 bohr
//       ([BCP, ray A, BCP, ray B] - three 0.7 spans; the RK4 arc-length
//       parameterization makes the sum exact), and it runs monotonically
//       outward from the BCP
//   (d) H2O/STO-3G: exactly two (3,-1) critical points (the H-H pair is
//       below the seed cutoff - a negative pin on the cutoff physics),
//       mirror-paired across the C2 axis, matching the pyscf rows.  The
//       minimal-basis density carries a non-nuclear attractor ~0.2 bohr
//       off each H (the H nucleus is not a density maximum without the
//       cusp), so the O-H bond paths terminate at the attractors and the
//       BCPs are reported through the truncated-path bucket - the
//       "reported, never a failure" contract pin
//   (e) the density shape mismatch is kInvalidArgument

#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/properties/qtaim.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

// The H2/STO-3G BCP row from the pyscf 2.14.0 cross-check (WSL 2026-08-29,
// unit='Bohr', cart=False; probe in tools/properties/): the BCP sits at
// (0.7, 0, 0) exactly by symmetry.  The Hessian comes from central
// differences of pyscf's analytic first derivatives (this WSL build's
// libcgto ships no GTOval_ipip), so the laplacian row's last digits carry
// a ~1e-8 FD residual - the pin tolerances below reflect that.
constexpr double kH2Sto3gBcpRho = 0.255926671299;
constexpr double kH2Sto3gBcpLaplacian = -0.840815721791;
constexpr double kH2Sto3gBcpLambda1 = -0.983843774616;
constexpr double kH2Sto3gBcpLambda3 = 1.126871827440;

// The H2O/STO-3G BCP rows from the same probe (one of the mirror pair;
// the other is (-x, y, z)).  Positions 1e-6, rho/laplacian 1e-6,
// ellipticity 1e-5 - provisional tolerances fixed from the
// probe run (the pyscf side is analytic for rho and g; only the Hessian
// carries the ~1e-8 FD residual, far inside these).
constexpr double kH2oSto3gBcpX = 1.086211393867;
constexpr double kH2oSto3gBcpY = 0.846519879634;
constexpr double kH2oSto3gBcpRho = 0.379146070644;
constexpr double kH2oSto3gBcpLaplacian = -2.580065809965;
constexpr double kH2oSto3gBcpLambda1 = -1.668715222737;
constexpr double kH2oSto3gBcpLambda2 = -1.604016954821;
constexpr double kH2oSto3gBcpLambda3 = 0.692666367594;
constexpr double kH2oSto3gBcpEllipticity = 0.040335152145;

// The evaluator's l = 0 normalization (ao_evaluator.cpp RadialNormalization,
// reimplemented for the reference): N_0(zeta) = (2 zeta/pi)^(3/4).
double SOrbitalNormalization(double exponent) {
    return std::pow(2.0 * exponent / kPi, 0.75);
}

// The contracted value of one s AO at a point, in the evaluator's exact
// convention (the closed-form reference of pin (b)).
double SOrbitalValue(const qcx::basisset::Shell& shell, std::size_t contraction, double distance) {
    double phi = 0.0;

    for (std::size_t i = 0; i < shell.exponents.size(); ++i)
    {
        phi += shell.coefficients[contraction][i] * SOrbitalNormalization(shell.exponents[i]) *
               std::exp(-shell.exponents[i] * distance * distance);
    }

    return phi;
}

bool DistanceEquals(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) < 1e-9;
}

using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

// One hydrogen at the origin (the fixtures only offer H2/He).
qcx::Result<qcx::molecule::Molecule> MakeSingleHydrogen() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({1, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}}, std::move(*coordinates), 0, 1);
}

// H = T + V from the one-electron engines (the dense_rhf_test.cpp pattern,
// shared with the sibling properties tests).
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildCoreHamiltonian(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet) {
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

// The dense general-l RHF path - the qtaim tests run the real converged
// density, then search it for critical points.
qcx::Result<qcx::scf::HfResult> RunDenseRhf(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet) {
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

    auto eri = qcx::integrals::BuildEriTensorGeneral(molecule, basisSet);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    return qcx::scf::RunRhfScf(molecule, ToMatrix(*overlap), ToMatrix(*core), *eri);
}

TEST(QtaimTest, SingleHydrogenHasNoCriticalPoints) {
    // Pin (a): no atom pairs -> no seeds -> the empty-result contract.
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    Eigen::MatrixXd density(1, 1);
    density(0, 0) = 2.0;

    auto result = qcx::properties::AnalyzeQtaim(*molecule, *basis, density);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->bondCriticalPoints.empty());
    EXPECT_TRUE(result->otherCriticalPoints.empty());
    EXPECT_TRUE(result->unconvergedSeeds.empty());
}

TEST(QtaimTest, H2Sto3gBondCriticalPoint) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF: Release-only";
    }

    // Pin (b): the converged H2/STO-3G RHF density has exactly one BCP at
    // (0.7, 0, 0) - the midpoint seed converges immediately, since g = 0
    // there by cylindrical symmetry (D_00 = D_11 in the RHF density).
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto result = qcx::properties::AnalyzeQtaim(*molecule, *basis, scf->density);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_EQ(result->bondCriticalPoints.size(), 1u);
    EXPECT_TRUE(result->otherCriticalPoints.empty());
    EXPECT_TRUE(result->unconvergedSeeds.empty());

    const auto& bcp = result->bondCriticalPoints[0];
    EXPECT_NEAR(bcp.positionBohr[0], 0.7, 1e-9);
    EXPECT_LT(std::abs(bcp.positionBohr[1]), 1e-12);
    EXPECT_LT(std::abs(bcp.positionBohr[2]), 1e-12);

    // rho at the BCP: the in-test closed form (the evaluator convention
    // reimplemented, tol 1e-10) and the baked pyscf row (tol 1e-8).
    const auto& shells = basis->Find(1)->shells;
    ASSERT_EQ(shells.size(), 1u);
    ASSERT_EQ(shells[0].coefficients.size(), 1u);

    const std::array<double, 3> centers = {0.0, 1.4};
    std::array<double, 2> phi{};

    for (std::size_t mu = 0; mu < 2; ++mu)
    {
        phi[mu] = SOrbitalValue(shells[0], 0, std::abs(0.7 - centers[mu]));
    }

    double rhoClosedForm = 0.0;

    for (std::size_t mu = 0; mu < 2; ++mu)
    {
        for (std::size_t nu = 0; nu < 2; ++nu)
        {
            rhoClosedForm +=
                scf->density(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) *
                phi[mu] * phi[nu];
        }
    }

    EXPECT_NEAR(bcp.density, rhoClosedForm, 1e-10);
    EXPECT_NEAR(bcp.density, kH2Sto3gBcpRho, 1e-8);
    EXPECT_NEAR(bcp.laplacian, kH2Sto3gBcpLaplacian, 1e-7);
    EXPECT_NEAR(bcp.eigenvalues[0], kH2Sto3gBcpLambda1, 1e-7);
    EXPECT_NEAR(bcp.eigenvalues[1], kH2Sto3gBcpLambda1, 1e-7);
    EXPECT_NEAR(bcp.eigenvalues[2], kH2Sto3gBcpLambda3, 1e-7);
    EXPECT_NEAR(bcp.ellipticity, 0.0, 1e-8);
    EXPECT_EQ(bcp.atomA, 0u);
    EXPECT_EQ(bcp.atomB, 1u);
}

TEST(QtaimTest, H2Sto3gBondPathTerminatesAtNuclei) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF: Release-only";
    }

    // Pin (c): the bond path of the H2 BCP.  The two rays start epsilon
    // off the BCP along the bond axis and follow +/-g/|g|; both terminate
    // at the two nuclei, the path lies on the bond axis (|y|, |z| < 1e-9),
    // and it runs monotonically outward from the BCP to each nucleus.
    // The path is the concatenation [BCP, ray A, BCP, ray B] (the BCP
    // starts each ray by construction), so its polyline length is
    // exactly 2.1 bohr = 3 x 0.7: the ray A span, the nucleus -> BCP
    // return segment, and the ray B span.  The RK4 arc-length
    // parameterization (unit-speed steps, length-conserving snap) makes
    // the sum exact to the axis noise - far inside the step-size
    // bound.
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;

    auto result = qcx::properties::AnalyzeQtaim(*molecule, *basis, scf->density);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_EQ(result->bondCriticalPoints.size(), 1u);

    const auto& bcp = result->bondCriticalPoints[0];
    const auto& path = bcp.bondPath;
    ASSERT_GE(path.size(), 4u);
    EXPECT_EQ(bcp.atomA, 0u);
    EXPECT_EQ(bcp.atomB, 1u);

    // The path starts at the BCP, which appears again between the two
    // rays (the BCP starts each ray).
    const auto& coordinates = molecule->CoordinatesBohr();
    const std::array<double, 3> nucleusA = {
        coordinates(0, 0), coordinates(0, 1), coordinates(0, 2)};
    const std::array<double, 3> nucleusB = {
        coordinates(1, 0), coordinates(1, 1), coordinates(1, 2)};

    EXPECT_NEAR(path[0][0], bcp.positionBohr[0], 1e-12);
    EXPECT_NEAR(path[0][1], 0.0, 1e-12);
    EXPECT_NEAR(path[0][2], 0.0, 1e-12);

    std::size_t middle = 1;

    while (middle < path.size() &&
           (std::abs(path[middle][0] - bcp.positionBohr[0]) > 1e-9 ||
            std::abs(path[middle][1]) > 1e-9 || std::abs(path[middle][2]) > 1e-9))
    {
        ++middle;
    }

    ASSERT_LT(middle, path.size());
    ASSERT_GE(middle, 2u);

    // The endpoints of the two rays are the two nuclei, in molecule order.
    const std::array<double, 3>& rayAEnd = path[middle - 1];
    const std::array<double, 3>& rayBEnd = path.back();
    const std::array<double, 3>* firstNucleus = nullptr;

    if (DistanceEquals(rayAEnd, nucleusA) && DistanceEquals(rayBEnd, nucleusB))
    {
        firstNucleus = &nucleusA;
    } else if (DistanceEquals(rayAEnd, nucleusB) && DistanceEquals(rayBEnd, nucleusA))
    { firstNucleus = &nucleusB; }

    ASSERT_NE(firstNucleus, nullptr);

    double pathLength = 0.0;

    for (std::size_t i = 0; i < path.size() - 1; ++i)
    {
        const double dx = path[i + 1][0] - path[i][0];
        const double dy = path[i + 1][1] - path[i][1];
        const double dz = path[i + 1][2] - path[i][2];
        pathLength += std::sqrt(dx * dx + dy * dy + dz * dz);

        EXPECT_LT(std::abs(path[i][1]), 1e-9);
        EXPECT_LT(std::abs(path[i][2]), 1e-9);
    }

    EXPECT_NEAR(pathLength, 2.1, 1e-8);

    // Monotonically outward from the BCP to each nucleus: each ray is
    // monotone in x, and the two rays run in opposite directions (the
    // eigensolver's eigenvector sign is arbitrary, so ray A may start
    // toward either nucleus).
    const bool rayAPlusX = path[1][0] > path[0][0];

    for (std::size_t i = 0; i < middle - 1; ++i)
    {
        if (rayAPlusX)
        {
            EXPECT_GT(path[i + 1][0], path[i][0]);
        } else
        {
            EXPECT_LT(path[i + 1][0], path[i][0]);
        }
    }

    for (std::size_t i = middle; i < path.size() - 1; ++i)
    {
        if (rayAPlusX)
        {
            EXPECT_LT(path[i + 1][0], path[i][0]);
        } else
        {
            EXPECT_GT(path[i + 1][0], path[i][0]);
        }
    }
}

TEST(QtaimTest, H2oSto3gTwoBondCriticalPoints) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF: Release-only";
    }

    // Pin (d): the converged H2O/STO-3G RHF density has EXACTLY two (3,-1)
    // critical points, the two O-H bonds.  The H-H pair (2.86 bohr) is
    // below the seed cutoff 1.5 * (r_cov + r_cov) = 1.76 bohr - a negative
    // pin on the cutoff physics: a third critical point would be spurious.
    //
    // The minimal-basis density carries a non-nuclear attractor (3,-3)
    // ~0.2 bohr off each H, toward O: rho is 0.3859 at the attractor but
    // only 0.3628 at the H nucleus, whose gradient flows into the attractor
    // (the Gaussian basis has no cusp, so the H nucleus is not a density
    // maximum).  The O-H bond path's H-side ray terminates at the
    // attractor, never at the nucleus, so both BCPs come back through the
    // truncated-path reporting - the "reported, never a failure"
    // contract.  pyscf's field reproduces the identical walk (the probe
    // replica stalls at the same point), so the pin is the density, not
    // the integrator; the BCP field rows below are contracted from the
    // density directly at the reported positions.
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto result = qcx::properties::AnalyzeQtaim(*molecule, *basis, scf->density);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->bondCriticalPoints.empty());
    EXPECT_TRUE(result->unconvergedSeeds.empty());
    ASSERT_EQ(result->otherCriticalPoints.size(), 2u);

    // The two critical points are the (3,-1) O-H bond critical points with
    // truncated bond paths: rank 3, signature -1, mirror-paired across the
    // C2 (y) axis, in the molecular plane.
    const auto& first = result->otherCriticalPoints[0];
    const auto& second = result->otherCriticalPoints[1];
    EXPECT_EQ(first.rank, 3);
    EXPECT_EQ(first.signatureSum, -1);
    EXPECT_EQ(second.rank, 3);
    EXPECT_EQ(second.signatureSum, -1);
    EXPECT_NEAR(second.positionBohr[0], -first.positionBohr[0], 1e-9);
    EXPECT_NEAR(second.positionBohr[1], first.positionBohr[1], 1e-9);
    EXPECT_NEAR(second.positionBohr[2], first.positionBohr[2], 1e-9);
    EXPECT_LT(std::abs(first.positionBohr[2]), 1e-9);
    EXPECT_LT(std::abs(second.positionBohr[2]), 1e-9);

    // The baked pyscf rows (positions 1e-6, fixed tolerances).
    // The seed loop decides which mirror sits in the first slot, so the
    // x-row comparison is sign-agnostic (the mirror-pair assertions above
    // already fix the relation between the two).
    EXPECT_NEAR(std::abs(first.positionBohr[0]), kH2oSto3gBcpX, 1e-6);
    EXPECT_NEAR(first.positionBohr[1], kH2oSto3gBcpY, 1e-6);

    // The field rows, contracted from the density at the reported position
    // (the reported structs carry only rank and signature).
    auto evaluator = qcx::grid::AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value()) << evaluator.error().message;
    const std::size_t n = evaluator->AOCount();
    std::vector<double> phi(n);
    std::vector<double> grad(3 * n);
    std::vector<double> hess(6 * n);

    const auto evaluate = [&](const std::array<double, 3>& point) {
        evaluator->EvaluateDerivatives(point, phi, grad, hess);
        double rho = 0.0;
        std::array<double, 6> packed{};

        for (std::size_t mu = 0; mu < n; ++mu)
        {
            for (std::size_t nu = 0; nu < n; ++nu)
            {
                const double d =
                    scf->density(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu));
                rho += d * phi[mu] * phi[nu];

                for (int j = 0; j < 3; ++j)
                {
                    for (int k = j; k < 3; ++k)
                    {
                        // The packed (xx, xy, xz, yy, yz, zz) index of (j, k).
                        const int pack = j == 0 ? k : (j == 1 ? k + 2 : 5);
                        const std::size_t muPack = 6 * mu + static_cast<std::size_t>(pack);
                        const std::size_t nuPack = 6 * nu + static_cast<std::size_t>(pack);
                        const std::size_t muJ = 3 * mu + static_cast<std::size_t>(j);
                        const std::size_t muK = 3 * mu + static_cast<std::size_t>(k);
                        const std::size_t nuJ = 3 * nu + static_cast<std::size_t>(j);
                        const std::size_t nuK = 3 * nu + static_cast<std::size_t>(k);
                        packed[static_cast<std::size_t>(pack)] +=
                            d * (phi[nu] * hess[muPack] + grad[muJ] * grad[nuK] +
                                 grad[muK] * grad[nuJ] + phi[mu] * hess[nuPack]);
                    }
                }
            }
        }

        return std::pair{rho, packed};
    };

    const auto [firstRho, firstHessian] = evaluate(first.positionBohr);
    const auto [secondRho, secondHessian] = evaluate(second.positionBohr);
    const Eigen::Matrix3d firstMatrix = (Eigen::Matrix3d() << firstHessian[0],
                                         firstHessian[1],
                                         firstHessian[2],
                                         firstHessian[1],
                                         firstHessian[3],
                                         firstHessian[4],
                                         firstHessian[2],
                                         firstHessian[4],
                                         firstHessian[5])
                                            .finished();
    const Eigen::Vector3d firstLambda = firstMatrix.selfadjointView<Eigen::Lower>().eigenvalues();

    // Equal rho, laplacian, ellipticity across the mirror pair.
    EXPECT_NEAR(secondRho, firstRho, 1e-9);

    // The baked pyscf rows (the fixed tolerances; the contracted
    // values at the converged positions carry the Newton residual).
    EXPECT_NEAR(firstRho, kH2oSto3gBcpRho, 1e-6);
    EXPECT_NEAR(firstLambda[0] + firstLambda[1] + firstLambda[2], kH2oSto3gBcpLaplacian, 1e-6);
    EXPECT_NEAR(firstLambda[0], kH2oSto3gBcpLambda1, 1e-6);
    EXPECT_NEAR(firstLambda[1], kH2oSto3gBcpLambda2, 1e-6);
    EXPECT_NEAR(firstLambda[2], kH2oSto3gBcpLambda3, 1e-6);
    EXPECT_NEAR(firstLambda[0] / firstLambda[1] - 1.0, kH2oSto3gBcpEllipticity, 1e-5);
}

TEST(QtaimTest, RejectsDensityShapeMismatch) {
    // Pin (e): a 2x2 density on the single-H basis is not n x n.
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    Eigen::MatrixXd density(2, 2);
    density.setZero();

    auto result = qcx::properties::AnalyzeQtaim(*molecule, *basis, density);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}

} // namespace
