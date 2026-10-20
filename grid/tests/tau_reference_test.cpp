// The tau acceptance gates for PR3, each an exact identity or limit that a
// named defect violates:
//   1. the uniform gas, where tau_x's enhancement is exactly 1 at tau_unif
//      and 13/12 at tau = 0 (the live defect a missing tau term produces);
//   2. the assembly's tau against the brute-force 1/2 sum_mu nu D grad.grad,
//      on a one-orbital density where 8 rho tau - sigma = 0 pointwise;
//   3. uniform coordinate scaling, exc(l^3 rho, l^5 tau, l^8 sigma) = l^4 exc,
//      with the vtau and vsigma scalings;
//   4. the von Weizsacker derivative identity, vtau = -8 rho vsigma, exact for
//      tau_x;
//   5. the tau potential as the finite difference of the engine's energy, with
//      the missing-half defect shown by the size of the tau contribution;
//   6. the tau screening sweep (Q-251's open risk): screened against dense tau,
//      grad tau, energy, potential and gradient on a tight-core/diffuse
//      fixture, with the density-weight selection reproduced from the public
//      rule;
//   7. end to end lives in driver/tests/ks_tau_test.cpp.
//
// Conventions under test are the design note's (pr3-tau-design.md): the tau is
// libxc/PySCF per spin, tau_s = 1/2 sum_mu nu D_s,munu grad phi_mu . grad phi_nu,
// Bohr^-5, and tau_x's Pauli variable is tau_P = tau_s - sigma_ss/(8 rho_s).
//
// Dependencies at build time: gates 1-5 and the energy/potential half of gate 6
// need B1's engine support (present in the tree when this file was written);
// the gradient half of gate 6 needs B2. Before B2 the gradient walk reads
// tau = 0 and its integrated energy disagrees with the energy path's - the
// assertion that says so is marked in the gate-6 gradient case.

#include "h2_sto3g.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/grid/shell_screening.hpp"
#include "qcx/grid/xc_gradient.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/memory/tensor.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <excgrid/geometry.hpp>
#include <excgrid/grid.hpp>
#include <excgrid/grid_derivatives.hpp>
#include <excgrid/kernel.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qcx::grid {
namespace {

// ---------------------------------------------------------------------------
// Kernel-level values and helpers
// ---------------------------------------------------------------------------

// (3/4)(6/pi)^(1/3), the spin-scaled Slater constant tau_x uses, per spin
// channel: the LSDA exchange energy density is -kSlaterSpinPrefactor rho^(4/3).
constexpr double kSlaterSpinPrefactor = 0.9305257363491;

// (3/10)(6 pi^2)^(2/3), the spin-scaled Thomas-Fermi constant, so the uniform
// electron gas's kinetic-energy density is kFermiConstant rho^(5/3) per spin.
constexpr double kFermiConstant = 4.5577998723455959;

// tau_unif(rho) = (3/10)(6 pi^2)^(2/3) rho^(5/3), per spin channel.
double UniformTau(double rho) noexcept {
    return kFermiConstant * std::pow(rho, 5.0 / 3.0);
}

// One point's named first-derivative tier, as tau_x publishes it.
struct PointValues {
    double exc = 0.0;
    double vrhoA = 0.0;
    double vrhoB = 0.0;
    double vsigmaAa = 0.0;
    double vsigmaAb = 0.0;
    double vsigmaBb = 0.0;
    double vtauA = 0.0;
    double vtauB = 0.0;
};

// One named-component evaluation. The positional adapter is deliberately not
// used: it cannot carry tau and reports a refusal as zeros.
std::optional<PointValues> EvaluateAt(const excgrid::XcFunctional& functional,
                                      double rhoA,
                                      double rhoB,
                                      double sigmaAa,
                                      double sigmaAb,
                                      double sigmaBb,
                                      double tauA,
                                      double tauB) {
    excgrid::PointInputs inputs;
    inputs.Set(excgrid::Component::RhoA, rhoA);
    inputs.Set(excgrid::Component::RhoB, rhoB);
    inputs.Set(excgrid::Component::SigmaAa, sigmaAa);
    inputs.Set(excgrid::Component::SigmaAb, sigmaAb);
    inputs.Set(excgrid::Component::SigmaBb, sigmaBb);
    inputs.Set(excgrid::Component::TauA, tauA);
    inputs.Set(excgrid::Component::TauB, tauB);

    excgrid::PointResult result;

    if (functional.EvaluatePoint(inputs, result) != excgrid::KernelStatus::kOk)
    {
        return std::nullopt;
    }

    const auto first = [&result](excgrid::Component id) {
        return result.first[excgrid::IndexOf(id)];
    };

    return PointValues{result.exc,
                       first(excgrid::Component::RhoA),
                       first(excgrid::Component::RhoB),
                       first(excgrid::Component::SigmaAa),
                       first(excgrid::Component::SigmaAb),
                       first(excgrid::Component::SigmaBb),
                       first(excgrid::Component::TauA),
                       first(excgrid::Component::TauB)};
}

// |actual - expected| / |expected|, with a floor so a zero expectation reports
// its absolute deviation rather than an infinity.
double RelativeDeviation(double actual, double expected) noexcept {
    return std::abs(actual - expected) / std::max(std::abs(expected), 1e-300);
}

// A number as the test records it.
std::string Scientific(double value) {
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%.6e", value);

    return std::string(buffer.data());
}

// The fixture's coarse grid: the engine and the reference walk share it, so a
// gate that walks every point many times stays cheap.
XcGridSettings GateSettings() {
    XcGridSettings settings;
    settings.radialPoints = 20;
    settings.angularPoints = 74;
    settings.blockTarget = 256;

    return settings;
}

// ---------------------------------------------------------------------------
// The brute-force fields the assemblies are held to
// ---------------------------------------------------------------------------

// One spin's density, density gradient, sigma and tau at a point, written as
// the double sums themselves rather than through the contraction that factors
// them:
//     rho      = sum_mu nu D_mu nu phi_mu phi_nu
//     grad rho = sum_mu nu D_mu nu (grad phi_mu phi_nu + phi_mu grad phi_nu)
//     tau      = 1/2 sum_mu nu D_mu nu grad phi_mu . grad phi_nu
struct BruteForcePoint {
    double rho = 0.0;
    std::array<double, 3> gradient{};
    double sigma = 0.0;
    double tau = 0.0;
};

BruteForcePoint BruteForce(const Eigen::MatrixXd& density,
                           std::span<const double> values,
                           std::span<const double> gradients) {
    const std::size_t aoCount = static_cast<std::size_t>(density.rows());
    BruteForcePoint result;

    for (std::size_t mu = 0; mu < aoCount; ++mu)
    {
        for (std::size_t nu = 0; nu < aoCount; ++nu)
        {
            const double element =
                density(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu));

            result.rho += element * values[mu] * values[nu];

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                result.gradient[axis] += element * (gradients[3 * mu + axis] * values[nu] +
                                                    values[mu] * gradients[3 * nu + axis]);
                result.tau += 0.5 * element * gradients[3 * mu + axis] * gradients[3 * nu + axis];
            }
        }
    }

    result.sigma = result.gradient[0] * result.gradient[0] +
                   result.gradient[1] * result.gradient[1] +
                   result.gradient[2] * result.gradient[2];

    return result;
}

// ---------------------------------------------------------------------------
// Gate 1: the uniform gas
// ---------------------------------------------------------------------------

TEST(TauReferenceTest, UniformGasGivesUnitEnhancementAndTheZeroTauWitness) {
    const excgrid::XcFunctional* tauX = excgrid::FindFunctional("tau_x");
    ASSERT_NE(tauX, nullptr);
    EXPECT_TRUE(tauX->RequiredMask().Test(excgrid::Component::TauA));
    EXPECT_TRUE(tauX->RequiredMask().Test(excgrid::Component::TauB));

    // rho = 1 keeps the kernel's rho + 1e-16 regulariser below the tolerance's
    // reach. Both spins carry the density, so the energy density is twice the
    // per-channel value with the enhancement as its factor.
    constexpr double rho = 1.0;
    const double tauUniform = UniformTau(rho);
    const double scale = -2.0 * kSlaterSpinPrefactor * std::pow(rho, 4.0 / 3.0);

    const auto uniform = EvaluateAt(*tauX, rho, rho, 0.0, 0.0, 0.0, tauUniform, tauUniform);
    ASSERT_TRUE(uniform.has_value());
    EXPECT_NEAR(uniform->exc, scale, 1e-12 * std::abs(scale))
        << "the enhancement at tau_unif is not exactly 1";

    const auto zeroTau = EvaluateAt(*tauX, rho, rho, 0.0, 0.0, 0.0, 0.0, 0.0);
    ASSERT_TRUE(zeroTau.has_value());
    EXPECT_NEAR(zeroTau->exc, (13.0 / 12.0) * scale, 1e-12 * std::abs(scale))
        << "the enhancement at tau = 0 is not 13/12";

    // The spin channels carry their own tau: a kernel that read one channel's
    // tau in both would still pass the symmetric case above.
    const auto alphaOnly = EvaluateAt(*tauX, rho, 0.0, 0.0, 0.0, 0.0, tauUniform, 0.0);
    ASSERT_TRUE(alphaOnly.has_value());
    EXPECT_NEAR(alphaOnly->exc, 0.5 * scale, 1e-12 * std::abs(scale))
        << "the alpha channel did not receive its own tau";

    RecordProperty("enhancement at tau_unif", Scientific(uniform->exc / scale));
    RecordProperty("enhancement at tau = 0", Scientific(zeroTau->exc / scale));

    // The deliberate broken case, stated as a measurable gap rather than an
    // implementation: a tau functional whose tau never arrives lands on the
    // tau = 0 value, a 1/12 relative defect the first assertion rejects by
    // hundreds of times its tolerance.
    EXPECT_GT(std::abs(zeroTau->exc - uniform->exc), 1e-2 * std::abs(scale))
        << "the zero-tau witness does not separate from the uniform gas";
}

// ---------------------------------------------------------------------------
// Gate 2: the tau mapping on a one-orbital density
// ---------------------------------------------------------------------------

// One hydrogen at the origin: a single AO, so the double sums below are the
// whole contraction and the identity has no other orbitals to hide behind.
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

TEST(TauReferenceTest, TheAssembledTauIsTheBruteForceContractionAndTheIsoOrbitalIdentityHolds) {
    const excgrid::XcFunctional* tauX = excgrid::FindFunctional("tau_x");
    ASSERT_NE(tauX, nullptr);

    auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto engine = XcGridEngine::Create(*molecule, *basis, "tau_x", GateSettings());
    ASSERT_TRUE(engine.has_value()) << engine.error().message;
    ASSERT_EQ(engine->AOCount(), 1U);

    Eigen::MatrixXd alpha(1, 1);
    alpha << 1.0;
    const Eigen::MatrixXd beta = Eigen::MatrixXd::Zero(1, 1);
    const auto assembled = engine->Evaluate(alpha, beta);
    ASSERT_TRUE(assembled.has_value()) << assembled.error().message;

    const AoEvaluator& evaluator = engine->Evaluator();
    std::vector<double> values(1);
    std::vector<double> gradients(3);

    double reference = 0.0;
    double doubled = 0.0;
    double halved = 0.0;
    double slater = 0.0;
    double worstIdentity = 0.0;
    double worstDoubled = 0.0;
    double worstHalved = 0.0;
    std::size_t checkedPoints = 0;

    for (const excgrid::Block& block : engine->Grid().Blocks())
    {
        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            const double quadratureWeight = block.weights[point];

            if (quadratureWeight == 0.0)
            {
                continue;
            }

            ++checkedPoints;
            evaluator.EvaluateGradients(block.points[point], values, gradients);

            const BruteForcePoint fields = BruteForce(alpha, values, gradients);

            // 8 rho tau - sigma = 0 pointwise on a one-orbital density: the
            // Pauli kinetic-energy density tau_P vanishes identically. The
            // magnitude is the pair's own scale, so the check is relative and
            // survives the far tail where both sides are tiny.
            const double eightRhoTau = 8.0 * fields.rho * fields.tau;
            const double magnitude = std::abs(eightRhoTau) + std::abs(fields.sigma);

            if (magnitude > 1e-200)
            {
                worstIdentity =
                    std::max(worstIdentity, std::abs(eightRhoTau - fields.sigma) / magnitude);
                worstDoubled = std::max(
                    worstDoubled,
                    std::abs(8.0 * fields.rho * (2.0 * fields.tau) - fields.sigma) / magnitude);
                worstHalved = std::max(
                    worstHalved,
                    std::abs(8.0 * fields.rho * (0.5 * fields.tau) - fields.sigma) / magnitude);
            }

            const auto kernel =
                EvaluateAt(*tauX, fields.rho, 0.0, fields.sigma, 0.0, 0.0, fields.tau, 0.0);
            ASSERT_TRUE(kernel.has_value());

            reference += quadratureWeight * kernel->exc;
            slater += quadratureWeight * (-kSlaterSpinPrefactor) * std::pow(fields.rho, 4.0 / 3.0);

            const auto withDoubled =
                EvaluateAt(*tauX, fields.rho, 0.0, fields.sigma, 0.0, 0.0, 2.0 * fields.tau, 0.0);
            ASSERT_TRUE(withDoubled.has_value());
            doubled += quadratureWeight * withDoubled->exc;

            const auto withHalved =
                EvaluateAt(*tauX, fields.rho, 0.0, fields.sigma, 0.0, 0.0, 0.5 * fields.tau, 0.0);
            ASSERT_TRUE(withHalved.has_value());
            halved += quadratureWeight * withHalved->exc;
        }
    }

    ASSERT_GT(checkedPoints, 100U) << "the grid has too few live points to be a fixture";

    // The convention check: the brute-force tau carries the identity.
    EXPECT_LT(worstIdentity, 1e-12) << "8 rho tau - sigma is not zero pointwise";
    // The deliberate broken cases. Doubling tau gives +sigma (0.5 of the
    // pair's magnitude), halving gives -sigma/2 (0.25); either is visible by
    // twelve orders against the criterion above.
    EXPECT_GT(worstDoubled, 0.4) << "a doubled tau does not violate the identity measurably";
    EXPECT_GT(worstHalved, 0.2) << "a halved tau does not violate the identity measurably";

    // The assembly's tau is the brute-force one: the engine's energy is the
    // reference's, built from the double sums above.
    EXPECT_NEAR(assembled->energy, reference, 1e-12)
        << "the engine's tau is not the brute-force contraction";

    // tau_x's exact condition at tau_P = 0 is 13/12 times the Slater energy,
    // not the Slater energy: the shipped reference is the bare second-order
    // expansion e = -(13/12) k_S rho^(4/3) + (k_S/(12 k_F)) rho^(-1/3) tau_P
    // (external/excgrid/src/meta_gga_reference.cpp), a shape reference that
    // fixes the uniform gas at enhancement 1 and deliberately does not carry
    // the iso-orbital cancellation a production meta-GGA would. The identity
    // gate above is the pointwise one; this gate says the kernel received the
    // brute-force tau and sigma well enough that tau_P cancels at every point.
    // The 1e-7 floor is floating point, not physics: tau and sigma are
    // independent double sums whose 8 rho tau - sigma residual is only zero to
    // rounding, and the kernel evaluates tau_P against rho + 1e-16. Both are
    // smooth small perturbations of an integrated ratio; a tau or sigma that
    // did not arrive moves this ratio by order 1e-2.
    EXPECT_LT(RelativeDeviation(reference, (13.0 / 12.0) * slater), 1e-7)
        << "tau_x at tau_P = 0 is not its own 13/12 Slater value";

    RecordProperty("engine energy (Hartree)", Scientific(assembled->energy));
    RecordProperty("brute-force energy (Hartree)", Scientific(reference));
    RecordProperty("energy with tau doubled (Hartree)", Scientific(doubled));
    RecordProperty("energy with tau halved (Hartree)", Scientific(halved));
    RecordProperty("worst 8 rho tau - sigma residual (relative)", Scientific(worstIdentity));

    // The broken cases reach the energy, so the gate's witness is visible in
    // the number that gets reported and not only in the identity.
    EXPECT_GT(std::abs(doubled - reference), 1e-4);
    EXPECT_GT(std::abs(assembled->energy - doubled), 0.5 * std::abs(doubled - reference));
    EXPECT_GT(std::abs(assembled->energy - halved), 0.5 * std::abs(halved - reference));
}

// ---------------------------------------------------------------------------
// Gate 3: uniform coordinate scaling
// ---------------------------------------------------------------------------

TEST(TauReferenceTest, UniformScalingHoldsForEnergyAndTheTauAndSigmaDerivatives) {
    const excgrid::XcFunctional* tauX = excgrid::FindFunctional("tau_x");
    ASSERT_NE(tauX, nullptr);

    // Under psi_l(r) = l^(3/2) psi(l r): rho -> l^3 rho, sigma -> l^8 sigma,
    // tau -> l^5 tau, and the energy density -> l^4 exc. The derivatives
    // follow: vtau -> l^-1 vtau (an energy per tau) and vsigma -> l^-4 vsigma.
    struct Case {
        double rhoA;
        double rhoB;
        double sigmaAa;
        double sigmaAb;
        double sigmaBb;
        double tauA;
        double tauB;
    };

    const std::array<Case, 3> cases = {
        Case{0.7, 0.3, 0.21, 0.05, 0.08, 1.1 * UniformTau(0.7), 0.9 * UniformTau(0.3)},
        Case{0.2, 0.6, 0.04, -0.01, 0.15, 0.8 * UniformTau(0.2), 1.3 * UniformTau(0.6)},
        Case{1.3, 0.9, 0.4, 0.0, 0.12, 1.4 * UniformTau(1.3), 1.0 * UniformTau(0.9)}};

    const std::array<double, 3> scales = {0.5, 1.7, 2.0};

    for (const Case& point : cases)
    {
        const auto base = EvaluateAt(*tauX,
                                     point.rhoA,
                                     point.rhoB,
                                     point.sigmaAa,
                                     point.sigmaAb,
                                     point.sigmaBb,
                                     point.tauA,
                                     point.tauB);
        ASSERT_TRUE(base.has_value());

        for (const double lambda : scales)
        {
            const double rhoScale = lambda * lambda * lambda;
            const double tauScale = std::pow(lambda, 5.0);
            const double sigmaScale = std::pow(lambda, 8.0);
            const double energyScale = std::pow(lambda, 4.0);

            const auto scaled = EvaluateAt(*tauX,
                                           rhoScale * point.rhoA,
                                           rhoScale * point.rhoB,
                                           sigmaScale * point.sigmaAa,
                                           sigmaScale * point.sigmaAb,
                                           sigmaScale * point.sigmaBb,
                                           tauScale * point.tauA,
                                           tauScale * point.tauB);
            ASSERT_TRUE(scaled.has_value());

            EXPECT_LT(RelativeDeviation(scaled->exc, energyScale * base->exc), 1e-10)
                << "exc did not scale as l^4 at lambda " << lambda;
            EXPECT_LT(RelativeDeviation(scaled->vtauA, base->vtauA / lambda), 1e-10)
                << "vtauA did not scale as l^-1 at lambda " << lambda;
            EXPECT_LT(RelativeDeviation(scaled->vtauB, base->vtauB / lambda), 1e-10)
                << "vtauB did not scale as l^-1 at lambda " << lambda;
            EXPECT_LT(RelativeDeviation(scaled->vsigmaAa, base->vsigmaAa / std::pow(lambda, 4.0)),
                      1e-10)
                << "vsigmaAa did not scale as l^-4 at lambda " << lambda;
            EXPECT_LT(RelativeDeviation(scaled->vsigmaBb, base->vsigmaBb / std::pow(lambda, 4.0)),
                      1e-10)
                << "vsigmaBb did not scale as l^-4 at lambda " << lambda;

            // The deliberate broken case: a functional with tau's power wrong
            // by one lands here and misses the scaled energy by the tau term's
            // share, which is percent-level, not roundoff.
            const auto wrongPower = EvaluateAt(*tauX,
                                               rhoScale * point.rhoA,
                                               rhoScale * point.rhoB,
                                               sigmaScale * point.sigmaAa,
                                               sigmaScale * point.sigmaAb,
                                               sigmaScale * point.sigmaBb,
                                               std::pow(lambda, 4.0) * point.tauA,
                                               std::pow(lambda, 4.0) * point.tauB);
            ASSERT_TRUE(wrongPower.has_value());
            EXPECT_GT(RelativeDeviation(wrongPower->exc, energyScale * base->exc), 1e-3)
                << "a tau exponent of 4 is not distinguishable at lambda " << lambda;
        }

        RecordProperty("base energy density (Hartree/Bohr^3)", Scientific(base->exc));
    }
}

// ---------------------------------------------------------------------------
// Gate 4: the von Weizsacker derivative identity
// ---------------------------------------------------------------------------

TEST(TauReferenceTest, TheVonWeizsackerIdentityHoldsForTauX) {
    const excgrid::XcFunctional* tauX = excgrid::FindFunctional("tau_x");
    ASSERT_NE(tauX, nullptr);

    // tau_x's enhancement is linear in tau_P = tau - sigma/(8 rho), so per
    // channel vtau = -8 rho vsigma exactly - the von Weizsacker coefficient of
    // the gradient in tau_P. It is sigma-independent and exact at every point.
    struct Case {
        double rhoA;
        double rhoB;
        double sigmaAa;
        double sigmaAb;
        double sigmaBb;
        double tauA;
        double tauB;
    };

    const std::array<Case, 4> cases = {
        Case{0.9, 0.4, 0.0, 0.0, 0.0, 1.0 * UniformTau(0.9), 1.0 * UniformTau(0.4)},
        Case{0.5, 0.2, 0.3, 0.02, 0.05, 1.2 * UniformTau(0.5), 0.7 * UniformTau(0.2)},
        Case{1.4, 1.1, 0.9, -0.2, 0.6, 0.6 * UniformTau(1.4), 1.5 * UniformTau(1.1)},
        Case{0.05, 0.03, 0.001, 0.0, 0.0005, 1.1 * UniformTau(0.05), 0.9 * UniformTau(0.03)}};

    for (const Case& point : cases)
    {
        const auto values = EvaluateAt(*tauX,
                                       point.rhoA,
                                       point.rhoB,
                                       point.sigmaAa,
                                       point.sigmaAb,
                                       point.sigmaBb,
                                       point.tauA,
                                       point.tauB);
        ASSERT_TRUE(values.has_value());

        const double expectedA = -8.0 * point.rhoA * values->vsigmaAa;
        const double expectedB = -8.0 * point.rhoB * values->vsigmaBb;

        EXPECT_LT(RelativeDeviation(values->vtauA, expectedA), 1e-12)
            << "vtauA != -8 rhoA vsigmaAa";
        EXPECT_LT(RelativeDeviation(values->vtauB, expectedB), 1e-12)
            << "vtauB != -8 rhoB vsigmaBb";
        // tau_x reads no sigma_ab; the named slot must stay exactly zero.
        EXPECT_DOUBLE_EQ(values->vsigmaAb, 0.0);

        // The deliberate broken case: a stub returning vtau = 0 (the adapter's
        // silent-zero failure) misses the identity by the whole of vtau.
        PointValues stub = *values;
        stub.vtauA = 0.0;
        stub.vtauB = 0.0;
        EXPECT_GT(RelativeDeviation(stub.vtauA, expectedA), 0.5)
            << "the identity cannot see a zero vtau";
        EXPECT_GT(RelativeDeviation(stub.vtauB, expectedB), 0.5)
            << "the identity cannot see a zero vtau";

        // vtauA does not depend on the other spin's tau: a swapped channel
        // would pass the symmetric cases and fail this one.
        const auto otherTau = EvaluateAt(*tauX,
                                         point.rhoA,
                                         point.rhoB,
                                         point.sigmaAa,
                                         point.sigmaAb,
                                         point.sigmaBb,
                                         point.tauA,
                                         2.0 * point.tauB + 0.1);
        ASSERT_TRUE(otherTau.has_value());
        EXPECT_DOUBLE_EQ(otherTau->vtauA, values->vtauA);

        RecordProperty("vtauA (Hartree Bohr^5)", Scientific(values->vtauA));
        RecordProperty("vsigmaAa (Hartree Bohr^5)", Scientific(values->vsigmaAa));
    }
}

// ---------------------------------------------------------------------------
// Gate 5: the potential is the energy's derivative
// ---------------------------------------------------------------------------

constexpr double kFiniteDifferenceStep = 1e-4;

// The engine assembly's measured finite-difference floor on this fixture:
// ~1e-10 on potentials of order 1e-1 (grid/tests/xc_assembly_test.cpp), so the
// tolerance carries twentyfold headroom over the roundoff of the difference.
constexpr double kFiniteDifferenceTolerance = 2e-9;

Eigen::MatrixXd H2ClosedShellDensity() {
    Eigen::MatrixXd density(2, 2);
    density << 1.0, 1.0, 1.0, 1.0;

    return density;
}

Eigen::MatrixXd SymmetricPerturbation() {
    Eigen::MatrixXd perturbation(2, 2);
    perturbation << 0.1, -0.05, -0.05, 0.1;

    return perturbation;
}

double TraceProduct(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right) {
    return (left.array() * right.array()).sum();
}

TEST(TauReferenceTest, TheTauPotentialIsTheEnergyDerivativeAndAMissingHalfIsVisible) {
    const excgrid::XcFunctional* tauX = excgrid::FindFunctional("tau_x");
    ASSERT_NE(tauX, nullptr);

    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto engine = XcGridEngine::Create(*molecule, *basis, "tau_x");
    ASSERT_TRUE(engine.has_value()) << engine.error().message;

    const Eigen::MatrixXd density = H2ClosedShellDensity();
    const Eigen::MatrixXd perturbation = SymmetricPerturbation();

    const auto center = engine->EvaluateClosedShell(density);
    const auto plus = engine->EvaluateClosedShell(density + kFiniteDifferenceStep * perturbation);
    const auto minus = engine->EvaluateClosedShell(density - kFiniteDifferenceStep * perturbation);
    ASSERT_TRUE(center.has_value());
    ASSERT_TRUE(plus.has_value());
    ASSERT_TRUE(minus.has_value());

    const double numeric = (plus->energy - minus->energy) / (2.0 * kFiniteDifferenceStep);
    const double analytic =
        0.5 * TraceProduct(perturbation, center->potentialAlpha + center->potentialBeta);

    EXPECT_NEAR(numeric, analytic, kFiniteDifferenceTolerance)
        << "finite difference " << numeric << " vs potential " << analytic;

    // The tau half of the analytic value, reconstructed from the public grid
    // and kernel: V_s += w * 1/2 * vtau_s * grad phi_mu . grad phi_nu, so its
    // share of Tr(P V_s) is the point sum below. A potential that carried
    // w * vtau (the missing 1/2) would double exactly this term.
    const AoEvaluator& evaluator = engine->Evaluator();
    const std::size_t aoCount = engine->AOCount();
    std::vector<double> values(aoCount);
    std::vector<double> gradients(3 * aoCount);

    double tauAlpha = 0.0;
    double tauBeta = 0.0;
    // The engine's closed-shell split: D_alpha = D_beta = D/2, held out of the
    // point loop because it is the same matrix at every point.
    const Eigen::MatrixXd spinDensity = 0.5 * density;

    for (const excgrid::Block& block : engine->Grid().Blocks())
    {
        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            const double quadratureWeight = block.weights[point];

            if (quadratureWeight == 0.0)
            {
                continue;
            }

            evaluator.EvaluateGradients(block.points[point], values, gradients);

            const BruteForcePoint fields = BruteForce(spinDensity, values, gradients);

            const auto kernel = EvaluateAt(*tauX,
                                           fields.rho,
                                           fields.rho,
                                           fields.sigma,
                                           fields.sigma,
                                           fields.sigma,
                                           fields.tau,
                                           fields.tau);
            ASSERT_TRUE(kernel.has_value());

            // grad(phi)^T P grad(phi) = sum_mu nu P_mu nu grad phi_mu . grad phi_nu.
            double gradientPerturbation = 0.0;

            for (std::size_t mu = 0; mu < aoCount; ++mu)
            {
                for (std::size_t nu = 0; nu < aoCount; ++nu)
                {
                    double dot = 0.0;

                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        dot += gradients[3 * mu + axis] * gradients[3 * nu + axis];
                    }

                    gradientPerturbation +=
                        perturbation(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) *
                        dot;
                }
            }

            tauAlpha += quadratureWeight * 0.5 * kernel->vtauA * gradientPerturbation;
            tauBeta += quadratureWeight * 0.5 * kernel->vtauB * gradientPerturbation;
        }
    }

    const double tauPart = 0.5 * (tauAlpha + tauBeta);
    ASSERT_GT(std::abs(tauPart), 1e-6)
        << "this fixture does not see the tau potential, so it cannot witness the half";

    RecordProperty("finite difference (Hartree)", Scientific(numeric));
    RecordProperty("analytic potential (Hartree)", Scientific(analytic));
    RecordProperty("tau contribution to the analytic value", Scientific(tauPart));

    // The deliberate broken case: dropping the 1/2 puts the analytic value at
    // analytic + tauPart, which the finite difference misses by the whole tau
    // contribution - three orders above its own tolerance.
    const double missingHalf = analytic + tauPart;
    EXPECT_GT(std::abs(numeric - missingHalf), 0.5 * std::abs(tauPart))
        << "the finite difference cannot separate the missing half";
}

// ---------------------------------------------------------------------------
// Gate 6: the tau screening sweep (Q-251's open risk)
// ---------------------------------------------------------------------------

// A fixture with both ends the density-weight envelope was not tuned for: an
// oxygen STO-3G core (its 1s exponent is 130.7) beside a diffuse hydrogen s
// function (exponent 0.03). The tau tail carries (2 zeta r)^2 relative to the
// density, so this is where a selection built on |phi| can be too aggressive.
constexpr std::string_view kTightCoreDiffuseBasis = R"(BASIS "ao basis" SPHERICAL PRINT
O    S
      0.1307093214E+03       0.1543289673E+00
      0.2380886605E+02       0.5353281423E+00
      0.6443608313E+01       0.4446345422E+00
O    SP
      0.5033151319E+01      -0.9996722919E-01       0.1559162750E+00
      0.1169596125E+01       0.3995128261E+00       0.6076837186E+00
      0.3803889600E+00       0.7001154689E+00       0.3919573931E+00
H    S
      3.4252509140E+00       1.5432896730E-01
      6.2391372980E-01       5.3532814230E-01
      1.6885540400E-01       4.4463454220E-01
H    S
      3.0000000000E-02       1.0000000000E+00
END
)";

qcx::Result<qcx::molecule::Molecule> MakeTightCoreDiffuseMolecule() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.8;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

// O/STO-3G carries 5 AOs (2s, 1p) and H contributes the STO-3G s plus the
// diffuse s: 7 in all. Both hydrogen slots carry density so the diffuse shell
// is populated whether the canonical order puts O first or not.
constexpr std::size_t kScreeningAOCount = 7;

Eigen::MatrixXd ScreeningAlphaDensity() {
    Eigen::MatrixXd density = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(kScreeningAOCount),
                                                    static_cast<Eigen::Index>(kScreeningAOCount));
    density.diagonal() << 0.6, 0.6, 0.4, 0.4, 0.4, 0.5, 0.8;

    return density;
}

Eigen::MatrixXd ScreeningBetaDensity() {
    Eigen::MatrixXd density = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(kScreeningAOCount),
                                                    static_cast<Eigen::Index>(kScreeningAOCount));
    density.diagonal() << 0.3, 0.3, 0.2, 0.2, 0.2, 0.25, 0.4;

    return density;
}

// The row-major copy ShellDensityWeights walks.
std::vector<double> RowMajorCopy(const Eigen::MatrixXd& density) {
    const Eigen::Index n = density.rows();
    std::vector<double> flat(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map(
        flat.data(), n, n);
    map = density;

    return flat;
}

struct ScreeningFixture {
    XcGridEngine engine;
    std::unique_ptr<excgrid::GridDerivativeProvider> provider;
};

std::optional<ScreeningFixture> MakeScreeningFixture() {
    const auto basis = qcx::basisset::ParseNwchemText(kTightCoreDiffuseBasis);

    if (!basis.has_value())
    {
        return std::nullopt;
    }

    const auto molecule = MakeTightCoreDiffuseMolecule();

    if (!molecule.has_value())
    {
        return std::nullopt;
    }

    const XcGridSettings settings = GateSettings();
    auto engine = XcGridEngine::Create(*molecule, *basis, "tau_x", settings);

    if (!engine.has_value())
    {
        return std::nullopt;
    }

    const auto geometry = ToExcgridGeometry(*molecule);

    if (!geometry.has_value())
    {
        return std::nullopt;
    }

    auto provider = excgrid::CreateGridDerivativeProvider(*geometry, ToExcgridParams(settings));

    if (!provider.has_value())
    {
        return std::nullopt;
    }

    return ScreeningFixture{std::move(*engine), std::move(*provider)};
}

// The engine's selection rule, reproduced from the public pieces: a shell is
// kept when the stronger spin's density weight, decayed past the shell's
// radial extent, exceeds the tolerance.
std::vector<std::size_t> SelectedShells(std::span<const ShellEnvelope> envelopes,
                                        std::span<const double> weightsAlpha,
                                        std::span<const double> weightsBeta,
                                        const std::array<double, 3>& point,
                                        double tolerance,
                                        std::size_t& shellTests) {
    std::vector<std::size_t> selected;

    for (std::size_t shell = 0; shell < envelopes.size(); ++shell)
    {
        ++shellTests;
        const double weight = std::max(weightsAlpha[shell], weightsBeta[shell]);

        if (ShellIsSignificant(envelopes[shell], point, weight, tolerance))
        {
            selected.push_back(shell);
        }
    }

    return selected;
}

std::vector<std::size_t> SelectedAos(std::span<const ShellEnvelope> envelopes,
                                     const std::vector<std::size_t>& shells) {
    std::vector<std::size_t> slots;

    for (const std::size_t shell : shells)
    {
        for (std::size_t ao = 0; ao < envelopes[shell].functionCount; ++ao)
        {
            slots.push_back(envelopes[shell].aoOffset + ao);
        }
    }

    return slots;
}

// One spin's tau and its spatial gradient over one AO block. The gradient is
// d_k tau = sum_nu (D grad phi)_nu . d_k grad phi_nu; the Hessian's six
// entries are (xx, xy, xz, yy, yz, zz), so the three derivative rows are
// (xx, xy, xz), (xy, yy, yz), (xz, yz, zz).
struct TauField {
    double tau = 0.0;
    std::array<double, 3> gradient{};
};

// The AO derivative fields TauAtPoint reads: the gradients (3 per AO) and the
// Hessians (6 per AO), each in the evaluator's layout.
struct TauDerivatives {
    std::span<const double> gradients;
    std::span<const double> hessians;
};

TauField TauAtPoint(const Eigen::MatrixXd& density,
                    TauDerivatives derivatives,
                    std::span<const std::size_t> slots) {
    TauField field;

    for (const std::size_t nu : slots)
    {
        std::array<double, 3> densityGradient{};

        for (const std::size_t mu : slots)
        {
            const double element =
                density(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu));

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                densityGradient[axis] += element * derivatives.gradients[3 * mu + axis];
            }
        }

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            field.tau += 0.5 * derivatives.gradients[3 * nu + axis] * densityGradient[axis];
        }

        const double* hessian = derivatives.hessians.data() + 6 * nu;
        field.gradient[0] += densityGradient[0] * hessian[0] + densityGradient[1] * hessian[1] +
                             densityGradient[2] * hessian[2];
        field.gradient[1] += densityGradient[0] * hessian[1] + densityGradient[1] * hessian[3] +
                             densityGradient[2] * hessian[4];
        field.gradient[2] += densityGradient[0] * hessian[2] + densityGradient[1] * hessian[4] +
                             densityGradient[2] * hessian[5];
    }

    return field;
}

struct TauScreeningError {
    double tauWeighted = 0.0; ///< sum |w| |tau_selected - tau_dense|, both spins.
    double tauReference = 0.0; ///< sum |w| |tau_dense|, both spins.
    double gradientWeighted = 0.0; ///< The same for |grad tau|.
    double gradientReference = 0.0;
    std::size_t shellTests = 0;
    std::size_t shellKeeps = 0;
};

TauScreeningError MeasureTauScreening(const XcGridEngine& engine,
                                      const Eigen::MatrixXd& densityAlpha,
                                      const Eigen::MatrixXd& densityBeta,
                                      std::span<const double> weightsAlpha,
                                      std::span<const double> weightsBeta,
                                      double tolerance) {
    const std::size_t aoCount = engine.AOCount();
    std::vector<double> denseValues(aoCount);
    std::vector<double> denseGradients(3 * aoCount);
    std::vector<double> denseHessians(6 * aoCount);
    std::vector<double> values(aoCount, 0.0);
    std::vector<double> gradients(3 * aoCount, 0.0);
    std::vector<double> hessians(6 * aoCount, 0.0);
    std::vector<std::size_t> allSlots(aoCount);
    std::iota(allSlots.begin(), allSlots.end(), 0U);

    TauScreeningError error;
    const std::span<const ShellEnvelope> envelopes = engine.ShellEnvelopes();

    for (const excgrid::Block& block : engine.Grid().Blocks())
    {
        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            const double quadratureWeight = block.weights[point];

            if (quadratureWeight == 0.0)
            {
                continue;
            }

            engine.Evaluator().EvaluateDerivatives(
                block.points[point], denseValues, denseGradients, denseHessians);

            const std::vector<std::size_t> shells = SelectedShells(envelopes,
                                                                   weightsAlpha,
                                                                   weightsBeta,
                                                                   block.points[point],
                                                                   tolerance,
                                                                   error.shellTests);
            error.shellKeeps += shells.size();

            if (shells.empty())
            {
                continue;
            }

            const std::vector<std::size_t> slots = SelectedAos(envelopes, shells);
            engine.Evaluator().EvaluateDerivativesSelected(
                block.points[point], shells, values, gradients, hessians);

            for (const Eigen::MatrixXd* density : {&densityAlpha, &densityBeta})
            {
                const TauField dense =
                    TauAtPoint(*density, TauDerivatives{denseGradients, denseHessians}, allSlots);
                const TauField screened =
                    TauAtPoint(*density, TauDerivatives{gradients, hessians}, slots);

                error.tauWeighted +=
                    std::abs(quadratureWeight) * std::abs(screened.tau - dense.tau);
                error.tauReference += std::abs(quadratureWeight) * std::abs(dense.tau);

                double denseNorm = 0.0;
                double differenceNorm = 0.0;

                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    denseNorm += dense.gradient[axis] * dense.gradient[axis];
                    const double difference = screened.gradient[axis] - dense.gradient[axis];
                    differenceNorm += difference * difference;
                }

                error.gradientWeighted += std::abs(quadratureWeight) * std::sqrt(differenceNorm);
                error.gradientReference += std::abs(quadratureWeight) * std::sqrt(denseNorm);
            }
        }
    }

    return error;
}

TEST(TauReferenceTest, TheScreeningSelectionCoversTauAtTheDensityTestsOwnOrder) {
    const auto fixture = MakeScreeningFixture();
    ASSERT_TRUE(fixture.has_value());
    ASSERT_EQ(fixture->engine.AOCount(), kScreeningAOCount)
        << "the tight-core/diffuse fixture's AO layout changed";

    const Eigen::MatrixXd alpha = ScreeningAlphaDensity();
    const Eigen::MatrixXd beta = ScreeningBetaDensity();

    const auto dense = fixture->engine.Evaluate(alpha, beta);
    ASSERT_TRUE(dense.has_value()) << dense.error().message;

    const auto weightsAlphaResult = ShellDensityWeights(
        fixture->engine.ShellEnvelopes(), RowMajorCopy(alpha), fixture->engine.AOCount());
    const auto weightsBetaResult = ShellDensityWeights(
        fixture->engine.ShellEnvelopes(), RowMajorCopy(beta), fixture->engine.AOCount());
    ASSERT_TRUE(weightsAlphaResult.has_value());
    ASSERT_TRUE(weightsBetaResult.has_value());

    // 1e-8 is the tolerance family this project runs at elsewhere; 1e-12 is the
    // near-dense control and 1e-4 the loose end of the sweep.
    const std::array<double, 5> tolerances = {1e-12, 1e-10, 1e-8, 1e-6, 1e-4};
    std::array<double, 5> tauRelative{};
    std::array<double, 5> gradientTauRelative{};
    std::array<double, 5> energyError{};
    std::array<double, 5> potentialError{};
    std::array<std::size_t, 5> keeps{};
    std::array<std::size_t, 5> tests{};

    for (std::size_t index = 0; index < tolerances.size(); ++index)
    {
        const auto screened = fixture->engine.EvaluateScreened(alpha, beta, tolerances[index]);
        ASSERT_TRUE(screened.has_value()) << screened.error().message;

        energyError[index] = std::abs(screened->energy - dense->energy);
        potentialError[index] =
            std::max((screened->potentialAlpha - dense->potentialAlpha).cwiseAbs().maxCoeff(),
                     (screened->potentialBeta - dense->potentialBeta).cwiseAbs().maxCoeff());

        const TauScreeningError tauError = MeasureTauScreening(fixture->engine,
                                                               alpha,
                                                               beta,
                                                               *weightsAlphaResult,
                                                               *weightsBetaResult,
                                                               tolerances[index]);
        tauRelative[index] = tauError.tauWeighted / std::max(tauError.tauReference, 1e-300);
        gradientTauRelative[index] =
            tauError.gradientWeighted / std::max(tauError.gradientReference, 1e-300);
        keeps[index] = screened->counts.shellKeeps;
        tests[index] = screened->counts.shellTests;

        RecordProperty("tolerance " + std::to_string(index) + ": tau relative L1",
                       Scientific(tauRelative[index]));
        RecordProperty("tolerance " + std::to_string(index) + ": grad tau relative L1",
                       Scientific(gradientTauRelative[index]));
        RecordProperty("tolerance " + std::to_string(index) + ": energy error (Hartree)",
                       Scientific(energyError[index]));
        RecordProperty("tolerance " + std::to_string(index) + ": potential max error",
                       Scientific(potentialError[index]));
        RecordProperty("tolerance " + std::to_string(index) + ": shells kept / tests",
                       std::to_string(keeps[index]) + " / " + std::to_string(tests[index]));
    }

    RecordProperty("dense energy (Hartree)", Scientific(dense->energy));

    // The sweep must bite at its loose end, or it measures nothing.
    EXPECT_LT(keeps[4], tests[4]) << "the loose tolerance keeps every shell";

    // Q-251's criterion. The density-weight test bounds the neglected density;
    // tau's tail carries an extra (2 zeta r)^2, so the question the gate asks
    // is whether that factor keeps the neglected tau at the density test's own
    // practical order at 1e-8. An error above these bars says the selection is
    // wrong for tau rather than merely loose.
    constexpr std::size_t kShippedToleranceIndex = 2; // 1e-8
    EXPECT_LT(tauRelative[kShippedToleranceIndex], 1e-4)
        << "the selection neglects tau far beyond the density's own order";
    EXPECT_LT(gradientTauRelative[kShippedToleranceIndex], 1e-4)
        << "the selection neglects grad tau far beyond the density's own order";
    EXPECT_LT(energyError[kShippedToleranceIndex], 1e-6)
        << "the neglected tau moves the energy above 1e-6 Hartree at 1e-8";
    EXPECT_LT(potentialError[kShippedToleranceIndex], 1e-6)
        << "the neglected tau moves the potential above 1e-6 Hartree at 1e-8";

    // Tightening the tolerance does not make the tau error worse.
    EXPECT_LE(tauRelative[kShippedToleranceIndex], tauRelative[4] + 1e-12)
        << "the tau error does not respond to the tolerance";
    EXPECT_LE(gradientTauRelative[kShippedToleranceIndex], gradientTauRelative[4] + 1e-12);

    // The near-dense control: at 1e-12 screening reproduces the dense path.
    EXPECT_LT(energyError[0], 1e-10);
    EXPECT_LT(potentialError[0], 1e-9);
    EXPECT_LT(tauRelative[0], 1e-10);

    // The deliberate broken case, as a number rather than an implementation: a
    // tau doubled on this metric scores a relative L1 of 1 (halved, 0.5),
    // three orders above the criterion the selection is held to.
    const double doubledRelativeL1 = 1.0;
    EXPECT_GT(doubledRelativeL1, 1e3 * tauRelative[kShippedToleranceIndex])
        << "the gate cannot separate a doubled tau from the screening residual";
}

TEST(TauReferenceTest, TheScreeningSweepBoundsTheTauGradientErrorAndMatchesTheEnergyPathsEnergy) {
    const auto fixture = MakeScreeningFixture();
    ASSERT_TRUE(fixture.has_value());

    const Eigen::MatrixXd alpha = ScreeningAlphaDensity();
    const Eigen::MatrixXd beta = ScreeningBetaDensity();
    const XcGridSettings settings = GateSettings();
    const XcBatchSettings batch;

    // The dense reference keeps every shell unconditionally (a negative
    // tolerance), so it is the walk without the selection rather than a
    // tighter selection that could itself be wrong.
    const auto dense = EvaluateXcGradient(fixture->engine,
                                          alpha,
                                          beta,
                                          *fixture->provider,
                                          XcIntegrationThresholds::FromEnergyPath(settings, -1.0),
                                          batch);
    ASSERT_TRUE(dense.has_value()) << dense.error().message;

    const double denseScale = dense->gradient.cwiseAbs().maxCoeff();

    const std::array<double, 5> tolerances = {1e-12, 1e-10, 1e-8, 1e-6, 1e-4};
    std::array<double, 5> gradientError{};
    std::array<double, 5> gradientRelative{};

    for (std::size_t index = 0; index < tolerances.size(); ++index)
    {
        const auto energyPath = fixture->engine.EvaluateScreened(alpha, beta, tolerances[index]);
        ASSERT_TRUE(energyPath.has_value()) << energyPath.error().message;

        const auto walk =
            EvaluateXcGradient(fixture->engine,
                               alpha,
                               beta,
                               *fixture->provider,
                               XcIntegrationThresholds::FromEnergyPath(settings, tolerances[index]),
                               batch);
        ASSERT_TRUE(walk.has_value()) << walk.error().message;

        gradientError[index] = (walk->gradient - dense->gradient).cwiseAbs().maxCoeff();
        gradientRelative[index] =
            gradientError[index] / std::max(denseScale, std::numeric_limits<double>::min());

        RecordProperty("gradient tolerance " + std::to_string(index) + ": max error (Hartree/Bohr)",
                       Scientific(gradientError[index]));
        RecordProperty("gradient tolerance " + std::to_string(index) + ": relative",
                       Scientific(gradientRelative[index]));

        EXPECT_DOUBLE_EQ(walk->thresholds.screeningTolerance, tolerances[index]);

        // The tau adapter regression: the walk integrates its own energy from
        // the same slices it differentiates. Before B2 the walk reads tau = 0
        // and this line separates it from the energy path by the whole tau
        // contribution, which is exactly what the gate is for.
        EXPECT_DOUBLE_EQ(walk->energy, energyPath->energy)
            << "the gradient walk integrated an energy the energy path does not recognise "
               "(B2's tau plumbing is missing or screening differently)";
    }

    RecordProperty("dense gradient scale (Hartree/Bohr)", Scientific(denseScale));

    // At the near-dense end the screened walk is the dense one; at 1e-8 the
    // selection's gradient error stays at the density test's order.
    EXPECT_LT(gradientError[0], 1e-8);
    EXPECT_LT(gradientRelative[2], 1e-4)
        << "the selection neglects the tau gradient far beyond the density's order";
    EXPECT_LE(gradientRelative[2], gradientRelative[4] + 1e-12)
        << "the gradient error does not respond to the tolerance";
}

} // namespace
} // namespace qcx::grid
