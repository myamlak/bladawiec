// — the driver's Kohn-Sham composition, pinned by value.
//
// Two layers, deliberately:
//
//  * The ARITHMETIC layer drives the composition's three callbacks with
//    halves and an XC evaluator whose every number is written down here, so
//    each term's placement and scale is checked exactly rather than
//    approximately: F_KS = H + J[D] + Vxc - c_HF K[D]/2, the Coulomb
//    callback is exactly J[D], and the contribution is exactly
//    Exc - c_HF 1/4 Tr[D K[D]]. A composition that adds the XC potential
//    twice, forgets the exact-exchange subtraction, or halves the wrong
//    quantity fails by value.
//
//  * The RUN layer closes the loop with the real H2/STO-3G integrals: a real
//    RunRhfScf / RunUhfScf through the composition, with the energy then
//    recomputed here from the density the run returned. That is what makes
//    "the SCF converged to this Kohn-Sham energy" checkable from outside,
//    and it is the same instrument seam_test.cpp uses on the bare seam.
//
// The real grid engine appears in the last two tests, where the functional
// is excgrid's own "slater" rather than a function written down here.
#include "h2_sto3g.hpp"
#include "internal/ks_composition.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/lean_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/molecule/mass_properties.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

using qcx::driver::internal::HalfFockFn;
using qcx::driver::internal::KsFunctional;
using qcx::driver::internal::MakeRksSeam;
using qcx::driver::internal::MakeUksSeam;
using qcx::driver::internal::ResolveKsFunctional;
using qcx::driver::internal::XcEvaluatorFn;

// The J and K supermatrices of one (uv|ws) tensor, in the flat row-major
// order the direct builders contract against: J = E d and K = E_x d for the
// flat density d.
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

Eigen::MatrixXd CoulombOf(const JkSuper& super, const Eigen::MatrixXd& density) {
    return UnflattenRowMajor(super.coulomb * FlattenRowMajor(density), density.rows());
}

Eigen::MatrixXd ExchangeOf(const JkSuper& super, const Eigen::MatrixXd& density) {
    return UnflattenRowMajor(super.exchange * FlattenRowMajor(density), density.rows());
}

// The two halves of the integrals convention, as densitY -> Fock maps:
//   coulombHalf(rho) = H + 2 J(rho) (a buildCoulombOnly member)
//   exchangeHalf(rho) = H - K(rho) (a buildExchangeOnly member)
HalfFockFn MakeCoulombHalf(const JkSuper& super, const Eigen::MatrixXd& coreHamiltonian) {
    return [super, coreHamiltonian](const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> {
        return coreHamiltonian + 2.0 * CoulombOf(super, rho);
    };
}

HalfFockFn MakeExchangeHalf(const JkSuper& super, const Eigen::MatrixXd& coreHamiltonian) {
    return [super, coreHamiltonian](const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> {
        return coreHamiltonian - ExchangeOf(super, rho);
    };
}

// A synthetic XC functional with every number written down: the energy is
// g (Tr d_alpha^2 + Tr d_beta^2) and the potentials are its exact
// derivatives, 2 g Tr d_s * I. Quadratic in the density, so a composition
// that evaluates the potential at the wrong density (or drops the
// functional's density dependence) is off by a measurable amount rather
// than by rounding.
struct SyntheticFunctional {
    double g = 0.0;
};

qcx::Result<qcx::grid::XcEvaluation> EvaluateSynthetic(const SyntheticFunctional& functional,
                                                       const Eigen::MatrixXd& densityAlpha,
                                                       const Eigen::MatrixXd& densityBeta) {
    const Eigen::Index n = densityAlpha.rows();
    const double traceAlpha = densityAlpha.trace();
    const double traceBeta = densityBeta.trace();

    qcx::grid::XcEvaluation evaluation;
    evaluation.energy = functional.g * (traceAlpha * traceAlpha + traceBeta * traceBeta);
    evaluation.potentialAlpha = (2.0 * functional.g * traceAlpha) * Eigen::MatrixXd::Identity(n, n);
    evaluation.potentialBeta = (2.0 * functional.g * traceBeta) * Eigen::MatrixXd::Identity(n, n);
    return evaluation;
}

XcEvaluatorFn MakeSyntheticEvaluator(const SyntheticFunctional& functional) {
    return
        [functional](const Eigen::MatrixXd& densityAlpha,
                     const Eigen::MatrixXd& densityBeta) -> qcx::Result<qcx::grid::XcEvaluation> {
            return EvaluateSynthetic(functional, densityAlpha, densityBeta);
        };
}

// A three-function system with no symmetry: an H2 whose second atom is
// pushed off the bond axis. Nothing downstream needs a real molecule here -
// the halves and the evaluator are analytic - but a density with all
// three functions populated is what makes the Fock checks non-degenerate.
Eigen::MatrixXd MakeTestDensity() {
    Eigen::MatrixXd density(3, 3);
    density << 0.90, 0.31, -0.22, 0.31, 0.74, 0.18, -0.22, 0.18, 0.63;
    return density;
}

Eigen::MatrixXd MakeTestCore() {
    Eigen::MatrixXd core(3, 3);
    core << -1.20, -0.34, 0.11, -0.34, -0.86, -0.27, 0.11, -0.27, -0.55;
    return core;
}

// ---- the arithmetic layer ------------------------------------------------

TEST(KsCompositionTest, ResolvesAShippedFunctionalAndRefusesAnUnknownOne) {
    const auto b3lyp = ResolveKsFunctional("b3lyp");
    ASSERT_TRUE(b3lyp.has_value()) << b3lyp.error().message;
    EXPECT_EQ(b3lyp->name, "b3lyp");
    EXPECT_DOUBLE_EQ(b3lyp->exchangeFraction, 0.20);
    EXPECT_TRUE(b3lyp->usesGradient);

    const auto pbe = ResolveKsFunctional("pbe");
    ASSERT_TRUE(pbe.has_value()) << pbe.error().message;
    EXPECT_DOUBLE_EQ(pbe->exchangeFraction, 0.0);
    EXPECT_TRUE(pbe->usesGradient);

    // An LDA: no gradient, no exact exchange.
    const auto svwn = ResolveKsFunctional("svwn");
    ASSERT_TRUE(svwn.has_value()) << svwn.error().message;
    EXPECT_DOUBLE_EQ(svwn->exchangeFraction, 0.0);
    EXPECT_FALSE(svwn->usesGradient);

    // An unknown name is refused WITH the shipped set - never substituted.
    const auto unknown = ResolveKsFunctional("blyp");
    ASSERT_FALSE(unknown.has_value());
    EXPECT_EQ(unknown.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(unknown.error().message.find("b3lyp"), std::string::npos);
    EXPECT_NE(unknown.error().message.find("pbe"), std::string::npos);
}

TEST(KsCompositionTest, RksPlacesEveryTermAndScalesTheExchangeByTheFraction) {
    const Eigen::MatrixXd core = MakeTestCore();
    const Eigen::MatrixXd density = MakeTestDensity();
    const double fraction = 0.35;

    // The oracle, in closed form: the bilinear maps the halves above encode.
    // Both are LINEAR in the density, which is what fixes the scale of the
    // exchange term: the composition hands the halves rho = D/2, so the
    // exchange half returns H - K[D]/2 and the Fock's exchange term is
    // -c_HF K[D]/2 - one half of the bilinear form at D, not all of it.
    const auto jOf = [&](const Eigen::MatrixXd& d) {
        return 0.5 * d.trace() * Eigen::MatrixXd::Identity(3, 3) + 0.25 * d;
    };
    const auto kOf = [&](const Eigen::MatrixXd& d) {
        return 0.3 * d.trace() * Eigen::MatrixXd::Identity(3, 3) + 0.6 * d;
    };

    const HalfFockFn coulombHalf = [&](const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> {
        return core + 2.0 * jOf(rho);
    };
    const HalfFockFn exchangeHalf =
        [&](const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> { return core - kOf(rho); };

    const SyntheticFunctional functional{0.7};
    const auto seam =
        MakeRksSeam(coulombHalf, exchangeHalf, core, fraction, MakeSyntheticEvaluator(functional));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    // The XC evaluation the composition must have made: at rho = D/2.
    const Eigen::MatrixXd rho = 0.5 * density;
    const auto xc = EvaluateSynthetic(functional, rho, rho);
    ASSERT_TRUE(xc.has_value());

    const Eigen::MatrixXd expectedFock = core + jOf(density) +
                                         0.5 * (xc->potentialAlpha + xc->potentialBeta) -
                                         0.5 * fraction * kOf(density);

    const auto fock = seam->fock(density);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;
    EXPECT_NEAR((*fock - expectedFock).cwiseAbs().maxCoeff(), 0.0, 1e-13);

    const auto coulomb = seam->coulomb(density);
    ASSERT_TRUE(coulomb.has_value()) << coulomb.error().message;
    EXPECT_NEAR((*coulomb - jOf(density)).cwiseAbs().maxCoeff(), 0.0, 1e-13);

    const auto contribution = seam->contribution(density);
    ASSERT_TRUE(contribution.has_value()) << contribution.error().message;
    const double expectedExchange = -0.25 * fraction * (density.cwiseProduct(kOf(density))).sum();
    EXPECT_NEAR(*contribution, xc->energy + expectedExchange, 1e-13);

    // The scale is checked by value, not just by shape: the exact-exchange
    // piece must carry the FRACTION, so a composition that dropped it (or
    // applied the full K, the Hartree-Fock term) fails here by a wide margin.
    const double fullExchange = -0.25 * (density.cwiseProduct(kOf(density))).sum();
    EXPECT_GT(std::abs(fraction * fullExchange), 1e-6);
    EXPECT_NEAR(*contribution - xc->energy, fraction * fullExchange, 1e-13);
}

TEST(KsCompositionTest, RksPureFunctionalNeverBuildsTheExchangeHalf) {
    const Eigen::MatrixXd core = MakeTestCore();
    const Eigen::MatrixXd density = MakeTestDensity();

    int coulombCalls = 0;
    int exchangeCalls = 0;

    const HalfFockFn coulombHalf = [&](const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> {
        ++coulombCalls;
        return core + 2.0 * rho;
    };
    const HalfFockFn exchangeHalf =
        [&](const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> {
        ++exchangeCalls;
        return core - rho;
    };

    const SyntheticFunctional functional{0.4};
    const auto seam =
        MakeRksSeam(coulombHalf, exchangeHalf, core, 0.0, MakeSyntheticEvaluator(functional));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    const auto fock = seam->fock(density);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;

    const Eigen::MatrixXd rho = 0.5 * density;
    const auto xc = EvaluateSynthetic(functional, rho, rho);
    ASSERT_TRUE(xc.has_value());
    const Eigen::MatrixXd expected =
        core + density + 0.5 * (xc->potentialAlpha + xc->potentialBeta);
    EXPECT_NEAR((*fock - expected).cwiseAbs().maxCoeff(), 0.0, 1e-13);

    // The whole point of a pure functional's cost: no K build at all.
    EXPECT_EQ(exchangeCalls, 0);
    EXPECT_EQ(coulombCalls, 1);

    // A non-zero fraction with no exchange half is a REFUSAL, not a silent
    // Hartree-Fock-exact-exchange-free run.
    const auto refused =
        MakeRksSeam(coulombHalf, HalfFockFn{}, core, 0.25, MakeSyntheticEvaluator(functional));
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(KsCompositionTest, RksBuildsEachDensityOnceAcrossTheThreeCallbacks) {
    const Eigen::MatrixXd core = MakeTestCore();
    const Eigen::MatrixXd density = MakeTestDensity();

    int coulombCalls = 0;
    int exchangeCalls = 0;
    int xcCalls = 0;

    const HalfFockFn coulombHalf = [&](const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> {
        ++coulombCalls;
        return core + 2.0 * rho;
    };
    const HalfFockFn exchangeHalf =
        [&](const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> {
        ++exchangeCalls;
        return core - rho;
    };
    const SyntheticFunctional functional{0.4};
    const XcEvaluatorFn evaluator =
        [&](const Eigen::MatrixXd& alpha,
            const Eigen::MatrixXd& beta) -> qcx::Result<qcx::grid::XcEvaluation> {
        ++xcCalls;
        return EvaluateSynthetic(functional, alpha, beta);
    };

    const auto seam = MakeRksSeam(coulombHalf, exchangeHalf, core, 0.5, evaluator);
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    // The loop's own order: the Fock build on one density, then the Coulomb
    // and the contribution on the next - and the Fock build on THAT one
    // again. One density must cost one fill.
    const Eigen::MatrixXd first = MakeTestDensity();
    Eigen::MatrixXd second = 0.9 * first;

    ASSERT_TRUE(seam->fock(first).has_value());
    EXPECT_EQ(coulombCalls, 1);
    EXPECT_EQ(exchangeCalls, 1);
    EXPECT_EQ(xcCalls, 1);

    ASSERT_TRUE(seam->coulomb(second).has_value());
    ASSERT_TRUE(seam->contribution(second).has_value());
    EXPECT_EQ(coulombCalls, 2);
    EXPECT_EQ(exchangeCalls, 2);
    EXPECT_EQ(xcCalls, 2);

    // The next iteration's Fock build re-asks for the second density: the
    // entry is already there, so nothing is rebuilt.
    ASSERT_TRUE(seam->fock(second).has_value());
    EXPECT_EQ(coulombCalls, 2);
    EXPECT_EQ(exchangeCalls, 2);
    EXPECT_EQ(xcCalls, 2);

    // And a re-ask of the FIRST density is a miss, not a stale hit.
    ASSERT_TRUE(seam->fock(first).has_value());
    EXPECT_EQ(coulombCalls, 3);
}

TEST(KsCompositionTest, UksPlacesEveryTermWithPerSpinPotentials) {
    const Eigen::MatrixXd core = MakeTestCore();
    const Eigen::MatrixXd densityAlpha = MakeTestDensity();
    const Eigen::MatrixXd densityBeta = 0.8 * densityAlpha;
    const double fraction = 0.25;

    const auto jOf = [](const Eigen::MatrixXd& d) {
        return 0.4 * d.trace() * Eigen::MatrixXd::Identity(3, 3) + 0.2 * d;
    };
    const auto kOf = [](const Eigen::MatrixXd& d) {
        return 0.5 * d.trace() * Eigen::MatrixXd::Identity(3, 3) + 0.7 * d;
    };

    const HalfFockFn coulombHalf = [&](const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> {
        return core + 2.0 * jOf(rho);
    };
    const HalfFockFn exchangeAlpha = [&](const Eigen::MatrixXd& d) -> qcx::Result<Eigen::MatrixXd> {
        return core - kOf(d);
    };
    const HalfFockFn exchangeBeta = exchangeAlpha;

    const SyntheticFunctional functional{0.55};
    const auto seam = MakeUksSeam(coulombHalf,
                                  exchangeAlpha,
                                  exchangeBeta,
                                  core,
                                  fraction,
                                  MakeSyntheticEvaluator(functional));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    const auto xc = EvaluateSynthetic(functional, densityAlpha, densityBeta);
    ASSERT_TRUE(xc.has_value());

    // Unlike the closed-shell case, the exchange halves take the RAW
    // per-spin densities (the direct-UHF convention), so the Fock's
    // exchange term is the full -c_HF K[d_sigma], with no further halving.
    const Eigen::MatrixXd coulombOfTotal = jOf(densityAlpha + densityBeta);

    const auto focks = seam->fock(densityAlpha, densityBeta);
    ASSERT_TRUE(focks.has_value()) << focks.error().message;

    const Eigen::MatrixXd expectedAlpha =
        core + coulombOfTotal + xc->potentialAlpha - fraction * kOf(densityAlpha);
    const Eigen::MatrixXd expectedBeta =
        core + coulombOfTotal + xc->potentialBeta - fraction * kOf(densityBeta);
    EXPECT_NEAR((focks->first - expectedAlpha).cwiseAbs().maxCoeff(), 0.0, 1e-13);
    EXPECT_NEAR((focks->second - expectedBeta).cwiseAbs().maxCoeff(), 0.0, 1e-13);

    // The per-spin potentials must differ: a composition that handed the
    // same potential to both spins would pass an alpha-only check.
    EXPECT_GT((expectedAlpha - expectedBeta).cwiseAbs().maxCoeff(), 1e-6);

    const auto coulomb = seam->coulomb(densityAlpha + densityBeta);
    ASSERT_TRUE(coulomb.has_value()) << coulomb.error().message;
    EXPECT_NEAR((*coulomb - jOf(densityAlpha + densityBeta)).cwiseAbs().maxCoeff(), 0.0, 1e-13);

    const auto contribution = seam->contribution(densityAlpha, densityBeta);
    ASSERT_TRUE(contribution.has_value()) << contribution.error().message;
    const double expectedExchange = -0.5 * fraction *
                                    ((densityAlpha.cwiseProduct(kOf(densityAlpha))).sum() +
                                     (densityBeta.cwiseProduct(kOf(densityBeta))).sum());
    EXPECT_NEAR(*contribution, xc->energy + expectedExchange, 1e-13);
}

TEST(KsCompositionTest, UksCacheKeepsTheCoulombAndPairKeysApart) {
    const Eigen::MatrixXd core = MakeTestCore();
    const Eigen::MatrixXd densityAlpha = MakeTestDensity();
    const Eigen::MatrixXd densityBeta = 0.6 * densityAlpha;

    int coulombCalls = 0;
    int exchangeCalls = 0;
    int xcCalls = 0;

    const HalfFockFn coulombHalf = [&](const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> {
        ++coulombCalls;
        return core + 2.0 * rho;
    };
    const HalfFockFn exchangeHalf = [&](const Eigen::MatrixXd& d) -> qcx::Result<Eigen::MatrixXd> {
        ++exchangeCalls;
        return core - d;
    };
    const SyntheticFunctional functional{0.3};
    const XcEvaluatorFn evaluator =
        [&](const Eigen::MatrixXd& alpha,
            const Eigen::MatrixXd& beta) -> qcx::Result<qcx::grid::XcEvaluation> {
        ++xcCalls;
        return EvaluateSynthetic(functional, alpha, beta);
    };

    const auto seam = MakeUksSeam(coulombHalf, exchangeHalf, exchangeHalf, core, 0.5, evaluator);
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    ASSERT_TRUE(seam->fock(densityAlpha, densityBeta).has_value());
    EXPECT_EQ(coulombCalls, 1);
    EXPECT_EQ(exchangeCalls, 2);
    EXPECT_EQ(xcCalls, 1);

    // The loop's next step: the Coulomb provider sees the TOTAL alone, and
    // the contribution sees the pair. Both are new densities here, so both
    // fill - but the total's fill must not rebuild the exchange halves.
    const Eigen::MatrixXd nextAlpha = 0.95 * densityAlpha;
    const Eigen::MatrixXd nextBeta = 0.95 * densityBeta;
    ASSERT_TRUE(seam->coulomb(nextAlpha + nextBeta).has_value());
    EXPECT_EQ(coulombCalls, 2);
    EXPECT_EQ(exchangeCalls, 2);

    ASSERT_TRUE(seam->contribution(nextAlpha, nextBeta).has_value());
    EXPECT_EQ(exchangeCalls, 4);
    EXPECT_EQ(xcCalls, 2);

    // And the next iteration's Fock build on that pair rebuilds nothing.
    ASSERT_TRUE(seam->fock(nextAlpha, nextBeta).has_value());
    EXPECT_EQ(coulombCalls, 2);
    EXPECT_EQ(exchangeCalls, 4);
    EXPECT_EQ(xcCalls, 2);
}

// ---- the refusals: the guards that had never fired -----------------------
//
// Every branch below is a guard no test had ever entered: the two
// composition makers refuse an argument they cannot work with, and until
// this block no caller had ever passed them one. Each refusal is pinned
// twice - the refused call asserts the error code and the load-bearing
// part of the message, and a CONTROL call differing in exactly one
// argument must return a seam - so a refusal that fired for some other
// reason than the one named here cannot pass.
//
// The failing halves and evaluators carry their OWN code and message, which
// is what makes the forwarding checkable: a failed fill hands back
// `std::unexpected(source.error())` verbatim, so what a caller sees must be
// the source's own error and not a reworded one. Calls are counted as well,
// because the counts are what say WHERE the failure happened - a failure
// that left the Coulomb half already built and the XC evaluation never
// reached is a mid-fill failure, and the count proves it rather than
// asserting it.

// A Fock half whose calls are counted and whose failure can be switched on
// between two asks: H + factor*rho, or H - factor*rho when `subtract`.
struct SwitchableHalf {
    Eigen::MatrixXd core;
    double factor = 1.0;
    bool subtract = false;
    bool fail = false;
    qcx::Error error{qcx::ErrorCode::kInternalError, "the half failed"};
    int calls = 0;
};

HalfFockFn MakeSwitchableHalf(const std::shared_ptr<SwitchableHalf>& state) {
    return [state](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        ++state->calls;

        if (state->fail)
        {
            return std::unexpected(state->error);
        }

        if (state->subtract)
        {
            return state->core - state->factor * density;
        }

        return state->core + state->factor * density;
    };
}

// The same instrument for the XC evaluator: the synthetic functional's own
// numbers, with a failure switch and a call count.
struct SwitchableEvaluator {
    SyntheticFunctional functional{0.4};
    bool fail = false;
    qcx::Error error{qcx::ErrorCode::kInternalError, "the evaluator failed"};
    int calls = 0;
};

XcEvaluatorFn MakeSwitchableEvaluator(const std::shared_ptr<SwitchableEvaluator>& state) {
    return [state](const Eigen::MatrixXd& densityAlpha,
                   const Eigen::MatrixXd& densityBeta) -> qcx::Result<qcx::grid::XcEvaluation> {
        ++state->calls;

        if (state->fail)
        {
            return std::unexpected(state->error);
        }

        return EvaluateSynthetic(state->functional, densityAlpha, densityBeta);
    };
}

TEST(KsCompositionTest, RksRefusesAnEmptyCoulombHalfOrEvaluator) {
    const Eigen::MatrixXd core = MakeTestCore();
    const auto coulombState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 2.0});
    const auto exchangeState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.7, true});
    const auto evaluatorState = std::make_shared<SwitchableEvaluator>();

    const HalfFockFn coulombHalf = MakeSwitchableHalf(coulombState);
    const HalfFockFn exchangeHalf = MakeSwitchableHalf(exchangeState);
    const XcEvaluatorFn evaluator = MakeSwitchableEvaluator(evaluatorState);

    // The two refusals are told apart by their messages, which is the only
    // thing a reader of the error holds: an empty Coulomb half and an empty
    // evaluator are two conditions and carry two names. (Before the split
    // both arms carried one text naming both callables, so the error could
    // not say which argument was empty.)
    constexpr const char* kEmptyCoulomb =
        "the closed-shell Kohn-Sham composition needs a callable Coulomb half seam";
    constexpr const char* kEmptyEvaluator =
        "the closed-shell Kohn-Sham composition needs a callable XC evaluator seam";

    // The Coulomb half empty, everything else in order. At c_HF = 0 the
    // exchange half is not consulted at all, so the emptied Coulomb half is
    // the only thing this call can be refused for.
    const auto noCoulomb = MakeRksSeam(HalfFockFn{}, exchangeHalf, core, 0.0, evaluator);
    ASSERT_FALSE(noCoulomb.has_value());
    EXPECT_EQ(noCoulomb.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(noCoulomb.error().message, kEmptyCoulomb);

    // The evaluator empty, everything else in order.
    const auto noEvaluator = MakeRksSeam(coulombHalf, exchangeHalf, core, 0.0, XcEvaluatorFn{});
    ASSERT_FALSE(noEvaluator.has_value());
    EXPECT_EQ(noEvaluator.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(noEvaluator.error().message, kEmptyEvaluator);

    // The distinction is the assertion: two conditions, two messages.
    EXPECT_NE(noCoulomb.error().message, noEvaluator.error().message);

    // The refusal is made at COMPOSITION time: neither refusal ran a half or
    // an evaluation, so nothing was built before the argument was checked.
    EXPECT_EQ(coulombState->calls, 0);
    EXPECT_EQ(exchangeState->calls, 0);
    EXPECT_EQ(evaluatorState->calls, 0);

    // The controls: the same call with the emptied argument restored.
    const auto bothPresent = MakeRksSeam(coulombHalf, exchangeHalf, core, 0.0, evaluator);
    ASSERT_TRUE(bothPresent.has_value()) << bothPresent.error().message;
}

TEST(KsCompositionTest, RksRefusesAnExchangeFractionOutsideTheUnitInterval) {
    const Eigen::MatrixXd core = MakeTestCore();
    const auto coulombState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 2.0});
    const auto exchangeState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.7, true});
    const auto evaluatorState = std::make_shared<SwitchableEvaluator>();

    const HalfFockFn coulombHalf = MakeSwitchableHalf(coulombState);
    const HalfFockFn exchangeHalf = MakeSwitchableHalf(exchangeState);
    const XcEvaluatorFn evaluator = MakeSwitchableEvaluator(evaluatorState);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    constexpr const char* kRangeMessage = "the exact-exchange fraction must lie in [0, 1]";

    // A NaN is refused as well as a plainly out-of-range number: the guard
    // is the NEGATION of the membership test, so a value that compares false
    // against both bounds takes the refusal arm instead of falling through
    // into a seam whose fraction is neither in range nor testable.
    for (const double fraction : {-0.25, 1.25, nan, infinity, -infinity})
    {
        const auto refused = MakeRksSeam(coulombHalf, exchangeHalf, core, fraction, evaluator);
        ASSERT_FALSE(refused.has_value()) << "fraction " << fraction << " was accepted";
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_EQ(refused.error().message, kRangeMessage) << "fraction " << fraction;
    }

    // Nothing was built by any of the refusals.
    EXPECT_EQ(coulombState->calls, 0);
    EXPECT_EQ(exchangeState->calls, 0);
    EXPECT_EQ(evaluatorState->calls, 0);

    // The boundaries are INCLUSIVE, and both are values a functional really
    // carries: c_HF = 0 for every pure functional, 1.0 for a full
    // exact-exchange hybrid. Refusing either would refuse a run the schema
    // can express.
    for (const double fraction : {0.0, 1.0})
    {
        const auto accepted = MakeRksSeam(coulombHalf, exchangeHalf, core, fraction, evaluator);
        ASSERT_TRUE(accepted.has_value()) << "fraction " << fraction;
    }

    // The ORDER of the two guards, pinned because only the message tells
    // them apart: an out-of-range fraction with no exchange half at all
    // satisfies both refusals, and it must be refused for the RANGE. A
    // reader sent the other message would look at the wrong argument.
    const auto bothWrong = MakeRksSeam(coulombHalf, HalfFockFn{}, core, 2.0, evaluator);
    ASSERT_FALSE(bothWrong.has_value());
    EXPECT_EQ(bothWrong.error().message, kRangeMessage);
}

TEST(KsCompositionTest, RksForwardsEachHalfsOwnFailure) {
    const Eigen::MatrixXd core = MakeTestCore();
    const Eigen::MatrixXd first = MakeTestDensity();
    const Eigen::MatrixXd second = 0.9 * first;
    const Eigen::MatrixXd third = 0.8 * first;

    const auto coulombState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 2.0});
    const auto exchangeState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.7, true});
    const auto evaluatorState = std::make_shared<SwitchableEvaluator>();
    coulombState->error = {qcx::ErrorCode::kOutOfMemory, "the Coulomb half ran out of room"};
    exchangeState->error = {qcx::ErrorCode::kDeviceError, "the exchange half lost its device"};
    evaluatorState->error = {qcx::ErrorCode::kInternalError, "the evaluator fell over"};

    const auto seam = MakeRksSeam(MakeSwitchableHalf(coulombState),
                                  MakeSwitchableHalf(exchangeState),
                                  core,
                                  0.5,
                                  MakeSwitchableEvaluator(evaluatorState));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    // The first half fails: nothing downstream of it is asked, so the error
    // is the Coulomb half's own and the fill never reached the exchange.
    coulombState->fail = true;
    const auto coulombFail = seam->fock(first);
    ASSERT_FALSE(coulombFail.has_value());
    EXPECT_EQ(coulombFail.error().code, qcx::ErrorCode::kOutOfMemory);
    EXPECT_EQ(coulombFail.error().message, "the Coulomb half ran out of room");
    EXPECT_EQ(coulombState->calls, 1);
    EXPECT_EQ(exchangeState->calls, 0);
    EXPECT_EQ(evaluatorState->calls, 0);

    // The exchange half fails with the Coulomb half already built: the
    // mid-fill failure, and the XC evaluation is still never reached.
    coulombState->fail = false;
    exchangeState->fail = true;
    const auto exchangeFail = seam->fock(second);
    ASSERT_FALSE(exchangeFail.has_value());
    EXPECT_EQ(exchangeFail.error().code, qcx::ErrorCode::kDeviceError);
    EXPECT_EQ(exchangeFail.error().message, "the exchange half lost its device");
    EXPECT_EQ(coulombState->calls, 2);
    EXPECT_EQ(exchangeState->calls, 1);
    EXPECT_EQ(evaluatorState->calls, 0);

    // And the evaluator's own failure, past both halves.
    exchangeState->fail = false;
    evaluatorState->fail = true;
    const auto evaluatorFail = seam->fock(third);
    ASSERT_FALSE(evaluatorFail.has_value());
    EXPECT_EQ(evaluatorFail.error().code, qcx::ErrorCode::kInternalError);
    EXPECT_EQ(evaluatorFail.error().message, "the evaluator fell over");
    EXPECT_EQ(coulombState->calls, 3);
    EXPECT_EQ(exchangeState->calls, 2);
    EXPECT_EQ(evaluatorState->calls, 1);
}

TEST(KsCompositionTest, RksKeepsThePreviousEntryIntactWhenAFillFails) {
    const Eigen::MatrixXd core = MakeTestCore();
    const Eigen::MatrixXd first = MakeTestDensity();
    const Eigen::MatrixXd second = 0.9 * first;
    const Eigen::MatrixXd third = 0.8 * first;
    const double fraction = 0.4;

    const auto coulombState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 2.0});
    const auto exchangeState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.7, true});
    const auto evaluatorState = std::make_shared<SwitchableEvaluator>();
    coulombState->error = {qcx::ErrorCode::kOutOfMemory, "the Coulomb half ran out of room"};
    exchangeState->error = {qcx::ErrorCode::kDeviceError, "the exchange half lost its device"};

    const auto seam = MakeRksSeam(MakeSwitchableHalf(coulombState),
                                  MakeSwitchableHalf(exchangeState),
                                  core,
                                  fraction,
                                  MakeSwitchableEvaluator(evaluatorState));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    // One good fill first, so a partially updated entry would have real
    // numbers to serve as if they belonged to the density that failed.
    const auto firstFock = seam->fock(first);
    ASSERT_TRUE(firstFock.has_value()) << firstFock.error().message;
    EXPECT_EQ(coulombState->calls, 1);
    EXPECT_EQ(exchangeState->calls, 1);
    EXPECT_EQ(evaluatorState->calls, 1);

    // The exchange half fails on a NEW density, after that density's Coulomb
    // half has been built and before its XC evaluation: the fill is
    // abandoned half-way.
    exchangeState->fail = true;
    const auto abandoned = seam->fock(second);
    ASSERT_FALSE(abandoned.has_value());
    EXPECT_EQ(abandoned.error().code, qcx::ErrorCode::kDeviceError);
    EXPECT_EQ(coulombState->calls, 2);
    EXPECT_EQ(exchangeState->calls, 2);
    EXPECT_EQ(evaluatorState->calls, 1);

    // The abandoned fill must leave no trace. The next ask for the SAME
    // density rebuilds from scratch and answers with THAT density's numbers:
    // a fill that had written its density key - or its Coulomb half -
    // before the exchange half failed would serve a HIT here instead, mixing
    // the new density's Coulomb half with the previous density's exchange
    // half and XC potential. Both the count and the value below would move.
    exchangeState->fail = false;
    const auto rebuilt = seam->fock(second);
    ASSERT_TRUE(rebuilt.has_value()) << rebuilt.error().message;
    EXPECT_EQ(coulombState->calls, 3);
    EXPECT_EQ(exchangeState->calls, 3);
    EXPECT_EQ(evaluatorState->calls, 2);

    const Eigen::MatrixXd rho = 0.5 * second;
    const auto evaluation = EvaluateSynthetic(evaluatorState->functional, rho, rho);
    ASSERT_TRUE(evaluation.has_value());
    const Eigen::MatrixXd expected =
        core + 2.0 * rho + 0.5 * (evaluation->potentialAlpha + evaluation->potentialBeta) +
        fraction * ((core - 0.7 * rho) - core);
    EXPECT_NEAR((*rebuilt - expected).cwiseAbs().maxCoeff(), 0.0, 1e-13);

    // A failure in the other half leaves no trace either. The Coulomb half is
    // asked first, so this one dies before anything of the fill has been
    // written; the next ask for the same density must still rebuild rather
    // than serve the entry that was already there. (The third failing arm -
    // the evaluation - is NOT safe, and is pinned by the leak test below.)
    coulombState->fail = true;
    const auto coulombFail = seam->fock(third);
    ASSERT_FALSE(coulombFail.has_value());
    EXPECT_EQ(coulombFail.error().code, qcx::ErrorCode::kOutOfMemory);
    EXPECT_EQ(coulombFail.error().message, "the Coulomb half ran out of room");
    EXPECT_EQ(coulombState->calls, 4);
    EXPECT_EQ(exchangeState->calls, 3);
    EXPECT_EQ(evaluatorState->calls, 2);

    coulombState->fail = false;
    const auto thirdFock = seam->fock(third);
    ASSERT_TRUE(thirdFock.has_value()) << thirdFock.error().message;
    EXPECT_EQ(coulombState->calls, 5);
    EXPECT_EQ(exchangeState->calls, 4);
    EXPECT_EQ(evaluatorState->calls, 3);
}

// THE DEFECT THIS FILE FOUND, and the invariant it now enforces.
//
// The composition states its own contract at the fill it performs
// (ks_composition.hpp, above the commit): a half-failed fill must never
// leave a partially updated entry behind. The fill used to break it for the
// EVALUATION. The exchange half was stored into the entry BEFORE the
// evaluator was asked, so an evaluation failure left the entry still keyed -
// and still marked valid - for the PREVIOUS density, while the exchange half
// inside it already belonged to the density that failed. The next ask for
// the previous density was then a HIT: no half was rebuilt, no error was
// reported, and the Fock came back carrying the wrong density's
// exact-exchange term. Measured on the pre-fix code at the parameters below,
// that Fock differed from the one the entry is keyed for by 0.0126 in its
// largest element - a silent wrong answer, not a failure.
//
// The pin is the invariant and not the defect: the entry a failed fill
// leaves behind is the one it found, contents and key alike, and the ask for
// the previous density is answered with the numbers it is keyed for. The
// second assertion keeps the first honest - the two candidates are
// measurably different, so the zero says WHICH one came back.
//
// The blast radius, stated rather than left to be implied: RunRhfScf returns
// a failed callback's error on the spot (scf/src/rhf.cpp:525), so a real run
// aborts at the first failed fill and never asks the seam again. The leak was
// observable only by a caller that keeps using the seam after a failed fill
// - which is exactly the case the contract above promises is safe.
TEST(KsCompositionTest, RksKeepsThePreviousEntryIntactWhenTheEvaluationFails) {
    const Eigen::MatrixXd core = MakeTestCore();
    const Eigen::MatrixXd first = MakeTestDensity();
    const Eigen::MatrixXd second = 0.9 * first;
    const double fraction = 0.4;

    const auto coulombState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 2.0});
    const auto exchangeState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.7, true});
    const auto evaluatorState = std::make_shared<SwitchableEvaluator>();
    evaluatorState->error = {qcx::ErrorCode::kInternalError, "the evaluator fell over"};

    const auto seam = MakeRksSeam(MakeSwitchableHalf(coulombState),
                                  MakeSwitchableHalf(exchangeState),
                                  core,
                                  fraction,
                                  MakeSwitchableEvaluator(evaluatorState));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    const auto filled = seam->fock(first);
    ASSERT_TRUE(filled.has_value()) << filled.error().message;

    // The evaluation fails on a new density, with that density's exchange
    // half already stored in the entry.
    evaluatorState->fail = true;
    const auto failed = seam->fock(second);
    ASSERT_FALSE(failed.has_value());
    EXPECT_EQ(failed.error().code, qcx::ErrorCode::kInternalError);
    evaluatorState->fail = false;

    // The ask for the PREVIOUS density: the entry the failed fill left
    // behind is intact, so this is a HIT - no half and no evaluation is
    // built again - and it holds the numbers that density is keyed for.
    const auto stale = seam->fock(first);
    ASSERT_TRUE(stale.has_value()) << stale.error().message;
    EXPECT_EQ(coulombState->calls, 2);
    EXPECT_EQ(exchangeState->calls, 2);
    EXPECT_EQ(evaluatorState->calls, 2);

    // And the numbers: the first density's Coulomb half and XC evaluation
    // with the FIRST density's exchange half. The second assertion is what
    // makes the first one mean something: the two candidates are measurably
    // different, so the zero above says which one the entry carries.
    const Eigen::MatrixXd rhoFirst = 0.5 * first;
    const Eigen::MatrixXd rhoSecond = 0.5 * second;
    const auto evaluation = EvaluateSynthetic(evaluatorState->functional, rhoFirst, rhoFirst);
    ASSERT_TRUE(evaluation.has_value());
    const auto fockOf = [&core, fraction, &evaluation](const Eigen::MatrixXd& rho,
                                                       const Eigen::MatrixXd& exchangeRho) {
        return core + 2.0 * rho + 0.5 * (evaluation->potentialAlpha + evaluation->potentialBeta) +
               fraction * ((core - 0.7 * exchangeRho) - core);
    };
    const Eigen::MatrixXd correct = fockOf(rhoFirst, rhoFirst);
    const Eigen::MatrixXd leaked = fockOf(rhoFirst, rhoSecond);
    EXPECT_NEAR((*stale - correct).cwiseAbs().maxCoeff(), 0.0, 1e-13)
        << "the entry is keyed on the first density, so it must carry that density's halves";
    EXPECT_GT((*stale - leaked).cwiseAbs().maxCoeff(), 1e-3)
        << "the two candidates must be distinguishable, or the pin above proves nothing";
}

TEST(KsCompositionTest, UksRefusesAnEmptyCoulombHalfOrEvaluator) {
    const Eigen::MatrixXd core = MakeTestCore();
    const auto coulombState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 2.0});
    const auto alphaState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.7, true});
    const auto betaState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.5, true});
    const auto evaluatorState = std::make_shared<SwitchableEvaluator>();

    const HalfFockFn coulombHalf = MakeSwitchableHalf(coulombState);
    const HalfFockFn alphaHalf = MakeSwitchableHalf(alphaState);
    const HalfFockFn betaHalf = MakeSwitchableHalf(betaState);
    const XcEvaluatorFn evaluator = MakeSwitchableEvaluator(evaluatorState);

    // c_HF = 0, so neither exchange half is required: the two calls below
    // differ from their controls in exactly one argument each. Each refusal
    // names the argument that was empty - and, because the two compositions
    // are separate call sites a reader of the error cannot see, the
    // composition that refused as well.
    constexpr const char* kEmptyCoulomb =
        "the unrestricted Kohn-Sham composition needs a callable Coulomb half seam";
    constexpr const char* kEmptyEvaluator =
        "the unrestricted Kohn-Sham composition needs a callable XC evaluator seam";

    const auto noCoulomb = MakeUksSeam(HalfFockFn{}, alphaHalf, betaHalf, core, 0.0, evaluator);
    ASSERT_FALSE(noCoulomb.has_value());
    EXPECT_EQ(noCoulomb.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(noCoulomb.error().message, kEmptyCoulomb);

    const auto noEvaluator =
        MakeUksSeam(coulombHalf, alphaHalf, betaHalf, core, 0.0, XcEvaluatorFn{});
    ASSERT_FALSE(noEvaluator.has_value());
    EXPECT_EQ(noEvaluator.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(noEvaluator.error().message, kEmptyEvaluator);

    // Nothing was built: the refusal is made before the first callback runs.
    EXPECT_EQ(coulombState->calls, 0);
    EXPECT_EQ(alphaState->calls, 0);
    EXPECT_EQ(betaState->calls, 0);
    EXPECT_EQ(evaluatorState->calls, 0);

    // The distinction is the assertion, and it holds across the two makers
    // as well: four ways to pass an empty callable, four distinct messages.
    // (Before the split all four carried one text.)
    const auto rksNoCoulomb = MakeRksSeam(HalfFockFn{}, HalfFockFn{}, core, 0.0, evaluator);
    const auto rksNoEvaluator = MakeRksSeam(coulombHalf, alphaHalf, core, 0.0, XcEvaluatorFn{});
    ASSERT_FALSE(rksNoCoulomb.has_value());
    ASSERT_FALSE(rksNoEvaluator.has_value());
    const std::vector<std::string> fourMessages{noCoulomb.error().message,
                                                noEvaluator.error().message,
                                                rksNoCoulomb.error().message,
                                                rksNoEvaluator.error().message};

    for (std::size_t i = 0; i < fourMessages.size(); ++i)
    {
        for (std::size_t j = i + 1; j < fourMessages.size(); ++j)
        {
            EXPECT_NE(fourMessages[i], fourMessages[j])
                << "messages " << i << " and " << j << " are the same text";
        }
    }

    // The controls.
    const auto bothPresent = MakeUksSeam(coulombHalf, alphaHalf, betaHalf, core, 0.0, evaluator);
    ASSERT_TRUE(bothPresent.has_value()) << bothPresent.error().message;
}

TEST(KsCompositionTest, UksRefusesAnExchangeFractionOutsideTheUnitInterval) {
    const Eigen::MatrixXd core = MakeTestCore();
    const auto coulombState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 2.0});
    const auto alphaState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.7, true});
    const auto betaState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.5, true});
    const auto evaluatorState = std::make_shared<SwitchableEvaluator>();

    const HalfFockFn coulombHalf = MakeSwitchableHalf(coulombState);
    const HalfFockFn alphaHalf = MakeSwitchableHalf(alphaState);
    const HalfFockFn betaHalf = MakeSwitchableHalf(betaState);
    const XcEvaluatorFn evaluator = MakeSwitchableEvaluator(evaluatorState);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    constexpr const char* kRangeMessage = "the exact-exchange fraction must lie in [0, 1]";

    for (const double fraction : {-0.25, 1.25, nan, infinity, -infinity})
    {
        const auto refused =
            MakeUksSeam(coulombHalf, alphaHalf, betaHalf, core, fraction, evaluator);
        ASSERT_FALSE(refused.has_value()) << "fraction " << fraction << " was accepted";
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_EQ(refused.error().message, kRangeMessage) << "fraction " << fraction;
    }

    EXPECT_EQ(coulombState->calls, 0);
    EXPECT_EQ(alphaState->calls, 0);
    EXPECT_EQ(betaState->calls, 0);
    EXPECT_EQ(evaluatorState->calls, 0);

    // The boundaries are inclusive here too.
    for (const double fraction : {0.0, 1.0})
    {
        const auto accepted =
            MakeUksSeam(coulombHalf, alphaHalf, betaHalf, core, fraction, evaluator);
        ASSERT_TRUE(accepted.has_value()) << "fraction " << fraction;
    }

    // The range guard runs before the exchange-half guard here as well: an
    // out-of-range fraction with BOTH halves missing is refused for the
    // range, which is the only one of the two messages that names it.
    const auto bothWrong =
        MakeUksSeam(coulombHalf, HalfFockFn{}, HalfFockFn{}, core, 2.0, evaluator);
    ASSERT_FALSE(bothWrong.has_value());
    EXPECT_EQ(bothWrong.error().message, kRangeMessage);
}

TEST(KsCompositionTest, UksRefusesAnExchangeHalfMissingFromEitherSpin) {
    const Eigen::MatrixXd core = MakeTestCore();
    const auto coulombState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 2.0});
    const auto alphaState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.7, true});
    const auto betaState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.5, true});
    const auto evaluatorState = std::make_shared<SwitchableEvaluator>();

    const HalfFockFn coulombHalf = MakeSwitchableHalf(coulombState);
    const HalfFockFn alphaHalf = MakeSwitchableHalf(alphaState);
    const HalfFockFn betaHalf = MakeSwitchableHalf(betaState);
    const XcEvaluatorFn evaluator = MakeSwitchableEvaluator(evaluatorState);

    // Three ways to be short a half at c_HF > 0: both, only the alpha, only
    // the beta. Each is refused BY NAME - the error says which half is
    // missing and not merely that the pair is incomplete. (Before the split
    // all three carried one text, so the distinction was the reader's guess.)
    // The expected message travels WITH its case rather than in a parallel
    // list: the pair is (alpha, beta), so the empty member is what names the
    // missing half, and an index-keyed list is how that gets read backwards.
    constexpr const char* kBothHalvesMissing =
        "the functional carries a non-zero exact-exchange fraction, so both per-spin exchange "
        "halves are required and must be callable";
    constexpr const char* kAlphaHalfMissing =
        "the functional carries a non-zero exact-exchange fraction, so the alpha spin exchange "
        "half is required and must be callable";
    constexpr const char* kBetaHalfMissing =
        "the functional carries a non-zero exact-exchange fraction, so the beta spin exchange "
        "half is required and must be callable";

    struct IncompletePair {
        HalfFockFn alpha;
        HalfFockFn beta;
        const char* message = nullptr;
    };

    const std::vector<IncompletePair> incomplete{{HalfFockFn{}, HalfFockFn{}, kBothHalvesMissing},
                                                 {alphaHalf, HalfFockFn{}, kBetaHalfMissing},
                                                 {HalfFockFn{}, betaHalf, kAlphaHalfMissing}};

    for (const auto& entry : incomplete)
    {
        const auto refused =
            MakeUksSeam(coulombHalf, entry.alpha, entry.beta, core, 0.25, evaluator);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_EQ(refused.error().message, entry.message);
    }

    // None of the three ran anything.
    EXPECT_EQ(coulombState->calls, 0);
    EXPECT_EQ(alphaState->calls, 0);
    EXPECT_EQ(betaState->calls, 0);
    EXPECT_EQ(evaluatorState->calls, 0);

    // The controls, and they are what separates this refusal from an
    // absence rule: the SAME missing halves at the same fraction are the
    // only change, and both halves present at c_HF > 0 is accepted; a pure
    // functional needs no exchange half at all, so the absence alone is
    // never what is refused.
    const auto complete = MakeUksSeam(coulombHalf, alphaHalf, betaHalf, core, 0.25, evaluator);
    ASSERT_TRUE(complete.has_value()) << complete.error().message;
    const auto pure = MakeUksSeam(coulombHalf, HalfFockFn{}, HalfFockFn{}, core, 0.0, evaluator);
    ASSERT_TRUE(pure.has_value()) << pure.error().message;
}

// THE SAME DEFECT ON THE UNRESTRICTED SIDE, with a wider leak, and the same
// invariant enforced: the pair fill stored BOTH spin halves before it asked
// the evaluator, so an evaluation failure left the entry keyed on the
// previous pair while both halves inside it belonged to the pair that
// failed. One stale spin would be bad enough; this was two. Measured on the
// pre-fix code at the parameters below, the two spins came back 0.0315 and
// 0.0243 away from the Fock the entry is keyed for.
//
// Leg 2 is the CONTROL that keeps leg 1 honest, and it is the reading that
// refuted the first draft of this test: a BETA failure does NOT leak the
// alpha half. The pair fill computes the two halves separately but stores
// them TOGETHER, after both calls have returned, so that failure - then and
// now - leaves the entry exactly as it was. The evaluation was therefore the
// only arm of this fill that leaked, which is what separates "the
// evaluation leaked" from any looser story about late failures.
TEST(KsCompositionTest, UksKeepsThePreviousPairIntactWhenTheEvaluationFails) {
    const Eigen::MatrixXd core = MakeTestCore();
    const Eigen::MatrixXd alpha = MakeTestDensity();
    const Eigen::MatrixXd beta = 0.6 * alpha;
    const Eigen::MatrixXd nextAlpha = 0.9 * alpha;
    const Eigen::MatrixXd nextBeta = 0.9 * beta;
    const double fraction = 0.5;

    // Leg 1: the evaluation is what fails, with both halves already stored.
    const auto coulombState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 2.0});
    const auto alphaState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.7, true});
    const auto betaState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.5, true});
    const auto evaluatorState = std::make_shared<SwitchableEvaluator>();
    evaluatorState->error = {qcx::ErrorCode::kInternalError, "the evaluator fell over"};

    const auto seam = MakeUksSeam(MakeSwitchableHalf(coulombState),
                                  MakeSwitchableHalf(alphaState),
                                  MakeSwitchableHalf(betaState),
                                  core,
                                  fraction,
                                  MakeSwitchableEvaluator(evaluatorState));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    const auto filled = seam->fock(alpha, beta);
    ASSERT_TRUE(filled.has_value()) << filled.error().message;

    evaluatorState->fail = true;
    const auto failed = seam->fock(nextAlpha, nextBeta);
    ASSERT_FALSE(failed.has_value());
    EXPECT_EQ(failed.error().code, qcx::ErrorCode::kInternalError);
    evaluatorState->fail = false;

    // The ask for the previous PAIR: the pair is a HIT, so neither spin half
    // is rebuilt. The Coulomb half is, and correctly so - it is keyed on the
    // total alone and the failed fill replaced that key with the other pair's.
    const auto stale = seam->fock(alpha, beta);
    ASSERT_TRUE(stale.has_value()) << stale.error().message;
    EXPECT_EQ(coulombState->calls, 3);
    EXPECT_EQ(alphaState->calls, 2);
    EXPECT_EQ(betaState->calls, 2);
    EXPECT_EQ(evaluatorState->calls, 2);

    const auto evaluation = EvaluateSynthetic(evaluatorState->functional, alpha, beta);
    ASSERT_TRUE(evaluation.has_value());
    const Eigen::MatrixXd coulombHalf = core + alpha + beta;
    const Eigen::MatrixXd correctAlpha =
        coulombHalf + evaluation->potentialAlpha + fraction * ((core - 0.7 * alpha) - core);
    const Eigen::MatrixXd correctBeta =
        coulombHalf + evaluation->potentialBeta + fraction * ((core - 0.5 * beta) - core);
    const Eigen::MatrixXd leakedAlpha =
        coulombHalf + evaluation->potentialAlpha + fraction * ((core - 0.7 * nextAlpha) - core);
    const Eigen::MatrixXd leakedBeta =
        coulombHalf + evaluation->potentialBeta + fraction * ((core - 0.5 * nextBeta) - core);
    EXPECT_NEAR((stale->first - correctAlpha).cwiseAbs().maxCoeff(), 0.0, 1e-13)
        << "the pair is keyed on the previous densities, so the alpha spin carries alpha";
    EXPECT_NEAR((stale->second - correctBeta).cwiseAbs().maxCoeff(), 0.0, 1e-13)
        << "and the beta spin carries beta";
    EXPECT_GT((stale->first - leakedAlpha).cwiseAbs().maxCoeff(), 1e-3)
        << "the two candidates must be distinguishable, or the pins above prove nothing";
    EXPECT_GT((stale->second - leakedBeta).cwiseAbs().maxCoeff(), 1e-3);

    // Leg 2: the BETA half is what fails. Nothing leaks - both halves are
    // stored together only after both have returned - so the previous pair's
    // Fock comes back with the numbers it is keyed for, on both spins.
    const auto coulombState2 = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 2.0});
    const auto alphaState2 = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.7, true});
    const auto betaState2 = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.5, true});
    const auto evaluatorState2 = std::make_shared<SwitchableEvaluator>();
    betaState2->error = {qcx::ErrorCode::kIOError, "the beta half failed to read"};

    const auto seam2 = MakeUksSeam(MakeSwitchableHalf(coulombState2),
                                   MakeSwitchableHalf(alphaState2),
                                   MakeSwitchableHalf(betaState2),
                                   core,
                                   fraction,
                                   MakeSwitchableEvaluator(evaluatorState2));
    ASSERT_TRUE(seam2.has_value()) << seam2.error().message;

    const auto filled2 = seam2->fock(alpha, beta);
    ASSERT_TRUE(filled2.has_value()) << filled2.error().message;

    betaState2->fail = true;
    const auto betaFailed = seam2->fock(nextAlpha, nextBeta);
    ASSERT_FALSE(betaFailed.has_value());
    EXPECT_EQ(betaFailed.error().code, qcx::ErrorCode::kIOError);
    betaState2->fail = false;

    const auto halfStale = seam2->fock(alpha, beta);
    ASSERT_TRUE(halfStale.has_value()) << halfStale.error().message;
    EXPECT_EQ(coulombState2->calls, 3);
    EXPECT_EQ(alphaState2->calls, 2);
    EXPECT_EQ(betaState2->calls, 2);
    EXPECT_NEAR((halfStale->first - correctAlpha).cwiseAbs().maxCoeff(), 0.0, 1e-13);
    EXPECT_NEAR((halfStale->second - correctBeta).cwiseAbs().maxCoeff(), 0.0, 1e-13);
    EXPECT_GT((halfStale->first - leakedAlpha).cwiseAbs().maxCoeff(), 1e-3);
}

TEST(KsCompositionTest, UksForwardsEachHalfsOwnFailure) {
    const Eigen::MatrixXd core = MakeTestCore();
    const Eigen::MatrixXd alpha = MakeTestDensity();
    const Eigen::MatrixXd beta = 0.6 * alpha;

    const auto coulombState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 2.0});
    const auto alphaState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.7, true});
    const auto betaState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.5, true});
    const auto evaluatorState = std::make_shared<SwitchableEvaluator>();
    coulombState->error = {qcx::ErrorCode::kOutOfMemory, "the Coulomb half ran out of room"};
    alphaState->error = {qcx::ErrorCode::kDeviceError, "the alpha half lost its device"};
    betaState->error = {qcx::ErrorCode::kIOError, "the beta half failed to read"};
    evaluatorState->error = {qcx::ErrorCode::kInternalError, "the evaluator fell over"};

    const auto seam = MakeUksSeam(MakeSwitchableHalf(coulombState),
                                  MakeSwitchableHalf(alphaState),
                                  MakeSwitchableHalf(betaState),
                                  core,
                                  0.5,
                                  MakeSwitchableEvaluator(evaluatorState));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    // The Coulomb half is asked first, on the spin-summed total, so its
    // failure stops the fill before either per-spin half is entered.
    coulombState->fail = true;
    const auto coulombFail = seam->fock(alpha, beta);
    ASSERT_FALSE(coulombFail.has_value());
    EXPECT_EQ(coulombFail.error().code, qcx::ErrorCode::kOutOfMemory);
    EXPECT_EQ(coulombFail.error().message, "the Coulomb half ran out of room");
    EXPECT_EQ(coulombState->calls, 1);
    EXPECT_EQ(alphaState->calls, 0);
    EXPECT_EQ(betaState->calls, 0);
    EXPECT_EQ(evaluatorState->calls, 0);

    // The alpha half fails first inside the pair: the beta half is not asked
    // at all, and the evaluation is never reached.
    coulombState->fail = false;
    alphaState->fail = true;
    const auto alphaFail = seam->fock(0.9 * alpha, 0.9 * beta);
    ASSERT_FALSE(alphaFail.has_value());
    EXPECT_EQ(alphaFail.error().code, qcx::ErrorCode::kDeviceError);
    EXPECT_EQ(alphaFail.error().message, "the alpha half lost its device");
    EXPECT_EQ(coulombState->calls, 2);
    EXPECT_EQ(alphaState->calls, 1);
    EXPECT_EQ(betaState->calls, 0);
    EXPECT_EQ(evaluatorState->calls, 0);

    // The beta half fails with the alpha half already built - the pair's own
    // mid-fill failure.
    alphaState->fail = false;
    betaState->fail = true;
    const auto betaFail = seam->fock(0.8 * alpha, 0.8 * beta);
    ASSERT_FALSE(betaFail.has_value());
    EXPECT_EQ(betaFail.error().code, qcx::ErrorCode::kIOError);
    EXPECT_EQ(betaFail.error().message, "the beta half failed to read");
    EXPECT_EQ(coulombState->calls, 3);
    EXPECT_EQ(alphaState->calls, 2);
    EXPECT_EQ(betaState->calls, 1);
    EXPECT_EQ(evaluatorState->calls, 0);

    // And the evaluator's own failure, past both spin halves.
    betaState->fail = false;
    evaluatorState->fail = true;
    const auto evaluatorFail = seam->fock(0.7 * alpha, 0.7 * beta);
    ASSERT_FALSE(evaluatorFail.has_value());
    EXPECT_EQ(evaluatorFail.error().code, qcx::ErrorCode::kInternalError);
    EXPECT_EQ(evaluatorFail.error().message, "the evaluator fell over");
    EXPECT_EQ(coulombState->calls, 4);
    EXPECT_EQ(alphaState->calls, 3);
    EXPECT_EQ(betaState->calls, 2);
    EXPECT_EQ(evaluatorState->calls, 1);
}

TEST(KsCompositionTest, UksKeepsThePairEntryIntactWhenAFillFails) {
    const Eigen::MatrixXd core = MakeTestCore();
    const double fraction = 0.5;
    const Eigen::MatrixXd alpha = MakeTestDensity();
    const Eigen::MatrixXd beta = 0.6 * alpha;
    const Eigen::MatrixXd nextAlpha = 0.9 * alpha;
    const Eigen::MatrixXd nextBeta = 0.9 * beta;

    const auto coulombState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 2.0});
    const auto alphaState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.7, true});
    const auto betaState = std::make_shared<SwitchableHalf>(SwitchableHalf{core, 0.5, true});
    const auto evaluatorState = std::make_shared<SwitchableEvaluator>();
    coulombState->error = {qcx::ErrorCode::kOutOfMemory, "the Coulomb half ran out of room"};
    alphaState->error = {qcx::ErrorCode::kDeviceError, "the alpha half lost its device"};

    const auto seam = MakeUksSeam(MakeSwitchableHalf(coulombState),
                                  MakeSwitchableHalf(alphaState),
                                  MakeSwitchableHalf(betaState),
                                  core,
                                  fraction,
                                  MakeSwitchableEvaluator(evaluatorState));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    // One good pair first, so an entry abandoned half-way would have real
    // numbers to serve as if they belonged to the pair that failed.
    const auto firstFock = seam->fock(alpha, beta);
    ASSERT_TRUE(firstFock.has_value()) << firstFock.error().message;
    EXPECT_EQ(coulombState->calls, 1);
    EXPECT_EQ(alphaState->calls, 1);
    EXPECT_EQ(betaState->calls, 1);
    EXPECT_EQ(evaluatorState->calls, 1);

    // The alpha half fails on a NEW pair. The composition writes the alpha
    // half into the entry BEFORE it asks the beta half (that is its own
    // order), so an abandoned fill has already touched the entry - which is
    // exactly why the density key is written last.
    alphaState->fail = true;
    const auto abandoned = seam->fock(nextAlpha, nextBeta);
    ASSERT_FALSE(abandoned.has_value());
    EXPECT_EQ(abandoned.error().code, qcx::ErrorCode::kDeviceError);
    EXPECT_EQ(coulombState->calls, 2);
    EXPECT_EQ(alphaState->calls, 2);
    EXPECT_EQ(betaState->calls, 1);
    EXPECT_EQ(evaluatorState->calls, 1);

    // The next ask for the SAME pair therefore rebuilds both spin halves and
    // the evaluation, and answers with that pair's numbers. A fill that had
    // written its density key before the alpha half failed would serve a HIT
    // here instead, pairing the new alpha half with the previous pair's beta
    // half - and the value below would move. The Coulomb half is NOT rebuilt:
    // it is keyed on the spin-summed total alone, and its own fill for this
    // pair's total did complete before the pair failed, which is the split
    // key behaving exactly as the cache test above pins it.
    alphaState->fail = false;
    const auto rebuilt = seam->fock(nextAlpha, nextBeta);
    ASSERT_TRUE(rebuilt.has_value()) << rebuilt.error().message;
    EXPECT_EQ(coulombState->calls, 2);
    EXPECT_EQ(alphaState->calls, 3);
    EXPECT_EQ(betaState->calls, 2);
    EXPECT_EQ(evaluatorState->calls, 2);

    const auto evaluation = EvaluateSynthetic(evaluatorState->functional, nextAlpha, nextBeta);
    ASSERT_TRUE(evaluation.has_value());
    const Eigen::MatrixXd rho = 0.5 * (nextAlpha + nextBeta);
    const Eigen::MatrixXd coulombHalf = core + 2.0 * rho;
    const Eigen::MatrixXd expectedAlpha =
        coulombHalf + evaluation->potentialAlpha + fraction * ((core - 0.7 * nextAlpha) - core);
    const Eigen::MatrixXd expectedBeta =
        coulombHalf + evaluation->potentialBeta + fraction * ((core - 0.5 * nextBeta) - core);
    EXPECT_NEAR((rebuilt->first - expectedAlpha).cwiseAbs().maxCoeff(), 0.0, 1e-13);
    EXPECT_NEAR((rebuilt->second - expectedBeta).cwiseAbs().maxCoeff(), 0.0, 1e-13);

    // A failure in the Coulomb half leaves no trace either: it is asked
    // before the pair, so the fill dies before anything of the pair has been
    // written, and the next ask for the same pair rebuilds both spin halves
    // and the evaluation rather than serving the entry that is still there.
    // (The one failing arm that DOES leave a trace - the evaluation, which
    // runs after both halves have been stored - is pinned by the leak test
    // below. The alpha and beta failures do not, which that test's second
    // leg is there to say.)
    coulombState->fail = true;
    const auto coulombFail = seam->fock(0.8 * nextAlpha, 0.8 * nextBeta);
    ASSERT_FALSE(coulombFail.has_value());
    EXPECT_EQ(coulombFail.error().code, qcx::ErrorCode::kOutOfMemory);
    EXPECT_EQ(coulombState->calls, 3);
    EXPECT_EQ(alphaState->calls, 3);
    EXPECT_EQ(betaState->calls, 2);
    EXPECT_EQ(evaluatorState->calls, 2);

    coulombState->fail = false;
    const auto laterFock = seam->fock(0.8 * nextAlpha, 0.8 * nextBeta);
    ASSERT_TRUE(laterFock.has_value()) << laterFock.error().message;
    EXPECT_EQ(coulombState->calls, 4);
    EXPECT_EQ(alphaState->calls, 4);
    EXPECT_EQ(betaState->calls, 3);
    EXPECT_EQ(evaluatorState->calls, 3);
}

// ---- the run layer: a real SCF through the composition -------------------

struct Integrals {
    qcx::basisset::BasisSet basis;
    qcx::molecule::Molecule molecule;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd core;
    JkSuper super;
    std::size_t n = 0;
};

qcx::Result<Integrals> MakeIntegrals() {
    auto basis = MakeSto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto molecule = MakeH2Sto3g();

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);

    if (!overlap.has_value() || !kinetic.has_value() || !nuclear.has_value() || !eri.has_value())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "the H2/STO-3G fixture integrals did not build"});
    }

    const std::size_t n = static_cast<std::size_t>(ToMatrix(*overlap).rows());

    return Integrals{std::move(*basis),
                     std::move(*molecule),
                     ToMatrix(*overlap),
                     ToMatrix(*kinetic) + ToMatrix(*nuclear),
                     BuildJkSuper(*eri, n),
                     n};
}

// The expected Kohn-Sham energy at one density, recomputed here:
// Tr[D H] + 1/2 Tr[D J[D]] + Exc - c_HF 1/4 Tr[D K[D]].
double ExpectedRksEnergy(const Integrals& integrals,
                         const Eigen::MatrixXd& density,
                         const SyntheticFunctional& functional,
                         double fraction) {
    const Eigen::MatrixXd rho = 0.5 * density;
    const auto xc = EvaluateSynthetic(functional, rho, rho);
    const Eigen::MatrixXd j = CoulombOf(integrals.super, density);
    const Eigen::MatrixXd k = ExchangeOf(integrals.super, density);

    return (density.cwiseProduct(integrals.core)).sum() + 0.5 * (density.cwiseProduct(j)).sum() +
           xc->energy - 0.25 * fraction * (density.cwiseProduct(k)).sum() +
           qcx::molecule::NuclearRepulsionEnergy(integrals.molecule);
}

TEST(KsCompositionTest, RksRunConvergesToTheKohnShamEnergy) {
    const auto integrals = MakeIntegrals();
    ASSERT_TRUE(integrals.has_value()) << integrals.error().message;

    const SyntheticFunctional functional{0.6};
    const double fraction = 0.3;
    const qcx::scf::RhfOptions options;

    const auto seam = MakeRksSeam(MakeCoulombHalf(integrals->super, integrals->core),
                                  MakeExchangeHalf(integrals->super, integrals->core),
                                  integrals->core,
                                  fraction,
                                  MakeSyntheticEvaluator(functional));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    const auto run = qcx::scf::RunRhfScf(integrals->molecule,
                                         integrals->overlap,
                                         integrals->core,
                                         options,
                                         seam->fock,
                                         seam->coulomb,
                                         seam->contribution);
    ASSERT_TRUE(run.has_value()) << run.error().message;
    ASSERT_TRUE(run->converged);

    // The run's own report against the formula above, at its own density.
    const double expected = ExpectedRksEnergy(*integrals, run->density, functional, fraction);
    EXPECT_NEAR(run->totalEnergy, expected, 1e-11);

    // A pure functional reaches a DIFFERENT fixed point than its hybrid -
    // otherwise the fraction would not be reaching the Fock matrix, and the
    // energy check above would pass on a run that ignored it.
    const auto pure = MakeRksSeam(MakeCoulombHalf(integrals->super, integrals->core),
                                  HalfFockFn{},
                                  integrals->core,
                                  0.0,
                                  MakeSyntheticEvaluator(functional));
    ASSERT_TRUE(pure.has_value()) << pure.error().message;
    const auto pureRun = qcx::scf::RunRhfScf(integrals->molecule,
                                             integrals->overlap,
                                             integrals->core,
                                             options,
                                             pure->fock,
                                             pure->coulomb,
                                             pure->contribution);
    ASSERT_TRUE(pureRun.has_value()) << pureRun.error().message;
    ASSERT_TRUE(pureRun->converged);
    EXPECT_NEAR(pureRun->totalEnergy,
                ExpectedRksEnergy(*integrals, pureRun->density, functional, 0.0),
                1e-11);
    EXPECT_GT(std::abs(pureRun->totalEnergy - run->totalEnergy), 1e-6);
}

TEST(KsCompositionTest, UksRunConvergesToTheKohnShamEnergy) {
    const auto integrals = MakeIntegrals();
    ASSERT_TRUE(integrals.has_value()) << integrals.error().message;

    const SyntheticFunctional functional{0.45};
    const double fraction = 0.2;
    qcx::scf::UhfOptions options;
    const int electrons = integrals->molecule.ElectronCount();
    const int nAlpha = (electrons + integrals->molecule.Multiplicity() - 1) / 2;
    const int nBeta = electrons - nAlpha;

    const auto seam = MakeUksSeam(MakeCoulombHalf(integrals->super, integrals->core),
                                  MakeExchangeHalf(integrals->super, integrals->core),
                                  MakeExchangeHalf(integrals->super, integrals->core),
                                  integrals->core,
                                  fraction,
                                  MakeSyntheticEvaluator(functional));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    const auto gwh = qcx::scf::BuildGwhGuess(integrals->overlap, integrals->core, nAlpha, nBeta);
    ASSERT_TRUE(gwh.has_value()) << gwh.error().message;
    options.initialDensityAlpha = gwh->first;
    options.initialDensityBeta = gwh->second;

    const auto run = qcx::scf::RunUhfScf(integrals->molecule,
                                         integrals->overlap,
                                         integrals->core,
                                         options,
                                         seam->fock,
                                         seam->coulomb,
                                         seam->contribution);
    ASSERT_TRUE(run.has_value()) << run.error().message;
    ASSERT_TRUE(run->converged);

    const Eigen::MatrixXd total = run->densityAlpha + run->densityBeta;
    const auto xc = EvaluateSynthetic(functional, run->densityAlpha, run->densityBeta);
    ASSERT_TRUE(xc.has_value());
    const double expected =
        (total.cwiseProduct(integrals->core)).sum() +
        0.5 * (total.cwiseProduct(CoulombOf(integrals->super, total))).sum() + xc->energy -
        0.5 * fraction *
            ((run->densityAlpha.cwiseProduct(ExchangeOf(integrals->super, run->densityAlpha)))
                 .sum() +
             (run->densityBeta.cwiseProduct(ExchangeOf(integrals->super, run->densityBeta)))
                 .sum()) +
        qcx::molecule::NuclearRepulsionEnergy(integrals->molecule);
    EXPECT_NEAR(run->totalEnergy, expected, 1e-11);
}

// ---- the real half builders ----------------------------------------------

// The composition consumes two halves whose contract it does not enforce:
// a buildCoulombOnly member must return H + 2 J(rho) and a buildExchangeOnly
// member H - K(rho), both at the SPATIAL density. Every test above writes the
// halves down IN that shape, so a real builder whose split flag meant
// something else would leave all of them green while the driver composed a
// different Fock. These two tests close that gap on the real integrals
// builders - the direct family's machinery member and its lean member, the
// two the driver actually wires - by checking their outputs against the exact
// ERI supermatrix and then running the composition over them.
using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

// A symmetric, positive-definite, non-idempotent spatial density on the
// H2/STO-3G fixture: the contract under test is algebra, not any density an
// SCF would produce, and a non-idempotent rho keeps the two halves from
// coinciding.
Eigen::MatrixXd MakeRho2() {
    Eigen::MatrixXd rho(2, 2);
    rho << 0.61, 0.24, 0.24, 0.58;
    return rho;
}

// One real builder behind a HalfFockFn: the driver's own adapter shape, with
// the Eigen <-> tensor boundary and a call through the builder's plain
// BuildFock (no stats collector, no ladder inputs).
template <typename Builder> HalfFockFn MakeRealHalf(const Builder& builder) {
    return [builder](const Eigen::MatrixXd& spatialDensity) -> qcx::Result<Eigen::MatrixXd> {
        auto tensor = ToTensor(spatialDensity);

        if (!tensor.has_value())
        {
            return std::unexpected(tensor.error());
        }

        auto fock = builder.BuildFock(*tensor);

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        return ToMatrix(*fock);
    };
}

// The convention check and the composition check, over one pair of real
// halves: the halves must reproduce H + 2J and H - K at rho exactly, and the
// seam built from them must reproduce F_KS = H + J[D] + Vxc - c K[D]/2, the
// Coulomb callback exactly J[D], and the contribution exactly
// Exc - c/4 Tr[D K[D]], all at D = 2 rho.
void CheckRealHalves(const Integrals& integrals,
                     const HalfFockFn& coulombHalf,
                     const HalfFockFn& exchangeHalf) {
    const Eigen::MatrixXd rho = MakeRho2();
    const Eigen::MatrixXd jRho = CoulombOf(integrals.super, rho);
    const Eigen::MatrixXd kRho = ExchangeOf(integrals.super, rho);

    const auto coulombHalfValue = coulombHalf(rho);
    ASSERT_TRUE(coulombHalfValue.has_value()) << coulombHalfValue.error().message;
    const auto exchangeHalfValue = exchangeHalf(rho);
    ASSERT_TRUE(exchangeHalfValue.has_value()) << exchangeHalfValue.error().message;

    EXPECT_LT((*coulombHalfValue - (integrals.core + 2.0 * jRho)).norm(), 1e-9);
    EXPECT_LT((*exchangeHalfValue - (integrals.core - kRho)).norm(), 1e-9);

    const double fraction = 0.3;
    const SyntheticFunctional functional{0.5};
    const auto seam = MakeRksSeam(
        coulombHalf, exchangeHalf, integrals.core, fraction, MakeSyntheticEvaluator(functional));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    const Eigen::MatrixXd density = 2.0 * rho;
    const auto xc = EvaluateSynthetic(functional, rho, rho);

    const auto fock = seam->fock(density);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;

    const Eigen::MatrixXd j = CoulombOf(integrals.super, density);
    const Eigen::MatrixXd k = ExchangeOf(integrals.super, density);
    const Eigen::MatrixXd expectedFock =
        integrals.core + j + 0.5 * (xc->potentialAlpha + xc->potentialBeta) - 0.5 * fraction * k;
    EXPECT_LT((*fock - expectedFock).norm(), 1e-9);

    const auto coulombMatrix = seam->coulomb(density);
    ASSERT_TRUE(coulombMatrix.has_value()) << coulombMatrix.error().message;
    EXPECT_LT((*coulombMatrix - j).norm(), 1e-9);

    const auto contribution = seam->contribution(density);
    ASSERT_TRUE(contribution.has_value()) << contribution.error().message;
    EXPECT_NEAR(
        *contribution, xc->energy - 0.25 * fraction * (density.cwiseProduct(k)).sum(), 1e-9);
}

TEST(KsCompositionTest, RealMachineryHalvesCarryTheConventionTheCompositionAssumes) {
    const auto integrals = MakeIntegrals();
    ASSERT_TRUE(integrals.has_value()) << integrals.error().message;

    auto coreTensor = ToTensor(integrals->core);
    ASSERT_TRUE(coreTensor.has_value()) << coreTensor.error().message;

    qcx::integrals::FockBuildOptions coulombOptions;
    coulombOptions.buildCoulombOnly = true;
    auto coulombBuilder = qcx::integrals::DirectJkFockBuilder::Create(
        integrals->molecule, integrals->basis, *coreTensor, coulombOptions);
    ASSERT_TRUE(coulombBuilder.has_value()) << coulombBuilder.error().message;

    qcx::integrals::FockBuildOptions exchangeOptions;
    exchangeOptions.buildExchangeOnly = true;
    auto exchangeBuilder = qcx::integrals::DirectJkFockBuilder::Create(
        integrals->molecule, integrals->basis, *coreTensor, exchangeOptions);
    ASSERT_TRUE(exchangeBuilder.has_value()) << exchangeBuilder.error().message;

    CheckRealHalves(*integrals, MakeRealHalf(*coulombBuilder), MakeRealHalf(*exchangeBuilder));
}

TEST(KsCompositionTest, RealLeanHalvesCarryTheConventionTheCompositionAssumes) {
    const auto integrals = MakeIntegrals();
    ASSERT_TRUE(integrals.has_value()) << integrals.error().message;

    auto coreTensor = ToTensor(integrals->core);
    ASSERT_TRUE(coreTensor.has_value()) << coreTensor.error().message;

    qcx::integrals::LeanFockBuildOptions coulombOptions;
    coulombOptions.buildCoulombOnly = true;
    auto coulombBuilder = qcx::integrals::LeanDirectFockBuilder::Create(
        integrals->molecule, integrals->basis, *coreTensor, coulombOptions);
    ASSERT_TRUE(coulombBuilder.has_value()) << coulombBuilder.error().message;

    qcx::integrals::LeanFockBuildOptions exchangeOptions;
    exchangeOptions.buildExchangeOnly = true;
    auto exchangeBuilder = qcx::integrals::LeanDirectFockBuilder::Create(
        integrals->molecule, integrals->basis, *coreTensor, exchangeOptions);
    ASSERT_TRUE(exchangeBuilder.has_value()) << exchangeBuilder.error().message;

    CheckRealHalves(*integrals, MakeRealHalf(*coulombBuilder), MakeRealHalf(*exchangeBuilder));
}

// ---- the RI-J link's halves ----------------------------------------------
//
// The RI-J link is the family whose Kohn-Sham composition the driver wires
// over its two split builders (run_driver.cpp MakeRiJLinkKsHalf), and its
// halves do NOT carry the direct family's H accounting: the Coulomb
// contraction returns 2 J_RI(rho) with NO core Hamiltonian, and the
// exchange-only call is the sole H carrier (H - K(rho)). The composition's
// HalfFockFn contract is the direct family's, where both halves carry one H
// each - so the Coulomb half the driver hands the composition is the
// builder's contraction PLUS H, and the exchange half is the builder's own
// result untouched. Getting that the wrong way round moves the Fock by an
// entire core Hamiltonian.
//
// The auxiliary set here is the STO-3G fixture itself. RI-J is defined for
// ANY fitting set; a coarse fit makes the reported TOTAL coarser and does not
// touch the CONVENTION these checks are about - which is why the
// self-consistency check below stays exact to 1e-10 on a fit nobody would
// ship. The fit's ACCURACY is not this file's subject and is not asserted
// anywhere here.

/// One RI-J link builder over the H2/STO-3G fixture, fitted on the fixture's
/// own set.
qcx::Result<qcx::integrals::RiJkFockBuilder> MakeRiJLinkBuilder(const Integrals& integrals,
                                                                const CpuTensor2& coreTensor) {
    return qcx::integrals::RiJkFockBuilder::Create(
        integrals.molecule, integrals.basis, integrals.basis, coreTensor);
}

/// The driver's Coulomb half, verbatim: H + 2 J_RI(rho). `coulombCalls` counts
/// the contractions, so a check can assert the composition issued exactly one
/// per density rather than trusting the cache by reading it.
HalfFockFn MakeRiJLinkCoulombHalf(const qcx::integrals::RiJkFockBuilder& builder,
                                  const Eigen::MatrixXd& coreHamiltonian,
                                  std::size_t* coulombCalls) {
    return [builder, coreHamiltonian, coulombCalls](
               const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> {
        if (coulombCalls != nullptr)
        {
            ++(*coulombCalls);
        }

        auto density = ToTensor(rho);

        if (!density.has_value())
        {
            return std::unexpected(density.error());
        }

        auto j = builder.BuildCoulombOnly(*density);

        if (!j.has_value())
        {
            return std::unexpected(j.error());
        }

        // The family's missing H, added by the caller - the ONE line that
        // separates this family's convention from the composition's.
        return coreHamiltonian + ToMatrix(*j);
    };
}

/// The driver's exchange half, verbatim: the builder's own H - K(rho), with no
/// accounting applied at all.
HalfFockFn MakeRiJLinkExchangeHalf(const qcx::integrals::RiJkFockBuilder& builder) {
    return [builder](const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> {
        auto density = ToTensor(rho);

        if (!density.has_value())
        {
            return std::unexpected(density.error());
        }

        auto half = builder.BuildExchangeOnly(*density);

        if (!half.has_value())
        {
            return std::unexpected(half.error());
        }

        return ToMatrix(*half);
    };
}

/// The builder's own Coulomb contraction at a spatial density, called from
/// OUTSIDE the seam - so a comparison against a seam value is a comparison
/// against an independent build and not against the seam's own cache.
Eigen::MatrixXd RiJCoulombOf(const qcx::integrals::RiJkFockBuilder& builder,
                             const Eigen::MatrixXd& spatialDensity) {
    auto density = ToTensor(spatialDensity);
    EXPECT_TRUE(density.has_value()) << "the spatial density did not convert to a tensor";

    if (!density.has_value())
    {
        return Eigen::MatrixXd::Zero(spatialDensity.rows(), spatialDensity.cols());
    }

    auto j = builder.BuildCoulombOnly(*density);
    EXPECT_TRUE(j.has_value()) << (j.has_value() ? "" : j.error().message);

    if (!j.has_value())
    {
        return Eigen::MatrixXd::Zero(spatialDensity.rows(), spatialDensity.cols());
    }

    return ToMatrix(*j);
}

/// The builder's own exchange half at a spatial density (H - K(rho)), from
/// outside the seam.
Eigen::MatrixXd RiJExchangeHalfOf(const qcx::integrals::RiJkFockBuilder& builder,
                                  const Eigen::MatrixXd& spatialDensity) {
    auto density = ToTensor(spatialDensity);
    EXPECT_TRUE(density.has_value()) << "the spatial density did not convert to a tensor";

    if (!density.has_value())
    {
        return Eigen::MatrixXd::Zero(spatialDensity.rows(), spatialDensity.cols());
    }

    auto half = builder.BuildExchangeOnly(*density);
    EXPECT_TRUE(half.has_value()) << (half.has_value() ? "" : half.error().message);

    if (!half.has_value())
    {
        return Eigen::MatrixXd::Zero(spatialDensity.rows(), spatialDensity.cols());
    }

    return ToMatrix(*half);
}

TEST(KsCompositionTest, RiJLinkHalvesCarryTheAccountingTheCompositionDoesNotExpect) {
    const auto integrals = MakeIntegrals();
    ASSERT_TRUE(integrals.has_value()) << integrals.error().message;

    auto coreTensor = ToTensor(integrals->core);
    ASSERT_TRUE(coreTensor.has_value()) << coreTensor.error().message;

    auto builder = MakeRiJLinkBuilder(*integrals, *coreTensor);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const Eigen::MatrixXd rho = MakeRho2();

    // The family's own contraction, before any half is formed: it carries no
    // core Hamiltonian, which is the whole reason the Coulomb half below has
    // one added. Stated as a NUMBER rather than as prose - the norm of H here
    // is what a wiring that skipped the addition would be wrong by.
    const Eigen::MatrixXd rawCoulomb = RiJCoulombOf(*builder, rho);
    EXPECT_GT(rawCoulomb.norm(), 1e-9);

    const auto coulombHalf = MakeRiJLinkCoulombHalf(*builder, integrals->core, nullptr)(rho);
    ASSERT_TRUE(coulombHalf.has_value()) << coulombHalf.error().message;
    EXPECT_LT((*coulombHalf - (integrals->core + rawCoulomb)).norm(), 1e-12)
        << "the driver's Coulomb half is the contraction plus H";

    // The other direction, as the failure it catches: this half WITHOUT the
    // added H differs from it by exactly |H| - which is why the addition is
    // not a detail. Stated as the norm of H itself, so the assertion carries
    // the size of the mistake it excludes rather than a bare threshold.
    EXPECT_NEAR((*coulombHalf - rawCoulomb).norm(), integrals->core.norm(), 1e-9)
        << "the added core Hamiltonian must be present: a half without it differs from this "
           "one by |H|, not by rounding";

    // The exchange half needs no accounting at all. A second build at the same
    // density is deterministic, so agreement here is exact up to the
    // summation's own reproducibility rather than up to a modelling
    // tolerance - and the WRONG accounting (the core Hamiltonian added a
    // second time, the shape a caller gets by treating the two halves alike)
    // is |H| away from it, which is the number asserted below.
    const Eigen::MatrixXd exchangeOfRho = RiJExchangeHalfOf(*builder, rho);
    const auto exchangeHalf = MakeRiJLinkExchangeHalf(*builder)(rho);
    ASSERT_TRUE(exchangeHalf.has_value()) << exchangeHalf.error().message;
    EXPECT_LT((*exchangeHalf - exchangeOfRho).norm(), 1e-12);
    EXPECT_NEAR(
        (*exchangeHalf - (integrals->core + exchangeOfRho)).norm(), integrals->core.norm(), 1e-9)
        << "the exchange half carries exactly one H: a second one would show up as |H| here";
}

TEST(KsCompositionTest, RiJLinkRksSeamConsumesTheBuildersOwnJAtTheSameDensity) {
    const auto integrals = MakeIntegrals();
    ASSERT_TRUE(integrals.has_value()) << integrals.error().message;

    auto coreTensor = ToTensor(integrals->core);
    ASSERT_TRUE(coreTensor.has_value()) << coreTensor.error().message;

    auto builder = MakeRiJLinkBuilder(*integrals, *coreTensor);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    std::size_t coulombCalls = 0;
    const double fraction = 0.3;
    const SyntheticFunctional functional{0.5};

    const auto seam = MakeRksSeam(MakeRiJLinkCoulombHalf(*builder, integrals->core, &coulombCalls),
                                  MakeRiJLinkExchangeHalf(*builder),
                                  integrals->core,
                                  fraction,
                                  MakeSyntheticEvaluator(functional));
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    const auto xc = EvaluateSynthetic(functional, MakeRho2(), MakeRho2());
    ASSERT_TRUE(xc.has_value()) << xc.error().message;

    // The density the loop asks about, and the operator the builder makes of
    // it - held in ONE place, so every expectation below is formed from the
    // same pair the seam was asked about.
    const Eigen::MatrixXd rho = MakeRho2();
    const Eigen::MatrixXd density = 2.0 * rho;
    const Eigen::MatrixXd jOfRho = RiJCoulombOf(*builder, rho);
    const Eigen::MatrixXd exchangeOfRho = RiJExchangeHalfOf(*builder, rho);
    // The composition's restricted chain rule: the engine returns dE/dD_s, so
    // the restricted potential is HALF the spin-sum.
    const Eigen::MatrixXd xcPotential = 0.5 * (xc->potentialAlpha + xc->potentialBeta);

    // The three questions, asked in the order the SCF loop asks them (the
    // Fock first, then the Coulomb matrix and the contribution) and all at ONE
    // density.
    const auto fock = seam->fock(density);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;
    const auto coulomb = seam->coulomb(density);
    ASSERT_TRUE(coulomb.has_value()) << coulomb.error().message;
    const auto contribution = seam->contribution(density);
    ASSERT_TRUE(contribution.has_value()) << contribution.error().message;

    // THE SEAM AGREEMENT. The composition contractually builds the Coulomb
    // half once per density, and the counter is what proves this rather than
    // the cache's docstring: three callbacks, one contraction. That is the
    // structural half of "the J the seam consumes is the J the Fock used".
    EXPECT_EQ(coulombCalls, 1u)
        << "three callbacks at one density must issue exactly one Coulomb build";

    // The numerical half, both directions:
    //   - no H survives in the number the energy formula contracts (the
    //     composition subtracts back out exactly the object the half added);
    //   - no K is folded into it (the exchange calls are separate callables
    //     and neither reaches this one).
    // Both failure modes are O(1) against this bound: a surviving H is |H| and
    // a folded-in K is |K|, while the residual here is the round trip through
    // one matrix addition.
    EXPECT_LT((*coulomb - jOfRho).norm(), 1e-12);
    EXPECT_GT((*coulomb - (jOfRho - integrals->core)).norm(), 1.0)
        << "the H the Coulomb half added must be gone from the matrix the energy contracts";

    // And the Fock WAS formed from that same J: H + J_RI[D] + Vxc, with the
    // exact-exchange correction subtracted out of the exchange half the
    // composition already holds rather than added as a second K build.
    const Eigen::MatrixXd expectedFock =
        integrals->core + jOfRho + xcPotential - fraction * (integrals->core - exchangeOfRho);
    EXPECT_LT((*fock - expectedFock).norm(), 1e-12);

    // The contribution is Exc - c_HF/2 Tr[D (H - exchangeHalf)], the
    // two-halves form of -c_HF 1/4 Tr[D K[D]].
    EXPECT_NEAR(*contribution,
                xc->energy -
                    0.5 * fraction * (density.cwiseProduct(integrals->core - exchangeOfRho)).sum(),
                1e-12);
}

TEST(KsCompositionTest, RiJLinkRksRunThroughTheRealEngineIsSelfConsistent) {
    const auto integrals = MakeIntegrals();
    ASSERT_TRUE(integrals.has_value()) << integrals.error().message;

    auto coreTensor = ToTensor(integrals->core);
    ASSERT_TRUE(coreTensor.has_value()) << coreTensor.error().message;

    auto builder = MakeRiJLinkBuilder(*integrals, *coreTensor);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    auto engine = qcx::grid::XcGridEngine::Create(integrals->molecule, integrals->basis, "slater");
    ASSERT_TRUE(engine.has_value()) << engine.error().message;
    EXPECT_DOUBLE_EQ(engine->ExchangeFraction(), 0.0);

    const XcEvaluatorFn evaluator =
        [&engine](const Eigen::MatrixXd& densityAlpha,
                  const Eigen::MatrixXd& densityBeta) -> qcx::Result<qcx::grid::XcEvaluation> {
        return engine->EvaluateScreened(densityAlpha, densityBeta, 0.0);
    };

    const auto seam = MakeRksSeam(MakeRiJLinkCoulombHalf(*builder, integrals->core, nullptr),
                                  HalfFockFn{},
                                  integrals->core,
                                  0.0,
                                  evaluator);
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    const qcx::scf::RhfOptions options;
    const auto run = qcx::scf::RunRhfScf(integrals->molecule,
                                         integrals->overlap,
                                         integrals->core,
                                         options,
                                         seam->fock,
                                         seam->coulomb,
                                         seam->contribution);
    ASSERT_TRUE(run.has_value()) << run.error().message;
    ASSERT_TRUE(run->converged);

    // The loop's own number, recomputed from the density it returned with a
    // FRESH contraction - the builder is asked again, outside the seam and
    // outside its cache:
    //
    //     E = Tr[D H] + 1/2 Tr[D J_RI[D]] + Exc(D) + E_nuc.
    //
    // This is the check the acceptance asks for, one level up from the
    // single-density one above: if the loop's energy seam had consumed a
    // Coulomb matrix that is not the Fock's - built at another density, or
    // still carrying the added H, or with K folded in - this identity would
    // fail by O(1) while every internal check kept passing.
    const Eigen::MatrixXd rho = 0.5 * run->density;
    const Eigen::MatrixXd jOfRun = RiJCoulombOf(*builder, rho);
    const auto xc = engine->Evaluate(rho, rho);
    ASSERT_TRUE(xc.has_value()) << xc.error().message;

    const double expected = (run->density.cwiseProduct(integrals->core)).sum() +
                            0.5 * (run->density.cwiseProduct(jOfRun)).sum() + xc->energy +
                            qcx::molecule::NuclearRepulsionEnergy(integrals->molecule);
    EXPECT_NEAR(run->totalEnergy, expected, 1e-10);

    std::printf("RKS slater/ri_j_link H2/STO-3G (STO-3G aux): total = %.12f Ha, "
                "electronic = %.12f, Exc = %.12f, iterations = %d\n",
                run->totalEnergy,
                run->electronicEnergy,
                xc->energy,
                run->iterations);
    EXPECT_GT(run->totalEnergy, -2.0);
    EXPECT_LT(run->totalEnergy, 0.0);
}

// ---- the real grid engine ------------------------------------------------
TEST(KsCompositionTest, SlaterRksRunThroughTheRealEngineIsSelfConsistent) {
    const auto integrals = MakeIntegrals();
    ASSERT_TRUE(integrals.has_value()) << integrals.error().message;

    auto engine = qcx::grid::XcGridEngine::Create(integrals->molecule, integrals->basis, "slater");
    ASSERT_TRUE(engine.has_value()) << engine.error().message;
    ASSERT_EQ(engine->AOCount(), integrals->n);
    EXPECT_DOUBLE_EQ(engine->ExchangeFraction(), 0.0);
    EXPECT_FALSE(engine->UsesGradient());

    // The production evaluator's shape: the screened two-spin entry point.
    // The composition calls it with (rho, rho) for a closed-shell run, which
    // is EvaluateClosedShellScreened's own D/2 split by construction.
    const XcEvaluatorFn evaluator =
        [&engine](const Eigen::MatrixXd& densityAlpha,
                  const Eigen::MatrixXd& densityBeta) -> qcx::Result<qcx::grid::XcEvaluation> {
        return engine->EvaluateScreened(densityAlpha, densityBeta, 0.0);
    };

    const auto seam = MakeRksSeam(MakeCoulombHalf(integrals->super, integrals->core),
                                  HalfFockFn{},
                                  integrals->core,
                                  0.0,
                                  evaluator);
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    const qcx::scf::RhfOptions options;
    const auto run = qcx::scf::RunRhfScf(integrals->molecule,
                                         integrals->overlap,
                                         integrals->core,
                                         options,
                                         seam->fock,
                                         seam->coulomb,
                                         seam->contribution);
    ASSERT_TRUE(run.has_value()) << run.error().message;

    // Whatever the walk did, the number reported must be the Kohn-Sham
    // energy of the density it returned - the engine's own Exc and Vxc at
    // that density, on top of the exact integrals.
    const Eigen::MatrixXd rho = 0.5 * run->density;
    const auto xc = engine->Evaluate(rho, rho);
    ASSERT_TRUE(xc.has_value()) << xc.error().message;
    const double expected =
        (run->density.cwiseProduct(integrals->core)).sum() +
        0.5 * (run->density.cwiseProduct(CoulombOf(integrals->super, run->density))).sum() +
        xc->energy + qcx::molecule::NuclearRepulsionEnergy(integrals->molecule);
    EXPECT_NEAR(run->totalEnergy, expected, 1e-10);

    // The converged RKS energy of H2/STO-3G with Slater exchange. A PIN, so
    // a change to the composition, the grid or the walk shows up as a
    // number rather than as a silently different run; it is not a
    // cross-package validation (no external reference computed it).
    std::printf("RKS slater H2/STO-3G: total = %.12f Ha, iterations = %d, converged = %d\n",
                run->totalEnergy,
                run->iterations,
                run->converged ? 1 : 0);
    std::printf("  electronic = %.12f, Exc = %.12f, rho_tr = %.12f\n",
                run->electronicEnergy,
                xc->energy,
                rho.trace());
    EXPECT_GT(run->totalEnergy, -2.0);
    EXPECT_LT(run->totalEnergy, 0.0);
}

TEST(KsCompositionTest, SlaterUksRunThroughTheRealEngineIsSelfConsistent) {
    const auto integrals = MakeIntegrals();
    ASSERT_TRUE(integrals.has_value()) << integrals.error().message;

    auto engine = qcx::grid::XcGridEngine::Create(integrals->molecule, integrals->basis, "slater");
    ASSERT_TRUE(engine.has_value()) << engine.error().message;

    const XcEvaluatorFn evaluator =
        [&engine](const Eigen::MatrixXd& densityAlpha,
                  const Eigen::MatrixXd& densityBeta) -> qcx::Result<qcx::grid::XcEvaluation> {
        return engine->EvaluateScreened(densityAlpha, densityBeta, 0.0);
    };

    const auto seam = MakeUksSeam(MakeCoulombHalf(integrals->super, integrals->core),
                                  HalfFockFn{},
                                  HalfFockFn{},
                                  integrals->core,
                                  0.0,
                                  evaluator);
    ASSERT_TRUE(seam.has_value()) << seam.error().message;

    const int electrons = integrals->molecule.ElectronCount();
    const int nAlpha = (electrons + integrals->molecule.Multiplicity() - 1) / 2;
    const int nBeta = electrons - nAlpha;

    qcx::scf::UhfOptions options;
    const auto gwh = qcx::scf::BuildGwhGuess(integrals->overlap, integrals->core, nAlpha, nBeta);
    ASSERT_TRUE(gwh.has_value()) << gwh.error().message;
    options.initialDensityAlpha = gwh->first;
    options.initialDensityBeta = gwh->second;

    const auto run = qcx::scf::RunUhfScf(integrals->molecule,
                                         integrals->overlap,
                                         integrals->core,
                                         options,
                                         seam->fock,
                                         seam->coulomb,
                                         seam->contribution);
    ASSERT_TRUE(run.has_value()) << run.error().message;

    const Eigen::MatrixXd total = run->densityAlpha + run->densityBeta;
    const auto xc = engine->Evaluate(run->densityAlpha, run->densityBeta);
    ASSERT_TRUE(xc.has_value()) << xc.error().message;
    const double expected = (total.cwiseProduct(integrals->core)).sum() +
                            0.5 * (total.cwiseProduct(CoulombOf(integrals->super, total))).sum() +
                            xc->energy + qcx::molecule::NuclearRepulsionEnergy(integrals->molecule);
    EXPECT_NEAR(run->totalEnergy, expected, 1e-10);

    std::printf("UKS slater H2/STO-3G: total = %.12f Ha, iterations = %d, converged = %d\n",
                run->totalEnergy,
                run->iterations,
                run->converged ? 1 : 0);
    EXPECT_TRUE(std::isfinite(run->totalEnergy));
    EXPECT_LT(run->totalEnergy, 0.0);
}

} // namespace
