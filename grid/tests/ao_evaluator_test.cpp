// AO-value-at-point: the evaluator must reproduce the closed-form value
// of every basis function, and the module's unit-norm renormalization of
// spherical contractions must be honored (int phi^2 over the molecular
// grid equals 1 for s, p, ... shells).

#include "h2_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/grid/molecular_grid.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <string_view>
#include <vector>

namespace qcx::grid {
namespace {

constexpr double kPi = 3.14159265358979323846;

// The evaluator's l = 0 normalization (ao_evaluator.cpp RadialNormalization,
// reimplemented for the reference): N_0(zeta) = (2 zeta/pi)^(3/4).
double SOrbitalNormalization(double exponent) {
    return std::pow(2.0 * exponent / kPi, 0.75);
}

// The evaluator's convention, reimplemented for the reference:
// phi = sum_i d_i N_l(zeta_i) e^{-zeta_i r^2} times the solid-harmonic
// factor, with N_l the integrals module's per-primitive normalization
// (unit self-overlap for every l).
// (l, r2) is the angular-momentum + squared-radius pair.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double RadialReference(int l,
                       double r2,
                       const std::vector<double>& exponents,
                       const std::vector<double>& coefficients) {
    double doubleFactorial = 1.0;

    for (int k = 3; k <= 2 * l - 1; k += 2)
    {
        doubleFactorial *= static_cast<double>(k);
    }

    double value = 0.0;

    for (std::size_t i = 0; i < exponents.size(); ++i)
    {
        const double norm = std::pow(2.0, l + 1.25) * std::pow(exponents[i], 0.5 * l + 0.75) /
                            std::sqrt(2.0 * std::pow(kPi, 1.5) * doubleFactorial);
        value += coefficients[i] * norm * std::exp(-exponents[i] * r2);
    }

    return value;
}

constexpr std::string_view kSto3gH = R"(
BASIS "ao basis" SPHERICAL PRINT
H S
3.42525091 0.15432897
0.62391373 0.53532814
0.16885540 0.44463454
END
)";

// One hydrogen at the origin (fixtures only offer H2/He).
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

TEST(AoEvaluatorTest, STypeMatchesClosedForm) {
    const auto basis = qcx::basisset::ParseNwchemText(kSto3gH);
    ASSERT_TRUE(basis.has_value());
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());
    EXPECT_EQ(evaluator->AOCount(), 1u);

    const auto* element = basis->Find(1);
    ASSERT_NE(element, nullptr);
    const auto& exponents = element->shells[0].exponents;
    const auto& coefficients = element->shells[0].coefficients[0];

    std::array<double, 1> values{};
    const std::array<std::array<double, 3>, 3> points = {{
        {0.3, -0.2, 0.1},
        {1.7, 0.9, -0.5},
        {0.0, 0.0, 2.0},
    }};

    for (const auto& point : points)
    {
        evaluator->Evaluate(point, values);
        const double r2 = point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
        EXPECT_NEAR(values[0], RadialReference(0, r2, exponents, coefficients), 1e-12)
            << "at " << point[0] << " " << point[1] << " " << point[2];
    }
}

TEST(AoEvaluatorTest, STypeGridNormIsOne) {
    // The spherical renormalization on parse makes int phi^2 = 1.
    const auto basis = qcx::basisset::ParseNwchemText(kSto3gH);
    ASSERT_TRUE(basis.has_value());
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());
    const auto grid = MolecularGrid::Create(*molecule, 50, 50);
    ASSERT_TRUE(grid.has_value());

    std::array<double, 1> values{};
    double norm = 0.0;

    for (std::size_t i = 0; i < grid->Size(); ++i)
    {
        evaluator->Evaluate(grid->Point(i), values);
        norm += grid->Weight(i) * values[0] * values[0];
    }

    EXPECT_NEAR(norm, 1.0, 1e-3);
}

constexpr std::string_view kP1 = R"(
BASIS "ao basis" SPHERICAL PRINT
H P
1.0 1.0
END
)";

TEST(AoEvaluatorTest, PShellMatchesClosedForm) {
    const auto basis = qcx::basisset::ParseNwchemText(kP1);
    ASSERT_TRUE(basis.has_value());
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());
    EXPECT_EQ(evaluator->AOCount(), 3u);

    // Stored order (m = -1, 0, +1): py, pz, px.
    const auto* element = basis->Find(1);
    ASSERT_NE(element, nullptr);
    const auto& exponents = element->shells[0].exponents;
    const auto& coefficients = element->shells[0].coefficients[0];

    // On the z axis px = py = 0 and pz = z * radial.
    std::array<double, 3> values{};
    const double d = 1.3;
    evaluator->Evaluate({0.0, 0.0, d}, values);
    const double radial = RadialReference(1, d * d, exponents, coefficients);
    EXPECT_NEAR(values[0], 0.0, 1e-15);
    EXPECT_NEAR(values[1], d * radial, 1e-12);
    EXPECT_NEAR(values[2], 0.0, 1e-15);

    // Off axis the Cartesian factors hold: px/py = x/y exactly, and pz
    // vanishes in the z = 0 plane.
    evaluator->Evaluate({2.0, 3.0, 0.0}, values);
    EXPECT_NEAR(values[2] / values[0], 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(values[1], 0.0, 1e-15);
}

TEST(AoEvaluatorTest, PGridNormIsOne) {
    // The stored coefficients are renormalized so that int phi^2 = 1 for
    // every p function.
    const auto basis = qcx::basisset::ParseNwchemText(kP1);
    ASSERT_TRUE(basis.has_value());
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());
    const auto grid = MolecularGrid::Create(*molecule, 60, 50);
    ASSERT_TRUE(grid.has_value());

    std::array<double, 3> values{};
    std::array<double, 3> norms{};

    for (std::size_t i = 0; i < grid->Size(); ++i)
    {
        evaluator->Evaluate(grid->Point(i), values);

        for (std::size_t m = 0; m < 3; ++m)
        {
            norms[m] += grid->Weight(i) * values[m] * values[m];
        }
    }

    for (const double norm : norms)
    {
        EXPECT_NEAR(norm, 1.0, 5e-3);
    }
}

TEST(AoEvaluatorTest, DerivativesMatchClosedFormForSPrimitive) {
    // The closed-form pin of the derivative path: one s contraction on a
    // hydrogen at the origin, evaluated at an off-origin point with all
    // coordinates nonzero.  S00 = 1, so phi = R, grad phi = grad R =
    // -2 sum_i d_i N_0(zeta_i) zeta_i e^{-zeta_i r2} Delta, and
    // (H_phi)_jk = sum_i d_i N_0(zeta_i) e^{-zeta_i r2}
    // (4 zeta_i^2 Delta_j Delta_k - 2 zeta_i delta_jk) - the l = 0 case
    // of the evaluator's radial closed forms.
    const auto basis = qcx::basisset::ParseNwchemText(kSto3gH);
    ASSERT_TRUE(basis.has_value());
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());
    ASSERT_EQ(evaluator->AOCount(), 1u);

    const auto* element = basis->Find(1);
    ASSERT_NE(element, nullptr);
    const auto& exponents = element->shells[0].exponents;
    const auto& coefficients = element->shells[0].coefficients[0];

    const std::array<double, 3> point = {0.3, -0.2, 0.1};
    const double r2 = point[0] * point[0] + point[1] * point[1] + point[2] * point[2];

    std::array<double, 1> values{};
    std::array<double, 3> gradients{};
    std::array<double, 6> hessians{};
    evaluator->EvaluateDerivatives(point, values, gradients, hessians);

    EXPECT_NEAR(values[0], RadialReference(0, r2, exponents, coefficients), 1e-10);

    double gradX = 0.0;
    double gradY = 0.0;
    double gradZ = 0.0;
    double hessXX = 0.0;
    double hessXY = 0.0;
    double hessXZ = 0.0;
    double hessYY = 0.0;
    double hessYZ = 0.0;
    double hessZZ = 0.0;

    for (std::size_t i = 0; i < exponents.size(); ++i)
    {
        const double factor =
            coefficients[i] * SOrbitalNormalization(exponents[i]) * std::exp(-exponents[i] * r2);
        const double twoZ = 2.0 * exponents[i];
        const double fourZ2 = 4.0 * exponents[i] * exponents[i];
        gradX += -twoZ * point[0] * factor;
        gradY += -twoZ * point[1] * factor;
        gradZ += -twoZ * point[2] * factor;
        hessXX += factor * (fourZ2 * point[0] * point[0] - twoZ);
        hessXY += factor * fourZ2 * point[0] * point[1];
        hessXZ += factor * fourZ2 * point[0] * point[2];
        hessYY += factor * (fourZ2 * point[1] * point[1] - twoZ);
        hessYZ += factor * fourZ2 * point[1] * point[2];
        hessZZ += factor * (fourZ2 * point[2] * point[2] - twoZ);
    }

    EXPECT_NEAR(gradients[0], gradX, 1e-10);
    EXPECT_NEAR(gradients[1], gradY, 1e-10);
    EXPECT_NEAR(gradients[2], gradZ, 1e-10);
    EXPECT_NEAR(hessians[0], hessXX, 1e-10);
    EXPECT_NEAR(hessians[1], hessXY, 1e-10);
    EXPECT_NEAR(hessians[2], hessXZ, 1e-10);
    EXPECT_NEAR(hessians[3], hessYY, 1e-10);
    EXPECT_NEAR(hessians[4], hessYZ, 1e-10);
    EXPECT_NEAR(hessians[5], hessZZ, 1e-10);
}

constexpr std::string_view kP1Cartesian = R"(
BASIS "ao basis" CARTESIAN PRINT
H P
1.0 1.0
END
)";

TEST(AoEvaluatorTest, DerivativesMatchFiniteDifferences) {
    // Central differences of the value path vs the analytic gradient and
    // Hessian, at the closed-form pin's point.  The three cases cover the
    // radial-only s path, the spherical solid-harmonic derivative terms
    // (p shell), and the Cartesian monomial path (no solid-harmonic
    // factor).  h = 1e-5 bohr; the tolerance is the central-difference
    // noise floor of the Hessian at that step (the f(x+h) - 2f(x) + f(x-h)
    // cancellation, eps * |f| / h^2 for O(1) values).  Measured residual
    // 2026-08-29: max |analytic - FD| = 9.1e-7 on the xx entry of the s
    // shell (gradients agree to < 1e-9), so 2e-6 carries a comfortable
    // margin over the floor.
    constexpr double kFiniteDifferenceStep = 1e-5;
    constexpr double kTolerance = 2e-6;
    const std::array<double, 3> point = {0.3, -0.2, 0.1};
    const std::array<std::string_view, 3> bases = {kSto3gH, kP1, kP1Cartesian};

    for (const std::string_view basisText : bases)
    {
        const auto basis = qcx::basisset::ParseNwchemText(basisText);
        ASSERT_TRUE(basis.has_value());
        const auto molecule = MakeSingleHydrogen();
        ASSERT_TRUE(molecule.has_value());

        const auto evaluator = AoEvaluator::Create(*molecule, *basis);
        ASSERT_TRUE(evaluator.has_value());
        const std::size_t count = evaluator->AOCount();

        std::vector<double> values(count);
        std::vector<double> gradients(3 * count);
        std::vector<double> hessians(6 * count);
        evaluator->EvaluateDerivatives(point, values, gradients, hessians);

        // The value path agrees with Evaluate by construction (the same
        // angular * radial expression).
        std::vector<double> valueReference(count);
        evaluator->Evaluate(point, valueReference);

        for (std::size_t mu = 0; mu < count; ++mu)
        {
            EXPECT_NEAR(values[mu], valueReference[mu], 1e-12);
        }

        // Central differences per component.
        const auto evaluateAt = [&](const std::array<double, 3>& shifted, std::size_t mu) {
            std::vector<double> f(count);
            evaluator->Evaluate(shifted, f);
            return f[mu];
        };

        for (std::size_t mu = 0; mu < count; ++mu)
        {
            for (std::size_t j = 0; j < 3; ++j)
            {
                std::array<double, 3> plus = point;
                std::array<double, 3> minus = point;
                plus[j] += kFiniteDifferenceStep;
                minus[j] -= kFiniteDifferenceStep;
                const double derivative =
                    (evaluateAt(plus, mu) - evaluateAt(minus, mu)) / (2.0 * kFiniteDifferenceStep);
                EXPECT_NEAR(gradients[3 * mu + j], derivative, kTolerance)
                    << "basis " << basisText << " gradient mu=" << mu << " j=" << j;
            }

            for (std::size_t j = 0; j < 3; ++j)
            {
                for (std::size_t k = j; k < 3; ++k)
                {
                    double secondDerivative = 0.0;

                    if (j == k)
                    {
                        std::array<double, 3> plus = point;
                        std::array<double, 3> minus = point;
                        plus[j] += kFiniteDifferenceStep;
                        minus[j] -= kFiniteDifferenceStep;
                        secondDerivative = (evaluateAt(plus, mu) - 2.0 * valueReference[mu] +
                                            evaluateAt(minus, mu)) /
                                           (kFiniteDifferenceStep * kFiniteDifferenceStep);
                    } else
                    {
                        std::array<double, 3> pp = point;
                        std::array<double, 3> pm = point;
                        std::array<double, 3> mp = point;
                        std::array<double, 3> mm = point;
                        pp[j] += kFiniteDifferenceStep;
                        pp[k] += kFiniteDifferenceStep;
                        pm[j] += kFiniteDifferenceStep;
                        pm[k] -= kFiniteDifferenceStep;
                        mp[j] -= kFiniteDifferenceStep;
                        mp[k] += kFiniteDifferenceStep;
                        mm[j] -= kFiniteDifferenceStep;
                        mm[k] -= kFiniteDifferenceStep;
                        secondDerivative = (evaluateAt(pp, mu) - evaluateAt(pm, mu) -
                                            evaluateAt(mp, mu) + evaluateAt(mm, mu)) /
                                           (4.0 * kFiniteDifferenceStep * kFiniteDifferenceStep);
                    }

                    // The packed (xx, xy, xz, yy, yz, zz) entry of (j, k).
                    const std::size_t hessianIndex = j == 0 ? k : (j == 1 ? k + 2 : 5);
                    EXPECT_NEAR(hessians[6 * mu + hessianIndex], secondDerivative, kTolerance)
                        << "basis " << basisText << " hessian mu=" << mu << " j=" << j
                        << " k=" << k;
                }
            }
        }
    }
}

TEST(AoEvaluatorTest, AOCountMatchesShellExpansion) {
    // Two hydrogens, one s shell each: 2 AOs.  Add a p shell on H: 3 more.
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());

    const auto basis = qcx::basisset::ParseNwchemText(kSto3gH);
    ASSERT_TRUE(basis.has_value());
    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());
    EXPECT_EQ(evaluator->AOCount(), 2u);

    // One BASIS block with both shell lines on H.
    constexpr std::string_view kSto3gHp = R"(
BASIS "ao basis" SPHERICAL PRINT
H S
3.42525091 0.15432897
0.62391373 0.53532814
0.16885540 0.44463454
H P
1.0 1.0
END
)";
    const auto merged = qcx::basisset::ParseNwchemText(kSto3gHp);
    ASSERT_TRUE(merged.has_value());
    const auto evaluator2 = AoEvaluator::Create(*molecule, *merged);
    ASSERT_TRUE(evaluator2.has_value());
    // Two atoms, each with an s and a p shell.
    EXPECT_EQ(evaluator2->AOCount(), 2u * (1u + 3u));
}

TEST(AoEvaluatorTest, CartesianShellUsesMonomials) {
    // Cartesian shells keep raw coefficients: phi = (2 zeta / pi)^(3/4)
    // x^a y^b z^c e^{-zeta r^2}.
    constexpr std::string_view kDcart = R"(
BASIS "ao basis" CARTESIAN PRINT
H D
1.0 1.0
END
)";
    const auto basis = qcx::basisset::ParseNwchemText(kDcart);
    ASSERT_TRUE(basis.has_value());
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());
    EXPECT_EQ(evaluator->AOCount(), 6u);

    // (a,b,c) order, a descending then b descending: xx, xy, xz, yy, yz, zz.
    // At (0,0,1) only zz survives, with value N * 1^2 * e^{-1}.
    std::array<double, 6> values{};
    evaluator->Evaluate({0.0, 0.0, 1.0}, values);
    const double expected = std::pow(2.0 / kPi, 0.75) * std::exp(-1.0);
    EXPECT_NEAR(values[5], expected, 1e-12);

    for (std::size_t i = 0; i < 5; ++i)
    {
        EXPECT_NEAR(values[i], 0.0, 1e-15) << "index " << i;
    }
}

// Single-primitive G/H/I shells (l = 4..6): the evaluator covers every
// angular momentum the basis parser accepts (up to I).  The solid
// harmonics are homogeneous of degree l, so on the z axis only the
// m = 0 row survives and equals z^l; the off-axis pins are exact values
// of the generated rows, whose x^l coefficients reduce to closed forms
// (e.g. S62's sqrt(105/512)).
constexpr std::string_view kG1 = R"(
BASIS "ao basis" SPHERICAL PRINT
H G
1.0 1.0
END
)";

constexpr std::string_view kH1 = R"(
BASIS "ao basis" SPHERICAL PRINT
H H
1.0 1.0
END
)";

constexpr std::string_view kI1 = R"(
BASIS "ao basis" SPHERICAL PRINT
H I
1.0 1.0
END
)";

TEST(AoEvaluatorTest, GShellMatchesClosedForm) {
    const auto basis = qcx::basisset::ParseNwchemText(kG1);
    ASSERT_TRUE(basis.has_value());
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());
    EXPECT_EQ(evaluator->AOCount(), 9u);

    const auto* element = basis->Find(1);
    ASSERT_NE(element, nullptr);
    const auto& exponents = element->shells[0].exponents;
    const auto& coefficients = element->shells[0].coefficients[0];

    // z axis: m = 0 (index 4) equals z^4 times the radial part; every
    // other row carries an x or y factor and vanishes.
    std::array<double, 9> values{};
    const double d = 1.3;
    evaluator->Evaluate({0.0, 0.0, d}, values);
    const double radialZ = RadialReference(4, d * d, exponents, coefficients);

    for (std::size_t m = 0; m < 9; ++m)
    {
        if (m == 4)
        {
            EXPECT_NEAR(values[m], d * d * d * d * radialZ, 1e-12);
        } else
        {
            EXPECT_NEAR(values[m], 0.0, 1e-15) << "m index " << m;
        }
    }

    // (1, 1, 0): rows m = -2, 0, +4 survive with -sqrt(5), 3/2 and
    // -sqrt(35)/2 times the radial part; all other rows vanish.
    evaluator->Evaluate({1.0, 1.0, 0.0}, values);
    const double radialOff = RadialReference(4, 2.0, exponents, coefficients);
    EXPECT_NEAR(values[0], 0.0, 1e-15);
    EXPECT_NEAR(values[1], 0.0, 1e-15);
    EXPECT_NEAR(values[2], -std::sqrt(5.0) * radialOff, 1e-12);
    EXPECT_NEAR(values[3], 0.0, 1e-15);
    EXPECT_NEAR(values[4], 1.5 * radialOff, 1e-12);
    EXPECT_NEAR(values[5], 0.0, 1e-15);
    EXPECT_NEAR(values[6], 0.0, 1e-15);
    EXPECT_NEAR(values[7], 0.0, 1e-15);
    EXPECT_NEAR(values[8], -0.5 * std::sqrt(35.0) * radialOff, 1e-12);
}

TEST(AoEvaluatorTest, HShellMatchesClosedForm) {
    const auto basis = qcx::basisset::ParseNwchemText(kH1);
    ASSERT_TRUE(basis.has_value());
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());
    EXPECT_EQ(evaluator->AOCount(), 11u);

    const auto* element = basis->Find(1);
    ASSERT_NE(element, nullptr);
    const auto& exponents = element->shells[0].exponents;
    const auto& coefficients = element->shells[0].coefficients[0];

    // z axis: m = 0 (index 5) equals z^5 times the radial part.
    std::array<double, 11> values{};
    const double d = 1.3;
    evaluator->Evaluate({0.0, 0.0, d}, values);
    const double radialZ = RadialReference(5, d * d, exponents, coefficients);

    for (std::size_t m = 0; m < 11; ++m)
    {
        if (m == 5)
        {
            EXPECT_NEAR(values[m], d * d * d * d * d * radialZ, 1e-12);
        } else
        {
            EXPECT_NEAR(values[m], 0.0, 1e-15) << "m index " << m;
        }
    }

    // (1, 0, 0): only the cosine rows with an x^l term survive: m = +1,
    // +3, +5 with sqrt(15)/8, -sqrt(35/128), sqrt(63/128) times the
    // radial part (P5(0) = 0 kills m = 0).
    evaluator->Evaluate({1.0, 0.0, 0.0}, values);
    const double radialOff = RadialReference(5, 1.0, exponents, coefficients);
    const double kPins[3] = {
        std::sqrt(15.0) / 8.0,
        -std::sqrt(35.0 / 128.0),
        std::sqrt(63.0 / 128.0),
    };

    for (std::size_t m = 0; m < 11; ++m)
    {
        const double expected = (m == 6 || m == 8 || m == 10) ? kPins[(m - 6) / 2] : 0.0;
        EXPECT_NEAR(values[m], expected * radialOff, 1e-12) << "m index " << m;
    }
}

TEST(AoEvaluatorTest, IShellMatchesClosedForm) {
    const auto basis = qcx::basisset::ParseNwchemText(kI1);
    ASSERT_TRUE(basis.has_value());
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());
    EXPECT_EQ(evaluator->AOCount(), 13u);

    const auto* element = basis->Find(1);
    ASSERT_NE(element, nullptr);
    const auto& exponents = element->shells[0].exponents;
    const auto& coefficients = element->shells[0].coefficients[0];

    // z axis: m = 0 (index 6) equals z^6 times the radial part.
    std::array<double, 13> values{};
    const double d = 1.3;
    evaluator->Evaluate({0.0, 0.0, d}, values);
    const double radialZ = RadialReference(6, d * d, exponents, coefficients);

    for (std::size_t m = 0; m < 13; ++m)
    {
        if (m == 6)
        {
            EXPECT_NEAR(values[m], d * d * d * d * d * d * radialZ, 1e-12);
        } else
        {
            EXPECT_NEAR(values[m], 0.0, 1e-15) << "m index " << m;
        }
    }

    // (1, 0, 0): the cosine rows with an x^6 term survive: m = 0, +2,
    // +4, +6 with -5/16, sqrt(105/512), -sqrt(63/256), sqrt(231/512)
    // times the radial part.
    evaluator->Evaluate({1.0, 0.0, 0.0}, values);
    const double radialOff = RadialReference(6, 1.0, exponents, coefficients);
    const double kPins[4] = {
        -5.0 / 16.0,
        std::sqrt(105.0 / 512.0),
        -std::sqrt(63.0 / 256.0),
        std::sqrt(231.0 / 512.0),
    };

    for (std::size_t m = 0; m < 13; ++m)
    {
        const double expected = (m == 6 || m == 8 || m == 10 || m == 12) ? kPins[(m - 6) / 2] : 0.0;
        EXPECT_NEAR(values[m], expected * radialOff, 1e-12) << "m index " << m;
    }
}

TEST(AoEvaluatorTest, HighLSphericalSumIsConstantOnUnitSphere) {
    // Sum over m of S_lm(r)^2 = 1 on the unit sphere (the addition
    // theorem under the Schlegel norm), so the squared AO values at a
    // unit point sum to the squared radial part for every l.
    constexpr std::string_view kShells[3] = {kG1, kH1, kI1};
    const int ls[3] = {4, 5, 6};
    const int counts[3] = {9, 11, 13};

    for (int t = 0; t < 3; ++t)
    {
        const auto basis = qcx::basisset::ParseNwchemText(kShells[t]);
        ASSERT_TRUE(basis.has_value());
        const auto molecule = MakeSingleHydrogen();
        ASSERT_TRUE(molecule.has_value());
        const auto evaluator = AoEvaluator::Create(*molecule, *basis);
        ASSERT_TRUE(evaluator.has_value());

        const auto* element = basis->Find(1);
        ASSERT_NE(element, nullptr);
        const auto& exponents = element->shells[0].exponents;
        const auto& coefficients = element->shells[0].coefficients[0];

        std::array<double, 13> values{};
        std::span<double> block(values.data(), counts[t]);

        const std::array<std::array<double, 3>, 2> unitPoints = {{
            {0.6, -0.8, 0.0},
            {1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)},
        }};

        for (const auto& point : unitPoints)
        {
            evaluator->Evaluate(point, block);
            double sum = 0.0;

            for (std::size_t m = 0; m < block.size(); ++m)
            {
                sum += block[m] * block[m];
            }

            const double radial = RadialReference(ls[t], 1.0, exponents, coefficients);
            EXPECT_NEAR(sum, radial * radial, 1e-12) << "l = " << ls[t];
        }
    }
}

TEST(AoEvaluatorTest, HighLGridNormsAreOne) {
    // int phi^2 over the molecular grid equals 1 for l = 4..6 single
    // primitives: the parser's spherical renormalization targets exactly
    // the evaluator's radial factor and Schlegel rows.  The angular grid
    // must integrate degree 2l = 12 exactly (Lebedev 74, degree 13).
    constexpr std::string_view kShells[3] = {kG1, kH1, kI1};
    const int ls[3] = {4, 5, 6};
    const int counts[3] = {9, 11, 13};

    for (int t = 0; t < 3; ++t)
    {
        const auto basis = qcx::basisset::ParseNwchemText(kShells[t]);
        ASSERT_TRUE(basis.has_value());
        const auto molecule = MakeSingleHydrogen();
        ASSERT_TRUE(molecule.has_value());
        const auto evaluator = AoEvaluator::Create(*molecule, *basis);
        ASSERT_TRUE(evaluator.has_value());
        const auto grid = MolecularGrid::Create(*molecule, 80, 74);
        ASSERT_TRUE(grid.has_value());

        std::array<double, 13> values{};
        std::span<double> block(values.data(), counts[t]);
        std::array<double, 13> norms{};

        for (std::size_t i = 0; i < grid->Size(); ++i)
        {
            evaluator->Evaluate(grid->Point(i), block);

            for (std::size_t m = 0; m < block.size(); ++m)
            {
                norms[m] += grid->Weight(i) * block[m] * block[m];
            }
        }

        for (int m = 0; m < counts[t]; ++m)
        {
            EXPECT_NEAR(norms[m], 1.0, 1e-3) << "l = " << ls[t] << ", m = " << m;
        }
    }
}

TEST(AoEvaluatorTest, ICartesianShellUsesMonomials) {
    constexpr std::string_view kIcart = R"(
BASIS "ao basis" CARTESIAN PRINT
H I
1.0 1.0
END
)";
    const auto basis = qcx::basisset::ParseNwchemText(kIcart);
    ASSERT_TRUE(basis.has_value());
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());
    EXPECT_EQ(evaluator->AOCount(), 28u);

    // (a,b,c) order, a descending then b descending: at (0,0,1) only the
    // zzzzzz monomial (last, index 27) survives, with value N z^6 e^-1.
    std::array<double, 28> values{};
    evaluator->Evaluate({0.0, 0.0, 1.0}, values);
    const double expected = std::pow(2.0 / kPi, 0.75) * std::exp(-1.0);

    for (std::size_t i = 0; i < 28; ++i)
    {
        if (i == 27)
        {
            EXPECT_NEAR(values[i], expected, 1e-12);
        } else
        {
            EXPECT_NEAR(values[i], 0.0, 1e-15) << "index " << i;
        }
    }
}

TEST(AoEvaluatorTest, CreateRejectsMissingElement) {
    const auto basis = qcx::basisset::ParseNwchemText(kSto3gH);
    ASSERT_TRUE(basis.has_value());

    // No basis for helium.
    const auto molecule = qcx::testing::MakeHeAtom();
    ASSERT_TRUE(molecule.has_value());
    EXPECT_FALSE(AoEvaluator::Create(*molecule, *basis).has_value());

    // Angular momenta beyond 6 are accepted by no parser (the SPDFGHI
    // labels stop at I), so the evaluator's defensive cap above 6 is
    // unreachable from parsed input; the l = 4..6 acceptance is covered
    // by the closed-form tests above.
}

TEST(AoEvaluatorTest, GradientTierMatchesTheFullTier) {
    // The gradient-only tier exists to skip the second derivatives, not to
    // approximate them: its values and gradients must be the ones the full
    // tier writes.  The split touches the angular derivative fill and the
    // per-shell radial assembly, so every shipped angular momentum is
    // exercised, s through i.
    //
    // EXPECT_DOUBLE_EQ rather than exact equality because the two tiers are
    // separate instantiations of the same arithmetic: the compiler is free to
    // contract or reassociate differently in each, and four ULP is the
    // tolerance for that, not for a numerical difference in the method.
    struct ShellCase {
        std::string_view label;
        std::string_view basis;
    };

    const std::array<ShellCase, 6> shellCases = {{
        {"s", kSto3gH},
        {"p", kP1},
        {"p-cartesian", kP1Cartesian},
        {"g", kG1},
        {"h", kH1},
        {"i", kI1},
    }};

    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const std::array<std::array<double, 3>, 3> points = {{
        {0.3, -0.2, 0.1},
        {1.7, 0.9, -0.5},
        {-0.4, 2.2, 0.8},
    }};

    for (const ShellCase& shellCase : shellCases)
    {
        const auto basis = qcx::basisset::ParseNwchemText(shellCase.basis);
        ASSERT_TRUE(basis.has_value()) << shellCase.label;
        const auto evaluator = AoEvaluator::Create(*molecule, *basis);
        ASSERT_TRUE(evaluator.has_value()) << shellCase.label;

        const std::size_t count = evaluator->AOCount();
        std::vector<double> fullValues(count);
        std::vector<double> fullGradients(3 * count);
        std::vector<double> hessians(6 * count);
        std::vector<double> tierValues(count);
        std::vector<double> tierGradients(3 * count);

        for (const std::array<double, 3>& point : points)
        {
            evaluator->EvaluateDerivatives(point, fullValues, fullGradients, hessians);
            evaluator->EvaluateGradients(point, tierValues, tierGradients);

            for (std::size_t function = 0; function < count; ++function)
            {
                EXPECT_DOUBLE_EQ(tierValues[function], fullValues[function])
                    << shellCase.label << " value " << function;
            }

            for (std::size_t slot = 0; slot < 3 * count; ++slot)
            {
                EXPECT_DOUBLE_EQ(tierGradients[slot], fullGradients[slot])
                    << shellCase.label << " gradient slot " << slot;
            }
        }
    }
}

TEST(AoEvaluatorTest, SelectedEvaluationMatchesTheFullOneAndTouchesOnlyItsShells) {
    // EvaluateSelected and EvaluateGradientsSelected are what a screened
    // assembly calls, so two properties matter.  Over the FULL selection they
    // must reproduce the unselected entry points exactly.  Over a PARTIAL
    // selection they must write their own shells' slots and leave every other
    // slot alone - which is a contract rather than an optimisation, because a
    // caller reuses one buffer across grid points and a value left over from
    // the previous point would be silently wrong rather than obviously absent.
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value());
    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());

    const std::size_t count = evaluator->AOCount();
    const std::span<const ShellRange> ranges = evaluator->ShellRanges();
    ASSERT_EQ(ranges.size(), 2u) << "the fixture must carry two shells";

    std::vector<std::size_t> allShells;
    allShells.reserve(ranges.size());

    for (std::size_t index = 0; index < ranges.size(); ++index)
    {
        allShells.push_back(index);
    }

    const std::array<double, 3> point = {0.7, -0.3, 0.4};
    constexpr double kSentinel = 1234.5;

    std::vector<double> referenceValues(count);
    std::vector<double> referenceGradients(3 * count);
    evaluator->EvaluateGradients(point, referenceValues, referenceGradients);

    {
        std::vector<double> values(count, kSentinel);
        evaluator->Evaluate(point, values);

        std::vector<double> selected(count, kSentinel);
        evaluator->EvaluateSelected(point, allShells, selected);

        for (std::size_t mu = 0; mu < count; ++mu)
        {
            EXPECT_EQ(values[mu], referenceValues[mu]);
            EXPECT_DOUBLE_EQ(selected[mu], referenceValues[mu]) << "value " << mu;
        }
    }

    {
        std::vector<double> selectedValues(count, kSentinel);
        std::vector<double> selectedGradients(3 * count, kSentinel);
        evaluator->EvaluateGradientsSelected(point, allShells, selectedValues, selectedGradients);

        for (std::size_t mu = 0; mu < count; ++mu)
        {
            EXPECT_DOUBLE_EQ(selectedValues[mu], referenceValues[mu]) << "value " << mu;
        }

        for (std::size_t slot = 0; slot < 3 * count; ++slot)
        {
            EXPECT_DOUBLE_EQ(selectedGradients[slot], referenceGradients[slot])
                << "gradient slot " << slot;
        }
    }

    for (std::size_t chosen = 0; chosen < ranges.size(); ++chosen)
    {
        const std::vector<std::size_t> selection = {chosen};
        std::vector<double> values(count, kSentinel);
        std::vector<double> gradients(3 * count, kSentinel);
        evaluator->EvaluateGradientsSelected(point, selection, values, gradients);

        const ShellRange range = ranges[chosen];

        for (std::size_t mu = 0; mu < count; ++mu)
        {
            const bool selected = mu >= range.aoOffset && mu < range.aoOffset + range.functionCount;

            if (selected)
            {
                EXPECT_DOUBLE_EQ(values[mu], referenceValues[mu])
                    << "shell " << chosen << " value " << mu;
            } else
            {
                EXPECT_EQ(values[mu], kSentinel) << "slot " << mu << " must be left untouched";
            }

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                const std::size_t slot = 3 * mu + axis;

                if (selected)
                {
                    EXPECT_DOUBLE_EQ(gradients[slot], referenceGradients[slot])
                        << "shell " << chosen << " gradient " << slot;
                } else
                {
                    EXPECT_EQ(gradients[slot], kSentinel)
                        << "gradient slot " << slot << " must be left untouched";
                }
            }
        }
    }
}

} // namespace
} // namespace qcx::grid
