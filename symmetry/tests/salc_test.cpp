// SALC generation: orthonormal rows, correct irrep counts, and the 8/8
// proof that the s-s overlap matrix block-diagonalizes over the irreps.
#include "corpus.hpp"
#include "qcx/linalg/dense_ops.hpp"
#include "qcx/symmetry/detection.hpp"
#include "qcx/symmetry/point_group.hpp"
#include "qcx/symmetry/salc.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace qcx::symmetry {
namespace {

using MakeFn = qcx::Result<qcx::molecule::Molecule> (*)();

constexpr double kOrthonormalityTolerance = 1e-8;
constexpr double kCrossBlockTolerance = 1e-8;
constexpr double kExponentPower = 2.0 / 3.0; ///< alpha(Z) = Z^(2/3).
constexpr double kOverlapPower = 1.5; ///< (2 sqrt(ab)/(a+b))^(3/2) prefactor.
constexpr double kZero = 0.0;

struct ProofCase {
    const char* name;
    MakeFn make;
    PointGroup expectedGroup;
};

const ProofCase kProofCases[] = {
    {"C1 generic", testing::MakeGenericC1, PointGroup::kC1},
    {"Ci meso", testing::MakeMesoDifluoroethane, PointGroup::kCi},
    {"C2 twisted H2O2", testing::MakeTwistedHydrogenPeroxide, PointGroup::kC2},
    {"Cs HOF", testing::MakeHypofluorousAcid, PointGroup::kCs},
    {"C2h trans-N2H2", testing::MakeTransDiazene, PointGroup::kC2h},
    {"D2 synthetic", testing::MakeD2Synthetic, PointGroup::kD2},
    {"C2v water", testing::MakeWater, PointGroup::kC2v},
    {"D2h ethene", testing::MakeEthene, PointGroup::kD2h},
    // Degenerate inertia: the axis frame needs the perpendicularity
    // constraints, not independent moment alignment (oblate and spherical).
    {"D2h benzene (D6h)", testing::MakeBenzene, PointGroup::kD2h},
    {"D2h SF6 (Oh)", testing::MakeSulfurHexafluoride, PointGroup::kD2h},
    // Linear molecules: the detector records the C2/mirror elements the
    // computational groups need (all atoms sit on the axis).
    {"C2v HF (Cinfv)", testing::MakeHydrogenFluoride, PointGroup::kC2v},
    {"D2h CO2 (Dinfh)", testing::MakeCarbonDioxide, PointGroup::kD2h},
};

// Normalized s-s Gaussian overlap with exponents alphaI, alphaJ at
// distance r: (2 sqrt(ab)/(a+b))^(3/2) exp(-ab r^2/(a+b)).
double SGaussianOverlap(double alphaI, double alphaJ, double rSq) {
    const double sum = alphaI + alphaJ;
    return std::pow(2.0 * std::sqrt(alphaI * alphaJ) / sum, kOverlapPower) *
           std::exp(-alphaI * alphaJ * rSq / sum);
}

TEST(SalcTest, SalcsSpanEveryGroupAndDiagonalizeOverlap) {
    for (const auto& testCase : kProofCases)
    {
        SCOPED_TRACE(testCase.name);
        const auto molecule = testCase.make();
        ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
        const SymmetryAnalysis analysis = DetectPointGroup(*molecule);
        ASSERT_EQ(analysis.computational, testCase.expectedGroup);

        const auto salcs = GenerateSalcs(*molecule, analysis);
        ASSERT_TRUE(salcs.has_value()) << salcs.error().message;
        const auto n = static_cast<Eigen::Index>(molecule->AtomCount());

        // Full basis: one row per atom, labeled by irrep, orthonormal.
        ASSERT_EQ(salcs->coefficients.rows(), n);
        ASSERT_EQ(salcs->coefficients.cols(), n);
        ASSERT_EQ(salcs->irrepLabels.size(), static_cast<std::size_t>(n));
        EXPECT_TRUE((salcs->coefficients * salcs->coefficients.transpose() -
                     Eigen::MatrixXd::Identity(n, n))
                        .norm() < kOrthonormalityTolerance);

        // The s-s overlap matrix of same-element atoms is invariant under
        // the group (symmetry-equivalent atoms share exponents), so the
        // SALC transform block-diagonalizes it: cross-irrep entries vanish.
        const auto& coords = molecule->CoordinatesBohr();
        const auto& atoms = molecule->Atoms();
        Eigen::MatrixXd overlap(n, n);

        for (Eigen::Index i = 0; i < n; ++i)
        {
            for (Eigen::Index j = 0; j < n; ++j)
            {
                const double alphaI =
                    std::pow(static_cast<double>(atoms[static_cast<std::size_t>(i)].atomicNumber),
                             kExponentPower);
                const double alphaJ =
                    std::pow(static_cast<double>(atoms[static_cast<std::size_t>(j)].atomicNumber),
                             kExponentPower);
                double rSq = kZero;

                for (int d = 0; d < 3; ++d)
                {
                    const double delta = coords(static_cast<std::size_t>(i), d) -
                                         coords(static_cast<std::size_t>(j), d);
                    rSq += delta * delta;
                }

                overlap(i, j) = SGaussianOverlap(alphaI, alphaJ, rSq);
            }
        }

        const auto block = qcx::linalg::DenseMultiply(
            salcs->coefficients,
            qcx::linalg::DenseMultiply(overlap, salcs->coefficients.transpose()).value());
        ASSERT_TRUE(block.has_value());

        for (Eigen::Index r = 0; r < n; ++r)
        {
            for (Eigen::Index s = 0; s < n; ++s)
            {
                if (salcs->irrepLabels[static_cast<std::size_t>(r)] !=
                    salcs->irrepLabels[static_cast<std::size_t>(s)])
                {
                    EXPECT_LT(std::abs((*block)(r, s)), kCrossBlockTolerance)
                        << "cross-irrep block at (" << r << ", " << s << ")";
                }
            }
        }
    }
}

TEST(SalcTest, WaterHasTwoHydrogenSalcs) {
    const auto molecule = testing::MakeWater();
    ASSERT_TRUE(molecule.has_value());
    const auto salcs = GenerateSalcs(*molecule, DetectPointGroup(*molecule));
    ASSERT_TRUE(salcs.has_value());
    // 3 atoms -> 3 SALCs; the two H's form one orbit whose pair spans two
    // distinct irreps (only the counts are asserted - labels are B1/B2
    // ambiguous by convention).
    EXPECT_EQ(salcs->coefficients.rows(), 3);
    std::vector<std::string_view> labels = salcs->irrepLabels;
    std::sort(labels.begin(), labels.end());
    EXPECT_EQ(labels[0], labels[1]);
    EXPECT_NE(labels[1], labels[2]);
}

} // namespace
} // namespace qcx::symmetry
