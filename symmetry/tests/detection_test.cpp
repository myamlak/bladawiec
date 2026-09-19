// Point-group detection against the exact-symmetry regression corpus.
#include "corpus.hpp"
#include "large_molecules.hpp"
#include "qcx/symmetry/detection.hpp"
#include "qcx/symmetry/point_group.hpp"
#include "qcx/symmetry/point_group_name.hpp"

#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <numbers>
#include <utility>
#include <vector>

namespace qcx::symmetry {
namespace {

using MakeFn = qcx::Result<qcx::molecule::Molecule> (*)();

struct CorpusCase {
    const char* name;
    MakeFn make;
    PointGroupName expectedGroup;
    PointGroup expectedComputational;
};

const CorpusCase kCorpus[] = {
    {"Water", testing::MakeWater, PointGroupName::kC2v, PointGroup::kC2v},
    {"Ethene", testing::MakeEthene, PointGroupName::kD2h, PointGroup::kD2h},
    {"Benzene", testing::MakeBenzene, PointGroupName::kD6h, PointGroup::kD2h},
    {"Ammonia", testing::MakeAmmonia, PointGroupName::kC3v, PointGroup::kCs},
    {"Methane", testing::MakeMethane, PointGroupName::kTd, PointGroup::kD2h},
    {"Sulfur hexafluoride", testing::MakeSulfurHexafluoride, PointGroupName::kOh, PointGroup::kD2h},
    {"Carbon dioxide", testing::MakeCarbonDioxide, PointGroupName::kDInfH, PointGroup::kD2h},
    {"Hydrogen fluoride", testing::MakeHydrogenFluoride, PointGroupName::kCInfV, PointGroup::kC2v},
    {"Twisted hydrogen peroxide",
     testing::MakeTwistedHydrogenPeroxide,
     PointGroupName::kC2,
     PointGroup::kC2},
    {"trans-diazene", testing::MakeTransDiazene, PointGroupName::kC2h, PointGroup::kC2h},
    {"Allene", testing::MakeAllene, PointGroupName::kD2d, PointGroup::kD2},
    {"meso-CHFCl-CHFCl", testing::MakeMesoDifluoroethane, PointGroupName::kCi, PointGroup::kCi},
    {"Hypofluorous acid", testing::MakeHypofluorousAcid, PointGroupName::kCs, PointGroup::kCs},
    {"Eclipsed ethane", testing::MakeEclipsedEthane, PointGroupName::kD3h, PointGroup::kC2v},
    {"Staggered ethane", testing::MakeStaggeredEthane, PointGroupName::kD3d, PointGroup::kC2h},
    {"S8 crown", testing::MakeS8Crown, PointGroupName::kD4d, PointGroup::kD2},
    {"Generic asymmetric", testing::MakeGenericC1, PointGroupName::kC1, PointGroup::kC1},
    {"D2 synthetic", testing::MakeD2Synthetic, PointGroupName::kD2, PointGroup::kD2},
    {"S4 synthetic", testing::MakeS4Synthetic, PointGroupName::kS4, PointGroup::kC2},
    {"S6 synthetic", testing::MakeS6Synthetic, PointGroupName::kS6, PointGroup::kCi},
    {"S8 synthetic", testing::MakeS8Synthetic, PointGroupName::kS8, PointGroup::kC2},
    {"T synthetic", testing::MakeTSynthetic, PointGroupName::kT, PointGroup::kD2},
    {"Th synthetic", testing::MakeThSynthetic, PointGroupName::kTh, PointGroup::kD2h},
};

// Rebuilds a molecule from its coordinates with a small deterministic
// perturbation applied (Molecule is immutable, so perturbation goes through
// Create).
qcx::Result<qcx::molecule::Molecule> Perturb(const qcx::molecule::Molecule& molecule,
                                             double magnitude) {
    const auto& source = molecule.CoordinatesBohr();
    auto coordinates =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({molecule.AtomCount(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    for (std::size_t i = 0; i < molecule.AtomCount(); ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            // Cast before subtracting: (i + d) % 3 is unsigned, so -1 would
            // underflow to SIZE_MAX and shift atoms by ~1e16 Bohr.
            const double shift = magnitude * static_cast<double>(static_cast<int>((i + d) % 3) - 1);
            (*coordinates)(i, d) = source(i, d) + shift;
        }
    }

    return qcx::molecule::Molecule::Create(
        molecule.Atoms(), std::move(*coordinates), molecule.Charge(), molecule.Multiplicity());
}

TEST(DetectionTest, CorpusGroupsAndReductions) {
    for (const auto& testCase : kCorpus)
    {
        SCOPED_TRACE(testCase.name);
        const auto molecule = testCase.make();
        ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
        const SymmetryAnalysis analysis = DetectPointGroup(*molecule);
        EXPECT_EQ(analysis.group, testCase.expectedGroup)
            << "detected " << ToString(analysis.group) << ", expected "
            << ToString(testCase.expectedGroup);
        EXPECT_EQ(analysis.computational, testCase.expectedComputational);
        EXPECT_EQ(analysis.computational, LargestAbelianSubgroup(analysis.group));
    }
}

TEST(DetectionTest, BuckminsterfullereneIsIh) {
    const auto molecule = qcx::molecule::testing::MakeBuckminsterfullerene();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const SymmetryAnalysis analysis = DetectPointGroup(*molecule);
    EXPECT_EQ(analysis.group, PointGroupName::kIh);
    EXPECT_EQ(analysis.computational, PointGroup::kD2h);
}

TEST(DetectionTest, ElementsIncludeGenerators) {
    // Benzene D6h: identity, inversion, a C6, mirrors, and the S6.
    const auto molecule = testing::MakeBenzene();
    ASSERT_TRUE(molecule.has_value());
    const SymmetryAnalysis analysis = DetectPointGroup(*molecule);
    bool hasIdentity = false;
    bool hasInversion = false;
    bool hasC6 = false;
    bool hasMirror = false;
    bool hasS6 = false;

    for (const auto& element : analysis.elements)
    {
        hasIdentity = hasIdentity || element.kind == OperationKind::kIdentity;
        hasInversion = hasInversion || element.kind == OperationKind::kInversion;
        hasC6 = hasC6 || (element.kind == OperationKind::kRotation && element.order == 6);
        hasMirror = hasMirror || element.kind == OperationKind::kSigma;
        hasS6 = hasS6 || (element.kind == OperationKind::kImproper && element.order == 6);
    }

    EXPECT_TRUE(hasIdentity);
    EXPECT_TRUE(hasInversion);
    EXPECT_TRUE(hasC6);
    EXPECT_TRUE(hasMirror);
    EXPECT_TRUE(hasS6);
    // Every recorded element must actually map the molecule onto itself.
    const auto& coords = molecule->CoordinatesBohr();
    const auto& atoms = molecule->Atoms();

    for (const auto& element : analysis.elements)
    {
        if (element.kind == OperationKind::kIdentity)
        {
            continue;
        }

        Eigen::Matrix3d transform;

        if (element.kind == OperationKind::kInversion)
        {
            transform = -Eigen::Matrix3d::Identity();
        } else
        {
            const Eigen::Matrix3d rotation =
                Eigen::AngleAxisd(2.0 * std::numbers::pi / static_cast<double>(element.order),
                                  element.axis)
                    .toRotationMatrix();
            transform =
                element.kind == OperationKind::kSigma
                    ? Eigen::Matrix3d::Identity() - 2.0 * element.axis * element.axis.transpose()
                    : rotation;

            if (element.kind == OperationKind::kImproper)
            {
                transform =
                    (Eigen::Matrix3d::Identity() - 2.0 * element.axis * element.axis.transpose()) *
                    rotation;
            }
        }

        for (std::size_t i = 0; i < molecule->AtomCount(); ++i)
        {
            const Eigen::Vector3d p(coords(i, 0), coords(i, 1), coords(i, 2));
            const Eigen::Vector3d q = transform * p;
            bool found = false;

            for (std::size_t j = 0; j < molecule->AtomCount(); ++j)
            {
                if (atoms[j].atomicNumber != atoms[i].atomicNumber)
                {
                    continue;
                }

                const Eigen::Vector3d c(coords(j, 0), coords(j, 1), coords(j, 2));

                if ((q - c).norm() <= 1e-4)
                {
                    found = true;
                    break;
                }
            }

            EXPECT_TRUE(found) << "element does not map the molecule onto itself";
        }
    }
}

TEST(DetectionTest, PerturbationWithinToleranceStillDetected) {
    // 1e-3 Bohr noise needs a tolerance that absorbs it; the strict default
    // deliberately falls back to C1 (documented misdetection hazard for
    // near-symmetric geometries).
    const auto molecule = testing::MakeBenzene();
    ASSERT_TRUE(molecule.has_value());
    const auto perturbed = Perturb(*molecule, 1e-3);
    ASSERT_TRUE(perturbed.has_value()) << perturbed.error().message;
    EXPECT_EQ(DetectPointGroup(*perturbed).group, PointGroupName::kC1);
    DetectOptions loose;
    loose.positionToleranceBohr = 0.01;
    EXPECT_EQ(DetectPointGroup(*perturbed, loose).group, PointGroupName::kD6h);
}

TEST(DetectionTest, LargePerturbationFallsBackToC1) {
    // Methane (4 atoms, non-planar) with a large perturbation: no symmetry
    // survives. A triatomic would stay at least Cs - any three points define
    // a plane, and the in-plane mirror always maps them onto themselves.
    const auto molecule = testing::MakeMethane();
    ASSERT_TRUE(molecule.has_value());
    const auto perturbed = Perturb(*molecule, 0.5);
    ASSERT_TRUE(perturbed.has_value()) << perturbed.error().message;
    EXPECT_EQ(DetectPointGroup(*perturbed).group, PointGroupName::kC1);
}

TEST(DetectionTest, SingleAtomIsC1) {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({1, 3});
    ASSERT_TRUE(coordinates.has_value());
    (*coordinates)(0, 0) = 1.0;
    (*coordinates)(0, 1) = 2.0;
    (*coordinates)(0, 2) = 3.0;
    auto molecule = qcx::molecule::Molecule::Create(
        {qcx::molecule::Atom{"He", 2, 0.0}}, std::move(*coordinates), 0, 1);
    ASSERT_TRUE(molecule.has_value());
    const SymmetryAnalysis analysis = DetectPointGroup(*molecule);
    EXPECT_EQ(analysis.group, PointGroupName::kC1);
    EXPECT_EQ(analysis.elements.size(), 1u);
}

TEST(DetectionTest, NearAxisAtomsKeepTheGroupUnderNoise) {
    // sym-1: atoms close to a symmetry axis make the FITTED rotation angle
    // inherit the position noise amplified by 1/r - a fixed 1e-4 rad angle
    // tolerance then silently under-detects to a lower group even though
    // the position tolerance accepts the fits. The derived tolerance
    // (2 * positionTolerance / min atom-to-axis radius) must keep the
    // group. Synthetic D2h: two atoms on the z axis (1.0 Bohr out) and two
    // at radius 0.02 Bohr about it; the C2(z) / C2(y) / inversion fits
    // move the near-axis pair, whose 2.5e-5 Bohr noise gives an angle
    // error of ~1.25e-3 rad - far over the fixed 1e-4, well under the
    // derived 2 * 1e-4 / 0.02 = 1e-2.
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({4, 3});
    ASSERT_TRUE(coordinates.has_value());
    constexpr double kAxisRadius = 0.02;
    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 1.0;
    (*coordinates)(1, 0) = 0.0;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = -1.0;
    (*coordinates)(2, 0) = kAxisRadius;
    (*coordinates)(2, 1) = 0.0;
    (*coordinates)(2, 2) = 0.0;
    (*coordinates)(3, 0) = -kAxisRadius;
    (*coordinates)(3, 1) = 0.0;
    (*coordinates)(3, 2) = 0.0;
    const std::vector<qcx::molecule::Atom> atoms = {
        qcx::molecule::Atom{"H", 1, 0.0},
        qcx::molecule::Atom{"H", 1, 0.0},
        qcx::molecule::Atom{"H", 1, 0.0},
        qcx::molecule::Atom{"H", 1, 0.0},
    };
    auto exact = qcx::molecule::Molecule::Create(atoms, std::move(*coordinates), 0, 1);
    ASSERT_TRUE(exact.has_value());
    EXPECT_EQ(DetectPointGroup(*exact).group, PointGroupName::kD2h);

    // Perturbation within the default position tolerance (2.5e-5 of 1e-4
    // per coordinate): with the fixed angle tolerance the C2(z) / C2(y) /
    // inversion fits (angle error ~1.25e-3 rad) fail classification and
    // the molecule degrades to C2h; the derived tolerance keeps D2h.
    const auto noisy = Perturb(*exact, 2.5e-5);
    ASSERT_TRUE(noisy.has_value()) << noisy.error().message;
    EXPECT_EQ(DetectPointGroup(*noisy).group, PointGroupName::kD2h);
}

} // namespace
} // namespace qcx::symmetry
