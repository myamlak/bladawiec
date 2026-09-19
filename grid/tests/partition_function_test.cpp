// Becke/SSF partition: the defining properties are (a) the weights sum
// to 1 at every point (partition of unity), (b) the cell boundary
// midpoints split evenly for identical atoms, (c) a point deep inside an
// atom's cell carries weight 1.

#include "h2_sto3g.hpp"
#include "qcx/grid/partition_function.hpp"
#include "qcx/memory/tensor.hpp"

#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace qcx::grid {
namespace {

// H2 (R = 1.4 bohr, as in the shared fixture) plus a ghost atom 1000 bohr
// away: the fixture for the exact-cutoff contract, where the far atom must
// vanish from the partition without touching the real atoms.
qcx::Result<qcx::molecule::Molecule> MakeH2WithGhostAtom() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.4;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    (*coordinates)(2, 0) = 1000.0;
    (*coordinates)(2, 1) = 0.0;
    (*coordinates)(2, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}, {"Ar", 18, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

TEST(PartitionFunctionTest, WeightsSumToUnityEverywhere) {
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());

    // A sweep of points: on the bond axis, off it, far outside, at a
    // nucleus.
    const std::array<std::array<double, 3>, 6> probes = {{
        {0.0, 0.0, 0.0},
        {0.7, 0.0, 0.0},
        {1.4, 0.0, 0.0},
        {0.5, 0.8, -0.3},
        {10.0, 0.0, 0.0},
        {-3.0, 2.0, 1.0},
    }};

    for (const auto& point : probes)
    {
        const std::vector<double> weights = BeckePartitionWeights(point, *molecule);
        ASSERT_EQ(weights.size(), 2u);
        double sum = 0.0;

        for (const double w : weights)
        {
            sum += w;
        }

        EXPECT_NEAR(sum, 1.0, 1e-12) << "at " << point[0] << " " << point[1] << " " << point[2];
    }
}

TEST(PartitionFunctionTest, IdenticalAtomsSplitTheMidpoint) {
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());

    // H2 has both hydrogens at... the fixture's bond length; the cell
    // boundary of two identical atoms sits halfway between them.
    const auto& coordinates = molecule->CoordinatesBohr();
    const double mid = 0.5 * (coordinates(0, 0) + coordinates(1, 0));
    const std::vector<double> weights = BeckePartitionWeights({mid, 0.0, 0.0}, *molecule);
    EXPECT_NEAR(weights[0], 0.5, 1e-6);
    EXPECT_NEAR(weights[1], 0.5, 1e-6);
}

TEST(PartitionFunctionTest, NearNucleusWeightIsOne) {
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());

    const std::vector<double> weights = BeckePartitionWeights({0.0, 0.0, 0.0}, *molecule);
    EXPECT_NEAR(weights[0], 1.0, 1e-12);
    EXPECT_NEAR(weights[1], 0.0, 1e-12);
}

TEST(PartitionFunctionTest, FarAwayPointBelongsToNearestAtom) {
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());

    // Far beyond atom 1 on its side of the bond axis.
    const auto& coordinates = molecule->CoordinatesBohr();
    const std::vector<double> weights =
        BeckePartitionWeights({2.0 * coordinates(1, 0), 0.0, 0.0}, *molecule);
    EXPECT_GT(weights[1], 0.999);
}

TEST(PartitionFunctionTest, BraggSlaterRadiiAreTabulated) {
    EXPECT_NEAR(BraggSlaterRadiusBohr(1), 0.35 * 1.8897261246257702, 1e-12);
    EXPECT_NEAR(BraggSlaterRadiusBohr(6), 0.70 * 1.8897261246257702, 1e-12);
    EXPECT_NEAR(BraggSlaterRadiusBohr(8), 0.60 * 1.8897261246257702, 1e-12);
}

TEST(PartitionFunctionTest, FarAtomsBeyondTheCutoffCostNothing) {
    // The exact-cutoff contract (P-D, scaling-audit track): an atom beyond
    // the partition's decay radius - here a ghost atom 1000 bohr away -
    // vanishes from the weights EXACTLY, leaves the real atoms' weights
    // bit-identical to the ghost-free molecule, and adds nothing to the
    // per-point pair loop (the switch-evaluation count is unchanged, so
    // the per-point cost is independent of atoms beyond the cutoff).
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());

    const auto ghostMolecule = MakeH2WithGhostAtom();
    ASSERT_TRUE(ghostMolecule.has_value());
    ASSERT_EQ(ghostMolecule->AtomCount(), 3u);

    // Canonical atom order (molecule.hpp): [H, H, Ar], so the ghost is
    // index 2 in both the raw weights and the distance list.
    const auto& coordinates = molecule->CoordinatesBohr();
    const std::array<std::array<double, 3>, 5> probes = {{
        {0.7, 0.0, 0.0}, // the bond midpoint (general-loop region)
        {0.5, 0.8, -0.3}, // off-axis
        {10.0, 0.0, 0.0}, // far outside
        {-3.0, 2.0, 1.0}, // far outside, off-axis
        {coordinates(1, 0), 0.0, 0.0}, // at the H2 nucleus (shortcut region)
    }};

    for (const auto& point : probes)
    {
        std::size_t countBare = 0;
        const std::vector<double> weightsBare = BeckeRawWeights(point, *molecule, countBare);

        std::size_t countGhost = 0;
        const std::vector<double> weightsGhost = BeckeRawWeights(point, *ghostMolecule, countGhost);

        EXPECT_EQ(countGhost, countBare) << "at " << point[0] << " " << point[1] << " " << point[2];
        EXPECT_DOUBLE_EQ(weightsGhost[2], 0.0)
            << "at " << point[0] << " " << point[1] << " " << point[2];
        EXPECT_DOUBLE_EQ(weightsGhost[0], weightsBare[0])
            << "at " << point[0] << " " << point[1] << " " << point[2];
        EXPECT_DOUBLE_EQ(weightsGhost[1], weightsBare[1])
            << "at " << point[0] << " " << point[1] << " " << point[2];
    }
}

} // namespace
} // namespace qcx::grid
