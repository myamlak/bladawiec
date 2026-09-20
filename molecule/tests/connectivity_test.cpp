// Connectivity: covalent-radius-sum heuristic, CSR structure, and graph consistency.
#include "large_molecules.hpp"
#include "qcx/molecule/connectivity.hpp"
#include "qcx/molecule/molecule.hpp"

#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <gtest/gtest.h>
#include <vector>

namespace qcx::molecule {
namespace {

// kAngstromToBohr comes from molecule.hpp (shared constant).

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- test helper mirrors Create.
qcx::Result<Molecule> MakeMolecule(std::vector<Atom> atoms,
                                   std::vector<std::vector<double>> rows,
                                   int charge = 0,
                                   int multiplicity = 1) {
    auto coords = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({rows.size(), 3});

    if (!coords.has_value())
    {
        return std::unexpected(coords.error());
    }

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            (*coords)(i, d) = rows[i][d];
        }
    }

    return Molecule::Create(std::move(atoms), std::move(*coords), charge, multiplicity);
}

void CheckCsrMatchesGraph(const Connectivity& connectivity, std::size_t atomCount) {
    const auto& csr = connectivity.csr;
    EXPECT_EQ(csr.offsets.size(), atomCount + 1);
    std::size_t totalDegrees = 0;

    for (std::size_t i = 0; i < atomCount; ++i)
    {
        const std::size_t degree = csr.offsets[i + 1] - csr.offsets[i];
        totalDegrees += degree;
        EXPECT_EQ(degree, static_cast<std::size_t>(boost::degree(i, connectivity.graph)));
        // Neighbors sorted ascending and symmetric.

        for (std::size_t k = csr.offsets[i]; k < csr.offsets[i + 1]; ++k)
        {
            const std::size_t j = csr.neighbors[k];

            if (k + 1 < csr.offsets[i + 1])
            {
                EXPECT_LT(csr.neighbors[k], csr.neighbors[k + 1]);
            }

            const auto& jList = csr.neighbors;
            EXPECT_NE(
                std::find(jList.begin() + csr.offsets[j], jList.begin() + csr.offsets[j + 1], i),
                jList.begin() + csr.offsets[j + 1]);
        }
    }

    EXPECT_EQ(totalDegrees, 2 * boost::num_edges(connectivity.graph));
}

TEST(ConnectivityTest, WaterHasTwoBonds) {
    // O at origin, H at (+-0.757, 0.586, 0) Angstrom.
    auto water = MakeMolecule({Atom{"O", 8, 0.0}, Atom{"H", 1, 0.0}, Atom{"H", 1, 0.0}},
                              {{0.0, 0.0, 0.0},
                               {0.757 * kAngstromToBohr, 0.586 * kAngstromToBohr, 0.0},
                               {-0.757 * kAngstromToBohr, 0.586 * kAngstromToBohr, 0.0}});
    ASSERT_TRUE(water.has_value());
    const auto connectivity = BuildConnectivity(*water);
    EXPECT_EQ(boost::num_edges(connectivity.graph), 2);
    // Canonical order H, H, O: the O atom (index 2) is the hub.
    EXPECT_EQ(connectivity.csr.offsets[3] - connectivity.csr.offsets[2], 2u);
    CheckCsrMatchesGraph(connectivity, 3);
}

TEST(ConnectivityTest, MethaneHasFourBonds) {
    // C at the origin, H at tetrahedral corners (1.09 Angstrom bonds).
    auto methane = MakeMolecule(
        {Atom{"C", 6, 0.0},
         Atom{"H", 1, 0.0},
         Atom{"H", 1, 0.0},
         Atom{"H", 1, 0.0},
         Atom{"H", 1, 0.0}},
        {{0.0, 0.0, 0.0},
         {0.629 * kAngstromToBohr, 0.629 * kAngstromToBohr, 0.629 * kAngstromToBohr},
         {-0.629 * kAngstromToBohr, -0.629 * kAngstromToBohr, 0.629 * kAngstromToBohr},
         {0.629 * kAngstromToBohr, -0.629 * kAngstromToBohr, -0.629 * kAngstromToBohr},
         {-0.629 * kAngstromToBohr, 0.629 * kAngstromToBohr, -0.629 * kAngstromToBohr}});
    ASSERT_TRUE(methane.has_value());
    const auto connectivity = BuildConnectivity(*methane);
    EXPECT_EQ(boost::num_edges(connectivity.graph), 4);

    for (std::size_t i = 0; i < 5; ++i)
    {
        const std::size_t degree = connectivity.csr.offsets[i + 1] - connectivity.csr.offsets[i];
        EXPECT_TRUE(degree == 1 || degree == 4);
    }

    CheckCsrMatchesGraph(connectivity, 5);
}

TEST(ConnectivityTest, EtheneHasFiveBonds) {
    // Planar C2H4: C=C 1.33 Angstrom, C-H 1.09 Angstrom.
    auto ethene = MakeMolecule({Atom{"C", 6, 0.0},
                                Atom{"C", 6, 0.0},
                                Atom{"H", 1, 0.0},
                                Atom{"H", 1, 0.0},
                                Atom{"H", 1, 0.0},
                                Atom{"H", 1, 0.0}},
                               {{0.665 * kAngstromToBohr, 0.0, 0.0},
                                {-0.665 * kAngstromToBohr, 0.0, 0.0},
                                {1.235 * kAngstromToBohr, 0.925 * kAngstromToBohr, 0.0},
                                {1.235 * kAngstromToBohr, -0.925 * kAngstromToBohr, 0.0},
                                {-1.235 * kAngstromToBohr, 0.925 * kAngstromToBohr, 0.0},
                                {-1.235 * kAngstromToBohr, -0.925 * kAngstromToBohr, 0.0}});
    ASSERT_TRUE(ethene.has_value());
    const auto connectivity = BuildConnectivity(*ethene);
    EXPECT_EQ(boost::num_edges(connectivity.graph), 5);
    CheckCsrMatchesGraph(connectivity, 6);
}

TEST(ConnectivityTest, FluorineDimerHasOneBond) {
    auto f2 =
        MakeMolecule({Atom{"F", 9, 0.0}, Atom{"F", 9, 0.0}},
                     {{0.71 * kAngstromToBohr, 0.0, 0.0}, {-0.71 * kAngstromToBohr, 0.0, 0.0}});
    ASSERT_TRUE(f2.has_value());
    EXPECT_EQ(boost::num_edges(BuildConnectivity(*f2).graph), 1);
}

TEST(ConnectivityTest, ElementWithoutRadiusNeverBonds) {
    // Bk has no published covalent radius (0 in the table): never bonds.
    auto bk2 = MakeMolecule({Atom{"Bk", 97, 0.0}, Atom{"Bk", 97, 0.0}},
                            {{0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}});
    ASSERT_TRUE(bk2.has_value());
    EXPECT_EQ(boost::num_edges(BuildConnectivity(*bk2).graph), 0);
}

TEST(ConnectivityTest, ToleranceIsRespected) {
    // H-H at 1.5 Bohr: bonded with the default 0.3 Angstrom tolerance, not with 0.2 Bohr.
    auto h2 =
        MakeMolecule({Atom{"H", 1, 0.0}, Atom{"H", 1, 0.0}}, {{0.75, 0.0, 0.0}, {-0.75, 0.0, 0.0}});
    ASSERT_TRUE(h2.has_value());
    EXPECT_EQ(boost::num_edges(BuildConnectivity(*h2).graph), 1);
    EXPECT_EQ(boost::num_edges(BuildConnectivity(*h2, 0.2).graph), 0);
}

TEST(ConnectivityTest, BuckminsterfullereneHasNinetyEdges) {
    auto c60 = testing::MakeBuckminsterfullerene();
    ASSERT_TRUE(c60.has_value()) << c60.error().message;
    EXPECT_EQ(c60->AtomCount(), 60u);
    const auto connectivity = BuildConnectivity(*c60);
    EXPECT_EQ(boost::num_edges(connectivity.graph), 90);

    for (std::size_t i = 0; i < 60; ++i)
    {
        EXPECT_EQ(boost::degree(i, connectivity.graph), 3);
    }

    CheckCsrMatchesGraph(connectivity, 60);
}

} // namespace
} // namespace qcx::molecule
