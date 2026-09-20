// The alkane fixture's non-C12 contract: every QFMM call site uses
// carbonCount = 12 (qfmm_fock_build_test.cpp, qfmm_monopole_test.cpp,
// qfmm_lmult_test.cpp), so the kInvalidArgument error path and the
// small-carbon geometry branches are only pinned here - a geometry bug in
// the methane/ethane branches would otherwise pass every suite.
// Test-only helper - not part of the public API.

#include "alkane_sto3g.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

namespace {

// Bohr per Angstrom (CODATA 2018), mirrored from the fixture.
constexpr double kBohrPerAngstrom = 1.889726125457828;

// The fixture's bond constants (alkane_sto3g.cpp:85-86): C-C 1.538 A and
// C-H 1.09 A, both in Bohr.
constexpr double kCCBohr = 2.9066;
constexpr double kCHBohr = 1.09 * kBohrPerAngstrom;

// Distance between canonical atom rows a and b, Bohr.
double Distance(const qcx::molecule::Molecule& molecule, std::size_t a, std::size_t b) {
    double d2 = 0.0;

    for (int d = 0; d < 3; ++d)
    {
        const double diff = molecule.CoordinatesBohr()(a, d) - molecule.CoordinatesBohr()(b, d);
        d2 += diff * diff;
    }

    return std::sqrt(d2);
}

} // namespace

TEST(AlkaneFixtureTest, ZeroCarbonsIsRejected) {
    const auto molecule = qcx::testing::MakeAlkaneSto3g(0);

    ASSERT_FALSE(molecule.has_value());
    EXPECT_EQ(molecule.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(AlkaneFixtureTest, MethaneIsTetrahedral) {
    const auto molecule = qcx::testing::MakeAlkaneSto3g(1);
    ASSERT_TRUE(molecule.has_value());
    ASSERT_EQ(molecule->AtomCount(), 5);

    // Canonical renumbering is by Z then position: the four hydrogens
    // (Z = 1) precede the single carbon (Z = 6).
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(molecule->Atoms()[i].atomicNumber, 1);
        EXPECT_NEAR(Distance(*molecule, i, 4), kCHBohr, 1e-9);
    }

    EXPECT_EQ(molecule->Atoms()[4].atomicNumber, 6);

    // Perfect tetrahedron: all six H-H edges equal R sqrt(8/3) for the
    // circumradius R = C-H, and the H-C-H angle is arccos(-1/3).
    const double edge = kCHBohr * std::sqrt(8.0 / 3.0);

    for (std::size_t a = 0; a < 4; ++a)
    {
        for (std::size_t b = a + 1; b < 4; ++b)
        {
            EXPECT_NEAR(Distance(*molecule, a, b), edge, 1e-9);
        }
    }

    // (H0 - C) . (H1 - C) = R^2 cos(109.47 deg) = -R^2 / 3.
    double dot = 0.0;

    for (int d = 0; d < 3; ++d)
    {
        dot += (molecule->CoordinatesBohr()(0, d) - molecule->CoordinatesBohr()(4, d)) *
               (molecule->CoordinatesBohr()(1, d) - molecule->CoordinatesBohr()(4, d));
    }

    EXPECT_NEAR(dot, -kCHBohr * kCHBohr / 3.0, 1e-9);
}

TEST(AlkaneFixtureTest, EthaneIsEightAtomsWithOneCCBond) {
    const auto molecule = qcx::testing::MakeAlkaneSto3g(2);
    ASSERT_TRUE(molecule.has_value());
    ASSERT_EQ(molecule->AtomCount(), 8);

    // Canonical order: the six hydrogens (Z = 1), then the two carbons
    // (Z = 6) at the last two rows.
    std::size_t carbons = 0;

    for (std::size_t i = 0; i < molecule->AtomCount(); ++i)
    {
        carbons += molecule->Atoms()[i].atomicNumber == 6 ? 1 : 0;
    }

    EXPECT_EQ(carbons, 2);
    EXPECT_NEAR(Distance(*molecule, 6, 7), kCCBohr, 1e-9);
}
