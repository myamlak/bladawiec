// Generated periodic table: structural invariants and spot-checked published values.
#include "qcx/molecule/elements.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace qcx::molecule {
namespace {

TEST(ElementsTest, HasAll118Elements) {
    EXPECT_EQ(kElements.size(), 118u);
    EXPECT_EQ(kElementCount, 118u);
}

TEST(ElementsTest, SpotCheckKnownElements) {
    const auto* h = FindElement(1);
    const auto* c = FindElement(6);
    const auto* fe = FindElement(26);
    const auto* u = FindElement(92);
    const auto* og = FindElement(118);
    ASSERT_NE(h, nullptr);
    ASSERT_NE(c, nullptr);
    ASSERT_NE(fe, nullptr);
    ASSERT_NE(u, nullptr);
    ASSERT_NE(og, nullptr);
    EXPECT_EQ(h->symbol, "H");
    EXPECT_EQ(h->name, "hydrogen");
    EXPECT_EQ(c->symbol, "C");
    EXPECT_EQ(fe->symbol, "Fe");
    EXPECT_EQ(u->symbol, "U");
    EXPECT_EQ(og->symbol, "Og");
    EXPECT_NEAR(fe->standardAtomicWeight, 55.845, 1e-2);
    EXPECT_GT(c->covalentRadiusBohr, 0.0);
    EXPECT_GT(c->vanDerWaalsRadiusBohr, 0.0);
}

TEST(ElementsTest, FindElementBySymbolIsInverseOfZ) {
    for (std::size_t i = 0; i < kElements.size(); ++i)
    {
        EXPECT_EQ(FindElement(kElements[i].symbol), &kElements[i]);
        EXPECT_EQ(FindElement(kElements[i].atomicNumber), &kElements[i]);
    }
}

TEST(ElementsTest, UnknownElementsRejected) {
    EXPECT_EQ(FindElement(0), nullptr);
    EXPECT_EQ(FindElement(119), nullptr);
    EXPECT_EQ(FindElement("Xx"), nullptr);
    EXPECT_EQ(FindElement("h"), nullptr); // exact case only
}

TEST(ElementsTest, UnpublishedCovalentRadiiAreZero) {
    // Cordero 2008 stops at Cm; everything beyond has radius 0 ("never bonds").
    EXPECT_DOUBLE_EQ(FindElement(97)->covalentRadiusBohr, 0.0);
    EXPECT_GT(FindElement(96)->covalentRadiusBohr, 0.0);
    EXPECT_GT(FindElement(2)->covalentRadiusBohr, 0.0); // He is published (0.28 Angstrom)
}

TEST(ElementsTest, IsotopeOffsetsCoverTheFlatArray) {
    std::size_t offset = 0;

    for (const auto& e : kElements)
    {
        EXPECT_EQ(e.isotopeOffset, offset);
        offset += e.isotopeCount;
        // A zero isotope count means "no measured isotope masses" (Cn, Nh,
        // Fl, Mc, Lv, Ts, Og); the default mass is then the standard weight.

        if (e.isotopeCount == 0)
        {
            EXPECT_DOUBLE_EQ(e.mostAbundantIsotopeMass, e.standardAtomicWeight);
        }

        for (std::size_t k = 0; k + 1 < e.isotopeCount; ++k)
        {
            EXPECT_LT(kIsotopes[e.isotopeOffset + k].mass, kIsotopes[e.isotopeOffset + k + 1].mass);
        }
    }

    EXPECT_EQ(offset, kIsotopes.size());
}

TEST(ElementsTest, VanDerWaalsRadiiFromAlvarez) {
    // Published Alvarez 2013 values (Angstrom, converted at generation time).
    EXPECT_NEAR(FindElement(1)->vanDerWaalsRadiusBohr, 1.20 * 1.8897261246257702, 1e-9);
    EXPECT_NEAR(FindElement(6)->vanDerWaalsRadiusBohr, 1.77 * 1.8897261246257702, 1e-9);
    EXPECT_GT(FindElement(92)->vanDerWaalsRadiusBohr, 0.0);
}

TEST(ElementsTest, HydrogenDefaultsToProtium) {
    const auto* h = FindElement("H");
    ASSERT_NE(h, nullptr);
    EXPECT_NEAR(h->standardAtomicWeight, 1.008, 1e-3);
    EXPECT_DOUBLE_EQ(h->mostAbundantIsotopeMass, 1.0078250321);
    EXPECT_GE(h->isotopeCount, 2u);
    EXPECT_DOUBLE_EQ(kIsotopes[h->isotopeOffset].mass, 1.0078250321);
    EXPECT_DOUBLE_EQ(kIsotopes[h->isotopeOffset].abundance, 99.9885);
    EXPECT_NEAR(kIsotopes[h->isotopeOffset + 1].mass, 2.014101778, 1e-9);
}

TEST(ElementsTest, XenonHasTheRareIsotopes) {
    const auto* xe = FindElement(54);
    ASSERT_NE(xe, nullptr);

    for (double rareMass : {135.90722, 129.9035079})
    {
        bool found = false;

        for (std::size_t k = 0; k < xe->isotopeCount; ++k)
        {
            if (std::abs(kIsotopes[xe->isotopeOffset + k].mass - rareMass) < 1e-6)
            {
                found = true;
            }
        }

        EXPECT_TRUE(found) << rareMass;
    }
}

TEST(ElementsTest, SuperheavyElementsFallBackToStandardWeight) {
    // No measured abundances for Cn/Og: the default isotope mass is the standard weight.
    EXPECT_DOUBLE_EQ(FindElement(112)->mostAbundantIsotopeMass,
                     FindElement(112)->standardAtomicWeight);
    EXPECT_GT(FindElement(118)->mostAbundantIsotopeMass, 0.0);
}

} // namespace
} // namespace qcx::molecule
