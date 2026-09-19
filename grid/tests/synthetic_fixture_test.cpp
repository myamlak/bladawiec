// The synthetic chain fixture (synthetic_fixture.hpp) is the system the
// screening gate measures its O(nGrid x nSig) complexity bound
// on, so these pins are ordered by what would silently disable the gate:
//
//   1. SPATIAL EXTENT - a chain that collapses, clamps or wraps would let
//      a screen pass while proving nothing.  The floor is the exact
//      end-to-end length of the requested chain, not a loose one, so a
//      compact geometry fails rather than skates through.
//   2. COUNTS - one AO and one shell per atom, which is the "atomCount
//      functions" contract a caller sizes its buffers from.
//   3. DETERMINISM - two builds of the same chain agree on every
//      coordinate exactly, so a gate measurement is attributable to the
//      algorithm rather than to the fixture.
//
// The extent pin reads the coordinates back rather than trusting the
// generator: Molecule::Create renumbers its atoms canonically (by Z, then
// position), so the endpoints are found by their position, never by row
// index.

#include "qcx/grid/ao_evaluator.hpp"
#include "synthetic_fixture.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

namespace qcx::grid {
namespace {

// The generator's documented contract, restated here so the pins fail
// loudly if the fixture's spacing, basis or geometry contract moves.
constexpr double kChainSpacingBohr = 1.4;
constexpr double kExtentSlackFraction = 0.99;

// The size the screening gate is calibrated on.
constexpr std::size_t kGateAtomCount = 5000;

// The distance between the two ends of the chain.
//
// The chain is collinear along x, so the endpoints are the atoms with the
// smallest and largest x; every coordinate is read through
// CoordinatesBohr(), which is the geometry the fixture actually produced.
double EndpointDistance(const qcx::molecule::Molecule& molecule) {
    const auto& coordinates = molecule.CoordinatesBohr();
    std::size_t lowestIndex = 0;
    std::size_t highestIndex = 0;

    for (std::size_t atom = 1; atom < molecule.AtomCount(); ++atom)
    {
        if (coordinates(atom, 0) < coordinates(lowestIndex, 0))
        {
            lowestIndex = atom;
        }

        if (coordinates(atom, 0) > coordinates(highestIndex, 0))
        {
            highestIndex = atom;
        }
    }

    double squaredDistance = 0.0;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const double delta = coordinates(highestIndex, axis) - coordinates(lowestIndex, axis);
        squaredDistance += delta * delta;
    }

    return std::sqrt(squaredDistance);
}

// The end-to-end length the chain of this size must reach: the exact
// generated length, less the fixture contract's 1% slack.
double RequiredExtentBohr(std::size_t atomCount) {
    return kExtentSlackFraction * static_cast<double>(atomCount - 1) * kChainSpacingBohr;
}

TEST(SyntheticFixtureTest, ExtentTracksTheAtomCount) {
    // The gate size first, then a small chain: the pair catches both a
    // collapsed geometry (the gate size would fail on its own) and a
    // generator that ignores its argument or clamps the total length.
    constexpr std::array<std::size_t, 3> kSizes = {kGateAtomCount, 32, 3};

    for (const std::size_t atomCount : kSizes)
    {
        const auto system = MakeSyntheticChain(atomCount);
        ASSERT_TRUE(system.has_value()) << system.error().message;
        ASSERT_EQ(system->molecule.AtomCount(), atomCount);

        const double distance = EndpointDistance(system->molecule);
        const double required = RequiredExtentBohr(atomCount);
        EXPECT_GE(distance, required)
            << "atomCount " << atomCount << ": end-to-end distance " << distance
            << " Bohr, required at least " << required << " Bohr";
    }
}

TEST(SyntheticFixtureTest, CountsAreOneFunctionPerAtom) {
    const auto system = MakeSyntheticChain(kGateAtomCount);
    ASSERT_TRUE(system.has_value()) << system.error().message;

    const auto evaluator = AoEvaluator::Create(system->molecule, system->basis);
    ASSERT_TRUE(evaluator.has_value()) << evaluator.error().message;
    EXPECT_EQ(evaluator->AOCount(), kGateAtomCount);
    EXPECT_EQ(evaluator->ShellRanges().size(), kGateAtomCount);
}

TEST(SyntheticFixtureTest, RepeatedBuildsAgreeOnEveryCoordinate) {
    const auto first = MakeSyntheticChain(kGateAtomCount);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    const auto second = MakeSyntheticChain(kGateAtomCount);
    ASSERT_TRUE(second.has_value()) << second.error().message;

    ASSERT_EQ(first->molecule.AtomCount(), kGateAtomCount);
    ASSERT_EQ(second->molecule.AtomCount(), kGateAtomCount);

    const auto& firstCoordinates = first->molecule.CoordinatesBohr();
    const auto& secondCoordinates = second->molecule.CoordinatesBohr();

    for (std::size_t atom = 0; atom < kGateAtomCount; ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_EQ(firstCoordinates(atom, axis), secondCoordinates(atom, axis))
                << "atom " << atom << " axis " << axis;
        }
    }
}

} // namespace
} // namespace qcx::grid
