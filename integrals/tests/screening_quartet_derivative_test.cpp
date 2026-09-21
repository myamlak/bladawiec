// The quartet-level product companion of the derivative-aware screen's
// tests: the product rule the companion applies, the checkpoint case one
// level up - a quartet whose value bound is under the threshold while its
// derivative bound is not, kept by the derivative rule and dropped by a
// value-only one - and the companion's value half against the engine, where
// Q_ab Q_cd must dominate every quartet of the fixture's pair list.
//
// The dominance check is the Cauchy-Schwarz statement the one-electron screen
// rests on too, one level up: with Q_ab the largest diagonal element of the
// pair's own (ab|ab) block, |(ab|cd)| <= Q_ab Q_cd for every function
// quadruple. The test measures it on the engine's own blocks rather than
// assuming it.
#include "h2o_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/molecule/molecule.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <span>
#include <utility>
#include <vector>

namespace {

using qcx::integrals::ScreenedQuartet;
using qcx::integrals::ShellPairIndex;
using qcx::integrals::ShellPairList;
using qcx::integrals::ShellQuartet;
using qcx::integrals::TwoElectronPairBound;

/// The function count of one shell of the pair list.
std::size_t ShellFunctions(const qcx::integrals::ShellInfo& shell) {
    const int angular = shell.isSpherical
                            ? shell.angularMomentum * 2 + 1
                            : (shell.angularMomentum + 1) * (shell.angularMomentum + 2) / 2;
    return shell.contractionCount * static_cast<std::size_t>(angular);
}

/// The packed-block element count of one quartet.
std::size_t QuartetElements(const ShellPairList& pairList, const ShellQuartet& quartet) {
    return ShellFunctions(pairList.shells[quartet.i]) * ShellFunctions(pairList.shells[quartet.j]) *
           ShellFunctions(pairList.shells[quartet.k]) * ShellFunctions(pairList.shells[quartet.l]);
}

/// The largest element magnitude of a block.
double LargestMagnitude(std::span<const double> block) {
    double largest = 0.0;

    for (const double value : block)
    {
        largest = std::max(largest, std::abs(value));
    }

    return largest;
}

/// The fixture's water molecule and STO-3G basis, with its pair list.
struct Fixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    ShellPairList pairList;
};

qcx::Result<Fixture> MakeFixture() {
    auto molecule = qcx::testing::MakeH2oSto3g();

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto basis = qcx::testing::MakeH2oSto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    return Fixture{std::move(*molecule), std::move(*basis), std::move(*pairList)};
}

/// The Schwarz bound of every canonical pair: the square root of the largest
/// diagonal element of the pair's own (ab|ab) block, read off the engine's
/// blocks.
/// \param fixture The fixture.
/// \returns One bound per canonical pair, or an Error.
qcx::Result<std::vector<double>> DiagonalBounds(const Fixture& fixture) {
    std::vector<ShellQuartet> diagonal;
    diagonal.reserve(fixture.pairList.pairs.size());

    for (const ShellPairIndex& pair : fixture.pairList.pairs)
    {
        diagonal.push_back(ShellQuartet{pair.i, pair.j, pair.i, pair.j});
    }

    auto batch = qcx::integrals::ComputeEriBatch(fixture.molecule, fixture.basis, diagonal);

    if (!batch.has_value())
    {
        return std::unexpected(batch.error());
    }

    std::vector<double> bounds(fixture.pairList.pairs.size(), 0.0);
    std::size_t offset = 0;

    for (std::size_t computed = 0; computed < batch->computed.size(); ++computed)
    {
        const std::size_t elements = QuartetElements(fixture.pairList, batch->computed[computed]);
        const double largest = LargestMagnitude(std::span<const double>(
            batch->values.data() + static_cast<std::ptrdiff_t>(offset), elements));
        // The diagonal quartet's block is (ij | ij): its largest element is
        // the pair's Schwarz numerator, squared-free.
        const std::size_t braPair = qcx::integrals::PairIndexOf(
            batch->computed[computed].i, batch->computed[computed].j, fixture.pairList);
        bounds[braPair] = std::max(bounds[braPair], std::sqrt(largest));
        offset += elements;
    }

    return bounds;
}

TEST(ScreeningQuartetDerivativeTest, TheProductBoundIsTheProductRuleOfItsTwoPairs) {
    const TwoElectronPairBound bra{2.0, 3.0};
    const TwoElectronPairBound ket{5.0, 7.0};
    const qcx::integrals::QuartetDerivativeBound bound =
        qcx::integrals::QuartetDerivativeProduct(bra, ket);
    // The value is the product of the two pair values; the derivative is the
    // product rule's two terms, one per pair the derivative may act on.
    EXPECT_DOUBLE_EQ(bound.value, 10.0);
    EXPECT_DOUBLE_EQ(bound.derivative, 3.0 * 5.0 + 2.0 * 7.0);
}

TEST(ScreeningQuartetDerivativeTest, TheScreenKeepsAQuartetWhoseDerivativeAloneIsAbove) {
    // A quartet whose value bound sits under the threshold while its
    // derivative bound does not: absent from the energy, present in the
    // gradient, and dropped by every screen written for the energy.
    const double threshold = 1.0e-10;
    const TwoElectronPairBound quiet{3.0e-6, 3.0e-6};
    const TwoElectronPairBound loud{1.0e-5, 1.0e-2};
    const qcx::integrals::QuartetDerivativeBound valueOnly =
        qcx::integrals::QuartetDerivativeProduct(quiet, quiet);
    const qcx::integrals::QuartetDerivativeBound kept =
        qcx::integrals::QuartetDerivativeProduct(quiet, loud);
    EXPECT_LE(valueOnly.value, threshold);
    EXPECT_LE(valueOnly.derivative, threshold);
    EXPECT_FALSE(qcx::integrals::SurvivesQuartetDerivativeScreen(valueOnly, threshold));
    // The kept quartet's VALUE bound is under the threshold too: only its
    // derivative bound is over it.
    EXPECT_LE(kept.value, threshold);
    EXPECT_GT(kept.derivative, threshold);
    EXPECT_TRUE(qcx::integrals::SurvivesQuartetDerivativeScreen(kept, threshold));

    const std::vector<TwoElectronPairBound> bounds = {quiet, loud};
    const std::vector<ScreenedQuartet> candidates = {
        ScreenedQuartet{0, 0}, ScreenedQuartet{0, 1}, ScreenedQuartet{1, 1}};
    const std::vector<ScreenedQuartet> retained =
        qcx::integrals::RetainedQuartets(bounds, candidates, threshold);
    ASSERT_EQ(retained.size(), 2u);
    EXPECT_EQ(retained[0].braPair, 0u);
    EXPECT_EQ(retained[0].ketPair, 1u);
    EXPECT_EQ(retained[1].braPair, 1u);
    EXPECT_EQ(retained[1].ketPair, 1u);
}

TEST(ScreeningQuartetDerivativeTest, TheProductBoundDominatesEveryQuartetOfTheFixture) {
    auto fixture = MakeFixture();
    ASSERT_TRUE(fixture.has_value()) << fixture.error().message;
    auto bounds = DiagonalBounds(*fixture);
    ASSERT_TRUE(bounds.has_value()) << bounds.error().message;
    std::vector<TwoElectronPairBound> pairBounds;

    for (const double bound : *bounds)
    {
        pairBounds.push_back(TwoElectronPairBound{bound, 0.0});
    }

    std::vector<ShellQuartet> requests;

    for (const ShellPairIndex& bra : fixture->pairList.pairs)
    {
        for (const ShellPairIndex& ket : fixture->pairList.pairs)
        {
            requests.push_back(ShellQuartet{bra.i, bra.j, ket.i, ket.j});
        }
    }

    auto order = qcx::integrals::CanonicalizeQuartetOrder(fixture->pairList, requests);
    ASSERT_TRUE(order.has_value()) << order.error().message;
    auto batch = qcx::integrals::ComputeEriBatch(fixture->molecule, fixture->basis, requests);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    ASSERT_EQ(order->size(), batch->computed.size());
    std::size_t offset = 0;
    std::size_t checked = 0;

    for (std::size_t computed = 0; computed < batch->computed.size(); ++computed)
    {
        const std::size_t elements = QuartetElements(fixture->pairList, batch->computed[computed]);
        const double largest = LargestMagnitude(std::span<const double>(
            batch->values.data() + static_cast<std::ptrdiff_t>(offset), elements));
        const std::size_t braPair = (*order)[computed].braPair;
        const std::size_t ketPair = (*order)[computed].ketPair;
        const qcx::integrals::QuartetDerivativeBound bound =
            qcx::integrals::QuartetDerivativeProduct(pairBounds[braPair], pairBounds[ketPair]);
        EXPECT_GE(bound.value * (1.0 + 1.0e-12), largest)
            << "pair " << braPair << " x pair " << ketPair << ": bound " << bound.value
            << " against a measured " << largest;
        ++checked;
        offset += elements;
    }

    EXPECT_EQ(checked, requests.size());
}

} // namespace
