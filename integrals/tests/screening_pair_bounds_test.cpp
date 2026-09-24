// The two-electron pair-bound producer's tests (screening.hpp
// ComputeTwoElectronPairBounds) - the pair numbers the derivative-aware
// screen multiplies.
//
// The value half is checked against two independent measurements of the same
// number: the Schwarz sweep's own pass (a different production walk, through
// the fp64 batch pipeline) and the engine's blocks read element by element.
// The derivative half is checked against the MEASURED derivative of the bound
// it bounds - a central difference of the pair's own Schwarz bound under a
// displacement of the pair's own centre coordinates - and then at the level
// the screen actually uses it, against the differentiated quartets of the
// fixture's pair list.
//
// The distinction the producer exists for is between the derivative OF the
// bound and the bound OF the derivative: a pair's bound and its derivative are
// two different numbers, and a screen that read the value bound as if IT
// bounded the derivative would drop every quartet whose value is small while
// its motion is not - the failure the derivative-aware screen exists to
// prevent.
#include "h2o_sto3g.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_eri_derivative.hpp"
#include "internal/shells_flat.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::integrals::ShellInfo;
using qcx::integrals::ShellPairIndex;
using qcx::integrals::ShellPairList;
using qcx::integrals::ShellQuartet;
using qcx::integrals::TwoElectronPairBound;
using qcx::integrals::internal::MdEriDerivativeQuartet;
using qcx::integrals::internal::MdEriDerivativeScratch;
using qcx::integrals::internal::MdShellInput;

/// The central-difference step of the pair-bound check: O(h^2) truncation
/// against an O(eps Q / h) cancellation error, near 1e-12 at this fixture's
/// O(1) bounds.
constexpr double kFiniteDifferenceStep = 1e-5;

/// The slack the never-under comparisons carry: the producer's bound and the
/// measurement sum the same products in a different order, so the bound may
/// sit a roundoff under the measured value and no more.
constexpr double kSlack = 1.0e-12;

/// The extra allowance a FINITE-DIFFERENCE comparison carries: the central
/// difference of an O(1) bound at this step has an O(h^2) truncation and an
/// O(eps / h) cancellation error near 1e-11 of its own, so a measured
/// derivative that sits that far above the analytic bound is the measurement.
constexpr double kMeasurementSlack = 1.0e-9;

/// The absolute allowance an exact block comparison carries: an element that
/// is zero by symmetry comes out of the builder at its own roundoff, and a
/// bound that is exactly zero has no relative form to carry it.
constexpr double kZeroBoundSlack = 1.0e-12;

/// The tier's own acceptance: the analytic derivative block against the central
/// difference of the value block the same tier builds, over the whole fixture's
/// quartet space. An O(1) block's difference carries the step's O(h^2) and the
/// element magnitudes' roundoff, so the deviation is read at 1e-8.
constexpr double kDerivativeConsistencyTolerance = 1.0e-8;

/// The function count of one shell of the pair list.
std::size_t ShellFunctions(const ShellInfo& shell) {
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

/// A double as a short scientific string, for the recorded properties.
std::string ShortDouble(double value) {
    char text[32] = {};
    std::snprintf(text, sizeof(text), "%.3e", value);
    return text;
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

/// The geometry with one flat coordinate displaced; errors when the canonical
/// renumbering would move an atom.
/// \param base The base molecule.
/// \param flatCoordinate The coordinate (3 * atom + axis) to displace.
/// \param delta The displacement (Bohr).
/// \returns The displaced molecule, or an Error.
qcx::Result<qcx::molecule::Molecule> MakeDisplaced(const qcx::molecule::Molecule& base,
                                                   std::size_t flatCoordinate,
                                                   double delta) {
    auto coordinates = CpuTensor2::Create({base.AtomCount(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    for (std::size_t atom = 0; atom < base.AtomCount(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            (*coordinates)(atom, axis) = base.CoordinatesBohr()(atom, axis);
        }
    }

    (*coordinates)(flatCoordinate / 3, flatCoordinate % 3) += delta;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        base.Atoms(), std::move(*coordinates), base.Charge(), base.Multiplicity());
}

/// The Schwarz bound of every canonical pair, measured off the engine's own
/// (ab|ab) blocks rather than read from any bound-producing pass.
/// \param molecule The molecule.
/// \param basis The basis set.
/// \param pairList Its pair list.
/// \returns One bound per canonical pair, or an Error.
qcx::Result<std::vector<double>> MeasuredDiagonalBounds(const qcx::molecule::Molecule& molecule,
                                                        const qcx::basisset::BasisSet& basis,
                                                        const ShellPairList& pairList) {
    std::vector<ShellQuartet> diagonal;
    diagonal.reserve(pairList.pairs.size());

    for (const ShellPairIndex& pair : pairList.pairs)
    {
        diagonal.push_back(ShellQuartet{pair.i, pair.j, pair.i, pair.j});
    }

    auto batch = qcx::integrals::ComputeEriBatch(molecule, basis, diagonal);

    if (!batch.has_value())
    {
        return std::unexpected(batch.error());
    }

    std::vector<double> bounds(pairList.pairs.size(), 0.0);
    std::size_t offset = 0;

    for (const ShellQuartet& computed : batch->computed)
    {
        const std::size_t elements = QuartetElements(pairList, computed);
        const double largest = LargestMagnitude(std::span<const double>(
            batch->values.data() + static_cast<std::ptrdiff_t>(offset), elements));
        const std::size_t braPair = qcx::integrals::PairIndexOf(computed.i, computed.j, pairList);
        bounds[braPair] = std::max(bounds[braPair], std::sqrt(largest));
        offset += elements;
    }

    return bounds;
}

/// The distinct atoms a quartet's four shells sit on, three flat coordinates
/// (3 * atom + axis) each, in ascending atom order.
/// \param atoms The quartet's four atoms.
/// \returns The flat coordinates.
std::vector<std::size_t> QuartetCoordinates(const std::array<std::size_t, 4>& atoms) {
    std::array<std::size_t, 4> distinct = atoms;
    std::sort(distinct.begin(), distinct.end());
    const auto last = std::unique(distinct.begin(), distinct.end());
    std::vector<std::size_t> coordinates;

    for (auto atom = distinct.begin(); atom != last; ++atom)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            coordinates.push_back(3 * (*atom) + static_cast<std::size_t>(axis));
        }
    }

    return coordinates;
}

/// One quartet at a displaced coordinate: the shells sitting on the
/// coordinate's atom move, the others stand. A quartet's value depends on its
/// shells' centres and exponents alone, so this is the quartet at the
/// displaced geometry, reached without rebuilding the molecule.
/// \param quartet The quartet.
/// \param flatCoordinate The coordinate (3 * atom + axis) to displace.
/// \param delta The displacement (Bohr).
/// \param storage The displaced shells; must outlive the result.
/// \returns The displaced quartet.
MdEriDerivativeQuartet DisplacedQuartet(const MdEriDerivativeQuartet& quartet,
                                        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                        std::size_t flatCoordinate,
                                        double delta,
                                        std::array<MdShellInput, 4>& storage) {
    for (std::size_t slot = 0; slot < 4; ++slot)
    {
        storage[slot] = *quartet.shells[slot];

        if (quartet.atoms[slot] != flatCoordinate / 3)
        {
            continue;
        }

        switch (flatCoordinate % 3)
        {
        case 0:
            storage[slot].cx += delta;
            break;
        case 1:
            storage[slot].cy += delta;
            break;
        default:
            storage[slot].cz += delta;
            break;
        }
    }

    MdEriDerivativeQuartet displaced;
    displaced.atoms = quartet.atoms;

    for (std::size_t slot = 0; slot < 4; ++slot)
    {
        displaced.shells[slot] = &storage[slot];
    }

    return displaced;
}

} // namespace

/// The producer's value half is the Schwarz bound: the same number the
/// sweeping pass reaches from the other direction, and the same number the
/// engine's blocks carry.
TEST(ScreeningPairBoundsTest, TheValueHalfIsTheSchwarzBoundTwoWays) {
    auto fixture = MakeFixture();
    ASSERT_TRUE(fixture.has_value()) << fixture.error().message;
    auto produced = qcx::integrals::ComputeTwoElectronPairBounds(fixture->molecule, fixture->basis);
    ASSERT_TRUE(produced.has_value()) << produced.error().message;
    auto sweep = qcx::integrals::ComputeSchwarzBounds(fixture->molecule, fixture->basis);
    ASSERT_TRUE(sweep.has_value()) << sweep.error().message;
    auto measured = MeasuredDiagonalBounds(fixture->molecule, fixture->basis, fixture->pairList);
    ASSERT_TRUE(measured.has_value()) << measured.error().message;
    ASSERT_EQ(produced->size(), fixture->pairList.pairs.size());
    ASSERT_EQ(sweep->size(), fixture->pairList.pairs.size());
    ASSERT_EQ(measured->size(), fixture->pairList.pairs.size());

    double worstSweep = 0.0;
    double worstMeasured = 0.0;

    for (std::size_t pair = 0; pair < produced->size(); ++pair)
    {
        worstSweep = std::max(worstSweep, std::abs((*produced)[pair].value - (*sweep)[pair]));
        worstMeasured =
            std::max(worstMeasured, std::abs((*produced)[pair].value - (*measured)[pair]));
        // The producer's bound dominates the measured block by construction;
        // a bound under it would screen away a contribution that is there.
        EXPECT_GE((*produced)[pair].value * (1.0 + kSlack), (*measured)[pair]) << "pair " << pair;
    }

    RecordProperty("worst_against_sweep", ShortDouble(worstSweep));
    RecordProperty("worst_against_measured", ShortDouble(worstMeasured));
    RecordProperty("pairs", std::to_string(produced->size()));
    EXPECT_LT(worstSweep, 1.0e-12);
    EXPECT_LT(worstMeasured, 1.0e-12);
}

/// The producer's derivative half dominates the measured derivative of the
/// bound it bounds: the central difference of the pair's own Schwarz bound
/// under a displacement of the pair's own centre coordinates.
TEST(ScreeningPairBoundsTest, TheDerivativeHalfDominatesTheMeasuredBoundDerivative) {
    auto fixture = MakeFixture();
    ASSERT_TRUE(fixture.has_value()) << fixture.error().message;
    auto produced = qcx::integrals::ComputeTwoElectronPairBounds(fixture->molecule, fixture->basis);
    ASSERT_TRUE(produced.has_value()) << produced.error().message;

    double worstRatio = 0.0;
    double largestDerivative = 0.0;
    std::size_t checked = 0;
    std::size_t expected = 0;

    for (std::size_t pair = 0; pair < fixture->pairList.pairs.size(); ++pair)
    {
        const ShellPairIndex& shells = fixture->pairList.pairs[pair];
        const std::size_t braAtom = fixture->pairList.shells[shells.i].atomIndex;
        const std::size_t ketAtom = fixture->pairList.shells[shells.j].atomIndex;
        const std::array<std::size_t, 2> atoms = {braAtom, ketAtom};
        const std::vector<std::size_t> coordinates =
            QuartetCoordinates(std::array<std::size_t, 4>{braAtom, ketAtom, braAtom, ketAtom});
        EXPECT_GE((*produced)[pair].derivative, 0.0);
        largestDerivative = std::max(largestDerivative, (*produced)[pair].derivative);
        expected += coordinates.size();

        for (const std::size_t coordinate : coordinates)
        {
            auto plus = MakeDisplaced(fixture->molecule, coordinate, kFiniteDifferenceStep);
            auto minus = MakeDisplaced(fixture->molecule, coordinate, -kFiniteDifferenceStep);

            ASSERT_TRUE(plus.has_value()) << plus.error().message;
            ASSERT_TRUE(minus.has_value()) << minus.error().message;

            auto plusBounds = MeasuredDiagonalBounds(*plus, fixture->basis, fixture->pairList);
            auto minusBounds = MeasuredDiagonalBounds(*minus, fixture->basis, fixture->pairList);

            ASSERT_TRUE(plusBounds.has_value()) << plusBounds.error().message;
            ASSERT_TRUE(minusBounds.has_value()) << minusBounds.error().message;

            const double measured =
                ((*plusBounds)[pair] - (*minusBounds)[pair]) / (2.0 * kFiniteDifferenceStep);

            if ((*produced)[pair].derivative > 0.0)
            {
                worstRatio =
                    std::max(worstRatio, std::abs(measured) / (*produced)[pair].derivative);
            }

            EXPECT_LE(std::abs(measured),
                      (*produced)[pair].derivative * (1.0 + kSlack) + kMeasurementSlack)
                << "pair " << pair << " (" << atoms[0] << ", " << atoms[1] << ") coordinate "
                << coordinate << ": measured " << measured << " against a bound of "
                << (*produced)[pair].derivative;
            ++checked;
        }
    }

    RecordProperty("checked_coordinates", std::to_string(checked));
    RecordProperty("largest_derivative", ShortDouble(largestDerivative));
    RecordProperty("worst_ratio", ShortDouble(worstRatio));
    // A producer that returned zeros everywhere would pass a dominance check
    // by being trivially true only if the measurement were zero too; the
    // fixture's bounds move with the geometry, so the derivative half is a
    // real number on every pair.
    EXPECT_GT(largestDerivative, 0.0);
    EXPECT_EQ(checked, expected);
}

/// The derivative tier's own consistency at the quartet level: the derivative
/// blocks of a quartet are the derivative of the value block the same tier
/// builds, against the central difference of that value under a displacement of
/// the quartet's own coordinates. The producer, the fitted walk and the exact
/// reference all stand on this one contraction, so it is measured on its own
/// before any of them is read.
TEST(ScreeningPairBoundsTest, TheTierDerivativeIsTheDerivativeOfItsOwnValue) {
    auto fixture = MakeFixture();
    ASSERT_TRUE(fixture.has_value()) << fixture.error().message;
    auto shells = qcx::integrals::internal::FlattenShells(
        fixture->molecule, fixture->basis, fixture->pairList);
    ASSERT_TRUE(shells.has_value()) << shells.error().message;

    MdEriDerivativeScratch scratch;
    std::vector<double> value;
    std::vector<double> derivatives;
    std::vector<double> plus;
    std::vector<double> minus;
    double worstDeviation = 0.0;
    std::string worstAt;
    std::size_t checked = 0;

    for (std::size_t bra = 0; bra < fixture->pairList.pairs.size(); ++bra)
    {
        for (std::size_t ket = 0; ket < fixture->pairList.pairs.size(); ++ket)
        {
            const ShellPairIndex& braPair = fixture->pairList.pairs[bra];
            const ShellPairIndex& ketPair = fixture->pairList.pairs[ket];
            const std::array<std::size_t, 4> indices = {braPair.i, braPair.j, ketPair.i, ketPair.j};
            MdEriDerivativeQuartet quartet;

            for (int slot = 0; slot < 4; ++slot)
            {
                const std::size_t index = indices[static_cast<std::size_t>(slot)];
                quartet.shells[static_cast<std::size_t>(slot)] = &(*shells)[index];
                quartet.atoms[static_cast<std::size_t>(slot)] =
                    fixture->pairList.shells[index].atomIndex;
            }

            const std::vector<std::size_t> coordinates = QuartetCoordinates(quartet.atoms);
            const std::size_t elements = qcx::integrals::internal::EriQuartetBlockElements(quartet);
            value.assign(elements, 0.0);
            qcx::integrals::internal::BuildEriQuartetValue(quartet, 1, value, scratch);
            derivatives.assign(coordinates.size() * elements, 0.0);
            auto built = qcx::integrals::internal::BuildEriQuartetDerivative(
                quartet, 1, coordinates, derivatives, scratch, fixture->molecule.AtomCount());
            ASSERT_TRUE(built.has_value()) << built.error().message;

            for (std::size_t coordinate = 0; coordinate < coordinates.size(); ++coordinate)
            {
                std::array<MdShellInput, 4> plusStorage;
                std::array<MdShellInput, 4> minusStorage;
                plus.assign(elements, 0.0);
                minus.assign(elements, 0.0);
                qcx::integrals::internal::BuildEriQuartetValue(
                    DisplacedQuartet(
                        quartet, coordinates[coordinate], kFiniteDifferenceStep, plusStorage),
                    1,
                    plus,
                    scratch);
                qcx::integrals::internal::BuildEriQuartetValue(
                    DisplacedQuartet(
                        quartet, coordinates[coordinate], -kFiniteDifferenceStep, minusStorage),
                    1,
                    minus,
                    scratch);

                for (std::size_t element = 0; element < elements; ++element)
                {
                    const double measured =
                        (plus[element] - minus[element]) / (2.0 * kFiniteDifferenceStep);
                    const double analytic = derivatives[coordinate * elements + element];
                    const double deviation = std::abs(analytic - measured);

                    if (deviation > worstDeviation)
                    {
                        worstDeviation = deviation;
                        worstAt = "pairs " + std::to_string(bra) + " x " + std::to_string(ket) +
                                  " coordinate " + std::to_string(coordinates[coordinate]) +
                                  " element " + std::to_string(element) + ": analytic " +
                                  ShortDouble(analytic) + " measured " + ShortDouble(measured);
                    }

                    ++checked;
                }
            }
        }
    }

    RecordProperty("checked_elements", std::to_string(checked));
    RecordProperty("worst_deviation", ShortDouble(worstDeviation));
    RecordProperty("worst_at", worstAt);
    EXPECT_LE(worstDeviation, kDerivativeConsistencyTolerance) << worstAt;
}

/// The product rule's two halves against the quartets the screen applies it
/// to: the value half is a ceiling on every quartet's value, and the derivative
/// half - the product rule of the two pairs' own bound derivatives - is
/// recorded against each quartet's measured derivative.
///
/// The derivative half is an estimate, not a ceiling, and the fixture's own
/// numbers say by how much: a pair whose two shells share a centre has a
/// stationary bound and contributes nothing to the half, while a quartet it
/// takes part in still moves with the other pair's centre. That is what the
/// count of quartets past the bound is here for - the value half is what no
/// quartet gets past.
TEST(ScreeningPairBoundsTest, TheValueHalfBoundsEveryQuartetAndTheDerivativeHalfIsMeasured) {
    auto fixture = MakeFixture();
    ASSERT_TRUE(fixture.has_value()) << fixture.error().message;
    auto produced = qcx::integrals::ComputeTwoElectronPairBounds(fixture->molecule, fixture->basis);
    ASSERT_TRUE(produced.has_value()) << produced.error().message;
    auto shells = qcx::integrals::internal::FlattenShells(
        fixture->molecule, fixture->basis, fixture->pairList);
    ASSERT_TRUE(shells.has_value()) << shells.error().message;

    MdEriDerivativeScratch scratch;
    std::vector<double> value;
    std::vector<double> derivatives;
    double worstValueRatio = 0.0;
    double worstDerivativeRatio = 0.0;
    std::string worstDerivativeAt;
    std::size_t past = 0;
    std::size_t checked = 0;

    for (std::size_t bra = 0; bra < fixture->pairList.pairs.size(); ++bra)
    {
        for (std::size_t ket = 0; ket < fixture->pairList.pairs.size(); ++ket)
        {
            const ShellPairIndex& braPair = fixture->pairList.pairs[bra];
            const ShellPairIndex& ketPair = fixture->pairList.pairs[ket];
            const std::array<std::size_t, 4> indices = {braPair.i, braPair.j, ketPair.i, ketPair.j};
            MdEriDerivativeQuartet quartet;

            for (int slot = 0; slot < 4; ++slot)
            {
                const std::size_t index = indices[static_cast<std::size_t>(slot)];
                quartet.shells[static_cast<std::size_t>(slot)] = &(*shells)[index];
                quartet.atoms[static_cast<std::size_t>(slot)] =
                    fixture->pairList.shells[index].atomIndex;
            }

            const std::vector<std::size_t> coordinates = QuartetCoordinates(quartet.atoms);
            const std::size_t elements = qcx::integrals::internal::EriQuartetBlockElements(quartet);
            value.assign(elements, 0.0);
            qcx::integrals::internal::BuildEriQuartetValue(quartet, 1, value, scratch);
            derivatives.assign(coordinates.size() * elements, 0.0);
            auto built = qcx::integrals::internal::BuildEriQuartetDerivative(
                quartet, 1, coordinates, derivatives, scratch, fixture->molecule.AtomCount());

            ASSERT_TRUE(built.has_value()) << built.error().message;

            const qcx::integrals::QuartetDerivativeBound bound =
                qcx::integrals::QuartetDerivativeProduct((*produced)[bra], (*produced)[ket]);
            const double measuredValue = LargestMagnitude(value);
            const double measuredDerivative = LargestMagnitude(derivatives);

            if (measuredValue > 0.0)
            {
                worstValueRatio = std::max(worstValueRatio, measuredValue / bound.value);
            }

            if (measuredDerivative > bound.derivative + kMeasurementSlack)
            {
                ++past;

                if (measuredDerivative / std::max(bound.derivative, kMeasurementSlack) >
                    worstDerivativeRatio)
                {
                    worstDerivativeRatio =
                        measuredDerivative / std::max(bound.derivative, kMeasurementSlack);
                    worstDerivativeAt = "pairs " + std::to_string(bra) + " x " +
                                        std::to_string(ket) + ": measured " +
                                        ShortDouble(measuredDerivative) + " against a bound of " +
                                        ShortDouble(bound.derivative);
                }
            }

            EXPECT_LE(measuredValue, bound.value * (1.0 + kSlack) + kZeroBoundSlack)
                << "pairs " << bra << " x " << ket << " value: measured " << measuredValue
                << " against a bound of " << bound.value;
            ++checked;
        }
    }

    RecordProperty("checked_quartets", std::to_string(checked));
    RecordProperty("worst_value_ratio", ShortDouble(worstValueRatio));
    RecordProperty("quartets_past_the_derivative_bound", std::to_string(past));
    RecordProperty("worst_derivative_ratio", ShortDouble(worstDerivativeRatio));
    RecordProperty("worst_derivative_at", worstDerivativeAt);
    // A producer that returned zeros everywhere would read as a perfect bound
    // in the ratios above; the fixture's bounds are O(1).
    EXPECT_GT(worstValueRatio, 0.0);
    EXPECT_EQ(checked, fixture->pairList.pairs.size() * fixture->pairList.pairs.size());
}

/// The retention rule's direction, on the fixture's own candidates: every
/// quartet a value-only screen keeps, the derivative-aware one keeps too - it
/// screens on the larger of the two bounds, never the smaller.
TEST(ScreeningPairBoundsTest, TheDerivativeScreenNeverDropsWhatAValueScreenKeeps) {
    auto fixture = MakeFixture();
    ASSERT_TRUE(fixture.has_value()) << fixture.error().message;
    auto produced = qcx::integrals::ComputeTwoElectronPairBounds(fixture->molecule, fixture->basis);
    ASSERT_TRUE(produced.has_value()) << produced.error().message;

    std::vector<qcx::integrals::ScreenedQuartet> candidates;

    for (std::size_t bra = 0; bra < produced->size(); ++bra)
    {
        for (std::size_t ket = 0; ket < produced->size(); ++ket)
        {
            candidates.push_back(qcx::integrals::ScreenedQuartet{bra, ket});
        }
    }

    const std::array<double, 4> thresholds = {1.0e-10, 1.0e-6, 1.0e-3, 1.0e-1};
    std::size_t index = 0;

    for (const double threshold : thresholds)
    {
        std::size_t valueOnly = 0;
        std::size_t derivativeAware = 0;

        for (const qcx::integrals::ScreenedQuartet& candidate : candidates)
        {
            const qcx::integrals::QuartetDerivativeBound bound =
                qcx::integrals::QuartetDerivativeProduct((*produced)[candidate.braPair],
                                                         (*produced)[candidate.ketPair]);

            if (bound.value > threshold)
            {
                ++valueOnly;
                EXPECT_TRUE(qcx::integrals::SurvivesQuartetDerivativeScreen(bound, threshold))
                    << "threshold " << threshold << ", pairs " << candidate.braPair << " x "
                    << candidate.ketPair;
            }

            if (qcx::integrals::SurvivesQuartetDerivativeScreen(bound, threshold))
            {
                ++derivativeAware;
            }
        }

        const std::string label = std::to_string(index);
        RecordProperty("value_only_" + label, std::to_string(valueOnly));
        RecordProperty("derivative_aware_" + label, std::to_string(derivativeAware));
        RecordProperty("threshold_" + label, ShortDouble(threshold));
        EXPECT_GE(derivativeAware, valueOnly);
        ++index;
    }
}

/// One contracted shell per atom, an f and a g shell on the oxygen and an f
/// shell on each hydrogen: the f-g pair's diagonal block reaches Hermite order
/// 2 (3 + 4) and the g-g pair's 2 (4 + 4), past the 2 kMaxShellL the kernel
/// table covers, so neither pair has an evaluable block in this build - while
/// an oxygen-hydrogen f pair adds to exactly kMaxShellL, whose block is
/// evaluable but whose derivative is not. A fitting basis on a first-row centre
/// carries the same pairs: def2-universal-jfit's oxygen is one f and one g shell
/// among fifteen.
inline constexpr std::string_view kPastTheKernelTableBasis = R"(BASIS "ao basis" SPHERICAL PRINT
O    S
      1.0     1.0
O    F
      1.0     1.0
O    G
      1.0     1.0
H    S
      1.0     1.0
H    F
      1.0     1.0
END
)";

/// A pair whose shells add past kMaxShellL carries no bound at either order,
/// and the screen's product rule keeps every quartet that touches one.
///
/// The alternative - a finite stand-in for a number nothing derives - is the
/// failure this test exists against: the screen would drop quartets on a
/// quantity that is not a bound, and the gradient would be wrong smoothly
/// rather than loudly.
TEST(ScreeningPairBoundsTest, APairPastTheKernelTableCarriesNoBound) {
    // The fixture's whole point is shells the kernel table does not cover, so
    // a build whose ceiling stops below them refuses the basis at its entry
    // point and there is no pair census to read.
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "the fixture's f and g shells exceed this build's kMaxEngineL "
                        "(CI lmax=2)";
    }

    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::basisset::ParseNwchemText(kPastTheKernelTableBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto produced = qcx::integrals::ComputeTwoElectronPairBounds(*molecule, *basis);
    ASSERT_TRUE(produced.has_value()) << produced.error().message;
    ASSERT_EQ(produced->size(), pairList->pairs.size());

    std::vector<std::size_t> unbounded;
    std::size_t bounded = 0;
    std::size_t sameAtom = 0;
    std::size_t crossAtom = 0;
    std::size_t measured = 0;

    for (std::size_t pair = 0; pair < pairList->pairs.size(); ++pair)
    {
        const ShellPairIndex& shells = pairList->pairs[pair];
        const int pairAngular =
            pairList->shells[shells.i].angularMomentum + pairList->shells[shells.j].angularMomentum;

        if (pairAngular > qcx::integrals::internal::kMaxShellL)
        {
            unbounded.push_back(pair);
            EXPECT_FALSE((*produced)[pair].hasBound()) << "pair " << pair;
        } else
        {
            ++bounded;
            EXPECT_TRUE((*produced)[pair].hasBound()) << "pair " << pair;
            EXPECT_GT((*produced)[pair].value, 0.0) << "pair " << pair;

            if (pairList->shells[shells.i].atomIndex == pairList->shells[shells.j].atomIndex)
            {
                // Both shells on one atom: the pair's motion is a rigid
                // translation of that atom, and the bound does not move with it.
                ++sameAtom;
                EXPECT_EQ((*produced)[pair].derivative, 0.0) << "pair " << pair;
            } else
            {
                // Two atoms: the bound moves with their separation, through the
                // tier's analytic route where the kernel table reaches
                // 2 (l_i + l_j) + 2 and through the measured radial difference
                // where it stops short.
                ++crossAtom;
                EXPECT_GT((*produced)[pair].derivative, 0.0) << "pair " << pair;
                measured += pairAngular == qcx::integrals::internal::kMaxShellL ? 1 : 0;
            }
        }
    }

    RecordProperty("unbounded_pairs", std::to_string(unbounded.size()));
    RecordProperty("bounded_pairs", std::to_string(bounded));
    RecordProperty("same_atom_pairs", std::to_string(sameAtom));
    RecordProperty("cross_atom_pairs", std::to_string(crossAtom));
    RecordProperty("pairs_past_the_analytic_tier", std::to_string(measured));
    ASSERT_FALSE(unbounded.empty());
    EXPECT_GT(bounded, 0u);
    EXPECT_GT(sameAtom, 0u);
    EXPECT_GT(measured, 0u);

    const TwoElectronPairBound& noBound = (*produced)[unbounded.front()];
    const qcx::integrals::QuartetDerivativeBound quartet =
        qcx::integrals::QuartetDerivativeProduct(noBound, noBound);
    EXPECT_FALSE(std::isfinite(quartet.value));
    EXPECT_FALSE(std::isfinite(quartet.derivative));
    EXPECT_TRUE(qcx::integrals::SurvivesQuartetDerivativeScreen(quartet, 1.0e6));

    // The other order of the same product: an unbounded pair against a pair
    // whose bound is exactly zero - a NaN would read as neither above nor below
    // any threshold.
    const TwoElectronPairBound zero;
    const qcx::integrals::QuartetDerivativeBound mixed =
        qcx::integrals::QuartetDerivativeProduct(zero, noBound);
    EXPECT_FALSE(std::isfinite(mixed.value));
    EXPECT_FALSE(std::isfinite(mixed.derivative));
    EXPECT_TRUE(qcx::integrals::SurvivesQuartetDerivativeScreen(mixed, 1.0e6));
}
