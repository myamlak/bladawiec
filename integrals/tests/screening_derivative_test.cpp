// The derivative-aware screening's tests: the checkpoint case - a one-electron
// block element whose value is below the threshold at a mirror geometry while
// its gradient is not, kept by the derivative rule and dropped by a value-only
// one - the bound's dominance of an independently measured difference, and the
// chunked pass's invariance.
//
// The fixture is a linear OH: the H sits on the O's z axis, so the O 2p_x and
// O 2p_y block elements are exactly zero and their gradients are not. That is
// the case a screen written for energies loses: the elements contribute
// nothing to the energy and everything to the gradient.
#include "h2o_sto3g.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_one_electron.hpp"
#include "internal/shells_flat.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::integrals::internal::MdPairData;

/// The central-difference step of the dominance check: the elements here are
/// O(0.1) with O(1e-16) noise, so a step far above the cancellation floor
/// still sits inside the truncation law's comfortable range.
constexpr double kDifferenceStep = 1.0e-5;

/// A linear OH with the oxygen at the origin and the hydrogen at (0, 0, r).
/// \param r The O-H distance (Bohr).
/// \returns The molecule, or an Error.
qcx::Result<qcx::molecule::Molecule> MakeLinearOh(double r) {
    auto coordinates = CpuTensor2::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 0.0;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = r;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

/// The same molecule with one flat coordinate (3 * atom + axis) moved by a
/// delta. The molecule's canonical atom order is the created order's, so the
/// caller passes the index it read back from the pair list's shells.
/// \param base The undisplaced molecule.
/// \param flat The flat coordinate to move.
/// \param delta The displacement (Bohr).
/// \returns The displaced molecule, or an Error.
qcx::Result<qcx::molecule::Molecule> DisplacedLinearOh(const qcx::molecule::Molecule& base,
                                                       std::size_t flat,
                                                       double delta) {
    auto coordinates = CpuTensor2::Create({base.AtomCount(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    const auto& source = base.CoordinatesBohr();

    for (std::size_t atom = 0; atom < base.AtomCount(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            (*coordinates)(atom, axis) = source(atom, axis);
        }
    }

    (*coordinates)(flat / 3, flat % 3) += delta;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        base.Atoms(), std::move(*coordinates), base.Charge(), base.Multiplicity());
}

/// The index of the first shell with the given angular momentum on the given
/// atom, or pairList.shells.size() when there is none.
/// \param pairList The pair list.
/// \param atomIndex The atom.
/// \param angularMomentum The shell's l.
/// \returns The shell index.
std::size_t FindShellOnAtom(const qcx::integrals::ShellPairList& pairList,
                            std::size_t atomIndex,
                            int angularMomentum) {
    for (std::size_t shell = 0; shell < pairList.shells.size(); ++shell)
    {
        if (pairList.shells[shell].atomIndex == atomIndex &&
            pairList.shells[shell].angularMomentum == angularMomentum)
        {
            return shell;
        }
    }

    return pairList.shells.size();
}

/// The molecule's nuclear charges, one per atom.
/// \param molecule The molecule.
/// \returns Z per atom in Bohr order.
std::vector<double> NuclearCharges(const qcx::molecule::Molecule& molecule) {
    std::vector<double> charges;

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        charges.push_back(static_cast<double>(atom.atomicNumber));
    }

    return charges;
}

/// One pair's overlap, kinetic and nuclear-attraction blocks at a geometry -
/// every operator the screening bound covers.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \param shellA The bra shell.
/// \param shellB The ket shell (shellB >= shellA).
/// \param blocks Out: three nFuncs blocks (overlap, kinetic, nuclear).
/// \returns Nothing, or an Error.
qcx::Result<void> OperatorBlocks(const qcx::molecule::Molecule& molecule,
                                 const qcx::basisset::BasisSet& basisSet,
                                 std::size_t shellA,
                                 std::size_t shellB,
                                 std::array<std::vector<double>, 3>& blocks) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto store = qcx::integrals::internal::BuildPairData(molecule, basisSet, *pairList);

    if (!store.has_value())
    {
        return std::unexpected(store.error());
    }

    const MdPairData& pair = (*store)[qcx::integrals::PairIndexOf(shellA, shellB, *pairList)];
    const std::vector<double> charges = NuclearCharges(molecule);
    const std::vector<Eigen::Vector3d> centers = qcx::integrals::internal::AtomPositions(molecule);

    for (std::vector<double>& block : blocks)
    {
        block.assign(pair.nFuncs, 0.0);
    }

    qcx::integrals::internal::BuildOverlapPair(pair, blocks[0].data());
    qcx::integrals::internal::BuildKineticPair(pair, blocks[1].data());
    qcx::integrals::internal::BuildNuclearPair(pair, charges, centers, blocks[2].data());
    return {};
}

/// The O 2p / H 1s pair of the linear OH, with the shells and atoms the tests
/// read.
struct NodalPair {
    std::size_t shellA = 0; ///< The hydrogen 1s.
    std::size_t shellB = 0; ///< The oxygen 2p.
    std::size_t pairIndex = 0; ///< The canonical pair index.
    std::size_t pAtom = 0; ///< The atom carrying the 2p.
};

/// Finds the O 2p / H 1s pair (the p shell by angular momentum, the s shell on
/// the other atom), plus both atom indices.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \returns The pair's indices, or an Error.
qcx::Result<NodalPair> FindNodalPair(const qcx::molecule::Molecule& molecule,
                                     const qcx::basisset::BasisSet& basisSet) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    std::size_t pShell = pairList->shells.size();

    for (std::size_t shell = 0; shell < pairList->shells.size(); ++shell)
    {
        if (pairList->shells[shell].angularMomentum == 1)
        {
            pShell = shell;
            break;
        }
    }

    if (pShell == pairList->shells.size())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "no l = 1 shell in the fixture"});
    }

    const std::size_t pAtom = pairList->shells[pShell].atomIndex;
    const std::size_t sAtom = 1 - pAtom;
    const std::size_t sShell = FindShellOnAtom(*pairList, sAtom, 0);

    if (sShell == pairList->shells.size())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "no s shell on the other atom"});
    }

    const std::size_t first = std::min(sShell, pShell);
    const std::size_t second = std::max(sShell, pShell);
    NodalPair pair;
    pair.shellA = first;
    pair.shellB = second;
    pair.pairIndex = qcx::integrals::PairIndexOf(first, second, *pairList);
    pair.pAtom = pAtom;
    return pair;
}

} // namespace

TEST(ScreeningDerivativeTest, TheNodalElementSurvivesTheDerivativeScreen) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    // A physical linear OH: the hydrogen on the oxygen's z axis.
    auto molecule = MakeLinearOh(1.83);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto nodal = FindNodalPair(*molecule, *basis);
    ASSERT_TRUE(nodal.has_value()) << nodal.error().message;

    auto bounds = qcx::integrals::ComputeDerivativeAwareBounds(*molecule, *basis);
    ASSERT_TRUE(bounds.has_value()) << bounds.error().message;
    ASSERT_LT(nodal->pairIndex, bounds->size());
    const auto& elements = (*bounds)[nodal->pairIndex].elements;
    ASSERT_FALSE(elements.empty());

    const double threshold =
        qcx::integrals::SchwarzThreshold(qcx::integrals::AccuracyPreset::kNormal);

    // The pair's own block: the p_x and p_y elements vanish at this geometry
    // while the p_z element does not, so the pair-level maximum is large and
    // only the element-level bound sees the case.
    std::array<std::vector<double>, 3> blocks;
    auto built = OperatorBlocks(*molecule, *basis, nodal->shellA, nodal->shellB, blocks);
    ASSERT_TRUE(built.has_value()) << built.error().message;
    ASSERT_EQ(blocks[0].size(), elements.size());
    double pairMax = 0.0;

    for (const std::vector<double>& block : blocks)
    {
        for (const double value : block)
        {
            pairMax = std::max(pairMax, std::abs(value));
        }
    }

    ASSERT_GT(pairMax, threshold) << "the fixture's p_z element is expected to be large";

    // The nodal element: below the threshold in value, above it in gradient.
    std::size_t nodalElement = elements.size();

    for (std::size_t element = 0; element < elements.size(); ++element)
    {
        if (elements[element].value < threshold && elements[element].derivative > threshold &&
            (nodalElement == elements.size() ||
             elements[element].derivative > elements[nodalElement].derivative))
        {
            nodalElement = element;
        }
    }

    ASSERT_LT(nodalElement, elements.size())
        << "no element is below the threshold in value and above it in gradient";
    const double elementValue = std::abs(blocks[0][nodalElement]);
    ASSERT_LT(elementValue, threshold);
    ASSERT_GT(elements[nodalElement].derivative, threshold);
    // The value-only rule drops this element; the derivative-aware rule keeps it.
    ASSERT_FALSE(elementValue > threshold);
    ASSERT_TRUE(qcx::integrals::SurvivesDerivativeScreen(elements[nodalElement], threshold));

    // Present in the screened set.
    const std::vector<qcx::integrals::ScreenedElement> retained =
        qcx::integrals::RetainedElements(*bounds, threshold);
    bool found = false;

    for (const qcx::integrals::ScreenedElement& entry : retained)
    {
        if (entry.pair == nodal->pairIndex && entry.element == nodalElement)
        {
            found = true;
        }
    }

    ASSERT_TRUE(found) << "the nodal element is absent from the screened set";
}

TEST(ScreeningDerivativeTest, TheDerivativeBoundDominatesTheMeasuredDifference) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeLinearOh(1.83);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto nodal = FindNodalPair(*molecule, *basis);
    ASSERT_TRUE(nodal.has_value()) << nodal.error().message;

    auto bounds = qcx::integrals::ComputeDerivativeAwareBounds(*molecule, *basis);
    ASSERT_TRUE(bounds.has_value()) << bounds.error().message;
    const auto& elements = (*bounds)[nodal->pairIndex].elements;

    // The element with the largest gradient: the p_x or p_y element here.
    std::size_t target = 0;

    for (std::size_t element = 0; element < elements.size(); ++element)
    {
        if (elements[element].derivative > elements[target].derivative)
        {
            target = element;
        }
    }

    ASSERT_GT(elements[target].derivative, 0.0);

    // The measured difference of that element, over every operator and every
    // coordinate of the pair's own two atoms: the bound must not be under it.
    double measured = 0.0;
    const std::array<std::size_t, 2> atoms{nodal->pAtom, 1 - nodal->pAtom};

    for (const std::size_t atom : atoms)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const std::size_t flat = 3 * atom + axis;
            auto plus = DisplacedLinearOh(*molecule, flat, kDifferenceStep);
            ASSERT_TRUE(plus.has_value()) << plus.error().message;
            auto minus = DisplacedLinearOh(*molecule, flat, -kDifferenceStep);
            ASSERT_TRUE(minus.has_value()) << minus.error().message;
            std::array<std::vector<double>, 3> plusBlocks;
            std::array<std::vector<double>, 3> minusBlocks;
            auto builtPlus =
                OperatorBlocks(*plus, *basis, nodal->shellA, nodal->shellB, plusBlocks);
            ASSERT_TRUE(builtPlus.has_value()) << builtPlus.error().message;
            auto builtMinus =
                OperatorBlocks(*minus, *basis, nodal->shellA, nodal->shellB, minusBlocks);
            ASSERT_TRUE(builtMinus.has_value()) << builtMinus.error().message;

            for (std::size_t op = 0; op < 3; ++op)
            {
                ASSERT_EQ(plusBlocks[op].size(), elements.size());
                measured = std::max(measured,
                                    std::abs(plusBlocks[op][target] - minusBlocks[op][target]) /
                                        (2.0 * kDifferenceStep));
            }
        }
    }

    // Never under: the bound is the maximum the blocks and their coordinates
    // carry, and the measured difference is one point of that maximum.
    EXPECT_GE(elements[target].derivative, measured * (1.0 - 1.0e-4));
    // And not a runaway: the maximum over the same operators and coordinates,
    // measured, is what the bound is.
    EXPECT_LE(elements[target].derivative, measured * (1.0 + 1.0e-4));
}

TEST(ScreeningDerivativeTest, TheChunkedPassAgreesWithTheSingleChunk) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto single = qcx::integrals::ComputeDerivativeAwareBounds(*molecule, *basis, 0);
    ASSERT_TRUE(single.has_value()) << single.error().message;
    // One byte forces the smallest chunk the payload rule can make - one pair
    // at a time, every pair its own arena build and release.
    auto chunked = qcx::integrals::ComputeDerivativeAwareBounds(*molecule, *basis, 1);
    ASSERT_TRUE(chunked.has_value()) << chunked.error().message;
    ASSERT_EQ(single->size(), chunked->size());

    for (std::size_t pair = 0; pair < single->size(); ++pair)
    {
        ASSERT_EQ((*single)[pair].nFuncs, (*chunked)[pair].nFuncs);
        ASSERT_EQ((*single)[pair].elements.size(), (*chunked)[pair].elements.size());

        for (std::size_t element = 0; element < (*single)[pair].elements.size(); ++element)
        {
            EXPECT_EQ((*single)[pair].elements[element].value,
                      (*chunked)[pair].elements[element].value);
            EXPECT_EQ((*single)[pair].elements[element].derivative,
                      (*chunked)[pair].elements[element].derivative);
        }
    }
}
