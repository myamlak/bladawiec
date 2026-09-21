// The MD derivative tier's tests: the analytic nuclear-coordinate
// derivatives of the one-electron pair blocks against the CENTRAL FINITE
// DIFFERENCE of the existing base builders at displaced geometries (the
// plan's B1 threshold, 1e-8), the exactly-zero mixed-operator-center case,
// the order-taking call shape (B4), and the refusal paths. The finite
// difference runs no second analytic path - it is BuildOverlapPair,
// BuildKineticPair and BuildNuclearPair, unchanged, on a displaced water
// molecule, so a drift between the derivative tier and the base builders
// shows up as a mismatch.
#include "h2o_sto3g.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_derivative.hpp"
#include "internal/md_one_electron.hpp"
#include "internal/shells_flat.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::integrals::internal::MdPairData;

/// The central-difference step of the first-derivative check: the
/// truncation is O(h^2) and the cancellation error O(eps/h), both near 1e-11
/// at the blocks' O(1) magnitudes - comfortably inside the 1e-8 threshold.
constexpr double kFirstStep = 1e-5;

/// The plan's analytic-versus-finite-difference threshold for the
/// one-electron derivative blocks.
constexpr double kFirstTolerance = 1e-8;

/// The central-difference step of the second-derivative check, from a
/// measured balance: the four-point central second difference carries an
/// f'''' h^2/12 truncation against an eps/h^2 cancellation error, and on the
/// hetero pair the largest nuclear-block discrepancy over every ordered
/// coordinate tuple measures 1.51e-06 (h = 1e-3), 1.37e-07 (h = 3e-4) and
/// 3.86e-08 (h = 1e-4) - the O(h^2) law down to the minimum, where the
/// cancellation error takes over. The step sits at that minimum.
constexpr double kSecondStep = 1e-4;

/// The second-derivative tolerance, RELATIVE with an absolute floor. It is
/// the finite difference's limit, not the analytic accuracy of the tier: the
/// discrepancy measured at kSecondStep, an order of magnitude above the 1e-8
/// the first-derivative check holds the same code to, stays an order below
/// this bound.
constexpr double kSecondTolerance = 1e-6;

/// The water molecule with the listed flat coordinates (3 * atom + axis)
/// displaced by the listed deltas; a repeated coordinate accumulates, so
/// (c, c) with (+h, +h) is the +2h point of a same-coordinate tuple. Errors
/// when the canonical renumbering would move an atom - the finite-difference
/// loop reads the displaced molecule's shells by the BASE geometry's shell
/// indices, which is only valid when the order is unchanged.
qcx::Result<qcx::molecule::Molecule> MakeDisplacedWater(
    const qcx::molecule::Molecule& base,
    std::span<const std::size_t> flatCoordinates,
    std::span<const double> deltas) {
    if (flatCoordinates.size() != deltas.size())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "displacement list shape mismatch"});
    }

    auto coordinates = CpuTensor2::Create({base.AtomCount(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    const auto& source = base.CoordinatesBohr();
    // The geometry the created molecule must come back as, row for row. The
    // guard below compares against THIS and not against the base: the moved
    // rows are supposed to differ from the base, and a comparison that
    // demanded they did not could never pass.
    std::vector<std::array<double, 3>> expected;
    expected.reserve(base.AtomCount());

    for (std::size_t atom = 0; atom < base.AtomCount(); ++atom)
    {
        std::array<double, 3> row{};

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            row[axis] = source(atom, axis);
            (*coordinates)(atom, axis) = source(atom, axis);
        }

        expected.push_back(row);
    }

    for (std::size_t k = 0; k < flatCoordinates.size(); ++k)
    {
        const std::size_t flat = flatCoordinates[k];

        if (flat >= 3 * base.AtomCount())
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "displaced coordinate out of range"});
        }

        (*coordinates)(flat / 3, flat % 3) += deltas[k];
        expected[flat / 3][flat % 3] += deltas[k];
    }

    coordinates->MarkHostDirty();
    auto displaced = qcx::molecule::Molecule::Create(
        base.Atoms(), std::move(*coordinates), base.Charge(), base.Multiplicity());

    if (!displaced.has_value())
    {
        return std::unexpected(displaced.error());
    }

    for (std::size_t atom = 0; atom < base.AtomCount(); ++atom)
    {
        if (displaced->Atoms()[atom].atomicNumber != base.Atoms()[atom].atomicNumber ||
            displaced->CoordinatesBohr()(atom, 0) != expected[atom][0] ||
            displaced->CoordinatesBohr()(atom, 1) != expected[atom][1] ||
            displaced->CoordinatesBohr()(atom, 2) != expected[atom][2])
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "the displaced geometry is not the base geometry with one row moved - "
                           "the shell indices of the finite-difference loop would name other "
                           "shells"});
        }
    }

    return displaced;
}

/// The nuclear charges the nuclear-attraction builders sum, one per atom.
std::vector<double> NuclearCharges(const qcx::molecule::Molecule& molecule) {
    std::vector<double> charges;
    charges.reserve(molecule.AtomCount());

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        charges.push_back(static_cast<double>(atom.atomicNumber));
    }

    return charges;
}

/// One shell pair's data at one geometry, with the atoms of its two shells.
struct PairAt {
    qcx::integrals::internal::MdPairData pair; ///< The contracted pair data.
    std::size_t braAtom = 0; ///< Atom carrying shell \p shellA.
    std::size_t ketAtom = 0; ///< Atom carrying shell \p shellB.
};

/// Builds the pair data of one shell pair at the molecule's geometry.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \param shellA The bra shell index.
/// \param shellB The ket shell index, shellB >= shellA.
/// \returns The pair and its two atoms, or an Error.
qcx::Result<PairAt> PairDataAt(const qcx::molecule::Molecule& molecule,
                               const qcx::basisset::BasisSet& basisSet,
                               std::size_t shellA,
                               std::size_t shellB) {
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

    const std::size_t index = qcx::integrals::PairIndexOf(shellA, shellB, *pairList);

    if (index >= store->size())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "shell pair out of range"});
    }

    PairAt out;
    out.pair = std::move((*store)[index]);
    out.braAtom = pairList->shells[shellA].atomIndex;
    out.ketAtom = pairList->shells[shellB].atomIndex;
    return out;
}

/// The fixture's one shell of the given angular momentum: its oxygen 2p for
/// l = 1. The fixture's molecule is canonically renumbered by (Z, x, y, z),
/// which puts the hydrogens FIRST - so the oxygen is not atom 0 and no test
/// may name an atom to reach its shells. The shell list is the only fact.
/// \returns The shell index, or the shell count when there is none.
std::size_t FindAngularShell(const qcx::integrals::ShellPairList& pairList, int angularMomentum) {
    for (std::size_t shell = 0; shell < pairList.shells.size(); ++shell)
    {
        if (pairList.shells[shell].angularMomentum == angularMomentum)
        {
            return shell;
        }
    }

    return pairList.shells.size();
}

/// The first atom that carries neither shell of the shell pair \p shells - a
/// nucleus the pair's blocks reach only through the nuclear attraction's
/// operator center.
/// \param pairList The fixture's shell pair list.
/// \param shells The shell pair whose two atoms are excluded.
/// \param atomCount The molecule's atom count.
/// \returns The atom index, or the atom count when there is none.
std::size_t FindThirdAtom(const qcx::integrals::ShellPairList& pairList,
                          qcx::integrals::ShellPairIndex shells,
                          std::size_t atomCount) {
    const std::size_t braAtom = pairList.shells[shells.i].atomIndex;
    const std::size_t ketAtom = pairList.shells[shells.j].atomIndex;

    for (std::size_t atom = 0; atom < atomCount; ++atom)
    {
        if (atom != braAtom && atom != ketAtom)
        {
            return atom;
        }
    }

    return atomCount;
}

/// One pair's three blocks (S, T and V) at one geometry, laid out
/// nFuncsA x nFuncsB row-major - the layout every builder of the module
/// writes.
struct PairBlocks {
    std::size_t nFuncs = 0; ///< Elements per block.
    std::vector<double> overlap; ///< The overlap block.
    std::vector<double> kinetic; ///< The kinetic-energy block.
    std::vector<double> nuclear; ///< The nuclear-attraction block.
};

/// The three BASE blocks of one shell pair at the molecule's geometry: the
/// finite difference's reference values, computed by the production builders
/// (md_one_electron.hpp) and nothing else.
qcx::Result<PairBlocks> BaseBlocks(const qcx::molecule::Molecule& molecule,
                                   const qcx::basisset::BasisSet& basisSet,
                                   std::size_t shellA,
                                   std::size_t shellB) {
    auto pairAt = PairDataAt(molecule, basisSet, shellA, shellB);

    if (!pairAt.has_value())
    {
        return std::unexpected(pairAt.error());
    }

    PairBlocks out;
    out.nFuncs = pairAt->pair.nFuncs;
    out.overlap.assign(out.nFuncs, 0.0);
    out.kinetic.assign(out.nFuncs, 0.0);
    out.nuclear.assign(out.nFuncs, 0.0);
    qcx::integrals::internal::BuildOverlapPair(pairAt->pair, out.overlap.data());
    qcx::integrals::internal::BuildKineticPair(pairAt->pair, out.kinetic.data());
    qcx::integrals::internal::BuildNuclearPair(pairAt->pair,
                                               NuclearCharges(molecule),
                                               qcx::integrals::internal::AtomPositions(molecule),
                                               out.nuclear.data());
    return out;
}

/// The three base blocks at the geometry with the listed coordinates
/// displaced - the finite difference's sample point.
qcx::Result<PairBlocks> DisplacedBlocks(const qcx::molecule::Molecule& molecule,
                                        const qcx::basisset::BasisSet& basisSet,
                                        std::size_t shellA,
                                        std::size_t shellB,
                                        std::span<const std::size_t> flatCoordinates,
                                        std::span<const double> deltas) {
    auto displaced = MakeDisplacedWater(molecule, flatCoordinates, deltas);

    if (!displaced.has_value())
    {
        return std::unexpected(displaced.error());
    }

    return BaseBlocks(*displaced, basisSet, shellA, shellB);
}

/// The three DERIVATIVE blocks of one shell pair - one block per tuple of
/// \p coordinates, tuple-major. The order is the one call argument that
/// selects the derivative: the same three functions serve every order.
/// \p coordinates must name atoms that carry one of the pair's two shells:
/// the overlap and kinetic blocks depend on no other nucleus, and their
/// builders refuse such a coordinate. A request that reaches a third nucleus
/// goes through NuclearDerivativeBlocks.
qcx::Result<PairBlocks> DerivativeBlocks(const qcx::molecule::Molecule& molecule,
                                         const qcx::basisset::BasisSet& basisSet,
                                         qcx::integrals::ShellPairIndex shells,
                                         int order,
                                         std::span<const std::size_t> coordinates) {
    auto pairAt = PairDataAt(molecule, basisSet, shells.i, shells.j);

    if (!pairAt.has_value())
    {
        return std::unexpected(pairAt.error());
    }

    const std::size_t count =
        qcx::integrals::internal::MdDerivativeBlockCount(pairAt->pair, order, coordinates.size());
    const std::vector<double> charges = NuclearCharges(molecule);
    const std::vector<Eigen::Vector3d> centers = qcx::integrals::internal::AtomPositions(molecule);
    PairBlocks out;
    out.nFuncs = pairAt->pair.nFuncs;
    out.overlap.assign(count, 0.0);
    out.kinetic.assign(count, 0.0);
    out.nuclear.assign(count, 0.0);
    qcx::integrals::internal::MdDerivativeScratch scratch;
    auto overlap = qcx::integrals::internal::BuildOverlapPairDerivative(
        pairAt->pair, pairAt->braAtom, pairAt->ketAtom, order, coordinates, out.overlap, scratch);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::internal::BuildKineticPairDerivative(
        pairAt->pair, pairAt->braAtom, pairAt->ketAtom, order, coordinates, out.kinetic, scratch);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::internal::BuildNuclearPairDerivative(pairAt->pair,
                                                                        pairAt->braAtom,
                                                                        pairAt->ketAtom,
                                                                        charges,
                                                                        centers,
                                                                        order,
                                                                        coordinates,
                                                                        out.nuclear,
                                                                        scratch);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    return out;
}

/// The NUCLEAR-attraction derivative block of one shell pair alone, one
/// block per tuple of \p coordinates, tuple-major. The nuclear attraction is
/// the one operator of the three whose coordinates may name a nucleus that
/// carries neither shell - such a nucleus enters through the operator center
/// alone - so its requests are exactly the ones the shared three-block
/// helper cannot serve.
qcx::Result<std::vector<double>> NuclearDerivativeBlocks(const qcx::molecule::Molecule& molecule,
                                                         const qcx::basisset::BasisSet& basisSet,
                                                         qcx::integrals::ShellPairIndex shells,
                                                         int order,
                                                         std::span<const std::size_t> coordinates) {
    auto pairAt = PairDataAt(molecule, basisSet, shells.i, shells.j);

    if (!pairAt.has_value())
    {
        return std::unexpected(pairAt.error());
    }

    const std::size_t count =
        qcx::integrals::internal::MdDerivativeBlockCount(pairAt->pair, order, coordinates.size());
    std::vector<double> out(count, 0.0);
    qcx::integrals::internal::MdDerivativeScratch scratch;
    auto nuclear = qcx::integrals::internal::BuildNuclearPairDerivative(
        pairAt->pair,
        pairAt->braAtom,
        pairAt->ketAtom,
        NuclearCharges(molecule),
        qcx::integrals::internal::AtomPositions(molecule),
        order,
        coordinates,
        out,
        scratch);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    return out;
}

/// The four-point central second difference of one block:
/// (f(++) - f(+-) - f(-+) + f(--)) / (4 h^2), the mixed second difference of
/// the two tuple slots. A tuple naming one coordinate twice collapses to
/// the three-point central difference with step 2h - f(+-) and f(-+) are the
/// undisplaced value there - so one expression serves both shapes.
std::vector<double> CrossDifference(const std::vector<double>& plusPlus,
                                    const std::vector<double>& plusMinus,
                                    const std::vector<double>& minusPlus,
                                    const std::vector<double>& minusMinus,
                                    double step) {
    std::vector<double> out(plusPlus.size());
    const double scale = 1.0 / (4.0 * step * step);

    for (std::size_t i = 0; i < out.size(); ++i)
    {
        out[i] = (plusPlus[i] - plusMinus[i] - minusPlus[i] + minusMinus[i]) * scale;
    }

    return out;
}

/// The pair's own six coordinates: both shells' atoms, three axes each, in
/// (x, y, z) per atom order.
std::vector<std::size_t> PairCoordinates(std::size_t braAtom, std::size_t ketAtom) {
    std::vector<std::size_t> coordinates;

    for (std::size_t atom : std::array<std::size_t, 2>{braAtom, ketAtom})
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            if (std::find(coordinates.begin(), coordinates.end(), 3 * atom + axis) ==
                coordinates.end())
            {
                coordinates.push_back(3 * atom + axis);
            }
        }
    }

    return coordinates;
}

/// The fixture's hetero pair: its l = 1 shell (the oxygen 2p) with an l = 0
/// shell of a DIFFERENT atom (a hydrogen 1s), in canonical order
/// (shellA <= shellB - the order PairIndexOf and the pair data take).
struct HeteroPair {
    std::size_t shellA = 0; ///< The lower-indexed shell of the pair.
    std::size_t shellB = 0; ///< The higher-indexed shell, shellB >= shellA.
};

HeteroPair FindHeteroPair(const qcx::integrals::ShellPairList& pairList) {
    HeteroPair out;
    out.shellA = FindAngularShell(pairList, 1);
    out.shellB = pairList.shells.size();

    if (out.shellA == pairList.shells.size())
    {
        return out;
    }

    for (std::size_t shell = 0; shell < pairList.shells.size(); ++shell)
    {
        if (pairList.shells[shell].angularMomentum == 0 &&
            pairList.shells[shell].atomIndex != pairList.shells[out.shellA].atomIndex)
        {
            out.shellB = shell;
            break;
        }
    }

    if (out.shellB < out.shellA)
    {
        std::swap(out.shellA, out.shellB);
    }

    return out;
}

TEST(MdDerivativeTest, OneAtomPairMatchesCentralDifferenceAt1e8) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const HeteroPair pair = FindHeteroPair(*pairList);
    ASSERT_LT(pair.shellA, pairList->shells.size());
    ASSERT_LT(pair.shellB, pairList->shells.size());

    // The one atom pair's own coordinates, derivative wrt both atoms.
    const std::vector<std::size_t> coordinates = PairCoordinates(
        pairList->shells[pair.shellA].atomIndex, pairList->shells[pair.shellB].atomIndex);
    ASSERT_EQ(coordinates.size(), 6);

    auto base = BaseBlocks(*molecule, *basis, pair.shellA, pair.shellB);
    ASSERT_TRUE(base.has_value()) << base.error().message;
    auto analytic = DerivativeBlocks(*molecule, *basis, {pair.shellA, pair.shellB}, 1, coordinates);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    ASSERT_EQ(analytic->overlap.size(), coordinates.size() * base->nFuncs);

    std::array<std::size_t, 1> flat{0};
    std::array<double, 1> delta{0.0};

    for (std::size_t k = 0; k < coordinates.size(); ++k)
    {
        flat[0] = coordinates[k];
        delta[0] = kFirstStep;
        auto plus = DisplacedBlocks(*molecule, *basis, pair.shellA, pair.shellB, flat, delta);
        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        delta[0] = -kFirstStep;
        auto minus = DisplacedBlocks(*molecule, *basis, pair.shellA, pair.shellB, flat, delta);
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        for (std::size_t e = 0; e < base->nFuncs; ++e)
        {
            SCOPED_TRACE("coordinate " + std::to_string(coordinates[k]) + ", element " +
                         std::to_string(e));
            const double scale = 1.0 / (2.0 * kFirstStep);
            EXPECT_NEAR(analytic->overlap[k * base->nFuncs + e],
                        (plus->overlap[e] - minus->overlap[e]) * scale,
                        kFirstTolerance);
            EXPECT_NEAR(analytic->kinetic[k * base->nFuncs + e],
                        (plus->kinetic[e] - minus->kinetic[e]) * scale,
                        kFirstTolerance);
            EXPECT_NEAR(analytic->nuclear[k * base->nFuncs + e],
                        (plus->nuclear[e] - minus->nuclear[e]) * scale,
                        kFirstTolerance);
        }
    }
}

TEST(MdDerivativeTest, NuclearAttractionDifferentiatesEveryNucleus) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const HeteroPair pair = FindHeteroPair(*pairList);
    ASSERT_LT(pair.shellB, pairList->shells.size());

    // Every nucleus of the molecule: the second hydrogen enters the
    // derivative through the operator center alone - the term the pair's two
    // atoms cannot reach.
    std::vector<std::size_t> coordinates;

    for (std::size_t atom = 0; atom < molecule->AtomCount(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            coordinates.push_back(3 * atom + axis);
        }
    }

    ASSERT_EQ(coordinates.size(), 9);
    auto base = BaseBlocks(*molecule, *basis, pair.shellA, pair.shellB);
    ASSERT_TRUE(base.has_value()) << base.error().message;
    auto analytic =
        NuclearDerivativeBlocks(*molecule, *basis, {pair.shellA, pair.shellB}, 1, coordinates);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;

    std::array<std::size_t, 1> flat{0};
    std::array<double, 1> delta{0.0};

    for (std::size_t k = 0; k < coordinates.size(); ++k)
    {
        flat[0] = coordinates[k];
        delta[0] = kFirstStep;
        auto plus = DisplacedBlocks(*molecule, *basis, pair.shellA, pair.shellB, flat, delta);
        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        delta[0] = -kFirstStep;
        auto minus = DisplacedBlocks(*molecule, *basis, pair.shellA, pair.shellB, flat, delta);
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        for (std::size_t e = 0; e < base->nFuncs; ++e)
        {
            SCOPED_TRACE("coordinate " + std::to_string(coordinates[k]) + ", element " +
                         std::to_string(e));
            EXPECT_NEAR((*analytic)[k * base->nFuncs + e],
                        (plus->nuclear[e] - minus->nuclear[e]) / (2.0 * kFirstStep),
                        kFirstTolerance);
        }
    }
}

TEST(MdDerivativeTest, SameCenterPairMatchesCentralDifference) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const std::size_t shell = FindAngularShell(*pairList, 1);
    ASSERT_LT(shell, pairList->shells.size());

    // A same-center pair: one coordinate offers BOTH the bra and the ket
    // operator (and, for the nuclear attraction, the operator center too),
    // the slot branch the hetero pair never reaches.
    const std::size_t atom = pairList->shells[shell].atomIndex;
    const std::array<std::size_t, 3> coordinates{3 * atom + 0, 3 * atom + 1, 3 * atom + 2};
    auto base = BaseBlocks(*molecule, *basis, shell, shell);
    ASSERT_TRUE(base.has_value()) << base.error().message;
    auto analytic = DerivativeBlocks(*molecule, *basis, {shell, shell}, 1, coordinates);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;

    std::array<std::size_t, 1> flat{0};
    std::array<double, 1> delta{0.0};

    for (std::size_t k = 0; k < coordinates.size(); ++k)
    {
        flat[0] = coordinates[k];
        delta[0] = kFirstStep;
        auto plus = DisplacedBlocks(*molecule, *basis, shell, shell, flat, delta);
        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        delta[0] = -kFirstStep;
        auto minus = DisplacedBlocks(*molecule, *basis, shell, shell, flat, delta);
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        for (std::size_t e = 0; e < base->nFuncs; ++e)
        {
            SCOPED_TRACE("coordinate " + std::to_string(coordinates[k]) + ", element " +
                         std::to_string(e));
            EXPECT_NEAR(analytic->overlap[k * base->nFuncs + e],
                        (plus->overlap[e] - minus->overlap[e]) / (2.0 * kFirstStep),
                        kFirstTolerance);
            EXPECT_NEAR(analytic->kinetic[k * base->nFuncs + e],
                        (plus->kinetic[e] - minus->kinetic[e]) / (2.0 * kFirstStep),
                        kFirstTolerance);
            EXPECT_NEAR(analytic->nuclear[k * base->nFuncs + e],
                        (plus->nuclear[e] - minus->nuclear[e]) / (2.0 * kFirstStep),
                        kFirstTolerance);
        }
    }
}

TEST(MdDerivativeTest, TranslationInvarianceSumsToZero) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const HeteroPair pair = FindHeteroPair(*pairList);
    ASSERT_LT(pair.shellB, pairList->shells.size());

    // S and T depend on the pair's centers alone: the two atoms' derivatives
    // sum to zero along every axis, at every element.
    const std::vector<std::size_t> pairCoordinates = PairCoordinates(
        pairList->shells[pair.shellA].atomIndex, pairList->shells[pair.shellB].atomIndex);
    auto base = BaseBlocks(*molecule, *basis, pair.shellA, pair.shellB);
    ASSERT_TRUE(base.has_value()) << base.error().message;
    auto pairDerivative =
        DerivativeBlocks(*molecule, *basis, {pair.shellA, pair.shellB}, 1, pairCoordinates);
    ASSERT_TRUE(pairDerivative.has_value()) << pairDerivative.error().message;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        for (std::size_t e = 0; e < base->nFuncs; ++e)
        {
            SCOPED_TRACE("axis " + std::to_string(axis) + ", element " + std::to_string(e));
            // PairCoordinates lists the bra atom's three axes then the ket's.
            const std::size_t braIndex = axis * base->nFuncs + e;
            const std::size_t ketIndex = (3 + axis) * base->nFuncs + e;
            EXPECT_NEAR(
                pairDerivative->overlap[braIndex] + pairDerivative->overlap[ketIndex], 0.0, 1e-12);
            EXPECT_NEAR(
                pairDerivative->kinetic[braIndex] + pairDerivative->kinetic[ketIndex], 0.0, 1e-12);
        }
    }

    // V carries every nucleus: translating the whole molecule leaves it
    // invariant, so the nine coordinate derivatives sum to zero per axis.
    std::vector<std::size_t> allCoordinates;

    for (std::size_t atom = 0; atom < molecule->AtomCount(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            allCoordinates.push_back(3 * atom + axis);
        }
    }

    auto nuclear =
        NuclearDerivativeBlocks(*molecule, *basis, {pair.shellA, pair.shellB}, 1, allCoordinates);
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        for (std::size_t e = 0; e < base->nFuncs; ++e)
        {
            SCOPED_TRACE("axis " + std::to_string(axis) + ", element " + std::to_string(e));
            double sum = 0.0;

            for (std::size_t atom = 0; atom < molecule->AtomCount(); ++atom)
            {
                sum += (*nuclear)[(3 * atom + axis) * base->nFuncs + e];
            }

            EXPECT_NEAR(sum, 0.0, 1e-12);
        }
    }
}

TEST(MdDerivativeTest, SecondOrderMatchesCentralDifference) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const HeteroPair pair = FindHeteroPair(*pairList);
    ASSERT_LT(pair.shellB, pairList->shells.size());
    const std::vector<std::size_t> coordinates = PairCoordinates(
        pairList->shells[pair.shellA].atomIndex, pairList->shells[pair.shellB].atomIndex);
    ASSERT_EQ(coordinates.size(), 6);
    auto base = BaseBlocks(*molecule, *basis, pair.shellA, pair.shellB);
    ASSERT_TRUE(base.has_value()) << base.error().message;

    // Every ordered tuple of the pair's coordinates, each its own request.
    for (std::size_t i = 0; i < coordinates.size(); ++i)
    {
        for (std::size_t j = 0; j < coordinates.size(); ++j)
        {
            const std::array<std::size_t, 2> tuple{coordinates[i], coordinates[j]};
            auto analytic =
                DerivativeBlocks(*molecule, *basis, {pair.shellA, pair.shellB}, 2, tuple);
            ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
            ASSERT_EQ(analytic->overlap.size(), base->nFuncs);

            std::array<double, 2> delta{kSecondStep, kSecondStep};
            auto plusPlus =
                DisplacedBlocks(*molecule, *basis, pair.shellA, pair.shellB, tuple, delta);
            ASSERT_TRUE(plusPlus.has_value()) << plusPlus.error().message;
            delta[1] = -kSecondStep;
            auto plusMinus =
                DisplacedBlocks(*molecule, *basis, pair.shellA, pair.shellB, tuple, delta);
            ASSERT_TRUE(plusMinus.has_value()) << plusMinus.error().message;
            delta[0] = -kSecondStep;
            auto minusMinus =
                DisplacedBlocks(*molecule, *basis, pair.shellA, pair.shellB, tuple, delta);
            ASSERT_TRUE(minusMinus.has_value()) << minusMinus.error().message;
            delta[1] = kSecondStep;
            auto minusPlus =
                DisplacedBlocks(*molecule, *basis, pair.shellA, pair.shellB, tuple, delta);
            ASSERT_TRUE(minusPlus.has_value()) << minusPlus.error().message;

            const std::vector<double> overlap = CrossDifference(plusPlus->overlap,
                                                                plusMinus->overlap,
                                                                minusPlus->overlap,
                                                                minusMinus->overlap,
                                                                kSecondStep);
            const std::vector<double> kinetic = CrossDifference(plusPlus->kinetic,
                                                                plusMinus->kinetic,
                                                                minusPlus->kinetic,
                                                                minusMinus->kinetic,
                                                                kSecondStep);
            const std::vector<double> nuclear = CrossDifference(plusPlus->nuclear,
                                                                plusMinus->nuclear,
                                                                minusPlus->nuclear,
                                                                minusMinus->nuclear,
                                                                kSecondStep);

            for (std::size_t e = 0; e < base->nFuncs; ++e)
            {
                SCOPED_TRACE("tuple " + std::to_string(tuple[0]) + ", " + std::to_string(tuple[1]) +
                             ", element " + std::to_string(e));
                EXPECT_NEAR(analytic->overlap[e],
                            overlap[e],
                            kSecondTolerance * (1.0 + std::abs(analytic->overlap[e])));
                EXPECT_NEAR(analytic->kinetic[e],
                            kinetic[e],
                            kSecondTolerance * (1.0 + std::abs(analytic->kinetic[e])));
                EXPECT_NEAR(analytic->nuclear[e],
                            nuclear[e],
                            kSecondTolerance * (1.0 + std::abs(analytic->nuclear[e])));
            }
        }
    }
}

TEST(MdDerivativeTest, MixedOperatorCentersOnDifferentNucleiAreExactlyZero) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const std::size_t shell = FindAngularShell(*pairList, 1);
    ASSERT_LT(shell, pairList->shells.size());

    // A same-center pair addressed by two OTHER nuclei: each slot names a
    // nucleus that carries neither shell, so each offers the operator center
    // alone, and the two name different nuclei - the second derivative is
    // identically zero, the cross-nucleus term the enumerator drops, and the
    // finite difference agrees.
    const std::size_t ownAtom = pairList->shells[shell].atomIndex;
    std::array<std::size_t, 2> others{};
    std::size_t otherCount = 0;

    for (std::size_t atom = 0; atom < molecule->AtomCount() && otherCount < others.size(); ++atom)
    {
        if (atom != ownAtom)
        {
            others[otherCount] = atom;
            ++otherCount;
        }
    }

    ASSERT_EQ(otherCount, others.size());
    const std::array<std::size_t, 2> tuple{3 * others[0] + 0, 3 * others[1] + 0};
    auto analytic = NuclearDerivativeBlocks(*molecule, *basis, {shell, shell}, 2, tuple);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    auto base = BaseBlocks(*molecule, *basis, shell, shell);
    ASSERT_TRUE(base.has_value()) << base.error().message;

    for (std::size_t e = 0; e < base->nFuncs; ++e)
    {
        EXPECT_EQ((*analytic)[e], 0.0);
    }

    std::array<double, 2> delta{kSecondStep, kSecondStep};
    auto plusPlus = DisplacedBlocks(*molecule, *basis, shell, shell, tuple, delta);
    ASSERT_TRUE(plusPlus.has_value()) << plusPlus.error().message;
    delta[1] = -kSecondStep;
    auto plusMinus = DisplacedBlocks(*molecule, *basis, shell, shell, tuple, delta);
    ASSERT_TRUE(plusMinus.has_value()) << plusMinus.error().message;
    delta[0] = -kSecondStep;
    auto minusMinus = DisplacedBlocks(*molecule, *basis, shell, shell, tuple, delta);
    ASSERT_TRUE(minusMinus.has_value()) << minusMinus.error().message;
    delta[1] = kSecondStep;
    auto minusPlus = DisplacedBlocks(*molecule, *basis, shell, shell, tuple, delta);
    ASSERT_TRUE(minusPlus.has_value()) << minusPlus.error().message;
    const std::vector<double> nuclear = CrossDifference(plusPlus->nuclear,
                                                        plusMinus->nuclear,
                                                        minusPlus->nuclear,
                                                        minusMinus->nuclear,
                                                        kSecondStep);

    for (std::size_t e = 0; e < base->nFuncs; ++e)
    {
        SCOPED_TRACE("element " + std::to_string(e));
        EXPECT_NEAR(nuclear[e], 0.0, kSecondTolerance);
    }
}

TEST(MdDerivativeTest, TheOrderIsACallArgumentNotASecondFunction) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const HeteroPair pair = FindHeteroPair(*pairList);
    ASSERT_LT(pair.shellB, pairList->shells.size());
    auto pairAt = PairDataAt(*molecule, *basis, pair.shellA, pair.shellB);
    ASSERT_TRUE(pairAt.has_value()) << pairAt.error().message;
    const MdPairData& data = pairAt->pair;
    // The canonical shell order puts the hydrogen 1s first, so the pair's
    // angular momentum sits on the ket: what matters here is that the pair
    // carries one.
    ASSERT_GE(data.la + data.lb, 1);

    const std::array<std::size_t, 1> single{0};
    const std::array<std::size_t, 2> two{0, 1};
    const std::array<std::size_t, 3> three{0, 1, 2};

    // The block span the caller must provide is the tuple count times the
    // pair block - one formula, every order.
    EXPECT_EQ(qcx::integrals::internal::MdDerivativeBlockCount(data, 1, single.size()),
              data.nFuncs);
    EXPECT_EQ(qcx::integrals::internal::MdDerivativeBlockCount(data, 1, two.size()),
              2 * data.nFuncs);
    EXPECT_EQ(qcx::integrals::internal::MdDerivativeBlockCount(data, 2, two.size()), data.nFuncs);
    EXPECT_EQ(qcx::integrals::internal::MdDerivativeBlockCount(data, 2, three.size()), 0);
    EXPECT_EQ(qcx::integrals::internal::MdDerivativeBlockCount(data, 0, single.size()), 0);

    // Asking for order 1 gives the first derivative: the two-coordinate
    // request's first block is the one-coordinate request's block, and the
    // second is the same function on the second coordinate.
    auto first = DerivativeBlocks(*molecule, *basis, {pair.shellA, pair.shellB}, 1, single);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    auto both = DerivativeBlocks(*molecule, *basis, {pair.shellA, pair.shellB}, 1, two);
    ASSERT_TRUE(both.has_value()) << both.error().message;
    ASSERT_EQ(both->overlap.size(), 2 * data.nFuncs);

    for (std::size_t e = 0; e < data.nFuncs; ++e)
    {
        EXPECT_DOUBLE_EQ(both->overlap[e], first->overlap[e]);
        EXPECT_DOUBLE_EQ(both->kinetic[e], first->kinetic[e]);
        EXPECT_DOUBLE_EQ(both->nuclear[e], first->nuclear[e]);
    }

    // A higher order is askable through the SAME function: order 2 over the
    // same two coordinates returns one block, and it is a second derivative
    // - not the first one back again.
    auto second = DerivativeBlocks(*molecule, *basis, {pair.shellA, pair.shellB}, 2, two);
    ASSERT_TRUE(second.has_value()) << second.error().message;
    ASSERT_EQ(second->overlap.size(), data.nFuncs);
    double largest = 0.0;

    for (std::size_t e = 0; e < data.nFuncs; ++e)
    {
        largest = std::max(largest, std::abs(second->overlap[e] - first->overlap[e]));
    }

    EXPECT_GT(largest, 1e-6);
}

TEST(MdDerivativeTest, DerivativeRequestsAreRefusedByName) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const HeteroPair pair = FindHeteroPair(*pairList);
    ASSERT_LT(pair.shellB, pairList->shells.size());
    auto pairAt = PairDataAt(*molecule, *basis, pair.shellA, pair.shellB);
    ASSERT_TRUE(pairAt.has_value()) << pairAt.error().message;
    const MdPairData& data = pairAt->pair;
    qcx::integrals::internal::MdDerivativeScratch scratch;
    std::vector<double> blocks(data.nFuncs, 0.0);
    const std::array<std::size_t, 1> coordinate{0};

    // An order outside the band the recurrence takes.
    auto zeroOrder = qcx::integrals::internal::BuildOverlapPairDerivative(
        data, pairAt->braAtom, pairAt->ketAtom, 0, coordinate, blocks, scratch);
    ASSERT_FALSE(zeroOrder.has_value());
    EXPECT_EQ(zeroOrder.error().code, qcx::ErrorCode::kInvalidArgument);
    auto tooHigh = qcx::integrals::internal::BuildOverlapPairDerivative(
        data,
        pairAt->braAtom,
        pairAt->ketAtom,
        qcx::integrals::internal::kMaxMdDerivativeOrder + 1,
        coordinate,
        blocks,
        scratch);
    ASSERT_FALSE(tooHigh.has_value());
    EXPECT_EQ(tooHigh.error().code, qcx::ErrorCode::kInvalidArgument);

    // A coordinate list that is not a whole number of tuples.
    const std::array<std::size_t, 3> odd{0, 1, 2};
    auto ragged = qcx::integrals::internal::BuildOverlapPairDerivative(
        data, pairAt->braAtom, pairAt->ketAtom, 2, odd, blocks, scratch);
    ASSERT_FALSE(ragged.has_value());
    EXPECT_EQ(ragged.error().code, qcx::ErrorCode::kInvalidArgument);

    // A block span that is not the tuple count times the pair block.
    std::vector<double> shortBlocks(data.nFuncs > 1 ? data.nFuncs - 1 : 1, 0.0);
    auto wrongSpan = qcx::integrals::internal::BuildOverlapPairDerivative(
        data, pairAt->braAtom, pairAt->ketAtom, 1, coordinate, shortBlocks, scratch);
    ASSERT_FALSE(wrongSpan.has_value());
    EXPECT_EQ(wrongSpan.error().code, qcx::ErrorCode::kInvalidArgument);

    // A coordinate naming an atom that carries neither shell of the pair.
    // The fixture's molecule is renumbered by (Z, x, y, z): the pair's atoms
    // are one hydrogen and the oxygen, so the remaining hydrogen is the only
    // atom that qualifies - found from the shell list, never named.
    const std::size_t unrelatedAtom =
        FindThirdAtom(*pairList, {pair.shellA, pair.shellB}, molecule->AtomCount());
    ASSERT_LT(unrelatedAtom, molecule->AtomCount());
    const std::array<std::size_t, 1> unrelated{3 * unrelatedAtom + 0};
    auto unrelatedOverlap = qcx::integrals::internal::BuildOverlapPairDerivative(
        data, pairAt->braAtom, pairAt->ketAtom, 1, unrelated, blocks, scratch);
    ASSERT_FALSE(unrelatedOverlap.has_value());
    EXPECT_EQ(unrelatedOverlap.error().code, qcx::ErrorCode::kInvalidArgument);
    auto unrelatedKinetic = qcx::integrals::internal::BuildKineticPairDerivative(
        data, pairAt->braAtom, pairAt->ketAtom, 1, unrelated, blocks, scratch);
    ASSERT_FALSE(unrelatedKinetic.has_value());
    EXPECT_EQ(unrelatedKinetic.error().code, qcx::ErrorCode::kInvalidArgument);

    // The nuclear attraction sums every nucleus, but not an atom that does
    // not exist.
    const std::vector<double> charges = NuclearCharges(*molecule);
    const std::vector<Eigen::Vector3d> centers = qcx::integrals::internal::AtomPositions(*molecule);
    const std::array<std::size_t, 1> outOfRange{3 * molecule->AtomCount()};
    auto badNucleus = qcx::integrals::internal::BuildNuclearPairDerivative(
        data, pairAt->braAtom, pairAt->ketAtom, charges, centers, 1, outOfRange, blocks, scratch);
    ASSERT_FALSE(badNucleus.has_value());
    EXPECT_EQ(badNucleus.error().code, qcx::ErrorCode::kInvalidArgument);

    // The third atom IS a valid nuclear-attraction coordinate.
    auto thirdNucleus = qcx::integrals::internal::BuildNuclearPairDerivative(
        data, pairAt->braAtom, pairAt->ketAtom, charges, centers, 1, unrelated, blocks, scratch);
    ASSERT_TRUE(thirdNucleus.has_value()) << thirdNucleus.error().message;
}

} // namespace
