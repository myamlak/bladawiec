// The two-electron derivative tier's tests: the analytic nuclear-coordinate
// derivatives of a shell quartet's packed block against the CENTRAL FINITE
// DIFFERENCE of the production batch engine (ComputeEriBatch) at displaced
// geometries. The finite difference runs no second analytic path - it is the
// engine the derivative tier differentiates - so a drift between the
// derivative's algebra and the value pipeline shows up as a mismatch. The
// zero-operator term of the same contraction is checked against the engine's
// own block first, which pins the layout before any derivative is involved.
#include "h2o_sto3g.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_eri_derivative.hpp"
#include "internal/shells_flat.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/eri_batch.hpp"
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
using qcx::integrals::ShellPairList;
using qcx::integrals::ShellQuartet;
using qcx::integrals::internal::MdEriDerivativeQuartet;
using qcx::integrals::internal::MdEriDerivativeScratch;
using qcx::integrals::internal::MdShellInput;

/// The central-difference step of the first-derivative check: the truncation
/// is O(h^2) and the cancellation error O(eps/h), both near 1e-11 at the
/// quartets' O(1) magnitudes - comfortably inside the 1e-7 threshold.
constexpr double kFirstStep = 1e-5;

/// The analytic-versus-finite-difference threshold of the two-electron
/// derivative blocks.
constexpr double kFirstTolerance = 1e-7;

/// The central-difference step of the second-derivative check: the four-point
/// cross difference carries an O(h^2) truncation against an O(eps/h^2)
/// cancellation error, and the measured worst discrepancy over the fixture's
/// tuples falls by the h^2 law down to this step, where the cancellation
/// takes over.
constexpr double kSecondStep = 1e-4;

/// The second-derivative tolerance, RELATIVE with an absolute floor. It is
/// the finite difference's limit, not the analytic accuracy of the tier.
constexpr double kSecondTolerance = 1e-6;

/// The value pin's relative tolerance: the derivative tier's contraction and
/// the batch engine's GEMM pipeline sum the same products in a different
/// order, so they agree to roundoff and not to the bit.
constexpr double kValueTolerance = 1e-12;

/// One shell of the fixture, with the atom that carries it.
struct ShellAt {
    std::size_t shell = 0; ///< Shell index.
    int angularMomentum = 0; ///< Its l.
    std::size_t atom = 0; ///< Its atom.
};

/// The shells of one atom, in the pair list's shell order.
std::vector<ShellAt> ShellsOfAtom(const ShellPairList& pairList, std::size_t atom) {
    std::vector<ShellAt> shells;

    for (std::size_t shell = 0; shell < pairList.shells.size(); ++shell)
    {
        if (pairList.shells[shell].atomIndex == atom)
        {
            shells.push_back(ShellAt{
                shell, pairList.shells[shell].angularMomentum, pairList.shells[shell].atomIndex});
        }
    }

    return shells;
}

/// The first atom whose nuclear charge is \p atomicNumber.
std::size_t AtomOfElement(const qcx::molecule::Molecule& molecule, int atomicNumber) {
    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        if (molecule.Atoms()[atom].atomicNumber == atomicNumber)
        {
            return atom;
        }
    }

    return molecule.AtomCount();
}

/// The geometry with the listed flat coordinates (3 * atom + axis) displaced
/// by the listed deltas; a repeated coordinate accumulates. Errors when the
/// canonical renumbering would move an atom - the finite-difference loop
/// reads the displaced molecule's shells by the BASE geometry's shell
/// indices, which is only valid when the order is unchanged.
qcx::Result<qcx::molecule::Molecule> MakeDisplaced(const qcx::molecule::Molecule& base,
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
            return std::unexpected(qcx::Error{
                qcx::ErrorCode::kInvalidArgument,
                "the displaced geometry is not the base geometry with one row moved - the shell "
                "indices of the finite-difference loop would name other shells"});
        }
    }

    return displaced;
}

/// One quartet's canonical form and packed block, as the production batch
/// engine returns them.
struct BatchBlock {
    ShellQuartet quartet{}; ///< The canonical quartet the engine evaluated.
    std::vector<double> values; ///< Its packed block.
};

/// The production engine's block for one quartet request.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \param request The quartet, in any form.
/// \returns The canonical quartet and its block, or an Error.
qcx::Result<BatchBlock> ReferenceBlock(const qcx::molecule::Molecule& molecule,
                                       const qcx::basisset::BasisSet& basisSet,
                                       const ShellQuartet& request) {
    auto batch = qcx::integrals::ComputeEriBatch(molecule, basisSet, {request});

    if (!batch.has_value())
    {
        return std::unexpected(batch.error());
    }

    if (batch->computed.size() != 1)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "the engine did not evaluate the one quartet"});
    }

    return BatchBlock{batch->computed[0], std::move(batch->values)};
}

/// The derivative quartet of one canonical quartet, with the fixture's
/// flattened shells and the pair list's atom indices.
/// \param pairList The shell pair list.
/// \param shells The flattened shell inputs (FlattenShells).
/// \param quartet The canonical quartet.
/// \returns The quartet the derivative builder takes.
MdEriDerivativeQuartet DerivativeQuartet(const ShellPairList& pairList,
                                         const std::vector<MdShellInput>& shells,
                                         const ShellQuartet& quartet) {
    const std::array<std::size_t, 4> indices = {quartet.i, quartet.j, quartet.k, quartet.l};
    MdEriDerivativeQuartet out;

    for (int slot = 0; slot < 4; ++slot)
    {
        const std::size_t index = indices[static_cast<std::size_t>(slot)];
        out.shells[static_cast<std::size_t>(slot)] = &shells[index];
        out.atoms[static_cast<std::size_t>(slot)] = pairList.shells[index].atomIndex;
    }

    return out;
}

/// The derivative tier's packed block of one canonical quartet at one
/// geometry, the term with no operator applied - the value the batch engine
/// must reproduce.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \param quartet The canonical quartet.
/// \returns The packed block, or an Error.
qcx::Result<std::vector<double>> ValueBlock(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet,
                                            const ShellQuartet& quartet) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto shells = qcx::integrals::internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    const MdEriDerivativeQuartet derivative = DerivativeQuartet(*pairList, *shells, quartet);
    std::vector<double> block(qcx::integrals::internal::EriQuartetBlockElements(derivative), 0.0);
    MdEriDerivativeScratch scratch;
    qcx::integrals::internal::BuildEriQuartetValue(derivative, 1, block, scratch);
    return block;
}

/// The analytic derivative blocks of one canonical quartet: one packed block
/// per tuple of \p coordinates, tuple-major.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \param quartets The canonical quartets.
/// \param order The derivative order.
/// \param coordinates The flat coordinates, order values per tuple.
/// \returns The blocks, one vector per quartet, or an Error.
qcx::Result<std::vector<std::vector<double>>> AnalyticBlocks(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    std::span<const ShellQuartet> quartets,
    int order,
    std::span<const std::size_t> coordinates) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto shells = qcx::integrals::internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    MdEriDerivativeScratch scratch;
    std::vector<std::vector<double>> blocks;

    for (const ShellQuartet& quartet : quartets)
    {
        const MdEriDerivativeQuartet derivative = DerivativeQuartet(*pairList, *shells, quartet);
        std::vector<double> block(qcx::integrals::internal::MdEriDerivativeBlockCount(
                                      derivative, order, coordinates.size()),
                                  0.0);
        auto built = qcx::integrals::internal::BuildEriQuartetDerivative(
            derivative, order, coordinates, block, scratch, molecule.AtomCount());

        if (!built.has_value())
        {
            return std::unexpected(built.error());
        }

        blocks.push_back(std::move(block));
    }

    return blocks;
}

/// The finite-difference reference of one canonical quartet: the engine's
/// block at the geometry with the listed coordinates displaced. The canonical
/// form at the displaced geometry must be the base one, or the block indices
/// would name another quartet.
/// \param molecule The base molecule.
/// \param basisSet The basis set.
/// \param request The quartet request the base block came from.
/// \param baseBlock The base geometry's canonical form and its block.
/// \param flatCoordinates The displaced coordinates.
/// \param deltas The displacements.
/// \returns The displaced block, or an Error.
qcx::Result<std::vector<double>> DisplacedBlock(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const ShellQuartet& request,
                                                const BatchBlock& baseBlock,
                                                std::span<const std::size_t> flatCoordinates,
                                                std::span<const double> deltas) {
    auto displaced = MakeDisplaced(molecule, flatCoordinates, deltas);

    if (!displaced.has_value())
    {
        return std::unexpected(displaced.error());
    }

    auto block = ReferenceBlock(*displaced, basisSet, request);

    if (!block.has_value())
    {
        return std::unexpected(block.error());
    }

    if (!(block->quartet == baseBlock.quartet))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the displaced geometry canonicalizes the quartet differently - the finite "
                       "difference would difference another block"});
    }

    return block->values;
}

/// A double as a short scientific string, for the recorded properties.
std::string ShortDouble(double value) {
    char text[32] = {};
    std::snprintf(text, sizeof(text), "%.3e", value);
    return text;
}

/// The largest absolute deviation of two equal-length block sets.
double WorstDeviation(const std::vector<double>& analytic, const std::vector<double>& reference) {
    double worst = 0.0;

    for (std::size_t i = 0; i < analytic.size(); ++i)
    {
        worst = std::max(worst, std::abs(analytic[i] - reference[i]));
    }

    return worst;
}

/// The max element magnitude of a block set, the scale a tolerance is read
/// against.
double LargestValue(const std::vector<double>& values) {
    double largest = 0.0;

    for (const double value : values)
    {
        largest = std::max(largest, std::abs(value));
    }

    return largest;
}

/// The fixture's water molecule and STO-3G basis.
struct Water {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
};

qcx::Result<Water> MakeWater() {
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

    return Water{std::move(*molecule), std::move(*basis)};
}

/// The flat coordinates of every atom and axis, in atom then axis order.
std::vector<std::size_t> AllCoordinates(std::size_t atomCount) {
    std::vector<std::size_t> coordinates;

    for (std::size_t atom = 0; atom < atomCount; ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            coordinates.push_back(3 * atom + axis);
        }
    }

    return coordinates;
}

/// The s-only quartet the fixture's hydrogens define: both hydrogens' 1s
/// shells on both sides - the simplest two-body family of the tier, one
/// contracted primitive set per shell and no angular momentum anywhere.
qcx::Result<ShellQuartet> HydrogenQuartet(const Water& water) {
    auto pairList = qcx::integrals::BuildShellPairs(water.molecule, water.basis);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    std::vector<std::size_t> hydrogens;
    std::vector<std::size_t> sShells;

    for (std::size_t atom = 0; atom < water.molecule.AtomCount(); ++atom)
    {
        if (water.molecule.Atoms()[atom].atomicNumber == 1)
        {
            hydrogens.push_back(atom);
        }
    }

    if (hydrogens.size() != 2)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "the fixture is not a two-hydrogen water"});
    }

    for (const std::size_t hydrogen : hydrogens)
    {
        const std::vector<ShellAt> shells = ShellsOfAtom(*pairList, hydrogen);

        if (shells.size() != 1 || shells[0].angularMomentum != 0)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                              "an STO-3G hydrogen is not one 1s shell"});
        }

        sShells.push_back(shells[0].shell);
    }

    return ShellQuartet{sShells[0], sShells[1], sShells[0], sShells[1]};
}

/// The mixed quartet the fixture's oxygen and one hydrogen define: the
/// oxygen's low s shell and the hydrogen's 1s on the bra, the two hydrogens'
/// 1s shells on the ket.
qcx::Result<ShellQuartet> OxygenHydrogenQuartet(const Water& water) {
    auto pairList = qcx::integrals::BuildShellPairs(water.molecule, water.basis);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    const std::size_t oxygen = AtomOfElement(water.molecule, 8);

    if (oxygen >= water.molecule.AtomCount())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "the fixture is not a water molecule"});
    }

    std::vector<std::size_t> oxygenS;
    std::vector<std::size_t> hydrogenS;

    for (const ShellAt& shell : ShellsOfAtom(*pairList, oxygen))
    {
        if (shell.angularMomentum == 0)
        {
            oxygenS.push_back(shell.shell);
        }
    }

    for (std::size_t atom = 0; atom < water.molecule.AtomCount(); ++atom)
    {
        if (water.molecule.Atoms()[atom].atomicNumber != 1)
        {
            continue;
        }

        for (const ShellAt& shell : ShellsOfAtom(*pairList, atom))
        {
            if (shell.angularMomentum == 0)
            {
                hydrogenS.push_back(shell.shell);
            }
        }
    }

    if (oxygenS.empty() || hydrogenS.size() != 2)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "the fixture has no oxygen/hydrogen s shells"});
    }

    return ShellQuartet{oxygenS[0], hydrogenS[0], hydrogenS[0], hydrogenS[1]};
}

/// The sp quartet: the oxygen's p shell against a hydrogen 1s on the bra, the
/// same two on the ket - the family that exercises the raised coefficient
/// tables and the spherical fold.
qcx::Result<ShellQuartet> OxygenPHydrogenQuartet(const Water& water) {
    auto pairList = qcx::integrals::BuildShellPairs(water.molecule, water.basis);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    const std::size_t oxygen = AtomOfElement(water.molecule, 8);
    std::vector<std::size_t> oxygenP;
    std::vector<std::size_t> hydrogenS;

    for (const ShellAt& shell : ShellsOfAtom(*pairList, oxygen))
    {
        if (shell.angularMomentum == 1)
        {
            oxygenP.push_back(shell.shell);
        }
    }

    for (std::size_t atom = 0; atom < water.molecule.AtomCount(); ++atom)
    {
        if (water.molecule.Atoms()[atom].atomicNumber != 1)
        {
            continue;
        }

        for (const ShellAt& shell : ShellsOfAtom(*pairList, atom))
        {
            if (shell.angularMomentum == 0)
            {
                hydrogenS.push_back(shell.shell);
            }
        }
    }

    if (oxygenP.empty() || hydrogenS.size() != 2)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "the fixture has no oxygen p shell"});
    }

    return ShellQuartet{oxygenP[0], hydrogenS[0], oxygenP[0], hydrogenS[1]};
}

/// One family's first-derivative acceptance: the analytic block against the
/// central difference of the engine's own block, over every atom and axis.
/// \param water The fixture.
/// \param request The family's quartet request.
/// \returns The worst deviation over the family's tuples and elements.
qcx::Result<double> FirstDerivativeDeviation(const Water& water, const ShellQuartet& request) {
    auto reference = ReferenceBlock(water.molecule, water.basis, request);

    if (!reference.has_value())
    {
        return std::unexpected(reference.error());
    }

    const std::vector<std::size_t> coordinates = AllCoordinates(water.molecule.AtomCount());
    auto analytic = AnalyticBlocks(water.molecule,
                                   water.basis,
                                   std::span<const ShellQuartet>(&reference->quartet, 1),
                                   1,
                                   coordinates);

    if (!analytic.has_value())
    {
        return std::unexpected(analytic.error());
    }

    double worst = 0.0;

    for (std::size_t tuple = 0; tuple < coordinates.size(); ++tuple)
    {
        const std::size_t flat = coordinates[tuple];
        const std::array<std::size_t, 1> one = {flat};
        const std::array<double, 1> plus = {kFirstStep};
        const std::array<double, 1> minus = {-kFirstStep};
        auto forward = DisplacedBlock(water.molecule, water.basis, request, *reference, one, plus);

        if (!forward.has_value())
        {
            return std::unexpected(forward.error());
        }

        auto backward =
            DisplacedBlock(water.molecule, water.basis, request, *reference, one, minus);

        if (!backward.has_value())
        {
            return std::unexpected(backward.error());
        }

        std::vector<double> difference(forward->size(), 0.0);

        for (std::size_t i = 0; i < difference.size(); ++i)
        {
            difference[i] = ((*forward)[i] - (*backward)[i]) / (2.0 * kFirstStep);
        }

        std::vector<double> tupleAnalytic(
            analytic->front().begin() + static_cast<std::ptrdiff_t>(tuple * difference.size()),
            analytic->front().begin() +
                static_cast<std::ptrdiff_t>((tuple + 1) * difference.size()));
        worst = std::max(worst, WorstDeviation(tupleAnalytic, difference));
    }

    return worst;
}

/// One named quartet family of the acceptance runs.
struct Family {
    std::string name; ///< The family's name, for a failure message.
    ShellQuartet quartet; ///< Its quartet request.
};

/// The families the acceptance runs cover, in widening order: s-only across
/// two centres, then oxygen s against hydrogen s, then the oxygen p shell
/// against hydrogen s.
/// \param water The fixture.
/// \returns The families, or an Error.
qcx::Result<std::vector<Family>> Families(const Water& water) {
    std::vector<Family> families;
    auto hydrogen = HydrogenQuartet(water);

    if (!hydrogen.has_value())
    {
        return std::unexpected(hydrogen.error());
    }

    families.push_back(Family{"hydrogen 1s x hydrogen 1s", *hydrogen});
    auto mixed = OxygenHydrogenQuartet(water);

    if (!mixed.has_value())
    {
        return std::unexpected(mixed.error());
    }

    families.push_back(Family{"oxygen 1s + hydrogen 1s x hydrogen 1s squared", *mixed});
    auto polarized = OxygenPHydrogenQuartet(water);

    if (!polarized.has_value())
    {
        return std::unexpected(polarized.error());
    }

    families.push_back(Family{"oxygen 2p + hydrogen 1s x oxygen 2p + hydrogen 1s", *polarized});
    return families;
}

TEST(TwoElectronDerivativeTest, ValueTermMatchesTheBatchEngine) {
    auto water = MakeWater();
    ASSERT_TRUE(water.has_value()) << water.error().message;
    auto families = Families(*water);
    ASSERT_TRUE(families.has_value()) << families.error().message;

    for (const Family& family : *families)
    {
        auto reference = ReferenceBlock(water->molecule, water->basis, family.quartet);
        ASSERT_TRUE(reference.has_value()) << reference.error().message;
        auto value = ValueBlock(water->molecule, water->basis, reference->quartet);
        ASSERT_TRUE(value.has_value()) << value.error().message;
        ASSERT_EQ(value->size(), reference->values.size()) << family.name;
        const double scale = std::max(1.0, LargestValue(reference->values));
        EXPECT_LE(WorstDeviation(*value, reference->values), kValueTolerance * scale)
            << family.name;
    }
}

// The four shells of the quartet all sit on one atom, so a rigid translation
// of that atom moves every centre and the block is invariant: its derivative
// with respect to that atom's coordinates is exactly zero, with no finite
// difference in the way.
/// One family's second-derivative acceptance: the analytic block against the
/// four-point cross difference of the engine's own block, over every ordered
/// pair of coordinates. A tuple naming one coordinate twice collapses to the
/// three-point difference with step 2h - its +/- samples are the undisplaced
/// value - so one expression serves both shapes.
/// \param water The fixture.
/// \param request The family's quartet request.
/// \returns The worst deviation over the family's tuples and elements.
qcx::Result<double> SecondDerivativeDeviation(const Water& water, const ShellQuartet& request) {
    auto reference = ReferenceBlock(water.molecule, water.basis, request);

    if (!reference.has_value())
    {
        return std::unexpected(reference.error());
    }

    const std::vector<std::size_t> coordinates = AllCoordinates(water.molecule.AtomCount());
    std::vector<std::size_t> tuples;

    for (const std::size_t first : coordinates)
    {
        for (const std::size_t second : coordinates)
        {
            tuples.push_back(first);
            tuples.push_back(second);
        }
    }

    auto analytic = AnalyticBlocks(water.molecule,
                                   water.basis,
                                   std::span<const ShellQuartet>(&reference->quartet, 1),
                                   2,
                                   tuples);

    if (!analytic.has_value())
    {
        return std::unexpected(analytic.error());
    }

    const std::size_t elements = reference->values.size();
    double worst = 0.0;

    for (std::size_t tuple = 0; tuple < tuples.size() / 2; ++tuple)
    {
        const std::array<std::size_t, 2> flat = {tuples[2 * tuple], tuples[2 * tuple + 1]};
        const std::array<double, 2> plus = {kSecondStep, kSecondStep};
        const std::array<double, 2> plusMinus = {kSecondStep, -kSecondStep};
        const std::array<double, 2> minusPlus = {-kSecondStep, kSecondStep};
        const std::array<double, 2> minus = {-kSecondStep, -kSecondStep};
        std::array<std::vector<double>, 4> blocks;

        for (std::size_t corner = 0; corner < 4; ++corner)
        {
            const std::array<double, 2> deltas = corner == 0   ? plus
                                                 : corner == 1 ? plusMinus
                                                 : corner == 2 ? minusPlus
                                                               : minus;
            auto block =
                DisplacedBlock(water.molecule, water.basis, request, *reference, flat, deltas);

            if (!block.has_value())
            {
                return std::unexpected(block.error());
            }

            blocks[corner] = std::move(*block);
        }

        const double scale = 1.0 / (4.0 * kSecondStep * kSecondStep);

        for (std::size_t element = 0; element < elements; ++element)
        {
            const double difference = (blocks[0][element] - blocks[1][element] -
                                       blocks[2][element] + blocks[3][element]) *
                                      scale;
            const double value = (*analytic)[0][tuple * elements + element];
            worst = std::max(worst, std::abs(value - difference));
        }
    }

    return worst;
}

TEST(TwoElectronDerivativeTest, SecondDerivativeMatchesCrossDifference) {
    auto water = MakeWater();
    ASSERT_TRUE(water.has_value()) << water.error().message;
    auto quartet = HydrogenQuartet(*water);
    ASSERT_TRUE(quartet.has_value()) << quartet.error().message;
    auto worst = SecondDerivativeDeviation(*water, *quartet);
    ASSERT_TRUE(worst.has_value()) << worst.error().message;
    RecordProperty("worst_deviation", ShortDouble(*worst));
    EXPECT_LE(*worst, kSecondTolerance) << "worst deviation " << *worst;
}

TEST(TwoElectronDerivativeTest, ATotallyTranslatingQuartetHasAnExactlyZeroDerivative) {
    auto water = MakeWater();
    ASSERT_TRUE(water.has_value()) << water.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(water->molecule, water->basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const std::size_t oxygen = AtomOfElement(water->molecule, 8);
    ASSERT_LT(oxygen, water->molecule.AtomCount());
    std::vector<std::size_t> oxygenS;

    for (const ShellAt& shell : ShellsOfAtom(*pairList, oxygen))
    {
        if (shell.angularMomentum == 0)
        {
            oxygenS.push_back(shell.shell);
        }
    }

    ASSERT_GE(oxygenS.size(), 2u);
    const ShellQuartet quartet{oxygenS[0], oxygenS[1], oxygenS[0], oxygenS[1]};
    auto reference = ReferenceBlock(water->molecule, water->basis, quartet);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    // The engine's own block: the translation invariance is a property of the
    // quartet, not of the derivative tier, so the reference is checked too.
    EXPECT_GT(LargestValue(reference->values), 0.0);
    ASSERT_EQ(reference->quartet.i, oxygenS[0]);
    const std::vector<std::size_t> coordinates = {3 * oxygen, 3 * oxygen + 1, 3 * oxygen + 2};
    auto analytic = AnalyticBlocks(water->molecule,
                                   water->basis,
                                   std::span<const ShellQuartet>(&reference->quartet, 1),
                                   1,
                                   coordinates);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    const double largest = LargestValue(*analytic->data());
    RecordProperty("translating_derivative", ShortDouble(largest));
    EXPECT_LE(largest, 1.0e-14) << "the translating quartet's derivative";
}

TEST(TwoElectronDerivativeTest, FirstDerivativeMatchesCentralDifference) {
    auto water = MakeWater();
    ASSERT_TRUE(water.has_value()) << water.error().message;
    auto families = Families(*water);
    ASSERT_TRUE(families.has_value()) << families.error().message;

    double overall = 0.0;

    for (const Family& family : *families)
    {
        auto worst = FirstDerivativeDeviation(*water, family.quartet);
        ASSERT_TRUE(worst.has_value()) << worst.error().message;
        overall = std::max(overall, *worst);
        EXPECT_LE(*worst, kFirstTolerance)
            << family.name << ": worst deviation against the central difference " << *worst;
    }

    // The worst deviation over every family, tuple and element - the number
    // the tier is accepted at, recorded rather than only asserted.
    RecordProperty("worst_deviation", ShortDouble(overall));
}

} // namespace
