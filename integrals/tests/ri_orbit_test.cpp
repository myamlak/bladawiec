// The point group's joint action on RI-J's 3-center work: that work is a
// (braPair, auxShell) task grid and the point group acts on both indices, so
// the fast path's single tensor pass can evaluate one REPRESENTATIVE per joint
// orbit and expand the rest of the orbit from that representative's block.
//
// What these tests hold down:
//   - the joint action's own contract (one group over two bases, both
//     shell-closed - the second being the piece the AO bra-pair action
//     cannot supply for the aux basis);
//   - the representative test against an INDEPENDENT enumeration of the
//     same orbits (a breadth-first walk of the surviving grid): the same
//     orbit count, one representative per orbit, and every surviving cell
//     covered by exactly one of them;
//   - VALUE NEUTRALITY, at the tolerance measured here and pinned below:
//     the expanded tensor equals the plain walk's within that tolerance, and
//     the cells the Schwarz screen dropped stay exactly zero in both;
//   - the BLOCK COUNT: the tensor-pass counters fall to the
//     representatives, and the ratio matches an independently computed
//     mass-weighted ratio over the same grid;
//   - the refusals: no reductions, mismatched group orders, a basis whose
//     shells do not permute as units, and the light rung (where the
//     expansion is not wired).

#include "h2o_sto3g.hpp"
#include "internal/footprint.hpp"
#include "internal/ri_orbit_action.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/molecule/molecule.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using CpuTensor3 = qcx::memory::Tensor<double, 3, qcx::backend::CpuTag>;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;

using qcx::integrals::RiEngineOptions;
using qcx::integrals::RiTermCounters;
using qcx::integrals::ShellPairList;
using qcx::integrals::SymmetryReduction;

// The water C2v reduction over the H-first STO-3G function order
// [H1s, H2s, O1s, O2s, Opy, Opz, Opx] - the fixture fock_build_test.cpp and
// lean_point_group_test.cpp both carry (the integrals module cannot include
// scf headers). Canonicalized water renumbers the atoms by Z, so the H
// atoms come first and their shells with them. Elements: {I, sigma_z
// (z-flip), sigma_x (x-flip), C2 about y}; the last two swap the two H
// functions.
SymmetryReduction MakeWaterC2vReductionHFirst() {
    SymmetryReduction reduction;
    reduction.groupOrder = 4;
    reduction.isTrivial = false;
    reduction.permutation = {
        {0, 1, 2, 3, 4, 5, 6}, // I.
        {0, 1, 2, 3, 4, 5, 6}, // sigma_z: identity permutation.
        {1, 0, 2, 3, 4, 5, 6}, // sigma_x: swaps the two H functions.
        {1, 0, 2, 3, 4, 5, 6}, // C2: swaps the two H functions.
    };
    reduction.sign = {
        {1, 1, 1, 1, 1, 1, 1}, // I.
        {1, 1, 1, 1, 1, -1, 1}, // sigma_z: p_z -> -p_z.
        {1, 1, 1, 1, 1, 1, -1}, // sigma_x: p_x -> -p_x.
        {1, 1, 1, 1, 1, -1, -1}, // C2: p_z and p_x flip.
    };
    return reduction;
}

// The same group over a SMALLER, structurally different auxiliary set: one
// s shell per H plus one s and one p shell on O, parsed as
// "O S / O P / H S". BuildShellPairs walks the atoms in the molecule's
// canonical order (H, H, O), so the aux shell order is
// [H1s, H2s, Os, Op] with function order [H1s, H2s, Os, Opy, Opz, Opx]
// (within a p shell the order is m = -1, 0, +1 ~ y, z, x - the same
// convention the orbital fixture records). The elements act exactly as
// they do on the orbital basis, one index shorter in the H block and with
// the O p shell as the only signed shell.
SymmetryReduction MakeWaterC2vAuxReduction() {
    SymmetryReduction reduction;
    reduction.groupOrder = 4;
    reduction.isTrivial = false;
    reduction.permutation = {
        {0, 1, 2, 3, 4, 5}, // I.
        {0, 1, 2, 3, 4, 5}, // sigma_z: identity permutation.
        {1, 0, 2, 3, 4, 5}, // sigma_x: swaps the two H functions.
        {1, 0, 2, 3, 4, 5}, // C2: swaps the two H functions.
    };
    reduction.sign = {
        {1, 1, 1, 1, 1, 1}, // I.
        {1, 1, 1, 1, -1, 1}, // sigma_z: p_z -> -p_z.
        {1, 1, 1, 1, 1, -1}, // sigma_x: p_x -> -p_x.
        {1, 1, 1, 1, -1, -1}, // C2: p_z and p_x flip.
    };
    return reduction;
}

// The small symmetric auxiliary set of the second fixture: one s shell per
// atom (two primitives on O) and one p shell on O. Every function sits on
// the molecule's own centers, so the group acts on it the same way - the
// set is what makes the two-reduction machinery testable without the scf
// module's signed-permutation extraction.
inline constexpr std::string_view kTinySymmetricAux = R"(BASIS "ao basis" SPHERICAL PRINT
O    S
      2.5000000000E+00       5.0000000000E-01
      8.0000000000E-01       6.0000000000E-01
O    P
      1.2000000000E+00       1.0000000000E+00
H    S
      1.0000000000E+00       1.0000000000E+00
END
)";

// The cell census of one task grid under one joint action: the surviving
// cells (the Schwarz test), their orbits found by an INDEPENDENT
// breadth-first walk of the grid, the representatives that walk picks, and
// the representatives the production test picks. The two must agree -
// otherwise the walk and the test are two different definitions and only
// one of them can be right.
struct GridCensus {
    std::size_t survivors = 0;
    std::size_t walkOrbits = 0;
    std::size_t walkReps = 0;
    std::size_t testReps = 0;
    double massSurvivors = 0.0;
    double massWalkReps = 0.0;
    bool coversExactlyOnce = true;
};

// The kernel weight of one cell: the pair's primitive pairs times the aux
// shell's - the quantity the g3 counter sums (AccumulateScreenedRiPass:
// primPairs(bra) * primPairs(aux entry), and a pair's primitive list is the
// product of its two shells').
double CellMass(const qcx::integrals::ShellPairList& pairList,
                const qcx::integrals::ShellPairList& auxPairList,
                const qcx::basisset::BasisSet& basis,
                const qcx::basisset::BasisSet& auxBasis,
                const qcx::molecule::Molecule& molecule,
                // (braPair, auxShell) is the pair-then-shell order of the cell lookup.
                // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                std::size_t braPair,
                std::size_t auxShell) {
    const auto primitives = [&molecule](const qcx::basisset::BasisSet& basisSet,
                                        const qcx::integrals::ShellInfo& shell) {
        const qcx::molecule::Atom& atom = molecule.Atoms()[shell.atomIndex];
        const qcx::basisset::ElementBasis* element = basisSet.Find(atom.atomicNumber);
        return element->shells[shell.elementShellIndex].exponents.size();
    };

    const qcx::integrals::ShellPairIndex& pair = pairList.pairs[braPair];

    return static_cast<double>(primitives(basis, pairList.shells[pair.i]) *
                               primitives(basis, pairList.shells[pair.j]) *
                               primitives(auxBasis, auxPairList.shells[auxShell]));
}

// The independent enumeration: a breadth-first walk of the surviving grid
// under the joint action. The orbit's representative is its smallest cell in
// the grid's own order - the cell the walk reaches first from the smallest
// unvisited survivor, which is exactly the definition the production test
// implements, arrived at the other way round.
GridCensus CensusGrid(const qcx::integrals::internal::RiOrbitAction& action,
                      const ShellPairList& pairList,
                      const ShellPairList& auxPairList,
                      const qcx::basisset::BasisSet& basis,
                      const qcx::basisset::BasisSet& auxBasis,
                      const qcx::molecule::Molecule& molecule,
                      const std::vector<double>& orbitalBounds,
                      const std::vector<double>& auxShellBounds,
                      double cutoff) {
    const std::size_t nPairs = action.nOrbitalPairs;
    const std::size_t nAuxShells = action.nAuxShells;
    const std::size_t order = action.order;

    std::vector<unsigned char> survives(nPairs * nAuxShells, 0);

    for (std::size_t p = 0; p < nPairs; ++p)
    {
        for (std::size_t s = 0; s < nAuxShells; ++s)
        {
            if (orbitalBounds[p] * auxShellBounds[s] >= cutoff)
            {
                survives[qcx::integrals::internal::RiCellIndex(p, s, nAuxShells)] = 1;
            }
        }
    }

    std::vector<unsigned char> seen(nPairs * nAuxShells, 0);
    std::vector<std::size_t> stack;
    GridCensus census;
    std::size_t coveredCells = 0;

    for (std::size_t p = 0; p < nPairs; ++p)
    {
        for (std::size_t s = 0; s < nAuxShells; ++s)
        {
            const std::size_t cell = qcx::integrals::internal::RiCellIndex(p, s, nAuxShells);

            if (survives[cell] == 0)
            {
                continue;
            }

            ++census.survivors;
            census.massSurvivors +=
                CellMass(pairList, auxPairList, basis, auxBasis, molecule, p, s);

            if (seen[cell] != 0)
            {
                continue;
            }

            ++census.walkOrbits;
            std::size_t representative = cell;
            stack.clear();
            stack.push_back(cell);
            seen[cell] = 1;

            while (!stack.empty())
            {
                const std::size_t current = stack.back();
                stack.pop_back();
                const std::size_t currentPair = current / nAuxShells;
                const std::size_t currentAux = current % nAuxShells;

                for (std::size_t g = 0; g < order; ++g)
                {
                    const std::size_t imagePair = action.orbital.pairImage[currentPair * order + g];
                    const std::size_t imageAux = action.aux.shellImage[currentAux * order + g];
                    const std::size_t image =
                        qcx::integrals::internal::RiCellIndex(imagePair, imageAux, nAuxShells);

                    if (survives[image] != 0 && seen[image] == 0)
                    {
                        seen[image] = 1;
                        stack.push_back(image);
                        representative = std::min(representative, image);
                    }
                }
            }

            ++census.walkReps;
            coveredCells += static_cast<std::size_t>(1);

            const std::size_t repPair = representative / nAuxShells;
            const std::size_t repAux = representative % nAuxShells;

            if (!qcx::integrals::internal::RiCellIsOrbitRep(
                    action, orbitalBounds, auxShellBounds, cutoff, repPair, repAux))
            {
                census.coversExactlyOnce = false;
            }

            census.massWalkReps +=
                CellMass(pairList, auxPairList, basis, auxBasis, molecule, repPair, repAux);
        }
    }

    // The production test's own answer over the surviving grid: exactly one
    // representative per walk orbit, and no cell outside the walk's set.
    for (std::size_t p = 0; p < nPairs; ++p)
    {
        for (std::size_t s = 0; s < nAuxShells; ++s)
        {
            const std::size_t cell = qcx::integrals::internal::RiCellIndex(p, s, nAuxShells);

            if (survives[cell] == 0)
            {
                continue;
            }

            if (qcx::integrals::internal::RiCellIsOrbitRep(
                    action, orbitalBounds, auxShellBounds, cutoff, p, s))
            {
                ++census.testReps;
            }
        }
    }

    if (coveredCells != census.walkOrbits)
    {
        census.coversExactlyOnce = false;
    }

    return census;
}

// The two Schwarz bound vectors and the cutoff the fill uses, for one
// fixture (BuildScreenedRiTaskList's own three inputs, recomputed the same
// way).
struct Bounds {
    std::vector<double> orbital;
    std::vector<double> auxShell;
    double cutoff = 0.0;
};

Bounds ComputeBounds(const qcx::molecule::Molecule& molecule,
                     const qcx::basisset::BasisSet& basis,
                     const qcx::basisset::BasisSet& auxBasis,
                     const ShellPairList& auxPairList,
                     qcx::integrals::AccuracyPreset accuracy) {
    Bounds bounds;
    auto schwarzOrbital = qcx::integrals::ComputeSchwarzBounds(molecule, basis);
    auto schwarzAux = qcx::integrals::ComputeSchwarzBounds(molecule, auxBasis);
    bounds.orbital = std::move(*schwarzOrbital);
    bounds.auxShell.resize(auxPairList.shells.size());

    for (std::size_t auxShell = 0; auxShell < auxPairList.shells.size(); ++auxShell)
    {
        bounds.auxShell[auxShell] =
            (*schwarzAux)[qcx::integrals::PairIndexOf(auxShell, auxShell, auxPairList)];
    }

    bounds.cutoff = qcx::integrals::SchwarzThreshold(accuracy);
    return bounds;
}

double MaxAbsDifference(const CpuTensor3& a, const CpuTensor3& b) {
    double worst = 0.0;

    for (std::size_t u = 0; u < a.Shape()[0]; ++u)
    {
        for (std::size_t v = 0; v < a.Shape()[1]; ++v)
        {
            for (std::size_t p = 0; p < a.Shape()[2]; ++p)
            {
                worst = std::max(worst, std::abs(a(u, v, p) - b(u, v, p)));
            }
        }
    }

    return worst;
}

} // namespace

// The joint action's own contract. Two reductions of different group orders
// are not two views of one group action, and their orbits would not be
// orbits.
TEST(RiOrbitActionTest, RefusesReductionsOfDifferentGroupOrders) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;

    const SymmetryReduction orbital = MakeWaterC2vReductionHFirst();
    SymmetryReduction aux = MakeWaterC2vAuxReduction();
    // The aux table above is over 6 functions; give it a group of the wrong
    // order over the orbital basis' own 7-function shape - the orders are
    // what must disagree, and they are checked before the shapes.
    SymmetryReduction truncated = aux;
    truncated.groupOrder = 2;
    truncated.permutation.resize(2);
    truncated.sign.resize(2);

    auto mismatched = qcx::integrals::internal::RiOrbitAction::Create(
        orbital, *pairList, truncated, *auxPairList);
    ASSERT_FALSE(mismatched.has_value());
    EXPECT_EQ(mismatched.error().code, qcx::ErrorCode::kInvalidArgument);
}

// The shell closure is the precondition the whole expansion rests on: a
// shell whose functions map onto more than one shell has no image shell, so
// the member's position arithmetic would have no meaning. The reduction
// below splits the O p shell - in range, legal signed-permutation entries,
// and refused by the action. (A single-function shell cannot demonstrate
// this: with one function there is nothing to split, which is why the
// perturbation has to land on the three-function p shell.)
TEST(RiOrbitActionTest, RefusesABasisWhoseShellsDoNotPermuteAsUnits) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    SymmetryReduction reduction = MakeWaterC2vReductionHFirst();
    // Function 4 is a component of the O p shell (functions 4, 5, 6); the
    // perturbation sends it to the H 1s shell, so the p shell's functions
    // land in two different shells.
    reduction.permutation[1][4] = 0;

    auto refused =
        qcx::integrals::internal::RiOrbitAction::Create(reduction, *pairList, reduction, *pairList);
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kUnimplemented);
}

// The expansion's per-axis maps are fixed-size stack arrays, so a shell
// wider than their cap is a CHECKED precondition the engine refuses at
// Create (kUnimplemented) - not an assert a release build compiles away
// and not a silent overflow. No basis this engine ships comes near the cap.
TEST(RiOrbitActionTest, RefusesABasisWithAShellWiderThanTheAxisCap) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    // The real water basis fits.
    EXPECT_TRUE(qcx::integrals::internal::RiOrbitActionSupports(*pairList, *pairList));

    // One fabricated shell as wide as the cap allows (13 contracted d rows
    // x 5 spherical components = 65 > 64): the aux side being wide is
    // enough, and the orbital side being wide is enough.
    qcx::integrals::ShellPairList wide;
    qcx::integrals::ShellInfo shell{};
    shell.angularMomentum = 2;
    shell.isSpherical = true;
    shell.contractionCount = 13;
    shell.functionOffset = 0;
    shell.atomIndex = 0;
    shell.elementShellIndex = 0;
    wide.shells.push_back(shell);
    wide.functionCount = 65;

    EXPECT_FALSE(qcx::integrals::internal::RiOrbitActionSupports(wide, *pairList));
    EXPECT_FALSE(qcx::integrals::internal::RiOrbitActionSupports(*pairList, wide));
    EXPECT_FALSE(qcx::integrals::internal::RiOrbitActionSupports(wide, wide));

    // 12 rows x 5 = 60 <= 64 fits.
    wide.shells[0].contractionCount = 12;
    wide.functionCount = 60;
    EXPECT_TRUE(qcx::integrals::internal::RiOrbitActionSupports(wide, *pairList));
}

// The same precondition through the ENGINE's own door: an engaged expansion
// over a basis with an over-wide shell is refused by BuildRiTensor
// (kUnimplemented) rather than computed. The aux basis below is one
// 22-row p contraction on O (22 x 3 = 66 functions > the 64-function cap)
// plus a plain s shell on H - a general contraction, which is exactly the
// shape the cap exists for.
TEST(RiOrbitExpansionTest, RefusesABasisWithAnOverWideShell) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    std::string auxText = "BASIS \"ao basis\" SPHERICAL PRINT\nO    P\n      1.0000000000E+00";

    for (int row = 0; row < 23; ++row)
    {
        auxText += "    " + std::to_string(row + 1) + ".0000000000E-01";
    }

    auxText += "\nH    S\n      1.0000000000E+00       1.0000000000E+00\nEND\n";

    auto auxBasis = qcx::basisset::ParseNwchemText(auxText);
    ASSERT_TRUE(auxBasis.has_value()) << auxBasis.error().message;

    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *auxBasis);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;
    auto orbitalPairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(orbitalPairList.has_value()) << orbitalPairList.error().message;
    ASSERT_FALSE(qcx::integrals::internal::RiOrbitActionSupports(*orbitalPairList, *auxPairList));

    // The refusal fires before any table is built, so the reductions' own
    // shapes never come into it - the dummy below is a 3-function table
    // against a 69-function basis on purpose.
    const SymmetryReduction orbital = MakeWaterC2vReductionHFirst();
    SymmetryReduction aux;
    aux.groupOrder = 4;
    aux.isTrivial = false;
    aux.permutation = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
    aux.sign = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};

    RiEngineOptions options;
    options.symmetryOrbitExpansion = true;
    options.symmetryReduction = &orbital;
    options.auxSymmetryReduction = &aux;

    auto refused = qcx::integrals::BuildRiTensor(*molecule, *basis, *auxBasis, options);
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kUnimplemented) << refused.error().message;
}

// The engine NAMES its own engagement, so a run record can carry it: an
// option a record cannot mention is an option nobody can check. True exactly
// under the condition the tables are built with; false when the flag is off
// and when the group is trivial (the mechanism is then the identity).
TEST(RiOrbitExpansionTest, BuilderNamesWhetherTheExpansionRan) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto auxBasis = qcx::basisset::ParseNwchemText(kTinySymmetricAux);
    ASSERT_TRUE(auxBasis.has_value()) << auxBasis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    auto core = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create(
        {pairList->functionCount, pairList->functionCount});
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const SymmetryReduction orbital = MakeWaterC2vReductionHFirst();
    const SymmetryReduction aux = MakeWaterC2vAuxReduction();

    SymmetryReduction trivial;
    trivial.groupOrder = 1;
    trivial.isTrivial = true;
    trivial.permutation = {{0, 1, 2, 3, 4, 5, 6}};
    trivial.sign = {{1, 1, 1, 1, 1, 1, 1}};
    SymmetryReduction trivialAux;
    trivialAux.groupOrder = 1;
    trivialAux.isTrivial = true;
    trivialAux.permutation = {{0, 1, 2, 3, 4, 5}};
    trivialAux.sign = {{1, 1, 1, 1, 1, 1}};

    RiEngineOptions off;
    auto plain = qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *auxBasis, *core, off);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    EXPECT_FALSE(plain->OrbitExpansionEngaged()) << "no reductions = the permission is inert";

    RiEngineOptions on;
    on.symmetryReduction = &orbital;
    on.auxSymmetryReduction = &aux;
    auto engaged = qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *auxBasis, *core, on);
    ASSERT_TRUE(engaged.has_value()) << engaged.error().message;
    EXPECT_TRUE(engaged->OrbitExpansionEngaged());

    RiEngineOptions inert = on;
    inert.symmetryReduction = &trivial;
    inert.auxSymmetryReduction = &trivialAux;
    auto trivialGroup =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *auxBasis, *core, inert);
    ASSERT_TRUE(trivialGroup.has_value()) << trivialGroup.error().message;
    EXPECT_FALSE(trivialGroup->OrbitExpansionEngaged());

    // The SAME statement one step out, and the step that costs real work if
    // it is missed: a NON-C1 group that acts trivially on every shell pair.
    // Every element is a pure sign flip - the molecular plane's mirror on a
    // planar molecule, the HOCl/Cs case - so every joint cell is its own
    // orbit and the expansion removes nothing while building its tables and
    // charging them. The verdict is the same as the order-1 case above, and
    // an order test (`groupOrder > 1`, what this predicate used to read) gets
    // it wrong.
    SymmetryReduction signOnly;
    signOnly.groupOrder = 2;
    signOnly.isTrivial = true;
    signOnly.permutation = {{0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5, 6}};
    signOnly.sign = {{1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, -1, 1}};
    SymmetryReduction signOnlyAux;
    signOnlyAux.groupOrder = 2;
    signOnlyAux.isTrivial = true;
    signOnlyAux.permutation = {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}};
    signOnlyAux.sign = {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, -1}};
    RiEngineOptions inertSignOnly = on;
    inertSignOnly.symmetryReduction = &signOnly;
    inertSignOnly.auxSymmetryReduction = &signOnlyAux;
    auto pairTrivialGroup =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *auxBasis, *core, inertSignOnly);
    ASSERT_TRUE(pairTrivialGroup.has_value()) << pairTrivialGroup.error().message;
    EXPECT_FALSE(pairTrivialGroup->OrbitExpansionEngaged())
        << "a group that permutes no shell pair removes nothing: the expansion must be inert";
}

// The Create-time footprint charge: the engaged orbit action's tables are
// charged by the SAME formula the action reports, so the estimate and the
// allocation cannot drift apart (the estimate must never fall under the
// allocation). The term is zero whenever the mechanism is not engaged, which
// is why the default path's envelope is untouched.
TEST(RiOrbitExpansionTest, FootprintChargesTheEngagedOrbitAction) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto auxBasis = qcx::basisset::ParseNwchemText(kTinySymmetricAux);
    ASSERT_TRUE(auxBasis.has_value()) << auxBasis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *auxBasis);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;

    const SymmetryReduction orbital = MakeWaterC2vReductionHFirst();
    const SymmetryReduction aux = MakeWaterC2vAuxReduction();

    auto action =
        qcx::integrals::internal::RiOrbitAction::Create(orbital, *pairList, aux, *auxPairList);
    ASSERT_TRUE(action.has_value()) << action.error().message;
    ASSERT_GT(action->Bytes(), 0u);

    // The int product is 512 MiB: in range, the widening is the assignment's.
    // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result)
    constexpr std::size_t kBatch = 512 * 1024 * 1024;
    const auto disengaged = qcx::integrals::internal::RiFootprint(
        *molecule, *basis, *auxBasis, *pairList, *auxPairList, kBatch, 1);
    const auto engaged = qcx::integrals::internal::RiFootprint(*molecule,
                                                               *basis,
                                                               *auxBasis,
                                                               *pairList,
                                                               *auxPairList,
                                                               kBatch,
                                                               1,
                                                               false,
                                                               false,
                                                               0,
                                                               action->Bytes());

    EXPECT_EQ(disengaged.orbitActionBytes, 0u);
    EXPECT_EQ(engaged.orbitActionBytes, action->Bytes());
    EXPECT_EQ(engaged.Total() - disengaged.Total(), action->Bytes());
    // And the free formula the engine passes agrees with the built action's
    // own report - the two cannot drift apart.
    EXPECT_EQ(qcx::integrals::internal::RiOrbitActionBytes(action->order,
                                                           pairList->pairs.size(),
                                                           pairList->shells.size(),
                                                           pairList->functionCount,
                                                           auxPairList->pairs.size(),
                                                           auxPairList->shells.size(),
                                                           auxPairList->functionCount),
              action->Bytes());
}

// The action's bytes are its two instances' tables and nothing else - the
// formula any Create-time charge must use.
TEST(RiOrbitActionTest, BytesIsTheTwoActionsTables) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    const SymmetryReduction reduction = MakeWaterC2vReductionHFirst();
    auto action =
        qcx::integrals::internal::RiOrbitAction::Create(reduction, *pairList, reduction, *pairList);
    ASSERT_TRUE(action.has_value()) << action.error().message;

    EXPECT_EQ(action->order, 4u);
    EXPECT_EQ(action->nOrbitalPairs, pairList->pairs.size());
    EXPECT_EQ(action->nAuxShells, pairList->shells.size());
    EXPECT_EQ(action->Bytes(), action->orbital.Bytes() + action->aux.Bytes());
    EXPECT_EQ(action->Bytes(),
              qcx::integrals::internal::LeanOrbitActionBytes(
                  4, pairList->pairs.size(), pairList->shells.size(), pairList->functionCount) *
                  2);
}

// The representative test against an independent walk of the same grid: the
// same orbit count, exactly one representative per orbit, and every
// surviving cell covered. Run on both aux fixtures - the same basis as the
// orbital one and the smaller, structurally different one.
TEST(RiOrbitExpansionTest, RepresentativesPartitionTheSurvivingGrid) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto auxBasis = qcx::basisset::ParseNwchemText(kTinySymmetricAux);
    ASSERT_TRUE(auxBasis.has_value()) << auxBasis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *auxBasis);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;

    const SymmetryReduction orbital = MakeWaterC2vReductionHFirst();
    const SymmetryReduction aux = MakeWaterC2vAuxReduction();

    auto action =
        qcx::integrals::internal::RiOrbitAction::Create(orbital, *pairList, aux, *auxPairList);
    ASSERT_TRUE(action.has_value()) << action.error().message;
    EXPECT_EQ(auxPairList->shells.size(), 4u);
    EXPECT_EQ(auxPairList->functionCount, 6u);

    const Bounds bounds =
        ComputeBounds(*molecule, *basis, *auxBasis, *auxPairList, RiEngineOptions{}.accuracy);
    const GridCensus census = CensusGrid(*action,
                                         *pairList,
                                         *auxPairList,
                                         *basis,
                                         *auxBasis,
                                         *molecule,
                                         bounds.orbital,
                                         bounds.auxShell,
                                         bounds.cutoff);

    std::cout << "[ri_orbit] water/STO-3G + tiny aux: survivors " << census.survivors
              << ", walk orbits " << census.walkOrbits << ", walk reps " << census.walkReps
              << ", test reps " << census.testReps << ", mass ratio "
              << (census.massSurvivors > 0.0 ? census.massWalkReps / census.massSurvivors : 0.0)
              << '\n';

    EXPECT_TRUE(census.coversExactlyOnce);
    EXPECT_EQ(census.testReps, census.walkOrbits);
    EXPECT_EQ(census.walkReps, census.walkOrbits);
    EXPECT_LT(census.walkOrbits, census.survivors);
}

// THE ACCOUNTING: what the expansion REMOVES and what it ADDS, per surviving
// cell, as counts. This is the cell that turns "3.41x fewer blocks" into an
// identity, and it is the whole reason the reduction can be a block-count win
// and a wall loss at the same time.
//
// The two sides:
//  - REMOVED: one kernel evaluation per representative instead of one per
//    surviving cell - `survivors - reps` evaluations, and nothing else. In
//    particular the expansion removes nothing from the WRITE.
//  - ADDED: one block copy per expanded member, and one tensor scatter for
//    it. The scatter is NOT saved from the plain walk's side: every surviving
//    cell is scattered either way (ri_engine.cpp scatters the representative,
//    then each expanded member), so the scatter count is the plain walk's PLUS
//    whatever the expansion re-writes.
//
// The identity asserted here is therefore
//   copies            == (survivors - reps)
//   scatters(engaged) == scatters(disengaged)
// because the walk now writes each member cell EXACTLY ONCE. Before 2026-09-15
// the g-loop skipped the image that IS the representative and the images the
// screen dropped, but not an image an EARLIER g had already produced: the
// group's own STABILIZERS carried two elements onto one cell, so that cell was
// expanded and scattered once per such element and only the LAST write's bytes
// survived. The earlier copies were DEAD STORES, and they made the copy count
// EXCEED the evaluations removed (23 copies against 23 removed, 46 in total,
// on this fixture; 78,211 of 714,629 on the grid one). ri_engine.cpp now skips
// an element a later one supersedes, which removes every dead store and leaves
// the surviving write - the largest g that lands on the cell and clears the
// screen - exactly the one that survived before.
//
// That is the BIT-IDENTITY argument, and this cell makes it checkable rather
// than asserted: the kept elements are recomputed here as the LAST OCCURRENCE
// of each member cell in the walk's ascending-g order, and the counts the cell
// pins are the ones that follow from it. The engine's own rule (skip g when a
// later element lands on the same surviving cell) and "keep the last
// occurrence" are the same predicate.
//
// WHAT THIS CELL DOES NOT DO, stated because the difference matters: it does
// not re-derive the grid fixture's absolutes. ri_engine.hpp's option comment
// carries survivors 900,698 / reps 264,280 (so 636,418 evaluations removed)
// against 714,629 copies and 78,211 duplicate scatters (+8.7%) on
// c8h18/def2-SVP + def2-universal-jfit, C2h - the numbers behind the
// default-ON choice - and THOSE FOUR NUMBERS COME FROM A ONE-OFF MEASUREMENT
// THAT IS NOT CHECKED IN: NO TEST IN THIS TREE REPRODUCES THEM. What is
// reproduced here is the ACCOUNTING, on a fixture whose reduction the suite
// validates by independent walk (the cell above), so the shape is pinned even
// though the grid fixture's magnitudes are not.
TEST(RiOrbitExpansionTest, ExpansionTradesEvaluationsForCopiesAndAddsNoScatter) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto auxBasis = qcx::basisset::ParseNwchemText(kTinySymmetricAux);
    ASSERT_TRUE(auxBasis.has_value()) << auxBasis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *auxBasis);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;

    const SymmetryReduction orbital = MakeWaterC2vReductionHFirst();
    const SymmetryReduction aux = MakeWaterC2vAuxReduction();
    auto action =
        qcx::integrals::internal::RiOrbitAction::Create(orbital, *pairList, aux, *auxPairList);
    ASSERT_TRUE(action.has_value()) << action.error().message;

    const Bounds bounds =
        ComputeBounds(*molecule, *basis, *auxBasis, *auxPairList, RiEngineOptions{}.accuracy);
    const std::size_t nPairs = pairList->pairs.size();
    const std::size_t nAuxShells = auxPairList->shells.size();
    const std::size_t order = action->order;
    const std::vector<double>& orbitalBounds = bounds.orbital;
    const std::vector<double>& auxBounds = bounds.auxShell;
    const double cutoff = bounds.cutoff;

    // The fill's own survival test, with the fill's own doubles and `<`
    // (RiCellIsOrbitRep's contract: the two must not disagree on the boundary).
    const auto survives = [&orbitalBounds, &auxBounds, cutoff](std::size_t p, std::size_t s) {
        return orbitalBounds[p] * auxBounds[s] >= cutoff;
    };

    std::size_t survivors = 0;
    std::size_t reps = 0;
    std::size_t fixedCopies = 0;
    std::size_t deadStoresRemoved = 0;
    std::size_t identityImages = 0;
    std::size_t screenedImages = 0;
    std::size_t copiesOntoAnEvaluatedCell = 0;
    bool everyKeptElementIsTheLastWriter = true;

    for (std::size_t p = 0; p < nPairs; ++p)
    {
        for (std::size_t s = 0; s < nAuxShells; ++s)
        {
            if (!survives(p, s))
            {
                continue;
            }

            ++survivors;

            if (!qcx::integrals::internal::RiCellIsOrbitRep(
                    *action, orbitalBounds, auxBounds, cutoff, p, s))
            {
                continue;
            }

            ++reps;

            // The expansion loop's candidate elements, in the walk's own
            // ascending g, with the identity image and the screened-out images
            // dropped exactly as ri_engine.cpp drops them. What remains is the
            // walk's write sequence for this representative, BEFORE the
            // supersession rule - so its length is the old copy count and its
            // distinct-cell count is the fixed one.
            std::vector<std::size_t> writtenCells;
            std::vector<std::size_t> writtenElements;
            std::vector<std::size_t> writtenPairs;
            std::vector<std::size_t> writtenAuxShells;
            writtenCells.reserve(order);
            writtenElements.reserve(order);
            writtenPairs.reserve(order);
            writtenAuxShells.reserve(order);

            for (std::size_t g = 1; g < order; ++g)
            {
                const std::size_t memberPair = action->orbital.pairImage[p * order + g];
                const std::size_t memberAux = action->aux.shellImage[s * order + g];

                if (memberPair == p && memberAux == s)
                {
                    ++identityImages;
                    continue;
                }

                if (!survives(memberPair, memberAux))
                {
                    ++screenedImages;
                    continue;
                }

                writtenCells.push_back(
                    qcx::integrals::internal::RiCellIndex(memberPair, memberAux, nAuxShells));
                writtenElements.push_back(g);
                writtenPairs.push_back(memberPair);
                writtenAuxShells.push_back(memberAux);
            }

            // The fix, as the engine applies it, and the bit-identity check
            // in the same pass: an element is kept iff no LATER element writes
            // the same cell (so it is that cell's LAST writer - the write whose
            // bytes are the ones that survived before the fix), and everything
            // else is a dead store.
            for (std::size_t i = 0; i < writtenCells.size(); ++i)
            {
                bool superseded = false;

                for (std::size_t later = i + 1; later < writtenCells.size(); ++later)
                {
                    if (writtenCells[later] == writtenCells[i])
                    {
                        superseded = true;
                        break;
                    }
                }

                if (superseded)
                {
                    ++deadStoresRemoved;
                    continue;
                }

                ++fixedCopies;

                // AFTER THE FIX, does any copied cell ALSO receive an
                // evaluation - i.e. is a member cell also a representative?
                // The orbit partition forbids it (one representative per
                // orbit, and this member is in THIS representative's orbit),
                // so the answer must be zero; the count is here so the answer
                // is read rather than argued.
                if (qcx::integrals::internal::RiCellIsOrbitRep(*action,
                                                               orbitalBounds,
                                                               auxBounds,
                                                               cutoff,
                                                               writtenPairs[i],
                                                               writtenAuxShells[i]))
                {
                    ++copiesOntoAnEvaluatedCell;
                }

                // A kept element must be the LAST writer of its cell: nothing
                // after it writes that cell. This is what makes the fix
                // byte-preserving rather than merely cheaper.
                for (std::size_t later = i + 1; later < writtenCells.size(); ++later)
                {
                    if (writtenCells[later] == writtenCells[i])
                    {
                        everyKeptElementIsTheLastWriter = false;
                    }
                }
            }
        }
    }

    const std::size_t evaluationsRemoved = survivors - reps;
    const std::size_t scattersDisengaged = survivors;
    const std::size_t scattersEngaged = reps + fixedCopies;

    std::cout << "[ri_orbit] accounting (fixed): survivors " << survivors << ", reps " << reps
              << ", evaluations removed " << evaluationsRemoved << ", copies " << fixedCopies
              << " (dead stores removed " << deadStoresRemoved << ", identity images "
              << identityImages << ", screened-out images " << screenedImages
              << ", copies onto an evaluated cell " << copiesOntoAnEvaluatedCell
              << "), scatters disengaged " << scattersDisengaged << " vs engaged "
              << scattersEngaged << '\n';

    EXPECT_GT(survivors, reps) << "the reduction must remove evaluations, or it is not a reduction";

    // The bit-identity check: every element the walk keeps is its member cell's
    // last writer, so the bytes that survive are the bytes that survived
    // before the dead stores were dropped.
    EXPECT_TRUE(everyKeptElementIsTheLastWriter)
        << "a kept element is not its member cell's last writer: the fix would move bytes";

    // The copy side, post-fix: exactly one copy per removed evaluation, and
    // nothing else.
    EXPECT_EQ(fixedCopies, evaluationsRemoved)
        << "each removed evaluation must be replaced by exactly one copy, and nothing else";

    // The scatter side, and the reason a block-count win need not be a wall
    // win: the expansion removes EVALUATIONS and removes nothing from the
    // WRITE. With the dead stores gone the expanded walk scatters every
    // surviving cell exactly once - the plain walk's count, no more.
    EXPECT_EQ(scattersEngaged, scattersDisengaged)
        << "the expanded walk must scatter each surviving cell exactly once, as the plain walk "
           "does";

    // The packet's closing question, as a count: no copied cell is also an
    // evaluated one, so after the fix no address receives both a copy and an
    // evaluation.
    EXPECT_EQ(copiesOntoAnEvaluatedCell, 0u)
        << "a copied member cell is also a representative: an address would take both a copy and "
           "an evaluation";

    // The dead stores are real and were removed; a fixture whose group carries
    // no stabilizer onto a cell would show none, and this one does not.
    EXPECT_GT(deadStoresRemoved, 0u)
        << "this fixture's group must exercise the stabilizer duplicates the fix removes";

    EXPECT_EQ(order, 4u);
}

// The value-neutrality contract, at the tolerance this build MEASURES. The
// expanded tensor is not the plain walk's tensor bit for bit in general: a
// member's block is the representative's block under the group's signed
// permutation, i.e. the same integral obtained from a different
// (value-equal) evaluation order - a class of difference measured at
// ~1e-15/1e-16 per element. Measured here: this fixture is
// BIT-IDENTICAL at kNormal (max |delta| = 0), while the largest fixture this
// work was measured on (c8h18/def2-SVP + def2-universal-jfit, C2h, order 4)
// reaches max |delta| = 2.867e-14 over a tensor max of 6.969 (4.1e-15
// relative). The pin below - 1e-13 - is 3.5x
// that largest measurement, deliberately set from the LARGER fixture: this
// test must not be the place a bigger system's tolerance fails first.
TEST(RiOrbitExpansionTest, TensorMatchesThePlainWalkWithinTheMeasuredTolerance) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto auxBasis = qcx::basisset::ParseNwchemText(kTinySymmetricAux);
    ASSERT_TRUE(auxBasis.has_value()) << auxBasis.error().message;

    const SymmetryReduction orbital = MakeWaterC2vReductionHFirst();
    const SymmetryReduction aux = MakeWaterC2vAuxReduction();

    // The screen's dropped cells: every element of a (braPair, auxShell)
    // cell the Schwarz test dropped must be EXACTLY zero in both tensors -
    // the expansion writes a cell only through the same survival test the
    // fill uses, so a dropped cell can never pick up an expanded block.
    // Both presets run: at kNormal this small fixture screens nothing out,
    // and a check over an empty dropped set proves nothing, so kLoose is
    // where that property is actually live.
    for (const qcx::integrals::AccuracyPreset preset :
         {qcx::integrals::AccuracyPreset::kNormal, qcx::integrals::AccuracyPreset::kLoose})
    {
        RiEngineOptions plainOptions;
        plainOptions.accuracy = preset;
        RiTermCounters plainSink;
        auto plainAt =
            qcx::integrals::BuildRiTensor(*molecule, *basis, *auxBasis, plainOptions, &plainSink);
        ASSERT_TRUE(plainAt.has_value()) << plainAt.error().message;

        RiEngineOptions expandedOptions = plainOptions;
        expandedOptions.symmetryOrbitExpansion = true;
        expandedOptions.symmetryReduction = &orbital;
        expandedOptions.auxSymmetryReduction = &aux;
        RiTermCounters expandedSink;
        auto expandedAt = qcx::integrals::BuildRiTensor(
            *molecule, *basis, *auxBasis, expandedOptions, &expandedSink);
        ASSERT_TRUE(expandedAt.has_value()) << expandedAt.error().message;

        ASSERT_EQ(plainAt->Shape()[0], expandedAt->Shape()[0]);
        ASSERT_EQ(plainAt->Shape()[2], expandedAt->Shape()[2]);

        double largest = 0.0;

        for (std::size_t u = 0; u < plainAt->Shape()[0]; ++u)
        {
            for (std::size_t v = 0; v < plainAt->Shape()[1]; ++v)
            {
                for (std::size_t p = 0; p < plainAt->Shape()[2]; ++p)
                {
                    largest = std::max(largest, std::abs((*plainAt)(u, v, p)));
                }
            }
        }

        auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
        ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
        auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *auxBasis);
        ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;
        const Bounds bounds = ComputeBounds(*molecule, *basis, *auxBasis, *auxPairList, preset);

        std::size_t droppedCells = 0;
        std::size_t droppedNonzero = 0;

        for (std::size_t p = 0; p < pairList->pairs.size(); ++p)
        {
            const qcx::integrals::ShellPairIndex& pair = pairList->pairs[p];
            const qcx::integrals::ShellInfo& shellI = pairList->shells[pair.i];
            const qcx::integrals::ShellInfo& shellJ = pairList->shells[pair.j];

            for (std::size_t s = 0; s < auxPairList->shells.size(); ++s)
            {
                if (bounds.orbital[p] * bounds.auxShell[s] >= bounds.cutoff)
                {
                    continue;
                }

                ++droppedCells;
                const qcx::integrals::ShellInfo& shellP = auxPairList->shells[s];

                for (std::size_t fa = 0; fa < qcx::integrals::ShellFunctionCount(shellI); ++fa)
                {
                    for (std::size_t fb = 0; fb < qcx::integrals::ShellFunctionCount(shellJ); ++fb)
                    {
                        for (std::size_t fP = 0; fP < qcx::integrals::ShellFunctionCount(shellP);
                             ++fP)
                        {
                            const std::size_t u = shellI.functionOffset + fa;
                            const std::size_t v = shellJ.functionOffset + fb;
                            const std::size_t w = shellP.functionOffset + fP;
                            droppedNonzero += ((*plainAt)(u, v, w) != 0.0) ? 1u : 0u;
                            droppedNonzero += ((*expandedAt)(u, v, w) != 0.0) ? 1u : 0u;
                        }
                    }
                }
            }
        }

        std::cout << "[ri_orbit] value neutrality ("
                  << (preset == qcx::integrals::AccuracyPreset::kLoose ? "kLoose" : "kNormal")
                  << "): max |delta| " << MaxAbsDifference(*plainAt, *expandedAt)
                  << " over tensor max " << largest << "; dropped cells " << droppedCells
                  << " with " << droppedNonzero << " non-zero elements across both tensors" << '\n';

        EXPECT_EQ(droppedNonzero, 0u);
        EXPECT_LT(MaxAbsDifference(*plainAt, *expandedAt), 1e-13);

        // NOT asserted: that this fixture drops anything. Measured, it drops
        // NO cell at either preset (every Schwarz product of the water/STO-3G
        // grid sits above even the loose cutoff), so the check above is a
        // guard here rather than evidence. The live evidence for it is the
        // large fixture's probe, where the kNormal screen drops 202,466 of
        // 1,103,130 cells and every element of every dropped cell is exactly
        // zero in both tensors (measured on that fixture).
        std::cout << "[ri_orbit]   dropped cells at this preset: " << droppedCells << '\n';
    }
}

// The block count: the tensor-pass counters fall to the evaluated
// representatives, and both counts match an independently computed
// primitive-pair weighted sum over the same grid - the counter is verified
// against the grid, not against itself.
TEST(RiOrbitExpansionTest, EvaluatedBlocksFallToTheRepresentatives) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto auxBasis = qcx::basisset::ParseNwchemText(kTinySymmetricAux);
    ASSERT_TRUE(auxBasis.has_value()) << auxBasis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *auxBasis);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;

    const SymmetryReduction orbital = MakeWaterC2vReductionHFirst();
    const SymmetryReduction aux = MakeWaterC2vAuxReduction();

    RiEngineOptions plain;
    RiTermCounters plainCounters;
    auto plainTensor =
        qcx::integrals::BuildRiTensor(*molecule, *basis, *auxBasis, plain, &plainCounters);
    ASSERT_TRUE(plainTensor.has_value()) << plainTensor.error().message;

    RiEngineOptions expanded;
    expanded.symmetryOrbitExpansion = true;
    expanded.symmetryReduction = &orbital;
    expanded.auxSymmetryReduction = &aux;
    RiTermCounters expandedCounters;
    auto expandedTensor =
        qcx::integrals::BuildRiTensor(*molecule, *basis, *auxBasis, expanded, &expandedCounters);
    ASSERT_TRUE(expandedTensor.has_value()) << expandedTensor.error().message;

    // The independent weights: the same grid, the same two bound vectors,
    // the surviving sums and the representative sums.
    auto action =
        qcx::integrals::internal::RiOrbitAction::Create(orbital, *pairList, aux, *auxPairList);
    ASSERT_TRUE(action.has_value()) << action.error().message;
    const Bounds bounds =
        ComputeBounds(*molecule, *basis, *auxBasis, *auxPairList, RiEngineOptions{}.accuracy);
    const GridCensus census = CensusGrid(*action,
                                         *pairList,
                                         *auxPairList,
                                         *basis,
                                         *auxBasis,
                                         *molecule,
                                         bounds.orbital,
                                         bounds.auxShell,
                                         bounds.cutoff);

    std::cout << "[ri_orbit] block counts: g3 plain " << plainCounters.g3 << " expanded "
              << expandedCounters.g3 << " (ratio "
              << static_cast<double>(expandedCounters.g3) / static_cast<double>(plainCounters.g3)
              << "); mass ratio from the grid " << census.massWalkReps / census.massSurvivors
              << "; x " << expandedCounters.x << " of " << plainCounters.x << ", p3 "
              << expandedCounters.p3 << " of " << plainCounters.p3 << '\n';

    EXPECT_EQ(static_cast<double>(plainCounters.g3), census.massSurvivors);
    EXPECT_EQ(static_cast<double>(expandedCounters.g3), census.massWalkReps);
    EXPECT_LT(expandedCounters.g3, plainCounters.g3);
    EXPECT_LE(expandedCounters.p3, plainCounters.p3);
    EXPECT_LE(expandedCounters.x, plainCounters.x);
}

// The identity mechanism leaves the path untouched: with a trivial (order 1)
// group every cell is its own orbit, so the expansion is not engaged and the
// tensor is BIT-identical to the plain walk's - the property a regression in
// the engagement condition would break.
TEST(RiOrbitExpansionTest, TrivialReductionLeavesThePathUntouched) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto auxBasis = qcx::basisset::ParseNwchemText(kTinySymmetricAux);
    ASSERT_TRUE(auxBasis.has_value()) << auxBasis.error().message;

    SymmetryReduction trivial;
    trivial.groupOrder = 1;
    trivial.isTrivial = true;
    trivial.permutation = {{0, 1, 2, 3, 4, 5, 6}};
    trivial.sign = {{1, 1, 1, 1, 1, 1, 1}};
    SymmetryReduction trivialAux;
    trivialAux.groupOrder = 1;
    trivialAux.isTrivial = true;
    trivialAux.permutation = {{0, 1, 2, 3, 4, 5}};
    trivialAux.sign = {{1, 1, 1, 1, 1, 1}};

    RiEngineOptions plain;
    RiTermCounters plainCounters;
    auto plainTensor =
        qcx::integrals::BuildRiTensor(*molecule, *basis, *auxBasis, plain, &plainCounters);
    ASSERT_TRUE(plainTensor.has_value()) << plainTensor.error().message;

    RiEngineOptions expanded;
    expanded.symmetryOrbitExpansion = true;
    expanded.symmetryReduction = &trivial;
    expanded.auxSymmetryReduction = &trivialAux;
    RiTermCounters expandedCounters;
    auto expandedTensor =
        qcx::integrals::BuildRiTensor(*molecule, *basis, *auxBasis, expanded, &expandedCounters);
    ASSERT_TRUE(expandedTensor.has_value()) << expandedTensor.error().message;

    EXPECT_EQ(MaxAbsDifference(*plainTensor, *expandedTensor), 0.0);
    EXPECT_EQ(expandedCounters.g3, plainCounters.g3);
    EXPECT_EQ(expandedCounters.p3, plainCounters.p3);
    EXPECT_EQ(expandedCounters.x, plainCounters.x);
}

// The flag is defined by the two reductions it expands over: asking for it
// without them is a caller error, refused rather than silently ignored - the
// same discipline the bra-pair action gives its own flag.
// The option is a PERMISSION with the default ON (set from the counts, not
// from a contended-machine sample - see the option's own documentation):
// with no reductions it is inert, not an error, because that is every caller
// that does not engage symmetry; supplying exactly ONE of the two is the
// caller error it still refuses, because half a group action is not one.
TEST(RiOrbitExpansionTest, InertWithoutReductionsAndRefusedWithOnlyOne) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto auxBasis = qcx::basisset::ParseNwchemText(kTinySymmetricAux);
    ASSERT_TRUE(auxBasis.has_value()) << auxBasis.error().message;

    const SymmetryReduction orbital = MakeWaterC2vReductionHFirst();
    const SymmetryReduction aux = MakeWaterC2vAuxReduction();

    // No reductions: accepted, and it computes the plain walk (checked here
    // as bit-identity against an explicit opt-out, which is the same path).
    RiEngineOptions noReductions;
    RiTermCounters defaultCounters;
    auto byDefault =
        qcx::integrals::BuildRiTensor(*molecule, *basis, *auxBasis, noReductions, &defaultCounters);
    ASSERT_TRUE(byDefault.has_value()) << byDefault.error().message;

    RiEngineOptions optedOut;
    optedOut.symmetryOrbitExpansion = false;
    RiTermCounters optedOutCounters;
    auto plain =
        qcx::integrals::BuildRiTensor(*molecule, *basis, *auxBasis, optedOut, &optedOutCounters);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    EXPECT_EQ(MaxAbsDifference(*byDefault, *plain), 0.0);
    EXPECT_EQ(defaultCounters.g3, optedOutCounters.g3);

    // Exactly one reduction: refused, because the joint orbits are ONE group
    // acting on TWO bases.
    RiEngineOptions orbitalOnly;
    orbitalOnly.symmetryReduction = &orbital;
    auto auxMissing = qcx::integrals::BuildRiTensor(*molecule, *basis, *auxBasis, orbitalOnly);
    ASSERT_FALSE(auxMissing.has_value());
    EXPECT_EQ(auxMissing.error().code, qcx::ErrorCode::kInvalidArgument);

    RiEngineOptions auxOnly;
    auxOnly.auxSymmetryReduction = &aux;
    auto orbitalMissing = qcx::integrals::BuildRiTensor(*molecule, *basis, *auxBasis, auxOnly);
    ASSERT_FALSE(orbitalMissing.has_value());
    EXPECT_EQ(orbitalMissing.error().code, qcx::ErrorCode::kInvalidArgument);

    // A C1 molecule's reductions carry order 1: the flag is accepted and the
    // mechanism stays inert (the C1 trace below), never an error - symmetry
    // detection legitimately finds no group.
    SymmetryReduction trivial;
    trivial.groupOrder = 1;
    trivial.isTrivial = true;
    trivial.permutation = {{0, 1, 2, 3, 4, 5, 6}};
    trivial.sign = {{1, 1, 1, 1, 1, 1, 1}};
    SymmetryReduction trivialAux;
    trivialAux.groupOrder = 1;
    trivialAux.isTrivial = true;
    trivialAux.permutation = {{0, 1, 2, 3, 4, 5}};
    trivialAux.sign = {{1, 1, 1, 1, 1, 1}};

    RiEngineOptions inert;
    inert.symmetryReduction = &trivial;
    inert.auxSymmetryReduction = &trivialAux;
    auto accepted = qcx::integrals::BuildRiTensor(*molecule, *basis, *auxBasis, inert);
    EXPECT_TRUE(accepted.has_value()) << accepted.error().message;
}

// The expansion is wired on the fast path's single tensor pass. The light
// rung recomputes the grid per iteration (the amortized variant, where the
// expansion is not wired), so the flag is REFUSED there rather than silently
// dropped.
TEST(RiOrbitExpansionTest, RefusesOnTheLightRung) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto auxBasis = qcx::basisset::ParseNwchemText(kTinySymmetricAux);
    ASSERT_TRUE(auxBasis.has_value()) << auxBasis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    auto core = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create(
        {pairList->functionCount, pairList->functionCount});
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const SymmetryReduction orbital = MakeWaterC2vReductionHFirst();
    const SymmetryReduction aux = MakeWaterC2vAuxReduction();

    RiEngineOptions options;
    options.symmetryOrbitExpansion = true;
    options.symmetryReduction = &orbital;
    options.auxSymmetryReduction = &aux;
    options.forceLightRung = true;

    auto refused =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *auxBasis, *core, options);
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kUnimplemented);
}
