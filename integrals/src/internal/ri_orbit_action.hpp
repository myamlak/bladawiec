#pragma once

/// \file
/// RI-J's JOINT TASK-GRID ORBIT ACTION (the joint-grid expansion the owner
/// ruled on 2026-09-13: "the expansion is worth doing, skip reading not"). The
/// 3c work RI-J does is a grid of (braPair,
/// auxShell) cells; the group acts on BOTH indices, so a cell's orbit is
/// joint and the walk can evaluate ONE representative per orbit and expand
/// the rest of the orbit from that representative's block.
///
/// THE PATTERN IS THE LANDED ONE, not a new one. The petite-list walk landed
/// exactly this shape for the 4-index walk
/// (lean_orbit_action.hpp: the per-index action tables, the walk-time
/// "is this cell its orbit's representative?" test, the contraction-time
/// block expansion) and measured 2,379,972 ERI blocks over 8,811,481
/// screened cells on c8h18/def2-SVP. What a 3c grid adds is a SECOND index
/// space: the auxiliary basis carries its own shells, its own function
/// order and its own signed permutation, so `LeanOrbitAction` - which takes
/// ONE reduction and ONE pair list - is instantiated TWICE here, once per
/// basis, and this header is the joint layer over the two instances. A
/// reduction over the AUX basis is a second input the caller supplies; it
/// cannot be derived from the orbital reduction, which exposes the
/// function-level signed permutation and no atom map
/// (symmetry_reduction.hpp).
///
/// WHAT IT BUYS, AND WHAT IT DOES NOT. The measured ceiling on the grid's
/// own fixture (c8h18/def2-SVP + def2-universal-jfit, C2h, order 4): the
/// screened task grid is 900,698 cells (5,253 orbital pairs x 210 aux
/// shells = 1,103,130 before the Schwarz screen) and carries 264,280
/// joint orbits = 0.29342, i.e. **3.41x fewer ERI blocks** (mass-weighted
/// 3.16x). That is a BLOCK-COUNT ratio and never a wall ratio - the rule
/// for its sibling number, and it binds here too. The blocks it removes are
/// evaluated ONCE, at Create, by the fast path's single tensor pass
/// (ri_engine.hpp RiTermCounters, the tensor-pass occurrence); the measured
/// wall of that one-time build was 2,210.87 ms on one ungated sample, so a
/// 3.41x cut of its block count bounds the RUN-LEVEL saving at ~1.5 seconds,
/// ONCE - not per iteration, and not per call. A reader who lifts 3.41x out
/// as an efficiency claim is quoting the wrong quantity.
///
/// WHAT IT COSTS AND WHAT IT CHANGES. The tables are O(order x (nPairs +
/// nShells + nFunctions)) bytes per basis (the two actions' own Bytes()
/// terms), built at Create from reductions the caller already has to build.
/// The expansion is NOT bit-identical to the plain walk: the member's block
/// is the representative's block under the group's signed permutation, so
/// the arithmetic that produced it is a different (value-equal) evaluation
/// order than evaluating that cell's own integrals - the same class of
/// difference the petite-list landing measured at ~1e-15/1e-16 per element. The
/// tolerance this build achieves is measured and pinned in
/// ri_orbit_test.cpp; no identity is claimed beyond what that test holds.
///
/// WHAT THE ACTION REQUIRES of the two reductions (validated at Create,
/// never assumed): the shared signed-permutation shape and the SHELL
/// CLOSURE of both bases - a shell whose functions map onto more than one
/// shell has no image shell, so the expansion's position arithmetic would
/// have no meaning. `LeanOrbitAction::Create` refuses both, and this Create
/// refuses the one joint failure left over: two reductions whose group
/// orders differ are not two instances of one group action, and their
/// orbits would not be orbits.

#include "lean_orbit_action.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <vector>

namespace qcx::integrals::internal {

/// The expansion's per-axis map width: one shell's function count, the
/// bound of the fixed-size arrays `RiExpandOrbitMemberBlock` maps its three
/// axes through. Every s/p/d/f/g shell of the bases this engine is run on
/// sits far below it (a spherical g shell is 9, an uncontracted Cartesian
/// h shell 28); a GENERAL contraction multiplies its row count in, so the
/// bound is not a theorem about any basis a caller might hand in - which
/// is why it is a CHECKED precondition (RiOrbitActionSupports, refused by
/// the engine at Create) rather than an assert that a release build
/// compiles away.
inline constexpr std::size_t kRiOrbitAxisCap = 64;

/// Whether both bases' shells fit the expansion's per-axis map width - the
/// precondition `RiExpandOrbitMemberBlock` runs under. The engine checks
/// this where the mechanism is engaged and refuses the expansion
/// (kUnimplemented) when a shell is wider, rather than letting a fixed-size
/// map overflow in a release build.
/// \param pairList The orbital pair list.
/// \param auxPairList The aux pair list.
/// \returns True when every shell of both bases fits kRiOrbitAxisCap.
inline bool RiOrbitActionSupports(const ShellPairList& pairList,
                                  const ShellPairList& auxPairList) noexcept {
    for (const ShellInfo& shell : pairList.shells)
    {
        if (ShellFunctionCount(shell) > kRiOrbitAxisCap)
        {
            return false;
        }
    }

    for (const ShellInfo& shell : auxPairList.shells)
    {
        if (ShellFunctionCount(shell) > kRiOrbitAxisCap)
        {
            return false;
        }
    }

    return true;
}

/// The retained byte size of one joint task-grid orbit action over two bases:
/// the two instances' tables, each by LeanOrbitActionBytes. RiOrbitAction::Bytes
/// reports the same number, so a Create-time charge and the tables cannot
/// drift apart.
///
/// The charge itself is NOT wired yet, and this comment says so rather than
/// implying it: the RI-J envelope (internal/footprint.hpp RiFootprint) carries
/// no orbit term, so an engaged expansion under a workspace budget
/// under-charges by these bytes (order x (pairs + shells + functions) x 4 B per
/// basis - tens of KB against the tensor's n^2 x nAux x 8 B, so the rung
/// decision cannot move, but the never-under discipline is not met). The
/// option off - the default - would add zero, so the default path's envelope is
/// unchanged either way. Adding the term is a follow-on for whoever holds
/// footprint.hpp.
/// \param order The group order (0 = no action, no charge).
/// \param orbitalPairs The canonical orbital pair count.
/// \param orbitalShells The orbital shell count.
/// \param orbitalFunctions The orbital function count.
/// \param auxPairs The canonical aux pair count.
/// \param auxShells The aux shell count.
/// \param auxFunctions The aux function count.
/// \returns The two tables' byte size.
inline std::size_t RiOrbitActionBytes(std::size_t order,
                                      std::size_t orbitalPairs,
                                      std::size_t orbitalShells,
                                      std::size_t orbitalFunctions,
                                      std::size_t auxPairs,
                                      std::size_t auxShells,
                                      std::size_t auxFunctions) noexcept {
    return LeanOrbitActionBytes(order, orbitalPairs, orbitalShells, orbitalFunctions) +
           LeanOrbitActionBytes(order, auxPairs, auxShells, auxFunctions);
}

/// The joint task-grid orbit action: the orbital basis's action (the bra
/// pair's images and the orbital functions' signed permutation) and the
/// auxiliary basis's action (the aux shell's image and the aux functions'
/// signed permutation), under ONE group.
///
/// The two `LeanOrbitAction`s are the same object the petite-list walk landed,
/// built over
/// the two bases' own pair lists. The orbital instance's `pairImage` is the
/// canonical image pair (its own canonicalization rule) and its
/// `shellImage` is the shell-level map the expansion's within-pair flips
/// read; the aux instance's `shellImage` is the aux shell's image, indexed
/// in the aux shell order the task grid uses (the aux pair list's shell
/// order, which is the aux pair store's order).
struct RiOrbitAction {
    /// The group order - the one both reductions carry (Create refuses a
    /// mismatch). The joint orbit of a cell is over elements 0..order-1 of
    /// this one group.
    std::size_t order = 1;
    /// The canonical orbital pair count (the task grid's bra axis).
    std::size_t nOrbitalPairs = 0;
    /// The aux shell count (the task grid's aux axis).
    std::size_t nAuxShells = 0;
    /// The orbital basis's action: pairImage = the canonical pair images,
    /// shellImage = the orbital shell images, functionImage/functionSign =
    /// the orbital functions' signed permutation.
    LeanOrbitAction orbital;
    /// The auxiliary basis's action: shellImage = the aux shell images,
    /// functionImage/functionSign = the aux functions' signed permutation.
    /// Its pairImage is built (the same object) but no task reads it - a
    /// task's aux index is a single shell, never a pair.
    LeanOrbitAction aux;

    /// Builds the joint action of one group over the orbital basis and the
    /// auxiliary basis.
    /// \param orbitalReduction The reduction over the orbital basis.
    /// \param pairList The orbital canonical pairs (the bra axis' index
    /// space).
    /// \param auxReduction The reduction over the auxiliary basis (the
    /// group's own action in the aux function index space).
    /// \param auxPairList The aux pair list (the aux axis' index space: its
    /// shells are the task grid's aux shells).
    /// \returns The joint action, or an Error (kInvalidArgument when the
    /// two reductions' group orders differ; otherwise the two
    /// LeanOrbitAction::Create refusals - a shape mismatch or a basis whose
    /// shells do not permute as units).
    static qcx::Result<RiOrbitAction> Create(const SymmetryReduction& orbitalReduction,
                                             const ShellPairList& pairList,
                                             const SymmetryReduction& auxReduction,
                                             const ShellPairList& auxPairList) {
        if (orbitalReduction.groupOrder != auxReduction.groupOrder)
        {
            return std::unexpected(qcx::Error{
                qcx::ErrorCode::kInvalidArgument,
                "the orbital and auxiliary symmetry reductions carry different group orders: "
                "the joint task-grid orbits need ONE group acting on both bases"});
        }

        auto orbital = LeanOrbitAction::Create(orbitalReduction, pairList);

        if (!orbital.has_value())
        {
            return std::unexpected(orbital.error());
        }

        auto aux = LeanOrbitAction::Create(auxReduction, auxPairList);

        if (!aux.has_value())
        {
            return std::unexpected(aux.error());
        }

        RiOrbitAction action;
        action.order = orbitalReduction.groupOrder;
        action.nOrbitalPairs = orbital->pairCount;
        action.nAuxShells = aux->shellCount;
        action.orbital = std::move(*orbital);
        action.aux = std::move(*aux);
        return action;
    }

    /// The action's retained byte size: both instances' tables, by the
    /// formula that charges them (RiOrbitActionBytes). Exposed so the
    /// Create-time charge and the tables cannot drift apart.
    /// \returns The two actions' tables' bytes.
    std::size_t Bytes() const noexcept {
        return RiOrbitActionBytes(order,
                                  orbital.pairCount,
                                  orbital.shellCount,
                                  orbital.functionCount,
                                  aux.pairCount,
                                  aux.shellCount,
                                  aux.functionCount);
    }
};

/// The walk's linear index of one 3c cell in the task grid's own order:
/// braPair major, auxShell minor (the order BuildScreenedRiTaskList fills
/// in, ri_engine.cpp). Monotone in that order, so the smallest index of an
/// orbit is the cell the fill meets FIRST - the orbit's representative, and
/// the same cell the measurement's orbit walk picks.
/// \param braPair The cell's canonical orbital pair index.
/// \param auxShell The cell's aux shell index.
/// \param nAuxShells The aux shell count (the aux axis' row stride).
/// \returns The cell's rank in the grid order.
inline std::size_t RiCellIndex(std::size_t braPair,
                               std::size_t auxShell,
                               std::size_t nAuxShells) noexcept {
    return braPair * nAuxShells + auxShell;
}

/// The walk's orbit-representative test over one SURVIVING 3c cell.
///
/// True when (braPair, auxShell) is the SMALLEST cell of its orbit among
/// the cells the grid KEEPS - the cell whose block the kernel computes and
/// the expansion covers the orbit from. The candidate set is the surviving
/// set (the cells that clear the Schwarz cutoff), not the raw grid: an
/// image the fill would have dropped must not win the minimum, or every
/// surviving member of that orbit would read as a non-representative and
/// the orbit's block would never be computed at all. With the candidate set
/// restricted this way every surviving cell is written exactly once - as
/// its orbit's minimum, or by that minimum's expansion.
///
/// The orbit of a cell (p, s) under the diagonal action is
/// {(pairImage[g][p], auxShellImage[g][s])}; every image is a canonical
/// pair crossed with an aux shell, the identity element (g = 0) returns the
/// cell, so the grid's own index decides the minimum.
///
/// The survival test MUST be the fill's own test - same doubles, same
/// cutoff, the same `<` - or the two disagree on the boundary and the
/// expansion's write set stops matching the computed set.
/// \param action The joint orbit action.
/// \param orbitalBounds The per-canonical-pair Schwarz bounds (Q_uv).
/// \param auxShellBounds The per-aux-shell Schwarz bounds (Q_P).
/// \param cutoff The grid's Schwarz cutoff (Q_uv * Q_P >= cutoff keeps).
/// \param braPair The cell's canonical orbital pair index.
/// \param auxShell The cell's aux shell index.
/// \returns True when the cell is its orbit's surviving minimum.
inline bool RiCellIsOrbitRep(const RiOrbitAction& action,
                             const std::vector<double>& orbitalBounds,
                             const std::vector<double>& auxShellBounds,
                             double cutoff,
                             std::size_t braPair,
                             std::size_t auxShell) noexcept {
    const std::size_t order = action.order;
    const std::size_t nAuxShells = action.nAuxShells;
    const std::size_t own = RiCellIndex(braPair, auxShell, nAuxShells);
    std::size_t best = own;

    for (std::size_t g = 1; g < order; ++g)
    {
        const std::size_t imagePair = action.orbital.pairImage[braPair * order + g];
        const std::size_t imageAux = action.aux.shellImage[auxShell * order + g];

        if (orbitalBounds[imagePair] * auxShellBounds[imageAux] < cutoff)
        {
            continue;
        }

        best = std::min(best, RiCellIndex(imagePair, imageAux, nAuxShells));
    }

    return best == own;
}

/// Expands one orbit representative's computed 3c block into one member's
/// block, expressed in the member's own task frame.
///
/// The expansion is the 3c invariance (g a, g b | g c) = s(a)s(b)s(c)
/// (a b | c) read as a block map: the member's block at the position of the
/// image functions equals the representative's block at the source position
/// times the three functions' signs. The signs are +-1 and the positions
/// come from the group's own function permutations, so the map introduces
/// no rounding of its own - what moves a byte against the plain walk is the
/// evaluation order of the ERI, not this arithmetic.
///
/// Both blocks are in the task frame the assembler's consumers use: the
/// packed layout eri_batch.hpp defines for one task - value at (a, b, c), a
/// in the cell's pair.i shell, b in its pair.j shell, c in its aux shell, at
/// EriBlockIndex(a, b, c, 0, nA, nB, nC, 1), whose index arithmetic this
/// function folds into the per-axis maps and row bases it hoists. The
/// representative's block is the kernel's output for the representative's
/// own task; the member's block comes out in the frame the kernel WOULD have
/// produced for the member's task, so the caller writes both through the
/// same tensor scatter.
///
/// The frame bookkeeping: the image pair's canonicalization may have put
/// the representative's pair.i images in the member pair's j slot (the
/// within-pair flip), which swaps the member's two bra axes. The three
/// image functions always land inside the member's three shells - the shell
/// closure Create validated - so the member's block is filled exactly once
/// per element.
/// \param action The joint orbit action (the images and signs).
/// \param pairList The orbital pair list the frames' shells come from.
/// \param auxPairList The aux pair list the member's aux shell comes from.
/// \param g The element mapping the representative's cell onto the member's
/// (g >= 1; the identity member is the representative itself, whose block
/// is the computed one and needs no expansion).
/// \param repBraPair The representative cell's canonical bra pair.
/// \param repAuxShell The representative cell's aux shell index.
/// \param memberBraPair The member cell's canonical bra pair (the image
/// pair, canonicalized).
/// \param memberAuxShell The member cell's aux shell index (the aux image).
/// \param repBlock The representative's computed block, in the
/// representative task's frame.
/// \param memberBlock The destination; must hold the member task's block
/// element count (the two member shells' and the member aux shell's
/// function counts multiplied - equal to the representative's by the shell
/// closure).
inline void RiExpandOrbitMemberBlock(const RiOrbitAction& action,
                                     const ShellPairList& pairList,
                                     const ShellPairList& auxPairList,
                                     std::size_t g,
                                     std::size_t repBraPair,
                                     std::size_t repAuxShell,
                                     std::size_t memberBraPair,
                                     std::size_t memberAuxShell,
                                     const double* repBlock,
                                     double* memberBlock) noexcept {
    const std::size_t order = action.order;
    const ShellPairIndex& repPair = pairList.pairs[repBraPair];
    const ShellPairIndex& memberPair = pairList.pairs[memberBraPair];

    // The representative's shells and their images. The image pair is the
    // canonical pair of the two image shells, so the image shells ARE the
    // member pair's two shells; which one came from the representative's
    // pair.i is the within-pair flip.
    const std::size_t imageShellI = action.orbital.shellImage[repPair.i * order + g];
    const std::size_t imageShellJ = action.orbital.shellImage[repPair.j * order + g];
    const bool flipBra = (imageShellI > imageShellJ);

    // The member task's frame: axis 0 is the member pair's .i shell (the
    // packed layout's row shell), axis 1 its .j shell. The member's aux
    // shell is the aux image by construction (the caller's memberAuxShell).
    const std::size_t memberShell[2] = {memberPair.i, memberPair.j};
    const std::size_t repOffset[3] = {pairList.shells[repPair.i].functionOffset,
                                      pairList.shells[repPair.j].functionOffset,
                                      auxPairList.shells[repAuxShell].functionOffset};
    const std::size_t repCount[3] = {ShellFunctionCount(pairList.shells[repPair.i]),
                                     ShellFunctionCount(pairList.shells[repPair.j]),
                                     ShellFunctionCount(auxPairList.shells[repAuxShell])};
    const std::size_t memberCount[3] = {ShellFunctionCount(pairList.shells[memberShell[0]]),
                                        ShellFunctionCount(pairList.shells[memberShell[1]]),
                                        ShellFunctionCount(auxPairList.shells[memberAuxShell])};
    // The function offset of the member shell each REPRESENTATIVE bra axis
    // lands in: the flip swaps the two.
    const std::size_t memberBraOffset[2] = {
        pairList.shells[memberShell[flipBra ? 1u : 0u]].functionOffset,
        pairList.shells[memberShell[flipBra ? 0u : 1u]].functionOffset};
    const std::size_t memberAuxOffset = auxPairList.shells[memberAuxShell].functionOffset;

    // The caller's member pair must BE the canonical pair of the two image
    // shells, or the member's frame and the images' positions are two
    // different things and the block comes out permuted.
    assert(((imageShellI == memberShell[0] && imageShellJ == memberShell[1]) ||
            (imageShellI == memberShell[1] && imageShellJ == memberShell[0])) &&
           "the member pair must be the canonical pair of the representative's image shells");
    assert(repCount[0] * repCount[1] * repCount[2] ==
               memberCount[0] * memberCount[1] * memberCount[2] &&
           "the expanded member block must have the representative's element count");

    // The per-axis image maps, hoisted: the position of a source function
    // inside the member's shell and that function's sign depend on the
    // SOURCE function alone, so they are computed once per axis entry
    // instead of once per block element. Measured on c8h18 (this build's own
    // probe): the per-element form made the expansion cost more than the
    // kernel evaluation it replaces.
    //
    // The axes are at most a shell's function count wide (a few functions
    // for the s/p/d shells of an orbital basis, up to a contracted f shell's
    // 7-9 on an aux one); the cap is the caller's checked precondition
    // (RiOrbitActionSupports), asserted here as well.
    constexpr std::size_t kAxisCap = kRiOrbitAxisCap;
    assert(repCount[0] <= kAxisCap && repCount[1] <= kAxisCap && repCount[2] <= kAxisCap &&
           "the expansion's per-axis map buffer is smaller than a shell");

    std::size_t position[3][kAxisCap];
    int sign[3][kAxisCap];

    for (std::size_t k = 0; k < repCount[0]; ++k)
    {
        const std::size_t source = repOffset[0] + k;
        position[0][k] = action.orbital.functionImage[source * order + g] - memberBraOffset[0];
        sign[0][k] = action.orbital.functionSign[source * order + g];
    }

    for (std::size_t k = 0; k < repCount[1]; ++k)
    {
        const std::size_t source = repOffset[1] + k;
        position[1][k] = action.orbital.functionImage[source * order + g] - memberBraOffset[1];
        sign[1][k] = action.orbital.functionSign[source * order + g];
    }

    for (std::size_t k = 0; k < repCount[2]; ++k)
    {
        const std::size_t source = repOffset[2] + k;
        position[2][k] = action.aux.functionImage[source * order + g] - memberAuxOffset;
        sign[2][k] = action.aux.functionSign[source * order + g];
    }

    // The packed layout's row shell is the member pair's .i: the
    // representative's pair.j images land there when the flip swapped the
    // two. The row and column bases are per (a, b) - the innermost loop is
    // then a flat run of the packed block's aux columns.
    for (std::size_t a = 0; a < repCount[0]; ++a)
    {
        for (std::size_t b = 0; b < repCount[1]; ++b)
        {
            // positionRow / positionCol of the original form: the flip sends
            // the representative's pair.j images to the member block's ROW.
            const std::size_t positionRow = flipBra ? position[1][b] : position[0][a];
            const std::size_t positionCol = flipBra ? position[0][a] : position[1][b];
            // The member block's (positionRow, positionCol) aux row, in the
            // packed layout: (positionCol * memberCount[0] + positionRow) *
            // memberCount[2] - EriBlockIndex(positionRow, positionCol, 0, 0,
            // memberCount..., 1) folded.
            const std::size_t rowBase =
                (positionCol * memberCount[0] + positionRow) * memberCount[2];
            const double* source = repBlock + (b * repCount[0] + a) * repCount[2];
            double* target = memberBlock + rowBase;
            const int signAB = sign[0][a] * sign[1][b];

            for (std::size_t c = 0; c < repCount[2]; ++c)
            {
                target[position[2][c]] = static_cast<double>(signAB * sign[2][c]) * source[c];
            }
        }
    }
}

} // namespace qcx::integrals::internal
