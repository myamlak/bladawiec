#pragma once

/// \file
/// The lean builder's ORBIT ACTION (the petite-list increment): the tables
/// of one Abelian reduction that let the
/// streaming row walk visit ONE canonical cell per symmetry orbit and expand
/// the orbit representative's ERI block over the orbit's members at
/// contraction time - the mechanism measured as 2,379,972 ERI
/// blocks over 8,811,481 screened cells (0.2701, 3.70x fewer blocks) on
/// c8h18/def2-SVP.
///
/// WHY THIS SHAPE, and not the machinery's quartet-orbit table. The petite
/// list proper (symmetry_reduction.hpp) carries the orbits THEMSELVES:
/// ClassPair::orbits holds one ClassOrbitMember per member quartet (braPair,
/// ketPair, generator and a four-slot permutation - ~56 bytes each). Consuming
/// that table in the lean builder would retain the member space: on the same
/// fixture 8,811,481 members is ~470 MB of member records, plus ~2.4M per-orbit
/// containers (a 24-byte vector header plus the allocator's own per-container
/// overhead, ~250 MB) - ~0.75 GB, the working set the measurement observed
/// when the c8h18 orbit expansions were materialized. The lean builder is the
/// <= 1000-basis-function default member of the direct family, whose
/// whole character is a small, immutable, bounded Create-time state, so the
/// member table is exactly what it must not grow. The action below buys the
/// same work reduction from a per-PAIR map of the group's action - O(nPairs x
/// |group|) bytes: 84 KB of pair images on the same fixture and ~6 KB for the
/// shell and function tables - plus the function-level signed permutation, so
/// a walk-time test can decide "is this cell its orbit's
/// representative?" in O(|group|) and the expansion is computed FROM the
/// representative's block at contraction time. Nothing per quartet is
/// retained.
///
/// WHAT THE ACTION REQUIRES of the reduction (validated at Create, never
/// assumed): the shared signed-permutation shape (ValidateSymmetryReductionShape,
/// lean_pair_record.hpp) AND that every shell maps onto ONE shell - the shell
/// closure the machinery's own pair map relies on (ImagePairIndex,
/// symmetry_reduction.cpp) and without which the expansion's position
/// arithmetic has no meaning. A reduction that fails either is refused, never
/// believed.

#include "lean_pair_record.hpp"
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

/// The retained byte size of one orbit action over a basis of the given
/// sizes: the four tables. The Create-time envelope charges exactly this
/// (EstimatePeakBytes' orbitActionBytes term) and LeanOrbitAction::Bytes
/// reports it, so the charge and the tables cannot drift apart.
/// \param order The group order (0 = no action, no charge).
/// \param pairCount The canonical pair count.
/// \param shellCount The shell count.
/// \param functionCount The basis function count.
/// \returns The tables' byte size.
inline std::size_t LeanOrbitActionBytes(std::size_t order,
                                        std::size_t pairCount,
                                        std::size_t shellCount,
                                        std::size_t functionCount) noexcept {
    return order * (sizeof(std::uint32_t) * (pairCount + shellCount + functionCount) +
                    sizeof(std::int8_t) * functionCount);
}

/// The pair- and function-level action of one Abelian reduction over one
/// basis: what the row walk's orbit-representative test and the block
/// expansion read. Every table is INDEX-major ([index * order + g]) - the
/// walk and the contraction both sweep one pair over the group, so the
/// group index is the inner stride.
struct LeanOrbitAction {
    /// The group order (the reduction's own; >= 2 whenever the action exists
    /// - order 1 is the trivial group and needs no tables).
    std::size_t order = 1;
    /// The canonical pair count (the pairImage row width).
    std::size_t pairCount = 0;
    /// The shell count (the shellImage row width).
    std::size_t shellCount = 0;
    /// The basis function count (the functionImage/functionSign row width).
    std::size_t functionCount = 0;
    /// [p * order + g] -> the canonical image pair of canonical pair p under
    /// element g (the image shells canonicalized by shell index - the
    /// machinery's ImagePairIndex rule). The representative test's and the
    /// member enumeration's only input.
    std::vector<std::uint32_t> pairImage;
    /// [s * order + g] -> the shell element g maps shell s onto.
    std::vector<std::uint32_t> shellImage;
    /// [f * order + g] -> the global function index element g maps function
    /// f onto (the reduction's own permutation, refit for the walk).
    std::vector<std::uint32_t> functionImage;
    /// [f * order + g] -> +1 or -1: the sign of that image (the reduction's
    /// own sign, refit as a signed byte).
    std::vector<std::int8_t> functionSign;

    /// Builds the action of one reduction over one canonical pair list.
    /// \param reduction The reduction (validated here: the shared shape
    /// checks plus the shell closure - a shell whose functions map onto more
    /// than one shell is refused, kUnimplemented).
    /// \param pairList The canonical pairs (and the shells) the tables index.
    /// \returns The action, or an Error.
    static qcx::Result<LeanOrbitAction> Create(const SymmetryReduction& reduction,
                                               const ShellPairList& pairList) {
        const std::size_t order = reduction.groupOrder;
        const std::size_t n = pairList.functionCount;
        const std::size_t nShells = pairList.shells.size();
        const std::size_t nPairs = pairList.pairs.size();

        auto shape = ValidateSymmetryReductionShape(reduction, n);

        if (!shape.has_value())
        {
            return std::unexpected(shape.error());
        }

        // shellOfFunction[f] = the shell containing function f (the shells
        // cover contiguous function ranges in ascending offset order).
        std::vector<std::size_t> shellOfFunction(n, 0);

        for (std::size_t s = 0; s < nShells; ++s)
        {
            const ShellInfo& shell = pairList.shells[s];

            for (std::size_t k = 0; k < ShellFunctionCount(shell); ++k)
            {
                shellOfFunction[shell.functionOffset + k] = s;
            }
        }

        LeanOrbitAction action;
        action.order = order;
        action.pairCount = nPairs;
        action.shellCount = nShells;
        action.functionCount = n;
        action.pairImage.resize(order * nPairs);
        action.shellImage.resize(order * nShells);
        action.functionImage.resize(order * n);
        action.functionSign.resize(order * n);

        for (std::size_t g = 0; g < order; ++g)
        {
            for (std::size_t f = 0; f < n; ++f)
            {
                action.functionImage[f * order + g] =
                    static_cast<std::uint32_t>(reduction.permutation[g][f]);
                action.functionSign[f * order + g] = static_cast<std::int8_t>(reduction.sign[g][f]);
            }

            // The shell images. Every shell must map onto ONE shell: the
            // expansion reads the image function's position inside the target
            // shell (a function-level permutation WITHIN a shell), which only
            // exists when the image functions of a shell stay together.
            for (std::size_t s = 0; s < nShells; ++s)
            {
                const ShellInfo& shell = pairList.shells[s];
                const std::size_t count = ShellFunctionCount(shell);
                const std::size_t imageShell =
                    shellOfFunction[reduction.permutation[g][shell.functionOffset]];

                for (std::size_t k = 0; k < count; ++k)
                {
                    if (shellOfFunction[reduction.permutation[g][shell.functionOffset + k]] !=
                        imageShell)
                    {
                        return std::unexpected(qcx::Error{
                            qcx::ErrorCode::kUnimplemented,
                            "the symmetry reduction maps one shell's functions onto more than one "
                            "shell: the orbit expansion needs the shells to permute as units"});
                    }
                }

                action.shellImage[s * order + g] = static_cast<std::uint32_t>(imageShell);
            }

            // The pair images: the canonical image pair of every canonical
            // pair (its shells' images, canonicalized by shell index).
            for (std::size_t p = 0; p < nPairs; ++p)
            {
                const std::size_t imageI = action.shellImage[pairList.pairs[p].i * order + g];
                const std::size_t imageJ = action.shellImage[pairList.pairs[p].j * order + g];
                action.pairImage[p * order + g] = static_cast<std::uint32_t>(
                    PairIndexOf(std::min(imageI, imageJ), std::max(imageI, imageJ), pairList));
            }
        }

        return action;
    }

    /// The action's retained byte size - the formula the Create-time envelope
    /// charges (EstimatePeakBytes' orbitActionBytes term), exposed so the
    /// charge and the tables cannot drift apart.
    /// \returns The four tables' bytes.
    std::size_t Bytes() const noexcept {
        return LeanOrbitActionBytes(order, pairCount, shellCount, functionCount);
    }
};

/// The walk's linear index of one canonical cell (braPair >= ketPair): the
/// row-major rank braPair * (braPair + 1) / 2 + ketPair. Monotone in the
/// walk's own visitation order (rows ascending, then the ket prefix), so the
/// smallest index of an orbit is the cell the walk meets FIRST - the orbit's
/// representative, and the same one the machinery's orbit builder picks (its
/// quartets are sorted lexicographically in (bra, ket), symmetry_reduction.cpp
/// BuildOrbits).
/// \param braPair The cell's bra pair index.
/// \param ketPair The cell's ket pair index (<= braPair).
/// \returns The cell's rank in the walk's order.
inline std::size_t LeanCellIndex(std::size_t braPair, std::size_t ketPair) noexcept {
    return (braPair * (braPair + 1)) / 2 + ketPair;
}

/// The walk's orbit-representative test over one surviving cell.
///
/// True when (braPair, ketPair) is the SMALLEST cell of its orbit among the
/// cells the walk KEEPS - the cell whose ERI block the flush computes and the
/// expansion covers the orbit from. The candidate set is the surviving set
/// (the cells that clear both the provable-zero test and the Schwarz cutoff),
/// not the raw cell space: an image the walk would have dropped must not win
/// the minimum, or every surviving member of that orbit would read as a
/// non-representative and the whole orbit would vanish from the build. With
/// the candidate set restricted this way every surviving cell is contracted
/// exactly once - as its orbit's minimum, or as one of that minimum's members.
///
/// The orbit of a canonical cell (i, j) under the diagonal action is
/// {(canon(imagePair(g, i)), canon(imagePair(g, j)))}; every image is itself
/// a canonical cell, and the identity element returns the cell, so the walk's
/// own index decides the minimum.
/// \param action The orbit action (pairImage is this test's only input).
/// \param records The per-pair records the survival of an image is tested on.
/// \param cutoff The walk's Schwarz cutoff (Q_bra * Q_ket >= cutoff keeps).
/// \param braPair The cell's bra pair index.
/// \param ketPair The cell's ket pair index (<= braPair).
/// \returns True when the cell is its orbit's surviving minimum.
inline bool LeanCellIsOrbitRep(const LeanOrbitAction& action,
                               const LeanShellPairRecord* records,
                               double cutoff,
                               std::size_t braPair,
                               std::size_t ketPair) noexcept {
    const std::size_t order = action.order;
    const std::size_t own = LeanCellIndex(braPair, ketPair);
    std::size_t best = own;

    for (std::size_t g = 1; g < order; ++g)
    {
        const std::size_t imageBra = action.pairImage[braPair * order + g];
        const std::size_t imageKet = action.pairImage[ketPair * order + g];
        const std::size_t cellBra = std::max(imageBra, imageKet);
        const std::size_t cellKet = std::min(imageBra, imageKet);

        if (LeanPairCellIsProvablyZero(records[cellBra], records[cellKet]))
        {
            continue;
        }

        if (records[cellBra].qSchwarz * records[cellKet].qSchwarz < cutoff)
        {
            continue;
        }

        best = std::min(best, LeanCellIndex(cellBra, cellKet));
    }

    return best == own;
}

/// The fp64 element count of one canonical pair's ERI block: the product of
/// its two shells' function counts (the same per-pair block mass the
/// machinery's pair store carries as nFuncs, computed here from the pair list
/// - a member's pair need never be built in the store).
/// \param pairList The pair list (function counts per shell).
/// \param pair The canonical pair index.
/// \returns The pair's block element count.
inline std::size_t LeanPairBlockElements(const ShellPairList& pairList, std::size_t pair) noexcept {
    return ShellFunctionCount(pairList.shells[pairList.pairs[pair].i]) *
           ShellFunctionCount(pairList.shells[pairList.pairs[pair].j]);
}

/// Expands one orbit representative's computed ERI block into one member's
/// block, expressed in the member's canonical-role frame.
///
/// The expansion is the ERI invariance (g mu g nu | g lambda g sigma) =
/// s(mu)s(nu)s(lambda)s(sigma) (mu nu | lambda sigma) read as a block map: the
/// member's block at the position of the image functions equals the
/// representative's block at the source position times the four functions'
/// signs. Signs are +-1 and positions come from the group's own function
/// permutation, so the map introduces no rounding of its own - what moves a
/// byte against the plain walk is the ERI's ordering identity, not this
/// arithmetic (the 2026-09-12 amendment, and the machinery's own
/// class path is pinned the same way, fock_build_test.cpp:772).
///
/// The frame bookkeeping the caller supplies: the representative's block is
/// indexed in the frame of the TASK the assembler emitted for the
/// representative's cell (its canonical-role pair order - the L-role swap may
/// have made the task's bra pair the cell's ket pair), and the member's block
/// must come out in the frame of the task the assembler WOULD emit for the
/// member's cell. The three maps composed here are the two role swaps and the
/// group action's own axis map (the within-pair flips and the whole-quartet
/// role swap the image pairs' canonicalization implies).
/// \param action The orbit action (the images and signs).
/// \param pairList The pair list the frames' shells come from.
/// \param g The generator mapping the representative cell onto the member
/// (g >= 1; the identity member is the representative itself and is contracted
/// straight from the tile).
/// \param repTaskBra The representative task's canonical-role bra pair.
/// \param repTaskKet The representative task's canonical-role ket pair.
/// \param memberTaskBra The member task's canonical-role bra pair.
/// \param memberTaskKet The member task's canonical-role ket pair.
/// \param imageBraPair The image of the representative CELL's bra pair under
/// g (before canonicalization: the whole-quartet role swap is read off it).
/// \param imageKetPair The image of the representative cell's ket pair under g.
/// \param repBlock The representative's computed block; must be the block of
/// \p repTaskBra / \p repTaskKet (not of the cell).
/// \param memberBlock The destination; must hold at least the member task's
/// block element count (LeanPairBlockElements of the two member pairs
/// multiplied).
inline void LeanExpandOrbitMemberBlock(const LeanOrbitAction& action,
                                       const ShellPairList& pairList,
                                       std::size_t g,
                                       std::size_t repTaskBra,
                                       std::size_t repTaskKet,
                                       std::size_t memberTaskBra,
                                       std::size_t memberTaskKet,
                                       std::size_t imageBraPair,
                                       std::size_t imageKetPair,
                                       const double* repBlock,
                                       double* memberBlock) noexcept {
    const std::size_t order = action.order;
    const std::size_t cellBra = std::max(repTaskBra, repTaskKet);
    const std::size_t cellKet = std::min(repTaskBra, repTaskKet);
    const std::size_t memberCellBra = std::max(memberTaskBra, memberTaskKet);

    // The two role swaps: the assembler hands a task the LOWER-L side as its
    // bra pair, so a task's bra pair is the cell's bra pair or its ket pair.
    const bool repRoleSwap = (repTaskBra != cellBra);
    const bool memberRoleSwap = (memberTaskBra != memberCellBra);

    // The representative cell's quartet, axis by axis (0, 1 = its bra pair's
    // two shells, 2, 3 = its ket pair's), and the images of those shells.
    const ShellPairIndex& repCellBraPair = pairList.pairs[cellBra];
    const ShellPairIndex& repCellKetPair = pairList.pairs[cellKet];
    const std::size_t repCellShell[4] = {
        repCellBraPair.i, repCellBraPair.j, repCellKetPair.i, repCellKetPair.j};
    const std::size_t imageBraShell[2] = {action.shellImage[repCellShell[0] * order + g],
                                          action.shellImage[repCellShell[1] * order + g]};
    const std::size_t imageKetShell[2] = {action.shellImage[repCellShell[2] * order + g],
                                          action.shellImage[repCellShell[3] * order + g]};

    // The within-pair flips (an image pair whose shells come out in the other
    // order) and the whole-quartet role swap (the image bra pair's index below
    // the image ket pair's - the member cell's bra side then holds the
    // representative's KET images).
    const bool flipBra = imageBraShell[0] > imageBraShell[1];
    const bool flipKet = imageKetShell[0] > imageKetShell[1];
    const bool groupRoleSwap = (imageBraPair < imageKetPair);

    // The member CELL axis receiving each representative CELL axis.
    std::size_t memberCellAxis[4];
    memberCellAxis[0] = flipBra ? 1 : 0;
    memberCellAxis[1] = flipBra ? 0 : 1;
    memberCellAxis[2] = flipKet ? 3 : 2;
    memberCellAxis[3] = flipKet ? 2 : 3;

    if (groupRoleSwap)
    {
        for (std::size_t k = 0; k < 4; ++k)
        {
            memberCellAxis[k] = (memberCellAxis[k] + 2) % 4;
        }
    }

    // The member TASK axis receiving each representative TASK axis: task ->
    // cell (the representative's role swap), the group action, then cell ->
    // task (the member's role swap). A permutation by construction - the
    // group maps shells bijectively, the canonicalizations permute axes - and
    // asserted below rather than assumed.
    std::size_t memberTaskAxis[4];

    for (std::size_t t = 0; t < 4; ++t)
    {
        const std::size_t repCellAxis = repRoleSwap ? (t + 2) % 4 : t;
        const std::size_t memberCellAxisOfRep = memberCellAxis[repCellAxis];
        memberTaskAxis[t] = memberRoleSwap ? (memberCellAxisOfRep + 2) % 4 : memberCellAxisOfRep;
    }

    const ShellPairIndex& repTaskBraPair = pairList.pairs[repTaskBra];
    const ShellPairIndex& repTaskKetPair = pairList.pairs[repTaskKet];
    const std::size_t repTaskShell[4] = {
        repTaskBraPair.i, repTaskBraPair.j, repTaskKetPair.i, repTaskKetPair.j};
    const ShellPairIndex& memberTaskBraPair = pairList.pairs[memberTaskBra];
    const ShellPairIndex& memberTaskKetPair = pairList.pairs[memberTaskKet];
    const std::size_t memberTaskShell[4] = {
        memberTaskBraPair.i, memberTaskBraPair.j, memberTaskKetPair.i, memberTaskKetPair.j};
    const std::size_t repOffset[4] = {pairList.shells[repTaskShell[0]].functionOffset,
                                      pairList.shells[repTaskShell[1]].functionOffset,
                                      pairList.shells[repTaskShell[2]].functionOffset,
                                      pairList.shells[repTaskShell[3]].functionOffset};
    const std::size_t repCount[4] = {ShellFunctionCount(pairList.shells[repTaskShell[0]]),
                                     ShellFunctionCount(pairList.shells[repTaskShell[1]]),
                                     ShellFunctionCount(pairList.shells[repTaskShell[2]]),
                                     ShellFunctionCount(pairList.shells[repTaskShell[3]])};
    const std::size_t memberOffset[4] = {pairList.shells[memberTaskShell[0]].functionOffset,
                                         pairList.shells[memberTaskShell[1]].functionOffset,
                                         pairList.shells[memberTaskShell[2]].functionOffset,
                                         pairList.shells[memberTaskShell[3]].functionOffset};
    const std::size_t memberCount[4] = {ShellFunctionCount(pairList.shells[memberTaskShell[0]]),
                                        ShellFunctionCount(pairList.shells[memberTaskShell[1]]),
                                        ShellFunctionCount(pairList.shells[memberTaskShell[2]]),
                                        ShellFunctionCount(pairList.shells[memberTaskShell[3]])};

    assert(memberTaskAxis[0] != memberTaskAxis[1] && memberTaskAxis[0] != memberTaskAxis[2] &&
           memberTaskAxis[0] != memberTaskAxis[3] && memberTaskAxis[1] != memberTaskAxis[2] &&
           memberTaskAxis[1] != memberTaskAxis[3] && memberTaskAxis[2] != memberTaskAxis[3] &&
           "the composed axis map must be a permutation of the four quartet axes");
    // The shell closure makes the images carry their sources' function counts,
    // so the two frames' blocks are the same size - the bijection contract the
    // expansion writes under (the machinery asserts the same identity for its
    // own expansion, fock_build.cpp).
    assert(repCount[0] * repCount[1] * repCount[2] * repCount[3] ==
               memberCount[0] * memberCount[1] * memberCount[2] * memberCount[3] &&
           "the expanded member block must have the representative's element count");

    for (std::size_t a = 0; a < repCount[0]; ++a)
    {
        const std::size_t sourceA = repOffset[0] + a;
        const std::size_t positionA =
            action.functionImage[sourceA * order + g] - memberOffset[memberTaskAxis[0]];
        const int signA = action.functionSign[sourceA * order + g];

        for (std::size_t b = 0; b < repCount[1]; ++b)
        {
            const std::size_t sourceB = repOffset[1] + b;
            const std::size_t positionB =
                action.functionImage[sourceB * order + g] - memberOffset[memberTaskAxis[1]];
            const int signB = action.functionSign[sourceB * order + g];

            for (std::size_t c = 0; c < repCount[2]; ++c)
            {
                const std::size_t sourceC = repOffset[2] + c;
                const std::size_t positionC =
                    action.functionImage[sourceC * order + g] - memberOffset[memberTaskAxis[2]];
                const int signC = action.functionSign[sourceC * order + g];

                for (std::size_t d = 0; d < repCount[3]; ++d)
                {
                    const std::size_t sourceD = repOffset[3] + d;
                    const std::size_t positionD =
                        action.functionImage[sourceD * order + g] - memberOffset[memberTaskAxis[3]];
                    const int signD = action.functionSign[sourceD * order + g];

                    std::size_t target[4];
                    target[memberTaskAxis[0]] = positionA;
                    target[memberTaskAxis[1]] = positionB;
                    target[memberTaskAxis[2]] = positionC;
                    target[memberTaskAxis[3]] = positionD;

                    memberBlock[EriBlockIndex(target[0],
                                              target[1],
                                              target[2],
                                              target[3],
                                              memberCount[0],
                                              memberCount[1],
                                              memberCount[2],
                                              memberCount[3])] =
                        static_cast<double>(signA * signB * signC * signD) *
                        repBlock[EriBlockIndex(
                            a, b, c, d, repCount[0], repCount[1], repCount[2], repCount[3])];
                }
            }
        }
    }
}

} // namespace qcx::integrals::internal
