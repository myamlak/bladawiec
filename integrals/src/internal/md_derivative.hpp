#pragma once

// The nuclear-coordinate derivative tier of the one-electron MD path: the
// analytic dS/dR, dT/dR and dV/dR of the overlap, kinetic-energy and
// nuclear-attraction pair blocks, to a caller-chosen derivative order. The
// order is a runtime argument of one function per integral kind - the
// second derivative is a new order, not a second function beside the first.
//
// Every application of one nuclear coordinate is one of three exact
// operators on the pair integral <a|O|b> [Helgaker2000] [McMurchie1978]:
//   bra center A, axis k:  2a <a+e_k|O|b> - i_k <a-e_k|O|b>
//   ket center B, axis k:  2b <a|O|b+e_k> - j_k <a|O|b-e_k>
//   operator center C, k:  the nuclear-attraction operator's own center
//                          moves with the atom that carries the nucleus
// The first two are the exact derivative of the bra (ket) function
// (x-A)^i e^{-a|r-A|^2}, whose A-derivative is 2a (x-A)^{i+e_k} -
// i_k (x-A)^{i-e_k} times the same exponential. They therefore carry the
// whole derivative - the Gaussian prefactor K = exp(-ab/p |A-B|^2)
// included - as a shifted pair integral, so no term of this file
// differentiates P, p or K explicitly.
//   The operator center is the same statement one index space over: with
//   the 1-center VRR table [t]^(m) of md_one_electron.hpp and xi = P - C,
//   d[t]^(0)/dxi_k = [t+e_k]^(0) (the seed obeys d[0]^(m)/dxi_k =
//   -2p xi_k [0]^(m+1), which the VRR's first term turns into [e_k]^(m)),
//   so d/dC_k is the +[t+e_k]^(0) index raise of the same recurrence.
//
// A k-th derivative is the product of k such operators, one per tuple slot,
// summed over the operators available at each slot. The three act on
// independent index spaces - the bra E index, the ket E index, the
// operator's Hermite index - so a tuple's terms are the cross product of
// its slots' choices, and only the operator-center choices of one and the
// same atom compose into a single raised index (and, with it, a sign).
//
// Two mechanisms evaluate the terms, each the natural one for its kernel:
//   - Overlap and nuclear attraction fold the operator through a per-axis E
//     table (BuildDerivedPairTables), so FoldPairETable, the shell
//     transforms and the contraction tail run on the differentiated
//     product - the spherical fold of a differentiated table is the
//     differentiated spherical fold, the transform being a constant linear
//     map. Only their tier range is not the plain pair's: the coefficients
//     2a E^{(i+1,j)}_t - i E^{(i-1,j)}_t of a raise reach tier la + lb + 1,
//     so a term's tables, its fold and the kernel it contracts against all
//     run to la + lb + raises (TermTierRaises) where the plain pair stops
//     at la + lb. Folding no further silently drops those tiers; the
//     overlap path is insensitive to them only because its kernel is the
//     Hermite selector at t = 0.
//   - Kinetic energy cannot: its coefficients (3 + 2|i|) a, -2a^2 and
//     -1/2 i_k(i_k-1) depend on the very index the operator shifts, so the
//     path evaluates its own per-component expression at the shifted index
//     (KineticTermComponent). Both mechanisms are the same operator, and
//     the finite-difference test pins whichever one each kernel uses.
//
// Output layout: one (nFuncsA x nFuncsB) row-major block per derivative
// tuple, tuple-major in request order - the layout of the base builders
// (md_one_electron.hpp), so a caller compares blocks directly. A tuple's
// coordinates are the flat encoding 3 * atom + axis (axis 0 = x, 1 = y,
// 2 = z) of the request.

#include "md_batch.hpp"
#include "md_boys.hpp"
#include "md_defs.hpp"
#include "md_hermite.hpp"
#include "qcx/error.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qcx::integrals::internal {

/// One nuclear coordinate of a derivative tuple: the atom carrying it and
/// the Cartesian axis.
struct MdDerivativeCoordinate {
    std::size_t atom; ///< Atom index (molecule order).
    int axis; ///< Cartesian axis: 0 = x, 1 = y, 2 = z.
};

/// The atom a derivative term carries when its combination applied no
/// operator-center operator.
inline constexpr std::size_t kNoDerivativeAtom = static_cast<std::size_t>(-1);

/// The operator applications of one derivative term, per center and axis: a
/// bra application raises the bra E index along its axis, a ket application
/// the ket E index, an operator-center application the 1-center Hermite
/// index.
struct MdDerivativeRaises {
    std::array<int, 3> bra{}; ///< Bra-center (A) applications per axis.
    std::array<int, 3> ket{}; ///< Ket-center (B) applications per axis.
    std::array<int, 3> op{}; ///< Operator-center (C) applications per axis.
};

/// One term of one derivative tuple: the operator applications of one
/// combination of the tuple's slot choices, and the one atom whose nucleus
/// table the raised Hermite index reads.
struct MdDerivativeTerm {
    MdDerivativeRaises raises{}; ///< The applications, per center and axis.
    std::size_t operatorAtom = kNoDerivativeAtom; ///< The operator-center atom, or none.
};

/// The most terms one tuple enumerates: every slot offers at most three
/// operators (bra, ket, operator center), so an order-k tuple has at most
/// 3^k of them - 9 at kMaxMdDerivativeOrder.
inline constexpr int kMaxMdDerivativeTermCount = 9;

/// The enumerated terms of one tuple.
struct MdDerivativeTerms {
    std::array<MdDerivativeTerm, kMaxMdDerivativeTermCount> terms{}; ///< The terms.
    int count = 0; ///< Number of terms holding a value.
};

/// The operator kinds one slot can apply. A slot offers a bra (ket)
/// application only when its atom carries the bra (ket) shell, and an
/// operator-center application only where the operator has a center of its
/// own - the nuclear-attraction path.
enum class MdDerivativeKind : unsigned char {
    kBra = 0, ///< The bra center's operator.
    kKet = 1, ///< The ket center's operator.
    kOperatorCenter = 2, ///< The operator center's operator.
};

/// The scratch of one derivative call, held by the caller across the tuples
/// and primitives so a call allocates once: the fold target, the in-place
/// operator copy, the 1-center VRR buffers (the raised tables) and the
/// derived per-axis tables. Every member is reused at the size it last
/// needed - the per-term extents vary, so the vectors keep their capacity.
struct MdDerivativeScratch {
    std::vector<double> folded; ///< FoldPairETable target.
    std::vector<double> operatorScratch; ///< The pre-update copy of ApplyDerivativeOperator.
    std::vector<double> centerTable; ///< The charge-summed 1-center table of the plain terms.
    std::vector<double> centerRead; ///< One nucleus' table, as the VRR wrote it.
    std::vector<double> centerGather; ///< centerRead's raised-index gather, the terms' kernel.
    std::vector<double> hermiteSelector; ///< The overlap's E at t = 0 selector.
    std::vector<double> perCart; ///< The kinetic path's per-cartesian-row buffer.
    std::vector<double> slice; ///< 1-center VRR ping-pong buffer.
    std::vector<double> prevSlice; ///< 1-center VRR ping-pong buffer.
    std::vector<double> seeds; ///< 1-center VRR targets.
    std::array<PerAxisETable, 3> derivedTables; ///< The term's operator-raised tables.
};

/// One nuclear coordinate decoded from the flat encoding 3 * atom + axis.
/// \param encoded The flat encoding.
/// \returns The atom and axis.
inline MdDerivativeCoordinate DecodeDerivativeCoordinate(std::size_t encoded) noexcept {
    return MdDerivativeCoordinate{encoded / 3, static_cast<int>(encoded % 3)};
}

/// One tuple's coordinates decoded from the flat encoding.
/// \param flat The flat coordinates, order values.
/// \param order The derivative order.
/// \returns The tuple, with the trailing slots unused.
inline std::array<MdDerivativeCoordinate, kMaxMdDerivativeOrder> DecodeTuple(
    std::span<const std::size_t> flat, int order) noexcept {
    std::array<MdDerivativeCoordinate, kMaxMdDerivativeOrder> tuple{};

    for (int slot = 0; slot < order; ++slot)
    {
        tuple[static_cast<std::size_t>(slot)] =
            DecodeDerivativeCoordinate(flat[static_cast<std::size_t>(slot)]);
    }

    return tuple;
}

/// The Cartesian power of one index along one axis.
/// \param index The index.
/// \param axis The axis: 0 = x, 1 = y, 2 = z.
/// \returns The power along that axis.
inline int CartesianPower(const CartIndex& index, int axis) noexcept {
    return axis == 0 ? index.ix : axis == 1 ? index.iy : index.iz;
}

/// One index shifted by one along one axis.
/// \param index The index.
/// \param axis The axis: 0 = x, 1 = y, 2 = z.
/// \param delta The shift (+1 for a raise, -1 for a lower).
/// \returns The shifted index.
inline CartIndex ShiftCartIndex(CartIndex index, int axis, int delta) noexcept {
    if (axis == 0)
    {
        index.ix += delta;
    } else if (axis == 1)
    {
        index.iy += delta;
    } else
    {
        index.iz += delta;
    }

    return index;
}

/// The flat storage index of one per-axis E table cell - the layout
/// PerAxisETable::At carries, at the table's own storage dims.
/// \param i Bra index.
/// \param j Ket index.
/// \param t Hermite index.
/// \param strideLb The table's storage ket extent (PerAxisETable::Reset's lb).
/// \param tMax The table's storage Hermite extent.
/// \returns The index into the table's values.
inline std::size_t PerAxisTableIndex(int i, int j, int t, int strideLb, int tMax) noexcept {
    return static_cast<std::size_t>((i * (strideLb + 1) + j) * (tMax + 1) + t);
}

/// The block-span element count of one derivative request: one pair block
/// per tuple. The caller sizes its output span with this before calling.
/// \param pair The pair the request addresses.
/// \param order The derivative order.
/// \param coordinateCount The flat coordinate count (order per tuple).
/// \returns The element count, or 0 for a request CheckDerivativeRequest
/// refuses (an order outside the supported band, or a coordinate count that
/// is not a whole number of tuples) - the two agree, so a caller sizing its
/// span here never reads past what the call writes.
inline std::size_t MdDerivativeBlockCount(const MdPairData& pair,
                                          int order,
                                          std::size_t coordinateCount) noexcept {
    if (order < 1 || order > kMaxMdDerivativeOrder ||
        coordinateCount % static_cast<std::size_t>(order) != 0)
    {
        return 0;
    }

    return (coordinateCount / static_cast<std::size_t>(order)) * pair.nFuncs;
}

/// Validates one derivative request's shape: the order is within the band
/// the recurrence takes, the coordinate list divides into whole tuples, and
/// the block span holds exactly one pair block per tuple.
/// \param pair The pair the request addresses.
/// \param order The derivative order.
/// \param coordinateCount The flat coordinate count.
/// \param blockCount The element count of the caller's output span.
/// \returns The tuple count, or an Error (kInvalidArgument) naming the bound
/// that failed.
inline qcx::Result<std::size_t> CheckDerivativeRequest(const MdPairData& pair,
                                                       int order,
                                                       std::size_t coordinateCount,
                                                       std::size_t blockCount) {
    if (order < 1 || order > kMaxMdDerivativeOrder)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "integrals: derivative order " + std::to_string(order) +
                                              " is outside 1.." +
                                              std::to_string(kMaxMdDerivativeOrder)});
    }

    if (coordinateCount % static_cast<std::size_t>(order) != 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "integrals: derivative coordinate count " + std::to_string(coordinateCount) +
                           " is not a whole number of order-" + std::to_string(order) + " tuples"});
    }

    const std::size_t tupleCount = coordinateCount / static_cast<std::size_t>(order);

    if (blockCount != tupleCount * pair.nFuncs)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "integrals: derivative block span holds " +
                                              std::to_string(blockCount) + " elements, expected " +
                                              std::to_string(tupleCount * pair.nFuncs)});
    }

    return tupleCount;
}

/// Enumerates the terms of one derivative tuple: the cross product of the
/// operator kinds its slots offer. Combinations whose operator-center
/// choices name two different nuclei are left out, and that is exact rather
/// than an approximation - each nucleus' table depends on that nucleus'
/// position alone, so a mixed derivative across two of them is identically
/// zero.
/// \param tuple The tuple's coordinates, one per derivative order.
/// \param braAtom The atom carrying the bra shell.
/// \param ketAtom The atom carrying the ket shell (a same-atom pair passes
/// the same index twice).
/// \param hasOperatorCenter True for an operator with a center of its own.
/// \returns The terms, or an Error (kInvalidArgument) naming the tuple slot
/// and atom of a coordinate the path cannot differentiate.
inline qcx::Result<MdDerivativeTerms> DerivativeTermsOf(
    std::span<const MdDerivativeCoordinate> tuple,
    std::size_t braAtom,
    std::size_t ketAtom,
    bool hasOperatorCenter) {
    const int order = static_cast<int>(tuple.size());
    std::array<std::array<MdDerivativeKind, 3>, kMaxMdDerivativeOrder> choices{};
    std::array<int, kMaxMdDerivativeOrder> choiceCount{};

    for (int slot = 0; slot < order; ++slot)
    {
        const MdDerivativeCoordinate coordinate = tuple[static_cast<std::size_t>(slot)];
        int count = 0;

        if (coordinate.atom == braAtom)
        {
            choices[static_cast<std::size_t>(slot)][static_cast<std::size_t>(count)] =
                MdDerivativeKind::kBra;
            ++count;
        }

        if (coordinate.atom == ketAtom)
        {
            choices[static_cast<std::size_t>(slot)][static_cast<std::size_t>(count)] =
                MdDerivativeKind::kKet;
            ++count;
        }

        if (hasOperatorCenter)
        {
            choices[static_cast<std::size_t>(slot)][static_cast<std::size_t>(count)] =
                MdDerivativeKind::kOperatorCenter;
            ++count;
        }

        if (count == 0)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "integrals: derivative tuple slot " + std::to_string(slot) +
                               " names atom " + std::to_string(coordinate.atom) +
                               ", which carries neither shell of the pair (atoms " +
                               std::to_string(braAtom) + " and " + std::to_string(ketAtom) + ")"});
        }

        choiceCount[static_cast<std::size_t>(slot)] = count;
    }

    MdDerivativeTerms terms;
    std::array<int, kMaxMdDerivativeOrder> pick{};
    bool done = order == 0;

    while (!done)
    {
        MdDerivativeRaises raises;
        std::size_t operatorAtom = kNoDerivativeAtom;
        bool crossesNuclei = false;

        for (int slot = 0; slot < order; ++slot)
        {
            const MdDerivativeCoordinate coordinate = tuple[static_cast<std::size_t>(slot)];
            const MdDerivativeKind kind =
                choices[static_cast<std::size_t>(slot)]
                       [static_cast<std::size_t>(pick[static_cast<std::size_t>(slot)])];

            if (kind == MdDerivativeKind::kBra)
            {
                ++raises.bra[static_cast<std::size_t>(coordinate.axis)];
            } else if (kind == MdDerivativeKind::kKet)
            {
                ++raises.ket[static_cast<std::size_t>(coordinate.axis)];
            } else
            {
                ++raises.op[static_cast<std::size_t>(coordinate.axis)];

                if (operatorAtom == kNoDerivativeAtom)
                {
                    operatorAtom = coordinate.atom;
                } else if (operatorAtom != coordinate.atom)
                { crossesNuclei = true; }
            }
        }

        if (!crossesNuclei)
        {
            terms.terms[static_cast<std::size_t>(terms.count)] =
                MdDerivativeTerm{raises, operatorAtom};
            ++terms.count;
        }

        int slot = 0;

        for (; slot < order; ++slot)
        {
            ++pick[static_cast<std::size_t>(slot)];

            if (pick[static_cast<std::size_t>(slot)] < choiceCount[static_cast<std::size_t>(slot)])
            {
                break;
            }

            pick[static_cast<std::size_t>(slot)] = 0;
        }

        done = slot == order;
    }

    return terms;
}

/// Applies one derivative operator along one axis, in place, to a per-axis E
/// table: the bra operator D[E](i, j, t) = 2a E(i+1, j, t) - i E(i-1, j, t),
/// or the ket operator with (b, j). The reads run one row (column) past the
/// write range, which the raised extent provides; the second read comes from
/// the pre-update table, hence the copy through \p scratch.
/// \param table The table, at its storage dims (strideLb, tMax).
/// \param braSide True for the bra operator, false for the ket operator.
/// \param exponent The side's primitive exponent (a or b).
/// \param rowMax Bra extent the caller still needs after this application.
/// \param colMax Ket extent the caller still needs after this application.
/// \param strideLb The table's storage ket extent.
/// \param tMax The table's storage Hermite extent.
/// \param scratch Caller storage for the pre-update copy.
inline void ApplyDerivativeOperator(PerAxisETable& table,
                                    bool braSide,
                                    double exponent,
                                    int rowMax,
                                    int colMax,
                                    int strideLb,
                                    int tMax,
                                    std::vector<double>& scratch) {
    // Sized by the table's own stride, not by the write region: the staging
    // index is PerAxisTableIndex over (strideLb, tMax), so a pass that had
    // already given up columns (colMax < strideLb) would run past a buffer
    // sized (rowMax+1)(colMax+1)(tMax+1).
    scratch.assign(static_cast<std::size_t>(rowMax + 1) * static_cast<std::size_t>(strideLb + 1) *
                       static_cast<std::size_t>(tMax + 1),
                   0.0);

    for (int i = 0; i <= rowMax; ++i)
    {
        for (int j = 0; j <= colMax; ++j)
        {
            for (int t = 0; t <= tMax; ++t)
            {
                double value = 0.0;

                if (braSide)
                {
                    value = 2.0 * exponent * table.At(i + 1, j, t);

                    if (i >= 1)
                    {
                        value -= static_cast<double>(i) * table.At(i - 1, j, t);
                    }
                } else
                {
                    value = 2.0 * exponent * table.At(i, j + 1, t);

                    if (j >= 1)
                    {
                        value -= static_cast<double>(j) * table.At(i, j - 1, t);
                    }
                }

                scratch[PerAxisTableIndex(i, j, t, strideLb, tMax)] = value;
            }
        }
    }

    for (int i = 0; i <= rowMax; ++i)
    {
        for (int j = 0; j <= colMax; ++j)
        {
            for (int t = 0; t <= tMax; ++t)
            {
                table.At(i, j, t) = scratch[PerAxisTableIndex(i, j, t, strideLb, tMax)];
            }
        }
    }
}

/// Builds one derivative term's per-axis tables into the caller's scratch:
/// the pair's tables at the term's raised extent, with the term's bra
/// applications then its ket applications applied along each axis. The bra
/// applications run first and keep the full ket range, so the ket
/// applications still read the columns the bra pass wrote; each pass then
/// gives up the rows (columns) it no longer needs.
///
/// Every axis is raised by the term's TOTAL raise count on each side, not by
/// that axis' own: a differentiated coefficient reaches tier
/// la + lb + raises, and the fold contracts that whole tier range against
/// the kernel, so a raise on one axis puts reads on the other axes' tables
/// too (the fold fills the tiers of all three axes for each Cartesian pair).
/// At the padded extent those reads stay inside the table - the cells past a
/// row's own diagonal are zero, so they contribute nothing - and the rows
/// the fold does read (<= la, <= lb) keep the values the unpadded extent
/// holds, since the recurrence reads only lower (i, j) cells.
/// \param pair The pair (its la/lb and the centers of the shift vectors).
/// \param prim The primitive pair.
/// \param raises The term's per-axis applications.
/// \param scratch The call's scratch (its derivedTables and operatorScratch).
inline void BuildDerivedPairTables(const MdPairData& pair,
                                   const MdPrimPair& prim,
                                   const MdDerivativeRaises& raises,
                                   MdDerivativeScratch& scratch) {
    const std::array<double, 3> shiftA = {prim.px - pair.ax, prim.py - pair.ay, prim.pz - pair.az};
    const std::array<double, 3> shiftB = {prim.px - pair.bx, prim.py - pair.by, prim.pz - pair.bz};
    const double beta = prim.p - prim.exponentA;
    const int braTotal = raises.bra[0] + raises.bra[1] + raises.bra[2];
    const int ketTotal = raises.ket[0] + raises.ket[1] + raises.ket[2];

    for (int axis = 0; axis < 3; ++axis)
    {
        const std::size_t slot = static_cast<std::size_t>(axis);
        const int braRaises = raises.bra[slot];
        const int ketRaises = raises.ket[slot];
        const int extentA = pair.la + braTotal;
        const int extentB = pair.lb + ketTotal;
        const int tMax = extentA + extentB;
        PerAxisETable& table = scratch.derivedTables[slot];
        table.Reset(extentA, extentB);
        FillPerAxisTable(extentA, extentB, shiftA, shiftB, prim.p, axis, table);
        int rowMax = extentA;
        int colMax = extentB;

        for (int application = 0; application < braRaises; ++application)
        {
            --rowMax;
            ApplyDerivativeOperator(table,
                                    true,
                                    prim.exponentA,
                                    rowMax,
                                    colMax,
                                    extentB,
                                    tMax,
                                    scratch.operatorScratch);
        }

        for (int application = 0; application < ketRaises; ++application)
        {
            --colMax;
            ApplyDerivativeOperator(
                table, false, beta, rowMax, colMax, extentB, tMax, scratch.operatorScratch);
        }
    }
}

/// Accumulates one term's folded contraction into a pair block - the tail of
/// the three base builders (md_one_electron.hpp), with \p g carrying the
/// operator's kernel: the overlap's Hermite selector, the nuclear-attraction
/// table, or the operator-center table read at the raised index.
/// \param pair The pair.
/// \param primIdx The primitive pair the fold belongs to (its weights).
/// \param folded The folded angular-only E slice of the term's tables.
/// \param g The kernel over the Hermite index, one value per folded slot.
/// \param coefficient The term's kernel factor and sign (the weights
/// multiply on top).
/// \param block The pair block to accumulate into.
inline void AccumulateFoldContraction(const MdPairData& pair,
                                      std::size_t primIdx,
                                      const std::vector<double>& folded,
                                      std::span<const double> g,
                                      double coefficient,
                                      double* block) {
    const int nHerm = static_cast<int>(g.size());
    const std::size_t nAngA = pair.nFuncsA / pair.rowsA;
    const std::size_t nAngB = pair.nFuncsB / pair.rowsB;

    for (std::size_t rowA = 0; rowA < pair.rowsA; ++rowA)
    {
        for (std::size_t rowB = 0; rowB < pair.rowsB; ++rowB)
        {
            const double weight = pair.braWeights[primIdx][rowA * pair.rowsB + rowB];

            for (std::size_t fb = 0; fb < nAngB; ++fb)
            {
                for (std::size_t fa = 0; fa < nAngA; ++fa)
                {
                    double value = 0.0;

                    for (int t = 0; t < nHerm; ++t)
                    {
                        value +=
                            folded[(fa * nAngB + fb) * nHerm + t] * g[static_cast<std::size_t>(t)];
                    }

                    block[(rowA * nAngA + fa) * pair.nFuncsB + rowB * nAngB + fb] +=
                        coefficient * weight * value;
                }
            }
        }
    }
}

/// Builds the 1-center VRR table of ONE nucleus: the table BuildNuclearPair
/// (md_one_electron.hpp) accumulates, run to l1 + |raise| so that the raised
/// index [t + raise]^(0) the operator-center derivative reads stays inside
/// it. The seeds, the recurrence and the accumulation are that builder's,
/// unchanged; a zero raise reproduces its table exactly.
/// \param prim The primitive pair.
/// \param center The nucleus position C (Bohr).
/// \param charge The nuclear charge Z.
/// \param l1 The pair's total angular momentum la + lb.
/// \param raise The operator-center applications per axis.
/// \param pref The pair's 2pi/p * prefactor kernel factor.
/// \param g Out: [t]^(0) for every |t| <= l1 + |raise|.
/// \param scratch The call's scratch (its VRR buffers).
inline void BuildCenterTable(const MdPrimPair& prim,
                             const Eigen::Vector3d& center,
                             double charge,
                             int l1,
                             const std::array<int, 3>& raise,
                             double pref,
                             std::vector<double>& g,
                             MdDerivativeScratch& scratch) {
    const int order = l1 + raise[0] + raise[1] + raise[2];
    const int nHerm = Hermite3DCount(order);
    g.assign(static_cast<std::size_t>(nHerm), 0.0);
    scratch.slice.assign(static_cast<std::size_t>(nHerm), 0.0);
    scratch.prevSlice.assign(static_cast<std::size_t>(nHerm), 0.0);
    scratch.seeds.assign(static_cast<std::size_t>(order + 1), 0.0);

    const double dcx = prim.px - center.x();
    const double dcy = prim.py - center.y();
    const double dcz = prim.pz - center.z();
    const double x = prim.p * (dcx * dcx + dcy * dcy + dcz * dcz);
    double tmp[1 + kMaxBoysOrder];
    BoysAllOrders(order, x, tmp);

    for (int m = 0; m <= order; ++m)
    {
        scratch.seeds[static_cast<std::size_t>(m)] = charge * pref * tmp[m];
    }

    const double c2 = -2.0 * prim.p;
    const double c1x = -2.0 * prim.p * dcx;
    const double c1y = -2.0 * prim.p * dcy;
    const double c1z = -2.0 * prim.p * dcz;
    std::vector<double>& slice = scratch.slice;
    std::vector<double>& prevSlice = scratch.prevSlice;
    g[0] += scratch.seeds[0];

    for (int sliceT = 1; sliceT <= order; ++sliceT)
    {
        slice[0] = scratch.seeds[static_cast<std::size_t>(sliceT)];

        for (int n = 1; n <= sliceT; ++n)
        {
            const int off = kH2Prefix[n];
            const int off1 = kH2Prefix[n - 1];
            const int off2 = n >= 2 ? kH2Prefix[n - 2] : 0;

            for (int ty = 0; ty <= n; ++ty)
            {
                for (int tz = 0; tz <= n - ty; ++tz)
                {
                    const int tx = n - ty - tz;
                    const int sub = SubIndex3(ty, tz, n);
                    double value = 0.0;

                    if (tx >= 1)
                    {
                        value = c1x * slice[off1 + SubIndex3(ty, tz, n - 1)];

                        if (tx >= 2)
                        {
                            value += c2 * static_cast<double>(tx - 1) *
                                     prevSlice[off2 + SubIndex3(ty, tz, n - 2)];
                        }
                    } else if (ty >= 1)
                    {
                        value = c1y * slice[off1 + SubIndex3(ty - 1, tz, n - 1)];

                        if (ty >= 2)
                        {
                            value += c2 * static_cast<double>(ty - 1) *
                                     prevSlice[off2 + SubIndex3(ty - 2, tz, n - 2)];
                        }
                    } else
                    {
                        value = c1z * slice[off1 + SubIndex3(ty, tz - 1, n - 1)];

                        if (tz >= 2)
                        {
                            value += c2 * static_cast<double>(tz - 1) *
                                     prevSlice[off2 + SubIndex3(ty, tz - 2, n - 2)];
                        }
                    }

                    slice[off + sub] = value;
                }
            }
        }

        // Tier sliceT of slice sliceT holds the [t]^(0) targets.
        const int off = kH2Prefix[sliceT];

        for (int sub = 0; sub < Hermite2DCount(sliceT); ++sub)
        {
            g[static_cast<std::size_t>(off + sub)] += slice[static_cast<std::size_t>(off + sub)];
        }

        // The second VRR source [t-2e]^(m+1) lives in the previous slice
        // (md_vrr.hpp documents the same ping-pong).
        slice.swap(prevSlice);
    }
}

/// The kinetic-energy per-Cartesian-component expression of BuildKineticPair
/// (md_one_electron.hpp) at one bra/ket index pair: the same terms, in the
/// same order, reading the same tables. The two copies are the mirror the
/// finite-difference test detects a drift between.
/// \param tables The per-axis tables, built to cover the indices read here.
/// \param ca The bra Cartesian index.
/// \param cb The ket Cartesian index.
/// \param a The bra primitive exponent.
/// \returns The component's kinetic-energy value (before the folds).
inline double KineticComponentAt(const std::array<PerAxisETable, 3>& tables,
                                 const CartIndex& ca,
                                 const CartIndex& cb,
                                 double a) {
    const PerAxisETable& ex = tables[0];
    const PerAxisETable& ey = tables[1];
    const PerAxisETable& ez = tables[2];
    const double e0 = ex.At(ca.ix, cb.ix, 0) * ey.At(ca.iy, cb.iy, 0) * ez.At(ca.iz, cb.iz, 0);
    const double e2x = ex.At(ca.ix + 2, cb.ix, 0) * ey.At(ca.iy, cb.iy, 0) * ez.At(ca.iz, cb.iz, 0);
    const double e2y = ex.At(ca.ix, cb.ix, 0) * ey.At(ca.iy + 2, cb.iy, 0) * ez.At(ca.iz, cb.iz, 0);
    const double e2z = ex.At(ca.ix, cb.ix, 0) * ey.At(ca.iy, cb.iy, 0) * ez.At(ca.iz + 2, cb.iz, 0);
    const double e2mx =
        ca.ix >= 2 ? ex.At(ca.ix - 2, cb.ix, 0) * ey.At(ca.iy, cb.iy, 0) * ez.At(ca.iz, cb.iz, 0)
                   : 0.0;
    const double e2my =
        ca.iy >= 2 ? ex.At(ca.ix, cb.ix, 0) * ey.At(ca.iy - 2, cb.iy, 0) * ez.At(ca.iz, cb.iz, 0)
                   : 0.0;
    const double e2mz =
        ca.iz >= 2 ? ex.At(ca.ix, cb.ix, 0) * ey.At(ca.iy, cb.iy, 0) * ez.At(ca.iz - 2, cb.iz, 0)
                   : 0.0;

    return (3.0 + 2.0 * static_cast<double>(ca.ix + ca.iy + ca.iz)) * a * e0 -
           2.0 * a * a * (e2x + e2y + e2z) -
           0.5 * (static_cast<double>(ca.ix * (ca.ix - 1)) * e2mx +
                  static_cast<double>(ca.iy * (ca.iy - 1)) * e2my +
                  static_cast<double>(ca.iz * (ca.iz - 1)) * e2mz);
}

/// One kinetic-energy derivative term: the operator recursion of the header
/// comment, over the tuple's slots in order. The lower branch of an
/// application whose index is 0 is dropped - its coefficient is 0 and the
/// lowered function with a negative power is identically zero, so the two
/// agree.
/// \param tables The tables built to the tuple's extents.
/// \param a The bra primitive exponent.
/// \param beta The ket primitive exponent (p - a).
/// \param tuple The tuple's coordinates.
/// \param braAtom The atom carrying the bra shell.
/// \param ketAtom The atom carrying the ket shell.
/// \param slot The slot to apply next (the outermost at 0).
/// \param ca The bra Cartesian index.
/// \param cb The ket Cartesian index.
/// \returns The term's value (before the folds).
inline double KineticTermComponent(const std::array<PerAxisETable, 3>& tables,
                                   double a,
                                   double beta,
                                   std::span<const MdDerivativeCoordinate> tuple,
                                   std::size_t braAtom,
                                   std::size_t ketAtom,
                                   std::size_t slot,
                                   CartIndex ca,
                                   CartIndex cb) {
    if (slot == tuple.size())
    {
        return KineticComponentAt(tables, ca, cb, a);
    }

    const MdDerivativeCoordinate coordinate = tuple[slot];
    double value = 0.0;

    if (coordinate.atom == braAtom)
    {
        const int index = CartesianPower(ca, coordinate.axis);
        value += 2.0 * a *
                 KineticTermComponent(tables,
                                      a,
                                      beta,
                                      tuple,
                                      braAtom,
                                      ketAtom,
                                      slot + 1,
                                      ShiftCartIndex(ca, coordinate.axis, 1),
                                      cb);

        if (index >= 1)
        {
            value -= static_cast<double>(index) *
                     KineticTermComponent(tables,
                                          a,
                                          beta,
                                          tuple,
                                          braAtom,
                                          ketAtom,
                                          slot + 1,
                                          ShiftCartIndex(ca, coordinate.axis, -1),
                                          cb);
        }
    }

    if (coordinate.atom == ketAtom)
    {
        const int index = CartesianPower(cb, coordinate.axis);
        value += 2.0 * beta *
                 KineticTermComponent(tables,
                                      a,
                                      beta,
                                      tuple,
                                      braAtom,
                                      ketAtom,
                                      slot + 1,
                                      ca,
                                      ShiftCartIndex(cb, coordinate.axis, 1));

        if (index >= 1)
        {
            value -= static_cast<double>(index) *
                     KineticTermComponent(tables,
                                          a,
                                          beta,
                                          tuple,
                                          braAtom,
                                          ketAtom,
                                          slot + 1,
                                          ca,
                                          ShiftCartIndex(cb, coordinate.axis, -1));
        }
    }

    return value;
}

/// The overlap block's derivative of one canonical pair: one tuple's
/// boundary is one (nFuncsA x nFuncsB) row-major block at
/// tuple * pair.nFuncs, in request order.
/// \param pair The pair.
/// \param braAtom The atom carrying the bra shell.
/// \param ketAtom The atom carrying the ket shell.
/// \param order The derivative order, 1..kMaxMdDerivativeOrder.
/// \param coordinates The flat coordinates: order values per tuple, each
/// 3 * atom + axis with axis 0..2.
/// \param blocks Out: MdDerivativeBlockCount(pair, order, count) elements.
/// \param scratch The call's scratch.
/// \returns Nothing, or an Error (kInvalidArgument) naming the bound or the
/// coordinate the path cannot differentiate.
inline qcx::Result<void> BuildOverlapPairDerivative(const MdPairData& pair,
                                                    std::size_t braAtom,
                                                    std::size_t ketAtom,
                                                    int order,
                                                    std::span<const std::size_t> coordinates,
                                                    std::span<double> blocks,
                                                    MdDerivativeScratch& scratch) {
    const auto checked = CheckDerivativeRequest(pair, order, coordinates.size(), blocks.size());

    if (!checked.has_value())
    {
        return std::unexpected(checked.error());
    }

    const std::size_t tupleCount = *checked;
    std::fill(blocks.begin(), blocks.end(), 0.0);
    const int nHerm = Hermite3DCount(pair.la + pair.lb);
    // The overlap reads the Hermite E at t = 0 alone: the selector kernel.
    scratch.hermiteSelector.assign(static_cast<std::size_t>(nHerm), 0.0);
    scratch.hermiteSelector[0] = 1.0;

    for (std::size_t tupleIndex = 0; tupleIndex < tupleCount; ++tupleIndex)
    {
        const std::span<const std::size_t> flat = coordinates.subspan(
            tupleIndex * static_cast<std::size_t>(order), static_cast<std::size_t>(order));
        const std::array<MdDerivativeCoordinate, kMaxMdDerivativeOrder> tuple =
            DecodeTuple(flat, order);
        const std::span<const MdDerivativeCoordinate> view(tuple.data(),
                                                           static_cast<std::size_t>(order));
        const auto terms = DerivativeTermsOf(view, braAtom, ketAtom, false);

        if (!terms.has_value())
        {
            return std::unexpected(terms.error());
        }

        double* block = blocks.data() + tupleIndex * pair.nFuncs;

        for (std::size_t primIdx = 0; primIdx < pair.primPairs.size(); ++primIdx)
        {
            const MdPrimPair& prim = pair.primPairs[primIdx];
            const double pref = std::pow(std::numbers::pi / prim.p, 1.5) * prim.prefactor;

            for (int termIndex = 0; termIndex < terms->count; ++termIndex)
            {
                BuildDerivedPairTables(
                    pair, prim, terms->terms[static_cast<std::size_t>(termIndex)].raises, scratch);
                FoldPairETable(pair.la,
                               pair.lb,
                               0,
                               scratch.derivedTables,
                               pair.isSphericalA,
                               pair.isSphericalB,
                               scratch.folded);
                AccumulateFoldContraction(
                    pair, primIdx, scratch.folded, scratch.hermiteSelector, pref, block);
            }
        }
    }

    return {};
}

/// The kinetic-energy block's derivative of one canonical pair: one tuple's
/// boundary is one (nFuncsA x nFuncsB) row-major block at
/// tuple * pair.nFuncs, in request order.
/// \param pair The pair.
/// \param braAtom The atom carrying the bra shell.
/// \param ketAtom The atom carrying the ket shell.
/// \param order The derivative order, 1..kMaxMdDerivativeOrder.
/// \param coordinates The flat coordinates: order values per tuple, each
/// 3 * atom + axis with axis 0..2.
/// \param blocks Out: MdDerivativeBlockCount(pair, order, count) elements.
/// \param scratch The call's scratch.
/// \returns Nothing, or an Error (kInvalidArgument) naming the bound or the
/// coordinate the path cannot differentiate.
inline qcx::Result<void> BuildKineticPairDerivative(const MdPairData& pair,
                                                    std::size_t braAtom,
                                                    std::size_t ketAtom,
                                                    int order,
                                                    std::span<const std::size_t> coordinates,
                                                    std::span<double> blocks,
                                                    MdDerivativeScratch& scratch) {
    const auto checked = CheckDerivativeRequest(pair, order, coordinates.size(), blocks.size());

    if (!checked.has_value())
    {
        return std::unexpected(checked.error());
    }

    const std::size_t tupleCount = *checked;
    std::fill(blocks.begin(), blocks.end(), 0.0);
    const int nCartA = CartesianCount(pair.la);
    const int nCartB = CartesianCount(pair.lb);
    const std::size_t nAngA = pair.nFuncsA / pair.rowsA;
    const std::size_t nAngB = pair.nFuncsB / pair.rowsB;

    for (std::size_t tupleIndex = 0; tupleIndex < tupleCount; ++tupleIndex)
    {
        const std::span<const std::size_t> flat = coordinates.subspan(
            tupleIndex * static_cast<std::size_t>(order), static_cast<std::size_t>(order));
        const std::array<MdDerivativeCoordinate, kMaxMdDerivativeOrder> tuple =
            DecodeTuple(flat, order);
        const std::span<const MdDerivativeCoordinate> view(tuple.data(),
                                                           static_cast<std::size_t>(order));
        const auto terms = DerivativeTermsOf(view, braAtom, ketAtom, false);

        if (!terms.has_value())
        {
            return std::unexpected(terms.error());
        }

        double* block = blocks.data() + tupleIndex * pair.nFuncs;
        // The tuple's raises bound the indices the recursion reaches: the
        // bra side adds its own raises plus the two rows of the |r-A|^2
        // terms, the ket side its own.
        std::array<int, 3> braRaises{};
        std::array<int, 3> ketRaises{};

        for (int slot = 0; slot < order; ++slot)
        {
            const MdDerivativeCoordinate coordinate = tuple[static_cast<std::size_t>(slot)];

            if (coordinate.atom == braAtom)
            {
                ++braRaises[static_cast<std::size_t>(coordinate.axis)];
            }

            if (coordinate.atom == ketAtom)
            {
                ++ketRaises[static_cast<std::size_t>(coordinate.axis)];
            }
        }

        for (std::size_t primIdx = 0; primIdx < pair.primPairs.size(); ++primIdx)
        {
            const MdPrimPair& prim = pair.primPairs[primIdx];
            const double pref = std::pow(std::numbers::pi / prim.p, 1.5) * prim.prefactor;
            const double a = prim.exponentA;
            const double beta = prim.p - a;
            const std::array<double, 3> shiftA = {
                prim.px - pair.ax, prim.py - pair.ay, prim.pz - pair.az};
            const std::array<double, 3> shiftB = {
                prim.px - pair.bx, prim.py - pair.by, prim.pz - pair.bz};

            for (int axis = 0; axis < 3; ++axis)
            {
                const std::size_t slot = static_cast<std::size_t>(axis);
                const int rows = pair.la + braRaises[slot] + 2;
                const int cols = pair.lb + ketRaises[slot];
                scratch.derivedTables[slot].Reset(rows, cols);
                FillPerAxisTable(
                    rows, cols, shiftA, shiftB, prim.p, axis, scratch.derivedTables[slot]);
            }

            // Per cartesian bra row (anga), the ket-folded kinetic terms -
            // the tail of BuildKineticPair, over the recursion's value.
            scratch.perCart.assign(static_cast<std::size_t>(nCartA) * nAngB, 0.0);

            for (int anga = 0; anga < nCartA; ++anga)
            {
                const CartIndex ca = kCartesianIndices[pair.la][anga];

                for (int angb = 0; angb < nCartB; ++angb)
                {
                    const CartIndex cb = kCartesianIndices[pair.lb][angb];
                    const double value = KineticTermComponent(
                        scratch.derivedTables, a, beta, view, braAtom, ketAtom, 0, ca, cb);

                    for (int fb = 0; fb < static_cast<int>(nAngB); ++fb)
                    {
                        const double gb = pair.isSphericalB ? kSolidHarmonicG[pair.lb][fb][angb]
                                                            : (fb == angb ? 1.0 : 0.0);

                        if (gb != 0.0)
                        {
                            scratch.perCart[static_cast<std::size_t>(anga) * nAngB +
                                            static_cast<std::size_t>(fb)] += gb * value;
                        }
                    }
                }
            }

            for (std::size_t rowA = 0; rowA < pair.rowsA; ++rowA)
            {
                for (std::size_t rowB = 0; rowB < pair.rowsB; ++rowB)
                {
                    const double weight = pair.braWeights[primIdx][rowA * pair.rowsB + rowB];

                    for (std::size_t fb = 0; fb < nAngB; ++fb)
                    {
                        for (std::size_t fa = 0; fa < nAngA; ++fa)
                        {
                            double value = 0.0;

                            if (pair.isSphericalA)
                            {
                                for (int anga = 0; anga < nCartA; ++anga)
                                {
                                    value +=
                                        kSolidHarmonicG[pair.la][fa][anga] *
                                        scratch
                                            .perCart[static_cast<std::size_t>(anga) * nAngB + fb];
                                }
                            } else
                            {
                                value = scratch.perCart[fa * nAngB + fb];
                            }

                            block[(rowA * nAngA + fa) * pair.nFuncsB + rowB * nAngB + fb] +=
                                weight * pref * value;
                        }
                    }
                }
            }
        }
    }

    return {};
}

/// The tiers one derivative term's raised coefficients reach: the term's
/// total raise count, the amount the fold and the kernel of its contraction
/// run past la + lb. A differentiated coefficient D[E]^{(i,j)}_t =
/// 2a E^{(i+1,j)}_t - i E^{(i-1,j)}_t reaches tier i + j + 1 per raise, and
/// the fold contracts that whole tier range against the kernel, so a term
/// without raises reads only the tiers the plain pair does.
inline int TermTierRaises(const MdDerivativeRaises& raises) {
    return raises.bra[0] + raises.bra[1] + raises.bra[2] + raises.ket[0] + raises.ket[1] +
           raises.ket[2];
}

/// The nuclear-attraction block's derivative of one canonical pair: one
/// tuple's boundary is one (nFuncsA x nFuncsB) row-major block at
/// tuple * pair.nFuncs, in request order. A coordinate may name any atom of
/// the molecule: a nucleus that carries neither shell enters through the
/// operator center, and a coordinate on a shell's own atom contributes both
/// its operators.
/// \param pair The pair.
/// \param braAtom The atom carrying the bra shell.
/// \param ketAtom The atom carrying the ket shell.
/// \param charges Nuclear charges Z (one per atom).
/// \param centers Atom centers (Bohr).
/// \param order The derivative order, 1..kMaxMdDerivativeOrder.
/// \param coordinates The flat coordinates: order values per tuple, each
/// 3 * atom + axis with axis 0..2.
/// \param blocks Out: MdDerivativeBlockCount(pair, order, count) elements.
/// \param scratch The call's scratch.
/// \returns Nothing, or an Error (kInvalidArgument) naming the bound or the
/// coordinate the path cannot differentiate.
inline qcx::Result<void> BuildNuclearPairDerivative(const MdPairData& pair,
                                                    std::size_t braAtom,
                                                    std::size_t ketAtom,
                                                    const std::vector<double>& charges,
                                                    const std::vector<Eigen::Vector3d>& centers,
                                                    int order,
                                                    std::span<const std::size_t> coordinates,
                                                    std::span<double> blocks,
                                                    MdDerivativeScratch& scratch) {
    const auto checked = CheckDerivativeRequest(pair, order, coordinates.size(), blocks.size());

    if (!checked.has_value())
    {
        return std::unexpected(checked.error());
    }

    const std::size_t tupleCount = *checked;
    std::fill(blocks.begin(), blocks.end(), 0.0);
    const int l1 = pair.la + pair.lb;

    for (std::size_t tupleIndex = 0; tupleIndex < tupleCount; ++tupleIndex)
    {
        const std::span<const std::size_t> flat = coordinates.subspan(
            tupleIndex * static_cast<std::size_t>(order), static_cast<std::size_t>(order));
        const std::array<MdDerivativeCoordinate, kMaxMdDerivativeOrder> tuple =
            DecodeTuple(flat, order);
        const std::span<const MdDerivativeCoordinate> view(tuple.data(),
                                                           static_cast<std::size_t>(order));

        for (int slot = 0; slot < order; ++slot)
        {
            const std::size_t atom = tuple[static_cast<std::size_t>(slot)].atom;

            if (atom >= charges.size())
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "integrals: derivative tuple slot " + std::to_string(slot) +
                                   " names atom " + std::to_string(atom) + ", outside the " +
                                   std::to_string(charges.size()) + " nuclei the operator sums"});
            }
        }

        const auto terms = DerivativeTermsOf(view, braAtom, ketAtom, true);

        if (!terms.has_value())
        {
            return std::unexpected(terms.error());
        }

        double* block = blocks.data() + tupleIndex * pair.nFuncs;

        for (std::size_t primIdx = 0; primIdx < pair.primPairs.size(); ++primIdx)
        {
            const MdPrimPair& prim = pair.primPairs[primIdx];
            const double pref = 2.0 * std::numbers::pi / prim.p * prim.prefactor;
            bool hasPlainTerm = false;
            int plainTierRaises = 0;

            for (int termIndex = 0; termIndex < terms->count; ++termIndex)
            {
                const MdDerivativeTerm& term = terms->terms[static_cast<std::size_t>(termIndex)];

                if (term.operatorAtom == kNoDerivativeAtom)
                {
                    hasPlainTerm = true;
                    plainTierRaises = std::max(plainTierRaises, TermTierRaises(term.raises));
                }
            }

            if (hasPlainTerm)
            {
                // The charge-summed table the plain terms contract against -
                // BuildNuclearPair's own accumulation, reproduced term for
                // term, run to the raised order a differentiated plain term
                // contracts over.
                const int plainHerm = Hermite3DCount(l1 + plainTierRaises);
                scratch.centerTable.assign(static_cast<std::size_t>(plainHerm), 0.0);

                for (std::size_t c = 0; c < charges.size(); ++c)
                {
                    BuildCenterTable(prim,
                                     centers[c],
                                     charges[c],
                                     l1 + plainTierRaises,
                                     std::array<int, 3>{},
                                     pref,
                                     scratch.centerRead,
                                     scratch);

                    for (int t = 0; t < plainHerm; ++t)
                    {
                        scratch.centerTable[static_cast<std::size_t>(t)] +=
                            scratch.centerRead[static_cast<std::size_t>(t)];
                    }
                }
            }

            for (int termIndex = 0; termIndex < terms->count; ++termIndex)
            {
                const MdDerivativeTerm& term = terms->terms[static_cast<std::size_t>(termIndex)];
                const int operators = term.raises.op[0] + term.raises.op[1] + term.raises.op[2];
                // The tiers this term reaches: its differentiated coefficients
                // run one tier past la + lb per raise, and the fold, the
                // contraction and the kernel all have to cover them.
                const int tierRaises = TermTierRaises(term.raises);
                const int tierMax = l1 + tierRaises;
                const int nHerm = Hermite3DCount(tierMax);
                BuildDerivedPairTables(pair, prim, term.raises, scratch);
                FoldPairETable(pair.la,
                               pair.lb,
                               tierRaises,
                               scratch.derivedTables,
                               pair.isSphericalA,
                               pair.isSphericalB,
                               scratch.folded);

                if (operators == 0)
                {
                    AccumulateFoldContraction(
                        pair,
                        primIdx,
                        scratch.folded,
                        std::span<const double>(scratch.centerTable.data(),
                                                static_cast<std::size_t>(nHerm)),
                        -1.0,
                        block);
                    continue;
                }

                // The raised table of the one nucleus the term's operator
                // choices name, read at the raised index: g[t + n] for
                // |t| <= l1 + raises. Its sign is (-1)^(operators+1): the
                // block accumulates -weight * sum_t folded * g
                // (BuildNuclearPair), and each application of d/dC_k
                // multiplies its nucleus' table by -1.
                BuildCenterTable(prim,
                                 centers[term.operatorAtom],
                                 charges[term.operatorAtom],
                                 tierMax,
                                 term.raises.op,
                                 pref,
                                 scratch.centerRead,
                                 scratch);
                scratch.centerGather.assign(static_cast<std::size_t>(nHerm), 0.0);

                for (int tx = 0; tx <= tierMax; ++tx)
                {
                    for (int ty = 0; ty <= tierMax - tx; ++ty)
                    {
                        for (int tz = 0; tz <= tierMax - tx - ty; ++tz)
                        {
                            scratch.centerGather[static_cast<std::size_t>(
                                Hermite3DIndex(tx, ty, tz))] =
                                scratch.centerRead[static_cast<std::size_t>(
                                    Hermite3DIndex(tx + term.raises.op[0],
                                                   ty + term.raises.op[1],
                                                   tz + term.raises.op[2]))];
                        }
                    }
                }

                AccumulateFoldContraction(pair,
                                          primIdx,
                                          scratch.folded,
                                          std::span<const double>(scratch.centerGather.data(),
                                                                  static_cast<std::size_t>(nHerm)),
                                          operators % 2 == 0 ? -1.0 : 1.0,
                                          block);
            }
        }
    }

    return {};
}

} // namespace qcx::integrals::internal
