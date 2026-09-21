#pragma once

// The derivative tier of the two-electron MD pipeline: the analytic
// nuclear-coordinate derivative of one shell quartet's packed block, through
// the same contraction the value pipeline performs.
//
// One primitive quadruple's block element is
//   (ab|cd) = sum_{t,u} E_t^(ab) E_u^(cd) [t+u]^(0),
// with E^(ab) the Hermite expansion coefficients of the pair product about
// its Gaussian centre P, [r]^(0) = (d/dP)^r [0]^(0) the pair-pair kernel, and
// the ket coefficients signed by (-1)^|u| - the P-derivative convention for
// an expansion about Q (md_batch.cpp). Differentiating the expansion
//   phi_a phi_b = sum_t E_t Lambda_t(r - P),   Lambda_t = (d/dP)^t Lambda_0
// with respect to A_x, whose two effects are the coefficient's own derivative
// and the product centre's (dP_x/dA_x = a/p), and equating the coefficients
// of the independent Lambda_t against
//   d/dA_x (phi_a phi_b) = (-a_x phi_(a-e_x) + 2a phi_(a+e_x)) phi_b
// gives - with the coefficient identity of md_hermite.hpp -
//   d E_t / dA_x = -a_x E_t^(a-e_x,b) + 2a E_t^(a+e_x,b) - (a/p) E_(t-e_x).
//
// Inside the contraction the last term and the kernel's own
//   d [t+u]^(0) / dA_x = (a/p) [t+u+e_x]^(0)
// are the same operation, since sum_t E_(t-e_x) K_t = sum_t E_t K_(t+e_x):
// the pair centre's translation and the coefficient's index shift cancel
// exactly, on both pairs. What survives is two operators per centre - the
// raised coefficient 2 e_self and the lowered -i_x - one of them per centre
// of the four. The pair prefactor's derivative is inside them too: 2a E_0^(a+e,b)
// IS d/dA_x of the (a, b) expansion's constant term exp(-ab/p |A-B|^2).
//
// A coordinate on an atom carrying several shells of the quartet offers every
// one of them, which is how a same-centre quartet differentiates; a
// coordinate on an atom carrying none of them leaves the block exactly zero.
// [McMurchie1978]

#include "md_batch.hpp"
#include "md_boys.hpp"
#include "md_defs.hpp"
#include "md_hermite.hpp"
#include "md_tables_gen.hpp"
#include "qcx/error.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <string>
#include <vector>

namespace qcx::integrals::internal {

/// The elementary operators one quartet shell offers a derivative slot: the
/// raised coefficient and the lowered coefficient. Every other term of the
/// differentiated expansion cancels inside the contraction.
inline constexpr int kEriOperatorKinds = 2;

/// The most operators one slot carries: each of the quartet's four shells may
/// sit on the coordinate's atom.
inline constexpr int kMaxEriSlotOperators = 4 * kEriOperatorKinds;

/// One elementary operator of a derivative term: what one nuclear coordinate
/// does to the primitive quartet it is a slot of.
enum class EriOperatorKind {
    kRaise, ///< 2 e_self times the coefficient raised one index along the axis.
    kLower, ///< -i_x times the coefficient lowered one index along the axis.
};

/// One elementary operator, with the shell it acts on.
struct EriOperator {
    EriOperatorKind kind = EriOperatorKind::kRaise; ///< What it does.
    int side = 0; ///< 0 = the bra pair (a, b), 1 = the ket pair (c, d).
    int sub = 0; ///< 0 = the pair's first shell, 1 = its second.
    int axis = 0; ///< The axis: 0 = x, 1 = y, 2 = z.
    double weight = 0.0; ///< The operator's scalar factor (2 e_self for a raise).
};

/// The elementary operators one nuclear coordinate offers, one group per
/// quartet shell sitting on the coordinate's atom.
struct EriOperatorList {
    std::array<EriOperator, kMaxEriSlotOperators> operators{}; ///< The operators.
    int count = 0; ///< Operators held.
};

/// One shell quartet of the two-electron derivative tier: the four shells in
/// the engine's canonical (i, j, k, l) order - la on centre A and lb on B,
/// lc on C and ld on D - and the atom each shell sits on. The block layout is
/// the packed quartet layout of eri_batch.hpp.
struct MdEriDerivativeQuartet {
    std::array<const MdShellInput*, 4> shells{}; ///< The shells, i, j, k, l.
    std::array<std::size_t, 4> atoms{}; ///< The atom each shell sits on.
};

/// One primitive quadruple of a quartet: the four exponents, the four centres
/// and the four primitives' row indices into their shells.
struct MdEriPrimitive {
    std::array<double, 4> exponents{}; ///< The primitive exponents.
    std::array<std::array<double, 3>, 4> centers{}; ///< The shell centres (Bohr).
    std::array<std::size_t, 4> indices{}; ///< The primitives' positions in their shells.
};

/// The working storage of one derivative call, held by the caller across the
/// tuples and the primitive quadruples so a call allocates once. Every member
/// is reused at the size it last needed.
struct MdEriDerivativeScratch {
    std::vector<double> kernel; ///< The K-free pair-pair kernel, by Hermite3DIndex.
    std::vector<double> sliceA; ///< The kernel recurrence's slice buffers.
    std::vector<double> sliceB;
    std::vector<double> seeds; ///< The Boys seeds of one primitive quadruple.
    std::array<PerAxisETable, 3> bra; ///< The bra pair's base E tables.
    std::array<PerAxisETable, 3> ket; ///< The ket pair's base E tables.
    std::array<PerAxisETable, 3> braWork; ///< The term's bra tables.
    std::array<PerAxisETable, 3> ketWork; ///< The term's ket tables.
    std::array<PerAxisETable, 3> braCopy; ///< The pre-update copies of the term's bra tables.
    std::array<PerAxisETable, 3> ketCopy; ///< The pre-update copies of the term's ket tables.
    std::vector<double> braFolded; ///< The bra pair's folded coefficient slice.
    std::vector<double> ketFolded; ///< The ket pair's folded coefficient slice.
};

/// The function count of one shell of a quartet.
/// \param shell The shell.
/// \returns The contraction rows times its angular components.
inline std::size_t EriShellFunctions(const MdShellInput& shell) noexcept {
    const MdShellContractions& contraction = shell.contractions;
    const int components = contraction.isSpherical ? SphericalCount(contraction.angularMomentum)
                                                   : CartesianCount(contraction.angularMomentum);
    return contraction.rows * static_cast<std::size_t>(components);
}

/// The element count of one quartet's packed block.
/// \param quartet The quartet.
/// \returns n_i n_j n_k n_l.
inline std::size_t EriQuartetBlockElements(const MdEriDerivativeQuartet& quartet) noexcept {
    return EriShellFunctions(*quartet.shells[0]) * EriShellFunctions(*quartet.shells[1]) *
           EriShellFunctions(*quartet.shells[2]) * EriShellFunctions(*quartet.shells[3]);
}

/// The element count of one derivative request: one packed block per tuple.
/// The caller sizes its output span with this before calling.
/// \param quartet The quartet the request addresses.
/// \param order The derivative order.
/// \param coordinateCount The flat coordinate count (order per tuple).
/// \returns The element count, or 0 for a request CheckEriDerivativeRequest
/// refuses - the two agree, so a caller sizing its span here never reads past
/// what the call writes.
inline std::size_t MdEriDerivativeBlockCount(const MdEriDerivativeQuartet& quartet,
                                             int order,
                                             std::size_t coordinateCount) noexcept {
    if (order < 1 || order > kMaxMdDerivativeOrder ||
        coordinateCount % static_cast<std::size_t>(order) != 0)
    {
        return 0;
    }

    return (coordinateCount / static_cast<std::size_t>(order)) * EriQuartetBlockElements(quartet);
}

/// Validates one derivative request's shape: the order is within the band,
/// the coordinate list divides into whole tuples, the block span holds one
/// packed block per tuple, every atom is one the caller's geometry has, and
/// the Hermite extent the request reaches stays inside the generated tables.
/// \param quartet The quartet the request addresses.
/// \param order The derivative order.
/// \param coordinateCount The flat coordinate count.
/// \param blockCount The element count of the caller's output span.
/// \param atomCount The atom count of the caller's geometry.
/// \returns The tuple count, or an Error (kInvalidArgument) naming the bound
/// that failed, or kUnimplemented for a quartet whose kernel table the
/// generated tier layout does not reach.
inline qcx::Result<std::size_t> CheckEriDerivativeRequest(const MdEriDerivativeQuartet& quartet,
                                                          int order,
                                                          std::size_t coordinateCount,
                                                          std::size_t blockCount,
                                                          std::size_t atomCount) {
    if (order < 1 || order > kMaxMdDerivativeOrder)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "integrals: two-electron derivative order " +
                                              std::to_string(order) + " is outside 1.." +
                                              std::to_string(kMaxMdDerivativeOrder)});
    }

    if (coordinateCount % static_cast<std::size_t>(order) != 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "integrals: " + std::to_string(coordinateCount) +
                           " derivative coordinates do not divide into tuples of order " +
                           std::to_string(order)});
    }

    const std::size_t tuples = coordinateCount / static_cast<std::size_t>(order);
    const std::size_t block = EriQuartetBlockElements(quartet);

    if (blockCount != tuples * block)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "integrals: the derivative block span holds " +
                                              std::to_string(blockCount) + " elements, not the " +
                                              std::to_string(tuples * block) + " of " +
                                              std::to_string(tuples) + " tuples"});
    }

    int totalL = 0;

    for (int shell = 0; shell < 4; ++shell)
    {
        const std::size_t atom = quartet.atoms[static_cast<std::size_t>(shell)];

        if (atom >= atomCount)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "integrals: derivative quartet names atom " +
                                                  std::to_string(atom) + ", outside the " +
                                                  std::to_string(atomCount) + " of the geometry"});
        }

        totalL += quartet.shells[static_cast<std::size_t>(shell)]->contractions.angularMomentum;
    }

    const int hermiteOrder = totalL + 2 * order;

    if (hermiteOrder > 2 * kMaxShellL)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kUnimplemented,
            "integrals: a total angular momentum of " + std::to_string(totalL) +
                " at derivative order " + std::to_string(order) + " reaches Hermite order " +
                std::to_string(hermiteOrder) + ", past the " + std::to_string(2 * kMaxShellL) +
                " the kernel table covers"});
    }

    return tuples;
}

/// Signs one folded coefficient slice with the ket pair's P-derivative
/// convention: (-1)^|u|, with |u| the 3D-Hermite tier of the index. The slice
/// is tier-major (md_defs.hpp), so the sign flips at every tier boundary.
/// \param index The 3D-Hermite index.
/// \returns +1 or -1.
inline double EriKetSign(int index) noexcept {
    int tier = 0;

    // The closed form, not the generated prefix table: the derivative tier's
    // extents run past the table's 2*kMaxShellL range on the classes it
    // admits, and a clamped walk would sign the wrong tiers.
    while (Hermite3DCount(tier) <= index)
    {
        ++tier;
    }

    return tier % 2 == 0 ? 1.0 : -1.0;
}

/// Fills the pair-pair kernel [r]^(0) for every |r| <= \p lMax: the Boys seeds
/// scaled by the primitive prefactor, lifted by the slice recurrence of the
/// value pipeline to tier |r| of slice |r| - the same recurrence and the same
/// tier layout, at the runtime extent this tier asks for.
/// \param lMax The highest Hermite order the call reads, <= 2*kMaxShellL.
/// \param mu The pair-pair exponent p q / (p + q).
/// \param dP The vector P - Q (Bohr), per axis.
/// \param prefactor The primitive quadruple's kernel constant K.
/// \param out The kernel table, Hermite3DCount(lMax) entries.
/// \param sliceA The recurrence's two slice ping-pong buffers.
/// \param sliceB
/// \param seeds The Boys seed buffer, resized to lMax + 1.
inline void BuildEriKernel(int lMax,
                           double mu,
                           const std::array<double, 3>& dP,
                           double prefactor,
                           std::vector<double>& out,
                           std::vector<double>& sliceA,
                           std::vector<double>& sliceB,
                           std::vector<double>& seeds) {
    assert(lMax >= 0 && lMax <= 2 * kMaxShellL);
    const std::size_t cells = static_cast<std::size_t>(Hermite3DCount(lMax));
    seeds.assign(static_cast<std::size_t>(lMax) + 1, 0.0);
    MdBoysBatch<double>(lMax, mu * (dP[0] * dP[0] + dP[1] * dP[1] + dP[2] * dP[2]), seeds.data());

    for (int m = 0; m <= lMax; ++m)
    {
        seeds[static_cast<std::size_t>(m)] *= prefactor;
    }

    out.assign(cells, 0.0);
    sliceA.assign(cells, 0.0);
    sliceB.assign(cells, 0.0);
    out[0] = seeds[0];
    sliceA[0] = seeds[0];

    const double c2 = -2.0 * mu;
    const std::array<double, 3> c1 = {-2.0 * mu * dP[0], -2.0 * mu * dP[1], -2.0 * mu * dP[2]};

    for (int sliceT = 1; sliceT <= lMax; ++sliceT)
    {
        sliceA[0] = seeds[static_cast<std::size_t>(sliceT)];

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
                    double value = 0.0;

                    if (tx >= 1)
                    {
                        value = c1[0] *
                                sliceA[static_cast<std::size_t>(off1 + SubIndex3(ty, tz, n - 1))];

                        if (tx >= 2)
                        {
                            value +=
                                c2 * static_cast<double>(tx - 1) *
                                sliceB[static_cast<std::size_t>(off2 + SubIndex3(ty, tz, n - 2))];
                        }
                    } else if (ty >= 1)
                    {
                        value =
                            c1[1] *
                            sliceA[static_cast<std::size_t>(off1 + SubIndex3(ty - 1, tz, n - 1))];

                        if (ty >= 2)
                        {
                            value += c2 * static_cast<double>(ty - 1) *
                                     sliceB[static_cast<std::size_t>(off2 +
                                                                     SubIndex3(ty - 2, tz, n - 2))];
                        }
                    } else
                    {
                        value =
                            c1[2] *
                            sliceA[static_cast<std::size_t>(off1 + SubIndex3(ty, tz - 1, n - 1))];

                        if (tz >= 2)
                        {
                            value += c2 * static_cast<double>(tz - 1) *
                                     sliceB[static_cast<std::size_t>(off2 +
                                                                     SubIndex3(ty, tz - 2, n - 2))];
                        }
                    }

                    sliceA[static_cast<std::size_t>(off + SubIndex3(ty, tz, n))] = value;
                }
            }
        }

        // Tier sliceT of slice sliceT IS [r]^(0) for |r| = sliceT, at the flat
        // index of the 3D-Hermite layout - the same positions in the output.
        for (int index = kH2Prefix[sliceT]; index < Hermite3DCount(sliceT); ++index)
        {
            out[static_cast<std::size_t>(index)] = sliceA[static_cast<std::size_t>(index)];
        }

        sliceA.swap(sliceB);
    }
}

/// Fills one side's per-axis E tables of one primitive pair at the extent the
/// terms reach: the pair's own angular momenta raised by \p raises, which is
/// where a differentiated coefficient's index steps land. The recurrence
/// reads only cells with t <= i + j, so the raised extent holds the plain
/// extent's values at the plain extent's indices (md_hermite.hpp).
/// \param pairAngular The pair's two angular momenta.
/// \param shiftA The vector P - A (Bohr), per axis.
/// \param shiftB The vector P - B (Bohr), per axis.
/// \param p The pair exponent.
/// \param raises The extent added to each side.
/// \param tables The tables, re-fitted and filled.
inline void BuildEriPairTables(const std::array<int, 2>& pairAngular,
                               const std::array<double, 3>& shiftA,
                               const std::array<double, 3>& shiftB,
                               double p,
                               int raises,
                               std::array<PerAxisETable, 3>& tables) {
    for (int axis = 0; axis < 3; ++axis)
    {
        PerAxisETable& table = tables[static_cast<std::size_t>(axis)];
        table.Reset(pairAngular[0] + raises, pairAngular[1] + raises);
        FillPerAxisTable(
            pairAngular[0] + raises, pairAngular[1] + raises, shiftA, shiftB, p, axis, table);
    }
}

/// Builds one primitive quadruple's kernel table and base E tables, and
/// returns the quadruple's kernel constant K - the value pipeline's
/// 2 pi^(5/2) / (p q sqrt(p+q)) E_ab E_cd.
///
/// The kernel table holds the K-FREE [r]^(0): the Boys seeds carry no K, and
/// the constant multiplies the whole term instead (AccumulateEriTerm's
/// prefactor). Keeping it out of the table is what lets the pair prefactor's
/// own derivative - which is a derivative of K - be one of the term's
/// elementary operators.
/// \param quartet The quartet.
/// \param primitive The primitive quadruple.
/// \param order The derivative order, the extent the tables are raised by.
/// \param lMax The kernel's Hermite extent.
/// \param scratch The call's scratch.
/// \returns K.
inline double BuildEriPrimitive(const MdEriDerivativeQuartet& quartet,
                                const MdEriPrimitive& primitive,
                                int order,
                                int lMax,
                                MdEriDerivativeScratch& scratch) {
    const MdShellContractions& contractionA = quartet.shells[0]->contractions;
    const MdShellContractions& contractionB = quartet.shells[1]->contractions;
    const MdShellContractions& contractionC = quartet.shells[2]->contractions;
    const MdShellContractions& contractionD = quartet.shells[3]->contractions;
    const double a = primitive.exponents[0];
    const double b = primitive.exponents[1];
    const double c = primitive.exponents[2];
    const double d = primitive.exponents[3];
    const double p = a + b;
    const double q = c + d;
    std::array<double, 3> centerP{};
    std::array<double, 3> centerQ{};

    for (int axis = 0; axis < 3; ++axis)
    {
        const std::size_t index = static_cast<std::size_t>(axis);
        centerP[index] = (a * primitive.centers[0][index] + b * primitive.centers[1][index]) / p;
        centerQ[index] = (c * primitive.centers[2][index] + d * primitive.centers[3][index]) / q;
    }

    std::array<double, 3> shiftPa{};
    std::array<double, 3> shiftPb{};
    std::array<double, 3> shiftQc{};
    std::array<double, 3> shiftQd{};
    double abSquared = 0.0;
    double cdSquared = 0.0;

    for (int axis = 0; axis < 3; ++axis)
    {
        const std::size_t index = static_cast<std::size_t>(axis);
        shiftPa[index] = centerP[index] - primitive.centers[0][index];
        shiftPb[index] = centerP[index] - primitive.centers[1][index];
        shiftQc[index] = centerQ[index] - primitive.centers[2][index];
        shiftQd[index] = centerQ[index] - primitive.centers[3][index];
        abSquared += (primitive.centers[0][index] - primitive.centers[1][index]) *
                     (primitive.centers[0][index] - primitive.centers[1][index]);
        cdSquared += (primitive.centers[2][index] - primitive.centers[3][index]) *
                     (primitive.centers[2][index] - primitive.centers[3][index]);
    }

    const std::array<int, 2> braAngular = {contractionA.angularMomentum,
                                           contractionB.angularMomentum};
    const std::array<int, 2> ketAngular = {contractionC.angularMomentum,
                                           contractionD.angularMomentum};
    BuildEriPairTables(braAngular, shiftPa, shiftPb, p, order, scratch.bra);
    BuildEriPairTables(ketAngular, shiftQc, shiftQd, q, order, scratch.ket);
    const double mu = p * q / (p + q);
    const std::array<double, 3> delta = {
        centerP[0] - centerQ[0], centerP[1] - centerQ[1], centerP[2] - centerQ[2]};
    const double kernel = 2.0 * std::pow(std::numbers::pi, 2.5) / (p * q * std::sqrt(p + q)) *
                          std::exp(-(a * b / p) * abSquared) * std::exp(-(c * d / q) * cdSquared);
    BuildEriKernel(
        lMax, mu, delta, 1.0, scratch.kernel, scratch.sliceA, scratch.sliceB, scratch.seeds);
    return kernel;
}

/// The elementary operators one nuclear coordinate offers: every quartet shell
/// sitting on the coordinate's atom contributes its four, at the primitive
/// exponents of the quadruple being contracted.
/// \param quartet The quartet.
/// \param atom The coordinate's atom.
/// \param axis The coordinate's axis: 0 = x, 1 = y, 2 = z.
/// \param primitive The primitive quadruple.
/// \param out The coordinate's operators.
inline void EriOperatorsOf(const MdEriDerivativeQuartet& quartet,
                           std::size_t atom,
                           int axis,
                           const MdEriPrimitive& primitive,
                           EriOperatorList& out) {
    out.count = 0;

    for (int shell = 0; shell < 4; ++shell)
    {
        if (quartet.atoms[static_cast<std::size_t>(shell)] != atom)
        {
            continue;
        }

        const int side = shell < 2 ? 0 : 1;
        const int sub = shell % 2;
        const double self = primitive.exponents[static_cast<std::size_t>(shell)];
        const std::array<EriOperator, kEriOperatorKinds> kinds = {{
            EriOperator{EriOperatorKind::kRaise, side, sub, axis, 2.0 * self},
            EriOperator{EriOperatorKind::kLower, side, sub, axis, 1.0},
        }};

        for (const EriOperator& op : kinds)
        {
            out.operators[static_cast<std::size_t>(out.count)] = op;
            ++out.count;
        }
    }
}

/// Copies one axis table into the term's pre-update scratch, so an operator
/// that rewrites the table reads the values the previous operator left.
/// \param table The table to copy.
/// \param copy The copy, re-fitted to the table's extent.
inline void CopyEriAxisTable(const PerAxisETable& table, PerAxisETable& copy) {
    copy.Reset(table.BraExtent(), table.Lb());

    for (int i = 0; i <= table.BraExtent(); ++i)
    {
        for (int j = 0; j <= table.Lb(); ++j)
        {
            for (int t = 0; t <= table.TMax(); ++t)
            {
                copy.At(i, j, t) = table.At(i, j, t);
            }
        }
    }
}

/// Applies one raise or lower operator to one axis table in place, reading the
/// pre-update copy. The read one index past an edge is zero rather than out of
/// range: the raised coefficient of the pair's top row does not exist, and the
/// lowered coefficient carries the output index i_x, which vanishes at 0.
/// \param table The axis table to rewrite.
/// \param copy The pre-update copy of the same table.
/// \param op The operator (a raise or a lower).
inline void ApplyEriTableOperator(PerAxisETable& table,
                                  const PerAxisETable& copy,
                                  const EriOperator& op) {
    const int braExtent = table.BraExtent();
    const int ketExtent = table.Lb();
    const bool overBra = op.sub == 0;
    const int step = op.kind == EriOperatorKind::kRaise ? 1 : -1;

    for (int i = 0; i <= braExtent; ++i)
    {
        for (int j = 0; j <= ketExtent; ++j)
        {
            const int source = overBra ? i + step : j + step;

            if (source < 0 || source > (overBra ? braExtent : ketExtent))
            {
                for (int t = 0; t <= table.TMax(); ++t)
                {
                    table.At(i, j, t) = 0.0;
                }

                continue;
            }

            const double factor = op.kind == EriOperatorKind::kRaise
                                      ? op.weight
                                      : -op.weight * static_cast<double>(overBra ? i : j);

            for (int t = 0; t <= table.TMax(); ++t)
            {
                table.At(i, j, t) =
                    factor * (overBra ? copy.At(source, j, t) : copy.At(i, source, t));
            }
        }
    }
}

/// Applies one operator to the term's state: the axis table it acts on is
/// rewritten in place through the pre-update copy.
/// \param op The operator.
/// \param bra The bra pair's working tables.
/// \param ket The ket pair's working tables.
/// \param braCopy The bra pair's pre-update copies.
/// \param ketCopy The ket pair's pre-update copies.
inline void ApplyEriOperator(const EriOperator& op,
                             std::array<PerAxisETable, 3>& bra,
                             std::array<PerAxisETable, 3>& ket,
                             std::array<PerAxisETable, 3>& braCopy,
                             std::array<PerAxisETable, 3>& ketCopy) {
    const std::size_t axis = static_cast<std::size_t>(op.axis);
    std::array<PerAxisETable, 3>& tables = op.side == 0 ? bra : ket;
    std::array<PerAxisETable, 3>& copies = op.side == 0 ? braCopy : ketCopy;
    CopyEriAxisTable(tables[axis], copies[axis]);
    ApplyEriTableOperator(tables[axis], copies[axis], op);
}

/// Copies one side's base tables into the term's working tables.
/// \param from The base tables.
/// \param to The working tables.
inline void CopyEriPairTables(const std::array<PerAxisETable, 3>& from,
                              std::array<PerAxisETable, 3>& to) {
    for (int axis = 0; axis < 3; ++axis)
    {
        CopyEriAxisTable(from[static_cast<std::size_t>(axis)], to[static_cast<std::size_t>(axis)]);
    }
}

/// Accumulates one term of one tuple into one quartet block: the composed
/// state's folded coefficient slices contracted against the raised kernel,
/// then weighted by the contractions of the primitive quadruple.
/// \param quartet The quartet.
/// \param primitive The primitive quadruple being contracted.
/// \param scratch The call's scratch (the composed tables and the slices).
/// \param tierRaises The raise count the fold runs to.
/// \param prefactor The primitive quadruple's kernel constant K.
/// \param block The tuple's packed block.
inline void AccumulateEriTerm(const MdEriDerivativeQuartet& quartet,
                              const MdEriPrimitive& primitive,
                              MdEriDerivativeScratch& scratch,
                              int tierRaises,
                              double prefactor,
                              std::span<double> block) {
    const MdShellContractions& contractionA = quartet.shells[0]->contractions;
    const MdShellContractions& contractionB = quartet.shells[1]->contractions;
    const MdShellContractions& contractionC = quartet.shells[2]->contractions;
    const MdShellContractions& contractionD = quartet.shells[3]->contractions;
    const int la = contractionA.angularMomentum;
    const int lb = contractionB.angularMomentum;
    const int lc = contractionC.angularMomentum;
    const int ld = contractionD.angularMomentum;
    const std::size_t rowsA = contractionA.rows;
    const std::size_t rowsB = contractionB.rows;
    const std::size_t rowsC = contractionC.rows;
    const std::size_t rowsD = contractionD.rows;
    const int nAngA = contractionA.isSpherical ? SphericalCount(la) : CartesianCount(la);
    const int nAngB = contractionB.isSpherical ? SphericalCount(lb) : CartesianCount(lb);
    const int nAngC = contractionC.isSpherical ? SphericalCount(lc) : CartesianCount(lc);
    const int nAngD = contractionD.isSpherical ? SphericalCount(ld) : CartesianCount(ld);
    const int nHermBra = Hermite3DCount(la + lb + tierRaises);
    const int nHermKet = Hermite3DCount(lc + ld + tierRaises);
    const int tBound = la + lb + tierRaises;
    const int uBound = lc + ld + tierRaises;
    const std::size_t nFuncsA = rowsA * static_cast<std::size_t>(nAngA);
    const std::size_t nFuncsC = rowsC * static_cast<std::size_t>(nAngC);
    const std::size_t nFuncsD = rowsD * static_cast<std::size_t>(nAngD);

    FoldPairETable(la,
                   lb,
                   tierRaises,
                   scratch.braWork,
                   contractionA.isSpherical,
                   contractionB.isSpherical,
                   scratch.braFolded);
    FoldPairETable(lc,
                   ld,
                   tierRaises,
                   scratch.ketWork,
                   contractionC.isSpherical,
                   contractionD.isSpherical,
                   scratch.ketFolded);

    // The P-derivative convention of the ket pair's half of the contraction.
    for (int index = 0; index < nHermKet; ++index)
    {
        const double sign = EriKetSign(index);

        for (int component = 0; component < nAngC * nAngD; ++component)
        {
            scratch.ketFolded[static_cast<std::size_t>(component) * nHermKet +
                              static_cast<std::size_t>(index)] *= sign;
        }
    }

    const double weight = prefactor;

    for (int fa = 0; fa < nAngA; ++fa)
    {
        for (int fb = 0; fb < nAngB; ++fb)
        {
            const std::size_t braBase =
                (static_cast<std::size_t>(fa) * nAngB + fb) * static_cast<std::size_t>(nHermBra);

            for (int fc = 0; fc < nAngC; ++fc)
            {
                for (int fd = 0; fd < nAngD; ++fd)
                {
                    const std::size_t ketBase = (static_cast<std::size_t>(fc) * nAngD + fd) *
                                                static_cast<std::size_t>(nHermKet);
                    double value = 0.0;

                    for (int tx = 0; tx <= tBound; ++tx)
                    {
                        for (int ty = 0; ty <= tBound - tx; ++ty)
                        {
                            for (int tz = 0; tz <= tBound - tx - ty; ++tz)
                            {
                                const double braValue =
                                    scratch.braFolded[braBase + static_cast<std::size_t>(
                                                                    Hermite3DIndex(tx, ty, tz))];

                                if (braValue == 0.0)
                                {
                                    continue;
                                }

                                for (int ux = 0; ux <= uBound; ++ux)
                                {
                                    for (int uy = 0; uy <= uBound - ux; ++uy)
                                    {
                                        for (int uz = 0; uz <= uBound - ux - uy; ++uz)
                                        {
                                            const double ketValue =
                                                scratch.ketFolded[ketBase +
                                                                  static_cast<std::size_t>(
                                                                      Hermite3DIndex(ux, uy, uz))];

                                            if (ketValue == 0.0)
                                            {
                                                continue;
                                            }

                                            value +=
                                                braValue * ketValue *
                                                scratch.kernel[static_cast<std::size_t>(
                                                    Hermite3DIndex(tx + ux, ty + uy, tz + uz))];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (std::size_t rowA = 0; rowA < rowsA; ++rowA)
                    {
                        for (std::size_t rowB = 0; rowB < rowsB; ++rowB)
                        {
                            for (std::size_t rowC = 0; rowC < rowsC; ++rowC)
                            {
                                for (std::size_t rowD = 0; rowD < rowsD; ++rowD)
                                {
                                    const std::size_t braRow =
                                        (rowB * static_cast<std::size_t>(nAngB) +
                                         static_cast<std::size_t>(fb)) *
                                            nFuncsA +
                                        rowA * static_cast<std::size_t>(nAngA) +
                                        static_cast<std::size_t>(fa);
                                    const std::size_t ketColumn =
                                        (rowD * static_cast<std::size_t>(nAngD) +
                                         static_cast<std::size_t>(fd)) *
                                            nFuncsC +
                                        rowC * static_cast<std::size_t>(nAngC) +
                                        static_cast<std::size_t>(fc);
                                    const double rowWeight =
                                        weight *
                                        contractionA.normalized[rowA][primitive.indices[0]] *
                                        contractionB.normalized[rowB][primitive.indices[1]] *
                                        contractionC.normalized[rowC][primitive.indices[2]] *
                                        contractionD.normalized[rowD][primitive.indices[3]];
                                    block[braRow * (nFuncsC * nFuncsD) + ketColumn] +=
                                        rowWeight * value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The packed block of one quartet through the derivative tier's contraction:
/// the term with no operator applied. It is the layout pin and the value the
/// finite-difference reference of the acceptance test converges to.
/// \param quartet The quartet.
/// \param order The derivative order the block is being built for (the raise
/// count the tables and the fold run to; the value itself is independent of
/// it).
/// \param block Out: the packed block.
/// \param scratch The call's scratch.
inline void BuildEriQuartetValue(const MdEriDerivativeQuartet& quartet,
                                 int order,
                                 std::span<double> block,
                                 MdEriDerivativeScratch& scratch) {
    std::fill(block.begin(), block.end(), 0.0);
    const MdShellContractions& contractionA = quartet.shells[0]->contractions;
    const MdShellContractions& contractionB = quartet.shells[1]->contractions;
    const MdShellContractions& contractionC = quartet.shells[2]->contractions;
    const MdShellContractions& contractionD = quartet.shells[3]->contractions;
    const int lMax = contractionA.angularMomentum + contractionB.angularMomentum +
                     contractionC.angularMomentum + contractionD.angularMomentum + 2 * order;

    for (std::size_t primA = 0; primA < contractionA.exponents.size(); ++primA)
    {
        for (std::size_t primB = 0; primB < contractionB.exponents.size(); ++primB)
        {
            for (std::size_t primC = 0; primC < contractionC.exponents.size(); ++primC)
            {
                for (std::size_t primD = 0; primD < contractionD.exponents.size(); ++primD)
                {
                    MdEriPrimitive primitive;
                    const std::array<std::size_t, 4> indices = {primA, primB, primC, primD};
                    primitive.indices = indices;

                    for (int shell = 0; shell < 4; ++shell)
                    {
                        const MdShellInput& input =
                            *quartet.shells[static_cast<std::size_t>(shell)];
                        const std::size_t index = indices[static_cast<std::size_t>(shell)];
                        primitive.exponents[static_cast<std::size_t>(shell)] =
                            input.contractions.exponents[index];
                        primitive.centers[static_cast<std::size_t>(shell)] = {
                            input.cx, input.cy, input.cz};
                    }

                    const double kernel =
                        BuildEriPrimitive(quartet, primitive, order, lMax, scratch);
                    CopyEriPairTables(scratch.bra, scratch.braWork);
                    CopyEriPairTables(scratch.ket, scratch.ketWork);
                    AccumulateEriTerm(quartet, primitive, scratch, 0, kernel, block);
                }
            }
        }
    }
}

/// The analytic nuclear-coordinate derivative of one quartet's packed block:
/// one tuple's block per order-tuple of flat coordinates, each 3 * atom +
/// axis, in request order. A coordinate names any atom of the caller's
/// geometry; one whose atom carries none of the quartet's shells contributes
/// an exactly zero block.
/// \param quartet The quartet.
/// \param order The derivative order, 1..kMaxMdDerivativeOrder.
/// \param coordinates The flat coordinates, order values per tuple.
/// \param blocks Out: MdEriDerivativeBlockCount elements.
/// \param scratch The call's scratch.
/// \param atomCount The atom count of the caller's geometry.
/// \returns Nothing, or an Error naming the bound that failed.
inline qcx::Result<void> BuildEriQuartetDerivative(const MdEriDerivativeQuartet& quartet,
                                                   int order,
                                                   std::span<const std::size_t> coordinates,
                                                   std::span<double> blocks,
                                                   MdEriDerivativeScratch& scratch,
                                                   std::size_t atomCount) {
    const auto checked =
        CheckEriDerivativeRequest(quartet, order, coordinates.size(), blocks.size(), atomCount);

    if (!checked.has_value())
    {
        return std::unexpected(checked.error());
    }

    const std::size_t tupleCount = *checked;
    const std::size_t blockElements = EriQuartetBlockElements(quartet);
    std::fill(blocks.begin(), blocks.end(), 0.0);

    const MdShellContractions& contractionA = quartet.shells[0]->contractions;
    const MdShellContractions& contractionB = quartet.shells[1]->contractions;
    const MdShellContractions& contractionC = quartet.shells[2]->contractions;
    const MdShellContractions& contractionD = quartet.shells[3]->contractions;
    const int lMax = contractionA.angularMomentum + contractionB.angularMomentum +
                     contractionC.angularMomentum + contractionD.angularMomentum + 2 * order;

    for (std::size_t primA = 0; primA < contractionA.exponents.size(); ++primA)
    {
        for (std::size_t primB = 0; primB < contractionB.exponents.size(); ++primB)
        {
            for (std::size_t primC = 0; primC < contractionC.exponents.size(); ++primC)
            {
                for (std::size_t primD = 0; primD < contractionD.exponents.size(); ++primD)
                {
                    MdEriPrimitive primitive;
                    primitive.indices = {primA, primB, primC, primD};

                    for (int shell = 0; shell < 4; ++shell)
                    {
                        const MdShellInput& input =
                            *quartet.shells[static_cast<std::size_t>(shell)];
                        primitive.exponents[static_cast<std::size_t>(shell)] =
                            input.contractions
                                .exponents[primitive.indices[static_cast<std::size_t>(shell)]];
                        primitive.centers[static_cast<std::size_t>(shell)] = {
                            input.cx, input.cy, input.cz};
                    }

                    const double kernel =
                        BuildEriPrimitive(quartet, primitive, order, lMax, scratch);

                    for (std::size_t tuple = 0; tuple < tupleCount; ++tuple)
                    {
                        const std::span<const std::size_t> flat =
                            coordinates.subspan(tuple * static_cast<std::size_t>(order),
                                                static_cast<std::size_t>(order));
                        std::array<EriOperatorList, kMaxMdDerivativeOrder> slots{};
                        int termCount = 1;

                        for (int slot = 0; slot < order; ++slot)
                        {
                            const std::size_t encoded = flat[static_cast<std::size_t>(slot)];
                            EriOperatorsOf(quartet,
                                           encoded / 3,
                                           static_cast<int>(encoded % 3),
                                           primitive,
                                           slots[static_cast<std::size_t>(slot)]);
                            termCount *= slots[static_cast<std::size_t>(slot)].count;
                        }

                        if (termCount == 0)
                        {
                            continue;
                        }

                        const std::span<double> block =
                            blocks.subspan(tuple * blockElements, blockElements);
                        std::array<int, kMaxMdDerivativeOrder> choice{};

                        for (int term = 0; term < termCount; ++term)
                        {
                            CopyEriPairTables(scratch.bra, scratch.braWork);
                            CopyEriPairTables(scratch.ket, scratch.ketWork);

                            int tierRaises = 0;

                            for (int slot = 0; slot < order; ++slot)
                            {
                                const EriOperator& op =
                                    slots[static_cast<std::size_t>(slot)]
                                        .operators[static_cast<std::size_t>(
                                            choice[static_cast<std::size_t>(slot)])];

                                if (op.kind == EriOperatorKind::kRaise)
                                {
                                    ++tierRaises;
                                }

                                ApplyEriOperator(op,
                                                 scratch.braWork,
                                                 scratch.ketWork,
                                                 scratch.braCopy,
                                                 scratch.ketCopy);
                            }

                            AccumulateEriTerm(
                                quartet, primitive, scratch, tierRaises, kernel, block);

                            for (int slot = 0; slot < order; ++slot)
                            {
                                if (++choice[static_cast<std::size_t>(slot)] <
                                    slots[static_cast<std::size_t>(slot)].count)
                                {
                                    break;
                                }

                                choice[static_cast<std::size_t>(slot)] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    return {};
}

} // namespace qcx::integrals::internal
