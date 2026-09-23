#pragma once

/// \file
/// Schwarz screening: the
/// per-shell-pair bounds Q_ab = sqrt of the largest diagonal element of the
/// (ab|ab) block. |(ab|cd)| <= Q_ab * Q_cd for every function quadruple.
///
/// The engine performs no screening internally - the caller (the dense
/// driver today, the direct Fock builders) supplies screened
/// quartet lists built from these bounds.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace qcx::integrals {

/// The Schwarz sweep's pair-data chunk cap: the byte mass of contracted pair
/// data (the E tables, the transforms and the weights BuildContractedPairTransform
/// commits) the sweep materializes per chunk, 24 B x the Hermite tables'
/// shape per primitive pair. The sweep needs the diagonal (ab|ab) block of
/// each canonical pair and nothing else, and a diagonal block's bra and ket
/// pair are the SAME pair, so the sweep is chunkable by pair at no
/// algorithmic cost: the pair store is built and dropped one chunk at a
/// time instead of whole (the 4,974-function C42H86/def2-QZVP point's whole
/// store is 9.95 GiB, which no path can afford as a screening pre-pass).
/// 512 MiB is the repo's standard batch cap
/// (EriBatchOptions::maxBatchBytes), so the sweep chunks at the same
/// granularity every other batched allocation uses.
/// \ingroup qcx-integrals
inline constexpr std::size_t kSchwarzChunkBytes = 512 * 1024 * 1024;

/// Computes the Schwarz bound of every canonical pair, in
/// ShellPairList::pairs order.
///
/// The diagonal (ab|ab) quartets are evaluated through the fp64 batch
/// pipeline, chunked over the pair list: the bra and ket pair of a diagonal
/// quartet are the same pair, so each chunk keeps only its own pairs'
/// contracted data resident (the geometry-only skeleton, sizeof(MdPairData)
/// per canonical pair, plus the chunk's pair data). A pair list whose whole
/// payload fits \p chunkBytes runs in ONE chunk and produces byte-for-byte
/// the values of the pre-chunking single-pass form.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
/// (kUnimplemented otherwise).
/// \param chunkBytes The pair-data chunk cap (kSchwarzChunkBytes by
/// default); 0 disables chunking - one chunk covering the whole pair list,
/// the pre-chunking form (the bit-identity reference of the chunked path).
/// \returns Q_ab per canonical pair, or an Error (a chunk's diagonal
/// quartets are evaluated through the fp64 batch pipeline).
/// \ingroup qcx-integrals
qcx::Result<std::vector<double>> ComputeSchwarzBounds(const qcx::molecule::Molecule& molecule,
                                                      const qcx::basisset::BasisSet& basisSet,
                                                      std::size_t chunkBytes = kSchwarzChunkBytes);

/// One block element's derivative-aware screening data, for the one-electron
/// blocks: the largest magnitude the element carries across the pair's
/// overlap, kinetic and nuclear-attraction blocks, and the largest magnitude
/// its derivative carries with respect to the pair's OWN centre coordinates
/// (the bra atom's three axes, then the ket atom's three - three only when
/// both shells sit on one atom).
///
/// Both are maxima, so neither is under the quantity it bounds. That is the
/// point of the element granularity: a rule that drops on the value alone
/// silently drops an element whose value vanishes at a symmetry or nodal
/// geometry while its gradient does not - absent from the energy, fully
/// present in the gradient, and invisible to every screen written for the
/// energy.
/// \ingroup qcx-integrals
struct DerivativeAwareBound {
    double value = 0.0; ///< Largest |S|, |T| or |V| the element carries.
    double derivative = 0.0; ///< Largest |dS/dX|, |dT/dX| or |dV/dX| it carries.
};

/// One canonical pair's bound, one entry per element of its (nFuncsA x nFuncsB)
/// block in block order - the order the one-electron builders write.
/// \ingroup qcx-integrals
struct PairDerivativeBounds {
    std::size_t nFuncs = 0; ///< Elements of the pair's block.
    std::vector<DerivativeAwareBound> elements; ///< nFuncs entries, block order.
};

/// One retained contribution: a block element of a canonical pair.
/// \ingroup qcx-integrals
struct ScreenedElement {
    std::size_t pair = 0; ///< Index into ShellPairList::pairs.
    std::size_t element = 0; ///< Index into the pair's block.
};

/// Computes the derivative-aware bound of every canonical pair, in
/// ShellPairList::pairs order.
///
/// The pass is chunked over the pair list exactly as the Schwarz sweep is
/// (the same kSchwarzChunkBytes cap and the same single-chunk byte-identity
/// when the payload fits): each chunk's contracted pair data is built, used
/// for that chunk's pairs and released, so the sweep never materializes the
/// whole store. Each pair costs one block build and one derivative build per
/// operator - the pair-level pass, whose cost is O(nPairs); it is the
/// companion of a quartet-level product bound, which is what screens the
/// O(n^2) pair blocks' O(n^2) quartets.
/// \param molecule Molecule providing the atom coordinates (Bohr) and the
/// nuclear charges.
/// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
/// (kUnimplemented otherwise).
/// \param chunkBytes The pair-data chunk cap (kSchwarzChunkBytes by default);
/// 0 disables chunking.
/// \returns The bound per canonical pair, or an Error (kUnimplemented for a
/// shell beyond kMaxEngineL; the block builders' errors otherwise).
/// \ingroup qcx-integrals
qcx::Result<std::vector<PairDerivativeBounds>> ComputeDerivativeAwareBounds(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    std::size_t chunkBytes = kSchwarzChunkBytes);

/// The derivative-aware retention rule: an element survives when EITHER bound
/// is above the threshold. The threshold is the caller's - the gradient path
/// reuses the energy path's Schwarz threshold rather than deriving its own.
/// \param bound The element's bound (ComputeDerivativeAwareBounds).
/// \param threshold The caller's threshold.
/// \returns True when the element contributes to the gradient.
/// \ingroup qcx-integrals
inline bool SurvivesDerivativeScreen(const DerivativeAwareBound& bound, double threshold) noexcept {
    return bound.value > threshold || bound.derivative > threshold;
}

/// The screened set: the contributions SurvivesDerivativeScreen retains, in
/// pair then element order - the list a gradient caller iterates instead of
/// the whole pair list.
/// \param bounds The bounds (ComputeDerivativeAwareBounds).
/// \param threshold The caller's threshold.
/// \returns The retained contributions.
/// \ingroup qcx-integrals
inline std::vector<ScreenedElement> RetainedElements(std::span<const PairDerivativeBounds> bounds,
                                                     double threshold) {
    std::vector<ScreenedElement> retained;

    for (std::size_t pair = 0; pair < bounds.size(); ++pair)
    {
        for (std::size_t element = 0; element < bounds[pair].elements.size(); ++element)
        {
            if (SurvivesDerivativeScreen(bounds[pair].elements[element], threshold))
            {
                retained.push_back(ScreenedElement{pair, element});
            }
        }
    }

    return retained;
}

/// One canonical pair's bound for the two-electron screen: the largest
/// magnitude the pair's own blocks carry, and the largest magnitude the
/// bound's own derivative carries with respect to the pair's own centre
/// coordinates.
///
/// The pair pass that produces these is the two-electron companion of
/// ComputeDerivativeAwareBounds, and the quantities are the two-electron
/// ones - the one-electron pair bounds are a different pair of numbers and
/// do not bound a quartet's blocks.
/// \ingroup qcx-integrals
struct TwoElectronPairBound {
    /// The pair's Schwarz bound: the square root of its largest diagonal
    /// element.
    double value = 0.0;
    double derivative = 0.0; ///< Largest |d(value)/dX| over the pair's centre coordinates.

    /// Both fields are infinite for a pair with no evaluable diagonal block
    /// (ComputeTwoElectronPairBounds), which is a pair the screen keeps.
    /// \returns True when the pair carries a bound.
    bool hasBound() const noexcept {
        return std::isfinite(value) && std::isfinite(derivative);
    }
};

/// Computes the two-electron pair bound of every canonical pair, in
/// ShellPairList::pairs order - the pair numbers QuartetDerivativeProduct
/// multiplies.
///
/// One diagonal (ab|ab) quartet per pair, whose diagonal elements give the
/// value: the same numbers the Schwarz sweep reads, so the two agree by
/// construction. The derivative is the bound's OWN derivative with respect to
/// the pair's centre coordinates (the bra atom's three axes, then the ket
/// atom's - three only when both shells sit on one atom), and it is read off
/// the same diagonal elements: Q_ab = sqrt(m) with m the largest diagonal
/// element, so |dQ_ab/dX| is |dm/dX| / (2 sqrt(m)) and |dm/dX| is at most the
/// largest |dq/dX| over the elements m maximizes.
///
/// The distinction is the whole point of the pair. Bounding the derivative by
/// the value instead gives zero derivative wherever the value vanishes - and a
/// pair whose bound passes through zero at a symmetric geometry is exactly the
/// one whose quartet contributions do not.
///
/// The tier holds no pair store - it builds one quartet's tables per pair and
/// releases them - so this pass is not chunked the way the Schwarz and
/// one-electron sweeps are, and takes no chunk cap.
///
/// The analytic derivative needs the kernel table to reach Hermite order
/// 2 (l_i + l_j) + 2, and the table covers 2 kMaxShellL; past that the
/// derivative is measured instead: one central difference of the bound along
/// the pair's separation, which bounds every centre-axis derivative of the
/// bound because the bound depends on the pair's geometry through that
/// separation alone.
///
/// A pair whose shells add past kMaxShellL has a diagonal block reaching
/// Hermite order 2 (l_i + l_j), past the table, so this build produces neither
/// the bound nor its derivative: both fields are infinite for such a pair, and
/// every quartet that touches it survives the screen. Fitting bases are where
/// such pairs appear - an f and a g shell on one centre is one pair of an
/// unusable block apiece - and a quartet that is kept costs work, while one
/// that is dropped on a number nothing derives costs the gradient.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
/// (kUnimplemented otherwise).
/// \returns The bound per canonical pair, or an Error (kUnimplemented for a
/// shell beyond kMaxEngineL; the tier's errors otherwise).
/// \ingroup qcx-integrals
qcx::Result<std::vector<TwoElectronPairBound>> ComputeTwoElectronPairBounds(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet);

/// One quartet's bound for the two-electron screen - the product companion of
/// TwoElectronPairBound one level up.
/// \ingroup qcx-integrals
struct QuartetDerivativeBound {
    double value = 0.0; ///< Largest |(ab|cd)| magnitude the quartet carries.
    double derivative = 0.0; ///< Largest |d(ab|cd)/dX| magnitude it carries.
};

/// The quartet-level product bound of two pair bounds.
///
/// A quartet's derivative splits into a bra-pair-differentiated term and a
/// ket-pair-differentiated term, each a bilinear form in its own pair's
/// differentiated data against the other pair's plain data, so the product
/// rule below reads one derivative from each pair.
///
/// The value half is the trivially conservative Schwarz product - two maxima,
/// so never under the quartet's value. The derivative half is the product of
/// the two bounds' own derivatives, which is the derivative of the bound; it
/// is an estimate and not a ceiling (a pair whose bound is stationary, as one
/// whose shells sit on a single atom is, contributes nothing to it while the
/// quartet's own derivative need not vanish). The two halves are read against
/// the same threshold and a quartet survives on the larger, so the derivative
/// half only ever keeps quartets the value half would drop, and the screen
/// keeps every quartet a value-only Schwarz screen keeps.
///
/// A pair with no bound (TwoElectronPairBound::hasBound) makes the product
/// unbounded at both orders, which keeps every quartet that touches it.
/// \param bra The bra pair's bound.
/// \param ket The ket pair's bound.
/// \returns The quartet's bound.
/// \ingroup qcx-integrals
inline QuartetDerivativeBound QuartetDerivativeProduct(const TwoElectronPairBound& bra,
                                                       const TwoElectronPairBound& ket) noexcept {
    // A pair with no bound keeps every quartet it touches, at both orders: the
    // product of an unbounded term with a zero would be a NaN, which reads as
    // neither above nor below the threshold.
    if (!bra.hasBound() || !ket.hasBound())
    {
        return QuartetDerivativeBound{std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity()};
    }

    return QuartetDerivativeBound{bra.value * ket.value,
                                  bra.derivative * ket.value + bra.value * ket.derivative};
}

/// The two-electron derivative screen's retention rule: a quartet survives
/// when either bound is above the threshold. The threshold is the caller's -
/// the gradient path reuses the energy path's Schwarz threshold rather than
/// deriving its own.
/// \param bound The quartet's bound (QuartetDerivativeProduct).
/// \param threshold The caller's threshold.
/// \returns True when the quartet contributes to the gradient.
/// \ingroup qcx-integrals
inline bool SurvivesQuartetDerivativeScreen(const QuartetDerivativeBound& bound,
                                            double threshold) noexcept {
    return bound.value > threshold || bound.derivative > threshold;
}

/// One candidate quartet: two canonical pair indices, the bra pair first.
/// \ingroup qcx-integrals
struct ScreenedQuartet {
    std::size_t braPair = 0; ///< Index into the caller's pair bounds and pair list.
    std::size_t ketPair = 0; ///< Index into the caller's pair bounds and pair list.
};

/// The two-electron screen over a caller's candidate list: the quartets
/// SurvivesQuartetDerivativeScreen retains, in candidate order. The candidates
/// are the caller's - the screen is a filter and never enumerates the
/// O(n^4) quartet space itself, which is what keeps it affordable.
/// \param bounds The pair bounds (one per canonical pair).
/// \param candidates The quartets to test.
/// \param threshold The caller's threshold.
/// \returns The retained quartets.
/// \ingroup qcx-integrals
inline std::vector<ScreenedQuartet> RetainedQuartets(std::span<const TwoElectronPairBound> bounds,
                                                     std::span<const ScreenedQuartet> candidates,
                                                     double threshold) {
    std::vector<ScreenedQuartet> retained;
    retained.reserve(candidates.size());

    for (const ScreenedQuartet& candidate : candidates)
    {
        if (candidate.braPair >= bounds.size() || candidate.ketPair >= bounds.size())
        {
            continue;
        }

        const QuartetDerivativeBound bound =
            QuartetDerivativeProduct(bounds[candidate.braPair], bounds[candidate.ketPair]);

        if (SurvivesQuartetDerivativeScreen(bound, threshold))
        {
            retained.push_back(candidate);
        }
    }

    return retained;
}

} // namespace qcx::integrals
