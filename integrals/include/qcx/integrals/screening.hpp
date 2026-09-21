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

#include <cstddef>
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
/// magnitude the pair's own blocks carry, and the largest magnitude their
/// derivative carries with respect to the pair's own centre coordinates.
///
/// The pair pass that produces these is the two-electron companion of
/// ComputeDerivativeAwareBounds, and the quantities are the two-electron
/// ones - the one-electron pair bounds are a different pair of numbers and
/// do not bound a quartet's blocks.
/// \ingroup qcx-integrals
struct TwoElectronPairBound {
    double value = 0.0; ///< Largest |(ab|..)| magnitude the pair carries.
    double derivative = 0.0; ///< Largest |d(ab|..)/dX| magnitude it carries.
};

/// One quartet's bound for the two-electron screen - the product companion of
/// TwoElectronPairBound one level up.
/// \ingroup qcx-integrals
struct QuartetDerivativeBound {
    double value = 0.0; ///< Largest |(ab|cd)| magnitude the quartet carries.
    double derivative = 0.0; ///< Largest |d(ab|cd)/dX| magnitude it carries.
};

/// The quartet-level product bound of two pair bounds.
///
/// A quartet's two centres are the product centres of two pairs: moving a
/// centre of the bra pair moves that pair's coefficient tables and the centre
/// its kernel is measured from, and nothing of the ket pair - and the other
/// way round. The derivative of the quartet is therefore a sum of one
/// bra-pair-differentiated term and one ket-pair-differentiated term, and
/// each is a bilinear form in its own pair's differentiated data against the
/// other pair's plain data. Bounding each factor by its pair's bound gives
/// the product rule below. It is O(1) per quartet and builds nothing, which
/// is the point: a screen that costs an integral evaluation per candidate is
/// not a screen.
///
/// Both bounds are maxima over their pair's elements, so neither is under the
/// quantity it bounds; the product of two maxima bounds the product, and the
/// sum bounds the sum.
/// \param bra The bra pair's bound.
/// \param ket The ket pair's bound.
/// \returns The quartet's bound.
/// \ingroup qcx-integrals
inline QuartetDerivativeBound QuartetDerivativeProduct(const TwoElectronPairBound& bra,
                                                       const TwoElectronPairBound& ket) noexcept {
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
