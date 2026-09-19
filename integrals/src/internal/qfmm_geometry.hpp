#pragma once

// The shell-pair geometry of the QFMM skeleton: the expansion
// center and the conservative bounding radius of one canonical shell pair.
// These two numbers feed the octree construction (qfmm_tree.hpp) and, via
// the tree's box geometry, the well-separatedness test (qfmm_tree.hpp's
// interaction lists).
//
// Design note: the expansion center is the MIDPOINT of
// the two shell centers, deliberately NOT a charge-weighted centroid of the
// per-primitive Gaussian-product centers. The centroid would center the
// multipole expansion marginally better, but the midpoint is trivial to
// compute with zero risk of a weighting-formula bug and always
// well-defined regardless of the primitives' exponent ratio - "skeleton,
// correctness over performance" is this stage's stated design principle.

#include "md_batch.hpp"

#include <cstddef>

namespace qcx::integrals::internal {

/// The geometric density cutoff of ComputeExtent (dimensionless): the
/// kTight rung of the per-preset extent ladder (QfmmExtentForPreset,
/// accuracy.hpp). A primitive
/// pair's Gaussian falloff radius sqrt(-ln(tau)/p) drops below this
/// threshold at tau = 1e-10. This is a GEOMETRIC cutoff deciding "how big
/// is this Gaussian's footprint" - deliberately NOT the accuracy preset's
/// Schwarz/density threshold, which answers a different question ("is this
/// quartet worth computing"); conflating the two threshold concepts makes
/// debugging harder. The
/// production coupling of extent to accuracy is the tail-charge argument
/// (documented on QfmmExtentForPreset): the pair's charge outside the box
/// is bounded by the Gaussian tail integral at tau, so a looser preset may
/// shrink the boxes (and grow the far field) at its own coarser budget.
inline constexpr double kQfmmExtentThreshold = 1e-10;

/// Shell-pair expansion center for QFMM multipole moments: the midpoint
/// of the two shell centers. NOT a charge-weighted centroid of the
/// per-primitive Gaussian-product centers - deliberately simpler (see the
/// file comment) and always well-defined regardless of the primitives'
/// exponent ratio.
struct QfmmPairGeometry {
    double centerX, centerY, centerZ; ///< Bohr.
    double extent; ///< Bohr - see ComputeExtent.
};

/// Which objects and radii the pair geometry is built from (the product-ball
/// fix, option-gated; the default keeps the recorded form bit-identically).
enum class QfmmExtentModel {
    /// The recorded form: expansion centre = the shell-pair MIDPOINT, extent
    /// = max over primitive products of (|product centre - midpoint| +
    /// falloff radius). The offset term is tau-INDEPENDENT and reaches ~R/2
    /// for a tight x distant-diffuse product - the inflation the counts
    /// measured (median 12.5 Bohr at 586 functions on a ~55-Bohr
    /// chain, where the products themselves are a few Bohr across).
    kMidpointBound,
    /// The corrected form: the multipole expansion is of the PRODUCT
    /// DISTRIBUTION, so the centre is the amplitude-weighted centroid of the
    /// pair's SIGNIFICANT primitive products (|d_a d_b| * prefactor at or
    /// above productWeightTau x the pair's largest) and the radius is the
    /// covering radius of those products about it. Insignificant products
    /// (the tight x distant-diffuse ones, whose prefactor is e^-40 and
    /// below) are excluded: they contribute nothing to the moments, and
    /// including them was what made every box molecule-sized.
    kProductBall,
};

/// The significance floor of kProductBall's product set: a primitive
/// product joins the pair's distribution when its amplitude weight is at
/// least this fraction of the pair's largest. Dimensionless, and separate
/// from the geometric tau of the falloff radius (which decides how far a
/// Gaussian reaches, not whether it is there at all).
inline constexpr double kQfmmProductWeightFloor = 1e-6;

/// The two terms of the recorded extent, per pair (a measurement):
/// the largest product offset from the shell-pair midpoint and the largest
/// product falloff radius, over ALL products and - separately - over the
/// SIGNIFICANT ones (the weight floor above). The current extent is the max
/// of (offset + falloff) over all products; the corrected radius is the
/// significant products' spread about their own centroid.
struct QfmmExtentSplit {
    double maxOffset = 0.0; ///< All products: max |product centre - midpoint| (Bohr).
    double maxFalloff = 0.0; ///< All products: max sqrt(-ln(tau) / p) (Bohr).
    double significantOffset = 0.0; ///< Significant products only (same quantity).
    double significantFalloff = 0.0; ///< Significant products only (same quantity).
    double productRadius = 0.0; ///< The kProductBall covering radius (Bohr).
    double productOffset = 0.0; ///< The significant products' centroid shift from the midpoint.
    double centerX = 0.0; ///< The kProductBall centre (the significant centroid, Bohr).
    double centerY = 0.0; ///< See centerX.
    double centerZ = 0.0; ///< See centerX.
};

/// The per-pair amplitude weight of one primitive product (the significance
/// test of kProductBall): the largest |d_a d_b| of the product's contraction
/// row pairs times its Gaussian prefactor.
double PrimitiveProductWeight(const MdPairData& pair, std::size_t primIndex);

/// The extent decomposition and the kProductBall radius of one pair.
/// \param pair The shell pair.
/// \param extentThreshold The geometric tau of the falloff radius.
/// \returns The split (see QfmmExtentSplit).
QfmmExtentSplit SplitPairExtent(const MdPairData& pair, double extentThreshold);

/// The shell-pair geometry under the given model: kMidpointBound reproduces
/// ComputePairGeometry exactly (bit-identical), kProductBall returns the
/// significant products' centroid and their covering radius.
/// \param pair The shell pair.
/// \param extentThreshold The geometric tau of the falloff radius.
/// \param model The geometry model.
/// \returns The geometry.
QfmmPairGeometry ComputePairGeometry(const MdPairData& pair,
                                     double extentThreshold,
                                     QfmmExtentModel model);

/// Conservative bounding radius of one shell pair: for every primitive
/// pair, the distance from ITS OWN Gaussian-product center to the
/// shell-pair expansion center, PLUS that primitive pair's own Gaussian
/// falloff radius at the given threshold. The max over all primitive
/// pairs - a deliberately loose (never tight) bound, appropriate for a
/// correctness-first skeleton: an over-large extent only costs a few
/// nodes ending up in the near field that a tighter bound might have
/// pushed to the far field - it never causes an INCORRECT far-field
/// classification, which is the property that matters here.
/// \param pair The shell pair (centers, per-primitive product centers and
/// combined exponents).
/// \param extentThreshold The geometric density cutoff tau; the falloff
/// radius of one primitive pair is sqrt(-ln(tau) / p).
/// \returns The conservative bounding radius in Bohr.
double ComputeExtent(const MdPairData& pair, double extentThreshold);

/// The shell-pair expansion center and bounding radius (Bohr) under the
/// RECORDED model: the midpoint of the two shell centers plus ComputeExtent.
/// \param pair The shell pair.
/// \param extentThreshold The geometric density cutoff (see ComputeExtent).
/// \returns The pair geometry.
QfmmPairGeometry ComputePairGeometry(const MdPairData& pair, double extentThreshold);

/// ComputePairGeometry for every pair of a pair store (the one-shot
/// geometry pass the octree construction consumes).
/// \param pairs The pair store.
/// \param extentThreshold The geometric density cutoff (see ComputeExtent).
/// \param model The geometry model (the recorded form by default, so every
/// existing caller keeps its exact behavior; kProductBall is the fix).
/// \returns One QfmmPairGeometry per input pair.
std::vector<QfmmPairGeometry> ComputePairGeometries(
    const std::vector<MdPairData>& pairs,
    double extentThreshold,
    QfmmExtentModel model = QfmmExtentModel::kMidpointBound);

} // namespace qcx::integrals::internal
