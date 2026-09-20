// Shell-pair geometry of the QFMM octree: the expansion center and
// conservative bounding radius per canonical shell pair. Pure geometry -
// no multipole math, no FMM-specific risk; the octree construction builds
// on these numbers.

#include "internal/qfmm_geometry.hpp"

#include <algorithm>
#include <cmath>

namespace qcx::integrals::internal {

double ComputeExtent(const MdPairData& pair, double extentThreshold) {
    const double cx = 0.5 * (pair.ax + pair.bx);
    const double cy = 0.5 * (pair.ay + pair.by);
    const double cz = 0.5 * (pair.az + pair.bz);
    double maxExtent = 0.0;

    for (const MdPrimPair& prim : pair.primPairs)
    {
        const double dx = prim.px - cx;
        const double dy = prim.py - cy;
        const double dz = prim.pz - cz;
        const double offsetFromCenter = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double falloffRadius = std::sqrt(-std::log(extentThreshold) / prim.p);
        maxExtent = std::max(maxExtent, offsetFromCenter + falloffRadius);
    }

    return maxExtent;
}

QfmmPairGeometry ComputePairGeometry(const MdPairData& pair, double extentThreshold) {
    QfmmPairGeometry geometry{};
    geometry.centerX = 0.5 * (pair.ax + pair.bx);
    geometry.centerY = 0.5 * (pair.ay + pair.by);
    geometry.centerZ = 0.5 * (pair.az + pair.bz);
    geometry.extent = ComputeExtent(pair, extentThreshold);
    return geometry;
}

double PrimitiveProductWeight(const MdPairData& pair, std::size_t primIndex) {
    double weight = 0.0;

    if (primIndex < pair.braWeights.size())
    {
        for (const double coefficient : pair.braWeights[primIndex])
        {
            weight = std::max(weight, std::abs(coefficient));
        }
    }

    if (primIndex < pair.primPairs.size())
    {
        weight *= std::abs(pair.primPairs[primIndex].prefactor);
    }

    return weight;
}

QfmmExtentSplit SplitPairExtent(const MdPairData& pair, double extentThreshold) {
    QfmmExtentSplit split;
    const double cx = 0.5 * (pair.ax + pair.bx);
    const double cy = 0.5 * (pair.ay + pair.by);
    const double cz = 0.5 * (pair.az + pair.bz);

    // The significance pass first: the weight floor is relative to the
    // pair's own largest product, so a pair whose every product is tiny
    // keeps its shape rather than collapsing to a point.
    double maxWeight = 0.0;

    for (std::size_t prim = 0; prim < pair.primPairs.size(); ++prim)
    {
        maxWeight = std::max(maxWeight, PrimitiveProductWeight(pair, prim));
    }

    // A product joins the pair's distribution when its amplitude clears BOTH
    // the pair's own relative floor and the ABSOLUTE floor (the geometric tau
    // read as an amplitude floor: nothing below it can reach the screening
    // threshold). A pair whose every product sits below the absolute floor is
    // negligible as a whole, and its geometry is then its single largest
    // product's compact ball - an all-negligible pair whose products spread
    // over the whole bond would otherwise keep inflating every box it lands
    // in (the depth-cap carrier the counts measured: 20-33% of pairs
    // still sat in depth-capped leaves with the relative floor alone).
    const double floorWeight = std::max(kQfmmProductWeightFloor * maxWeight, extentThreshold);
    std::size_t largestProduct = 0;
    double largestWeight = 0.0;

    for (std::size_t prim = 0; prim < pair.primPairs.size(); ++prim)
    {
        const double weight = PrimitiveProductWeight(pair, prim);

        if (weight > largestWeight)
        {
            largestWeight = weight;
            largestProduct = prim;
        }
    }

    const bool negligible = largestWeight < floorWeight;
    const auto isSignificant = [&](std::size_t prim) {
        return negligible ? prim == largestProduct
                          : PrimitiveProductWeight(pair, prim) >= floorWeight;
    };

    double centroidX = 0.0;
    double centroidY = 0.0;
    double centroidZ = 0.0;
    double significantWeight = 0.0;

    for (std::size_t prim = 0; prim < pair.primPairs.size(); ++prim)
    {
        const MdPrimPair& product = pair.primPairs[prim];
        const double dx = product.px - cx;
        const double dy = product.py - cy;
        const double dz = product.pz - cz;
        const double offset = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double falloff = std::sqrt(-std::log(extentThreshold) / product.p);
        split.maxOffset = std::max(split.maxOffset, offset);
        split.maxFalloff = std::max(split.maxFalloff, falloff);

        if (isSignificant(prim))
        {
            split.significantOffset = std::max(split.significantOffset, offset);
            split.significantFalloff = std::max(split.significantFalloff, falloff);
            const double weight = PrimitiveProductWeight(pair, prim);
            centroidX += weight * product.px;
            centroidY += weight * product.py;
            centroidZ += weight * product.pz;
            significantWeight += weight;
        }
    }

    if (significantWeight > 0.0)
    {
        centroidX /= significantWeight;
        centroidY /= significantWeight;
        centroidZ /= significantWeight;
    } else
    {
        // No product carried any weight at all (the distant tight x diffuse
        // pairs, whose prefactor underflows to zero - the radius tail:
        // shell separation 56 Bohr, every amplitude 0). The pair's charge is
        // then wherever its single largest product sits: the midpoint fallback
        // would hand those pairs a molecule-sized ball and keep them chaining
        // to the depth cap, which is exactly the inflation this model exists
        // to remove.
        centroidX = pair.primPairs[largestProduct].px;
        centroidY = pair.primPairs[largestProduct].py;
        centroidZ = pair.primPairs[largestProduct].pz;
    }

    split.centerX = centroidX;
    split.centerY = centroidY;
    split.centerZ = centroidZ;
    const double centroidDx = centroidX - cx;
    const double centroidDy = centroidY - cy;
    const double centroidDz = centroidZ - cz;
    split.productOffset =
        std::sqrt(centroidDx * centroidDx + centroidDy * centroidDy + centroidDz * centroidDz);

    for (std::size_t prim = 0; prim < pair.primPairs.size(); ++prim)
    {
        if (!isSignificant(prim))
        {
            continue;
        }

        const MdPrimPair& product = pair.primPairs[prim];
        const double dx = product.px - centroidX;
        const double dy = product.py - centroidY;
        const double dz = product.pz - centroidZ;
        const double reach = std::sqrt(dx * dx + dy * dy + dz * dz) +
                             std::sqrt(-std::log(extentThreshold) / product.p);
        split.productRadius = std::max(split.productRadius, reach);
    }

    return split;
}

QfmmPairGeometry ComputePairGeometry(const MdPairData& pair,
                                     double extentThreshold,
                                     QfmmExtentModel model) {
    if (model == QfmmExtentModel::kMidpointBound)
    {
        return ComputePairGeometry(pair, extentThreshold);
    }

    const QfmmExtentSplit split = SplitPairExtent(pair, extentThreshold);
    QfmmPairGeometry geometry{};

    if (split.productRadius <= 0.0)
    {
        return ComputePairGeometry(pair, extentThreshold);
    }

    geometry.centerX = split.centerX;
    geometry.centerY = split.centerY;
    geometry.centerZ = split.centerZ;
    geometry.extent = split.productRadius;
    return geometry;
}

std::vector<QfmmPairGeometry> ComputePairGeometries(const std::vector<MdPairData>& pairs,
                                                    double extentThreshold,
                                                    QfmmExtentModel model) {
    std::vector<QfmmPairGeometry> geometries;
    geometries.reserve(pairs.size());

    for (const MdPairData& pair : pairs)
    {
        geometries.push_back(ComputePairGeometry(pair, extentThreshold, model));
    }

    return geometries;
}

} // namespace qcx::integrals::internal
