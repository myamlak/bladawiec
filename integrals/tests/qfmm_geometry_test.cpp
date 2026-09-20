// Shell-pair geometry of the QFMM octree: hand-computed checks of the
// expansion center (the exact midpoint of the two shell centers) and the
// conservative bounding radius (sqrt(-ln(tau)/p) falloff plus the
// primitive-center offset, maxed over primitive pairs). Five-line
// arithmetic checks - the gate before anything built on top of
// ComputeExtent is trusted.

#include "internal/qfmm_geometry.hpp"
#include "qcx/integrals/accuracy.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace {

using qcx::integrals::internal::ComputeExtent;
using qcx::integrals::internal::ComputePairGeometries;
using qcx::integrals::internal::ComputePairGeometry;
using qcx::integrals::internal::kQfmmExtentThreshold;
using qcx::integrals::internal::MdPairData;
using qcx::integrals::internal::MdPrimPair;
using qcx::integrals::internal::QfmmPairGeometry;

/// A one-primitive-pair s/s pair spanning the shell centers (ax, ay, az)
/// and (bx, by, bz), with its single Gaussian-product center at P and the
/// combined exponent p. Only the fields ComputePairGeometry reads are set.
MdPairData MakePair(double ax,
                    double ay,
                    // (az, bx) are the two centers' z/x coordinates - the
                    // triplets feed the same field.
                    //
                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                    double az,
                    double bx,
                    double by,
                    // (bz, px) are the second center's z and the product
                    // center's x - same-type coordinates of one pair.
                    //
                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                    double bz,
                    double px,
                    double py,
                    double pz,
                    double p) {
    MdPairData pair{};
    pair.ax = ax;
    pair.ay = ay;
    pair.az = az;
    pair.bx = bx;
    pair.by = by;
    pair.bz = bz;
    MdPrimPair prim{};
    prim.p = p;
    prim.px = px;
    prim.py = py;
    prim.pz = pz;
    pair.primPairs.push_back(prim);
    return pair;
}

/// Adds a second primitive pair with its own product center and exponent.
void AddPrimitive(MdPairData& pair, double px, double py, double pz, double p) {
    MdPrimPair prim{};
    prim.p = p;
    prim.px = px;
    prim.py = py;
    prim.pz = pz;
    pair.primPairs.push_back(prim);
}

// The hand-computed falloff radius of one primitive with p = 1 at tau =
// 1e-10: sqrt(-ln(1e-10) / 1) = sqrt(10 ln 10). Transcribed from the math,
// not from the implementation; the separate constant pin below makes a
// stale kQfmmExtentThreshold fail loudly.
const double kExpectedFalloffP1 = std::sqrt(10.0 * std::log(10.0));

TEST(QfmmGeometryTest, CenterIsExactMidpoint) {
    // Two s shells 2 Bohr apart along x: the center is (1, 0, 0) exactly.
    const QfmmPairGeometry geometry = ComputePairGeometry(
        MakePair(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0), kQfmmExtentThreshold);
    EXPECT_DOUBLE_EQ(geometry.centerX, 1.0);
    EXPECT_DOUBLE_EQ(geometry.centerY, 0.0);
    EXPECT_DOUBLE_EQ(geometry.centerZ, 0.0);

    // An asymmetric placement: (-1, 2, -3) and (3, 4, 5) midpoints at
    // (1, 3, 1), regardless of where the primitive product center sits.
    const QfmmPairGeometry offAxis = ComputePairGeometry(
        MakePair(-1.0, 2.0, -3.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 1.0), kQfmmExtentThreshold);
    EXPECT_DOUBLE_EQ(offAxis.centerX, 1.0);
    EXPECT_DOUBLE_EQ(offAxis.centerY, 3.0);
    EXPECT_DOUBLE_EQ(offAxis.centerZ, 1.0);
}

TEST(QfmmGeometryTest, ExtentMatchesHandComputedFalloff) {
    // Primitive product center ON the shell-pair center, p = 1: the extent
    // is exactly the falloff radius sqrt(10 ln 10).
    const MdPairData onCenter = MakePair(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
    EXPECT_NEAR(ComputeExtent(onCenter, kQfmmExtentThreshold), kExpectedFalloffP1, 1e-12);

    // A threshold change flows through: at tau = 1e-4 the falloff is
    // sqrt(4 ln 10) = 2 sqrt(ln 10).
    EXPECT_NEAR(ComputeExtent(onCenter, 1e-4), 2.0 * std::sqrt(std::log(10.0)), 1e-12);
}

TEST(QfmmGeometryTest, ExtentIncludesPrimitiveCenterOffset) {
    // The primitive product center sits 0.5 Bohr off the shell-pair center
    // with p = 4: extent = 0.5 + sqrt(10 ln 10 / 4) = 0.5 + sqrt(10 ln 10)/2.
    const MdPairData offset = MakePair(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.5, 0.0, 0.0, 4.0);
    EXPECT_NEAR(ComputeExtent(offset, kQfmmExtentThreshold), 0.5 + 0.5 * kExpectedFalloffP1, 1e-12);
}

TEST(QfmmGeometryTest, ExtentIsTheMaxOverPrimitivePairs) {
    // The loose pair (p = 1, extent sqrt(10 ln 10)) dominates the tight
    // offset pair (p = 4, extent 0.5 + sqrt(10 ln 10)/2).
    MdPairData pair = MakePair(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.5, 0.0, 0.0, 4.0);
    AddPrimitive(pair, 1.0, 0.0, 0.0, 1.0);
    EXPECT_NEAR(ComputeExtent(pair, kQfmmExtentThreshold), kExpectedFalloffP1, 1e-12);

    // And a prim pair with no offset at all (p = 0.25: falloff
    // sqrt(10 ln 10)/0.5 = 2 sqrt(10 ln 10)) widens it further.
    MdPairData wider = MakePair(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
    AddPrimitive(wider, 1.0, 0.0, 0.0, 0.25);
    EXPECT_NEAR(ComputeExtent(wider, kQfmmExtentThreshold), 2.0 * kExpectedFalloffP1, 1e-12);
}

TEST(QfmmGeometryTest, ComputePairGeometriesIsPerPair) {
    const MdPairData first = MakePair(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
    const MdPairData second = MakePair(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.5, 0.0, 0.0, 4.0);
    const auto geometries = ComputePairGeometries({first, second}, kQfmmExtentThreshold);

    ASSERT_EQ(geometries.size(), 2u);
    EXPECT_DOUBLE_EQ(geometries[0].centerX, 1.0);
    EXPECT_NEAR(geometries[0].extent, kExpectedFalloffP1, 1e-12);
    EXPECT_NEAR(geometries[1].extent, 0.5 + 0.5 * kExpectedFalloffP1, 1e-12);
}

TEST(QfmmGeometryTest, ExtentThresholdConstantIsPinned) {
    // The named constant is the 1e-10 geometric cutoff the hand-computed
    // falloff above assumes; a change here fails this test loudly instead
    // of silently changing every extent. It is ALSO the kTight rung of the
    // per-preset extent ladder (QfmmExtentForPreset, accuracy.hpp): the committed
    // value survives as the tight preset's footprint bound.
    EXPECT_DOUBLE_EQ(kQfmmExtentThreshold, 1e-10);
    EXPECT_DOUBLE_EQ(kQfmmExtentThreshold,
                     qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kTight));
}

TEST(QfmmGeometryTest, ExtentLadderIsMonotoneInThePreset) {
    // The per-preset extent ladder
    // {1e-6, 1e-8, 1e-10} for {kLoose, kNormal, kTight} - the tight preset
    // keeps the loosest (largest) footprint bound, so the extent is
    // monotone non-decreasing kLoose -> kTight on any real pair (the
    // tail-charge coupling: a looser preset admits a larger missed-charge
    // tail at its own coarser budget and shrinks the boxes).
    EXPECT_DOUBLE_EQ(qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kLoose),
                     1e-6);
    EXPECT_DOUBLE_EQ(qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kNormal),
                     1e-8);
    EXPECT_DOUBLE_EQ(qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kTight),
                     1e-10);

    const MdPairData pair = MakePair(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
    const double loose = ComputeExtent(
        pair, qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kLoose));
    const double normal = ComputeExtent(
        pair, qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kNormal));
    const double tight = ComputeExtent(
        pair, qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kTight));

    EXPECT_GE(tight, normal);
    EXPECT_GE(normal, loose);
    // The hand-computed anchor: at p = 1 the kTight extent is exactly
    // sqrt(10 ln 10) (the kExpectedFalloffP1 of the falloff tests).
    EXPECT_NEAR(tight, kExpectedFalloffP1, 1e-12);
    // The kLoose box is strictly smaller than the kTight box for this pair
    // (the loose preset's admitted tail is 8 orders coarser, so the
    // falloff radius shrinks by sqrt(23.0/13.8) ~ 1.29x).
    EXPECT_LT(loose, tight);
}

} // namespace
