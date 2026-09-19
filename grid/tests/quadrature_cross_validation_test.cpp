// Cross-validation of the two quadrature stacks this repository carries:
// qcx::grid::MolecularGrid (the properties consumers) and
// excgrid::BlockGrid (the XC consumer), built over the same molecule at
// identical settings and compared on a quantity both can produce.
//
// The quantity is an analytic integral.  f(r) = exp(-a |r - R|^2) has the
// closed form (pi/a)^(3/2), so ONE number checks each stack against theory and
// against the other.  The integrand is centred on the H2 midpoint rather than on
// an atom, so neither stack can flatter itself with an atom-centred integrand.
//
// What the measurement says, and the difference between the two claims:
//   - THE STACKS AGREE TO ROUNDOFF.  They differ by ~1e-14 on a value of ~1.97
//     (relative ~6e-15), which is summation order and nothing else: excgrid
//     re-batches points spatially and qcx keeps atom-major order, and each sums
//     the same terms in its own order.
//   - THAT IS NOT THE SAME AS ACCURACY.  Both stacks sit ~1.5e-6 away from the
//     analytic value, which is this quadrature's own accuracy floor for this
//     integrand at these settings - not an error in either implementation.  The
//     refinement check below is what separates the two readings: a quadrature
//     error shrinks when the grid is refined, an assembly error does not.
//
// A NOTE ON A QUANTITY THAT DID NOT WORK.  Summed quadrature weights were the
// other candidate for this comparison and they are NOT usable: the untrimmed
// molecular grid's total weight is ~5.7e11, because the outermost radial points
// carry r^2 factors that make the sum diverge, so a stack-to-stack difference
// there is dominated by roundoff on a meaningless number.  It is asserted below
// only as a roundoff bound, to record why the integral is the quantity to use.
//
// HONEST LIMIT: this certifies ONE configuration - identical settings.  The
// shipped settings differ (excgrid defaults to 75x302 with trimming and
// re-batching; the driver and properties paths pin 80x194), and their agreement
// is NOT what this test measures.

#include "h2_sto3g.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/grid/molecular_grid.hpp"
#include "qcx/grid/xc_grid_engine.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <excgrid/grid.hpp>
#include <gtest/gtest.h>

namespace qcx::grid {
namespace {

constexpr double kPi = 3.14159265358979323846;

// The shared settings: both stacks are asked for the same quadrature.
constexpr std::size_t kRadialPoints = 75;
constexpr std::size_t kAngularPoints = 302;
constexpr double kAlpha = 0.5;
constexpr std::size_t kRadialExponent = 2;
constexpr double kTrimWeight = 1e-15;

// The refined settings, for the refinement check below.
constexpr std::size_t kRefinedRadialPoints = 110;
constexpr std::size_t kRefinedAngularPoints = 434;

// f(r) = exp(-a |r - R|^2), integral over R^3 = (pi/a)^(3/2).
constexpr double kGaussianExponent = 2.0;

// Measured on this fixture: the stacks agree to 1.1e-14 and sit 1.476e-6 from
// the analytic value.  The pin carries about fiftyfold headroom over the
// accuracy floor, and the refinement check below is what justifies reading that
// floor as quadrature error.
constexpr double kCrossStackTolerance = 1e-12;
constexpr double kAccuracyTolerance = 5e-5;

double Gaussian(const std::array<double, 3>& point, const std::array<double, 3>& center) {
    double distanceSquared = 0.0;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const double delta = point[axis] - center[axis];
        distanceSquared += delta * delta;
    }

    return std::exp(-kGaussianExponent * distanceSquared);
}

std::array<double, 3> Midpoint(const qcx::molecule::Molecule& molecule) {
    const auto& coordinates = molecule.CoordinatesBohr();
    std::array<double, 3> center{0.0, 0.0, 0.0};

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            center[axis] += coordinates(atom, axis) / static_cast<double>(molecule.AtomCount());
        }
    }

    return center;
}

// (radial, angular) follows GridParams' own field order.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
excgrid::GridParams SharedParams(std::size_t radial, std::size_t angular) {
    excgrid::GridParams params;
    params.radialPoints = radial;
    params.angularPoints = angular;
    params.alpha = kAlpha;
    params.radialExponent = kRadialExponent;
    params.trimWeight = kTrimWeight;

    return params;
}

} // namespace

TEST(QuadratureCrossValidationTest, AnalyticGaussianIntegralAgreesAcrossBothStacks) {
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());

    // qcx side: no trimming, atom-major order.
    const auto qcxGrid = MolecularGrid::Create(*molecule, kRadialPoints, kAngularPoints, kAlpha);
    ASSERT_TRUE(qcxGrid.has_value());

    // excgrid side: same quadrature, its own trimming and re-batching.
    const auto geometry = ToExcgridGeometry(*molecule);
    ASSERT_TRUE(geometry.has_value());
    const auto blocks =
        excgrid::BlockGrid::Create(*geometry, SharedParams(kRadialPoints, kAngularPoints));
    ASSERT_TRUE(blocks.has_value());

    const std::array<double, 3> center = Midpoint(*molecule);

    double qcxWeight = 0.0;
    double qcxIntegral = 0.0;
    double qcxTrimmedWeight = 0.0;
    std::size_t qcxTrimmedPoints = 0;
    std::size_t qcxZeroWeightPoints = 0;

    for (std::size_t point = 0; point < qcxGrid->Size(); ++point)
    {
        const double weight = qcxGrid->Weight(point);
        qcxWeight += weight;
        qcxIntegral += weight * Gaussian(qcxGrid->Point(point), center);

        if (std::abs(weight) < kTrimWeight)
        {
            qcxTrimmedWeight += std::abs(weight);
            ++qcxTrimmedPoints;

            if (weight == 0.0)
            {
                ++qcxZeroWeightPoints;
            }
        }
    }

    double excgridWeight = 0.0;
    double excgridIntegral = 0.0;

    for (const excgrid::Block& block : blocks->Blocks())
    {
        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            excgridWeight += block.weights[point];
            excgridIntegral += block.weights[point] * Gaussian(block.points[point], center);
        }
    }

    const double exact = std::pow(kPi / kGaussianExponent, 1.5);

    // Structural claim: excgrid keeps exactly the points qcx keeps, minus the
    // ones below its trim threshold.
    EXPECT_EQ(blocks->TotalPointCount() + qcxTrimmedPoints, qcxGrid->Size());
    EXPECT_GT(qcxTrimmedPoints, 0u) << "the fixture must exercise the trim rule";

    // On THIS fixture every trimmed point weighs exactly zero, so the trimmed
    // tail is empty and the stacks have nothing but summation order between
    // them.  Recorded because it is the reason the cross-stack bound below is
    // as tight as it is.
    EXPECT_EQ(qcxTrimmedWeight, 0.0);
    EXPECT_EQ(qcxZeroWeightPoints, qcxTrimmedPoints);

    // The stacks against each other.  The bound is the trimmed tail (empty here)
    // plus the roundoff a re-ordered sum of this size can accumulate.
    EXPECT_LE(std::abs(qcxIntegral - excgridIntegral), kCrossStackTolerance)
        << "qcx " << qcxIntegral << " excgrid " << excgridIntegral;

    // Against theory, at an accuracy the refinement check below justifies.
    EXPECT_NEAR(qcxIntegral, exact, kAccuracyTolerance) << "qcx integral " << qcxIntegral;
    EXPECT_NEAR(excgridIntegral, exact, kAccuracyTolerance)
        << "excgrid integral " << excgridIntegral;

    // The total-weight sum, asserted only as a roundoff bound: it is ~5.7e11
    // and its disagreement between the stacks is pure summation order, which is
    // why the integral above is the quantity this test uses.
    EXPECT_LE(std::abs(qcxWeight - excgridWeight), 1e-12 * qcxWeight)
        << "qcx total " << qcxWeight << " excgrid total " << excgridWeight;
}

TEST(QuadratureCrossValidationTest, TheAccuracyFloorIsQuadratureErrorNotAssemblyError) {
    // The reading that makes the tolerance above defensible: if the residual
    // against the analytic value is this quadrature's own error, refining the
    // grid must shrink it.  An assembly error would not move with the grid.
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());
    const std::array<double, 3> center = Midpoint(*molecule);
    const double exact = std::pow(kPi / kGaussianExponent, 1.5);

    const auto coarse = MolecularGrid::Create(*molecule, kRadialPoints, kAngularPoints, kAlpha);
    const auto refined =
        MolecularGrid::Create(*molecule, kRefinedRadialPoints, kRefinedAngularPoints, kAlpha);
    ASSERT_TRUE(coarse.has_value());
    ASSERT_TRUE(refined.has_value());

    const auto integrate = [&center](const MolecularGrid& grid) {
        double value = 0.0;

        for (std::size_t point = 0; point < grid.Size(); ++point)
        {
            value += grid.Weight(point) * Gaussian(grid.Point(point), center);
        }

        return value;
    };

    const double coarseResidual = std::abs(integrate(*coarse) - exact);
    const double refinedResidual = std::abs(integrate(*refined) - exact);

    EXPECT_LT(refinedResidual, coarseResidual)
        << "coarse residual " << coarseResidual << " refined residual " << refinedResidual
        << ": the residual does not shrink with the grid, so it is not quadrature error";
}

} // namespace qcx::grid
