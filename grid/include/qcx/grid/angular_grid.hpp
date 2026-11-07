#pragma once

#include "qcx/error.hpp"
#include "qcx/grid/internal/lebedev_tables.hpp"
#include "qcx/symmetry/point_group_name.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace qcx::grid {

/// \defgroup qcx-grid Grid module
/// Numerical integration grids: radial and angular quadratures, the
/// atomic and molecular product grids, the Becke fuzzy-cell partition,
/// and the AO-at-point evaluator.
/// \{

/// A Lebedev-Laikov angular quadrature on the unit sphere.
///
/// The point set is closed under the octahedral group and the weights
/// integrate every polynomial of total degree <= Degree() exactly.  The
/// tables are SOLVED for from these two defining conditions by
/// tools/grid/gen_lebedev.py (no copied tables) and certified per size to
/// max |moment residual| < 1e-60.  Weights are normalized so that the
/// weights sum to 4 pi, the area of the unit sphere - the angular
/// convention of the molecular product grid.
/// \ingroup qcx-grid
class AngularGrid {
public:
    /// The available point counts, smallest first.
    /// \ingroup qcx-grid
    static constexpr std::span<const std::size_t, internal::kLebedevSizeCount> kAvailableSizes =
        internal::kLebedevSizes;

    /// Creates the quadrature with the given point count.
    /// \param pointCount A size from kAvailableSizes.
    /// \returns The grid, or kInvalidArgument for a non-Lebedev count.
    /// \ingroup qcx-grid
    static Result<AngularGrid> Create(std::size_t pointCount);

    /// The number of quadrature nodes.
    /// \returns The node count.
    [[nodiscard]] std::size_t Size() const noexcept {
        return _size;
    }

    /// The algebraic degree of exactness: every polynomial of total degree
    /// <= Degree() integrates exactly.
    /// \returns The algebraic degree of exactness.
    [[nodiscard]] std::size_t Degree() const noexcept {
        return _degree;
    }

    /// The \p index-th node, a unit vector on the sphere.
    /// \param index The node index in [0, Size()).
    /// \returns The node as a unit vector on the sphere.
    [[nodiscard]] std::array<double, 3> Point(std::size_t index) const noexcept;

    /// The \p index-th quadrature weight; the weights sum to 4 pi.
    /// \param index The node index in [0, Size()).
    /// \returns The quadrature weight (the weights sum to 4 pi).
    [[nodiscard]] double Weight(std::size_t index) const noexcept;

private:
    AngularGrid(std::size_t offset, std::size_t size, std::size_t degree) noexcept;

    // Start of this grid's nodes in the flat internal tables.
    std::size_t _offset;
    std::size_t _size;
    std::size_t _degree;
};

/// One orbit of a symmetry-adapted angular rule: every point of the orbit
/// carries the same weight, and the orbit as a whole carries \ref weight.
///
/// The multiplicity is the number of distinct points the group produces
/// from the representative, so the per-point weight is weight / multiplicity
/// and an irreducible-wedge evaluation may replace the whole orbit by the
/// representative with the full orbit weight.
/// \ingroup qcx-grid
struct AngularOrbit {
    std::array<double, 3> representative; ///< The orbit's first point, a unit vector.
    std::size_t multiplicity; ///< Number of distinct points in the orbit.
    double weight; ///< Total quadrature weight of the orbit.
};

/// A symmetry-adapted angular quadrature on the unit sphere, built from the
/// orbit construction of [DaulDaul1997].
///
/// A rule with nodes p_k and weights w_k is adapted to a point group G when
/// the node set is a union of G-orbits and every point of an orbit carries
/// the same weight.  Such a rule is invariant under G by construction: the
/// reduction to the irreducible wedge and the invariance are the same act.
/// Daul and Daul construct rules of this form for an arbitrary point group,
/// identical to Lebedev for octahedral symmetry.  This implementation builds
/// them from the certified Lebedev rule: the orbit of each base node is
/// generated, and the base node's weight is shared equally over its orbit.
/// A base rule that carries a few negative weights, as the certified 74,
/// 230 and 266-point rules do, passes them on unchanged.  The construction
/// therefore reproduces the base rule's exactness degree
/// rather than assuming it - for any polynomial f of degree <= Degree(),
/// averaging the base rule over G leaves its integral unchanged.  The
/// published rules aim at the fewest points that reach a given order; this
/// construction does not minimise the point count, it trades that claim for
/// the exactness inherited from a certified base rule.
///
/// Only the angular nodes and their weights differ between rules: the radial
/// quadrature, the r^2 volume element of the product grid and the Becke
/// partition function are untouched by the adaptation, and the tests pin all
/// three.
///
/// Groups whose operations are all elements of the octahedral group need no
/// new rule: the Lebedev set is already invariant under them, and Create
/// returns it unchanged.  A chiral icosahedral or trigonal rule is invariant
/// under its full point group but not under an arbitrary rotation of the
/// molecule; a fixed quadrature cannot be rotationally invariant, which is
/// the orientation sensitivity measured for popular grids in
/// [BootsmaWheeler2019] and is a limitation, not a target.
/// \ingroup qcx-grid
class SymmetryAdaptedAngularGrid {
public:
    /// Builds the rule for a point group in its canonical orientation.
    ///
    /// The principal axis is the z axis (the five-fold axis of the
    /// icosahedral groups lies in the yz plane); mirrors and twofold axes
    /// are placed by the generator convention in the implementation.  For
    /// an octahedral-compatible group the certified Lebedev rule is
    /// returned unchanged.  The linear groups produce the polar-preserving
    /// azimuthal ring rule: the base rule's polar nodes and their total
    /// weights are kept, and each ring is sampled at n equally spaced
    /// azimuths, n > Degree().
    /// \param pointCount A Lebedev size (AngularGrid::kAvailableSizes).
    /// \param group The point group to adapt the rule to.
    /// \returns The rule, or kInvalidArgument for a non-Lebedev count.
    /// \ingroup qcx-grid
    static Result<SymmetryAdaptedAngularGrid> Create(std::size_t pointCount,
                                                     symmetry::PointGroupName group);

    /// Builds the rule from an explicit group action on three-space.
    ///
    /// The operations need not be closed; the closure under matrix
    /// multiplication is taken, capped at 240 elements.  Passing the
    /// molecule's own symmetry operations places the rule in the molecular
    /// frame, which is what a molecule not in canonical orientation needs.
    /// \param pointCount A Lebedev size (AngularGrid::kAvailableSizes).
    /// \param operations Row-major 3x3 orthogonal matrices, one per
    ///        generator or group element.
    /// \returns The rule, or kInvalidArgument for a non-Lebedev count or a
    ///          set generating more than 240 operations.
    /// \ingroup qcx-grid
    static Result<SymmetryAdaptedAngularGrid> CreateFromOperations(
        std::size_t pointCount, std::span<const std::array<double, 9>> operations);

    /// The number of quadrature nodes.
    /// \returns The node count.
    [[nodiscard]] std::size_t Size() const noexcept {
        return _points.size();
    }

    /// The algebraic degree of exactness inherited from the base rule.
    /// \returns The exactness degree.
    [[nodiscard]] std::size_t Degree() const noexcept {
        return _degree;
    }

    /// The \p index-th node, a unit vector on the sphere.
    /// \param index The node index in [0, Size()).
    /// \returns The node as a unit vector on the sphere.
    [[nodiscard]] const std::array<double, 3>& Point(std::size_t index) const noexcept {
        const std::array<double, 3>& point = _points[index];

        return point;
    }

    /// The \p index-th quadrature weight; the weights sum to 4 pi.
    /// \param index The node index in [0, Size()).
    /// \returns The quadrature weight (the weights sum to 4 pi).
    [[nodiscard]] double Weight(std::size_t index) const noexcept {
        const double weight = _weights[index];

        return weight;
    }

    /// The number of group orbits.
    /// \returns The orbit count.
    [[nodiscard]] std::size_t OrbitCount() const noexcept {
        return _orbits.size();
    }

    /// The \p orbit-th orbit.
    /// \param orbit The orbit index in [0, OrbitCount()).
    /// \returns The orbit's representative, multiplicity and total weight.
    [[nodiscard]] const AngularOrbit& Orbit(std::size_t orbit) const noexcept {
        const AngularOrbit& data = _orbits[orbit];

        return data;
    }

    /// The orbit owning node \p index.
    /// \param index The node index in [0, Size()).
    /// \returns The orbit index.
    [[nodiscard]] std::size_t OrbitOf(std::size_t index) const noexcept {
        const std::size_t orbit = _pointOrbit[index];

        return orbit;
    }

private:
    SymmetryAdaptedAngularGrid(std::vector<std::array<double, 3>> points,
                               std::vector<double> weights,
                               std::vector<std::size_t> pointOrbit,
                               std::vector<AngularOrbit> orbits,
                               std::size_t degree) noexcept;

    // The three construction paths: the base rule already invariant (copied
    // with its orbit decomposition), the orbit average, and the linear ring
    // rule.  The group is closed before they are called.
    static Result<SymmetryAdaptedAngularGrid> BuildIdentityRule(
        std::size_t pointCount, const std::vector<std::array<double, 9>>& group);

    static Result<SymmetryAdaptedAngularGrid> BuildOrbitRule(
        std::size_t pointCount, const std::vector<std::array<double, 9>>& group);

    static Result<SymmetryAdaptedAngularGrid> BuildLinearRule(std::size_t pointCount);

    std::vector<std::array<double, 3>> _points;
    std::vector<double> _weights;
    std::vector<std::size_t> _pointOrbit;
    std::vector<AngularOrbit> _orbits;
    std::size_t _degree;
};

/// \}

} // namespace qcx::grid
