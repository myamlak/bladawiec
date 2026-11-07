#include "qcx/grid/angular_grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qcx::grid {

// (offset, size, degree) are the Lebedev table lookup triple: the rule
// row offset, the point count, and the polynomial degree.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
AngularGrid::AngularGrid(std::size_t offset, std::size_t size, std::size_t degree) noexcept :
    _offset(offset), _size(size), _degree(degree) {}

Result<AngularGrid> AngularGrid::Create(std::size_t pointCount) {
    for (std::size_t i = 0; i < internal::kLebedevSizeCount; ++i)
    {
        if (internal::kLebedevSizes[i] == pointCount)
        {
            return AngularGrid(
                internal::kLebedevOffsets[i], pointCount, internal::kLebedevDegrees[i]);
        }
    }

    return std::unexpected(Error{ErrorCode::kInvalidArgument,
                                 "not a Lebedev point count: " + std::to_string(pointCount)});
}

std::array<double, 3> AngularGrid::Point(std::size_t index) const noexcept {
    // The tables are flat x,y,z triples; node k lives at 3 * (offset + k).
    const std::size_t base = 3 * (_offset + index);
    return std::array<double, 3>{internal::kLebedevPoints[base],
                                 internal::kLebedevPoints[base + 1],
                                 internal::kLebedevPoints[base + 2]};
}

double AngularGrid::Weight(std::size_t index) const noexcept {
    return internal::kLebedevWeights[_offset + index];
}

namespace {

using symmetry::PointGroupName;

using Matrix3 = std::array<double, 9>;
using Point3 = std::array<double, 3>;

constexpr double kTwoPi = 2.0 * std::numbers::pi;
constexpr double kMatrixTolerance = 1e-9;
constexpr double kPointTolerance = 1e-9;
constexpr double kHashScale = 1e8;
constexpr std::size_t kMaxGroupOrder = 240;
constexpr std::size_t kNoIndex = static_cast<std::size_t>(-1);

Matrix3 IdentityMatrix() noexcept {
    return Matrix3{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
}

Matrix3 InversionMatrix() noexcept {
    return Matrix3{-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0};
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (a, b) reads as a product.
Matrix3 Multiply(const Matrix3& a, const Matrix3& b) noexcept {
    Matrix3 result{};

    for (std::size_t row = 0; row < 3; ++row)
    {
        for (std::size_t column = 0; column < 3; ++column)
        {
            result[3 * row + column] = a[3 * row] * b[column] + a[3 * row + 1] * b[3 + column] +
                                       a[3 * row + 2] * b[6 + column];
        }
    }

    return result;
}

Point3 Apply(const Matrix3& matrix, const Point3& point) noexcept {
    return Point3{matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2],
                  matrix[3] * point[0] + matrix[4] * point[1] + matrix[5] * point[2],
                  matrix[6] * point[0] + matrix[7] * point[1] + matrix[8] * point[2]};
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (a, b) reads as a comparison.
bool SameMatrix(const Matrix3& a, const Matrix3& b) noexcept {
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(a[i] - b[i]) >= kMatrixTolerance)
        {
            return false;
        }
    }

    return true;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (a, b) reads as a comparison.
bool SamePoint(const Point3& a, const Point3& b) noexcept {
    return std::abs(a[0] - b[0]) < kPointTolerance && std::abs(a[1] - b[1]) < kPointTolerance &&
           std::abs(a[2] - b[2]) < kPointTolerance;
}

// Rodrigues' rotation matrix about a normalised axis.
Matrix3 Rotation(const Point3& axis, double angle) noexcept {
    const double norm = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    const double x = axis[0] / norm;
    const double y = axis[1] / norm;
    const double z = axis[2] / norm;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double t = 1.0 - c;

    return Matrix3{t * x * x + c,
                   t * x * y - s * z,
                   t * x * z + s * y,
                   t * x * y + s * z,
                   t * y * y + c,
                   t * y * z - s * x,
                   t * x * z - s * y,
                   t * y * z + s * x,
                   t * z * z + c};
}

// Householder reflection in the plane through the origin with the given normal.
Matrix3 Reflection(const Point3& normal) noexcept {
    const double norm =
        std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    const double x = normal[0] / norm;
    const double y = normal[1] / norm;
    const double z = normal[2] / norm;

    return Matrix3{1.0 - 2.0 * x * x,
                   -2.0 * x * y,
                   -2.0 * x * z,
                   -2.0 * x * y,
                   1.0 - 2.0 * y * y,
                   -2.0 * y * z,
                   -2.0 * x * z,
                   -2.0 * y * z,
                   1.0 - 2.0 * z * z};
}

// S_n = C_n followed by the reflection in the plane perpendicular to the axis.
Matrix3 Improper(const Point3& axis, double angle) noexcept {
    return Multiply(Reflection(axis), Rotation(axis, angle));
}

// The symmetry families the canonical recipes are built from.
enum class GroupFamily {
    kIdentity,
    kInversion,
    kMirror,
    kCn,
    kCnv,
    kCnh,
    kDn,
    kDnh,
    kDnd,
    kSn,
    kT,
    kTd,
    kTh,
    kO,
    kOh,
    kI,
    kIh,
    kLinear,
};

struct GroupRecipe {
    GroupFamily family;
    int order;
};

// The canonical orientation: principal axis z, C2(x) and the vertical planes
// placed by the standard convention.  Linear groups have no finite recipe.
GroupRecipe RecipeOf(PointGroupName group) noexcept {
    switch (group)
    {
    case PointGroupName::kC1:
        return {GroupFamily::kIdentity, 1};
    case PointGroupName::kCi:
        return {GroupFamily::kInversion, 1};
    case PointGroupName::kCs:
        return {GroupFamily::kMirror, 1};
    case PointGroupName::kC2:
        return {GroupFamily::kCn, 2};
    case PointGroupName::kC3:
        return {GroupFamily::kCn, 3};
    case PointGroupName::kC4:
        return {GroupFamily::kCn, 4};
    case PointGroupName::kC5:
        return {GroupFamily::kCn, 5};
    case PointGroupName::kC6:
        return {GroupFamily::kCn, 6};
    case PointGroupName::kC7:
        return {GroupFamily::kCn, 7};
    case PointGroupName::kC8:
        return {GroupFamily::kCn, 8};
    case PointGroupName::kC2v:
        return {GroupFamily::kCnv, 2};
    case PointGroupName::kC3v:
        return {GroupFamily::kCnv, 3};
    case PointGroupName::kC4v:
        return {GroupFamily::kCnv, 4};
    case PointGroupName::kC5v:
        return {GroupFamily::kCnv, 5};
    case PointGroupName::kC6v:
        return {GroupFamily::kCnv, 6};
    case PointGroupName::kC7v:
        return {GroupFamily::kCnv, 7};
    case PointGroupName::kC8v:
        return {GroupFamily::kCnv, 8};
    case PointGroupName::kC2h:
        return {GroupFamily::kCnh, 2};
    case PointGroupName::kC3h:
        return {GroupFamily::kCnh, 3};
    case PointGroupName::kC4h:
        return {GroupFamily::kCnh, 4};
    case PointGroupName::kC5h:
        return {GroupFamily::kCnh, 5};
    case PointGroupName::kC6h:
        return {GroupFamily::kCnh, 6};
    case PointGroupName::kC7h:
        return {GroupFamily::kCnh, 7};
    case PointGroupName::kC8h:
        return {GroupFamily::kCnh, 8};
    case PointGroupName::kD2:
        return {GroupFamily::kDn, 2};
    case PointGroupName::kD3:
        return {GroupFamily::kDn, 3};
    case PointGroupName::kD4:
        return {GroupFamily::kDn, 4};
    case PointGroupName::kD5:
        return {GroupFamily::kDn, 5};
    case PointGroupName::kD6:
        return {GroupFamily::kDn, 6};
    case PointGroupName::kD7:
        return {GroupFamily::kDn, 7};
    case PointGroupName::kD8:
        return {GroupFamily::kDn, 8};
    case PointGroupName::kD2h:
        return {GroupFamily::kDnh, 2};
    case PointGroupName::kD3h:
        return {GroupFamily::kDnh, 3};
    case PointGroupName::kD4h:
        return {GroupFamily::kDnh, 4};
    case PointGroupName::kD5h:
        return {GroupFamily::kDnh, 5};
    case PointGroupName::kD6h:
        return {GroupFamily::kDnh, 6};
    case PointGroupName::kD7h:
        return {GroupFamily::kDnh, 7};
    case PointGroupName::kD8h:
        return {GroupFamily::kDnh, 8};
    case PointGroupName::kD2d:
        return {GroupFamily::kDnd, 2};
    case PointGroupName::kD3d:
        return {GroupFamily::kDnd, 3};
    case PointGroupName::kD4d:
        return {GroupFamily::kDnd, 4};
    case PointGroupName::kD5d:
        return {GroupFamily::kDnd, 5};
    case PointGroupName::kD6d:
        return {GroupFamily::kDnd, 6};
    case PointGroupName::kD7d:
        return {GroupFamily::kDnd, 7};
    case PointGroupName::kD8d:
        return {GroupFamily::kDnd, 8};
    case PointGroupName::kS4:
        return {GroupFamily::kSn, 4};
    case PointGroupName::kS6:
        return {GroupFamily::kSn, 6};
    case PointGroupName::kS8:
        return {GroupFamily::kSn, 8};
    case PointGroupName::kT:
        return {GroupFamily::kT, 1};
    case PointGroupName::kTd:
        return {GroupFamily::kTd, 1};
    case PointGroupName::kTh:
        return {GroupFamily::kTh, 1};
    case PointGroupName::kO:
        return {GroupFamily::kO, 1};
    case PointGroupName::kOh:
        return {GroupFamily::kOh, 1};
    case PointGroupName::kI:
        return {GroupFamily::kI, 1};
    case PointGroupName::kIh:
        return {GroupFamily::kIh, 1};
    case PointGroupName::kCInfV:
    case PointGroupName::kDInfH:
        return {GroupFamily::kLinear, 1};
    }

    return {GroupFamily::kIdentity, 1};
}

std::vector<Matrix3> CanonicalGenerators(const GroupRecipe& recipe) {
    const Point3 zAxis{0.0, 0.0, 1.0};
    const Point3 xAxis{1.0, 0.0, 0.0};
    const Point3 bodyDiagonal{1.0, 1.0, 1.0};
    const double angle = kTwoPi / static_cast<double>(recipe.order);
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0; // The icosahedral vertex.

    switch (recipe.family)
    {
    case GroupFamily::kIdentity:
        return {};
    case GroupFamily::kInversion:
        return {InversionMatrix()};
    case GroupFamily::kMirror:
        return {Reflection(xAxis)};
    case GroupFamily::kCn:
        return {Rotation(zAxis, angle)};
    case GroupFamily::kCnv:
        return {Rotation(zAxis, angle), Reflection(xAxis)};
    case GroupFamily::kCnh:
        return {Rotation(zAxis, angle), Reflection(zAxis)};
    case GroupFamily::kDn:
        return {Rotation(zAxis, angle), Rotation(xAxis, std::numbers::pi)};
    case GroupFamily::kDnh:
        return {Rotation(zAxis, angle), Rotation(xAxis, std::numbers::pi), Reflection(zAxis)};
    case GroupFamily::kDnd: {
        // The diagonal mirrors CONTAIN the bisectors of the horizontal
        // twofold axes, which themselves lie at multiples of pi / n from x.
        // A plane with normal azimuth a contains the directions a +/- pi/2,
        // so the bisector condition fixes a at pi/(2n) for even n and at
        // pi/n for odd n; the wrong parity gives D-n-h, not D-n-d.
        double azimuth = std::numbers::pi / (2.0 * static_cast<double>(recipe.order));

        if (recipe.order % 2 != 0)
        {
            azimuth += 0.5 * std::numbers::pi;
        }

        return {Rotation(zAxis, angle),
                Rotation(xAxis, std::numbers::pi),
                Reflection(Point3{std::cos(azimuth), std::sin(azimuth), 0.0})};
    }

    case GroupFamily::kSn:
        return {Improper(zAxis, angle)};
    case GroupFamily::kT:
        return {Rotation(zAxis, std::numbers::pi),
                Rotation(bodyDiagonal, 2.0 * std::numbers::pi / 3.0)};
    case GroupFamily::kTd:
        return {Improper(zAxis, std::numbers::pi / 2.0),
                Rotation(bodyDiagonal, 2.0 * std::numbers::pi / 3.0)};
    case GroupFamily::kTh:
        return {Rotation(zAxis, std::numbers::pi),
                Rotation(bodyDiagonal, 2.0 * std::numbers::pi / 3.0),
                InversionMatrix()};
    case GroupFamily::kO:
        return {Rotation(zAxis, std::numbers::pi / 2.0),
                Rotation(bodyDiagonal, 2.0 * std::numbers::pi / 3.0)};
    case GroupFamily::kOh:
        return {Rotation(zAxis, std::numbers::pi / 2.0),
                Rotation(bodyDiagonal, 2.0 * std::numbers::pi / 3.0),
                InversionMatrix()};
    case GroupFamily::kI:
        // A fivefold axis through an icosahedron vertex and a threefold axis
        // through a face centre; the pair generates the rotation group of
        // order 60 (verified in the tests by the group order).
        return {Rotation(Point3{0.0, 1.0, phi}, kTwoPi / 5.0),
                Rotation(bodyDiagonal, 2.0 * std::numbers::pi / 3.0)};
    case GroupFamily::kIh:
        return {Rotation(Point3{0.0, 1.0, phi}, kTwoPi / 5.0),
                Rotation(bodyDiagonal, 2.0 * std::numbers::pi / 3.0),
                InversionMatrix()};
    case GroupFamily::kLinear:
        return {};
    }

    return {};
}

// The groups whose canonical operations are all elements of the octahedral
// group.  The certified Lebedev set is invariant under each of them, so the
// adapted rule is the base rule itself.
bool IsOctahedralCompatible(PointGroupName group) noexcept {
    switch (group)
    {
    case PointGroupName::kC1:
    case PointGroupName::kCi:
    case PointGroupName::kCs:
    case PointGroupName::kC2:
    case PointGroupName::kC2v:
    case PointGroupName::kC2h:
    case PointGroupName::kD2:
    case PointGroupName::kD2h:
    case PointGroupName::kC4:
    case PointGroupName::kC4v:
    case PointGroupName::kC4h:
    case PointGroupName::kS4:
    case PointGroupName::kD4:
    case PointGroupName::kD4h:
    case PointGroupName::kD2d:
    case PointGroupName::kT:
    case PointGroupName::kTh:
    case PointGroupName::kTd:
    case PointGroupName::kO:
    case PointGroupName::kOh:
        return true;
    default:
        return false;
    }
}

bool ContainsMatrix(const std::vector<Matrix3>& group, const Matrix3& matrix) noexcept {
    for (const Matrix3& element : group)
    {
        if (SameMatrix(element, matrix))
        {
            return true;
        }
    }

    return false;
}

Result<std::vector<Matrix3>> CloseGroup(std::span<const Matrix3> generators) {
    std::vector<Matrix3> group{IdentityMatrix()};
    bool changed = true;

    while (changed)
    {
        changed = false;
        const std::size_t count = group.size();

        for (std::size_t i = 0; i < count; ++i)
        {
            for (const Matrix3& generator : generators)
            {
                const Matrix3 product = Multiply(generator, group[i]);

                if (ContainsMatrix(group, product))
                {
                    continue;
                }

                if (group.size() >= kMaxGroupOrder)
                {
                    return std::unexpected(Error{ErrorCode::kInvalidArgument,
                                                 "the operations generate more than " +
                                                     std::to_string(kMaxGroupOrder) +
                                                     " group elements"});
                }

                group.push_back(product);
                changed = true;
            }
        }
    }

    return group;
}

bool GroupPreservesLebedev(std::size_t pointCount, const std::vector<Matrix3>& group) {
    const auto base = AngularGrid::Create(pointCount);

    if (!base.has_value())
    {
        return false;
    }

    for (const Matrix3& element : group)
    {
        for (std::size_t i = 0; i < base->Size(); ++i)
        {
            const Point3 image = Apply(element, base->Point(i));
            bool found = false;

            for (std::size_t j = 0; j < base->Size(); ++j)
            {
                if (SamePoint(base->Point(j), image))
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                return false;
            }
        }
    }

    return true;
}

} // namespace

// The identity path: the base set is already invariant, so the rule is the
// certified Lebedev rule; only the orbit decomposition is new.
Result<SymmetryAdaptedAngularGrid> SymmetryAdaptedAngularGrid::BuildIdentityRule(
    std::size_t pointCount, const std::vector<Matrix3>& group) {
    const auto base = AngularGrid::Create(pointCount);

    if (!base.has_value())
    {
        return std::unexpected(base.error());
    }

    const std::size_t size = base->Size();
    std::vector<Point3> points(size);
    std::vector<double> weights(size);

    for (std::size_t i = 0; i < size; ++i)
    {
        points[i] = base->Point(i);
        weights[i] = base->Weight(i);
    }

    std::vector<std::size_t> pointOrbit(size, kNoIndex);
    std::vector<AngularOrbit> orbits;

    for (std::size_t i = 0; i < size; ++i)
    {
        if (pointOrbit[i] != kNoIndex)
        {
            continue;
        }

        std::vector<std::size_t> members;

        for (const Matrix3& element : group)
        {
            const Point3 image = Apply(element, points[i]);
            std::size_t member = kNoIndex;

            for (std::size_t j = 0; j < size; ++j)
            {
                if (SamePoint(points[j], image))
                {
                    member = j;
                    break;
                }
            }

            if (member == kNoIndex)
            {
                return std::unexpected(Error{ErrorCode::kInternalError,
                                             "the group does not map the Lebedev set onto itself"});
            }

            if (std::find(members.begin(), members.end(), member) == members.end())
            {
                members.push_back(member);
            }
        }

        const std::size_t orbit = orbits.size();
        double weight = 0.0;

        for (const std::size_t member : members)
        {
            pointOrbit[member] = orbit;
            weight += weights[member];
        }

        orbits.push_back(AngularOrbit{points[i], members.size(), weight});
    }

    return SymmetryAdaptedAngularGrid(std::move(points),
                                      std::move(weights),
                                      std::move(pointOrbit),
                                      std::move(orbits),
                                      base->Degree());
}

namespace {

using PointKey = std::array<long long, 3>;

struct PointKeyHash {
    std::size_t operator()(const PointKey& key) const noexcept {
        // A 64-bit mix of the three quantized coordinates; only equality
        // matters, so a cheap combining hash is enough.
        std::size_t hash = static_cast<std::size_t>(key[0]) * 73856093u;

        hash ^= static_cast<std::size_t>(key[1]) * 19349663u;
        hash ^= static_cast<std::size_t>(key[2]) * 83492791u;

        return hash;
    }
};

PointKey Quantize(const Point3& point) noexcept {
    return PointKey{std::llround(point[0] * kHashScale),
                    std::llround(point[1] * kHashScale),
                    std::llround(point[2] * kHashScale)};
}

using PointTable = std::unordered_map<PointKey, std::vector<std::size_t>, PointKeyHash>;

std::size_t FindPoint(const PointTable& table,
                      const std::vector<Point3>& points,
                      const Point3& query) {
    const PointKey key = Quantize(query);

    for (long long dx = -1; dx <= 1; ++dx)
    {
        for (long long dy = -1; dy <= 1; ++dy)
        {
            for (long long dz = -1; dz <= 1; ++dz)
            {
                const PointKey probe{key[0] + dx, key[1] + dy, key[2] + dz};
                const auto entry = table.find(probe);

                if (entry == table.end())
                {
                    continue;
                }

                for (const std::size_t index : entry->second)
                {
                    if (SamePoint(points[index], query))
                    {
                        return index;
                    }
                }
            }
        }
    }

    return kNoIndex;
}

} // namespace

// The orbit path: every base node is replaced by its group orbit, and the
// node weight is shared equally over the orbit.  Averaging the certified
// base rule over the group preserves every moment the base integrates, so
// the adapted rule is exact to the base degree by construction.
Result<SymmetryAdaptedAngularGrid> SymmetryAdaptedAngularGrid::BuildOrbitRule(
    std::size_t pointCount, const std::vector<Matrix3>& group) {
    const auto base = AngularGrid::Create(pointCount);

    if (!base.has_value())
    {
        return std::unexpected(base.error());
    }

    std::vector<Point3> points;
    std::vector<double> weights;
    std::vector<std::size_t> pointOrbit;
    std::vector<AngularOrbit> orbits;
    PointTable table;

    for (std::size_t i = 0; i < base->Size(); ++i)
    {
        const Point3 node = base->Point(i);
        const double nodeWeight = base->Weight(i);
        std::vector<Point3> orbitPoints;

        for (const Matrix3& element : group)
        {
            const Point3 image = Apply(element, node);
            bool seen = false;

            for (const Point3& existing : orbitPoints)
            {
                if (SamePoint(existing, image))
                {
                    seen = true;
                    break;
                }
            }

            if (!seen)
            {
                orbitPoints.push_back(image);
            }
        }

        std::size_t orbit = kNoIndex;

        for (const Point3& orbitPoint : orbitPoints)
        {
            const std::size_t existing = FindPoint(table, points, orbitPoint);

            if (existing != kNoIndex)
            {
                orbit = pointOrbit[existing];
                break;
            }
        }

        if (orbit == kNoIndex)
        {
            orbit = orbits.size();
            orbits.push_back(AngularOrbit{orbitPoints.front(), orbitPoints.size(), 0.0});
        }

        const double share = nodeWeight / static_cast<double>(orbitPoints.size());

        for (const Point3& orbitPoint : orbitPoints)
        {
            std::size_t index = FindPoint(table, points, orbitPoint);

            if (index == kNoIndex)
            {
                index = points.size();
                points.push_back(orbitPoint);
                weights.push_back(0.0);
                pointOrbit.push_back(orbit);
                table[Quantize(orbitPoint)].push_back(index);
            }

            weights[index] += share;
        }

        orbits[orbit].weight += nodeWeight;
    }

    return SymmetryAdaptedAngularGrid(std::move(points),
                                      std::move(weights),
                                      std::move(pointOrbit),
                                      std::move(orbits),
                                      base->Degree());
}

// The linear path: the polar quadrature is kept in full - a ring per distinct
// z node, carrying the base rule's total weight at that height - and each
// ring is sampled at n equally spaced azimuths.  For n > degree the
// azimuthal trapezoid is exact for every harmonic of degree <= degree, so
// the rule is exact to the base degree; it is invariant under the n-fold
// azimuthal group and its mirrors, not under the continuous C-infinity axis,
// which no finite point set can be.
Result<SymmetryAdaptedAngularGrid> SymmetryAdaptedAngularGrid::BuildLinearRule(
    std::size_t pointCount) {
    const auto base = AngularGrid::Create(pointCount);

    if (!base.has_value())
    {
        return std::unexpected(base.error());
    }

    std::size_t azimuths = std::max<std::size_t>(base->Degree() + 1, 4);

    // An even count makes the ring closed under the vertical mirrors of the
    // linear groups (phi -> pi - phi) as well as under its rotations.
    if (azimuths % 2 != 0)
    {
        ++azimuths;
    }

    struct PolarRing {
        double z;
        double weight;
    };

    std::vector<PolarRing> rings;

    for (std::size_t i = 0; i < base->Size(); ++i)
    {
        const Point3 node = base->Point(i);
        bool found = false;

        for (PolarRing& ring : rings)
        {
            if (std::abs(ring.z - node[2]) < kPointTolerance)
            {
                ring.weight += base->Weight(i);
                found = true;
                break;
            }
        }

        if (!found)
        {
            rings.push_back(PolarRing{node[2], base->Weight(i)});
        }
    }

    std::vector<Point3> points;
    std::vector<double> weights;
    std::vector<std::size_t> pointOrbit;
    std::vector<AngularOrbit> orbits;

    for (const PolarRing& ring : rings)
    {
        const std::size_t orbit = orbits.size();
        const double radius = std::sqrt(std::max(0.0, 1.0 - ring.z * ring.z));

        if (radius < kPointTolerance)
        {
            points.push_back(Point3{0.0, 0.0, ring.z});
            weights.push_back(ring.weight);
            pointOrbit.push_back(orbit);
            orbits.push_back(AngularOrbit{points.back(), 1, ring.weight});
            continue;
        }

        const std::size_t first = points.size();

        for (std::size_t k = 0; k < azimuths; ++k)
        {
            const double phi = kTwoPi * static_cast<double>(k) / static_cast<double>(azimuths);
            points.push_back(Point3{radius * std::cos(phi), radius * std::sin(phi), ring.z});
            weights.push_back(ring.weight / static_cast<double>(azimuths));
            pointOrbit.push_back(orbit);
        }

        orbits.push_back(AngularOrbit{points[first], azimuths, ring.weight});
    }

    return SymmetryAdaptedAngularGrid(std::move(points),
                                      std::move(weights),
                                      std::move(pointOrbit),
                                      std::move(orbits),
                                      base->Degree());
}

// (points, weights, pointOrbit, orbits, degree) are the five facets of one
// constructed rule; the constructor stores them.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
SymmetryAdaptedAngularGrid::SymmetryAdaptedAngularGrid(std::vector<std::array<double, 3>> points,
                                                       std::vector<double> weights,
                                                       std::vector<std::size_t> pointOrbit,
                                                       std::vector<AngularOrbit> orbits,
                                                       std::size_t degree) noexcept :
    _points(std::move(points)), _weights(std::move(weights)), _pointOrbit(std::move(pointOrbit)),
    _orbits(std::move(orbits)), _degree(degree) {}

Result<SymmetryAdaptedAngularGrid> SymmetryAdaptedAngularGrid::Create(std::size_t pointCount,
                                                                      PointGroupName group) {
    const GroupRecipe recipe = RecipeOf(group);

    if (recipe.family == GroupFamily::kLinear)
    {
        return BuildLinearRule(pointCount);
    }

    const std::vector<Matrix3> generators = CanonicalGenerators(recipe);
    auto closure = CloseGroup(generators);

    if (!closure.has_value())
    {
        return std::unexpected(closure.error());
    }

    if (IsOctahedralCompatible(group))
    {
        return BuildIdentityRule(pointCount, *closure);
    }

    return BuildOrbitRule(pointCount, *closure);
}

Result<SymmetryAdaptedAngularGrid> SymmetryAdaptedAngularGrid::CreateFromOperations(
    std::size_t pointCount, std::span<const std::array<double, 9>> operations) {
    auto group = CloseGroup(operations);

    if (!group.has_value())
    {
        return std::unexpected(group.error());
    }

    if (GroupPreservesLebedev(pointCount, *group))
    {
        return BuildIdentityRule(pointCount, *group);
    }

    return BuildOrbitRule(pointCount, *group);
}

} // namespace qcx::grid
