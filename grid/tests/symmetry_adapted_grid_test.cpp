// The symmetry-adapted angular rule, from [DaulDaul1997], built as the
// group average of the certified Lebedev rule: invariance under the group,
// exactness inherited from the base rule, the octahedral fixtures where the
// two rules must coincide, and the radial and partition paths, which the
// angular adaptation must leave alone.

#include "corpus.hpp"
#include "excgrid/geometry.hpp"
#include "excgrid/grid.hpp"
#include "large_molecules.hpp"
#include "qcx/grid/angular_grid.hpp"
#include "qcx/grid/atomic_grid.hpp"
#include "qcx/grid/partition_function.hpp"
#include "qcx/grid/radial_grid.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/symmetry/detection.hpp"
#include "qcx/symmetry/point_group_name.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iomanip>
#include <numbers>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qcx::grid {
namespace {

using symmetry::PointGroupName;

using Matrix3 = std::array<double, 9>;
using Point3 = std::array<double, 3>;

constexpr double kTwoPi = 2.0 * std::numbers::pi;
constexpr double kNodeTolerance = 1e-9;
constexpr double kWeightTolerance = 1e-12;
constexpr std::size_t kNoIndex = static_cast<std::size_t>(-1);

Point3 Apply(const Matrix3& matrix, const Point3& point) {
    return Point3{matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2],
                  matrix[3] * point[0] + matrix[4] * point[1] + matrix[5] * point[2],
                  matrix[6] * point[0] + matrix[7] * point[1] + matrix[8] * point[2]};
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (a, b) reads as a comparison.
bool SamePoint(const Point3& a, const Point3& b) {
    return std::abs(a[0] - b[0]) < kNodeTolerance && std::abs(a[1] - b[1]) < kNodeTolerance &&
           std::abs(a[2] - b[2]) < kNodeTolerance;
}

// The detector reports each verified element once, with the maximal rotation
// order per axis; the closure over these matrices is the full group.
std::vector<Matrix3> OperationsOf(const symmetry::SymmetryAnalysis& analysis) {
    std::vector<Matrix3> operations;
    operations.reserve(analysis.elements.size());

    for (const symmetry::SymmetryElement& element : analysis.elements)
    {
        Eigen::Matrix3d matrix = Eigen::Matrix3d::Identity();
        const Eigen::Vector3d axis = element.axis;

        switch (element.kind)
        {
        case symmetry::OperationKind::kIdentity:
            break;
        case symmetry::OperationKind::kInversion:
            matrix = -Eigen::Matrix3d::Identity();
            break;
        case symmetry::OperationKind::kRotation:
            matrix = Eigen::AngleAxisd(kTwoPi / static_cast<double>(element.order), axis)
                         .toRotationMatrix();
            break;
        case symmetry::OperationKind::kSigma:
            matrix = Eigen::Matrix3d::Identity() - 2.0 * axis * axis.transpose();
            break;
        case symmetry::OperationKind::kImproper:
            matrix = (Eigen::Matrix3d::Identity() - 2.0 * axis * axis.transpose()) *
                     Eigen::AngleAxisd(kTwoPi / static_cast<double>(element.order), axis)
                         .toRotationMatrix();
            break;
        }

        Matrix3 raw{};

        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                raw[3 * row + column] = matrix(row, column);
            }
        }

        operations.push_back(raw);
    }

    return operations;
}

using NodeKey = std::array<long long, 3>;

NodeKey KeyOf(const Point3& point) {
    constexpr double kScale = 1e8;
    return NodeKey{std::llround(point[0] * kScale),
                   std::llround(point[1] * kScale),
                   std::llround(point[2] * kScale)};
}

struct NodeKeyHash {
    std::size_t operator()(const NodeKey& key) const noexcept {
        std::size_t hash = static_cast<std::size_t>(key[0]) * 73856093u;
        hash ^= static_cast<std::size_t>(key[1]) * 19349663u;
        hash ^= static_cast<std::size_t>(key[2]) * 83492791u;

        return hash;
    }
};

// A hash index over a node set, so the invariance check is linear in the
// point count rather than quadratic.
class NodeIndex {
public:
    explicit NodeIndex(const std::vector<Point3>& nodes) : _nodes(nodes) {
        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            _buckets[KeyOf(nodes[i])].push_back(i);
        }
    }

    std::size_t Find(const Point3& query) const {
        const NodeKey key = KeyOf(query);

        for (long long dx = -1; dx <= 1; ++dx)
        {
            for (long long dy = -1; dy <= 1; ++dy)
            {
                for (long long dz = -1; dz <= 1; ++dz)
                {
                    const auto bucket =
                        _buckets.find(NodeKey{key[0] + dx, key[1] + dy, key[2] + dz});

                    if (bucket == _buckets.end())
                    {
                        continue;
                    }

                    for (const std::size_t index : bucket->second)
                    {
                        if (SamePoint(_nodes[index], query))
                        {
                            return index;
                        }
                    }
                }
            }
        }

        return kNoIndex;
    }

private:
    std::vector<Point3> _nodes;
    std::unordered_map<NodeKey, std::vector<std::size_t>, NodeKeyHash> _buckets;
};

std::vector<Point3> NodesOf(const SymmetryAdaptedAngularGrid& rule) {
    std::vector<Point3> nodes(rule.Size());

    for (std::size_t i = 0; i < rule.Size(); ++i)
    {
        nodes[i] = rule.Point(i);
    }

    return nodes;
}

std::vector<Point3> NodesOf(const AngularGrid& rule) {
    std::vector<Point3> nodes(rule.Size());

    for (std::size_t i = 0; i < rule.Size(); ++i)
    {
        nodes[i] = rule.Point(i);
    }

    return nodes;
}

// Every orbit carries one per-point weight, the point count is the sum of
// the multiplicities, and the weights sum to the sphere area.
void ExpectWellFormed(const SymmetryAdaptedAngularGrid& rule) {
    double totalWeight = 0.0;

    for (std::size_t i = 0; i < rule.Size(); ++i)
    {
        const Point3& point = rule.Point(i);
        const double norm =
            std::sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
        EXPECT_NEAR(norm, 1.0, 1e-12) << "node " << i;
        // A certified size can carry a few negative weights (74, 230 and 266
        // in the shipped tables); the adaptation inherits them, so the sign
        // belongs to the base rule and is not asserted here.
        EXPECT_TRUE(std::isfinite(rule.Weight(i))) << "node " << i;
        totalWeight += rule.Weight(i);
    }

    EXPECT_NEAR(totalWeight, 4.0 * std::numbers::pi, 1e-11);

    std::vector<std::size_t> counts(rule.OrbitCount(), 0);
    std::vector<double> weightSums(rule.OrbitCount(), 0.0);
    std::vector<double> firstWeight(rule.OrbitCount(), 0.0);

    for (std::size_t i = 0; i < rule.Size(); ++i)
    {
        const std::size_t orbit = rule.OrbitOf(i);
        ASSERT_LT(orbit, rule.OrbitCount()) << "node " << i;

        if (counts[orbit] == 0)
        {
            firstWeight[orbit] = rule.Weight(i);
        } else
        {
            EXPECT_NEAR(rule.Weight(i), firstWeight[orbit], kWeightTolerance)
                << "node " << i << " of orbit " << orbit;
        }

        ++counts[orbit];
        weightSums[orbit] += rule.Weight(i);
    }

    for (std::size_t orbit = 0; orbit < rule.OrbitCount(); ++orbit)
    {
        const AngularOrbit& data = rule.Orbit(orbit);
        EXPECT_GT(data.multiplicity, 0u) << "orbit " << orbit;
        EXPECT_EQ(counts[orbit], data.multiplicity) << "orbit " << orbit;
        EXPECT_NEAR(weightSums[orbit], data.weight, kWeightTolerance) << "orbit " << orbit;

        const double norm = std::sqrt(data.representative[0] * data.representative[0] +
                                      data.representative[1] * data.representative[1] +
                                      data.representative[2] * data.representative[2]);
        EXPECT_NEAR(norm, 1.0, 1e-12) << "orbit " << orbit;

        // The representative is the point the orbit was created from, so it
        // is one of the stored nodes.
        bool found = false;

        for (std::size_t i = 0; i < rule.Size(); ++i)
        {
            if (rule.OrbitOf(i) == orbit && rule.Point(i) == data.representative)
            {
                found = true;
                break;
            }
        }

        EXPECT_TRUE(found) << "orbit " << orbit;
    }
}

void ExpectInvariant(const SymmetryAdaptedAngularGrid& rule,
                     const std::vector<Matrix3>& operations) {
    const std::vector<Point3> nodes = NodesOf(rule);
    const NodeIndex index(nodes);

    for (const Matrix3& operation : operations)
    {
        for (std::size_t i = 0; i < rule.Size(); ++i)
        {
            const std::size_t image = index.Find(Apply(operation, rule.Point(i)));
            ASSERT_NE(image, kNoIndex) << "node " << i << " has no image under the operation";
            EXPECT_NEAR(rule.Weight(image), rule.Weight(i), kWeightTolerance)
                << "node " << i << " changed weight under the operation";
        }
    }
}

// The exact integral of x^a y^b z^c over the unit sphere.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (a, b, c) is the exponent triple.
double ExactMoment(int a, int b, int c) {
    if (a % 2 != 0 || b % 2 != 0 || c % 2 != 0)
    {
        return 0.0;
    }

    const double total = static_cast<double>(a + b + c);

    return 2.0 * std::tgamma(0.5 * static_cast<double>(a + 1)) *
           std::tgamma(0.5 * static_cast<double>(b + 1)) *
           std::tgamma(0.5 * static_cast<double>(c + 1)) / std::tgamma(0.5 * (total + 3.0));
}

// The largest |rule moment - exact moment| over every monomial of total
// degree <= rule.Degree().  Both quadrature classes expose the same four
// members, so the same measure serves the adapted rule and the base rule.
template <typename Rule> double MaxMomentResidual(const Rule& rule) {
    const std::size_t degree = rule.Degree();
    const std::size_t size = rule.Size();
    std::vector<double> powerX(size * (degree + 1));
    std::vector<double> powerY(size * (degree + 1));
    std::vector<double> powerZ(size * (degree + 1));

    for (std::size_t i = 0; i < size; ++i)
    {
        const Point3& point = rule.Point(i);
        double px = 1.0;
        double py = 1.0;
        double pz = 1.0;

        for (std::size_t e = 0; e <= degree; ++e)
        {
            powerX[i * (degree + 1) + e] = px;
            powerY[i * (degree + 1) + e] = py;
            powerZ[i * (degree + 1) + e] = pz;
            px *= point[0];
            py *= point[1];
            pz *= point[2];
        }
    }

    double worst = 0.0;

    for (std::size_t a = 0; a <= degree; ++a)
    {
        for (std::size_t b = 0; a + b <= degree; ++b)
        {
            for (std::size_t c = 0; a + b + c <= degree; ++c)
            {
                double sum = 0.0;

                for (std::size_t i = 0; i < size; ++i)
                {
                    sum += rule.Weight(i) * powerX[i * (degree + 1) + a] *
                           powerY[i * (degree + 1) + b] * powerZ[i * (degree + 1) + c];
                }

                worst = std::max(worst,
                                 std::abs(sum - ExactMoment(static_cast<int>(a),
                                                            static_cast<int>(b),
                                                            static_cast<int>(c))));
            }
        }
    }

    return worst;
}

// A record value for a measured residual: six decimal places would round
// every certified number to zero.
std::string Scientific(double value) {
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(2) << value;

    return stream.str();
}

std::size_t FiniteGroupOrder(PointGroupName group) {
    switch (group)
    {
    case PointGroupName::kC1:
        return 1;
    case PointGroupName::kCi:
    case PointGroupName::kCs:
        return 2;
    case PointGroupName::kC2:
        return 2;
    case PointGroupName::kC3:
        return 3;
    case PointGroupName::kC4:
        return 4;
    case PointGroupName::kC5:
        return 5;
    case PointGroupName::kC6:
        return 6;
    case PointGroupName::kC7:
        return 7;
    case PointGroupName::kC8:
        return 8;
    case PointGroupName::kC2v:
    case PointGroupName::kC2h:
    case PointGroupName::kS4:
        return 4;
    case PointGroupName::kC3v:
    case PointGroupName::kC3h:
    case PointGroupName::kS6:
        return 6;
    case PointGroupName::kC4v:
    case PointGroupName::kC4h:
    case PointGroupName::kS8:
        return 8;
    case PointGroupName::kC5v:
    case PointGroupName::kC5h:
        return 10;
    case PointGroupName::kC6v:
    case PointGroupName::kC6h:
        return 12;
    case PointGroupName::kC7v:
    case PointGroupName::kC7h:
        return 14;
    case PointGroupName::kC8v:
    case PointGroupName::kC8h:
        return 16;
    case PointGroupName::kD2:
        return 4;
    case PointGroupName::kD3:
        return 6;
    case PointGroupName::kD4:
        return 8;
    case PointGroupName::kD5:
        return 10;
    case PointGroupName::kD6:
        return 12;
    case PointGroupName::kD7:
        return 14;
    case PointGroupName::kD8:
        return 16;
    case PointGroupName::kD2h:
    case PointGroupName::kD2d:
        return 8;
    case PointGroupName::kTd:
    case PointGroupName::kTh:
    case PointGroupName::kO:
        return 24;
    case PointGroupName::kD3h:
    case PointGroupName::kD3d:
        return 12;
    case PointGroupName::kD4h:
    case PointGroupName::kD4d:
        return 16;
    case PointGroupName::kD5h:
    case PointGroupName::kD5d:
        return 20;
    case PointGroupName::kD6h:
    case PointGroupName::kD6d:
        return 24;
    case PointGroupName::kD7h:
    case PointGroupName::kD7d:
        return 28;
    case PointGroupName::kD8h:
    case PointGroupName::kD8d:
        return 32;
    case PointGroupName::kT:
        return 12;
    case PointGroupName::kOh:
        return 48;
    case PointGroupName::kI:
        return 60;
    case PointGroupName::kIh:
        return 120;
    case PointGroupName::kCInfV:
    case PointGroupName::kDInfH:
        return 0;
    }

    return 0;
}

// A hexagonal ring of six equivalent carbons with one fluorine on the
// sixfold axis: the axial atom removes the horizontal mirror and every
// twofold axis, leaving exactly C6v (order 12).  The coordinates are
// constructed from full-precision trigonometry, so the C6 images match the
// detector's tolerance rather than a rounded coordinate table's.
Result<molecule::Molecule> MakeC6vSynthetic() {
    std::vector<molecule::Atom> atoms;
    std::vector<std::vector<double>> rows;

    for (int k = 0; k < 6; ++k)
    {
        const double theta = static_cast<double>(k) * std::numbers::pi / 3.0;
        rows.push_back({1.39 * std::cos(theta), 1.39 * std::sin(theta), 0.0});
        atoms.push_back(molecule::Atom{"C", 6, 0.0});
    }

    atoms.push_back(molecule::Atom{"F", 9, 0.0});
    rows.push_back({0.0, 0.0, 1.35});

    auto coordinates = memory::Tensor<double, 2, backend::CpuTag>::Create({rows.size(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            (*coordinates)(i, d) = rows[i][d] * molecule::kAngstromToBohr;
        }
    }

    return molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

} // namespace

// ---------------------------------------------------------------------------
// Behaviour per group class: every named group builds a well-formed,
// invariant rule, and the orbit multiplicities divide the group order.
// ---------------------------------------------------------------------------

TEST(SymmetryAdaptedGridTest, EveryNamedGroupBuildsAWellFormedRule) {
    const std::vector<PointGroupName> groups = {
        PointGroupName::kC1,    PointGroupName::kCi,  PointGroupName::kCs,  PointGroupName::kC2,
        PointGroupName::kC3,    PointGroupName::kC4,  PointGroupName::kC5,  PointGroupName::kC6,
        PointGroupName::kC7,    PointGroupName::kC8,  PointGroupName::kC2v, PointGroupName::kC3v,
        PointGroupName::kC4v,   PointGroupName::kC5v, PointGroupName::kC6v, PointGroupName::kC7v,
        PointGroupName::kC8v,   PointGroupName::kC2h, PointGroupName::kC3h, PointGroupName::kC4h,
        PointGroupName::kC5h,   PointGroupName::kC6h, PointGroupName::kC7h, PointGroupName::kC8h,
        PointGroupName::kD2,    PointGroupName::kD3,  PointGroupName::kD4,  PointGroupName::kD5,
        PointGroupName::kD6,    PointGroupName::kD7,  PointGroupName::kD8,  PointGroupName::kD2h,
        PointGroupName::kD3h,   PointGroupName::kD4h, PointGroupName::kD5h, PointGroupName::kD6h,
        PointGroupName::kD7h,   PointGroupName::kD8h, PointGroupName::kD2d, PointGroupName::kD3d,
        PointGroupName::kD4d,   PointGroupName::kD5d, PointGroupName::kD6d, PointGroupName::kD7d,
        PointGroupName::kD8d,   PointGroupName::kS4,  PointGroupName::kS6,  PointGroupName::kS8,
        PointGroupName::kT,     PointGroupName::kTd,  PointGroupName::kTh,  PointGroupName::kO,
        PointGroupName::kOh,    PointGroupName::kI,   PointGroupName::kIh,  PointGroupName::kCInfV,
        PointGroupName::kDInfH,
    };

    EXPECT_EQ(groups.size(), 57u);

    for (const PointGroupName group : groups)
    {
        const auto rule = SymmetryAdaptedAngularGrid::Create(74, group);
        ASSERT_TRUE(rule.has_value()) << "group index " << static_cast<int>(group);
        ExpectWellFormed(*rule);

        const std::string name(symmetry::ToString(group));
        RecordProperty(name + ".points", std::to_string(rule->Size()));
        RecordProperty(name + ".orbits", std::to_string(rule->OrbitCount()));

        const std::size_t order = FiniteGroupOrder(group);

        if (order == 0)
        {
            continue;
        }

        std::size_t generic = 0;

        for (std::size_t orbit = 0; orbit < rule->OrbitCount(); ++orbit)
        {
            const std::size_t multiplicity = rule->Orbit(orbit).multiplicity;
            EXPECT_EQ(order % multiplicity, 0u)
                << "group index " << static_cast<int>(group) << " orbit " << orbit;
            generic = std::max(generic, multiplicity);
        }

        // The largest orbit is the group order, with one exception that is a
        // property of the base set rather than of the rule: every node of a
        // Lebedev set lies on at least one mirror plane of O_h - (a, a, b)
        // on a diagonal mirror, (a, 0, b) on a coordinate plane, a z = 0 node
        // on the horizontal one - so no node is free and the largest orbit
        // is |O_h| / 2.
        const std::size_t freeOrbit = group == PointGroupName::kOh ? order / 2 : order;
        EXPECT_EQ(generic, freeOrbit) << "group index " << static_cast<int>(group);
    }
}

TEST(SymmetryAdaptedGridTest, OctahedralCompatibleGroupsReturnTheLebedevRuleUnchanged) {
    const std::vector<PointGroupName> groups = {
        PointGroupName::kC1,  PointGroupName::kCi,  PointGroupName::kCs,  PointGroupName::kC2,
        PointGroupName::kC2v, PointGroupName::kC2h, PointGroupName::kD2,  PointGroupName::kD2h,
        PointGroupName::kC4,  PointGroupName::kC4v, PointGroupName::kC4h, PointGroupName::kS4,
        PointGroupName::kD4,  PointGroupName::kD4h, PointGroupName::kD2d, PointGroupName::kT,
        PointGroupName::kTh,  PointGroupName::kTd,  PointGroupName::kO,   PointGroupName::kOh,
    };

    const auto base = AngularGrid::Create(74);
    ASSERT_TRUE(base.has_value());

    for (const PointGroupName group : groups)
    {
        const auto rule = SymmetryAdaptedAngularGrid::Create(74, group);
        ASSERT_TRUE(rule.has_value());
        ASSERT_EQ(rule->Size(), base->Size());
        EXPECT_EQ(rule->Degree(), base->Degree());
        ExpectWellFormed(*rule);

        double worst = 0.0;

        for (std::size_t i = 0; i < base->Size(); ++i)
        {
            const Point3 point = base->Point(i);
            worst = std::max(worst, std::abs(rule->Point(i)[0] - point[0]));
            worst = std::max(worst, std::abs(rule->Point(i)[1] - point[1]));
            worst = std::max(worst, std::abs(rule->Point(i)[2] - point[2]));
            worst = std::max(worst, std::abs(rule->Weight(i) - base->Weight(i)));
        }

        EXPECT_EQ(worst, 0.0) << "group index " << static_cast<int>(group);
    }
}

// The D-nd class is where a mirror placed on a twofold axis instead of on its
// bisector silently produces D-n-h (same order, same count of improper
// elements).  The reference here is independent of the library recipe: an
// S-2n axis about z plus one horizontal C2 generate D-nd by definition.
TEST(SymmetryAdaptedGridTest, DiagonalGroupsMatchTheIndependentlyBuiltDndOperations) {
    struct Case {
        PointGroupName group;
        int order;
    };

    const std::vector<Case> cases = {{PointGroupName::kD2d, 2},
                                     {PointGroupName::kD3d, 3},
                                     {PointGroupName::kD4d, 4},
                                     {PointGroupName::kD5d, 5},
                                     {PointGroupName::kD6d, 6},
                                     {PointGroupName::kD7d, 7},
                                     {PointGroupName::kD8d, 8}};

    for (const Case& item : cases)
    {
        const auto rule = SymmetryAdaptedAngularGrid::Create(74, item.group);
        ASSERT_TRUE(rule.has_value()) << "group index " << static_cast<int>(item.group);

        // S-2n about z: rotate by 2 pi / (2n) = pi / n, then reflect in z = 0.
        const double angle = std::numbers::pi / static_cast<double>(item.order);
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const Matrix3 s2n{c, -s, 0.0, s, c, 0.0, 0.0, 0.0, -1.0};
        const Matrix3 c2x{1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0};

        ExpectInvariant(*rule, {s2n, c2x});
    }
}

TEST(SymmetryAdaptedGridTest, EmptyOperationSetReproducesTheLebedevRule) {
    const std::vector<Matrix3> noOperations = {};
    const auto rule = SymmetryAdaptedAngularGrid::CreateFromOperations(50, noOperations);
    ASSERT_TRUE(rule.has_value());

    const auto base = AngularGrid::Create(50);
    ASSERT_TRUE(base.has_value());
    ASSERT_EQ(rule->Size(), base->Size());

    for (std::size_t i = 0; i < base->Size(); ++i)
    {
        EXPECT_EQ(rule->Point(i), base->Point(i));
        EXPECT_EQ(rule->Weight(i), base->Weight(i));
    }
}

// ---------------------------------------------------------------------------
// Exactness: the group average keeps every moment the base rule integrates.
// ---------------------------------------------------------------------------

TEST(SymmetryAdaptedGridTest, MomentsAreExactToTheBaseDegreeForEveryClass) {
    const std::vector<PointGroupName> groups = {
        PointGroupName::kC1,  PointGroupName::kCi,    PointGroupName::kCs,    PointGroupName::kC2,
        PointGroupName::kC2v, PointGroupName::kD2h,   PointGroupName::kC3v,   PointGroupName::kC3h,
        PointGroupName::kD3d, PointGroupName::kD6h,   PointGroupName::kS6,    PointGroupName::kS8,
        PointGroupName::kT,   PointGroupName::kTd,    PointGroupName::kOh,    PointGroupName::kI,
        PointGroupName::kIh,  PointGroupName::kCInfV, PointGroupName::kDInfH,
    };

    for (const PointGroupName group : groups)
    {
        const auto rule = SymmetryAdaptedAngularGrid::Create(110, group);
        ASSERT_TRUE(rule.has_value()) << "group index " << static_cast<int>(group);
        ASSERT_EQ(rule->Degree(), 17u) << "group index " << static_cast<int>(group);
        const double residual = MaxMomentResidual(*rule);
        RecordProperty(std::string(symmetry::ToString(group)) + ".moment_residual",
                       Scientific(residual));
        EXPECT_LT(residual, 1e-10) << "group index " << static_cast<int>(group);
    }
}

// ---------------------------------------------------------------------------
// The octahedral fixtures: the molecular groups of water, methane and sulfur
// hexafluoride are subgroups of O_h, so the adapted rule must BE the Lebedev
// rule - same points, same weights, same degree.
// ---------------------------------------------------------------------------

TEST(SymmetryAdaptedGridTest, OctahedralFixturesReproduceTheLebedevPointCountAndWeights) {
    struct Fixture {
        const char* name;
        Result<molecule::Molecule> (*make)();
    };

    const std::vector<Fixture> fixtures = {
        {"water/C2v", symmetry::testing::MakeWater},
        {"methane/Td", symmetry::testing::MakeMethane},
        {"synthetic T", symmetry::testing::MakeTSynthetic},
        {"synthetic Th", symmetry::testing::MakeThSynthetic},
        {"sulfur hexafluoride/Oh", symmetry::testing::MakeSulfurHexafluoride},
    };

    const auto base = AngularGrid::Create(302);
    ASSERT_TRUE(base.has_value());

    for (const Fixture& fixture : fixtures)
    {
        const auto molecule = fixture.make();
        ASSERT_TRUE(molecule.has_value()) << fixture.name;

        const symmetry::SymmetryAnalysis analysis = symmetry::DetectPointGroup(*molecule);
        const std::vector<Matrix3> operations = OperationsOf(analysis);
        const auto rule = SymmetryAdaptedAngularGrid::CreateFromOperations(302, operations);
        ASSERT_TRUE(rule.has_value()) << fixture.name;
        ASSERT_EQ(rule->Size(), base->Size()) << fixture.name;
        ASSERT_EQ(rule->Degree(), base->Degree()) << fixture.name;

        double worst = 0.0;

        for (std::size_t i = 0; i < base->Size(); ++i)
        {
            const Point3 point = base->Point(i);
            worst = std::max(worst, std::abs(rule->Point(i)[0] - point[0]));
            worst = std::max(worst, std::abs(rule->Point(i)[1] - point[1]));
            worst = std::max(worst, std::abs(rule->Point(i)[2] - point[2]));
            worst = std::max(worst, std::abs(rule->Weight(i) - base->Weight(i)));
        }

        EXPECT_EQ(worst, 0.0) << fixture.name;

        // The quadrature accuracy of the two rules, measured by the same
        // worst moment residual over the inherited degree: because the rules
        // coincide point for point, the two numbers must coincide too.
        const double adaptedResidual = MaxMomentResidual(*rule);
        const double lebedevResidual = MaxMomentResidual(*base);
        EXPECT_EQ(adaptedResidual, lebedevResidual) << fixture.name;

        RecordProperty(std::string(fixture.name) + ".points", std::to_string(rule->Size()));
        RecordProperty(std::string(fixture.name) + ".degree", std::to_string(rule->Degree()));
        RecordProperty(std::string(fixture.name) + ".max_deviation_from_lebedev",
                       Scientific(worst));
        RecordProperty(std::string(fixture.name) + ".max_moment_residual_adapted",
                       Scientific(adaptedResidual));
        RecordProperty(std::string(fixture.name) + ".max_moment_residual_lebedev",
                       Scientific(lebedevResidual));
        ExpectInvariant(*rule, operations);
    }
}

// ---------------------------------------------------------------------------
// The non-octahedral fixtures: ammonia (C3v) and benzene (D6h) are invariant
// under the molecule's own operations, while the plain Lebedev rule is not -
// which is the defect this rule exists to remove.
// ---------------------------------------------------------------------------

TEST(SymmetryAdaptedGridTest, TrigonalAndHexagonalFixturesAreInvariantWhereLebedevIsNot) {
    struct Fixture {
        const char* name;
        Result<molecule::Molecule> (*make)();
        PointGroupName group;
    };

    const std::vector<Fixture> fixtures = {
        {"ammonia/C3v", symmetry::testing::MakeAmmonia, PointGroupName::kC3v},
        {"benzene/D6h", symmetry::testing::MakeBenzene, PointGroupName::kD6h},
    };

    const auto base = AngularGrid::Create(110);
    ASSERT_TRUE(base.has_value());
    const std::vector<Point3> baseNodes = NodesOf(*base);
    const NodeIndex baseIndex(baseNodes);

    for (const Fixture& fixture : fixtures)
    {
        const auto molecule = fixture.make();
        ASSERT_TRUE(molecule.has_value()) << fixture.name;

        const symmetry::SymmetryAnalysis analysis = symmetry::DetectPointGroup(*molecule);
        EXPECT_EQ(analysis.group, fixture.group) << fixture.name;
        const std::vector<Matrix3> operations = OperationsOf(analysis);

        const auto rule = SymmetryAdaptedAngularGrid::CreateFromOperations(110, operations);
        ASSERT_TRUE(rule.has_value()) << fixture.name;
        ExpectWellFormed(*rule);
        ExpectInvariant(*rule, operations);

        // The Lebedev rule itself fails the same invariance check: at least
        // one node image is not a base node.
        std::size_t missing = 0;

        for (const Matrix3& operation : operations)
        {
            for (const Point3& node : baseNodes)
            {
                if (baseIndex.Find(Apply(operation, node)) == kNoIndex)
                {
                    ++missing;
                }
            }
        }

        EXPECT_GT(missing, 0u) << fixture.name;
        const double residual = MaxMomentResidual(*rule);
        RecordProperty(std::string(fixture.name) + ".points", std::to_string(rule->Size()));
        RecordProperty(std::string(fixture.name) + ".degree", std::to_string(rule->Degree()));
        RecordProperty(std::string(fixture.name) + ".lebedev_missing_images",
                       std::to_string(missing));
        RecordProperty(std::string(fixture.name) + ".max_moment_residual", Scientific(residual));
        EXPECT_LT(residual, 1e-10) << fixture.name;
    }
}

// ---------------------------------------------------------------------------
// The remaining two classes the plan names, C6v and Ih, each with a molecule:
// a synthetic sixfold ring (C6v) and the C60 truncated icosahedron (Ih).  Both
// are outside the octahedral group, so the rule is the orbit rule over the
// molecule's own operations, and its point count is strictly larger than the
// base rule's.
// ---------------------------------------------------------------------------

TEST(SymmetryAdaptedGridTest, HexagonalAndIcosahedralFixturesAreDetectedAndAdapted) {
    struct Fixture {
        const char* name;
        Result<molecule::Molecule> (*make)();
        PointGroupName group;
    };

    const std::vector<Fixture> fixtures = {
        {"synthetic C6v", MakeC6vSynthetic, PointGroupName::kC6v},
        {"buckminsterfullerene/Ih",
         molecule::testing::MakeBuckminsterfullerene,
         PointGroupName::kIh},
    };

    const auto base = AngularGrid::Create(74);
    ASSERT_TRUE(base.has_value());

    for (const Fixture& fixture : fixtures)
    {
        const auto molecule = fixture.make();
        ASSERT_TRUE(molecule.has_value()) << fixture.name;

        const symmetry::SymmetryAnalysis analysis = symmetry::DetectPointGroup(*molecule);
        EXPECT_EQ(analysis.group, fixture.group) << fixture.name;
        const std::vector<Matrix3> operations = OperationsOf(analysis);

        const auto rule = SymmetryAdaptedAngularGrid::CreateFromOperations(74, operations);
        ASSERT_TRUE(rule.has_value()) << fixture.name;
        ExpectWellFormed(*rule);
        ExpectInvariant(*rule, operations);
        EXPECT_GT(rule->Size(), base->Size()) << fixture.name;

        const double residual = MaxMomentResidual(*rule);
        RecordProperty(std::string(fixture.name) + ".points", std::to_string(rule->Size()));
        RecordProperty(std::string(fixture.name) + ".orbits", std::to_string(rule->OrbitCount()));
        RecordProperty(std::string(fixture.name) + ".degree", std::to_string(rule->Degree()));
        RecordProperty(std::string(fixture.name) + ".max_moment_residual", Scientific(residual));
        EXPECT_LT(residual, 1e-10) << fixture.name;
    }
}

// ---------------------------------------------------------------------------
// The linear groups: the polar quadrature is kept in full and the azimuth is
// sampled on an even ring wider than the exactness degree.
// ---------------------------------------------------------------------------

TEST(SymmetryAdaptedGridTest, LinearGroupsKeepEveryPolarNodeAndWedgeTheAzimuth) {
    const std::vector<PointGroupName> groups = {PointGroupName::kCInfV, PointGroupName::kDInfH};

    const auto base = AngularGrid::Create(302);
    ASSERT_TRUE(base.has_value());
    const std::size_t azimuths = base->Degree() + 1; // 30, even
    ASSERT_EQ(azimuths % 2, 0u);

    std::vector<double> baseHeights;

    for (std::size_t i = 0; i < base->Size(); ++i)
    {
        const double z = base->Point(i)[2];
        bool seen = false;

        for (const double height : baseHeights)
        {
            if (std::abs(height - z) < kNodeTolerance)
            {
                seen = true;
                break;
            }
        }

        if (!seen)
        {
            baseHeights.push_back(z);
        }
    }

    for (const PointGroupName group : groups)
    {
        const auto rule = SymmetryAdaptedAngularGrid::Create(302, group);
        ASSERT_TRUE(rule.has_value());
        ExpectWellFormed(*rule);
        EXPECT_EQ(rule->Degree(), base->Degree());

        // Ring structure: every stored point sits at one of the base heights,
        // each height carries either the two poles or a full ring, and every
        // height of the base rule survives.
        std::vector<std::size_t> ringSize(baseHeights.size(), 0);
        std::vector<double> ringWeight(baseHeights.size(), 0.0);

        for (std::size_t i = 0; i < rule->Size(); ++i)
        {
            const double z = rule->Point(i)[2];
            std::size_t height = kNoIndex;

            for (std::size_t h = 0; h < baseHeights.size(); ++h)
            {
                if (std::abs(baseHeights[h] - z) < kNodeTolerance)
                {
                    height = h;
                    break;
                }
            }

            ASSERT_NE(height, kNoIndex) << "node " << i << " is at a new height";
            ++ringSize[height];
            ringWeight[height] += rule->Weight(i);
        }

        for (std::size_t h = 0; h < baseHeights.size(); ++h)
        {
            const bool pole = std::abs(std::abs(baseHeights[h]) - 1.0) < kNodeTolerance;
            EXPECT_EQ(ringSize[h], pole ? 1u : azimuths) << "height " << baseHeights[h];

            double expected = 0.0;

            for (std::size_t i = 0; i < base->Size(); ++i)
            {
                if (std::abs(base->Point(i)[2] - baseHeights[h]) < kNodeTolerance)
                {
                    expected += base->Weight(i);
                }
            }

            EXPECT_NEAR(ringWeight[h], expected, 1e-12) << "height " << baseHeights[h];
        }

        // Invariance under the finite subgroup the rule does carry: the
        // azimuthal rotation and the vertical mirror (and inversion for
        // D-infinity-h).
        Matrix3 rotation{};
        const double angle = kTwoPi / static_cast<double>(azimuths);
        rotation[0] = std::cos(angle);
        rotation[1] = -std::sin(angle);
        rotation[3] = std::sin(angle);
        rotation[4] = std::cos(angle);
        rotation[8] = 1.0;
        const Matrix3 mirror{-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        std::vector<Matrix3> operations = {rotation, mirror};

        if (group == PointGroupName::kDInfH)
        {
            operations.push_back(Matrix3{-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0});
        }

        ExpectInvariant(*rule, operations);
        EXPECT_LT(MaxMomentResidual(*rule), 1e-10);
    }
}

// The linear row of the taxonomy, with a molecule: the detector names the
// group, and that named group builds the full-polar ring rule exactly.
TEST(SymmetryAdaptedGridTest, LinearFixturesDetectTheLinearGroupAndTakeTheRingRule) {
    struct Fixture {
        const char* name;
        Result<molecule::Molecule> (*make)();
        PointGroupName group;
    };

    const std::vector<Fixture> fixtures = {
        {"hydrogen fluoride/C-infinity-v",
         symmetry::testing::MakeHydrogenFluoride,
         PointGroupName::kCInfV},
        {"carbon dioxide/D-infinity-h",
         symmetry::testing::MakeCarbonDioxide,
         PointGroupName::kDInfH},
    };

    const auto base = AngularGrid::Create(110);
    ASSERT_TRUE(base.has_value());
    const std::size_t azimuths = base->Degree() + 1; // 18, even
    ASSERT_EQ(azimuths % 2, 0u);

    for (const Fixture& fixture : fixtures)
    {
        const auto molecule = fixture.make();
        ASSERT_TRUE(molecule.has_value()) << fixture.name;

        const symmetry::SymmetryAnalysis analysis = symmetry::DetectPointGroup(*molecule);
        EXPECT_EQ(analysis.group, fixture.group) << fixture.name;

        const auto rule = SymmetryAdaptedAngularGrid::Create(110, fixture.group);
        ASSERT_TRUE(rule.has_value()) << fixture.name;
        ExpectWellFormed(*rule);
        EXPECT_EQ(rule->Degree(), base->Degree()) << fixture.name;

        // The finite subgroup the rule carries at this degree: the 18-fold
        // azimuthal rotation and the vertical mirror (and inversion for
        // D-infinity-h).
        const double angle = kTwoPi / static_cast<double>(azimuths);
        const Matrix3 rotation{std::cos(angle),
                               -std::sin(angle),
                               0.0,
                               std::sin(angle),
                               std::cos(angle),
                               0.0,
                               0.0,
                               0.0,
                               1.0};
        const Matrix3 mirror{-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        std::vector<Matrix3> operations = {rotation, mirror};

        if (fixture.group == PointGroupName::kDInfH)
        {
            operations.push_back(Matrix3{-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0});
        }

        ExpectInvariant(*rule, operations);
        const double residual = MaxMomentResidual(*rule);
        RecordProperty(std::string(fixture.name) + ".points", std::to_string(rule->Size()));
        RecordProperty(std::string(fixture.name) + ".max_moment_residual", Scientific(residual));
        EXPECT_LT(residual, 1e-10) << fixture.name;
    }
}

// ---------------------------------------------------------------------------
// The radial quadrature is untouched: both atomic-grid factories share the
// same radial builder and the same r^2 product loop, so every radial shell
// keeps its node and its total radial weight exactly.
// ---------------------------------------------------------------------------

TEST(SymmetryAdaptedGridTest, RadialQuadratureIsUntouchedByTheAngularSource) {
    const std::vector<PointGroupName> groups = {
        PointGroupName::kOh,
        PointGroupName::kC3v,
        PointGroupName::kD6h,
        PointGroupName::kDInfH,
    };

    constexpr std::size_t kRadialPoints = 75;
    constexpr std::size_t kAngularPoints = 302;
    constexpr double kAlpha = 0.5;
    const std::array<double, 3> center{0.1, -0.2, 0.3};

    const auto plain = AtomicGrid::Create(kRadialPoints, kAngularPoints, kAlpha, center);
    ASSERT_TRUE(plain.has_value());
    ASSERT_EQ(plain->Size() % kAngularPoints, 0u);

    const auto radial = RadialGrid::Create(kRadialPoints, kAlpha);
    ASSERT_TRUE(radial.has_value());

    for (const PointGroupName group : groups)
    {
        const auto adapted =
            AtomicGrid::CreateSymmetryAdapted(kRadialPoints, kAngularPoints, group, kAlpha, center);
        ASSERT_TRUE(adapted.has_value()) << "group index " << static_cast<int>(group);

        const auto rule = SymmetryAdaptedAngularGrid::Create(kAngularPoints, group);
        ASSERT_TRUE(rule.has_value());
        ASSERT_EQ(adapted->Size(), kRadialPoints * rule->Size());
        EXPECT_EQ(adapted->Center(), center);

        // Each shell: the radius from the center equals the radial node, and
        // the shell weight equals 4 pi times the radial weight times r^2.
        for (std::size_t shell = 0; shell < kRadialPoints; ++shell)
        {
            const std::array<double, 3> offset = {
                adapted->Point(shell * rule->Size())[0] - center[0],
                adapted->Point(shell * rule->Size())[1] - center[1],
                adapted->Point(shell * rule->Size())[2] - center[2]};
            const double radius =
                std::sqrt(offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]);
            EXPECT_NEAR(radius, radial->Point(shell), 1e-12) << "shell " << shell;

            // The plain-grid shell at the same radial index has the same
            // radius; both factories read the same radial rule.
            const std::array<double, 3> reference = {
                plain->Point(shell * kAngularPoints)[0] - center[0],
                plain->Point(shell * kAngularPoints)[1] - center[1],
                plain->Point(shell * kAngularPoints)[2] - center[2]};
            const double referenceRadius =
                std::sqrt(reference[0] * reference[0] + reference[1] * reference[1] +
                          reference[2] * reference[2]);
            EXPECT_NEAR(radius, referenceRadius, 1e-12) << "shell " << shell;

            double shellWeight = 0.0;
            double referenceWeight = 0.0;

            for (std::size_t j = 0; j < rule->Size(); ++j)
            {
                shellWeight += adapted->Weight(shell * rule->Size() + j);
            }

            for (std::size_t j = 0; j < kAngularPoints; ++j)
            {
                referenceWeight += plain->Weight(shell * kAngularPoints + j);
            }

            EXPECT_NEAR(
                shellWeight, referenceWeight, 1e-11 * std::max(1.0, std::abs(referenceWeight)))
                << "shell " << shell;
            const double analytic = 4.0 * std::numbers::pi * radial->Weight(shell) *
                                    radial->Point(shell) * radial->Point(shell);
            EXPECT_NEAR(shellWeight, analytic, 1e-11 * std::max(1.0, std::abs(analytic)))
                << "shell " << shell;
        }
    }
}

// The partition function consumes a point and a molecule and nothing else, so
// the angular source cannot enter it.  The check: the partition of unity
// holds on every adapted-grid point of a real molecule, exactly as it does on
// the plain grid's points.
TEST(SymmetryAdaptedGridTest, PartitionFunctionIsAFunctionOfPositionOnly) {
    const auto molecule = symmetry::testing::MakeMethane();
    ASSERT_TRUE(molecule.has_value());

    const auto adapted =
        AtomicGrid::CreateSymmetryAdapted(25, 74, PointGroupName::kTd, 0.5, {0.0, 0.0, 0.0});
    ASSERT_TRUE(adapted.has_value());

    const auto plain = AtomicGrid::Create(25, 74, 0.5, {0.0, 0.0, 0.0});
    ASSERT_TRUE(plain.has_value());

    // Td is an octahedral-compatible group: the adapted grid IS the plain
    // grid, so the partition weights coincide point for point.
    ASSERT_EQ(adapted->Size(), plain->Size());

    for (std::size_t i = 0; i < adapted->Size(); i += 37)
    {
        EXPECT_EQ(adapted->Point(i), plain->Point(i));
        EXPECT_EQ(adapted->Weight(i), plain->Weight(i));

        const std::vector<double> weights = BeckePartitionWeights(adapted->Point(i), *molecule);
        ASSERT_EQ(weights.size(), molecule->AtomCount());
        double sum = 0.0;

        for (const double weight : weights)
        {
            sum += weight;
        }

        EXPECT_NEAR(sum, 1.0, 1e-12) << "point " << i;
    }

    // A non-octahedral rule is a different point set, and the same partition
    // function still holds the partition of unity on it.
    const auto trigonal =
        AtomicGrid::CreateSymmetryAdapted(25, 74, PointGroupName::kC3v, 0.5, {0.0, 0.0, 0.0});
    ASSERT_TRUE(trigonal.has_value());
    ASSERT_NE(trigonal->Size(), plain->Size());

    for (std::size_t i = 0; i < trigonal->Size(); i += 53)
    {
        const std::vector<double> weights = BeckePartitionWeights(trigonal->Point(i), *molecule);
        double sum = 0.0;

        for (const double weight : weights)
        {
            sum += weight;
        }

        EXPECT_NEAR(sum, 1.0, 1e-12) << "point " << i;
    }
}

// ---------------------------------------------------------------------------
// The block-grid boundary: the rule crosses it as nodes and weights, so the
// excgrid build consumes a symmetry-adapted rule without knowing point
// groups.  The count is exact because the trim is disabled.
// ---------------------------------------------------------------------------

TEST(SymmetryAdaptedGridTest, BlockGridConsumesTheAdaptedRuleAsNodesAndWeights) {
    const auto rule = SymmetryAdaptedAngularGrid::Create(50, PointGroupName::kC3v);
    ASSERT_TRUE(rule.has_value());

    std::vector<Point3> nodes(rule->Size());
    std::vector<double> weights(rule->Size());

    for (std::size_t i = 0; i < rule->Size(); ++i)
    {
        nodes[i] = rule->Point(i);
        weights[i] = rule->Weight(i);
    }

    excgrid::Geometry geometry;
    geometry.atoms.push_back({8, {0.0, 0.0, 0.0}});
    geometry.atoms.push_back({1, {0.0, 0.0, 1.8}});

    excgrid::GridParams params;
    params.radialPoints = 12;
    params.alpha = 0.5;
    params.trimWeight = 0.0;
    params.blockTarget = 64;

    const excgrid::AngularRuleView view{nodes, weights};
    const auto grid = excgrid::BlockGrid::Create(geometry, params, view);
    ASSERT_TRUE(grid.has_value());
    EXPECT_EQ(grid->TotalPointCount(), 2u * params.radialPoints * rule->Size());

    double totalWeight = 0.0;

    for (const excgrid::Block& block : grid->Blocks())
    {
        for (const double weight : block.weights)
        {
            totalWeight += weight;
        }
    }

    EXPECT_GT(totalWeight, 0.0);

    const excgrid::AngularRuleView empty{};
    EXPECT_FALSE(excgrid::BlockGrid::Create(geometry, params, empty).has_value());

    // A view whose weights do not match its nodes is rejected rather than
    // silently truncating or reading past the end.
    const excgrid::AngularRuleView mismatched{
        nodes, std::span<const double>(weights.data(), weights.size() - 1)};
    EXPECT_FALSE(excgrid::BlockGrid::Create(geometry, params, mismatched).has_value());
}

// ---------------------------------------------------------------------------
// Error contract.
// ---------------------------------------------------------------------------

TEST(SymmetryAdaptedGridTest, InvalidPointCountAndOpenOperationSetAreRejected) {
    const auto badSize = SymmetryAdaptedAngularGrid::Create(100, PointGroupName::kC3v);
    EXPECT_FALSE(badSize.has_value());

    // A rotation by an angle that is not a rational fraction of 2 pi never
    // closes; the 240-element cap must reject it rather than run forever.
    Matrix3 irrational{};
    const double angle = 1.0;
    irrational[0] = std::cos(angle);
    irrational[1] = -std::sin(angle);
    irrational[3] = std::sin(angle);
    irrational[4] = std::cos(angle);
    irrational[8] = 1.0;
    const std::vector<Matrix3> operations = {irrational};
    const auto open = SymmetryAdaptedAngularGrid::CreateFromOperations(26, operations);
    EXPECT_FALSE(open.has_value());
}

} // namespace qcx::grid
