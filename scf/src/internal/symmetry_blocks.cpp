// Implementation of the AO-space symmetry decomposition (symmetry_blocks.hpp).
//
// Math: the computational group is one of the eight Abelian point groups, so
// every element is an involution and its irreps are the sign patterns of the
// generators. For each group element we build the exact AO-space matrix A(g):
// the atom permutation it induces (greedy distance matching about the center
// of mass, like symmetry's own matcher) times the exact angular action on
// each shell. The angular action of a rotation/reflection/inversion matrix on
// the shell's functions (real spherical harmonics in the engine's m-order, or
// the z-slowest Cartesian monomials) is computed through an evaluation
// matrix: D = E^{-1} F with E and F the basis evaluated on sample points and
// on their images. The projector P(chi) = (1/|G|) sum_g chi(g) A(g) is an
// orthogonal projector onto the chi irrep subspace; its orthonormal range is
// one block of u. Because F commutes with every A(g) (the molecule is
// symmetric), U^T F U is block-diagonal up to integral noise.

#include "symmetry_blocks.hpp"

#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/molecule/mass_properties.hpp"
#include "qcx/symmetry/character_tables.hpp"
#include "scf_common.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numbers>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace qcx::scf::internal {
namespace {

constexpr double kPositionToleranceBohr = 1e-4; // Atom matching (detection default).
constexpr double kAxisParallelTolerance = 1e-6; // |dot| above this counts as parallel.
constexpr double kMatrixDedupTolerance = 1e-8; // Matrix equality within the group.
constexpr double kProjectorEigenvalueCutoff = 0.5; // Rank cutoff of a projector's spectrum.
constexpr double kAngularOrthogonalityTolerance = 1e-8; // Guard on the D-matrix construction.
constexpr double kMomentLinearityThreshold = 1e-6; // Mass properties (detection default).

// The generator-signs gate (SymmetryBlocks::generatorSigns): the realized
// generator action must reproduce a symmetrized column as +-1 times itself.
// Both operands are exact up to rounding - A(g) is an involution of an
// orthogonal representation and u's columns are orthonormal up to the
// projector eigensolve - so the residual is round-off and the repo's usual
// 1e-8 matrix gate applies; a larger one means the action is not a signed
// identity and the guard's reference would be a fiction.
constexpr double kGeneratorSignedActionTolerance = 1e-8;

// Strict-weak x/y/z ordering for the greedy matcher's sorted array.
bool LexLess(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    if (a.x() != b.x())
    {
        return a.x() < b.x();
    }

    if (a.y() != b.y())
    {
        return a.y() < b.y();
    }

    return a.z() < b.z();
}

// Greedy distance-matched permutation of the atoms under `transform` about
// the center of mass; mirrors symmetry's own PermutationFor (salc.cpp).
qcx::Result<std::vector<std::size_t>> AtomPermutationFor(const qcx::molecule::Molecule& molecule,
                                                         const Eigen::Matrix3d& transform,
                                                         double toleranceBohr) {
    const qcx::molecule::MassProperties mass =
        qcx::molecule::ComputeMassProperties(molecule, kMomentLinearityThreshold);
    const std::size_t n = molecule.AtomCount();

    struct MatchEntry {
        Eigen::Vector3d position;
        int atomicNumber;
        std::size_t index;
    };

    const auto& coordinates = molecule.CoordinatesBohr();
    std::vector<MatchEntry> entries;
    entries.reserve(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        entries.push_back(
            MatchEntry{Eigen::Vector3d(coordinates(i, 0), coordinates(i, 1), coordinates(i, 2)) -
                           mass.centerOfMass,
                       molecule.Atoms()[i].atomicNumber,
                       i});
    }

    std::vector<MatchEntry> sorted = entries;
    std::sort(sorted.begin(), sorted.end(), [](const MatchEntry& a, const MatchEntry& b) {
        return LexLess(a.position, b.position);
    });
    const double tolSq = toleranceBohr * toleranceBohr;
    std::vector<std::size_t> sigma(n);
    std::vector<bool> matched(n, false);

    for (const MatchEntry& entry : sorted)
    {
        const Eigen::Vector3d q = transform * entry.position;
        const double xMin = q.x() - toleranceBohr;
        const double xMax = q.x() + toleranceBohr;
        auto it =
            std::lower_bound(sorted.begin(), sorted.end(), xMin, [](const MatchEntry& e, double x) {
                return e.position.x() < x;
            });
        bool found = false;

        for (; it != sorted.end() && it->position.x() <= xMax; ++it)
        {
            if (it->atomicNumber != entry.atomicNumber)
            {
                continue;
            }

            const auto j = static_cast<std::size_t>(it - sorted.begin());

            if (!matched[j] && (q - it->position).squaredNorm() <= tolSq)
            {
                matched[j] = true;
                sigma[entry.index] = it->index;
                found = true;
                break;
            }
        }

        if (!found)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "an operation of the computational group does not "
                                              "permute the atoms"});
        }
    }

    return sigma;
}

double Factorial(int n) {
    double result = 1.0;

    for (int k = 2; k <= n; ++k)
    {
        result *= static_cast<double>(k);
    }

    return result;
}

// Associated Legendre function P_l^m(cos theta), 0 <= m <= l <= 6, WITHOUT
// the Condon-Shortley phase: the angular action must reproduce the engine's
// real solid harmonics (kSolidHarmonicG, Racah-normalized, unphased -
// md_tables_gen.hpp), and the phase would flip every m != 0 function, an
// equivalence the representation validation cannot see (conjugating by a
// diagonal +-1 matrix preserves traces and products) yet which breaks the
// block-diagonality of u^T F u for dense group actions. The standard
// (2m-1)!! seed plus the (l-1, l-2) recurrence, stable at these small
// degrees.
double AssociatedLegendre(int l, int m, double x) {
    double pmm = 1.0;
    const double somx2 = std::sqrt(std::max(0.0, 1.0 - x * x));
    double fact = 1.0;

    for (int i = 1; i <= m; ++i)
    {
        pmm *= fact * somx2;
        fact += 2.0;
    }

    if (l == m)
    {
        return pmm;
    }

    double pmmp1 = x * (2.0 * m + 1.0) * pmm;

    if (l == m + 1)
    {
        return pmmp1;
    }

    double pll = 0.0;

    for (int ll = m + 2; ll <= l; ++ll)
    {
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - static_cast<double>(ll - 1 + m) * pmm) /
              static_cast<double>(ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }

    return pll;
}

// Real spherical harmonics Y_{l,m} in the engine's m ascending -l..+l order
// (cos(m phi) for m >= 0, sin(m phi) for m < 0), orthonormal on the unit
// sphere: the real combination of the complex harmonics with the
// Condon-Shortley phase. Any fixed sign convention works - the D-matrix
// construction only needs a self-consistent orthonormal evaluator.
// (l, m) are the degree and the order - distinct quantities; every call
// site passes them in fixed order.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double RealSphericalHarmonic(int l, int m, const Eigen::Vector3d& r) {
    const double radius = r.norm();

    if (radius < 1e-14)
    {
        return 0.0;
    }

    const double x = r.x() / radius;
    const double y = r.y() / radius;
    const double z = r.z() / radius;
    const int am = std::abs(m);
    const double norm = std::sqrt((2.0 * l + 1.0) / (4.0 * std::numbers::pi) * Factorial(l - am) /
                                  Factorial(l + am));
    const double plm = AssociatedLegendre(l, am, z);

    if (am == 0)
    {
        return norm * plm;
    }

    const double sinTheta = std::sqrt(std::max(0.0, 1.0 - z * z));

    if (sinTheta < 1e-14)
    {
        return 0.0;
    }

    const double phi = std::atan2(y, x);

    if (m > 0)
    {
        return std::sqrt(2.0) * norm * plm * std::cos(static_cast<double>(am) * phi);
    }

    return std::sqrt(2.0) * norm * plm * std::sin(static_cast<double>(am) * phi);
}

// One basis function of a shell: spherical harmonic `index - l`, or the
// z-slowest Cartesian monomial of degree l (shell_pairs.hpp ordering: a
// descending, c ascending, b = l - a - c).
double ShellBasisValue(int l, bool isSpherical, int index, const Eigen::Vector3d& r) {
    if (isSpherical)
    {
        return RealSphericalHarmonic(l, index - l, r);
    }

    int a = l;
    int c = 0;

    for (int i = 0; i < index; ++i)
    {
        if (c < l - a)
        {
            ++c;
        } else
        {
            --a;
            c = 0;
        }
    }

    const int b = l - a - c;
    double value = 1.0;

    for (int k = 0; k < a; ++k)
    {
        value *= r.x();
    }

    for (int k = 0; k < b; ++k)
    {
        value *= r.y();
    }

    for (int k = 0; k < c; ++k)
    {
        value *= r.z();
    }

    return value;
}

// Sample points on the unit sphere for the evaluation matrix: two latitudes
// away from the poles and the z-axis (where the azimuth is undefined), with
// golden-angle phase steps that keep the phases distinct.
std::vector<Eigen::Vector3d> SamplePoints(int count) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(static_cast<std::size_t>(count));

    for (int k = 0; k < count; ++k)
    {
        const double theta = std::numbers::pi * (0.25 + 0.5 * static_cast<double>(k % 2));
        const double phi = 2.0 * std::numbers::pi * static_cast<double>(k) * 0.6180339887498949;
        points.push_back(Eigen::Vector3d(
            std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)));
    }

    return points;
}

// The exact action of `matrix` on the shell's angular functions, in the
// engine's function order: the nAng x nAng matrix D with
// (g f_m)(r) = sum_m' D(m', m) f_m'(matrix * r). Computed by evaluating the
// basis on sample points and inverting the evaluation matrix; the inversion
// is guarded by an orthogonality self-check (D must be orthogonal).
qcx::Result<Eigen::MatrixXd> AngularAction(int l, bool isSpherical, const Eigen::Matrix3d& matrix) {
    const int nAng = isSpherical ? 2 * l + 1 : (l + 1) * (l + 2) / 2;
    const auto points = SamplePoints(nAng);
    Eigen::MatrixXd evaluation = Eigen::MatrixXd::Zero(nAng, nAng);
    Eigen::MatrixXd transformed = Eigen::MatrixXd::Zero(nAng, nAng);

    for (int k = 0; k < nAng; ++k)
    {
        for (int m = 0; m < nAng; ++m)
        {
            evaluation(k, m) =
                ShellBasisValue(l, isSpherical, m, points[static_cast<std::size_t>(k)]);
            transformed(k, m) =
                ShellBasisValue(l, isSpherical, m, matrix * points[static_cast<std::size_t>(k)]);
        }
    }

    Eigen::MatrixXd action = evaluation.fullPivLu().solve(transformed);
    const Eigen::MatrixXd orthoError =
        action.transpose() * action - Eigen::MatrixXd::Identity(nAng, nAng);

    if (orthoError.cwiseAbs().maxCoeff() > kAngularOrthogonalityTolerance)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "angular action is not orthogonal (sample points "
                                          "degenerate)"});
    }

    return action;
}

// The n x n matrix of the operation on the whole AO space: the atom
// permutation relocates each shell to its image, and the image shell's
// angular functions mix under the operation's exact angular action (the same
// D block per contraction row, matching the engine's function order).
qcx::Result<Eigen::MatrixXd> BuildAoAction(const qcx::integrals::ShellPairList& shellList,
                                           const std::vector<std::size_t>& permutation,
                                           const Eigen::Matrix3d& matrix) {
    const Eigen::Index n = static_cast<Eigen::Index>(shellList.functionCount);
    Eigen::MatrixXd action = Eigen::MatrixXd::Zero(n, n);

    for (const auto& shell : shellList.shells)
    {
        const std::size_t imageAtom = permutation[shell.atomIndex];
        std::optional<std::size_t> imageShell;

        for (std::size_t t = 0; t < shellList.shells.size(); ++t)
        {
            if (shellList.shells[t].atomIndex == imageAtom &&
                shellList.shells[t].elementShellIndex == shell.elementShellIndex)
            {
                imageShell = t;
                break;
            }
        }

        if (!imageShell.has_value())
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "an operation maps a shell onto a missing element-shell"});
        }

        const auto angular = AngularAction(shell.angularMomentum, shell.isSpherical, matrix);

        if (!angular.has_value())
        {
            return std::unexpected(angular.error());
        }

        const Eigen::Index nAng = static_cast<Eigen::Index>(angular->rows());
        const Eigen::Index imageOffset =
            static_cast<Eigen::Index>(shellList.shells[*imageShell].functionOffset);

        for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(shell.contractionCount); ++c)
        {
            for (Eigen::Index a = 0; a < nAng; ++a)
            {
                for (Eigen::Index ap = 0; ap < nAng; ++ap)
                {
                    const Eigen::Index col =
                        static_cast<Eigen::Index>(shell.functionOffset) + c * nAng + a;
                    const Eigen::Index row = imageOffset + c * nAng + ap;
                    action(row, col) = (*angular)(ap, a);
                }
            }
        }
    }

    return action;
}

// The generators of the computational group realized from the detected
// elements, the group as the deduplicated subset products (every generator
// is an involution and the group is Abelian), and the irreps as the distinct
// sign patterns on the generators (an Abelian group of order m has exactly m
// irreps). kC1 yields the empty generator set.
struct GroupData {
    std::vector<Eigen::Matrix3d> generators;
    std::vector<Eigen::Matrix3d> elements; // All group elements, elements[0] = I.
    std::vector<std::vector<int>> characters; // [irrep][element]: +-1.
    bool isTrivial = false; // The computational group is C1.
};

// Deterministic tie-break for the axis selection: smaller direction first.
bool DirectionLess(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    if (a.x() != b.x())
    {
        return a.x() < b.x();
    }

    if (a.y() != b.y())
    {
        return a.y() < b.y();
    }

    return a.z() < b.z();
}

qcx::Result<GroupData> BuildGroupData(const qcx::symmetry::SymmetryAnalysis& analysis,
                                      const qcx::molecule::Molecule& molecule) {
    if (analysis.computational == qcx::symmetry::PointGroup::kC1)
    {
        return GroupData{};
    }

    std::vector<Eigen::Vector3d> c2Axes;
    std::vector<Eigen::Vector3d> mirrorNormals;
    bool hasInversion = false;

    for (const auto& element : analysis.elements)
    {
        if (element.kind == qcx::symmetry::OperationKind::kRotation && element.order % 2 == 0)
        {
            c2Axes.push_back(element.axis);
        } else if (element.kind == qcx::symmetry::OperationKind::kSigma)
        {
            mirrorNormals.push_back(element.axis);
        } else if (element.kind == qcx::symmetry::OperationKind::kInversion)
        { hasInversion = true; }
    }

    const auto dedupDirections = [](const std::vector<Eigen::Vector3d>& directions) {
        std::vector<Eigen::Vector3d> unique;

        for (const auto& direction : directions)
        {
            const Eigen::Vector3d normalized = direction.normalized();
            bool seen = false;

            for (const auto& existing : unique)
            {
                if (std::abs(normalized.dot(existing)) > 1.0 - kAxisParallelTolerance)
                {
                    seen = true;
                    break;
                }
            }

            if (!seen)
            {
                unique.push_back(normalized);
            }
        }

        return unique;
    };

    c2Axes = dedupDirections(c2Axes);
    mirrorNormals = dedupDirections(mirrorNormals);
    const auto c2 = [](const std::vector<Eigen::Vector3d>& axes, std::size_t i) {
        return Eigen::AngleAxisd(std::numbers::pi, axes[i]).toRotationMatrix();
    };
    const auto rotationAbout = [](const Eigen::Vector3d& axis) {
        return Eigen::AngleAxisd(std::numbers::pi, axis).toRotationMatrix();
    };
    const auto mirror = [](const std::vector<Eigen::Vector3d>& normals, std::size_t i) {
        return Eigen::Matrix3d::Identity() - 2.0 * normals[i] * normals[i].transpose();
    };
    const Eigen::Matrix3d inversion = -Eigen::Matrix3d::Identity();

    // salc.cpp's D2/D2h axis labeling: the generators are the three mutually
    // perpendicular C2 axes selected by principal-moment alignment - 'z' hugs
    // the highest moment, 'y' the middle moment subject to perpendicularity,
    // and 'x' = z x y must itself be a detected axis. The first few recorded
    // axes are NOT safe: benzene's detector records all six in-plane C2' lines
    // (60 deg apart, not mutually perpendicular), so the naive pick does not
    // realize D2h. The mass properties are cheap and the eigensolve is
    // deterministic.
    const qcx::molecule::MassProperties mass =
        qcx::molecule::ComputeMassProperties(molecule, kMomentLinearityThreshold);
    const auto bestAligned = [&mass](const std::vector<Eigen::Vector3d>& directions,
                                     int column,
                                     const Eigen::Vector3d* ref) -> std::optional<Eigen::Vector3d> {
        double bestDot = -1.0;
        Eigen::Vector3d best = Eigen::Vector3d::Zero();

        for (const auto& direction : directions)
        {
            if (ref != nullptr && std::abs(direction.dot(*ref)) >= kAxisParallelTolerance)
            {
                continue;
            }

            const double dot = std::abs(direction.dot(mass.principalAxes.col(column)));

            if (dot > bestDot + kAxisParallelTolerance ||
                (std::abs(dot - bestDot) <= kAxisParallelTolerance && bestDot >= 0.0 &&
                 DirectionLess(direction, best)))
            {
                bestDot = dot;
                best = direction;
            }
        }

        if (bestDot < 0.0)
        {
            return std::nullopt;
        }

        return best;
    };

    std::vector<Eigen::Matrix3d> generators;

    switch (analysis.computational)
    {
    case qcx::symmetry::PointGroup::kCs:

        if (mirrorNormals.size() >= 1)
        {
            generators.push_back(mirror(mirrorNormals, 0));
        }

        break;
    case qcx::symmetry::PointGroup::kCi:

        if (hasInversion)
        {
            generators.push_back(inversion);
        }

        break;
    case qcx::symmetry::PointGroup::kC2:

        if (c2Axes.size() >= 1)
        {
            generators.push_back(c2(c2Axes, 0));
        }

        break;
    case qcx::symmetry::PointGroup::kC2v:

        if (mirrorNormals.size() >= 2)
        {
            generators.push_back(mirror(mirrorNormals, 0));
            generators.push_back(mirror(mirrorNormals, 1));
        }

        break;
    case qcx::symmetry::PointGroup::kC2h:

        if (c2Axes.size() >= 1 && hasInversion)
        {
            generators.push_back(c2(c2Axes, 0));
            generators.push_back(inversion);
        }

        break;
    case qcx::symmetry::PointGroup::kD2:
    case qcx::symmetry::PointGroup::kD2h: {
        const auto zBest = bestAligned(c2Axes, 0, nullptr);

        if (c2Axes.size() >= 3 && zBest.has_value())
        {
            const Eigen::Vector3d zAxis = *zBest;
            const auto yBest = bestAligned(c2Axes, 1, &zAxis);

            if (yBest.has_value())
            {
                const Eigen::Vector3d yAxis = *yBest;
                const Eigen::Vector3d xAxis = zAxis.cross(yAxis).normalized();
                const bool xIsC2 = std::any_of(
                    c2Axes.begin(), c2Axes.end(), [&xAxis](const Eigen::Vector3d& candidate) {
                        return std::abs(candidate.dot(xAxis)) > 1.0 - kAxisParallelTolerance;
                    });

                if (xIsC2)
                {
                    generators.push_back(rotationAbout(zAxis));
                    generators.push_back(rotationAbout(yAxis));
                    generators.push_back(rotationAbout(xAxis));

                    // Td (and T/Th/Oh/Ih by the psi4 convention) reduce to
                    // the nominal D2h, but Td contains no inversion: only
                    // the D2 of the three perpendicular C2s is realizable
                    // from the detected elements. Blocking by that real
                    // subgroup is equally valid - F commutes with every
                    // element of the full molecular group, hence with every
                    // element of the subgroup - and the tests below pin the
                    // blocked == plain equivalence either way.
                    if (hasInversion)
                    {
                        generators.push_back(inversion);
                    }
                }
            }
        }

        break;
    }

    case qcx::symmetry::PointGroup::kC1:
        return GroupData{};
    }

    const std::size_t k = generators.size();

    if (k == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                          "the computational group cannot be realized from the "
                                          "detected elements"});
    }

    // The subset products close under the relations of the generator set
    // (C2x C2y = C2z in D2/D2h, sigma0 sigma1 = C2 in C2v): D2 is generated
    // by three C2 axes but has order 4. A nominal D2h whose molecule lacks
    // inversion (Td-type cubic) realizes only the D2 of its three C2s.
    const std::size_t expectedOrder = [&analysis, &hasInversion]() {
        switch (analysis.computational)
        {
        case qcx::symmetry::PointGroup::kCi:
        case qcx::symmetry::PointGroup::kC2:
        case qcx::symmetry::PointGroup::kCs:
            return std::size_t{2};
        case qcx::symmetry::PointGroup::kC2h:
        case qcx::symmetry::PointGroup::kD2:
        case qcx::symmetry::PointGroup::kC2v:
            return std::size_t{4};
        case qcx::symmetry::PointGroup::kD2h:
            return hasInversion ? std::size_t{8} : std::size_t{4};
        case qcx::symmetry::PointGroup::kC1:
            return std::size_t{1};
        }

        return std::size_t{1}; // Unreachable; silences -Wreturn-type.
    }();
    std::vector<Eigen::Matrix3d> elements;
    std::vector<std::uint32_t> elementMasks;
    elements.reserve(expectedOrder);
    elementMasks.reserve(expectedOrder);

    for (std::uint32_t mask = 0; mask < (std::uint32_t{1} << k); ++mask)
    {
        Eigen::Matrix3d product = Eigen::Matrix3d::Identity();

        for (std::size_t j = 0; j < k; ++j)
        {
            if ((mask & (std::uint32_t{1} << j)) != 0)
            {
                product = generators[j] * product;
            }
        }

        bool seen = false;

        for (const auto& existing : elements)
        {
            if ((product - existing).norm() < kMatrixDedupTolerance)
            {
                seen = true;
                break;
            }
        }

        if (!seen)
        {
            elements.push_back(product);
            elementMasks.push_back(mask);
        }
    }

    if (elements.size() != expectedOrder)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                          "the generators do not close into the computational "
                                          "group"});
    }

    // The irreps: the distinct sign patterns on the generators, evaluated on
    // every group element through its generating subset. Patterns that
    // differ only by a relation of the generator set coincide on the group
    // and are deduplicated; an Abelian group of order m has exactly m.
    std::vector<std::vector<int>> characters;

    for (std::uint32_t pattern = 0; pattern < (std::uint32_t{1} << k); ++pattern)
    {
        std::vector<int> chi;
        chi.reserve(elements.size());

        for (const std::uint32_t mask : elementMasks)
        {
            int value = 1;

            for (std::size_t j = 0; j < k; ++j)
            {
                if ((mask & (std::uint32_t{1} << j)) != 0 &&
                    (pattern & (std::uint32_t{1} << j)) != 0)
                {
                    value = -value;
                }
            }

            chi.push_back(value);
        }

        bool seen = false;

        for (const auto& existing : characters)
        {
            if (existing == chi)
            {
                seen = true;
                break;
            }
        }

        if (!seen)
        {
            characters.push_back(std::move(chi));
        }
    }

    if (characters.size() != elements.size())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                          "the generator sign patterns do not realize the "
                                          "group's irreps"});
    }

    return GroupData{std::move(generators), std::move(elements), std::move(characters)};
}

// Union-find orbits of atoms under the group's permutations (mirrors
// symmetry's ComputeOrbits); the AO orbits are the function sets of the atom
// orbits (the angular action stays within the shell images).
std::vector<std::vector<std::size_t>> ComputeAoOrbits(
    const qcx::integrals::ShellPairList& shellList,
    const std::vector<std::vector<std::size_t>>& atomPermutations,
    std::size_t atomCount) {
    std::vector<std::size_t> parent(atomCount);
    std::iota(parent.begin(), parent.end(), 0);
    const auto findRoot = [&parent](std::size_t i) {
        while (parent[i] != i)
        {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }

        return i;
    };

    for (const auto& permutation : atomPermutations)
    {
        for (std::size_t i = 0; i < atomCount; ++i)
        {
            parent[findRoot(i)] = findRoot(permutation[i]);
        }
    }

    std::vector<std::vector<std::size_t>> atomOrbits(atomCount);

    for (std::size_t i = 0; i < atomCount; ++i)
    {
        atomOrbits[findRoot(i)].push_back(i);
    }

    std::vector<std::vector<std::size_t>> aoOrbits;

    for (const auto& atomOrbit : atomOrbits)
    {
        if (atomOrbit.empty())
        {
            continue;
        }

        std::vector<std::size_t> functions;

        for (const auto& shell : shellList.shells)
        {
            bool inOrbit = false;

            for (const std::size_t atom : atomOrbit)
            {
                if (shell.atomIndex == atom)
                {
                    inOrbit = true;
                    break;
                }
            }

            if (!inOrbit)
            {
                continue;
            }

            for (std::size_t f = 0; f < qcx::integrals::ShellFunctionCount(shell); ++f)
            {
                functions.push_back(shell.functionOffset + f);
            }
        }

        aoOrbits.push_back(std::move(functions));
    }

    return aoOrbits;
}

// ---- Full-group closure machinery ----
// The runtime mirrors tools/gen_full_group_tables.py exactly - the
// classification (classify), the greedy conjugacy partition
// (conjugacy_classes), the slot assignment (assign_slots) and the table sort
// (sort_classes) - so the runtime classes of a detected group match the
// generated tables' classes (kind, order, j, slot) key-for-key. The closures
// of the recorded finite operations generate the full detected group.

constexpr double kClosureDedupTolerance = 1e-8; // The generator's matrix equality.
// The A(g) A(h) = A(gh) guard tolerance: the angular actions carry the
// D-matrix construction noise (<= 1e-8), so the composition check runs at
// 1e-7.
constexpr double kRepresentationGuardTolerance = 1e-7;
// The axis-canonicalization zero-snap (the generator's canonical_axis):
// sub-tolerance components snap to exact zero before the sign
// canonicalization (CanonicalAxis and the frame-canonicalized image).
constexpr double kAxisCanonicalizationTolerance = 1e-8;

// A rotation matrix about a (not necessarily normalized) axis.
Eigen::Matrix3d RotationMatrix(const Eigen::Vector3d& axis, double angle) {
    return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}

// The reflection in the plane with the given normal.
Eigen::Matrix3d MirrorMatrix(const Eigen::Vector3d& normal) {
    const Eigen::Vector3d n = normal.normalized();
    return Eigen::Matrix3d::Identity() - 2.0 * n * n.transpose();
}

// The generator's canonical_axis(): snap sub-tolerance components to exact
// zero, normalize to unit length, then sign-canonicalize (first nonzero
// component positive); zero for an all-zero axis. The normalization makes
// the axis representative-independent: the raw axial vector's magnitude
// varies per element (2|sin theta|), so the same physical axis would rank
// differently for different members of one conjugacy class (the D5d S10
// classes: (0,0,1.902) vs (0,0,1.176) for the same z) and scramble the
// principal tie-break and the on-principal-axis slot test.
Eigen::Vector3d CanonicalAxis(const Eigen::Vector3d& a) {
    Eigen::Vector3d out = a;

    for (int k = 0; k < 3; ++k)
    {
        if (std::abs(out(k)) <= kAxisCanonicalizationTolerance)
        {
            out(k) = 0.0;
        }
    }

    int first = -1;

    for (int k = 0; k < 3; ++k)
    {
        if (std::abs(out(k)) > kAxisCanonicalizationTolerance)
        {
            first = k;
            break;
        }
    }

    if (first < 0)
    {
        return Eigen::Vector3d::Zero();
    }

    out /= out.norm();

    if (out(first) < 0.0)
    {
        out = -out;
    }

    return out;
}

// The same physical axis up to sign (the LINE comparison, matching the
// parallel-axis dot test of the axis-fixing scans): the absolute dot of
// the unit canonical axes. A componentwise comparison of the absolute
// values is not a line test - it conflates genuinely distinct axes that
// share component magnitudes: the tilted pair (0, 0.7071, 0.7071) and
// (0, 0.7071, -0.7071) are perpendicular lines with identical
// |components|, and the D2h tilted fixture's sigma(xz) class was
// mis-slotted onto the principal axis by exactly that. The dot form
// distinguishes the lines while remaining sign-insensitive (the two
// directions of one axis are the same line).
bool SameAxis(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return std::abs(a.dot(b)) > 1.0 - kAxisParallelTolerance;
}

// Strict-weak lexicographic axis ordering (the generator's axis_key).
bool AxisKeyLess(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    if (a.x() != b.x())
    {
        return a.x() < b.x();
    }

    if (a.y() != b.y())
    {
        return a.y() < b.y();
    }

    return a.z() < b.z();
}

// The generator's axis_of(g, target): the axial vector of the skew part
// (works where eigh is invalid), Euclidean-norm gated at 1e-6; the eigh
// fallback covers theta = pi and the mirrors (the skew part vanishes).
// Sign-canonicalized and unit-normalized via CanonicalAxis (the C3 axis is
// z, not sqrt(3) z).
Eigen::Vector3d AxisOf(const Eigen::Matrix3d& g, double target) {
    const Eigen::Matrix3d m = target > 0.0 ? g : -g;
    const Eigen::Vector3d axial(m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1));

    if (axial.norm() > 1e-6)
    {
        return CanonicalAxis(axial);
    }

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(g);
    Eigen::Index best = 0;

    for (Eigen::Index i = 1; i < 3; ++i)
    {
        if (std::abs(solver.eigenvalues()(i) - target) <
            std::abs(solver.eigenvalues()(best) - target))
        {
            best = i;
        }
    }

    return CanonicalAxis(solver.eigenvectors().col(best));
}

// The runtime classification of one element, mirroring the generator's
// classify(): (kind, order, j, axis) with the identity/inversion max-abs
// checks first, the order the minimal power with g^k = E (check-before-
// multiply, capped at 24), det > 0 a rotation, det < 0 with |tr - 1| < 1e-8
// a mirror (order 1, j 1), else an improper rotation; j = round(theta *
// order / 2pi) with theta measured from the action on a perpendicular test
// vector (the bare trace cannot tell theta from 2pi - theta), Python-style
// modulo, j == 0 -> order. The axis is canonicalized (zero for
// identity/inversion).
struct ClassifiedOperation {
    qcx::symmetry::OperationKind kind = qcx::symmetry::OperationKind::kIdentity;
    int order = 1;
    int power = 1;
    Eigen::Vector3d axis = Eigen::Vector3d::Zero();
};

ClassifiedOperation ClassifyOperation(const Eigen::Matrix3d& g) {
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

    if ((g - identity).cwiseAbs().maxCoeff() < kClosureDedupTolerance)
    {
        return {qcx::symmetry::OperationKind::kIdentity, 1, 1, Eigen::Vector3d::Zero()};
    }

    if ((g + identity).cwiseAbs().maxCoeff() < kClosureDedupTolerance)
    {
        return {qcx::symmetry::OperationKind::kInversion, 1, 1, Eigen::Vector3d::Zero()};
    }

    const double det = g.determinant();
    const double tr = g.trace();
    Eigen::Matrix3d h = g;
    int order = 1;

    for (int k = 1; k <= 24; ++k)
    {
        if ((h - identity).cwiseAbs().maxCoeff() < kClosureDedupTolerance)
        {
            order = k;
            break;
        }

        h = h * g;
    }

    qcx::symmetry::OperationKind kind = qcx::symmetry::OperationKind::kImproper;

    if (det > 0.0)
    {
        kind = qcx::symmetry::OperationKind::kRotation;
    } else if (std::abs(tr - 1.0) < kClosureDedupTolerance)
    { return {qcx::symmetry::OperationKind::kSigma, 1, 1, AxisOf(g, -1.0)}; }

    const Eigen::Vector3d axis = AxisOf(g, 1.0);
    Eigen::Vector3d ref(1.0, 0.0, 0.0);

    if (std::abs(axis.dot(ref)) > 0.9)
    {
        ref = Eigen::Vector3d(0.0, 1.0, 0.0);
    }

    const Eigen::Vector3d v = (ref - (ref.dot(axis) / axis.dot(axis)) * axis).normalized();
    const Eigen::Vector3d u = g * v;
    const double theta = std::atan2(u.dot(axis.cross(v)), u.dot(v));
    int power = static_cast<int>(
        std::lround(theta * static_cast<double>(order) / (2.0 * std::numbers::pi)));
    power = ((power % order) + order) % order;

    if (power == 0)
    {
        power = order;
    }

    return {kind, order, power, axis};
}

// The generator seeds from the detected elements: the identity is skipped,
// inversion is -I, a rotation of order n yields its powers R(2 pi j / n,
// axis) for j = 1..n-1 (the detection records the maximal rotation per
// axis), a mirror the reflection in its plane, an improper S2n the powers
// (R(2 pi / 2n, axis) sigma_h)^k for k = 1..2n-1 - the generator's
// convention (rot . sigma_h for the improper seed).
std::vector<Eigen::Matrix3d> ClosureSeeds(const qcx::symmetry::SymmetryAnalysis& analysis) {
    std::vector<Eigen::Matrix3d> seeds;

    for (const auto& element : analysis.elements)
    {
        switch (element.kind)
        {
        case qcx::symmetry::OperationKind::kIdentity:
            break;

        case qcx::symmetry::OperationKind::kInversion:
            seeds.push_back(-Eigen::Matrix3d::Identity());
            break;

        case qcx::symmetry::OperationKind::kRotation:

            for (int j = 1; j < element.order; ++j)
            {
                seeds.push_back(RotationMatrix(element.axis,
                                               2.0 * std::numbers::pi * static_cast<double>(j) /
                                                   static_cast<double>(element.order)));
            }

            break;

        case qcx::symmetry::OperationKind::kSigma:
            seeds.push_back(MirrorMatrix(element.axis));
            break;

        case qcx::symmetry::OperationKind::kImproper: {
            // The recorded order field of an S2n is 2n.
            const int order = element.order;
            const Eigen::Matrix3d seed =
                RotationMatrix(element.axis, 2.0 * std::numbers::pi / static_cast<double>(order)) *
                MirrorMatrix(element.axis);
            Eigen::Matrix3d power = seed;

            for (int k = 1; k < order; ++k)
            {
                seeds.push_back(power);
                power = power * seed;
            }

            break;
        }
        }
    }

    return seeds;
}

// One discovery step of the closure BFS: the element `parent` times the
// generator `seed` produced the new element `product`. The representation
// guard verifies A(parent) A(seed) = A(product) on the witnesses only - each
// element's action is a verified product along its discovery chain, which
// certifies the whole closure inductively.
struct ClosureWitness {
    std::size_t parent;
    std::size_t seed;
    std::size_t product;
};

// The BFS closure mirroring the generator's closure(): wave over the
// snapshot with p = a g, dedup at 1e-8 max-abs against the whole list,
// repeat until no growth; identity first.
std::vector<Eigen::Matrix3d> BuildClosure(const std::vector<Eigen::Matrix3d>& seeds,
                                          std::vector<ClosureWitness>& witnesses) {
    std::vector<Eigen::Matrix3d> closure;
    closure.push_back(Eigen::Matrix3d::Identity());
    std::size_t waveStart = 0;

    while (waveStart < closure.size())
    {
        const std::size_t waveEnd = closure.size();

        for (std::size_t i = waveStart; i < waveEnd; ++i)
        {
            for (std::size_t s = 0; s < seeds.size(); ++s)
            {
                const Eigen::Matrix3d product = closure[i] * seeds[s];
                const bool duplicate = std::any_of(
                    closure.begin(), closure.end(), [&product](const Eigen::Matrix3d& e) {
                        return (e - product).cwiseAbs().maxCoeff() < kClosureDedupTolerance;
                    });

                if (!duplicate)
                {
                    // The witness records the seed's CLOSURE index, not its
                    // seeds-array index: the realized actions live in the
                    // closure order, and the guard verifies
                    // A(closure[i]) A(closure[seedIndex]) = A(closure[new]).
                    // (Every seed is added in the first wave - products of
                    // the identity - so the scan always finds it; a seed
                    // added by this very push is the product itself.)
                    std::size_t seedIndex = closure.size();

                    for (std::size_t t = 0; t < closure.size(); ++t)
                    {
                        if ((closure[t] - seeds[s]).cwiseAbs().maxCoeff() < kClosureDedupTolerance)
                        {
                            seedIndex = t;
                            break;
                        }
                    }

                    witnesses.push_back({i, seedIndex, closure.size()});
                    closure.push_back(product);
                }
            }
        }

        waveStart = waveEnd;
    }

    return closure;
}

// The greedy conjugacy partition mirroring the generator's
// conjugacy_classes(): scan the unassigned elements in order, collecting the
// conjugates x g x^-1 of the first unassigned element under all group
// elements.
std::vector<std::vector<int>> ConjugacyClasses(const std::vector<Eigen::Matrix3d>& elements) {
    const std::size_t count = elements.size();
    std::vector<Eigen::Matrix3d> inverses;
    inverses.reserve(count);

    for (const auto& element : elements)
    {
        inverses.push_back(element.inverse());
    }

    std::vector<bool> assigned(count, false);
    std::vector<std::vector<int>> classes;

    for (std::size_t i = 0; i < count; ++i)
    {
        if (assigned[i])
        {
            continue;
        }

        std::vector<int> cls;
        const Eigen::Matrix3d& g = elements[i];

        for (std::size_t j = i; j < count; ++j)
        {
            if (assigned[j])
            {
                continue;
            }

            const Eigen::Matrix3d& h = elements[j];
            bool conjugate = false;

            for (std::size_t k = 0; k < count && !conjugate; ++k)
            {
                const Eigen::Matrix3d transformed = elements[k] * g * inverses[k];
                conjugate = (transformed - h).cwiseAbs().maxCoeff() < kClosureDedupTolerance;
            }

            if (conjugate)
            {
                cls.push_back(static_cast<int>(j));
                assigned[j] = true;
            }
        }

        classes.push_back(std::move(cls));
    }

    return classes;
}

// One runtime class: the (kind, order, j, slot) table key, the lex-min
// canonical member axis, and the canonical member axes themselves (the
// frame-aligned orbit ranking runs on them).
struct RuntimeClassInfo {
    qcx::symmetry::OperationKind kind = qcx::symmetry::OperationKind::kIdentity;
    int order = 1;
    int power = 1;
    qcx::symmetry::ClassAxis slot = qcx::symmetry::ClassAxis::kNone;
    Eigen::Vector3d minAxis = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> memberAxes; // Canonical; empty for identity/inversion.
};

// The principal tie-break key of one maximal-order rotation class (used
// only when equal-power classes of DIFFERENT axes compete - the D2 family):
// a class whose axis carries an improper order-4 class wins outright (the
// S4-axis C2 of the D2d groups); otherwise the class whose axis hosts a
// mirror fixing the most atoms wins (the D2h groups - ethene's C2z under
// the molecular-plane mirror), then the class whose axis hosts the most
// atom-fixing elements at all. Residual ties keep the raw lex convention.
// The counts are the per-element atom-fixed counts maxed over the elements
// whose axis is parallel to the candidate's.
struct PrincipalTieKey {
    int stage = 1; // 0 = S4-axis rule, 1 = atom-fixed counts (then lex).
    int mirrorCount = 0; // Negated: larger wins under <.
    int allCount = 0; // Negated: larger wins under <.

    bool operator<(const PrincipalTieKey& other) const {
        if (stage != other.stage)
        {
            return stage < other.stage;
        }

        if (mirrorCount != other.mirrorCount)
        {
            return mirrorCount < other.mirrorCount;
        }

        return allCount < other.allCount;
    }

    bool operator==(const PrincipalTieKey& other) const {
        return stage == other.stage && mirrorCount == other.mirrorCount &&
               allCount == other.allCount;
    }
};

PrincipalTieKey PrincipalTieKeyOf(const RuntimeClassInfo& info,
                                  const std::vector<RuntimeClassInfo>& infos,
                                  const std::vector<FullGroupElement>& elements,
                                  const std::vector<int>& fixedCounts) {
    const Eigen::Vector3d axis = info.minAxis;

    for (const RuntimeClassInfo& other : infos)
    {
        if (other.kind == qcx::symmetry::OperationKind::kImproper && other.order == 4 &&
            other.minAxis.norm() > 0.0 && SameAxis(other.minAxis, axis))
        {
            return {0, 0, 0};
        }
    }

    int mirrorBest = 0;
    int allBest = 0;

    for (std::size_t g = 0; g < elements.size(); ++g)
    {
        if (elements[g].axis.norm() == 0.0 || !SameAxis(elements[g].axis, axis))
        {
            continue;
        }

        allBest = std::max(allBest, fixedCounts[g]);

        if (elements[g].kind == qcx::symmetry::OperationKind::kSigma)
        {
            mirrorBest = std::max(mirrorBest, fixedCounts[g]);
        }
    }

    return {1, -mirrorBest, -allBest};
}

// The principal class: the maximal-order proper rotation, ties by the
// class-minimal power, then the D2-family rule above, then the
// lexicographically smallest canonical axis; a lone sigma class (Cs) when
// no proper rotation exists. Factored out so the standard frame can be
// built on the principal before the slot assignment.
std::optional<std::size_t> SelectPrincipal(const std::vector<RuntimeClassInfo>& infos,
                                           const std::vector<FullGroupElement>& elements,
                                           const std::vector<int>& fixedCounts) {
    std::optional<std::size_t> principal;
    int maxOrder = 0;

    for (std::size_t c = 0; c < infos.size(); ++c)
    {
        if (infos[c].kind == qcx::symmetry::OperationKind::kRotation && infos[c].order > maxOrder)
        {
            maxOrder = infos[c].order;
            principal = c;
        }
    }

    for (std::size_t c = 0; c < infos.size(); ++c)
    {
        if (principal.has_value() && infos[c].kind == qcx::symmetry::OperationKind::kRotation &&
            infos[c].order == maxOrder && c != *principal)
        {
            // The tie-break is DETERMINISTIC: the class-minimal power
            // first. Classes of the same axis (C5 and C5^2 in the order-5
            // groups) differ in power, so the smaller power wins regardless
            // of the detected axes' precision (the principal class is the
            // class of the principal generator's first power); the D2-family
            // rule and the axis comparison only separate equal-power
            // classes of different axes (the D2 family).
            const RuntimeClassInfo& best = infos[*principal];

            if (infos[c].power != best.power)
            {
                if (infos[c].power < best.power)
                {
                    principal = c;
                }
            } else
            {
                const PrincipalTieKey candidate =
                    PrincipalTieKeyOf(infos[c], infos, elements, fixedCounts);
                const PrincipalTieKey incumbent =
                    PrincipalTieKeyOf(best, infos, elements, fixedCounts);

                if (candidate < incumbent ||
                    (candidate == incumbent && AxisKeyLess(infos[c].minAxis, best.minAxis)))
                {
                    principal = c;
                }
            }
        }
    }

    return principal;
}

// The standard-orientation frame (the table generator's construction frame,
// reproduced for an arbitrary molecule orientation): z is the principal
// axis, x the "prime" direction - the axis of a realized mirror or
// perpendicular C2 that fixes the most atoms (the molecular plane's normal
// for planar molecules, the through-atoms C2' for the D families - the
// generator's SIGMA_XZ / C2_X conventions; ties prefer the rotation over
// the mirror, then the lexicographically smallest axis - the rotation
// preference is what keeps the frame covariant under in-plane rotation of
// the D6h benzene, where a sigma_v and a C2' fix the same four atoms but
// their raw-lex winner changes with the orientation), y = z x x. The orbit
// ranking runs in this frame so the slot assignment matches the generated
// tables for molecules in the standard orientation (e.g. the C2v
// sigma_v/sigma_v' order: the molecular-plane mirror's normal ranks last)
// and stays deterministic otherwise. With no perpendicular mirror/C2 (pure
// rotation groups) x is an arbitrary axis perpendicular to z (only the
// ranking of same-(kind, order, j) orbit classes uses it, and those groups
// have none).
Eigen::Matrix3d ComputeStandardFrame(const Eigen::Vector3d& principal,
                                     const std::vector<FullGroupElement>& elements,
                                     const std::vector<int>& fixedCounts) {
    Eigen::Vector3d xBest;
    int bestCount = -1;
    bool bestIsRotation = false;

    for (std::size_t g = 0; g < elements.size(); ++g)
    {
        const FullGroupElement& element = elements[g];
        const bool isRotation = element.kind == qcx::symmetry::OperationKind::kRotation;
        const bool candidate = element.kind == qcx::symmetry::OperationKind::kSigma ||
                               (isRotation && element.order == 2);

        if (!candidate || element.axis.norm() == 0.0 ||
            std::abs(element.axis.dot(principal)) > kAxisParallelTolerance)
        {
            continue;
        }

        if (fixedCounts[g] > bestCount ||
            (fixedCounts[g] == bestCount &&
             ((isRotation && !bestIsRotation) ||
              (isRotation == bestIsRotation && AxisKeyLess(element.axis, xBest)))))
        {
            bestCount = fixedCounts[g];
            xBest = element.axis;
            bestIsRotation = isRotation;
        }
    }

    if (bestCount < 0)
    {
        Eigen::Vector3d fallback(1.0, 0.0, 0.0);

        if (std::abs(fallback.dot(principal)) > kAxisParallelTolerance)
        {
            fallback = Eigen::Vector3d(0.0, 1.0, 0.0);
        }

        xBest = (fallback - principal * fallback.dot(principal)).normalized();
    }

    const Eigen::Vector3d z = principal.normalized();
    const Eigen::Vector3d x = (xBest - z * xBest.dot(z)).normalized();
    const Eigen::Vector3d y = z.cross(x);
    Eigen::Matrix3d frame;
    frame.row(0) = x;
    frame.row(1) = y;
    frame.row(2) = z;
    return frame;
}

// The frame-canonicalized image of an axis: zero-snapped (sub-tolerance
// frame components snap to exact zero), unit-normalized, sign-flipped so
// the first nonzero frame-coordinate component is positive - the
// generator's canonical_axis applied in the standard frame. The snap makes
// the lexicographic ordering exact where the frame alignment is exact.
Eigen::Vector3d FrameCanonicalize(const Eigen::Vector3d& v) {
    Eigen::Vector3d out = v;

    for (int i = 0; i < 3; ++i)
    {
        if (std::abs(out(i)) < kAxisCanonicalizationTolerance)
        {
            out(i) = 0.0;
        }
    }

    const double norm = out.norm();

    if (norm == 0.0)
    {
        return Eigen::Vector3d::Zero();
    }

    out /= norm;

    for (int i = 0; i < 3; ++i)
    {
        if (out(i) != 0.0)
        {
            if (out(i) < 0.0)
            {
                out = -out;
            }

            break;
        }
    }

    return out;
}

// The frame-aligned orbit key of a class: the lex-min frame-canonicalized
// member axis. Ranking same-(kind, order, j) classes by the frame image of
// the raw class minAxis is not rotation-covariant - the minAxis is a
// molecule-frame quantity whose lex-min member can jump discontinuously
// under rotation - so every member axis is canonicalized in the frame and
// the class key is the lex-min over them.
Eigen::Vector3d ClassFrameKey(const RuntimeClassInfo& info, const Eigen::Matrix3d& frame) {
    Eigen::Vector3d best = Eigen::Vector3d::Zero();
    bool haveBest = false;

    for (const Eigen::Vector3d& axis : info.memberAxes)
    {
        const Eigen::Vector3d key = FrameCanonicalize(frame * axis);

        if (!haveBest || AxisKeyLess(key, best))
        {
            best = key;
            haveBest = true;
        }
    }

    return best;
}

// The generator's assign_slots(), with the orbit ranking run in the
// standard-orientation frame (ComputeStandardFrame): the principal class is
// the maximal-order proper rotation (ties by the D2-family rule, then the
// lexicographically smallest canonical axis); when no proper rotation
// exists a lone sigma class (Cs) is principal. Sigma and improper classes
// on the principal axis are principal too; everything else is an orbit
// ranked among same-(kind, order, j) classes by the frame-canonicalized
// member axes, skipping the principal member.
void AssignSlots(std::vector<RuntimeClassInfo>& infos,
                 const Eigen::Matrix3d& frame,
                 const std::vector<FullGroupElement>& elements,
                 const std::vector<int>& fixedCounts) {
    const std::optional<std::size_t> principal = SelectPrincipal(infos, elements, fixedCounts);
    std::optional<Eigen::Vector3d> principalAxis;

    if (principal.has_value())
    {
        principalAxis = infos[*principal].minAxis;
    } else
    {
        std::optional<std::size_t> loneSigma;

        for (std::size_t c = 0; c < infos.size(); ++c)
        {
            if (infos[c].kind != qcx::symmetry::OperationKind::kSigma)
            {
                continue;
            }

            if (loneSigma.has_value())
            {
                loneSigma = std::nullopt;
                break;
            }

            loneSigma = c;
        }

        if (loneSigma.has_value())
        {
            principalAxis = infos[*loneSigma].minAxis;
        }
    }

    for (std::size_t c = 0; c < infos.size(); ++c)
    {
        const RuntimeClassInfo& info = infos[c];

        // ANY member on the principal line: the lex-min minAxis
        // is a single noisy representative that can jump between
        // the lines of a multi-axis class (the Ih S10 class
        // spans all six fivefold axes, so its members contain
        // the principal line whichever one the min lands on),
        // while a class whose members avoid the principal line
        // is an orbit class regardless of its min.
        const bool onPrincipalLine = (info.kind == qcx::symmetry::OperationKind::kSigma ||
                                      info.kind == qcx::symmetry::OperationKind::kImproper) &&
                                     principalAxis.has_value() &&
                                     std::any_of(info.memberAxes.begin(),
                                                 info.memberAxes.end(),
                                                 [&principalAxis](const Eigen::Vector3d& axis) {
                                                     return SameAxis(axis, *principalAxis);
                                                 });
        const bool isPrincipalClass = principal.has_value() && c == *principal;

        if (info.kind == qcx::symmetry::OperationKind::kIdentity ||
            info.kind == qcx::symmetry::OperationKind::kInversion ||
            (!onPrincipalLine && !isPrincipalClass))
        {
            infos[c].slot = qcx::symmetry::ClassAxis::kNone;
        } else
        {
            infos[c].slot = qcx::symmetry::ClassAxis::kPrincipal;
        }
    }

    // The orbit ranks within each (kind, order, j) group, in the
    // frame-aligned min-axis order.
    std::vector<std::size_t> order;
    order.reserve(infos.size());

    for (std::size_t c = 0; c < infos.size(); ++c)
    {
        order.push_back(c);
    }

    const auto groupLess = [&infos](std::size_t a, std::size_t b) {
        const RuntimeClassInfo& x = infos[a];
        const RuntimeClassInfo& y = infos[b];

        if (x.kind != y.kind)
        {
            return static_cast<int>(x.kind) < static_cast<int>(y.kind);
        }

        if (x.order != y.order)
        {
            return x.order < y.order;
        }

        return x.power < y.power;
    };
    std::sort(order.begin(), order.end(), groupLess);
    std::size_t rankStart = 0;

    while (rankStart < order.size())
    {
        std::size_t rankEnd = rankStart + 1;

        while (rankEnd < order.size() && !groupLess(order[rankStart], order[rankEnd]) &&
               !groupLess(order[rankEnd], order[rankStart]))
        {
            ++rankEnd;
        }

        std::vector<std::size_t> members(order.begin() + static_cast<std::ptrdiff_t>(rankStart),
                                         order.begin() + static_cast<std::ptrdiff_t>(rankEnd));
        std::sort(members.begin(), members.end(), [&infos, &frame](std::size_t a, std::size_t b) {
            return AxisKeyLess(ClassFrameKey(infos[a], frame), ClassFrameKey(infos[b], frame));
        });
        int orbitRank = 0;

        for (const std::size_t c : members)
        {
            if (infos[c].slot == qcx::symmetry::ClassAxis::kNone &&
                infos[c].kind != qcx::symmetry::OperationKind::kIdentity &&
                infos[c].kind != qcx::symmetry::OperationKind::kInversion)
            {
                infos[c].slot = static_cast<qcx::symmetry::ClassAxis>(
                    static_cast<int>(qcx::symmetry::ClassAxis::kOrbit0) + orbitRank);
                ++orbitRank;
            }
        }

        rankStart = rankEnd;
    }
}

// The (kind priority, order, j, slot priority, min axis) table sort of the
// generator's sort_classes().
bool ClassInfoLess(const RuntimeClassInfo& a, const RuntimeClassInfo& b) {
    const auto kindPriority = [](qcx::symmetry::OperationKind kind) {
        switch (kind)
        {
        case qcx::symmetry::OperationKind::kIdentity:
            return 0;

        case qcx::symmetry::OperationKind::kRotation:
            return 1;

        case qcx::symmetry::OperationKind::kInversion:
            return 2;

        case qcx::symmetry::OperationKind::kImproper:
            return 3;

        case qcx::symmetry::OperationKind::kSigma:
            return 4;
        }

        return 4;
    };
    const auto slotPriority = [](qcx::symmetry::ClassAxis slot) {
        switch (slot)
        {
        case qcx::symmetry::ClassAxis::kPrincipal:
            return 0;

        case qcx::symmetry::ClassAxis::kOrbit0:
            return 1;

        case qcx::symmetry::ClassAxis::kOrbit1:
            return 2;

        case qcx::symmetry::ClassAxis::kOrbit2:
            return 3;

        case qcx::symmetry::ClassAxis::kNone:
            return 4;
        }

        return 4;
    };

    if (kindPriority(a.kind) != kindPriority(b.kind))
    {
        return kindPriority(a.kind) < kindPriority(b.kind);
    }

    if (a.order != b.order)
    {
        return a.order < b.order;
    }

    if (a.power != b.power)
    {
        return a.power < b.power;
    }

    if (slotPriority(a.slot) != slotPriority(b.slot))
    {
        return slotPriority(a.slot) < slotPriority(b.slot);
    }

    return AxisKeyLess(a.minAxis, b.minAxis);
}

// The runtime classes of a closure (no slots yet): classify every member of
// every conjugacy class, take the class-minimal power (min j over the
// members - a class mixes conjugate powers by construction, and the
// min-index member's own power is discovery-order dependent), the min-axis
// over the members, and the canonical member axes. The slot assignment and
// the table-order sort run afterwards (the standard frame needs the
// principal first).
std::vector<RuntimeClassInfo> ClassifyRawClasses(const std::vector<Eigen::Matrix3d>& elements,
                                                 const std::vector<std::vector<int>>& classes) {
    std::vector<RuntimeClassInfo> infos(classes.size());

    for (std::size_t c = 0; c < classes.size(); ++c)
    {
        const auto& cls = classes[c];
        RuntimeClassInfo& info = infos[c];

        for (std::size_t m = 0; m < cls.size(); ++m)
        {
            const ClassifiedOperation op =
                ClassifyOperation(elements[static_cast<std::size_t>(cls[m])]);

            if (m == 0)
            {
                info.kind = op.kind;
                info.order = op.order;
                info.power = op.power;
            } else if (op.power < info.power)
            {
                // The canonical class key power: the class-minimal member
                // power (the generated tables canonicalize the same way).
                info.power = op.power;
            }

            if (m == 0 || AxisKeyLess(op.axis, info.minAxis))
            {
                info.minAxis = op.axis;
            }

            if (op.axis.norm() > 0.0)
            {
                info.memberAxes.push_back(op.axis);
            }
        }
    }

    return infos;
}

// The slotted classes sorted by the generator's table key, paired with the
// raw-class index of each sorted entry (the raw classes stay in the
// discovery order).
std::vector<std::pair<RuntimeClassInfo, std::size_t>> SortClassInfos(
    std::vector<RuntimeClassInfo> infos) {
    std::vector<std::size_t> perm(infos.size());

    for (std::size_t c = 0; c < perm.size(); ++c)
    {
        perm[c] = c;
    }

    std::sort(perm.begin(), perm.end(), [&infos](std::size_t a, std::size_t b) {
        return ClassInfoLess(infos[a], infos[b]);
    });
    std::vector<std::pair<RuntimeClassInfo, std::size_t>> out;
    out.reserve(infos.size());

    for (const std::size_t c : perm)
    {
        out.push_back({infos[c], c});
    }

    return out;
}

// The monomial exponents (a, b, c) of the shell's z-slowest Cartesian order
// (a descending, c ascending, b = l - a - c - the ShellBasisValue walk).
struct MonomialExponents {
    int a;
    int b;
    int c;
};

// (l, index) are the degree and the Cartesian order index - distinct
// quantities, fixed call order.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
MonomialExponents MonomialExponentsOf(int l, int index) {
    int a = l;
    int c = 0;

    for (int i = 0; i < index; ++i)
    {
        if (c < l - a)
        {
            ++c;
        } else
        {
            --a;
            c = 0;
        }
    }

    return {a, l - a - c, c};
}

// The index of the (a, b, c) monomial in the same order (a descending, c
// ascending): the number of entries of the higher a's plus c.
// (l, a) are the degree and the a exponent - distinct quantities, fixed
// call order.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int MonomialIndexOf(int l, int a, int c) {
    const int d = l - a;
    return d * (d + 1) / 2 + c;
}

// The raw closed-form Lz^2 map on the monomials:
//   Lz^2 x^a y^b z^c = (2ab + a + b) x^a y^b z^c
//                      - a(a-1) x^(a-2) y^(b+2) z^c
//                      - b(b-1) x^(a+2) y^(b-2) z^c
// The map is NOT symmetric in the monomial basis for l >= 3 (e.g. the
// x^2 y <-> y^3 couplings are -2 and -6); its eigenvalues are the true
// spectrum, and the public function symmetrizes it by the Gram congruence.
Eigen::MatrixXd RawAngularMomentumSquaredMap(int l) {
    const int nAng = (l + 1) * (l + 2) / 2;
    Eigen::MatrixXd map = Eigen::MatrixXd::Zero(nAng, nAng);

    for (int i = 0; i < nAng; ++i)
    {
        const MonomialExponents e = MonomialExponentsOf(l, i);
        map(i, i) = 2.0 * static_cast<double>(e.a) * static_cast<double>(e.b) + e.a + e.b;

        if (e.a >= 2)
        {
            map(i, MonomialIndexOf(l, e.a - 2, e.c)) -= static_cast<double>(e.a) * (e.a - 1);
        }

        if (e.b >= 2)
        {
            map(i, MonomialIndexOf(l, e.a + 2, e.c)) -= static_cast<double>(e.b) * (e.b - 1);
        }
    }

    return map;
}

// The Gram matrix of the monomials over the unit sphere:
//   int x^(2a) y^(2b) z^(2c) dOmega = 2 Gamma(a+1/2) Gamma(b+1/2)
//                                     Gamma(c+1/2) / Gamma(a+b+c+3/2)
// (odd exponents vanish).
Eigen::MatrixXd MonomialGram(int l) {
    const int nAng = (l + 1) * (l + 2) / 2;
    Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(nAng, nAng);

    for (int i = 0; i < nAng; ++i)
    {
        const MonomialExponents ei = MonomialExponentsOf(l, i);

        for (int j = 0; j <= i; ++j)
        {
            const MonomialExponents ej = MonomialExponentsOf(l, j);
            const int a = ei.a + ej.a;
            const int b = ei.b + ej.b;
            const int c = ei.c + ej.c;

            if (a % 2 != 0 || b % 2 != 0 || c % 2 != 0)
            {
                continue;
            }

            const double value = 2.0 * std::tgamma(static_cast<double>(a) / 2.0 + 0.5) *
                                 std::tgamma(static_cast<double>(b) / 2.0 + 0.5) *
                                 std::tgamma(static_cast<double>(c) / 2.0 + 0.5) /
                                 std::tgamma(static_cast<double>(a + b + c) / 2.0 + 1.5);
            gram(i, j) = value;
            gram(j, i) = value;
        }
    }

    return gram;
}

// The axis letter of an axis-aligned canonical axis ("x"/"y"/"z"), or a
// decimal "(x, y, z)" fallback for a tilted axis (the linear closures are
// aligned, so the letters cover the recorded finite subsets).
std::string AxisLetter(const Eigen::Vector3d& axis) {
    const Eigen::Vector3d unit = axis.norm() > 1e-8 ? axis.normalized() : Eigen::Vector3d::Zero();
    static const char* kLetters[] = {"x", "y", "z"};

    for (int k = 0; k < 3; ++k)
    {
        Eigen::Vector3d e = Eigen::Vector3d::Zero();
        e(k) = 1.0;

        if ((unit - e).cwiseAbs().maxCoeff() < 1e-4)
        {
            return kLetters[k];
        }
    }

    return "(" + std::to_string(axis.x()) + ", " + std::to_string(axis.y()) + ", " +
           std::to_string(axis.z()) + ")";
}

// The plane letters of an aligned mirror normal ("xy" for a z normal), else
// the decimal fallback.
std::string PlaneName(const Eigen::Vector3d& normal) {
    const Eigen::Vector3d unit =
        normal.norm() > 1e-8 ? normal.normalized() : Eigen::Vector3d::Zero();

    if ((unit - Eigen::Vector3d::UnitX()).cwiseAbs().maxCoeff() < 1e-4)
    {
        return "yz";
    }

    if ((unit - Eigen::Vector3d::UnitY()).cwiseAbs().maxCoeff() < 1e-4)
    {
        return "xz";
    }

    if ((unit - Eigen::Vector3d::UnitZ()).cwiseAbs().maxCoeff() < 1e-4)
    {
        return "xy";
    }

    return "(" + std::to_string(normal.x()) + ", " + std::to_string(normal.y()) + ", " +
           std::to_string(normal.z()) + ")";
}

// The descriptor of one closure element for the linear path's documented
// symmetrization subset: "E", "i", "C2z", "S4z", "sigma_xz", ...
std::string ElementDescriptorString(const FullGroupElement& element) {
    switch (element.kind)
    {
    case qcx::symmetry::OperationKind::kIdentity:
        return "E";

    case qcx::symmetry::OperationKind::kInversion:
        return "i";

    case qcx::symmetry::OperationKind::kRotation:
        return "C" + std::to_string(element.order) + AxisLetter(element.axis);

    case qcx::symmetry::OperationKind::kSigma:
        return "sigma_" + PlaneName(element.axis);

    case qcx::symmetry::OperationKind::kImproper:
        return "S" + std::to_string(element.order) + AxisLetter(element.axis);
    }

    return "?";
}

// The degenerate-partner canonicalization shared by both paths: for every
// dim >= 2 irrep with at least two MOs, diagonalize the converged Fock
// restricted to the subspace and replace the partners by the eigenbasis -
// applied only when the subspace is entirely occupied or entirely virtual
// (the density is invariant then), skipped and recorded otherwise (the
// aufbau-straddle rule, record-don't-rotate). `dimensionOf` maps a
// label to its irrep dimension.
//
// THE MANIFOLD KEY IS (ENERGY CLUSTER, IRREP LABEL), NOT THE LABEL ALONE
// The columns are ordered by their Fock expectation and cut into
// clusters whose consecutive gaps are within kManifoldEnergyTolerance, and the
// groups are formed INSIDE those clusters. Two spatially separate manifolds
// that share one irrep label - ammonia's E irrep of C3v carries two partner
// pairs - are therefore two records rather than one; grouped under the label
// alone their union reaches the aufbau-straddle test as a single subspace, so
// partner pairs that each sit entirely on one side of the boundary are
// reported as straddling it and skipped. The density is invariant either way
// (each manifold is entirely occupied or entirely virtual), so no energy moves:
// what moves is the record and the rotation decision it drives.
template <typename DimensionOf>
void CanonicalizeDegeneratePartners(const Eigen::MatrixXd& fock,
                                    const std::vector<std::string>& labels,
                                    int occupiedCount,
                                    Eigen::MatrixXd& coefficients,
                                    std::vector<DegenerateSubspaceRecord>& canonicalized,
                                    std::vector<DegenerateSubspaceRecord>& straddled,
                                    const DimensionOf& dimensionOf) {
    const int columnCount = static_cast<int>(labels.size());

    // The Fock expectation of every column. The SCF path hands columns that are
    // energy-ordered already, but the ordering is DERIVED here rather than
    // assumed: this helper is also reached from paths whose column order is the
    // caller's. The (energy, column) pair is a total order, so the clustering
    // is deterministic even when two columns tie exactly.
    std::vector<std::pair<double, int>> ordered; // (energy, column)
    ordered.reserve(static_cast<std::size_t>(columnCount));

    for (int j = 0; j < columnCount; ++j)
    {
        const Eigen::VectorXd column = coefficients.col(j);
        ordered.emplace_back(column.dot(fock * column), j);
    }

    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
    });

    std::vector<DegenerateSubspaceRecord> manifolds;
    std::size_t start = 0;

    while (start < ordered.size())
    {
        std::size_t end = start + 1;

        while (end < ordered.size() &&
               ordered[end].first - ordered[end - 1].first <= kManifoldEnergyTolerance)
        {
            ++end;
        }

        std::map<std::string, std::vector<int>> byLabel;

        for (std::size_t k = start; k < end; ++k)
        {
            byLabel[labels[static_cast<std::size_t>(ordered[k].second)]].push_back(
                ordered[k].second);
        }

        for (auto& entry : byLabel)
        {
            if (dimensionOf(entry.first) < 2 || entry.second.size() < 2)
            {
                continue;
            }

            std::vector<int>& indices = entry.second;
            std::sort(indices.begin(), indices.end());
            manifolds.push_back(DegenerateSubspaceRecord{entry.first, indices});
        }

        start = end;
    }

    // Emitted in ascending order of their lowest column, so the record order is
    // the coefficient order rather than the energy ordering.
    std::sort(manifolds.begin(), manifolds.end(), [](const auto& a, const auto& b) {
        return a.moIndices.front() < b.moIndices.front();
    });

    for (auto& record : manifolds)
    {
        std::vector<int>& indices = record.moIndices;
        const int occupiedIn = static_cast<int>(std::count_if(
            indices.begin(), indices.end(), [occupiedCount](int j) { return j < occupiedCount; }));

        if (occupiedIn != 0 && occupiedIn != static_cast<int>(indices.size()))
        {
            straddled.push_back(std::move(record));
            continue;
        }

        const int subspaceSize = static_cast<int>(indices.size());
        Eigen::MatrixXd subspace(coefficients.rows(), subspaceSize);

        for (int t = 0; t < subspaceSize; ++t)
        {
            subspace.col(t) = coefficients.col(indices[static_cast<std::size_t>(t)]);
        }

        const Eigen::MatrixXd restricted = subspace.transpose() * fock * subspace;
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(restricted);
        const Eigen::MatrixXd rotated = subspace * solver.eigenvectors();

        for (int t = 0; t < subspaceSize; ++t)
        {
            coefficients.col(indices[static_cast<std::size_t>(t)]) = rotated.col(t);
        }

        canonicalized.push_back(std::move(record));
    }
}

} // namespace

qcx::Result<SymmetryBlocks> BuildSymmetryBlocks(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const qcx::symmetry::SymmetryAnalysis& analysis) {
    const auto shellList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!shellList.has_value())
    {
        return std::unexpected(shellList.error());
    }

    const Eigen::Index n = static_cast<Eigen::Index>(shellList->functionCount);

    if (analysis.computational == qcx::symmetry::PointGroup::kC1)
    {
        SymmetryBlocks trivial;
        trivial.u = Eigen::MatrixXd::Identity(n, n);
        trivial.blockSizes.push_back(n);
        trivial.isTrivial = true;
        return trivial;
    }

    const auto groupData = BuildGroupData(analysis, molecule);

    if (!groupData.has_value())
    {
        return std::unexpected(groupData.error());
    }

    std::vector<Eigen::MatrixXd> actions;
    actions.reserve(groupData->elements.size());
    std::vector<std::vector<std::size_t>> permutations;
    permutations.reserve(groupData->elements.size());

    for (const auto& element : groupData->elements)
    {
        const auto permutation = AtomPermutationFor(molecule, element, kPositionToleranceBohr);

        if (!permutation.has_value())
        {
            return std::unexpected(permutation.error());
        }

        const auto action = BuildAoAction(*shellList, *permutation, element);

        if (!action.has_value())
        {
            return std::unexpected(action.error());
        }

        permutations.push_back(*permutation);
        actions.push_back(*action);
    }

    // Representation validation: the realized matrices must satisfy
    // A(g) A(h) = A(gh) - a wrong group (e.g. a generator set that does not
    // realize the computational group) would silently corrupt the projector
    // ranks below instead of failing loudly.
    for (std::size_t g = 0; g < actions.size(); ++g)
    {
        for (std::size_t h = 0; h < actions.size(); ++h)
        {
            const auto product = actions[g] * actions[h];
            bool found = false;

            for (std::size_t k = 0; k < actions.size(); ++k)
            {
                if ((actions[k] - product).norm() < kMatrixDedupTolerance)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                                  "the group action is not a representation of "
                                                  "the computational group"});
            }
        }
    }

    // The symmetrizing unitary: per irrep, the orthonormal range of the
    // projector P = (1/|G|) sum_g chi(g) A(g) (an orthogonal projector, since
    // the A(g) form a representation of the group and every element is an
    // involution). Columns are ordered irrep-major.
    const double groupOrderInv = 1.0 / static_cast<double>(groupData->elements.size());
    SymmetryBlocks blocks;
    blocks.u = Eigen::MatrixXd::Zero(n, n);
    Eigen::Index offset = 0;

    for (const auto& chi : groupData->characters)
    {
        Eigen::MatrixXd projector = Eigen::MatrixXd::Zero(n, n);

        for (std::size_t g = 0; g < groupData->elements.size(); ++g)
        {
            projector += static_cast<double>(chi[g]) * actions[g];
        }

        projector *= groupOrderInv;
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(projector);
        // The eigenvalues ascend, so the > 0.5 subspace (eigenvalue 1, the
        // range of the projector) sits on the RIGHT of the eigenvectors.
        const Eigen::Index blockSize = static_cast<Eigen::Index>(
            (solver.eigenvalues().array() > kProjectorEigenvalueCutoff).count());
        blocks.u.middleCols(offset, blockSize) = solver.eigenvectors().rightCols(blockSize);
        blocks.blockSizes.push_back(blockSize);
        offset += blockSize;
    }

    if (offset != n)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                          "the projector ranks do not sum to the basis size"});
    }

    blocks.aoOrbits = ComputeAoOrbits(*shellList, permutations, molecule.AtomCount());

    // The generators' signs on the symmetrized basis (the adaptation
    // guard's reference, see SymmetryBlocks::generatorSigns). Each generator
    // is realized through the SAME two steps the elements above went through
    // (the atom permutation, then the exact angular action), and the realized
    // action must be a signed identity on every column of u: U^T A(g) U is
    // exactly diag(signs) then, which is what makes the guard's O(n^2)
    // per-generator commutator equal to the O(n^3) one it replaces. A
    // generator that fails either check is a broken group realization, not a
    // chemistry statement - the caller falls back to the plain path, which is
    // always correct (kUnimplemented is the fall-back code, exactly as for an
    // unrealizable group).
    for (const Eigen::Matrix3d& generator : groupData->generators)
    {
        const auto permutation = AtomPermutationFor(molecule, generator, kPositionToleranceBohr);

        if (!permutation.has_value())
        {
            return std::unexpected(permutation.error());
        }

        const auto action = BuildAoAction(*shellList, *permutation, generator);

        if (!action.has_value())
        {
            return std::unexpected(action.error());
        }

        const Eigen::MatrixXd images = (*action) * blocks.u;
        Eigen::VectorXd signs(n);

        for (Eigen::Index i = 0; i < n; ++i)
        {
            const double projection = blocks.u.col(i).dot(images.col(i));

            if (std::abs(std::abs(projection) - 1.0) > kGeneratorSignedActionTolerance ||
                (images.col(i) - std::copysign(1.0, projection) * blocks.u.col(i)).norm() >
                    kGeneratorSignedActionTolerance)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kUnimplemented,
                               "a generator of the computational group is not a signed identity on "
                               "the symmetrized basis"});
            }

            signs[i] = std::copysign(1.0, projection);
        }

        blocks.generatorSigns.push_back(std::move(signs));
    }

    return blocks;
}

// The largest subgroup of the realized computational group whose elements
// all pass the axis-alignment gate, returned as the original element indices
// (identity first). The groups here are tiny (order <= 8, every element an
// involution of an Abelian group), so every subset of the non-identity
// elements is enumerated; a subset is a candidate when it is closed under
// the group multiplication and every member aligns. The search is
// deterministic: masks enumerate in ascending order and ties keep the first
// (smallest) mask, so the subgroup whose members carry the smallest element
// indices wins equal-size races.
std::vector<std::size_t> LargestAxisAlignedSubgroup(const std::vector<Eigen::Matrix3d>& elements,
                                                    const std::vector<bool>& axisAligned) {
    const std::size_t order = elements.size();

    // The multiplication table over the original element indices. BuildGroupData
    // deduplicates the subset products with the same tolerance, so every
    // product matches an element; a mismatch would leave a sentinel that
    // fails the membership checks below.
    std::vector<std::vector<std::size_t>> table(order, std::vector<std::size_t>(order, order));

    for (std::size_t g = 0; g < order; ++g)
    {
        for (std::size_t h = 0; h < order; ++h)
        {
            const Eigen::Matrix3d product = elements[g] * elements[h];

            for (std::size_t k = 0; k < order; ++k)
            {
                if ((product - elements[k]).norm() < kMatrixDedupTolerance)
                {
                    table[g][h] = k;
                    break;
                }
            }
        }
    }

    std::vector<std::size_t> best;
    best.push_back(0); // The identity alone always aligns and always closes.

    for (std::uint32_t mask = 1; order > 1 && mask < (1u << (order - 1)); ++mask)
    {
        std::vector<std::size_t> candidate;
        candidate.reserve(order);
        candidate.push_back(0);

        for (std::size_t g = 1; g < order; ++g)
        {
            if ((mask & (1u << (g - 1))) != 0)
            {
                candidate.push_back(g);
            }
        }

        bool valid = std::all_of(candidate.begin(), candidate.end(), [&axisAligned](std::size_t g) {
            return axisAligned[g];
        });

        if (valid)
        {
            for (const std::size_t g : candidate)
            {
                for (const std::size_t h : candidate)
                {
                    if (std::find(candidate.begin(), candidate.end(), table[g][h]) ==
                        candidate.end())
                    {
                        valid = false;
                        break;
                    }
                }

                if (!valid)
                {
                    break;
                }
            }
        }

        if (valid && candidate.size() > best.size())
        {
            best = std::move(candidate);
        }
    }

    return best;
}

qcx::Result<qcx::integrals::SymmetryReduction> BuildSymmetryReduction(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::symmetry::SymmetryAnalysis& analysis) {
    const auto shellList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!shellList.has_value())
    {
        return std::unexpected(shellList.error());
    }

    if (analysis.computational == qcx::symmetry::PointGroup::kC1)
    {
        return qcx::integrals::SymmetryReduction{};
    }

    const auto groupData = BuildGroupData(analysis, molecule);

    if (!groupData.has_value())
    {
        return std::unexpected(groupData.error());
    }

    const std::size_t n = shellList->functionCount;
    const std::size_t order = groupData->elements.size();
    // The petite list requires the AO action to be a signed permutation in
    // the global function basis: every realized element must be a signed
    // permutation of the global coordinate axes, so the angular action is a
    // +/-1 diagonal over the spherical components. For a molecule whose
    // symmetry planes/axes are not the global coordinate planes/axes the
    // angular action is a dense m-mixing matrix and the reduction is
    // inapplicable. Failures are collected per element instead of bailing:
    // the aligned elements close into a subgroup, and blocking by that
    // subgroup is equally valid - F commutes with every element of the full
    // molecular group, hence with every element of the subgroup (the methane
    // Td -> D2 precedent). The largest such subgroup is selected below; when
    // only the identity aligns, kUnimplemented lets the caller fall back to
    // the plain path. (The sample-point angular action is exact up to the
    // solve's rounding, ~1e-12, far below the 1e-8 acceptance threshold; a
    // dense row's entries are bounded away from +-1 by construction, so the
    // threshold cannot accept a near-axis molecule whose mixing would break
    // the machine-precision equivalence the petite list promises.)
    constexpr double kSignedPermutationTolerance = 1e-8;

    std::vector<std::vector<std::size_t>> permutations(order);
    std::vector<std::vector<int>> signs(order);
    std::vector<bool> axisAligned(order, false);
    std::optional<qcx::Error> firstGateError;

    for (std::size_t g = 0; g < order; ++g)
    {
        const auto atomPermutation =
            AtomPermutationFor(molecule, groupData->elements[g], kPositionToleranceBohr);

        if (!atomPermutation.has_value())
        {
            return std::unexpected(atomPermutation.error());
        }

        const auto action = BuildAoAction(*shellList, *atomPermutation, groupData->elements[g]);

        if (!action.has_value())
        {
            return std::unexpected(action.error());
        }

        std::vector<std::size_t> permutation(n);
        std::vector<int> sign(n);
        std::optional<qcx::Error> gateError;

        for (Eigen::Index i = 0; i < action->rows(); ++i)
        {
            std::size_t image = n;
            int imageSign = 0;

            for (Eigen::Index j = 0; j < action->cols(); ++j)
            {
                const double value = (*action)(i, j);

                if (std::abs(value) < kSignedPermutationTolerance)
                {
                    continue;
                }

                if (image != n || std::abs(std::abs(value) - 1.0) > kSignedPermutationTolerance)
                {
                    gateError = qcx::Error{qcx::ErrorCode::kUnimplemented,
                                           "the AO action is not a signed permutation in the "
                                           "global function basis (the molecule's symmetry frame "
                                           "must be the global coordinate frame)"};
                    break;
                }

                image = static_cast<std::size_t>(j);
                imageSign = value > 0.0 ? 1 : -1;
            }

            if (gateError.has_value())
            {
                break;
            }

            if (image == n)
            {
                gateError = qcx::Error{qcx::ErrorCode::kUnimplemented,
                                       "the AO action row is zero - not a signed permutation"};
                break;
            }

            permutation[static_cast<std::size_t>(i)] = image;
            sign[static_cast<std::size_t>(i)] = imageSign;
        }

        if (gateError.has_value())
        {
            if (!firstGateError.has_value())
            {
                firstGateError = std::move(gateError);
            }

            continue;
        }

        permutations[g] = std::move(permutation);
        signs[g] = std::move(sign);
        axisAligned[g] = true;
    }

    // The realized element subset: the full computational group when every
    // element aligns, otherwise the largest axis-aligned subgroup.
    std::vector<std::size_t> subset;
    subset.reserve(order);

    if (std::all_of(axisAligned.begin(), axisAligned.end(), [](bool aligned) { return aligned; }))
    {
        for (std::size_t g = 0; g < order; ++g)
        {
            subset.push_back(g);
        }
    } else
    {
        subset = LargestAxisAlignedSubgroup(groupData->elements, axisAligned);

        if (subset.size() <= 1)
        {
            return std::unexpected(
                firstGateError.has_value()
                    ? *firstGateError
                    : qcx::Error{qcx::ErrorCode::kUnimplemented,
                                 "the AO action is not a signed permutation in the global "
                                 "function basis (the molecule's symmetry frame must be the "
                                 "global coordinate frame)"});
        }
    }

    qcx::integrals::SymmetryReduction reduction;
    reduction.groupOrder = subset.size();
    reduction.permutation.resize(subset.size());
    reduction.sign.resize(subset.size());

    for (std::size_t r = 0; r < subset.size(); ++r)
    {
        const std::size_t g = subset[r];
        reduction.permutation[r] = std::move(permutations[g]);
        reduction.sign[r] = std::move(signs[g]);
    }

    // THE ENGAGEMENT TEST IS THE GROUP'S ACTION ON SHELL PAIRS (the
    // mechanism-value audit, 2026-09-13), not the group's order. The flag
    // below has always meant "no symmetry to exploit", but it was set from
    // "a non-C1 group was found", which is a different statement. An
    // element that permutes no function - a pure sign flip, which is
    // exactly what an element induces when its coordinate permutation
    // fixes every atom - maps every canonical shell pair onto itself, so
    // every orbit is a singleton, the class tables enumerate the plain
    // cells and the reduction removes nothing while still paying for the
    // classification, the class tables and the orbit-action tables. The
    // measured case: hocl/Cs (every atom in the mirror plane, 13 BF) is
    // classified Cs, engaged, and buys EXACTLY 1.0000x on every ERI
    // counter - 987 blocks either way. A planar molecule is the general
    // shape of it (the out-of-plane mirror permutes no atom), which is why
    // benzene's D2h realisation plateaus at 3.99 of an 8-element group
    // rather than reaching it.
    //
    // Consumers read this flag to decide whether to engage at all - the
    // class path (fock_build.cpp), the density symmetrization
    // (symmetry_reduction.cpp) and the lean builder's mask and orbit
    // derivations (lean_fock_build.cpp) - so setting it from the pair
    // action makes every one of them produce the C1 path, byte for byte,
    // for a group that cannot reduce the ERI work.
    //
    // shellOfFunction[f] = the shell containing function f. The image of a
    // shell under an element is the shell containing the image of its
    // first function; the walk below VERIFIES that the element maps each
    // shell INTO one shell rather than assuming it, because a signed
    // permutation that split a shell across two others would make the pair
    // action undefined and the first-function shortcut wrong.
    std::vector<std::size_t> shellOfFunction(n, 0);

    for (std::size_t s = 0; s < shellList->shells.size(); ++s)
    {
        const qcx::integrals::ShellInfo& shell = shellList->shells[s];

        for (std::size_t f = shell.functionOffset;
             f < shell.functionOffset + qcx::integrals::ShellFunctionCount(shell);
             ++f)
        {
            shellOfFunction[f] = s;
        }
    }

    std::vector<std::size_t> shellImage(shellList->shells.size(), 0);
    bool pairActionIsIdentity = true;

    for (std::size_t r = 0; r < subset.size() && pairActionIsIdentity; ++r)
    {
        const std::vector<std::size_t>& permutation = reduction.permutation[r];
        bool shellClosed = true;

        for (std::size_t s = 0; s < shellList->shells.size() && shellClosed; ++s)
        {
            const qcx::integrals::ShellInfo& shell = shellList->shells[s];
            const std::size_t image = shellOfFunction[permutation[shell.functionOffset]];
            shellImage[s] = image;

            for (std::size_t f = shell.functionOffset;
                 f < shell.functionOffset + qcx::integrals::ShellFunctionCount(shell) &&
                 shellClosed;
                 ++f)
            {
                if (shellOfFunction[permutation[f]] != image)
                {
                    shellClosed = false;
                }
            }
        }

        // An element that does not map shells onto shells cannot be a
        // shell-pair action at all; the signed-permutation gate above makes
        // that unreachable, and treating it as non-trivial keeps the
        // failure loud (the tables still carry the raw action) instead of
        // silently claiming a reduction that does not exist.
        if (!shellClosed)
        {
            pairActionIsIdentity = false;
            break;
        }

        for (const qcx::integrals::ShellPairIndex& pair : shellList->pairs)
        {
            const std::size_t imageI = shellImage[pair.i];
            const std::size_t imageJ = shellImage[pair.j];
            // The canonical pair is UNORDERED (i <= j), so an element that
            // swaps the pair's two shells fixes the pair and its orbit stays
            // a singleton - the within-pair flip the lean walk's own member
            // dedup already handles.
            const bool fixesPair =
                (imageI == pair.i && imageJ == pair.j) || (imageI == pair.j && imageJ == pair.i);

            if (!fixesPair)
            {
                pairActionIsIdentity = false;
                break;
            }
        }
    }

    reduction.isTrivial = pairActionIsIdentity;

    // Representation validation on the quantized data: the extracted signed
    // permutations must multiply like the selected group - the image of i
    // under the product A(g) A(h) is pi_h(pi_g(i)) with the sign product
    // s_g(i) * s_h(pi_g(i)), and the result must match one element exactly.
    // A wrong realization (e.g. a generator set that does not realize the
    // computational group) would silently corrupt the class signs instead
    // of failing loudly.
    for (std::size_t r = 0; r < subset.size(); ++r)
    {
        for (std::size_t s = 0; s < subset.size(); ++s)
        {
            bool found = false;

            for (std::size_t t = 0; t < subset.size(); ++t)
            {
                bool matches = true;

                for (std::size_t i = 0; i < n && matches; ++i)
                {
                    const std::size_t mid = reduction.permutation[r][i];
                    const std::size_t productImage = reduction.permutation[s][mid];
                    const int productSign = reduction.sign[r][i] * reduction.sign[s][mid];

                    if (reduction.permutation[t][i] != productImage ||
                        reduction.sign[t][i] != productSign)
                    {
                        matches = false;
                    }
                }

                if (matches)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kUnimplemented,
                               "the extracted signed permutations do not form a representation "
                               "of the computational group"});
            }
        }
    }

    return reduction;
}

qcx::Result<Eigen::MatrixXd> DiagonalizeFockBlocked(
    const Eigen::MatrixXd& fock,
    const Eigen::MatrixXd& x,
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::symmetry::SymmetryAnalysis& analysis) {
    const auto blocks = BuildSymmetryBlocks(molecule, basisSet, analysis);

    if (!blocks.has_value())
    {
        if (blocks.error().code == qcx::ErrorCode::kUnimplemented)
        {
            return DiagonalizeFock(fock, x);
        }

        return std::unexpected(blocks.error());
    }

    const auto data = BuildBlockedDiagonalizeData(x, *blocks);

    if (!data.has_value())
    {
        return std::unexpected(data.error());
    }

    return DiagonalizeFockBlocked(fock, x, *blocks, *data);
}

qcx::Result<BlockedDiagonalizeData> BuildBlockedDiagonalizeData(const Eigen::MatrixXd& x,
                                                                const SymmetryBlocks& blocks) {
    BlockedDiagonalizeData data;
    data.isTrivial = blocks.isTrivial;

    if (blocks.isTrivial)
    {
        return data;
    }

    // A rectangular X (the linear-dependence removal dropped directions)
    // has no per-irrep square block: U_b^T X U_b does not even multiply
    // here, because X's kept dimension is not the blocks' n. Mark the data
    // trivial so the blocked solver falls back to the plain n-dimensional
    // solve - the fallback's reasoning lives in DiagonalizeFockBlocked.
    if (x.rows() != x.cols())
    {
        data.isTrivial = true;
        return data;
    }

    const Eigen::Index n = static_cast<Eigen::Index>(x.rows());
    Eigen::Index offset = 0;

    for (const Eigen::Index blockSize : blocks.blockSizes)
    {
        if (blockSize == 0)
        {
            continue;
        }

        // X_bb = U_b^T X U_b: X = S^{-1/2} is group-invariant (S commutes
        // with the group action), so U^T X U is block-diagonal up to noise
        // and this is its diagonal block. Computed ONCE per run - the
        // per-iteration transform sandwiches F with it.
        Eigen::MatrixXd xBlock = blocks.u.block(0, offset, n, blockSize).transpose() * x *
                                 blocks.u.block(0, offset, n, blockSize);
        data.offsets.push_back(offset);
        data.sizes.push_back(blockSize);
        data.xBlocks.push_back(std::move(xBlock));
        offset += blockSize;
    }

    // Y = X U: the AO -> symmetrized-orthonormal transform of the adaptation
    // guard (MeasureSymmetryAdaptation - one n x n product here, once per
    // run, instead of two per iteration per spin). Built only where the guard
    // can be reached, i.e. where the blocked solve is not already delegating
    // to the plain path.
    data.symmetrizedBasis = x * blocks.u;
    return data;
}

SymmetryAdaptation MeasureSymmetryAdaptation(const Eigen::MatrixXd& fock,
                                             const SymmetryBlocks& blocks,
                                             const BlockedDiagonalizeData& data) {
    SymmetryAdaptation adaptation;

    if (blocks.isTrivial || data.symmetrizedBasis.size() == 0)
    {
        return adaptation;
    }

    // M = Y^T F Y: the Fock in the symmetrized orthonormal basis (the blocked
    // solve's own X^T F X, symmetrized). Two multiplies per spin per
    // diagonalization; everything after this is O(n^2) per generator, because
    // each generator's action in THIS basis is exactly its signed diagonal.
    const Eigen::MatrixXd m = data.symmetrizedBasis.transpose() * fock * data.symmetrizedBasis;
    const double fockNorm = m.norm();

    if (fockNorm <= 0.0)
    {
        // The zero matrix commutes with everything; a matrix that cannot
        // violate the precondition is not one to refuse on.
        return adaptation;
    }

    for (const Eigen::VectorXd& signs : blocks.generatorSigns)
    {
        // [M, diag(signs)] has entries (signs_j - signs_i) M_ij: the
        // commutator with a diagonal matrix is a scaling per entry, so the
        // Frobenius norm is a sum of squares over the entries whose signs
        // differ - no product to form.
        const Eigen::RowVectorXd signRow = signs.transpose();
        double squaredNorm = 0.0;

        for (Eigen::Index i = 0; i < m.rows(); ++i)
        {
            const Eigen::RowVectorXd scaled =
                (signRow.array() - signs[i]).matrix().cwiseProduct(m.row(i));
            squaredNorm += scaled.squaredNorm();
        }

        adaptation.generatorCommutatorNorms.push_back(std::sqrt(squaredNorm) / fockNorm);
    }

    for (const double norm : adaptation.generatorCommutatorNorms)
    {
        adaptation.maxCommutatorNorm = std::max(adaptation.maxCommutatorNorm, norm);
    }

    return adaptation;
}

qcx::Result<Eigen::MatrixXd> DiagonalizeFockBlocked(const Eigen::MatrixXd& fock,
                                                    const Eigen::MatrixXd& x,
                                                    const SymmetryBlocks& blocks,
                                                    const BlockedDiagonalizeData& data) {
    if (blocks.isTrivial || data.isTrivial)
    {
        return DiagonalizeFock(fock, x);
    }

    // The linear-dependence removal (scf_common.hpp OrthogonalizeOverlap)
    // makes X RECTANGULAR, n x numKept, and the per-irrep scheme below needs
    // the square case: its block sizes partition n, not numKept, and the
    // removed directions are not distributed one-per-irrep. Falling back to
    // the plain n-dimensional solve is EXACT (both paths diagonalize the same
    // X^T F X; the blocked form is a speed optimization, not a different
    // answer), and a run that removed directions is by definition the
    // ill-conditioned case where correctness outranks the speed. The blocked
    // path still handles every well-conditioned run unchanged, which is all
    // of them except the diffuse-basis case this removal exists for.
    if (x.rows() != x.cols())
    {
        return DiagonalizeFock(fock, x);
    }

    const Eigen::Index n = static_cast<Eigen::Index>(fock.rows());
    Eigen::VectorXd eigenvalues(n);
    // The per-irrep eigenvectors (block-embedded) and X_diag times them:
    // C = X U W = U (X_diag W) up to the off-block noise of U^T X U,
    // so the back-transform never forms a full n x n product with X.
    Eigen::MatrixXd w = Eigen::MatrixXd::Zero(n, n);
    Eigen::MatrixXd xw = Eigen::MatrixXd::Zero(n, n);
    const std::size_t numBlocks = data.sizes.size();

    // (the threaded eigensolve exception to the BLAS-sequential
    // rule): the per-irrep transforms and solves are independent (disjoint
    // outputs, pure Eigen kernels - no BLAS inside the region), so the whole
    // block loop runs on the OpenMP team. The results are bit-identical
    // across team sizes: every block's work is a pure function of its inputs.
    qcx::backend::Backend<qcx::backend::CpuTag>{}.ParallelFor(numBlocks, [&](std::size_t b) {
        const Eigen::Index blockSize = data.sizes[b];
        const Eigen::Index offset = data.offsets[b];
        // The symmetrized orthogonal-basis matrix per irrep:
        // g_b = X_bb^T (U_b^T F U_b) X_bb. X = S^{-1/2} commutes with the
        // group, so U^T X U is block-diagonal up to noise and the per-irrep
        // sandwich replaces the two full n x n transforms.
        const auto uBlock = blocks.u.block(0, offset, n, blockSize);
        const Eigen::MatrixXd fBlock = uBlock.transpose() * fock * uBlock;
        const Eigen::MatrixXd gBlock = data.xBlocks[b].transpose() * fBlock * data.xBlocks[b];
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(gBlock);
        const Eigen::MatrixXd& blockVectors = solver.eigenvectors();
        eigenvalues.segment(offset, blockSize) = solver.eigenvalues();
        w.block(offset, offset, blockSize, blockSize) = blockVectors;
        xw.block(offset, offset, blockSize, blockSize) = data.xBlocks[b] * blockVectors;
    });

    // The plain path sorts globally ascending; the per-block solves sorted
    // within their blocks only, so the embedded result is re-sorted.
    std::vector<Eigen::Index> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&eigenvalues](Eigen::Index i, Eigen::Index j) {
        return eigenvalues(i) < eigenvalues(j);
    });
    // C = X U W P = (U X_diag W) P: the re-sort permutation applies to the
    // result's columns AFTER the products (a column permutation is exact
    // data movement), so the back-transform costs one n x n GEMM instead of
    // two.
    const Eigen::MatrixXd uTimesXw = blocks.u * xw;
    Eigen::MatrixXd coefficients(n, n);

    for (Eigen::Index k = 0; k < n; ++k)
    {
        coefficients.col(k) = uTimesXw.col(order[static_cast<std::size_t>(k)]);
    }

    return coefficients;
}

double OffBlockNorm(const Eigen::MatrixXd& matrix, const SymmetryBlocks& blocks) {
    const Eigen::MatrixXd transformed = blocks.u.transpose() * matrix * blocks.u;
    Eigen::MatrixXd offBlock = transformed;
    Eigen::Index offset = 0;

    for (const Eigen::Index blockSize : blocks.blockSizes)
    {
        offBlock.block(offset, offset, blockSize, blockSize).setZero();
        offset += blockSize;
    }

    const double offMax = offBlock.cwiseAbs().maxCoeff();
    const double scale = std::max(transformed.cwiseAbs().maxCoeff(), 1e-30);
    return offMax / scale;
}

// ---- Full-group labeling stage ----

qcx::Result<FullGroupRealization> BuildFullGroupRealization(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::symmetry::SymmetryAnalysis& analysis) {
    const auto shellList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!shellList.has_value())
    {
        return std::unexpected(shellList.error());
    }

    const auto seeds = ClosureSeeds(analysis);
    std::vector<ClosureWitness> witnesses;
    const auto closure = BuildClosure(seeds, witnesses);
    FullGroupRealization out;
    out.elements.reserve(closure.size());
    out.actions.reserve(closure.size());
    std::vector<int> fixedCounts;
    fixedCounts.reserve(closure.size());

    for (const auto& matrix : closure)
    {
        const ClassifiedOperation op = ClassifyOperation(matrix);
        FullGroupElement element;
        element.matrix = matrix;
        element.kind = op.kind;
        element.order = op.order;
        element.power = op.power;
        element.axis = op.axis;
        const auto permutation = AtomPermutationFor(molecule, matrix, kPositionToleranceBohr);

        if (!permutation.has_value())
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "a full-group operation does not permute the atoms"});
        }

        const auto action = BuildAoAction(*shellList, *permutation, matrix);

        if (!action.has_value())
        {
            return std::unexpected(action.error());
        }

        fixedCounts.push_back(static_cast<int>(std::count_if(
            permutation->begin(),
            permutation->end(),
            [index = std::size_t{0}](std::size_t image) mutable { return image == index++; })));
        out.elements.push_back(std::move(element));
        out.actions.push_back(*action);
    }

    // The representation guard on the closure-defining witness pairs:
    // A(closure[parent]) A(closure[seed]) = A(closure[product]) along every
    // discovery chain (all witness indices are closure indices - the guard
    // used to index the actions by the seeds-array position, which only
    // coincided with the closure position by construction) certifies the
    // action of every element inductively.
    for (const auto& witness : witnesses)
    {
        const Eigen::MatrixXd product = out.actions[witness.parent] * out.actions[witness.seed];
        const double error = (product - out.actions[witness.product]).cwiseAbs().maxCoeff();

        if (error > kRepresentationGuardTolerance)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "the realized AO action is not a representation of the closure"});
        }
    }

    // The classes in the generator's table order, with the classOfElement
    // indices into that sorted list. The orbit slots are ranked in the
    // standard-orientation frame so the assignment matches the generated
    // tables for any molecule orientation (ComputeStandardFrame).
    const auto rawClasses = ConjugacyClasses(closure);
    std::vector<RuntimeClassInfo> infos = ClassifyRawClasses(closure, rawClasses);
    const std::optional<std::size_t> principalIndex =
        SelectPrincipal(infos, out.elements, fixedCounts);
    Eigen::Vector3d principalAxis = Eigen::Vector3d::Zero();

    if (principalIndex.has_value())
    {
        principalAxis = infos[*principalIndex].minAxis;
    } else
    {
        for (const RuntimeClassInfo& info : infos)
        {
            if (info.kind == qcx::symmetry::OperationKind::kSigma)
            {
                principalAxis = info.minAxis;
                break;
            }
        }
    }

    const Eigen::Matrix3d frame = ComputeStandardFrame(principalAxis, out.elements, fixedCounts);
    AssignSlots(infos, frame, out.elements, fixedCounts);
    const auto classInfos = SortClassInfos(std::move(infos));
    out.classOfElement.assign(closure.size(), 0);

    for (std::size_t c = 0; c < classInfos.size(); ++c)
    {
        const RuntimeClassInfo& info = classInfos[c].first;
        FullGroupClass cls;
        cls.kind = info.kind;
        cls.order = info.order;
        cls.power = info.power;
        cls.slot = info.slot;
        cls.minAxis = info.minAxis;
        out.classes.push_back(std::move(cls));

        for (const int member : rawClasses[classInfos[c].second])
        {
            out.classOfElement[static_cast<std::size_t>(member)] = static_cast<int>(c);
        }
    }

    out.groupOrder = static_cast<int>(closure.size());
    return out;
}

Eigen::MatrixXd AngularMomentumSquaredAboutZ(int l, bool isSpherical) {
    if (isSpherical)
    {
        Eigen::MatrixXd operatorMatrix = Eigen::MatrixXd::Zero(2 * l + 1, 2 * l + 1);

        for (int index = 0; index < 2 * l + 1; ++index)
        {
            const double m = static_cast<double>(index - l);
            operatorMatrix(index, index) = m * m;
        }

        return operatorMatrix;
    }

    // The Cartesian monomial map is not symmetric for l >= 3 (the x^2 y <->
    // y^3 couplings are -2 and -6), so SelfAdjointEigenSolver needs the
    // Gram congruence. The raw map M stores each image in a row, i.e.
    // M(i, j) = <image of monomial i | monomial j>; the operator matrix is
    // therefore M^T, and self-adjointness in the Gram inner product reads
    // M G = G M^T. The congruence S = G^(1/2) M^T G^(-1/2) is then symmetric
    // (S^T = S follows from M G = G M^T) and similar to the operator, so the
    // spectrum is preserved.
    const Eigen::MatrixXd map = RawAngularMomentumSquaredMap(l).transpose();
    const Eigen::MatrixXd gram = MonomialGram(l);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> gramSolver(gram);
    const Eigen::MatrixXd sqrtGram = gramSolver.eigenvectors() *
                                     gramSolver.eigenvalues().cwiseSqrt().asDiagonal() *
                                     gramSolver.eigenvectors().transpose();
    const Eigen::MatrixXd inverseSqrtGram =
        gramSolver.eigenvectors() *
        gramSolver.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() *
        gramSolver.eigenvectors().transpose();
    return sqrtGram * map * inverseSqrtGram;
}

qcx::Result<qcx::scf::SymmetryLabels> SymmetryLabelAndSymmetrize(
    const Eigen::MatrixXd& fock,
    const Eigen::MatrixXd& coefficients,
    const Eigen::MatrixXd& density,
    int occupiedCount,
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::symmetry::SymmetryAnalysis& analysis) {
    const Eigen::Index n = coefficients.rows();

    if (coefficients.cols() != n || density.rows() != n || density.cols() != n ||
        fock.rows() != n || fock.cols() != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "non-square or mismatched matrices"});
    }

    if (occupiedCount < 0 || occupiedCount > n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "occupiedCount outside the MO range"});
    }

    const auto realization = BuildFullGroupRealization(molecule, basisSet, analysis);

    if (!realization.has_value())
    {
        return std::unexpected(realization.error());
    }

    SymmetryLabels out;
    out.fullGroup = analysis.group;
    out.abelianReduction = analysis.computational;
    out.coefficients = coefficients;
    out.density = density;

    const bool linear = analysis.group == qcx::symmetry::PointGroupName::kCInfV ||
                        analysis.group == qcx::symmetry::PointGroupName::kDInfH;

    if (linear)
    {
        // The principal axis: the first rotation element of the analysis
        // (the detection records the molecular-axis C2 first for linear
        // molecules).
        Eigen::Vector3d principal = Eigen::Vector3d::Zero();

        for (const auto& element : analysis.elements)
        {
            if (element.kind == qcx::symmetry::OperationKind::kRotation)
            {
                principal = element.axis.normalized();
                break;
            }
        }

        if (principal.norm() == 0.0)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "a linear molecule without a principal axis"});
        }

        // The z-axis operator conjugated to the principal axis: the global
        // matrix M = D Lz^2 D^T with D the angular action of the rotation
        // taking z to the principal axis (the same per-shell block on every
        // contraction row, matching the engine's function order).
        const auto shellList = qcx::integrals::BuildShellPairs(molecule, basisSet);

        if (!shellList.has_value())
        {
            return std::unexpected(shellList.error());
        }

        const Eigen::Matrix3d frame =
            Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), principal)
                .toRotationMatrix();
        Eigen::MatrixXd angularSquared = Eigen::MatrixXd::Zero(n, n);

        for (const auto& shell : shellList->shells)
        {
            const auto angular = AngularAction(shell.angularMomentum, shell.isSpherical, frame);

            if (!angular.has_value())
            {
                return std::unexpected(angular.error());
            }

            const Eigen::MatrixXd aboutZ =
                AngularMomentumSquaredAboutZ(shell.angularMomentum, shell.isSpherical);
            const Eigen::MatrixXd block = *angular * aboutZ * angular->transpose();
            const Eigen::Index nAng = static_cast<Eigen::Index>(block.rows());

            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(shell.contractionCount); ++c)
            {
                for (Eigen::Index a = 0; a < nAng; ++a)
                {
                    for (Eigen::Index ap = 0; ap < nAng; ++ap)
                    {
                        const Eigen::Index row =
                            static_cast<Eigen::Index>(shell.functionOffset) + c * nAng + a;
                        const Eigen::Index col =
                            static_cast<Eigen::Index>(shell.functionOffset) + c * nAng + ap;
                        angularSquared(row, col) = block(a, ap);
                    }
                }
            }
        }

        // The inversion and sigma-v elements of the closure (the sigma-v
        // planes are perpendicular to the principal axis).
        int inversionIndex = -1;
        int sigmaVIndex = -1;

        for (int g = 0; g < realization->groupOrder; ++g)
        {
            const FullGroupElement& element = realization->elements[static_cast<std::size_t>(g)];

            if (element.kind == qcx::symmetry::OperationKind::kInversion)
            {
                inversionIndex = g;
            }

            if (element.kind == qcx::symmetry::OperationKind::kSigma &&
                std::abs(element.axis.dot(principal)) < kAxisParallelTolerance && sigmaVIndex < 0)
            {
                sigmaVIndex = g;
            }
        }

        // The per-MO labels: |lambda| from the operator's expectation, the
        // Sigma+/- parity from the sigma-v element, g/u from the inversion.
        static const char* kLambdaLetters[] = {"S", "Pi", "D", "F", "G", "H"};
        out.labels.reserve(static_cast<std::size_t>(n));
        out.irrepIndices.assign(static_cast<std::size_t>(n), -1);
        // The linear path DETERMINES its labels: |lambda| is read off the
        // operator's eigenvalue and the parities off the elements' actions, so
        // there is no weight fraction to decide on and no UNKNOWN to report
        // (the reportable unknown is the scored rule's).
        out.labelOutcomes.assign(static_cast<std::size_t>(n), LabelOutcome::kDetermined);

        for (int j = 0; j < n; ++j)
        {
            const Eigen::VectorXd column = coefficients.col(j);
            const double lambdaSquared = std::max(0.0, column.dot(angularSquared * column));
            const int lambda = static_cast<int>(std::lround(std::sqrt(lambdaSquared)));
            std::string label;

            if (lambda <= 5)
            {
                label = kLambdaLetters[lambda];
            } else
            {
                label = "Lz" + std::to_string(lambda);
            }

            if (lambda == 0 && sigmaVIndex >= 0)
            {
                const double parity = column.dot(
                    realization->actions[static_cast<std::size_t>(sigmaVIndex)] * column);
                label += parity > 0.0 ? "+" : "-";
            }

            if (inversionIndex >= 0)
            {
                const double parity = column.dot(
                    realization->actions[static_cast<std::size_t>(inversionIndex)] * column);
                label += parity > 0.0 ? "g" : "u";
            }

            out.labels.push_back(std::move(label));
        }

        const auto dimensionOf = [](const std::string& label) {
            return label.starts_with("S") ? 1 : 2;
        };
        CanonicalizeDegeneratePartners(fock,
                                       out.labels,
                                       occupiedCount,
                                       out.coefficients,
                                       out.canonicalized,
                                       out.straddled,
                                       dimensionOf);

        // The documented finite symmetrization subset (the continuum
        // symmetries cannot be finitely averaged).
        for (const auto& element : realization->elements)
        {
            out.symmetrizationSubset.push_back(ElementDescriptorString(element));
        }
    } else
    {
        const auto* table = qcx::symmetry::FullGroupTableFor(analysis.group);

        if (table == nullptr)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "no full-group character table for the detected group"});
        }

        if (realization->groupOrder != table->order)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "the closure does not reproduce the character table's order"});
        }

        // The runtime classes matched against the table classes by their
        // (kind, order, j, slot) keys, with the class sizes verified. The
        // construction is shared with the arbitrary-vector entry point
        // (BuildIsotypicProjections), so the two cannot drift.
        const auto projections = BuildIsotypicProjections(*realization, *table);

        if (!projections.has_value())
        {
            return std::unexpected(projections.error());
        }

        // The per-MO labels: the rule over the isotypic scores. A column
        // whose outcome is not kDetermined carries the UNKNOWN label, no irrep
        // index and its own outcome - reported, never guessed.
        out.labels.assign(static_cast<std::size_t>(n), std::string());
        out.irrepIndices.assign(static_cast<std::size_t>(n), -1);
        out.labelOutcomes.assign(static_cast<std::size_t>(n), LabelOutcome::kUnknown);

        for (int j = 0; j < n; ++j)
        {
            const VectorLabel labeled = ClassifyVectorLabel(
                coefficients.col(j), projections->projectors, projections->labels);
            out.labels[static_cast<std::size_t>(j)] = labeled.label;
            out.irrepIndices[static_cast<std::size_t>(j)] = labeled.irrepIndex;
            out.labelOutcomes[static_cast<std::size_t>(j)] = labeled.outcome;
        }

        const auto dimensionOf = [&table](const std::string& label) {
            for (int i = 0; i < table->irrepCount; ++i)
            {
                if (table->irrepLabels[i] == label)
                {
                    return table->dimensions[i];
                }
            }

            return 0;
        };
        CanonicalizeDegeneratePartners(fock,
                                       out.labels,
                                       occupiedCount,
                                       out.coefficients,
                                       out.canonicalized,
                                       out.straddled,
                                       dimensionOf);
    }

    // The symmetrized density D_sym = (1/|G|) sum_g A(g) D A(g)^T over the
    // full group (the table path) or the finite recorded subset (the linear
    // path). Energies are NOT recomputed from it.
    Eigen::MatrixXd symmetrized = Eigen::MatrixXd::Zero(n, n);

    for (const auto& action : realization->actions)
    {
        symmetrized += action * out.density * action.transpose();
    }

    out.density = symmetrized / static_cast<double>(realization->groupOrder);
    out.averagedElementCount = realization->groupOrder;
    return out;
}

qcx::Result<IsotypicProjections> BuildIsotypicProjections(
    const FullGroupRealization& realization, const qcx::symmetry::FullGroupTable& table) {
    const Eigen::Index n = realization.actions.empty()
                               ? Eigen::Index{0}
                               : static_cast<Eigen::Index>(realization.actions.front().rows());

    // The runtime classes matched against the table classes by their
    // (kind, order, j, slot) keys, with the class sizes verified. A mismatch is
    // a real detection/classification inconsistency, so it is refused with the
    // key that makes it diagnosable rather than absorbed.
    std::vector<int> tableClassOf(static_cast<std::size_t>(realization.groupOrder), -1);

    for (std::size_t c = 0; c < realization.classes.size(); ++c)
    {
        const FullGroupClass& cls = realization.classes[c];
        int tableClass = -1;

        for (int t = 0; t < table.classCount; ++t)
        {
            const qcx::symmetry::OperationClass& candidate = table.classes[t];

            if (candidate.kind == cls.kind && candidate.order == cls.order &&
                candidate.power == cls.power && candidate.axis == cls.slot)
            {
                tableClass = t;
                break;
            }
        }

        if (tableClass < 0)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "a runtime class matches no table class (kind " +
                               std::to_string(static_cast<int>(cls.kind)) + ", order " +
                               std::to_string(cls.order) + ", power " + std::to_string(cls.power) +
                               ", slot " + std::to_string(static_cast<int>(cls.slot)) + ")"});
        }

        const int classSize = static_cast<int>(std::count(realization.classOfElement.begin(),
                                                          realization.classOfElement.end(),
                                                          static_cast<int>(c)));

        if (classSize != table.classSizes[tableClass])
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "a runtime class size differs from the table"});
        }

        for (int g = 0; g < realization.groupOrder; ++g)
        {
            if (realization.classOfElement[static_cast<std::size_t>(g)] == static_cast<int>(c))
            {
                tableClassOf[static_cast<std::size_t>(g)] = tableClass;
            }
        }
    }

    // The real-form normalization: a complex-conjugate pair stored as one
    // realified row carries the 2|G| row dot; nu = round(row dot / |G|) (the
    // full_group_tables_test.cpp convention).
    std::vector<double> nu(static_cast<std::size_t>(table.irrepCount));

    for (int i = 0; i < table.irrepCount; ++i)
    {
        double rowDot = 0.0;

        for (int t = 0; t < table.classCount; ++t)
        {
            const double chi = table.characters[i][t];
            rowDot += static_cast<double>(table.classSizes[t]) * chi * chi;
        }

        nu[static_cast<std::size_t>(i)] =
            static_cast<double>(std::lround(rowDot / static_cast<double>(table.order)));
    }

    // The isotypic projectors P = (dim / (|G| nu)) sum_g chi(class g) A(g):
    // idempotent in the real form (verified against the character-table
    // identities), tr P over the isotypic = dim * multiplicity, and
    // sum_i P_i = I by character-column orthogonality - which is the identity
    // ClassifyVectorLabel's consistency gate measures.
    IsotypicProjections out;
    // irrepLabels is a FIXED std::array<std::string_view, 16>, so the count is
    // the table's, not the array's: the trailing slots are not irreps.
    out.labels.reserve(static_cast<std::size_t>(table.irrepCount));

    for (int i = 0; i < table.irrepCount; ++i)
    {
        out.labels.emplace_back(table.irrepLabels[static_cast<std::size_t>(i)]);
    }

    for (int i = 0; i < table.irrepCount; ++i)
    {
        Eigen::MatrixXd projector = Eigen::MatrixXd::Zero(n, n);

        for (int g = 0; g < realization.groupOrder; ++g)
        {
            projector += table.characters[i][tableClassOf[static_cast<std::size_t>(g)]] *
                         realization.actions[static_cast<std::size_t>(g)];
        }

        projector *=
            static_cast<double>(table.dimensions[i]) /
            (static_cast<double>(realization.groupOrder) * nu[static_cast<std::size_t>(i)]);
        out.projectors.push_back(std::move(projector));
    }

    return out;
}

qcx::scf::VectorLabel ClassifyVectorLabel(const Eigen::VectorXd& vector,
                                          const std::vector<Eigen::MatrixXd>& projectors,
                                          const std::vector<std::string>& labels) {
    qcx::scf::VectorLabel out;
    out.normSquared = vector.squaredNorm();
    out.scoreSum = 0.0;

    int best = -1;
    double bestScore = 0.0;

    for (std::size_t i = 0; i < projectors.size(); ++i)
    {
        // The score is the SQUARED norm of the isotypic component:
        // v^T P_i v = ||P_i v||^2 because P_i is an orthogonal projector.
        const double score = vector.dot(projectors[i] * vector);
        out.scoreSum += score;

        if (best < 0 || score > bestScore)
        {
            bestScore = score;
            best = static_cast<int>(i);
        }
    }

    out.bestScore = best < 0 ? 0.0 : bestScore;

    if (out.scoreSum > 0.0)
    {
        out.weightFraction = out.bestScore / out.scoreSum;
    }

    // Nothing to label: the weight fraction would be 0/0 and no irrep is named.
    if (out.normSquared <= qcx::scf::kLabelMinVectorNormSquared)
    {
        out.outcome = qcx::scf::LabelOutcome::kUnknown;
        return out;
    }

    out.projectorResidual = std::abs(out.scoreSum - out.normSquared) / out.normSquared;

    // The projectors must RESOLVE THE IDENTITY. A failure here is not an
    // unknown: it is an incomplete projector set or a wrong action.
    if (out.projectorResidual > qcx::scf::kLabelProjectorConsistencyTolerance)
    {
        out.outcome = qcx::scf::LabelOutcome::kProjectorDefect;
        return out;
    }

    // DETERMINED exactly when the best isotypic component carries essentially
    // all the weight - which covers the NEAR-TIE (two irreps at ~half each)
    // and the spread mixture with the same test.
    if (best >= 0 && best < static_cast<int>(labels.size()) &&
        out.bestScore >= (1.0 - qcx::scf::kLabelDeterminedTolerance) * out.scoreSum)
    {
        out.outcome = qcx::scf::LabelOutcome::kDetermined;
        out.label = labels[static_cast<std::size_t>(best)];
        out.irrepIndex = best;
        return out;
    }

    // No single irrep carries the weight. The label stays UNKNOWN and no index
    // is named: the argmax would be a winner decided by noise.
    out.outcome = qcx::scf::LabelOutcome::kUnknown;
    return out;
}

} // namespace qcx::scf::internal

namespace qcx::scf {

qcx::Result<std::vector<VectorLabel>> LabelVectors(
    const Eigen::MatrixXd& vectors,
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::symmetry::SymmetryAnalysis& analysis) {
    const bool linear = analysis.group == qcx::symmetry::PointGroupName::kCInfV ||
                        analysis.group == qcx::symmetry::PointGroupName::kDInfH;

    if (linear)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kUnimplemented,
            "LabelVectors: the detected full group is a linear group, which has no finite "
            "character table - its labels are read off the |lambda| operator rather than scored "
            "from isotypic projectors, so the scored rule this entry point documents does not "
            "apply there"});
    }

    const qcx::symmetry::FullGroupTable* table = qcx::symmetry::FullGroupTableFor(analysis.group);

    if (table == nullptr)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       "LabelVectors: no full-group character table for the detected group"});
    }

    const auto realization =
        qcx::scf::internal::BuildFullGroupRealization(molecule, basisSet, analysis);

    if (!realization.has_value())
    {
        return std::unexpected(realization.error());
    }

    if (realization->groupOrder != table->order)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "LabelVectors: the closure does not reproduce the character table's order"});
    }

    const auto projections = qcx::scf::internal::BuildIsotypicProjections(*realization, *table);

    if (!projections.has_value())
    {
        return std::unexpected(projections.error());
    }

    const Eigen::Index functionCount =
        projections->projectors.empty() ? Eigen::Index{0} : projections->projectors.front().rows();

    if (vectors.rows() != functionCount)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "LabelVectors: the vector matrix has " + std::to_string(vectors.rows()) +
                           " rows against the basis's " + std::to_string(functionCount)});
    }

    std::vector<VectorLabel> out;
    out.reserve(static_cast<std::size_t>(vectors.cols()));

    for (Eigen::Index j = 0; j < vectors.cols(); ++j)
    {
        out.push_back(qcx::scf::internal::ClassifyVectorLabel(
            vectors.col(j), projections->projectors, projections->labels));
    }

    return out;
}

qcx::Result<VectorLabel> LabelVector(const Eigen::VectorXd& vector,
                                     const qcx::molecule::Molecule& molecule,
                                     const qcx::basisset::BasisSet& basisSet,
                                     const qcx::symmetry::SymmetryAnalysis& analysis) {
    const Eigen::MatrixXd single = vector;
    auto labeled = LabelVectors(single, molecule, basisSet, analysis);

    if (!labeled.has_value())
    {
        return std::unexpected(labeled.error());
    }

    return labeled->front();
}

} // namespace qcx::scf
