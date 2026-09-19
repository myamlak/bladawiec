#include "qcx/symmetry/salc.hpp"

#include "qcx/backend/cpu_backend.hpp"
#include "qcx/molecule/mass_properties.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <numeric>
#include <optional>
#include <vector>

namespace qcx::symmetry {
namespace {

// Thresholds and tolerances (named per the no-magic-numbers rule).
constexpr double kMomentLinearityThreshold = 1e-6; ///< Linear-top cutoff for mass properties.
constexpr double kAxisAlignmentTolerance = 1e-6; ///< |dot| tolerance for axis matching.
constexpr double kAxisPerpendicularTolerance = 1e-6; ///< |dot| below this counts as perpendicular.
constexpr double kMultiplicityTolerance = 1e-8; ///< Below this, a multiplicity counts as zero.
constexpr double kClosureTolerance = 1e-8; ///< Matrix norm below which ops are the same.

struct MatchEntry {
    Eigen::Vector3d position; ///< CoM-centered.
    int atomicNumber; ///< Z.
    std::size_t index; ///< Canonical atom index (molecule order).
};

// Strict-weak x/y/z ordering for the greedy matcher's sorted array.
bool LexLess(const MatchEntry& a, const MatchEntry& b) {
    if (a.position.x() != b.position.x())
    {
        return a.position.x() < b.position.x();
    }

    if (a.position.y() != b.position.y())
    {
        return a.position.y() < b.position.y();
    }

    return a.position.z() < b.position.z();
}

/// Greedy distance-matched permutation of the atom set under transform.
qcx::Result<std::vector<std::size_t>> PermutationFor(const std::vector<MatchEntry>& sorted,
                                                     const Eigen::Matrix3d& transform,
                                                     double tolerance) {
    const double tolSq = tolerance * tolerance;
    std::vector<std::size_t> sigma(sorted.size());
    std::vector<bool> matched(sorted.size(), false);

    for (std::size_t i = 0; i < sorted.size(); ++i)
    {
        const Eigen::Vector3d q = transform * sorted[i].position;
        const double xMin = q.x() - tolerance;
        const double xMax = q.x() + tolerance;
        auto it =
            std::lower_bound(sorted.begin(), sorted.end(), xMin, [](const MatchEntry& e, double x) {
                return e.position.x() < x;
            });
        bool found = false;

        for (; it != sorted.end() && it->position.x() <= xMax; ++it)
        {
            if (it->atomicNumber != sorted[i].atomicNumber)
            {
                continue;
            }

            const auto j = static_cast<std::size_t>(it - sorted.begin());

            if (!matched[j] && (q - it->position).squaredNorm() <= tolSq)
            {
                matched[j] = true;
                // The consumers index sigma by canonical atom: the source is
                // sorted[i].index, the image's canonical index is it->index.
                sigma[sorted[i].index] = it->index;
                found = true;
                break;
            }
        }

        if (!found)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "an operation of the computational group does not permute the atoms"});
        }
    }

    return sigma;
}

/// Deterministic tie-break: smaller direction first.
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

/// A concrete operation of the computational group: its matrix and the atom
/// permutation it induces.
struct ConcreteOp {
    Eigen::Matrix3d matrix; ///< The operation as a 3x3 matrix.
    std::vector<std::size_t> permutation; ///< atom i -> image.
};

/// Union-find orbits of equivalent atoms under the group's permutations.
std::vector<std::vector<std::size_t>> ComputeOrbits(const std::vector<ConcreteOp>& ops,
                                                    std::size_t n) {
    std::vector<std::size_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    const auto findRoot = [&parent](std::size_t i) {
        while (parent[i] != i)
        {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }

        return i;
    };

    for (const auto& op : ops)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            parent[findRoot(i)] = findRoot(op.permutation[i]);
        }
    }

    std::vector<std::vector<std::size_t>> orbits(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        orbits[findRoot(i)].push_back(i);
    }

    std::vector<std::vector<std::size_t>> nonEmptyOrbits;

    for (const auto& orbit : orbits)
    {
        if (!orbit.empty())
        {
            nonEmptyOrbits.push_back(orbit);
        }
    }

    return nonEmptyOrbits;
}

/// Chi-weighted SALC projection: one orthonormal row per (orbit, irrep)
/// pair whose multiplicity is non-zero.
///
/// The computational groups are Abelian, so a transitive permutation
/// representation decomposes into distinct 1-dimensional irreps
/// (multiplicity-free): every orbit x irrep pair with m = 1 yields exactly
/// one SALC, distinct orbits have disjoint supports, and no Gram-Schmidt is
/// needed. The result's orthonormality is asserted by the tests, not
/// recomputed here.
struct ProjectionResult {
    std::vector<std::string_view> irrepLabels; ///< Irrep of each row.
    Eigen::MatrixXd coefficients; ///< N x N orthonormal rows (SALCs x atoms).
};

ProjectionResult ProjectRows(const CharacterTable& table,
                             const std::vector<ConcreteOp>& ops,
                             const std::vector<std::vector<std::size_t>>& orbits,
                             std::size_t n) {
    // Irrep multiplicities per orbit: m = (1/|G|) sum_g chiOrbit(g) chi(g).
    struct SalcTask {
        std::size_t seed; ///< First atom of the orbit.
        int irrep; ///< Table row.
        std::size_t row; ///< Row of the coefficient matrix.
    };

    std::vector<SalcTask> tasks;
    std::vector<std::string_view> labels;
    std::size_t rowCount = 0;

    for (const auto& orbit : orbits)
    {
        const std::size_t seed = *std::min_element(orbit.begin(), orbit.end());

        for (int irrep = 0; irrep < table.order; ++irrep)
        {
            double multiplicity = 0.0;

            for (std::size_t g = 0; g < ops.size(); ++g)
            {
                std::size_t fixedPoints = 0;

                for (const std::size_t atom : orbit)
                {
                    if (ops[g].permutation[atom] == atom)
                    {
                        ++fixedPoints;
                    }
                }

                multiplicity += static_cast<double>(fixedPoints) *
                                static_cast<double>(table.characters[irrep][g]);
            }

            multiplicity /= static_cast<double>(table.order);

            if (std::abs(multiplicity) > kMultiplicityTolerance)
            {
                labels.push_back(table.irrepLabels[static_cast<std::size_t>(irrep)]);
                tasks.push_back(SalcTask{seed, irrep, rowCount});
                ++rowCount;
            }
        }
    }

    // Projection, parallel over the (orbit, irrep) tasks with disjoint row
    // writes: each coefficient is the chi-weighted image of the seed atom.
    Eigen::MatrixXd coefficients =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(rowCount), static_cast<Eigen::Index>(n));
    qcx::backend::Backend<qcx::backend::CpuTag>{}.ParallelForDynamic(
        tasks.size(), [&](std::size_t taskIndex) {
            const SalcTask& task = tasks[taskIndex];
            Eigen::VectorXd row = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n));

            for (std::size_t g = 0; g < ops.size(); ++g)
            {
                const std::size_t image = ops[g].permutation[task.seed];
                row(static_cast<Eigen::Index>(image)) +=
                    static_cast<double>(table.characters[task.irrep][g]);
            }

            row.normalize();
            coefficients.row(static_cast<Eigen::Index>(task.row)) = row;
        });

    return ProjectionResult{labels, coefficients};
}

} // namespace

qcx::Result<SalcSet> GenerateSalcs(const qcx::molecule::Molecule& molecule,
                                   const SymmetryAnalysis& analysis,
                                   double toleranceBohr) {
    const CharacterTable& table = CharacterTableFor(analysis.computational);
    const std::size_t n = molecule.AtomCount();

    const auto mass = qcx::molecule::ComputeMassProperties(molecule, kMomentLinearityThreshold);
    std::vector<MatchEntry> entries;
    entries.reserve(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& coords = molecule.CoordinatesBohr();
        const Eigen::Vector3d position =
            Eigen::Vector3d(coords(i, 0), coords(i, 1), coords(i, 2)) - mass.centerOfMass;
        entries.push_back(MatchEntry{position, molecule.Atoms()[i].atomicNumber, i});
    }

    std::vector<MatchEntry> sorted = entries;
    std::sort(sorted.begin(), sorted.end(), LexLess);

    // Collect the verified rotation axes and mirror normals. The C2 axes
    // come from the recorded order-2 rotations AND the C2 powers of
    // even-order ones (C4^2, C6^3, C8^4 share the axis): degenerate-inertia
    // molecules like benzene record the six-fold axis but never a C2 about
    // it, and the D2/D2h tables need that axis.
    std::vector<Eigen::Vector3d> c2Axes;
    std::vector<Eigen::Vector3d> mirrorNormals;
    bool hasInversion = false;

    for (const auto& element : analysis.elements)
    {
        if (element.kind == OperationKind::kRotation && element.order % 2 == 0)
        {
            c2Axes.push_back(element.axis);
        } else if (element.kind == OperationKind::kSigma)
        {
            mirrorNormals.push_back(element.axis);
        } else if (element.kind == OperationKind::kInversion)
        { hasInversion = true; }
    }

    // Which axis letters the table's columns reference.
    std::vector<char> rotationLetters;
    std::vector<char> mirrorLetters;

    for (int i = 0; i < table.order; ++i)
    {
        const OperationTemplate& op = table.operations[i];

        if (op.kind == OperationKind::kRotation)
        {
            rotationLetters.push_back(op.axis);
        } else if (op.kind == OperationKind::kSigma)
        { mirrorLetters.push_back(op.axis); }
    }

    // Best |dot| with the given moment column among `directions`, optionally
    // restricted to directions perpendicular to `ref`; deterministic
    // tie-break by canonicalized direction comparison.
    const auto bestAligned = [&mass](const std::vector<Eigen::Vector3d>& directions,
                                     int column,
                                     const Eigen::Vector3d* ref) -> std::optional<Eigen::Vector3d> {
        double bestDot = -1.0;
        Eigen::Vector3d best = Eigen::Vector3d::Zero();

        for (const auto& direction : directions)
        {
            if (ref != nullptr && std::abs(direction.dot(*ref)) >= kAxisPerpendicularTolerance)
            {
                continue;
            }

            const double dot = std::abs(direction.dot(mass.principalAxes.col(column)));

            if (dot > bestDot + kAxisAlignmentTolerance ||
                (std::abs(dot - bestDot) <= kAxisAlignmentTolerance && bestDot >= 0.0 &&
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

    // Axis labeling: 'z' hugs the highest principal moment; for the
    // multi-C2 groups (D2/D2h) 'y' is the remaining C2 axis best aligned
    // with the middle moment subject to perpendicularity with 'z', and 'x'
    // is z x y - which must itself be a detected C2 axis. Independent
    // per-letter moment alignment breaks on degenerate-inertia molecules
    // (ties can pick the same axis, or non-perpendicular axes, and the ops
    // then do not realize the table's group); the perpendicularity
    // constraints plus the closure check below make the realization exact
    // or loudly failing.
    std::array<std::optional<Eigen::Vector3d>, 3> fixedAxes; // z, y, x.
    std::array<std::optional<Eigen::Vector3d>, 3> fixedNormals; // Mirror normals, z, y, x.

    if (!rotationLetters.empty())
    {
        const auto zBest = bestAligned(c2Axes, 0, nullptr);

        if (!zBest.has_value())
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "group requires a C2 axis, not detected"});
        }

        // Guarded local copies: the optional derefs happen once, immediately
        // after their check (clang-tidy's bugprone-unchecked-optional-access
        // cannot see guards through std::array subscripting).
        const Eigen::Vector3d zAxis = *zBest;
        fixedAxes[0] = zAxis;

        if (rotationLetters.size() >= 2)
        {
            const auto yBest = bestAligned(c2Axes, 1, &zAxis);

            if (!yBest.has_value())
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "group requires a perpendicular C2 axis, not detected"});
            }

            const Eigen::Vector3d yAxis = *yBest;
            fixedAxes[1] = yAxis;
            const Eigen::Vector3d xAxis = zAxis.cross(yAxis).normalized();
            const bool xIsC2 = std::any_of(
                c2Axes.begin(), c2Axes.end(), [&xAxis](const Eigen::Vector3d& candidate) {
                    return std::abs(candidate.dot(xAxis)) > 1.0 - kAxisAlignmentTolerance;
                });

            if (!xIsC2)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "group requires three perpendicular C2 axes, not detected"});
            }

            fixedAxes[2] = xAxis;
        }

        // Mirror normals: a mirror column whose letter matches a fixed axis
        // uses that axis (D2h, C2h). For molecules with inversion but no
        // detected mirrors - C60/Ih has none - sigma = i x C2 realizes the
        // column, and the permutation check below validates the composition.
        // C2v (exactly one rotation and two mirror columns) fixes 'y' from
        // the detected normals perpendicular to 'z' and completes
        // 'x' = z x y.
        if (rotationLetters.size() == 1 && mirrorLetters.size() == 2)
        {
            const auto yNormalBest = bestAligned(mirrorNormals, 1, &zAxis);

            if (!yNormalBest.has_value())
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                                  "group requires a mirror, not detected"});
            }

            const Eigen::Vector3d normalY = *yNormalBest;
            fixedNormals[1] = normalY;
            fixedNormals[2] = zAxis.cross(normalY).normalized();
        }
    } else
    {
        // Cs: no rotations; the single mirror column takes its normal from
        // the detected mirrors.
        for (const char letter : mirrorLetters)
        {
            const int column = 'z' - letter;
            const auto normalBest = bestAligned(mirrorNormals, column, nullptr);

            if (!normalBest.has_value())
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                                  "group requires a mirror, not detected"});
            }

            fixedNormals[column] = *normalBest;
        }
    }

    // Mirrors tied to fixed axes (D2h, C2h): the normal IS the axis.
    for (const char letter : mirrorLetters)
    {
        const int column = 'z' - letter;

        if (fixedNormals[column].has_value())
        {
            continue;
        }

        const auto& axisOpt = fixedAxes[column];

        if (!axisOpt.has_value())
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "group requires a mirror, not detected"});
        }

        const Eigen::Vector3d normal = *axisOpt;
        fixedNormals[column] = normal;
    }

    // Concrete operation for each table column, plus its permutation.
    std::vector<ConcreteOp> ops;
    ops.reserve(static_cast<std::size_t>(table.order));

    for (int opIndex = 0; opIndex < table.order; ++opIndex)
    {
        const OperationTemplate& op = table.operations[opIndex];
        Eigen::Matrix3d matrix = Eigen::Matrix3d::Identity();

        if (op.kind == OperationKind::kInversion)
        {
            if (!hasInversion)
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                                  "group requires inversion, not detected"});
            }

            matrix = -Eigen::Matrix3d::Identity();
        } else if (op.kind == OperationKind::kRotation)
        {
            const int column = 'z' - op.axis;
            const auto& axisOpt = fixedAxes[column];

            if (!axisOpt.has_value())
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                                  "group requires a C2 axis, not detected"});
            }

            const Eigen::Vector3d axis = *axisOpt;
            matrix = Eigen::AngleAxisd(2.0 * std::numbers::pi / static_cast<double>(op.order), axis)
                         .toRotationMatrix();
        } else if (op.kind == OperationKind::kSigma)
        {
            const int column = 'z' - op.axis;
            const auto& normalOpt = fixedNormals[column];

            if (!normalOpt.has_value())
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                                  "group requires a mirror, not detected"});
            }

            const Eigen::Vector3d normal = *normalOpt;
            matrix = Eigen::Matrix3d::Identity() - 2.0 * normal * normal.transpose();
        }

        const auto sigma = PermutationFor(sorted, matrix, toleranceBohr);

        if (!sigma.has_value())
        {
            return std::unexpected(sigma.error());
        }

        ops.push_back(ConcreteOp{matrix, *sigma});
    }

    // Closure validation: the concrete matrices must realize the table's
    // group - every pairwise product lands on one of the matrices, all
    // distinct. A degenerate-inertia axis pick that misses this would
    // silently corrupt the multiplicities, so it fails loudly instead.
    for (std::size_t i = 0; i < ops.size(); ++i)
    {
        for (std::size_t j = 0; j < ops.size(); ++j)
        {
            const Eigen::Matrix3d product = ops[i].matrix * ops[j].matrix;
            bool closed = false;

            for (const auto& op : ops)
            {
                if ((product - op.matrix).norm() < kClosureTolerance)
                {
                    closed = true;
                    break;
                }
            }

            if (!closed)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "the concrete operations do not close under composition"});
            }
        }
    }

    for (std::size_t i = 0; i < ops.size(); ++i)
    {
        for (std::size_t j = i + 1; j < ops.size(); ++j)
        {
            if ((ops[i].matrix - ops[j].matrix).norm() < kClosureTolerance)
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                                  "duplicate operation in the concrete group"});
            }
        }
    }

    const auto orbits = ComputeOrbits(ops, n);
    const auto projection = ProjectRows(table, ops, orbits, n);

    return SalcSet{analysis.computational, projection.irrepLabels, projection.coefficients};
}

} // namespace qcx::symmetry
