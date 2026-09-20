#include "qcx/symmetry/detection.hpp"

#include "qcx/backend/cpu_backend.hpp"
#include "qcx/molecule/mass_properties.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

namespace qcx::symmetry {
namespace {

// |dot| above this counts as parallel (both vectors normalized).
constexpr double kAxisParallelTolerance = 1e-6;
// Base rotation-angle matching tolerance for classifying a fit as Cn
// (rad); the per-fit derived tolerance (AngleToleranceFor) may widen it
// for near-axis geometries.
constexpr double kAngleTolerance = 1e-4;
// Matrix norm below which a fit counts as the identity (Classify).
constexpr double kFitIdentityTolerance = 1e-8;
// Column norm above which rotation+I yields an axis direction.
constexpr double kAxisColumnTolerance = 1e-9;
// Skew-part norm below which the axis recovery needs the rotation+I fallback.
constexpr double kAxisSkewTolerance = 1e-12;
// Matrix-norm difference below which two fits are the same transformation.
constexpr double kDedupTolerance = 1e-8;
// Upper bound on enumerated atom permutations (a deterministic DFS prefix;
// every corpus group is far below this - Ih, the largest, has 120).
constexpr std::size_t kMaxPermutations = 256;

struct MatchEntry {
    Eigen::Vector3d position; ///< CoM-centered.
    int atomicNumber; ///< Z.
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

/// Flips the sign so each direction pair {-d, d} has one canonical form.
Eigen::Vector3d Canonicalize(Eigen::Vector3d direction) {
    constexpr double eps = 1e-14;

    if (direction.x() < -eps || (std::abs(direction.x()) <= eps && direction.y() < -eps) ||
        (std::abs(direction.x()) <= eps && std::abs(direction.y()) <= eps && direction.z() < -eps))
    {
        direction = -direction;
    }

    return direction;
}

bool IsParallel(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return std::abs(a.dot(b)) > 1.0 - kAxisParallelTolerance;
}

/// Enumerates the automorphisms of the labeled atom set (same element, all
/// pair distances preserved within tolerance): a deterministic iterative
/// depth-first search. Each automorphism is a symmetry permutation.
std::vector<std::vector<std::size_t>> EnumeratePermutations(const std::vector<MatchEntry>& entries,
                                                            double tolerance) {
    const std::size_t n = entries.size();
    // Squared distances, lower triangle.
    std::vector<double> distances(n * n);

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < i; ++j)
        {
            distances[i * n + j] = (entries[i].position - entries[j].position).squaredNorm();
        }
    }

    const double distanceTolerance = 2.0 * tolerance; // Triangle-inequality bound.

    std::vector<std::vector<std::size_t>> permutations;
    std::vector<std::size_t> sigma(n);
    std::vector<bool> used(n, false);
    std::size_t depth = 0;
    std::size_t nextTarget = 0;
    std::vector<std::size_t> firstTarget(n, 0); // Last tried target per depth.

    while (true)
    {
        if (depth == n)
        {
            permutations.push_back(sigma);

            if (permutations.size() >= kMaxPermutations)
            {
                return permutations;
            }

            --depth;
            used[sigma[depth]] = false;
            nextTarget = firstTarget[depth] + 1;
            continue;
        }

        const auto& atom = entries[depth];
        bool assigned = false;

        for (std::size_t j = nextTarget; j < n; ++j)
        {
            if (used[j] || entries[j].atomicNumber != atom.atomicNumber)
            {
                continue;
            }

            bool consistent = true;

            for (std::size_t k = 0; k < depth; ++k)
            {
                const double before = distances[depth * n + k];
                const std::size_t a = sigma[k] > j ? sigma[k] : j;
                const std::size_t b = sigma[k] > j ? j : sigma[k];

                if (std::abs(std::sqrt(before) - std::sqrt(distances[a * n + b])) >
                    distanceTolerance)
                {
                    consistent = false;
                    break;
                }
            }

            if (!consistent)
            {
                continue;
            }

            sigma[depth] = j;
            used[j] = true;
            firstTarget[depth] = j;
            ++depth;
            nextTarget = 0;
            assigned = true;
            break;
        }

        if (!assigned)
        {
            if (depth == 0)
            {
                break;
            }

            --depth;
            used[sigma[depth]] = false;
            nextTarget = firstTarget[depth] + 1;
        }
    }

    return permutations;
}

/// Both least-squares orthogonal fits mapping positions onto their
/// permutation images: the proper rotation and the improper (reflecting)
/// one [Kabsch1976]. Kabsch's classic formula returns only the det-corrected
/// (proper) matrix, which would silently drop every mirror and inversion;
/// the improper fit is the det-flip variant of the same SVD, and the
/// rank-deficient completion below is the nullspace treatment of
/// [Kabsch1978].
struct OrthogonalFits {
    Eigen::Matrix3d proper; ///< det = +1 fit.
    Eigen::Matrix3d improper; ///< det = -1 fit.
};

OrthogonalFits FitOrthogonal(const std::vector<MatchEntry>& entries,
                             const std::vector<std::size_t>& sigma) {
    Eigen::Matrix3d h = Eigen::Matrix3d::Zero();

    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        h += entries[sigma[i]].position * entries[i].position.transpose();
    }

    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(h, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto& u = svd.matrixU();
    const auto& v = svd.matrixV();
    // Rank-deficient H (planar/collinear sets) has arbitrary column pairings
    // at sigma = 0, which would poison U*V^T with a garbage rank-1 term; the
    // kernel completes with the identity instead, the natural extension of
    // the least-squares rotation. ALL kernel columns complete: a rank-1 H
    // (a near-collinear set below the linear cutoff) has two zero singular
    // values, and completing only one leaves the other arbitrary pairing in
    // place - safe (the garbage fit then fails the set-invariance check and
    // the molecule falls to C1) but silently missing the symmetry, so the
    // full kernel gets the identity completion.
    constexpr double kSingularZeroTolerance = 1e-12;
    const double eps = kSingularZeroTolerance * std::max(1.0, svd.singularValues()(0));
    std::vector<int> kernelColumns;
    int smallestNonzero = 0;

    for (int k = 0; k < 3; ++k)
    {
        if (svd.singularValues()(k) <= eps)
        {
            kernelColumns.push_back(k);
        } else
        {
            smallestNonzero = k;
        }
    }

    Eigen::Matrix3d base = u * v.transpose();

    for (const int k : kernelColumns)
    {
        base += u.col(k) * (u.col(k) - v.col(k)).transpose();
    }

    // The opposite-determinant partner flips the kernel columns when they
    // exist (zero cost) and the smallest singular direction otherwise.
    Eigen::Matrix3d partner = base;

    if (!kernelColumns.empty())
    {
        for (const int k : kernelColumns)
        {
            partner -= 2.0 * u.col(k) * u.col(k).transpose();
        }
    } else
    {
        partner -= 2.0 * u.col(smallestNonzero) * v.col(smallestNonzero).transpose();
    }

    if (base.determinant() > 0.0)
    {
        return OrthogonalFits{base, partner};
    }

    return OrthogonalFits{partner, base};
}

struct OpInfo {
    OperationKind kind = OperationKind::kIdentity;
    int order = 1; ///< n of Cn/Sn; 1 for identity, inversion, mirrors.
    Eigen::Vector3d axis = Eigen::Vector3d::Zero(); ///< Rotation axis or mirror normal.
};

/// The rotation axis of a proper rotation matrix: the skew part for
/// theta != pi, any column of rotation + I for theta = pi (the skew part
/// vanishes there).
Eigen::Vector3d FindRotationAxis(const Eigen::Matrix3d& rotation) {
    Eigen::Vector3d axis(rotation(2, 1) - rotation(1, 2),
                         rotation(0, 2) - rotation(2, 0),
                         rotation(1, 0) - rotation(0, 1));

    if (axis.norm() < kAxisSkewTolerance)
    {
        // theta = pi: any column of rotation + I lies along the axis.
        const Eigen::Matrix3d plusIdentity = rotation + Eigen::Matrix3d::Identity();

        for (int col = 0; col < 3; ++col)
        {
            if (plusIdentity.col(col).norm() > kAxisColumnTolerance)
            {
                axis = plusIdentity.col(col);
                break;
            }
        }
    }

    return axis.normalized();
}

/// The per-fit angle tolerance, derived from the position tolerance and
/// the geometry: a position error of up to positionTolerance on an
/// atom at radius r from the fit's axis perturbs the recovered rotation
/// angle by up to ~positionTolerance / r, so near-axis atoms need a wider
/// angle tolerance than the fixed kAngleTolerance - which silently
/// under-detected them to a lower group. Atoms within 2 * positionTolerance
/// of the axis are rotationally indeterminate about it (their image under
/// any rotation lands within tolerance of themselves) and carry no angular
/// information; the derived term covers the rest.
double AngleToleranceFor(const Eigen::Matrix3d& transform,
                         const std::vector<MatchEntry>& entries,
                         double positionTolerance) {
    const bool proper = transform.determinant() > 0.0;
    const Eigen::Vector3d axis = FindRotationAxis(proper ? transform : -transform);
    double minRadius = std::numeric_limits<double>::infinity();

    for (const auto& entry : entries)
    {
        const double radius = (entry.position - axis * entry.position.dot(axis)).norm();

        if (radius > 2.0 * positionTolerance)
        {
            minRadius = std::min(minRadius, radius);
        }
    }

    if (!std::isfinite(minRadius))
    {
        return kAngleTolerance; // no atom constrains the angle
    }

    return std::max(kAngleTolerance, 2.0 * positionTolerance / minRadius);
}

/// Classifies an orthogonal fit: identity, Cn, inversion, mirror, or Sn.
/// A fit that is not a small-n rotation (numerically impossible for a finite
/// atom set, but guarded) comes back as the identity and is ignored.
/// \param angleTolerance The per-fit matching tolerance (AngleToleranceFor);
/// the caller derives it from the position tolerance and the geometry.
OpInfo Classify(const Eigen::Matrix3d& transform, double angleTolerance) {
    if (transform.isApprox(Eigen::Matrix3d::Identity(), kFitIdentityTolerance))
    {
        return OpInfo{OperationKind::kIdentity, 1, Eigen::Vector3d::Zero()};
    }

    const bool proper = transform.determinant() > 0.0;
    const Eigen::Matrix3d rotation = proper ? transform : -transform;
    const double cosThetaPrime = std::clamp(0.5 * (rotation.trace() - 1.0), -1.0, 1.0);
    const double thetaPrime = std::acos(cosThetaPrime);

    if (thetaPrime < angleTolerance)
    {
        // The proper part is the identity: E, or -E = inversion.
        return OpInfo{proper ? OperationKind::kIdentity : OperationKind::kInversion,
                      1,
                      Eigen::Vector3d::Zero()};
    }

    // An improper fit is sigma_h * R(theta): its proper part is -T =
    // R(theta + pi), so the rotation angle is recovered as the acute angle
    // equivalent of (thetaPrime - pi).
    double theta = thetaPrime;

    if (!proper)
    {
        const double phi = std::fmod(thetaPrime - std::numbers::pi + 2.0 * std::numbers::pi,
                                     2.0 * std::numbers::pi);
        theta = phi <= std::numbers::pi ? phi : 2.0 * std::numbers::pi - phi;
    }

    if (theta < angleTolerance)
    {
        // Pure mirror: S1 = sigma with the normal along the C2 axis of -T.
        return OpInfo{OperationKind::kSigma, 1, Canonicalize(FindRotationAxis(rotation))};
    }

    int n = 0;

    for (int candidate = 2; candidate <= 12; ++candidate)
    {
        if (std::abs(theta - 2.0 * std::numbers::pi / static_cast<double>(candidate)) <
            angleTolerance)
        {
            n = candidate;
            break;
        }
    }

    if (n == 0)
    {
        return OpInfo{OperationKind::kIdentity, 1, Eigen::Vector3d::Zero()};
    }

    const Eigen::Vector3d axis = Canonicalize(FindRotationAxis(rotation));

    if (!proper)
    {
        return OpInfo{OperationKind::kImproper, n, axis};
    }

    return OpInfo{OperationKind::kRotation, n, axis};
}

struct AxisFindings {
    Eigen::Vector3d direction; ///< Normalized axis (or mirror normal).
    int maxRotationOrder = 0; ///< Maximal verified Cn (0 = none).
    bool s4 = false; ///< Verified S4 about the direction.
    bool s6 = false; ///< Verified S6.
    bool s8 = false; ///< Verified S8.
    bool mirror = false; ///< Verified mirror with this normal.
};

// Cn name for the principal rotation order n (2..8).
PointGroupName CnGroup(int n) {
    switch (n)
    {
    case 2:
        return PointGroupName::kC2;
    case 3:
        return PointGroupName::kC3;
    case 4:
        return PointGroupName::kC4;
    case 5:
        return PointGroupName::kC5;
    case 6:
        return PointGroupName::kC6;
    case 7:
        return PointGroupName::kC7;
    case 8:
        return PointGroupName::kC8;
    default:
        return PointGroupName::kC1;
    }
}

// Cnv name for the principal rotation order n (2..8).
PointGroupName CnvGroup(int n) {
    switch (n)
    {
    case 2:
        return PointGroupName::kC2v;
    case 3:
        return PointGroupName::kC3v;
    case 4:
        return PointGroupName::kC4v;
    case 5:
        return PointGroupName::kC5v;
    case 6:
        return PointGroupName::kC6v;
    case 7:
        return PointGroupName::kC7v;
    case 8:
        return PointGroupName::kC8v;
    default:
        return PointGroupName::kC1;
    }
}

// Cnh name for the principal rotation order n (2..8).
PointGroupName CnhGroup(int n) {
    switch (n)
    {
    case 2:
        return PointGroupName::kC2h;
    case 3:
        return PointGroupName::kC3h;
    case 4:
        return PointGroupName::kC4h;
    case 5:
        return PointGroupName::kC5h;
    case 6:
        return PointGroupName::kC6h;
    case 7:
        return PointGroupName::kC7h;
    case 8:
        return PointGroupName::kC8h;
    default:
        return PointGroupName::kC1;
    }
}

// Dn name for the principal rotation order n (2..8).
PointGroupName DnGroup(int n) {
    switch (n)
    {
    case 2:
        return PointGroupName::kD2;
    case 3:
        return PointGroupName::kD3;
    case 4:
        return PointGroupName::kD4;
    case 5:
        return PointGroupName::kD5;
    case 6:
        return PointGroupName::kD6;
    case 7:
        return PointGroupName::kD7;
    case 8:
        return PointGroupName::kD8;
    default:
        return PointGroupName::kC1;
    }
}

// Dnh name for the principal rotation order n (2..8).
PointGroupName DnhGroup(int n) {
    switch (n)
    {
    case 2:
        return PointGroupName::kD2h;
    case 3:
        return PointGroupName::kD3h;
    case 4:
        return PointGroupName::kD4h;
    case 5:
        return PointGroupName::kD5h;
    case 6:
        return PointGroupName::kD6h;
    case 7:
        return PointGroupName::kD7h;
    case 8:
        return PointGroupName::kD8h;
    default:
        return PointGroupName::kC1;
    }
}

// Dnd name for the principal rotation order n (2..8).
PointGroupName DndGroup(int n) {
    switch (n)
    {
    case 2:
        return PointGroupName::kD2d;
    case 3:
        return PointGroupName::kD3d;
    case 4:
        return PointGroupName::kD4d;
    case 5:
        return PointGroupName::kD5d;
    case 6:
        return PointGroupName::kD6d;
    case 7:
        return PointGroupName::kD7d;
    case 8:
        return PointGroupName::kD8d;
    default:
        return PointGroupName::kC1;
    }
}

/// Point-group name from the verified per-axis findings ([Cotton1990]
/// classification tree, psi4-style; the documented reduction to the
/// computational group is in LargestAbelianSubgroup).
PointGroupName ClassifyFromFindings(const std::vector<AxisFindings>& findings, bool hasInversion) {
    std::size_t principalIndex = 0;
    std::vector<std::size_t> c2AxisIndices;
    std::size_t higherOrderCount = 0;

    for (std::size_t k = 0; k < findings.size(); ++k)
    {
        if (findings[k].maxRotationOrder >= 3)
        {
            ++higherOrderCount;
            principalIndex = k;
        } else if (findings[k].maxRotationOrder == 2)
        { c2AxisIndices.push_back(k); }
    }

    std::vector<Eigen::Vector3d> mirrorNormals;

    for (const auto& finding : findings)
    {
        if (finding.mirror)
        {
            mirrorNormals.push_back(finding.direction);
        }
    }

    const bool anyMirror = !mirrorNormals.empty();
    const bool anyS4 =
        std::any_of(findings.begin(), findings.end(), [](const AxisFindings& f) { return f.s4; });
    const bool anyS8 =
        std::any_of(findings.begin(), findings.end(), [](const AxisFindings& f) { return f.s8; });

    if (higherOrderCount >= 2)
    {
        // Cubic branch: two distinct high-order axes never coexist outside
        // T/Td/Th (C3), O/Oh (C4), I/Ih (C5).
        const bool anyC5 = std::any_of(findings.begin(), findings.end(), [](const AxisFindings& f) {
            return f.maxRotationOrder == 5;
        });
        const bool anyC4 = std::any_of(findings.begin(), findings.end(), [](const AxisFindings& f) {
            return f.maxRotationOrder == 4;
        });

        if (anyC5)
        {
            return (hasInversion || anyMirror) ? PointGroupName::kIh : PointGroupName::kI;
        }

        if (anyC4)
        {
            return (hasInversion || anyMirror) ? PointGroupName::kOh : PointGroupName::kO;
        }

        // Th = T x Ci: the inversion plus the C3 axes forces the sigmaH
        // mirrors, so inversion (with or without detected mirrors) means
        // Th; mirrors without inversion mean Td; neither means T.
        return hasInversion ? PointGroupName::kTh
                            : (anyMirror ? PointGroupName::kTd : PointGroupName::kT);
    }

    if (higherOrderCount == 1)
    {
        const AxisFindings& principal = findings[principalIndex];
        const int nOrder = principal.maxRotationOrder;
        const bool sigmaH = std::any_of(mirrorNormals.begin(),
                                        mirrorNormals.end(),
                                        [&principal](const Eigen::Vector3d& normal) {
                                            return IsParallel(normal, principal.direction);
                                        });
        const bool hasS2n = principal.s4 || principal.s6 || principal.s8;
        const std::size_t perpC2Count = static_cast<std::size_t>(
            std::count_if(c2AxisIndices.begin(),
                          c2AxisIndices.end(),
                          [&findings, &principal](const std::size_t k) {
                              return !IsParallel(findings[k].direction, principal.direction);
                          }));
        const bool sigmaV = std::any_of(mirrorNormals.begin(),
                                        mirrorNormals.end(),
                                        [&principal](const Eigen::Vector3d& normal) {
                                            return !IsParallel(normal, principal.direction);
                                        });

        if (sigmaH && perpC2Count >= 2)
        {
            return DnhGroup(nOrder);
        }

        if (sigmaH)
        {
            return CnhGroup(nOrder);
        }

        if (hasS2n)
        {
            if (perpC2Count >= 2)
            {
                return DndGroup(nOrder);
            }

            if (principal.s4)
            {
                return PointGroupName::kS4;
            }

            if (principal.s6)
            {
                return PointGroupName::kS6;
            }

            return PointGroupName::kS8;
        }

        if (perpC2Count >= 2)
        {
            return DnGroup(nOrder);
        }

        if (sigmaV)
        {
            return CnvGroup(nOrder);
        }

        return CnGroup(nOrder);
    }

    if (anyS4 || anyS8)
    {
        // Before the C2 count: D2d has three C2 axes (S4/S8 axes imply
        // their C2 power; a lone C2 + S2n without the perpendicular set
        // is the pure S4/S8 group).
        return c2AxisIndices.size() >= 2 ? PointGroupName::kD2d
                                         : (anyS4 ? PointGroupName::kS4 : PointGroupName::kS8);
    }

    if (c2AxisIndices.size() >= 3)
    {
        return (hasInversion || anyMirror) ? PointGroupName::kD2h : PointGroupName::kD2;
    }

    if (c2AxisIndices.size() == 1)
    {
        const Eigen::Vector3d& axis = findings[c2AxisIndices[0]].direction;
        const bool sigmaH = std::any_of(
            mirrorNormals.begin(), mirrorNormals.end(), [&axis](const Eigen::Vector3d& normal) {
                return IsParallel(normal, axis);
            });
        const bool sigmaV = std::any_of(
            mirrorNormals.begin(), mirrorNormals.end(), [&axis](const Eigen::Vector3d& normal) {
                return !IsParallel(normal, axis);
            });

        if (sigmaH || hasInversion)
        {
            return PointGroupName::kC2h;
        }

        if (sigmaV)
        {
            return PointGroupName::kC2v;
        }

        return PointGroupName::kC2;
    }

    // Exactly two perpendicular C2 axes cannot occur (their product is the
    // third C2); defensively this falls to the reflection groups below.
    if (hasInversion)
    {
        return PointGroupName::kCi;
    }

    if (anyMirror)
    {
        return PointGroupName::kCs;
    }

    return PointGroupName::kC1;
}

/// Appends the verified operations to \p elements in deterministic (finding)
/// order: inversion, rotations, mirrors, improper rotations.
void RecordElements(const std::vector<AxisFindings>& findings,
                    bool hasInversion,
                    std::vector<SymmetryElement>& elements) {
    if (hasInversion)
    {
        elements.push_back(SymmetryElement{OperationKind::kInversion, 1, Eigen::Vector3d::Zero()});
    }

    for (const auto& finding : findings)
    {
        if (finding.maxRotationOrder >= 2)
        {
            elements.push_back(SymmetryElement{
                OperationKind::kRotation, finding.maxRotationOrder, finding.direction});
        }
    }

    for (const auto& finding : findings)
    {
        if (finding.mirror)
        {
            elements.push_back(SymmetryElement{OperationKind::kSigma, 1, finding.direction});
        }
    }

    for (const auto& finding : findings)
    {
        if (finding.s4)
        {
            elements.push_back(SymmetryElement{OperationKind::kImproper, 4, finding.direction});
        }

        if (finding.s6)
        {
            elements.push_back(SymmetryElement{OperationKind::kImproper, 6, finding.direction});
        }

        if (finding.s8)
        {
            elements.push_back(SymmetryElement{OperationKind::kImproper, 8, finding.direction});
        }
    }
}

} // namespace

SymmetryAnalysis DetectPointGroup(const qcx::molecule::Molecule& molecule,
                                  const DetectOptions& options) {
    const std::size_t n = molecule.AtomCount();
    const double tol = options.positionToleranceBohr;
    SymmetryAnalysis analysis;
    analysis.group = PointGroupName::kC1;
    analysis.elements.push_back(
        SymmetryElement{OperationKind::kIdentity, 1, Eigen::Vector3d::Zero()});

    if (n == 1)
    {
        analysis.computational = PointGroup::kC1;
        return analysis;
    }

    const auto mass = qcx::molecule::ComputeMassProperties(molecule, options.linearRatio);
    std::vector<MatchEntry> entries;
    entries.reserve(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& atom = molecule.Atoms()[i];
        const auto& coords = molecule.CoordinatesBohr();
        const Eigen::Vector3d position =
            Eigen::Vector3d(coords(i, 0), coords(i, 1), coords(i, 2)) - mass.centerOfMass;
        entries.push_back(MatchEntry{position, atom.atomicNumber});
    }

    if (mass.isLinear)
    {
        // Collinear atom sets make the Kabsch fit rank-deficient (rotations
        // about the axis are all minimizers), so linear molecules take a
        // direct inversion check: -p_i must land on a same-element atom.
        bool inversion = true;

        for (const auto& entry : entries)
        {
            bool found = false;

            for (const auto& target : entries)
            {
                if (target.atomicNumber == entry.atomicNumber &&
                    (target.position + entry.position).norm() <= tol)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                inversion = false;
                break;
            }
        }

        analysis.group = inversion ? PointGroupName::kDInfH : PointGroupName::kCInfV;

        // The computational groups need concrete C2 and mirror elements: C2
        // about the molecular axis (permutes nothing on a collinear set but
        // realizes the table's rotation column), the vertical mirrors, and
        // for DinfH the perpendicular C2 axes plus sigmaH - all genuine
        // elements of CinfV/DinfH. The molecular axis is the smallest
        // principal moment of a linear top.
        const Eigen::Vector3d axis = mass.principalAxes.col(2).normalized();
        Eigen::Vector3d perp1 = axis.cross(Eigen::Vector3d::UnitX());

        if (perp1.norm() < kAxisSkewTolerance)
        {
            perp1 = axis.cross(Eigen::Vector3d::UnitY());
        }

        perp1.normalize();
        const Eigen::Vector3d perp2 = axis.cross(perp1).normalized();
        analysis.elements.push_back(SymmetryElement{OperationKind::kRotation, 2, axis});
        analysis.elements.push_back(SymmetryElement{OperationKind::kSigma, 1, perp1});
        analysis.elements.push_back(SymmetryElement{OperationKind::kSigma, 1, perp2});

        if (inversion)
        {
            analysis.elements.push_back(
                SymmetryElement{OperationKind::kInversion, 1, Eigen::Vector3d::Zero()});
            analysis.elements.push_back(SymmetryElement{OperationKind::kRotation, 2, perp1});
            analysis.elements.push_back(SymmetryElement{OperationKind::kRotation, 2, perp2});
            analysis.elements.push_back(SymmetryElement{OperationKind::kSigma, 1, axis});
        }

        analysis.computational = LargestAbelianSubgroup(analysis.group);
        return analysis;
    }

    // Enumerate the symmetry permutations of the atom set (element labels +
    // distance matrix preserved), fit each to an orthogonal transformation
    // via Kabsch, and deduplicate by effect. This finds every symmetry
    // element from the structure itself - axis-guessing from pair directions
    // misses e.g. D2d's diagonal C2' axes.
    const auto permutations = EnumeratePermutations(entries, tol);
    std::vector<Eigen::Matrix3d> transforms;
    // A fit is a symmetry operation iff it maps the atom set onto itself
    // (same elements, within tolerance) - NOT iff it realizes its generating
    // permutation, whose direction (Cn vs Cn^-1, say) is an enumeration
    // artifact of the DFS while the fit's is fixed by the SVD. Greedy
    // distance matching against the sorted atoms answers the set question.
    std::vector<MatchEntry> sortedEntries = entries;
    std::sort(sortedEntries.begin(), sortedEntries.end(), LexLess);
    const auto setInvariant = [&sortedEntries, tol](const Eigen::Matrix3d& fit) {
        std::vector<bool> matched(sortedEntries.size(), false);

        for (const auto& entry : sortedEntries)
        {
            const Eigen::Vector3d q = fit * entry.position;
            const double xMin = q.x() - tol;
            const double xMax = q.x() + tol;
            auto it =
                std::lower_bound(sortedEntries.begin(),
                                 sortedEntries.end(),
                                 xMin,
                                 [](const MatchEntry& e, double x) { return e.position.x() < x; });
            bool found = false;

            for (; it != sortedEntries.end() && it->position.x() <= xMax; ++it)
            {
                if (it->atomicNumber != entry.atomicNumber)
                {
                    continue;
                }

                const auto j = static_cast<std::size_t>(it - sortedEntries.begin());

                if (!matched[j] && (q - it->position).norm() <= tol)
                {
                    matched[j] = true;
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                return false;
            }
        }

        return true;
    };
    // Accepted fits are deduplicated by the transformation matrix itself -
    // effect-based dedup would merge distinct group elements that act
    // identically on a planar set (e.g. water's sigma(yz) and its C2),
    // which SALC generation must keep apart.
    const auto considerFit = [&transforms](const Eigen::Matrix3d& fit) {
        for (const auto& existing : transforms)
        {
            if ((fit - existing).norm() < kDedupTolerance)
            {
                return;
            }
        }

        transforms.push_back(fit);
    };

    for (const auto& sigma : permutations)
    {
        const OrthogonalFits fits = FitOrthogonal(entries, sigma);

        if (setInvariant(fits.proper))
        {
            considerFit(fits.proper);
        }

        if (setInvariant(fits.improper))
        {
            considerFit(fits.improper);
        }
    }

    // Classify each distinct transformation (parallel, disjoint writes);
    // the angle tolerance derives per fit from the position tolerance and
    // the geometry (AngleToleranceFor).
    std::vector<OpInfo> ops(transforms.size());
    qcx::backend::Backend<qcx::backend::CpuTag>{}.ParallelForDynamic(
        transforms.size(), [&](std::size_t k) {
            ops[k] = Classify(transforms[k], AngleToleranceFor(transforms[k], entries, tol));
        });

    // Fold the operations into per-axis findings (merge parallel axes).
    bool hasInversion = false;
    std::vector<AxisFindings> findings;
    const auto findingFor = [&findings](const Eigen::Vector3d& axis) -> std::optional<std::size_t> {
        for (std::size_t k = 0; k < findings.size(); ++k)
        {
            if (IsParallel(findings[k].direction, axis))
            {
                return k;
            }
        }

        return std::nullopt;
    };

    for (const auto& op : ops)
    {
        if (op.kind == OperationKind::kIdentity)
        {
            continue;
        }

        if (op.kind == OperationKind::kInversion)
        {
            hasInversion = true;
            continue;
        }

        const auto existing = findingFor(op.axis);

        if (!existing.has_value())
        {
            findings.push_back(AxisFindings{});
            findings.back().direction = op.axis;
        }

        AxisFindings& slot = findings[existing.value_or(findings.size() - 1)];

        switch (op.kind)
        {
        case OperationKind::kRotation:
            slot.maxRotationOrder = std::max(slot.maxRotationOrder, op.order);
            break;
        case OperationKind::kSigma:
            slot.mirror = true;
            break;
        case OperationKind::kImproper:
            slot.s4 = slot.s4 || op.order == 4;
            slot.s6 = slot.s6 || op.order == 6;
            slot.s8 = slot.s8 || op.order == 8;
            break;
        case OperationKind::kIdentity:
        case OperationKind::kInversion:
            break;
        }
    }

    // Classification tree ([Cotton1990], psi4-style; the documented
    // reduction to the computational group is in LargestAbelianSubgroup).
    analysis.group = ClassifyFromFindings(findings, hasInversion);

    // Record the verified operations in deterministic (finding) order.
    RecordElements(findings, hasInversion, analysis.elements);

    analysis.computational = LargestAbelianSubgroup(analysis.group);
    return analysis;
}

} // namespace qcx::symmetry
