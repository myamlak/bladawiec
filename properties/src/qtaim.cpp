// Bader QTAIM bond critical points: the (3,-1)
// bond critical points of the electron density rho = sum_mu,nu D_mu,nu
// phi_mu phi_nu (D = P_alpha + P_beta, the spin-summed density; Bohr units) with
// per-BCP position, density, Laplacian, ordered Hessian eigenvalues,
// ellipticity, and the two bond paths (the gradient paths from the BCP to
// its bonded nuclei).  The search: covalent-radius atom-pair seeds ->
// damped Newton on grad rho = 0 (Jacobian H_rho) -> Hessian-eigenvalue
// classification -> RK4 bond paths -> dedupe.  Failed searches are REPORTED
// in the result (unconvergedSeeds, otherCriticalPoints), never a failure.

#include "qcx/properties/qtaim.hpp"

#include "internal/density_field.hpp"
#include "qcx/molecule/elements.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace qcx::properties {

namespace {

// Search constants: provisional until the topology pins freeze, then
// amended in place the way the module's pin constants are amended.
constexpr double kSeedDistanceFactor = 1.5;
constexpr double kNewtonGradientTolerance = 1e-8; // e/bohr^4
// The Newton STEP gate, at the same 1e-8 as its gradient sibling rather
// than the 1e-10 it carried: the sibling is the binding leg on a smooth
// surface (a step tolerance far below the gradient tolerance cannot decide
// a stop the gradient leg has not already decided), so the tighter value
// bought nothing measured and sat four orders under the quantity it gates
// - a BCP position is not known to 1e-10 bohr. Retired under the owner
// ruling of 2026-09-18 (no convergence gate at or below 1e-10).
constexpr double kNewtonDeltaTolerance = 1e-8; // bohr
constexpr int kNewtonMaxIterations = 50;
constexpr int kNewtonMaxDamps = 6;
constexpr double kNuclearAttractorDistance = 0.25; // bohr; closer points are the nuclear maxima
constexpr double kDedupeDistance = 1e-3; // bohr
constexpr double kPathStep = 0.02; // bohr (RK4)
constexpr int kPathMaxSteps = 2500;
constexpr double kPathStartEpsilon = 1e-3; // bohr off the BCP
constexpr double kPathNucleusDistance = 0.1; // bohr; the endpoint snaps to the nucleus
constexpr double kPathGradientFloor = 1e-12; // e/bohr^4; below it the direction is undefined

double Distance(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    double squared = 0.0;

    for (std::size_t j = 0; j < 3; ++j)
    {
        const double delta = a[j] - b[j];
        squared += delta * delta;
    }

    return std::sqrt(squared);
}

// a + scale * b, componentwise.
std::array<double, 3> AddScaled(const std::array<double, 3>& a,
                                const std::array<double, 3>& b,
                                double scale) {
    std::array<double, 3> sum{};

    for (std::size_t j = 0; j < 3; ++j)
    {
        sum[j] = a[j] + scale * b[j];
    }

    return sum;
}

// The packed (xx, xy, xz, yy, yz, zz) Hessian as a 3x3 matrix.
Eigen::Matrix3d HessianMatrix(const std::array<double, 6>& hessian) {
    Eigen::Matrix3d matrix;
    matrix << hessian[0], hessian[1], hessian[2], hessian[1], hessian[3], hessian[4], hessian[2],
        hessian[4], hessian[5];
    return matrix;
}

// A converged critical point: its position, the field values there, and
// the classification from the Hessian eigenvalue signs.
struct CriticalPoint {
    std::array<double, 3> positionBohr{};
    internal::DensityField::Values values{};
    int rank{}; // 3
    int signatureSum{}; // -1 | +1 | +3 (eigenvalue-sign sum)
    std::array<double, 3> eigenvalues{}; // ascending
    Eigen::Vector3d bondDirection; // the unique positive eigenvector ((3,-1) only)
};

CriticalPoint Classify(const std::array<double, 3>& positionBohr,
                       const internal::DensityField::Values& values) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(HessianMatrix(values.hessian));
    const Eigen::Vector3d lambda = solver.eigenvalues(); // ascending

    CriticalPoint point;
    point.positionBohr = positionBohr;
    point.values = values;
    point.rank = 3;
    point.signatureSum = 0;
    point.bondDirection = Eigen::Vector3d::Zero();

    for (int j = 0; j < 3; ++j)
    {
        point.eigenvalues[j] = lambda[j];
        point.signatureSum += lambda[j] > 0.0 ? 1 : (lambda[j] < 0.0 ? -1 : 0);

        if (lambda[j] > 0.0)
        {
            // The (3,-1) bond direction: the eigenvector of the unique
            // positive eigenvalue (the bond axis).
            point.bondDirection = solver.eigenvectors().col(j);
        }
    }

    return point;
}

// Damped Newton on grad rho = 0 with the Hessian as Jacobian, from \p seed:
// solve H_rho Delta r = -g, halve the step up to kNewtonMaxDamps times when
// |g| does not decrease, stop at |g| <= kNewtonGradientTolerance or
// |Delta r| <= kNewtonDeltaTolerance, kNewtonMaxIterations iterations.
// Returns the converged critical point, or nullopt when the seed fails its
// budget (the caller reports the seed).
std::optional<CriticalPoint> ConvergeToCriticalPoint(const internal::DensityField& field,
                                                     const std::array<double, 3>& seed) {
    std::array<double, 3> point = seed;

    for (int iteration = 0; iteration < kNewtonMaxIterations; ++iteration)
    {
        const internal::DensityField::Values values = field.EvaluateAll(point);
        const Eigen::Vector3d gradient(values.gradient[0], values.gradient[1], values.gradient[2]);

        if (gradient.norm() <= kNewtonGradientTolerance)
        {
            return Classify(point, values);
        }

        const Eigen::Vector3d delta =
            HessianMatrix(values.hessian).colPivHouseholderQr().solve(-gradient);

        if (delta.norm() <= kNewtonDeltaTolerance)
        {
            return Classify(point, values);
        }

        double factor = 1.0;
        bool accepted = false;

        for (int damp = 0; damp <= kNewtonMaxDamps; ++damp)
        {
            const std::array<double, 3> candidate = {point[0] + factor * delta[0],
                                                     point[1] + factor * delta[1],
                                                     point[2] + factor * delta[2]};
            const auto candidateValues = field.EvaluateAll(candidate);
            const Eigen::Vector3d candidateGradient(candidateValues.gradient[0],
                                                    candidateValues.gradient[1],
                                                    candidateValues.gradient[2]);

            if (candidateGradient.norm() < gradient.norm())
            {
                point = candidate;
                accepted = true;
                break;
            }

            factor *= 0.5;
        }

        if (!accepted)
        {
            return std::nullopt;
        }
    }

    return std::nullopt;
}

// One RK4 step of dr/ds = +g/|g| at \p r: h * g/|g|, or nullopt when |g| is
// below kPathGradientFloor (the direction is undefined away from a nucleus).
std::optional<std::array<double, 3>> PathSubstep(const internal::DensityField& field,
                                                 const std::array<double, 3>& r) {
    const auto values = field.EvaluateAll(r);
    const Eigen::Vector3d gradient(values.gradient[0], values.gradient[1], values.gradient[2]);
    const double norm = gradient.norm();

    if (norm < kPathGradientFloor)
    {
        return std::nullopt;
    }

    return std::array<double, 3>{kPathStep * gradient.x() / norm,
                                 kPathStep * gradient.y() / norm,
                                 kPathStep * gradient.z() / norm};
}

// One ray of a bond path: dr/ds = +g/|g| with RK4 (step kPathStep) from
// \p start, until the path comes within kPathNucleusDistance of a nucleus;
// the endpoint is snapped to that nucleus.  Returns the ray's points and
// the terminating atom index, or nullopt when the ray does not terminate at
// a nucleus (the step budget or the |g| floor).
std::optional<std::pair<std::vector<std::array<double, 3>>, std::size_t>> IntegrateBondRay(
    const internal::DensityField& field,
    const std::array<double, 3>& start,
    const std::vector<std::array<double, 3>>& nuclei) {
    std::vector<std::array<double, 3>> ray;
    ray.push_back(start);
    std::array<double, 3> point = start;

    for (int step = 0; step < kPathMaxSteps; ++step)
    {
        const auto k1 = PathSubstep(field, point);

        if (!k1.has_value())
        {
            return std::nullopt;
        }

        const auto k2 = PathSubstep(field, AddScaled(point, *k1, 0.5));

        if (!k2.has_value())
        {
            return std::nullopt;
        }

        const auto k3 = PathSubstep(field, AddScaled(point, *k2, 0.5));

        if (!k3.has_value())
        {
            return std::nullopt;
        }

        const auto k4 = PathSubstep(field, AddScaled(point, *k3, 1.0));

        if (!k4.has_value())
        {
            return std::nullopt;
        }

        std::array<double, 3> next{};

        for (std::size_t j = 0; j < 3; ++j)
        {
            next[j] = point[j] + ((*k1)[j] + 2.0 * (*k2)[j] + 2.0 * (*k3)[j] + (*k4)[j]) / 6.0;
        }

        for (std::size_t atom = 0; atom < nuclei.size(); ++atom)
        {
            if (Distance(next, nuclei[atom]) <= kPathNucleusDistance)
            {
                ray.push_back(nuclei[atom]);
                return std::pair{std::move(ray), atom};
            }
        }

        ray.push_back(next);
        point = next;
    }

    return std::nullopt;
}

} // namespace

qcx::Result<QtaimResult> AnalyzeQtaim(const qcx::molecule::Molecule& molecule,
                                      const qcx::basisset::BasisSet& basisSet,
                                      const Eigen::MatrixXd& density) {
    auto field = internal::DensityField::Create(molecule, basisSet, density);

    if (!field.has_value())
    {
        return std::unexpected(field.error());
    }

    const Eigen::Index n = static_cast<Eigen::Index>(field->AOCount());

    if (density.rows() != n || density.cols() != n)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the density must be n x n with n the AO count"});
    }

    const std::size_t atomCount = molecule.AtomCount();
    const auto& coordinates = molecule.CoordinatesBohr();
    const auto& atoms = molecule.Atoms();
    std::vector<std::array<double, 3>> nuclei(atomCount);

    for (std::size_t atom = 0; atom < atomCount; ++atom)
    {
        nuclei[atom] = {coordinates(atom, 0), coordinates(atom, 1), coordinates(atom, 2)};
    }

    QtaimResult result;
    std::vector<std::array<double, 3>> convergedPositions; // the dedupe set

    // Seeds: every atom pair within 1.5 * (r_cov(A) + r_cov(B)); a zero
    // covalent radius means the element has no published radius and the
    // pair never bonds.
    for (std::size_t atomA = 0; atomA < atomCount; ++atomA)
    {
        for (std::size_t atomB = atomA + 1; atomB < atomCount; ++atomB)
        {
            const double radiusA =
                qcx::molecule::FindElement(atoms[atomA].atomicNumber)->covalentRadiusBohr;
            const double radiusB =
                qcx::molecule::FindElement(atoms[atomB].atomicNumber)->covalentRadiusBohr;

            if (radiusA <= 0.0 || radiusB <= 0.0)
            {
                continue;
            }

            if (Distance(nuclei[atomA], nuclei[atomB]) > kSeedDistanceFactor * (radiusA + radiusB))
            {
                continue;
            }

            std::array<double, 3> midpoint{};

            for (std::size_t j = 0; j < 3; ++j)
            {
                midpoint[j] = 0.5 * (nuclei[atomA][j] + nuclei[atomB][j]);
            }

            const auto converged = ConvergeToCriticalPoint(*field, midpoint);

            if (!converged.has_value())
            {
                result.unconvergedSeeds.push_back(midpoint);
                continue;
            }

            // Dedupe converged critical points closer than kDedupeDistance
            // (keep the first).
            bool duplicate = false;

            for (const auto& previous : convergedPositions)
            {
                if (Distance(previous, converged->positionBohr) <= kDedupeDistance)
                {
                    duplicate = true;
                    break;
                }
            }

            if (duplicate)
            {
                continue;
            }

            convergedPositions.push_back(converged->positionBohr);

            // Converged points within kNuclearAttractorDistance of a nucleus
            // are the density's own maxima (the nuclear attractors), not new
            // critical points.
            bool nearNucleus = false;

            for (const auto& nucleus : nuclei)
            {
                if (Distance(nucleus, converged->positionBohr) <= kNuclearAttractorDistance)
                {
                    nearNucleus = true;
                    break;
                }
            }

            if (nearNucleus)
            {
                continue;
            }

            // Classification: exactly one positive eigenvalue -> (3,-1) bond
            // critical point; anything else is reported as it stands.
            if (converged->signatureSum != -1)
            {
                result.otherCriticalPoints.push_back(OtherCriticalPoint{
                    converged->positionBohr, converged->rank, converged->signatureSum});
                continue;
            }

            // The bond paths: the two gradient rays from the BCP, each
            // starting kPathStartEpsilon off it along the bond direction
            // (+/- the unique positive eigenvector); a BCP is a bond only
            // when BOTH rays terminate at a nucleus.
            std::array<double, 3> offset{};

            for (std::size_t j = 0; j < 3; ++j)
            {
                offset[j] =
                    kPathStartEpsilon * converged->bondDirection[static_cast<Eigen::Index>(j)];
            }

            const auto rayA =
                IntegrateBondRay(*field, AddScaled(converged->positionBohr, offset, 1.0), nuclei);
            const auto rayB =
                IntegrateBondRay(*field, AddScaled(converged->positionBohr, offset, -1.0), nuclei);

            if (!rayA.has_value() || !rayB.has_value() || rayA->second == rayB->second)
            {
                // A truncated path (or both rays on the same atom) is not a
                // bond: report the critical point, not a failure.
                result.otherCriticalPoints.push_back(OtherCriticalPoint{
                    converged->positionBohr, converged->rank, converged->signatureSum});
                continue;
            }

            const std::size_t firstAtom = std::min(rayA->second, rayB->second);
            const std::size_t secondAtom = std::max(rayA->second, rayB->second);

            // The bond path: the BCP, ray A (BCP side -> nucleus), the BCP
            // again, ray B.  The BCP starts each ray.
            std::vector<std::array<double, 3>> bondPath;
            bondPath.push_back(converged->positionBohr);
            bondPath.insert(bondPath.end(), rayA->first.begin(), rayA->first.end());
            bondPath.push_back(converged->positionBohr);
            bondPath.insert(bondPath.end(), rayB->first.begin(), rayB->first.end());

            // Ellipticity: lambda1/lambda2 - 1; the division is safe because
            // the (3,-1) filter above (signatureSum == -1, ascending order)
            // guarantees exactly two negative eigenvalues, so lambda2 < 0.
            result.bondCriticalPoints.push_back(BondCriticalPoint{
                firstAtom,
                secondAtom,
                converged->positionBohr,
                converged->values.rho,
                converged->eigenvalues[0] + converged->eigenvalues[1] + converged->eigenvalues[2],
                converged->eigenvalues[0] / converged->eigenvalues[1] - 1.0,
                converged->eigenvalues,
                std::move(bondPath)});
        }
    }

    return result;
}

} // namespace qcx::properties
