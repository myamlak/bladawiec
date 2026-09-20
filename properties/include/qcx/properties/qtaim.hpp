#pragma once

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <vector>

namespace qcx::properties {

/// One (3,-1) bond critical point of the electron density (Bader QTAIM):
/// a saddle point of rho with exactly one positive Hessian eigenvalue, and
/// the two gradient paths (bond paths) from it to its bonded nuclei.
struct BondCriticalPoint {
    /// The first bonded nucleus, molecule order.
    std::size_t atomA;
    /// The second bonded nucleus, molecule order.
    std::size_t atomB;
    /// The critical point, Bohr.
    std::array<double, 3> positionBohr;
    /// The density at the critical point, e/bohr^3.
    double density;
    /// The Laplacian lambda1 + lambda2 + lambda3, e/bohr^5.
    double laplacian;
    /// The ellipticity lambda1/lambda2 - 1 >= 0 (lambda1 <= lambda2 < 0).
    double ellipticity;
    /// The Hessian eigenvalues, ascending: lambda1 <= lambda2 < 0 < lambda3.
    std::array<double, 3> eigenvalues;
    /// The bond path: the two gradient rays BCP -> nucleus concatenated,
    /// each seeded 1e-3 bohr (kPathStartEpsilon) off the BCP along the bond
    /// direction (the BCP itself opens both rays and appears twice),
    /// endpoints snapped to the nuclei, Bohr.
    std::vector<std::array<double, 3>> bondPath;
};

/// A converged critical point that is not a (3,-1) bond critical point, or
/// a bond critical point whose bond path did not terminate at a nucleus.
/// Reported, never a failure (the failed-search policy).
struct OtherCriticalPoint {
    /// The critical point, Bohr.
    std::array<double, 3> positionBohr;
    /// The Hessian rank (3).
    int rank;
    /// The Hessian eigenvalue-sign sum (-1 | +1 | +3).
    int signatureSum;
};

/// The Bader QTAIM analysis of one electron density.
struct QtaimResult {
    /// The bond critical points, each with its bond path.
    std::vector<BondCriticalPoint> bondCriticalPoints;
    /// Converged non-bond critical points and bond critical points with
    /// truncated bond paths.
    std::vector<OtherCriticalPoint> otherCriticalPoints;
    /// The starting midpoints of the Newton searches that did not converge
    /// within the iteration budget.
    std::vector<std::array<double, 3>> unconvergedSeeds;
};

/// Locates the (3,-1) bond critical points of the electron density
///     rho(r) = sum_mu,nu D_mu,nu phi_mu(r) phi_nu(r),
/// with D the spin-summed AO density D = P_alpha + P_beta (the RHF caller
/// passes HfResult::density, the UHF caller densityAlpha + densityBeta —
/// the properties module's convention) and phi the contracted AOs of the
/// grid module's AoEvaluator (Bohr coordinates, the integrals module's
/// per-primitive normalization; the evaluator's AO ordering is the
/// ordering the density is written in).  Values are in e/bohr^3 (rho),
/// e/bohr^4 (grad rho), e/bohr^5 (Laplacian, Hessian eigenvalues).
///
/// The search: seeds at the midpoints of every atom pair within
/// 1.5 * (r_cov(A) + r_cov(B)) of the element table's covalent radii
/// (Cordero2008; a zero radius means the pair never bonds), damped Newton
/// on grad rho = 0 with the Hessian as Jacobian, classification by the
/// Hessian eigenvalue signs (exactly one positive -> (3,-1)), and RK4
/// bond paths dr/ds = +/- g/|g| from each (3,-1) point to its bonded
/// nuclei.  Unconverged seeds and truncated bond paths are reported in
/// the result, never a run failure.
/// \param molecule The molecule whose density is analyzed; Bohr
/// coordinates are the module contract (do not pass Angstrom).
/// \param basisSet The molecular basis set (must cover every element).
/// \param density The spin-summed AO density D, n x n with n the AO count.
/// \returns The critical points, or an Error (kInvalidArgument when the
/// density is not n x n).
/// \ingroup qcx-properties
qcx::Result<QtaimResult> AnalyzeQtaim(const qcx::molecule::Molecule& molecule,
                                      const qcx::basisset::BasisSet& basisSet,
                                      const Eigen::MatrixXd& density);

} // namespace qcx::properties
