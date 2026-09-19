#pragma once

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <vector>

namespace qcx::properties {

/// The ESP point-set scheme.
enum class EspFitScheme {
    kChelpg, ///< CHELPG cubic lattice [BrenemanWiberg1990].
    kMerzKollman ///< Merz-Kollman vdW shells [SinghKollman1984,
                 ///< BeslerMerzKollman1990].
};

/// Tuning knobs for the ESP point sets; the defaults are the published
/// values of the schemes.
struct EspFitOptions {
    /// CHELPG: cubic lattice spacing, 0.3 A = 0.5669 bohr
    /// (BrenemanWiberg1990).
    double chelpgSpacingBohr = 0.5669178374;
    /// CHELPG: points are kept at least this far from the nearest nucleus,
    /// 2.0 A = 3.7795 bohr.
    double chelpgInnerRadiusBohr = 3.7794522484;
    /// CHELPG: points are kept at most this far from the nearest nucleus,
    /// 3.0 A = 5.6692 bohr.
    double chelpgOuterRadiusBohr = 5.6691783746;
    /// MK: angular nodes per shell, a Lebedev count from
    /// qcx::grid::AngularGrid::kAvailableSizes (302 = degree 17; the
    /// published MK samplings are all comparable once the point set is
    /// dense).
    std::size_t mkLebedevPoints = 302;
};

/// One ESP charge fit.
struct EspFitResult {
    /// The fitted per-atom charges, molecule atom order; sum = molecular
    /// charge by construction.
    Eigen::VectorXd charges;
    /// RMS deviation of the fitted potential from the quantum potential
    /// over the fit points (hartree).
    double rmsError = 0.0;
    /// Number of fit points actually used.
    std::size_t pointCount = 0;
};

/// Fits atom-centered point charges to the molecular electrostatic
/// potential (ESP charges): with
///     V_QM(p) = V_elec(p) + V_nuc(p),
///     V_elec(p) = -sum_uv D_uv <u| 1/|r - p| |v>  (the density
///     contraction, integrals::BuildElectronPotentialAtPoints),
///     V_nuc(p) = sum_A Z_A / |p - R_A|           (analytic here),
/// the charges minimize sum_p (sum_A q_A / |p - R_A| - V_QM(p))^2 subject
/// to the total-charge constraint sum_A q_A = Q_mol = sum_A Z_A - N_e,
/// N_e = tr(D S).  The constrained problem is solved through the augmented
/// normal equations with the Lagrange multiplier row/column (Eigen LDLT).
///
/// Point sets:
///  - CHELPG: a cubic lattice at 0.3 A spacing spanning the molecule plus
///    3.0 A, keeping the points whose distance to the nearest nucleus lies
///    between 2.0 and 3.0 A (BrenemanWiberg1990).
///  - MK: concentric shells around every atom at 1.4, 1.6, 1.8, and 2.0
///    times the element's van der Waals radius (molecule/elements.hpp),
///    each shell sampled on a Lebedev grid (SinghKollman1984,
///    BeslerMerzKollman1990).
/// \param molecule The molecule whose atoms carry the charges.
/// \param basisSet The molecular basis set (must cover every element).
/// \param density The spin-summed molecular density D = D_alpha + D_beta,
/// as converged by the SCF (HfResult::density).
/// \param scheme The point-set scheme.
/// \param options Scheme tuning knobs.
/// \returns The fitted charges, or an Error (kInvalidArgument for a
/// density shape mismatch or a non-Lebedev MK count, kUnimplemented for a
/// shell beyond the parser cap, kInvalidArgument when an atom has no
/// basis entry or no van der Waals radius).
/// \ingroup qcx-properties
qcx::Result<EspFitResult> AnalyzeEspCharges(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet,
                                            const Eigen::MatrixXd& density,
                                            EspFitScheme scheme,
                                            const EspFitOptions& options = {});

} // namespace qcx::properties
