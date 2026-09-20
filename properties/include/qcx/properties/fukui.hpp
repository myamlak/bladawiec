#pragma once

#include "qcx/error.hpp"
#include "qcx/properties/populations.hpp"

#include <Eigen/Core>
#include <vector>

namespace qcx::properties {

/// Condensed Fukui indices ([Parr1984], [YangMortier1986]): the per-atom
/// response of the electron population to adding or removing an electron,
/// condensed to atoms through the Mulliken population difference
///     f_A^+ = q_A(N + 1) - q_A(N)     (nucleophilic attack site)
///     f_A^- = q_A(N) - q_A(N - 1)     (electrophilic attack site)
///     f_A^0 = (f_A^+ + f_A^-) / 2     (radical attack site)
/// with q_A(N) the Mulliken gross population of atom A at electron count
/// N.  Each vector sums to one (adding or removing an electron changes
/// the total population by exactly one).
struct FukuiIndices {
    Eigen::VectorXd nucleophilic; ///< f_A^+ per atom; sums to 1.
    Eigen::VectorXd electrophilic; ///< f_A^- per atom; sums to 1.
    Eigen::VectorXd radical; ///< f_A^0 per atom; sums to 1.
};

/// Condensed Fukui indices from three per-spin density matrices.
///
/// The densities are the converged SCF results for the neutral system,
/// its anion (N + 1 electrons) and its cation (N - 1 electrons), at the
/// same geometry and basis.  Each density pair is Mulliken-analyzed
/// ([Mulliken1955]) and the atomic populations differenced per the
/// formulas above.
/// \param overlap The n x n overlap matrix S.
/// \param neutralAlpha, neutralBeta The neutral per-spin densities.
/// \param anionAlpha, anionBeta The N + 1 per-spin densities.
/// \param cationAlpha, cationBeta The N - 1 per-spin densities.
/// \param aoRanges Per-atom function ranges partitioning [0, n).
/// \returns The Fukui indices, or an Error (kInvalidArgument when the
/// matrices disagree in shape or the ranges exceed n - the underlying
/// AnalyzeMulliken validation).
/// \ingroup qcx-properties
qcx::Result<FukuiIndices> AnalyzeFukui(const Eigen::MatrixXd& overlap,
                                       const Eigen::MatrixXd& neutralAlpha,
                                       const Eigen::MatrixXd& neutralBeta,
                                       const Eigen::MatrixXd& anionAlpha,
                                       const Eigen::MatrixXd& anionBeta,
                                       const Eigen::MatrixXd& cationAlpha,
                                       const Eigen::MatrixXd& cationBeta,
                                       const std::vector<AoRange>& aoRanges);

} // namespace qcx::properties
