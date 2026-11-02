#pragma once

/// \file
/// The restricted open-shell assembly of the Roothaan effective Fock, in the
/// ORTHONORMAL solver space. Kept beside rohf.cpp rather than in rohf.cpp's
/// anonymous namespace so the block-table acceptance test can pin the
/// assembly itself, apart from the loop and apart from the integrals: the
/// block identities below are the whole content of the method, and a test
/// that reached them only through a converged run could not say which of the
/// two was wrong.

#include <Eigen/Dense>

namespace qcx::scf::internal {

/// The Roothaan effective Fock from the two solver-space spin Focks and the
/// closed/open projectors:
///
///   Delta = F_alpha - F_beta,   P_v = I - P_c - P_o
///   F_eff = (F_alpha + F_beta)/2
///         + 1/2 [ P_o Delta P_o - P_c Delta P_o - P_o Delta P_c
///                 + P_o Delta P_v + P_v Delta P_o ].
///
/// Its blocks, in the closed/open/virtual basis, are the Roothaan ones:
///
/// | block | value | stationarity condition |
/// | --- | --- | --- |
/// | closed-closed | (F_alpha + F_beta)/2 | - |
/// | open-open | F_alpha | - |
/// | closed-open | F_beta | F_beta on the closed-open block = 0 |
/// | closed-virtual | (F_alpha + F_beta)/2 | its vanishing |
/// | open-virtual | F_alpha | F_alpha on the open-virtual block = 0 |
/// | virtual-virtual | (F_alpha + F_beta)/2 | irrelevant |
///
/// The closed-open block is F_beta exactly because the Delta terms there sum
/// to -Delta/2 over the average; the open-virtual block is F_alpha for the
/// same reason with the opposite sign. Those two identities are what makes
/// the three vanishing blocks the Roothaan stationarity conditions, and the
/// block-table test pins them one by one.
/// \param fockAlpha The alpha spin Fock, X^T F_alpha X.
/// \param fockBeta The beta spin Fock, X^T F_beta X.
/// \param closedProjector P_c of the current closed space.
/// \param openProjector P_o of the current open space; zero columns' worth
/// (the zero matrix) in the closed-shell limit.
/// \returns F_eff, symmetric when the inputs are.
/// \ingroup qcx-scf
inline Eigen::MatrixXd AssembleRoothaanFock(const Eigen::MatrixXd& fockAlpha,
                                            const Eigen::MatrixXd& fockBeta,
                                            const Eigen::MatrixXd& closedProjector,
                                            const Eigen::MatrixXd& openProjector) {
    const Eigen::MatrixXd delta = fockAlpha - fockBeta;
    const Eigen::MatrixXd virtualProjector =
        Eigen::MatrixXd::Identity(fockAlpha.rows(), fockAlpha.cols()) - closedProjector -
        openProjector;
    const Eigen::MatrixXd openDelta = openProjector * delta;

    Eigen::MatrixXd effective = 0.5 * (fockAlpha + fockBeta);
    effective += 0.5 * (openDelta * openProjector - closedProjector * delta * openProjector -
                        openDelta * closedProjector + openDelta * virtualProjector +
                        virtualProjector * delta * openProjector);
    return effective;
}

} // namespace qcx::scf::internal
