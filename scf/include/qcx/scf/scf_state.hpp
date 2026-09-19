#pragma once

/// \file
/// The serializable SCF restart state: the checkpoint/restart
/// contract between the scf module and the storage module's
/// SaveScfCheckpoint/LoadScfCheckpoint.

#include "qcx/scf/diis.hpp"

#include <Eigen/Dense>

namespace qcx::scf {

/// The serializable state of one SCF run needed to continue it (the
/// checkpoint/restart): the iterate density, the DIIS
/// histories, and the convergence gate's previous-iteration energies.
/// RHF fills density/diis; UHF fills the per-spin fields. This is the
/// INPUT-side state; the result fields (HfResult::density & co) are
/// the OUTPUT-side converged state and hold the same density variable.
/// \ingroup qcx-scf
struct ScfRestartState {
    Eigen::MatrixXd
        density; ///< RHF: spin-summed D = 2 C_occ C_occ^T (the FockBuilderFn convention).
    Eigen::MatrixXd densityAlpha; ///< UHF alpha-spin density.
    Eigen::MatrixXd densityBeta; ///< UHF beta-spin density.
    DiisState diis; ///< RHF DIIS history.
    DiisState diisAlpha; ///< UHF alpha-spin DIIS history.
    DiisState diisBeta; ///< UHF beta-spin DIIS history.
    double previousTotalEnergy = 0.0; ///< Gate state: last iterate's energy, and the energy OF
                                      ///< the density beside it - the exit refreshes the pair
                                      ///< together, so a resumed run's first energy leg
                                      ///< compares like with like.
    double previousElectronicEnergy = 0.0; ///< Gate state.
};

} // namespace qcx::scf
