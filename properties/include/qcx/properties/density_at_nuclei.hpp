#pragma once

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>

namespace qcx::properties {

/// The electron density at every nucleus,
///     rho(R_A) = sum_mu,nu D_mu,nu phi_mu(R_A) phi_nu(R_A),
/// with D the spin-summed AO density D = P_alpha + P_beta (RHF:
/// HfResult::density; UHF: densityAlpha + densityBeta - the properties
/// module's convention) and phi the contracted AOs of the grid module's
/// AoEvaluator (Bohr coordinates, the integrals module's per-primitive
/// normalization).  Values are in electrons/bohr^3; this is the Bader
/// QTAIM prerequisite (density at the nuclei pins the topology search
/// seeds).
/// \param molecule The molecule whose nuclei carry the evaluation points;
/// the atoms' Bohr coordinates are used as-is (Bohr units are the module
/// contract - do not pass Angstrom).
/// \param basisSet The molecular basis set (must cover every element).
/// \param density The spin-summed AO density D, n x n with n the AO count
/// of the basis in the integrals module's ordering (the ordering the
/// AoEvaluator writes its values in).
/// \returns The per-atom densities rho(R_A) in molecule atom order, or an
/// Error (kInvalidArgument when the density is not n x n).
/// \ingroup qcx-properties
qcx::Result<Eigen::VectorXd> AnalyzeDensityAtNuclei(const qcx::molecule::Molecule& molecule,
                                                    const qcx::basisset::BasisSet& basisSet,
                                                    const Eigen::MatrixXd& density);

} // namespace qcx::properties
