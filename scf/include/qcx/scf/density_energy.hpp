#pragma once

/// \file
/// The Fock matrix and total energy evaluated at an arbitrary density
/// (the ETS-NOCV embedding seam).
///
/// The RHF/UHF SCF loops only produce Fock/energy at their own iterate
/// densities. The ETS-NOCV energy decomposition instead needs F[P] and
/// E[P] at densities that are not SCF fixed points - the frozen fragment
/// combination and the transition-state 1/2(P_frag + P_mol) - so the
/// analysis drives this evaluator rather than a second SCF run.

#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Dense>

namespace qcx::scf {

/// The closed-shell Fock matrix and RHF energy at an arbitrary density.
/// \ingroup qcx-scf
struct RhfDensityEnergy {
    Eigen::MatrixXd fock; ///< F = H + J[D] - K[D]/2 at the given density
                          ///< (the spin-summed convention of rhf.hpp's
                          ///< FockBuilderFn note).
    double electronicEnergy = 0.0; ///< 1/2 Tr[D (H + F)] (Hartree).
    double totalEnergy = 0.0; ///< electronic plus nuclear repulsion (Hartree).
};

/// The per-spin Fock matrices and UHF energy at arbitrary densities.
/// \ingroup qcx-scf
struct UhfDensityEnergy {
    Eigen::MatrixXd fockAlpha; ///< F_a = H + J(D_a + D_b) - K(D_a).
    Eigen::MatrixXd fockBeta; ///< F_b = H + J(D_a + D_b) - K(D_b).
    double electronicEnergy = 0.0; ///< 1/2 Tr[D_a (H + F_a) + D_b (H + F_b)] (Hartree).
    double totalEnergy = 0.0; ///< electronic plus nuclear repulsion (Hartree).
};

/// Evaluates the closed-shell Fock matrix and total energy at an arbitrary
/// density - not necessarily an SCF fixed point. The supermatrix J/K
/// contract the density with the dense repulsion tensor exactly as the
/// RHF loop of rhf.cpp does, so the evaluator reproduces the loop's
/// Fock/energy at a converged density to the SCF convergence residue
/// (~1e-11; the ETS energy-identity pin depends on that).
/// \param molecule Validated molecule; the nuclear repulsion comes from
/// molecule::NuclearRepulsionEnergy. The electron count is NOT read - the
/// energy of a density is defined regardless of its trace.
/// \param coreHamiltonian Core Hamiltonian H = T + V (n x n).
/// \param eri Dense (uv|ws) repulsion tensor with shape {n, n, n, n},
/// host-canonical.
/// \param density The evaluation density, n x n, in the spin-summed
/// closed-shell convention D = 2 C_occ C_occ^T.
/// \param includeExchange True (default) evaluates the full Roothaan Fock
/// F = H + J - K/2; false drops the exchange (F = H + J) - the classical
/// Coulomb-only functional whose cross-fragment pieces form the ETS
/// electrostatic term of the NOCV decomposition.
/// \returns The Fock matrix and energies, or an Error (kInvalidArgument
/// for a shape mismatch).
/// \ingroup qcx-scf
qcx::Result<RhfDensityEnergy> EvaluateRhfDensityEnergy(
    const qcx::molecule::Molecule& molecule,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& density,
    bool includeExchange = true);

/// Evaluates the UHF per-spin Fock matrices and total energy at arbitrary
/// densities, mirroring the uhf.cpp conventions: the Coulomb field of the
/// combined density and each spin's own exchange
/// (F_sigma = H + J(D_a + D_b) - K(D_sigma)).
/// \param molecule Validated molecule; the nuclear repulsion comes from
/// molecule::NuclearRepulsionEnergy. The electron count is NOT read.
/// \param coreHamiltonian Core Hamiltonian H = T + V (n x n).
/// \param eri Dense (uv|ws) repulsion tensor with shape {n, n, n, n},
/// host-canonical.
/// \param densityAlpha The alpha density D_a = C_a C_a^T (n x n).
/// \param densityBeta The beta density D_b = C_b C_b^T (n x n).
/// \param includeExchange True (default) evaluates the full UHF Fock
/// F_sigma = H + J(D_a + D_b) - K(D_sigma); false drops the exchange -
/// the classical Coulomb-only functional of the ETS electrostatic term.
/// \returns The per-spin Fock matrices and energies, or an Error
/// (kInvalidArgument for a shape mismatch).
/// \ingroup qcx-scf
qcx::Result<UhfDensityEnergy> EvaluateUhfDensityEnergy(
    const qcx::molecule::Molecule& molecule,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    bool includeExchange = true);

} // namespace qcx::scf
