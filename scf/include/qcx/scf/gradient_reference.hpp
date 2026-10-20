#pragma once

/// \file
/// The converged SCF state a total-gradient calculation differentiates.
///
/// The response machinery needs the reference's orbitals, orbital energies
/// and spin-summed density, and the evidence that the reference converged.
/// HfResult carries all of it, but a gradient consumer that took the fields
/// directly would also take a non-converged last iterate or two objects from
/// different iterations. ExtractGradientReference is the one gate in front of
/// that use: a consumer holding a ScfGradientReference holds a validated
/// state and the numbers it was validated with.

#include "qcx/error.hpp"
#include "qcx/scf/rhf.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <string>
#include <utility>

namespace qcx::scf {

/// The converged SCF state a total-gradient calculation differentiates.
///
/// Shapes are the AO dimension \c numBasisFunctions; the occupied orbitals
/// are the first \c numOccupied columns of \c coefficients and the virtual
/// manifold the remaining \c numVirtual. The energies are the run's own, so a
/// consumer that reproduces a term from this state can state the energy it
/// belongs to.
/// \ingroup qcx-scf
struct ScfGradientReference {
    std::size_t numBasisFunctions = 0; ///< The AO dimension n.
    std::size_t numOccupied = 0; ///< Occupied spatial orbitals.
    std::size_t numVirtual = 0; ///< Virtual spatial orbitals, n - occupied.
    double totalEnergy = 0.0; ///< Electronic plus nuclear repulsion, Hartree.
    double electronicEnergy = 0.0; ///< 1/2 Tr[D (H + F[D])], Hartree.
    int iterations = 0; ///< The SCF iterations the state cost.
    double achievedEnergyDelta = 0.0; ///< The convergence gate's energy leg at the stop.
    double achievedRmsDensityChange = 0.0; ///< The convergence gate's density leg at the stop.
    Eigen::MatrixXd coefficients; ///< MO coefficients C, n x n, columns ascending by energy.
    Eigen::VectorXd orbitalEnergies; ///< Eigenvalues of F[D] in the MO basis, n.
    Eigen::MatrixXd density; ///< The spin-summed density D = 2 C_occ C_occ^T, n x n.
};

/// Extracts the gradient reference from a converged RHF result.
///
/// The extraction refuses what a gradient cannot use:
/// - a non-converged result, whose last iterate is not a stationary point of
///   the energy the gradient differentiates;
/// - a result whose returned density and occupied orbitals describe different
///   states, measured as ||D - 2 C_occ C_occ^T||_F above the rounding bound -
///   the same quantity \c HfResult::stateConsistencyResidual reports, and a
///   wrong \p numOccupied shows up here as well;
/// - a shape mismatch: \c coefficients must be square, \c density n x n and
///   \c orbitalEnergies n long, with \p numOccupied between 1 and n - 1 (a
///   response needs an occupied and a virtual manifold).
/// \param result The RHF result to extract from.
/// \param numOccupied The number of occupied spatial orbitals (electrons / 2).
/// \returns The validated reference, or kInvalidArgument naming the refusal.
/// \ingroup qcx-scf
inline qcx::Result<ScfGradientReference> ExtractGradientReference(const HfResult& result,
                                                                  std::size_t numOccupied) {
    if (!result.converged)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "scf: the gradient reference must be a converged SCF result; this one reports "
            "converged = false after " +
                std::to_string(result.iterations) +
                " iterations, so its last iterate is not the energy's stationary point"});
    }

    const Eigen::Index basis = result.coefficients.rows();

    if (basis < 2 || result.coefficients.cols() != basis)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "scf: the gradient reference's MO coefficients must be a square n x n "
                       "matrix with n >= 2; the result carries " +
                           std::to_string(result.coefficients.rows()) + " x " +
                           std::to_string(result.coefficients.cols())});
    }

    const std::size_t n = static_cast<std::size_t>(basis);

    if (result.density.rows() != basis || result.density.cols() != basis)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "scf: the gradient reference's density must be n x n with n = " + std::to_string(n) +
                "; the result carries " + std::to_string(result.density.rows()) + " x " +
                std::to_string(result.density.cols())});
    }

    if (result.orbitalEnergies.size() != basis)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "scf: the gradient reference's orbital energies must have n = " + std::to_string(n) +
                " entries; the result carries " + std::to_string(result.orbitalEnergies.size())});
    }

    if (numOccupied == 0 || numOccupied >= n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "scf: the gradient reference's occupied count must be between 1 and n - 1 "
                       "= " +
                           std::to_string(n - 1) +
                           " (a response needs both an occupied and a "
                           "virtual manifold); the request states " +
                           std::to_string(numOccupied)});
    }

    // The state the response differentiates is D = 2 C_occ C_occ^T. Rebuilding
    // it from the returned orbitals and comparing with the returned density is
    // what makes the reference one state rather than two numbers that happen
    // to sit in one struct; a wrong numOccupied lands here too.
    constexpr double kDensityConsistencyTolerance = 1e-8;
    const Eigen::MatrixXd occupied =
        result.coefficients.leftCols(static_cast<Eigen::Index>(numOccupied));
    const Eigen::MatrixXd projected = 2.0 * occupied * occupied.transpose();
    const double densityResidual = (projected - result.density).norm();

    if (densityResidual > kDensityConsistencyTolerance)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "scf: the gradient reference's density is not 2 C_occ C_occ^T for " +
                           std::to_string(numOccupied) +
                           " occupied orbitals (||D - 2 C_occ "
                           "C_occ^T||_F = " +
                           std::to_string(densityResidual) +
                           "); the returned density and orbitals describe different states"});
    }

    ScfGradientReference reference;
    reference.numBasisFunctions = n;
    reference.numOccupied = numOccupied;
    reference.numVirtual = n - numOccupied;
    reference.totalEnergy = result.totalEnergy;
    reference.electronicEnergy = result.electronicEnergy;
    reference.iterations = result.iterations;
    reference.achievedEnergyDelta = result.achievedEnergyDelta;
    reference.achievedRmsDensityChange = result.achievedRmsDensityChange;
    reference.coefficients = result.coefficients;
    reference.orbitalEnergies = result.orbitalEnergies;
    reference.density = result.density;
    return reference;
}

} // namespace qcx::scf
