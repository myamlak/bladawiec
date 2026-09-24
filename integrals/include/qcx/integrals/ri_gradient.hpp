#pragma once

/// \file
/// The nuclear-coordinate gradient of the resolution-of-identity Coulomb
/// term at fixed density - the companion of the RI energy path's
/// BuildCoulombOnly and BuildFock.
///
/// The fitting coefficients are a function of the geometry, so the gradient
/// of (mu nu|P) * c_P is not the gradient of (mu nu|P) alone. Written as
/// E_J = 1/2 d^T M^-1 d with d_P = sum_uv D_uv (uv|P) and M the auxiliary
/// metric, the derivative is the sum of two terms - one from the three-index
/// tensor, one from the two-index metric - and no four-index object is
/// formed at any point.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>

namespace qcx::integrals {

/// What one RI Coulomb gradient assembly counted.
///
/// The columns the energy path already keeps, so the two walks are
/// comparable: a gradient that truncates a different task set or screens by a
/// different rule differs from the energy path here first.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-integrals
struct RiCoulombGradientCounts {
    std::size_t orbitalPairs = 0; ///< Canonical orbital pairs the tensor term walked.
    std::size_t auxShells = 0; ///< Auxiliary shells the tensor term walked.
    std::size_t tensorQuartets = 0; ///< Three-index quartets differentiated.
    std::size_t metricQuartets = 0; ///< Auxiliary metric quartets differentiated.
};

/// One RI Coulomb gradient, at fixed density.
///
/// The density matrix is held fixed: `gradient` is the derivative of
/// E_J(R; D) with respect to the nuclear positions, so the density matrix's
/// own response to the displacement - the SCF's business, not this walk's -
/// is not in it. It is the contribution the gradient assembly owes the total
/// nuclear gradient, and the whole of what the fitted Coulomb term
/// contributes to it.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-integrals
struct RiCoulombGradient {
    /// dE_J/dR, 3N entries in Hartree/Bohr, atom-major: entry 3*i + d is
    /// direction d of atom i, the same ordering the nuclear coordinates use.
    Eigen::VectorXd gradient;
    /// The energy this gradient belongs to, Hartree: 1/2 d^T M^-1 d, the
    /// functional whose derivative `gradient` is.
    ///
    /// Reported on its own because it is the one number that separates the
    /// two ways a gradient can be wrong: a gradient that differentiates a
    /// different contraction differs from the energy path here, whereas one
    /// that agrees with its own energy's finite differences while the energy
    /// does not move with it differs nowhere else.
    double energy = 0.0;
    /// The counted cost of this assembly.
    RiCoulombGradientCounts counts;
};

/// Assembles the nuclear-coordinate gradient of the fitted Coulomb term.
///
/// The energy is E_J = 1/2 sum_PQ d_P (M^-1)_PQ d_Q, with d_P =
/// sum_uv D_uv (uv|P) and M_PQ = (P|Q); both d and M move with the nuclei.
/// Writing w = M^-1 d, the chain rule gives
///
///   dE_J/dR = sum_P w_P d(d_P)/dR - 1/2 sum_PQ w_P w_Q d(M_PQ)/dR
///
/// The second term is the fitting coefficients' own dependence on the
/// geometry: M^-1 differentiates to -M^-1 M' M^-1, and the w^T M' w form is
/// what survives. Dropping it leaves a gradient that is smooth, plausible
/// and wrong - and that a small symmetric fixture's finite differences can
/// pass, because the term is small exactly where the auxiliary basis is
/// nearly complete.
///
/// Both terms are evaluated through the two-electron derivative tier on the
/// same quartet representation the energy path evaluates the integrals in:
/// the auxiliary shell is paired with a phantom s function of zero exponent,
/// so (uv|P) is the quartet (uv|P|s) and (P|Q) is (s|P|s|Q). The tier
/// differentiates them as it differentiates any other quartet, which is why
/// no three-centre derivative tier exists here - and the aux index is
/// contracted before any block is accumulated, so the largest object the
/// walk holds is one quartet's block.
///
/// The density matrix is held fixed, so this is the skeleton contribution
/// only [Helgaker2000].
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Orbital basis set; every shell must satisfy l <= kMaxEngineL
/// (kUnimplemented otherwise).
/// \param auxBasisSet Auxiliary basis set, same condition. The energy path's
/// own, or the two assemblies are not comparable.
/// \param density The density matrix, nBasis x nBasis, symmetric, in the
/// basis set's function order.
/// \param options The RI engine options; metricFloorEpsilon and accuracy are
/// read from it exactly as the energy path reads them, so the metric inverse
/// this gradient differentiates is the one the energy used.
/// \returns The gradient, the energy it belongs to and the counted cost, or an
/// Error: kInvalidArgument for a density whose shape does not match the basis
/// set or a degenerate auxiliary metric, kUnimplemented for a shell beyond
/// kMaxEngineL, and the tensor and metric builders' own errors otherwise.
/// \ingroup qcx-integrals
qcx::Result<RiCoulombGradient> ComputeRiCoulombGradient(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::basisset::BasisSet& auxBasisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    const RiEngineOptions& options = {});

} // namespace qcx::integrals
