#pragma once

/// \file
/// The nuclear-coordinate derivatives of the resolution-of-identity Coulomb
/// term at fixed density - the companion of the RI energy path's
/// BuildCoulombOnly and BuildFock.
///
/// The fitting coefficients are a function of the geometry, so the derivative
/// of (mu nu|P) * c_P is not the derivative of (mu nu|P) alone. Written as
/// E_J = 1/2 d^T M^-1 d with d_P = sum_uv D_uv (uv|P) and M the auxiliary
/// metric, the derivatives follow from d^(U) and M^(U) alone: no four-index
/// object is formed and no nBasis^2 x nAux tensor is built.
///
/// One entry takes the derivative order. Order 1 differentiates the reported
/// energy once; order 2 is its Hessian. The energy and every derivative read
/// one retained task set - the union of the energy path's predicate and the
/// derivative-aware one - so the reported derivatives belong to the reported
/// energy, not to a differently screened functional.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <span>

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

/// One RI Coulomb derivative assembly's settings.
/// \ingroup qcx-integrals
struct RiCoulombDerivativeOptions {
    /// The derivative order: 1 for the gradient, 2 for the Hessian. The
    /// derivative tier covers 1..2; a value outside that band is refused
    /// (kInvalidArgument).
    int order = 1;
    /// Screen the three-index tasks. On, a task is retained when it passes
    /// either the energy path's density-free predicate or the
    /// derivative-aware one; the same retained set then serves the energy and
    /// every derivative. Off, every task contributes - the exact
    /// contraction. The metric is differentiated whole in both cases.
    bool useDensityScreening = true;
    /// The accuracy preset whose Schwarz threshold the task screen reads.
    AccuracyPreset accuracy = AccuracyPreset::kNormal;
    /// The relative eigenvalue floor of the auxiliary metric's inverse - the
    /// energy path's own setting, so the coefficients this entry
    /// differentiates are the coefficients the energy used.
    double metricFloorEpsilon = 1e-10;
    /// The byte cap on one derivative block batch: a task's differentiated
    /// blocks are requested in batches of whole tuples whose packed mass
    /// stays at or below this value. One tuple heavier than the cap is its
    /// own batch - the cap is a target, not a bound on a single block - and
    /// zero is refused.
    std::size_t maxBatchBytes = 512 * 1024 * 1024;
};

/// What one RI Coulomb derivative assembly counted.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-integrals
struct RiCoulombDerivativeCounts {
    std::size_t orbitalPairs = 0; ///< Canonical orbital pairs offered to the walk.
    std::size_t auxShells = 0; ///< Auxiliary shells offered to the walk.
    std::size_t tensorTasksOffered = 0; ///< orbitalPairs * auxShells.
    /// Three-index tasks the retained set kept and whose blocks were
    /// differentiated. Equal at a base geometry and a small displacement when
    /// no task sits on the screen's threshold.
    std::size_t tensorQuartets = 0;
    /// Auxiliary metric quartets differentiated. The metric is not screened,
    /// so this is every canonical auxiliary pair.
    std::size_t metricQuartets = 0;
    int order = 1; ///< The derivative order that was assembled.
    std::size_t tuples = 0; ///< Derivative tuples assembled.
};

/// One RI Coulomb derivative, at fixed density.
///
/// `values` is tuple-major: each tuple of `order` flat coordinates
/// (3 * atom + axis) contributes one entry, in the request's tuple order.
/// Entry t is the order-`order` derivative of `energy` with respect to that
/// tuple's coordinates.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-integrals
struct RiCoulombDerivative {
    /// The requested derivatives, one per tuple, in request tuple order.
    Eigen::VectorXd values;
    /// The energy the derivatives belong to, Hartree: 1/2 d^T M^-1 d.
    double energy = 0.0;
    /// The counted cost of this assembly.
    RiCoulombDerivativeCounts counts;
};

/// Assembles requested nuclear-coordinate derivatives of the fitted Coulomb
/// term.
///
/// The energy is E_J = 1/2 sum_PQ d_P (M^-1)_PQ d_Q, with d_P =
/// sum_uv D_uv (uv|P) and M_PQ = (P|Q); both d and M move with the nuclei.
/// Writing w = M^-1 d, the derivative tier expands the coefficients'
/// dependence on the geometry through the differentiated auxiliary metric:
/// for a tuple T of coordinates, the sub-tuples U of T obey
///
///   d^(U)_P = sum_uv D_uv d(uv|P)^(U),
///   w^(U) = M^-1 (d^(U) - sum_{nonempty V subset U} M^(V) w^(U \ V)),
///   E^(T) = 1/2 sum_{U subset T} d^(U) . w^(T \ U),
///
/// with w^(empty) = w and d^(empty) = d. Order 1 evaluates the closed form
/// of the same expression, d^(a) . w - 1/2 w^T M^(a) w; order 2 walks the
/// recursion, which is its own second derivative.
///
/// Both d and M are evaluated through the two-electron derivative tier on
/// the same quartet representation the energy path evaluates the integrals
/// in: the auxiliary shell is paired with a phantom s function of zero
/// exponent, so (uv|P) is the quartet (uv|P|s) and (P|Q) is (s|P|s|Q). The
/// tier differentiates them as it differentiates any other quartet, and the
/// auxiliary index is contracted one task at a time, so no nBasis^2 x nAux
/// tensor is formed at any point. The largest objects the walk holds are the
/// nAux x nAux auxiliary metric and its eigenfactor, one length-nAux
/// derivative vector per coordinate, and one task's derivative block.
///
/// One retained set serves the energy and every derivative: a task is kept
/// when it passes the energy path's density-free predicate or the
/// derivative-aware one, and the metric is differentiated whole. The
/// reported energy is therefore the functional the reported derivatives
/// differentiate. The floored inverse is the energy path's; the floor is
/// held at the base geometry, so a task whose metric eigenvalue sits exactly
/// on the floor is outside the contract (as in the order-1 entry).
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
/// \param coordinates The flat coordinates (3 * atom + axis), `order` values
/// per tuple, in request order. Its size must be a multiple of the order and
/// every value must name a coordinate of this geometry.
/// \param options The assembly settings.
/// \returns The derivatives, the energy they belong to and the counted cost,
/// or an Error: kInvalidArgument for a density whose shape does not match the
/// basis set, an order outside 1..2, a coordinate list that does not divide
/// into whole tuples or names a coordinate of another geometry, a zero batch
/// cap, or a degenerate auxiliary metric; kUnimplemented for a shell beyond
/// kMaxEngineL or a differentiated quartet past the derivative tier's
/// Hermite table; and the metric builders' own errors otherwise.
/// \ingroup qcx-integrals
qcx::Result<RiCoulombDerivative> ComputeRiCoulombDerivative(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::basisset::BasisSet& auxBasisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    std::span<const std::size_t> coordinates,
    const RiCoulombDerivativeOptions& options = {});

/// Assembles the enumerated nuclear-coordinate derivatives of the fitted
/// Coulomb term: every flat coordinate at order 1 (all 3N, in atom-major
/// order) or every ordered coordinate pair at order 2 (all 9N^2, first
/// coordinate outermost). Each overload's contract is the explicit-coordinate
/// entry's.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Orbital basis set; every shell must satisfy l <= kMaxEngineL
/// (kUnimplemented otherwise).
/// \param auxBasisSet Auxiliary basis set, same condition.
/// \param density The density matrix, nBasis x nBasis, symmetric.
/// \param options The assembly settings; `order` selects the enumeration.
/// \returns The derivatives, the energy they belong to and the counted cost,
/// or the explicit-coordinate entry's errors.
/// \ingroup qcx-integrals
qcx::Result<RiCoulombDerivative> ComputeRiCoulombDerivative(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::basisset::BasisSet& auxBasisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    const RiCoulombDerivativeOptions& options = {});

/// Assembles the nuclear-coordinate gradient of the fitted Coulomb term.
///
/// This is the order-1 derivative entry: the call is exactly
/// ComputeRiCoulombDerivative at order 1 with the enumeration of every flat
/// coordinate, and the two produce the same entries, in the same order and
/// with the same bits. The energy is E_J = 1/2 sum_PQ d_P (M^-1)_PQ d_Q,
/// with d_P = sum_uv D_uv (uv|P) and M_PQ = (P|Q); both d and M move with
/// the nuclei. Writing w = M^-1 d, the chain rule gives
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
/// contracted before any block is accumulated, so no nBasis^2 x nAux tensor
/// exists at any point.
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
/// kMaxEngineL, and the metric builder's own errors otherwise.
/// \ingroup qcx-integrals
qcx::Result<RiCoulombGradient> ComputeRiCoulombGradient(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::basisset::BasisSet& auxBasisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    const RiEngineOptions& options = {});

} // namespace qcx::integrals
