#pragma once

/// \file
/// The total analytic nuclear gradient at a converged closed-shell reference:
/// the electronic terms, the nuclear repulsion and the dispersion correction,
/// kept apart in the result so a consumer can read each one.
///
/// The electronic gradient is the density's skeleton terms plus the orbital
/// response's own contribution, `dE_elec/dR = dTr(D h)/dR + dE_J/dR -
/// (1/2) dE_K/dR + Tr(F D(1))`. Every derivative integral comes from the
/// integrals module's derivative tier and the response is solved by the
/// response module's state, so this module adds the term algebra and nothing
/// else. The one thing it forms itself is the nuclear-repulsion derivative:
/// `molecule` exposes the repulsion ENERGY and no derivative of it, and the
/// pair sum differentiates in closed form.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/dispersion/dispersion.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/response/response_solver.hpp"
#include "qcx/response/response_state.hpp"
#include "qcx/scf/gradient_reference.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace qcx::properties {

/// The total gradient's terms, each a 3N vector in Hartree/Bohr,
/// atom-major: entry `3 * a + d` differentiates with respect to the `d`-th
/// Cartesian component of atom `a`, the ordering the nuclear coordinates use.
///
/// One vector per term rather than a sum and a note, because the failure a
/// gradient assembly can hide is a term that is silently absent: a term that
/// can be read on its own can be checked on its own.
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
/// \ingroup qcx-properties
struct TotalGradientTerms {
    /// `d Tr(D h)/dR` at the fixed reference density, `h = T + V`.
    Eigen::VectorXd oneElectron;
    /// `dE_nuc/dR`, the analytic pair sum.
    Eigen::VectorXd nuclearRepulsion;
    /// `dE_J/dR`, `E_J = 1/2 sum D D (uv|ls)`.
    Eigen::VectorXd coulomb;
    /// `-(1/2) dE_K/dR`, `E_K = 1/2 sum D D (ul|vs)` - the exchange as the
    /// closed-shell energy subtracts it.
    Eigen::VectorXd exchange;
    /// The energy-weighted response term `Tr(F D(1))`, one entry per
    /// displacement.
    Eigen::VectorXd response;
    /// `dE_disp/dR`; empty when no dispersion correction was requested (an
    /// absent term stays absent rather than being a zero that a consumer
    /// could read as a computed one).
    Eigen::VectorXd dispersion;
};

/// The energies the gradient's terms belong to, in Hartree.
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
/// \ingroup qcx-properties
struct TotalGradientEnergies {
    /// The reference's own electronic energy, `1/2 Tr[D (H + F[D])]`, as the
    /// SCF reported it.
    double electronic = 0.0;
    /// The electronic energy the derivative walks' contractions rebuild -
    /// `Tr(DT) + Tr(DV) + E_J - (1/2) E_K`. It exists to be compared with
    /// `electronic`: the residual is one of the gates this entry refuses on.
    double reconstructedElectronic = 0.0;
    /// The nuclear repulsion at the reference geometry.
    double nuclearRepulsion = 0.0;
    /// The dispersion correction; zero when none was requested.
    double dispersion = 0.0;
    /// `electronic + nuclearRepulsion + dispersion`.
    double total = 0.0;
};

/// What the underlying walks counted, kept so the cost of a gradient stays
/// readable.
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
/// \ingroup qcx-properties
struct TotalGradientCounts {
    /// The one-electron derivative walk's counts.
    integrals::OneElectronGradientCounts oneElectron;
    /// The two-electron derivative Fock-block walk's counts.
    integrals::TwoElectronFockDerivativeCounts twoElectron;
};

/// A converged reference's total analytic gradient, term by term.
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
/// \ingroup qcx-properties
struct TotalGradient {
    /// The total gradient: the terms summed in the order oneElectron,
    /// nuclearRepulsion, coulomb, exchange, response, dispersion. It is that
    /// sum and not an independently assembled number, so the decomposition
    /// and the total can never disagree.
    Eigen::VectorXd gradient;
    /// The decomposition.
    TotalGradientTerms terms;
    /// The energies the gradient belongs to.
    TotalGradientEnergies energies;
    /// The walks' counted cost.
    TotalGradientCounts counts;
    /// Atoms in the reference molecule.
    std::size_t atomCount = 0;
    /// Basis functions.
    std::size_t numBasisFunctions = 0;
    /// Occupied spatial orbitals.
    std::size_t numOccupied = 0;
    /// Virtual spatial orbitals.
    std::size_t numVirtual = 0;
    /// What the response solves cost; one cold solve per coordinate.
    response::ResponseStateReport response;
    /// The reference's stationarity measure, `max|2 F_ai|` in the MO basis.
    /// The entry refuses when it exceeds TotalGradientOptions'
    /// stationarityTolerance.
    double stationarityResidual = 0.0;
    /// `|reconstructedElectronic - electronic|`. The entry refuses when it
    /// exceeds TotalGradientOptions' energyTolerance.
    double energyResidual = 0.0;
    /// The names of the right-hand-side contributions the response solves
    /// read, in the order they were wired in. A missing contribution is
    /// invisible in the total; this list is what makes it visible.
    std::vector<std::string> rightHandSideTerms;
};

/// Total-gradient settings.
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
/// \ingroup qcx-properties
struct TotalGradientOptions {
    /// The dense ERI tensor's build settings. `accuracy` is the ONE screening
    /// preset of the calculation - it is read by the tensor build and by the
    /// two-electron derivative walk (the one-electron walk forms no pair
    /// screen) - and `maxTensorBytes` is the admission cap of the reference
    /// path (a dense tensor is `O(n^4)` by construction).
    integrals::EriDenseOptions eri;
    /// Whether the two-electron derivative walk screens its quartets. Off is
    /// the exact contraction and costs accordingly.
    bool useDensityScreening = true;
    /// The per-coordinate response solvers' settings. One solver is kept per
    /// displacement inside the response state.
    response::ResponseSolveOptions solver;
    /// The dispersion correction to add. Empty adds no term and leaves
    /// `TotalGradientTerms::dispersion` empty; a D4 request is refused by the
    /// dispersion module's gradient entry, whose refusal is propagated.
    std::optional<dispersion::ResolvedDispersion> dispersion;
    /// The gate on `max|2 F_ai|`: a reference above it is not a stationary
    /// point of the energy the gradient differentiates, so its gradient is
    /// refused before it is emitted.
    double stationarityTolerance = 1e-6;
    /// The gate on `|E_reconstructed - E_reference|`: the derivative walks'
    /// own energies must agree with the reference's electronic energy.
    double energyTolerance = 1e-8;
};

/// Builds the response state a total-gradient calculation solves against.
///
/// The state owns copies of everything its operator reads, so the heavy work
/// happens here once: the AO ERI tensor is built through the dense reference
/// builder (admission-capped), transformed to the MO basis by four
/// quarter-transforms, the AO Fock is evaluated at the reference density by
/// the SCF module's own evaluator (`EvaluateRhfDensityEnergy` - the
/// Roothaan Fock, not a second integral path), and the screening structure is
/// stamped with the reference geometry's digest.
///
/// The screening structure is at AO-pair granularity, one entry per pair
/// `mu <= nu`, because that is the shape the response state declares. Each
/// AO pair carries the bound of its containing shell pair, which is the
/// shell pair's own maximum over the block - the conservative direction. The
/// state carries and guards these bounds; it does not interpret them.
///
/// The reference is re-validated here for what this entry reads: its arrays
/// must be square and shaped by the same basis-function count the basis set
/// gives, the molecule must be closed shell, and its electron count must be
/// twice the stated occupied count. The state's own factory then refuses a
/// non-converged reference, a density that is not `2 C_occ C_occ^T` for the
/// stated occupied count, an asymmetric Fock, and a screening digest from
/// another geometry.
/// \param molecule The reference molecule, coordinates in Bohr.
/// \param basisSet The basis set the reference was run in.
/// \param reference The converged SCF state to differentiate (scf::
/// ExtractGradientReference).
/// \param eriOptions The dense tensor's build settings.
/// \param solverOptions The settings every per-coordinate solver starts with.
/// \returns The state, or an Error: kInvalidArgument naming the shape, count
/// or consistency refusal; kUnimplemented for a non-closed-shell molecule;
/// the tensor build's own kOutOfMemory or admission refusal; the response
/// module's refusal otherwise.
/// \ingroup qcx-properties
qcx::Result<response::ResponseState> BuildResponseState(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::scf::ScfGradientReference& reference,
    const integrals::EriDenseOptions& eriOptions = {},
    const response::ResponseSolveOptions& solverOptions = {});

/// Assembles the total analytic gradient at a converged closed-shell
/// reference.
///
/// The terms, in the order the result's gradient accumulates them:
///
/// 1. `dTr(D h)/dR` from the one-electron derivative walk;
/// 2. `dE_nuc/dR`, the analytic pair sum;
/// 3. `dE_J/dR` and `-(1/2) dE_K/dR` from the two-electron derivative Fock
///    walk (one call covers every displacement; its skeletons are the same
///    contractions its blocks differentiate);
/// 4. `Tr(F D(1))` per displacement, the orbital response solved by the
///    response state against the right-hand side
///    `H'_ov + J'_ov - (1/2) K'_ov` and the overlap derivative `C^T S' C`;
/// 5. `dE_disp/dR` when a dispersion correction was requested.
///
/// Two gates run before anything is emitted, because both failures produce a
/// plausible-looking gradient from a reference the gradient does not belong
/// to: the reference must be stationary (`max|2 F_ai|` within the option) and
/// the derivative walks' own energies must reproduce the reference's
/// electronic energy (within the option). A refused call returns no partial
/// result - the result carries no gradient until both gates have passed.
/// \param molecule The reference molecule (its coordinates are the geometry
/// the gradient is taken at).
/// \param basisSet The basis set the reference was run in.
/// \param reference The converged SCF state to differentiate.
/// \param options The build, screening, solver, dispersion and gate settings.
/// \returns The gradient with its decomposition, energies, counts and solve
/// report, or an Error: the response state's refusals, the derivative walks'
/// own errors, the stationarity or energy-reconstruction refusal
/// (kInvalidArgument, naming the residual and the tolerance), or the
/// dispersion entry's refusal (kUnimplemented for a D4 gradient).
/// \ingroup qcx-properties
qcx::Result<TotalGradient> ComputeTotalGradient(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const qcx::scf::ScfGradientReference& reference,
                                                const TotalGradientOptions& options = {});

} // namespace qcx::properties
