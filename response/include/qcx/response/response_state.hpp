#pragma once

/// \file
/// The state a displacement sweep carries between geometries: the reference
/// wavefunction and Fock, the screening structure it was solved at, and one
/// response solver per displacement ordinal.
///
/// The orbital-response solver already stores its previous answers; what a
/// sweep needs beyond that is an owner for the reference quantities the
/// operator and the right-hand side read, and a place where one solver per
/// coordinate guarantees that slot `k` is reused for coordinate `k` and not
/// for whichever displacement happened to be solved last. This is that owner.

#include "qcx/error.hpp"
#include "qcx/response/response_gradient.hpp"
#include "qcx/response/response_operator.hpp"
#include "qcx/response/response_solver.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace qcx::response {

/// \ingroup qcx-response
/// The screening structure a displacement sweep reuses.
///
/// The response operator and right-hand side do not screen; the derivative
/// integral blocks a gradient consumer assembles above them do. The pair
/// bounds live here so that a sweep carries them with the reference they were
/// built for instead of rebuilding them at every displacement, and so that a
/// bound set from another geometry is REFUSED rather than silently read
/// against this one.
///
/// One entry per AO pair `(mu, nu)`, `mu <= nu`, row-major in `mu` then `nu`:
/// `n (n + 1) / 2` entries at `n` basis functions. The two arrays are the same
/// pair set and must be the same length; what each bound means is the
/// builder's contract.
struct ResponseScreening {
    /// Digest of the nuclear coordinates these bounds belong to, as
    /// MakeResponseReferenceIdentity computes it. A mismatch is a stale
    /// structure, not a licence to re-screen.
    std::uint64_t geometry = 0;

    /// The pair's integral bound, one entry per AO pair.
    std::vector<double> pairValues;

    /// The pair's coordinate-derivative bound, one entry per AO pair.
    std::vector<double> pairDerivativeBounds;
};

/// \ingroup qcx-response
/// The reference SCF's convergence record, carried so a gradient consumer can
/// state what its reference cost and how tightly it was converged.
struct ResponseStateHistory {
    std::size_t scfIterations = 0; ///< Iterations the reference SCF spent.
    double achievedEnergyDelta = 0.0; ///< The SCF gate's energy leg at its stop.
    double achievedRmsDensityChange = 0.0; ///< The SCF gate's density leg at its stop.
};

/// \ingroup qcx-response
/// The converged reference a ResponseState is built from.
///
/// Every array is copied at construction; the spans only have to outlive the
/// Create or Rebase call. The MO coefficients are row-major
/// `numBasisFunctions x (numOccupied + numVirtual)`, occupied orbitals first;
/// the density and Fock matrices are `numBasisFunctions^2` row-major; the
/// two-electron tensor is the orbital count to the fourth, chemists' notation
/// `(pq|rs)` indexed `((p n + q) n + r) n + s`.
struct ResponseStateReference {
    bool converged = false; ///< The reference SCF's convergence flag.
    std::size_t numBasisFunctions = 0; ///< The AO count n.
    std::size_t numOccupied = 0; ///< Occupied orbitals.
    std::size_t numVirtual = 0; ///< Virtual orbitals.
    std::span<const double> coordinatesBohr; ///< The nuclear coordinates, 3 per atom, Bohr.
    std::span<const double> moCoefficients; ///< MO coefficients, row-major n x orbitals.
    std::span<const double> orbitalEnergies; ///< Canonical orbital energies, occupied first.
    std::span<const double> density; ///< The spin-summed density D = 2 C_occ C_occ^T, row-major.
    std::span<const double> fock; ///< The AO Fock at that density, row-major.
    std::span<const double> moTwoElectron; ///< The MO two-electron tensor, orbitals^4.
    ResponseStateHistory history; ///< The reference SCF's convergence record.
};

/// \ingroup qcx-response
/// One solved displacement, with what its solve cost.
struct ResponseStateSolution {
    OrbitalResponse response; ///< The solved response, stamped with the state's reference.
    bool warmStarted = false; ///< The solve began from the stored answer for this ordinal.
    std::size_t iterations = 0; ///< Conjugate-gradient iterations the solve spent.
};

/// \ingroup qcx-response
/// The state's cumulative solve record.
///
/// The totals span every solve the state has run, across rebases: a sweep that
/// restarted its counters at each geometry could not be told from one that
/// restarted its work. `matvecs` and the worst residual are deliberately NOT
/// here: the response solve entry returns the solved response rather than its
/// report, so this state did not observe them and does not invent them.
struct ResponseStateReport {
    std::size_t solves = 0; ///< Right-hand sides solved.
    std::size_t warmStarted = 0; ///< Solves that began from a stored answer.
    std::size_t iterations = 0; ///< Conjugate-gradient iterations in total.
};

/// \ingroup qcx-response
/// The reference state, screening structure and per-coordinate solvers a
/// nuclear-displacement sweep runs against.
///
/// The state owns copies of everything its operator and right-hand side read,
/// so a caller can rebase it to a new geometry without invalidating the old
/// solve data. One solver is kept per displacement ordinal,
/// `3 * atom + direction`: slot `k` always solves coordinate `k`, which is
/// what makes its stored vector the previous geometry's answer to the same
/// question. Rebase rebuilds the operator against the new reference and keeps
/// those vectors - the reuse the sweep exists for.
///
/// The factory refuses what a response cannot use: a non-converged reference,
/// a density that is not `2 C_occ C_occ^T` for the stated occupied count, an
/// asymmetric Fock, arrays whose shapes do not describe one state, and a
/// screening structure whose digest belongs to another geometry. Rebase
/// refuses those and any change of basis-function count, occupied/virtual
/// count or atom count.
///
/// The class is move-only: the operator borrows the state's own arrays, so a
/// copy would leave it reading another object's storage.
class ResponseState {
public:
    /// Builds the state from a converged reference.
    /// \param reference The reference quantities; every array is copied.
    /// \param screening The screening structure, its digest matching the
    /// reference's coordinates.
    /// \param solverOptions The options every per-coordinate solver starts
    /// with.
    /// \returns The state, or an Error (kInvalidArgument) naming the refusal.
    static qcx::Result<ResponseState> Create(ResponseStateReference reference,
                                             ResponseScreening screening,
                                             ResponseSolveOptions solverOptions = {});

    /// Replaces the reference and screening structure, keeping every stored
    /// solution vector.
    ///
    /// This is the sweep's rebase: the operator is rebuilt against the new
    /// reference and each coordinate's solver keeps the answer it stored for
    /// that same coordinate. The new reference goes through the same
    /// validation as Create, and its layout must match this state's - a
    /// changed basis-function, occupied, virtual or atom count is refused
    /// rather than solved with the wrong-shaped stored vectors.
    /// \param reference The new reference; every array is copied.
    /// \param screening The new screening structure, its digest matching the
    /// new coordinates.
    /// \returns An Error (kInvalidArgument) naming the refusal. A refused
    /// rebase leaves the state unchanged.
    qcx::Result<void> Rebase(ResponseStateReference reference, ResponseScreening screening);

    /// Solves the orbital response to one nuclear displacement.
    ///
    /// The request's reference quantities are the state's own, and the solve
    /// runs on the solver kept for this displacement's ordinal, so it starts
    /// from that ordinal's previous answer when one was stored.
    /// \param displacement The atom and axis; must lie inside the molecule.
    /// \param rightHandSide The assembled contributions, built for this
    /// state's layout and basis-function count.
    /// \param moOverlapDerivative `C^T S^R C`, the MO-basis overlap
    /// derivative, orbital count squared and row-major; its symmetric part
    /// fixes the redundant rotations.
    /// \returns The solved response with its cost, or an Error: kInvalidArgument
    /// for a displacement outside the molecule or a right-hand side of another
    /// layout, otherwise whatever the assembly, the solve or the preconditioner
    /// refused. A failed solve leaves the state's report unchanged.
    qcx::Result<ResponseStateSolution> SolveDisplacement(
        NuclearDisplacement displacement,
        const NuclearDisplacementRightHandSide& rightHandSide,
        std::span<const double> moOverlapDerivative);

    /// \returns The occupied/virtual structure.
    ResponseLayout Layout() const {
        return _layout;
    }

    /// \returns The AO count.
    std::size_t NumBasisFunctions() const {
        return _numBasisFunctions;
    }

    /// \returns The atom count, 3 per displacement.
    std::size_t NumAtoms() const {
        return _coordinatesBohr.size() / 3;
    }

    /// \returns The displacement count, 3 * NumAtoms().
    std::size_t NumDisplacements() const {
        return _coordinatesBohr.size();
    }

    /// \returns The screening structure's geometry digest and pair bounds.
    const ResponseScreening& Screening() const {
        return _screening;
    }

    /// \returns The reference's identity digests, stamped on every response.
    const ResponseReferenceIdentity& Identity() const {
        return _identity;
    }

    /// \returns The reference SCF's convergence record.
    const ResponseStateHistory& History() const {
        return _history;
    }

    /// \returns The cumulative solve record.
    const ResponseStateReport& Report() const {
        return _report;
    }

    /// \returns The reference coordinates, 3 per atom, Bohr.
    std::span<const double> CoordinatesBohr() const {
        return _coordinatesBohr;
    }

    /// \returns The reference MO coefficients, row-major.
    std::span<const double> MoCoefficients() const {
        return _moCoefficients;
    }

    /// \returns The reference orbital energies, occupied first.
    std::span<const double> OrbitalEnergies() const {
        return _orbitalEnergies;
    }

    /// \returns The reference spin-summed AO density, row-major.
    std::span<const double> Density() const {
        return _density;
    }

    /// \returns The reference AO Fock, row-major.
    std::span<const double> Fock() const {
        return _fock;
    }

    /// \returns The MO two-electron tensor, chemists' notation, orbital count
    /// to the fourth.
    std::span<const double> MoTwoElectron() const {
        return _moTwoElectron;
    }

    ResponseState(const ResponseState&) = delete;
    ResponseState& operator=(const ResponseState&) = delete;
    ResponseState(ResponseState&&) = default;
    ResponseState& operator=(ResponseState&&) = delete;

private:
    ResponseState();

    /// Migrates a validated reference into the state's own storage and
    /// rebuilds the operator against it.
    /// \param reference The validated reference.
    /// \returns An Error from the operator's construction; on failure the
    /// state's own storage is untouched.
    qcx::Result<void> Install(ResponseStateReference reference);

    ResponseLayout _layout;
    std::size_t _numBasisFunctions = 0;
    std::vector<double> _coordinatesBohr;
    std::vector<double> _moCoefficients;
    std::vector<double> _orbitalEnergies;
    std::vector<double> _density;
    std::vector<double> _fock;
    std::vector<double> _moTwoElectron;
    ResponseStateHistory _history;
    ResponseScreening _screening;
    ResponseReferenceIdentity _identity;
    ResponseSolveOptions _solverOptions;
    std::optional<OrbitalHessianOperator> _operator;
    std::vector<ResponseSolver<OrbitalHessianOperator>> _solvers;
    ResponseStateReport _report;
};

} // namespace qcx::response
