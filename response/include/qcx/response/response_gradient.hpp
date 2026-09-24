#pragma once

/// \file
/// The nuclear-displacement right-hand side: the perturbation that turns the
/// orbital-response solver into a gradient, as distinct from the applied-field
/// perturbation that gives a polarizability.
///
/// The right-hand side is everything in the derivative of the Fock matrix that
/// the response term does not account for - the one-electron derivative, the
/// two-electron derivative, and for a functional the derivative of the
/// exchange-correlation potential. Each is supplied through one seam, so a
/// caller can wire in whichever engine it has without this module linking one.

#include "qcx/error.hpp"
#include "qcx/response/response_operator.hpp"
#include "qcx/response/response_solver.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace qcx::response {

/// \ingroup qcx-response
/// One nuclear displacement: which atom moved, and along which Cartesian axis.
///
/// Its own type rather than a packed index so that the two coordinates of a
/// displacement cannot be handed over in each other's place, and so that the
/// sign convention of the direction is stated once, here.
struct NuclearDisplacement {
    std::size_t atom = 0; ///< Atom index into the molecule's coordinate list.
    std::size_t direction = 0; ///< Cartesian axis: 0 = x, 1 = y, 2 = z.
};

/// \ingroup qcx-response
/// Names a displacement, for reporting and for a refusal that says which one it
/// refused.
/// \param displacement The displacement to name.
/// \returns Its name, of the form "atom<k>-<axis>".
std::string ToString(const NuclearDisplacement& displacement);

/// \ingroup qcx-response
/// The identity of the reference state a solved response belongs to.
///
/// A response is a derivative at a point: it is the answer for one geometry and
/// one density, and it is meaningless at any other. The two digests are what
/// makes that checkable rather than assumed, and they are carried on the cache
/// handle so that a stale answer is refused by name.
///
/// Both are order-sensitive hashes of the bits of the arrays they summarize, so
/// any change at all - a moved atom, a reconverged density, a different
/// iteration - gives a different identity. That is deliberately finer than it
/// needs to be: an invalidation that fires when nothing changed costs a re-solve,
/// and one that stays silent when something did leaves the energy right and the
/// gradient wrong.
struct ResponseReferenceIdentity {
    std::uint64_t geometry = 0; ///< Digest of the nuclear coordinates, in Bohr.
    std::uint64_t density = 0; ///< Digest of the reference AO density.

    /// \param other The identity to compare against.
    /// \returns Whether both digests are equal.
    bool operator==(const ResponseReferenceIdentity& other) const = default;
};

/// \ingroup qcx-response
/// Digests the reference state a response is solved at.
/// \param coordinatesBohr The nuclear coordinates, 3 per atom, in Bohr.
/// \param density The spin-summed AO density, row-major and symmetric.
/// \returns The identity, or an Error (kInvalidArgument when either array is
/// empty or the coordinates do not number a whole number of atoms).
qcx::Result<ResponseReferenceIdentity> MakeResponseReferenceIdentity(
    std::span<const double> coordinatesBohr, std::span<const double> density);

/// \ingroup qcx-response
/// What one right-hand-side contribution is handed.
///
/// The reference quantities are borrowed rather than copied, and they are the
/// ones a contribution cannot obtain for itself from the request's own
/// contents: the derivatives it differentiates are the caller's to fetch.
struct FockDerivativeRequest {
    ResponseLayout layout; ///< Occupied/virtual structure of the response.
    NuclearDisplacement displacement; ///< The displacement being responded to.
    std::size_t numBasisFunctions = 0; ///< AO count (may exceed the MO count).
    /// The reference MO coefficients, numBasisFunctions x numOrbitals,
    /// row-major. Column p is orbital p, occupied orbitals first.
    std::span<const double> moCoefficients;
    /// The reference spin-summed AO density, numBasisFunctions^2, row-major.
    std::span<const double> density;

    /// \returns The orbital count, numOccupied + numVirtual.
    std::size_t NumOrbitals() const {
        return layout.numOccupied + layout.numVirtual;
    }
};

/// \ingroup qcx-response
/// One term of the occupied-virtual block of the nuclear-derivative Fock matrix.
///
/// The callback ADDS its term into \p block, which arrives zeroed: contributions
/// accumulate, so the assembly is the sum of what was wired in and nothing else.
/// A contribution that needs a derivative integral fetches it itself; that is
/// what keeps this module free of a link to an integral engine.
///
/// \param request The reference and the displacement.
/// \param block The occupied-virtual block to add into, flattened occupied-major
/// as ResponseLayout documents, size layout.Dimension().
using FockDerivativeContributionFn =
    std::function<qcx::Result<void>(const FockDerivativeRequest& request, std::span<double> block)>;

/// \ingroup qcx-response
/// The nuclear-displacement right-hand side, assembled from named contributions.
///
/// The assembled object is B of `A U = -B`: the occupied-virtual block of the
/// derivative Fock matrix, before the sign flip the solve applies.
///
/// Names are kept because the failure this seam can hide is a MISSING
/// contribution, and a missing contribution is invisible in the sum. A caller
/// that reports the names it wired in reports what its answer includes.
class NuclearDisplacementRightHandSide {
public:
    /// Builds an empty right-hand side for one occupied/virtual structure.
    /// \param layout Occupied/virtual counts; both must be non-zero.
    /// \param numBasisFunctions AO count; must be at least the orbital count.
    /// \returns The right-hand side, or an Error (kInvalidArgument on a zero
    /// block or an AO count below the orbital count).
    static qcx::Result<NuclearDisplacementRightHandSide> Create(ResponseLayout layout,
                                                                std::size_t numBasisFunctions);

    /// Wires in one contribution under a name.
    /// \param name A name for the term, kept for reporting; must be non-empty
    /// and not already used.
    /// \param contribution The term; must be non-null.
    /// \returns An Error (kInvalidArgument) on a duplicate or empty name, or a
    /// null callback.
    qcx::Result<void> Add(std::string_view name, FockDerivativeContributionFn contribution);

    /// Assembles B into \p occupiedVirtualBlock, which arrives zeroed.
    /// \param request The reference and the displacement to assemble for.
    /// \param occupiedVirtualBlock Output, size Dimension().
    /// \returns An Error (kInvalidArgument) on a shape mismatch or a request
    /// that does not describe this right-hand side's layout; otherwise the
    /// first contribution's error.
    qcx::Result<void> Assemble(const FockDerivativeRequest& request,
                               std::span<double> occupiedVirtualBlock) const;

    /// \returns The names of the contributions, in the order they were added.
    const std::vector<std::string>& Names() const {
        return _names;
    }

    /// \returns The occupied/virtual structure.
    const ResponseLayout& Layout() const {
        return _layout;
    }

    /// \returns The flattened dimension, numOccupied * numVirtual.
    std::size_t Dimension() const {
        return _layout.Dimension();
    }

private:
    NuclearDisplacementRightHandSide(ResponseLayout layout, std::size_t numBasisFunctions);

    ResponseLayout _layout;
    std::size_t _numBasisFunctions = 0;
    std::vector<std::string> _names;
    std::vector<FockDerivativeContributionFn> _contributions;
};

/// \ingroup qcx-response
/// A solved orbital response to one nuclear displacement.
///
/// The perturbed density is the primary object and the amplitudes are the
/// secondary one, which is the order a consumer wants them in: the density is
/// what a gradient or a derivative property contracts against, and the
/// amplitudes are what produced it.
///
/// SCOPE: this carries the FIRST-ORDER response only. It is not a gradient, and
/// the energy-weighted term that turns it into one is a separate contraction -
/// see EnergyWeightedTerm - because whether that term belongs in a given
/// property is the consumer's decision, not this object's.
struct OrbitalResponse {
    /// The perturbed density, numBasisFunctions^2, row-major and symmetric,
    /// in the AO basis. This is the object a gradient contracts against.
    std::vector<double> perturbedDensity;

    /// The response amplitudes: the full first-order MO rotation U, written so
    /// that the perturbed coefficient matrix is C (1 + U). Stored as the
    /// numOrbitals^2 row-major matrix U(q, p) - the amount of orbital q mixed
    /// into orbital p. The occupied-virtual block of this matrix is the
    /// solution of `A U = -B`; the occupied-occupied and virtual-virtual blocks
    /// are fixed by orthonormality instead, and the two are written into one
    /// matrix because a perturbed orbital is one object.
    std::vector<double> amplitudes;

    std::size_t numBasisFunctions = 0; ///< AO count.
    std::size_t numOrbitals = 0; ///< MO count, occupied + virtual.
    NuclearDisplacement displacement; ///< The displacement this responds to.
    ResponseReferenceIdentity reference; ///< The reference it was solved at.

    /// \param mu Row index.
    /// \param nu Column index.
    /// \returns The perturbed density's element (mu, nu), row-major.
    /// \pre The indices are within numBasisFunctions.
    double Density(std::size_t mu, std::size_t nu) const {
        return std::span<const double>(perturbedDensity)[mu * numBasisFunctions + nu];
    }

    /// \param q Row index.
    /// \param p Column index.
    /// \returns The amplitude U(q, p), row-major.
    /// \pre The indices are within numOrbitals.
    double Amplitude(std::size_t q, std::size_t p) const {
        return std::span<const double>(amplitudes)[q * numOrbitals + p];
    }
};

/// \ingroup qcx-response
/// The handle a solved response is filed under, and the thing that makes a stale
/// one refusable.
///
/// A response is only valid at the reference it was solved at and for the
/// displacement it was solved for. Both are in the key, so a lookup that does
/// not carry today's identity misses - which is the conservative direction: it
/// costs a re-solve, where the other direction costs a quietly wrong gradient.
struct ResponseCacheHandle {
    NuclearDisplacement displacement; ///< The displacement it was solved for.
    ResponseReferenceIdentity reference; ///< The reference it was solved at.

    /// \param other The handle to compare against.
    /// \returns Whether both the displacement and the reference agree.
    bool operator==(const ResponseCacheHandle& other) const = default;
};

/// \ingroup qcx-response
/// The explicit response cache: what has been solved, and at what reference.
///
/// It is explicit rather than implicit because a stale response is the worst
/// failure this module can produce - the energy stays right, because the energy
/// does not read the response, and only the gradient moves. Every lookup goes
/// through the handle, and every handle carries the reference.
class OrbitalResponseCache {
public:
    /// Files a solved response under its handle, replacing any entry with the
    /// same displacement. The entry's own reference is what the next lookup
    /// compares against.
    /// \param response The response to file; its displacement is the key.
    /// \returns An Error (kInvalidArgument) when the response is empty or its
    /// reference digest pair is zero, which is what an unfilled one looks like.
    qcx::Result<void> Store(OrbitalResponse response);

    /// Retrieves a response, but only one that was solved at the handle's own
    /// reference.
    /// \param handle The displacement and reference wanted.
    /// \returns The stored response, or nullptr when nothing is filed for that
    /// displacement or the filed one belongs to a different reference.
    const OrbitalResponse* Lookup(const ResponseCacheHandle& handle) const;

    /// Drops every entry.
    void Invalidate() {
        _entries.clear();
    }

    /// \returns How many responses are filed.
    std::size_t Size() const {
        return _entries.size();
    }

private:
    std::vector<OrbitalResponse> _entries;
};

/// \ingroup qcx-response
/// The energy-weighted term: the reference Fock matrix contracted with the
/// perturbed density, Tr(F D(1)) - the whole of what the response contributes to
/// a gradient through the density.
///
/// This is one object and not two. The perturbed density carries both the
/// response amplitudes and the redundant rotations that the orthonormality
/// constraint fixes from the overlap derivative; contracting the amplitudes
/// alone - an `L^T U` - keeps half of the first and none of the second. The
/// remainder is neither small nor zero at a converged reference, and no
/// symmetry check sees its absence.
///
/// \param fock The Fock matrix at the reference density in the AO basis,
/// numBasisFunctions^2, row-major: the same matrix the MO-basis Fock, and hence
/// the orbital gradient, is transformed from.
/// \param response The solved response, whose perturbed density is contracted.
/// \returns Tr(F D(1)), or an Error (kInvalidArgument) on a shape mismatch.
qcx::Result<double> EnergyWeightedTerm(std::span<const double> fock,
                                       const OrbitalResponse& response);

/// \ingroup qcx-response
/// Extracts L, the orbital gradient, from a MO-basis Fock matrix.
///
/// L_ai = 2 F_ai for a closed-shell reference with real orbitals, which is the
/// statement that the density is twice the occupied block. It is the reference's
/// convergence measure - zero exactly when the reference is stationary - and not
/// a gradient contribution on its own: what the perturbed density contributes is
/// the whole of Tr(F D(1)), which is EnergyWeightedTerm's job.
/// \param moFock The MO-basis Fock matrix, numOrbitals^2, row-major.
/// \param layout Occupied/virtual counts.
/// \param orbitalGradient Output, size numOccupied * numVirtual.
/// \returns An Error (kInvalidArgument) on a shape mismatch.
qcx::Result<void> OrbitalGradient(std::span<const double> moFock,
                                  ResponseLayout layout,
                                  std::span<double> orbitalGradient);

/// \ingroup qcx-response
/// Solves the nuclear-displacement response for one displacement.
///
/// The steps, in the order the decomposition puts them in:
///
/// 1. Assemble B, the occupied-virtual block of the derivative Fock matrix,
///    from the wired-in contributions.
/// 2. Solve `A U = -B` for the occupied-virtual amplitudes, with the supplied
///    solver - which is what makes a displacement sweep reuse its previous
///    answer.
/// 3. Fix the occupied-occupied and virtual-virtual blocks of U from
///    orthonormality: the perturbed orbitals must stay orthonormal, so
///    `U + U^T = -C^T S^R C` and the redundant rotations are the symmetric half
///    of that condition.
/// 4. Form the perturbed AO density from the whole of U.
///
/// The energy-weighted term is deliberately NOT applied here. It is a separate
/// contraction because whether it belongs in a property depends on whether that
/// property's energy is stationary at the reference, and this function cannot
/// know that.
/// \param hessian The orbital-Hessian operator at the reference.
/// \param rightHandSide The assembled right-hand side.
/// \param request The reference quantities and the displacement.
/// \param moOverlapDerivative C^T S^R C, the MO-basis overlap derivative,
/// numOrbitals^2 row-major; its symmetric part fixes the redundant rotations.
/// \param reference The reference identity to stamp on the answer.
/// \param solver The solver to run, kept by the caller so its stored vectors
/// survive the next displacement.
/// \returns The response, or an Error - kInvalidArgument on a shape mismatch,
/// otherwise whatever the assembly, the solve or the preconditioner refused.
qcx::Result<OrbitalResponse> SolveNuclearDisplacementResponse(
    const OrbitalHessianOperator& hessian,
    const NuclearDisplacementRightHandSide& rightHandSide,
    const FockDerivativeRequest& request,
    std::span<const double> moOverlapDerivative,
    const ResponseReferenceIdentity& reference,
    ResponseSolver<OrbitalHessianOperator>& solver);

} // namespace qcx::response
