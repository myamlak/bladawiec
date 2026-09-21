#pragma once

/// \file
/// The response-operator seam: the matrix-free action of the orbital-Hessian
/// (coupled-perturbed Hartree-Fock) matrix, and the dense operator the solver's
/// own tests are written on. Nothing here assembles the operator's matrix, and
/// nothing here owns the data a physical operator acts on.

#include "qcx/error.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qcx::response {

/// \defgroup qcx-response Response module
/// The orbital response to a perturbation: the matrix-free action of the
/// closed-shell orbital-Hessian (coupled-perturbed Hartree-Fock) matrix, and
/// the iterative solver that applies it to one or more right-hand sides without
/// ever assembling it. A response consumes a converged wavefunction - the MO
/// two-electron tensor and the orbital energies - and is the piece an analytic
/// second derivative sits on.
/// \{

/// \ingroup qcx-response
/// The occupied/virtual block structure a response operator acts on.
///
/// A response vector is flattened occupied-major: element
/// `i * numVirtual + a` carries the amplitude by which occupied orbital `i`
/// mixes with virtual orbital `a`. Every operator in this header uses that
/// layout, and so do the solvers in `response_solver.hpp`.
struct ResponseLayout {
    std::size_t numOccupied = 0; ///< Occupied orbitals (the row count).
    std::size_t numVirtual = 0; ///< Virtual orbitals (the column count).

    /// \returns The flattened dimension, numOccupied * numVirtual.
    std::size_t Dimension() const {
        return numOccupied * numVirtual;
    }
};

/// \ingroup qcx-response
/// The canonical orbital energies an orbital-Hessian action reads, occupied
/// orbitals first and the virtual orbitals after them.
///
/// Carried as its own type rather than as a bare span so that the integral
/// tensor it is indexed against cannot be handed over in its place: the two are
/// both spans of doubles, and the lengths that tell them apart are runtime
/// facts.
struct OrbitalEnergies {
    /// One energy per orbital, size ResponseLayout's numOccupied + numVirtual.
    std::span<const double> values;
};

/// \ingroup qcx-response
/// The MO-basis two-electron integrals an orbital-Hessian action reads, in
/// chemists' notation (pq|rs), full n^4, indexed `((p * n + q) * n + r) * n + s`.
///
/// Not copied: how the tensor is stored stays the caller's decision, and the
/// span only has to outlive the operator built from it.
struct MoTwoElectronTensor {
    /// The flattened tensor, size n^4 at n = numOccupied + numVirtual.
    std::span<const double> values;
};

/// \ingroup qcx-response
/// The closed-shell Hartree-Fock orbital-Hessian operator, applied matrix-free.
///
/// This is the coupled-perturbed Hartree-Fock matrix that the orbital response
/// to a perturbation is solved against - the "A + B" combination of the
/// orbital-rotation Hessian in the static limit [Helgaker2000]. With occupied
/// orbitals i and j, virtual orbitals a and b, the canonical orbital energies
/// e, and two-electron integrals in chemists' notation (pq|rs),
///
///     A(ai, bj) = (e_a - e_i) d_ab d_ij + 4 (ai|bj) - (ab|ij) - (aj|ib)
///
/// The action is matrix-free: the n_o n_v squared matrix is never formed, and
/// the integral tensor is borrowed rather than copied, so how it is stored
/// stays the caller's decision.
///
/// \pre The spans passed to Create outlive the operator.
class OrbitalHessianOperator {
public:
    /// Builds the action from the MO-basis quantities it consumes.
    /// \param layout Occupied/virtual counts; both must be non-zero.
    /// \param orbitalEnergies Canonical orbital energies, occupied orbitals
    /// first, size `layout.numOccupied + layout.numVirtual`.
    /// \param moTwoElectron MO-basis two-electron integrals in chemists'
    /// notation (pq|rs), full n^4 at `n = layout.numOccupied + layout.numVirtual`.
    /// Neither is copied.
    /// \returns The operator, or an Error (kInvalidArgument for a zero block, a
    /// length mismatch, or an integral array that is not n^4 long).
    static qcx::Result<OrbitalHessianOperator> Create(ResponseLayout layout,
                                                      OrbitalEnergies orbitalEnergies,
                                                      MoTwoElectronTensor moTwoElectron);

    /// Applies the operator: y = A x.
    /// \param x Trial vector, size Dimension().
    /// \param y Result vector, size Dimension(); must not alias \p x.
    /// \returns An Error (kInvalidArgument) on a length mismatch.
    qcx::Result<void> Apply(std::span<const double> x, std::span<double> y) const;

    /// Writes the diagonal preconditioner this operator's solves start from:
    /// the canonical orbital-energy difference `e_a - e_i` at each amplitude.
    /// It is deliberately *not* the operator's own diagonal - the coupled
    /// solve's diagonal preconditioner is the energy difference, and the
    /// small-gap fallback keys on it.
    /// \param diagonal Output, size Dimension().
    /// \returns An Error (kInvalidArgument) on a length mismatch.
    qcx::Result<void> Preconditioner(std::span<double> diagonal) const;

    /// \returns The occupied/virtual block structure.
    const ResponseLayout& Layout() const {
        return _layout;
    }

    /// \returns The flattened dimension, numOccupied * numVirtual.
    std::size_t Dimension() const {
        return _layout.Dimension();
    }

private:
    OrbitalHessianOperator(ResponseLayout layout,
                           OrbitalEnergies orbitalEnergies,
                           MoTwoElectronTensor moTwoElectron);

    ResponseLayout _layout;
    std::size_t _numOrbitals = 0;
    std::span<const double> _orbitalEnergies;
    std::span<const double> _moTwoElectron;
};

/// \ingroup qcx-response
/// A dense response operator: the small-band member of the solver ladder, and
/// the made-up system the iterative solver is checked against.
///
/// It exists so the solver can be exercised on an arbitrary real symmetric
/// matrix with a preconditioner of the caller's choosing, without an integral
/// engine anywhere near it. The matrix is copied at construction.
class DenseResponseOperator {
public:
    /// Builds the operator from a dense symmetric matrix.
    /// \param layout Occupied/virtual counts; both must be non-zero.
    /// \param matrix Row-major matrix, size Dimension()^2.
    /// \returns The operator, or an Error (kInvalidArgument for a zero block or
    /// a length mismatch).
    static qcx::Result<DenseResponseOperator> Create(ResponseLayout layout,
                                                     std::span<const double> matrix);

    /// Builds the operator with a preconditioner that is *not* the matrix's own
    /// diagonal - the situation every physical response operator is in, where
    /// the preconditioner is the orbital-energy difference and the operator's
    /// diagonal carries two-electron terms as well.
    /// \param layout Occupied/virtual counts; both must be non-zero.
    /// \param matrix Row-major matrix, size Dimension()^2.
    /// \param preconditioner Diagonal preconditioner, size Dimension().
    /// \returns The operator, or an Error (kInvalidArgument for a zero block or
    /// a length mismatch).
    static qcx::Result<DenseResponseOperator> Create(ResponseLayout layout,
                                                     std::span<const double> matrix,
                                                     std::span<const double> preconditioner);

    /// Applies the operator: y = A x.
    /// \param x Trial vector, size Dimension().
    /// \param y Result vector, size Dimension(); must not alias \p x.
    /// \returns An Error (kInvalidArgument) on a length mismatch.
    qcx::Result<void> Apply(std::span<const double> x, std::span<double> y) const;

    /// Writes the diagonal preconditioner handed to Create.
    /// \param diagonal Output, size Dimension().
    /// \returns An Error (kInvalidArgument) on a length mismatch.
    qcx::Result<void> Preconditioner(std::span<double> diagonal) const;

    /// \returns The occupied/virtual block structure.
    const ResponseLayout& Layout() const {
        return _layout;
    }

    /// \returns The flattened dimension, numOccupied * numVirtual.
    std::size_t Dimension() const {
        return _layout.Dimension();
    }

private:
    DenseResponseOperator(ResponseLayout layout,
                          std::vector<double> matrix,
                          std::vector<double> preconditioner);

    ResponseLayout _layout;
    std::vector<double> _matrix;
    std::vector<double> _preconditioner;
};

/// \}
} // namespace qcx::response
