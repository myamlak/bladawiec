#pragma once

/// \file
/// The matrix-free two-electron action of the orbital Hessian: the term a dense
/// MO tensor would carry, applied to a trial vector without ever forming the
/// orbitals^4 array.
///
/// The dense operator in response_operator.hpp reads the MO-basis tensor
/// `(pq|rs)` and contracts three of its indices for every amplitude. The same
/// action has a much smaller representation. With occupied orbitals i, j,
/// virtual orbitals a, b, MO coefficients C and a trial vector x flattened
/// occupied-major (element `i * numVirtual + a`), form the AO matrix
///
///     M_{mu nu} = sum_{j b} C_{mu, n_o + b} x_{j b} C_{nu, j}
///
/// and symmetrise it, `S = (M + M^T) / 2`. The Coulomb term
/// `4 sum (ai|bj) x_{jb}` is `4 C^T J(S) C` on the occupied-virtual block,
/// and the two exchange terms are `-2 c_HF C^T K(S) C`: `(ab|ij)` contracts
/// `K(M)` and `(aj|ib)` contracts `K(M^T)`, whose sum is `2 K(S)`. The whole
/// two-electron action is therefore
///
///     y_{j b} += 2 sum_{mu nu} C_{mu, n_o + b} F(S)_{mu nu} C_{nu, j},
///
/// where `F(S) = 2 J(S) - c_HF K(S)` is the closed-shell Fock the direct
/// engine returns for the density argument `S` (with a zero core, its
/// two-electron part is exactly that contraction). The direct screened engine
/// evaluates F without storing an n^4 array, so the action costs one Fock
/// build (two when the exact-exchange fraction is strictly between zero and
/// one) and four n^2-by-n transforms per application, and holds O(n^2) memory.
///
/// The action is identical to the dense tensor's in exact arithmetic. With
/// the density screen on, it drops the same small quartets the rest of the
/// engine drops; the exact band is the derivative options with the screen
/// off, which is what the dense band's `EriDenseOptions::screen = false`
/// gives on the other side.
///
/// Cost, and where this kernel belongs. One application is one full direct
/// Fock build (two when the exact-exchange fraction is strictly between zero
/// and one) plus four n^2-by-n transforms; the dense band's application is a
/// precomputed orbitals^4 contraction. The matrix-free application is
/// therefore not competitive per application wherever the dense tensor fits:
/// the factor-of-three per-application target is unmet, and closing the gap is
/// not a goal of this kernel - the gap is structural, not a tuning defect. The
/// total-gradient state build's automatic band policy therefore chooses the
/// dense band whenever the a-priori estimate is admitted, and this band only
/// when it is not: the memory fallback, not the hot path. The timed comparison
/// is the `qcx-bench-response-matvec` quiet-window run - both arms on one
/// fixture and trial vector, warm-up applications discarded, the
/// per-application minimum quoted beside the fixed-work canary.

#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/response/response_operator.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace qcx::basisset {
class BasisSet;
}

namespace qcx::molecule {
class Molecule;
}

namespace qcx::response {

/// \ingroup qcx-response
/// The screening settings the matrix-free two-electron action runs under.
///
/// They are the integrals engine's own settings, not a second accuracy
/// scheme: the action is one Fock build, and it screens exactly as a Fock
/// build does. `accuracy` is the run's single preset; the two flags are the
/// engine's two-level density screen; the certified fp32 lane is pinned off
/// by the class itself (an action applied to many trial vectors must not
/// change its arithmetic between applications).
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
struct MatrixFreeEriDerivativeOptions {
    /// The screening and certified-lane preset handed to the Fock builder.
    integrals::AccuracyPreset accuracy = integrals::AccuracyPreset::kNormal;
    /// Whether each Fock build screens its quartets by the density's pair
    /// maxima. Off is the exact contraction and costs accordingly.
    bool useDensityScreening = true;
    /// Whether the kept quartets' elements are re-filtered per target. Only
    /// meaningful with the density screen on; off with it gives the exact
    /// contraction.
    bool usePerElementScreening = true;
    /// The Fock builder's per-class batch cap.
    std::size_t maxBatchBytes = 512 * 1024 * 1024;
    /// The Fock builder's parallel chunk override: 0 = auto (the builder's
    /// own split over the run's OpenMP ceiling), 1 = the serial path, >= 2 a
    /// fixed split. A timing instrument pins 1 so an application's cost is not
    /// a scheduling reading; the production gradient leaves it auto and the
    /// run's thread ceiling governs.
    std::size_t maxParallelChunks = 0;
};

/// \ingroup qcx-response
/// The two-electron term of the orbital-Hessian action, computed on the fly.
///
/// An instance owns one or two direct Fock builders (created once, with the
/// molecule, basis and core Hamiltonian they need), a copy of the MO
/// coefficients and the exact-exchange fraction. `Apply` adds the action of
/// the two-electron term into a result buffer; it never forms an n^4 array
/// and never touches the dense MO tensor.
///
/// The class also builds the reference Fock at a density through the same
/// engine and the same screening settings the action uses: a reference
/// assembled by another path would make the operator the response solves
/// against differ from the matrix the reference converged, and the mismatch
/// would look like a convergence defect.
///
/// One application runs at a time: the builders and the scratch matrices are
/// reused between applications, so a caller must not apply an instance from
/// two threads concurrently.
class MatrixFreeEriDerivative {
public:
    /// Builds the action.
    /// \param molecule The reference molecule, coordinates in Bohr.
    /// \param basisSet The basis the reference was run in.
    /// \param coreHamiltonian The core Hamiltonian H = T + V, row-major
    /// numBasisFunctions^2; the reference Fock methods add it.
    /// \param layout The occupied/virtual structure.
    /// \param numBasisFunctions The AO count n; at least the orbital count.
    /// \param moCoefficients The MO coefficients, row-major
    /// numBasisFunctions x (numOccupied + numVirtual), occupied orbitals
    /// first. Copied.
    /// \param exchangeFraction The exact-exchange fraction c_HF the action
    /// scales its exchange term by, in [0, 1].
    /// \param options The builder's screening and batch settings.
    /// \returns The action, or an Error (kInvalidArgument) naming a shape or
    /// fraction mismatch, or the Fock builder's own error.
    static qcx::Result<MatrixFreeEriDerivative> Create(const qcx::molecule::Molecule& molecule,
                                                       const qcx::basisset::BasisSet& basisSet,
                                                       std::span<const double> coreHamiltonian,
                                                       ResponseLayout layout,
                                                       std::size_t numBasisFunctions,
                                                       std::span<const double> moCoefficients,
                                                       double exchangeFraction,
                                                       MatrixFreeEriDerivativeOptions options = {});

    /// Adds the two-electron term of the action into \p y:
    /// `y += 2 C^T F(S) C` on the occupied-virtual block, with S the
    /// symmetrised trial-vector density and F the scaled closed-shell Fock.
    /// \param x Trial vector, size Layout().Dimension().
    /// \param y Result to add into, size Layout().Dimension().
    /// \returns An Error (kInvalidArgument) on a length mismatch.
    qcx::Result<void> Apply(std::span<const double> x, std::span<double> y) const;

    /// Builds the Roothaan Fock `H + 2 J(rho) - K(rho)` at a spatial density
    /// through the same builder the action uses.
    /// \param spatialDensity The spatial closed-shell density rho = D / 2,
    /// row-major numBasisFunctions^2 and symmetric.
    /// \returns The Fock, row-major numBasisFunctions^2, or an Error
    /// (kInvalidArgument for a shape mismatch; kUnimplemented when the
    /// derivative was built with c_HF = 0, which carries no exchange builder).
    qcx::Result<std::vector<double>> ReferenceFock(std::span<const double> spatialDensity) const;

    /// Builds the Coulomb-only Fock `H + 2 J(rho)` at a spatial density
    /// through the same builder the action uses.
    /// \param spatialDensity The spatial closed-shell density rho = D / 2,
    /// row-major numBasisFunctions^2 and symmetric.
    /// \returns The Fock, row-major numBasisFunctions^2, or an Error
    /// (kInvalidArgument for a shape mismatch).
    qcx::Result<std::vector<double>> CoulombFock(std::span<const double> spatialDensity) const;

    /// \returns The occupied/virtual structure.
    ResponseLayout Layout() const;

    /// \returns The AO count.
    std::size_t NumBasisFunctions() const;

    /// \returns The exact-exchange fraction the action scales its exchange
    /// term by.
    double ExchangeFraction() const;

    /// \returns The MO coefficients the action reads, row-major.
    std::span<const double> MoCoefficients() const;

    /// \returns The screening settings the action runs under.
    const MatrixFreeEriDerivativeOptions& Options() const;

    /// Moves the action's builders and scratch storage; the moved-from object
    /// holds no state and must not be applied.
    MatrixFreeEriDerivative(MatrixFreeEriDerivative&&) noexcept;
    /// Releases the target's builders and takes the source's; the moved-from
    /// object holds no state and must not be applied.
    /// \returns The target, \p *this.
    MatrixFreeEriDerivative& operator=(MatrixFreeEriDerivative&&) noexcept;
    /// Copying is deleted: the builders and the mutable scratch storage are
    /// the action's identity, and a copy would alias the scratch.
    MatrixFreeEriDerivative(const MatrixFreeEriDerivative&) = delete;
    /// Copy assignment is deleted for the same reason as the copy
    /// constructor.
    /// \returns The target; never called, the function is deleted.
    MatrixFreeEriDerivative& operator=(const MatrixFreeEriDerivative&) = delete;
    /// Releases the builders and scratch storage.
    ~MatrixFreeEriDerivative();

private:
    /// The builders, coefficients and scratch matrices the class owns.
    struct State;
    /// Takes ownership of the built state; only Create constructs one.
    explicit MatrixFreeEriDerivative(std::unique_ptr<State> state);

    // Mutable because an application reuses the builders' and the scratch
    // matrices' storage; the class is conceptually constant.
    /// The owned state; null only in a moved-from object.
    mutable std::unique_ptr<State> _state;
};

} // namespace qcx::response
