#pragma once

#include "qcx/error.hpp"
#include "qcx/response/response_operator.hpp"

#include <cstddef>
#include <span>
#include <string>

namespace qcx::response {

/// A density functional's contribution to the orbital-Hessian action.
///
/// The action is the kernel's own: for the XC part of the orbital Hessian it is
/// the second functional derivative contracted twice with the density response
/// to the trial vector, and it adds its result into the caller's buffer. The
/// seam cannot tell a kernel from a right-hand side, so the term is wired in as
/// the quadratic part of the energy derivative - not as a correction.
///
/// The name travels with the callback so a solve report or a refusal can say
/// which kernel it used. Public aggregate: the fields are the API
/// (aggregate-struct exemption).
/// \ingroup qcx-response
struct KohnShamKernelTerm {
    std::string name; ///< The functional's name, for reporting.
    HessianExtraTermFn apply; ///< Adds the kernel's action into the result.
};

/// The Kohn-Sham orbital-Hessian operator: the Hartree-Fock action with a
/// scaled exact-exchange term and the density functional's kernel added.
///
/// It is the operator a restricted Kohn-Sham response solves against,
///     A = (e_a - e_i) delta + 4 (ai|bj) - c_HF [(ab|ij) + (aj|ib)] + A_xc,
/// with c_HF from the functional's ExchangeFraction() and A_xc the kernel term.
/// The class owns no grid and no density: the kernel callback carries them, and
/// it must outlive the operator.
///
/// \pre The kernel's callback and every span handed to Create outlive the
/// operator.
/// \ingroup qcx-response
class KohnShamOrbitalHessian {
public:
    /// Builds the operator.
    /// \param layout Occupied/virtual counts; both must be non-zero.
    /// \param orbitalEnergies Canonical orbital energies, occupied orbitals
    /// first, size `layout.numOccupied + layout.numVirtual`.
    /// \param moTwoElectron MO-basis two-electron integrals in chemists'
    /// notation, full n^4. Neither span is copied.
    /// \param exchangeFraction The exact-exchange fraction c_HF, in [0, 1].
    /// \param kernel The kernel term; its name must be non-empty and its
    /// callback non-null.
    /// \returns The operator, or an Error: the operator's own refusals
    /// (kInvalidArgument) and kInvalidArgument for an empty name or a null
    /// callback.
    /// \ingroup qcx-response
    static qcx::Result<KohnShamOrbitalHessian> Create(ResponseLayout layout,
                                                      OrbitalEnergies orbitalEnergies,
                                                      MoTwoElectronTensor moTwoElectron,
                                                      double exchangeFraction,
                                                      KohnShamKernelTerm kernel);

    /// Applies the operator: y = A x, with every term.
    /// \param x Trial vector, size Dimension().
    /// \param y Result vector, size Dimension(); must not alias \p x.
    /// \returns An Error (kInvalidArgument) on a length mismatch, or the
    /// kernel's own error.
    /// \ingroup qcx-response
    qcx::Result<void> Apply(std::span<const double> x, std::span<double> y) const {
        return _base.Apply(x, y);
    }

    /// Writes the diagonal preconditioner: the canonical orbital-energy
    /// difference `e_a - e_i`.
    /// \param diagonal Output, size Dimension().
    /// \returns An Error (kInvalidArgument) on a length mismatch.
    /// \ingroup qcx-response
    qcx::Result<void> Preconditioner(std::span<double> diagonal) const {
        return _base.Preconditioner(diagonal);
    }

    /// The complete action as the underlying operator: the scaled
    /// exact-exchange term and the kernel are both in it, so it is what a
    /// nuclear-displacement solve uses.
    /// \returns The operator.
    /// \ingroup qcx-response
    const OrbitalHessianOperator& BaseOperator() const {
        return _base;
    }

    /// The kernel term's name.
    /// \returns The name.
    /// \ingroup qcx-response
    const std::string& KernelName() const noexcept {
        return _kernelName;
    }

    /// \returns The occupied/virtual block structure.
    /// \ingroup qcx-response
    const ResponseLayout& Layout() const {
        return _base.Layout();
    }

    /// \returns The flattened dimension, numOccupied * numVirtual.
    /// \ingroup qcx-response
    std::size_t Dimension() const {
        return _base.Dimension();
    }

    /// \returns The exact-exchange fraction the action scales its exchange
    /// integrals by.
    /// \ingroup qcx-response
    double ExchangeFraction() const noexcept {
        return _base.ExchangeFraction();
    }

private:
    KohnShamOrbitalHessian(OrbitalHessianOperator base, std::string kernelName);

    OrbitalHessianOperator _base;
    std::string _kernelName;
};

} // namespace qcx::response
