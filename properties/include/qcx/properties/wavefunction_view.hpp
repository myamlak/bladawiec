#pragma once

#include "qcx/io/run_input.hpp"

#include <Eigen/Core>
#include <functional>
#include <optional>
#include <type_traits>

namespace qcx::properties {

/// A read-only view of the wavefunction one property analysis runs against.
///
/// One object carries what a density property is allowed to read: the total
/// density, the alpha and beta spin densities when the run is unrestricted,
/// the method kind, and the convergence state the driver already holds. A
/// property takes this view instead of raw density matrices plus a method word
/// it would have to interpret itself.
///
/// The view owns nothing. Every density is a reference to driver-owned
/// storage, so constructing or copying the view copies no matrix, and the type
/// is trivially copyable: handing it to a property costs a handful of pointers.
///
/// Density convention: the per-spin convention populations.hpp states. The
/// total density is P_alpha + P_beta with NO factor of 2 anywhere; on a
/// restricted run it is HfResult::density (D = 2 rho), whose two spin halves
/// would coincide. The view does not materialize those halves: a restricted
/// closed-shell run has one density and an identically zero spin density, so a
/// property that needs the spin densities declares that need and is refused
/// where they are not supplied, rather than receiving fabricated halves.
///
/// The trap this shape exists to avoid: "restricted" does not mean "the two
/// spins are equal". A restricted open-shell wavefunction has a real spin
/// density and is not expressible through this two-constructor view. When that
/// kind lands it extends the view with the spin density; it must never be
/// approximated by the closed-shell halves.
/// \ingroup qcx-properties
class WavefunctionView {
public:
    /// Views a restricted (closed-shell) run: one total density, no spin
    /// resolution.
    /// \param totalDensity The run's total density P_alpha + P_beta.
    /// \param method The method kind the driver ran.
    /// \param converged True when the SCF reported convergence; false means
    /// the densities are the last iterate.
    /// \pre \p method is qcx::io::MethodType::kRhf or
    /// qcx::io::MethodType::kRks.
    WavefunctionView(const Eigen::MatrixXd& totalDensity,
                     qcx::io::MethodType method,
                     bool converged) noexcept :
        _totalDensity(totalDensity), _method(method), _converged(converged) {}

    /// Views an unrestricted run: the total density plus both spin densities.
    /// \param totalDensity The run's total density P_alpha + P_beta.
    /// \param alphaDensity The alpha spin density P_alpha.
    /// \param betaDensity The beta spin density P_beta.
    /// \param method The method kind the driver ran.
    /// \param converged True when the SCF reported convergence; false means
    /// the densities are the last iterate.
    /// \pre \p method is qcx::io::MethodType::kUhf or
    /// qcx::io::MethodType::kUks.
    /// \pre \p alphaDensity + \p betaDensity equals \p totalDensity. The
    /// caller owns the sum: the view copies no matrix and cannot check it.
    WavefunctionView(const Eigen::MatrixXd& totalDensity,
                     const Eigen::MatrixXd& alphaDensity,
                     const Eigen::MatrixXd& betaDensity,
                     qcx::io::MethodType method,
                     bool converged) noexcept :
        _totalDensity(totalDensity), _alphaDensity(alphaDensity), _betaDensity(betaDensity),
        _method(method), _converged(converged) {}

    /// The total density P_alpha + P_beta.
    /// \returns A reference to the driver-owned matrix.
    const Eigen::MatrixXd& TotalDensity() const noexcept {
        return _totalDensity.get();
    }

    /// The alpha spin density.
    /// \returns A reference to the driver-owned matrix.
    /// \pre HasSpinDensities() is true.
    const Eigen::MatrixXd& AlphaDensity() const noexcept {
        return _alphaDensity->get();
    }

    /// The beta spin density.
    /// \returns A reference to the driver-owned matrix.
    /// \pre HasSpinDensities() is true.
    const Eigen::MatrixXd& BetaDensity() const noexcept {
        return _betaDensity->get();
    }

    /// The method kind the driver ran.
    /// \returns One of the four qcx::io::MethodType enumerators.
    qcx::io::MethodType Method() const noexcept {
        return _method;
    }

    /// The convergence state the driver holds.
    /// \returns True when the SCF reported convergence; false means the
    /// densities are the last iterate and the view states it rather than
    /// hiding it.
    bool Converged() const noexcept {
        return _converged;
    }

    /// Whether the run is an unrestricted kind (UHF or UKS).
    /// \returns True for qcx::io::MethodType::kUhf and
    /// qcx::io::MethodType::kUks.
    bool IsUnrestricted() const noexcept {
        return _method == qcx::io::MethodType::kUhf || _method == qcx::io::MethodType::kUks;
    }

    /// Whether the alpha and beta spin densities are available.
    /// \returns True when the view was built by the unrestricted
    /// constructor.
    bool HasSpinDensities() const noexcept {
        return _alphaDensity.has_value();
    }

private:
    std::reference_wrapper<const Eigen::MatrixXd> _totalDensity;
    std::optional<std::reference_wrapper<const Eigen::MatrixXd>> _alphaDensity;
    std::optional<std::reference_wrapper<const Eigen::MatrixXd>> _betaDensity;
    qcx::io::MethodType _method;
    bool _converged;
};

/// The view is a reference aggregate: no matrix is copied, so copying it is a
/// pointer copy and a property takes it by value for free.
static_assert(std::is_trivially_copyable_v<WavefunctionView>);

} // namespace qcx::properties
