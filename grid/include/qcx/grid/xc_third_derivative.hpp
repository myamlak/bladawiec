#pragma once

#include "qcx/error.hpp"

#include <excgrid/kernel.hpp>
#include <span>

namespace qcx::grid {

/// Evaluates one point through excgrid's third-derivative seam.
///
/// No third-derivative tier is built yet: every shipped functional inherits
/// the seam's default refusal, and this tree has no contraction arithmetic to
/// contribute. The path exists so a caller reaches the seam through qcx's error
/// type today and needs no change when a tier is shipped; a refusal is an
/// error, never a zero-filled contraction. The layout of `contracted` is the
/// tier's, defined with the tier.
/// \param functional The functional kernel. It must outlive the call.
/// \param inputs The point's components and the mask naming them.
/// \param rightHandSide The vector to contract with, by active component.
/// \param result Filled on success; untouched otherwise.
/// \param contracted Filled on success; untouched otherwise.
/// \returns Success when the seam answers kOk; kUnimplemented carrying the
/// refusal's description otherwise.
/// \ingroup qcx-grid
qcx::Result<void> EvaluateXcKernelThirdDerivativeContraction(
    const excgrid::XcFunctional& functional,
    const excgrid::PointInputs& inputs,
    std::span<const double> rightHandSide,
    excgrid::PointResult& result,
    std::span<double> contracted);

} // namespace qcx::grid
