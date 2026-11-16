#include "qcx/grid/xc_third_derivative.hpp"

#include <expected>
#include <string>

namespace qcx::grid {

qcx::Result<void> EvaluateXcKernelThirdDerivativeContraction(
    const excgrid::XcFunctional& functional,
    const excgrid::PointInputs& inputs,
    std::span<const double> rightHandSide,
    excgrid::PointResult& result,
    std::span<double> contracted) {
    const excgrid::KernelStatus status =
        functional.EvaluatePointWithThirdDerivatives(inputs, rightHandSide, result, contracted);

    if (status != excgrid::KernelStatus::kOk)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       std::string("EvaluateXcKernelThirdDerivativeContraction: the functional ") +
                           std::string(excgrid::DescribeStatus(status))});
    }

    return {};
}

} // namespace qcx::grid
