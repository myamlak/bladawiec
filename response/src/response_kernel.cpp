#include "qcx/response/response_kernel.hpp"

#include <utility>

namespace qcx::response {

KohnShamOrbitalHessian::KohnShamOrbitalHessian(OrbitalHessianOperator base,
                                               std::string kernelName) :
    _base(std::move(base)), _kernelName(std::move(kernelName)) {}

qcx::Result<KohnShamOrbitalHessian> KohnShamOrbitalHessian::Create(
    ResponseLayout layout,
    OrbitalEnergies orbitalEnergies,
    MoTwoElectronTensor moTwoElectron,
    double exchangeFraction,
    KohnShamKernelTerm kernel) {
    if (kernel.name.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the kernel term must carry a name"});
    }

    if (kernel.apply == nullptr)
    {
        // A missing tier is a refusal at the surface that builds the kernel;
        // silently dropping the term here would solve against the wrong
        // operator.
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the kernel term must carry an action"});
    }

    auto base = OrbitalHessianOperator::Create(
        layout, orbitalEnergies, moTwoElectron, exchangeFraction, std::move(kernel.apply));

    if (!base.has_value())
    {
        return std::unexpected(base.error());
    }

    return KohnShamOrbitalHessian(std::move(*base), std::move(kernel.name));
}

} // namespace qcx::response
