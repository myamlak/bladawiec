// Two-electron repulsion integrals: the dense (uv|ws) tensor over s
// functions. The s-s-s-s primitive integral is the closed-form
// Gaussian-product expression (Helgaker2000) evaluated through BoysSingle
// (Helgaker2000); no screening and no symmetry compression yet.
#include "qcx/integrals/two_electron.hpp"

#include "internal/shells_flat.hpp"

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace qcx::integrals {
namespace {

using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;

// The s-s-s-s primitive sum of quartet (i, j, k, l): the closed-form
// Gaussian-product expression (Helgaker2000)
//   2 pi^(5/2) / (p q sqrt(p+q)) exp(-ab/p R_AB^2 - cd/q R_CD^2) F0(alpha R_PQ^2)
// accumulated over all primitive quadruples.
double PrimitiveEriSum(double twoPiFiveHalves,
                       const std::vector<Eigen::Vector3d>& positions,
                       const std::vector<internal::SFunctionData>& functions,
                       std::size_t i,
                       std::size_t j,
                       std::size_t k,
                       std::size_t l) {
    const Eigen::Vector3d& centerI = positions[functions[i].atomIndex];
    const Eigen::Vector3d& centerJ = positions[functions[j].atomIndex];
    const Eigen::Vector3d& centerK = positions[functions[k].atomIndex];
    const Eigen::Vector3d& centerL = positions[functions[l].atomIndex];
    double value = 0.0;

    for (const internal::PrimitiveData& primitiveA : functions[i].primitives)
    {
        for (const internal::PrimitiveData& primitiveB : functions[j].primitives)
        {
            const double p = primitiveA.exponent + primitiveB.exponent;
            const double abOverP = primitiveA.exponent * primitiveB.exponent / p;
            const double rAb2 = (centerI - centerJ).squaredNorm();
            const Eigen::Vector3d centerP =
                (primitiveA.exponent * centerI + primitiveB.exponent * centerJ) / p;

            for (const internal::PrimitiveData& primitiveC : functions[k].primitives)
            {
                for (const internal::PrimitiveData& primitiveD : functions[l].primitives)
                {
                    const double q = primitiveC.exponent + primitiveD.exponent;
                    const double alpha = p * q / (p + q);
                    const double cdOverQ = primitiveC.exponent * primitiveD.exponent / q;
                    const double rCd2 = (centerK - centerL).squaredNorm();
                    const Eigen::Vector3d centerQ =
                        (primitiveC.exponent * centerK + primitiveD.exponent * centerL) / q;
                    const double rPq2 = (centerP - centerQ).squaredNorm();
                    value +=
                        primitiveA.coefficient * primitiveB.coefficient * primitiveC.coefficient *
                        primitiveD.coefficient * (twoPiFiveHalves / (p * q * std::sqrt(p + q))) *
                        std::exp(-abOverP * rAb2 - cdOverQ * rCd2) * BoysSingle(0, alpha * rPq2);
                }
            }
        }
    }

    return value;
}

} // namespace

qcx::Result<CpuTensor4> BuildEriTensor(const qcx::molecule::Molecule& molecule,
                                       const qcx::basisset::BasisSet& basisSet,
                                       std::size_t maxTensorBytes) {
    auto functions = internal::BuildSFunctionData(molecule, basisSet);

    if (!functions.has_value())
    {
        return std::unexpected(functions.error());
    }

    const std::size_t n = functions->size();
    const std::vector<Eigen::Vector3d> positions = internal::AtomPositions(molecule);

    // The admission gate: the tensor is 8 n^4 B (64.8 GB at n = 300), a
    // reference-only mass - refuse gracefully before the allocation instead
    // of dying on it (the s-only build is the test/reference path; the
    // direct screened builder is the scale path). The n^4 product saturates
    // at size_t max, so the check can never under-estimate through
    // wraparound.
    const std::size_t n2 = n > std::numeric_limits<std::size_t>::max() / n
                               ? std::numeric_limits<std::size_t>::max()
                               : n * n;

    const std::size_t n4 = n2 > std::numeric_limits<std::size_t>::max() / n2
                               ? std::numeric_limits<std::size_t>::max()
                               : n2 * n2;
    const std::size_t tensorBytes = n4 > std::numeric_limits<std::size_t>::max() / 8
                                        ? std::numeric_limits<std::size_t>::max()
                                        : n4 * 8;

    if (tensorBytes > maxTensorBytes)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "the dense s-only ERI tensor build estimates " + std::to_string(tensorBytes) +
                " bytes (8 n^4) against the maxTensorBytes cap of " +
                std::to_string(maxTensorBytes) +
                " bytes. The dense tensor is a reference path: raise maxTensorBytes to admit "
                "this build deliberately, or run the direct screened builder (the scale path)."});
    }

    auto tensor = CpuTensor4::Create({n, n, n, n});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    // 2 pi^(5/2), the s-s-s-s prefactor constant (Helgaker2000).
    const double twoPiFiveHalves = 2.0 * std::pow(std::numbers::pi, 2.5);

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            for (std::size_t k = 0; k < n; ++k)
            {
                for (std::size_t l = 0; l < n; ++l)
                {
                    (*tensor)(i, j, k, l) =
                        PrimitiveEriSum(twoPiFiveHalves, positions, *functions, i, j, k, l);
                }
            }
        }
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

} // namespace qcx::integrals
