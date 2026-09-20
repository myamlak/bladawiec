// The general-l dense ERI tensor (eri_dense.hpp): the screening caller.
// Canonical quartets are evaluated through the fp64 batch pipeline and
// mirrored into their 8-fold partners bit-exactly (pure index
// transpositions - no arithmetic, so the mirroring is exact).

#include "qcx/integrals/eri_dense.hpp"

#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace qcx::integrals {
namespace {

using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;

// The admission-gate arithmetic: products saturate at size_t max (an exact
// refusal bound - anything that saturates cannot fit any cap), so the gate
// can never under-estimate through wraparound.
constexpr std::size_t SaturatingMul(std::size_t a, std::size_t b) noexcept {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
    {
        return std::numeric_limits<std::size_t>::max();
    }

    return a * b;
}

// The saturating add twin (the gate sums three terms).
constexpr std::size_t SaturatingAdd(std::size_t a, std::size_t b) noexcept {
    return b > std::numeric_limits<std::size_t>::max() - a ? std::numeric_limits<std::size_t>::max()
                                                           : a + b;
}

// The admission-gate refusal: the dense tensor is a reference path, so an
// unaffordable build refuses gracefully (never a mid-enumeration
// allocation death) and points at the ladder - the direct screened builder
// is the scale path, or raise maxTensorBytes to admit deliberately.
std::string DenseTensorRefusal(std::size_t estimateBytes, std::size_t capBytes) {
    return "the dense ERI tensor build estimates " + std::to_string(estimateBytes) +
           " bytes (the canonical quartet list + the batch payload + the 8 n^4 tensor) "
           "against the maxTensorBytes cap of " +
           std::to_string(capBytes) +
           " bytes. The dense tensor is a reference path: raise maxTensorBytes to admit "
           "this build deliberately, or run the direct screened builder (the scale path).";
}

} // namespace

std::size_t DenseEriTensorEstimateBytes(std::size_t nBasis, std::size_t nPairs) noexcept {
    const std::size_t n4 =
        SaturatingMul(SaturatingMul(nBasis, nBasis), SaturatingMul(nBasis, nBasis));
    const std::size_t worstListBytes =
        SaturatingMul(SaturatingMul(nPairs, nPairs + 1) / 2, sizeof(ShellQuartet));
    const std::size_t worstBatchBytes = SaturatingMul(n4, 8);
    const std::size_t tensorBytes = SaturatingMul(n4, 8);
    return SaturatingAdd(SaturatingAdd(worstListBytes, worstBatchBytes), tensorBytes);
}

qcx::Result<CpuTensor4> BuildEriTensorGeneral(const qcx::molecule::Molecule& molecule,
                                              const qcx::basisset::BasisSet& basisSet,
                                              const EriDenseOptions& options) {
    if (options.maxBatchBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxBatchBytes must be positive"});
    }

    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    for (const ShellInfo& shell : pairList->shells)
    {
        if (!SupportsL(shell.angularMomentum))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "shell angular momentum exceeds kMaxEngineL of this build"});
        }
    }

    // The admission gate, a priori (all-survive, before anything is
    // enumerated): the shared estimate (DenseEriTensorEstimateBytes - the
    // canonical quartet list alone is nPairs(nPairs+1)/2 ShellQuartet
    // entries (32 B each - 16.3 GiB at 45,150 pairs), the batch payload
    // at most 8 n^4 B and the tensor 8 n^4 B). Estimating before the
    // enumeration keeps the list itself from being the death allocation.
    const std::size_t nPairs = pairList->pairs.size();
    const std::size_t n = pairList->functionCount;
    const std::size_t worstCaseBytes = DenseEriTensorEstimateBytes(n, nPairs);

    if (worstCaseBytes > options.maxTensorBytes)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       DenseTensorRefusal(worstCaseBytes, options.maxTensorBytes)});
    }

    std::vector<double> schwarz;

    if (options.screen)
    {
        auto bounds = ComputeSchwarzBounds(molecule, basisSet);

        if (!bounds.has_value())
        {
            return std::unexpected(bounds.error());
        }

        schwarz = std::move(*bounds);
    }

    // The canonical quartet list: pair (i,j) >= pair (k,l), Schwarz-screened
    // when enabled. Every tensor slot is covered by exactly one canonical
    // quartet's 8-fold orbit.
    std::vector<ShellQuartet> quartets;

    for (std::size_t bra = 0; bra < nPairs; ++bra)
    {
        for (std::size_t ket = 0; ket <= bra; ++ket)
        {
            if (options.screen && schwarz[bra] * schwarz[ket] < SchwarzThreshold(options.accuracy))
            {
                continue;
            }

            quartets.push_back(ShellQuartet{pairList->pairs[bra].i,
                                            pairList->pairs[bra].j,
                                            pairList->pairs[ket].i,
                                            pairList->pairs[ket].j});
        }
    }

    EriBatchOptions batchOptions;
    batchOptions.accuracy = options.accuracy;
    batchOptions.maxBatchBytes = options.maxBatchBytes;
    auto batch = ComputeEriBatch(molecule, basisSet, quartets, batchOptions);

    if (!batch.has_value())
    {
        return std::unexpected(batch.error());
    }

    auto tensor = CpuTensor4::Create({n, n, n, n});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    std::vector<std::size_t> shellCount(pairList->shells.size());

    for (std::size_t s = 0; s < pairList->shells.size(); ++s)
    {
        shellCount[s] = ShellFunctionCount(pairList->shells[s]);
    }

    // The 8-fold mirror: one canonical block writes all eight partners by
    // index transposition (bit-exact - no arithmetic).
    std::size_t base = 0;

    for (const ShellQuartet& quartet : batch->computed)
    {
        const std::size_t nI = shellCount[quartet.i];
        const std::size_t nJ = shellCount[quartet.j];
        const std::size_t nK = shellCount[quartet.k];
        const std::size_t nL = shellCount[quartet.l];
        const std::size_t oI = pairList->shells[quartet.i].functionOffset;
        const std::size_t oJ = pairList->shells[quartet.j].functionOffset;
        const std::size_t oK = pairList->shells[quartet.k].functionOffset;
        const std::size_t oL = pairList->shells[quartet.l].functionOffset;

        for (std::size_t fi = 0; fi < nI; ++fi)
        {
            for (std::size_t fj = 0; fj < nJ; ++fj)
            {
                for (std::size_t fk = 0; fk < nK; ++fk)
                {
                    for (std::size_t fl = 0; fl < nL; ++fl)
                    {
                        const double value =
                            batch->values[base + EriBlockIndex(fi, fj, fk, fl, nI, nJ, nK, nL)];
                        (*tensor)(oI + fi, oJ + fj, oK + fk, oL + fl) = value;
                        (*tensor)(oJ + fj, oI + fi, oK + fk, oL + fl) = value;
                        (*tensor)(oI + fi, oJ + fj, oL + fl, oK + fk) = value;
                        (*tensor)(oJ + fj, oI + fi, oL + fl, oK + fk) = value;
                        (*tensor)(oK + fk, oL + fl, oI + fi, oJ + fj) = value;
                        (*tensor)(oL + fl, oK + fk, oI + fi, oJ + fj) = value;
                        (*tensor)(oK + fk, oL + fl, oJ + fj, oI + fi) = value;
                        (*tensor)(oL + fl, oK + fk, oJ + fj, oI + fi) = value;
                    }
                }
            }
        }

        base += nI * nJ * nK * nL;
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

} // namespace qcx::integrals
