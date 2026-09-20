// The batched ERI entry points (eri_batch.hpp): pair flattening, pair-data
// construction, class-batch assembly, and the dispatch runs over the
// per-class kernels. The engine performs no screening internally - the
// caller supplies the quartet list.

#include "qcx/integrals/eri_batch.hpp"

#include "internal/md_batch.hpp"
#include "internal/md_engine.hpp"
#include "qcx/integrals/limits.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace qcx::integrals {
namespace {

// Every shell of the pair list must be covered by this build's class matrix.
qcx::Result<void> ValidateL(const ShellPairList& pairList) {
    for (const ShellInfo& shell : pairList.shells)
    {
        if (!SupportsL(shell.angularMomentum))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "shell angular momentum exceeds kMaxEngineL of this build"});
        }
    }

    return {};
}

} // namespace

qcx::Result<EriBatch> ComputeEriBatch(const qcx::molecule::Molecule& molecule,
                                      const qcx::basisset::BasisSet& basisSet,
                                      const std::vector<ShellQuartet>& quartets,
                                      const EriBatchOptions& options) {
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

    auto lValidation = ValidateL(*pairList);

    if (!lValidation.has_value())
    {
        return std::unexpected(lValidation.error());
    }

    auto pairStore = internal::BuildPairData(molecule, basisSet, *pairList);

    if (!pairStore.has_value())
    {
        return std::unexpected(pairStore.error());
    }

    std::vector<ShellQuartet> computed;
    auto batches = internal::AssembleClassBatches(
        *pairStore, *pairList, quartets, options.maxBatchBytes, computed);

    if (!batches.has_value())
    {
        return std::unexpected(batches.error());
    }

    std::size_t total = 0;

    for (internal::MdClassBatch& batch : *batches)
    {
        for (const internal::MdQuartetTask& task : batch.tasks)
        {
            total += (*pairStore)[task.braPair].nFuncs * (*pairStore)[task.ketPair].nFuncs;
        }
    }

    EriBatch result;
    result.computed = std::move(computed);
    result.values.resize(total);

    std::size_t base = 0;

    for (internal::MdClassBatch& batch : *batches)
    {
        batch.outF64 = result.values.data() + base;

        for (const internal::MdQuartetTask& task : batch.tasks)
        {
            base += (*pairStore)[task.braPair].nFuncs * (*pairStore)[task.ketPair].nFuncs;
        }
    }

    auto run = internal::RunBatches(*batches);

    if (!run.has_value())
    {
        return std::unexpected(run.error());
    }

    return result;
}

qcx::Result<CertifiedBatch> ComputeEriBatchCertified(const qcx::molecule::Molecule& molecule,
                                                     const qcx::basisset::BasisSet& basisSet,
                                                     const std::vector<ShellQuartet>& quartets,
                                                     const EriBatchOptions& options) {
#if !QcxIntegralsF32
    return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                      "this build has no fp32 pipeline (QCX_INTEGRALS_F32=OFF)"});
#else
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

    auto lValidation = ValidateL(*pairList);

    if (!lValidation.has_value())
    {
        return std::unexpected(lValidation.error());
    }

    auto pairStore = internal::BuildPairData(molecule, basisSet, *pairList);

    if (!pairStore.has_value())
    {
        return std::unexpected(pairStore.error());
    }

    std::vector<ShellQuartet> computed;
    auto batches = internal::AssembleClassBatches(
        *pairStore, *pairList, quartets, options.maxBatchBytes, computed);

    if (!batches.has_value())
    {
        return std::unexpected(batches.error());
    }

    std::size_t total = 0;

    for (internal::MdClassBatch& batch : *batches)
    {
        for (const internal::MdQuartetTask& task : batch.tasks)
        {
            total += (*pairStore)[task.braPair].nFuncs * (*pairStore)[task.ketPair].nFuncs;
        }
    }

    CertifiedBatch result;
    result.computed = std::move(computed);
    result.values.resize(total);
    result.errorBounds.assign(result.computed.size(), 0.0);

    std::size_t base = 0;

    for (internal::MdClassBatch& batch : *batches)
    {
        batch.outF32 = result.values.data() + base;
        batch.errorBounds = result.errorBounds.data();

        for (const internal::MdQuartetTask& task : batch.tasks)
        {
            base += (*pairStore)[task.braPair].nFuncs * (*pairStore)[task.ketPair].nFuncs;
        }
    }

    auto run = internal::RunBatches(*batches);

    if (!run.has_value())
    {
        return std::unexpected(run.error());
    }

    return result;
#endif
}

} // namespace qcx::integrals
