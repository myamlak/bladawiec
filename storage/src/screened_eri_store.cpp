// The Create-time screened pre-warm of a full-ERI store
// (screened_eri_store.hpp carries the design note): the Schwarz admission
// sweep, the canonicalization into the store's own order, and the class-run
// append loop.

#include "qcx/storage/screened_eri_store.hpp"

#include "qcx/integrals/screening.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace qcx::storage {

namespace {

using qcx::integrals::CanonicalQuartetInfo;
using qcx::integrals::ShellPairIndex;
using qcx::integrals::ShellPairList;
using qcx::integrals::ShellQuartet;

// The block element count of one quartet - the store's own rule
// (eri_store.cpp's BlockElementCount: nI*nJ*nK*nL over the shared
// ShellFunctionCount definition, the single definition every engine reads).
std::size_t BlockElementCount(const ShellPairList& pairList, const ShellQuartet& quartet) {
    return qcx::integrals::ShellFunctionCount(pairList.shells[quartet.i]) *
           qcx::integrals::ShellFunctionCount(pairList.shells[quartet.j]) *
           qcx::integrals::ShellFunctionCount(pairList.shells[quartet.k]) *
           qcx::integrals::ShellFunctionCount(pairList.shells[quartet.l]);
}

// The unreduced 4-index payload of one basis function count (8 n^4).
std::size_t DenseEriBytes(std::size_t functionCount) {
    return 8 * functionCount * functionCount * functionCount * functionCount;
}

} // namespace

qcx::Result<ScreenedEriStorePlan> PlanScreenedEriStore(const qcx::molecule::Molecule& molecule,
                                                       const qcx::basisset::BasisSet& basisSet,
                                                       qcx::integrals::AccuracyPreset accuracy) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto bounds = qcx::integrals::ComputeSchwarzBounds(molecule, basisSet);

    if (!bounds.has_value())
    {
        return std::unexpected(bounds.error());
    }

    const std::vector<ShellPairIndex>& pairs = pairList->pairs;
    const std::vector<double>& schwarz = *bounds;
    const double threshold = qcx::integrals::SchwarzThreshold(accuracy);

    ScreenedEriStorePlan plan;
    plan.denseBytes = DenseEriBytes(pairList->functionCount);

    // The canonical pair-pair half: bra pair index >= ket pair index IS the
    // (i,j) >= (k,l) rule of the 8-fold canonical form, so each unordered
    // pair-pair is admitted once, already correctly oriented - the
    // canonicalizer re-sorts and re-checks it rather than trusting this.
    std::vector<ShellQuartet> admitted;

    for (std::size_t bra = 0; bra < pairs.size(); ++bra)
    {
        for (std::size_t ket = 0; ket <= bra; ++ket)
        {
            if (schwarz[bra] * schwarz[ket] < threshold)
            {
                continue;
            }

            admitted.push_back({pairs[bra].i, pairs[bra].j, pairs[ket].i, pairs[ket].j});
        }
    }

    // An empty admission is a legitimate outcome - a coarse preset over a
    // small system admits nothing - and never an error: the canonicalizer
    // refuses an empty request, so the empty plan returns here. A store with
    // no pre-warmed quartet serves no hits, which degrades the hit rate and
    // changes no answer (the pre-warm is a cache fill; see the header).
    if (admitted.empty())
    {
        return plan;
    }

    auto canonical = qcx::integrals::CanonicalizeQuartetOrder(*pairList, admitted);

    if (!canonical.has_value())
    {
        return std::unexpected(canonical.error());
    }

    plan.quartets = std::move(*canonical);

    for (const CanonicalQuartetInfo& info : plan.quartets)
    {
        plan.payloadBytes += 8 * BlockElementCount(*pairList, info.quartet);
    }

    return plan;
}

qcx::Result<std::size_t> FillScreenedEriStore(const std::filesystem::path& storePath,
                                              const qcx::molecule::Molecule& molecule,
                                              const qcx::basisset::BasisSet& basisSet,
                                              std::string_view orbitalBasisName,
                                              const ScreenedEriStorePlan& plan,
                                              qcx::integrals::AccuracyPreset accuracy,
                                              std::size_t maxBatchBytes) {
    if (maxBatchBytes == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "FillScreenedEriStore: maxBatchBytes must be positive"});
    }

    // The plan carries shell indices, not function counts: the pair list is
    // rebuilt to size each run's bytes. It is the same construction the
    // planner ran, so a plan from PlanScreenedEriStore sized this way is
    // exact. It is built BEFORE the store so a degenerate system leaves no
    // half-created file behind.
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    StoreOptions storeOptions;
    storeOptions.accuracy = accuracy;
    storeOptions.provenance = "qcx-screened-full-eri-prewarm";

    auto store =
        EriStore::Create(storePath, molecule, basisSet, orbitalBasisName, "", storeOptions);

    if (!store.has_value())
    {
        return std::unexpected(store.error());
    }

    std::size_t written = 0;
    std::size_t begin = 0;
    const std::size_t count = plan.quartets.size();

    while (begin < count)
    {
        // One class run: a maximal stretch of consecutive equal (lBra, lKet)
        // in the canonical order (the canonicalizer's sort is class-major
        // over (lKet, lBra), so equal classes ARE contiguous), cut again at
        // the byte cap. The cap never empties a run - a single block over it
        // is the engine's to refuse by name, never a silent skip.
        const CanonicalQuartetInfo& head = plan.quartets[begin];
        std::size_t end = begin;
        std::size_t runBytes = 0;

        while (end < count && plan.quartets[end].lBra == head.lBra &&
               plan.quartets[end].lKet == head.lKet)
        {
            const std::size_t blockBytes =
                8 * BlockElementCount(*pairList, plan.quartets[end].quartet);

            if (end > begin && runBytes + blockBytes > maxBatchBytes)
            {
                break;
            }

            runBytes += blockBytes;
            ++end;
        }

        std::vector<ShellQuartet> request;
        request.reserve(end - begin);

        for (std::size_t index = begin; index < end; ++index)
        {
            request.push_back(plan.quartets[index].quartet);
        }

        qcx::integrals::EriBatchOptions batchOptions;
        batchOptions.accuracy = accuracy;
        batchOptions.maxBatchBytes = maxBatchBytes;

        auto batch = qcx::integrals::ComputeEriBatch(molecule, basisSet, request, batchOptions);

        if (!batch.has_value())
        {
            return std::unexpected(batch.error());
        }

        // The engine's computed order is the store's canonical order, so the
        // run appends as one chunk whose manifest class is the run's own.
        auto appended = store->AppendBatch(batch->computed, batch->values);

        if (!appended.has_value())
        {
            return std::unexpected(appended.error());
        }

        written += batch->values.size() * sizeof(double);
        begin = end;
    }

    return written;
}

} // namespace qcx::storage
