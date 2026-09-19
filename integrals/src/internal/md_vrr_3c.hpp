#pragma once

// The 3-center RI engine: the
// orbital-bra x aux-ket integrals (uv|P) through the same matrix-form MD
// kernels as the 2e engine (md_vrr.hpp - the VRR, the transform GEMMs, and
// the per-class dispatch are shared verbatim). The 3c-specific pieces are
// the aux pair data and the canonicalization-free assembly:
//
//   - The aux shell P is the ket "pair" of a single shell, built as the
//     phantom pair (P, s_0): P crossed with a unit s phantom at the same
//     center - the same construction the (P|Q) metric uses
//     (ri_engine.cpp). The phantom turns the pair into the aux function
//     phi_P itself (p = zeta, prefactor E_cd = exp(-a*0/p * |C-C|^2) = 1)
//     and the full E-table fold with the ket-derivative sign rides the
//     contracted ket transform T_P(q~, fP) like any 2e ket pair.
//   - The class dispatch table already instantiates every (L_bra, L_ket)
//     pair, so the aux shell's angular momentum is just the ket class -
//     no new kernels. Bra and aux pairs share one pair store (the kernels
//     index both sides of a task through batch.pairStore).
//   - No 8-fold symmetry in (uv|P): the assembly is canonicalization-free;
//     it only sorts into the ket-group order the kernel's pass 1 expects
//     and caps the batches like the 2e assembler.
//
// The fp32 certified lane is a 2e-engine feature; the RI path runs
// fp64 only in this step.

#include "md_attribution.hpp"
#include "md_batch.hpp"
#include "md_defs.hpp"
#include "md_engine.hpp"
#include "md_hermite.hpp"
#include "qcx/integrals/limits.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace qcx::integrals {

/// The term-counter block, owned by ri_engine.hpp: forward-declared
/// here because this header only passes a reference to it (the counting rule
/// below takes the sink, and adding the public header's include graph to
/// every 3c include site would buy nothing).
struct RiTermCounters;

/// The chunked build's cross-chunk dedup state (ri_engine.hpp
/// RiScreenedPassState), forward-declared for the same reason: the counting
/// rule below only passes a pointer to it.
struct RiScreenedPassState;

} // namespace qcx::integrals

namespace qcx::integrals::internal {

/// Builds the ket-side pair data of one aux shell as the phantom pair
/// (P, s_0): the shell P crossed with a unit s phantom at the same center -
/// the same construction the (P|Q) metric uses (ri_engine.cpp), so the
/// pair function is the aux function phi_P itself: p = zeta, prefactor
/// E_cd = exp(-a*0/p * |C-C|^2) = 1. The full E-table fold with the
/// ket-derivative sign rides the contracted ket transform (the
/// 2026-08-22 RI-J finding: the former "identity Hermite expansion" - a
/// bare c_rho * G-fold without E tables - was wrong for lP >= 1, because
/// with zero shifts the E recurrence still yields E(1, 0, 1) = 1/(2 zeta)
/// etc.; the missing fold scaled and sign-flipped the l >= 1 columns of I
/// by a (bra, aux)-dependent factor, the 2026-08-22 RI-J finding). The entry
/// is a valid MdPairData ket (la = lP, lb = 0); the bra-only fields of the
/// pair store stay empty - an aux shell is never a bra in the RI path.
inline MdPairData BuildAuxPairData(const MdShellContractions& aux,
                                   const std::array<double, 3>& center) {
    MdShellContractions phantom;
    phantom.angularMomentum = 0;
    phantom.isSpherical = false;
    phantom.rows = 1;
    phantom.exponents = {0.0};
    phantom.normalized = {{1.0}};

    MdPairData pair;
    // BuildContractedPairTransform sets the row/vector fields only: the
    // pair class and the two centers are the caller's (BuildPairStore sets
    // all eight around its own call), so this entry declares them here.
    // Leaving them to the transform left two bools and six centers
    // indeterminate in the returned pair. The aux shell is the bra side of
    // this phantom pair and the ket class is la = lP, lb = 0.
    pair.la = aux.angularMomentum;
    pair.lb = 0;
    pair.isSphericalA = aux.isSpherical;
    pair.isSphericalB = false;
    pair.ax = center[0];
    pair.ay = center[1];
    pair.az = center[2];
    pair.bx = center[0];
    pair.by = center[1];
    pair.bz = center[2];
    BuildContractedPairTransform(pair, aux, phantom, center, center);
    return pair;
}

/// One 3c task: an orbital pair crossed with an aux shell.
struct RiTask {
    std::size_t braPair; ///< Index into the orbital pair store.
    std::size_t auxShell; ///< Index into the aux pair store (appended after the orbital pairs).
};

/// The assembled 3c batches of one task list: the tasks in final output
/// order, the per-class batches with their packed output sizes. Pure
/// assembly - no kernel work, no output allocation; the batch output
/// pointers stay unset. RunRiBatches allocates the full packed buffer on
/// top of the assembly; the light rung (ri_engine.cpp) consumes the
/// batches one at a time against a single maxBatchBytes-class buffer
/// instead (the per-iteration two-pass recompute).
struct RiBatchAssembly {
    /// The tasks in final output order (the batches' order). The
    /// tagged allocator: the assembly is the light rung's per-iteration
    /// recompute slice (light_rung_slice) and the fast path's transient
    /// Create-time assembly (rij_fast_tensor) - the tag is the scope active
    /// at the AssembleRiBatches call.
    TaggedVector<RiTask> computed;
    /// The per-class batches (outF64/outF32 null); plain - the batches flow
    /// into RunBatches (md_batch.cpp), whose container type is not tagged.
    std::vector<MdClassBatch> batches;
    std::size_t totalOutput = 0; ///< The packed element count of every task.
    std::size_t maxBatchOutput = 0; ///< The largest batch's packed element count.
};

/// The TENSOR-PASS occurrence's x/p3/g3 accumulation over one
/// kernel evaluation of a screened 3c task list - the rule the monolithic
/// BuildRiTensor runs (ri_engine.cpp) and the chunked build shares through
/// this declaration (ri_engine_chunk.cpp BuildRiTensorChunk), so both report
/// the same numbers for the same tasks instead of each carrying its own copy.
/// A chunked caller partitions ONE such evaluation, so it passes the
/// build's cross-chunk dedup state and the orbital pair stamps span the
/// calls; a null state keeps the standalone per-call stamps (one call = one
/// whole evaluation).
/// Defined in ri_engine.cpp.
/// \param into The counter block the partial is added to.
/// \param combinedStore The task list's pair store: the first
/// \p nOrbitalPairs entries are the orbital pairs, the rest the aux pairs.
/// \param nOrbitalPairs The orbital pair count (the store's first block).
/// \param tasks The evaluated tasks.
/// \param passState The caller's cross-chunk dedup state (ri_engine.hpp
/// RiScreenedPassState), shared by every chunk call of one chunked build -
/// only its orbital pair stamps are read and written here. Null (the
/// default) allocates the stamps for this call alone.
void AccumulateScreenedRiPass(RiTermCounters& into,
                              const std::vector<MdPairData>& combinedStore,
                              std::size_t nOrbitalPairs,
                              std::span<const RiTask> tasks,
                              RiScreenedPassState* passState = nullptr);

/// Assembles 3c tasks into per-class batches (the RunRiBatches front
/// half): the index validation, the class-major/ket-group sort, the batch
/// capping and the packed-size sums. \p combinedStore holds the orbital
/// pairs (the first \p orbitalPairCount entries) followed by the aux
/// pairs - the kernels index both sides of a task through
/// batch.pairStore.
inline qcx::Result<RiBatchAssembly> AssembleRiBatches(const std::vector<MdPairData>& combinedStore,
                                                      std::size_t orbitalPairCount,
                                                      const std::vector<RiTask>& tasks,
                                                      std::size_t maxBatchBytes) {
    if (maxBatchBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxBatchBytes must be positive"});
    }

    if (tasks.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the task list is empty"});
    }

    const std::size_t nOrbitalPairs = orbitalPairCount;

    struct Item {
        RiTask task;
        int lBra;
        int lKet;
        std::size_t braRowPairs;
        std::size_t outputSize;
        std::size_t convertSize; // The kernel's per-batch convert workspace.
        std::size_t scratchSize;
    };

    TaggedVector<Item> items;
    items.reserve(tasks.size());

    for (const RiTask& task : tasks)
    {
        if (task.braPair >= nOrbitalPairs || task.auxShell >= combinedStore.size() ||
            task.auxShell < nOrbitalPairs)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "RI task index out of range"});
        }

        const MdPairData& bra = combinedStore[task.braPair];
        const MdPairData& aux = combinedStore[task.auxShell];
        const int lBra = bra.la + bra.lb;
        const int lKet = aux.la; // la = lP, lb = 0.
        const std::size_t hermBra = static_cast<std::size_t>(Hermite3DCount(lBra));
        const std::size_t hermKet = static_cast<std::size_t>(Hermite3DCount(lKet));
        const std::size_t mPq = bra.rowPairs * bra.primPairs.size() * hermBra;
        items.push_back(Item{task,
                             lBra,
                             lKet,
                             bra.rowPairs,
                             bra.nFuncs * aux.nFuncs,
                             // The kernel's convert workspace, max over its
                             // two uses (md_vrr.hpp): the ket-transform
                             // copy (hermKet x nFuncsKet) and the
                             // bra-transform copy (mBra x kBra =
                             // bra.nFuncs x mPq).
                             std::max(hermKet * aux.nFuncs, bra.nFuncs * mPq),
                             mPq * hermKet + mPq * aux.nFuncs});
    }

    // Class-major, then the ket-group order of the kernel's pass 1: tasks
    // with the same aux shell and bra row-pair count contiguous.
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        if (a.lKet != b.lKet)
        {
            return a.lKet < b.lKet;
        }

        if (a.lBra != b.lBra)
        {
            return a.lBra < b.lBra;
        }

        if (a.task.auxShell != b.task.auxShell)
        {
            return a.task.auxShell < b.task.auxShell;
        }

        if (a.braRowPairs != b.braRowPairs)
        {
            return a.braRowPairs < b.braRowPairs;
        }

        return a.task.braPair < b.task.braPair;
    });

    TaggedVector<RiTask> computed;
    computed.reserve(items.size());

    for (const Item& item : items)
    {
        computed.push_back(item.task);
    }

    // Plain: the batches flow into RunBatches (md_batch.cpp), whose
    // container type is not tagged.
    std::vector<MdClassBatch> batches;
    std::size_t runStart = 0;

    while (runStart < items.size())
    {
        std::size_t runEnd = runStart + 1;

        while (runEnd < items.size() && items[runEnd].lBra == items[runStart].lBra &&
               items[runEnd].lKet == items[runStart].lKet)
        {
            ++runEnd;
        }

        std::size_t batchStart = runStart;
        std::size_t batchBytes = 0;

        for (std::size_t t = runStart; t < runEnd; ++t)
        {
            // The kernel allocates the convert workspace once per batch,
            // sized by the max over the batch's tasks - folding each item's
            // convertSize in over-counts (sum >= max), deliberately
            // conservative so the cap bounds the whole per-batch allocation.
            const std::size_t itemBytes =
                (items[t].outputSize + items[t].scratchSize + items[t].convertSize) *
                sizeof(double);

            if (batchBytes + itemBytes > maxBatchBytes && t > batchStart)
            {
                MdClassBatch batch;
                batch.lBra = items[batchStart].lBra;
                batch.lKet = items[batchStart].lKet;
                batch.pairStore = &combinedStore;
                batch.boundsBase = batchStart;
                std::size_t outputOffset = 0;

                for (std::size_t s = batchStart; s < t; ++s)
                {
                    batch.tasks.push_back(
                        MdQuartetTask{items[s].task.braPair, items[s].task.auxShell, outputOffset});
                    outputOffset += items[s].outputSize;
                }

                batches.push_back(std::move(batch));
                batchStart = t;
                batchBytes = 0;
            }

            batchBytes += itemBytes;
        }

        MdClassBatch batch;
        batch.lBra = items[batchStart].lBra;
        batch.lKet = items[batchStart].lKet;
        batch.pairStore = &combinedStore;
        batch.boundsBase = batchStart;
        std::size_t outputOffset = 0;

        for (std::size_t s = batchStart; s < runEnd; ++s)
        {
            batch.tasks.push_back(
                MdQuartetTask{items[s].task.braPair, items[s].task.auxShell, outputOffset});
            outputOffset += items[s].outputSize;
        }

        batches.push_back(std::move(batch));
        runStart = runEnd;
    }

    std::size_t total = 0;

    for (const Item& item : items)
    {
        total += item.outputSize;
    }

    std::size_t maxBatchOutput = 0;

    for (const MdClassBatch& batch : batches)
    {
        std::size_t batchOutput = 0;

        for (const MdQuartetTask& task : batch.tasks)
        {
            batchOutput += combinedStore[task.braPair].nFuncs * combinedStore[task.ketPair].nFuncs;
        }

        maxBatchOutput = std::max(maxBatchOutput, batchOutput);
    }

    return RiBatchAssembly{std::move(computed), std::move(batches), total, maxBatchOutput};
}

/// Assembles 3c tasks into per-class batches and runs them through the
/// shared dispatch. \p combinedStore holds the orbital pairs (the first
/// \p orbitalPairCount entries) followed by the aux pairs - the kernels
/// index both sides of a task through batch.pairStore. \p computed
/// receives the (braPair, auxShell) task list in final output order;
/// \p values receives the packed blocks (layout: bra.nFuncs x aux.nFuncs
/// per task, the eri_batch.hpp row order (f_b * n_i + f_a) on the bra
/// side, (row * nAng + fang) on the aux side).
inline qcx::Result<void> RunRiBatches(std::vector<MdPairData>& combinedStore,
                                      std::size_t orbitalPairCount,
                                      const std::vector<RiTask>& tasks,
                                      std::size_t maxBatchBytes,
                                      TaggedVector<RiTask>& computed,
                                      TaggedVector<double>& values) {
    auto assembly = AssembleRiBatches(combinedStore, orbitalPairCount, tasks, maxBatchBytes);

    if (!assembly.has_value())
    {
        return std::unexpected(assembly.error());
    }

    computed = std::move(assembly->computed);
    values.assign(assembly->totalOutput, 0.0);
    std::size_t base = 0;

    for (MdClassBatch& batch : assembly->batches)
    {
        batch.outF64 = values.data() + base;

        for (const MdQuartetTask& task : batch.tasks)
        {
            base += combinedStore[task.braPair].nFuncs * combinedStore[task.ketPair].nFuncs;
        }
    }

    return RunBatches(assembly->batches);
}

} // namespace qcx::integrals::internal
