#pragma once

/// \file
/// The LightPath rung's Create-time footprint arithmetic, shared by the builders that
/// select the rung and by the enclosing builders that price a nested half at it: the
/// per-pair payload bound, the chunk auto-sizing closed form, the chunk row window, the
/// peak-chunk markers, the chunk bookkeeping byte terms and the light estimate itself.
/// The light rung's chunk loop (fock_build.cpp BuildFock) and the estimate consulted here
/// read the SAME functions, so the charge a decision makes and the bytes the rung holds
/// cannot drift apart. Moved verbatim out of fock_build.cpp's anonymous namespace (the
/// RI-J's nested-exchange pricing calls them: ri_engine.cpp).

#include "internal/footprint.hpp"
#include "internal/md_batch.hpp"
#include "qcx/integrals/accuracy.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace qcx::integrals::internal {

// The loose per-pair upper bound of the LightPath chunk arena: the largest
// pair's FastPath payload (PairStoreBytesPerPair minus the geometry-only
// MdPairData the light store already carries) - the "mean per-class
// pair cost" taken as the max, so the chunk sizing is conservative
// (correctness-neutral, never under).
inline std::size_t MaxPairPayload(const qcx::molecule::Molecule& molecule,
                                  const qcx::basisset::BasisSet& basisSet,
                                  const ShellPairList& pairList) {
    std::size_t maxPayload = 0;

    for (const ShellPairIndex& pair : pairList.pairs)
    {
        const ShellInfo& a = pairList.shells[pair.i];
        const ShellInfo& b = pairList.shells[pair.j];
        const std::size_t nPrimA = internal::ShellPrimitiveCount(molecule, basisSet, a);
        const std::size_t nPrimB = internal::ShellPrimitiveCount(molecule, basisSet, b);
        const std::size_t payload = internal::PairStoreBytesPerPair(a.angularMomentum,
                                                                    b.angularMomentum,
                                                                    a.isSpherical,
                                                                    b.isSpherical,
                                                                    a.contractionCount,
                                                                    b.contractionCount,
                                                                    nPrimA * nPrimB);

        if (payload > sizeof(internal::MdPairData))
        {
            maxPayload = std::max(maxPayload, payload - sizeof(internal::MdPairData));
        }
    }

    return maxPayload;
}

// The LightPath chunk auto-sizing (the closed form): the largest
// chunk whose LOOSE upper bound fits the remaining budget after the fixed
// terms - the light store, the structural block, the per-thread scratch
// arena, and the fixed-extra terms (the chunk-pair bookkeeping and the
// retained shells - none of them shrinks with C) - then the chunk's C bra
// rows plus at most C x maxRow distinct kets at the max per-pair payload,
// and at most C x maxRow counted pattern rows. The loose bounds give a
// conservative floor (fewer, smaller chunks than the honest optimum -
// correctness-neutral, never under). maxRow is the counted max
// neighbor-row length, or nPairs at the a-priori exclusion (the
// all-survive bound).
// (remaining, lightStoreBytes) are distinct terms of the footprint
// estimate; the single call site passes named variables.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline std::size_t AutoLightChunkPairs(std::size_t remaining,
                                       std::size_t lightStoreBytes,
                                       std::size_t structuralBytes,
                                       std::size_t fullScratchBytes,
                                       // (fixedExtraBytes, maxPairPayload) are distinct terms
                                       // of the footprint estimate; the single call site
                                       // passes named variables.
                                       //
                                       // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                       std::size_t fixedExtraBytes,
                                       std::size_t maxPairPayload,
                                       // (maxRow, nPairs) are distinct terms of the footprint
                                       // estimate; the single call site passes named variables.
                                       //
                                       // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                       std::size_t maxRow,
                                       std::size_t nPairs) {
    const std::size_t fixed =
        lightStoreBytes + structuralBytes + fullScratchBytes + fixedExtraBytes;
    const std::size_t perRow = maxPairPayload * (1 + maxRow) + 8 * maxRow;

    if (remaining <= fixed || perRow == 0)
    {
        return 1;
    }

    return std::clamp((remaining - fixed) / perRow, std::size_t{1}, nPairs);
}

// The LightPath chunk boundaries: the bra-row range of chunk c
// of chunkPairs-sized chunks over nPairs rows. The SAME formula serves the
// Create-time marker pass and the BuildFock chunk loop, so the estimate's
// peak terms describe the loop's actual chunks. Rows are half-open; the
// last chunk takes the remainder.
constexpr std::size_t LightChunkRowStart(std::size_t chunkPairs, std::size_t chunk) noexcept {
    return chunk * chunkPairs;
}

constexpr std::size_t LightChunkRowEnd(std::size_t chunkPairs,
                                       std::size_t chunk,
                                       std::size_t nPairs) noexcept {
    return std::min((chunk + 1) * chunkPairs, nPairs);
}

constexpr std::size_t LightChunkCount(std::size_t nPairs, std::size_t chunkPairs) noexcept {
    return (nPairs + chunkPairs - 1) / chunkPairs;
}

// The LightPath peak-chunk markers: the largest chunk's distinct ket count
// and its counted surviving rows, over the decision's swept CSR (the
// marker pass is O(pattern) once with a stamp vector - the "peak =
// max chunk, never the sum": the tail chunks' kets are the tail rows'
// neighborhood, small at scale). At the a-priori exclusion (no sweep) the
// exclusion's own counted peak row width stands in - a chunk's ket set is
// the union of its rows', each bounded by the widest row - and only where
// no count exists at all (the knob path) do the all-survive bounds stand:
// maxChunkKets = nPairs, peakChunkPattern = chunkPairs x nPairs.
struct LightChunkMarkers {
    std::size_t maxChunkKets = 0;
    std::size_t peakChunkPattern = 0;
};

// (neighborRowOffsets, neighborIndices) are the CSR offset/index pair; the
// single call site passes them in fixed order.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline LightChunkMarkers PeakChunkMarkers(const std::vector<std::size_t>& neighborRowOffsets,
                                          const std::vector<std::size_t>& neighborIndices,
                                          std::size_t nPairs,
                                          std::size_t chunkPairs) {
    LightChunkMarkers markers;
    const std::size_t numChunks = LightChunkCount(nPairs, chunkPairs);
    // The stamp vector: the distinct-ket set of each chunk's bra window.
    std::vector<std::size_t> stamps(nPairs, 0);
    std::size_t stampCounter = 0;

    for (std::size_t c = 0; c < numChunks; ++c)
    {
        const std::size_t rowStart = LightChunkRowStart(chunkPairs, c);
        const std::size_t rowEnd = LightChunkRowEnd(chunkPairs, c, nPairs);
        std::size_t distinctKets = 0;
        std::size_t patternRows = 0;
        ++stampCounter;

        for (std::size_t bra = rowStart; bra < rowEnd; ++bra)
        {
            const std::size_t begin = neighborRowOffsets[bra];
            const std::size_t end = neighborRowOffsets[bra + 1];
            patternRows += end - begin;

            for (std::size_t k = begin; k < end; ++k)
            {
                const std::size_t ket = neighborIndices[k];

                if (stamps[ket] != stampCounter)
                {
                    stamps[ket] = stampCounter;
                    ++distinctKets;
                }
            }
        }

        markers.maxChunkKets = std::max(markers.maxChunkKets, distinctKets);
        markers.peakChunkPattern = std::max(markers.peakChunkPattern, patternRows);
    }

    return markers;
}

// The LightPath per-call chunk-pair bookkeeping (BuildFock's chunk loop):
// the stamp vector (nPairs counters, one pair-space pass per chunk) and
// the chunk pair set (the built pairs' full-space indices - a subset of
// the pair space, so nPairs entries bound it), both live for the whole
// build - the peak is the union, never a sum over chunks.
inline std::size_t LightChunkIndexBytes(std::size_t nPairs) noexcept {
    return 2 * (nPairs * sizeof(std::size_t) + internal::kVectorHeaderBytes);
}

// The LightPath retained flattened shells: one MdShellInput per ShellInfo
// (FlattenShells, built once at Create for the per-chunk transform fills -
// the state._lightShells member).
inline std::size_t LightShellsBytes(std::size_t nShells) noexcept {
    return nShells * sizeof(internal::MdShellInput) + internal::kVectorHeaderBytes;
}

// The LightPath footprint terms: the light store (nPairs
// geometry-only entries), the peak chunk's arena ((chunkPairs +
// maxChunkKets) x maxPairPayload - the chunk's bra rows plus its distinct
// kets at the max per-pair payload; on the class path the chunk's orbit
// representatives are extra pairs, bounded by 2 x the chunk's counted
// rows), the peak chunk's counted surviving rows (the per-chunk CSR, 8
// bytes per row), the per-call chunk-pair bookkeeping (the stamp vector
// and the chunk pair set), the retained flattened shells, the per-thread
// batch arena, and the shared structural block (the DirectFootprint
// structural term - pattern-independent; the other DirectFootprint terms
// are not consulted). The ERI cache is disengaged on the LightPath (the
// miss-assembly would read the full pair store), so the cache term is
// zero by construction.
struct LightFootprintTerms {
    std::size_t lightStoreBytes = 0;
    std::size_t chunkArenaBytes = 0;
    std::size_t chunkPatternBytes = 0;
    std::size_t chunkIndexBytes = 0;
    std::size_t lightShellsBytes = 0;
    std::size_t scratchBytes = 0;
    std::size_t structuralBytes = 0;
    std::size_t classTableBytes = 0; ///< The class path's table charge (ClassTableBytes),
                                     ///< whole-build structural - 0 on the plain path.
    std::size_t screenedQuartetBytes = 0; ///< The whole-run screened-quartet task machinery
                                          ///< (ScreenedQuartetBytes - the whole-run charge: the
                                          ///< task-master survivor, kScreenedTaskBytesPerQuartet
                                          ///< per screened quartet plus the certified lane
                                          ///< delta).

    std::size_t Total() const noexcept {
        return lightStoreBytes + chunkArenaBytes + chunkPatternBytes + chunkIndexBytes +
               lightShellsBytes + scratchBytes + structuralBytes + classTableBytes +
               screenedQuartetBytes;
    }
};

inline LightFootprintTerms LightFootprint(const qcx::molecule::Molecule& molecule,
                                          const qcx::basisset::BasisSet& basisSet,
                                          const ShellPairList& pairList,
                                          std::size_t chunkPairs,
                                          // (maxChunkKets, peakChunkPattern) are distinct
                                          // footprint terms; the single call site passes
                                          // named variables.
                                          //
                                          // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                          std::size_t maxChunkKets,
                                          std::size_t peakChunkPattern,
                                          // (maxPairPayload, maxBatchBytes) are distinct
                                          // footprint terms; the single call site passes
                                          // named variables.
                                          //
                                          // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                          std::size_t maxPairPayload,
                                          std::size_t maxBatchBytes,
                                          std::size_t threadCount,
                                          bool classPath,
                                          qcx::integrals::AccuracyPreset accuracy,
                                          std::size_t classTableBytes = 0) {
    LightFootprintTerms terms;
    terms.lightStoreBytes = pairList.pairs.size() * sizeof(internal::MdPairData);
    terms.chunkIndexBytes = LightChunkIndexBytes(pairList.pairs.size());
    terms.lightShellsBytes = LightShellsBytes(pairList.shells.size());
    // The class path's orbit representatives are pairs outside the chunk's
    // bra rows and kets (the petite list's canonical quartets can sit
    // anywhere in the pair space); each screened member quartet maps to one
    // rep, so the extra pairs are bounded by 2 x the chunk's counted rows.
    const std::size_t classRepPairs = classPath ? 2 * peakChunkPattern : 0;
    terms.chunkArenaBytes = (chunkPairs + maxChunkKets + classRepPairs) * maxPairPayload;
    terms.chunkPatternBytes = peakChunkPattern * 8;
    terms.scratchBytes = maxBatchBytes * threadCount;
    // Structural-only query: the exchange term is not part of the light
    // model and is not consulted - exchangeEngaged false is threaded
    // explicitly.
    terms.structuralBytes =
        internal::DirectFootprint(
            molecule, basisSet, pairList, 0, maxBatchBytes, threadCount, false, 0, accuracy, false)
            .structuralBytes;
    // The class table is whole-build structural on the light rung too: the
    // table is built once at Create regardless of the mode (the chunk loop
    // consumes it), so the charge rides the light estimate when the
    // admission gate engaged the class path.
    terms.classTableBytes = classTableBytes;
    // The screened-quartet task machinery rides the light estimate
    // UNCONDITIONALLY - the patternExcluded path included (the fraction
    // law's input is the function count, available a priori): the light
    // rung runs its chunks off the sorted
    // task masters, whole-call resident on every light run, and the
    // whole-run law is never-under the per-chunk peak by construction
    // (the per-chunk surviving set is a subset of the whole-run set).
    terms.screenedQuartetBytes = internal::ScreenedQuartetBytes(pairList.functionCount, accuracy);
    return terms;
}

} // namespace qcx::integrals::internal
