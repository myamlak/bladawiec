// Schwarz screening: Q_ab = sqrt of the largest diagonal element of the
// (ab|ab) block, per canonical pair (screening.hpp). The diagonal quartets
// are evaluated through the fp64 batch pipeline - the engine performs no
// screening internally, this is the caller-side bound construction.
//
// The sweep is CHUNKED over the pair list (the unmodeled 10.68 GiB finding):
// a diagonal
// quartet's bra and ket pair are the SAME pair, so the contracted pair data
// the batch pipeline needs for a chunk is exactly that chunk's pairs, and
// building the WHOLE store (9.95 GiB at C42H86/def2-QZVP, 4,974 functions)
// to then read one diagonal block out of each pair's own quartet is
// allocation the sweep never needs. The geometry-only skeleton stays
// resident (sizeof(MdPairData) per canonical pair) and each chunk's pairs
// are built into it, used, and RELEASED (md_batch.hpp ReleaseChunkPairData) -
// the release is complete, not ClearChunkPairData's capacity-preserving
// teardown, because the sweep never revisits a pair and a preserved capacity
// is a retained byte (measured: the capacity-preserving form ramped to
// ~11 GiB on the 4,974-function point, the full release is what makes the
// peak the chunk). One chunk covering the whole pair list (chunkBytes = 0,
// or any list whose payload fits) reproduces the pre-chunking single-pass
// values byte for byte: the same builder over the same pairs and the same
// quartets.

#include "qcx/integrals/screening.hpp"

#include "internal/md_batch.hpp"
#include "internal/md_derivative.hpp"
#include "internal/md_engine.hpp"
#include "internal/md_one_electron.hpp"
#include "internal/shells_flat.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace qcx::integrals {

qcx::Result<std::vector<double>> ComputeSchwarzBounds(const qcx::molecule::Molecule& molecule,
                                                      const qcx::basisset::BasisSet& basisSet,
                                                      std::size_t chunkBytes) {
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

    const std::size_t nPairs = pairList->pairs.size();
    std::vector<double> bounds(nPairs, 0.0);

    if (nPairs == 0)
    {
        return bounds;
    }

    auto shells = internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    // The resident skeleton: one geometry-only MdPairData per canonical
    // pair, whose transform vectors are the not-built marker
    // (BuildChunkPairData fills exactly the listed pairs).
    std::vector<internal::MdPairData> store(nPairs);
    internal::FillPairGeometry(store, *shells, *pairList);

    std::vector<std::size_t> chunkPairs;
    std::vector<ShellQuartet> chunkQuartets;
    std::vector<ShellQuartet> computed;
    std::vector<double> values;
    const std::size_t maxBatchBytes = EriBatchOptions{}.maxBatchBytes;

    for (std::size_t start = 0; start < nPairs;)
    {
        chunkPairs.clear();
        chunkQuartets.clear();
        std::size_t chunkPayloadBytes = 0;
        std::size_t end = start;

        // The chunk runs to the first pair whose payload would push the
        // chunk's materialized pair data past the cap; a single pair heavier
        // than the cap is its own chunk (never-under: the chunk is what it
        // is, the cap is a target not a bound on the pair itself).
        while (end < nPairs)
        {
            const std::size_t payload =
                internal::ChunkPairPayloadBytes(molecule, basisSet, *pairList, end);

            if (end > start && chunkBytes != 0 && chunkPayloadBytes + payload > chunkBytes)
            {
                break;
            }

            const ShellPairIndex& pairIndex = pairList->pairs[end];
            const ShellQuartet diagonal{pairIndex.i, pairIndex.j, pairIndex.i, pairIndex.j};
            chunkPayloadBytes += payload;
            chunkPairs.push_back(end);
            chunkQuartets.push_back(diagonal);
            ++end;
        }

        internal::BuildChunkPairData(store, *shells, *pairList, chunkPairs);

        auto batches = internal::AssembleClassBatches(
            store, *pairList, chunkQuartets, maxBatchBytes, computed);

        if (!batches.has_value())
        {
            return std::unexpected(batches.error());
        }

        std::size_t total = 0;

        for (const internal::MdClassBatch& batch : *batches)
        {
            for (const internal::MdQuartetTask& task : batch.tasks)
            {
                total += store[task.braPair].nFuncs * store[task.ketPair].nFuncs;
            }
        }

        values.assign(total, 0.0);
        std::size_t base = 0;

        for (internal::MdClassBatch& batch : *batches)
        {
            batch.outF64 = values.data() + base;

            for (const internal::MdQuartetTask& task : batch.tasks)
            {
                base += store[task.braPair].nFuncs * store[task.ketPair].nFuncs;
            }
        }

        auto run = internal::RunBatches(*batches);

        if (!run.has_value())
        {
            return std::unexpected(run.error());
        }

        // The batch reports the computed quartets in output order (the
        // assembly may permute them); each is a diagonal (i,j,i,j) block.
        std::size_t offset = 0;

        for (const ShellQuartet& quartet : computed)
        {
            const ShellInfo& shellI = pairList->shells[quartet.i];
            const ShellInfo& shellJ = pairList->shells[quartet.j];
            const std::size_t nI =
                shellI.contractionCount *
                (shellI.isSpherical ? static_cast<std::size_t>(2 * shellI.angularMomentum + 1)
                                    : static_cast<std::size_t>((shellI.angularMomentum + 1) *
                                                               (shellI.angularMomentum + 2) / 2));
            const std::size_t nJ =
                shellJ.contractionCount *
                (shellJ.isSpherical ? static_cast<std::size_t>(2 * shellJ.angularMomentum + 1)
                                    : static_cast<std::size_t>((shellJ.angularMomentum + 1) *
                                                               (shellJ.angularMomentum + 2) / 2));
            // The diagonal elements (fg|fg) of the (ab|ab) block: the packed
            // layout offset ((fb*nI + fa)*nJ + fb)*nI + fa.
            double maxDiagonal = 0.0;

            for (std::size_t fa = 0; fa < nI; ++fa)
            {
                for (std::size_t fb = 0; fb < nJ; ++fb)
                {
                    const std::size_t index = ((fb * nI + fa) * nJ + fb) * nI + fa;
                    maxDiagonal = std::max(maxDiagonal, values[offset + index]);
                }
            }

            bounds[PairIndexOf(quartet.i, quartet.j, *pairList)] = std::sqrt(maxDiagonal);
            offset += nI * nJ * nI * nJ;
        }

        // The arena teardown: the chunk's payload is RELEASED, not merely
        // cleared - the sweep never revisits a pair, so the capacities
        // ClearChunkPairData preserves would accumulate over the chunk loop
        // and retain the whole store (md_batch.hpp ReleaseChunkPairData). The
        // emptied braTransform is the not-built marker either way.
        internal::ReleaseChunkPairData(store, chunkPairs);

        start = end;
    }

    return bounds;
}

namespace {

/// Folds one block's magnitudes into a running per-element maximum.
/// \param target The running per-element maxima.
/// \param block The block's elements, target.size() of them.
void AccumulateBlockMaxima(std::vector<double>& target, std::span<const double> block) {
    for (std::size_t element = 0; element < target.size(); ++element)
    {
        target[element] = std::max(target[element], std::abs(block[element]));
    }
}

/// The molecule's nuclear charges, one per atom.
/// \param molecule The molecule.
/// \returns Z per atom in Bohr order.
std::vector<double> NuclearCharges(const qcx::molecule::Molecule& molecule) {
    std::vector<double> charges;
    charges.reserve(molecule.AtomCount());

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        charges.push_back(static_cast<double>(atom.atomicNumber));
    }

    return charges;
}

} // namespace

qcx::Result<std::vector<PairDerivativeBounds>> ComputeDerivativeAwareBounds(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    std::size_t chunkBytes) {
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

    const std::size_t nPairs = pairList->pairs.size();
    std::vector<PairDerivativeBounds> bounds(nPairs);

    if (nPairs == 0)
    {
        return bounds;
    }

    auto shells = internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    const std::vector<double> charges = NuclearCharges(molecule);
    const std::vector<Eigen::Vector3d> centers = internal::AtomPositions(molecule);

    // The resident skeleton, filled one chunk at a time (ComputeSchwarzBounds'
    // arena discipline): the pass never materializes the whole store.
    std::vector<internal::MdPairData> store(nPairs);
    internal::FillPairGeometry(store, *shells, *pairList);

    std::vector<std::size_t> chunkPairs;
    std::vector<std::size_t> coordinates;
    // assign() keeps the capacity, so after the first pairs of a given shape
    // the pass allocates nothing per pair.
    std::vector<double> block;
    std::vector<double> derivatives;
    std::vector<double> value;
    std::vector<double> derivative;
    internal::MdDerivativeScratch scratch;

    for (std::size_t start = 0; start < nPairs;)
    {
        chunkPairs.clear();
        std::size_t chunkPayloadBytes = 0;
        std::size_t end = start;

        while (end < nPairs)
        {
            const std::size_t payload =
                internal::ChunkPairPayloadBytes(molecule, basisSet, *pairList, end);

            if (end > start && chunkBytes != 0 && chunkPayloadBytes + payload > chunkBytes)
            {
                break;
            }

            chunkPayloadBytes += payload;
            chunkPairs.push_back(end);
            ++end;
        }

        internal::BuildChunkPairData(store, *shells, *pairList, chunkPairs);

        for (const std::size_t pairIndex : chunkPairs)
        {
            const internal::MdPairData& pair = store[pairIndex];
            const std::size_t braAtom = pairList->shells[pairList->pairs[pairIndex].i].atomIndex;
            const std::size_t ketAtom = pairList->shells[pairList->pairs[pairIndex].j].atomIndex;

            // The pair's own coordinates: the bra atom's axes, then the ket
            // atom's - three only when both shells sit on one atom.
            coordinates.clear();

            for (int axis = 0; axis < 3; ++axis)
            {
                coordinates.push_back(3 * braAtom + static_cast<std::size_t>(axis));
            }

            if (ketAtom != braAtom)
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    coordinates.push_back(3 * ketAtom + static_cast<std::size_t>(axis));
                }
            }

            value.assign(pair.nFuncs, 0.0);
            derivative.assign(pair.nFuncs, 0.0);
            block.assign(pair.nFuncs, 0.0);
            derivatives.assign(internal::MdDerivativeBlockCount(pair, 1, coordinates.size()), 0.0);

            internal::BuildOverlapPair(pair, block.data());
            AccumulateBlockMaxima(value, block);
            internal::BuildKineticPair(pair, block.data());
            AccumulateBlockMaxima(value, block);
            internal::BuildNuclearPair(pair, charges, centers, block.data());
            AccumulateBlockMaxima(value, block);

            auto overlap = internal::BuildOverlapPairDerivative(
                pair, braAtom, ketAtom, 1, coordinates, derivatives, scratch);

            if (!overlap.has_value())
            {
                return std::unexpected(overlap.error());
            }

            // The builders write one tuple's (nFuncsA x nFuncsB) block per
            // requested coordinate, so an element's derivative is the element's
            // offset inside each tuple's block.
            for (std::size_t tuple = 0; tuple < coordinates.size(); ++tuple)
            {
                AccumulateBlockMaxima(
                    derivative,
                    std::span<const double>(derivatives).subspan(tuple * pair.nFuncs, pair.nFuncs));
            }

            auto kinetic = internal::BuildKineticPairDerivative(
                pair, braAtom, ketAtom, 1, coordinates, derivatives, scratch);

            if (!kinetic.has_value())
            {
                return std::unexpected(kinetic.error());
            }

            for (std::size_t tuple = 0; tuple < coordinates.size(); ++tuple)
            {
                AccumulateBlockMaxima(
                    derivative,
                    std::span<const double>(derivatives).subspan(tuple * pair.nFuncs, pair.nFuncs));
            }

            auto nuclear = internal::BuildNuclearPairDerivative(
                pair, braAtom, ketAtom, charges, centers, 1, coordinates, derivatives, scratch);

            if (!nuclear.has_value())
            {
                return std::unexpected(nuclear.error());
            }

            for (std::size_t tuple = 0; tuple < coordinates.size(); ++tuple)
            {
                AccumulateBlockMaxima(
                    derivative,
                    std::span<const double>(derivatives).subspan(tuple * pair.nFuncs, pair.nFuncs));
            }

            PairDerivativeBounds pairBound;
            pairBound.nFuncs = pair.nFuncs;
            pairBound.elements.resize(pair.nFuncs);

            for (std::size_t element = 0; element < pair.nFuncs; ++element)
            {
                pairBound.elements[element] =
                    DerivativeAwareBound{value[element], derivative[element]};
            }

            bounds[pairIndex] = std::move(pairBound);
        }

        internal::ReleaseChunkPairData(store, chunkPairs);

        start = end;
    }

    return bounds;
}

} // namespace qcx::integrals
