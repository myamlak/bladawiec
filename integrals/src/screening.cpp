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
#include "internal/md_eri_derivative.hpp"
#include "internal/md_one_electron.hpp"
#include "internal/shells_flat.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
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

namespace {

/// The radial finite-difference step (Bohr) of a pair bound whose derivative
/// the tier cannot reach: O(h^2) truncation against an O(eps Q / h)
/// cancellation error, near 1e-13 of a bound of O(1) at this step.
constexpr double kPairBoundRadialStep = 1.0e-4;

/// The factor the radial difference carries so its truncation cannot put the
/// bound under the derivative it bounds.
constexpr double kPairBoundRadialSafety = 1.0e-6;

/// The pair's diagonal quartet with the shells of one atom displaced along the
/// pair's separation. A quartet's value depends on its shells' centres and
/// exponents alone, so this is the same pair at another separation, reached
/// without rebuilding the molecule.
/// \param quartet The pair's diagonal quartet.
/// \param atom The atom whose shells are displaced.
/// \param direction The unit direction of the pair's separation.
/// \param delta The displacement (Bohr).
/// \param storage The displaced shells; must outlive the result.
/// \returns The displaced quartet.
internal::MdEriDerivativeQuartet ShiftedQuartet(const internal::MdEriDerivativeQuartet& quartet,
                                                std::size_t atom,
                                                const std::array<double, 3>& direction,
                                                double delta,
                                                std::array<internal::MdShellInput, 4>& storage) {
    for (std::size_t slot = 0; slot < 4; ++slot)
    {
        storage[slot] = *quartet.shells[slot];

        if (quartet.atoms[slot] == atom)
        {
            storage[slot].cx += delta * direction[0];
            storage[slot].cy += delta * direction[1];
            storage[slot].cz += delta * direction[2];
        }
    }

    internal::MdEriDerivativeQuartet shifted;
    shifted.atoms = quartet.atoms;

    for (std::size_t slot = 0; slot < 4; ++slot)
    {
        shifted.shells[slot] = &storage[slot];
    }

    return shifted;
}

/// The largest diagonal element of one (ab|ab) block: the element the bound is
/// the square root of, at the offsets the engine's own block layout names.
/// \param block The block.
/// \param nBra The first shell's function count.
/// \param nKet The second shell's function count.
/// \returns The largest diagonal element.
double LargestDiagonal(std::span<const double> block, std::size_t nBra, std::size_t nKet) {
    double largest = 0.0;

    for (std::size_t fa = 0; fa < nBra; ++fa)
    {
        for (std::size_t fb = 0; fb < nKet; ++fb)
        {
            const std::size_t diagonal = EriBlockIndex(fa, fb, fa, fb, nBra, nKet, nBra, nKet);
            largest = std::max(largest, block[diagonal]);
        }
    }

    return largest;
}

} // namespace

qcx::Result<std::vector<TwoElectronPairBound>> ComputeTwoElectronPairBounds(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet) {
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
    std::vector<TwoElectronPairBound> bounds(nPairs);

    if (nPairs == 0)
    {
        return bounds;
    }

    auto shells = internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    std::vector<std::size_t> coordinates;
    std::vector<double> block;
    std::vector<double> derivatives;
    internal::MdEriDerivativeScratch scratch;

    for (std::size_t pairIndex = 0; pairIndex < nPairs; ++pairIndex)
    {
        const ShellPairIndex& pair = pairList->pairs[pairIndex];
        const std::size_t braAtom = pairList->shells[pair.i].atomIndex;
        const std::size_t ketAtom = pairList->shells[pair.j].atomIndex;
        const int pairAngular =
            pairList->shells[pair.i].angularMomentum + pairList->shells[pair.j].angularMomentum;

        // A pair whose shells add past kMaxShellL has a diagonal block reaching
        // Hermite order 2 (l_i + l_j), past the 2 kMaxShellL the kernel table
        // covers: this build can produce neither the block nor its derivative,
        // so the pair carries no bound at either order and every quartet that
        // touches it stays. A finite stand-in would be a number nothing
        // derives, and a dropped quartet is a gradient that is smoothly wrong.
        if (pairAngular > internal::kMaxShellL)
        {
            bounds[pairIndex] = TwoElectronPairBound{std::numeric_limits<double>::infinity(),
                                                     std::numeric_limits<double>::infinity()};
            continue;
        }

        const std::array<std::size_t, 4> indices = {pair.i, pair.j, pair.i, pair.j};
        internal::MdEriDerivativeQuartet quartet;

        for (int slot = 0; slot < 4; ++slot)
        {
            const std::size_t index = indices[static_cast<std::size_t>(slot)];
            quartet.shells[static_cast<std::size_t>(slot)] = &(*shells)[index];
            quartet.atoms[static_cast<std::size_t>(slot)] = pairList->shells[index].atomIndex;
        }

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

        const std::size_t nBra = internal::EriShellFunctions(*quartet.shells[0]);
        const std::size_t nKet = internal::EriShellFunctions(*quartet.shells[1]);
        const std::size_t elements = nBra * nKet * nBra * nKet;

        block.assign(elements, 0.0);
        internal::BuildEriQuartetValue(quartet, 0, block, scratch);
        const double largestDiagonal = LargestDiagonal(block, nBra, nKet);

        TwoElectronPairBound bound;
        bound.value = std::sqrt(largestDiagonal);

        // A zero Schwarz numerator is an exactly zero block: every diagonal
        // element of a real Gaussian pair's own block is a self-repulsion, so
        // the block vanishing is the pair contributing nothing at either
        // order - and the division below has no denominator.
        if (largestDiagonal > 0.0)
        {
            if (ketAtom == braAtom)
            {
                // Both shells sit on one atom, so that atom's movement is a
                // rigid translation of the pair and the bound does not move
                // with it: the derivative is zero, exactly.
                bound.derivative = 0.0;
            } else if (2 * pairAngular + 2 <= 2 * internal::kMaxShellL)
            {
                derivatives.assign(
                    internal::MdEriDerivativeBlockCount(quartet, 1, coordinates.size()), 0.0);
                auto built = internal::BuildEriQuartetDerivative(
                    quartet, 1, coordinates, derivatives, scratch, molecule.AtomCount());

                if (!built.has_value())
                {
                    return std::unexpected(built.error());
                }

                // The bound's own derivative: m is the largest of the diagonal
                // elements, so |dm/dX| is at most the largest |dq/dX| over
                // them, and dQ_ab/dX is that over 2 sqrt(m). The pair's block
                // depends on its two centres through their separation alone, so
                // the ket atom's coordinates carry the bra's derivative with
                // the opposite sign and the largest over the pair's
                // coordinates is the one number that bounds every centre the
                // pair carries.
                double largestDerivative = 0.0;

                for (std::size_t tuple = 0; tuple < coordinates.size(); ++tuple)
                {
                    const std::span<const double> tupleBlock =
                        std::span<const double>(derivatives).subspan(tuple * elements, elements);

                    for (std::size_t fa = 0; fa < nBra; ++fa)
                    {
                        for (std::size_t fb = 0; fb < nKet; ++fb)
                        {
                            const std::size_t diagonal = ((fb * nBra + fa) * nKet + fb) * nBra + fa;
                            largestDerivative =
                                std::max(largestDerivative, std::abs(tupleBlock[diagonal]));
                        }
                    }
                }

                bound.derivative = largestDerivative / (2.0 * bound.value);
            } else
            {
                // The tier's derivative of this pair's diagonal quartet would
                // reach Hermite order 2 (l_i + l_j) + 2, past the table, so the
                // bound's derivative is measured instead. The bound depends on
                // the pair's geometry only through the two shells' separation,
                // so its derivative along that separation is the largest of the
                // pair's centre-axis derivatives - each of the others is this
                // one times a direction cosine.
                const internal::MdShellInput& braShell = (*shells)[pair.i];
                const internal::MdShellInput& ketShell = (*shells)[pair.j];
                const std::array<double, 3> separation = {ketShell.cx - braShell.cx,
                                                          ketShell.cy - braShell.cy,
                                                          ketShell.cz - braShell.cz};
                const double distance =
                    std::sqrt(separation[0] * separation[0] + separation[1] * separation[1] +
                              separation[2] * separation[2]);
                // Two shells on distinct atoms at one point: every direction
                // changes the separation by its own step, so the difference
                // below is zero there - which is the derivative.
                const std::array<double, 3> direction = {
                    distance > 0.0 ? separation[0] / distance : 1.0,
                    distance > 0.0 ? separation[1] / distance : 0.0,
                    distance > 0.0 ? separation[2] / distance : 0.0};
                std::array<internal::MdShellInput, 4> storage;
                std::array<double, 2> shiftedDiagonal = {};
                const std::array<double, 2> deltas = {kPairBoundRadialStep, -kPairBoundRadialStep};

                for (std::size_t step = 0; step < deltas.size(); ++step)
                {
                    const internal::MdEriDerivativeQuartet shifted =
                        ShiftedQuartet(quartet, braAtom, direction, deltas[step], storage);
                    block.assign(elements, 0.0);
                    internal::BuildEriQuartetValue(shifted, 0, block, scratch);
                    shiftedDiagonal[step] = LargestDiagonal(block, nBra, nKet);
                }

                bound.derivative =
                    std::abs(std::sqrt(shiftedDiagonal[0]) - std::sqrt(shiftedDiagonal[1])) /
                    (2.0 * kPairBoundRadialStep) * (1.0 + kPairBoundRadialSafety);
            }
        }

        bounds[pairIndex] = bound;
    }

    return bounds;
}

} // namespace qcx::integrals
