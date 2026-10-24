// The one-electron derivative surface (one_electron.hpp): the nuclear-
// coordinate derivatives of the overlap, kinetic-energy and nuclear-
// attraction matrices, contracted with a fixed density.
//
// One walk of the canonical shell pairs. The pair data is built in the same
// chunked shape the matrix builders walk (one_electron.cpp: a build-once walk
// materializes the geometry-only skeleton plus ONE chunk of contracted pair
// data, releases it, and moves on), so this path pays the same resident bound
// as the energy paths it differentiates.
//
// Each pair's three derivative blocks come from the derivative tier
// (internal/md_derivative.hpp), at the coordinates each operator reaches: the
// overlap and kinetic operators act only on the centre carrying their shell,
// so their request names the pair's two atoms; the nuclear-attraction
// operator's centre moves with every nucleus, so its request names all 3N.
//
// The density contraction is the same expression for values and derivatives: a
// diagonal shell pair's block covers the full shell square, an off-diagonal
// pair's block covers one triangle of a symmetric matrix, and the mirrored
// density supplies the other. No matrix is formed - the walk accumulates the
// 3N gradients directly, in one fixed pair order, so the arithmetic is
// deterministic.

#include "internal/md_batch.hpp"
#include "internal/md_derivative.hpp"
#include "internal/md_one_electron.hpp"
#include "internal/shells_flat.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <span>
#include <vector>

namespace qcx::integrals {
namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

// The next pair-data chunk of the walk: the pairs from `start` up to the first
// pair whose payload would push the chunk's materialized pair data past
// `capBytes`, filled into `chunkPairs`; returns the chunk's end (the next
// chunk's start). The policy is the matrix builders' NextPairChunk
// (one_electron.cpp) over the caller's cap: a single pair heavier than the cap
// is its own chunk - never-under, the cap is a target, not a bound on the pair
// itself - and a zero cap therefore walks one pair at a time.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): start, cap.
std::size_t NextPairChunk(const qcx::molecule::Molecule& molecule,
                          const qcx::basisset::BasisSet& basisSet,
                          const ShellPairList& pairList,
                          std::size_t start,
                          std::size_t capBytes,
                          std::vector<std::size_t>& chunkPairs) {
    chunkPairs.clear();
    std::size_t chunkPayloadBytes = 0;
    std::size_t end = start;

    while (end < pairList.pairs.size())
    {
        const std::size_t payload =
            internal::ChunkPairPayloadBytes(molecule, basisSet, pairList, end);

        if (end > start && chunkPayloadBytes + payload > capBytes)
        {
            break;
        }

        chunkPayloadBytes += payload;
        chunkPairs.push_back(end);
        ++end;
    }

    return end;
}

// The nuclear charges the attraction builder sums, one per atom.
std::vector<double> NuclearCharges(const qcx::molecule::Molecule& molecule) {
    std::vector<double> charges;
    charges.reserve(molecule.AtomCount());

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        charges.push_back(static_cast<double>(atom.atomicNumber));
    }

    return charges;
}

// The density contraction of one pair block: sum_uv D_uv B_uv over the function
// pairs the block covers. A canonical pair whose two shells are the same shell
// covers the whole square; one whose shells differ covers the (i, j) triangle
// alone, and the (j, i) triangle's terms are the mirrored density times the
// same block, because the integral matrices are symmetric.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): offsetA, offsetB.
double ContractBlock(const CpuTensor2& density,
                     const internal::MdPairData& pair,
                     std::size_t offsetA,
                     std::size_t offsetB,
                     const double* block) {
    const bool diagonal = offsetA == offsetB;
    double sum = 0.0;

    for (std::size_t fa = 0; fa < pair.nFuncsA; ++fa)
    {
        for (std::size_t fb = 0; fb < pair.nFuncsB; ++fb)
        {
            double weight = density(offsetA + fa, offsetB + fb);

            if (!diagonal)
            {
                weight += density(offsetB + fb, offsetA + fa);
            }

            sum += weight * block[fa * pair.nFuncsB + fb];
        }
    }

    return sum;
}

// Scatters one operator's derivative blocks into its gradient: one
// (nFuncsA x nFuncsB) block per requested coordinate, tuple-major in request
// order - the derivative tier's layout.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): offsetA, offsetB.
void AccumulateGradient(const internal::MdPairData& pair,
                        const CpuTensor2& density,
                        std::size_t offsetA,
                        std::size_t offsetB,
                        std::span<const std::size_t> coordinates,
                        const std::vector<double>& blocks,
                        Eigen::VectorXd& gradient) {
    for (std::size_t tuple = 0; tuple < coordinates.size(); ++tuple)
    {
        const double value =
            ContractBlock(density, pair, offsetA, offsetB, blocks.data() + tuple * pair.nFuncs);
        gradient(static_cast<Eigen::Index>(coordinates[tuple])) += value;
    }
}

// One canonical pair's contribution: the three value contractions and the
// three derivative walks, all on the pair data the chunk store holds.
// \param pairList The shell pair list (function offsets and atoms).
// \param pair The pair's contracted data.
// \param pairIndex The pair's canonical index (into pairList.pairs).
// \param density The fixed density matrix.
// \param charges Nuclear charges Z, one per atom.
// \param centers Atom centers (Bohr).
// \param allCoordinates Every flat nuclear coordinate (3 * atom + axis).
// \param scratch The derivative tier's scratch, held across the walk.
// \param ownCoordinates Scratch for the pair's own two atoms' coordinates.
// \param blocks Scratch for the derivative blocks.
// \param values Scratch for the value block.
// \param result Out: the gradients, contractions and counts.
// \returns Nothing, or an Error from the derivative tier.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): the scratch vectors.
qcx::Result<void> AccumulatePair(const ShellPairList& pairList,
                                 const internal::MdPairData& pair,
                                 std::size_t pairIndex,
                                 const CpuTensor2& density,
                                 const std::vector<double>& charges,
                                 const std::vector<Eigen::Vector3d>& centers,
                                 std::span<const std::size_t> allCoordinates,
                                 internal::MdDerivativeScratch& scratch,
                                 std::vector<std::size_t>& ownCoordinates,
                                 std::vector<double>& blocks,
                                 std::vector<double>& values,
                                 OneElectronGradient& result) {
    const ShellPairIndex& index = pairList.pairs[pairIndex];
    const std::size_t offsetA = pairList.shells[index.i].functionOffset;
    const std::size_t offsetB = pairList.shells[index.j].functionOffset;
    const std::size_t braAtom = pairList.shells[index.i].atomIndex;
    const std::size_t ketAtom = pairList.shells[index.j].atomIndex;

    // The values the gradients belong to: the same builders the matrix path
    // runs, on this pair's own contracted data.
    values.assign(pair.nFuncs, 0.0);
    internal::BuildOverlapPair(pair, values.data());
    result.overlapContraction += ContractBlock(density, pair, offsetA, offsetB, values.data());
    internal::BuildKineticPair(pair, values.data());
    result.kineticContraction += ContractBlock(density, pair, offsetA, offsetB, values.data());
    internal::BuildNuclearPair(pair, charges, centers, values.data());
    result.nuclearContraction += ContractBlock(density, pair, offsetA, offsetB, values.data());

    // The coordinates the overlap and kinetic operators reach: the atoms
    // carrying the pair's shells, three axes each (three only when both
    // shells sit on one atom).
    ownCoordinates.clear();

    for (int axis = 0; axis < 3; ++axis)
    {
        ownCoordinates.push_back(3 * braAtom + static_cast<std::size_t>(axis));
    }

    if (ketAtom != braAtom)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            ownCoordinates.push_back(3 * ketAtom + static_cast<std::size_t>(axis));
        }
    }

    blocks.assign(ownCoordinates.size() * pair.nFuncs, 0.0);
    auto overlap = internal::BuildOverlapPairDerivative(
        pair, braAtom, ketAtom, 1, ownCoordinates, blocks, scratch);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    AccumulateGradient(
        pair, density, offsetA, offsetB, ownCoordinates, blocks, result.overlapGradient);

    blocks.assign(ownCoordinates.size() * pair.nFuncs, 0.0);
    auto kinetic = internal::BuildKineticPairDerivative(
        pair, braAtom, ketAtom, 1, ownCoordinates, blocks, scratch);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    AccumulateGradient(
        pair, density, offsetA, offsetB, ownCoordinates, blocks, result.kineticGradient);

    // The nuclear-attraction operator's centre moves with every nucleus, so
    // its derivative reaches every coordinate - including those of nuclei
    // carrying neither shell, which enter through the operator centre alone.
    blocks.assign(allCoordinates.size() * pair.nFuncs, 0.0);
    auto nuclear = internal::BuildNuclearPairDerivative(
        pair, braAtom, ketAtom, charges, centers, 1, allCoordinates, blocks, scratch);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    AccumulateGradient(
        pair, density, offsetA, offsetB, allCoordinates, blocks, result.nuclearGradient);

    ++result.counts.orbitalPairs;
    result.counts.overlapTuples += ownCoordinates.size();
    result.counts.kineticTuples += ownCoordinates.size();
    result.counts.nuclearTuples += allCoordinates.size();
    return {};
}

} // namespace

qcx::Result<OneElectronGradient> ComputeOneElectronGradient(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    const OneElectronGradientOptions& options) {
    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    const std::size_t n = pairList->functionCount;

    if (density.Shape()[0] != n || density.Shape()[1] != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the density matrix is not nBasis x nBasis for this basis set"});
    }

    auto shells = internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    const std::size_t nAtoms = molecule.AtomCount();
    const Eigen::Index gradientSize = static_cast<Eigen::Index>(3 * nAtoms);
    OneElectronGradient result;
    result.overlapGradient = Eigen::VectorXd::Zero(gradientSize);
    result.kineticGradient = Eigen::VectorXd::Zero(gradientSize);
    result.nuclearGradient = Eigen::VectorXd::Zero(gradientSize);

    const std::vector<double> charges = NuclearCharges(molecule);
    const std::vector<Eigen::Vector3d> centers = internal::AtomPositions(molecule);
    std::vector<std::size_t> allCoordinates(3 * nAtoms);

    for (std::size_t flat = 0; flat < allCoordinates.size(); ++flat)
    {
        allCoordinates[flat] = flat;
    }

    // The resident skeleton: one geometry-only MdPairData per canonical pair,
    // whose empty transform vectors are the not-built marker
    // (BuildChunkPairData fills exactly the listed pairs).
    std::vector<internal::MdPairData> store(pairList->pairs.size());
    internal::FillPairGeometry(store, *shells, *pairList);

    internal::MdDerivativeScratch scratch;
    std::vector<std::size_t> chunkPairs;
    std::vector<std::size_t> ownCoordinates;
    std::vector<double> blocks;
    std::vector<double> values;

    for (std::size_t start = 0; start < pairList->pairs.size();)
    {
        const std::size_t end = NextPairChunk(
            molecule, basisSet, *pairList, start, options.maxPairChunkBytes, chunkPairs);
        internal::BuildChunkPairData(store, *shells, *pairList, chunkPairs);

        // One serial pair order: every gradient entry is shared by many pairs,
        // so a per-pair parallel accumulation would need a reduction; the walk
        // keeps the arithmetic one fixed sum instead.
        for (const std::size_t pairIndex : chunkPairs)
        {
            auto accumulated = AccumulatePair(*pairList,
                                              store[pairIndex],
                                              pairIndex,
                                              density,
                                              charges,
                                              centers,
                                              std::span<const std::size_t>(allCoordinates),
                                              scratch,
                                              ownCoordinates,
                                              blocks,
                                              values,
                                              result);

            if (!accumulated.has_value())
            {
                return std::unexpected(accumulated.error());
            }
        }

        internal::ReleaseChunkPairData(store, chunkPairs);
        start = end;
    }

    return result;
}

} // namespace qcx::integrals
