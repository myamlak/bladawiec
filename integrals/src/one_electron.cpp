// One-electron integral matrices: overlap, kinetic energy, and nuclear
// attraction - general angular momentum through the MD machinery
// (internal/md_one_electron.hpp; runtime l, no class instantiations). The
// matrices are symmetric, so canonical pairs are evaluated once and their
// blocks mirrored transposed.
#include "qcx/integrals/one_electron.hpp"

#include "internal/footprint.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_one_electron.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace qcx::integrals {
namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

// The 1e prep's pair-data chunk cap: the shared chunk policy of every
// build-once walk (internal::kPairChunkBytes - the screening sweep chunks at
// the same granularity), so the prep's resident pair data is one standard
// batch, not the whole store.
constexpr std::size_t kPairPrepChunkBytes = internal::kPairChunkBytes;

// The output matrix skeleton every builder needs.
qcx::Result<CpuTensor2> CreateSquareMatrix(std::size_t n) {
    auto matrix = CpuTensor2::Create({n, n});

    if (!matrix.has_value())
    {
        return std::unexpected(matrix.error());
    }

    return std::move(*matrix);
}

// The next pair-data chunk of a build-once walk: the pairs from `start` up to
// the first pair whose payload would push the chunk's materialized pair data
// past kPairPrepChunkBytes, filled into `chunkPairs`; returns the chunk's end
// (the next chunk's start). A single pair heavier than the cap is its own
// chunk - never-under: the chunk is what it is, the cap is a target, not a
// bound on the pair itself.
//
// ONE sizing policy for every build-once 1e walk (the matrix build and the
// point potentials below): the cap, the payload formula
// (internal::ChunkPairPayloadBytes) and the never-under rule live here, so
// the two callers cannot drift apart in what they call a chunk.
std::size_t NextPairChunk(const qcx::molecule::Molecule& molecule,
                          const qcx::basisset::BasisSet& basisSet,
                          const ShellPairList& pairList,
                          std::size_t start,
                          std::vector<std::size_t>& chunkPairs) {
    chunkPairs.clear();
    std::size_t chunkPayloadBytes = 0;
    std::size_t end = start;

    while (end < pairList.pairs.size())
    {
        const std::size_t payload =
            internal::ChunkPairPayloadBytes(molecule, basisSet, pairList, end);

        if (end > start && chunkPayloadBytes + payload > kPairPrepChunkBytes)
        {
            break;
        }

        chunkPayloadBytes += payload;
        chunkPairs.push_back(end);
        ++end;
    }

    return end;
}

// The shared build: per canonical pair the (nFuncsA x nFuncsB) block from
// the pair data, mirrored transposed into the symmetric matrix. Pairs write
// disjoint regions, so the loop parallelizes.
//
// The pair data is CHUNKED (the run5000-fixes shape, screening.cpp's sweep):
// the prep needs each pair's own block and nothing else, so materializing the
// WHOLE contracted pair store - 9.95 GiB per call at C42H86/def2-QZVP
// (4,974 functions, 1,239,525 pairs) - is allocation it never needs. The
// geometry-only skeleton stays resident (sizeof(MdPairData) per canonical
// pair, 0.27 GiB at that size), ONE kPairPrepChunkBytes chunk of contracted
// pair data is built into it, the chunk's pairs are written into the matrix,
// and the chunk is RELEASED (md_batch.hpp ReleaseChunkPairData - a build-once
// walk must not use the capacity-preserving ClearChunkPairData, or the
// capacities accumulate to the whole store one chunk at a time; the
// 4,974-function point's pre-chunking form SAMPLED at 10.748 GiB, and the
// chunked form's own figure is still a scheduled measurement). Every
// builder (overlap, kinetic, nuclear attraction, dipole, quadrupole) takes
// this same path, and the run makes several sequential calls, so the term
// falls from one whole store PER CALL to the skeleton plus one chunk.
//
// The values are bit-identical to the pre-chunking build: the same builder
// function over the same pair data (BuildChunkPairData fills exactly
// BuildPairData's contracted fields - md_batch.hpp), the same pair order, the
// same per-pair writes to the same matrix cells.
qcx::Result<CpuTensor2> BuildPairMatrix(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const std::function<void(const internal::MdPairData&, double*)>& builder) {
    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto matrix = CreateSquareMatrix(pairList->functionCount);

    if (!matrix.has_value())
    {
        return std::unexpected(matrix.error());
    }

    auto shells = internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    const std::size_t nPairs = pairList->pairs.size();

    // The resident skeleton: one geometry-only MdPairData per canonical pair,
    // whose empty transform vectors are the not-built marker
    // (BuildChunkPairData fills exactly the listed pairs).
    std::vector<internal::MdPairData> store(nPairs);
    internal::FillPairGeometry(store, *shells, *pairList);

    // Dynamic scheduling: the per-pair cost varies widely with the shell
    // angular momenta (an s-s pair is a single fold, a d-d pair an order
    // more), so static chunks would skew the load. The schedule stays
    // per-pair and dynamic WITHIN the chunk, so the load averaging the
    // unchunked loop had survives the chunking. No num_threads here - the
    // backend default applies (the callers that need explicit counts pass
    // them at their own seam).
    qcx::backend::Backend<qcx::backend::CpuTag> backend;
    std::vector<std::size_t> chunkPairs;

    for (std::size_t start = 0; start < nPairs;)
    {
        const std::size_t end = NextPairChunk(molecule, basisSet, *pairList, start, chunkPairs);
        internal::BuildChunkPairData(store, *shells, *pairList, chunkPairs);

        backend.ParallelForDynamic(chunkPairs.size(), [&](std::size_t slot) {
            const std::size_t pairIdx = chunkPairs[slot];
            const ShellPairIndex& pairIndex = pairList->pairs[pairIdx];
            const internal::MdPairData& pair = store[pairIdx];
            const std::size_t oI = pairList->shells[pairIndex.i].functionOffset;
            const std::size_t oJ = pairList->shells[pairIndex.j].functionOffset;
            std::vector<double> block(pair.nFuncs);
            builder(pair, block.data());

            for (std::size_t fa = 0; fa < pair.nFuncsA; ++fa)
            {
                for (std::size_t fb = 0; fb < pair.nFuncsB; ++fb)
                {
                    const double value = block[fa * pair.nFuncsB + fb];
                    (*matrix)(oI + fa, oJ + fb) = value;
                    (*matrix)(oJ + fb, oI + fa) = value;
                }
            }
        });

        internal::ReleaseChunkPairData(store, chunkPairs);
        start = end;
    }

    matrix->MarkHostDirty();
    return std::move(*matrix);
}

std::vector<double> NuclearCharges(const qcx::molecule::Molecule& molecule) {
    std::vector<double> charges;
    charges.reserve(molecule.AtomCount());

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        charges.push_back(static_cast<double>(atom.atomicNumber));
    }

    return charges;
}

std::vector<Eigen::Vector3d> AtomPositions(const qcx::molecule::Molecule& molecule) {
    const auto& coordinates = molecule.CoordinatesBohr();
    std::vector<Eigen::Vector3d> positions;
    positions.reserve(molecule.AtomCount());

    for (std::size_t atomIndex = 0; atomIndex < molecule.AtomCount(); ++atomIndex)
    {
        positions.emplace_back(
            coordinates(atomIndex, 0), coordinates(atomIndex, 1), coordinates(atomIndex, 2));
    }

    return positions;
}

} // namespace

// The setup's peak resident bytes: the never-under bound of the matrix
// builders above AND the Schwarz screening sweeps a budgeted builder runs
// in front of its budget decision, computable before either runs. The
// composition is internal::SetupPeakBytes' (footprint.hpp: the ramp's
// skeleton, one pair-data chunk and the caller's live matrices, plus one
// SchwarzSweepBytes per basis in effect) - this function only supplies the
// canonical pair lists it is computed over, the same BuildShellPairs calls
// those builders and sweeps make, so its error is theirs.
qcx::Result<double> SetupRampPeakBytes(const qcx::molecule::Molecule& molecule,
                                       const qcx::basisset::BasisSet& basisSet,
                                       const qcx::basisset::BasisSet* auxBasisSet,
                                       std::size_t liveMatrices) {
    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    // The aux pair list, built only when an aux is in effect and by the
    // SAME call the engine's own sweep makes, so the charge and the sweep
    // cannot disagree about the list either.
    ShellPairList auxPairList;

    if (auxBasisSet != nullptr)
    {
        auto builtAux = BuildShellPairs(molecule, *auxBasisSet);

        if (!builtAux.has_value())
        {
            return std::unexpected(builtAux.error());
        }

        auxPairList = std::move(*builtAux);
    }

    return static_cast<double>(
        internal::SetupPeakBytes(molecule,
                                 basisSet,
                                 *pairList,
                                 liveMatrices,
                                 auxBasisSet,
                                 auxBasisSet != nullptr ? &auxPairList : nullptr));
}

qcx::Result<double> SetupAdmissionReserveBytes(const qcx::molecule::Molecule& molecule,
                                               const qcx::basisset::BasisSet& basisSet,
                                               const qcx::basisset::BasisSet* auxBasisSet,
                                               std::size_t liveMatrices) {
    // Composed, never re-derived: the reserve IS the ramp-and-sweeps bound
    // times the one slack constant, so a change to the bound moves both floors
    // at once and they cannot drift apart.
    auto peakBytes = SetupRampPeakBytes(molecule, basisSet, auxBasisSet, liveMatrices);

    if (!peakBytes.has_value())
    {
        return std::unexpected(peakBytes.error());
    }

    return *peakBytes * kSetupAdmissionSlack;
}

qcx::Result<CpuTensor2> BuildOverlapMatrix(const qcx::molecule::Molecule& molecule,
                                           const qcx::basisset::BasisSet& basisSet) {
    return BuildPairMatrix(molecule, basisSet, [](const internal::MdPairData& pair, double* block) {
        internal::BuildOverlapPair(pair, block);
    });
}

qcx::Result<CpuTensor2> BuildKineticMatrix(const qcx::molecule::Molecule& molecule,
                                           const qcx::basisset::BasisSet& basisSet) {
    return BuildPairMatrix(molecule, basisSet, [](const internal::MdPairData& pair, double* block) {
        internal::BuildKineticPair(pair, block);
    });
}

qcx::Result<CpuTensor2> BuildNuclearAttractionMatrix(const qcx::molecule::Molecule& molecule,
                                                     const qcx::basisset::BasisSet& basisSet) {
    const std::vector<double> charges = NuclearCharges(molecule);
    const std::vector<Eigen::Vector3d> positions = AtomPositions(molecule);

    return BuildPairMatrix(molecule,
                           basisSet,
                           [&charges, &positions](const internal::MdPairData& pair, double* block) {
                               internal::BuildNuclearPair(pair, charges, positions, block);
                           });
}

qcx::Result<CpuTensor2> BuildNuclearAttractionMatrix(const qcx::molecule::Molecule& molecule,
                                                     const qcx::basisset::BasisSet& basisSet,
                                                     std::span<const std::size_t> centerIndices) {
    if (centerIndices.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "centerIndices must not be empty"});
    }

    const std::size_t atomCount = molecule.AtomCount();
    const auto& coordinates = molecule.CoordinatesBohr();
    std::vector<double> charges;
    std::vector<Eigen::Vector3d> positions;
    charges.reserve(centerIndices.size());
    positions.reserve(centerIndices.size());

    for (const std::size_t index : centerIndices)
    {
        if (index >= atomCount)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "a nuclear-attraction center index is out of range"});
        }

        charges.push_back(static_cast<double>(molecule.Atoms()[index].atomicNumber));
        positions.emplace_back(coordinates(index, 0), coordinates(index, 1), coordinates(index, 2));
    }

    return BuildPairMatrix(molecule,
                           basisSet,
                           [&charges, &positions](const internal::MdPairData& pair, double* block) {
                               internal::BuildNuclearPair(pair, charges, positions, block);
                           });
}

qcx::Result<std::vector<double>> BuildElectronPotentialAtPoints(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& points) {
    const auto& pointShape = points.Shape();

    if (pointShape[1] != 3)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "points must be {Npoints, 3} (Bohr)"});
    }

    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto shells = internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    const std::size_t n = pairList->functionCount;

    if (density.Shape()[0] != n || density.Shape()[1] != n)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "density must be n x n with n the AO count"});
    }

    const std::size_t nPairs = pairList->pairs.size();

    // The pair data is CHUNKED (the shared BuildPairMatrix shape above, one
    // sizing policy): the geometry-only skeleton stays resident
    // (sizeof(MdPairData) per canonical pair), ONE kPairPrepChunkBytes chunk
    // of contracted pair data is built into it, every point of the chunk
    // consumes it, and the chunk is RELEASED (internal::ReleaseChunkPairData;
    // the capacity-preserving ClearChunkPairData would ramp back to the whole
    // store one chunk at a time). Before this, the whole store was
    // materialized up front - 9.95 GiB at C42H86/def2-QZVP (4,974 functions,
    // 1,239,525 pairs), the same per-call term the matrix build carried -
    // for a walk that reads each pair's block and never revisits it.
    std::vector<internal::MdPairData> store(nPairs);
    internal::FillPairGeometry(store, *shells, *pairList);

    const std::size_t nPoints = pointShape[0];

    // The per-point potentials: every point reduces ALL canonical pairs in
    // pair order (one serial accumulation order per point - deterministic,
    // no cross-thread merging, the same sum the former reduction computed),
    // so the parallel axis is the POINT axis and the retained working set is
    // one double per point plus one transient per-point pair block. The
    // former pair-parallel layout kept one row of nPoints doubles per pair -
    // nPairs x nPoints x 8 B = Theta(N^3) for an atom-scaled point set
    // (45,150 pairs x ~300K CHELPG points x 8 B ~= 108 GiB at c60, and
    // ~3.6 GiB even on the MK grid), materialized with no fit check - and
    // is gone by construction here (the blocks are recomputed per point,
    // pure redundancy the pair store already serves).
    std::vector<double> values(nPoints, 0.0);
    const std::vector<double> charges(1, 1.0);

    qcx::backend::Backend<qcx::backend::CpuTag> backend;
    std::vector<std::size_t> chunkPairs;

    for (std::size_t start = 0; start < nPairs;)
    {
        const std::size_t end = NextPairChunk(molecule, basisSet, *pairList, start, chunkPairs);
        internal::BuildChunkPairData(store, *shells, *pairList, chunkPairs);

        // The chunk's widest pair, not the whole store's: only this chunk's
        // pairs are read here, and the scratch is sized once per point and
        // reused across the chunk's pair walk.
        std::size_t maxPairFuncs = 0;

        for (const std::size_t pairIdx : chunkPairs)
        {
            maxPairFuncs = std::max(maxPairFuncs, store[pairIdx].nFuncs);
        }

        backend.ParallelFor(nPoints, [&](std::size_t p) {
            // The per-point state: the point center as the one-entry center
            // vector BuildNuclearPair consumes, and the pair block scratch,
            // sized once to the widest pair of the chunk.
            std::vector<Eigen::Vector3d> center{
                Eigen::Vector3d(points(p, 0), points(p, 1), points(p, 2))};
            std::vector<double> block(maxPairFuncs);

            // The accumulator is the point's OWN cell, carried across the
            // chunk loop - deliberately not a per-chunk local summed in at
            // the chunk's end: that second rounding (a chunk partial sum,
            // then the add) would make the chunked walk's arithmetic differ
            // from the unchunked walk's, whose single accumulator added every
            // term in pair order. Accumulating the cell in place adds the
            // very same terms in the very same order, so the values are
            // bit-identical to the pre-chunking build - which is what
            // OneElectronTest.ElectronPotentialAtPointsIsBitIdenticalWithChunkedPairData
            // holds, on a fixture whose store needs more than one chunk.
            double& value = values[p];

            for (const std::size_t pairIdx : chunkPairs)
            {
                const ShellPairIndex& pairIndex = pairList->pairs[pairIdx];
                const internal::MdPairData& pair = store[pairIdx];
                const std::size_t oI = pairList->shells[pairIndex.i].functionOffset;
                const std::size_t oJ = pairList->shells[pairIndex.j].functionOffset;
                const bool diagonal = pairIndex.i == pairIndex.j;
                internal::BuildNuclearPair(pair, charges, center, block.data());

                if (diagonal)
                {
                    // The block covers the full shell square; contracting the
                    // mirrored sum would double it.
                    for (std::size_t fa = 0; fa < pair.nFuncsA; ++fa)
                    {
                        for (std::size_t fb = 0; fb < pair.nFuncsB; ++fb)
                        {
                            value += density(oI + fa, oJ + fb) * block[fa * pair.nFuncsB + fb];
                        }
                    }
                } else
                {
                    // block = -<u|1/|r - p||v> (the nuclear-attraction sign),
                    // so V_elec = -sum D <1/r> = +sum D block. The canonical
                    // pair covers the (i, j) region only; the mirrored density
                    // supplies the (j, i) region (block is symmetric).
                    for (std::size_t fa = 0; fa < pair.nFuncsA; ++fa)
                    {
                        for (std::size_t fb = 0; fb < pair.nFuncsB; ++fb)
                        {
                            value += (density(oI + fa, oJ + fb) + density(oJ + fb, oI + fa)) *
                                     block[fa * pair.nFuncsB + fb];
                        }
                    }
                }
            }
        });

        internal::ReleaseChunkPairData(store, chunkPairs);
        start = end;
    }

    return values;
}

qcx::Result<std::array<CpuTensor2, 3>> BuildDipoleMatrix(const qcx::molecule::Molecule& molecule,
                                                         const qcx::basisset::BasisSet& basisSet,
                                                         const std::array<double, 3>& origin) {
    auto buildAxis = [&molecule, &basisSet, &origin](int axis) {
        return BuildPairMatrix(
            molecule, basisSet, [axis, &origin](const internal::MdPairData& pair, double* block) {
                internal::BuildDipolePair(pair, axis, origin, block);
            });
    };
    auto x = buildAxis(0);

    if (!x.has_value())
    {
        return std::unexpected(x.error());
    }

    auto y = buildAxis(1);

    if (!y.has_value())
    {
        return std::unexpected(y.error());
    }

    auto z = buildAxis(2);

    if (!z.has_value())
    {
        return std::unexpected(z.error());
    }

    // Aggregated from the individual results: CpuTensor2 is not
    // default-constructible, so std::array cannot be default-built first.
    return std::array<CpuTensor2, 3>{std::move(*x), std::move(*y), std::move(*z)};
}

qcx::Result<std::array<CpuTensor2, 6>> BuildQuadrupoleMatrix(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const std::array<double, 3>& origin) {
    // The upper triangle xx, xy, xz, yy, yz, zz: the kl and lk integrals
    // coincide ((r_k - R_k)(r_l - R_l) commutes), so only six components
    // are built (the quadrupole-symmetrization fix).
    constexpr std::array<std::pair<int, int>, 6> kComponents = {
        {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}}};
    auto buildComponent = [&molecule, &basisSet, &origin](int axisK, int axisL) {
        return BuildPairMatrix(
            molecule,
            basisSet,
            [axisK, axisL, &origin](const internal::MdPairData& pair, double* block) {
                internal::BuildQuadrupolePair(pair, axisK, axisL, origin, block);
            });
    };
    auto xx = buildComponent(0, 0);

    if (!xx.has_value())
    {
        return std::unexpected(xx.error());
    }

    auto xy = buildComponent(0, 1);

    if (!xy.has_value())
    {
        return std::unexpected(xy.error());
    }

    auto xz = buildComponent(0, 2);

    if (!xz.has_value())
    {
        return std::unexpected(xz.error());
    }

    auto yy = buildComponent(1, 1);

    if (!yy.has_value())
    {
        return std::unexpected(yy.error());
    }

    auto yz = buildComponent(1, 2);

    if (!yz.has_value())
    {
        return std::unexpected(yz.error());
    }

    auto zz = buildComponent(2, 2);

    if (!zz.has_value())
    {
        return std::unexpected(zz.error());
    }

    return std::array<CpuTensor2, 6>{std::move(*xx),
                                     std::move(*xy),
                                     std::move(*xz),
                                     std::move(*yy),
                                     std::move(*yz),
                                     std::move(*zz)};
}

} // namespace qcx::integrals
