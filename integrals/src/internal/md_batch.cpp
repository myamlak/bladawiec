// The batch machinery of the matrix-form MD engine: pair-data construction
// (the Hermite transforms of md_hermite.hpp), the class-batch assembly with
// 8-fold canonicalization, the dispatch runs, and the class-spec table
// lookup (md_dispatch_gen.hpp, tools/gen_md_tables.py).

#include "internal/md_batch.hpp"

#include "internal/footprint.hpp"
#include "internal/md_dispatch_gen.hpp"
#include "internal/md_engine.hpp"
#include "internal/md_hermite.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/memory/first_touch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <numbers>
#include <utility>
#include <vector>

namespace qcx::integrals::internal {

qcx::Result<std::vector<MdShellInput>> FlattenShells(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::integrals::ShellPairList& pairList) {
    std::vector<MdShellInput> shells;
    shells.reserve(pairList.shells.size());
    const auto& atoms = molecule.Atoms();
    const auto& coordinates = molecule.CoordinatesBohr();

    for (const qcx::integrals::ShellInfo& info : pairList.shells)
    {
        const qcx::basisset::ElementBasis* element =
            basisSet.Find(atoms[info.atomIndex].atomicNumber);

        if (element == nullptr)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "basis set has no entry for " + atoms[info.atomIndex].symbol});
        }

        const qcx::basisset::Shell& shell = element->shells[info.elementShellIndex];
        MdShellInput input;
        input.contractions.angularMomentum = info.angularMomentum;
        input.contractions.isSpherical = info.isSpherical;
        input.contractions.rows = info.contractionCount;
        input.contractions.exponents = shell.exponents;
        input.contractions.normalized.reserve(shell.coefficients.size());
        input.cx = coordinates(info.atomIndex, 0);
        input.cy = coordinates(info.atomIndex, 1);
        input.cz = coordinates(info.atomIndex, 2);

        for (const std::vector<double>& row : shell.coefficients)
        {
            std::vector<double> normalized;
            normalized.reserve(shell.exponents.size());

            for (std::size_t primitive = 0; primitive < shell.exponents.size(); ++primitive)
            {
                const double exponent = shell.exponents[primitive];
                // (2a/pi)^(3/4) (the radial normalization convention) times
                // the solid-harmonic unit normalization of md_defs.hpp (the
                // standard QC normalization contract the basis-set
                // coefficients assume).
                // Cartesian shells keep the radial factor only for now.
                const double solid =
                    info.isSpherical ? SolidNormalization(info.angularMomentum, exponent) : 1.0;
                normalized.push_back(row[primitive] *
                                     std::pow(2.0 * exponent / std::numbers::pi, 0.75) * solid);
            }

            input.contractions.normalized.push_back(std::move(normalized));
        }

        shells.push_back(std::move(input));
    }

    return shells;
}

void BuildContractedPairTransform(MdPairData& pair,
                                  const MdShellContractions& shellA,
                                  const MdShellContractions& shellB,
                                  const std::array<double, 3>& centerA,
                                  const std::array<double, 3>& centerB) {
    const int la = shellA.angularMomentum;
    const int lb = shellB.angularMomentum;
    const std::size_t rowsA = shellA.rows;
    const std::size_t rowsB = shellB.rows;
    const std::size_t nAngA =
        static_cast<std::size_t>(shellA.isSpherical ? SphericalCount(la) : CartesianCount(la));
    const std::size_t nAngB =
        static_cast<std::size_t>(shellB.isSpherical ? SphericalCount(lb) : CartesianCount(lb));
    const std::size_t nFuncsA = rowsA * nAngA;
    const std::size_t nFuncsB = rowsB * nAngB;
    const int nHerm = Hermite3DCount(la + lb);
    pair.nFuncs = nFuncsA * nFuncsB;
    pair.nFuncsA = nFuncsA;
    pair.nFuncsB = nFuncsB;
    pair.rowsA = rowsA;
    pair.rowsB = rowsB;
    pair.rowPairs = rowsA * rowsB;

    const double rAb2 = (centerA[0] - centerB[0]) * (centerA[0] - centerB[0]) +
                        (centerA[1] - centerB[1]) * (centerA[1] - centerB[1]) +
                        (centerA[2] - centerB[2]) * (centerA[2] - centerB[2]);
    const std::size_t nPrimPairs = shellA.exponents.size() * shellB.exponents.size();
    // The per-prim-pair elements are re-fitted, never re-created: resize
    // keeps the elements (their vector headers included) when the pair is
    // rebuilt with the same primitive counts, and with them the three
    // per-axis E-table vectors each carries (PerAxisETable::Reset re-uses
    // their storage). The repeat build of a pair is the norm on the chunk
    // arena - the store's prims survive the teardown (ClearChunkPairData) -
    // so this is what keeps the ~3 x nPrimPairs heap operations of a repeat
    // build out of the flush loop - measured at roughly 1.5x the operation
    // count of the pair-vector retention. Every field of
    // every element is rewritten below - the six geometry fields here, the
    // tables by the loop's BuildPerAxisTables call - so a reused element
    // cannot carry a stale value into the transforms.
    pair.primPairs.resize(nPrimPairs);
    pair.braWeights.resize(nPrimPairs);
    pair.ketTransforms.resize(nPrimPairs);
    // The bra transform: nFuncs x (rowPairs x primPairs x nHerm) - the
    // angular fold per prim pair weighted by d_a d_b at the matching
    // row-pair slot. The VRR's pq rows are per prim pair and UNWEIGHTED,
    // so the bra contraction happens exactly once, in this GEMM.
    pair.braTransform.assign(
        pair.nFuncs * pair.rowPairs * nPrimPairs * static_cast<std::size_t>(nHerm), 0.0);

    // The folded angular-only E slice per primitive pair (scratch).
    std::vector<double> folded;

    // The ket-derivative sign d_Q^u [0] = (-1)^|u| d_P^u [0]: the folded
    // ket transform entries carry (-1)^(tx+ty+tz) with u the ket's 3D
    // Hermite index (the 2026-08-18 odd-ket-degree regression - the fast
    // path never applied it while the unfolded reference and the mpmath
    // grids do; the tier parity is the |u| parity because the linear
    // index is total-major). The bra transform carries no sign (pure P
    // derivatives).
    std::vector<double> ketSign(static_cast<std::size_t>(nHerm), 1.0);

    for (int t = 0; t < nHerm; ++t)
    {
        int n = 0;

        while (n < 2 * kMaxShellL && kH2Prefix[n + 1] <= t)
        {
            ++n;
        }

        ketSign[static_cast<std::size_t>(t)] = (n % 2 == 0) ? 1.0 : -1.0;
    }

    std::size_t primIdx = 0;

    for (std::size_t primA = 0; primA < shellA.exponents.size(); ++primA)
    {
        const double a = shellA.exponents[primA];

        for (std::size_t primB = 0; primB < shellB.exponents.size(); ++primB)
        {
            const double b = shellB.exponents[primB];
            const double p = a + b;
            const double px = (a * centerA[0] + b * centerB[0]) / p;
            const double py = (a * centerA[1] + b * centerB[1]) / p;
            const double pz = (a * centerA[2] + b * centerB[2]) / p;
            const std::array<double, 3> shiftA = {
                px - centerA[0], py - centerA[1], pz - centerA[2]};
            const std::array<double, 3> shiftB = {
                px - centerB[0], py - centerB[1], pz - centerB[2]};

            MdPrimPair& prim = pair.primPairs[primIdx];
            prim.p = p;
            prim.exponentA = a;
            prim.prefactor = std::exp(-(a * b / p) * rAb2);
            prim.px = px;
            prim.py = py;
            prim.pz = pz;
            // In place on the pair's own tables: the re-fit keeps their
            // vectors (the copy a returned array would cost is gone too).
            BuildPerAxisTables(la, lb, shiftA, shiftB, p, prim.perAxisTables);

            // The row-pair contraction weights (CWR part 1).
            std::vector<double>& weights = pair.braWeights[primIdx];
            weights.assign(rowsA * rowsB, 0.0);

            for (std::size_t rowA = 0; rowA < rowsA; ++rowA)
            {
                for (std::size_t rowB = 0; rowB < rowsB; ++rowB)
                {
                    weights[rowA * rowsB + rowB] =
                        shellA.normalized[rowA][primA] * shellB.normalized[rowB][primB];
                }
            }

            // The folded transform of this primitive pair; the bra transform
            // stores the weighted fold at the per-prim-pair slot, the ket
            // transform stores the transposed orientation with the weights
            // folded in (the ket contraction has no other site).
            FoldPairETable(
                la, lb, prim.perAxisTables, shellA.isSpherical, shellB.isSpherical, nHerm, folded);
            pair.ketTransforms[primIdx].assign(static_cast<std::size_t>(nHerm) * pair.nFuncs, 0.0);

            for (std::size_t rowA = 0; rowA < rowsA; ++rowA)
            {
                for (std::size_t rowB = 0; rowB < rowsB; ++rowB)
                {
                    const double weight = weights[rowA * rowsB + rowB];
                    const std::size_t rowAB = rowA * rowsB + rowB;

                    for (std::size_t fb = 0; fb < nAngB; ++fb)
                    {
                        for (std::size_t fa = 0; fa < nAngA; ++fa)
                        {
                            const std::size_t fRow =
                                (rowB * nAngB + fb) * nFuncsA + rowA * nAngA + fa;

                            for (int t = 0; t < nHerm; ++t)
                            {
                                const double value = folded[(fa * nAngB + fb) * nHerm + t];
                                pair.braTransform[(fRow * pair.rowPairs + rowAB) * nPrimPairs *
                                                      static_cast<std::size_t>(nHerm) +
                                                  primIdx * static_cast<std::size_t>(nHerm) +
                                                  static_cast<std::size_t>(t)] += value * weight;
                                pair.ketTransforms[primIdx]
                                                  [static_cast<std::size_t>(t) * pair.nFuncs +
                                                   fRow] +=
                                    value * weight * ketSign[static_cast<std::size_t>(t)];
                            }
                        }
                    }
                }
            }

            ++primIdx;
        }
    }

    // The certified bound helpers: max |.|-row-sum of the contracted bra transform
    // (over the row-pair and prim-pair slots) and max |.|-column-sum of the
    // ket transforms summed over primitives.
    double braRowSum = 0.0;

    for (std::size_t f = 0; f < pair.nFuncs; ++f)
    {
        double sum = 0.0;

        for (std::size_t rowAB = 0; rowAB < pair.rowPairs; ++rowAB)
        {
            for (std::size_t b = 0; b < pair.primPairs.size(); ++b)
            {
                for (int t = 0; t < nHerm; ++t)
                {
                    sum += std::abs(
                        pair.braTransform[(((f * pair.rowPairs + rowAB) * pair.primPairs.size()) +
                                           b) *
                                              static_cast<std::size_t>(nHerm) +
                                          static_cast<std::size_t>(t)]);
                }
            }
        }

        braRowSum = std::max(braRowSum, sum);
    }

    double ketColSum = 0.0;

    for (std::size_t f = 0; f < pair.nFuncs; ++f)
    {
        double sum = 0.0;

        for (const std::vector<double>& transform : pair.ketTransforms)
        {
            for (int t = 0; t < nHerm; ++t)
            {
                sum += std::abs(transform[static_cast<std::size_t>(t) * pair.nFuncs + f]);
            }
        }

        ketColSum = std::max(ketColSum, sum);
    }

    pair.braRowSum = braRowSum;
    pair.ketColSum = ketColSum;
}

qcx::Result<std::vector<MdPairData>> BuildPairData(const qcx::molecule::Molecule& molecule,
                                                   const qcx::basisset::BasisSet& basisSet,
                                                   const qcx::integrals::ShellPairList& pairList) {
    auto shells = FlattenShells(molecule, basisSet, pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    std::vector<MdPairData> pairStore;
    pairStore.reserve(pairList.pairs.size());
    // Place the reserved capacity across the consuming team before the
    // serial fill constructs the elements: an anonymous page stays on the
    // NUMA node of the thread that first accesses it, and every screening
    // and contraction thread reads this store each pass. The pass writes
    // raw bytes into the reserved storage - the fill below is the first
    // element access either way.
    qcx::memory::TouchPagesAcrossTeam(pairStore.data(), pairStore.capacity() * sizeof(MdPairData));

    for (const qcx::integrals::ShellPairIndex& pairIndex : pairList.pairs)
    {
        const MdShellInput& shellA = (*shells)[pairIndex.i];
        const MdShellInput& shellB = (*shells)[pairIndex.j];
        const std::array<double, 3> centerA = {shellA.cx, shellA.cy, shellA.cz};
        const std::array<double, 3> centerB = {shellB.cx, shellB.cy, shellB.cz};
        MdPairData pair;
        pair.la = shellA.contractions.angularMomentum;
        pair.lb = shellB.contractions.angularMomentum;
        pair.isSphericalA = shellA.contractions.isSpherical;
        pair.isSphericalB = shellB.contractions.isSpherical;
        pair.ax = shellA.cx;
        pair.ay = shellA.cy;
        pair.az = shellA.cz;
        pair.bx = shellB.cx;
        pair.by = shellB.cy;
        pair.bz = shellB.cz;
        BuildContractedPairTransform(
            pair, shellA.contractions, shellB.contractions, centerA, centerB);
        pairStore.push_back(std::move(pair));
    }

    return pairStore;
}

void FillPairGeometry(std::vector<MdPairData>& store,
                      const std::vector<MdShellInput>& shells,
                      const qcx::integrals::ShellPairList& pairList) {
    for (std::size_t pairIndex = 0; pairIndex < pairList.pairs.size(); ++pairIndex)
    {
        const qcx::integrals::ShellPairIndex& index = pairList.pairs[pairIndex];
        const MdShellInput& shellA = shells[index.i];
        const MdShellInput& shellB = shells[index.j];
        MdPairData& pair = store[pairIndex];
        pair.la = shellA.contractions.angularMomentum;
        pair.lb = shellB.contractions.angularMomentum;
        pair.isSphericalA = shellA.contractions.isSpherical;
        pair.isSphericalB = shellB.contractions.isSpherical;
        pair.ax = shellA.cx;
        pair.ay = shellA.cy;
        pair.az = shellA.cz;
        pair.bx = shellB.cx;
        pair.by = shellB.cy;
        pair.bz = shellB.cz;
    }
}

void BuildChunkPairData(std::vector<MdPairData>& store,
                        const std::vector<MdShellInput>& shells,
                        const qcx::integrals::ShellPairList& pairList,
                        const std::vector<std::size_t>& pairs) {
    for (const std::size_t pairIndex : pairs)
    {
        MdPairData& pair = store[pairIndex];

        // The not-built marker (the ClearChunkPairData contract): a pair
        // whose braTransform is empty was not built. On the machinery's
        // chunk pass the marker is off by construction here (the chunk
        // lists every chunk pair once; the class-orbit representatives can
        // coincide with the chunk's own pairs), so that guard exists for
        // the clear side, which must never wipe a pair the chunk did not
        // build. The lean builder's flush pass is the opposite caller: it
        // never tears down, so the marker holds across its flushes and this
        // guard is its skip (lean_fock_build.cpp). The two callers each
        // keep their own teardown choice and must stay consistent with it.
        if (!pair.braTransform.empty())
        {
            continue;
        }

        const MdShellInput& shellA = shells[pairList.pairs[pairIndex].i];
        const MdShellInput& shellB = shells[pairList.pairs[pairIndex].j];
        const std::array<double, 3> centerA = {shellA.cx, shellA.cy, shellA.cz};
        const std::array<double, 3> centerB = {shellB.cx, shellB.cy, shellB.cz};
        // The verbatim FastPath builder: the values are bit-identical to
        // the full store's by construction (the same function, the same
        // inputs, the same pair order) - the pairStore-seam contract. The
        // geometry fields the light store already carries (la/lb/
        // isSpherical/centers) are not rewritten here; the builder fills
        // the computed fields (nFuncs, rows, rowPairs) and the transforms.
        BuildContractedPairTransform(
            pair, shellA.contractions, shellB.contractions, centerA, centerB);
    }
}

void ClearChunkPairData(std::vector<MdPairData>& store, const std::vector<std::size_t>& pairs) {
    for (const std::size_t pairIndex : pairs)
    {
        MdPairData& pair = store[pairIndex];

        if (pair.braTransform.empty())
        {
            continue;
        }

        // The chunk arena teardown: the flat bra transform and the certified sums
        // are cleared so the next chunk's build starts clean; the geometry
        // fields survive (the chunk pass never rewrites them). The flat
        // clear() is load-bearing twice over: it frees the chunk's largest
        // payload, and it is the not-built marker the chunk pass tests
        // (BuildChunkPairData above).
        //
        // The per-prim-pair containers are deliberately NOT cleared - the
        // elements, their nested braWeights/ketTransforms vectors and the
        // three per-axis E-table vectors each element carries are the arena
        // the pair builder re-uses. BuildContractedPairTransform re-fits
        // every one of them (resize/re-fill/assign rewrite every slot of
        // every prim pair, so no stale value survives - the not-built
        // marker is braTransform alone, never primPairs.size()), and that
        // keeps the ~2 x nPrimPairs heap operations of the pair build plus
        // the ~3 x nPrimPairs of the per-axis E tables off every repeat
        // build of the same pair. The retention is bounded by the built
        // pairs' own payloads - the store is per-WINDOW on the lean path
        // (each window builds its own copy of the template) and per-builder
        // on the machinery path - so it plateaus, never grows
        // (the arena is capacity-preserved by contract: the reused
        // containers are not Reservations, so per-chunk alloc/free cannot
        // stack under cumulative high-water accounting).
        pair.braTransform.clear();
        pair.braRowSum = 0.0;
        pair.ketColSum = 0.0;
    }
}

void ReleaseChunkPairData(std::vector<MdPairData>& store, const std::vector<std::size_t>& pairs) {
    for (const std::size_t pairIndex : pairs)
    {
        MdPairData& pair = store[pairIndex];

        // The build-once teardown (md_batch.hpp): swap-with-empty frees the
        // containers outright instead of retaining their capacities the way
        // ClearChunkPairData above deliberately does. The emptied
        // braTransform is the not-built marker both teardowns leave.
        std::vector<MdPrimPair>().swap(pair.primPairs);
        std::vector<std::vector<double>>().swap(pair.braWeights);
        std::vector<double>().swap(pair.braTransform);
        std::vector<std::vector<double>>().swap(pair.ketTransforms);
        pair.braRowSum = 0.0;
        pair.ketColSum = 0.0;
    }
}

std::size_t ChunkPairPayloadBytes(const qcx::molecule::Molecule& molecule,
                                  const qcx::basisset::BasisSet& basisSet,
                                  const qcx::integrals::ShellPairList& pairList,
                                  std::size_t pairIndex) {
    if (pairIndex >= pairList.pairs.size())
    {
        return 0;
    }

    const qcx::integrals::ShellInfo& a = pairList.shells[pairList.pairs[pairIndex].i];
    const qcx::integrals::ShellInfo& b = pairList.shells[pairList.pairs[pairIndex].j];
    const std::size_t nPrimA = ShellPrimitiveCount(molecule, basisSet, a);
    const std::size_t nPrimB = ShellPrimitiveCount(molecule, basisSet, b);

    return PairStoreBytesPerPair(a.angularMomentum,
                                 b.angularMomentum,
                                 a.isSpherical,
                                 b.isSpherical,
                                 a.contractionCount,
                                 b.contractionCount,
                                 nPrimA * nPrimB) -
           sizeof(MdPairData);
}

// The streaming-rung partial: the per-task
// batch metrics shared by the eager AssembleClassBatches and the
// MdClassBatchCursor's incremental partition, so the two batch builders
// cannot drift. The Item entry one batch walk carries per task.
struct Item {
    qcx::integrals::ShellQuartet canonical;
    int lBra;
    int lKet;
    std::size_t ketPair;
    std::size_t braPair;
    std::size_t braRowPairs; ///< The bra pair's row-pair count (group key).
    std::size_t outputSize; ///< Block elements (fp64 count).
    std::size_t scratchSize; ///< pq + acc elements (fp64 count).
    std::size_t convertSize; ///< The kernel's per-batch convert workspace (fp64 count).
};

/// The per-task payload sizes of one canonical task (fp64 element counts).
struct TaskPayloadSizes {
    std::size_t outputSize;
    std::size_t scratchSize;
    std::size_t convertSize;
};

/// The single payload-arithmetic source of the batch builders.
inline TaskPayloadSizes PayloadSizesOf(const MdPairData& bra,
                                       const MdPairData& ket,
                                       int lBra,
                                       int lKet) noexcept {
    const std::size_t hermBra = static_cast<std::size_t>(Hermite3DCount(lBra));
    const std::size_t hermKet = static_cast<std::size_t>(Hermite3DCount(lKet));
    const std::size_t mPq = bra.rowPairs * bra.primPairs.size() * hermBra;
    // The kernel's convert buffer is sized by the max of the two transform
    // workspaces (md_vrr.hpp: kHermKet x nFuncsKet per g, mBra x kBra in
    // pass 2).
    const std::size_t convertSize = std::max(hermKet * ket.nFuncs, bra.nFuncs * mPq);

    return TaskPayloadSizes{bra.nFuncs * ket.nFuncs, mPq * hermKet + mPq * ket.nFuncs, convertSize};
}

/// The per-item byte mass of one task: the output/scratch/convert payload
/// plus the Item entry, its canonical ShellQuartet copy (the computed
/// buffer), the per-task MdQuartetTask and the per-task offset - the 152 B
/// per item on top of the payload the batch cap bounds with.
inline std::size_t TaskItemBytes(std::size_t outputSize,
                                 std::size_t scratchSize,
                                 std::size_t convertSize) noexcept {
    return (outputSize + scratchSize + convertSize) * sizeof(double) + sizeof(Item) +
           sizeof(qcx::integrals::ShellQuartet) + sizeof(MdQuartetTask) + sizeof(std::size_t);
}

qcx::Result<std::vector<MdClassBatch>> AssembleClassBatches(
    const std::vector<MdPairData>& pairStore,
    const qcx::integrals::ShellPairList& pairList,
    const std::vector<qcx::integrals::ShellQuartet>& quartets,
    std::size_t maxBatchBytes,
    std::vector<qcx::integrals::ShellQuartet>& computed) {
    if (maxBatchBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxBatchBytes must be positive"});
    }

    // The canonicalization and class-major ordering live in the public
    // CanonicalizeQuartetOrder - the single implementation shared
    // with the storage decorator, so a served reload matches the engine
    // byte for byte.
    auto ordered = qcx::integrals::CanonicalizeQuartetOrder(pairList, quartets);

    if (!ordered.has_value())
    {
        return std::unexpected(ordered.error());
    }

    std::vector<Item> items;
    items.reserve(ordered->size());

    for (const qcx::integrals::CanonicalQuartetInfo& info : *ordered)
    {
        const MdPairData& bra = pairStore[info.braPair];
        const MdPairData& ket = pairStore[info.ketPair];
        // The payload sizes ride the shared per-task metrics (the file-scope
        // helpers above) - the single arithmetic source the incremental
        // MdClassBatchCursor replicates, so the two batch builders cannot
        // drift.
        const TaskPayloadSizes payload = PayloadSizesOf(bra, ket, info.lBra, info.lKet);
        items.push_back(Item{info.quartet,
                             info.lBra,
                             info.lKet,
                             info.ketPair,
                             info.braPair,
                             info.braRowPairs,
                             payload.outputSize,
                             payload.scratchSize,
                             payload.convertSize});
    }

    computed.clear();
    computed.reserve(items.size());

    for (const Item& item : items)
    {
        computed.push_back(item.canonical);
    }

    // Split each class run into batches capped by output + scratch bytes.
    // The output blocks of one batch are contiguous in task order; the
    // caller points outF64/outF32 at the batch's base.
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
            // conservative so the cap bounds the whole per-batch allocation
            // (mirrors the RI path in md_vrr_3c.hpp). The batch's sibling
            // allocations ride the cap too (the n34b charge; TaskItemBytes
            // above), so the cap bounds the batch's entire per-batch
            // memory, not just the values (the batch boundary shifts are
            // value-neutral: the output blocks stay contiguous in task
            // order with the same per-task offsets).
            const std::size_t itemBytes =
                TaskItemBytes(items[t].outputSize, items[t].scratchSize, items[t].convertSize);

            if (batchBytes + itemBytes > maxBatchBytes && t > batchStart)
            {
                MdClassBatch batch;
                batch.lBra = items[batchStart].lBra;
                batch.lKet = items[batchStart].lKet;
                batch.pairStore = &pairStore;
                batch.boundsBase = batchStart;
                std::size_t outputOffset = 0;

                for (std::size_t s = batchStart; s < t; ++s)
                {
                    batch.tasks.push_back(
                        MdQuartetTask{items[s].braPair, items[s].ketPair, outputOffset});
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
        batch.pairStore = &pairStore;
        batch.boundsBase = batchStart;
        std::size_t outputOffset = 0;

        for (std::size_t s = batchStart; s < runEnd; ++s)
        {
            batch.tasks.push_back(MdQuartetTask{items[s].braPair, items[s].ketPair, outputOffset});
            outputOffset += items[s].outputSize;
        }

        batches.push_back(std::move(batch));
        runStart = runEnd;
    }

    return batches;
}

// The streaming-rung partial: the
// canonical-sorted masters and the incremental batch partition.

MdCanonicalTaskForm CanonicalTaskFormOf(const qcx::integrals::ShellPairList& pairList,
                                        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                        std::size_t braPair,
                                        std::size_t ketPair) noexcept {
    // The role rules of CanonicalizeQuartetOrder applied to a task's
    // pair roles: the canonical bra pair is the numerically larger pair
    // index (rule 1 - the screened lists and orbit reps already arrive
    // bra >= ket, so that swap never fires on them), then the L-role swap
    // makes the canonical bra pair the lower-L side. The pair totals are
    // the shells' angular-momentum sums (la + lb of each pair).
    std::size_t bra = braPair;
    std::size_t ket = ketPair;

    if (bra < ket)
    {
        std::swap(bra, ket);
    }

    const qcx::integrals::ShellPairIndex& braShells = pairList.pairs[bra];
    const qcx::integrals::ShellPairIndex& ketShells = pairList.pairs[ket];
    int lBra =
        pairList.shells[braShells.i].angularMomentum + pairList.shells[braShells.j].angularMomentum;
    int lKet =
        pairList.shells[ketShells.i].angularMomentum + pairList.shells[ketShells.j].angularMomentum;

    if (lBra > lKet)
    {
        std::swap(bra, ket);
        std::swap(lBra, lKet);
    }

    return MdCanonicalTaskForm{bra, ket, lBra, lKet};
}

// The five-key ascending comparator over two raw tasks' canonical forms
// - CanonicalizeQuartetOrder's exact sort order. The bra-row-pairs key is
// the canonical bra pair's shell contraction product (CanonicalQuartetInfo.
// braRowPairs; equals MdPairData.rowPairs by construction). Distinct
// canonical quartets never tie on all five keys, so the master order is
// deterministic.
inline bool CanonicalTaskLess(const qcx::integrals::ShellPairList& pairList,
                              const MdQuartetTask& a,
                              const MdQuartetTask& b) noexcept {
    const MdCanonicalTaskForm formA = CanonicalTaskFormOf(pairList, a.braPair, a.ketPair);
    const MdCanonicalTaskForm formB = CanonicalTaskFormOf(pairList, b.braPair, b.ketPair);

    if (formA.lKet != formB.lKet)
    {
        return formA.lKet < formB.lKet;
    }

    if (formA.lBra != formB.lBra)
    {
        return formA.lBra < formB.lBra;
    }

    if (formA.ketPair != formB.ketPair)
    {
        return formA.ketPair < formB.ketPair;
    }

    const qcx::integrals::ShellPairIndex& braShellsA = pairList.pairs[formA.braPair];
    const qcx::integrals::ShellPairIndex& braShellsB = pairList.pairs[formB.braPair];
    const std::size_t rowPairsA = pairList.shells[braShellsA.i].contractionCount *
                                  pairList.shells[braShellsA.j].contractionCount;
    const std::size_t rowPairsB = pairList.shells[braShellsB.i].contractionCount *
                                  pairList.shells[braShellsB.j].contractionCount;

    if (rowPairsA != rowPairsB)
    {
        return rowPairsA < rowPairsB;
    }

    return formA.braPair < formB.braPair;
}

void SortScreenedTasks(const qcx::integrals::ShellPairList& pairList,
                       std::vector<MdQuartetTask>& tasks) {
    if (tasks.size() < 2)
    {
        return;
    }

    std::sort(
        tasks.begin(), tasks.end(), [&pairList](const MdQuartetTask& a, const MdQuartetTask& b) {
            return CanonicalTaskLess(pairList, a, b);
        });
}

void SortScreenedTasks(const qcx::integrals::ShellPairList& pairList,
                       std::vector<MdQuartetTask>& tasks,
                       std::vector<double>& weights) {
    const std::size_t n = tasks.size();

    if (n < 2)
    {
        return;
    }

    // The weights are parallel to the tasks by construction (the screening
    // pushes both in lockstep), so one permutation serves both. The index
    // vector is the fp32 sort's whole transient - 8 B x n, freed before the
    // batch walks run.
    std::vector<std::size_t> order(n);

    for (std::size_t p = 0; p < n; ++p)
    {
        order[p] = p;
    }

    std::sort(order.begin(), order.end(), [&pairList, &tasks](std::size_t a, std::size_t b) {
        return CanonicalTaskLess(pairList, tasks[a], tasks[b]);
    });

    // Applies the permutation in place, using the transient order as its own
    // visited marker: position p must receive the element that started at
    // order[p] (source semantics). Following the cycle p -> order[p] -> ...
    // shifts each element one step toward its destination and marks the
    // cycle's nodes (order[x] = x), so later positions of the same cycle are
    // skipped. Tasks and weights move in lockstep in the single walk.
    for (std::size_t p = 0; p < n; ++p)
    {
        if (order[p] == p)
        {
            continue;
        }

        MdQuartetTask heldTask = tasks[p];
        double heldWeight = weights[p];
        std::size_t cur = p;

        for (;;)
        {
            const std::size_t src = order[cur];

            if (src == p)
            {
                break;
            }

            tasks[cur] = tasks[src];
            weights[cur] = weights[src];
            order[cur] = cur;
            cur = src;
        }

        tasks[cur] = heldTask;
        weights[cur] = heldWeight;
        order[cur] = cur;
    }
}

MdClassBatchCursor::MdClassBatchCursor(const std::vector<MdPairData>& pairStore,
                                       const qcx::integrals::ShellPairList& pairList,
                                       const std::vector<MdQuartetTask>& tasks,
                                       std::size_t maxBatchBytes) :
    _pairStore(pairStore), _pairList(pairList), _tasks(tasks), _maxBatchBytes(maxBatchBytes) {}

void MdClassBatchCursor::Reset() {
    _runEnd = 0;
    _batchStart = 0;
    _pos = 0;
    _batchBytes = 0;
}

bool MdClassBatchCursor::Next(MdClassBatch& out,
                              bool countOnly,
                              std::vector<std::size_t>* batchStarts) {
    for (;;)
    {
        if (_batchStart < _runEnd)
        {
            // Mid-run scan: examine positions until the cap cut fires or the
            // run boundary is reached. The cut rule and the byte
            // accumulation are AssembleClassBatches' verbatim (the run's
            // first position never cuts - t > batchStart - and a cut
            // position's bytes join the reopened batch).
            while (_pos < _runEnd)
            {
                const MdCanonicalTaskForm form =
                    CanonicalTaskFormOf(_pairList, _tasks[_pos].braPair, _tasks[_pos].ketPair);
                const MdPairData& bra = _pairStore[form.braPair];
                const MdPairData& ket = _pairStore[form.ketPair];
                const TaskPayloadSizes payload = PayloadSizesOf(bra, ket, form.lBra, form.lKet);
                const std::size_t itemBytes =
                    TaskItemBytes(payload.outputSize, payload.scratchSize, payload.convertSize);

                if (_batchBytes + itemBytes > _maxBatchBytes && _pos > _batchStart)
                {
                    if (batchStarts != nullptr)
                    {
                        batchStarts->push_back(_batchStart);
                    }

                    if (!countOnly)
                    {
                        FillBatch(out, _batchStart, _pos);
                    }

                    _batchStart = _pos;
                    _batchBytes = itemBytes;
                    ++_pos;
                    return true;
                }

                _batchBytes += itemBytes;
                ++_pos;
            }

            // The run boundary - or a cap cut that consumed the run's FINAL
            // position (the cut's reopened batch [batchStart, runEnd) holds
            // that position, and _pos already reached _runEnd): the tail
            // batch [batchStart, runEnd) is owed (never empty - the run's
            // last position always joined a batch). The entry condition is
            // _batchStart < _runEnd, not _pos < _runEnd, so the owed tail
            // survives the cut's early return; a dropped tail would leave
            // that position's quartet out of every emitted batch.
            if (batchStarts != nullptr)
            {
                batchStarts->push_back(_batchStart);
            }

            if (!countOnly)
            {
                FillBatch(out, _batchStart, _runEnd);
            }

            _batchStart = _runEnd;
            return true;
        }

        if (_pos >= _tasks.size())
        {
            return false;
        }

        // Open the next class run: equal (lBra, lKet) canonical forms are
        // contiguous under the canonical sort, so the run's extent is a
        // forward scan from the current position.
        const MdCanonicalTaskForm runClass =
            CanonicalTaskFormOf(_pairList, _tasks[_pos].braPair, _tasks[_pos].ketPair);
        _runEnd = _pos + 1;

        while (_runEnd < _tasks.size())
        {
            const MdCanonicalTaskForm form =
                CanonicalTaskFormOf(_pairList, _tasks[_runEnd].braPair, _tasks[_runEnd].ketPair);

            if (form.lBra != runClass.lBra || form.lKet != runClass.lKet)
            {
                break;
            }

            ++_runEnd;
        }

        _batchStart = _pos;
        _batchBytes = 0;
    }
}

void FillMdClassBatch(MdClassBatch& out,
                      const std::vector<MdPairData>& pairStore,
                      const qcx::integrals::ShellPairList& pairList,
                      const std::vector<MdQuartetTask>& tasks,
                      std::size_t from,
                      std::size_t to) {
    const MdCanonicalTaskForm runClass =
        CanonicalTaskFormOf(pairList, tasks[from].braPair, tasks[from].ketPair);
    out.lBra = runClass.lBra;
    out.lKet = runClass.lKet;
    out.pairStore = &pairStore;
    out.boundsBase = from;
    out.tasks.clear();
    out.tasks.reserve(to - from);
    std::size_t outputOffset = 0;

    for (std::size_t p = from; p < to; ++p)
    {
        const MdCanonicalTaskForm form =
            CanonicalTaskFormOf(pairList, tasks[p].braPair, tasks[p].ketPair);
        out.tasks.push_back(MdQuartetTask{form.braPair, form.ketPair, outputOffset});
        const MdPairData& bra = pairStore[form.braPair];
        const MdPairData& ket = pairStore[form.ketPair];
        outputOffset += bra.nFuncs * ket.nFuncs;
    }
}

void MdClassBatchCursor::FillBatch(MdClassBatch& out, std::size_t from, std::size_t to) const {
    FillMdClassBatch(out, _pairStore, _pairList, _tasks, from, to);
}

const MdClassFnF64* ClassSpecF64(int lBra, int lKet) noexcept {
    if (lBra < 0 || lKet < 0 || lBra > 2 * kMaxShellL || lKet > 2 * kMaxShellL)
    {
        return nullptr;
    }

    return &kMdClassTableF64[lBra][lKet];
}

const MdClassFnF32* ClassSpecF32(int lBra, int lKet) noexcept {
    if (lBra < 0 || lKet < 0 || lBra > 2 * kMaxShellL || lKet > 2 * kMaxShellL)
    {
        return nullptr;
    }

    return &kMdClassTableF32[lBra][lKet];
}

qcx::Result<void> RunBatches(const std::vector<MdClassBatch>& batches, std::size_t regionThreads) {
    // Validate the dispatch of every batch up front (the parallel body
    // cannot report errors).
    for (const MdClassBatch& batch : batches)
    {
        const MdClassFnF64* fn = ClassSpecF64(batch.lBra, batch.lKet);
        const bool fp64 = fn != nullptr && *fn != nullptr;
        const MdClassFnF32* fn32 = ClassSpecF32(batch.lBra, batch.lKet);
        const bool fp32 = fn32 != nullptr && *fn32 != nullptr;

        if ((batch.outF64 != nullptr && !fp64) || (batch.outF32 != nullptr && !fp32))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "no instantiated class kernel for this build's QcxIntegralsLMax"});
        }
    }

    qcx::backend::Backend<qcx::backend::CpuTag> backend;
    // The batch loop chunked over the OpenMP team with a FIXED static chunk
    // assignment (the scaling-audit contract). The batch vector is
    // class-major - AssembleClassBatches emits each (L_bra, L_ket) run
    // contiguously - so the static contiguous chunks ARE the per-class
    // chunk assignment. Bit-identity across team sizes holds by
    // construction: every batch writes only its own disjoint output block
    // (the caller points outF64/outF32 at each batch's base) and the certified
    // bounds occupy global per-task slots (boundsBase + t, one write per
    // task - md_vrr.hpp), so NO cross-batch accumulation exists for a
    // schedule to reorder; dynamic scheduling is deliberately avoided (the
    // schedule-ordered-accumulation trap) and costs
    // scheduler overhead per handoff at scale.
    backend.ParallelFor(
        batches.size(),
        [&batches](std::size_t i) {
            const MdClassBatch& batch = batches[i];
            const qcx::Result<void> result = batch.outF64 != nullptr
                                                 ? (*ClassSpecF64(batch.lBra, batch.lKet))(batch)
                                                 : (*ClassSpecF32(batch.lBra, batch.lKet))(batch);

            // The dispatch was validated up front and the kernels' remaining
            // failure modes are excluded by construction, so a failure here is
            // a programming or vendor-state bug - loud, not silent (the parallel
            // body cannot return it).
            if (!result.has_value())
            {
                std::terminate();
            }
        },
        static_cast<int>(regionThreads));

    return {};
}

std::size_t BatchElementCount(const MdClassBatch& batch) noexcept {
    std::size_t total = 0;

    for (const MdQuartetTask& task : batch.tasks)
    {
        total += (*batch.pairStore)[task.braPair].nFuncs * (*batch.pairStore)[task.ketPair].nFuncs;
    }

    return total;
}

bool EqualKetGroupStraddles(const MdClassBatch& before, const MdClassBatch& after) noexcept {
    if (before.lBra != after.lBra || before.lKet != after.lKet)
    {
        return false;
    }

    if (before.tasks.empty() || after.tasks.empty())
    {
        return false;
    }

    // The boundary tasks: the last of the earlier batch and the first of
    // the next. Same ket pair AND the same bra contraction structure (the
    // row-pair and prim-pair counts - the two quantities that fix the
    // bra-transform shapes of the ket reuse) make them one equal-ket group
    // (md_vrr.hpp's group walk).
    const MdQuartetTask& lastTask = before.tasks.back();
    const MdQuartetTask& firstTask = after.tasks.front();

    if (lastTask.ketPair != firstTask.ketPair)
    {
        return false;
    }

    const MdPairData& lastBra = (*before.pairStore)[lastTask.braPair];
    const MdPairData& firstBra = (*after.pairStore)[firstTask.braPair];
    return lastBra.rowPairs == firstBra.rowPairs &&
           lastBra.primPairs.size() == firstBra.primPairs.size();
}

std::size_t GpuSplitBatchIndex(const std::vector<MdClassBatch>& batches,
                               double deviceShare) noexcept {
    if (batches.empty())
    {
        return 0;
    }

    const double cpuShare = 1.0 - std::clamp(deviceShare, 0.0, 1.0);

    double totalMass = 0.0;

    for (const MdClassBatch& batch : batches)
    {
        totalMass += static_cast<double>(BatchElementCount(batch));
    }

    // One ascending pass over the boundaries, accumulating the prefix
    // mass; every group-clean boundary's distance to the target prefix
    // mass (cpuShare x total) is compared, and only a strictly smaller
    // distance replaces the incumbent - so the scan keeps the EARLIEST
    // boundary of an exact tie. Boundary 0 (empty prefix) is the
    // incumbent before the loop; boundary size (the full prefix - the
    // empty suffix, the CPU-only fallback at deviceShare 0) is compared
    // after the last batch; interior boundaries are skipped when an
    // equal-ket group straddles them - the split never cuts a group.
    double prefixMass = 0.0;
    std::size_t bestIndex = 0;
    double bestDistance = std::abs(cpuShare * totalMass);

    for (std::size_t i = 1; i <= batches.size(); ++i)
    {
        prefixMass += static_cast<double>(BatchElementCount(batches[i - 1]));

        if (i < batches.size() && EqualKetGroupStraddles(batches[i - 1], batches[i]))
        {
            continue;
        }

        const double distance = std::abs(prefixMass - cpuShare * totalMass);

        if (distance < bestDistance)
        {
            bestDistance = distance;
            bestIndex = i;
        }
    }

    return bestIndex;
}

} // namespace qcx::integrals::internal
