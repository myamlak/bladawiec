#pragma once

/// \file
/// Create-time footprint estimation for the workspace-budget mode selection:
/// exact byte formulas for the MD pair store (E tables, transforms,
/// weights), the neighbor pattern, the per-thread batch scratch and the
/// structural terms, evaluated from the
/// shell data available at Create(). The formulas mirror the engine's
/// allocation shapes - BuildContractedPairTransform in md_batch.cpp is the
/// ground truth - and are never-under by construction: every term models
/// what is actually allocated (payload bytes plus the std::vector 3-pointer
/// overhead and the struct sizes), so a mode decision made on this estimate
/// can never over-commit the workspace budget. With the filtered parse, any
/// pre-Create prediction that sums the parsed set equals the engine number
/// by construction. The per-builder decision
/// layer (the direct/RI/QFMM estimate composition, the mode selection and
/// the ladder refusal diagnostics) lives in this header too.
///
/// Unmodeled (documented): the QFMM state's retained far-field
/// pair vector - 16 B per
/// well-separated node pair, cardinality unknowable at the mode decision
/// (the interaction lists are built after it; every other retained QFMM
/// allocation is modeled in QfmmOuterStoreBytes below); and the GPU path
/// (eri_cuda.cpp): the host tables mirror the pair store (GB-class at
/// 5000-scale) and the device buffers are a separate domain, but
/// GpuJkFockBuilder has no workspace-budget seam - the pointer is ignored,
/// so nothing can be charged to the ladder's remaining - and the seam is
/// the GPU track's device-budget feature decision (the GPU
/// family's admission accounting stays the driver
/// gate's, whose base now carries the per-iteration working set). The
/// RI-J's FIRST-iteration terms are now MODELED (RiFirstIterationBytes,
/// folded into RiFootprintTerms::Total); the nested exchange half's
/// per-iteration output buffers are MODELED as the DirectFootprint
/// exchangePerCallBytes term - the premise that the
/// driver base's n^2 term covers them was refuted at C24H50, measured
/// ~12.6 GiB/call vs the base's ~21-170 MB; the base stays the driver-side
/// n^2 working-set term). The term
/// charges the per-batch bound: the per-batch
/// restructure bounds both lanes' per-call values at the batch cap, so
/// exchangePerCallBytes is the structural F64Bound + F32Live peak of the
/// values-term redesign, not the whole-call fitted mass.

#include "internal/md_batch.hpp"
#include "internal/md_defs.hpp"
#include "internal/qfmm_geometry.hpp"
#include "internal/qfmm_tables_gen.hpp"
#include "internal/qfmm_tree.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace qcx::integrals::internal {

/// MSVC std::vector object overhead: three pointers per vector.
inline constexpr std::size_t kVectorHeaderBytes = 24;

/// The E-table payload of one shell pair: three per-axis tables of
/// (la+1)(lb+1)(la+lb+1) doubles per primitive pair (md_hermite.hpp
/// PerAxisETable; the per-axis vector HEADERS sit inside sizeof(MdPrimPair)
/// and are not part of this payload term).
/// \param la, lb The two shells' angular momenta.
/// \param nPrimPairs The primitive-pair count (contractionCount product).
inline std::size_t ETableBytes(int la, int lb, std::size_t nPrimPairs) noexcept {
    return 24 * static_cast<std::size_t>(la + 1) * static_cast<std::size_t>(lb + 1) *
           static_cast<std::size_t>(la + lb + 1) * nPrimPairs;
}

/// The bra/ket transform payload of one shell pair (md_batch.cpp): the
/// braTransform GEMM, nFuncs x rowPairs x nPrimPairs x nHerm doubles, plus
/// the ketTransforms, nHerm x nFuncs doubles per primitive pair (the
/// class-folded orientation carries the pair's FULL function count).
/// \param la, lb The two shells' angular momenta.
/// \param nFuncs The pair's function count (nFuncsA x nFuncsB).
/// \param rowPairs rowsA x rowsB.
/// \param nPrimPairs The primitive-pair count.
inline std::size_t TransformBytes(
    int la, int lb, std::size_t nFuncs, std::size_t rowPairs, std::size_t nPrimPairs) noexcept {
    const std::size_t nHerm = static_cast<std::size_t>(Hermite3DCount(la + lb));
    return 8 * nPrimPairs * nHerm * nFuncs * (rowPairs + 1);
}

/// The row-pair contraction weights payload (braWeights, rowsA x rowsB
/// doubles per primitive pair).
/// \param rowPairs rowsA x rowsB.
/// \param nPrimPairs The primitive-pair count.
inline std::size_t WeightsBytes(std::size_t rowPairs, std::size_t nPrimPairs) noexcept {
    return 8 * nPrimPairs * rowPairs;
}

/// The full pair-store bytes of ONE shell pair: the MdPairData struct (its
/// vector headers included), the primitive-pair structs (their E-table
/// vector headers included), the inner braWeights/ketTransforms vector
/// headers, and the three payloads - exactly what BuildContractedPairTransform
/// commits. Capacity equals size on this path (one assign/reserve per
/// vector), so the formula is exact, never a guess.
/// \param la, lb The two shells' angular momenta.
/// \param isSphericalA, isSphericalB The shells' spherical/cartesian choice.
/// \param rowsA, rowsB The shells' contraction rows.
/// \param nPrimPairs The primitive-pair count.
inline std::size_t PairStoreBytesPerPair(int la,
                                         int lb,
                                         bool isSphericalA,
                                         bool isSphericalB,
                                         std::size_t rowsA,
                                         std::size_t rowsB,
                                         std::size_t nPrimPairs) noexcept {
    const std::size_t nAngA =
        static_cast<std::size_t>(isSphericalA ? SphericalCount(la) : CartesianCount(la));
    const std::size_t nAngB =
        static_cast<std::size_t>(isSphericalB ? SphericalCount(lb) : CartesianCount(lb));
    const std::size_t nFuncsA = rowsA * nAngA;
    const std::size_t nFuncsB = rowsB * nAngB;
    const std::size_t rowPairs = rowsA * rowsB;
    return sizeof(MdPairData) + nPrimPairs * sizeof(MdPrimPair) +
           kVectorHeaderBytes * 2 * nPrimPairs + ETableBytes(la, lb, nPrimPairs) +
           TransformBytes(la, lb, nFuncsA * nFuncsB, rowPairs, nPrimPairs) +
           WeightsBytes(rowPairs, nPrimPairs);
}

/// The primitive count of one flattened shell: the exponents of the owning
/// basis-set shell, looked up exactly as FlattenShells does (the
/// elementShellIndex addresses the element's shells, the atomIndex the
/// molecule's atom for the atomic number). ShellInfo itself carries only
/// the contraction-row count, so the basis set is the primitive-count
/// authority. Zero on a missing element - unreachable after the engine's
/// own Create-time validation (BuildShellPairs errors first).
/// \param molecule The molecule (atom lookup for the atomic number).
/// \param basisSet The basis set (the shell's exponents).
/// \param shell The flattened shell.
inline std::size_t ShellPrimitiveCount(const qcx::molecule::Molecule& molecule,
                                       const qcx::basisset::BasisSet& basisSet,
                                       const ShellInfo& shell) noexcept {
    const qcx::basisset::ElementBasis* element =
        basisSet.Find(molecule.Atoms()[shell.atomIndex].atomicNumber);

    if (element == nullptr)
    {
        return 0;
    }

    return element->shells[shell.elementShellIndex].exponents.size();
}

/// The pair-store bytes of a whole shell-pair list: the sum of
/// PairStoreBytesPerPair over the canonical pairs - the store the direct
/// builder's Create materializes (the shell data read from the pair list:
/// angular momentum, spherical flag, contraction rows; the primitive count
/// from the basis set, exactly as FlattenShells reads it).
/// \param molecule The molecule (atom lookup for the atomic number).
/// \param basisSet The basis set (the shells' primitive counts).
/// \param pairList The canonical shell pairs.
inline std::size_t PairStoreBytes(const qcx::molecule::Molecule& molecule,
                                  const qcx::basisset::BasisSet& basisSet,
                                  const ShellPairList& pairList) noexcept {
    std::size_t total = 0;

    for (const ShellPairIndex& pair : pairList.pairs)
    {
        const ShellInfo& a = pairList.shells[pair.i];
        const ShellInfo& b = pairList.shells[pair.j];
        const std::size_t nPrimA = ShellPrimitiveCount(molecule, basisSet, a);
        const std::size_t nPrimB = ShellPrimitiveCount(molecule, basisSet, b);
        total += PairStoreBytesPerPair(a.angularMomentum,
                                       b.angularMomentum,
                                       a.isSpherical,
                                       b.isSpherical,
                                       a.contractionCount,
                                       b.contractionCount,
                                       nPrimA * nPrimB);
    }

    return total;
}

/// The setup ramp's peak resident bytes of ONE walk pair matrix: the shape
/// BuildPairMatrix (one_electron.cpp) realizes - the
/// geometry-only skeleton, sizeof(MdPairData) per canonical pair, filled
/// once and retained for the whole walk; ONE pair-data chunk, the store's
/// contracted payload capped at kPairChunkBytes, which is the same cap and
/// the same payload the walk's own chunker cuts on (NextPairChunk over
/// ChunkPairPayloadBytes), so a store that fits one chunk materializes
/// whole and this term charges it whole; and \p liveMatrices n x n
/// matrices at 8 bytes per function pair, the caller's own concurrency
/// across the ramp (the driver holds the kinetic and nuclear matrices
/// together and then the core Hamiltonian beside the overlap matrix).
///
/// Never-under by construction: every term is the allocation the walk
/// performs, so a caller that refuses on this number refuses before the
/// ramp's first allocation rather than inside it. This is the same
/// composition the lean member's own envelope charges as its runStoreBytes
/// term (EstimatePeakBytes, lean_fock_build.cpp) - that term is this bound
/// with the matrices counted separately - so the two can never disagree
/// about what the ramp costs.
/// \param molecule The molecule (atom lookup for the atomic number).
/// \param basisSet The basis set (the shells' primitive counts).
/// \param pairList The canonical shell pairs the walk traverses.
/// \param liveMatrices The n x n matrices the caller holds live across the
/// ramp.
inline std::size_t SetupRampBytes(const qcx::molecule::Molecule& molecule,
                                  const qcx::basisset::BasisSet& basisSet,
                                  const ShellPairList& pairList,
                                  std::size_t liveMatrices) noexcept {
    const std::size_t nPairs = pairList.pairs.size();
    const std::size_t n = pairList.functionCount;
    const std::size_t skeletonBytes = sizeof(MdPairData) * nPairs;
    const std::size_t storePayloadBytes =
        PairStoreBytes(molecule, basisSet, pairList) - skeletonBytes;

    return skeletonBytes + std::min(kPairChunkBytes, storePayloadBytes) + 8 * liveMatrices * n * n;
}

/// The per-task item bytes of ONE diagonal (ab|ab) quartet inside a sweep
/// chunk - the mirror of md_batch.cpp's TaskItemBytes over
/// PayloadSizesOf with lBra = lKet = la + lb (the diagonal case the sweep
/// evaluates): the output block (nFuncsA x nFuncsB squared), the kernel
/// workspace mPq x (hermKet + nFuncs), the convert buffer (the max of the
/// two transform workspaces) and the 152 B per item of the assembly's
/// records (the file-local Item, the canonical ShellQuartet copy, the
/// MdQuartetTask and the per-task offset - md_batch.cpp's own sum).
/// \param la, lb The pair's angular momenta.
/// \param isSphericalA, isSphericalB The shells' spherical/cartesian choice.
/// \param rowsA, rowsB The shells' contraction rows.
/// \param nPrimPairs The primitive-pair count.
inline std::size_t DiagonalTaskItemBytes(int la,
                                         int lb,
                                         bool isSphericalA,
                                         bool isSphericalB,
                                         std::size_t rowsA,
                                         std::size_t rowsB,
                                         std::size_t nPrimPairs) noexcept {
    const std::size_t nAngA =
        static_cast<std::size_t>(isSphericalA ? SphericalCount(la) : CartesianCount(la));
    const std::size_t nAngB =
        static_cast<std::size_t>(isSphericalB ? SphericalCount(lb) : CartesianCount(lb));
    const std::size_t nFuncs = rowsA * nAngA * rowsB * nAngB;
    const std::size_t rowPairs = rowsA * rowsB;
    const std::size_t herm = static_cast<std::size_t>(Hermite3DCount(la + lb));
    const std::size_t mPq = rowPairs * nPrimPairs * herm;
    const std::size_t outputSize = nFuncs * nFuncs;
    const std::size_t scratchSize = mPq * herm + mPq * nFuncs;
    const std::size_t convertSize = std::max(herm * nFuncs, nFuncs * mPq);

    return (outputSize + scratchSize + convertSize) * sizeof(double) + 152;
}

/// The Schwarz sweep's chunk plan and byte terms, as the sweep realizes
/// them: the largest chunk's contracted pair data (the greedy sizer's own
/// bound) and that chunk's batch-arena mass (the sum of its diagonal
/// tasks' item bytes - AssembleClassBatches partitions by exactly this
/// quantity - floored at the batch cap plus one item, the most a single
/// batch can hold).
struct SchwarzSweepTerms {
    std::size_t skeletonBytes = 0; ///< The geometry-only store: sizeof(MdPairData) per pair.
    std::size_t chunkPayloadBytes = 0; ///< The largest chunk's contracted pair data.
    std::size_t chunkBatchBytes = 0; ///< The largest chunk's batch arena.
    std::size_t boundsBytes = 0; ///< The returned Q vector (8 B per pair).
};

/// The Schwarz sweep's terms over a pair list (SchwarzSweepBytes below is
/// their sum) - the same walk the sweep's own chunker runs, so the charge
/// and the allocation cannot drift.
/// \param molecule The molecule (atom lookup for the atomic number).
/// \param basisSet The basis set (the shells' primitive counts).
/// \param pairList The canonical shell pairs the sweep walks.
/// \param chunkBytes The sweep's pair-data chunk cap; 0 is the unchunked
/// sweep (one chunk over the whole list).
inline SchwarzSweepTerms SchwarzSweepTermsOf(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basisSet,
                                             const ShellPairList& pairList,
                                             std::size_t chunkBytes) noexcept {
    SchwarzSweepTerms terms;
    terms.skeletonBytes = pairList.pairs.size() * sizeof(MdPairData) + kVectorHeaderBytes;
    terms.boundsBytes = 8 * pairList.pairs.size() + kVectorHeaderBytes;

    // The greedy chunker (the sweep's own): a chunk runs to the first pair
    // whose payload would push its materialized pair data past the cap (the
    // chunk's first pair always joins, so a single pair heavier than the cap
    // is its own chunk).
    std::size_t chunkPayload = 0;
    std::size_t chunkItems = 0;
    std::size_t maxItem = 0;

    const auto closeChunk = [&]() noexcept {
        terms.chunkPayloadBytes = std::max(terms.chunkPayloadBytes, chunkPayload);
        terms.chunkBatchBytes = std::max(terms.chunkBatchBytes, chunkItems);
        chunkPayload = 0;
        chunkItems = 0;
    };

    for (const ShellPairIndex& pair : pairList.pairs)
    {
        const ShellInfo& a = pairList.shells[pair.i];
        const ShellInfo& b = pairList.shells[pair.j];
        const std::size_t nPrimA = ShellPrimitiveCount(molecule, basisSet, a);
        const std::size_t nPrimB = ShellPrimitiveCount(molecule, basisSet, b);
        const std::size_t nPrimPairs = nPrimA * nPrimB;
        const std::size_t payload = PairStoreBytesPerPair(a.angularMomentum,
                                                          b.angularMomentum,
                                                          a.isSpherical,
                                                          b.isSpherical,
                                                          a.contractionCount,
                                                          b.contractionCount,
                                                          nPrimPairs) -
                                    sizeof(MdPairData);
        const std::size_t item = DiagonalTaskItemBytes(a.angularMomentum,
                                                       b.angularMomentum,
                                                       a.isSpherical,
                                                       b.isSpherical,
                                                       a.contractionCount,
                                                       b.contractionCount,
                                                       nPrimPairs);
        maxItem = std::max(maxItem, item);

        if (chunkPayload != 0 && chunkBytes != 0 && chunkPayload + payload > chunkBytes)
        {
            closeChunk();
        }

        chunkPayload += payload;
        chunkItems += item;
    }

    closeChunk();

    // The batch arena: a chunk's batches run one at a time, each holding at
    // most the cap plus its single over-cap item (AssembleClassBatches'
    // boundary test), never more than the chunk's whole item sum.
    const std::size_t batchCap = EriBatchOptions{}.maxBatchBytes;
    terms.chunkBatchBytes = std::min(terms.chunkBatchBytes, batchCap + maxItem);
    return terms;
}

/// The screening pre-pass's resident byte bound (the screening.cpp
/// ComputeSchwarzBounds sweep every builder's Create runs to get its Schwarz
/// bounds) - the allocation NO estimate charged until this term
/// (the pre-chunking sweep built the WHOLE pair store, 9.95 GiB at
/// C42H86/def2-QZVP -
/// 10,683,182,128 B as the running process allocated it - as a Create-time
/// transient no mode record mentioned, so a rung whose modeled total was
/// small was not thereby safe; a ladder run died of `bad allocation` inside
/// it).
///
/// The chunked sweep (screening.cpp) no longer materializes the whole store,
/// so the charge is the shape that remains, counted from the same inputs the
/// sweep reads and by the same walk its own sizer runs
/// (SchwarzSweepTermsOf): the geometry-only skeleton, the largest chunk's
/// contracted pair data, that chunk's batch arena, and the returned bounds
/// vector. Every term is a sizeof or a per-pair arithmetic product over the
/// canonical pair list, so the charge is never-under by construction - and
/// it scales down with a small system, where the batch arena is the chunk's
/// own item sum rather than the 512 MiB cap.
/// \param molecule The molecule (atom lookup for the atomic number).
/// \param basisSet The basis set (the shells' primitive counts).
/// \param pairList The canonical shell pairs the sweep walks.
/// \param chunkBytes The sweep's pair-data chunk cap (kSchwarzChunkBytes is
/// the ComputeSchwarzBounds default - pass the caller's own value when it
/// sweeps under a different one); 0 is the unchunked sweep, whose whole
/// payload is the chunk term.
inline std::size_t SchwarzSweepBytes(const qcx::molecule::Molecule& molecule,
                                     const qcx::basisset::BasisSet& basisSet,
                                     const ShellPairList& pairList,
                                     std::size_t chunkBytes = kSchwarzChunkBytes) noexcept {
    const SchwarzSweepTerms terms = SchwarzSweepTermsOf(molecule, basisSet, pairList, chunkBytes);
    return terms.skeletonBytes + terms.chunkPayloadBytes + terms.chunkBatchBytes +
           terms.boundsBytes;
}

/// The SETUP's peak resident bytes: the ramp (SetupRampBytes) PLUS the
/// Schwarz screening sweeps that run in front of every budgeted builder's
/// own Create-time decision (SchwarzSweepBytes over both bases).
///
/// Why the sweeps are charged here rather than only at the decision: the
/// sweep is the pre-pass Create runs BEFORE it consults the workspace
/// budget, and it realizes the MD kernels' per-thread batch arenas
/// (md_attribution.hpp ThreadScratchVector, tag kScratch) sized by the
/// caller's maxBatchBytes - never by the budget. Measured (2026-09-18,
/// C50H102/def2-SVP, n = 1210): 683.6 MiB of live arenas and a 1690.7 MiB
/// process peak accumulated BEFORE the RI-J estimate was evaluated, which
/// is why a cap that clears the ramp alone still died inside the sweep -
/// silently, with 0xC0000409 and no diagnostic. The bound's only consumer
/// is the driver's pre-gate admission, which has no rung to fall to, so
/// this is the last place the refusal can still be a refusal.
///
/// Never-under by construction. The phases are sequential (the ramp
/// finishes before Create runs its sweeps) AND the two sweeps are
/// sequential, so their SUM is an upper bound of the peak, and the sum is
/// what is charged - the conservative direction on purpose. Under-charging
/// recreates the death; over-charging refuses only runs that would have
/// died. The sweep term is SchwarzSweepBytes' own composition, not a
/// second copy of the arithmetic, so the charge and the sweep's allocation
/// cannot drift.
/// \param molecule The molecule (atom lookup for the atomic number).
/// \param basisSet The orbital basis set.
/// \param pairList The canonical orbital pairs.
/// \param liveMatrices The caller's live n x n matrices across the ramp
/// (the driver's kRampLiveMatrices).
/// \param auxBasisSet The auxiliary basis set, or null when none is in
/// effect: a family that consumes no aux runs no aux sweep, and charging
/// one would refuse runs that never allocate it.
/// \param auxPairList The canonical auxiliary pairs, read only when
/// \p auxBasisSet is non-null (the same BuildShellPairs call the engine's
/// own sweep makes).
inline std::size_t SetupPeakBytes(const qcx::molecule::Molecule& molecule,
                                  const qcx::basisset::BasisSet& basisSet,
                                  const ShellPairList& pairList,
                                  std::size_t liveMatrices,
                                  const qcx::basisset::BasisSet* auxBasisSet,
                                  const ShellPairList* auxPairList) noexcept {
    std::size_t total = SetupRampBytes(molecule, basisSet, pairList, liveMatrices) +
                        SchwarzSweepBytes(molecule, basisSet, pairList);

    if (auxBasisSet != nullptr && auxPairList != nullptr)
    {
        total += SchwarzSweepBytes(molecule, *auxBasisSet, *auxPairList);
    }

    return total;
}

/// The all-survive neighbor-pattern upper bound: 8 bytes per surviving
/// canonical pair of the upper triangle (ket <= bra - the counting pass of
/// BuildNeighborList in fock_screen.hpp). The never-under dense reference
/// (the counted exclusion): the mode selection's exclusion (i) PRE-GATE -
/// when even the full pattern fits, the Schwarz-counted pattern fits a
/// fortiori and no counting pass runs - and the light path's peak-chunk
/// markers at the exclusion (the full pair space bounds every chunk). The
/// exact-count test CountSchwarzSurvivingPairs reproduces this value
/// exactly on all-survive inputs, so the counted decision never admits a
/// pattern the all-survive form would have refused.
/// \param nPairs The shell-pair count.
inline std::size_t AllSurvivePatternBytes(std::size_t nPairs) noexcept {
    return 8 * nPairs * (nPairs + 1) / 2;
}

/// The pair-class table's Create-time structural bytes (the pair-class root
/// fix of 2026-08-31): the class path materializes the FULL canonical
/// member-quartet space at Create - nPairs(nPairs+1)/2 ClassOrbitMember
/// entries (56 B each, symmetry_reduction.hpp) plus the class level - a
/// theta(N^4) structure no estimate charged until the c60 0xC0000409
/// deaths (57.1 GiB of members at 45,150 pairs). The formula is the
/// never-under worst case - the only shape Create-time-safe without
/// building the table itself (the actual class structure is unknowable
/// before the build): singleton orbits (one class per pair, one class
/// pair and one orbit per member quartet - the real case for linear-chain
/// molecules, where the action fixes every pair), each member with its
/// orbit's vector header, plus the per-pair class bookkeeping
/// (classOfPair entries, the classes' member lists and the PairClass
/// structs) and the BuildOrbits building transient (the raw mP x mQ
/// quartet list, 16 B per entry, over the loop's whole mass - charged as
/// if concurrent; the sum of the raw products is bounded by nPairs^2).
/// Over-charging a well-symmetric molecule's actual table is a
/// throughput-only regression (the path disengages where the table WOULD
/// fit); under-charging recreates the death. Every term is a sizeof of
/// the actual struct, so the formula tracks layout changes.
///
/// The root restructure (2026-08-31) keeps this formula as the Create-time
/// admission ceiling but never materializes the full eager table: the
/// class pairs are Schwarz-screened at BuildPairClasses and the orbit
/// expansions are generated ON DEMAND per reached class pair
/// (GenerateClassPairOrbits), so the live table's peak is the screened
/// subset of the formula's shape - the estimate remains a never-under
/// ceiling of the actual allocation, and the admission band is
/// unchanged.
/// \param nPairs The canonical shell-pair count.
/// \param nFunctions The basis function count (the shellOfFunction support
/// data of the on-demand orbit generation: the stored per-function shell
/// index, nFunctions x 8 B plus its vector header - charged here so the
/// estimate stays never-under over the whole Create-time allocation).
inline std::size_t ClassTableBytes(std::size_t nPairs, std::size_t nFunctions) noexcept {
    const std::size_t memberQuartets = nPairs * (nPairs + 1) / 2;
    return memberQuartets * (sizeof(ClassOrbitMember) + sizeof(ClassPair) + sizeof(ClassOrbit) +
                             2 * kVectorHeaderBytes) +
           nPairs * (2 * sizeof(std::size_t) + sizeof(PairClass) + kVectorHeaderBytes) +
           nFunctions * sizeof(std::size_t) + kVectorHeaderBytes + 16 * nPairs * nPairs +
           3 * kVectorHeaderBytes;
}

/// The class path's materialization charge from the counts the class
/// decomposition yields (CountClassMaterialization): the never-under bound
/// of what the path can ALLOCATE, as opposed to ClassTableBytes' singleton-
/// orbit ceiling over the whole canonical quartet space.
///
/// The terms, each a sizeof of the struct it charges, mirroring the shape
/// the root restructure (2026-08-31) actually materializes:
/// - the kept class pairs (one ClassPair each, with its vector header),
/// - the member quartets of the kept pairs: a reached class pair fills its
///   whole member-quartet set with ClassOrbitMember entries, and its orbits
///   (each with a ClassOrbit entry and a vector header) are at most one per
///   member - the same per-member convention ClassTableBytes uses,
/// - the class level: one PairClass per class plus the member-list and
///   classOfPair entries (one pair index each),
/// - the shellOfFunction support data (the on-demand generation's
///   per-function shell index),
/// - the BuildOrbits building transient (the raw mP x mQ quartet list, 16 B
///   per entry) over the same member-quartet mass.
/// The counts are ordered-pair measures, so every term is never-under the
/// corresponding materialized quantity (see CountClassMaterialization).
///
/// This is the gate's charge when the ClassTableBytes ceiling does not fit:
/// the ceiling is the ONLY shape Create-time-safe without the class
/// decomposition, and this one is the shape the path reaches once the
/// decomposition is in hand. Both are upper bounds of the same allocation,
/// so the gate takes their minimum - a charge that can never engage the
/// class path LESS often than the ceiling alone did.
/// \param counts The class decomposition's counts.
/// \param nPairs The canonical shell-pair count.
/// \param nFunctions The basis function count.
inline std::size_t ClassMaterializationBytes(const ClassMaterializationCounts& counts,
                                             std::size_t nPairs,
                                             std::size_t nFunctions) noexcept {
    return counts.classPairs * (sizeof(ClassPair) + kVectorHeaderBytes) +
           counts.memberQuartets *
               (sizeof(ClassOrbitMember) + sizeof(ClassOrbit) + 2 * kVectorHeaderBytes) +
           counts.classes * (sizeof(PairClass) + kVectorHeaderBytes) +
           2 * nPairs * sizeof(std::size_t) + nFunctions * sizeof(std::size_t) +
           2 * kVectorHeaderBytes + 16 * counts.memberQuartets;
}

/// The scratch clamp: the largest maxBatchBytes whose per-thread arenas
/// (batch x threadCount, arenaCount of them) keep the estimate at or below
/// the remaining budget. Batch re-partitioning is value-neutral (the
/// schedule-independence contract of the batch machinery), so clamping is
/// numerically safe and is the first response when the estimate exceeds the
/// budget. The deficit is closed across ALL the stack's sequential batch
/// arenas together (the RI-J's 3c arena plus the nested exchange's - the
/// CWA sums sequential commits): the batch shrinks by
/// ceil(deficit / (arenaCount x threadCount)), so the fired estimate lands
/// at or below the remaining bytes. Refusing only when even a unit batch
/// cannot fit across the arenas keeps the fit band as wide as the stack
/// really is.
/// \param maxBatchBytes The options' batch cap.
/// \param threadCount The OpenMP team size read at Create.
/// \param estimateBytes The full estimate with the unclamped batch.
/// \param remainingBytes The budget's remaining bytes.
/// \param arenaCount The number of sequential per-thread batch arenas in
/// the stack (1 for the direct/QFMM builders, 2 for the RI-J).
/// \param slopeBytes The charged estimate's exact per-byte-of-cap linear
/// coefficient when the caller knows it - the fix-A exchange-engaged fast
/// path passes threadCount + slotCount x (2|3), so the shrink removes the
/// true deficit (the legacy arenaCount x threadCount model under-removes
/// the k-scaled exchange term); 0 (the default) keeps the legacy model.
/// \returns The clamped batch, or 0 when even a unit batch cannot fit
/// (LightPath mandatory).
inline std::size_t ClampBatchBytes(std::size_t maxBatchBytes,
                                   std::size_t threadCount,
                                   std::size_t estimateBytes,
                                   std::size_t remainingBytes,
                                   std::size_t arenaCount = 1,
                                   std::size_t slopeBytes = 0) noexcept {
    if (estimateBytes <= remainingBytes)
    {
        return maxBatchBytes;
    }

    const std::size_t deficit = estimateBytes - remainingBytes;
    const std::size_t slope = slopeBytes != 0 ? slopeBytes : arenaCount * threadCount;
    const std::size_t shrink = (deficit + slope - 1) / slope;

    return shrink >= maxBatchBytes ? 0 : maxBatchBytes - shrink;
}

/// The surviving screened-exchange mass fraction fitted at
/// C24H50/def2-SVP kNormal (T ~= 1.04e9 doubles over 586^4) for the
/// exchange term's whole-call envelope (the fitted envelope).
/// ExchangePerCallBytes no longer charges it - the per-batch
/// restructured the exchange into per-batch assemble-and-contract for
/// BOTH lanes, so the per-call live values sit at the batch cap, and the
/// values-term redesign made the term the structural per-batch
/// bound - but the constant is RETAINED as the never-under floor of the
/// GPU lane's per-call survival law (device_footprint.hpp: the GPU
/// survival fraction never drops below the CPU exchange fraction).
/// The fit's bracket history stands: the survival law's n-dependence was
/// the recorded honest unknown, bracketed by two measured points - the
/// C12H26 trace's n^4-relative screened set (nq/n^4 = 2.225e-3 to
/// 2.336e-3 over the period-2 totals 17.55M/18.42M at n=298) is ~16x the
/// fit point's nq/n^4 = 1.42e-4 (15.7x-16.5x; the review-corrected
/// bracket - the earlier "~55x" mixed the survival fraction 0.0088 with
/// the quartet fraction), so the constant stayed at the fit point and
/// was re-verified at every new admission class.
inline constexpr double kExchangeSurvivalFraction = 0.0088;

/// The computed-quartet fraction fitted at C24H50 kNormal (~16.8M
/// quartets over 586^4, the 8 B per-quartet bounds vector of the old
/// whole-call exchange term). Like kExchangeSurvivalFraction, no longer
/// charged by ExchangePerCallBytes (per-batch now) - retained as
/// the GPU lane's per-call quartet floor (device_footprint.hpp) and as
/// the ScreenedQuartetFraction law's anchor at n = 586.
inline constexpr double kExchangeQuartetFraction = 1.42e-4;

/// The measured screened-quartet fraction of n^4 over a WHOLE run (the
/// whole-run charge): the screened-quartet count
/// grows like a fixed fraction of n^4 at small n and ~ n^2 at the
/// 5000-scale (the 10-30M-quartet plateau) - the piecewise log-log law
/// through the measured anchors:
/// (298, 2.5e-3) - the C12H26 trace's nq/n^4 = 2.225e-3..2.336e-3
/// at n = 298, charged at the CONSERVATIVE max (the never-under end of
/// the bracket); (586, 1.42e-4) - the C24H50 fit point (equals the
/// shipped kExchangeQuartetFraction); and (5000, 1/n^2 = 4e-8) - the
/// ~n^2 survival anchor. Flat 2.5e-3 below 298 (the screening
/// law's small-n start), 1/n^2 at and above 5000 (the measured band's
/// conservative end; the fraction there equals kExchangeQuartetFraction
/// x kExchangeSurvivalFraction x n^2 / n^4, the n^2 survival). The
/// whole-run law is never-under the per-chunk peak by construction: the
/// LightPath chunk pass's per-chunk surviving set is a subset of the
/// whole-run surviving set, so the whole-run count bounds every chunk.
/// \param functionCount The orbital function count.
inline double ScreenedQuartetFraction(std::size_t functionCount) noexcept {
    const double n = static_cast<double>(functionCount);

    if (n <= 298.0)
    {
        return 2.5e-3;
    }

    if (n >= 5000.0)
    {
        return 1.0 / (n * n);
    }

    if (n <= 586.0)
    {
        return 2.5e-3 * std::pow(n / 298.0, -4.2415);
    }

    return 1.42e-4 * std::pow(n / 586.0, -3.8131);
}

/// The whole-call screened-quartet task-list survivor per quartet (the
/// honest charge - the rows this decomposition replaces):
/// the fp64 task master's MdQuartetTask entry. The earlier whole-call
/// rows of the old 224 B/q control mass (the whole-run charge - the per-half
/// quartets vectors, the CanonicalQuartetInfo ordered list, the Item and
/// computed assembly buffers, the densityWeightOf routing map and the
/// whole-call offsets, together ~200 B/q) no longer exist: the masters
/// are sorted IN PLACE once per call and the batches materialize one at
/// a time from the sorted masters (MdClassBatchCursor at k = 1, the
/// k-slot slices of the batchStarts table at k > 1), so the sorted
/// masters are the irreducible whole-call survivor (the canonical sort's
/// input must be resident). The
/// per-batch machinery (the task slices, offsets and values) is
/// batch-capped and rides the exchange terms' k x batch-cap accounting;
/// the batch-boundary metadata (the per-half starts tables, 8 B x
/// (numBatches + 1) per half per call, the batchStarts dry-walk tables
/// of the k > 1 rung, fock_build.cpp RunPass and RunClassPass) is
/// whole-call but batch-count-sized, and is CHARGED at that same k x
/// term: DirectFootprint adds BatchStartsTablesBytes to the slot-folded
/// exchange bound when the k > 1 slots can run (the tables exist only
/// there - the k = 1 loop materializes its batches from the cursor
/// directly and never builds them, so the charge must not engage at
/// k = 1).
inline constexpr double kScreenedTaskBytesPerQuartet = static_cast<double>(sizeof(MdQuartetTask));

/// The certified fp32 lane's per-quartet whole-call delta: a lane-routed
/// quartet's task master carries the same MdQuartetTask entry plus the
/// parallel fp32DensityWeights entry (the positional weight
/// transport) - sizeof(double) on top of kScreenedTaskBytesPerQuartet.
/// The charge adds the delta whenever the preset authorizes the lane
/// (kLoose/kNormal - the certified gate; kTight's mixed threshold is 0
/// and the lane stays off, ExchangePerCallBytes' lane-split convention),
/// at the never-under end of the split: the certified gate
/// (cClass x kCertifiedEpsilon x Q_bra x Q_ket x dMax <= mixedThreshold)
/// can route the WHOLE density-screened set through the lane on a
/// weak-density system, so the delta rides the full charged set.
inline constexpr double kCertifiedLaneBytesPerQuartet = static_cast<double>(sizeof(double));

/// The whole-run screened-quartet task-machinery bytes (the whole-run charge
/// term, amended to the survivor): ScreenedQuartetFraction(n) x n^4 x
/// the per-quartet survivor - kScreenedTaskBytesPerQuartet, plus
/// kCertifiedLaneBytesPerQuartet when the preset authorizes the certified
/// lane. The masters exist on every direct path (Coulomb-only included -
/// ScreenAll fills them regardless of the exchange pass), so the charge
/// is unconditional. Never-under by construction: the fraction law's n^4
/// envelope sits at or above every measured screened set (the C12H26 max
/// 2.336e-3 at 298 <= 2.5e-3; the C24H50 fit point; the n^2 5000 anchor),
/// and the lane delta covers the whole set at the lane-authorized presets.
/// \param functionCount The orbital function count.
/// \param accuracy The screening preset (the certified lane split).
inline std::size_t ScreenedQuartetBytes(std::size_t functionCount,
                                        AccuracyPreset accuracy) noexcept {
    const double n = static_cast<double>(functionCount);
    const double n4 = n * n * n * n;
    const double laneDelta =
        accuracy == AccuracyPreset::kTight ? 0.0 : kCertifiedLaneBytesPerQuartet;
    return static_cast<std::size_t>(ScreenedQuartetFraction(functionCount) * n4 *
                                    (kScreenedTaskBytesPerQuartet + laneDelta));
}

/// The exchange per-slot per-call live-mass bound (the values-term
/// redesign): the structural per-call peak terms replacing the
/// whole-call fitted envelope (the 0.0088 kExchangeSurvivalFraction fit
/// above is no longer charged here - it survives only as the GPU lane's
/// floor and the anchor). The exchange was restructured
/// into per-batch assemble-and-contract loops for BOTH lanes: one
/// maxBatchBytes-bounded batch (or cache slice) is live at a time per
/// half and each batch's buffers are discarded before the next is built,
/// so a call's live values peak at one batch's mass, never the
/// whole-call screened mass (the death-allocation class - the
/// 3.88 GiB fp32 valuesF32 resize at C24H50 - is gone by construction).
/// The charged terms: F64Bound = 2 x maxBatchBytes, the fp64 half's
/// per-call bound (the batch's packed values plus the batch-class engine
/// workspace; the measured post-restructure C24H50 fp64 bound is ~1 GiB at the
/// 512 MiB default cap); and F32Live = maxBatchBytes, the certified fp32
/// half's per-call bound (the lane's valuesF32 at 4 B per element plus
/// the per-quartet bounds at 8 B, its batches and cache slices capped at
/// the batch cap like the fp64 half's - the certified cache-slice cap
/// counts the lane's own request mass, fock_build.cpp RunPass, so no
/// fp32 slice rides the ~1.5x overshoot of the old fp64-mass slice cap).
/// The preset splits the lanes: kTight runs the certified lane OFF (all
/// fp64 - F64Bound only), kLoose/kNormal route
/// through the certified lane and charge BOTH bounds (the fp64 refused
/// share's half and the certified half run sequentially per call, so the
/// CWA sums them). Never-under by construction: every assembled batch or
/// cache slice of either lane sits at or below its cap, so each slot's
/// per-call peak sits at or below its charged term at every preset; a
/// lone over-cap quartet batch (itemBytes > maxBatchBytes, the review
/// amendment's exception) stays inside the terms' slack at every
/// basis an admission can reach. The term is a Reservation charged once
/// at Create through Total() (the per-call buffers are
/// fresh allocations, the CWA note, not per-thread scratch), and scales
/// with the batch cap: the ladder's clamped re-estimates shrink it with
/// the batch. The bounded-concurrency batch loop composes the
/// concurrency into the charge: the estimate multiplies the per-slot
/// bound by the authorized slot count (DirectFootprint's slotCount), so
/// the CWA covers k slots x one live batch each, never 1 x k batches -
/// the under-model failure class the k-factor closes.
/// \param maxBatchBytes The batch cap (already clamped when re-estimating).
/// \param accuracy The screening preset (the lane split).
inline std::size_t ExchangePerCallBytes(std::size_t maxBatchBytes,
                                        AccuracyPreset accuracy) noexcept {
    const std::size_t f64Bound = 2 * maxBatchBytes;
    const std::size_t f32Live = maxBatchBytes;

    if (accuracy == AccuracyPreset::kTight)
    {
        return f64Bound;
    }

    return f64Bound + f32Live;
}

/// The k > 1 rung's per-call batch-boundary metadata: the per-half
/// batchStarts tables of the fix-A concurrent batch loop (fock_build.cpp
/// RunPass and RunClassPass - the cursor's dry walk fills one start per
/// emitted batch, 8 B x (numBatches + 1) entries per half per call, the
/// sentinel end entry included), allocated ONLY when the k > 1 slots can
/// run (the k = 1 loop materializes its batches from the cursor directly
/// and never builds the tables - DirectFootprint gates the charge on
/// slotCount > 1, no phantom at k = 1). The tables are shared across the
/// slots - one per half per call, never k-folded - so the charge stands
/// beside the slot fold at the exchange k x accounting it rides.
/// Never-under by construction: a batch closes when the next item would
/// overflow the cap and no item exceeds itemBytesMax, so every batch but
/// the last carries more than maxBatchBytes - itemBytesMax of item bytes
/// and numBatches(half) <= (mass(half) - 8) / (maxBatchBytes -
/// itemBytesMax + 1) + 1; the half's per-call mass is at most the
/// envelope's item count (ScreenedQuartetFraction's n^4 law, the same
/// envelope the screened charge uses - the half's screened items are a
/// subset of the whole-run screened set it sits above) times
/// itemBytesMax, so the mass ceiling below never-unders the table's
/// entry count. itemBytesMax is the lane's largest assembled item: the
/// canonical quartet block of the basis's largest pair-function product
/// (the block the batch engine cuts on), at the lane's element size -
/// 8 B per element fp64, 4 B plus the 8 B per-quartet bound fp32
/// (ExchangePerCallBytes' lane-split convention: kTight runs the
/// certified lane off, its half stays empty and only the fp64 table is
/// charged). The envelope is floored at one screened quartet (the flat
/// law's n^4 product runs below 1 only below n = 5, where a surviving
/// quartet still exists).
/// \param pairList The canonical shell pairs (the pair-function ceiling
/// of the basis).
/// \param accuracy The screening preset (the certified lane split).
/// \param maxBatchBytes The batch cap (already clamped when
/// re-estimating).
inline std::size_t BatchStartsTablesBytes(const ShellPairList& pairList,
                                          AccuracyPreset accuracy,
                                          std::size_t maxBatchBytes) noexcept {
    const double n = static_cast<double>(pairList.functionCount);
    const double envelopeCount =
        std::max(1.0, ScreenedQuartetFraction(pairList.functionCount) * n * n * n * n);

    // The basis's largest pair-function product: the canonical quartet's
    // block is the bra pair's functions times the ket pair's, so the
    // largest item the engine can assemble is the square of the largest
    // pair product (a bra and a ket pair of that size over four shells).
    std::size_t maxPairFuncs = 1;

    for (const ShellPairIndex& pair : pairList.pairs)
    {
        maxPairFuncs = std::max(maxPairFuncs,
                                ShellFunctionCount(pairList.shells[pair.i]) *
                                    ShellFunctionCount(pairList.shells[pair.j]));
    }

    const double blockElements =
        static_cast<double>(maxPairFuncs) * static_cast<double>(maxPairFuncs);

    // The per-half table's never-under entry count: the item-count
    // ceiling (envelopeCount + 1) and the mass ceiling above (the +2
    // covers the sentinel end entry and the tail-batch slack; the
    // denominator clamps to 1 at the sub-item caps, where the item
    // count is the binding bound).
    const auto tableEntries = [envelopeCount, maxBatchBytes](double itemBytesMax) -> std::size_t {
        const double denom = std::max(1.0, static_cast<double>(maxBatchBytes) - itemBytesMax + 1.0);

        return static_cast<std::size_t>(
            std::min(envelopeCount + 1.0, std::ceil(envelopeCount * itemBytesMax / denom) + 2.0));
    };

    // The fp64 half's table (its item is the block at 8 B per element);
    // the certified half's table only at the lane-authorizing presets
    // (its item is the block at 4 B plus the 8 B per-quartet bound).
    const std::size_t fp64TableBytes = tableEntries(blockElements * 8.0) * sizeof(std::size_t);

    if (accuracy == AccuracyPreset::kTight)
    {
        return fp64TableBytes;
    }

    return fp64TableBytes + tableEntries(blockElements * 4.0 + 8.0) * sizeof(std::size_t);
}

/// The direct builder's Create-time footprint terms (the fired-estimate
/// terms of the mode record; every field bytes).
struct DirectFootprintTerms {
    std::size_t pairStoreBytes = 0; ///< The MD pair data (E tables, transforms, weights).
    std::size_t patternBytes = 0; ///< The neighbor CSR indices (8 bytes per surviving pair).
    std::size_t scratchBytes = 0; ///< The per-thread batch arena (batch x threads).
    std::size_t structuralBytes = 0; ///< Pair list, Schwarz vector, CSR offsets, core-H copies.
    std::size_t cacheBytes = 0; ///< The ERI cache: two lanes x maxCacheBytes plus the entry-map
                                ///< allowance (plain path only).
    std::size_t exchangePerCallBytes = 0; ///< The nested exchange's per-call live-mass bound
                                          ///< (ExchangePerCallBytes x the authorized slot count -
                                          ///< the structural F64Bound + F32Live per-slot terms of
                                          ///< the values-term redesign, k-folded by slot count),
                                          ///< plus the k > 1 rung's batch-boundary metadata at
                                          ///< slotCount > 1 (BatchStartsTablesBytes - the per-half
                                          ///< batchStarts tables of the dry walk, shared
                                          ///< across the slots, never k-folded).
    std::size_t classTableBytes = 0; ///< The pair-class table's structural charge
                                     ///< (ClassTableBytes - the never-under ceiling of the
                                     ///< on-demand table, the class path only; 0 on the
                                     ///< plain path or when the admission gate disengaged).
    std::size_t screenedQuartetBytes = 0; ///< The whole-run screened-quartet task machinery
                                          ///< (ScreenedQuartetBytes - the whole-call
                                          ///< survivor, kScreenedTaskBytesPerQuartet per
                                          ///< screened quartet plus the certified lane delta,
                                          ///< the whole-run charge).
    std::size_t schwarzSweepBytes = 0; ///< The Create-time screening pre-pass's EXCESS over
                                       ///< the store term this estimate already carries
                                       ///< (SchwarzSweepBytes' phase-max check below; 0 at
                                       ///< every reachable size, and positive only if a
                                       ///< future sweep ever outgrows the store it feeds).

    /// The full Create-time footprint of the direct builder.
    std::size_t Total() const noexcept {
        return pairStoreBytes + patternBytes + scratchBytes + structuralBytes + cacheBytes +
               exchangePerCallBytes + classTableBytes + screenedQuartetBytes + schwarzSweepBytes;
    }
};

/// The direct builder's full Create-time footprint estimate: the pair
/// store, the neighbor pattern at the COUNTED surviving pairs (the sweep
/// result - the decision layer runs the actual BuildNeighborList, never a
/// mirror), the per-thread batch arena, the structural terms (the retained
/// ShellPairList, Schwarz vector, CSR row offsets and the two core-H
/// copies, 8 bytes each), the ERI cache when engaged and the exchange
/// per-call live-mass bound (ExchangePerCallBytes - the structural
/// F64Bound + F32Live per-batch terms of the values-term redesign)
/// when the estimated stack runs an exchange pass.
/// \param molecule The molecule (atom lookup for the atomic number).
/// \param basisSet The basis set (the shells' primitive counts).
/// \param pairList The canonical shell pairs.
/// \param patternCount The counted surviving neighbor pairs (the sweep).
/// \param maxBatchBytes The batch cap (already clamped when re-estimating).
/// \param threadCount The OpenMP team size read at Create.
/// \param useCache True when the ERI cache engages (plain path only).
/// \param maxCacheBytes The caller's cache cap (two lanes).
/// \param accuracy The screening preset (the exchange term's lane split).
/// \param exchangeEngaged True when the estimated builder runs an exchange
/// pass (the fused RHF and buildExchangeOnly modes); false for Coulomb-only
/// stacks (buildCoulombOnly - the QFMM near-field), where the exchange
/// term would be phantom and must not be charged (the exchange fold-in).
/// \param classTableBytes The class path's table charge (ClassTableBytes),
/// passed by the Create-time admission gate - 0 when the class path is off
/// or disengaged (the plain path charges no table).
/// \param slotCount The authorized concurrent batch slots (fix A): the
/// exchange term multiplies its per-slot bound by the slot count, so the
/// charge covers k slots x one live batch each (the k-factor accounting);
/// 1 (the default) on the legacy, cache-engaged, Coulomb-only and LightPath
/// rungs, and > 1 on the budgeted fast path with the exchange engaged -
/// free-standing runs and a
/// pre-reserved nested half too: the nested exchange's own Create-time
/// decision authorizes k from the POST-reservation remaining (the outer
/// reserves only its own terms and leaves the exchange band), so its
/// k-fold charge fits its band by construction (ri_engine's nesting-order
/// reservation contract; the k = 1 model stays the no-slack, no-budget
/// and non-exchange-engaged default).
inline DirectFootprintTerms DirectFootprint(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet,
                                            const ShellPairList& pairList,
                                            std::size_t patternCount,
                                            std::size_t maxBatchBytes,
                                            std::size_t threadCount,
                                            bool useCache,
                                            std::size_t maxCacheBytes,
                                            AccuracyPreset accuracy,
                                            bool exchangeEngaged = true,
                                            std::size_t classTableBytes = 0,
                                            std::size_t slotCount = 1) noexcept {
    DirectFootprintTerms terms;
    terms.pairStoreBytes = PairStoreBytes(molecule, basisSet, pairList);
    terms.patternBytes = 8 * patternCount;
    terms.scratchBytes = maxBatchBytes * threadCount;
    terms.structuralBytes = pairList.shells.size() * sizeof(ShellInfo) +
                            pairList.pairs.size() * sizeof(ShellPairIndex) +
                            8 * pairList.shells.size() + 8 * (pairList.pairs.size() + 1) +
                            2 * 8 * pairList.functionCount * pairList.functionCount;
    // The k > 1 rung's per-half batchStarts tables ride this term too
    // (BatchStartsTablesBytes - the batch-boundary metadata of the
    // streaming-rung charge, kScreenedTaskBytesPerQuartet's claim above):
    // the tables are allocated only when the k > 1 slots can run, so the
    // addend engages only at slotCount > 1, and it stands beside the slot
    // fold (one table per half per call, shared across the slots).
    terms.exchangePerCallBytes =
        exchangeEngaged
            ? slotCount * ExchangePerCallBytes(maxBatchBytes, accuracy) +
                  (slotCount > 1 ? BatchStartsTablesBytes(pairList, accuracy, maxBatchBytes) : 0)
            : 0;
    // The entry map: the unordered_map of key -> Entry (~90-100 B per
    // cached quartet - 1-3 GB at the 10-30M quartets the screening law
    // reaches at 5000-scale). The map is not
    // bounded by the payload cap (the arenas hold the values, the entries
    // are stable under arena growth), so the allowance is one
    // maxCacheBytes - the sanctioned ~3x total.
    terms.cacheBytes = useCache ? 3 * maxCacheBytes : 0;
    terms.classTableBytes = classTableBytes;
    // The screened-quartet task machinery is charged unconditionally: the
    // per-call sorted task masters exist on every direct path (Coulomb-only
    // included - ScreenAll fills them regardless of the exchange pass),
    // and the fraction law's n^4 envelope is never-under the measured
    // screened sets.
    terms.screenedQuartetBytes = ScreenedQuartetBytes(pairList.functionCount, accuracy);
    // The screening pre-pass (SchwarzSweepBytes), charged as a PHASE MAX and
    // not a sum: Create runs the chunked ComputeSchwarzBounds sweep first
    // and builds the pair store after it, so the two allocations never
    // overlap and the Create-time peak is the larger of the two phases. The
    // builder's store is the larger by construction (the sweep's largest
    // chunk is one chunkBytes-sized slice of exactly the payload
    // pairStoreBytes charges whole), so the term below is the excess - which
    // is the check the model was missing, not a second charge: at
    // C42H86/def2-QZVP the sweep peaks at ~1.3 GiB against the 9.95 GiB
    // pairStoreBytes beside it, and a run of any size reaches the same
    // dominance. Where the excess is non-zero the estimate grows by exactly
    // that much, so a sweep that ever outgrows its own store is charged.
    const std::size_t sweepBytes = SchwarzSweepBytes(molecule, basisSet, pairList);
    terms.schwarzSweepBytes =
        sweepBytes > terms.pairStoreBytes ? sweepBytes - terms.pairStoreBytes : 0;
    return terms;
}

/// The metric pair store of the RI-J builder (ri_engine.cpp
/// BuildAuxMetric): the vector of MdPairData entries of the shared-phantom
/// pair list - one l = 0 phantom shell at index 0 plus one filled
/// (phantom, P) pair per aux shell, compact (no pair-index holes): exactly
/// nAuxShells + 1 entries (the former per-shell phantom layout carried a
/// ~3 nAuxShells^2 / 2 entry tail).
/// \param nAuxShells The auxiliary shell count.
inline std::size_t MetricPairStoreBytes(std::size_t nAuxShells) noexcept {
    return (nAuxShells + 1) * sizeof(MdPairData);
}

/// The aux ket store of the RI-J combined store (ri_engine.cpp
/// BuildRiTensor / BuildAuxMetric): one BuildAuxPairData per aux SHELL -
/// the phantom (s_0, P) pair, the (0, lP) orientation of the store, the
/// phantom bra l = 0 with one row and one primitive crossed with the aux
/// shell - never the canonical aux PAIR list (the former auxStoreBytes
/// shape summed nAuxShells^2 / 2 full pair stores, a ~10^3-10^4x over-count
/// at scale that dominated the RI-J fit band where the tensor is marginal,
/// the over-count fix). The metric's filled pairs are the same store
/// (the metric loop shares this helper).
/// \param molecule The molecule (atom lookup for the atomic number).
/// \param auxBasisSet The auxiliary basis set (the shells' primitive counts).
/// \param auxPairList The canonical auxiliary pairs (the shells only).
inline std::size_t AuxKetsStoreBytes(const qcx::molecule::Molecule& molecule,
                                     const qcx::basisset::BasisSet& auxBasisSet,
                                     const ShellPairList& auxPairList) noexcept {
    std::size_t total = 0;

    for (const ShellInfo& shell : auxPairList.shells)
    {
        total += PairStoreBytesPerPair(0,
                                       shell.angularMomentum,
                                       false,
                                       shell.isSpherical,
                                       1,
                                       shell.contractionCount,
                                       ShellPrimitiveCount(molecule, auxBasisSet, shell));
    }

    return total;
}

/// The per-quartet assembly bytes of the class-batch metric builds
/// (md_batch.cpp AssembleClassBatches): the ShellQuartet input entry and
/// its canonical computed copy (32 B each), the assembly Item (88 B), the
/// per-task MdQuartetTask (24 B) and the per-task offset size_t - 176 B
/// per quartet on top of the 8 B per metric element, the metric-shape
/// mirror of the direct path's screened-quartet task mass
/// (kScreenedTaskBytesPerQuartet counts the same MdQuartetTask for the
/// denser direct-path shape). The batch machinery caps its own per-batch
/// memory (the itemBytes folding inside the cap), so the CAP is
/// batch-capped; the blocked-metric rung's per-STRIP lists (quartets,
/// items, computed, tasks) are capped by the strip's byte mass instead -
/// this constant is the never-under per-quartet allowance of that mass.
inline constexpr std::size_t kMetricAssemblyBytesPerQuartet = 176;

/// The blocked-metric rung's byte mass of the metric row q (ri_engine.cpp
/// BuildBlockedAuxMetric): the strip-wise (P|Q) build closes a strip
/// before the next row would push its byte mass over the batch cap, and
/// this is the never-under per-row allowance - the (q + 1) phantom
/// quartets of the row (the shared-phantom triangle rows (s_0, P_p | s_0,
/// P_q) over p <= q, one row per aux shell) at
/// kMetricAssemblyBytesPerQuartet each, plus the row's values (8 B per
/// element over the row's block columns: the row q's shells p <= q cover
/// exactly the functions [0, functionOffset + functionCount) of the
/// shell, so the values' function sum is (functionOffset +
/// functionCount) x the shell's own count).
/// \param shellIndex The row number (the aux shell index q - the row
/// holds the (q + 1) quartets of the shells p <= q).
/// \param shell The row's aux shell (functionOffset / ShellFunctionCount
/// give the row's value bytes).
inline std::size_t MetricRowStripBytes(std::size_t shellIndex, const ShellInfo& shell) noexcept {
    return kMetricAssemblyBytesPerQuartet * (shellIndex + 1) +
           8 * ShellFunctionCount(shell) * (shell.functionOffset + ShellFunctionCount(shell));
}

/// The blocked-metric rung's per-strip arena (ri_engine.cpp
/// BuildBlockedAuxMetric): the never-under byte bound of the largest
/// strip. The greedy strip partition accumulates rows while the running
/// mass (MetricRowStripBytes each) stays at or under the batch cap, so no
/// closed strip exceeds the cap; a row the cap alone cannot hold becomes
/// a forced single-row strip carrying its own row's mass (never-under -
/// the row alone is the strip, whatever the cap). The arena charges the
/// strip's values buffer and assembly lists together - the lists'
/// per-quartet bytes are part of the rows' masses above, so the bound
/// covers the lists too.
/// \param auxPairList The canonical auxiliary pairs (the metric rows).
/// \param maxBatchBytes The batch cap (the clamped value when clamped).
inline std::size_t BlockedMetricStripBytes(const ShellPairList& auxPairList,
                                           std::size_t maxBatchBytes) noexcept {
    std::size_t maxRowMass = 0;

    for (std::size_t q = 0; q < auxPairList.shells.size(); ++q)
    {
        maxRowMass = std::max(maxRowMass, MetricRowStripBytes(q, auxPairList.shells[q]));
    }

    return std::max(maxBatchBytes, maxRowMass);
}

/// The RI-J builder's FIRST Fock iteration's tensor-class terms: the per-call
/// allocations the Create-time
/// estimate historically omitted, so the first BuildFock overflowed the
/// gate (the fast rung's materialized riMatrix.transpose() copy and the
/// light rung's per-iteration recompute slice). Every input is available
/// at Create(). The formulas are never-under by construction: the transpose
/// is exactly the tensorBytes class again (8 n^2 nAux), the slice envelope
/// is 104 B per surviving task (the counted treatment: the
/// per-iteration batch loops of AssembleRiBatches iterate exactly the
/// screened task list, so the count the engine's Create-time decision
/// already computes - CountSchwarzSurvivingRiTasks, fock_screen.hpp - makes
/// charge == realized by equality; zero, the no-count default, charges the
/// unconditional dense grid as the never-under reference for the direct
/// tests and the compact fixtures, where every cell survives and the forms
/// coincide); the Item (56 B) + computed RiTask (16 B) +
/// per-batch MdQuartetTask (24 B) copies of AssembleRiBatches,
/// md_vrr_3c.hpp, verify 96 <= 104), the values buffer rides the clamped
/// batch cap, and the small-vector allowances cover the per-iteration
/// n^2-class working set (the dVec/fock/dMatrix and j product copies plus
/// the returned Fock tensor) and the nAux-class v/w vectors, with a
/// conservative w-step allowance and slack (ri_engine.cpp:1445-1486).
/// \param pairList The canonical orbital pairs.
/// \param auxPairList The canonical auxiliary pairs.
/// \param maxBatchBytes The batch cap (clamped or not).
/// \param lightRung True for the light-rung terms (the recompute slice and
/// the values buffer instead of the transpose).
/// \param screenedTaskCount The screened task-grid count the slice charges
/// when non-zero (the counted treatment: the surviving
/// (orbital pair, aux shell) cells the per-iteration batch loops iterate -
/// the same count RiFootprint's task-list charge consumes). Zero (the
/// default) charges the unconditional pairList.pairs x auxPairList.shells
/// grid instead - the never-under reference for callers without a count;
/// admission only widens.
struct RiFirstIterationTerms {
    std::size_t transposeBytes = 0; ///< Fast: the riMatrix.transpose() copy (8 n^2 nAux),
                                    ///< clamp-ineligible (basis-fixed).
    std::size_t sliceBytes = 0; ///< Light: the per-iteration recompute slice (104 bytes per
                                ///< surviving task - the counted treatment; the
                                ///< unconditional grid only as the caller's no-count never-under
                                ///< reference), clamp-ineligible.
    std::size_t valuesBufferBytes = 0; ///< Light: the per-iteration values buffer
                                       ///< (batch-capped, clamp-eligible).
    std::size_t smallVectorsBytes = 0; ///< Both: the per-iteration small-vector allowances.

    /// The first-iteration terms of the selected rung (the other rung's
    /// terms are zero).
    std::size_t Total() const noexcept {
        return transposeBytes + sliceBytes + valuesBufferBytes + smallVectorsBytes;
    }
};

inline RiFirstIterationTerms RiFirstIterationBytes(const ShellPairList& pairList,
                                                   const ShellPairList& auxPairList,
                                                   std::size_t maxBatchBytes,
                                                   bool lightRung,
                                                   std::size_t screenedTaskCount = 0) noexcept {
    RiFirstIterationTerms terms;
    const std::size_t n = pairList.functionCount;
    const std::size_t nAuxFuncs = auxPairList.functionCount;

    if (lightRung)
    {
        // The AssembleRiBatches per-iteration slice over the SCREENED task
        // count (the counted treatment, mirroring the
        // taskListBytes charge below: the count is the surviving-cell count
        // the per-iteration batch loops iterate, so charge == realized by
        // equality; the unconditional dense grid is the no-count caller's
        // never-under reference - the dense form is a special case and
        // admission only widens), the batch-capped values buffer, and the
        // light rung's n^2-class working set - the per-iteration product
        // vectors plus the returned Fock tensor (32 n^2; no dMatrix/
        // jColumn, the products run batch-at-a-time). The 8 nAuxFuncs^2
        // term is a conservative w-step allowance (the metric-eigenvector
        // products are a lazy Eigen view, never materialized).
        terms.sliceBytes =
            104 * (screenedTaskCount != 0 ? screenedTaskCount
                                          : pairList.pairs.size() * auxPairList.shells.size());
        terms.valuesBufferBytes = maxBatchBytes;
        terms.smallVectorsBytes = 8 * nAuxFuncs * nAuxFuncs + 32 * n * n;
    } else
    {
        // The materialized riMatrix.transpose() copy - a same-class copy of
        // the retained tensor, 8 n^2 nAux - and the fast rung's n^2-class
        // working set: dVec/fock/dMatrix, the jColumn product with its
        // fresh col(0) copy, and the returned Fock tensor (8 n^2 each,
        // ri_engine.cpp:1445-1486). The nAux-class v/w vectors carry slack
        // (40 nAuxFuncs); the 8 nAuxFuncs^2 term is a conservative w-step
        // allowance (the metric-eigenvector products are a lazy Eigen view,
        // never materialized).
        terms.transposeBytes = 8 * n * n * nAuxFuncs;
        terms.smallVectorsBytes = 8 * nAuxFuncs * nAuxFuncs + 48 * n * n + 40 * nAuxFuncs;
    }

    return terms;
}

/// The RI-J builder's own Create-time footprint terms (the fired-estimate
/// terms of the mode record; every field bytes), including the first Fock
/// iteration's tensor-class terms. The nested direct-exchange
/// half's estimate is separate (FockModeInfo::exchangeBytes) - its
/// per-iteration output buffers are the driver base's analytical n^2
/// term's subject (never double-charged here).
struct RiFootprintTerms {
    std::size_t orbitalStoreBytes = 0; ///< The orbital pair store (transient in BuildRiTensor).
    std::size_t auxStoreBytes = 0; ///< The aux ket store (one BuildAuxPairData per aux shell).
    std::size_t taskListBytes = 0; ///< The screened task-list charge (16 bytes per surviving task -
                                   ///< the counted treatment; the unconditional grid
                                   ///< only as the caller's no-count never-under reference).
    std::size_t tensorBytes =
        0; ///< The (uv|P) values buffer (8 n^2 nAux) - the exclusion (ii) term.
    std::size_t riMatrixBytes = 0; ///< The retained n^2 x nAux Eigen copy of the tensor.
    std::size_t metricBytes = 0; ///< The metric store (the shared-phantom pair store) and the
                                 ///< metric/eigen-class peak: the unblocked path's three live
                                 ///< nAuxFuncs^2 matrices (the TensorToEigen input retained
                                 ///< through the State copy, the solver's m_eivec and the
                                 ///< State eigenvector copy); the blocked-metric rung
                                 ///< carries two (its input dies before the State copy)
                                 ///< plus the per-strip arena instead of the full values
                                 ///< buffer.
    std::size_t scratchBytes = 0; ///< The 3c batch arena (batch x threads).
    std::size_t transposeBytes =
        0; ///< Fast: the per-iteration riMatrix.transpose() copy (8 n^2 nAux), clamp-ineligible.
    std::size_t sliceBytes = 0; ///< Light: the per-iteration recompute slice (104 bytes per
                                ///< surviving task - the counted treatment; the
                                ///< unconditional grid only as the caller's no-count never-under
                                ///< reference), clamp-ineligible.
    std::size_t valuesBufferBytes = 0; ///< Light: the per-iteration values buffer
                                       ///< (batch-capped, clamp-eligible).
    std::size_t smallVectorsBytes = 0; ///< Both: the per-iteration small-vector allowances.
    std::size_t retainedListBytes = 0; ///< The retained ShellPairList copies (16 B per pair +
                                       ///< 40 B per shell over both lists).
    std::size_t orbitActionBytes = 0; ///< The engaged task-grid orbit action's tables (both
                                      ///< bases; ri_orbit_action.hpp RiOrbitActionBytes), or 0
                                      ///< when the mechanism is not engaged.
    std::size_t schwarzSweepBytes = 0; ///< The Create-time screening pre-pass's EXCESS over
                                       ///< the store terms this estimate already carries (both
                                       ///< bases, the phase-max check RiFootprint applies; 0 at
                                       ///< every reachable size).

    /// The RI-J builder's own Create-time footprint (before the exchange
    /// half's estimate), INCLUDING the first Fock iteration's tensor-class
    /// terms - the reservation subject.
    std::size_t Total() const noexcept {
        return orbitalStoreBytes + auxStoreBytes + taskListBytes + tensorBytes + riMatrixBytes +
               metricBytes + scratchBytes + transposeBytes + sliceBytes + valuesBufferBytes +
               smallVectorsBytes + retainedListBytes + orbitActionBytes + schwarzSweepBytes;
    }
};

/// The RI-J builder's own Create-time footprint estimate: the orbital pair
/// store and the aux ket store, the screened task-list charge (the counted
/// treatment - screenedTaskCount x 16 B, the surviving
/// (bra pair x aux shell) cells CountSchwarzSurvivingRiTasks counts under
/// the 3c Schwarz cutoff and the engine's exact reserve realizes; the
/// unconditional nOrbitalPairs x nAuxShells grid, 16 B per task, is the
/// no-count reference - see \param screenedTaskCount), the (uv|P) values buffer
/// and the retained Eigen copy (8 n^2 nAux each - nAuxFuncs is the
/// FUNCTION count, never the shell count), the metric store with its
/// values buffer and the metric/eigen-class cluster (the unblocked
/// path's three live nAuxFuncs^2 matrices - the TensorToEigen input
/// retained through the State copy, the solver's m_eivec and the State
/// eigenvector copy - derived from the vendored Eigen internals: the
/// SelfAdjointEigenSolver holds exactly ONE nAuxFuncs^2 working matrix,
/// m_eivec, into which compute() copies the input's lower triangle; the
/// eigen-solver reconciliation of 2026-09-03), the 3c batch arena, and
/// the first Fock iteration's tensor-class terms (the fast
/// rung's transpose copy or the light rung's recompute slice, the
/// per-iteration small vectors, the light values buffer and the
/// retained lists).
/// \param molecule The molecule (atom lookup for the atomic number).
/// \param basisSet The orbital basis set.
/// \param auxBasisSet The auxiliary basis set.
/// \param pairList The canonical orbital pairs.
/// \param auxPairList The canonical auxiliary pairs.
/// \param maxBatchBytes The batch cap (already clamped when re-estimating).
/// \param threadCount The OpenMP team size read at Create.
/// \param lightRung True for the light-rung estimate (the per-iteration 3c
/// recompute mode): the (uv|P) values buffer and the retained Eigen copy
/// are never built, so their terms are zero. Everything else (stores, task
/// list, metric, scratch) is shared with the fast path - the light rung
/// retains the pair stores, the screened task list and the shell-pair
/// lists instead of the tensor (the retained lists' bytes are modeled as
/// retainedListBytes, the reconciliation item closed here).
/// \param blockedMetricRung True for the blocked-metric rung estimate (the
/// light rung with the metric built strip-wise): the
/// metric/eigen-class term drops from the unblocked 3 x 8 nAuxFuncs^2 to
/// the 2 x 8 nAuxFuncs^2 the blocked peak actually carries (the
/// strip-built solver input plus the solver's m_eivec - the input is
/// freed before the State eigenvector copy, so that copy's peak (m_eivec
/// plus the State matrix) stays at the same two), plus the per-strip
/// arena (BlockedMetricStripBytes - the strip's values buffer and
/// assembly lists; the strips never exceed max(maxBatchBytes, the
/// largest row's mass)). Meaningful only with lightRung (the blocked
/// metric is a light rung with a blocked metric build); ignored
/// otherwise.
/// \param screenedTaskCount The screened RI task-grid count: the surviving
/// (orbital pair, aux shell) cells of the
/// BuildScreenedRiTaskList grid under the preset's Schwarz cutoff - the
/// count the engine's decision block computes (CountSchwarzSurvivingRiTasks
/// over the pair-list Schwarz bounds and the aux diagonal) and the build's
/// exact reserve realizes, so the charge (this count x 16 B per RiTask)
/// equals the allocation - never-under by equality, the strongest
/// form. Zero (the default) charges the unconditional nOrbitalPairs x
/// nAuxShells grid instead - the dense never-under reference for callers
/// without a count (direct tests and the compact fixtures, where every
/// cell survives and the forms coincide); admission only widens.
/// \param orbitActionBytes The engaged task-grid orbit action's retained
/// tables (ri_orbit_action.hpp
/// RiOrbitActionBytes), or 0 (the default) when the mechanism is not
/// engaged - which is when no tables exist, so the default charges
/// nothing and the default path's envelope is untouched. The engine
/// passes the exact formula the action's own Bytes() reports, so charge ==
/// realized by equality (the strongest form), the same discipline the
/// screened task list's charge follows.
inline RiFootprintTerms RiFootprint(const qcx::molecule::Molecule& molecule,
                                    const qcx::basisset::BasisSet& basisSet,
                                    const qcx::basisset::BasisSet& auxBasisSet,
                                    const ShellPairList& pairList,
                                    const ShellPairList& auxPairList,
                                    std::size_t maxBatchBytes,
                                    std::size_t threadCount,
                                    bool lightRung = false,
                                    bool blockedMetricRung = false,
                                    std::size_t screenedTaskCount = 0,
                                    std::size_t orbitActionBytes = 0) noexcept {
    RiFootprintTerms terms;
    terms.orbitalStoreBytes = PairStoreBytes(molecule, basisSet, pairList);
    terms.auxStoreBytes = AuxKetsStoreBytes(molecule, auxBasisSet, auxPairList);
    terms.orbitActionBytes = orbitActionBytes;
    const std::size_t nOrbitalPairs = pairList.pairs.size();
    const std::size_t nAuxShells = auxPairList.shells.size();
    const std::size_t nAuxFuncs = auxPairList.functionCount;
    // The screened task list: the realized reserve is
    // the surviving-cell count x 16 B per RiTask (BuildScreenedRiTaskList's
    // exact reserve - the two-pointer count is bit-identical to the fill,
    // so charge == realized); the unconditional nOrbitalPairs x nAuxShells
    // grid is only the no-count (0) reference - the never-under dense form.
    terms.taskListBytes =
        (screenedTaskCount != 0 ? screenedTaskCount : nOrbitalPairs * nAuxShells) * 2 *
        sizeof(std::size_t);
    const std::size_t n = pairList.functionCount;

    if (!lightRung)
    {
        terms.tensorBytes = 8 * n * n * nAuxFuncs;
        terms.riMatrixBytes = 8 * n * n * nAuxFuncs;
    }

    // The metric/eigen-class peak, counted from the SAME solver internals
    // on both rungs (the eigen-solver reconciliation of 2026-09-03): the
    // vendored Eigen SelfAdjointEigenSolver holds exactly ONE nAuxFuncs^2
    // working matrix - m_eivec, into which compute() copies the input's
    // lower triangle (SelfAdjointEigenSolver.h:371-375 for the member
    // list - no m_matrix exists - and :436 for the copy); m_workspace,
    // m_eivalues, m_subdiag and m_hcoeffs are nAuxFuncs-class vectors,
    // and the tridiagonal reduction with the implicit-QR diagonalization
    // run fully in place (Tridiagonalization.h:332-365, the Householder Q
    // accumulation uses an n-vector workspace only). The rungs then
    // differ by the CALLER's input retention alone: the unblocked path
    // keeps its TensorToEigen (P|Q) copy alive through the State
    // constructor, whose eigenvector copy out of m_eivec is the third
    // live matrix (peak 3); the blocked rung frees its strip-built input
    // before that copy (ri_engine.cpp), so its solve peak (input plus
    // m_eivec) and its State-copy peak (m_eivec plus the copy) both carry
    // two. The old counts (6 = values buffer + five-matrix allowance on
    // the unblocked side, 3 with a phantom solver m_matrix on the blocked
    // side) shared one wrong premise - that the solver keeps a separate
    // copy of its input; the values buffer and the metric Tensor die
    // before the solve on the unblocked path, so they never join the
    // peak.
    if (blockedMetricRung)
    {
        // The blocked-metric rung's metric term: the
        // strip-wise build writes the (P|Q) matrix directly into the
        // solver's input - no full values buffer, no memory Tensor and no
        // TensorToEigen copy - so the eigen-class term is the two
        // nAuxFuncs^2 matrices the blocked peak actually carries (the
        // strip-built input and m_eivec), with the per-strip arena
        // (BlockedMetricStripBytes: the strip's values buffer and assembly
        // lists) riding on top. The filled (phantom, P) pairs are the same
        // per-shell ket store as the combined store's kets
        // (AuxKetsStoreBytes) - one shared ground truth.
        terms.metricBytes = MetricPairStoreBytes(nAuxShells) +
                            AuxKetsStoreBytes(molecule, auxBasisSet, auxPairList) +
                            2 * 8 * nAuxFuncs * nAuxFuncs +
                            BlockedMetricStripBytes(auxPairList, maxBatchBytes);
    } else
    {
        // The unblocked rung (fast, light and legacy paths): the metric
        // values buffer and the metric Tensor die before the solve, but
        // the TensorToEigen input matrix is retained through the State
        // constructor's eigenvector copy - the cluster's peak is the three
        // live nAuxFuncs^2 matrices (the retained input, m_eivec and the
        // State copy). The filled (phantom, P) pairs are the same
        // per-shell ket store as the combined store's kets
        // (AuxKetsStoreBytes) - one shared ground truth.
        terms.metricBytes = MetricPairStoreBytes(nAuxShells) + 3 * 8 * nAuxFuncs * nAuxFuncs +
                            AuxKetsStoreBytes(molecule, auxBasisSet, auxPairList);
    }

    terms.scratchBytes = maxBatchBytes * threadCount;

    // The first Fock iteration's tensor-class terms. The
    // nested exchange half's per-iteration output buffers are the
    // DirectFootprint exchangePerCallBytes term's subject - the premise
    // refuted at C24H50, measured ~12.6 GiB/call before the per-batch
    // restructure; the term now charges the per-batch bound, F64Bound +
    // F32Live of the values-term redesign); the driver base's n^2 term stays
    // the driver-side working set.
    const RiFirstIterationTerms firstIteration =
        RiFirstIterationBytes(pairList, auxPairList, maxBatchBytes, lightRung, screenedTaskCount);
    terms.transposeBytes = firstIteration.transposeBytes;
    terms.sliceBytes = firstIteration.sliceBytes;
    terms.valuesBufferBytes = firstIteration.valuesBufferBytes;
    terms.smallVectorsBytes = firstIteration.smallVectorsBytes;

    // The retained ShellPairList copies: 16 B per pair + 40 B per shell
    // over both lists (sizeof(ShellPairIndex) / sizeof(ShellInfo)). The
    // lists are retained only via the light payload (ri_engine.cpp:46-52,
    // nullopt on the fast rung, freed at Create return); charging both
    // rungs unconditionally is a defensible worst case under the CWA
    // heap-retention note.
    terms.retainedListBytes =
        (pairList.pairs.size() + auxPairList.pairs.size()) * sizeof(ShellPairIndex) +
        (pairList.shells.size() + auxPairList.shells.size()) * sizeof(ShellInfo);
    // The screening pre-pass over BOTH bases (SchwarzSweepBytes), as the
    // same PHASE MAX the direct footprint applies: the RI-J decision sweeps
    // the orbital and the aux pair list (ri_engine.cpp:252 and :259 in one
    // Create; :1689/:1736 in the budget decision), and each sweep's
    // chunked peak is dominated by the stores this estimate already charges
    // (orbitalStoreBytes - the full orbital store - and auxStoreBytes). The
    // term is the excess, so a sweep that ever outgrows its store is still
    // charged.
    const std::size_t orbitalSweepBytes = SchwarzSweepBytes(molecule, basisSet, pairList);
    const std::size_t auxSweepBytes = SchwarzSweepBytes(molecule, auxBasisSet, auxPairList);
    const std::size_t storeBytes = terms.orbitalStoreBytes + terms.auxStoreBytes;
    const std::size_t sweepBytes = orbitalSweepBytes + auxSweepBytes;
    terms.schwarzSweepBytes = sweepBytes > storeBytes ? sweepBytes - storeBytes : 0;
    return terms;
}

/// The QFMM outer store: every retained allocation of the QFMM state that
/// no other estimate covers (the nested near-field builder's own footprint
/// models ITS copies - the QFMM's own copies are the terms below). The
/// octree node count follows the leaf model - leaves = ceil(nPairs /
/// maxLeafSize), nodes = 2 * leaves - 1 (the flat node vector of
/// BuildQfmmTree); the per-leaf pair-index payloads (every pair lands in
/// exactly one leaf, so the payload sum is nPairs entries regardless of
/// the split shape), the child adjacency (the upper bound of 8 non-empty
/// octants per non-leaf node), the leafOfPair vector retained TWICE (the
/// tree result and the near-field restriction's copy), the restriction
/// bitset (nodes x nodes bits), the retained pair store (the QFMM's own
/// BuildPairData copy - the near-field builder's internal copy is its own
/// pairStoreBytes), the moment table (the CAP stride kQfmmMomentCount
/// blocks per function pair at any resolved lMult - blocks beyond the
/// resolved order stay zero, qfmm_lmult_test.cpp), the retained per-pair
/// geometries and the retained ShellPairList copy are all modeled exactly.
/// Only the far-field pair vector is unmodeled (the file comment).
/// \param molecule The molecule (atom lookup for the atomic number).
/// \param basisSet The basis set (the shells' primitive counts).
/// \param pairList The canonical shell pairs.
/// \param maxLeafSize The octree leaf-size cap.
inline std::size_t QfmmOuterStoreBytes(const qcx::molecule::Molecule& molecule,
                                       const qcx::basisset::BasisSet& basisSet,
                                       const ShellPairList& pairList,
                                       std::size_t maxLeafSize) noexcept {
    const std::size_t nPairs = pairList.pairs.size();
    const std::size_t leaves = (nPairs + maxLeafSize - 1) / maxLeafSize;
    const std::size_t nodes = 2 * leaves - 1;

    std::size_t total = nodes * sizeof(QfmmTreeNode);

    // The leaf pair-index payloads (the per-node vector HEADERS are inside
    // sizeof(QfmmTreeNode)).
    total += nPairs * sizeof(std::size_t);

    // The children adjacency: one vector header per node plus the bound of
    // 8 non-empty octants per non-leaf node (leaves - 1 of them).
    total += nodes * kVectorHeaderBytes + (leaves - 1) * 8 * sizeof(std::size_t);

    // The leafOfPair vector, retained twice (tree result + restriction).
    total += 2 * (nPairs * sizeof(std::size_t) + kVectorHeaderBytes);

    // The near-field restriction bitset: nodes x nodes bits.
    total += ((nodes * nodes + 63) / 64) * sizeof(std::size_t) + kVectorHeaderBytes;

    // The retained pair store: the QFMM's own BuildPairData copy.
    total += PairStoreBytes(molecule, basisSet, pairList);

    // The moment table: kQfmmMomentCount (the CAP stride) blocks per
    // function pair at any resolved lMult, the per-pair offsets and the
    // two vector headers.
    std::size_t momentFuncs = 0;

    for (const ShellPairIndex& pair : pairList.pairs)
    {
        momentFuncs += ShellFunctionCount(pairList.shells[pair.i]) *
                       ShellFunctionCount(pairList.shells[pair.j]);
    }

    total += kQfmmMomentCount * momentFuncs * sizeof(double) + (nPairs + 1) * sizeof(std::size_t) +
             2 * kVectorHeaderBytes;

    // The retained per-pair geometries (center + extent).
    total += nPairs * sizeof(QfmmPairGeometry) + kVectorHeaderBytes;

    // The retained ShellPairList copy (the nested builder re-sweeps its
    // own; the QFMM state keeps this one).
    total += pairList.shells.size() * sizeof(ShellInfo) +
             pairList.pairs.size() * sizeof(ShellPairIndex) + kVectorHeaderBytes;

    return total;
}

/// The four-rung refusal diagnostic (the shared text): the fast path
/// cannot fit the remaining workspace budget, and the light rung is the
/// next fit - the memory-fit failure ladder is direct screened,
/// batched/blocked, recompute, and disk LAST (a throughput choice, never
/// exclusion-only). The a-priori exclusion sentences append their own
/// clause.
/// \param builderName The builder family (direct / RI-J / QFMM).
/// \param estimateBytes The estimate that could not fit (0 on an
/// a-priori exclusion - no estimate was made).
/// \param remainingBytes The budget's remaining bytes at the decision.
/// \param patternExcluded True when exclusion (i) fired (the counted
/// Schwarz-screened pattern alone cannot fit - the fast path is skipped).
/// \param tensorExcluded True when exclusion (ii) fired (the RI-J tensor
/// alone cannot fit - the light rung is mandatory).
inline std::string FastPathRefusal(std::string_view builderName,
                                   std::size_t estimateBytes,
                                   std::size_t remainingBytes,
                                   bool patternExcluded,
                                   bool tensorExcluded) {
    std::string message = "the " + std::string(builderName) +
                          " fast path cannot fit the remaining workspace budget (" +
                          std::to_string(remainingBytes) + " bytes)";

    if (!patternExcluded && !tensorExcluded)
    {
        message +=
            ": the Create-time footprint estimate is " + std::to_string(estimateBytes) + " bytes";
    }

    message +=
        ". The memory-fit ladder is direct screened, batched/blocked, recompute, and disk LAST "
        "(a throughput choice, never exclusion-only): the light rung is the next fit";

    if (patternExcluded)
    {
        message += "; the counted Schwarz-screened neighbor pattern alone exceeds the budget";
    }

    if (tensorExcluded)
    {
        message += "; the (uv|P) tensor alone exceeds the budget (the light rung is mandatory)";
    }

    message += ".";
    return message;
}

/// The RI-J light-rung refusal: the recompute mode's own Create-time
/// footprint (the pair stores, the screened task list, the metric with its
/// eigendecomposition, the batch arenas and the nested exchange half)
/// cannot fit either. The next ladder step is the blocked-metric rung
/// which is BUILT and selectable - ri_engine.cpp's attempt()
/// takes it as its second argument, records it in the mode record and
/// dispatches BuildBlockedAuxMetric.
/// \param builderName The refusing builder ("RI-J").
/// \param estimateBytes The light-rung Create-time estimate that failed.
/// \param remainingBytes The budget's remaining bytes at the decision.
/// \param tensorExcluded True when the fast path was skipped a priori
/// (exclusion (ii): the tensor alone exceeds the budget).
/// \param patternExcluded True when the counted Schwarz-screened neighbor
/// pattern alone exceeds the budget (exclusion (i) under a forced light
/// rung - the nested exchange half's term, present in the light estimate
/// too).
inline std::string LightRungRefusal(std::string_view builderName,
                                    std::size_t estimateBytes,
                                    std::size_t remainingBytes,
                                    bool tensorExcluded,
                                    bool patternExcluded = false) {
    std::string message = "the " + std::string(builderName) +
                          " light rung cannot fit the remaining workspace budget (" +
                          std::to_string(remainingBytes) +
                          " bytes): the light-rung Create-time footprint estimate is " +
                          std::to_string(estimateBytes) + " bytes";

    if (tensorExcluded)
    {
        message += " (the (uv|P) tensor alone exceeds the budget - the fast path was never "
                   "attempted)";
    }

    message += ". The memory-fit ladder is direct screened, batched/blocked, recompute, and "
               "disk LAST (a throughput choice, never exclusion-only): the blocked-metric rung "
               "is the next rung.";

    if (patternExcluded)
    {
        message += " The counted Schwarz-screened neighbor pattern alone exceeds the budget.";
    }

    return message;
}

} // namespace qcx::integrals::internal
