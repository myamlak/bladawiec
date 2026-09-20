#pragma once

// The batch machinery of the matrix-form MD engine: compact per-pair data
// (primitive pairs, the contracted transforms, the certified bound helpers) and
// the per-{L_bra, L_ket}-class batches the instantiated kernels consume.
// The kernels live in md_vrr.hpp (the class template) and the pair/transform
// construction in md_hermite.hpp; the assembly itself in md_batch.cpp.

#include "md_defs.hpp"
#include "md_hermite.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/molecule/molecule.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace qcx::integrals::internal {

/// The pair-data chunk cap of the build-once walks (the Schwarz sweep,
/// one_electron.cpp's 1e prep): 512 MiB, the repo's standard batch
/// granularity (EriBatchOptions::maxBatchBytes, and the public
/// kSchwarzChunkBytes names the same cap for the sweep's own chunk
/// argument), so a chunker's resident pair data is one standard batch rather
/// than the whole store. One value for every chunker: the chunk is a policy
/// the callers share, and a second literal is how two chunkers drift.
inline constexpr std::size_t kPairChunkBytes = 512 * 1024 * 1024;

/// One primitive pair (a, b) of one shell pair: the Gaussian-product data
/// and the per-axis 1D Hermite coefficients of the (x-A)^ix (x-B)^jx
/// expansion - the seed of both transform matrices and of the 1e builders.
struct MdPrimPair {
    double p; ///< a + b.
    double exponentA; ///< a (the 1e kinetic builder needs it).
    double prefactor; ///< exp(-ab/p |A-B|^2).
    double px, py, pz; ///< The Gaussian-product center P (Bohr).
    /// Per axis x/y/z: E_t^{(ix,jx)} with ix in 0..la, jx in 0..lb,
    /// t in 0..la+lb. The (ix,jx) enumerate the Cartesian powers
    /// (kCartesianIndices); the spherical fold happens once per transform,
    /// not here.
    std::array<PerAxisETable, 3> perAxisTables;
};

/// The contracted data of one shell pair (i, j) - built once per pair and
/// shared by every quartet that contains it.
struct MdPairData {
    int la; ///< Angular momentum of shell i.
    int lb; ///< Angular momentum of shell j.
    bool isSphericalA; ///< Shell i spherical flag.
    bool isSphericalB; ///< Shell j spherical flag.
    std::size_t nFuncs; ///< Function rows of the pair block: n_i * n_j.
    std::size_t nFuncsA; ///< Function count of shell i.
    std::size_t nFuncsB; ///< Function count of shell j.
    std::size_t rowsA; ///< Contraction rows of shell i.
    std::size_t rowsB; ///< Contraction rows of shell j.
    std::size_t rowPairs; ///< Contraction row pairs: rows_i * rows_j.
    double ax, ay, az; ///< Center of shell i (Bohr).
    double bx, by, bz; ///< Center of shell j (Bohr).
    std::vector<MdPrimPair> primPairs; ///< In (a, b) primitive order.
    /// Per primitive pair, the bra-contraction weights d_a[rowA] * d_b[rowB]
    /// flattened [primPair][rowA * rows_j + rowB] - the row-pair-aware
    /// weight of the contraction-with-recurrence part 1 (general
    /// contractions; segmented bases degenerate to one row pair).
    std::vector<std::vector<double>> braWeights;
    /// The contracted bra transform T_ab^ctd: nFuncs x
    /// (rowPairs * primPairs * Hermite3DCount(la+lb)), row-major, rows in
    /// (f_b * n_i + f_a) order, the prim-pair slots carrying the weights
    /// d_a d_b (the contraction rides the p~ index - the VRR's pq rows are
    /// per prim pair and UNWEIGHTED, so the GEMM contracts the bra side
    /// exactly once).
    std::vector<double> braTransform;
    /// Per primitive pair (g, d): the ket transform T_ket^{(g,d)} with the
    /// weight d_g * d_d folded in: Hermite3DCount(la+lb) x nFuncs, row-major
    /// over t (the transpose orientation of braTransform).
    std::vector<std::vector<double>> ketTransforms;
    /// Certified-bound helpers: the max row-sum of |braTransform| and
    /// the max column-sum of the |ketTransforms| summed over primitives.
    double braRowSum;
    double ketColSum;
};

/// One quartet task inside a class batch.
struct MdQuartetTask {
    std::size_t braPair; ///< Index into the shared pair store.
    std::size_t ketPair; ///< Index into the shared pair store.
    std::size_t outputOffset; ///< Element offset of the packed block (batch-relative).
};

/// The canonical role form of one raw screened task (the streaming-rung
/// partial): the canonicalization of CanonicalizeQuartetOrder applied to
/// an already pair-index-canonical task (braPair >= ketPair on every
/// screened list and orbit rep, so its rule-1 shell-pair swap never fires -
/// only the L-role swap applies).
struct MdCanonicalTaskForm {
    std::size_t braPair; ///< The canonical bra pair (the lower-L role side).
    std::size_t ketPair; ///< The canonical ket pair.
    int lBra; ///< Total bra class: la + lb of braPair.
    int lKet; ///< Total ket class: lc + ld of ketPair.
};

/// Derives the canonical form of one raw task's pair roles - the exact
/// role rules of CanonicalizeQuartetOrder (the bra pair stays the larger
/// pair index; the L-role swap makes the canonical bra pair the lower-L
/// side). The sorted masters' sort-key source and the batch cursor's
/// per-position metrics source, so the sorted-master walks reproduce the
/// eager canonicalizer's forms exactly.
/// \param pairList The shell/pair list (pair ordering must match).
/// \param braPair The task's bra pair index.
/// \param ketPair The task's ket pair index.
/// \returns The canonical roles. Callers guarantee the pairs come from the
/// validated pair list (each shell l <= kMaxEngineL), so the canonical
/// l <= 2 * kMaxEngineL cap of CanonicalizeQuartetOrder always holds.
MdCanonicalTaskForm CanonicalTaskFormOf(const qcx::integrals::ShellPairList& pairList,
                                        std::size_t braPair,
                                        std::size_t ketPair) noexcept;

/// Sorts a screened task list IN PLACE into the canonical emission order:
/// CanonicalizeQuartetOrder's exact five-key ascending comparator over each
/// task's canonical form (lKet, lBra, ketPair, braRowPairs, braPair). The
/// sorted master is the whole-call survivor the per-half batch walks consume:
/// every master position's canonical form is
/// what the eager canonicalizer produced for the same raw tasks, so the
/// per-half batch sequences stay byte-identical.
/// \param pairList The shell/pair list.
/// \param tasks The task list to sort in place.
void SortScreenedTasks(const qcx::integrals::ShellPairList& pairList,
                       std::vector<MdQuartetTask>& tasks);

/// The fp32-lane overload: permutes the density weights alongside the tasks
/// (the weights stay parallel to the sorted master, so the contraction's
/// weight lookup becomes a direct index). The permutation is applied in
/// place through a transient index vector - 8 B x n, the documented fp32
/// sort transient of the streaming-rung audit.
/// \param weights The per-task density weights, parallel to \p tasks (the
/// ScreenAll fp32-lane output); permuted in place with the tasks.
void SortScreenedTasks(const qcx::integrals::ShellPairList& pairList,
                       std::vector<MdQuartetTask>& tasks,
                       std::vector<double>& weights);

/// One class batch: tasks of the same (L_bra, L_ket) class, sorted by ket
/// pair so equal-ket tasks are contiguous - the ket-transform GEMM batches
/// over those groups (md_vrr.hpp). The output pointers are batch-relative;
/// errorBounds is parallel to the assembled task order and starts at
/// boundsBase.
struct MdClassBatch {
    int lBra; ///< Total class of the bra pair: la + lb.
    int lKet; ///< Total class of the ket pair: lc + ld.
    const std::vector<MdPairData>* pairStore; ///< Shared pair store.
    std::vector<MdQuartetTask> tasks; ///< Sorted as described above.
    double* outF64 = nullptr; ///< fp64 output base (one batch of the caller's values).
    float* outF32 = nullptr; ///< fp32 output base (the certified lane).
    double* errorBounds = nullptr; ///< Per-quartet certified bounds (fp32 lane only).
    std::size_t boundsBase = 0; ///< Global index of the first task's bound.
};

/// The fp64 block element mass of one class batch: the sum over its tasks
/// of the bra pair's nFuncs times the ket pair's - the same per-batch
/// element count the layout passes accumulate (fock_build.cpp), here the
/// per-batch WORK SURROGATE the gpu_split partition accumulates. Cost-based
/// by construction, never a raw task or batch count.
/// \param batch The batch; must carry its pair store (assembled batches do).
/// \returns The total element count of the batch's tasks.
std::size_t BatchElementCount(const MdClassBatch& batch) noexcept;

/// True when one equal-ket group straddles the boundary between two
/// consecutive batches of one canonical batch sequence: the batches share
/// the (lBra, lKet) class and the boundary tasks (the last of \p before,
/// the first of \p after) share the kernel's equal-ket group key - the
/// ket pair and the bra contraction structure of md_vrr.hpp's group walk
/// (the row-pair count and the prim-pair count - the two quantities that
/// fix the GEMM shapes of a ket-transform reuse). A batch cut can land
/// inside an equal-ket run (the byte-cap cut rule of AssembleClassBatches
/// never inspects the grouping), so a partition between batches must check
/// the pair explicitly. The batches must be consecutive emissions of one
/// assembly, so their task spans tile the canonical master in order and
/// never overlap; batches of different classes return false (a batch never
/// spans a class run, and the kernel groups only within a class).
/// \param before The earlier batch.
/// \param after The next batch of the same sequence.
/// \returns True when an equal-ket group crosses the boundary.
bool EqualKetGroupStraddles(const MdClassBatch& before, const MdClassBatch& after) noexcept;

/// The deterministic gpu_split partition index of one canonical batch
/// sequence (the batch-partition contract of 2026-09-04): the split
/// point over CUMULATIVE ESTIMATED BATCH COST - the BatchElementCount
/// mass - at the NEAREST EQUAL-KET GROUP BOUNDARY, never inside a group.
/// The CPU side takes the canonical prefix batches [0, index), the device
/// side the canonical suffix [index, size); the target prefix mass is
/// (1 - deviceShare) x the sequence total, and the index is the
/// group-clean boundary (0, size, or any boundary where
/// EqualKetGroupStraddles is false) whose prefix mass is nearest the
/// target - exact ties resolve to the earlier boundary. Boundaries 0 and
/// size are always group-clean (nothing straddles outside the stream):
/// deviceShare 0 -> the empty suffix (index = size; the CPU-only
/// fallback, the k=1 path intact) and deviceShare 1 -> the empty
/// prefix (index = 0). Pure and deterministic: identical batches and
/// share produce the identical index. Not called by any production path:
/// the flag-gated executor (and
/// this file's tests) consume it.
/// \param batches The canonical batch vector in emission order.
/// \param deviceShare The device side's share of the cumulative element
/// mass, clamped to [0, 1].
/// \returns The suffix's first batch index, in [0, size].
std::size_t GpuSplitBatchIndex(const std::vector<MdClassBatch>& batches,
                               double deviceShare) noexcept;

/// Builds the contracted pair data of one shell pair: the per-primitive-pair
/// geometry (p, P, prefactor, per-axis E tables), the per-primitive-pair
/// bra-contraction weights (row-pair-aware: general contractions are
/// handled, segmented ones degenerate to one row pair), the bra transform
/// T_ab^ctd, the per-primitive-pair ket transforms, and the certified bound
/// helpers. \p centerA/\p centerB are the shell centers (Bohr). Implemented
/// in md_batch.cpp.
void BuildContractedPairTransform(MdPairData& pair,
                                  const MdShellContractions& shellA,
                                  const MdShellContractions& shellB,
                                  const std::array<double, 3>& centerA,
                                  const std::array<double, 3>& centerB);

/// The flattened per-shell contraction data (one entry per ShellInfo) with
/// the shell center - the input shape of the pair/aux data builders.
struct MdShellInput {
    MdShellContractions contractions;
    double cx, cy, cz; ///< Center (Bohr).
};

/// Flattens the shells of a pair list into contraction data.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set the shells came from.
/// \param pairList The flattened shell/pair list.
/// \returns The shell inputs in pairList.shells order, or an Error
/// (kInvalidArgument when an atom has no basis entry).
qcx::Result<std::vector<MdShellInput>> FlattenShells(const qcx::molecule::Molecule& molecule,
                                                     const qcx::basisset::BasisSet& basisSet,
                                                     const qcx::integrals::ShellPairList& pairList);

/// Builds the contracted pair data of every canonical pair (the per-axis E
/// tables and both transforms - md_hermite.hpp).
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell l <= kMaxEngineL (checked by the
/// callers through BuildShellPairs).
/// \param pairList The flattened shell/pair list.
/// \returns The pair store, or an Error.
qcx::Result<std::vector<MdPairData>> BuildPairData(const qcx::molecule::Molecule& molecule,
                                                   const qcx::basisset::BasisSet& basisSet,
                                                   const qcx::integrals::ShellPairList& pairList);

/// Fills the geometry-only fields of a pair store - la/lb, the spherical
/// flags and the two centers per pair, exactly the fields BuildPairData
/// derives from the flattened shells - leaving the computed fields and every
/// transform vector EMPTY. The result is the skeleton a chunk-store consumer
/// keeps resident while it builds one chunk at a time (BuildChunkPairData
/// fills the listed pairs' computed fields and transforms into it, its empty
/// braTransform marking the not-built pairs). \p store must already have one
/// default-constructed entry per canonical pair (the pair index IS the store
/// index, so the skeleton is the full pair list's shape at
/// sizeof(MdPairData) per pair - no per-pair payload).
/// \param store The skeleton store (resized by the caller).
/// \param shells The pair list's FlattenShells output.
/// \param pairList The flattened shell/pair list.
void FillPairGeometry(std::vector<MdPairData>& store,
                      const std::vector<MdShellInput>& shells,
                      const qcx::integrals::ShellPairList& pairList);

/// The LightPath chunk-view builder: fills the contracted transforms of the
/// listed FULL-SPACE pair
/// indices into a store that already carries their geometry - the Create-
/// time light store, reused as the per-chunk arena. The transforms are the
/// exact BuildContractedPairTransform values (bit-identical by construction
/// - the same function, the same pair order), so quartets assembled against
/// the chunk store contract identically to the FastPath store. \p shells
/// must be the pair list's FlattenShells output.
void BuildChunkPairData(std::vector<MdPairData>& store,
                        const std::vector<MdShellInput>& shells,
                        const qcx::integrals::ShellPairList& pairList,
                        const std::vector<std::size_t>& pairs);

/// The LightPath chunk-view teardown: clears the flat bra transform and the
/// certified sums of the listed pairs so the next chunk's build starts clean. The
/// geometry fields, the prim pairs (their nested weights/transforms and the
/// per-axis E tables each carries) and every vector capacity survive - the
/// chunk arena is capacity-preserved by contract (the reused
/// containers are not Reservations, so per-chunk alloc/free cannot stack
/// under cumulative high-water accounting), and the pair builder re-fits every slot of the
/// prim containers it re-uses. A pair whose braTransform is empty is not
/// built - the marker the chunk pass reads (never primPairs.size()).
void ClearChunkPairData(std::vector<MdPairData>& store, const std::vector<std::size_t>& pairs);

/// Releases the listed pairs' contracted payload COMPLETELY - the primitive
/// pairs (each carrying its three per-axis E-table vectors), the contraction
/// weights and both transforms - so the store's resident payload is the open
/// chunk and nothing else. Deliberately NOT ClearChunkPairData: that teardown
/// preserves the per-prim-pair containers because its callers re-use them for
/// pairs they will build again, and a caller that walks the pair list ONCE
/// (each pair built exactly once, never revisited) that keeps them retains a
/// byte per pair - measured on the C42H86/def2-QZVP (4,974-function) point:
/// the process working set ramped to ~11 GiB with the capacity-preserving
/// teardown against the ~1.3 GiB the chunk bound describes. A build-once walk
/// must use THIS teardown for its peak to be the chunk; the emptied
/// braTransform is the not-built marker either way.
/// \param store The build-once walk's store.
/// \param pairs The chunk's pair indices.
void ReleaseChunkPairData(std::vector<MdPairData>& store, const std::vector<std::size_t>& pairs);

/// The pair payload bytes a chunk store materializes for ONE canonical pair:
/// the pair store's per-pair bytes minus the geometry-only MdPairData the
/// skeleton already carries. The chunkers size their chunks with it - one
/// formula, every chunker (the same arithmetic fock_build.cpp's
/// MaxPairPayload uses for the LightPath arena).
/// \param molecule The molecule (atom lookup for the atomic number).
/// \param basisSet The basis set (the shells' primitive counts).
/// \param pairList The canonical shell pairs.
/// \param pairIndex The pair whose payload to measure.
/// \returns The pair's payload bytes (0 for an out-of-range index).
std::size_t ChunkPairPayloadBytes(const qcx::molecule::Molecule& molecule,
                                  const qcx::basisset::BasisSet& basisSet,
                                  const qcx::integrals::ShellPairList& pairList,
                                  std::size_t pairIndex);

/// Canonicalizes the caller quartets (i <= j, k <= l, pair index (i,j) >=
/// (k,l), then L_bra <= L_ket) and groups them into per-class batches, each
/// capped at maxBatchBytes (output + scratch + convert, conservative - the
/// convert sum over-counts the kernel's per-batch max, deliberately). The
/// computed forms are reported through \p computed in final output order.
/// \param pairStore The pair data (pair ordering must match pairList).
/// \param pairList The shell/pair list (function offsets per shell).
/// \param quartets The caller quartets, in any form.
/// \param maxBatchBytes Per-class batch cap; must be positive.
/// \param computed Receives the canonicalized quartets in final order.
/// \returns The class batches, or an Error.
qcx::Result<std::vector<MdClassBatch>> AssembleClassBatches(
    const std::vector<MdPairData>& pairStore,
    const qcx::integrals::ShellPairList& pairList,
    const std::vector<qcx::integrals::ShellQuartet>& quartets,
    std::size_t maxBatchBytes,
    std::vector<qcx::integrals::ShellQuartet>& computed);

/// Materializes the class batch over master positions [from, to) - the
/// shared fill of the cursor and of the k > 1 slots' per-pull slicing
/// (each slot slices its claimed batch's master range directly; the fill
/// is MdClassBatchCursor::FillBatch's exact body, so the slot-built
/// descriptors are byte-identical to the cursor's emissions): the run's
/// class, boundsBase = from, and the tasks with the canonical-role pairs
/// and batch-relative cumulative output offsets. \p tasks must be the
/// canonical-sorted master \p from/\p to index.
void FillMdClassBatch(MdClassBatch& out,
                      const std::vector<MdPairData>& pairStore,
                      const qcx::integrals::ShellPairList& pairList,
                      const std::vector<MdQuartetTask>& tasks,
                      std::size_t from,
                      std::size_t to);

/// The incremental per-half batch partitioner over a canonical-sorted task
/// list (the streaming-rung partial):
/// emits the class batches ONE at a time with AssembleClassBatches' exact
/// run/cut rules over the same per-task payload bytes (the shared metrics
/// of md_batch.cpp), so the emitted batch sequence is byte-identical to the
/// eager builder's for the same raw-task multiset - the per-half pin. The
/// batches carry the canonical-role pairs, the per-run class, the
/// batch-relative cumulative output offsets and boundsBase = the batch's
/// first master position, exactly as AssembleClassBatches reports them.
class MdClassBatchCursor {
public:
    MdClassBatchCursor(const std::vector<MdPairData>& pairStore,
                       const qcx::integrals::ShellPairList& pairList,
                       const std::vector<MdQuartetTask>& tasks,
                       std::size_t maxBatchBytes);

    /// Advances to the next batch in canonical emission order. \p out is
    /// filled only when \p countOnly is false (its previous contents are
    /// discarded; fresh MdClassBatch objects per emission are fine).
    /// \param out Receives the batch (pairStore = the store of
    /// construction; outF64/outF32/errorBounds stay untouched - the
    /// assembly step fills them from its own buffers).
    /// \param countOnly Dry-run mode: only the run/cut boundaries are
    /// computed - the O(n) batch-count pass the kEff decision needs before
    /// the k-slot loops open - and the task payload is not materialized.
    /// \param batchStarts Optional collector of each emitted batch's first
    /// master position - the k > 1 slots' per-batch slice table (8 B x
    /// numBatches, the batch-boundary metadata of the streaming-rung
    /// charge). Appended to per emission in
    /// both modes.
    /// \returns False when the list is exhausted (a count pass's total).
    bool Next(MdClassBatch& out,
              bool countOnly = false,
              std::vector<std::size_t>* batchStarts = nullptr);

    /// Rewinds to the list head (the producer's second, materializing walk
    /// after the count pass).
    void Reset();

private:
    const std::vector<MdPairData>& _pairStore; ///< The shared pair store.
    const qcx::integrals::ShellPairList& _pairList; ///< The shell/pair list.
    /// The canonical-sorted master; must outlive the cursor and stay sorted
    /// (the callers sort once per call and never mutate between walks).
    const std::vector<MdQuartetTask>& _tasks;
    const std::size_t _maxBatchBytes; ///< The per-class batch cap.
    std::size_t _runEnd = 0; ///< End of the current class run (discovered lazily).
    std::size_t _batchStart = 0; ///< First master position of the open batch.
    std::size_t _pos = 0; ///< The next master position to examine.
    std::size_t _batchBytes = 0; ///< Item bytes accumulated over [_batchStart, _pos).

    /// Materializes the batch over master positions [from, to): the run's
    /// class, boundsBase = from, and the tasks with the canonical-role pairs
    /// and batch-relative cumulative output offsets (AssembleClassBatches'
    /// exact per-batch fill).
    void FillBatch(MdClassBatch& out, std::size_t from, std::size_t to) const;
};

/// Runs every batch through the class dispatch table (md_engine.hpp) on the
/// CPU backend (ParallelFor over the class-major batch vector - the fixed
/// per-class chunk assignment of the scaling audit). Every
/// batch writes only its own disjoint output block and its own global certified
/// bound slots, so the run is bit-identical across team sizes by
/// construction (no cross-batch accumulation exists).
/// \param batches The assembled class batches (output pointers filled in).
/// \param regionThreads The region's thread count (0 = DefaultOmpTeamSize());
/// the bounded-concurrency seam - each concurrent slot's assemble
/// region forks its slot's share of the team (q = max(1, team/k)), never the
/// full team, so the k-slot thread product stays at or under the concurrency
/// policy's team.
/// \returns An Error (kUnimplemented) when a class has no instantiation -
/// cannot happen after the l <= kMaxEngineL check.
qcx::Result<void> RunBatches(const std::vector<MdClassBatch>& batches,
                             std::size_t regionThreads = 0);

} // namespace qcx::integrals::internal
