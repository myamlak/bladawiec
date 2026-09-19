// The host orchestration of the CUDA MD ERI lane (the device kernels and
// the class table live in eri_cuda.cu and
// the generated eri_cuda_dispatch_gen.hpp). Responsibilities:
//
//   - Create: canonical shell pairs -> the compact device tables (pair
//     metadata, primitives, the replicated bra transforms, the per-g ket
//     transforms, the Boys tables), uploaded once into DeviceBuffers; the
//     cuBLAS handle lifecycle lives with the engine.
//   - ComputeBatch/ComputeBatchCertified: the CPU-side canonicalization
//     (AssembleClassBatches, verbatim), the per-class variant heuristic and
//     override validation, the fused-kernel chunk geometry and task-range
//     partitioning, or the V2 per-run strided-batched cuBLAS transform
//     GEMMs, then the result copy-back and assembly.
//   - ComputeRiBatch: the 3c path - the same kernels over the combined
//     orbital+aux pair store with a per-call upload (the CPU assembly of
//     md_vrr_3c.hpp RunRiBatches is replicated here, because that inline
//     helper dispatches the CPU kernels - the sort and the run/batch split
//     are copied verbatim).
//
// fp64 values match the CPU lane to the tolerance the tests pin (1e-12 *
// scale): the device pow/exp differ from MSVC by 1-2 ulp. The certified
// fp32 lane carries the exact CPU bound formula (md_vrr.hpp).

#include "qcx/integrals/eri_cuda.hpp"

#include "boys_coefficients.hpp"
#include "internal/device_footprint.hpp"
#include "internal/eri_cuda_fock.hpp"
#include "internal/fock_screen.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_defs.hpp"
#include "internal/md_vrr_3c.hpp"
#include "internal/precision_ladder.hpp"
#include "internal/tensor_eigen_bridge.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/gpu_fock_build.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/cuda_allocator_traits.hpp"
#include "qcx/memory/device_buffer.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define QCX_ERI_CUDA_HEADER_ONLY
#include "eri_cuda.cu"

namespace qcx::integrals {

using qcx::backend::CudaTag;
using qcx::integrals::internal::Hermite3DCount;
using qcx::integrals::internal::MdClassBatch;
using qcx::integrals::internal::MdPairData;
using qcx::integrals::internal::MdQuartetTask;
using qcx::integrals::internal::cuda::EriCudaBoysTables;
using qcx::integrals::internal::cuda::EriCudaClassArgs;
using qcx::integrals::internal::cuda::EriCudaClassDispatchEntry;
using qcx::integrals::internal::cuda::EriCudaFockArgs;
using qcx::integrals::internal::cuda::EriCudaFockTaskMeta;
using qcx::integrals::internal::cuda::EriCudaPairMeta;
using qcx::integrals::internal::cuda::EriCudaTask;
using qcx::integrals::internal::cuda::kEriCudaCertifiedEpsilon;
using qcx::integrals::internal::cuda::kEriCudaFamilyV0;
using qcx::integrals::internal::cuda::kEriCudaFamilyV1;
using qcx::integrals::internal::cuda::kEriCudaFamilyV2;
using qcx::integrals::internal::cuda::kEriCudaFockPrecisionFp32;
using qcx::integrals::internal::cuda::kEriCudaFockPrecisionFp64;
using qcx::integrals::internal::cuda::kEriCudaFp32Unavailable;
using qcx::integrals::internal::cuda::kEriCudaFusedThreadsV0;
using qcx::integrals::internal::cuda::kEriCudaFusedThreadsV1;
using qcx::integrals::internal::cuda::kEriCudaInvalidFamily;
using qcx::integrals::internal::cuda::kEriCudaMaxSharedBytes;
using qcx::integrals::internal::cuda::kEriCudaNotRegistered;
using qcx::integrals::internal::cuda::kEriCudaOk;
using qcx::integrals::internal::cuda::kEriCudaPrecisionFp32;
using qcx::integrals::internal::cuda::kEriCudaPrecisionFp64;
using qcx::integrals::internal::cuda::kEriCudaSharedOverflow;
using qcx::memory::DeviceBuffer;

namespace {

Error DeviceError(const std::string& what) {
    return Error{ErrorCode::kDeviceError, what};
}

Error Unimplemented(const std::string& what) {
    return Error{ErrorCode::kUnimplemented, what};
}

Error InvalidArgument(const std::string& what) {
    return Error{ErrorCode::kInvalidArgument, what};
}

/// The device-budget refusal: the four-rung memory-fit
/// ladder fires with the device context named (the engine's current CUDA
/// device, and the attached budget's declared device when they differ),
/// kInvalidArgument, before any device allocation of the Create - the light
/// rung (the per-call statics upload) is the last device rung, and when even
/// it cannot fit the declared budget the refusal names the standing ladder
/// and the reinstatement options instead of letting a CUDA allocation fail
/// (the ladder fires before device OOM, never after).
/// \param deviceOrdinal The engine's current device (cudaGetDevice).
/// \param budgetDeviceId The device the attached budget was declared for.
/// \param fastEstimateBytes The fast rung's footprint total (statics
/// retained at Create).
/// \param lightEstimateBytes The light rung's per-call working set.
/// \param remainingBytes The budget's Remaining at the decision.
Error DeviceBudgetRefusal(int deviceOrdinal,
                          int budgetDeviceId,
                          std::size_t fastEstimateBytes,
                          std::size_t lightEstimateBytes,
                          std::size_t remainingBytes) {
    std::string message =
        "the gpu fock build cannot fit the device workspace budget on device " +
        std::to_string(deviceOrdinal) + " (" + std::to_string(remainingBytes) +
        " bytes remaining): the fast rung needs " + std::to_string(fastEstimateBytes) +
        " bytes (statics retained at Create) and the light rung needs " +
        std::to_string(lightEstimateBytes) +
        " bytes (statics uploaded per call). The memory-fit ladder is direct screened, "
        "batched/blocked, recompute, and disk LAST (a throughput choice, never "
        "exclusion-only): the light rung is the last device rung, and no device rung "
        "fits the budget - raise the device budget, use the CPU builder, or name the "
        "disk-backed store.";

    if (budgetDeviceId != deviceOrdinal)
    {
        message += " The attached budget declares device " + std::to_string(budgetDeviceId) + ".";
    }

    return Error{ErrorCode::kInvalidArgument, std::move(message)};
}

Result<void> CheckCudaError(cudaError_t err, const char* what) {
    if (err == cudaSuccess)
    {
        return {};
    }

    return std::unexpected(DeviceError(std::string(what) + ": " + cudaGetErrorString(err)));
}

Result<void> CheckCublasStatus(cublasStatus_t status, const char* what) {
    if (status == CUBLAS_STATUS_SUCCESS)
    {
        return {};
    }

    return std::unexpected(DeviceError(std::string(what) + ": " + cublasGetStatusName(status)));
}

/// Maps a per-class launcher return code (0 = ok, negative = the kernel
/// codes of eri_cuda.cu, positive = cudaError_t) to a qcx Error.
Result<void> CheckLaunchCode(int code, const char* what) {
    if (code == kEriCudaOk)
    {
        return {};
    }

    if (code == kEriCudaNotRegistered)
    {
        return std::unexpected(
            Unimplemented(std::string(what) + ": the class is not instantiated"));
    }

    if (code == kEriCudaFp32Unavailable)
    {
        return std::unexpected(
            Unimplemented(std::string(what) + ": the fp32 pipeline is not instantiated"));
    }

    if (code == kEriCudaInvalidFamily)
    {
        return std::unexpected(Unimplemented(std::string(what) + ": unsupported variant"));
    }

    if (code == kEriCudaSharedOverflow)
    {
        return std::unexpected(
            DeviceError(std::string(what) + ": the shared-memory budget is exceeded"));
    }

    return std::unexpected(
        DeviceError(std::string(what) + ": " + cudaGetErrorString(static_cast<cudaError_t>(code))));
}

/// The per-class variant heuristic:
///
///   kV0 (128 threads) for L <= 4: the per-thread working set - the pq band
///     (kHermBra * kHermKet), the acc rows (kHermBra * nFuncsKet), and the
///     rotating 3-tier slice - stays below ~1024 fp64 words (the register
///     budget of one 128-thread block); the shared pool covers the batch.
///   kV1 (256 threads) for 5 <= L <= 8: the per-thread set overflows
///     registers, so the intermediates move to shared memory; 256 threads
///     keep the row-pair parallelism high.
///   kV2 for L >= 9: the per-row shared budget no longer fits the 44 KB
///     pool at a useful chunk depth, so the VRR writes global memory and
///     the transforms run as cuBLAS strided-batched GEMMs.
int SelectFamily(int lBra, int lKet) {
    const int l = lBra + lKet;

    if (l <= 4)
    {
        return kEriCudaFamilyV0;
    }

    if (l <= 8)
    {
        return kEriCudaFamilyV1;
    }

    return kEriCudaFamilyV2;
}

/// The fused kernel's shared-memory depth: the 44 KB pool (48 KB minus the
/// driver reserve and the per-chunk slot-bound block) divided by the
/// per-row bytes - one row = one (rowAB, primB) slot of pq
/// (kHermBra * kHermKet) plus acc (kHermBra * maxNFuncsKet) - capped by the
/// task's slot count (maxRowPairs * maxNBraPrims) and the block's thread
/// count (every slot must have an owner). 0 means the class cannot fit
/// (the launcher then reports kEriCudaSharedOverflow; at cudaLmax=6 every
/// fused class fits).
int ComputeChunkRows(int lBra,
                     int lKet,
                     int maxRowPairs,
                     int maxNBraPrims,
                     int maxNFuncsKet,
                     int threads,
                     bool certified) {
    const int kHermBra = Hermite3DCount(lBra);
    const int kHermKet = Hermite3DCount(lKet);
    const std::size_t elementSize = certified ? sizeof(float) : sizeof(double);
    const std::size_t bytesPerRow = static_cast<std::size_t>(kHermBra) *
                                    (static_cast<std::size_t>(kHermKet) + maxNFuncsKet) *
                                    elementSize;
    // The kernel stages the per-chunk slot bounds first (chunkRows doubles)
    // and pads the pq/acc block to 8-byte alignment; the launcher's
    // sharedBytes formula mirrors this exactly: one chunk row costs
    // bytesPerRow plus one bound double.
    const std::size_t budget = static_cast<std::size_t>(kEriCudaMaxSharedBytes - 8);
    const std::size_t perRow = bytesPerRow + sizeof(double);

    if (bytesPerRow == 0 || budget < perRow)
    {
        return 0;
    }

    std::size_t rows = budget / perRow;
    const std::size_t maxSlots = static_cast<std::size_t>(maxRowPairs) * maxNBraPrims;

    if (rows > maxSlots)
    {
        rows = maxSlots;
    }

    if (rows > static_cast<std::size_t>(threads))
    {
        rows = static_cast<std::size_t>(threads);
    }

    return static_cast<int>(rows);
}

/// Flattens the Boys constants (boys_coefficients.hpp) into the device
/// struct: per order, the pieces of kPieceStart[order]..kPieceStart[order+1]
/// with the coefficients re-based into one running array (the boys_cuda.cu
/// upload pattern; the device FindPiece is a linear scan over the order's
/// pieces, identical semantics to the CPU kPieceStart loop).
Result<void> BuildBoysTables(EriCudaBoysTables& tables) {
    int runningOffset = 0;

    for (int order = 0; order <= detail::kMaxOrder; ++order)
    {
        const int first = detail::kPieceStart[order];
        const int last = detail::kPieceStart[order + 1];
        const int count = last - first;

        if (count > internal::cuda::kEriCudaMaxPieces)
        {
            return std::unexpected(DeviceError("the Boys piece layout exceeds the device tables"));
        }

        tables.count64[order] = count;

        for (int p = 0; p < count; ++p)
        {
            const detail::OrderPiece& piece = detail::kPieces[first + p];
            tables.a64[order][p] = piece.a;
            tables.b64[order][p] = piece.b;
            tables.deg64[order][p] = piece.deg;
            tables.offset64[order][p] = runningOffset;

            if (runningOffset + piece.deg + 1 > internal::cuda::kEriCudaMaxCoeffs)
            {
                return std::unexpected(
                    DeviceError("the Boys coefficient layout exceeds the device tables"));
            }

            for (int c = 0; c <= piece.deg; ++c)
            {
                tables.coeffs64[runningOffset] = detail::kCoeffs[piece.offset + c];
                ++runningOffset;
            }
        }
    }

    tables.bDeg64 = detail::kBDeg;

    for (int c = 0; c <= detail::kBDeg; ++c)
    {
        tables.bcoeffs64[c] = detail::kBcoeffs[c];
    }

    runningOffset = 0;

    for (int order = 0; order <= detail::kMaxOrder; ++order)
    {
        const int first = detail::f32::kPieceStart[order];
        const int last = detail::f32::kPieceStart[order + 1];
        const int count = last - first;

        if (count > internal::cuda::kEriCudaMaxPieces)
        {
            return std::unexpected(
                DeviceError("the Boys f32 piece layout exceeds the device tables"));
        }

        tables.count32[order] = count;

        for (int p = 0; p < count; ++p)
        {
            const detail::f32::OrderPiece& piece = detail::f32::kPieces[first + p];
            tables.a32[order][p] = piece.a;
            tables.b32[order][p] = piece.b;
            tables.deg32[order][p] = piece.deg;
            tables.offset32[order][p] = runningOffset;

            if (runningOffset + piece.deg + 1 > internal::cuda::kEriCudaMaxCoeffs)
            {
                return std::unexpected(
                    DeviceError("the Boys f32 coefficient layout exceeds the device tables"));
            }

            for (int c = 0; c <= piece.deg; ++c)
            {
                tables.coeffs32[runningOffset] = detail::f32::kCoeffs[piece.offset + c];
                ++runningOffset;
            }
        }
    }

    tables.bDeg32 = detail::f32::kBDeg;

    for (int c = 0; c <= detail::f32::kBDeg; ++c)
    {
        tables.bcoeffs32[c] = detail::f32::kBcoeffs[c];
    }

    return {};
}

/// The flattened pair tables one Compute* call runs against (the 2e path:
/// the engine's cached copies; the RI path: a per-call combined store).
struct HostTables {
    std::vector<EriCudaPairMeta> pairMeta;
    std::vector<double> primData;
    std::vector<double> braTransforms;
    std::vector<double> ketTransforms;
    std::vector<float> braTransformsF32;
    std::vector<float> ketTransformsF32;
};

Result<HostTables> BuildHostTables(const std::vector<MdPairData>& pairStore) {
    HostTables tables;
    tables.pairMeta.reserve(pairStore.size());

    for (const MdPairData& pair : pairStore)
    {
        const int lBra = pair.la + pair.lb;
        const std::size_t hermBra = static_cast<std::size_t>(Hermite3DCount(lBra));
        const std::size_t kBra = pair.rowPairs * pair.primPairs.size() * hermBra;

        EriCudaPairMeta meta{};
        meta.la = pair.la;
        meta.lb = pair.lb;
        meta.nFuncs = static_cast<int>(pair.nFuncs);
        meta.rowPairs = static_cast<int>(pair.rowPairs);
        meta.braRowSum = pair.braRowSum;
        meta.ketColSum = pair.ketColSum;
        meta.primOffset = tables.primData.size();
        meta.primCount = static_cast<int>(pair.primPairs.size());
        meta.braTransformOffset = tables.braTransforms.size();
        meta.braTransformK = static_cast<int>(kBra);
        meta.ketTransformOffset = tables.ketTransforms.size();

        for (const internal::MdPrimPair& prim : pair.primPairs)
        {
            tables.primData.push_back(prim.p);
            tables.primData.push_back(prim.exponentA);
            tables.primData.push_back(prim.prefactor);
            tables.primData.push_back(prim.px);
            tables.primData.push_back(prim.py);
            tables.primData.push_back(prim.pz);
        }

        // The full 4D bra transform: a verbatim copy
        // of the CPU's pair.braTransform, [f][rowAB][primB][t] with the
        // per-primitive contraction weights folded in - the exact layout
        // md_batch.cpp builds (flat index (((f * rowPairs + rowAB) *
        // primPairs) + b) * nHerm + t), so the device contracts the same
        // matrix the CPU lane contracts. The old b=0-replicated fold-only
        // table (exact only for the s-class, where the angular fold is 1)
        // is gone - the fused kernels read the per-slot slices directly,
        // and the V2 bra GEMM consumes the table verbatim with the
        // run-uniform k dimension.
        for (const double v : pair.braTransform)
        {
            tables.braTransforms.push_back(v);
        }

        for (const std::vector<double>& transform : pair.ketTransforms)
        {
            tables.ketTransforms.insert(
                tables.ketTransforms.end(), transform.begin(), transform.end());
        }

        // The certified-lane float copies: the kernels read the float
        // transforms directly - the conversion is covered by the certified
        // bound, exactly like the CPU lane's convert buffers. The full-4D
        // layout mirrors the fp64 one element for element.
        for (const double v : pair.braTransform)
        {
            tables.braTransformsF32.push_back(static_cast<float>(v));
        }

        for (const std::vector<double>& transform : pair.ketTransforms)
        {
            for (const double v : transform)
            {
                tables.ketTransformsF32.push_back(static_cast<float>(v));
            }
        }

        tables.pairMeta.push_back(meta);
    }

    return tables;
}

/// Partitions the batch's tasks into contiguous ranges whose slot sums stay
/// within chunkRows (a single oversized task keeps its own range - the
/// kernel splits it via the task-local chunk loop). A task's slot count is
/// rowPairs * primCount: the (rowAB, primB) slot space.
/// Pairs into \p ranges.
void PartitionTaskRanges(const std::vector<MdQuartetTask>& tasks,
                         const std::vector<MdPairData>& pairStore,
                         std::size_t chunkRows,
                         std::vector<int>& ranges) {
    ranges.clear();
    std::size_t rangeStart = 0;
    std::size_t slotSum = 0;

    for (std::size_t t = 0; t < tasks.size(); ++t)
    {
        const MdPairData& bra = pairStore[tasks[t].braPair];
        const std::size_t slots = bra.rowPairs * bra.primPairs.size();

        if (t > rangeStart && slotSum + slots > chunkRows)
        {
            ranges.push_back(static_cast<int>(rangeStart));
            ranges.push_back(static_cast<int>(t));
            rangeStart = t;
            slotSum = 0;
        }

        slotSum += slots;
    }

    ranges.push_back(static_cast<int>(rangeStart));
    ranges.push_back(static_cast<int>(tasks.size()));
}

/// The shared per-batch runner: resolves the variant (heuristic or
/// override), then dispatches to the fused kernels (kV0/kV1) or the V2
/// global-memory + cuBLAS pipeline. All pointers are device pointers; the
/// batch's output region starts at outF64/outF32 + outBase.
Result<void> RunBatchOnDevice(const EriCudaOptions& options,
                              const std::vector<MdPairData>& pairStore,
                              const EriCudaClassDispatchEntry* entry,
                              int lBra,
                              int lKet,
                              const EriCudaBoysTables* boysPtr,
                              const EriCudaPairMeta* pairMetaPtr,
                              // The host-resident pair meta: the host reads the
                              // GEMM offsets from it (pairMetaPtr is device-only
                              // memory - never dereferenced on the host).
                              const EriCudaPairMeta* hostPairMetaPtr,
                              const double* primDataPtr,
                              const double* braTransformsPtr,
                              const double* ketTransformsPtr,
                              const float* braTransformsF32Ptr,
                              const float* ketTransformsF32Ptr,
                              const MdClassBatch& batch,
                              double* outF64,
                              float* outF32,
                              double* errorBounds,
                              std::size_t outBase,
                              bool certified,
                              cudaStream_t stream,
                              cublasHandle_t cublas,
                              // The per-call transfer accounting: the
                              // task/ranges uploads of this batch
                              // are the screened-lists class of the residency
                              // contract.
                              EriCudaTransferAccounting& accounting) {
    const std::size_t nTasks = batch.tasks.size();

    if (nTasks == 0)
    {
        return {};
    }

    // The variant resolution. kV0/kV1 override within the fused family
    // (they share one instantiation - the thread count is runtime, see
    // EriCudaLaunchFusedClass); kV2 on a fused class or a fused override on
    // a kV2 class is kUnimplemented (the instantiation does not exist).
    int family = entry->family;
    int threads =
        family == kEriCudaFamilyV2
            ? 0
            : (family == kEriCudaFamilyV0 ? kEriCudaFusedThreadsV0 : kEriCudaFusedThreadsV1);

    if (options.variant != EriCudaVariant::kAuto)
    {
        const int requested = options.variant == EriCudaVariant::kV0   ? kEriCudaFamilyV0
                              : options.variant == EriCudaVariant::kV1 ? kEriCudaFamilyV1
                                                                       : kEriCudaFamilyV2;

        if (requested == kEriCudaFamilyV2)
        {
            if (family != kEriCudaFamilyV2)
            {
                return std::unexpected(
                    Unimplemented("kV2 is only instantiated for L >= 9 classes"));
            }
        } else if (family != kEriCudaFamilyV2)
        {
            // The fused families share the instantiation; the thread count
            // follows the override.
            family = requested;
            threads =
                requested == kEriCudaFamilyV0 ? kEriCudaFusedThreadsV0 : kEriCudaFusedThreadsV1;
        } else
        {
            return std::unexpected(
                Unimplemented("the fused variants are only instantiated for L <= 8 classes"));
        }
    }

    // The per-batch device task list.
    auto tasksBuffer = DeviceBuffer<EriCudaTask, CudaTag>::Create(nTasks);

    if (!tasksBuffer.has_value())
    {
        return std::unexpected(tasksBuffer.error());
    }

    std::vector<EriCudaTask>& tasksHost = tasksBuffer->HostView();

    for (std::size_t t = 0; t < nTasks; ++t)
    {
        tasksHost[t] = EriCudaTask{
            batch.tasks[t].braPair, batch.tasks[t].ketPair, batch.tasks[t].outputOffset};
    }

    auto sync = tasksBuffer->SyncToDevice();

    if (!sync.has_value())
    {
        return std::unexpected(sync.error());
    }

    accounting.listUploadBytes += tasksBuffer->Size() * sizeof(EriCudaTask);

    int maxRowPairs = 1;
    int maxNFuncsKet = 1;
    int maxNBraPrims = 1;

    for (const MdQuartetTask& task : batch.tasks)
    {
        const MdPairData& bra = pairStore[task.braPair];
        const MdPairData& ket = pairStore[task.ketPair];
        maxRowPairs = std::max(maxRowPairs, static_cast<int>(bra.rowPairs));
        maxNFuncsKet = std::max(maxNFuncsKet, static_cast<int>(ket.nFuncs));
        maxNBraPrims = std::max(maxNBraPrims, static_cast<int>(bra.primPairs.size()));
    }

    if (family == kEriCudaFamilyV2)
    {
        // -----------------------------------------------------------------
        // kV2: the VRR kernel writes global memory; the ket transform runs
        // as one strided-batched GEMM per (run, g), the bra transform as
        // one GEMM per task. Runs = maximal consecutive (ketPair, rowPairs,
        // primCount) subruns of the sorted batch (the CPU's ket-group
        // geometry: the GEMM shapes and the pq stride are uniform per run,
        // no padding). The primCount key is part of that layout -
        // the pq block is per-(rowAB, primB) slot, so its row count
        // rowPairs * vrrMaxNBraPrims * kHermBra must equal the bra
        // transform's own k dimension (rowPairs * primCount * kHermBra) for
        // the verbatim 4D table to contract exactly.
        // -----------------------------------------------------------------
        const int kHermBra = Hermite3DCount(lBra);
        const int kHermKet = Hermite3DCount(lKet);

        for (std::size_t runStart = 0; runStart < nTasks;)
        {
            std::size_t runEnd = runStart + 1;

            while (runEnd < nTasks &&
                   batch.tasks[runEnd].ketPair == batch.tasks[runStart].ketPair &&
                   pairStore[batch.tasks[runEnd].braPair].rowPairs ==
                       pairStore[batch.tasks[runStart].braPair].rowPairs &&
                   pairStore[batch.tasks[runEnd].braPair].primPairs.size() ==
                       pairStore[batch.tasks[runStart].braPair].primPairs.size())
            {
                ++runEnd;
            }

            const std::size_t runSize = runEnd - runStart;
            const MdPairData& ket = pairStore[batch.tasks[runStart].ketPair];
            const int rowPairs =
                static_cast<int>(pairStore[batch.tasks[runStart].braPair].rowPairs);
            const int vrrMaxNBraPrims =
                static_cast<int>(pairStore[batch.tasks[runStart].braPair].primPairs.size());
            // The per-task pq block: (rowAB, primB) slots of kHermBra Hermite
            // rows times kHermKet - the V2 VRR kernel's hole-free slot space
            // (thread (t, b) owns column b of every rowAB).
            const int mPq = rowPairs * vrrMaxNBraPrims * kHermBra;
            const int nFk = static_cast<int>(ket.nFuncs);

            EriCudaClassArgs args{};
            args.boys = boysPtr;
            args.pairMeta = pairMetaPtr;
            args.primData = primDataPtr;
            args.braTransforms = braTransformsPtr;
            args.ketTransforms = ketTransformsPtr;
            args.braTransformsF32 = braTransformsF32Ptr;
            args.ketTransformsF32 = ketTransformsF32Ptr;
            args.tasks =
                static_cast<const internal::cuda::EriCudaTask*>(tasksBuffer->DeviceHandle().Raw());
            args.boundsBase = batch.boundsBase;
            args.outF64 = outF64 + outBase;
            args.outF32 = outF32 != nullptr ? outF32 + outBase : nullptr;
            args.errorBounds = errorBounds;
            args.maxNFuncsKet = nFk;
            args.maxRowPairs = rowPairs;
            args.precision = certified ? kEriCudaPrecisionFp32 : kEriCudaPrecisionFp64;
            args.vrrG = 0;
            args.runStart = static_cast<int>(runStart);
            args.runSize = static_cast<int>(runSize);
            args.vrrMaxNBraPrims = vrrMaxNBraPrims;
            args.certifiedScale = kEriCudaCertifiedEpsilon *
                                  internal::kClassAmplification[lBra][lKet] *
                                  (1.0 + static_cast<double>(internal::kClassRounding[lBra][lKet]));
            args.stream = stream;

            // The per-run scratch: pq (mPq x kHermKet per task) zeroed per
            // g, acc (mPq x nFk per task) zeroed once at g == 0 (the GEMMs
            // accumulate with beta = 1), bound (one double per task) zeroed
            // once at g == 0 - every g's VRR atomicAdds into it, the full
            // g-summed CPU taskBound[t] semantics (md_vrr.hpp).
            const std::size_t pqWords = runSize * static_cast<std::size_t>(mPq) * kHermKet;
            const std::size_t accWords = runSize * static_cast<std::size_t>(mPq) * nFk;

            if (certified)
            {
                auto pqBuffer = DeviceBuffer<float, CudaTag>::Create(pqWords);

                if (!pqBuffer.has_value())
                {
                    return std::unexpected(pqBuffer.error());
                }

                auto accBuffer = DeviceBuffer<float, CudaTag>::Create(accWords);

                if (!accBuffer.has_value())
                {
                    return std::unexpected(accBuffer.error());
                }

                auto boundBuffer = DeviceBuffer<double, CudaTag>::Create(runSize);

                if (!boundBuffer.has_value())
                {
                    return std::unexpected(boundBuffer.error());
                }

                args.pq = pqBuffer->DeviceHandle().Raw();
                args.acc = accBuffer->DeviceHandle().Raw();
                args.bound = boundBuffer->DeviceHandle().Raw();

                for (int g = 0; g < static_cast<int>(ket.primPairs.size()); ++g)
                {
                    args.vrrG = g;

                    if (g == 0)
                    {
                        auto zeroAcc = CheckCudaError(
                            cudaMemsetAsync(args.acc, 0, accWords * sizeof(float), stream),
                            "cudaMemsetAsync acc");

                        if (!zeroAcc.has_value())
                        {
                            return std::unexpected(zeroAcc.error());
                        }

                        auto zeroBound = CheckCudaError(
                            cudaMemsetAsync(args.bound, 0, runSize * sizeof(double), stream),
                            "cudaMemsetAsync bound");

                        if (!zeroBound.has_value())
                        {
                            return std::unexpected(zeroBound.error());
                        }
                    }

                    auto zeroPq =
                        CheckCudaError(cudaMemsetAsync(args.pq, 0, pqWords * sizeof(float), stream),
                                       "cudaMemsetAsync pq");

                    if (!zeroPq.has_value())
                    {
                        return std::unexpected(zeroPq.error());
                    }

                    auto vrr = CheckLaunchCode(entry->launchVrr(args), "kV2 VRR kernel");

                    if (!vrr.has_value())
                    {
                        return std::unexpected(vrr.error());
                    }

                    const float one = 1.0f;
                    const float* ketG =
                        ketTransformsF32Ptr +
                        hostPairMetaPtr[batch.tasks[runStart].ketPair].ketTransformOffset +
                        static_cast<std::size_t>(g) * kHermKet * nFk;
                    auto gemm = CheckCublasStatus(
                        cublasSgemmStridedBatched(cublas,
                                                  CUBLAS_OP_N,
                                                  CUBLAS_OP_N,
                                                  nFk,
                                                  mPq,
                                                  kHermKet,
                                                  &one,
                                                  ketG,
                                                  nFk,
                                                  0,
                                                  static_cast<const float*>(args.pq),
                                                  kHermKet,
                                                  static_cast<std::ptrdiff_t>(mPq) * kHermKet,
                                                  &one,
                                                  static_cast<float*>(args.acc),
                                                  nFk,
                                                  static_cast<std::ptrdiff_t>(mPq) * nFk,
                                                  static_cast<int>(runSize)),
                        "cublasSgemmStridedBatched (ket)");

                    if (!gemm.has_value())
                    {
                        return std::unexpected(gemm.error());
                    }
                }

                for (std::size_t t = runStart; t < runEnd; ++t)
                {
                    const MdPairData& bra = pairStore[batch.tasks[t].braPair];
                    const EriCudaPairMeta& braMeta = hostPairMetaPtr[batch.tasks[t].braPair];
                    const int mBra = static_cast<int>(bra.nFuncs);
                    const int kBra = braMeta.braTransformK;
                    const float one = 1.0f;
                    const float zero = 0.0f;
                    const float* accT = static_cast<const float*>(args.acc) +
                                        (t - runStart) * static_cast<std::size_t>(mPq) * nFk;
                    const float* braT = braTransformsF32Ptr + braMeta.braTransformOffset;
                    float* outT = outF32 + outBase + batch.tasks[t].outputOffset;
                    auto gemm = CheckCublasStatus(cublasSgemm(cublas,
                                                              CUBLAS_OP_N,
                                                              CUBLAS_OP_N,
                                                              nFk,
                                                              mBra,
                                                              kBra,
                                                              &one,
                                                              accT,
                                                              nFk,
                                                              braT,
                                                              kBra,
                                                              &zero,
                                                              outT,
                                                              nFk),
                                                  "cublasSgemm (bra)");

                    if (!gemm.has_value())
                    {
                        return std::unexpected(gemm.error());
                    }
                }

                args.bound = boundBuffer->DeviceHandle().Raw();
                auto boundResult = CheckLaunchCode(entry->launchBound(args), "kV2 bound finalize");

                if (!boundResult.has_value())
                {
                    return std::unexpected(boundResult.error());
                }
            } else
            {
                auto pqBuffer = DeviceBuffer<double, CudaTag>::Create(pqWords);

                if (!pqBuffer.has_value())
                {
                    return std::unexpected(pqBuffer.error());
                }

                auto accBuffer = DeviceBuffer<double, CudaTag>::Create(accWords);

                if (!accBuffer.has_value())
                {
                    return std::unexpected(accBuffer.error());
                }

                args.pq = pqBuffer->DeviceHandle().Raw();
                args.acc = accBuffer->DeviceHandle().Raw();

                for (int g = 0; g < static_cast<int>(ket.primPairs.size()); ++g)
                {
                    args.vrrG = g;

                    if (g == 0)
                    {
                        auto zeroAcc = CheckCudaError(
                            cudaMemsetAsync(args.acc, 0, accWords * sizeof(double), stream),
                            "cudaMemsetAsync acc");

                        if (!zeroAcc.has_value())
                        {
                            return std::unexpected(zeroAcc.error());
                        }
                    }

                    auto zeroPq = CheckCudaError(
                        cudaMemsetAsync(args.pq, 0, pqWords * sizeof(double), stream),
                        "cudaMemsetAsync pq");

                    if (!zeroPq.has_value())
                    {
                        return std::unexpected(zeroPq.error());
                    }

                    auto vrr = CheckLaunchCode(entry->launchVrr(args), "kV2 VRR kernel");

                    if (!vrr.has_value())
                    {
                        return std::unexpected(vrr.error());
                    }

                    const double one = 1.0;
                    const double* ketG =
                        ketTransformsPtr +
                        hostPairMetaPtr[batch.tasks[runStart].ketPair].ketTransformOffset +
                        static_cast<std::size_t>(g) * kHermKet * nFk;
                    auto gemm = CheckCublasStatus(
                        cublasDgemmStridedBatched(cublas,
                                                  CUBLAS_OP_N,
                                                  CUBLAS_OP_N,
                                                  nFk,
                                                  mPq,
                                                  kHermKet,
                                                  &one,
                                                  ketG,
                                                  nFk,
                                                  0,
                                                  static_cast<const double*>(args.pq),
                                                  kHermKet,
                                                  static_cast<std::ptrdiff_t>(mPq) * kHermKet,
                                                  &one,
                                                  static_cast<double*>(args.acc),
                                                  nFk,
                                                  static_cast<std::ptrdiff_t>(mPq) * nFk,
                                                  static_cast<int>(runSize)),
                        "cublasDgemmStridedBatched (ket)");

                    if (!gemm.has_value())
                    {
                        return std::unexpected(gemm.error());
                    }
                }

                for (std::size_t t = runStart; t < runEnd; ++t)
                {
                    const MdPairData& bra = pairStore[batch.tasks[t].braPair];
                    const EriCudaPairMeta& braMeta = hostPairMetaPtr[batch.tasks[t].braPair];
                    const int mBra = static_cast<int>(bra.nFuncs);
                    const int kBra = braMeta.braTransformK;
                    const double one = 1.0;
                    const double zero = 0.0;
                    const double* accT = static_cast<const double*>(args.acc) +
                                         (t - runStart) * static_cast<std::size_t>(mPq) * nFk;
                    const double* braT = braTransformsPtr + braMeta.braTransformOffset;
                    double* outT = outF64 + outBase + batch.tasks[t].outputOffset;
                    auto gemm = CheckCublasStatus(cublasDgemm(cublas,
                                                              CUBLAS_OP_N,
                                                              CUBLAS_OP_N,
                                                              nFk,
                                                              mBra,
                                                              kBra,
                                                              &one,
                                                              accT,
                                                              nFk,
                                                              braT,
                                                              kBra,
                                                              &zero,
                                                              outT,
                                                              nFk),
                                                  "cublasDgemm (bra)");

                    if (!gemm.has_value())
                    {
                        return std::unexpected(gemm.error());
                    }
                }
            }

            runStart = runEnd;
        }

        return {};
    }

    // -----------------------------------------------------------------
    // kV0/kV1: the fused kernel - one block per task-range chunk; the
    // threads map to the (rowAB, primB) slots of the current task's
    // flattened slot space; the VRR, the ket GEMM (MicroGemmAdd order),
    // and the bra transform (MicroGemm order, chunk-accumulated) all run
    // inside the kernel.
    // -----------------------------------------------------------------
    const int chunkRows =
        ComputeChunkRows(lBra, lKet, maxRowPairs, maxNBraPrims, maxNFuncsKet, threads, certified);

    if (chunkRows < 1)
    {
        return std::unexpected(
            DeviceError("the fused class does not fit the shared-memory budget"));
    }

    std::vector<int> rangePairs;
    PartitionTaskRanges(batch.tasks, pairStore, static_cast<std::size_t>(chunkRows), rangePairs);
    const int nChunks = static_cast<int>(rangePairs.size() / 2);

    auto rangesBuffer = DeviceBuffer<int, CudaTag>::Create(rangePairs.size());

    if (!rangesBuffer.has_value())
    {
        return std::unexpected(rangesBuffer.error());
    }

    std::vector<int>& rangesHost = rangesBuffer->HostView();

    for (std::size_t i = 0; i < rangePairs.size(); ++i)
    {
        rangesHost[i] = rangePairs[i];
    }

    auto rangesSync = rangesBuffer->SyncToDevice();

    if (!rangesSync.has_value())
    {
        return std::unexpected(rangesSync.error());
    }

    accounting.listUploadBytes += rangesBuffer->Size() * sizeof(int);

    const double certifiedScale = kEriCudaCertifiedEpsilon *
                                  internal::kClassAmplification[lBra][lKet] *
                                  (1.0 + static_cast<double>(internal::kClassRounding[lBra][lKet]));

    EriCudaClassArgs args{};
    args.boys = boysPtr;
    args.pairMeta = pairMetaPtr;
    args.primData = primDataPtr;
    args.braTransforms = braTransformsPtr;
    args.ketTransforms = ketTransformsPtr;
    args.braTransformsF32 = braTransformsF32Ptr;
    args.ketTransformsF32 = ketTransformsF32Ptr;
    args.tasks = static_cast<const internal::cuda::EriCudaTask*>(tasksBuffer->DeviceHandle().Raw());
    args.boundsBase = batch.boundsBase;
    args.outF64 = outF64 + outBase;
    args.outF32 = outF32 != nullptr ? outF32 + outBase : nullptr;
    args.errorBounds = errorBounds;
    args.maxNFuncsKet = maxNFuncsKet;
    args.maxRowPairs = maxRowPairs;
    args.certifiedScale = certifiedScale;
    args.chunkRows = chunkRows;
    args.threads = threads;
    args.precision = certified ? kEriCudaPrecisionFp32 : kEriCudaPrecisionFp64;
    args.chunkRanges = static_cast<const int*>(rangesBuffer->DeviceHandle().Raw());
    args.nChunks = nChunks;
    args.stream = stream;

    return CheckLaunchCode(entry->launchFused(args), "fused kernel");
}

/// The device-side statics of the engine: the pair
/// tables and the Boys copy, uploaded once and retained on the FastPath,
/// uploaded per call (EnsureStatics) and released at the call end
/// (ReleaseStatics) on the LightPath. The per-call paths read the device
/// pointers through the active copy, never through a retained member - the
/// mode switch is invisible to them beyond the EnsureStatics call at the
/// call entry.
struct DeviceStatics {
    DeviceBuffer<EriCudaBoysTables, CudaTag> boys;
    DeviceBuffer<EriCudaPairMeta, CudaTag> pairMeta;
    DeviceBuffer<double, CudaTag> primData;
    DeviceBuffer<double, CudaTag> braTransforms;
    DeviceBuffer<double, CudaTag> ketTransforms;
    DeviceBuffer<float, CudaTag> braTransformsF32;
    DeviceBuffer<float, CudaTag> ketTransformsF32;

    /// Creates the seven buffers, fills them from the host tables and
    /// stages them to the device (the Create upload block). Every buffer
    /// starts host-resident; the host copies are filled and SyncToDevice
    /// stages them.
    /// \param hostTables The host tables (the canonical BuildHostTables
    /// output; the caller keeps it alive).
    /// \param boysHost The host Boys tables.
    /// \returns The statics, or an Error from any buffer create/sync step.
    static Result<DeviceStatics> Upload(const HostTables& hostTables,
                                        const EriCudaBoysTables& boysHost) {
        auto boysBuffer = DeviceBuffer<EriCudaBoysTables, CudaTag>::Create(1);

        if (!boysBuffer.has_value())
        {
            return std::unexpected(boysBuffer.error());
        }

        boysBuffer->HostView()[0] = boysHost;

        auto boysSync = boysBuffer->SyncToDevice();

        if (!boysSync.has_value())
        {
            return std::unexpected(boysSync.error());
        }

        auto pairMetaBuffer =
            DeviceBuffer<EriCudaPairMeta, CudaTag>::Create(hostTables.pairMeta.size());

        if (!pairMetaBuffer.has_value())
        {
            return std::unexpected(pairMetaBuffer.error());
        }

        pairMetaBuffer->HostView() = hostTables.pairMeta;

        auto pairMetaSync = pairMetaBuffer->SyncToDevice();

        if (!pairMetaSync.has_value())
        {
            return std::unexpected(pairMetaSync.error());
        }

        auto primDataBuffer = DeviceBuffer<double, CudaTag>::Create(hostTables.primData.size());

        if (!primDataBuffer.has_value())
        {
            return std::unexpected(primDataBuffer.error());
        }

        primDataBuffer->HostView() = hostTables.primData;

        auto primDataSync = primDataBuffer->SyncToDevice();

        if (!primDataSync.has_value())
        {
            return std::unexpected(primDataSync.error());
        }

        auto braTransformsBuffer =
            DeviceBuffer<double, CudaTag>::Create(hostTables.braTransforms.size());

        if (!braTransformsBuffer.has_value())
        {
            return std::unexpected(braTransformsBuffer.error());
        }

        braTransformsBuffer->HostView() = hostTables.braTransforms;

        auto braTransformsSync = braTransformsBuffer->SyncToDevice();

        if (!braTransformsSync.has_value())
        {
            return std::unexpected(braTransformsSync.error());
        }

        auto ketTransformsBuffer =
            DeviceBuffer<double, CudaTag>::Create(hostTables.ketTransforms.size());

        if (!ketTransformsBuffer.has_value())
        {
            return std::unexpected(ketTransformsBuffer.error());
        }

        ketTransformsBuffer->HostView() = hostTables.ketTransforms;

        auto ketTransformsSync = ketTransformsBuffer->SyncToDevice();

        if (!ketTransformsSync.has_value())
        {
            return std::unexpected(ketTransformsSync.error());
        }

        auto braTransformsF32Buffer =
            DeviceBuffer<float, CudaTag>::Create(hostTables.braTransformsF32.size());

        if (!braTransformsF32Buffer.has_value())
        {
            return std::unexpected(braTransformsF32Buffer.error());
        }

        braTransformsF32Buffer->HostView() = hostTables.braTransformsF32;

        auto braTransformsF32Sync = braTransformsF32Buffer->SyncToDevice();

        if (!braTransformsF32Sync.has_value())
        {
            return std::unexpected(braTransformsF32Sync.error());
        }

        auto ketTransformsF32Buffer =
            DeviceBuffer<float, CudaTag>::Create(hostTables.ketTransformsF32.size());

        if (!ketTransformsF32Buffer.has_value())
        {
            return std::unexpected(ketTransformsF32Buffer.error());
        }

        ketTransformsF32Buffer->HostView() = hostTables.ketTransformsF32;

        auto ketTransformsF32Sync = ketTransformsF32Buffer->SyncToDevice();

        if (!ketTransformsF32Sync.has_value())
        {
            return std::unexpected(ketTransformsF32Sync.error());
        }

        // The aggregate initialization: every buffer was created, filled
        // and staged above (each step checked), so the aggregate can be
        // built with the moved buffers in member order.
        return DeviceStatics{std::move(*boysBuffer),
                             std::move(*pairMetaBuffer),
                             std::move(*primDataBuffer),
                             std::move(*braTransformsBuffer),
                             std::move(*ketTransformsBuffer),
                             std::move(*braTransformsF32Buffer),
                             std::move(*ketTransformsF32Buffer)};
    }
};

} // namespace

// ---------------------------------------------------------------------------
// The engine implementation
// ---------------------------------------------------------------------------

struct EriCudaEngine::Impl {
    EriCudaOptions options;
    int deviceOrdinal = 0;
    cublasHandle_t cublas = nullptr;
    /// The pair-store tables in host memory (the 2e path runs the engine's
    /// device copies; the RI path re-uploads a combined store per call and
    /// reuses the orbital half of these cached copies).
    std::vector<MdPairData> pairStore;
    ShellPairList pairList;
    HostTables hostTables;
    EriCudaBoysTables boysHost{};
    /// The uploaded device statics. On the FastPath this
    /// copy is uploaded once at Create and retained; on the LightPath it
    /// stays empty and the per-call paths upload and release the statics
    /// through activeStatics (the per-call pointer block in every path reads
    /// the active copy - the mode switch is invisible to them beyond the
    /// EnsureStatics call at the call entry).
    std::optional<DeviceStatics> statics;
    /// The per-call statics copy of the LightPath: set by EnsureStatics at
    /// the call entry, cleared by ReleaseStatics at the call end. Never set
    /// on the FastPath (the retained statics are the active copy there).
    std::optional<DeviceStatics> activeStatics;
    /// The molecule the RI path needs for the auxiliary pair list (the
    /// caller keeps it alive - documented on EriCudaEngine::Create).
    const qcx::molecule::Molecule* molecule = nullptr;
    /// The Create-time device footprint estimate (design section 5), filled
    /// once at Create and exposed through EriCudaEngine::DeviceFootprint -
    /// the byte terms the device workspace budget is charged against.
    GpuDeviceFootprint deviceFootprint;
    /// The Create-time mode decision: the AM-B
    /// FastPath/LightPath decision generalized to the device context,
    /// recorded once at Create.
    FockBuildMode mode = FockBuildMode::kFastPath;
    /// The Create-time mode record: the decision plus
    /// the byte terms it ran on. Purely observational (EriCudaModeInfo).
    EriCudaModeInfo modeInfo;

    // The per-call transfer accounting:
    // the Fock-path traffic sums of the last BuildFock call. Reset at every
    // BuildFock entry; every Fock-path transfer site increments the class it
    // belongs to (the BuildFock debug assert verifies the accounted math).
    EriCudaTransferAccounting transferAccounting;

    // The device-timeline span of the last BuildFock call's device passes
    // (the corrected-method instrument):
    // cudaEventElapsedTime between the pass-bracket events below, summed
    // over the call's passes (RunDevicePass records once per pass). Reset
    // at every BuildFock entry; the launches of the Fock path enqueue on
    // the default stream, which the events are recorded on. Purely
    // observational, like the transfer accounting.
    double fockDeviceSpanMs = 0.0;
    /// The pass-bracket events (the same record as fockDeviceSpanMs): one
    /// pair per engine, recorded on the default stream around each device
    /// pass of the Fock path. Null when the creation failed at engine
    /// Create (the instrument then silently stays off).
    cudaEvent_t passStartEvent = nullptr;
    cudaEvent_t passEndEvent = nullptr;

    Impl(const EriCudaOptions& optionsIn,
         int deviceOrdinalIn,
         cublasHandle_t cublasIn,
         const qcx::molecule::Molecule& moleculeIn,
         std::vector<MdPairData> pairStoreIn,
         ShellPairList pairListIn,
         HostTables hostTablesIn,
         EriCudaBoysTables boysHostIn,
         std::optional<DeviceStatics> staticsIn,
         GpuDeviceFootprint deviceFootprintIn,
         FockBuildMode modeIn,
         EriCudaModeInfo modeInfoIn) :
        options(optionsIn), deviceOrdinal(deviceOrdinalIn), cublas(cublasIn),
        pairStore(std::move(pairStoreIn)), pairList(std::move(pairListIn)),
        hostTables(std::move(hostTablesIn)), boysHost(boysHostIn), statics(std::move(staticsIn)),
        molecule(&moleculeIn), deviceFootprint(deviceFootprintIn), mode(modeIn),
        modeInfo(modeInfoIn) {
        // The pass-bracket events (best-effort: a creation
        // failure leaves the event null and the span instrument silently
        // off - the engine Create probes just passed on this device, so a
        // failure is not expected in practice).
        if (cudaEventCreate(&passStartEvent) != cudaSuccess)
        {
            passStartEvent = nullptr;
        }

        if (cudaEventCreate(&passEndEvent) != cudaSuccess)
        {
            passEndEvent = nullptr;
        }
    }

    /// Ensures the device statics are uploaded and returns the active copy
    /// (the retained copy on the FastPath, a per-call copy on the
    /// LightPath). Call at the entry of every per-call path that reads the
    /// statics pointers.
    /// \returns The active copy, or an Error from the upload.
    Result<DeviceStatics*> EnsureStatics();

    /// Releases the per-call statics copy (LightPath): the buffers are
    /// destroyed and their mapping returned to the driver, so the light
    /// rung's device residency outside a call is exactly the per-call
    /// working set. No-op on the FastPath (the retained copy stays).
    void ReleaseStatics() noexcept;

    /// An RAII guard that releases the per-call statics at the end of a
    /// per-call path: the release runs at every exit of
    /// the enclosing call, early returns included - the light rung never
    /// leaks device residency. No-op on the FastPath.
    class StaticsGuard {
    public:
        explicit StaticsGuard(Impl& impl) : _impl(&impl) {}

        StaticsGuard(const StaticsGuard&) = delete;
        StaticsGuard& operator=(const StaticsGuard&) = delete;

        ~StaticsGuard() {
            _impl->ReleaseStatics();
        }

    private:
        Impl* _impl;
    };

    ~Impl() {
        if (cublas != nullptr)
        {
            cublasDestroy(cublas);
        }

        if (passStartEvent != nullptr)
        {
            cudaEventDestroy(passStartEvent);
        }

        if (passEndEvent != nullptr)
        {
            cudaEventDestroy(passEndEvent);
        }
    }
};

EriCudaEngine::EriCudaEngine(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}

EriCudaEngine::~EriCudaEngine() = default;

Result<EriCudaEngine> EriCudaEngine::Create(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet,
                                            const EriCudaOptions& options) {
    if (options.maxBatchBytes == 0)
    {
        return std::unexpected(InvalidArgument("maxBatchBytes must be positive"));
    }

    int deviceCount = 0;
    auto countError = CheckCudaError(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");

    if (!countError.has_value())
    {
        return std::unexpected(countError.error());
    }

    if (deviceCount == 0)
    {
        return std::unexpected(DeviceError("no CUDA device available"));
    }

    int deviceOrdinal = 0;
    auto ordinalError = CheckCudaError(cudaGetDevice(&deviceOrdinal), "cudaGetDevice");

    if (!ordinalError.has_value())
    {
        return std::unexpected(ordinalError.error());
    }

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
                Unimplemented("a shell exceeds this build's angular-momentum cap"));
        }
    }

    if (pairList->pairs.empty())
    {
        return std::unexpected(InvalidArgument("the basis has no shell pairs"));
    }

    auto pairStore = internal::BuildPairData(molecule, basisSet, *pairList);

    if (!pairStore.has_value())
    {
        return std::unexpected(pairStore.error());
    }

    auto hostTables = BuildHostTables(*pairStore);

    if (!hostTables.has_value())
    {
        return std::unexpected(hostTables.error());
    }

    EriCudaBoysTables boysHost{};
    auto boysTables = BuildBoysTables(boysHost);

    if (!boysTables.has_value())
    {
        return std::unexpected(boysTables.error());
    }

    // The Create-time device footprint estimate: the exact uploaded-table
    // bytes of the host tables just
    // built, the fixed Boys copy, the per-call working set and the
    // structural allowance. Computed here - before the first device
    // allocation - so the device-budget refusal (step 3) can fire with the
    // estimate in hand and no device memory committed.
    const std::size_t tablesBytes =
        hostTables->pairMeta.size() * sizeof(EriCudaPairMeta) +
        hostTables->primData.size() * sizeof(double) +
        (hostTables->braTransforms.size() + hostTables->ketTransforms.size()) * sizeof(double) +
        (hostTables->braTransformsF32.size() + hostTables->ketTransformsF32.size()) * sizeof(float);

    // The kV2 class mapping (the per-class variant heuristic of
    // RunBatchOnDevice): the global-memory scratch and the cuBLAS workspace
    // engage only when some class resolves to kV2 - under kAuto the class
    // L = lBra + lKet reaches 4 * lMax, kV2 for L >= 9, or under the kV2
    // override outright (a fused override on a kV2 class is kUnimplemented
    // at dispatch; the footprint keeps the conservative fused mapping).
    int lMax = 0;

    for (const ShellInfo& shell : pairList->shells)
    {
        lMax = std::max(lMax, shell.angularMomentum);
    }

    const bool hasKv2Classes = options.variant == EriCudaVariant::kV2 ||
                               (options.variant == EriCudaVariant::kAuto && 4 * lMax >= 9);
    const GpuDeviceFootprint deviceFootprint =
        internal::ComputeGpuDeviceFootprint(pairList->functionCount,
                                            options.maxBatchBytes,
                                            tablesBytes,
                                            sizeof(EriCudaBoysTables),
                                            hasKv2Classes);

    // The Create-time mode decision:
    // the AM-B FastPath/LightPath decision generalized to the device
    // context. The fast rung retains the statics from Create (the legacy
    // behavior); the light rung uploads them per call (EnsureStatics) and
    // releases them at the call end (ReleaseStatics). The decision is
    // budget-driven when a device workspace budget was given, and the
    // forced-LightPath knob always wins (the mode-forcing test surface, the
    // same role as lightPathChunkPairs on the CPU builder).
    const bool lightPath = options.forceLightPath ||
                           (options.deviceWorkspaceBudget != nullptr &&
                            deviceFootprint.Total() > options.deviceWorkspaceBudget->Remaining());
    const FockBuildMode mode = lightPath ? FockBuildMode::kLightPath : FockBuildMode::kFastPath;

    EriCudaModeInfo modeInfo;
    modeInfo.mode = mode;
    modeInfo.fastEstimateBytes = deviceFootprint.Total();
    modeInfo.lightEstimateBytes = deviceFootprint.matricesBytes +
                                  deviceFootprint.batchScratchBytes +
                                  deviceFootprint.perCallOutputBytes;
    modeInfo.remainingAtDecision =
        options.deviceWorkspaceBudget != nullptr ? options.deviceWorkspaceBudget->Remaining() : 0;

    // The device-budget refusal: when
    // the light rung - the last device rung - cannot fit the budget either,
    // the ladder fires HERE, before the first device allocation of this
    // Create (the statics upload and cublasCreate below), so no CUDA
    // allocation can fail for a budget the caller declared. The fast rung
    // fits by construction whenever the budget-driven decision kept it
    // (Total <= Remaining made the decision), so only the light rung can
    // refuse; the forced-LightPath knob refuses the same way - a forced
    // light rung under a budget that cannot hold it is a caller
    // contradiction, named and refused at Create, never deferred to the
    // per-call uploads.
    if (options.deviceWorkspaceBudget != nullptr && lightPath)
    {
        const std::size_t remaining = options.deviceWorkspaceBudget->Remaining();

        if (modeInfo.lightEstimateBytes > remaining)
        {
            return std::unexpected(DeviceBudgetRefusal(deviceOrdinal,
                                                       options.deviceWorkspaceBudget->DeviceId(),
                                                       deviceFootprint.Total(),
                                                       modeInfo.lightEstimateBytes,
                                                       remaining));
        }
    }

    // The device statics upload. The FastPath uploads the statics once at
    // Create and retains them; the LightPath defers the upload to the
    // per-call EnsureStatics and leaves the retained copy empty - the
    // per-call working set (matrices, batch scratch, per-call output) is
    // the only device residency the light rung commits outside a call.
    std::optional<DeviceStatics> statics;

    if (mode == FockBuildMode::kFastPath)
    {
        auto uploaded = DeviceStatics::Upload(*hostTables, boysHost);

        if (!uploaded.has_value())
        {
            return std::unexpected(uploaded.error());
        }

        statics = std::move(*uploaded);
    }

    cublasHandle_t cublas = nullptr;
    auto createHandle = CheckCublasStatus(cublasCreate(&cublas), "cublasCreate");

    if (!createHandle.has_value())
    {
        return std::unexpected(createHandle.error());
    }

    auto impl = std::make_unique<Impl>(options,
                                       deviceOrdinal,
                                       cublas,
                                       molecule,
                                       std::move(*pairStore),
                                       std::move(*pairList),
                                       std::move(*hostTables),
                                       boysHost,
                                       std::move(statics),
                                       deviceFootprint,
                                       mode,
                                       modeInfo);
    return EriCudaEngine{std::move(impl)};
}

GpuDeviceFootprint EriCudaEngine::DeviceFootprint() const {
    return _impl->deviceFootprint;
}

FockBuildMode EriCudaEngine::Mode() const noexcept {
    return _impl->mode;
}

EriCudaModeInfo EriCudaEngine::ModeInfo() const noexcept {
    return _impl->modeInfo;
}

EriCudaTransferAccounting EriCudaEngine::TransferAccounting() const noexcept {
    return _impl->transferAccounting;
}

Result<DeviceStatics*> EriCudaEngine::Impl::EnsureStatics() {
    // The FastPath keeps the retained copy; the LightPath uploads the
    // statics at the call entry and releases them at the call end - the
    // per-call pointer block of every path reads the active copy.
    if (statics.has_value())
    {
        return &*statics;
    }

    if (activeStatics.has_value())
    {
        return &*activeStatics;
    }

    auto uploaded = DeviceStatics::Upload(hostTables, boysHost);

    if (!uploaded.has_value())
    {
        return std::unexpected(uploaded.error());
    }

    activeStatics = std::move(*uploaded);

    // The per-call statics upload: the LightPath's upload
    // is the statics class of the accounting; the FastPath retained the
    // statics at Create (0 per call).
    transferAccounting.staticsUploadBytes +=
        deviceFootprint.tablesBytes + deviceFootprint.boysBytes;

    return &*activeStatics;
}

void EriCudaEngine::Impl::ReleaseStatics() noexcept {
    // Releases the per-call copy. Releasing sets the optional to nullopt,
    // which destroys the buffers at the call end - the mapping is returned
    // to the driver, so the light rung's device residency outside a call is
    // exactly the per-call working set. No-op on the FastPath.
    if (activeStatics.has_value())
    {
        activeStatics.reset();
    }
}

Result<EriCudaBatch> EriCudaEngine::ComputeBatch(const std::vector<ShellQuartet>& quartets,
                                                 void* stream) {
    if (quartets.empty())
    {
        return std::unexpected(InvalidArgument("the quartet list is empty"));
    }

    auto device = CheckCudaError(cudaSetDevice(_impl->deviceOrdinal), "cudaSetDevice");

    if (!device.has_value())
    {
        return std::unexpected(device.error());
    }

    auto streamSet = CheckCublasStatus(
        cublasSetStream(_impl->cublas, static_cast<cudaStream_t>(stream)), "cublasSetStream");

    if (!streamSet.has_value())
    {
        return std::unexpected(streamSet.error());
    }

    std::vector<ShellQuartet> computed;
    auto batches = internal::AssembleClassBatches(
        _impl->pairStore, _impl->pairList, quartets, _impl->options.maxBatchBytes, computed);

    if (!batches.has_value())
    {
        return std::unexpected(batches.error());
    }

    std::size_t totalWords = 0;

    for (const MdClassBatch& batch : *batches)
    {
        for (const MdQuartetTask& task : batch.tasks)
        {
            totalWords +=
                _impl->pairStore[task.braPair].nFuncs * _impl->pairStore[task.ketPair].nFuncs;
        }
    }

    if (totalWords == 0)
    {
        return EriCudaBatch{std::vector<double>{}, std::move(computed)};
    }

    auto outBuffer = DeviceBuffer<double, CudaTag>::Create(totalWords);

    if (!outBuffer.has_value())
    {
        return std::unexpected(outBuffer.error());
    }

    // The device statics: the FastPath reads the
    // retained copy; the LightPath uploads the statics at the call entry
    // and releases them at the call end - the pointer block reads the
    // active copy either way, and the guard runs the release at every
    // exit, early returns included.
    auto statics = _impl->EnsureStatics();

    if (!statics.has_value())
    {
        return std::unexpected(statics.error());
    }

    Impl::StaticsGuard staticsGuard(*_impl);

    const auto* boysPtr =
        static_cast<const EriCudaBoysTables*>((*statics)->boys.DeviceHandle().Raw());
    const auto* pairMetaPtr =
        static_cast<const EriCudaPairMeta*>((*statics)->pairMeta.DeviceHandle().Raw());
    const auto* primDataPtr = static_cast<const double*>((*statics)->primData.DeviceHandle().Raw());
    const auto* braTransformsPtr =
        static_cast<const double*>((*statics)->braTransforms.DeviceHandle().Raw());
    const auto* ketTransformsPtr =
        static_cast<const double*>((*statics)->ketTransforms.DeviceHandle().Raw());
    const auto* braTransformsF32Ptr =
        static_cast<const float*>((*statics)->braTransformsF32.DeviceHandle().Raw());
    const auto* ketTransformsF32Ptr =
        static_cast<const float*>((*statics)->ketTransformsF32.DeviceHandle().Raw());
    double* outF64 = static_cast<double*>(outBuffer->DeviceHandle().Raw());
    std::size_t outBase = 0;

    for (const MdClassBatch& batch : *batches)
    {
        const int lBra = batch.lBra;
        const int lKet = batch.lKet;
        const EriCudaClassDispatchEntry* entry = QcxEriCudaFindClass(lBra, lKet);

        if (entry == nullptr)
        {
            return std::unexpected(Unimplemented("class (" + std::to_string(lBra) + "," +
                                                 std::to_string(lKet) +
                                                 ") is outside the instantiated CUDA matrix"));
        }

        auto result = RunBatchOnDevice(_impl->options,
                                       _impl->pairStore,
                                       entry,
                                       lBra,
                                       lKet,
                                       boysPtr,
                                       pairMetaPtr,
                                       _impl->hostTables.pairMeta.data(),
                                       primDataPtr,
                                       braTransformsPtr,
                                       ketTransformsPtr,
                                       braTransformsF32Ptr,
                                       ketTransformsF32Ptr,
                                       batch,
                                       outF64,
                                       nullptr,
                                       nullptr,
                                       outBase,
                                       false,
                                       static_cast<cudaStream_t>(stream),
                                       _impl->cublas,
                                       _impl->transferAccounting);

        if (!result.has_value())
        {
            return std::unexpected(result.error());
        }

        for (const MdQuartetTask& task : batch.tasks)
        {
            outBase +=
                _impl->pairStore[task.braPair].nFuncs * _impl->pairStore[task.ketPair].nFuncs;
        }
    }

    auto sync = CheckCudaError(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)),
                               "cudaStreamSynchronize");

    if (!sync.has_value())
    {
        return std::unexpected(sync.error());
    }

    // The kernels and the cuBLAS GEMMs wrote the device side directly
    // through the raw handle: declare the device copy canonical before the
    // read-back (DeviceBuffer's residency machine, otherwise SyncToHost is
    // a no-op and the host copy stays the fresh zeros).
    outBuffer->MarkDeviceDirty();

    auto copy = outBuffer->SyncToHost();

    if (!copy.has_value())
    {
        return std::unexpected(copy.error());
    }

    return EriCudaBatch{std::move(outBuffer->HostView()), std::move(computed)};
}

Result<EriCudaCertifiedBatch> EriCudaEngine::ComputeBatchCertified(
    const std::vector<ShellQuartet>& quartets, void* stream) {
#if !QcxIntegralsF32
    return std::unexpected(
        Unimplemented("the certified fp32 pipeline is not instantiated (QCX_INTEGRALS_F32)"));
#else
    if (quartets.empty())
    {
        return std::unexpected(InvalidArgument("the quartet list is empty"));
    }

    auto device = CheckCudaError(cudaSetDevice(_impl->deviceOrdinal), "cudaSetDevice");

    if (!device.has_value())
    {
        return std::unexpected(device.error());
    }

    auto streamSet = CheckCublasStatus(
        cublasSetStream(_impl->cublas, static_cast<cudaStream_t>(stream)), "cublasSetStream");

    if (!streamSet.has_value())
    {
        return std::unexpected(streamSet.error());
    }

    std::vector<ShellQuartet> computed;
    auto batches = internal::AssembleClassBatches(
        _impl->pairStore, _impl->pairList, quartets, _impl->options.maxBatchBytes, computed);

    if (!batches.has_value())
    {
        return std::unexpected(batches.error());
    }

    std::size_t totalWords = 0;

    for (const MdClassBatch& batch : *batches)
    {
        for (const MdQuartetTask& task : batch.tasks)
        {
            totalWords +=
                _impl->pairStore[task.braPair].nFuncs * _impl->pairStore[task.ketPair].nFuncs;
        }
    }

    if (totalWords == 0)
    {
        return EriCudaCertifiedBatch{
            std::vector<float>{}, std::vector<double>{}, std::move(computed)};
    }

    auto outBuffer = DeviceBuffer<float, CudaTag>::Create(totalWords);

    if (!outBuffer.has_value())
    {
        return std::unexpected(outBuffer.error());
    }

    auto boundsBuffer = DeviceBuffer<double, CudaTag>::Create(computed.size());

    if (!boundsBuffer.has_value())
    {
        return std::unexpected(boundsBuffer.error());
    }

    // The device statics: the FastPath reads the
    // retained copy; the LightPath uploads the statics at the call entry
    // and releases them at the call end - the pointer block reads the
    // active copy either way, and the guard runs the release at every
    // exit, early returns included.
    auto statics = _impl->EnsureStatics();

    if (!statics.has_value())
    {
        return std::unexpected(statics.error());
    }

    Impl::StaticsGuard staticsGuard(*_impl);

    const auto* boysPtr =
        static_cast<const EriCudaBoysTables*>((*statics)->boys.DeviceHandle().Raw());
    const auto* pairMetaPtr =
        static_cast<const EriCudaPairMeta*>((*statics)->pairMeta.DeviceHandle().Raw());
    const auto* primDataPtr = static_cast<const double*>((*statics)->primData.DeviceHandle().Raw());
    const auto* braTransformsPtr =
        static_cast<const double*>((*statics)->braTransforms.DeviceHandle().Raw());
    const auto* ketTransformsPtr =
        static_cast<const double*>((*statics)->ketTransforms.DeviceHandle().Raw());
    const auto* braTransformsF32Ptr =
        static_cast<const float*>((*statics)->braTransformsF32.DeviceHandle().Raw());
    const auto* ketTransformsF32Ptr =
        static_cast<const float*>((*statics)->ketTransformsF32.DeviceHandle().Raw());
    float* outF32 = static_cast<float*>(outBuffer->DeviceHandle().Raw());
    double* errorBounds = static_cast<double*>(boundsBuffer->DeviceHandle().Raw());
    std::size_t outBase = 0;

    for (const MdClassBatch& batch : *batches)
    {
        const int lBra = batch.lBra;
        const int lKet = batch.lKet;
        const EriCudaClassDispatchEntry* entry = QcxEriCudaFindClass(lBra, lKet);

        if (entry == nullptr)
        {
            return std::unexpected(Unimplemented("class (" + std::to_string(lBra) + "," +
                                                 std::to_string(lKet) +
                                                 ") is outside the instantiated CUDA matrix"));
        }

        auto result = RunBatchOnDevice(_impl->options,
                                       _impl->pairStore,
                                       entry,
                                       lBra,
                                       lKet,
                                       boysPtr,
                                       pairMetaPtr,
                                       _impl->hostTables.pairMeta.data(),
                                       primDataPtr,
                                       braTransformsPtr,
                                       ketTransformsPtr,
                                       braTransformsF32Ptr,
                                       ketTransformsF32Ptr,
                                       batch,
                                       nullptr,
                                       outF32,
                                       errorBounds,
                                       outBase,
                                       true,
                                       static_cast<cudaStream_t>(stream),
                                       _impl->cublas,
                                       _impl->transferAccounting);

        if (!result.has_value())
        {
            return std::unexpected(result.error());
        }

        for (const MdQuartetTask& task : batch.tasks)
        {
            outBase +=
                _impl->pairStore[task.braPair].nFuncs * _impl->pairStore[task.ketPair].nFuncs;
        }
    }

    auto sync = CheckCudaError(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)),
                               "cudaStreamSynchronize");

    if (!sync.has_value())
    {
        return std::unexpected(sync.error());
    }

    // Device-side writes through the raw handles (fused/V2 kernels and the
    // cuBLAS GEMMs): mark both output buffers dirty so SyncToHost copies.
    outBuffer->MarkDeviceDirty();
    boundsBuffer->MarkDeviceDirty();

    auto copy = outBuffer->SyncToHost();

    if (!copy.has_value())
    {
        return std::unexpected(copy.error());
    }

    auto boundsCopy = boundsBuffer->SyncToHost();

    if (!boundsCopy.has_value())
    {
        return std::unexpected(boundsCopy.error());
    }

    return EriCudaCertifiedBatch{
        std::move(outBuffer->HostView()), std::move(boundsBuffer->HostView()), std::move(computed)};
#endif
}

Result<EriCudaRiBatch> EriCudaEngine::ComputeRiBatch(const qcx::basisset::BasisSet& auxBasis,
                                                     const std::vector<ShellTriple>& triples,
                                                     void* stream) {
    if (triples.empty())
    {
        return std::unexpected(InvalidArgument("the triple list is empty"));
    }

    auto device = CheckCudaError(cudaSetDevice(_impl->deviceOrdinal), "cudaSetDevice");

    if (!device.has_value())
    {
        return std::unexpected(device.error());
    }

    auto streamSet = CheckCublasStatus(
        cublasSetStream(_impl->cublas, static_cast<cudaStream_t>(stream)), "cublasSetStream");

    if (!streamSet.has_value())
    {
        return std::unexpected(streamSet.error());
    }

    // The auxiliary pair list and the flattened aux shells (the centers
    // come from the molecule, which the engine keeps alive).
    auto auxPairList = BuildShellPairs(*_impl->molecule, auxBasis);

    if (!auxPairList.has_value())
    {
        return std::unexpected(auxPairList.error());
    }

    auto auxShells = internal::FlattenShells(*_impl->molecule, auxBasis, *auxPairList);

    if (!auxShells.has_value())
    {
        return std::unexpected(auxShells.error());
    }

    const std::size_t nOrbitalShells = _impl->pairList.shells.size();
    const std::size_t nAuxShells = auxShells->size();
    const std::size_t nOrbitalPairs = _impl->pairList.pairs.size();

    // The combined pair store: the orbital pairs followed by one aux pair
    // per aux shell (BuildAuxPairData - the md_vrr_3c.hpp phantom
    // construction (P, s_0), verbatim).
    std::vector<MdPairData> combined = _impl->pairStore;

    for (const internal::MdShellInput& shell : *auxShells)
    {
        const std::array<double, 3> center = {shell.cx, shell.cy, shell.cz};
        combined.push_back(internal::BuildAuxPairData(shell.contractions, center));
    }

    // The task list: one (braPair, auxShell) per triple, with the i <= j
    // and range validation (the RunRiBatches contract).
    struct Item {
        internal::RiTask task;
        int lBra;
        int lKet;
        std::size_t braRowPairs;
        std::size_t outputSize;
        std::size_t scratchSize;
    };

    std::vector<Item> items;
    items.reserve(triples.size());

    for (const ShellTriple& triple : triples)
    {
        if (triple.i > triple.j || triple.i >= nOrbitalShells || triple.j >= nOrbitalShells)
        {
            return std::unexpected(InvalidArgument("the orbital shell indices violate i <= j"));
        }

        if (triple.k >= nAuxShells)
        {
            return std::unexpected(InvalidArgument("the aux shell index is out of range"));
        }

        const std::size_t braPair = PairIndexOf(triple.i, triple.j, _impl->pairList);
        const std::size_t auxShell = nOrbitalPairs + triple.k;
        const MdPairData& bra = combined[braPair];
        const MdPairData& aux = combined[auxShell];
        const int lBra = bra.la + bra.lb;
        const int lKet = aux.la; // la = lP, lb = 0.
        const EriCudaClassDispatchEntry* entry = QcxEriCudaFindClass(lBra, lKet);

        if (entry == nullptr)
        {
            return std::unexpected(Unimplemented("3c class (" + std::to_string(lBra) + "," +
                                                 std::to_string(lKet) +
                                                 ") is outside the instantiated CUDA matrix"));
        }

        const std::size_t hermBra = static_cast<std::size_t>(Hermite3DCount(lBra));
        const std::size_t hermKet = static_cast<std::size_t>(Hermite3DCount(lKet));
        const std::size_t mPq = bra.rowPairs * hermBra;
        items.push_back(Item{internal::RiTask{braPair, auxShell},
                             lBra,
                             lKet,
                             bra.rowPairs,
                             bra.nFuncs * aux.nFuncs,
                             mPq * hermKet + mPq * aux.nFuncs});
    }

    // The RunRiBatches sort (verbatim): class-major, then the ket-group
    // order of the kernel's pass 1.
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

    std::vector<ShellTriple> computed;
    computed.reserve(items.size());

    for (const Item& item : items)
    {
        const ShellPairIndex& pair = _impl->pairList.pairs[item.task.braPair];
        computed.push_back(ShellTriple{pair.i, pair.j, item.task.auxShell - nOrbitalPairs});
    }

    std::size_t total = 0;

    for (const Item& item : items)
    {
        total += item.outputSize;
    }

    // The per-class batches with the RunRiBatches cap (output + scratch,
    // conservative; the batch boundaries do not affect the packed layout -
    // the per-task global offsets below do).
    std::vector<std::size_t> globalOffset(items.size(), 0);
    std::size_t cumulative = 0;

    for (std::size_t i = 0; i < items.size(); ++i)
    {
        globalOffset[i] = cumulative;
        cumulative += items[i].outputSize;
    }

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
            const std::size_t itemBytes =
                (items[t].outputSize + items[t].scratchSize) * sizeof(double);

            if (batchBytes + itemBytes > _impl->options.maxBatchBytes && t > batchStart)
            {
                MdClassBatch batch;
                batch.lBra = items[batchStart].lBra;
                batch.lKet = items[batchStart].lKet;
                batch.pairStore = &combined;
                batch.boundsBase = batchStart;

                for (std::size_t s = batchStart; s < t; ++s)
                {
                    batch.tasks.push_back(
                        MdQuartetTask{items[s].task.braPair,
                                      items[s].task.auxShell,
                                      globalOffset[s] - globalOffset[batchStart]});
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
        batch.pairStore = &combined;
        batch.boundsBase = batchStart;

        for (std::size_t s = batchStart; s < runEnd; ++s)
        {
            batch.tasks.push_back(MdQuartetTask{items[s].task.braPair,
                                                items[s].task.auxShell,
                                                globalOffset[s] - globalOffset[batchStart]});
        }

        batches.push_back(std::move(batch));
        runStart = runEnd;
    }

    // The per-call combined upload (fp64 only - the RI path has no
    // certified lane in this step).
    auto hostTables = BuildHostTables(combined);

    if (!hostTables.has_value())
    {
        return std::unexpected(hostTables.error());
    }

    auto pairMetaBuffer =
        DeviceBuffer<EriCudaPairMeta, CudaTag>::Create(hostTables->pairMeta.size());

    if (!pairMetaBuffer.has_value())
    {
        return std::unexpected(pairMetaBuffer.error());
    }

    pairMetaBuffer->HostView() = hostTables->pairMeta;

    auto pairMetaSync = pairMetaBuffer->SyncToDevice();

    if (!pairMetaSync.has_value())
    {
        return std::unexpected(pairMetaSync.error());
    }

    auto primDataBuffer = DeviceBuffer<double, CudaTag>::Create(hostTables->primData.size());

    if (!primDataBuffer.has_value())
    {
        return std::unexpected(primDataBuffer.error());
    }

    primDataBuffer->HostView() = hostTables->primData;

    auto primDataSync = primDataBuffer->SyncToDevice();

    if (!primDataSync.has_value())
    {
        return std::unexpected(primDataSync.error());
    }

    auto braTransformsBuffer =
        DeviceBuffer<double, CudaTag>::Create(hostTables->braTransforms.size());

    if (!braTransformsBuffer.has_value())
    {
        return std::unexpected(braTransformsBuffer.error());
    }

    braTransformsBuffer->HostView() = hostTables->braTransforms;

    auto braTransformsSync = braTransformsBuffer->SyncToDevice();

    if (!braTransformsSync.has_value())
    {
        return std::unexpected(braTransformsSync.error());
    }

    auto ketTransformsBuffer =
        DeviceBuffer<double, CudaTag>::Create(hostTables->ketTransforms.size());

    if (!ketTransformsBuffer.has_value())
    {
        return std::unexpected(ketTransformsBuffer.error());
    }

    ketTransformsBuffer->HostView() = hostTables->ketTransforms;

    auto ketTransformsSync = ketTransformsBuffer->SyncToDevice();

    if (!ketTransformsSync.has_value())
    {
        return std::unexpected(ketTransformsSync.error());
    }

    auto outBuffer = DeviceBuffer<double, CudaTag>::Create(total);

    if (!outBuffer.has_value())
    {
        return std::unexpected(outBuffer.error());
    }

    // The Boys copy of the device statics: the RI path
    // builds its own pair tables per call but shares the statics' Boys
    // copy - through the active copy, with the per-call release guard, the
    // same as the 2e paths.
    auto statics = _impl->EnsureStatics();

    if (!statics.has_value())
    {
        return std::unexpected(statics.error());
    }

    Impl::StaticsGuard staticsGuard(*_impl);

    const auto* boysPtr =
        static_cast<const EriCudaBoysTables*>((*statics)->boys.DeviceHandle().Raw());
    const auto* pairMetaPtr =
        static_cast<const EriCudaPairMeta*>(pairMetaBuffer->DeviceHandle().Raw());
    const auto* primDataPtr = static_cast<const double*>(primDataBuffer->DeviceHandle().Raw());
    const auto* braTransformsPtr =
        static_cast<const double*>(braTransformsBuffer->DeviceHandle().Raw());
    const auto* ketTransformsPtr =
        static_cast<const double*>(ketTransformsBuffer->DeviceHandle().Raw());
    double* outF64 = static_cast<double*>(outBuffer->DeviceHandle().Raw());
    std::size_t outBase = 0;

    for (const MdClassBatch& batch : batches)
    {
        const int lBra = batch.lBra;
        const int lKet = batch.lKet;
        const EriCudaClassDispatchEntry* entry = QcxEriCudaFindClass(lBra, lKet);

        if (entry == nullptr)
        {
            return std::unexpected(Unimplemented("3c class (" + std::to_string(lBra) + "," +
                                                 std::to_string(lKet) +
                                                 ") is outside the instantiated CUDA matrix"));
        }

        auto result = RunBatchOnDevice(_impl->options,
                                       combined,
                                       entry,
                                       lBra,
                                       lKet,
                                       boysPtr,
                                       pairMetaPtr,
                                       hostTables->pairMeta.data(),
                                       primDataPtr,
                                       braTransformsPtr,
                                       ketTransformsPtr,
                                       nullptr,
                                       nullptr,
                                       batch,
                                       outF64,
                                       nullptr,
                                       nullptr,
                                       outBase,
                                       false,
                                       static_cast<cudaStream_t>(stream),
                                       _impl->cublas,
                                       _impl->transferAccounting);

        if (!result.has_value())
        {
            return std::unexpected(result.error());
        }

        for (const MdQuartetTask& task : batch.tasks)
        {
            outBase += combined[task.braPair].nFuncs * combined[task.ketPair].nFuncs;
        }
    }

    auto sync = CheckCudaError(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)),
                               "cudaStreamSynchronize");

    if (!sync.has_value())
    {
        return std::unexpected(sync.error());
    }

    // The 3c kernels wrote the device side directly: mark the buffer dirty
    // before the read-back (see the 2e path).
    outBuffer->MarkDeviceDirty();

    auto copy = outBuffer->SyncToHost();

    if (!copy.has_value())
    {
        return std::unexpected(copy.error());
    }

    return EriCudaRiBatch{std::move(outBuffer->HostView()), std::move(computed)};
}

// ---------------------------------------------------------------------------
// The GPU Fock builder
// ---------------------------------------------------------------------------
//
// GpuJkFockBuilder (gpu_fock_build.hpp): the DirectJkFockBuilder contract
// with the J/K contraction moved onto the device. Create builds the
// EriCudaEngine (its pair store and device tables are reused through the
// friend access) and the cached Schwarz neighbor list; BuildFock runs the
// SHARED screening path (internal/fock_screen.hpp - bit-identical candidate
// lists to the CPU builder), evaluates the screened quartets through the
// engine's batch machinery (AssembleClassBatches + RunBatchOnDevice, the
// exact ComputeBatch flow), and contracts each batch with the generic
// per-quartet kernel of eri_cuda_fock.cu (EriFockContractKernel - one block
// per quartet, atomicAdd accumulations, the elementwise port of
// FockContractor::AccumulateBlock). The certified fp32 lane reads back the
// per-quartet bounds and accumulates the density-weighted certified sum
// exactly like the CPU lane.

struct GpuJkFockBuilder::State {
    FockBuildOptions _options;
    ShellPairList _pairList;
    std::vector<MdPairData> _pairStore;
    std::vector<double> _schwarz;
    // The cached Schwarz neighbor list as a SparsityPattern: the identical
    // CSR the CPU builder constructs, so both
    // builders screen the same candidate lists.
    qcx::memory::SparsityPattern<qcx::backend::CpuTag> _neighborPattern;
    qcx::memory::Tensor<double, 2, qcx::backend::CpuTag> _coreHamiltonianTensor;
    Eigen::MatrixXd _coreHamiltonian;
    /// The engine whose device tables every BuildFock call contracts
    /// through (mutable: the launches use its stream and cuBLAS handle).
    mutable EriCudaEngine _engine;

    State(FockBuildOptions optionsIn,
          ShellPairList pairListIn,
          std::vector<MdPairData> pairStoreIn,
          std::vector<double> schwarzIn,
          qcx::memory::SparsityPattern<qcx::backend::CpuTag> neighborPatternIn,
          qcx::memory::Tensor<double, 2, qcx::backend::CpuTag> coreHamiltonianTensorIn,
          Eigen::MatrixXd coreHamiltonianIn,
          EriCudaEngine engineIn) :
        _options(optionsIn), _pairList(std::move(pairListIn)), _pairStore(std::move(pairStoreIn)),
        _schwarz(std::move(schwarzIn)), _neighborPattern(std::move(neighborPatternIn)),
        _coreHamiltonianTensor(std::move(coreHamiltonianTensorIn)),
        _coreHamiltonian(std::move(coreHamiltonianIn)), _engine(std::move(engineIn)) {}

    /// Runs one contraction pass over the screened tasks - the fp64 pass, or
    /// the certified fp32 lane (the FockContractor::RunPass mirror; the
    /// generic per-quartet kernel of eri_cuda_fock.cu as the contractor).
    /// The orchestration is the ComputeBatch / ComputeBatchCertified flow
    /// verbatim - AssembleClassBatches, per-batch RunBatchOnDevice with the
    /// running outBase, one stream sync per pass - with each batch contracted
    /// on the device, and the fp32 lane's per-quartet bounds read back and
    /// accumulated into the density-weighted certified sum.
    /// \param tasks The screened tasks of this pass.
    /// \param fp32 true for the certified lane, false for the fp64 lane.
    /// \param densityWeightOf The fp32-routing weight per (bra, ket) task in
    /// both orientations (the CPU builder's map).
    /// \param fockBuffer The device Fock, pre-initialized with H (the kernel
    /// accumulates into it).
    /// \param densityBuffer The device density, the row-major u*n+v
    /// flattening of the Eigen copy.
    /// \param certifiedBoundSum The running certified sum; fp32 passes
    /// accumulate into it.
    /// \returns An Error from any batch/launch/copy step.
    qcx::Result<void> RunDevicePass(const std::vector<internal::MdQuartetTask>& tasks,
                                    bool fp32,
                                    const std::unordered_map<std::size_t, double>& densityWeightOf,
                                    qcx::memory::DeviceBuffer<double, CudaTag>& fockBuffer,
                                    qcx::memory::DeviceBuffer<double, CudaTag>& densityBuffer,
                                    double& certifiedBoundSum) const;
};

qcx::Result<void> GpuJkFockBuilder::State::RunDevicePass(
    const std::vector<internal::MdQuartetTask>& tasks,
    bool fp32,
    const std::unordered_map<std::size_t, double>& densityWeightOf,
    qcx::memory::DeviceBuffer<double, CudaTag>& fockBuffer,
    qcx::memory::DeviceBuffer<double, CudaTag>& densityBuffer,
    double& certifiedBoundSum) const {
    const std::size_t n = _pairList.functionCount;
    const std::size_t nPairs = _pairList.pairs.size();

    if (tasks.empty())
    {
        return {};
    }

    std::vector<ShellQuartet> quartets;

    for (const internal::MdQuartetTask& task : tasks)
    {
        const ShellPairIndex& braPair = _pairList.pairs[task.braPair];
        const ShellPairIndex& ketPair = _pairList.pairs[task.ketPair];
        quartets.push_back(ShellQuartet{braPair.i, braPair.j, ketPair.i, ketPair.j});
    }

    std::vector<ShellQuartet> computed;
    auto batches = internal::AssembleClassBatches(
        _pairStore, _pairList, quartets, _options.maxBatchBytes, computed);

    if (!batches.has_value())
    {
        return std::unexpected(batches.error());
    }

    std::size_t totalWords = 0;

    for (const MdClassBatch& batch : *batches)
    {
        for (const MdQuartetTask& task : batch.tasks)
        {
            totalWords += _pairStore[task.braPair].nFuncs * _pairStore[task.ketPair].nFuncs;
        }
    }

    if (totalWords == 0)
    {
        return {};
    }

    std::optional<DeviceBuffer<double, CudaTag>> outF64Buffer;
    std::optional<DeviceBuffer<float, CudaTag>> outF32Buffer;
    std::optional<DeviceBuffer<double, CudaTag>> boundsBuffer;

    if (fp32)
    {
        auto outF32Created = DeviceBuffer<float, CudaTag>::Create(totalWords);

        if (!outF32Created.has_value())
        {
            return std::unexpected(outF32Created.error());
        }

        outF32Buffer.emplace(std::move(*outF32Created));

        auto boundsCreated = DeviceBuffer<double, CudaTag>::Create(computed.size());

        if (!boundsCreated.has_value())
        {
            return std::unexpected(boundsCreated.error());
        }

        boundsBuffer.emplace(std::move(*boundsCreated));
    } else
    {
        auto outF64Created = DeviceBuffer<double, CudaTag>::Create(totalWords);

        if (!outF64Created.has_value())
        {
            return std::unexpected(outF64Created.error());
        }

        outF64Buffer.emplace(std::move(*outF64Created));
    }

    // The device statics: the FastPath reads the
    // retained copy; the LightPath uploads the statics at the call entry
    // of the first pass and releases them at the end of BuildFock (the
    // per-call release guard lives there - a pass is not a call boundary).
    auto statics = _engine._impl->EnsureStatics();

    if (!statics.has_value())
    {
        return std::unexpected(statics.error());
    }

    const auto* boysPtr =
        static_cast<const EriCudaBoysTables*>((*statics)->boys.DeviceHandle().Raw());
    const auto* pairMetaPtr =
        static_cast<const EriCudaPairMeta*>((*statics)->pairMeta.DeviceHandle().Raw());
    const auto* primDataPtr = static_cast<const double*>((*statics)->primData.DeviceHandle().Raw());
    const auto* braTransformsPtr =
        static_cast<const double*>((*statics)->braTransforms.DeviceHandle().Raw());
    const auto* ketTransformsPtr =
        static_cast<const double*>((*statics)->ketTransforms.DeviceHandle().Raw());
    const auto* braTransformsF32Ptr =
        static_cast<const float*>((*statics)->braTransformsF32.DeviceHandle().Raw());
    const auto* ketTransformsF32Ptr =
        static_cast<const float*>((*statics)->ketTransformsF32.DeviceHandle().Raw());
    const double* densityPtr = static_cast<const double*>(densityBuffer.DeviceHandle().Raw());
    double* fockPtr = static_cast<double*>(fockBuffer.DeviceHandle().Raw());
    std::size_t outBase = 0;

    // ---- the device-span bracket opens here (the
    // corrected-method instrument of benchmarks/gpu_eri_benchmark.cpp):
    // the queue is empty at this record, so it fires at the pass's first
    // launch; the close after the sync below fires when the queue drains
    // - the elapsed is the device-timeline span of the launches INSIDE
    // the concurrent BuildFock (the pass's batch loop keeps the default
    // stream continuously fed, so the span is the true device share, not
    // a host-wall measurement). The pre-loop host work (quartet assembly,
    // AssembleClassBatches, buffer allocation, the LightPath statics
    // upload) stays outside the bracket and counts as host time.
    if (_engine._impl->passStartEvent != nullptr && _engine._impl->passEndEvent != nullptr)
    {
        auto startRecorded = CheckCudaError(cudaEventRecord(_engine._impl->passStartEvent, nullptr),
                                            "cudaEventRecord");

        if (!startRecorded.has_value())
        {
            return std::unexpected(startRecorded.error());
        }
    }

    for (const MdClassBatch& batch : *batches)
    {
        const int lBra = batch.lBra;
        const int lKet = batch.lKet;
        const EriCudaClassDispatchEntry* entry = QcxEriCudaFindClass(lBra, lKet);

        if (entry == nullptr)
        {
            return std::unexpected(Unimplemented("class (" + std::to_string(lBra) + "," +
                                                 std::to_string(lKet) +
                                                 ") is outside the instantiated CUDA matrix"));
        }

        auto result =
            RunBatchOnDevice(_engine._impl->options,
                             _engine._impl->pairStore,
                             entry,
                             lBra,
                             lKet,
                             boysPtr,
                             pairMetaPtr,
                             _engine._impl->hostTables.pairMeta.data(),
                             primDataPtr,
                             braTransformsPtr,
                             ketTransformsPtr,
                             braTransformsF32Ptr,
                             ketTransformsF32Ptr,
                             batch,
                             fp32 ? static_cast<double*>(nullptr)
                                  : static_cast<double*>(outF64Buffer->DeviceHandle().Raw()),
                             fp32 ? static_cast<float*>(outF32Buffer->DeviceHandle().Raw())
                                  : static_cast<float*>(nullptr),
                             fp32 ? static_cast<double*>(boundsBuffer->DeviceHandle().Raw())
                                  : static_cast<double*>(nullptr),
                             outBase,
                             fp32,
                             nullptr,
                             _engine._impl->cublas,
                             _engine._impl->transferAccounting);

        if (!result.has_value())
        {
            return std::unexpected(result.error());
        }

        // The per-task contraction metadata: the quartet's shell geometry
        // and the K-orbit divisor, computed host-side in the batch's task
        // order (the generic kernel's only per-class knowledge arrives
        // here).
        std::vector<EriCudaFockTaskMeta> metas;
        metas.reserve(batch.tasks.size());

        for (const MdQuartetTask& task : batch.tasks)
        {
            const ShellPairIndex& braPair = _pairList.pairs[task.braPair];
            const ShellPairIndex& ketPair = _pairList.pairs[task.ketPair];
            const ShellInfo& shellA = _pairList.shells[braPair.i];
            const ShellInfo& shellB = _pairList.shells[braPair.j];
            const ShellInfo& shellC = _pairList.shells[ketPair.i];
            const ShellInfo& shellD = _pairList.shells[ketPair.j];
            const double kMultiplicity =
                ((braPair.i == braPair.j) ? 2.0 : 1.0) * ((ketPair.i == ketPair.j) ? 2.0 : 1.0) *
                ((braPair.i == ketPair.i && braPair.j == ketPair.j) ? 2.0 : 1.0);
            metas.push_back(
                EriCudaFockTaskMeta{static_cast<unsigned int>(task.outputOffset),
                                    static_cast<unsigned int>(shellA.functionOffset),
                                    static_cast<unsigned int>(shellB.functionOffset),
                                    static_cast<unsigned int>(shellC.functionOffset),
                                    static_cast<unsigned int>(shellD.functionOffset),
                                    static_cast<unsigned int>(ShellFunctionCount(shellA)),
                                    static_cast<unsigned int>(ShellFunctionCount(shellB)),
                                    static_cast<unsigned int>(ShellFunctionCount(shellC)),
                                    static_cast<unsigned int>(ShellFunctionCount(shellD)),
                                    kMultiplicity,
                                    braPair.i == braPair.j,
                                    ketPair.i == ketPair.j,
                                    task.braPair == task.ketPair});
        }

        auto metaBuffer = DeviceBuffer<EriCudaFockTaskMeta, CudaTag>::Create(metas.size());

        if (!metaBuffer.has_value())
        {
            return std::unexpected(metaBuffer.error());
        }

        metaBuffer->HostView() = metas;

        auto metaSync = metaBuffer->SyncToDevice();

        if (!metaSync.has_value())
        {
            return std::unexpected(metaSync.error());
        }

        // The per-batch meta upload: the screened-lists class of the
        // accounting (one entry per screened task of the batch, every pass).
        _engine._impl->transferAccounting.listUploadBytes +=
            metaBuffer->Size() * sizeof(EriCudaFockTaskMeta);

        // Both integral pointers are valid: the unused lane receives the
        // same buffer (the kernel dereferences only the selected lane - the
        // documented EriCudaFockArgs contract).
        const double* batchF64 =
            fp32 ? reinterpret_cast<const double*>(outF32Buffer->DeviceHandle().Raw())
                 : static_cast<const double*>(outF64Buffer->DeviceHandle().Raw());
        const float* batchF32 =
            fp32 ? static_cast<const float*>(outF32Buffer->DeviceHandle().Raw())
                 : reinterpret_cast<const float*>(outF64Buffer->DeviceHandle().Raw());
        EriCudaFockArgs args;
        args.tasks = static_cast<const EriCudaFockTaskMeta*>(metaBuffer->DeviceHandle().Raw());
        args.integralsF64 = batchF64 + outBase;
        args.integralsF32 = batchF32 + outBase;
        args.density = densityPtr;
        args.fock = fockPtr;
        args.nTasks = batch.tasks.size();
        args.n = static_cast<int>(n);
        args.precision = fp32 ? kEriCudaFockPrecisionFp32 : kEriCudaFockPrecisionFp64;
        args.buildExchangeOnly = _options.buildExchangeOnly;
        args.buildCoulombOnly = _options.buildCoulombOnly;
        args.threads = 256;
        args.stream = nullptr;

        auto launch = CheckLaunchCode(QcxEriCudaFockContraction(args), "fock contraction kernel");

        if (!launch.has_value())
        {
            return std::unexpected(launch.error());
        }

        for (const MdQuartetTask& task : batch.tasks)
        {
            outBase += _pairStore[task.braPair].nFuncs * _pairStore[task.ketPair].nFuncs;
        }
    }

    auto sync = CheckCudaError(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");

    if (!sync.has_value())
    {
        return std::unexpected(sync.error());
    }

    // ---- the device-span bracket closes here: the sync
    // above drained every launch of the pass, so the end record fires at
    // the queue-empty moment right after it and the elapsed between the
    // two records is the pass's device-timeline span (accumulated per
    // pass into the call's span). The fp32 lane's bounds read-back and
    // the certified-sum accumulation below fall outside the bracket
    // (host-side work of the call).
    if (_engine._impl->passStartEvent != nullptr && _engine._impl->passEndEvent != nullptr)
    {
        auto endRecorded = CheckCudaError(cudaEventRecord(_engine._impl->passEndEvent, nullptr),
                                          "cudaEventRecord");

        if (!endRecorded.has_value())
        {
            return std::unexpected(endRecorded.error());
        }

        auto endSynced = CheckCudaError(cudaEventSynchronize(_engine._impl->passEndEvent),
                                        "cudaEventSynchronize");

        if (!endSynced.has_value())
        {
            return std::unexpected(endSynced.error());
        }

        float spanMs = 0.0F;

        if (cudaError_t elapsedError = cudaEventElapsedTime(
                &spanMs, _engine._impl->passStartEvent, _engine._impl->passEndEvent);
            elapsedError != cudaSuccess)
        {
            return std::unexpected(DeviceError(std::string("cudaEventElapsedTime: ") +
                                               cudaGetErrorString(elapsedError)));
        }

        _engine._impl->fockDeviceSpanMs += static_cast<double>(spanMs);
    }

    if (fp32)
    {
        // The per-quartet certified bounds were written device-side through
        // the raw handle: mark the buffer dirty and read the
        // density-weighted sum back (the CPU lane's accumulation).
        boundsBuffer->MarkDeviceDirty();

        auto boundsCopy = boundsBuffer->SyncToHost();

        if (!boundsCopy.has_value())
        {
            return std::unexpected(boundsCopy.error());
        }

        // The certified lane's per-quartet bound-sum read-back (the bounds
        // class of the accounting; 0 when the fp32 lane did not run).
        _engine._impl->transferAccounting.boundsDownloadBytes +=
            boundsBuffer->Size() * sizeof(double);

        for (std::size_t t = 0; t < computed.size(); ++t)
        {
            const ShellQuartet& quartet = computed[t];
            const std::size_t braPair = PairIndexOf(quartet.i, quartet.j, _pairList);
            const std::size_t ketPair = PairIndexOf(quartet.k, quartet.l, _pairList);
            const auto weightIt = densityWeightOf.find(braPair * nPairs + ketPair);

            if (weightIt == densityWeightOf.end())
            {
                return internal::CertifiedRoutingError();
            }

            certifiedBoundSum += weightIt->second * boundsBuffer->HostView()[t];
        }
    }

    return {};
}

GpuJkFockBuilder::GpuJkFockBuilder(std::shared_ptr<const State> state) : _state(std::move(state)) {}

qcx::Result<GpuJkFockBuilder> GpuJkFockBuilder::Create(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
    const FockBuildOptions& options) {
    if (options.maxBatchBytes == 0)
    {
        return std::unexpected(InvalidArgument("maxBatchBytes must be positive"));
    }

    // The device side: EriCudaEngine::Create probes the device count (the
    // kDeviceError self-skip convention), validates the shell cap, and
    // builds the canonical pair store and the device tables. The friend
    // access reads the canonical pair list/store back instead of rebuilding
    // them (the engine's Create already ran BuildShellPairs + BuildPairData).
    EriCudaOptions engineOptions;
    engineOptions.maxBatchBytes = options.maxBatchBytes;
    // Forward the device workspace budget and the forced-LightPath knob
    // into the engine options - the Create-time mode
    // decision runs inside the engine (the budget's Remaining is read once,
    // at the decision).
    engineOptions.deviceWorkspaceBudget = options.deviceWorkspaceBudget;
    engineOptions.forceLightPath = options.forceGpuLightPath;

    auto engine = EriCudaEngine::Create(molecule, basisSet, engineOptions);

    if (!engine.has_value())
    {
        return std::unexpected(engine.error());
    }

    const ShellPairList& pairList = engine->_impl->pairList;
    const std::vector<MdPairData>& pairStore = engine->_impl->pairStore;
    const std::size_t n = pairList.functionCount;

    if (coreHamiltonian.Shape()[0] != n || coreHamiltonian.Shape()[1] != n)
    {
        return std::unexpected(InvalidArgument("core Hamiltonian shape mismatch"));
    }

    auto schwarz = ComputeSchwarzBounds(molecule, basisSet);

    if (!schwarz.has_value())
    {
        return std::unexpected(schwarz.error());
    }

    // The cached Schwarz neighbor list: the identical CSR the CPU builder
    // constructs (internal/fock_screen.hpp), so both builders screen the
    // same candidate lists.
    std::vector<std::size_t> neighborRowOffsets;
    std::vector<std::size_t> neighborIndices;
    internal::BuildNeighborList(
        pairList, *schwarz, options.accuracy, neighborRowOffsets, neighborIndices);
    auto neighborPattern = qcx::memory::BuildSparsityFromAdjacency<qcx::backend::CpuTag>(
        neighborRowOffsets, neighborIndices);

    if (!neighborPattern.has_value())
    {
        return std::unexpected(neighborPattern.error());
    }

    // Keep the original Tensor too - CoreHamiltonian() returns it by
    // reference (the DirectJkFockBuilder::Create pattern).
    auto coreHamiltonianClone = coreHamiltonian.Clone();

    if (!coreHamiltonianClone.has_value())
    {
        return std::unexpected(coreHamiltonianClone.error());
    }

    const Eigen::MatrixXd core = internal::TensorToEigen(coreHamiltonian);

    // The pair list/store are copied into the State before the engine move
    // completes; the copies happen in the State ctor (after the arguments
    // are evaluated), but the moved-from engine's Impl stays alive inside
    // the State, so the referenced tables are readable throughout.
    auto state = std::make_shared<State>(options,
                                         pairList,
                                         pairStore,
                                         std::move(*schwarz),
                                         std::move(*neighborPattern),
                                         std::move(*coreHamiltonianClone),
                                         std::move(core),
                                         std::move(*engine));

    return GpuJkFockBuilder(std::move(state));
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> GpuJkFockBuilder::BuildFock(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    double* certifiedBoundSumOut,
    const PrecisionLadderInputs* ladderInputs) const {
    const State& state = *_state;
    const std::size_t n = state._pairList.functionCount;

    // The per-call release guard: the LightPath uploads
    // the statics at the first pass (the RunDevicePass EnsureStatics) and
    // releases them at the end of the call - the guard runs the release at
    // every exit, early returns included. No-op on the FastPath (the
    // retained copy stays).
    EriCudaEngine::Impl::StaticsGuard staticsGuard(*state._engine._impl);

    // The per-call transfer accounting:
    // reset at the call entry, incremented at every Fock-path transfer site,
    // verified by the debug assert below.
    state._engine._impl->transferAccounting = {};

    // The per-call device-timeline span (the instrument):
    // reset at the call entry, accumulated per pass by RunDevicePass.
    state._engine._impl->fockDeviceSpanMs = 0.0;

    if (density.Shape()[0] != n || density.Shape()[1] != n)
    {
        return std::unexpected(InvalidArgument("density shape mismatch"));
    }

    const Eigen::MatrixXd d = internal::TensorToEigen(density);

    // The contractions read both orientations of every unordered pair block
    // and the K transpose-writes assume the symmetry (the documented
    // BuildFock precondition, Debug-checked like the CPU builder).
    assert(d.isApprox(d.transpose()));

    // ---- the shared screening pass (bit-identical to the CPU builder) ----
    const double densityThreshold = DensityThreshold(state._options.accuracy);
    const double mixedThreshold = MixedPrecisionThreshold(state._options.accuracy);
    // The ladder's band classification replaces the certified routing when
    // engaged (the routing gate stays the ladder-off path, bit-identical).
    const bool ladderEngaged = ladderInputs != nullptr && ladderInputs->schedule != nullptr;
    // The lane's device-governed default (a device probe's verdict,
    // 2026-09-12) - the same ONE resolution point the
    // CPU builder reads, so the two builders cannot disagree.
    const bool certifiedLane =
        ResolveCertifiedLane(state._options) && mixedThreshold > 0.0 && !ladderEngaged;

    const std::size_t nPairs = state._pairList.pairs.size();
    std::vector<internal::MdQuartetTask> fp64Quartets;
    std::vector<internal::MdQuartetTask> fp32Quartets;
    std::vector<double> fp32DensityWeights;

    // The per-call max-density vector engages the quartet-level
    // product gate on the GPU screening path exactly as on the CPU path (the
    // shared ScreenAll). The device kernel predates the per-element
    // re-filter, so the GPU contraction filters nothing - the parity pins
    // run usePerElementScreening = false (the flag contract).
    std::vector<double> shellPairMaxDensity;

    // The ladder's classification consumes the per-pair maxima, so the
    // ladder builds the vector even when the per-element screening flag
    // did not (the flag still gates nothing here - the device kernel
    // predates the re-filter).
    if (state._options.usePerElementScreening || ladderEngaged)
    {
        shellPairMaxDensity = internal::BuildShellPairMaxDensity(d, state._pairList);
    }

    const internal::ScreeningContext context{state._pairList,
                                             state._schwarz,
                                             state._pairStore,
                                             state._options,
                                             shellPairMaxDensity.empty() ? nullptr
                                                                         : &shellPairMaxDensity};
    internal::ScreenAll(context,
                        d,
                        state._neighborPattern,
                        densityThreshold,
                        mixedThreshold,
                        certifiedLane,
                        fp64Quartets,
                        fp32Quartets,
                        fp32DensityWeights);

    // The fp32 routing weight per (bra, ket) task in both orientations (the
    // CPU builder's map): the class-swap in AssembleClassBatches may swap a
    // quartet's bra/ket roles, so the computed list must be looked up either
    // way; a miss is a routing bookkeeping bug and must not under-count the
    // certified sum silently.
    // The ladder pass (Package B): the B_screen co-term, the budget, and
    // the per-batch band partition replacing the certified routing's two
    // lists (the band lists dispatch between class runs, never inside a
    // kernel).
    std::vector<internal::MdQuartetTask> ladderFp16;
    std::vector<internal::MdQuartetTask> ladderCertified;
    std::vector<internal::MdQuartetTask> ladderMixed;

    if (ladderEngaged)
    {
        const double bScreen =
            internal::ScreenBoundSum(context, d, state._neighborPattern, densityThreshold);
        const EriBudgetInputs budgetInputs{
            bScreen, ladderInputs->riError, ladderInputs->densityError};
        const double classificationBudget =
            ladderInputs->schedule->ClassificationBudget(budgetInputs);
        internal::LadderPartition partition = internal::PartitionBatches(
            context, d, fp64Quartets, classificationBudget, *ladderInputs->schedule);

        fp64Quartets = std::move(partition.fp64);
        ladderFp16 = std::move(partition.fp16);
        ladderCertified = std::move(partition.fp32Certified);
        ladderMixed = std::move(partition.fp32Mixed);
        fp32Quartets = ladderFp16;
        fp32Quartets.insert(fp32Quartets.end(), ladderCertified.begin(), ladderCertified.end());
        fp32Quartets.insert(fp32Quartets.end(), ladderMixed.begin(), ladderMixed.end());
        fp32DensityWeights = std::move(partition.fp16Weights);
        fp32DensityWeights.insert(fp32DensityWeights.end(),
                                  partition.certifiedWeights.begin(),
                                  partition.certifiedWeights.end());
        fp32DensityWeights.insert(
            fp32DensityWeights.end(), partition.mixedWeights.begin(), partition.mixedWeights.end());

        // The schedule commits the A-PRIORI classification sum (the
        // delivery contract's quantity: <= T by construction, one
        // fraction-of-share per batch); the builder's certified-sum
        // out-parameter keeps the delivered kernel-bound sum (the
        // referee's delivered-error quantity).
        ladderInputs->schedule->CommitCoarseBoundSum(kFp16BoundScale * partition.fp16BoundSum +
                                                     partition.certifiedBoundSum +
                                                     partition.mixedBoundSum);
    }

    std::unordered_map<std::size_t, double> densityWeightOf;
    densityWeightOf.reserve(fp32Quartets.size() * 2);

    for (std::size_t t = 0; t < fp32Quartets.size(); ++t)
    {
        const std::size_t key = fp32Quartets[t].braPair * nPairs + fp32Quartets[t].ketPair;
        densityWeightOf[key] = fp32DensityWeights[t];
        densityWeightOf[fp32Quartets[t].ketPair * nPairs + fp32Quartets[t].braPair] =
            fp32DensityWeights[t];
    }

#if !QcxIntegralsF32
    if (!fp32Quartets.empty())
    {
        return std::unexpected(
            Unimplemented("the certified fp32 pipeline is not instantiated (QCX_INTEGRALS_F32)"));
    }
#endif

    // ---- the device buffers: the flat row-major Fock (pre-initialized
    // with the core Hamiltonian - the kernel accumulates into it) and the
    // density (the row-major u*n+v flattening of the Eigen copy).
    auto fockBuffer = DeviceBuffer<double, CudaTag>::Create(n * n);

    if (!fockBuffer.has_value())
    {
        return std::unexpected(fockBuffer.error());
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            fockBuffer->HostView()[i * n + j] = state._coreHamiltonian(i, j);
        }
    }

    auto fockSync = fockBuffer->SyncToDevice();

    if (!fockSync.has_value())
    {
        return std::unexpected(fockSync.error());
    }

    state._engine._impl->transferAccounting.fockUploadBytes += fockBuffer->Size() * sizeof(double);

    auto densityBuffer = DeviceBuffer<double, CudaTag>::Create(n * n);

    if (!densityBuffer.has_value())
    {
        return std::unexpected(densityBuffer.error());
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            densityBuffer->HostView()[i * n + j] =
                d(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
    }

    auto densitySync = densityBuffer->SyncToDevice();

    if (!densitySync.has_value())
    {
        return std::unexpected(densitySync.error());
    }

    state._engine._impl->transferAccounting.densityUploadBytes +=
        densityBuffer->Size() * sizeof(double);

    auto device =
        CheckCudaError(cudaSetDevice(state._engine._impl->deviceOrdinal), "cudaSetDevice");

    if (!device.has_value())
    {
        return std::unexpected(device.error());
    }

    auto streamSet =
        CheckCublasStatus(cublasSetStream(state._engine._impl->cublas, nullptr), "cublasSetStream");

    if (!streamSet.has_value())
    {
        return std::unexpected(streamSet.error());
    }

    double certifiedBoundSum = 0.0;

    // One pass over the screened tasks: the fp64 pass, or the certified fp32
    // lane. The per-pass pipeline is the named State method RunDevicePass
    // (the FockContractor::RunPass mirror) - AssembleClassBatches,
    // per-batch RunBatchOnDevice with the running outBase, the generic
    // EriFockContractKernel of eri_cuda_fock.cu, one stream sync per pass.
    // The ladder runs the per-band passes with their own sum accumulators
    // (the dispatch dimension - precision switches happen between class
    // runs) and commits the band-scaled ladder sum.
    if (ladderEngaged)
    {
        double fp64Sum = 0.0;
        double fp16Sum = 0.0;
        double certifiedSum = 0.0;
        double mixedSum = 0.0;

        auto runBandPass = [&](const std::vector<internal::MdQuartetTask>& tasks,
                               bool fp32,
                               double& bandSum) -> qcx::Result<void> {
            const double before = certifiedBoundSum;
            auto pass = state.RunDevicePass(
                tasks, fp32, densityWeightOf, *fockBuffer, *densityBuffer, certifiedBoundSum);

            if (!pass.has_value())
            {
                return pass;
            }

            bandSum = certifiedBoundSum - before;
            return {};
        };

        auto pass64 = runBandPass(fp64Quartets, false, fp64Sum);

        if (!pass64.has_value())
        {
            return std::unexpected(pass64.error());
        }

        auto pass16 = runBandPass(ladderFp16, true, fp16Sum);

        if (!pass16.has_value())
        {
            return std::unexpected(pass16.error());
        }

        auto passCertified = runBandPass(ladderCertified, true, certifiedSum);

        if (!passCertified.has_value())
        {
            return std::unexpected(passCertified.error());
        }

        auto passMixed = runBandPass(ladderMixed, true, mixedSum);

        if (!passMixed.has_value())
        {
            return std::unexpected(passMixed.error());
        }

        certifiedBoundSum = kFp16BoundScale * fp16Sum + certifiedSum + mixedSum;
    } else
    {
        auto fp64Pass = state.RunDevicePass(
            fp64Quartets, false, densityWeightOf, *fockBuffer, *densityBuffer, certifiedBoundSum);

        if (!fp64Pass.has_value())
        {
            return std::unexpected(fp64Pass.error());
        }

        auto fp32Pass = state.RunDevicePass(
            fp32Quartets, true, densityWeightOf, *fockBuffer, *densityBuffer, certifiedBoundSum);

        if (!fp32Pass.has_value())
        {
            return std::unexpected(fp32Pass.error());
        }
    }

    // The device Fock (pre-initialized with H, accumulated by both passes)
    // read back through the residency machine.
    fockBuffer->MarkDeviceDirty();

    auto fockCopy = fockBuffer->SyncToHost();

    if (!fockCopy.has_value())
    {
        return std::unexpected(fockCopy.error());
    }

    state._engine._impl->transferAccounting.fockDownloadBytes +=
        fockBuffer->Size() * sizeof(double);

    // The residency-contract debug assert: every Fock-path transfer site is
    // wired into the accounting, so the
    // accounted sums reproduce this call's in-scope math - the matrix
    // classes are exactly n^2 doubles each (the per-call fresh buffers sync
    // once), the statics class is the LightPath's per-call upload (0 on the
    // FastPath), the bounds class is capped by the fp32 lane's screened
    // quartets, and the list class is bounded below by the screened
    // task/meta counts (exact on the fused classes; the kV2 lane uploads the
    // ranges lists additionally, so the floor, not the equality). A transfer
    // site added without its increment breaks the contract loudly here.
    {
        const EriCudaTransferAccounting& accounted = state._engine._impl->transferAccounting;
        const std::size_t taskCount = fp64Quartets.size() + fp32Quartets.size();
        const bool lightPath = state._engine._impl->mode == FockBuildMode::kLightPath;
        const std::size_t staticsBytes = state._engine._impl->deviceFootprint.tablesBytes +
                                         state._engine._impl->deviceFootprint.boysBytes;
        assert(accounted.densityUploadBytes == n * n * sizeof(double));
        assert(accounted.fockUploadBytes == n * n * sizeof(double));
        assert(accounted.fockDownloadBytes == n * n * sizeof(double));
        assert(accounted.listUploadBytes >=
               taskCount * (sizeof(EriCudaTask) + sizeof(EriCudaFockTaskMeta)));
        assert(accounted.boundsDownloadBytes <= fp32Quartets.size() * sizeof(double));
        assert(accounted.staticsUploadBytes == (lightPath ? staticsBytes : 0));
    }

    auto tensor = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({n, n});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*tensor)(i, j) = fockBuffer->HostView()[i * n + j];
        }
    }

    tensor->MarkHostDirty();

    if (certifiedBoundSumOut != nullptr)
    {
        *certifiedBoundSumOut = certifiedBoundSum;
    }

    return std::move(*tensor);
}

const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& GpuJkFockBuilder::CoreHamiltonian()
    const {
    return _state->_coreHamiltonianTensor;
}

GpuDeviceFootprint GpuJkFockBuilder::DeviceFootprint() const {
    return _state->_engine.DeviceFootprint();
}

FockBuildMode GpuJkFockBuilder::Mode() const noexcept {
    return _state->_engine.Mode();
}

EriCudaModeInfo GpuJkFockBuilder::ModeInfo() const noexcept {
    return _state->_engine.ModeInfo();
}

EriCudaTransferAccounting GpuJkFockBuilder::TransferAccounting() const noexcept {
    return _state->_engine.TransferAccounting();
}

double GpuJkFockBuilder::LastBuildDeviceSpanMs() const noexcept {
    return _state->_engine._impl->fockDeviceSpanMs;
}

} // namespace qcx::integrals
