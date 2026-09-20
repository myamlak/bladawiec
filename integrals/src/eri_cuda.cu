// CUDA lane of the matrix-form MD ERI engine: the same per-class
// {L_bra, L_ket} pipeline as the CPU kernels of
// md_vrr.hpp - Boys seeds, the slice-triangle VRR with the CWR-part-1
// scatter, then the ket and bra transforms - with three device variants
// selected per class by the heuristic of eri_cuda.cpp:
//
//   kV0 (fused, 128 threads): L <= 4 - register-resident slice triangle.
//   kV1 (fused, 256 threads): 5 <= L <= 8 - shared-memory intermediates.
//   kV2 (global + cuBLAS):    L >= 9   - VRR to global memory, strided-
//        batched cuBLAS GEMMs for the ket transform, per-task GEMMs for the
//        bra transform.
//
// The fused kernels own one task-range chunk per block with the threads
// mapped to the (rowAB, primB) slots of the flattened slot space; the VRR algebra,
// the transform orders, the pq block layout, and the per-quartet certified
// bound mirror the CPU implementation exactly - per-class parity, not
// approximate agreement (fp64 values agree to ~1e-12 * scale; the certified
// fp32 lane carries the same a-priori bound as the CPU fp32 lane).
//
// C++20 and a CUDA-safe include list only: qcx C++23 headers would
// poison the nvcc TU. The host layer (eri_cuda.cpp) includes this file with
// QCX_ERI_CUDA_HEADER_ONLY defined and compiles Part 1 + the registry under
// MSVC; the per-class instantiation TUs (eri_cuda_instantiate.cu.in) do the
// same under nvcc and define the per-class extern "C" launchers.

#include <cstddef>
#include <cuda_runtime.h>

// The file-level include guard: the registry TU (this file compiled by
// nvcc) includes eri_cuda.cu both as its own body (Parts 1-4) and through
// eri_cuda_dispatch_gen.hpp (the header-style include of Part 4) - the
// second inclusion must emit nothing.
#ifndef QCX_ERI_CUDA_FILE_INCLUDED
#define QCX_ERI_CUDA_FILE_INCLUDED

// ---------------------------------------------------------------------------
// Part 1: host-safe constants and structs (compiled by both MSVC and nvcc)
// ---------------------------------------------------------------------------

namespace qcx::integrals::internal::cuda {

/// Return codes of the per-class launchers (positive values are cudaError_t
/// from the launch; the qcx::Result mapping lives in eri_cuda.cpp).
/// \{
inline constexpr int kEriCudaOk = 0;
inline constexpr int kEriCudaNotRegistered = -1;
inline constexpr int kEriCudaFp32Unavailable = -2;
inline constexpr int kEriCudaSharedOverflow = -3;
inline constexpr int kEriCudaInvalidFamily = -4;
/// \}

/// The variant families (the instantiated family of a class is fixed by the
/// heuristic; kV0/kV1 overrides stay within the fused family).
/// \{
inline constexpr int kEriCudaFamilyV0 = 0;
inline constexpr int kEriCudaFamilyV1 = 1;
inline constexpr int kEriCudaFamilyV2 = 2;
/// \}

/// The working scalar lane.
/// \{
inline constexpr int kEriCudaPrecisionFp64 = 0;
inline constexpr int kEriCudaPrecisionFp32 = 1;
/// \}

/// Fused-kernel thread counts per family.
/// \{
inline constexpr int kEriCudaFusedThreadsV0 = 128;
inline constexpr int kEriCudaFusedThreadsV1 = 256;
/// \}

/// Dynamic shared-memory budget of the fused kernels (44 KB: 48 KB minus
/// the 4 KB the driver reserves; opt-in is not needed).
inline constexpr int kEriCudaMaxSharedBytes = 45056;

/// Highest class angular momentum instantiated: lKet <= 2*cudaLmax and
/// lBra <= 2*cudaLmax (the union matrix of eri_cuda.cpp); 12 at cudaLmax=6.
inline constexpr int kEriCudaMaxClassL = 12;

/// The certified seed/roundoff epsilon (md_vrr.hpp kCertifiedEpsilon): the
/// BoysBatchF32 absolute tolerance and the fp32 roundoff bound of the
/// transform terms.
inline constexpr double kEriCudaCertifiedEpsilon = 1e-7;

/// Boys table capacity (boys_coefficients.hpp): 12 pieces per order row,
/// 2304 coefficients total, 33 order rows, 24 region-B coefficients.
/// \{
inline constexpr int kEriCudaMaxPieces = 12;
inline constexpr int kEriCudaMaxCoeffs = 2304;
inline constexpr int kEriCudaMaxBoysRows = 33;
inline constexpr int kEriCudaMaxBcoeffs = 24;

/// \}

/// The flattened Boys tables, one copy of each lane (the CPU constants of
/// boys_coefficients.hpp; the upload in eri_cuda.cpp flattens per-order rows
/// with a running offset exactly like boys_cuda.cu). Read from global
/// memory through the kernel arguments - a deviation from boys_cuda.cu's
/// __constant__ tables, chosen so the struct travels with the EriCudaClassArgs.
struct EriCudaBoysTables {
    double coeffs64[kEriCudaMaxCoeffs];
    int offset64[kEriCudaMaxBoysRows][kEriCudaMaxPieces];
    double a64[kEriCudaMaxBoysRows][kEriCudaMaxPieces];
    double b64[kEriCudaMaxBoysRows][kEriCudaMaxPieces];
    int deg64[kEriCudaMaxBoysRows][kEriCudaMaxPieces];
    int count64[kEriCudaMaxBoysRows];
    double bcoeffs64[kEriCudaMaxBcoeffs];
    int bDeg64;
    float coeffs32[kEriCudaMaxCoeffs];
    int offset32[kEriCudaMaxBoysRows][kEriCudaMaxPieces];
    float a32[kEriCudaMaxBoysRows][kEriCudaMaxPieces];
    float b32[kEriCudaMaxBoysRows][kEriCudaMaxPieces];
    int deg32[kEriCudaMaxBoysRows][kEriCudaMaxPieces];
    int count32[kEriCudaMaxBoysRows];
    float bcoeffs32[kEriCudaMaxBcoeffs];
    int bDeg32;
};

/// Compact per-pair data of the device (the md_batch.hpp MdPairData fields
/// the kernels read, flattened to contiguous tables uploaded at Create).
struct EriCudaPairMeta {
    int la; ///< Angular momentum of shell i.
    int lb; ///< Angular momentum of shell j.
    int nFuncs; ///< Function rows of the pair block.
    int rowPairs; ///< Contraction row pairs (rows_i * rows_j).
    double braRowSum; ///< Certified-bound helper (max |T_bra| row sum).
    double ketColSum; ///< Certified-bound helper (max |T_ket| column sum, over primitives).
    std::size_t primOffset; ///< Element offset into primData (6 doubles per primitive).
    int primCount; ///< Number of primitive pairs.
    std::size_t braTransformOffset; ///< Element offset into the full 4D braTransforms.
    int braTransformK; ///< rowPairs * primCount * Hermite3DCount(la + lb) - the GEMM k dimension.
    std::size_t ketTransformOffset; ///< Element offset into ketTransforms.
};

/// One quartet task (mirrors MdQuartetTask).
struct EriCudaTask {
    std::size_t braPair;
    std::size_t ketPair;
    std::size_t outputOffset;
};

struct EriCudaClassArgs;

/// The per-class dispatch entry filled by the instantiation TUs.
struct EriCudaClassDispatchEntry {
    int family;
    int (*launchFused)(EriCudaClassArgs&);
    int (*launchVrr)(EriCudaClassArgs&);
    int (*launchBound)(EriCudaClassArgs&);
};

/// The per-class launch arguments. Every pointer refers to a DeviceBuffer
/// allocation (memory module); stream is a cudaStream_t passed as void*.
struct EriCudaClassArgs {
    const EriCudaBoysTables* boys;
    const EriCudaPairMeta* pairMeta;
    const double* primData;
    const double* braTransforms;
    const double* ketTransforms;
    const float* braTransformsF32;
    const float* ketTransformsF32;
    const EriCudaTask* tasks;
    std::size_t boundsBase;
    double* outF64;
    float* outF32;
    double* errorBounds;
    void* pq;
    void* acc;
    void* bound;
    int maxNFuncsKet;
    int maxRowPairs;
    double certifiedScale;
    int chunkRows;
    int threads;
    int precision;
    int vrrG;
    int runStart;
    int runSize;
    int vrrMaxNBraPrims;
    const int* chunkRanges;
    int nChunks;
    void* stream;
};

} // namespace qcx::integrals::internal::cuda

extern "C" {
/// Looks up the dispatch entry of one class (the generated table
/// eri_cuda_dispatch_gen.hpp, included by the registry TU below).
/// \returns The entry, or nullptr when the class is not instantiated.
const qcx::integrals::internal::cuda::EriCudaClassDispatchEntry* QcxEriCudaFindClass(int lBra,
                                                                                     int lKet);
} // extern "C"

// ---------------------------------------------------------------------------
// Part 2: device code (nvcc only)
// ---------------------------------------------------------------------------
#if defined(__CUDACC__)

#include <math.h>
#include <type_traits>

namespace qcx::integrals::internal::cuda {
namespace {

// The Boys region boundaries and constants (boys.cpp; the f32 lane needs the
// double constants for the region-B seed divisor).
__device__ constexpr double kEriCudaX0d = 1.18998481521084840e+01;
__device__ constexpr double kEriCudaX1d = 2.89893377388207400e+01;
__device__ constexpr double kEriCudaHalfSqrtPi = 0.886226925452758014; // sqrt(pi)/2
__device__ constexpr float kEriCudaHalfSqrtPiF32 = 0.88622693f;
__device__ constexpr double kEriCudaPi = 3.14159265358979323846264338327950288;

/// Number of 3D-Hermite functions with tx + ty + tz <= total (md_defs.hpp).
/// Host-callable: the launchers evaluate the class layout in constexpr
/// contexts before the device launch.
__host__ __device__ __forceinline__ constexpr int DevHermite3DCount(int total) {
    return (total + 1) * (total + 2) * (total + 3) / 6;
}

/// Start of tier n (|t| = n) in the flat slice layout (md_defs.hpp
/// kH2Prefix): the count of 3D-Hermite functions with total < n.
__host__ __device__ __forceinline__ constexpr int DevH2PrefixValue(int n) {
    return n == 0 ? 0 : DevHermite3DCount(n - 1);
}

/// Sub-index of (ty, tz) within a fixed-total tier (md_defs.hpp SubIndex3).
__device__ __forceinline__ constexpr int DevSubIndex3(int ty, int tz, int total) {
    return ty * (total + 1) - ty * (ty - 1) / 2 + tz;
}

/// Flat 3D-Hermite index (md_defs.hpp Hermite3DIndex).
__device__ __forceinline__ constexpr int DevHermite3DIndex(int tx, int ty, int tz) {
    const int n = tx + ty + tz;
    return DevH2PrefixValue(n) + DevSubIndex3(ty, tz, n);
}

/// Split Clenshaw evaluation (boys.cpp ClenshawSplit): the even/odd split
/// with the D-family recurrence, __fma_rn for std::fma, and the deg == 2
/// special the even/odd split assumes away. The generator emits only even
/// deg >= 4 beyond that.
__device__ __forceinline__ double DevClenshawSplit64(const double* c, int deg, double t) {
    if (deg == 0)
    {
        return c[0];
    }

    if (deg == 1)
    {
        return __fma_rn(t, c[1], c[0]);
    }

    const double v = __fma_rn(2.0, t * t, -1.0);
    const double twoV = v + v;

    if (deg == 2)
    {
        // T_2(t) = 2t^2 - 1 = v; the even/odd split below assumes deg >= 4.
        return __fma_rn(t, c[1], __fma_rn(v, c[2], c[0]));
    }

    const int m = deg / 2;
    double b1 = c[2 * m];
    double b2 = 0.0;

    for (int k = m - 1; k >= 1; --k)
    {
        const double b0 = __fma_rn(twoV, b1, c[2 * k] - b2);
        b2 = b1;
        b1 = b0;
    }

    const double even = __fma_rn(v, b1, c[0] - b2);
    double o1 = c[2 * m - 1];
    double o2 = 0.0;

    for (int k = m - 2; k >= 1; --k)
    {
        const double o0 = __fma_rn(twoV, o1, c[2 * k + 1] - o2);
        o2 = o1;
        o1 = o0;
    }

    const double odd = __fma_rn(twoV - 1.0, o1, c[1] - o2);
    return __fma_rn(t, odd, even);
}

/// Float-lane split Clenshaw (boys.cpp ClenshawSplitF32).
__device__ __forceinline__ float DevClenshawSplit32(const float* c, int deg, float t) {
    if (deg == 0)
    {
        return c[0];
    }

    if (deg == 1)
    {
        return __fmaf_rn(t, c[1], c[0]);
    }

    const float v = __fmaf_rn(2.0f, t * t, -1.0f);
    const float twoV = v + v;

    if (deg == 2)
    {
        return __fmaf_rn(t, c[1], __fmaf_rn(v, c[2], c[0]));
    }

    const int m = deg / 2;
    float b1 = c[2 * m];
    float b2 = 0.0f;

    for (int k = m - 1; k >= 1; --k)
    {
        const float b0 = __fmaf_rn(twoV, b1, c[2 * k] - b2);
        b2 = b1;
        b1 = b0;
    }

    const float even = __fmaf_rn(v, b1, c[0] - b2);
    float o1 = c[2 * m - 1];
    float o2 = 0.0f;

    for (int k = m - 2; k >= 1; --k)
    {
        const float o0 = __fmaf_rn(twoV, o1, c[2 * k + 1] - o2);
        o2 = o1;
        o1 = o0;
    }

    const float odd = __fmaf_rn(twoV - 1.0f, o1, c[1] - o2);
    return __fmaf_rn(t, odd, even);
}

/// Region-A seed F_order(x) (boys.cpp ChebyshevValue): the per-order piece
/// lookup then the split Clenshaw.
__device__ __forceinline__ double DevChebyshevValue64(const EriCudaBoysTables& tables,
                                                      int order,
                                                      double x) {
    const int last = tables.count64[order];
    int p = last - 1;

    for (int i = 0; i < last - 1; ++i)
    {
        if (x < tables.b64[order][i])
        {
            p = i;
            break;
        }
    }

    const double* c = tables.coeffs64 + tables.offset64[order][p];
    const double t =
        2.0 * (x - tables.a64[order][p]) / (tables.b64[order][p] - tables.a64[order][p]) - 1.0;
    return DevClenshawSplit64(c, tables.deg64[order][p], t);
}

/// Float-lane region-A seed (boys.cpp ChebyshevValueF32).
__device__ __forceinline__ float DevChebyshevValue32(const EriCudaBoysTables& tables,
                                                     int order,
                                                     float x) {
    const int last = tables.count32[order];
    int p = last - 1;

    for (int i = 0; i < last - 1; ++i)
    {
        if (x < tables.b32[order][i])
        {
            p = i;
            break;
        }
    }

    const float* c = tables.coeffs32 + tables.offset32[order][p];
    const float t =
        2.0f * (x - tables.a32[order][p]) / (tables.b32[order][p] - tables.a32[order][p]) - 1.0f;
    return DevClenshawSplit32(c, tables.deg32[order][p], t);
}

/// Region-B seed F_0(x) on [x0, x1) (boys.cpp RegionBSeed).
__device__ __forceinline__ double DevRegionBSeed64(const EriCudaBoysTables& tables, double x) {
    const double t = 2.0 * (x - kEriCudaX0d) / (kEriCudaX1d - kEriCudaX0d) - 1.0;
    return DevClenshawSplit64(tables.bcoeffs64, tables.bDeg64, t);
}

/// Float-lane region-B seed (boys.cpp RegionBSeedF32): the divisor is the
/// DOUBLE difference cast to float - the CPU casts kX1 - kX0, not each bound.
__device__ __forceinline__ float DevRegionBSeed32(const EriCudaBoysTables& tables, float x) {
    const float t = 2.0f * (x - static_cast<float>(kEriCudaX0d)) /
                        static_cast<float>(kEriCudaX1d - kEriCudaX0d) -
                    1.0f;
    return DevClenshawSplit32(tables.bcoeffs32, tables.bDeg32, t);
}

/// The Boys batch of order 0..nmax (boys.cpp BoysBatch, verbatim: division
/// in region C - not rsqrt; region-B upward recursion; region-A downward
/// recursion with exp(-x) via exp).
template <typename T>
__device__ __forceinline__ void DevBoysBatch(const EriCudaBoysTables& tables,
                                             int nmax,
                                             double x,
                                             T* out);

template <>
__device__ __forceinline__ void DevBoysBatch<double>(const EriCudaBoysTables& tables,
                                                     int nmax,
                                                     double x,
                                                     double* out) {
    if (x == 0.0)
    {
        for (int l = 0; l <= nmax; ++l)
        {
            out[l] = 1.0 / (2.0 * l + 1.0);
        }

        return;
    }

    if (x < kEriCudaX0d)
    {
        double f = DevChebyshevValue64(tables, nmax, x);
        out[nmax] = f;
        const double expx = 0.5 * exp(-x);

        for (int l = nmax - 1; l >= 0; --l)
        {
            f = (x * f + expx) / (l + 0.5);
            out[l] = f;
        }

        return;
    }

    if (x < kEriCudaX1d)
    {
        double f = DevRegionBSeed64(tables, x);
        out[0] = f;
        const double expx = 0.5 * exp(-x);

        for (int l = 1; l <= nmax; ++l)
        {
            f = ((l - 0.5) * f - expx) / x;
            out[l] = f;
        }

        return;
    }

    double f = kEriCudaHalfSqrtPi / sqrt(x);
    out[0] = f;

    for (int l = 1; l <= nmax; ++l)
    {
        f = (l - 0.5) * f / x;
        out[l] = f;
    }
}

template <>
__device__ __forceinline__ void DevBoysBatch<float>(const EriCudaBoysTables& tables,
                                                    int nmax,
                                                    double x,
                                                    float* out) {
    if (x == 0.0)
    {
        for (int l = 0; l <= nmax; ++l)
        {
            out[l] = 1.0f / (2.0f * static_cast<float>(l) + 1.0f);
        }

        return;
    }

    const float xx = static_cast<float>(x);
    const float x0 = static_cast<float>(kEriCudaX0d);
    const float x1 = static_cast<float>(kEriCudaX1d);

    if (xx < x0)
    {
        // The seed must be double precision: the downward recursion amplifies
        // a float seed error far beyond the certified 1.5e-7 float budget
        // (boys.cpp BoysBatchF32). The recursion itself stays in float.
        const double seed = DevChebyshevValue64(tables, nmax, x);
        out[nmax] = static_cast<float>(seed);
        float f = out[nmax];
        // expf, not __expf: the downward recursion amplifies the e^{-x}
        // rounding error (the boys_cuda.cu precedent).
        const float expx = 0.5f * expf(-xx);

        for (int l = nmax - 1; l >= 0; --l)
        {
            f = (xx * f + expx) / (static_cast<float>(l) + 0.5f);
            out[l] = f;
        }

        return;
    }

    float f = DevRegionBSeed32(tables, xx);
    out[0] = f;

    if (xx < x1)
    {
        const float expx = 0.5f * expf(-xx);

        for (int l = 1; l <= nmax; ++l)
        {
            f = ((static_cast<float>(l) - 0.5f) * f - expx) / xx;
            out[l] = f;
        }

        return;
    }

    f = kEriCudaHalfSqrtPiF32 / sqrtf(xx);
    out[0] = f;

    for (int l = 1; l <= nmax; ++l)
    {
        f = (static_cast<float>(l) - 0.5f) * f / xx;
        out[l] = f;
    }
}

/// One primitive quadruple of one task: the Boys seeds, the slice triangle
/// (rotating 3-tier buffer - tier n of the current slice lives at rotation
/// n % 3, and tier 0 always at rotation 0; tier m's last reader is tier m+2
/// of the same slice, so the overwrite at rotation 0 by tier 3/6/9 is safe),
/// the prevSlice retain (the second VRR source of slice T + 1 lives in
/// slice T's tiers; each tier is captured while its rotation is still
/// intact - tier n - 2 at the bottom of tier n, the last two in the slice
/// tail - the end-of-slice copy from the rotations captured the stale
/// tier t + 3 for t <= T - 3, the L >= 4 wrong-output bug fixed
/// 2026-08-23), and the CWR-part-1 scatter into the pq block. Mirrors
/// detail::RunVrrQuadruple (md_vrr.hpp) exactly, including the certified
/// bound accumulation. The CPU's out-of-range kH2Prefix[n-2] read at n == 1
/// (undefined but never dereferenced) is not reproduced: the second source
/// term is guarded inside the tx/ty/tz >= 2 branches (identical values).
///
/// The scatter is UNWEIGHTED and per (rowAB, primB) slot, exactly like the
/// CPU's per-primitive pq rows (md_vrr.hpp: the row index is
/// (rowAB * nPrimBra + primIdx)): the bra contraction weights live in the
/// full 4D bra transform, not here (the old
/// weight-folded scatter plus the b=0-only replicated transform was exact
/// only for the s-class, where the angular fold is 1).
///
/// \param pqPrimStride The pq stride between contraction-row slots: the
/// number of primB slots per rowAB row of the pq block (the fused kernels
/// own one (rowAB, primB) slot per thread, so their band is a single slot
/// and pass 1; V2's per-task block is padded to the run's vrrMaxNBraPrims).
/// \param rowPairs Number of contraction row pairs the scatter covers (1
/// when each thread owns one (rowAB, primB) slot - the fused kernels; the
/// task's row pairs in V2).
/// \param primB The bra primitive whose primData block this thread reads.
/// \param primG The ket primitive index (primData offset and, in the fused
/// kernels, the per-g loop of the thread's pq band).
/// \param scatterSlot The slot's primB index inside its rowAB row of the pq
/// block. For V2 it equals primB (thread (t, b) owns column b); the fused
/// kernels own exactly one slot per thread, whose slot index within the
/// row is 0 - passing primB there would write primB slots past the
/// thread's band (the per-primitive band restructure bug, fixed 2026-08-22).
/// \param pq The pq block (fused: this thread's single-slot band; V2: the
/// task's padded block).
/// \param bound The certified-bound accumulator (per task per g); V2
/// scatters one thread per bra primitive, the fused kernels one thread per
/// (rowAB, primB) slot with the rowAB == 0 slots carrying the task's sum.
/// \tparam AtomicBound V2 passes true - the (task, b) threads of the per-g
/// launches all accumulate into the same run-level bound slot, so the
/// contribution must be an atomicAdd; the fused kernels pass false and hand
/// each thread its own local accumulator.
template <int L, int LBra, int LKet, typename T, bool AtomicBound>
__device__ __forceinline__ void DevVrrQuadruple(const EriCudaBoysTables& tables,
                                                const EriCudaPairMeta& bra,
                                                const EriCudaPairMeta& ket,
                                                const double* primData,
                                                int pqPrimStride,
                                                int rowPairs,
                                                int primB,
                                                int primG,
                                                int scatterSlot,
                                                T* pq,
                                                double* bound) {
    constexpr int kMaxTier = (L + 1) * (L + 2) / 2;
    constexpr int kHermBra = DevHermite3DCount(LBra);
    constexpr int kHermKet = DevHermite3DCount(LKet);
    constexpr int kRowStride = kHermBra * kHermKet;
    // primOffset is the ELEMENT offset of the pair's primitive block in the
    // flat primData (6 doubles per primitive; the host sets it to
    // tables.primData.size() - eri_cuda.cpp BuildHostTables). The old
    // (primOffset + primB) * 6 double-counted the per-prim stride: every
    // non-first pair read 6x past its block (past the table for the last
    // pair) - garbage exponents and centers, NaN seeds for any x > 0 task
    // (the x == 0 one-center case never exercised the data).
    const double* const primBData = primData + bra.primOffset + primB * 6;
    const double* const primGData = primData + ket.primOffset + primG * 6;
    const double p = primBData[0];
    const double q = primGData[0];
    const double alpha = p * q / (p + q);
    const double dqx = primBData[3] - primGData[3];
    const double dqy = primBData[4] - primGData[4];
    const double dqz = primBData[5] - primGData[5];
    const double x = alpha * (dqx * dqx + dqy * dqy + dqz * dqz);
    // K = 2 pi^(5/2) / (p q sqrt(p+q)) * E_ab * E_cd (the device pow may
    // differ from MSVC by 1-2 ulp - covered by the parity tolerances).
    const double prefactor =
        2.0 * pow(kEriCudaPi, 2.5) / (p * q * sqrt(p + q)) * primBData[2] * primGData[2];

    T seeds[L + 1];
    DevBoysBatch<T>(tables, L, x, seeds);
    double seedSum = 0.0;

    for (int m = 0; m <= L; ++m)
    {
        seeds[m] *= static_cast<T>(prefactor);
        seedSum += fabs(static_cast<double>(seeds[m])) + kEriCudaCertifiedEpsilon;
    }

    if (bound != nullptr)
    {
        double growth = 1.0;

        if constexpr (L > 0)
        {
            growth = pow(2.0 * sqrt(alpha), L);
            growth = growth > 1.0 ? growth : 1.0;
        }

        if constexpr (AtomicBound)
        {
            atomicAdd(bound, growth * seedSum);
        } else
        {
            *bound += growth * seedSum;
        }
    }

    const T c2 = static_cast<T>(-2.0 * alpha);
    const T c1x = static_cast<T>(-2.0 * alpha * dqx);
    const T c1y = static_cast<T>(-2.0 * alpha * dqy);
    const T c1z = static_cast<T>(-2.0 * alpha * dqz);

    // The rotating 3-tier slice buffer (see the comment above). kMaxTier is
    // the largest tier: (L+1)(L+2)/2 entries. The previous slice lives in
    // prevSlice (slot-per-tier): the second VRR source [t-2e]^(m+1) is in
    // slice T-1, and the rotation alone overwrites it (the CPU ping-pong fix).
    T slice[3 * kMaxTier];
    T prevSlice[(L + 1) * kMaxTier];

    // The tier-0 scatter: the seed [0]^(0) is the (0,0,0) Hermite, the only
    // element of slice 0, written to the (rowAB, primB) pq slot per
    // contraction row pair (the CPU's per-primitive seed write, md_vrr.hpp).
    // Every slot receives exactly one contribution per (primB, primG), so
    // the store order is unobservable and no atomics are needed.
    for (int rowAB = 0; rowAB < rowPairs; ++rowAB)
    {
        pq[(rowAB * pqPrimStride + scatterSlot) * kRowStride] += seeds[0];
    }

    for (int sliceT = 1; sliceT <= L; ++sliceT)
    {
        // Tier 0 of slice T is the seed [0]^(T); tier 0 always lives at
        // rotation 0 (tier 3/6/9 overwrite it - safe, see above). The
        // seed's retain for slice T + 1 is captured from rotation 0 at
        // the bottom of tier 2 (or in the slice tail when T == 1), before
        // tier 3 overwrites the slot - no register staging needed.
        slice[0] = seeds[sliceT];

        for (int n = 1; n <= sliceT; ++n)
        {
            const int rotN = (n % 3) * kMaxTier;
            const int rotN1 = ((n - 1) % 3) * kMaxTier;

            for (int ty = 0; ty <= n; ++ty)
            {
                for (int tz = 0; tz <= n - ty; ++tz)
                {
                    const int tx = n - ty - tz;
                    const int sub = DevSubIndex3(ty, tz, n);
                    T value;

                    if (tx >= 1)
                    {
                        value = c1x * slice[rotN1 + DevSubIndex3(ty, tz, n - 1)];

                        if (tx >= 2)
                        {
                            value += c2 * static_cast<T>(tx - 1) *
                                     prevSlice[(n - 2) * kMaxTier + DevSubIndex3(ty, tz, n - 2)];
                        }
                    } else if (ty >= 1)
                    {
                        value = c1y * slice[rotN1 + DevSubIndex3(ty - 1, tz, n - 1)];

                        if (ty >= 2)
                        {
                            value +=
                                c2 * static_cast<T>(ty - 1) *
                                prevSlice[(n - 2) * kMaxTier + DevSubIndex3(ty - 2, tz, n - 2)];
                        }
                    } else
                    {
                        value = c1z * slice[rotN1 + DevSubIndex3(ty, tz - 1, n - 1)];

                        if (tz >= 2)
                        {
                            value +=
                                c2 * static_cast<T>(tz - 1) *
                                prevSlice[(n - 2) * kMaxTier + DevSubIndex3(ty, tz - 2, n - 2)];
                        }
                    }

                    slice[rotN + sub] = value;
                }
            }

            // Capture tier n - 2 for slice T + 1 (the second VRR source of
            // slice T + 1 lives in slice T's tiers - the previous-slice
            // fix). Its rotation is still intact here - tier n + 1 is the
            // first to overwrite it - and its last reader in this slice
            // (tier n's c2 term) has already finished, so the capture
            // cannot corrupt the current slice's reads (the immediate
            // per-tier retain was wrong: it fed the current slice's c2
            // reads with the current slice's tiers, fixed 2026-08-23).
            if (n >= 2)
            {
                const int count = (n - 1) * n / 2;
                const int rotN2 = ((n - 2) % 3) * kMaxTier;

                for (int i = 0; i < count; ++i)
                {
                    prevSlice[(n - 2) * kMaxTier + i] = slice[rotN2 + i];
                }
            }
        }

        // Scatter tier sliceT into pq: [p~|q~] = [r~]^(0) for every split
        // p~ + q~ = r~ with |p~| <= LBra and |q~| <= LKet, weighted per
        // contraction row pair (CWR part 1).
        const int pLo = sliceT - LKet > 0 ? sliceT - LKet : 0;
        const int pHi = LBra < sliceT ? LBra : sliceT;

        for (int ty = 0; ty <= sliceT; ++ty)
        {
            for (int tz = 0; tz <= sliceT - ty; ++tz)
            {
                const int tx = sliceT - ty - tz;
                const T value = slice[(sliceT % 3) * kMaxTier + DevSubIndex3(ty, tz, sliceT)];

                for (int ps = pLo; ps <= pHi; ++ps)
                {
                    for (int px = 0; px <= ps; ++px)
                    {
                        for (int py = 0; py <= ps - px; ++py)
                        {
                            const int pz = ps - px - py;
                            const int qx = tx - px;
                            const int qy = ty - py;
                            const int qz = tz - pz;

                            if (qx < 0 || qy < 0 || qz < 0)
                            {
                                continue;
                            }

                            const int pqBase = DevHermite3DIndex(px, py, pz) * kHermKet +
                                               DevHermite3DIndex(qx, qy, qz);

                            for (int rowAB = 0; rowAB < rowPairs; ++rowAB)
                            {
                                pq[(rowAB * pqPrimStride + scatterSlot) * kRowStride + pqBase] +=
                                    value;
                            }
                        }
                    }
                }
            }
        }

        // Retain the last two tiers (sliceT - 1, sliceT) for slice T + 1:
        // their rotations are intact - no later tier of this slice
        // overwrites them (the end-of-slice loop that copied every tier
        // from its rotation captured the stale tier t + 3 for t <= T - 3,
        // the L >= 4 wrong-output bug, fixed 2026-08-23). For sliceT == 1
        // this also retains tier 0 from rotation 0, which still holds the
        // staged seed (tier 3 does not exist in this slice).
        for (int t = sliceT - 1; t <= sliceT; ++t)
        {
            const int count = (t + 1) * (t + 2) / 2;

            for (int i = 0; i < count; ++i)
            {
                prevSlice[t * kMaxTier + i] = slice[(t % 3) * kMaxTier + i];
            }
        }
    }
}

/// The fused kernel (variants V0/V1): one block owns one task-range chunk
/// (chunkRanges, host-partitioned so the block's task slot sum stays within
/// chunkRows); the threads map to the (rowAB, primB) slots of the current
/// task's flattened slot space (rowAB-major: slot = rowAB * nPrimBra + b),
/// and tasks with more slots than chunkRows are split by the task-local
/// chunk loop. Per chunk: each thread's acc row is zeroed once, then per
/// ket primitive its pq slot is zeroed and the VRR scatters in (weight-free:
/// the bra contraction weights live in the full
/// 4D bra transform), the ket transform accumulates in the exact
/// MicroGemmAdd order (md_gemm.hpp), and the bra transform runs as a
/// per-element partial sum in globally k-ascending order - the exact
/// MicroGemm order, accumulated chunk by chunk so the block owns each task
/// completely (no cross-block races).
///
/// The certified bound: the bound of a (primB, primG) quadruple is
/// rowAB-independent (it counts the VRR seed sums only), and the flattened
/// rowAB-major slot order puts every primB's rowAB == 0 slot inside the
/// first nPrimBra slots - so the rowAB == 0 slots contribute the task's
/// full g- and b-summed bound exactly once across the chunks. The per-chunk
/// slot bounds are staged in shared memory (boundSh) and summed by tid 0
/// into the task's running total; the last chunk finalizes errorBounds.
template <int LBra, int LKet, typename T>
__global__ void EriFusedKernel(const EriCudaClassArgs args) {
    constexpr int L = LBra + LKet;
    constexpr int kHermBra = DevHermite3DCount(LBra);
    constexpr int kHermKet = DevHermite3DCount(LKet);
    constexpr bool kCertified = std::is_same_v<T, float>;
    const int tid = static_cast<int>(threadIdx.x);
    const int block = static_cast<int>(blockIdx.x);
    const int rangeStart = args.chunkRanges[2 * block];
    const int rangeEnd = args.chunkRanges[2 * block + 1];

    const T* braT;
    const T* ketT;

    if constexpr (std::is_same_v<T, double>)
    {
        braT = args.braTransforms;
        ketT = args.ketTransforms;
    } else
    {
        braT = args.braTransformsF32;
        ketT = args.ketTransformsF32;
    }

    // The shared layout: the per-chunk slot bounds first (double-aligned),
    // then the per-slot pq bands and acc rows (T-aligned; the fp32 lane's
    // T array after the double block stays 4-byte aligned).
    extern __shared__ char sharedRaw[];
    double* const boundSh = reinterpret_cast<double*>(sharedRaw);
    T* const pqSh = reinterpret_cast<T*>(sharedRaw + args.chunkRows * sizeof(double));
    T* const accSh = pqSh + args.chunkRows * kHermBra * kHermKet;

    for (int t = rangeStart; t < rangeEnd; ++t)
    {
        const EriCudaTask& task = args.tasks[t];
        const EriCudaPairMeta& bra = args.pairMeta[task.braPair];
        const EriCudaPairMeta& ket = args.pairMeta[task.ketPair];
        const int nPrimBra = bra.primCount;
        const int totalSlots = bra.rowPairs * nPrimBra;
        const int nFk = ket.nFuncs;
        const int kBra = bra.braTransformK;
        const int mBra = bra.nFuncs;
        const int elems = mBra * nFk;
        double boundTotal = 0.0;
        T* outBlock;

        if constexpr (kCertified)
        {
            outBlock = args.outF32 + task.outputOffset;
        } else
        {
            outBlock = args.outF64 + task.outputOffset;
        }

        for (int chunkStart = 0; chunkStart < totalSlots; chunkStart += args.chunkRows)
        {
            const int slotsInChunk =
                args.chunkRows < totalSlots - chunkStart ? args.chunkRows : totalSlots - chunkStart;

            if (tid < slotsInChunk)
            {
                const int slot = chunkStart + tid;
                const int rowAB = slot / nPrimBra;
                const int primB = slot % nPrimBra;

                // The acc row of this thread's slot: zeroed once per chunk
                // (the CPU relies on fresh thread-local scratch; the device
                // must be explicit - md_vrr.hpp's fill covers only the pq
                // block).
                T* const accRow = accSh + tid * kHermBra * args.maxNFuncsKet;

                for (int w = 0; w < kHermBra * args.maxNFuncsKet; ++w)
                {
                    accRow[w] = T{};
                }

                T* const pqBand = pqSh + tid * kHermBra * kHermKet;
                double boundAcc = 0.0;

                for (int g = 0; g < ket.primCount; ++g)
                {
                    for (int w = 0; w < kHermBra * kHermKet; ++w)
                    {
                        pqBand[w] = T{};
                    }

                    // The bound accumulates over all g, exactly like the
                    // CPU's taskBound[t] (md_vrr.hpp: the last-pair-only
                    // overwrite dropped 8 of 9 STO-3G pairs from the
                    // a-priori bound - fixed on the CPU 2026-08-22, mirrored
                    // here). Zeroed once per task before the g loop.
                    DevVrrQuadruple<L, LBra, LKet, T, false>(*args.boys,
                                                             bra,
                                                             ket,
                                                             args.primData,
                                                             1,
                                                             1,
                                                             primB,
                                                             g,
                                                             0,
                                                             pqBand,
                                                             kCertified ? &boundAcc : nullptr);

                    // The ket transform: MicroGemmAdd order (j outer, l
                    // middle, i inner), accumulating into the acc rows. The
                    // transform of ket primitive g lives at
                    // ketTransformOffset + g * (kHermKet * nFk): the flat
                    // table is [pair][g][t * nF + f] (the old global-base
                    // read ignored the pair and the primitive).
                    const T* const ketTg = ketT + ket.ketTransformOffset + g * (kHermKet * nFk);

                    for (int j = 0; j < nFk; ++j)
                    {
                        for (int l = 0; l < kHermKet; ++l)
                        {
                            const T bValue = ketTg[l * nFk + j];

                            for (int i = 0; i < kHermBra; ++i)
                            {
                                accRow[i * args.maxNFuncsKet + j] +=
                                    pqBand[i * kHermKet + l] * bValue;
                            }
                        }
                    }
                }

                // The bound is rowAB-independent: only the rowAB == 0 slots
                // carry it, and the rowAB-major slot order covers every
                // primB exactly once (slots [0, nPrimBra)); later chunks'
                // rowAB >= 1 slots contribute zero.
                boundSh[tid] = kCertified && rowAB == 0 ? boundAcc : 0.0;
            }

            __syncthreads();

            if constexpr (kCertified)
            {
                if (tid == 0)
                {
                    for (int s = 0; s < slotsInChunk; ++s)
                    {
                        boundTotal += boundSh[s];
                    }
                }
            }

            // The bra transform: a per-element partial sum over this chunk's
            // k range in global k-ascending order (MicroGemm order), starting
            // from the global partial of the earlier chunks. The element
            // distribution stride is slotsInChunk (not the thread count) so
            // the last chunk's residues are covered exactly once. The k loop
            // walks the chunk's slots in order - the fixed form of the old
            // (l - chunkStart * kHermBra) index, which went negative for
            // chunkStart >= 1 and never covered the chunk's rows exactly.
            if (tid < slotsInChunk)
            {
                // The pair's full 4D bra transform base (the old
                // global-base read ignored braTransformOffset - wrong for
                // any non-first bra pair).
                const T* const braTp = braT + bra.braTransformOffset;

                for (int idx = tid; idx < elems; idx += slotsInChunk)
                {
                    const int f = idx / nFk;
                    const int f2 = idx % nFk;
                    T value = chunkStart == 0 ? T{} : outBlock[idx];

                    for (int s = 0; s < slotsInChunk; ++s)
                    {
                        const int kGlobal = (chunkStart + s) * kHermBra;

                        for (int l = 0; l < kHermBra; ++l)
                        {
                            value += braTp[f * kBra + kGlobal + l] *
                                     accSh[(s * kHermBra + l) * args.maxNFuncsKet + f2];
                        }
                    }

                    outBlock[idx] = value;
                }
            }

            __syncthreads();
        }

        if constexpr (kCertified)
        {
            if (tid == 0)
            {
                // The full g- and b-summed bound (md_vrr.hpp's taskBound[t]).
                args.errorBounds[args.boundsBase + t] =
                    args.certifiedScale * bra.braRowSum * ket.ketColSum * boundTotal;
            }
        }
    }
}

/// The global-memory VRR kernel (variant V2): one thread per (task, bra
/// primitive) of the run, the task's padded pq block per thread's column.
/// The run split by (ketPair, rowPairs, primCount) makes every task's
/// primCount exactly vrrMaxNBraPrims, so the block is hole-free: thread
/// (t, b) owns the (rowAB, b) slots of task t for every contraction row
/// pair, and the UNWEIGHTED scatter (the bra
/// contraction weights live in the full 4D bra transform, consumed by the
/// per-task bra GEMM) needs no atomics.
///
/// The certified bound accumulates atomically per (task, b) thread into the
/// task's run-level slot: the bound buffer is zeroed once per run
/// (cudaMemsetAsync in RunBatchOnDevice) and the per-g launches sum like the
/// CPU's taskBound[t] - the last-g-only gate dropped 8 of 9 STO-3G pairs
/// from the a-priori bound (mirrors the CPU fix of 2026-08-22).
template <int LBra, int LKet, typename T>
__global__ void EriVrrGlobalKernel(const EriCudaClassArgs args) {
    constexpr int L = LBra + LKet;
    constexpr int kHermBra = DevHermite3DCount(LBra);
    constexpr int kHermKet = DevHermite3DCount(LKet);
    constexpr bool kCertified = std::is_same_v<T, float>;
    const int tPrime = static_cast<int>(blockIdx.x * blockDim.x) + static_cast<int>(threadIdx.x);

    if (tPrime >= args.runSize * args.vrrMaxNBraPrims)
    {
        return;
    }

    const int t = args.runStart + tPrime / args.vrrMaxNBraPrims;
    const int b = tPrime % args.vrrMaxNBraPrims;
    const EriCudaTask& task = args.tasks[t];
    const EriCudaPairMeta& bra = args.pairMeta[task.braPair];
    const EriCudaPairMeta& ket = args.pairMeta[task.ketPair];

    // The hole-free pq block: mPq rows of kHermKet per task, the row
    // (rowAB, primB) Hermite index (rowAB * vrrMaxNBraPrims + b) * kHermBra
    // + t - the layout the per-task bra GEMM contracts against the pair's
    // verbatim 4D bra transform (both rowAB-major).
    const int rowPairs = args.maxRowPairs;
    const int mPq = rowPairs * args.vrrMaxNBraPrims * kHermBra;
    T* const pq = static_cast<T*>(args.pq) + (t - args.runStart) * mPq * kHermKet;
    double* bound = nullptr;

    if constexpr (kCertified)
    {
        bound = static_cast<double*>(args.bound) + (t - args.runStart);
    }

    DevVrrQuadruple<L, LBra, LKet, T, true>(*args.boys,
                                            bra,
                                            ket,
                                            args.primData,
                                            args.vrrMaxNBraPrims,
                                            rowPairs,
                                            b,
                                            args.vrrG,
                                            b,
                                            pq,
                                            bound);
}

/// The V2 bound finalize: errorBounds[boundsBase + t] =
/// certifiedScale * braRowSum * ketColSum * bound[t - runStart] (the CPU's
/// per-task bound assembly; not T-templated - the bound is a double).
__global__ void EriBoundFinalizeKernel(const EriCudaClassArgs args) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x) + static_cast<int>(threadIdx.x);

    if (i >= args.runSize)
    {
        return;
    }

    const int t = args.runStart + i;
    const EriCudaTask& task = args.tasks[t];
    const EriCudaPairMeta& bra = args.pairMeta[task.braPair];
    const EriCudaPairMeta& ket = args.pairMeta[task.ketPair];
    args.errorBounds[args.boundsBase + t] =
        args.certifiedScale * bra.braRowSum * ket.ketColSum * static_cast<double*>(args.bound)[i];
}

} // namespace

// ---------------------------------------------------------------------------
// Part 3: per-class launcher templates (nvcc only; wrapped in extern "C"
// functions by the instantiation TUs)
// ---------------------------------------------------------------------------

/// Validates the host-computed chunk geometry against the shared-memory
/// budget and launches the fused kernel. The chunkRows formula must agree
/// with eri_cuda.cpp's ComputeChunkRows: the per-row bytes are
/// kHermBra * (kHermKet + maxNFuncsKet) * sizeof(T) plus the per-chunk slot
/// bound block (chunkRows doubles), budget 45056. The thread count is
/// runtime (args.threads, 128/256): the kernel is thread-count-agnostic
/// (every slot guard is tid < slotsInChunk), so a single fused
/// instantiation per class serves both kV0 and kV1 - and the variant
/// override really changes the launch geometry.
template <int LBra, int LKet, typename T> int EriCudaLaunchFusedClass(EriCudaClassArgs& args) {
    constexpr int kHermBra = DevHermite3DCount(LBra);
    constexpr int kHermKet = DevHermite3DCount(LKet);
    const long long bytesPerRow =
        static_cast<long long>(kHermBra) * (kHermKet + args.maxNFuncsKet) * sizeof(T);
    const long long boundBytes = static_cast<long long>(args.chunkRows) * sizeof(double);
    const long long sharedBytes =
        boundBytes + static_cast<long long>(args.chunkRows) * bytesPerRow + 8;

    if (args.threads != kEriCudaFusedThreadsV0 && args.threads != kEriCudaFusedThreadsV1)
    {
        return kEriCudaInvalidFamily;
    }

    if (args.chunkRows < 1 || args.chunkRows > args.threads || args.nChunks < 1 ||
        sharedBytes > kEriCudaMaxSharedBytes)
    {
        return kEriCudaSharedOverflow;
    }

    EriFusedKernel<LBra, LKet, T><<<args.nChunks,
                                    args.threads,
                                    static_cast<std::size_t>(sharedBytes),
                                    static_cast<cudaStream_t>(args.stream)>>>(args);
    return static_cast<int>(cudaGetLastError());
}

/// Launches the V2 VRR kernel for one (run, g). Defense-in-depth:
/// the host never schedules an empty or degenerate run (eri_cuda.cpp
/// rejects empty batches before any launcher), but a zero task count must
/// not reach the launch config: grid 0 is an invalid launch (spurious
/// cudaErrorInvalidConfiguration) and the int product overflows for large
/// runs.
template <int LBra, int LKet, typename T> int EriCudaLaunchVrrClass(EriCudaClassArgs& args) {
    constexpr int kThreads = 256;

    if (args.runSize <= 0 || args.vrrMaxNBraPrims <= 0)
    {
        return kEriCudaOk;
    }

    // size_t arithmetic: the int product and the ceil division both
    // overflow (UB) near INT_MAX.
    const std::size_t total =
        static_cast<std::size_t>(args.runSize) * static_cast<std::size_t>(args.vrrMaxNBraPrims);
    const std::size_t blocks = (total + kThreads - 1) / kThreads;
    EriVrrGlobalKernel<LBra, LKet, T>
        <<<blocks, kThreads, 0, static_cast<cudaStream_t>(args.stream)>>>(args);
    return static_cast<int>(cudaGetLastError());
}

/// Launches the V2 bound finalize for one run. Declared here (this file is
/// header-included by every instantiation TU) and defined in Part 4, the
/// registry TU: a non-template launcher compiled into every TU would be a
/// duplicate definition at link time (the kernel itself is deliberately not
/// T-templated - the certified bound is always a double).
int EriCudaLaunchBoundClass(EriCudaClassArgs& args);

} // namespace qcx::integrals::internal::cuda
#endif // defined(__CUDACC__)

// ---------------------------------------------------------------------------
// Part 4: the class registry (one TU only: the .cu compiled by nvcc; the
// header-style includes define QCX_ERI_CUDA_HEADER_ONLY). The table itself
// is generated by integrals/CMakeLists.txt (eri_cuda_dispatch_gen.hpp): a
// static-link archive drops object files nothing references, so the table
// must reference every per-class launcher - the same keep-alive pattern as
// the CPU lane's generated md_dispatch_gen.hpp. No runtime registration.
// ---------------------------------------------------------------------------
#if !defined(QCX_ERI_CUDA_HEADER_ONLY)
#include "md_cuda_instantiate/eri_cuda_dispatch_gen.hpp"

namespace qcx::integrals::internal::cuda {

/// The single definition of the V2 bound-finalize launcher (Part 3 declares
/// it for the instantiation TUs; the registry TU owns the one definition).
int EriCudaLaunchBoundClass(EriCudaClassArgs& args) {
    constexpr int kThreads = 64;

    // The same defense as the VRR launcher: an empty run must not
    // reach the launch config, and the ceil division must not overflow.
    if (args.runSize <= 0)
    {
        return kEriCudaOk;
    }

    const std::size_t blocks = (static_cast<std::size_t>(args.runSize) + kThreads - 1) / kThreads;
    EriBoundFinalizeKernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(args.stream)>>>(args);
    return static_cast<int>(cudaGetLastError());
}

extern "C" const EriCudaClassDispatchEntry* QcxEriCudaFindClass(int lBra, int lKet) {
    if (lBra < 0 || lBra > kEriCudaMaxClassL || lKet < 0 || lKet > kEriCudaMaxClassL)
    {
        return nullptr;
    }

    const EriCudaClassDispatchEntry& entry = gClassTable[lBra][lKet];

    if (entry.launchFused == nullptr && entry.launchVrr == nullptr)
    {
        return nullptr;
    }

    return &entry;
}

} // namespace qcx::integrals::internal::cuda
#endif // !defined(QCX_ERI_CUDA_HEADER_ONLY)

#endif // QCX_ERI_CUDA_FILE_INCLUDED
