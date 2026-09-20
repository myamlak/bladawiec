// The generic GPU Fock contraction: the J/K accumulation
// kernel of the GpuJkFockBuilder, and its single extern "C" launcher. One
// kernel for EVERY class - the per-quartet geometry arrives per task in
// EriCudaFockTaskMeta (no per-class instantiation, unlike the ERI kernels
// of eri_cuda.cu), and the integral block is indexed with the packed
// EriBlockIndex layout of eri_batch.hpp:
//
//   EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)
//       = (fb * nA + fa) * (nC * nD) + (fd * nC + fc)
//
// The six accumulation phases are an exact elementwise port of the CPU
// FockContractor::AccumulateBlock (fock_build.cpp): J bra, J ket
// (skipped for diagonal quartets), and the four K targets, each with the
// same density arrangement reads, the same transpose writes, and the same
// kMultiplicity division. Every Fock write is an atomicAdd: within one
// quartet the phases can collide on one element (a diagonal quartet's J
// bra and K1 targets overlap), and across quartets the 8-fold symmetry
// maps distinct canonical quartets onto shared elements - plain writes
// would race. Hardware double atomicAdd is available on CC >= 6.0 (the
// T1000 is CC 7.5).
//
// C++20 and a CUDA-safe include list only: the only qcx header is
// the host-safe eri_cuda_fock.hpp contract.

#include "internal/eri_cuda_fock.hpp"

#include <cstddef>
#include <cuda_runtime.h>

namespace qcx::integrals::internal::cuda {
namespace {

// The packed-block offset of the (fa, fb | fc, fd) element - the single
// layout contract of eri_batch.hpp (EriBlockIndex), reimplemented device-
// side because eri_batch.hpp is a host header. The originally proposed
// formula ((fa * nB + fb) * nC + fc) * nD + fd was verified WRONG against
// the live implementation; the form below is the one every consumer must
// use.
__device__ std::size_t EriFockBlockIndex(std::size_t fa,
                                         std::size_t fb,
                                         std::size_t fc,
                                         std::size_t fd,
                                         std::size_t nA,
                                         std::size_t nB,
                                         std::size_t nC,
                                         std::size_t nD) {
    return (fb * nA + fa) * (nC * nD) + (fd * nC + fc);
}

// The element read through the once-per-thread lane pointers: the fp32
// lane widens on read (a float buffer cannot be interpreted as double), the
// fp64 lane reads directly. readF32 is null in the fp64 lane; the pointer
// selection itself happened once, before the phase loops.
__device__ double EriFockLaneRead(const double* readF64,
                                  const float* readF32,
                                  std::size_t blockIndex) {
    return readF32 != nullptr ? static_cast<double>(readF32[blockIndex]) : readF64[blockIndex];
}

// The Fock contraction of one batch of canonical quartets: one block per
// task, kThreads threads, each phase a strided slot loop over its (fa, fb)
// / (fc, fd) / ... pairs with the OTHER pair summed sequentially per
// thread (an exact elementwise port of AccumulateBlock's loop nest).
__global__ void EriFockContractKernel(const EriCudaFockArgs args) {
    const std::size_t t = static_cast<std::size_t>(blockIdx.x);

    if (t >= args.nTasks)
    {
        return;
    }

    const EriCudaFockTaskMeta& meta = args.tasks[t];
    const std::size_t n = static_cast<std::size_t>(args.n);
    const std::size_t nA = static_cast<std::size_t>(meta.nA);
    const std::size_t nB = static_cast<std::size_t>(meta.nB);
    const std::size_t nC = static_cast<std::size_t>(meta.nC);
    const std::size_t nD = static_cast<std::size_t>(meta.nD);
    const std::size_t offsetA = static_cast<std::size_t>(meta.offsetA);
    const std::size_t offsetB = static_cast<std::size_t>(meta.offsetB);
    const std::size_t offsetC = static_cast<std::size_t>(meta.offsetC);
    const std::size_t offsetD = static_cast<std::size_t>(meta.offsetD);
    const bool fp32 = args.precision == kEriCudaFockPrecisionFp32;
    // The batch layout of RunBatchOnDevice: task t's block sits at
    // outBase + tasks[t].outputOffset within the buffer the host hands
    // down (args.integralsF64 already includes outBase).
    const std::size_t blockBase = static_cast<std::size_t>(meta.outputOffset);
    // The precision is uniform per kernel: the lane is resolved once per
    // thread. Both handles are valid (the host aliases the unused lane to
    // the same buffer), so the untaken pointer is never formed from a null;
    // the phase loops read through EriFockLaneRead, which dereferences the
    // once-selected pointer - the per-element lane ternaries of the first
    // integration are gone (the lane is overhead-bound).
    const double* readF64 = fp32 ? nullptr : args.integralsF64 + blockBase;
    const float* readF32 = fp32 ? args.integralsF32 + blockBase : nullptr;

    // J: the bra side, and the ket side unless the quartet is its own ket.
    if (!args.buildExchangeOnly)
    {
        const std::size_t jBraSize = nA * nB;

        for (std::size_t slot = static_cast<std::size_t>(threadIdx.x); slot < jBraSize;
             slot += blockDim.x)
        {
            const std::size_t fa = slot / nB;
            const std::size_t fb = slot % nB;
            double j = 0.0;

            for (std::size_t fc = 0; fc < nC; ++fc)
            {
                for (std::size_t fd = 0; fd < nD; ++fd)
                {
                    // Both orientations of the unordered pair block (the
                    // canonical list represents (c, d) and (d, c) once).
                    const double densityElement =
                        meta.sameKetPair ? args.density[(offsetC + fc) * n + (offsetD + fd)]
                                         : args.density[(offsetC + fc) * n + (offsetD + fd)] +
                                               args.density[(offsetD + fd) * n + (offsetC + fc)];
                    const std::size_t blockIndex =
                        EriFockBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD);
                    j += EriFockLaneRead(readF64, readF32, blockIndex) * densityElement;
                }
            }

            const double twoJ = 2.0 * j;
            atomicAdd(&args.fock[(offsetA + fa) * n + (offsetB + fb)], twoJ);

            if (!meta.sameBraPair)
            {
                atomicAdd(&args.fock[(offsetB + fb) * n + (offsetA + fa)], twoJ);
            }
        }

        if (!meta.isDiagonal)
        {
            const std::size_t jKetSize = nC * nD;

            for (std::size_t slot = static_cast<std::size_t>(threadIdx.x); slot < jKetSize;
                 slot += blockDim.x)
            {
                const std::size_t fc = slot / nD;
                const std::size_t fd = slot % nD;
                double j = 0.0;

                for (std::size_t fa = 0; fa < nA; ++fa)
                {
                    for (std::size_t fb = 0; fb < nB; ++fb)
                    {
                        const double densityElement =
                            meta.sameBraPair
                                ? args.density[(offsetA + fa) * n + (offsetB + fb)]
                                : args.density[(offsetA + fa) * n + (offsetB + fb)] +
                                      args.density[(offsetB + fb) * n + (offsetA + fa)];
                        const std::size_t blockIndex =
                            EriFockBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD);
                        j += EriFockLaneRead(readF64, readF32, blockIndex) * densityElement;
                    }
                }

                const double twoJ = 2.0 * j;
                atomicAdd(&args.fock[(offsetC + fc) * n + (offsetD + fd)], twoJ);

                if (!meta.sameKetPair)
                {
                    atomicAdd(&args.fock[(offsetD + fd) * n + (offsetC + fc)], twoJ);
                }
            }
        }
    }

    // K: the four exchange targets, each with its arrangement density
    // read, plus the transposed element of every target; coincident
    // arrangements divide by the orbit size (kScaled = k / kMultiplicity,
    // the CPU formula).
    if (!args.buildCoulombOnly)
    {
        const double kScaledFactor = 1.0 / meta.kMultiplicity;

        // K1: targets (A, C) with the density d(B, D).
        const std::size_t k1Size = nA * nC;

        for (std::size_t slot = static_cast<std::size_t>(threadIdx.x); slot < k1Size;
             slot += blockDim.x)
        {
            const std::size_t fa = slot / nC;
            const std::size_t fc = slot % nC;
            double k = 0.0;

            for (std::size_t fb = 0; fb < nB; ++fb)
            {
                for (std::size_t fd = 0; fd < nD; ++fd)
                {
                    const std::size_t blockIndex =
                        EriFockBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD);
                    k += EriFockLaneRead(readF64, readF32, blockIndex) *
                         args.density[(offsetB + fb) * n + (offsetD + fd)];
                }
            }

            const double kScaled = k * kScaledFactor;
            atomicAdd(&args.fock[(offsetA + fa) * n + (offsetC + fc)], -kScaled);
            atomicAdd(&args.fock[(offsetC + fc) * n + (offsetA + fa)], -kScaled);
        }

        // K2: targets (A, D) with the density d(B, C).
        const std::size_t k2Size = nA * nD;

        for (std::size_t slot = static_cast<std::size_t>(threadIdx.x); slot < k2Size;
             slot += blockDim.x)
        {
            const std::size_t fa = slot / nD;
            const std::size_t fd = slot % nD;
            double k = 0.0;

            for (std::size_t fb = 0; fb < nB; ++fb)
            {
                for (std::size_t fc = 0; fc < nC; ++fc)
                {
                    const std::size_t blockIndex =
                        EriFockBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD);
                    k += EriFockLaneRead(readF64, readF32, blockIndex) *
                         args.density[(offsetB + fb) * n + (offsetC + fc)];
                }
            }

            const double kScaled = k * kScaledFactor;
            atomicAdd(&args.fock[(offsetA + fa) * n + (offsetD + fd)], -kScaled);
            atomicAdd(&args.fock[(offsetD + fd) * n + (offsetA + fa)], -kScaled);
        }

        // K3: targets (B, C) with the density d(A, D).
        const std::size_t k3Size = nB * nC;

        for (std::size_t slot = static_cast<std::size_t>(threadIdx.x); slot < k3Size;
             slot += blockDim.x)
        {
            const std::size_t fb = slot / nC;
            const std::size_t fc = slot % nC;
            double k = 0.0;

            for (std::size_t fa = 0; fa < nA; ++fa)
            {
                for (std::size_t fd = 0; fd < nD; ++fd)
                {
                    const std::size_t blockIndex =
                        EriFockBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD);
                    k += EriFockLaneRead(readF64, readF32, blockIndex) *
                         args.density[(offsetA + fa) * n + (offsetD + fd)];
                }
            }

            const double kScaled = k * kScaledFactor;
            atomicAdd(&args.fock[(offsetB + fb) * n + (offsetC + fc)], -kScaled);
            atomicAdd(&args.fock[(offsetC + fc) * n + (offsetB + fb)], -kScaled);
        }

        // K4: targets (B, D) with the density d(A, C).
        const std::size_t k4Size = nB * nD;

        for (std::size_t slot = static_cast<std::size_t>(threadIdx.x); slot < k4Size;
             slot += blockDim.x)
        {
            const std::size_t fb = slot / nD;
            const std::size_t fd = slot % nD;
            double k = 0.0;

            for (std::size_t fa = 0; fa < nA; ++fa)
            {
                for (std::size_t fc = 0; fc < nC; ++fc)
                {
                    const std::size_t blockIndex =
                        EriFockBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD);
                    k += EriFockLaneRead(readF64, readF32, blockIndex) *
                         args.density[(offsetA + fa) * n + (offsetC + fc)];
                }
            }

            const double kScaled = k * kScaledFactor;
            atomicAdd(&args.fock[(offsetB + fb) * n + (offsetD + fd)], -kScaled);
            atomicAdd(&args.fock[(offsetD + fd) * n + (offsetB + fb)], -kScaled);
        }
    }
}

} // namespace

// The generic-launch counterpart of the per-class registry launchers of
// eri_cuda.cu Part 4: the one definition of the extern "C" entry the host
// builder calls (declared in eri_cuda_fock.hpp).
extern "C" int QcxEriCudaFockContraction(EriCudaFockArgs& args) {
    if (args.nTasks == 0 || args.threads < 1)
    {
        return kEriCudaFockOk;
    }

    EriFockContractKernel<<<static_cast<unsigned int>(args.nTasks),
                            static_cast<unsigned int>(args.threads),
                            0,
                            static_cast<cudaStream_t>(args.stream)>>>(args);
    return static_cast<int>(cudaGetLastError());
}

} // namespace qcx::integrals::internal::cuda
