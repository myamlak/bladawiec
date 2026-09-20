#pragma once

/// \file
/// Create-time device-footprint estimation for the GPU device
/// workspace budget: the byte terms EriCudaEngine::Create fills the public
/// GpuDeviceFootprint record with - the uploaded pair tables (the exact
/// hostTables sizes, passed in by the caller who built them), the fixed
/// Boys copy, the per-call density/Fock matrices (2 x 8 n^2), the kV2
/// per-batch global scratch (the pq/acc/bound arenas, one batch at a time),
/// the per-call output mass (the screened-exchange envelope, the gate
/// re-pinned n^-1 law floored at footprint.hpp's earlier fitted fractions) and the
/// structural allowance (the cuBLAS workspace and the driver pool). The
/// formulas mirror the allocation shapes of eri_cuda.cpp
/// (RunBatchOnDevice / RunDevicePass); every term is a byte bound, never a
/// guess, so a mode decision made on this estimate can never over-commit
/// the device budget. The residency contract's RI/tree slot is a named
/// zero until the RI path's device residency lands (the RI tensors become
/// device-resident; their bytes charge this slot).

#include "internal/footprint.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_cuda.hpp"

#include <algorithm>
#include <cstddef>

namespace qcx::integrals::internal {

/// The cuBLAS workspace: the cuBLAS 12 default, 32 MiB, allocated lazily at
/// the first transform GEMM (kV2 classes only) and retained until the
/// handle is destroyed.
inline constexpr std::size_t kGpuCublasWorkspaceBytes = 32 * 1024 * 1024;

/// The driver-pool allowance: the per-Create device-side overhead (the
/// cuBLAS module and the ERI cubins, loaded by Create and retained for the
/// process lifetime) plus the driver's first-mapping quantum - the FIRST
/// small allocation of a context maps a 2 MiB granule regardless of its
/// size (measured on the T1000 via the 64 KiB probe at the device-budget
/// gate). Calibrated on the Quadro T1000 (2026-08-30): every measured
/// create delta is consistent with an 8 MiB per-Create pool plus the 2 MiB
/// granule (first create 10.0 MiB, a second create 8.0 MiB, C12H26/STO-3G
/// with 4.33 MB of uploads 12.0 MiB); the constant is set at the 10 MiB
/// bound so the estimate never under-charges the driver's mapping. The
/// uploads themselves are charged at byte truth and map granularity-lazily
/// (the measured create can under-run the estimate - errs safe).
inline constexpr std::size_t kGpuDriverPoolBytes = 10 * 1024 * 1024;

/// The per-call screened-quartet envelope of the GPU lane, re-pinned at the
/// device-budget gate (2026-08-30): the earlier flat fractions (0.0088 /
/// 1.42e-4) under-charge the C12H26/STO-3G measurement 7.7x (the measured
/// per-call growth was 30.0 MiB vs the flat-fit 4.2 MB - the same C12
/// refutation the CPU seam records). The C12 anchor is
/// f(86) = 0.068331 (from the measured 30.0 MiB growth: the estimate
/// charges 8T + 24Q + the matrices; the amp is set so the term matches the
/// measurement to 0.0003%), and the n^-1 laws through it land 14% ABOVE
/// the C24H50 single-point fit at n = 586 (0.01003 vs 0.0088) - a two-point
/// statement that errs safe everywhere (the amplitude laws are floored at
/// the earlier fitted fractions, so the estimate never charges less than the old
/// single-point fit at any size). Small-n extrapolations over-charge (fine:
/// byte bounds, never guesses).
inline constexpr double kGpuSurvivalAmp = 5.8765; // f(n) = 5.8765 / n.
inline constexpr double kGpuQuartetAmp = 0.09412; // q(n) = 0.09412 / n.

/// The per-call output mass of one contraction pass (the peak pass of
/// BuildFock's sequential structure): the packed out buffer - one word per
/// function-pair block element of every screened quartet, 8 B/word in the
/// fp64 pass (larger than the certified lane's 4 B/word plus its bounds:
/// 8T > 4T + 8Q since Q/T ~= 0.016, so the fp32-only bounds never make the
/// peak pass) - plus the per-pass task/meta/ranges buffers (24 B per
/// screened quartet, both lanes). The screened-quartet envelope is the
/// n^-1 law of kGpuSurvivalAmp / kGpuQuartetAmp (the device-budget gate
/// re-pin; see the constants above), floored at the earlier fitted fractions of
/// footprint.hpp.
/// \param functionCount The orbital function count (pairList.functionCount).
inline std::size_t GpuPerCallOutputBytes(std::size_t functionCount) noexcept {
    const double n = static_cast<double>(functionCount);
    const double n4 = n * n * n * n;
    const double survival =
        std::max(kGpuSurvivalAmp / n, static_cast<double>(kExchangeSurvivalFraction));
    const double quartets =
        std::max(kGpuQuartetAmp / n, static_cast<double>(kExchangeQuartetFraction));
    return static_cast<std::size_t>(8.0 * survival * n4 + 24.0 * quartets * n4);
}

/// Fills the engine's Create-time device footprint record: the caller
/// passes the exact uploaded-table bytes (the hostTables
/// sizes it just built - the ground truth, never re-derived) and the Boys
/// copy size, and the function adds the per-call working set and the
/// structural allowance. The kV2 global scratch and the cuBLAS workspace
/// engage only when some class maps to kV2 (the caller's variant
/// resolution: L = lBra + lKet >= 9 under kAuto, or the kV2 override).
/// \param functionCount The orbital function count (pairList.functionCount).
/// \param maxBatchBytes The options' batch cap (the kV2 arena bound).
/// \param tablesBytes The exact uploaded pair-table bytes (hostTables).
/// \param boysBytes sizeof of the Boys tables copy.
/// \param hasKv2Classes True when the variant resolution maps some class to
/// kV2 (the global-memory VRR + cuBLAS path).
/// \returns The footprint record, every term filled.
inline GpuDeviceFootprint ComputeGpuDeviceFootprint(std::size_t functionCount,
                                                    std::size_t maxBatchBytes,
                                                    std::size_t tablesBytes,
                                                    std::size_t boysBytes,
                                                    bool hasKv2Classes) noexcept {
    GpuDeviceFootprint terms;
    terms.tablesBytes = tablesBytes;
    terms.boysBytes = boysBytes;
    terms.matricesBytes = 2 * 8 * functionCount * functionCount;
    terms.batchScratchBytes = hasKv2Classes ? maxBatchBytes : 0;
    terms.perCallOutputBytes = GpuPerCallOutputBytes(functionCount);
    terms.riResidencyBytes = 0;
    terms.structuralBytes = (hasKv2Classes ? kGpuCublasWorkspaceBytes : 0) + kGpuDriverPoolBytes;
    return terms;
}

} // namespace qcx::integrals::internal
