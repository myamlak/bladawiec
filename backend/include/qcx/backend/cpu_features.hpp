#pragma once

// The host CPU-feature detection: one predicate per requirement, defined
// once, so no two translation units anywhere in the tree can answer the same
// question differently. Two consumers read it - the integrals module's
// /arch:AVX2 translation units (fock_contract_simd.cpp, md_bra_simd.cpp) and
// this module's own host compute-profile probe
// (host_compute_profile.hpp), whose AVX2 measurement lane is gated on
// CpuHasAvx2Fma().
//
// WHY IT LIVES IN backend. It was integrals/src/internal/cpu_features.hpp
// until the host compute-profile probe was added. The
// probe is a hardware probe and belongs beside the device probe in this
// module, but backend sits BELOW integrals in the module DAG (core -> backend
// -> ... -> integrals), so a probe here cannot include an integrals header.
// Moving the header down keeps ONE definition; the alternative - a second
// copy on this side - would have broken that, and a duplicated predicate is
// how the RDT-M/FMA defect recorded below happened in the first place.
//
// WHY IT IS ONE PLACE. Both TUs used to carry a private detector and they
// disagreed. md_bra_simd.cpp asked for OSXSAVE + AVX2 + FMA + the XCR0 vector
// state - the strictest of the three, and the only one that checks the OS has
// actually enabled the state - but read the FMA flag from
// CPUID.(EAX=7,ECX=0):EBX bit 12, which is RDT-M/PQM (a Xeon-only monitoring
// feature); the architectural FMA flag is CPUID.01H:ECX bit 12. On every CPU
// without RDT-M the MD transforms therefore answered "no FMA", fell back to
// the scalar md_gemm.hpp kernels, and the whole AVX2 micro-GEMM tier was dead
// code. Measured on the reference machine (2026-09-12): 01H:ECX.12 = 1,
// 07H:EBX.12 = 0, XCR0 = 0x1f, and an FMA instruction executes and returns
// the FMA rounding. The second detector (fock_contract_simd.cpp) asked for
// OSXSAVE + AVX2 and no FMA bit at all, the boys_simd.cpp shape.
//
// The predicates are named for the requirement, not for the module: a
// consumer whose AVX2 code contains no FMA gates on CpuHasAvx2(), one whose
// code can emit FMA gates on CpuHasAvx2Fma(). md_bra_simd.cpp is an FMA user
// in the direct sense (it calls _mm256_fmadd_pd); fock_contract_simd.cpp's
// /arch:AVX2 body is an auto-vectorized one whose mul+add the GNU toolchain
// may fuse (the divergence fock_build_test.cpp's dispatch A/B band records),
// while MSVC at its default /fp:precise does not contract.
//
// The boys submodule's own gate (external/boys/src/boys_simd.cpp
// DetectAvx2()) is still its own: OSXSAVE + AVX2, no FMA bit, though its
// kernels call _mm256_fmadd_pd. That file belongs to the submodule, so the
// divergence is recorded here rather than fixed here - and the two agree on
// every machine that reports FMA, which is every machine that reports AVX2.
//
// The XCR0 half is the reason this is a header and not a macro: it is the
// only part of the check that a bare feature bit cannot answer, and it is
// what makes "AVX2 available" mean "AVX2 may execute" rather than "the CPU
// has it".

// MSVC: <intrin.h> carries __cpuid/__cpuidex/_xgetbv. GNU/Clang: <cpuid.h>
// carries the __get_cpuid family, <immintrin.h> the _xgetbv wrapper, which is
// reached through ReadXcr0() below rather than called directly (the wrapper
// needs a target feature that a default build does not enable). The four-word
// results are std::array, not raw arrays: the intrinsics take a pointer (MSVC)
// or four pointers (GNU), so nothing here needs the raw-array form
// reserved for ABIs that truly mandate it.
//
// ALL THREE HEADERS, AND EVERY BODY BELOW, ARE x86-64 ONLY: CPUID and XSAVE
// have no other architecture, and the headers that declare their intrinsics do
// not exist there. The guards read ONE name, defined next.
#include <array>

// The tree's architecture test. It is the COMPILER's own macros rather than a
// build-system definition on purpose: this is a public header, a private
// target-level define would not reach a consumer of it, and two translation
// units answering this question differently would be an ODR defect instead of
// a compile error. The build system derives its own answer from the same two
// macros (the CMake probe that decides which per-file target features are
// legal), so the flag set and these guards cannot drift apart.
#if defined(__x86_64__) || defined(_M_X64)
#define QcxArchX86_64 1
#else
#define QcxArchX86_64 0
#endif

#if QcxArchX86_64
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

namespace qcx::backend {

#if QcxArchX86_64 && !defined(_MSC_VER)
/// Reads XCR0, the register that reports which extended processor state the OS
/// has enabled for this process.
///
/// The indirection is the point: GCC and Clang expand _xgetbv only where the
/// XSAVE target feature is on, so a bare call from a translation unit built
/// without it is refused at compile time ("target specific option mismatch") -
/// the compiler cannot see the run-time OSXSAVE test that guards the call. The
/// target attribute carries that feature into this one function, so the
/// instruction is emitted here and every caller stays compilable in a default
/// build; MSVC needs no such guard and calls the intrinsic directly.
///
/// Callers must have checked OSXSAVE: this read faults where the OS never
/// enabled the state.
/// \returns The current XCR0 value.
/// \ingroup qcx-backend
__attribute__((target("xsave"))) inline unsigned long long ReadXcr0() noexcept {
    return _xgetbv(0);
}
#endif

/// Whether the AVX2 instruction set may execute on this machine: the CPU
/// reports it (CPUID.(EAX=7,ECX=0):EBX bit 5) and the OS has the vector state
/// enabled (OSXSAVE, then XCR0 bits 1 and 2) - a CPU whose kernel never
/// enabled AVX would fault on the first AVX2 instruction, so the feature bit
/// alone is not the answer.
/// \returns True when AVX2 instructions are safe to execute.
/// \ingroup qcx-backend
inline bool CpuHasAvx2() noexcept {
#if !QcxArchX86_64
    // No CPUID on this architecture and no AVX2 instruction to guard: false is
    // the whole answer, and it keeps the caller's vector path unentered rather
    // than letting a compile error be the way that is discovered.
    return false;
#else
#ifdef _MSC_VER
    std::array<int, 4> leaf0 = {};
    __cpuid(leaf0.data(), 0);

    if (leaf0[0] < 7)
    {
        return false;
    }

    std::array<int, 4> leaf1 = {};
    __cpuid(leaf1.data(), 1);

    if ((leaf1[2] & (1u << 27)) == 0) // OSXSAVE
    {
        return false;
    }

    std::array<int, 4> leaf7 = {};
    __cpuidex(leaf7.data(), 7, 0);
    const unsigned long long xcr0 = _xgetbv(0);
    const bool avx2 = (leaf7[1] & (1u << 5)) != 0;
    return avx2 && (xcr0 & 0x6) == 0x6;
#else
    std::array<unsigned int, 4> leaf0 = {};
    __get_cpuid(0, &leaf0[0], &leaf0[1], &leaf0[2], &leaf0[3]);

    if (leaf0[0] < 7)
    {
        return false;
    }

    std::array<unsigned int, 4> leaf1 = {};
    __get_cpuid(1, &leaf1[0], &leaf1[1], &leaf1[2], &leaf1[3]);

    if ((leaf1[2] & (1u << 27)) == 0) // OSXSAVE
    {
        return false;
    }

    std::array<unsigned int, 4> leaf7 = {};

    if (__get_cpuid_count(7, 0, &leaf7[0], &leaf7[1], &leaf7[2], &leaf7[3]) == 0)
    {
        return false;
    }

    const unsigned long long xcr0 = ReadXcr0();
    const bool avx2 = (leaf7[1] & (1u << 5)) != 0;
    return avx2 && (xcr0 & 0x6) == 0x6;
#endif
#endif
}

/// Whether AVX2 *with FMA* may execute: CpuHasAvx2() plus the architectural
/// FMA flag, CPUID.01H:ECX bit 12. It is leaf 1 ECX, not leaf 7 EBX, because
/// FMA predates the structured-extended-features leaf the way AVX does - the
/// two are neighbours in the same word (AVX at bit 28, FMA at bit 12) - and
/// leaf 7 EBX bit 12 is RDT-M/PQM instead. Reading the wrong leaf reports "no
/// FMA" on every machine that is not a Xeon with the monitoring feature, and
/// a kernel dispatched on that answer silently runs its scalar fallback.
/// \returns True when the FMA instructions of this module's AVX2 kernels are
/// safe to execute.
/// \ingroup qcx-backend
inline bool CpuHasAvx2Fma() noexcept {
#if !QcxArchX86_64
    // The FMA flag is a CPUID leaf and there is no CPUID here; CpuHasAvx2() is
    // false on this architecture, so this is the same answer it would give.
    return false;
#else
    if (!CpuHasAvx2())
    {
        return false;
    }

#ifdef _MSC_VER
    std::array<int, 4> leaf1 = {};
    __cpuid(leaf1.data(), 1);
    return (leaf1[2] & (1u << 12)) != 0;
#else
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0)
    {
        return false;
    }

    return (ecx & (1u << 12)) != 0;
#endif
#endif
}

} // namespace qcx::backend
