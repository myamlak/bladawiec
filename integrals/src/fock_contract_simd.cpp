// The AVX2 copy of the Fock-contraction kernel: the one
// kernel body from internal/fock_contract_kernel.hpp, compiled with
// /arch:AVX2 (auto-vectorization only - no hand-written intrinsics at this
// stage) and entered only after the runtime cpuid
// check (qcx::backend::CpuHasAvx2(), qcx/backend/cpu_features.hpp) passes. The scalar copy
// instantiates inside fock_build.cpp (no /arch flag on that TU). The dispatch
// shape mirrors boys_simd.cpp: CPUID + OSXSAVE detection cached in a
// function-local static, the caller gating on it before the AVX2 path is ever
// entered - never assume the deployment machine has AVX2 just because the dev
// machine does. The detection is no longer a copy of boys_simd.cpp's: it is
// the module's shared one, so a wrong-bit edit cannot reach one TU and not
// the other.
//
// The translation unit is compiled with /arch:AVX2 and skips the module PCH
// (clang rejects a PCH built without the target feature the TU compiles
// with) - the same pair of CMake properties the other SIMD TUs carry.

#include "internal/fock_contract_kernel.hpp"

#include <cstddef>
#include <qcx/backend/cpu_features.hpp>

namespace qcx::integrals::internal {

bool FockAvx2Available() noexcept {
    // The detection lives in qcx/backend/cpu_features.hpp, one predicate per
    // requirement. This TU gates on the AVX2 contract and NOT on FMA, which
    // is what it needs: MSVC at its default /fp:precise does not contract,
    // so this /arch:AVX2 copy of the kernel differs from the scalar copy in
    // vector width alone (the fock_build_test dispatch A/B band is bit-exact
    // on MSVC Release for that reason). On GNU the -mfma flag this TU is
    // built with does let -ffp-contract=fast fuse, which is why that band is
    // a few ulps there and not zero - read cpu_features.hpp before changing
    // this to the FMA predicate.
    static const bool available = qcx::backend::CpuHasAvx2();
    return available;
}

void AccumulateBlockAvx2(const double* block,
                         const ShellQuartet& quartet,
                         std::size_t pairBra,
                         std::size_t pairKet,
                         const FockContractContext& context) {
    AccumulateBlockKernel<AccumulateBlockAvx2Tag>(block, quartet, pairBra, pairKet, context);
}

} // namespace qcx::integrals::internal
