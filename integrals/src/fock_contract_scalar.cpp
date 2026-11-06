// The baseline copy of the Fock-contraction kernel, out of line and out of
// the AVX2 tier: this unit is compiled without /arch:AVX2 or
// -mavx2 -mfma -mf16c (integrals/CMakeLists.txt states the per-source
// override), so the template body it instantiates is code a processor
// without AVX2 can execute.
//
// It exists because the fallback was not one. The scalar instantiation lived
// in fock_build.cpp, a unit this tree compiles at the AVX2 tier, so on an
// AVX2 host the "scalar copy" was the vector tier's codegen and on a host
// without AVX2 the entry could fault - and no build exercised the difference.
// The dispatcher (FockContractor::AccumulateBlock) now enters this unit when
// FockAvx2Available() is false or the caller pinned the scalar contract.
//
// This file includes no Boys header, deliberately: the library's tier
// registry refuses a program whose translation units disagree, and a unit at
// the baseline tier in an AVX2 program would be refused at start-up. The
// kernel needs the quartet and the contraction context, both of which are
// this module's own; the contraction pin is program-wide, so the two copies
// deliver bit-identical output (fock_build_test.cpp pins it).

#include "internal/fock_contract_kernel.hpp"

namespace qcx::integrals::internal {

void AccumulateBlockScalar(const double* block,
                           const ShellQuartet& quartet,
                           std::size_t pairBra,
                           std::size_t pairKet,
                           const FockContractContext& context) {
    AccumulateBlockKernel<AccumulateBlockScalarTag>(block, quartet, pairBra, pairKet, context);
}

bool FockScalarCopyIsBaseline() noexcept {
#if defined(__AVX2__)
    return false;
#else
    return true;
#endif
}

} // namespace qcx::integrals::internal
