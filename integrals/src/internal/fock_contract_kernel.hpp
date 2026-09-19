// The shared per-quartet Fock-contraction kernel: ONE body,
// compiled twice - the scalar copy instantiates inside fock_build.cpp (that
// TU has no /arch flag) and an /arch:AVX2 copy instantiates in
// fock_contract_simd.cpp, selected at runtime by the cpuid check
// (FockAvx2Available - the boys_simd.cpp dispatch shape). The Tag only
// separates the two compiled copies; the TU's architecture flag is what
// actually changes the codegen. Both copies must deliver bit-identical
// output - the A/B tests in fock_build_test.cpp pin the contract, and
// FockBuildOptions::forceScalarContract exists so the fallback is
// exercisable on AVX2-capable machines.

#pragma once

#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace qcx::integrals::internal {

// Everything the kernel needs besides the block/quartet/pair indices: the
// FockContractor members it touches, narrowed to the kernel's actual reads
// and writes (fock_build.cpp builds this from the contractor).
struct FockContractContext {
    const ShellPairList& pairList;
    const Eigen::MatrixXd& density;
    Eigen::MatrixXd& fock;
    bool buildExchangeOnly;
    bool buildCoulombOnly;
    // The per-preset density threshold (DensityThreshold of the
    // builder's accuracy preset - the same tau the screening gate used) and
    // the per-call shell-compressed max-density vector (raw max |D_block|
    // per canonical pair, BuildShellPairMaxDensity in fock_screen.hpp), or
    // nullptr when the per-element screening flag is off (null vector = the
    // filter never engages). The per-element re-filter's thresholds are
    // tau / max|D_T| per target, once per quartet.
    double densityThreshold = 0.0;
    const std::vector<double>* pairMaxDensity = nullptr;
    // Drop counting: the per-quartet skipped-element
    // count lands in the target this points at (a per-chunk partial on the
    // parallel paths, a BuildFock local on the serial paths) - null = no
    // counting. Only populated when the filter is active.
    std::size_t* elementDrops = nullptr;
};

// The per-target threshold of the per-element re-filter: tau_T
// = tau / max|D_T| with niedoida's floor guard (max|D_T| < 1e-20 ->
// tau/(1e-20) - the standard_k_matrix_generator.cpp current_threshold
// shape). \p factor is 2.0 on the J sections: the J sections
// accumulate BOTH density orientations and accumulate 2.0*j, so one skipped
// J element can contribute through two multiplies - halving the threshold
// per multiply keeps the element's total contribution <= tau. The K
// families are single-orientation (factor 1.0, strict <= tau per element).
inline double PerElementDensityThreshold(double maxDensity,
                                         double densityThreshold,
                                         double factor) {
    const double denominator = std::max(maxDensity, 1e-20);
    return densityThreshold / (factor * denominator);
}

struct AccumulateBlockScalarTag {};

struct AccumulateBlockAvx2Tag {};

// Contracts one computed quartet block into the Fock matrix (either
// precision; the fp32 lane casts per element in the caller - covered by the
// certified bound). J accumulates on the bra and ket pair blocks of every
// canonical quartet; K accumulates on the four exchange targets with the
// density permuted per block - integrals are never transposed.
template <typename Tag>
void AccumulateBlockKernel(const double* block,
                           const ShellQuartet& quartet,
                           std::size_t pairBra,
                           std::size_t pairKet,
                           const FockContractContext& context) {
    const ShellPairList& pairList = context.pairList;
    const std::size_t a = quartet.i;
    const std::size_t b = quartet.j;
    const std::size_t c = quartet.k;
    const std::size_t dIndex = quartet.l;
    const std::size_t oA = pairList.shells[a].functionOffset;
    const std::size_t oB = pairList.shells[b].functionOffset;
    const std::size_t oC = pairList.shells[c].functionOffset;
    const std::size_t oD = pairList.shells[dIndex].functionOffset;
    const std::size_t nA = ShellFunctionCount(pairList.shells[a]);
    const std::size_t nB = ShellFunctionCount(pairList.shells[b]);
    const std::size_t nC = ShellFunctionCount(pairList.shells[c]);
    const std::size_t nD = ShellFunctionCount(pairList.shells[dIndex]);
    const bool diagonal = pairBra == pairKet;

    // The per-element re-filter. When the pair-max
    // vector is present, the six per-target density maxima are O(1)
    // canonical-pair lookups (the same six values the screening gate read -
    // cross pairs canonicalized before PairIndexOf), each section
    // computes its per-target threshold tau_T = tau / max|D_T| once per
    // quartet, and elements with |g| < tau_T are skipped (niedoida's
    // current_threshold shape, keep at equality). A zero block max skips
    // the whole target section - its contributions are exactly zero, and
    // the division-by-zero case is excluded explicitly. The
    // per-quartet drop count lands in context.elementDrops ("the
    // filter is engaged" must be measurable).
    const bool filterActive = context.pairMaxDensity != nullptr && !context.pairMaxDensity->empty();
    double dAb = 0.0;
    double dCd = 0.0;
    double dAc = 0.0;
    double dBd = 0.0;
    double dAd = 0.0;
    double dBc = 0.0;

    if (filterActive)
    {
        const std::vector<double>& pairMax = *context.pairMaxDensity;
        const auto canonical = [&pairList](std::size_t i, std::size_t j) {
            return PairIndexOf(std::min(i, j), std::max(i, j), pairList);
        };
        dAb = pairMax[pairBra];
        dCd = pairMax[pairKet];
        dAc = pairMax[canonical(a, c)];
        dBd = pairMax[canonical(b, dIndex)];
        dAd = pairMax[canonical(a, dIndex)];
        dBc = pairMax[canonical(b, c)];
    }

    std::size_t dropped = 0;

    // J: the bra side, and the ket side unless the quartet is its own
    // ket (the diagonal blocks). Skipped in the RiJkFockBuilder mode
    // (buildExchangeOnly) - RI provides J there. The density reads sum
    // BOTH orientations of an unordered pair block (the canonical list
    // represents (x,y) and (y,x) by one quartet).
    if (!context.buildExchangeOnly)
    {
        // J bra: target block (a,b), density block (c,d). The J sections
        // accumulate both orientations and 2.0*j, so the halved threshold
        // tau/(2*max|D_T|) bounds the element's total
        // contribution by tau.
        if (!filterActive || dCd != 0.0)
        {
            const double jBraThreshold =
                filterActive ? PerElementDensityThreshold(dCd, context.densityThreshold, 2.0) : 0.0;

            for (std::size_t fa = 0; fa < nA; ++fa)
            {
                for (std::size_t fb = 0; fb < nB; ++fb)
                {
                    double j = 0.0;

                    for (std::size_t fc = 0; fc < nC; ++fc)
                    {
                        for (std::size_t fd = 0; fd < nD; ++fd)
                        {
                            const double g = block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)];

                            if (filterActive && std::abs(g) < jBraThreshold)
                            {
                                ++dropped;
                                continue;
                            }

                            const double densityElement =
                                (c == dIndex)
                                    ? context.density(static_cast<Eigen::Index>(oC + fc),
                                                      static_cast<Eigen::Index>(oD + fd))
                                    : context.density(static_cast<Eigen::Index>(oC + fc),
                                                      static_cast<Eigen::Index>(oD + fd)) +
                                          context.density(static_cast<Eigen::Index>(oD + fd),
                                                          static_cast<Eigen::Index>(oC + fc));
                            j += g * densityElement;
                        }
                    }

                    context.fock(static_cast<Eigen::Index>(oA + fa),
                                 static_cast<Eigen::Index>(oB + fb)) += 2.0 * j;

                    if (a != b)
                    {
                        context.fock(static_cast<Eigen::Index>(oB + fb),
                                     static_cast<Eigen::Index>(oA + fa)) += 2.0 * j;
                    }
                }
            }
        }

        if (!diagonal)
        {
            // J ket: target block (c,d), density block (a,b).
            if (!filterActive || dAb != 0.0)
            {
                const double jKetThreshold =
                    filterActive ? PerElementDensityThreshold(dAb, context.densityThreshold, 2.0)
                                 : 0.0;

                for (std::size_t fc = 0; fc < nC; ++fc)
                {
                    for (std::size_t fd = 0; fd < nD; ++fd)
                    {
                        double j = 0.0;

                        for (std::size_t fa = 0; fa < nA; ++fa)
                        {
                            for (std::size_t fb = 0; fb < nB; ++fb)
                            {
                                const double g =
                                    block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)];

                                if (filterActive && std::abs(g) < jKetThreshold)
                                {
                                    ++dropped;
                                    continue;
                                }

                                const double densityElement =
                                    (a == b)
                                        ? context.density(static_cast<Eigen::Index>(oA + fa),
                                                          static_cast<Eigen::Index>(oB + fb))
                                        : context.density(static_cast<Eigen::Index>(oA + fa),
                                                          static_cast<Eigen::Index>(oB + fb)) +
                                              context.density(static_cast<Eigen::Index>(oB + fb),
                                                              static_cast<Eigen::Index>(oA + fa));
                                j += g * densityElement;
                            }
                        }

                        context.fock(static_cast<Eigen::Index>(oC + fc),
                                     static_cast<Eigen::Index>(oD + fd)) += 2.0 * j;

                        if (c != dIndex)
                        {
                            context.fock(static_cast<Eigen::Index>(oD + fd),
                                         static_cast<Eigen::Index>(oC + fc)) += 2.0 * j;
                        }
                    }
                }
            }
        }
    }

    // K: the four exchange targets of the quartet, each contracted
    // with its arrangement density read, plus the transposed element
    // of every target. The canonical list holds one orientation of
    // every quartet, so a target write covers the target arrangement
    // and its transpose (the pair-swapped and role-swapped twins -
    // density symmetry makes the reads equal), and coincident
    // arrangements divide by the orbit size: 2 per degenerate pair
    // and 2 more for the self-role-swapped quartet (a,b) == (c,d).
    // Skipped entirely in the UHF J-only mode (buildCoulombOnly) - the
    // caller assembles the exchange part from a separate builder (the
    // direct-UHF adapter).
    if (!context.buildCoulombOnly)
    {
        const double kMultiplicity = ((a == b) ? 2.0 : 1.0) * ((c == dIndex) ? 2.0 : 1.0) *
                                     ((a == c && b == dIndex) ? 2.0 : 1.0);

        // K1: target (a,c), density block (b,d) - single orientation, so
        // the strict per-element bound tau/max|D_bd|.
        if (!filterActive || dBd != 0.0)
        {
            const double k1Threshold =
                filterActive ? PerElementDensityThreshold(dBd, context.densityThreshold, 1.0) : 0.0;

            for (std::size_t fa = 0; fa < nA; ++fa)
            {
                for (std::size_t fc = 0; fc < nC; ++fc)
                {
                    double k = 0.0;

                    for (std::size_t fb = 0; fb < nB; ++fb)
                    {
                        for (std::size_t fd = 0; fd < nD; ++fd)
                        {
                            const double g = block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)];

                            if (filterActive && std::abs(g) < k1Threshold)
                            {
                                ++dropped;
                                continue;
                            }

                            k += g * context.density(static_cast<Eigen::Index>(oB + fb),
                                                     static_cast<Eigen::Index>(oD + fd));
                        }
                    }

                    const double kScaled = k / kMultiplicity;
                    context.fock(static_cast<Eigen::Index>(oA + fa),
                                 static_cast<Eigen::Index>(oC + fc)) -= kScaled;
                    context.fock(static_cast<Eigen::Index>(oC + fc),
                                 static_cast<Eigen::Index>(oA + fa)) -= kScaled;
                }
            }
        }

        // K2: target (a,d), density block (b,c).
        if (!filterActive || dBc != 0.0)
        {
            const double k2Threshold =
                filterActive ? PerElementDensityThreshold(dBc, context.densityThreshold, 1.0) : 0.0;

            for (std::size_t fa = 0; fa < nA; ++fa)
            {
                for (std::size_t fd = 0; fd < nD; ++fd)
                {
                    double k = 0.0;

                    for (std::size_t fb = 0; fb < nB; ++fb)
                    {
                        for (std::size_t fc = 0; fc < nC; ++fc)
                        {
                            const double g = block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)];

                            if (filterActive && std::abs(g) < k2Threshold)
                            {
                                ++dropped;
                                continue;
                            }

                            k += g * context.density(static_cast<Eigen::Index>(oB + fb),
                                                     static_cast<Eigen::Index>(oC + fc));
                        }
                    }

                    const double kScaled = k / kMultiplicity;
                    context.fock(static_cast<Eigen::Index>(oA + fa),
                                 static_cast<Eigen::Index>(oD + fd)) -= kScaled;
                    context.fock(static_cast<Eigen::Index>(oD + fd),
                                 static_cast<Eigen::Index>(oA + fa)) -= kScaled;
                }
            }
        }

        // K3: target (b,c), density block (a,d).
        if (!filterActive || dAd != 0.0)
        {
            const double k3Threshold =
                filterActive ? PerElementDensityThreshold(dAd, context.densityThreshold, 1.0) : 0.0;

            for (std::size_t fb = 0; fb < nB; ++fb)
            {
                for (std::size_t fc = 0; fc < nC; ++fc)
                {
                    double k = 0.0;

                    for (std::size_t fa = 0; fa < nA; ++fa)
                    {
                        for (std::size_t fd = 0; fd < nD; ++fd)
                        {
                            const double g = block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)];

                            if (filterActive && std::abs(g) < k3Threshold)
                            {
                                ++dropped;
                                continue;
                            }

                            k += g * context.density(static_cast<Eigen::Index>(oA + fa),
                                                     static_cast<Eigen::Index>(oD + fd));
                        }
                    }

                    const double kScaled = k / kMultiplicity;
                    context.fock(static_cast<Eigen::Index>(oB + fb),
                                 static_cast<Eigen::Index>(oC + fc)) -= kScaled;
                    context.fock(static_cast<Eigen::Index>(oC + fc),
                                 static_cast<Eigen::Index>(oB + fb)) -= kScaled;
                }
            }
        }

        // K4: target (b,d), density block (a,c).
        if (!filterActive || dAc != 0.0)
        {
            const double k4Threshold =
                filterActive ? PerElementDensityThreshold(dAc, context.densityThreshold, 1.0) : 0.0;

            for (std::size_t fb = 0; fb < nB; ++fb)
            {
                for (std::size_t fd = 0; fd < nD; ++fd)
                {
                    double k = 0.0;

                    for (std::size_t fa = 0; fa < nA; ++fa)
                    {
                        for (std::size_t fc = 0; fc < nC; ++fc)
                        {
                            const double g = block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)];

                            if (filterActive && std::abs(g) < k4Threshold)
                            {
                                ++dropped;
                                continue;
                            }

                            k += g * context.density(static_cast<Eigen::Index>(oA + fa),
                                                     static_cast<Eigen::Index>(oC + fc));
                        }
                    }

                    const double kScaled = k / kMultiplicity;
                    context.fock(static_cast<Eigen::Index>(oB + fb),
                                 static_cast<Eigen::Index>(oD + fd)) -= kScaled;
                    context.fock(static_cast<Eigen::Index>(oD + fd),
                                 static_cast<Eigen::Index>(oB + fb)) -= kScaled;
                }
            }
        }
    }

    if (dropped != 0 && context.elementDrops != nullptr)
    {
        *context.elementDrops += dropped;
    }
}

// True when the CPU reports AVX2 (cpuid + OSXSAVE, cached) - the gate the
// dispatch reads. False on CPUs without AVX2 (and always on the CI lanes'
// x64 baselines): the scalar copy is then the only copy ever entered.
bool FockAvx2Available() noexcept;

// The /arch:AVX2 copy of the kernel (fock_contract_simd.cpp). Call only
// when FockAvx2Available() holds; otherwise the codegen the TU compiled
// with may fault on the host CPU.
void AccumulateBlockAvx2(const double* block,
                         const ShellQuartet& quartet,
                         std::size_t pairBra,
                         std::size_t pairKet,
                         const FockContractContext& context);

} // namespace qcx::integrals::internal
