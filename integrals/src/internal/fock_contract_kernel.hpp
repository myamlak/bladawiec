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
    // The per-preset density threshold (DensityThreshold of the builder's
    // accuracy preset - the same tau the screening gate used) and the
    // per-call shell-compressed max-density vector (raw max |D_block| per
    // canonical pair, BuildShellPairMaxDensity in fock_screen.hpp), or
    // nullptr when the per-element screening flag is off (null vector = the
    // filter never engages). The per-element re-filter's threshold is
    // tau / (factor * max|D_T|) per target, computed once per quartet; the
    // factor is 2.0 on the J sections and 1.0 on the K sections, and its
    // criterion sits with PerElementDensityThreshold below.
    double densityThreshold = 0.0;
    const std::vector<double>* pairMaxDensity = nullptr;
    // The per-quartet skipped-element count lands in the target this points
    // at (a per-chunk partial on the parallel paths, a BuildFock local on
    // the serial paths) - null = no counting. Only populated when the
    // filter is active.
    std::size_t* elementDrops = nullptr;
};

// The per-target threshold of the per-element re-filter. The
// density-weighted prescreening of Häser & Ahlrichs (J. Comput. Chem. 10,
// 104 (1989)) and of Almlöf, Fægri & Korsell (J. Comput. Chem. 3, 385
// (1982)) drops an integral whose density-weighted magnitude cannot move
// the result past the requested accuracy. Here it is applied per element
// rather than per shell quartet: a target block contracted against the
// density block T receives at most |g| * max|D_T| through one element g,
// so the drop test |g| * max|D_T| < tau is |g| < tau / max|D_T|.
//
// \p factor sets how large a multiple of tau one dropped element's
// contribution to a single Fock TARGET element may reach: the binding is
// factor * tau, so the J sections (which double a J element twice over -
// see the J comment below) pass 2.0 and the single-orientation,
// undoubled K sections pass 1.0. The factor is engineering, not physics:
// raising it drops fewer elements and costs more work, and the converged
// result is the same either way.
//
// The 1e-20 floor is a guard on the quotient, not a screening decision.
// Every caller already refuses its section on an exactly zero block max,
// so the floor is reachable only by a block whose density max is itself
// at or below 1e-20 - a block with no representable weight. There
// tau / 1e-20 stands orders of magnitude above any integral a normalized
// basis can yield, so the section is dropped whole, which is what a
// vanishing weight justifies. Any floor small enough that tau / floor
// still exceeds every block magnitude takes the identical decisions, so
// the exact value is numerical hygiene rather than a tuned constant.
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

    // The per-element re-filter. When the pair-max vector is present, the
    // six per-target density maxima are O(1) canonical-pair lookups (the
    // same six values the screening gate read - cross pairs canonicalized
    // before PairIndexOf), each section computes its per-target threshold
    // tau_T once per quartet, and elements with |g| < tau_T are skipped.
    // The comparison is strict, so an element exactly at its threshold is
    // kept: keeping is the conservative side, since a kept element can
    // only add work and never error. A zero block max skips the whole
    // target section - its contributions are exactly zero, and the
    // division is excluded explicitly. The per-quartet drop count lands in
    // context.elementDrops so that "the filter is engaged" stays
    // measurable.
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
        // J bra: target block (a,b), density block (c,d). A J element
        // reaches one Fock target element through two doublings - the read
        // sums both orientations of the symmetric density block, and the
        // accumulated sum carries the closed-shell factor 2 - so the
        // factor-2 threshold bounds a dropped element's contribution by
        // 2*tau (the K sections below pass 1.0 for a bound of tau).
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

    // K: the four exchange targets of the quartet. The exchange part of
    // the Fock matrix is K_mu_nu = sum_{lambda,sigma} D_lambda_sigma
    // (mu lambda | nu sigma), so every ordered assignment
    // (mu, lambda, nu, sigma) of the quartet's four shells contributes
    // once. The eightfold permutation symmetry of a real two-electron
    // integral - invariance under the bra swap mu <-> lambda, the ket
    // swap nu <-> sigma, and the pair exchange
    // (mu lambda) <-> (nu sigma) - partitions those assignments by which
    // pairing of the four shells forms the bra pair. This block holds the
    // integrals of exactly one such class, the one whose bra pair is
    // {a, b}; inside it mu is either a or b and nu is either c or d, four
    // combinations - hence four targets. Each reads the stored block
    // through one of the three invariances:
    //   K1  target (a,c) with density block (b,d): the stored assignment;
    //   K2  target (a,d) with density block (b,c): the ket swap;
    //   K3  target (b,c) with density block (a,d): the bra swap;
    //   K4  target (b,d) with density block (a,c): both swaps.
    // The other two classes need integrals that no block of this quartet
    // holds - they live in the quartets whose bra pair is {a, c} or
    // {a, d}, which the canonical list carries in their own right - so
    // each class is covered exactly once across the build. Every target
    // writes its transpose as well: the pair of writes stands for the two
    // ordered (lambda, sigma) orientations of the density block, whose
    // reads are equal because D is symmetric.
    //
    // Those four targets emit the orbit's eight ordered assignments. An
    // involution that fixes the shell tuple - the bra swap when a == b,
    // the ket swap when c == d, the pair exchange when (a, b) == (c, d) -
    // makes two of the assignments one and the same accumulation, which
    // doubles the count, so the accumulated value is divided by the
    // product of the fixing involutions, returning every assignment to a
    // single count.
    //
    // Skipped entirely in the UHF J-only mode (buildCoulombOnly) - the
    // caller assembles the exchange part from a separate builder.
    if (!context.buildCoulombOnly)
    {
        // The product of the involutions that fix the shell tuple (above).
        const double kMultiplicity = ((a == b) ? 2.0 : 1.0) * ((c == dIndex) ? 2.0 : 1.0) *
                                     ((a == c && b == dIndex) ? 2.0 : 1.0);

        // K1: target (a,c), density block (b,d) - single orientation and
        // undoubled, so the strict per-element bound tau/max|D_bd|.
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
