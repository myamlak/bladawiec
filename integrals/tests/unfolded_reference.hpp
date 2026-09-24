#pragma once

// The unfolded reference path of the MD algebra: a scalar per-quartet
// implementation with independent nested loops (no class templates, no
// slice-tier traversal, no batched transforms) - the plan's mandated
// independent validation of the fast path's recurrences. It re-derives the
// 1D E coefficients from the primitive data, runs the VRR over explicit
// (tx,ty,tz,m) loops, and contracts through its own transform loops; the
// only shared pieces are the validated data tables (kSolidHarmonicG, the
// index maps). Test-only - not part of the public API.

#include "internal/md_batch.hpp"
#include "internal/md_defs.hpp"
#include "internal/md_hermite.hpp"
#include "internal/md_tables_gen.hpp"
#include "qcx/integrals/boys.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace qcx::integrals::test {

namespace {

// The engine internals this path deliberately re-derives everything else
// from: the validated data tables and the pair data layout.
using internal::CartesianCount;
using internal::CartIndex;
using internal::Hermite3DCount;
using internal::Hermite3DIndex;
using internal::kCartesianIndices;
using internal::kSolidHarmonicG;
using internal::MdPairData;
using internal::MdPrimPair;

// Own 1D E recurrence (independent of BuildPerAxisTables): multiplying the
// expansion by (x-A) maps E'_t = (P-A) E_t + (t+1) E_{t+1} + E_{t-1}/(2p),
// applied per old coefficient: out[t] += shift e[t]; out[t-1] += t e[t];
// out[t+1] += e[t]/(2p).
std::vector<double> ETable1D(
    int iPower, int jPower, double shiftA, double shiftB, double p, int tMax) {
    // Sized by the pair's total t-range (la+lb): the fold reads t up to la+lb
    // for every cartesian pair, and entries beyond i+j are exactly zero
    // (the recurrence never writes past iPower+jPower).
    std::vector<double> e(tMax + 1, 0.0);
    e[0] = 1.0;

    for (int step = 0; step < iPower + jPower; ++step)
    {
        const double shift = step < iPower ? shiftA : shiftB;
        // next is sized to i+j+1, which may be smaller than e's tMax+1
        // range; e's tail past i+j is exactly zero, so stepping only over
        // next's range is exact - the old e.size() bound wrote past next
        // (ASan heap-buffer-overflow in the (p,p) mixed-power case, 2026-08-27).
        std::vector<double> next(iPower + jPower + 1, 0.0);

        for (std::size_t t = 0; t < next.size(); ++t)
        {
            next[t] += shift * e[t];

            if (t >= 1)
            {
                next[t - 1] += static_cast<double>(t) * e[t];
            }

            if (t + 1 < next.size())
            {
                next[t + 1] += e[t] / (2.0 * p);
            }
        }

        e = next;
    }

    // Zero-extend back to the fold's t-range: the caller reads t up to tMax
    // for every cartesian pair, and entries past i+j are exactly zero by
    // construction (the recurrence never writes past iPower+jPower). Without
    // the extension the table shrank to i+j+1 after the first step and the
    // caller's reads ran off the end - ASan heap-buffer-overflow, 2026-08-27.
    e.resize(tMax + 1, 0.0);

    return e;
}

// The folded angular transform of one primitive pair:
// T[(fa, fb)][t] for the 3D-Hermite indices |t| <= la+lb, angular-only
// (contraction rows applied by the caller), enumerated by explicit loops.
std::vector<double> FoldedTransform(const MdPairData& pair, std::size_t primIdx) {
    const int la = pair.la;
    const int lb = pair.lb;
    const int nHerm = Hermite3DCount(la + lb);
    const std::size_t nAngA = pair.nFuncsA / pair.rowsA;
    const std::size_t nAngB = pair.nFuncsB / pair.rowsB;
    const double p = pair.primPairs[primIdx].p;
    const double px = pair.primPairs[primIdx].px;
    const double py = pair.primPairs[primIdx].py;
    const double pz = pair.primPairs[primIdx].pz;
    std::vector<double> out(nAngA * nAngB * nHerm, 0.0);

    for (std::size_t fa = 0; fa < nAngA; ++fa)
    {
        for (std::size_t fb = 0; fb < nAngB; ++fb)
        {
            for (int tx = 0; tx <= la + lb; ++tx)
            {
                for (int ty = 0; ty <= la + lb - tx; ++ty)
                {
                    for (int tz = 0; tz <= la + lb - tx - ty; ++tz)
                    {
                        double value = 0.0;

                        for (int anga = 0; anga < CartesianCount(la); ++anga)
                        {
                            const CartIndex ca = kCartesianIndices[la][anga];
                            const double wa =
                                pair.isSphericalA
                                    ? kSolidHarmonicG[la][fa][anga]
                                    : (fa == static_cast<std::size_t>(anga) ? 1.0 : 0.0);

                            for (int angb = 0; angb < CartesianCount(lb); ++angb)
                            {
                                const CartIndex cb = kCartesianIndices[lb][angb];
                                const double wb =
                                    pair.isSphericalB
                                        ? kSolidHarmonicG[lb][fb][angb]
                                        : (fb == static_cast<std::size_t>(angb) ? 1.0 : 0.0);
                                const std::vector<double> exTab =
                                    ETable1D(ca.ix, cb.ix, px - pair.ax, px - pair.bx, p, la + lb);
                                const std::vector<double> eyTab =
                                    ETable1D(ca.iy, cb.iy, py - pair.ay, py - pair.by, p, la + lb);
                                const std::vector<double> ezTab =
                                    ETable1D(ca.iz, cb.iz, pz - pair.az, pz - pair.bz, p, la + lb);
                                value += wa * wb * exTab[tx] * eyTab[ty] * ezTab[tz];
                            }
                        }

                        out[(fa * nAngB + fb) * nHerm + Hermite3DIndex(tx, ty, tz)] = value;
                    }
                }
            }
        }
    }

    return out;
}

} // namespace

/// The full (ij|kl) function block of one canonical quartet through the
/// unfolded path, packed in the batch layout contract: rows
/// (f_j * n_i + f_i), columns (f_l * n_k + f_k), row-major.
inline std::vector<double> UnfoldedEriBlock(const MdPairData& bra, const MdPairData& ket) {
    const int lTotal = bra.la + bra.lb + ket.la + ket.lb;
    const std::size_t nHermBra = static_cast<std::size_t>(Hermite3DCount(bra.la + bra.lb));
    const std::size_t nHermKet = static_cast<std::size_t>(Hermite3DCount(ket.la + ket.lb));
    const std::size_t nAngBraA = bra.nFuncsA / bra.rowsA;
    const std::size_t nAngBraB = bra.nFuncsB / bra.rowsB;
    const std::size_t nAngKetA = ket.nFuncsA / ket.rowsA;
    const std::size_t nAngKetB = ket.nFuncsB / ket.rowsB;
    std::vector<double> block(bra.nFuncs * ket.nFuncs, 0.0);

    for (std::size_t primB = 0; primB < bra.primPairs.size(); ++primB)
    {
        const MdPrimPair& braPrim = bra.primPairs[primB];
        const std::vector<double> tBra = FoldedTransform(bra, primB);

        for (std::size_t primK = 0; primK < ket.primPairs.size(); ++primK)
        {
            const MdPrimPair& ketPrim = ket.primPairs[primK];
            const std::vector<double> tKet = FoldedTransform(ket, primK);
            const double p = braPrim.p;
            const double q = ketPrim.p;
            const double alpha = p * q / (p + q);
            const double dqx = braPrim.px - ketPrim.px;
            const double dqy = braPrim.py - ketPrim.py;
            const double dqz = braPrim.pz - ketPrim.pz;
            const double x = alpha * (dqx * dqx + dqy * dqy + dqz * dqz);
            const double prefactor = 2.0 * std::pow(std::numbers::pi, 2.5) /
                                     (p * q * std::sqrt(p + q)) * braPrim.prefactor *
                                     ketPrim.prefactor;

            // Seeds and the VRR over explicit (tx,ty,tz,m) loops:
            // slices[m][Hermite3DIndex(t)] = [t]^(m) for |t| + m <= lTotal.
            std::vector<double> seeds(lTotal + 1);
            qcx::integrals::BoysAllOrders(lTotal, x, seeds.data());
            const std::size_t sliceSize = static_cast<std::size_t>(Hermite3DCount(lTotal));
            std::vector<std::vector<double>> slices(lTotal + 1,
                                                    std::vector<double>(sliceSize, 0.0));

            for (int m = 0; m <= lTotal; ++m)
            {
                slices[m][0] = prefactor * seeds[m];
            }

            for (int m = lTotal - 1; m >= 0; --m)
            {
                for (int n = 1; n + m <= lTotal; ++n)
                {
                    for (int ty = 0; ty <= n; ++ty)
                    {
                        for (int tz = 0; tz <= n - ty; ++tz)
                        {
                            const int tx = n - ty - tz;
                            double value = 0.0;

                            if (tx >= 1)
                            {
                                value = -2.0 * alpha * dqx *
                                        slices[m + 1][Hermite3DIndex(tx - 1, ty, tz)];

                                if (tx >= 2)
                                {
                                    value += -2.0 * alpha * (tx - 1) *
                                             slices[m + 1][Hermite3DIndex(tx - 2, ty, tz)];
                                }
                            } else if (ty >= 1)
                            {
                                value = -2.0 * alpha * dqy *
                                        slices[m + 1][Hermite3DIndex(tx, ty - 1, tz)];

                                if (ty >= 2)
                                {
                                    value += -2.0 * alpha * (ty - 1) *
                                             slices[m + 1][Hermite3DIndex(tx, ty - 2, tz)];
                                }
                            } else
                            {
                                value = -2.0 * alpha * dqz *
                                        slices[m + 1][Hermite3DIndex(tx, ty, tz - 1)];

                                if (tz >= 2)
                                {
                                    value += -2.0 * alpha * (tz - 1) *
                                             slices[m + 1][Hermite3DIndex(tx, ty, tz - 2)];
                                }
                            }

                            slices[m][Hermite3DIndex(tx, ty, tz)] = value;
                        }
                    }
                }
            }

            // The transforms with the row weights:
            // block[row][col] += w_ab w_cd sum_p~ sum_q~ T_bra[(fa,fb),p~]
            // [p~+q~]^(0) T_ket[(fc,fd),q~].
            for (std::size_t rbA = 0; rbA < bra.rowsA; ++rbA)
            {
                for (std::size_t rbB = 0; rbB < bra.rowsB; ++rbB)
                {
                    const double wBra = bra.braWeights[primB][rbA * bra.rowsB + rbB];

                    for (std::size_t rkA = 0; rkA < ket.rowsA; ++rkA)
                    {
                        for (std::size_t rkB = 0; rkB < ket.rowsB; ++rkB)
                        {
                            const double wKet = ket.braWeights[primK][rkA * ket.rowsB + rkB];

                            for (std::size_t fa = 0; fa < nAngBraA; ++fa)
                            {
                                for (std::size_t fb = 0; fb < nAngBraB; ++fb)
                                {
                                    const std::size_t row =
                                        (rbB * nAngBraB + fb) * bra.nFuncsA + rbA * nAngBraA + fa;

                                    for (std::size_t fc = 0; fc < nAngKetA; ++fc)
                                    {
                                        for (std::size_t fd = 0; fd < nAngKetB; ++fd)
                                        {
                                            const std::size_t col =
                                                (rkB * nAngKetB + fd) * ket.nFuncsA +
                                                rkA * nAngKetA + fc;
                                            double value = 0.0;

                                            for (int px = 0; px <= bra.la + bra.lb; ++px)
                                            {
                                                for (int py = 0; py <= bra.la + bra.lb - px; ++py)
                                                {
                                                    for (int pz = 0;
                                                         pz <= bra.la + bra.lb - px - py;
                                                         ++pz)
                                                    {
                                                        const double tBraValue =
                                                            tBra[(fa * nAngBraB + fb) * nHermBra +
                                                                 Hermite3DIndex(px, py, pz)];

                                                        for (int qx = 0; qx <= ket.la + ket.lb;
                                                             ++qx)
                                                        {
                                                            for (int qy = 0;
                                                                 qy <= ket.la + ket.lb - qx;
                                                                 ++qy)
                                                            {
                                                                for (int qz = 0;
                                                                     qz <=
                                                                     ket.la + ket.lb - qx - qy;
                                                                     ++qz)
                                                                {
                                                                    const double tKetValue =
                                                                        tKet[(fc * nAngKetB + fd) *
                                                                                 nHermKet +
                                                                             Hermite3DIndex(
                                                                                 qx, qy, qz)];
                                                                    const double target =
                                                                        slices[0][Hermite3DIndex(
                                                                            px + qx,
                                                                            py + qy,
                                                                            pz + qz)];
                                                                    // d_Q^u [0] = (-1)^|u|
                                                                    // d_P^u [0]: the ket Hermite
                                                                    // derivatives carry the sign of
                                                                    // the ket degree (the
                                                                    // odd-ket-degree regression,
                                                                    // 2026-08-18).
                                                                    const double sign =
                                                                        ((qx + qy + qz) % 2 == 0)
                                                                            ? 1.0
                                                                            : -1.0;
                                                                    value += tBraValue * tKetValue *
                                                                             target * sign;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            block[row * ket.nFuncs + col] += wBra * wKet * value;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return block;
}

} // namespace qcx::integrals::test
