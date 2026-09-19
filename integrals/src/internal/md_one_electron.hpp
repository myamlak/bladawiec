#pragma once

// General-l one-electron kernels of the MD machinery (runtime l, no class
// instantiations): overlap and kinetic per primitive pair as small
// contractions of the Hermite E coefficients, nuclear attraction with early
// charge summation. [Helgaker2000] [McMurchie1978]
//
// Identities (E tables of md_hermite.hpp):
//   S_ab = sum_ab d_a d_b (pi/p)^(3/2) E_ab E_0
//   T_ab = sum_ab d_a d_b (pi/p)^(3/2) E_ab
//          [(3 + 2|i|) a E_0
//           - 2a^2 (E_0^(ix+2) + E_0^(iy+2) + E_0^(iz+2))
//           - 1/2 (ix(ix-1) E_0^(ix-2) + iy(iy-1) E_0^(iy-2)
//                  + iz(iz-1) E_0^(iz-2))]
// (-1/2 del^2 (P phi_a) = (3a - 2a^2 |r-A|^2) P phi_a + 2a (r-A).grad P
// phi_a - 1/2 (del^2 P) phi_a with P the bra polynomial and phi_a =
// e^{-a|r-A|^2}. At zero shift (r-A).grad P = |i| P - the 2a|i| term
// folds into the E_0 coefficient, there is no E_0^(ix-1) correction - the
// |r-A|^2 terms raise the bra cartesian index by 2 per axis (the
// (la+2, lb) tables at the raised rows, t = 0) and the del^2 P terms
// lower it by 2 (the original pair tables).)
//   V_ab = -sum_ab d_a d_b (2pi/p) E_ab sum_t E_t sum_c Z_c [t]^(0)_c
// with the 1-center VRR [t+e]^(m) = -2p (P-C)_dir [t]^(m+1)
// - 2p t_dir [t-e]^(m+1) (the ERI VRR with a -> p, (P-Q) -> (P-C)).
//
// Early charge summation: the VRR coefficients carry (P-C), so the
// recurrence runs PER NUCLEUS; the charge sum folds into the [t]^(0)
// targets afterwards, and the bra contraction (the E fold) runs ONCE.

#include "md_batch.hpp"
#include "md_boys.hpp"
#include "md_defs.hpp"
#include "md_hermite.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace qcx::integrals::internal {

/// The overlap block of one canonical pair, (nFuncsA x nFuncsB) row-major.
inline void BuildOverlapPair(const MdPairData& pair, double* block) {
    const int nHerm = Hermite3DCount(pair.la + pair.lb);
    const std::size_t nAngA = pair.nFuncsA / pair.rowsA;
    const std::size_t nAngB = pair.nFuncsB / pair.rowsB;
    std::fill(block, block + pair.nFuncs, 0.0);
    std::vector<double> folded;

    for (std::size_t primIdx = 0; primIdx < pair.primPairs.size(); ++primIdx)
    {
        const MdPrimPair& prim = pair.primPairs[primIdx];
        const double pref = std::pow(std::numbers::pi / prim.p, 1.5) * prim.prefactor;
        FoldPairETable(pair.la,
                       pair.lb,
                       prim.perAxisTables,
                       pair.isSphericalA,
                       pair.isSphericalB,
                       nHerm,
                       folded);

        for (std::size_t rowA = 0; rowA < pair.rowsA; ++rowA)
        {
            for (std::size_t rowB = 0; rowB < pair.rowsB; ++rowB)
            {
                const double weight = pair.braWeights[primIdx][rowA * pair.rowsB + rowB];

                for (std::size_t fb = 0; fb < nAngB; ++fb)
                {
                    for (std::size_t fa = 0; fa < nAngA; ++fa)
                    {
                        // E at t = 0 (Hermite3DIndex(0,0,0) = 0).
                        block[(rowA * nAngA + fa) * pair.nFuncsB + rowB * nAngB + fb] +=
                            weight * pref * folded[(fa * nAngB + fb) * nHerm];
                    }
                }
            }
        }
    }
}

/// The kinetic-energy block of one canonical pair, (nFuncsA x nFuncsB)
/// row-major.
inline void BuildKineticPair(const MdPairData& pair, double* block) {
    const int nCartA = CartesianCount(pair.la);
    const int nCartB = CartesianCount(pair.lb);
    const std::size_t nAngA = pair.nFuncsA / pair.rowsA;
    const std::size_t nAngB = pair.nFuncsB / pair.rowsB;
    std::fill(block, block + pair.nFuncs, 0.0);

    for (std::size_t primIdx = 0; primIdx < pair.primPairs.size(); ++primIdx)
    {
        const MdPrimPair& prim = pair.primPairs[primIdx];
        const double pref = std::pow(std::numbers::pi / prim.p, 1.5) * prim.prefactor;
        const double a = prim.exponentA;
        // -1/2 del^2 phi_a = (3a - 2a^2 |r-A|^2) phi_a: each |r-A|^2 factor
        // raises the bra cartesian index by 2 along one axis, so the three
        // second terms are the t = 0 slots of the (la+2, lb) pair tables at
        // the raised bra rows (the E-shift identity of the header comment).
        const std::array<double, 3> shiftA = {
            prim.px - pair.ax, prim.py - pair.ay, prim.pz - pair.az};
        const std::array<double, 3> shiftB = {
            prim.px - pair.bx, prim.py - pair.by, prim.pz - pair.bz};
        const auto raised = BuildPerAxisTables(pair.la + 2, pair.lb, shiftA, shiftB, prim.p);
        const PerAxisETable& ex = raised[0];
        const PerAxisETable& ey = raised[1];
        const PerAxisETable& ez = raised[2];
        const PerAxisETable& bx = prim.perAxisTables[0];
        const PerAxisETable& by = prim.perAxisTables[1];
        const PerAxisETable& bz = prim.perAxisTables[2];
        // Per cartesian bra row (anga), the ket-folded kinetic terms.
        std::vector<double> perCart(static_cast<std::size_t>(nCartA) * nAngB);

        for (int anga = 0; anga < nCartA; ++anga)
        {
            const CartIndex ca = kCartesianIndices[pair.la][anga];

            for (int angb = 0; angb < nCartB; ++angb)
            {
                const CartIndex cb = kCartesianIndices[pair.lb][angb];
                // -1/2 del^2 (P phi_a) = (3a - 2a^2 |r-A|^2) P phi_a
                //   + 2a (r-A).grad P phi_a - 1/2 (del^2 P) phi_a,
                // each term an E_0 read: |r-A|^2 raises the bra index by 2
                // (the (la+2, lb) tables), grad P lowers it by 1 and
                // del^2 P by 2 (the original pair tables).
                const double e0 =
                    ex.At(ca.ix, cb.ix, 0) * ey.At(ca.iy, cb.iy, 0) * ez.At(ca.iz, cb.iz, 0);
                const double e2x =
                    ex.At(ca.ix + 2, cb.ix, 0) * ey.At(ca.iy, cb.iy, 0) * ez.At(ca.iz, cb.iz, 0);
                const double e2y =
                    ex.At(ca.ix, cb.ix, 0) * ey.At(ca.iy + 2, cb.iy, 0) * ez.At(ca.iz, cb.iz, 0);
                const double e2z =
                    ex.At(ca.ix, cb.ix, 0) * ey.At(ca.iy, cb.iy, 0) * ez.At(ca.iz + 2, cb.iz, 0);
                const double e2mx = ca.ix >= 2 ? bx.At(ca.ix - 2, cb.ix, 0) *
                                                     by.At(ca.iy, cb.iy, 0) * bz.At(ca.iz, cb.iz, 0)
                                               : 0.0;
                const double e2my = ca.iy >= 2
                                        ? bx.At(ca.ix, cb.ix, 0) * by.At(ca.iy - 2, cb.iy, 0) *
                                              bz.At(ca.iz, cb.iz, 0)
                                        : 0.0;
                const double e2mz = ca.iz >= 2 ? bx.At(ca.ix, cb.ix, 0) * by.At(ca.iy, cb.iy, 0) *
                                                     bz.At(ca.iz - 2, cb.iz, 0)
                                               : 0.0;
                const double value =
                    (3.0 + 2.0 * static_cast<double>(ca.ix + ca.iy + ca.iz)) * a * e0 -
                    2.0 * a * a * (e2x + e2y + e2z) -
                    0.5 * (static_cast<double>(ca.ix * (ca.ix - 1)) * e2mx +
                           static_cast<double>(ca.iy * (ca.iy - 1)) * e2my +
                           static_cast<double>(ca.iz * (ca.iz - 1)) * e2mz);

                for (int fb = 0; fb < static_cast<int>(nAngB); ++fb)
                {
                    const double gb = pair.isSphericalB ? kSolidHarmonicG[pair.lb][fb][angb]
                                                        : (fb == angb ? 1.0 : 0.0);

                    if (gb != 0.0)
                    {
                        perCart[static_cast<std::size_t>(anga) * nAngB + fb] += gb * value;
                    }
                }
            }
        }

        for (std::size_t rowA = 0; rowA < pair.rowsA; ++rowA)
        {
            for (std::size_t rowB = 0; rowB < pair.rowsB; ++rowB)
            {
                const double weight = pair.braWeights[primIdx][rowA * pair.rowsB + rowB];

                for (std::size_t fb = 0; fb < nAngB; ++fb)
                {
                    for (std::size_t fa = 0; fa < nAngA; ++fa)
                    {
                        double value = 0.0;

                        if (pair.isSphericalA)
                        {
                            for (int anga = 0; anga < nCartA; ++anga)
                            {
                                value += kSolidHarmonicG[pair.la][fa][anga] *
                                         perCart[static_cast<std::size_t>(anga) * nAngB + fb];
                            }
                        } else
                        {
                            value = perCart[static_cast<std::size_t>(fa) * nAngB + fb];
                        }

                        block[(rowA * nAngA + fa) * pair.nFuncsB + rowB * nAngB + fb] +=
                            weight * pref * value;
                    }
                }
            }
        }
    }
}

/// The nuclear-attraction block of one canonical pair with early charge
/// summation, (nFuncsA x nFuncsB) row-major.
/// \param charges Nuclear charges Z (one per atom).
/// \param centers Atom centers (Bohr).
inline void BuildNuclearPair(const MdPairData& pair,
                             const std::vector<double>& charges,
                             const std::vector<Eigen::Vector3d>& centers,
                             double* block) {
    const int l1 = pair.la + pair.lb;
    const int nHerm = Hermite3DCount(l1);
    const std::size_t nAngA = pair.nFuncsA / pair.rowsA;
    const std::size_t nAngB = pair.nFuncsB / pair.rowsB;
    std::fill(block, block + pair.nFuncs, 0.0);
    std::vector<double> folded;
    std::vector<double> g(static_cast<std::size_t>(nHerm));
    std::vector<double> slice(static_cast<std::size_t>(nHerm));
    std::vector<double> prevSlice(static_cast<std::size_t>(nHerm));
    std::vector<double> seeds(static_cast<std::size_t>(l1 + 1));

    for (std::size_t primIdx = 0; primIdx < pair.primPairs.size(); ++primIdx)
    {
        const MdPrimPair& prim = pair.primPairs[primIdx];
        const double pref = 2.0 * std::numbers::pi / prim.p * prim.prefactor;
        std::fill(g.begin(), g.end(), 0.0);

        // The VRR runs per nucleus (its coefficients carry (P-C)); the
        // charge sum folds into the [t]^(0) targets.
        for (std::size_t c = 0; c < charges.size(); ++c)
        {
            const double dcx = prim.px - centers[c].x();
            const double dcy = prim.py - centers[c].y();
            const double dcz = prim.pz - centers[c].z();
            const double x = prim.p * (dcx * dcx + dcy * dcy + dcz * dcz);
            double tmp[1 + kMaxBoysOrder];
            BoysBatch(l1, x, tmp);

            for (int m = 0; m <= l1; ++m)
            {
                seeds[m] = charges[c] * pref * tmp[m];
            }

            const double c2 = -2.0 * prim.p;
            const double c1x = -2.0 * prim.p * dcx;
            const double c1y = -2.0 * prim.p * dcy;
            const double c1z = -2.0 * prim.p * dcz;
            g[0] += seeds[0];

            for (int sliceT = 1; sliceT <= l1; ++sliceT)
            {
                slice[0] = seeds[sliceT];

                for (int n = 1; n <= sliceT; ++n)
                {
                    const int off = kH2Prefix[n];
                    const int off1 = kH2Prefix[n - 1];
                    const int off2 = n >= 2 ? kH2Prefix[n - 2] : 0;

                    for (int ty = 0; ty <= n; ++ty)
                    {
                        for (int tz = 0; tz <= n - ty; ++tz)
                        {
                            const int tx = n - ty - tz;
                            const int sub = SubIndex3(ty, tz, n);
                            double value = 0.0;

                            if (tx >= 1)
                            {
                                value = c1x * slice[off1 + SubIndex3(ty, tz, n - 1)];

                                if (tx >= 2)
                                {
                                    value += c2 * static_cast<double>(tx - 1) *
                                             prevSlice[off2 + SubIndex3(ty, tz, n - 2)];
                                }
                            } else if (ty >= 1)
                            {
                                value = c1y * slice[off1 + SubIndex3(ty - 1, tz, n - 1)];

                                if (ty >= 2)
                                {
                                    value += c2 * static_cast<double>(ty - 1) *
                                             prevSlice[off2 + SubIndex3(ty - 2, tz, n - 2)];
                                }
                            } else
                            {
                                value = c1z * slice[off1 + SubIndex3(ty, tz - 1, n - 1)];

                                if (tz >= 2)
                                {
                                    value += c2 * static_cast<double>(tz - 1) *
                                             prevSlice[off2 + SubIndex3(ty, tz - 2, n - 2)];
                                }
                            }

                            slice[off + sub] = value;
                        }
                    }
                }

                // Tier sliceT of slice sliceT holds the [t]^(0) targets.
                const int off = kH2Prefix[sliceT];

                for (int sub = 0; sub < Hermite2DCount(sliceT); ++sub)
                {
                    g[off + sub] += slice[off + sub];
                }

                // The second VRR source [t-2e]^(m+1) lives in the previous
                // slice (md_vrr.hpp documents the same fix).
                slice.swap(prevSlice);
            }
        }

        // One bra contraction after the charge summation.
        FoldPairETable(pair.la,
                       pair.lb,
                       prim.perAxisTables,
                       pair.isSphericalA,
                       pair.isSphericalB,
                       nHerm,
                       folded);

        for (std::size_t rowA = 0; rowA < pair.rowsA; ++rowA)
        {
            for (std::size_t rowB = 0; rowB < pair.rowsB; ++rowB)
            {
                const double weight = pair.braWeights[primIdx][rowA * pair.rowsB + rowB];

                for (std::size_t fb = 0; fb < nAngB; ++fb)
                {
                    for (std::size_t fa = 0; fa < nAngA; ++fa)
                    {
                        double value = 0.0;

                        for (int t = 0; t < nHerm; ++t)
                        {
                            value += folded[(fa * nAngB + fb) * nHerm + t] * g[t];
                        }

                        block[(rowA * nAngA + fa) * pair.nFuncsB + rowB * nAngB + fb] -=
                            weight * value;
                    }
                }
            }
        }
    }
}

/// The dipole block of one canonical pair in one Cartesian direction,
/// (nFuncsA x nFuncsB) row-major: <u| r_k - R_k |v> through the Hermite
/// identity u_k Lambda_t = Lambda_{t+e_k}/(2p) + t_k Lambda_{t-e_k}
/// [Helgaker2000 Sec 9.5.1]: (pi/p)^{3/2} [d_k E_0 + E_{e_k}] with
/// d_k = P_k - R_k and E_t the folded pair table (the 1/(2p) of the
/// identity is already inside E_{e_k} - the fold recurrence of
/// md_hermite.hpp). E_{e_k} vanishes for la + lb = 0 (the fold of an s-s
/// pair is pure t = 0), where the entry would be out of the table range.
/// \param axis The Cartesian direction (0 = x, 1 = y, 2 = z).
/// \param origin The reference point R (Bohr) the operator is measured
/// from; the caller's documented convention.
inline void BuildDipolePair(const MdPairData& pair,
                            int axis,
                            const std::array<double, 3>& origin,
                            double* block) {
    const int nHerm = Hermite3DCount(pair.la + pair.lb);
    const std::size_t nAngA = pair.nFuncsA / pair.rowsA;
    const std::size_t nAngB = pair.nFuncsB / pair.rowsB;
    std::fill(block, block + pair.nFuncs, 0.0);
    std::vector<double> folded;
    const int tUnit = axis == 0   ? Hermite3DIndex(1, 0, 0)
                      : axis == 1 ? Hermite3DIndex(0, 1, 0)
                                  : Hermite3DIndex(0, 0, 1);

    for (std::size_t primIdx = 0; primIdx < pair.primPairs.size(); ++primIdx)
    {
        const MdPrimPair& prim = pair.primPairs[primIdx];
        const double pref = std::pow(std::numbers::pi / prim.p, 1.5) * prim.prefactor;
        FoldPairETable(pair.la,
                       pair.lb,
                       prim.perAxisTables,
                       pair.isSphericalA,
                       pair.isSphericalB,
                       nHerm,
                       folded);
        const double dK = axis == 0   ? prim.px - origin[0]
                          : axis == 1 ? prim.py - origin[1]
                                      : prim.pz - origin[2];

        for (std::size_t rowA = 0; rowA < pair.rowsA; ++rowA)
        {
            for (std::size_t rowB = 0; rowB < pair.rowsB; ++rowB)
            {
                const double weight = pair.braWeights[primIdx][rowA * pair.rowsB + rowB];

                for (std::size_t fb = 0; fb < nAngB; ++fb)
                {
                    for (std::size_t fa = 0; fa < nAngA; ++fa)
                    {
                        const std::size_t base = (fa * nAngB + fb) * nHerm;
                        const double eK = pair.la + pair.lb >= 1 ? folded[base + tUnit] : 0.0;
                        block[(rowA * nAngA + fa) * pair.nFuncsB + rowB * nAngB + fb] +=
                            weight * pref * (dK * folded[base] + eK);
                    }
                }
            }
        }
    }
}

/// The quadrupole block of one canonical pair, (nFuncsA x nFuncsB)
/// row-major: <u| (r_k - R_k)(r_l - R_l) |v> through the same Hermite
/// identity as BuildDipolePair [Helgaker2000 Sec 9.5.1]:
///   k == l: (pi/p)^{3/2} [(1/(2p) + d_k^2) E_0 + 2 d_k E_{e_k} + 2 E_{2e_k}]
///   k != l: (pi/p)^{3/2} [E_{e_k+e_l} + d_k E_{e_l} + d_l E_{e_k}
///                         + d_k d_l E_0]
/// with d_k = P_k - R_k. The second-order Hermite terms vanish for
/// la + lb < 2 (parity/degree: the fold of a p-s pair has no t = 2
/// component), where the entries would be out of the table range; the
/// first-order terms vanish for la + lb = 0 likewise.
/// \param axisK, \param axisL The Cartesian directions (0 = x, 1 = y,
/// 2 = z); both orders of an off-diagonal pair integrate identically, the
/// caller builds the upper triangle.
/// \param origin The reference point R (Bohr); see BuildDipolePair.
inline void BuildQuadrupolePair(const MdPairData& pair,
                                int axisK,
                                int axisL,
                                const std::array<double, 3>& origin,
                                double* block) {
    const int nHerm = Hermite3DCount(pair.la + pair.lb);
    const std::size_t nAngA = pair.nFuncsA / pair.rowsA;
    const std::size_t nAngB = pair.nFuncsB / pair.rowsB;
    std::fill(block, block + pair.nFuncs, 0.0);
    std::vector<double> folded;
    const bool hasFirstOrder = pair.la + pair.lb >= 1;
    const bool hasSecondOrder = pair.la + pair.lb >= 2;
    const int tK = axisK == 0   ? Hermite3DIndex(1, 0, 0)
                   : axisK == 1 ? Hermite3DIndex(0, 1, 0)
                                : Hermite3DIndex(0, 0, 1);
    const int tL = axisL == 0   ? Hermite3DIndex(1, 0, 0)
                   : axisL == 1 ? Hermite3DIndex(0, 1, 0)
                                : Hermite3DIndex(0, 0, 1);
    const int tKL = Hermite3DIndex(axisK == 0 || axisL == 0 ? 1 : 0,
                                   axisK == 1 || axisL == 1 ? 1 : 0,
                                   axisK == 2 || axisL == 2 ? 1 : 0);
    const int tKK = Hermite3DIndex(axisK == 0 ? 2 : 0, axisK == 1 ? 2 : 0, axisK == 2 ? 2 : 0);

    for (std::size_t primIdx = 0; primIdx < pair.primPairs.size(); ++primIdx)
    {
        const MdPrimPair& prim = pair.primPairs[primIdx];
        const double pref = std::pow(std::numbers::pi / prim.p, 1.5) * prim.prefactor;
        FoldPairETable(pair.la,
                       pair.lb,
                       prim.perAxisTables,
                       pair.isSphericalA,
                       pair.isSphericalB,
                       nHerm,
                       folded);
        const double dK = axisK == 0   ? prim.px - origin[0]
                          : axisK == 1 ? prim.py - origin[1]
                                       : prim.pz - origin[2];
        const double dL = axisL == 0   ? prim.px - origin[0]
                          : axisL == 1 ? prim.py - origin[1]
                                       : prim.pz - origin[2];

        for (std::size_t rowA = 0; rowA < pair.rowsA; ++rowA)
        {
            for (std::size_t rowB = 0; rowB < pair.rowsB; ++rowB)
            {
                const double weight = pair.braWeights[primIdx][rowA * pair.rowsB + rowB];

                for (std::size_t fb = 0; fb < nAngB; ++fb)
                {
                    for (std::size_t fa = 0; fa < nAngA; ++fa)
                    {
                        const std::size_t base = (fa * nAngB + fb) * nHerm;
                        const double eK = hasFirstOrder ? folded[base + tK] : 0.0;
                        double value;

                        if (axisK == axisL)
                        {
                            const double eKK = hasSecondOrder ? folded[base + tKK] : 0.0;
                            value =
                                (0.5 / prim.p + dK * dK) * folded[base] + 2.0 * dK * eK + 2.0 * eKK;
                        } else
                        {
                            const double eKL = hasSecondOrder ? folded[base + tKL] : 0.0;
                            const double eL = hasFirstOrder ? folded[base + tL] : 0.0;
                            value = eKL + dK * eL + dL * eK + dK * dL * folded[base];
                        }

                        block[(rowA * nAngA + fa) * pair.nFuncsB + rowB * nAngB + fb] +=
                            weight * pref * value;
                    }
                }
            }
        }
    }
}

} // namespace qcx::integrals::internal
