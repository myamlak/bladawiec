#pragma once

// Runtime Hermite expansion: the per-axis 1D E coefficients of the
// (x-A)^ix (x-B)^jx expansions, the 3D pair transforms with the spherical
// folds, and the contracted pair transform of the 4-step pipeline
// The engine is Cartesian-Hermite throughout; the
// real solid-harmonic transform (kSolidHarmonicG) folds once per transform
// matrix, never inside the recurrences. [McMurchie1978]
//
// The 1D coefficients satisfy (derived from (x-A) Lambda_t =
// Lambda_{t+1}/(2p) + t Lambda_{t-1} + (P-A)_x Lambda_t):
//     E_t^{(i+1,j)} = (P-A)_x E_t^{(i,j)} + (t+1) E_{t+1}^{(i,j)}
//                     + E_{t-1}^{(i,j)} / (2p)
//     E_t^{(i,j+1)} = (P-B)_x E_t^{(i,j)} + (t+1) E_{t+1}^{(i,j)}
//                     + E_{t-1}^{(i,j)} / (2p)
// with E^{(0,0)} = delta_{t,0}; entries outside 0..i+j are zero, and
// E_t^{(i,j)} vanishes unless t = i+j (mod 2) (kParityMask).

#include "md_defs.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace qcx::integrals::internal {

/// Per-primitive-pair E-table scratch sizing: per-pair (la, lb) instead of
/// the fixed worst-case array. The fixed table held
/// (kMaxShellL+3)(kMaxShellL+1)(2*kMaxShellL+3) = 945 doubles per axis,
/// 3 axes per primitive pair, every primitive pair of every shell pair -
/// 204 KB per shell pair, 3.32 GB for c60_sto3g's 16,290 pairs per builder
/// Create (the incident of 2026-08-29). The per-pair size is now
/// exact: (la+1)(lb+1)(la+lb+1) doubles (sto-3g pairs need 1). The kinetic
/// builders fill (la+2, lb) tables (the raised bra rows of the E-shift
/// identity), covered by passing la+2 to the constructor; the recurrence
/// writes the same values at the same indices, so the tables are
/// bit-identical to the fixed-array build.
/// The folded angular-only 3D E slice of FoldPairETable: nCartA x nCartB x
/// Hermite3DCount(la + lb), maximized at la = lb = kMaxShellL =
/// CartesianCount(6)^2 * Hermite3DCount(12) = 356720 elements. The buffer is
/// thread_local: the 1e builders run inside a ParallelFor (one_electron.cpp),
/// and ~2.7 MB per buffer is too large for the per-call stack.
inline constexpr std::size_t kMaxFoldElements =
    static_cast<std::size_t>(CartesianCount(kMaxShellL)) * CartesianCount(kMaxShellL) *
    Hermite3DCount(2 * kMaxShellL);

/// One per-axis 1D E table: E[(ix, jx)][t], flattened
/// [(ix*(lb+1)+jx)*(la+lb+1)+t]. The storage is re-fittable in place
/// (Reset) so a rebuilt table keeps its vector instead of allocating a new
/// one - the pair builder's reuse path (BuildContractedPairTransform).
class PerAxisETable {
public:
    /// The empty table (MdPrimPair construction sites default it; the aux
    /// kets of the RI path never fill the tables).
    PerAxisETable() : PerAxisETable(0, 0) {}

    /// The table for angular momenta (la, lb): (la+1)(lb+1)(la+lb+1)
    /// doubles, the exact rows the FillPerAxisTable recurrence writes.
    PerAxisETable(int la, int lb) {
        Reset(la, lb);
    }

    /// Re-fits the table to (la, lb) WITHOUT releasing its storage: the
    /// size reset the reuse path needs (the fill is a fresh assign of the
    /// same extent, so a matching re-fit neither allocates nor frees).
    /// Every cell is set to 0.0 - exactly what a freshly constructed
    /// table's vector holds - because the recurrence below writes only the
    /// cells with t <= i + j: the cells past the diagonal read zero on a
    /// constructed table and must read zero on a re-fitted one too (the
    /// bit-identity contract of the reuse).
    void Reset(int la, int lb) {
        _lb = lb;
        _tMax = la + lb;
        _values.assign(static_cast<std::size_t>(la + 1) * (lb + 1) * (_tMax + 1), 0.0);
    }

    /// Coefficient E_t^{(ix,jx)}.
    double& At(int ix, int jx, int t) noexcept {
        return _values[Index(ix, jx, t)];
    }

    double At(int ix, int jx, int t) const noexcept {
        return _values[Index(ix, jx, t)];
    }

private:
    std::size_t Index(int ix, int jx, int t) const noexcept {
        return static_cast<std::size_t>((ix * (_lb + 1) + jx) * (_tMax + 1) + t);
    }

    int _lb;
    int _tMax;
    std::vector<double> _values;
};

/// Fills one per-axis 1D E table by the recurrence above. \p shiftA[d] is
/// (P-A) along axis d, \p shiftB[d] is (P-B), \p p the pair exponent. The
/// table dims (la, lb) are the caller's choice - the 1e kinetic builders
/// fill (la+2, lb) tables (the raised bra rows of the E-shift identity).
inline void FillPerAxisTable(int la,
                             int lb,
                             const std::array<double, 3>& shiftA,
                             const std::array<double, 3>& shiftB,
                             double p,
                             int axis,
                             PerAxisETable& table) {
    assert(la <= kMaxShellL + 2 && lb <= kMaxShellL);
    const double invTwoP = 0.5 / p;
    table.At(0, 0, 0) = 1.0;

    // (i, j) in row-major order: every (i, j) reads only (i, j-1) and
    // (i-1, j), both already filled.
    for (int i = 0; i <= la; ++i)
    {
        for (int j = 0; j <= lb; ++j)
        {
            if (i == 0 && j == 0)
            {
                continue;
            }

            const int parentI = j > 0 ? i : i - 1;
            const int parentJ = j > 0 ? j - 1 : j;
            const double shift = j > 0 ? shiftB[axis] : shiftA[axis];

            for (int t = 0; t <= i + j; ++t)
            {
                double value = shift * table.At(parentI, parentJ, t);

                if (t + 1 <= parentI + parentJ)
                {
                    value += static_cast<double>(t + 1) * table.At(parentI, parentJ, t + 1);
                }

                if (t - 1 >= 0)
                {
                    value += invTwoP * table.At(parentI, parentJ, t - 1);
                }

                table.At(i, j, t) = value;
            }
        }
    }
}

/// Builds the per-axis 1D E tables of one primitive pair by the recurrence
/// above, into caller-owned tables: each table is re-fitted to (la, lb) by
/// Reset (capacity kept, every cell zeroed) and filled. The pair builder's
/// reuse path calls this on the pair's own tables, so the repeat build of a
/// pair neither allocates nor frees the three per-axis vectors. The values
/// are bit-identical to the returned-array form's - the same recurrence over
/// the same zeroed extents, no other read. \p shiftA[d] is (P-A) along axis
/// d, \p shiftB[d] is (P-B), \p p the pair exponent.
inline void BuildPerAxisTables(int la,
                               int lb,
                               const std::array<double, 3>& shiftA,
                               const std::array<double, 3>& shiftB,
                               double p,
                               std::array<PerAxisETable, 3>& tables) {
    for (int axis = 0; axis < 3; ++axis)
    {
        PerAxisETable& table = tables[static_cast<std::size_t>(axis)];
        table.Reset(la, lb);
        FillPerAxisTable(la, lb, shiftA, shiftB, p, axis, table);
    }
}

/// Builds the per-axis 1D E tables of one primitive pair by the recurrence
/// above. \p shiftA[d] is (P-A) along axis d, \p shiftB[d] is (P-B), \p p
/// the pair exponent.
inline std::array<PerAxisETable, 3> BuildPerAxisTables(int la,
                                                       int lb,
                                                       const std::array<double, 3>& shiftA,
                                                       const std::array<double, 3>& shiftB,
                                                       double p) {
    std::array<PerAxisETable, 3> tables;
    BuildPerAxisTables(la, lb, shiftA, shiftB, p, tables);

    return tables;
}

/// The contraction rows of one shell with normalized coefficients
/// (d = c * (2a/pi)^(3/4), the standard radial convention).
struct MdShellContractions {
    int angularMomentum; ///< l.
    bool isSpherical; ///< Spherical flag.
    std::size_t rows; ///< Contraction rows.
    std::vector<double> exponents; ///< One per primitive.
    std::vector<std::vector<double>> normalized; ///< [row][primitive].
};

/// Builds the folded angular-only 3D E slice E[(fa, fb), t] of one
/// primitive pair from its per-axis tables: the per-axis products with the
/// spherical folds applied (identity for Cartesian shells), rows excluded.
/// Shared by the transform construction (md_batch.cpp) and the 1e builders
/// (md_one_electron.hpp).
inline void FoldPairETable(int la,
                           int lb,
                           const std::array<PerAxisETable, 3>& tables,
                           bool isSphericalA,
                           bool isSphericalB,
                           int nHerm,
                           std::vector<double>& out) {
    assert(la <= kMaxShellL && lb <= kMaxShellL && nHerm <= Hermite3DCount(2 * kMaxShellL));
    const int nCartA = CartesianCount(la);
    const int nCartB = CartesianCount(lb);
    const int nAngA = isSphericalA ? SphericalCount(la) : nCartA;
    const int nAngB = isSphericalB ? SphericalCount(lb) : nCartB;
    out.assign(static_cast<std::size_t>(nAngA) * nAngB * nHerm, 0.0);
    // Fixed-size scratch: thread_local so the ~2.7 MB buffers neither
    // hit the per-call stack nor allocate per pair, and stay safe under the
    // ParallelFor of the 1e builders.
    static thread_local std::array<double, kMaxFoldElements> e3d;
    static thread_local std::array<double, kMaxFoldElements> mid;
    const PerAxisETable& ex = tables[0];
    const PerAxisETable& ey = tables[1];
    const PerAxisETable& ez = tables[2];

    for (int anga = 0; anga < nCartA; ++anga)
    {
        const CartIndex ca = kCartesianIndices[la][anga];

        for (int angb = 0; angb < nCartB; ++angb)
        {
            const CartIndex cb = kCartesianIndices[lb][angb];

            for (int tx = 0; tx <= la + lb; ++tx)
            {
                for (int ty = 0; ty <= la + lb - tx; ++ty)
                {
                    for (int tz = 0; tz <= la + lb - tx - ty; ++tz)
                    {
                        const double value = ex.At(ca.ix, cb.ix, tx) * ey.At(ca.iy, cb.iy, ty) *
                                             ez.At(ca.iz, cb.iz, tz);
                        e3d[(static_cast<std::size_t>(anga) * nCartB + angb) * nHerm +
                            Hermite3DIndex(tx, ty, tz)] = value;
                    }
                }
            }
        }
    }

    // Fold shell b, then shell a.
    for (int anga = 0; anga < nCartA; ++anga)
    {
        for (int fb = 0; fb < nAngB; ++fb)
        {
            for (int t = 0; t < nHerm; ++t)
            {
                double value = 0.0;

                if (isSphericalB)
                {
                    for (int angb = 0; angb < nCartB; ++angb)
                    {
                        value += kSolidHarmonicG[lb][fb][angb] *
                                 e3d[(static_cast<std::size_t>(anga) * nCartB + angb) * nHerm + t];
                    }
                } else
                {
                    value = e3d[(static_cast<std::size_t>(anga) * nCartB + fb) * nHerm + t];
                }

                mid[(static_cast<std::size_t>(anga) * nAngB + fb) * nHerm + t] = value;
            }
        }
    }

    for (int fa = 0; fa < nAngA; ++fa)
    {
        for (int fb = 0; fb < nAngB; ++fb)
        {
            for (int t = 0; t < nHerm; ++t)
            {
                double value = 0.0;

                if (isSphericalA)
                {
                    for (int anga = 0; anga < nCartA; ++anga)
                    {
                        value += kSolidHarmonicG[la][fa][anga] *
                                 mid[(static_cast<std::size_t>(anga) * nAngB + fb) * nHerm + t];
                    }
                } else
                {
                    value = mid[(static_cast<std::size_t>(fa) * nAngB + fb) * nHerm + t];
                }

                out[(static_cast<std::size_t>(fa) * nAngB + fb) * nHerm + t] = value;
            }
        }
    }
}

} // namespace qcx::integrals::internal
