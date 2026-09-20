// The general-L multipole machinery: the moment table at L_mult up to
// kQfmmMaxLMult, the density-weighted leaf aggregation, the M2M/M2L/L2L
// passes over the VERIFIED emitted translation tables (qfmm_tables_gen.hpp -
// generated, never hand-derived), and the far-field J accumulation in the
// direct builder's 2J convention. The monopole gate is the L_mult = 0
// degenerate case of the same seams: the (0, 0) moment block IS the overlap
// block through the trusted overlap engine, and the (0, 0) rows of the
// emitted tables are exactly {1.0, 0, ...} with kQfmmM2LDenomPower[0][0] = 1,
// so the M2M reduces to the exact 1.0 coefficient, the M2L to Coulomb's law
// at the node centers, and the L2L to the identity - every pass degrades to
// the monopole code path bit-identically (the tests pin this). The moments
// themselves are the "add one angular momentum" trick: the raised
// (la + lMult, lb)
// per-axis Hermite E tables give, at their t = 0 slot, the overlap-type
// integrals of the pair with the bra raised by (p, q, r); the Cartesian
// moments about the pair center follow by the binomial shift
// (x - P)^a = Σ_p C(a, p) (x - A)^p (A - P)^{a-p}, contracted to the
// Racah real solid harmonics by kSolidHarmonicG (the same transform the
// spherical folds use).

#include "internal/qfmm_multipole.hpp"

#include "internal/qfmm_tables_gen.hpp"
#include "qcx/backend/cpu_backend.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <utility>
#include <vector>

namespace qcx::integrals::internal {

namespace {

/// The (l, m) -> flat index of the moment blocks: idx = l^2 + l + m (m in
/// -l..+l), the packing of qfmm_tables_gen.hpp's kQfmmMomentCount layout.
inline constexpr int MomentIndex(int l, int m) noexcept {
    return l * l + l + m;
}

/// C(n, k) for the moment-order bound n <= kQfmmMaxLMult (the binomials of
/// the angular-momentum shift: the a, b, c degrees run to the multipole order).
inline constexpr std::array<std::array<int, kQfmmMaxLMult + 1>, kQfmmMaxLMult + 1>
MakeBinomialTable() noexcept {
    std::array<std::array<int, kQfmmMaxLMult + 1>, kQfmmMaxLMult + 1> table{};

    for (int n = 0; n <= kQfmmMaxLMult; ++n)
    {
        table[n][0] = 1;

        for (int k = 1; k <= n; ++k)
        {
            table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
        }
    }

    return table;
}

inline constexpr auto kBinomial = MakeBinomialTable();

/// x^k by repeated multiplication; x^0 = 1.0 exactly (the L_mult = 0
/// reductions rely on the exact 1.0).
// (x, k) is the (base, exponent) math signature; the few call sites pass
// fixed named arguments.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline double Power(double x, int k) noexcept {
    double value = 1.0;

    for (int i = 0; i < k; ++i)
    {
        value *= x;
    }

    return value;
}

/// The flat index of one entry of a per-axis E table in the PerAxisETable
/// layout at the caller-chosen (la, lb) dims.
inline std::size_t RaisedIndex(int ix, int jx, int t, int lb, int tMax) noexcept {
    return (static_cast<std::size_t>(ix) * (lb + 1) + jx) * (tMax + 1) + t;
}

/// The (-1)^|mono| of the M2M read: the moment shift M_t(new) = Σ_s
/// kQfmmM2M[t][s] · (-1)^|mono| d^mono M_s (the banner of
/// qfmm_tables_gen.hpp). Exactly +1.0 / -1.0.
inline double MonomialSign(int mono) noexcept {
    const std::array<int, 3>& exponents = kQfmmMonomialExponents[static_cast<std::size_t>(mono)];
    return ((exponents[0] + exponents[1] + exponents[2]) % 2 == 0) ? 1.0 : -1.0;
}

/// d^mono by repeated multiplication per axis; the monomial (0, 0, 0)
/// yields exactly 1.0.
// (mono) is the exponent index and (dx, dy, dz) the per-axis increments -
// distinct quantities, fixed call order.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline double MonomialPower(int mono, double dx, double dy, double dz) noexcept {
    const std::array<int, 3>& exponents = kQfmmMonomialExponents[static_cast<std::size_t>(mono)];
    return Power(dx, exponents[0]) * Power(dy, exponents[1]) * Power(dz, exponents[2]);
}

/// The M2M coefficient of the moment shift: M_t(new) = Σ_s Σ_mono
/// kQfmmM2M[t][s][mono] · (-1)^|mono| · d^mono · M_s. The coefficients are
/// read from the sparse run layout of qfmm_tables_gen.hpp (the entries
/// kQfmmM2MValues[k], k = the run of (t, s), with the monomial index
/// kQfmmM2MMonomial[k]). At (t, s) = (0, 0) the emitted run is exactly the
/// single entry {(monomial 0, 1.0)}, so the value is exactly 1.0 for any d -
/// the base of the monopole reduction (pinned by the tests).
// (t, s) are table indices and (dx, dy, dz) the center shift - distinct
// quantities, fixed call order.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline double EvalM2M(int t, int s, double dx, double dy, double dz) noexcept {
    double value = 0.0;
    const std::size_t row =
        static_cast<std::size_t>(t) * kQfmmMomentCount + static_cast<std::size_t>(s);
    const std::uint32_t begin = kQfmmM2MRunOffset[row];
    const std::uint32_t end = kQfmmM2MRunOffset[row + 1];

    for (std::uint32_t k = begin; k < end; ++k)
    {
        const int mono = kQfmmM2MMonomial[k];
        value += kQfmmM2MValues[k] * MonomialSign(mono) * MonomialPower(mono, dx, dy, dz);
    }

    return value;
}

/// The M2L coefficient: L_t(d) = Σ_mono kQfmmM2L[t][s][mono] · d^mono ·
/// M_s / |d|^kQfmmM2LDenomPower[t][s] (the table carries the
/// addition-theorem rescaling; the denominator is applied by the caller),
/// read from the sparse run layout (see EvalM2M). At (t, s) = (0, 0):
/// exactly 1.0.
// (t, s) are table indices and (dx, dy, dz) the center shift - distinct
// quantities, fixed call order.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline double EvalM2L(int t, int s, double dx, double dy, double dz) noexcept {
    double value = 0.0;
    const std::size_t row =
        static_cast<std::size_t>(t) * kQfmmMomentCount + static_cast<std::size_t>(s);
    const std::uint32_t begin = kQfmmM2LRunOffset[row];
    const std::uint32_t end = kQfmmM2LRunOffset[row + 1];

    for (std::uint32_t k = begin; k < end; ++k)
    {
        const int mono = kQfmmM2LMonomial[k];
        value += kQfmmM2LValues[k] * MonomialPower(mono, dx, dy, dz);
    }

    return value;
}

/// The L2L coefficient: the M2M table read transposed with +d (no sign) -
/// the local shift L_child = Σ_s kQfmmM2M[s][t] · d^mono · L_parent (the
/// banner), from the run of (sp, t) in the sparse layout.
// (sp, t) are table indices and (dx, dy, dz) the center shift - distinct
// quantities, fixed call order.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline double EvalL2L(int sp, int t, double dx, double dy, double dz) noexcept {
    double value = 0.0;
    const std::size_t row =
        static_cast<std::size_t>(sp) * kQfmmMomentCount + static_cast<std::size_t>(t);
    const std::uint32_t begin = kQfmmM2MRunOffset[row];
    const std::uint32_t end = kQfmmM2MRunOffset[row + 1];

    for (std::uint32_t k = begin; k < end; ++k)
    {
        const int mono = kQfmmM2MMonomial[k];
        value += kQfmmM2MValues[k] * MonomialPower(mono, dx, dy, dz);
    }

    return value;
}

/// Fills one per-axis 1D E table at the caller-chosen (la, lb) dims into
/// caller-owned storage in the PerAxisETable flat layout. The shared
/// PerAxisETable scratch (kMaxPerAxisTableValues, md_hermite.hpp) bounds
/// la <= kMaxShellL + 2 (the kinetic builders' bound); the moment path
/// raises the bra to la + lMult - up to kMaxShellL + 3 at (la, lMult) =
/// (6, 3) - so the oversized tables are filled here by the SAME recurrence
/// (md_hermite.hpp's FillPerAxisTable, mirrored verbatim) instead of
/// growing the shared bound, which is load-bearing for the other builders.
/// Only the t = 0 slot is consumed downstream.
void FillRaisedPerAxisTable(int la,
                            int lb,
                            const std::array<double, 3>& shiftA,
                            const std::array<double, 3>& shiftB,
                            // (p, axis) are the exponent and the coordinate index -
                            // distinct quantities.
                            //
                            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                            double p,
                            int axis,
                            std::vector<double>& out) {
    const int tMax = la + lb;
    out.assign(static_cast<std::size_t>(la + 1) * (lb + 1) * (tMax + 1), 0.0);
    const double invTwoP = 0.5 / p;
    out[RaisedIndex(0, 0, 0, lb, tMax)] = 1.0;

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
                double value = shift * out[RaisedIndex(parentI, parentJ, t, lb, tMax)];

                if (t + 1 <= parentI + parentJ)
                {
                    value += static_cast<double>(t + 1) *
                             out[RaisedIndex(parentI, parentJ, t + 1, lb, tMax)];
                }

                if (t - 1 >= 0)
                {
                    value += invTwoP * out[RaisedIndex(parentI, parentJ, t - 1, lb, tMax)];
                }

                out[RaisedIndex(i, j, t, lb, tMax)] = value;
            }
        }
    }
}

/// The chunk resolution of the parallel far-field passes (the mirror of
/// fock_screen.hpp's ChunkCountFor on a pass's own task count - the same
/// 0/1/>=2 semantics as FockBuildOptions::maxParallelChunks): 0 = auto
/// (min(taskCount, DefaultOmpTeamSize())); 1 = the serial fallback (the
/// pass-parameter default, so direct-call callers keep the bit-identical
/// serial path); >= 2 a fixed split (min(taskCount, chunkCount)). The
/// result is 0 only for an empty task count, which the passes treat as the
/// serial fallback (no tasks to chunk).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline std::size_t FarFieldChunkCount(std::size_t chunkCount, std::size_t taskCount) noexcept {
    const std::size_t maxChunks =
        chunkCount != 0 ? chunkCount : static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());

    return std::min(taskCount, maxChunks);
}

/// The chunk boundaries over [0, taskCount) for numChunks chunks: evenly
/// split, endpoints exact. No cache-line alignment of the interior
/// boundaries (the direct builder's 8-double rule): the chunked M2L writes
/// per-chunk partial buffers (no element shared with the adjacent chunk)
/// and the other chunked passes are single-writer per element, so there is
/// no adjacent-chunk shared write to separate.
inline std::vector<std::size_t> FarFieldChunkStarts(std::size_t taskCount, std::size_t numChunks) {
    std::vector<std::size_t> starts(numChunks + 1);
    starts[0] = 0;
    starts[numChunks] = taskCount;

    for (std::size_t c = 1; c < numChunks; ++c)
    {
        starts[c] = (taskCount * c) / numChunks;
    }

    return starts;
}

/// The moment blocks of one pair at lMult > 0: per primitive pair, the
/// raised (la + lMult, lb) per-axis E tables (the angular-momentum-raising trick), the
/// t = 0-only raised E3d (only the t = 0 slot is consumed: the moment is
/// an overlap-type integral), the shell folds, and the per-(p, q, r)
/// raised overlap-type integrals contracted over primitives and rows;
/// then the Cartesian moments about the pair center by the binomial shift
/// and the solid-harmonic contraction into the (l, m) blocks (l <= lMult;
/// the higher blocks stay zero). The (0, 0) block comes out of this path
/// too (a = b = c = 0: M_000 = O_000 = the overlap).
/// The raised-offset cube size: (lMult + 1)^3 - the layout of the
/// O_{p, q, r} and M_{a, b, c} scratches and of the slot table.
inline std::size_t OffsetCount(int lMult) noexcept {
    const int span = lMult + 1;

    return static_cast<std::size_t>(span) * span * span;
}

/// slot(anga, p, q, r): the index in the raised row set of
/// (ix + p, iy + q, iz + r), where (ix, iy, iz) = kCartesianIndices[la]
/// [anga]. The raised row set is the union of the rows the offsets
/// touch - the rows of total degree la .. la + 3*lMult (the rows the
/// moment transform reads, O_000 at (p, q, r) = (0, 0, 0) included; the
/// shell-Cartesian set of the raised shell alone - the total-degree
/// la + lMult rows - would miss the lower rows the moments need).
void BuildRaisedRowMap(int la,
                       int lMult,
                       std::vector<CartIndex>& raisedRows,
                       std::vector<std::size_t>& slot) {
    const int laPrime = la + lMult;
    const int nCartA = CartesianCount(la);
    const int offsetSpan = lMult + 1;
    const std::size_t offsetCount = OffsetCount(lMult);
    const int span = laPrime + 1;
    std::vector<int> raisedRow(static_cast<std::size_t>(span) * span * span, -1);
    slot.assign(offsetCount * static_cast<std::size_t>(nCartA), 0);

    for (int p = 0; p <= lMult; ++p)
    {
        for (int q = 0; q <= lMult; ++q)
        {
            for (int r = 0; r <= lMult; ++r)
            {
                const std::size_t offsetIdx =
                    (static_cast<std::size_t>(p) * offsetSpan + q) * offsetSpan + r;

                for (int anga = 0; anga < nCartA; ++anga)
                {
                    const CartIndex base = kCartesianIndices[la][anga];
                    const int ixR = base.ix + p;
                    const int iyR = base.iy + q;
                    const int izR = base.iz + r;
                    int& row = raisedRow[(static_cast<std::size_t>(ixR) * span + iyR) * span + izR];

                    if (row < 0)
                    {
                        row = static_cast<int>(raisedRows.size());
                        raisedRows.push_back(CartIndex{ixR, iyR, izR});
                    }

                    slot[offsetIdx * static_cast<std::size_t>(nCartA) + anga] =
                        static_cast<std::size_t>(row);
                }
            }
        }
    }
}

/// The per-primitive raised-overlap accumulation: the t = 0 slices of the
/// raised (la + lMult, lb) per-axis tables, the t = 0-only raised E3d, the
/// shell folds, and the per-(p, q, r) raised overlap-type integrals
/// contracted over primitives and rows into `raised`.
void AccumulateRaisedOverlaps(const MdPairData& pair,
                              int lMult,
                              const std::vector<CartIndex>& raisedRows,
                              const std::vector<std::size_t>& slot,
                              std::vector<double>& raised) {
    const int laPrime = pair.la + lMult;
    const int nCartA = CartesianCount(pair.la);
    const int nCartB = CartesianCount(pair.lb);
    const std::size_t nAngA = pair.nFuncsA / pair.rowsA;
    const std::size_t nAngB = pair.nFuncsB / pair.rowsB;
    const int offsetSpan = lMult + 1;
    const std::size_t rowCount = raisedRows.size();
    // The per-primitive scratch, reused across primitive pairs.
    std::vector<double> e3d0(rowCount * static_cast<std::size_t>(nCartB));
    std::vector<double> midB(rowCount * nAngB);
    std::vector<double> folded(nAngA * nAngB);

    for (std::size_t primIdx = 0; primIdx < pair.primPairs.size(); ++primIdx)
    {
        const MdPrimPair& prim = pair.primPairs[primIdx];
        const double pref = std::pow(std::numbers::pi / prim.p, 1.5) * prim.prefactor;
        const std::array<double, 3> shiftA = {
            prim.px - pair.ax, prim.py - pair.ay, prim.pz - pair.az};
        const std::array<double, 3> shiftB = {
            prim.px - pair.bx, prim.py - pair.by, prim.pz - pair.bz};

        // The raised (la + lMult, lb) per-axis tables. The shared
        // PerAxisETable scratch bounds la' <= kMaxShellL + 2; the moment
        // path raises the bra to la + lMult - up to kMaxShellL + 3 at
        // (la, lMult) = (6, 3) - so the oversized case fills caller-owned
        // storage by the same recurrence instead (see
        // FillRaisedPerAxisTable).
        std::array<PerAxisETable, 3> sharedTables;
        std::array<std::vector<double>, 3> localTables;
        const bool fitsShared = laPrime <= kMaxShellL + 2;

        if (fitsShared)
        {
            sharedTables = BuildPerAxisTables(laPrime, pair.lb, shiftA, shiftB, prim.p);
        } else
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                FillRaisedPerAxisTable(
                    laPrime, pair.lb, shiftA, shiftB, prim.p, axis, localTables[axis]);
            }
        }

        // The t = 0 slices of the raised tables (the only slot the moment
        // path consumes), in the (i, j) flat layout.
        std::array<std::vector<double>, 3> e0;

        for (int axis = 0; axis < 3; ++axis)
        {
            e0[static_cast<std::size_t>(axis)].resize(static_cast<std::size_t>(laPrime + 1) *
                                                      (pair.lb + 1));

            for (int ix = 0; ix <= laPrime; ++ix)
            {
                for (int jx = 0; jx <= pair.lb; ++jx)
                {
                    const std::size_t index =
                        static_cast<std::size_t>(ix) * (pair.lb + 1) + static_cast<std::size_t>(jx);
                    e0[static_cast<std::size_t>(axis)][index] =
                        fitsShared
                            ? sharedTables[static_cast<std::size_t>(axis)].At(ix, jx, 0)
                            : localTables[static_cast<std::size_t>(axis)]
                                         [RaisedIndex(ix, jx, 0, pair.lb, laPrime + pair.lb)];
                }
            }
        }

        // The t = 0-only raised E3d: per (raised bra row, ket cartesian
        // row) the product of the per-axis t = 0 slices - no Hermite-3D
        // index, only the t = 0 slot is ever consumed (the moment is an
        // overlap-type integral).
        for (std::size_t row = 0; row < rowCount; ++row)
        {
            const CartIndex& cr = raisedRows[row];

            for (int angb = 0; angb < nCartB; ++angb)
            {
                const CartIndex& cb = kCartesianIndices[pair.lb][angb];
                e3d0[row * static_cast<std::size_t>(nCartB) + static_cast<std::size_t>(angb)] =
                    e0[0][static_cast<std::size_t>(cr.ix) * (pair.lb + 1) +
                          static_cast<std::size_t>(cb.ix)] *
                    e0[1][static_cast<std::size_t>(cr.iy) * (pair.lb + 1) +
                          static_cast<std::size_t>(cb.iy)] *
                    e0[2][static_cast<std::size_t>(cr.iz) * (pair.lb + 1) +
                          static_cast<std::size_t>(cb.iz)];
            }
        }

        // Fold shell B (identity for Cartesian shells).
        for (std::size_t row = 0; row < rowCount; ++row)
        {
            for (std::size_t fb = 0; fb < nAngB; ++fb)
            {
                double value = 0.0;

                if (pair.isSphericalB)
                {
                    for (int angb = 0; angb < nCartB; ++angb)
                    {
                        value += kSolidHarmonicG[pair.lb][fb][angb] *
                                 e3d0[row * static_cast<std::size_t>(nCartB) +
                                      static_cast<std::size_t>(angb)];
                    }
                } else
                {
                    value = e3d0[row * static_cast<std::size_t>(nCartB) + fb];
                }

                midB[row * nAngB + fb] = value;
            }
        }

        // Per raised offset (p, q, r): fold shell A and accumulate the
        // raised overlap-type integral O_{p, q, r} over primitives and
        // rows (the BuildOverlapPair nesting: rows outer, then fb, then
        // fa).
        for (int p = 0; p <= lMult; ++p)
        {
            for (int q = 0; q <= lMult; ++q)
            {
                for (int r = 0; r <= lMult; ++r)
                {
                    const std::size_t offsetIdx =
                        (static_cast<std::size_t>(p) * offsetSpan + q) * offsetSpan + r;

                    for (std::size_t fb = 0; fb < nAngB; ++fb)
                    {
                        for (std::size_t fa = 0; fa < nAngA; ++fa)
                        {
                            double value = 0.0;

                            if (pair.isSphericalA)
                            {
                                for (int anga = 0; anga < nCartA; ++anga)
                                {
                                    value +=
                                        kSolidHarmonicG[pair.la][fa][anga] *
                                        midB[slot[offsetIdx * static_cast<std::size_t>(nCartA) +
                                                  static_cast<std::size_t>(anga)] *
                                                 nAngB +
                                             fb];
                                }
                            } else
                            {
                                value =
                                    midB[slot[offsetIdx * static_cast<std::size_t>(nCartA) + fa] *
                                             nAngB +
                                         fb];
                            }

                            folded[fa * nAngB + fb] = value;
                        }
                    }

                    for (std::size_t rowA = 0; rowA < pair.rowsA; ++rowA)
                    {
                        for (std::size_t rowB = 0; rowB < pair.rowsB; ++rowB)
                        {
                            const double weight =
                                pair.braWeights[primIdx][rowA * pair.rowsB + rowB];

                            for (std::size_t fb = 0; fb < nAngB; ++fb)
                            {
                                for (std::size_t fa = 0; fa < nAngA; ++fa)
                                {
                                    raised[offsetIdx * pair.nFuncs +
                                           (rowA * nAngA + fa) * pair.nFuncsB + rowB * nAngB +
                                           fb] += weight * pref * folded[fa * nAngB + fb];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The Cartesian moments about the pair center: M_{a,b,c} = Σ_{p<=a,
/// q<=b, r<=c} C(a,p) C(b,q) C(c,r) (Ax-Px)^{a-p} (Ay-Py)^{b-q}
/// (Az-Pz)^{c-r} O_{p,q,r} per function pair - the binomial expansion
/// of (x - C)^a = ((x - A) + (A - C))^a, with O_{p,q,r} the
/// fully-contracted raised overlap-type integral.
void ShiftToCartesianMoments(const MdPairData& pair,
                             const QfmmPairGeometry& geometry,
                             int lMult,
                             const std::vector<double>& raised,
                             std::vector<double>& moments) {
    const int offsetSpan = lMult + 1;
    const double dxA = pair.ax - geometry.centerX;
    const double dyA = pair.ay - geometry.centerY;
    const double dzA = pair.az - geometry.centerZ;

    for (int a = 0; a <= lMult; ++a)
    {
        for (int b = 0; b <= lMult; ++b)
        {
            for (int c = 0; c <= lMult; ++c)
            {
                if (a + b + c > lMult)
                {
                    continue;
                }

                const std::size_t mIdx =
                    (static_cast<std::size_t>(a) * offsetSpan + b) * offsetSpan + c;

                for (std::size_t fun = 0; fun < pair.nFuncs; ++fun)
                {
                    double value = 0.0;

                    for (int p = 0; p <= a; ++p)
                    {
                        for (int q = 0; q <= b; ++q)
                        {
                            for (int r = 0; r <= c; ++r)
                            {
                                const std::size_t oIdx =
                                    (static_cast<std::size_t>(p) * offsetSpan + q) * offsetSpan + r;
                                value += static_cast<double>(kBinomial[a][p] * kBinomial[b][q] *
                                                             kBinomial[c][r]) *
                                         Power(dxA, a - p) * Power(dyA, b - q) * Power(dzA, c - r) *
                                         raised[oIdx * pair.nFuncs + fun];
                            }
                        }
                    }

                    moments[mIdx * pair.nFuncs + fun] = value;
                }
            }
        }
    }
}

/// The solid-harmonic contraction: the (l, m) moment block = Σ_cart
/// kQfmmSolidHarmonicG[l][m + l][cart] · M_{cart} - the R_{l,m} rows of
/// the same transform the spherical folds use, from the QFMM header's own
/// emission (the shell tables kCartesianIndices / kSolidHarmonicG stop at
/// kMaxShellL = 6 and the moment path reaches l = lMult; the emitted rows
/// are bit-identical to the shell rows for l <= 6 - pinned by
/// QfmmLMultTest.MomentSolidHarmonicRowsMatchTheShellTablesUpToShellL).
/// Blocks with l > lMult stay zero (the table was zero-initialized).
void ContractToSolidHarmonics(const MdPairData& pair,
                              int lMult,
                              const std::vector<double>& moments,
                              double* blocks) {
    const int offsetSpan = lMult + 1;

    for (int l = 0; l <= lMult; ++l)
    {
        const int nCartL = CartesianCount(l);

        for (int m = -l; m <= l; ++m)
        {
            double* block = blocks + static_cast<std::size_t>(MomentIndex(l, m)) * pair.nFuncs;

            for (std::size_t fun = 0; fun < pair.nFuncs; ++fun)
            {
                double value = 0.0;

                for (int cart = 0; cart < nCartL; ++cart)
                {
                    const auto& cm = kQfmmCartesianIndices[l][cart];
                    const std::size_t mIdx =
                        (static_cast<std::size_t>(cm[0]) * offsetSpan + cm[1]) * offsetSpan + cm[2];
                    value +=
                        kQfmmSolidHarmonicG[l][m + l][cart] * moments[mIdx * pair.nFuncs + fun];
                }

                block[fun] = value;
            }
        }
    }
}

void BuildPairMoments(const MdPairData& pair,
                      const QfmmPairGeometry& geometry,
                      int lMult,
                      double* blocks) {
    std::vector<CartIndex> raisedRows;
    std::vector<std::size_t> slot;

    BuildRaisedRowMap(pair.la, lMult, raisedRows, slot);

    // The raised-overlap scratch O_{p, q, r} per function pair (the full
    // offset cube; the moment transform reads the p + q + r <= lMult
    // subset) and the Cartesian-moment scratch M_{a, b, c} (same layout).
    std::vector<double> raised(OffsetCount(lMult) * pair.nFuncs, 0.0);
    std::vector<double> moments(OffsetCount(lMult) * pair.nFuncs, 0.0);

    AccumulateRaisedOverlaps(pair, lMult, raisedRows, slot, raised);
    ShiftToCartesianMoments(pair, geometry, lMult, raised, moments);
    ContractToSolidHarmonics(pair, lMult, moments, blocks);
}

} // namespace

qcx::Result<QfmmMomentTable> BuildMomentTable(const ShellPairList& pairList,
                                              const std::vector<MdPairData>& pairStore,
                                              const std::vector<QfmmPairGeometry>& pairGeometries,
                                              const std::vector<int>& pairOrders) {
    if (pairOrders.size() != pairStore.size())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "one pair order per pair-store entry is required"});
    }

    for (const int order : pairOrders)
    {
        if (order < -1 || order > kQfmmMaxLMult)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "a pair order must be -1 or in 0..kQfmmMaxLMult"});
        }
    }

    if (pairGeometries.size() != pairStore.size())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "one pair geometry per pair is required"});
    }

    if (pairStore.size() != pairList.pairs.size())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "one pair-store entry per listed pair is required"});
    }

    QfmmMomentTable table;
    table.offsets.reserve(pairStore.size() + 1);
    table.counts.reserve(pairStore.size());
    table.offsets.push_back(0);

    // The on-demand layout: pair p's count is (order + 1)^2 blocks for
    // order >= 0 (the (0, 0) block included at every nonzero count), 0
    // blocks for order == -1 - a pair in a near-field-only leaf, whose
    // moments never feed any interaction, allocates nothing.
    std::size_t total = 0;

    for (std::size_t p = 0; p < pairStore.size(); ++p)
    {
        const int order = pairOrders[p];
        const std::size_t count =
            order < 0 ? 0
                      : static_cast<std::size_t>(order + 1) * static_cast<std::size_t>(order + 1);
        total += count * pairStore[p].nFuncs;
        table.offsets.push_back(total);
        table.counts.push_back(count);
    }

    // Value-initialized; the moments beyond a pair's own order do not
    // exist (uniform orders reproduce the old full-stride build's content
    // exactly - the old always-81-block stride only ever carried zeros
    // there).
    table.blocks.resize(total);

    for (std::size_t p = 0; p < pairStore.size(); ++p)
    {
        if (table.counts[p] == 0)
        {
            continue;
        }

        if (pairOrders[p] == 0)
        {
            // The monopole path, unchanged: the (0, 0) moment block of an
            // order-0 pair IS the overlap block through the trusted
            // overlap engine (the block offsets put it at
            // blocks[offsets[p]]).
            BuildOverlapPair(pairStore[p], table.blocks.data() + table.offsets[p]);
        } else
        {
            BuildPairMoments(pairStore[p],
                             pairGeometries[p],
                             pairOrders[p],
                             table.blocks.data() + table.offsets[p]);
        }
    }

    return table;
}

std::vector<int> SelectMultipoleOrders(
    const QfmmTreeBuildResult& tree,
    const std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    double epsInt,
    int orderCap,
    QfmmOrderSelectionStats* stats) {
    std::vector<int> orders;
    orders.reserve(farFieldPairs.size());

    QfmmOrderSelectionStats local;
    QfmmOrderSelectionStats& record = (stats != nullptr) ? *stats : local;
    record = QfmmOrderSelectionStats{};
    record.pairCount = farFieldPairs.size();
    record.orderCap = orderCap;
    record.epsInt = epsInt;

    // The source radius is the node's own charge radius: the moments are of
    // the charge the node carries, and this radius is that charge's covering
    // ball (built bottom-up by the octree from the pair extents). The larger
    // of the two bounds both M2L directions of the unordered pair, and it is
    // the same quantity admission reads, so "admitted as far" and
    // "representable at some allowed degree" are one test.
    for (const auto& [a, b] : farFieldPairs)
    {
        const QfmmTreeNode& nodeA = tree.nodes[a];
        const QfmmTreeNode& nodeB = tree.nodes[b];
        const double dx = nodeA.centerX - nodeB.centerX;
        const double dy = nodeA.centerY - nodeB.centerY;
        const double dz = nodeA.centerZ - nodeB.centerZ;
        const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double radius = std::max(nodeA.radius, nodeB.radius);

        const int order = MinimalMultipoleOrder(radius, distance, epsInt, orderCap);

        if (order < 0)
        {
            // No allowed degree meets the budget: the interaction runs at the
            // cap with the budget missed, and the record now says so. Reached
            // only through the geometric admission (an explicit theta) - the
            // error arm keeps such pairs out of this list.
            ++record.fellThroughToCap;
            orders.push_back(orderCap);
        } else
        {
            orders.push_back(order);
        }

        const double ratio = (distance > 0.0) ? (radius / distance) : 1.0;

        if (ratio > record.worstRatio)
        {
            record.worstRatio = ratio;
            record.worstBound = MultipoleTruncationBound(radius, distance, orderCap);
        }
    }

    return orders;
}

std::vector<int> ComputeNeededNodeOrders(
    const QfmmTreeBuildResult& tree,
    const std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
    const std::vector<int>& pairOrders) {
    // q(n) = the maximum over the far pairs touching n or an ancestor of n
    // of (pairOrder + 1); the per-node order is q - 1, with -1 (q == 0)
    // marking a near-field-only node. The node order is pre-order (parents
    // before children), so the touch-max first and the pre-order
    // propagation second visit every node after its ancestors.
    std::vector<int> q(tree.nodes.size(), 0);

    for (std::size_t i = 0; i < farFieldPairs.size(); ++i)
    {
        const int need = pairOrders[i] + 1;
        q[farFieldPairs[i].first] = std::max(q[farFieldPairs[i].first], need);
        q[farFieldPairs[i].second] = std::max(q[farFieldPairs[i].second], need);
    }

    for (std::size_t n = 0; n < tree.nodes.size(); ++n)
    {
        for (const std::size_t child : tree.children[n])
        {
            q[child] = std::max(q[child], q[n]);
        }
    }

    std::vector<int> orders;
    orders.reserve(tree.nodes.size());

    for (const int need : q)
    {
        orders.push_back(need - 1);
    }

    return orders;
}

std::vector<double> AggregateLeafMoments(const ShellPairList& pairList,
                                         const QfmmTreeBuildResult& tree,
                                         const QfmmMomentTable& table,
                                         const std::vector<QfmmPairGeometry>& pairGeometries,
                                         const Eigen::MatrixXd& density,
                                         const std::vector<int>& nodeOrders,
                                         std::size_t chunkCount) {
    // kQfmmMomentCount moments per NODE (internal nodes stay zero here -
    // the M2M pass fills them), so the leaf moments can index
    // AggregateNodeMoments directly.
    std::vector<double> leafMoments(tree.nodes.size() * kQfmmMomentCount, 0.0);

    // The chunked pass divides the NODE ranges: every leaf's moments are
    // accumulated by exactly one chunk (single-writer per element), so the
    // chunked evaluation is bit-identical to the serial in any schedule.
    auto aggregateRange = [&](std::size_t begin, std::size_t end) {
        for (std::size_t nodeIndex = begin; nodeIndex < end; ++nodeIndex)
        {
            const QfmmTreeNode& node = tree.nodes[nodeIndex];

            if (!node.isLeaf)
            {
                continue;
            }

            // A near-field-only leaf (order -1): its pairs hold zero blocks
            // and its moments are never read - skip the whole leaf.
            const int order = nodeOrders[nodeIndex];

            if (order < 0)
            {
                continue;
            }

            for (const std::size_t pairIndex : node.pairIndices)
            {
                const ShellPairIndex& pair = pairList.pairs[pairIndex];
                const std::size_t oA = pairList.shells[pair.i].functionOffset;
                const std::size_t oB = pairList.shells[pair.j].functionOffset;
                const std::size_t nA = ShellFunctionCount(pairList.shells[pair.i]);
                const std::size_t nB = ShellFunctionCount(pairList.shells[pair.j]);
                const bool diagonal = pair.i == pair.j;
                const std::size_t nFuncs = nA * nB;
                const double* block = table.blocks.data() + table.offsets[pairIndex];
                // The density-weighted moments of this pair, one per (l, m):
                // pairMoments[s] = Σ_{fb,fa} D_{fa,fb} · block_s(fa, fb), with
                // the density read in the direct builder's J convention. The
                // contraction reads the pair's count - the blocks beyond the
                // pair's order do not exist (the pair's order equals its
                // leaf's by construction).
                std::array<double, kQfmmMomentCount> pairMoments{};

                for (std::size_t fb = 0; fb < nB; ++fb)
                {
                    for (std::size_t fa = 0; fa < nA; ++fa)
                    {
                        const Eigen::Index funA = static_cast<Eigen::Index>(oA + fa);
                        const Eigen::Index funB = static_cast<Eigen::Index>(oB + fb);
                        const double densityElement =
                            diagonal ? density(funA, funB)
                                     : density(funA, funB) + density(funB, funA);
                        const std::size_t fun = fa * nB + fb;

                        for (std::size_t s = 0; s < table.counts[pairIndex]; ++s)
                        {
                            pairMoments[s] += densityElement * block[s * nFuncs + fun];
                        }
                    }
                }

                // The M2M of the pair's moments to the leaf center.
                const double dx = node.centerX - pairGeometries[pairIndex].centerX;
                const double dy = node.centerY - pairGeometries[pairIndex].centerY;
                const double dz = node.centerZ - pairGeometries[pairIndex].centerZ;

                for (int lt = 0; lt <= order; ++lt)
                {
                    for (int mt = -lt; mt <= lt; ++mt)
                    {
                        const std::size_t t = static_cast<std::size_t>(MomentIndex(lt, mt));

                        for (int ls = 0; ls <= order; ++ls)
                        {
                            for (int ms = -ls; ms <= ls; ++ms)
                            {
                                const std::size_t s = static_cast<std::size_t>(MomentIndex(ls, ms));
                                leafMoments[nodeIndex * kQfmmMomentCount + t] +=
                                    EvalM2M(static_cast<int>(t), static_cast<int>(s), dx, dy, dz) *
                                    pairMoments[s];
                            }
                        }
                    }
                }
            }
        }
    };

    const std::size_t numChunks = FarFieldChunkCount(chunkCount, tree.nodes.size());

    if (numChunks <= 1)
    {
        // The serial fallback: today's canonical path, unchanged.
        aggregateRange(0, tree.nodes.size());
    } else
    {
        const std::vector<std::size_t> chunkStarts =
            FarFieldChunkStarts(tree.nodes.size(), numChunks);
        qcx::backend::Backend<qcx::backend::CpuTag> backend;

        backend.ParallelFor(numChunks, [&](std::size_t chunk) {
            aggregateRange(chunkStarts[chunk], chunkStarts[chunk + 1]);
        });
    }

    return leafMoments;
}

std::vector<double> AggregateNodeMoments(const QfmmTreeBuildResult& tree,
                                         const std::vector<double>& leafMoments,
                                         const std::vector<int>& nodeOrders) {
    std::vector<double> nodeMoments = leafMoments;

    // Reverse pre-order: children before parents.
    for (std::size_t nodeIndex = tree.nodes.size(); nodeIndex-- > 0;)
    {
        if (tree.nodes[nodeIndex].isLeaf)
        {
            continue;
        }

        // A -1 node is touched by no far pair (a touching pair would give
        // it order >= 0), and its ancestors are all -1 too (the q
        // propagation runs downward, so an active ancestor would force
        // this node at least as active) - its moments are never read.
        // Skip it entirely; its descendants may be active on their own
        // (a pair touching a child does not touch the parent), and the
        // reverse pre-order reaches them independently.
        const int order = nodeOrders[nodeIndex];

        if (order < 0)
        {
            continue;
        }

        double* nodeBlock = nodeMoments.data() + nodeIndex * kQfmmMomentCount;

        for (const std::size_t child : tree.children[nodeIndex])
        {
            const double* childBlock = nodeMoments.data() + child * kQfmmMomentCount;
            const double dx = tree.nodes[nodeIndex].centerX - tree.nodes[child].centerX;
            const double dy = tree.nodes[nodeIndex].centerY - tree.nodes[child].centerY;
            const double dz = tree.nodes[nodeIndex].centerZ - tree.nodes[child].centerZ;

            // The child carries at least this node's order (children never
            // fall below their parents), so every read s <= order sits in
            // the child's content.
            for (int lt = 0; lt <= order; ++lt)
            {
                for (int mt = -lt; mt <= lt; ++mt)
                {
                    const std::size_t t = static_cast<std::size_t>(MomentIndex(lt, mt));
                    double value = 0.0;

                    for (int ls = 0; ls <= order; ++ls)
                    {
                        for (int ms = -ls; ms <= ls; ++ms)
                        {
                            const std::size_t s = static_cast<std::size_t>(MomentIndex(ls, ms));
                            value += EvalM2M(static_cast<int>(t), static_cast<int>(s), dx, dy, dz) *
                                     childBlock[s];
                        }
                    }

                    nodeBlock[t] += value;
                }
            }
        }
    }

    return nodeMoments;
}

/// The M2L accumulation over the far pairs [begin, end): the unordered far
/// pair (a, b) contributes the translated multipole field of b at a and of
/// a at b. The emitted kQfmmM2L[t][s](d) polynomial is the local-expansion
/// coefficient at the TARGET of the moment-s field with d = cTarget -
/// cSource (the generator's V7 convention), so the a-side reads it at d =
/// cA - cB and the b-side at the REVERSED displacement - the odd-parity
/// terms (the dipole part) flip sign between the two directions. At
/// L_mult = 0 the two reads coincide exactly (the (0, 0) row is {1.0, 0,
/// ...}), the symmetric monopole Coulomb term. The distance is never zero: a
/// well-separated far pair satisfies distance >= (wA + wB) / theta > 0, so
/// the centers cannot coincide. `potentials` is the caller-owned target -
/// the shared per-node buffer on the serial path, or a chunk's own partial
/// buffer on the chunked path (a chunk's pairs write only its
/// partial's nodes).
///
/// The two directions share ONE table evaluation. The emitted row (t, s) of
/// kQfmmM2L is a homogeneous polynomial of degree l_t + l_s (the
/// generator's convention - tools/gen_qfmm_translation_tables.py, the
/// header banner of qfmm_tables_gen.hpp), and Power/MonomialPower evaluate
/// a monomial at -d as the exact sign image of its value at d, so
///   EvalM2L(t, s, -dx, -dy, -dz) == (-1)^(l_t + l_s) · EvalM2L(t, s, dx,
///   dy, dz)
/// holds BIT FOR BIT, term by term in the same run order (the accumulation
/// rounds to the exact negation of its own running sum). The reversed
/// direction is therefore read as the forward coefficient times that exact
/// sign - no symmetry of the geometry, the tree or the density is assumed,
/// no table is consulted, and not one bit of the result moves.

/// The fold applied to one (t, s) row: the reversed direction's coefficient
/// is the forward one times the row's exact sign, so no second evaluation
/// happens. The fold is COUNTED here, where it is applied - a caller that
/// stops folding stops counting, which is what makes the fold visible from
/// outside the pass (the alternative, counting the loop's iterations, would
/// report the same number whether or not the fold ran). `stats` is reached
/// only on the counting instantiation; the production one compiles the
/// increment away entirely.
template <bool kCount>
double FoldedReversedCoefficient(double parity,
                                 double forward,
                                 QfmmM2LCoefficientStats& stats) noexcept {
    if constexpr (kCount)
    {
        ++stats.foldedRows;
    }

    return parity * forward;
}

/// `stats` (kCount true only) accumulates this call's fold count. The
/// instantiation is chosen by the caller (BuildFarFieldPotentials) so the
/// production path carries no counting at all.
template <bool kCount>
void ApplyM2LToImpl(double* potentials,
                    const QfmmTreeBuildResult& tree,
                    const std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
                    const std::vector<double>& nodeMoments,
                    const std::vector<int>& pairOrders,
                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                    std::size_t begin,
                    std::size_t end,
                    QfmmM2LCoefficientStats* stats) {
    for (std::size_t i = begin; i < end; ++i)
    {
        const auto& [a, b] = farFieldPairs[i];
        const QfmmTreeNode& nodeA = tree.nodes[a];
        const QfmmTreeNode& nodeB = tree.nodes[b];
        const double dx = nodeA.centerX - nodeB.centerX;
        const double dy = nodeA.centerY - nodeB.centerY;
        const double dz = nodeA.centerZ - nodeB.centerZ;
        const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

        // The pair's own truncation order bounds both directions: the
        // source content s <= L exists (the nodes' orders never fall below
        // the pairs touching them) and the target writes stay within the
        // nodes' content.
        const int order = pairOrders[i];
        // The pair's own fold counter, added to the sink once per pair (the
        // inner loop's increment stays register-resident).
        [[maybe_unused]] QfmmM2LCoefficientStats pairStats;

        for (int lt = 0; lt <= order; ++lt)
        {
            for (int mt = -lt; mt <= lt; ++mt)
            {
                const std::size_t t = static_cast<std::size_t>(MomentIndex(lt, mt));

                for (int ls = 0; ls <= order; ++ls)
                {
                    // The reversed direction's exact sign: the row is
                    // homogeneous of degree lt + ls.
                    const double parity = ((lt + ls) % 2 == 0) ? 1.0 : -1.0;

                    for (int ms = -ls; ms <= ls; ++ms)
                    {
                        const std::size_t s = static_cast<std::size_t>(MomentIndex(ls, ms));
                        const double coeffA =
                            EvalM2L(static_cast<int>(t), static_cast<int>(s), dx, dy, dz);
                        const double coeffB =
                            FoldedReversedCoefficient<kCount>(parity, coeffA, pairStats);

                        // |d|^kQfmmM2LDenomPower[t][s] by repeated
                        // multiplication from the distance (exactly the
                        // distance at power 1 - the monopole denominator).
                        double denom = distance;

                        for (int power = 1; power < kQfmmM2LDenomPower[t][s]; ++power)
                        {
                            denom *= distance;
                        }

                        potentials[a * kQfmmMomentCount + t] +=
                            coeffA * (nodeMoments[b * kQfmmMomentCount + s] / denom);
                        potentials[b * kQfmmMomentCount + t] +=
                            coeffB * (nodeMoments[a * kQfmmMomentCount + s] / denom);
                    }
                }
            }
        }

        if constexpr (kCount)
        {
            stats->foldedRows += pairStats.foldedRows;
        }
    }
}

/// The production M2L entry: the fold, no instrument (the 7-parameter
/// form, without the fold-counting template parameter).
void ApplyM2LTo(double* potentials,
                const QfmmTreeBuildResult& tree,
                const std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
                const std::vector<double>& nodeMoments,
                const std::vector<int>& pairOrders,
                // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                std::size_t begin,
                std::size_t end) {
    ApplyM2LToImpl<false>(
        potentials, tree, farFieldPairs, nodeMoments, pairOrders, begin, end, nullptr);
}

/// The L2L descent of one node: every child's local expansion gains the
/// translated parent content (the top-down pass; a child is written only
/// by its own parent's step, so each node block has exactly one writer).
/// The child's order caps the descent, and a -1 child is skipped entirely:
/// its potentials are never read (no far pair touches it or any ancestor -
/// the order bands are non-increasing upward, so its whole lineage is -1,
/// and it holds exact zeros). Parent content beyond the parent's own order
/// is never written (exact +0.0), so the zero-adds a cap above that would
/// read are inert; the caps below always read real content.
void ApplyL2LNode(std::vector<double>& potentials,
                  const QfmmTreeBuildResult& tree,
                  const std::vector<int>& nodeOrders,
                  std::size_t nodeIndex) {
    const double* nodeBlock = potentials.data() + nodeIndex * kQfmmMomentCount;

    for (const std::size_t child : tree.children[nodeIndex])
    {
        const int order = nodeOrders[child];

        if (order < 0)
        {
            continue;
        }

        double* childBlock = potentials.data() + child * kQfmmMomentCount;
        const double dx = tree.nodes[child].centerX - tree.nodes[nodeIndex].centerX;
        const double dy = tree.nodes[child].centerY - tree.nodes[nodeIndex].centerY;
        const double dz = tree.nodes[child].centerZ - tree.nodes[nodeIndex].centerZ;

        for (int lt = 0; lt <= order; ++lt)
        {
            for (int mt = -lt; mt <= lt; ++mt)
            {
                const std::size_t t = static_cast<std::size_t>(MomentIndex(lt, mt));
                double value = 0.0;

                for (int ls = 0; ls <= order; ++ls)
                {
                    for (int ms = -ls; ms <= ls; ++ms)
                    {
                        const std::size_t s = static_cast<std::size_t>(MomentIndex(ls, ms));
                        value += EvalL2L(static_cast<int>(s), static_cast<int>(t), dx, dy, dz) *
                                 nodeBlock[s];
                    }
                }

                childBlock[t] += value;
            }
        }
    }
}

/// The chunked L2L: the
/// root step first (its children's blocks must be final before any subtree
/// descends), then each root-child subtree descends independently - the
/// nodes are pre-order indexed and every subtree occupies a contiguous
/// index range, so a chunk is a plain interval and the ascending in-range
/// order keeps every parent before its children. Every node block is
/// written exactly once (by its own parent's step), so the chunked L2L is
/// bit-identical to the serial in any schedule.
void ApplyL2LChunked(std::vector<double>& potentials,
                     const QfmmTreeBuildResult& tree,
                     const std::vector<int>& nodeOrders,
                     std::size_t numChunks) {
    // The root's step: node 0's children are the subtree roots; their
    // blocks are read by the first step of every chunk.
    ApplyL2LNode(potentials, tree, nodeOrders, 0);

    // The chunk boundaries in the ROOT-CHILDREN index space; the chunk's
    // node interval runs from its first child to the next chunk's first
    // child (the last chunk to the node count) - consecutive root-child
    // subtrees tile the index space after the root.
    const std::vector<std::size_t>& rootChildren = tree.children[0];
    const std::vector<std::size_t> childStarts =
        FarFieldChunkStarts(rootChildren.size(), numChunks);
    qcx::backend::Backend<qcx::backend::CpuTag> backend;

    backend.ParallelFor(numChunks, [&](std::size_t chunk) {
        const std::size_t nodeBegin = rootChildren[childStarts[chunk]];
        const std::size_t nodeEnd = childStarts[chunk + 1] < rootChildren.size()
                                        ? rootChildren[childStarts[chunk + 1]]
                                        : tree.nodes.size();

        for (std::size_t nodeIndex = nodeBegin; nodeIndex < nodeEnd; ++nodeIndex)
        {
            ApplyL2LNode(potentials, tree, nodeOrders, nodeIndex);
        }
    });
}

std::vector<double> BuildFarFieldPotentials(
    const QfmmTreeBuildResult& tree,
    const std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
    const std::vector<double>& nodeMoments,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const std::vector<int>& pairOrders,
    const std::vector<int>& nodeOrders,
    std::size_t chunkCount,
    QfmmM2LCoefficientStats* coefficientStats) {
    // kQfmmMomentCount potentials per node (only leaves are consumed
    // downstream).
    std::vector<double> potentials(tree.nodes.size() * kQfmmMomentCount, 0.0);

    // The M2L first (the join), chunked over the far pairs (the one true
    // reduction of the far field: many pairs write the same node's
    // potentials). The chunk discipline mirrors the direct builder's
    // regions with one deliberate difference: instead of ParallelReduce's
    // critical-section merge (thread-completion order - the near field
    // accepts its last-bits scatter there), the chunk partials are
    // joined HERE in fixed chunk order after the parallel phase, so the
    // chunked far field is bit-reproducible run to run - the determinism
    // invariant the QFMM gates need at every chunk count. The chunked
    // result differs from the serial only in the fp grouping of each
    // node's adds (each chunk sums ITS pairs in list order, the join adds
    // the chunk sums in chunk order) - last bits, budget-irrelevant for
    // the approximate far field (the serial pin keeps
    // the bit-exact path for the gates).
    const std::size_t farPairCount = farFieldPairs.size();
    const std::size_t m2lChunks = FarFieldChunkCount(chunkCount, farPairCount);

    // The fold witness is reset here: the counters describe THIS call, not
    // the caller's history (a shared sink cannot silently accumulate).
    if (coefficientStats != nullptr)
    {
        *coefficientStats = QfmmM2LCoefficientStats{};
    }

    if (m2lChunks <= 1)
    {
        if (coefficientStats != nullptr)
        {
            ApplyM2LToImpl<true>(potentials.data(),
                                 tree,
                                 farFieldPairs,
                                 nodeMoments,
                                 pairOrders,
                                 0,
                                 farPairCount,
                                 coefficientStats);
        } else
        {
            ApplyM2LTo(
                potentials.data(), tree, farFieldPairs, nodeMoments, pairOrders, 0, farPairCount);
        }
    } else
    {
        // The per-chunk partial potentials: chunk c's partial holds the
        // adds of its far pairs in list order (chunks × nodes × moments
        // doubles - the direct builder's per-chunk partial Fock has the
        // same shape on a far bigger scale).
        const std::size_t stride = potentials.size();
        std::vector<double> partials(m2lChunks * stride, 0.0);
        const std::vector<std::size_t> chunkStarts = FarFieldChunkStarts(farPairCount, m2lChunks);
        qcx::backend::Backend<qcx::backend::CpuTag> backend;
        // One witness per chunk (the chunks run concurrently), summed in
        // chunk order after the join - deterministic, like the join itself.
        std::vector<QfmmM2LCoefficientStats> chunkStats(m2lChunks);

        backend.ParallelFor(m2lChunks, [&](std::size_t chunk) {
            if (coefficientStats != nullptr)
            {
                ApplyM2LToImpl<true>(partials.data() + chunk * stride,
                                     tree,
                                     farFieldPairs,
                                     nodeMoments,
                                     pairOrders,
                                     chunkStarts[chunk],
                                     chunkStarts[chunk + 1],
                                     &chunkStats[chunk]);
            } else
            {
                ApplyM2LTo(partials.data() + chunk * stride,
                           tree,
                           farFieldPairs,
                           nodeMoments,
                           pairOrders,
                           chunkStarts[chunk],
                           chunkStarts[chunk + 1]);
            }
        });

        // The fixed-order join: partial 0 + partial 1 + ... in chunk order.
        for (std::size_t c = 0; c < m2lChunks; ++c)
        {
            const double* partial = partials.data() + c * stride;

            for (std::size_t k = 0; k < stride; ++k)
            {
                potentials[k] += partial[k];
            }

            if (coefficientStats != nullptr)
            {
                coefficientStats->foldedRows += chunkStats[c].foldedRows;
            }
        }
    }

    // The L2L after the join: every node's local expansion must hold ALL
    // of its M2L content before its children descend (a child needs the
    // parent's full expansion, not a chunk's share). Per root-child
    // subtree, in ascending pre-order within the subtree.
    const std::size_t subtreeCount = tree.children[0].size();
    const std::size_t l2lChunks = FarFieldChunkCount(chunkCount, subtreeCount);

    if (l2lChunks <= 1)
    {
        // The serial top-down pass, unchanged.
        for (std::size_t nodeIndex = 0; nodeIndex < tree.nodes.size(); ++nodeIndex)
        {
            ApplyL2LNode(potentials, tree, nodeOrders, nodeIndex);
        }
    } else
    {
        ApplyL2LChunked(potentials, tree, nodeOrders, l2lChunks);
    }

    return potentials;
}

void AccumulateFarFieldJ(Eigen::MatrixXd& fock,
                         const ShellPairList& pairList,
                         const QfmmTreeBuildResult& tree,
                         const QfmmMomentTable& table,
                         const std::vector<double>& potentials,
                         const std::vector<QfmmPairGeometry>& pairGeometries,
                         const std::vector<int>& nodeOrders,
                         std::size_t chunkCount) {
    // The chunked accumulation runs over the NODE ranges. Every canonical
    // pair lives in exactly one leaf, and the distinct pairs' function-block
    // index sets (both orientations) are disjoint, so the fock writes are
    // single-writer per element - the chunked evaluation is bit-identical to
    // the serial in any schedule.
    auto accumulateRange = [&](std::size_t begin, std::size_t end) {
        for (std::size_t nodeIndex = begin; nodeIndex < end; ++nodeIndex)
        {
            const QfmmTreeNode& node = tree.nodes[nodeIndex];

            if (!node.isLeaf)
            {
                continue;
            }

            // A -1 leaf is skipped before anything dereferences it: its
            // pairs hold zero blocks (nothing to touch), and its potentials
            // are the exact zeros nothing wrote.
            const int order = nodeOrders[nodeIndex];

            if (order < 0)
            {
                continue;
            }

            const double* nodePotentials = potentials.data() + nodeIndex * kQfmmMomentCount;

            for (const std::size_t pairIndex : node.pairIndices)
            {
                const ShellPairIndex& pair = pairList.pairs[pairIndex];
                const std::size_t oA = pairList.shells[pair.i].functionOffset;
                const std::size_t oB = pairList.shells[pair.j].functionOffset;
                const std::size_t nA = ShellFunctionCount(pairList.shells[pair.i]);
                const std::size_t nB = ShellFunctionCount(pairList.shells[pair.j]);
                const bool diagonal = pair.i == pair.j;
                const std::size_t nFuncs = nA * nB;
                const double* block = table.blocks.data() + table.offsets[pairIndex];
                // The pair's moments are about the PAIR center; the
                // potentials are the local expansion about the NODE center.
                // The J contraction needs both at the same center, so the
                // moment blocks are translated to the node center first -
                // the same EvalM2M shift the leaf aggregation applies on
                // the source side. At order 0 the translation is exactly
                // the identity (the (0, 0) row is exactly {1.0, 0, ...}),
                // so the monopole path is unchanged bit-for-bit. The caps
                // stay within the leaf's order: the blocks exist exactly to
                // the pair's order (which equals the leaf's), and the
                // potential reads to the leaf's content.
                const double dx = node.centerX - pairGeometries[pairIndex].centerX;
                const double dy = node.centerY - pairGeometries[pairIndex].centerY;
                const double dz = node.centerZ - pairGeometries[pairIndex].centerZ;

                for (int lt = 0; lt <= order; ++lt)
                {
                    for (int mt = -lt; mt <= lt; ++mt)
                    {
                        const std::size_t t = static_cast<std::size_t>(MomentIndex(lt, mt));
                        const double potential = nodePotentials[t];

                        for (int ls = 0; ls <= order; ++ls)
                        {
                            for (int ms = -ls; ms <= ls; ++ms)
                            {
                                const std::size_t s = static_cast<std::size_t>(MomentIndex(ls, ms));
                                const double coeff =
                                    EvalM2M(static_cast<int>(t), static_cast<int>(s), dx, dy, dz) *
                                    potential;

                                for (std::size_t fb = 0; fb < nB; ++fb)
                                {
                                    for (std::size_t fa = 0; fa < nA; ++fa)
                                    {
                                        // The 2J factor of the direct
                                        // builder's convention, both
                                        // orientations.
                                        const Eigen::Index funA =
                                            static_cast<Eigen::Index>(oA + fa);
                                        const Eigen::Index funB =
                                            static_cast<Eigen::Index>(oB + fb);
                                        const double contribution =
                                            2.0 * coeff * block[s * nFuncs + fa * nB + fb];
                                        fock(funA, funB) += contribution;

                                        if (!diagonal)
                                        {
                                            fock(funB, funA) += contribution;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    const std::size_t numChunks = FarFieldChunkCount(chunkCount, tree.nodes.size());

    if (numChunks <= 1)
    {
        // The serial fallback: today's canonical path, unchanged.
        accumulateRange(0, tree.nodes.size());
    } else
    {
        const std::vector<std::size_t> chunkStarts =
            FarFieldChunkStarts(tree.nodes.size(), numChunks);
        qcx::backend::Backend<qcx::backend::CpuTag> backend;

        backend.ParallelFor(numChunks, [&](std::size_t chunk) {
            accumulateRange(chunkStarts[chunk], chunkStarts[chunk + 1]);
        });
    }
}

} // namespace qcx::integrals::internal
