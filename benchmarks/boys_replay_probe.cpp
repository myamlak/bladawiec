// The Boys replay probe: does the sorted region-partitioned path's advantage
// survive paying for the sort, on the arguments a real Fock build produces?
//
// The claim under test. The kernel's earlier runs carry a 3.2x (2026-08-29)
// to 3.68x (2026-09-08) penalty for unsorted input, and the batch entry that
// accepts arbitrary arguments and groups them internally (BoysAllN) is the
// shipped answer to it: the sorted region-partitioned path is that much faster
// than the general path, so handing it to every caller would deliver that
// factor. Both sides of that ratio are one construction - one order
// (n = 8) over 4.2M arguments drawn uniformly from [1e-4, 60], one side
// pre-partitioned by region and run through the region-specialised AVX2 lanes,
// the other through a mixed kernel that evaluates all three region paths per
// four-lane vector and blends. Neither side is a path a caller takes now: the
// general path is BoysAllOrders, which dispatches per call and evaluates only
// the path that call needs, and the grouped path is the many-argument entry
// BoysAllN, which classifies and groups its own arguments. Nor is that stream
// the argument distribution a build presents.
//
// What this probe measures. It enumerates the screened-in quartets of a real
// alkane/def2-SVP build in the engine's own emission order and, per call,
// takes the argument x = p q / (p + q) * |P - Q|^2 and its order L = LBra +
// LKet, then replays that stream two ways:
//
//   general - BoysAllOrders(L, x) per call, in the engine's order, which is what
//             the engine does today;
//   sorted  - a counting sort of the calls by (region, order), with the region
//             computed inside the timed pass, then one BoysAllN call per
//             bucket, which returns that bucket's whole ladder. The sort and
//             the entry are both inside the sorted timing: a sorted path that
//             wins only when the sort is free is not a win a batch entry
//             delivers.
//
// Both paths produce the same values. The general path returns orders 0..L per
// call; the sorted path asks the many-argument entry for the whole ladder of a
// bucket in one call. The entry's own classification and grouping of its
// arguments are inside the measured span: a caller pays for them whichever
// order its arguments arrive in, and the bucket split this probe pays for is
// charged on top.
//
// The argument is decomposed too, by counting only. x = mu * R^2, with mu the
// primitive pair's reduced exponent p q / (p + q) and R the separation of the
// two Gaussian product centres, so the two causes of a large argument can be
// told apart: a large separation belongs to distant pairs, a large exponent sum
// to tight core functions that may sit on adjacent atoms. Each high-argument
// tail is attributed exactly and in one pass into exponent-alone (mu >= T),
// separation-alone (R^2 >= T) and product-only, with the log-moments of both
// factors for a variance decomposition and the intra-pair atom distances of the
// tail - the three together decide whether a near-field-only large-molecule run
// would see a reduced tail or a differently-sourced one.
//
// Counted versus timed. The census half is a count and says nothing about cost.
// The replay half is wall-clock on whatever machine runs it: per chunk it takes
// the least-interfered of N rounds and prints the largest round beside it, so
// the load the machine carried is visible in the output rather than asserted
// away. A ratio read off a loaded machine is indicative only, and the spread
// printed with it is the evidence for that.
//
// Usage: qcx-bench-boys-replay [--basis <family>] [--carbons <n>]
//                              [--chunk <calls>] [--rounds <n>]
//                              [--replay-calls <n>] [--no-build]
#include "alkane_sto3g.hpp"
#include "boys/boys.hpp"
#include "boys_coefficients.hpp"
#include "internal/fock_screen.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/boys.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/lean_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/tensor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

/// The highest pair class the engine can dispatch to: two shells of
/// kMaxShellL = 6.
constexpr int kMaxPairClass = 12;

/// The highest quartet class.
constexpr int kMaxClass = 24;

/// The orders a bucket is keyed by: one bucket per (region, order).
constexpr int kOrderCount = qcx::integrals::kMaxBoysOrder + 1;

/// The region codes of the sort, in the library's own boundaries: A is
/// [0, kX0), B is [kX0, kX1), C is [kX1, inf). Region A owns x == 0.
constexpr int kRegionCount = 3;

/// The bucket count: (region, order) pairs.
constexpr std::size_t kBucketCount = static_cast<std::size_t>(kRegionCount) * kOrderCount;

/// The region boundaries, read from the library's own table.
constexpr double kRegionAX0 = boys::detail::kX0;
constexpr double kRegionBX1 = boys::detail::kX1;

/// The tail thresholds the factor attribution is taken at: the library's own
/// region-C boundary, then two round numbers above it.
constexpr std::array<double, 3> kTailThresholds{kRegionBX1, 100.0, 1000.0};

/// The near-field radii (bohr, product centres) the tails are read against.
constexpr std::array<double, 3> kNearFieldRadii{5.0, 10.0, 20.0};

/// The argument histogram's edges: half-decade bins from 1e-4 to 1e6, with
/// bin 0 reserved for x == 0 exactly.
constexpr int kHistBins = 21;
constexpr double kHistT0 = -4.0;
constexpr double kHistT1 = 6.0;

/// The joint (mu, R^2) table's decade bins.
constexpr int kMuBins = 12;
constexpr double kMuT0 = -6.0;
constexpr int kR2Bins = 8;
constexpr double kR2T0 = -2.0;

/// The intra-pair atom-distance bins, in bohr.
constexpr std::array<double, 8> kAtomBins{1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 14.0, 1e30};

double gSink = 0.0;

/// Prints an intra-pair atom-distance histogram as bohr ranges.
/// \param counts The bin counts.
/// \param total The count the shares are taken against.
void PrintAtomDistance(const std::array<std::size_t, kAtomBins.size()>& counts, double total) {
    static constexpr std::array<const char*, kAtomBins.size()> kLabels{"  d < 1",
                                                                       "1 <= d < 2",
                                                                       "2 <= d < 3",
                                                                       "3 <= d < 4",
                                                                       "4 <= d < 6",
                                                                       "6 <= d < 9",
                                                                       "9 <= d < 14",
                                                                       "d >= 14"};

    for (std::size_t b = 0; b < counts.size(); ++b)
    {
        std::printf(
            "      %-12s %12.4f%%", kLabels[b], 100.0 * static_cast<double>(counts[b]) / total);
        std::printf("%s", b % 2 == 1 ? "\n" : "");
    }

    std::printf("%s", kAtomBins.size() % 2 == 1 ? "\n" : "");
}

/// The region code of \p x, in the library's own boundaries.
/// \param x The Boys argument, >= 0.
/// \returns 0 (region A), 1 (region B) or 2 (region C).
inline int RegionOf(double x) noexcept {
    if (x < kRegionAX0)
    {
        return 0;
    }

    return x < kRegionBX1 ? 1 : 2;
}

/// The bucket of one call: its region times the order count plus its order.
/// \param region The call's region code.
/// \param order The call's class L.
/// \returns The bucket index.
inline std::size_t BucketOf(int region, int order) noexcept {
    return static_cast<std::size_t>(region) * kOrderCount + static_cast<std::size_t>(order);
}

/// One shell's primitive exponents and its center.
struct ShellPrims {
    std::vector<double> exponents;
    std::array<double, 3> center{0.0, 0.0, 0.0};
};

/// One canonical shell pair's role data.
struct PairRole {
    std::size_t rowPairs = 0; ///< contractionCount(a) * contractionCount(b).
    double atomDistance = 0.0; ///< The separation of the pair's two shell centers.
    std::vector<std::array<double, 4>> prim; ///< (p, Px, Py, Pz) rows.
};

/// The three ways the product x = mu * R^2 can put a call above a threshold,
/// plus the log-moments of both factors over those calls.
struct TailStats {
    std::size_t calls = 0;
    std::size_t exponentAlone = 0; ///< mu >= T: the separation plays no part.
    std::size_t separationAlone = 0; ///< R^2 >= T: the exponent sum plays no part.
    std::size_t productOnly = 0; ///< neither factor alone reaches T.
    std::array<std::size_t, kNearFieldRadii.size()> nearField{}; ///< R <= the radius.
    std::array<std::size_t, kAtomBins.size()> atomDistance{};

    // The log10 moments of both factors, for the variance decomposition:
    // Var(log x) = Var(log mu) + Var(log R^2) + 2 Cov.
    double sumMu = 0.0;
    double sumMu2 = 0.0;
    double sumR2 = 0.0;
    double sumR22 = 0.0;
    double sumCross = 0.0;

    /// Adds one tail call's log10 factors.
    /// \param logMu log10 of the reduced exponent.
    /// \param logR2 log10 of the squared product-centre separation.
    void AddLogs(double logMu, double logR2) {
        sumMu += logMu;
        sumMu2 += logMu * logMu;
        sumR2 += logR2;
        sumR22 += logR2 * logR2;
        sumCross += logMu * logR2;
    }
};

/// The counted census of the walk: the call count, the argument distribution
/// and the decomposition of the argument into its two factors.
struct Census {
    std::size_t quartets = 0;
    std::size_t calls = 0;
    std::size_t values = 0; ///< Sum of L + 1 over the calls.
    std::vector<std::size_t> callsByClass;
    std::array<std::size_t, kHistBins + 1> histogram{};
    std::array<std::size_t, kRegionCount> regionCalls{};
    std::array<TailStats, kTailThresholds.size()> tails{};
    std::array<std::size_t, kAtomBins.size()> atomDistance{};
    std::vector<std::size_t> jointAll; ///< [mu bin][R^2 bin], every call.
    std::vector<std::size_t> jointTail; ///< [mu bin][R^2 bin], the top tail only.
    std::size_t zeroCalls = 0;

    // The same log-moments over every call, not just the tails.
    double sumMu = 0.0;
    double sumMu2 = 0.0;
    double sumR2 = 0.0;
    double sumR22 = 0.0;
    double sumCross = 0.0;
    double minMu = std::numeric_limits<double>::max();
    double maxMu = 0.0;
    double minR = std::numeric_limits<double>::max();
    double maxR = 0.0;
};

/// The half-decade argument bin of \p x (0 is x == 0).
/// \param x The Boys argument, >= 0.
/// \returns The bin index.
std::size_t HistogramBin(double x) {
    if (x <= 0.0)
    {
        return 0;
    }

    const double t = (std::log10(x) - kHistT0) / (kHistT1 - kHistT0);
    const double clamped = std::clamp(t, 0.0, 1.0);

    return 1 + static_cast<std::size_t>(clamped * static_cast<double>(kHistBins - 1) + 0.5);
}

/// The joint table's mu bin of \p mu, decade resolution (bin 0 = underflow,
/// the last bin = overflow).
/// \param mu The reduced exponent, > 0.
/// \returns The bin index.
std::size_t MuBin(double mu) {
    const double decades = std::log10(mu) - kMuT0;

    if (decades <= 0.0)
    {
        return 0;
    }

    const auto bin = static_cast<std::size_t>(decades) + 1;

    return bin < static_cast<std::size_t>(kMuBins) ? bin : static_cast<std::size_t>(kMuBins);
}

/// The joint table's R^2 bin of \p r2, decade resolution.
/// \param r2 The squared product-centre separation, > 0.
/// \returns The bin index.
std::size_t R2Bin(double r2) {
    const double decades = std::log10(r2) - kR2T0;

    if (decades <= 0.0)
    {
        return 0;
    }

    const auto bin = static_cast<std::size_t>(decades) + 1;

    return bin < static_cast<std::size_t>(kR2Bins) ? bin : static_cast<std::size_t>(kR2Bins);
}

/// The intra-pair atom-distance bin of \p d, in bohr.
/// \param d The separation of a pair's two shell centers; 0 for a one-center pair.
/// \returns The bin index.
std::size_t AtomBin(double d) {
    for (std::size_t i = 0; i < kAtomBins.size(); ++i)
    {
        if (d < kAtomBins[i])
        {
            return i;
        }
    }

    return kAtomBins.size() - 1;
}

/// One chunk's buffers and the accumulators the walk fills.
///
/// The chunk is the replay's unit: a batch of calls the size a caller could
/// hand over at once. Everything the two paths need is inside it, so the sort
/// and the entry run over resident memory exactly as they would for a real
/// batch of that size.
struct Replay {
    std::size_t chunkCalls = 1u << 18;
    int rounds = 3;
    std::size_t limit = 0; ///< Replay at most this many calls (0 = all).

    std::vector<double> x;
    std::vector<int> order;
    std::vector<double> sorted;
    std::vector<std::size_t> counts;
    std::vector<std::size_t> cursor;
    std::vector<double> scratch;
    std::vector<double> generalOut;

    std::size_t calls = 0;
    std::size_t values = 0;
    std::size_t chunks = 0;

    double generalMin = 0.0;
    double generalMax = 0.0;
    double sortMin = 0.0;
    double sortMax = 0.0;
    double laneMin = 0.0;
    double laneMax = 0.0;

    std::array<double, kRegionCount> generalRegionMin{};
    std::array<double, kRegionCount> generalRegionMax{};
    std::vector<double> bucketLaneNanos; ///< The min round's per-bucket lanes.
    std::vector<std::size_t> bucketValues;
    std::array<std::size_t, kRegionCount> regionCalls{};
};

/// The nanoseconds a span took.
/// \param span The measured span.
/// \returns The span in nanoseconds.
double Nanos(std::chrono::steady_clock::duration span) {
    return std::chrono::duration<double, std::nano>(span).count();
}

/// The counting sort of one chunk by (region, order): the partition a batch
/// entry would pay for, region decided inside the pass.
/// \param replay The chunk's buffers.
/// \param n The chunk's call count.
void SortChunk(Replay& replay, std::size_t n) {
    std::fill(replay.counts.begin(), replay.counts.end(), std::size_t{0});

    for (std::size_t i = 0; i < n; ++i)
    {
        ++replay.counts[BucketOf(RegionOf(replay.x[i]), replay.order[i])];
    }

    std::size_t running = 0;

    for (std::size_t b = 0; b < kBucketCount; ++b)
    {
        replay.cursor[b] = running;
        running += replay.counts[b];
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        replay.sorted[replay.cursor[BucketOf(RegionOf(replay.x[i]), replay.order[i])]++] =
            replay.x[i];
    }
}

/// Runs one bucket through the many-argument entry: the whole ladder of the
/// bucket's order in one call, grouped internally.
/// \param n The bucket's order.
/// \param xs The bucket's arguments.
/// \param out Receives the ladder, plane-major: out[k * count + i].
/// \param count The bucket's element count.
void RunBucketEntry(int n, const double* xs, double* out, std::size_t count) {
    qcx::integrals::BoysAllN(n, xs, out, count);
}

/// Sets a chunk up before its accumulator loop: the working arrays the sort and
/// the lanes need.
/// \param replay The chunk's state.
void PrepareChunk(Replay& replay) {
    replay.x.assign(replay.chunkCalls, 0.0);
    replay.order.assign(replay.chunkCalls, 0);
    replay.sorted.assign(replay.chunkCalls, 0.0);
    replay.generalOut.assign(static_cast<std::size_t>(kOrderCount), 0.0);
    // The chip's widest need is one plane of every order, so the buffer is
    // sized for a chunk that lands entirely in one bucket: allocated once,
    // outside the timed passes, so no round pays for it.
    replay.scratch.assign(static_cast<std::size_t>(kOrderCount) * replay.chunkCalls, 0.0);
    replay.counts.assign(kBucketCount, 0);
    replay.cursor.assign(kBucketCount, 0);
    replay.bucketLaneNanos.assign(kBucketCount, 0.0);
    replay.bucketValues.assign(kBucketCount, 0);
}

/// Replays one full chunk through both paths and accumulates the timings.
/// \param replay The chunk's state; its arrays hold \ref Replay::chunkCalls calls.
void ReplayChunk(Replay& replay) {
    const std::size_t n = replay.chunkCalls;

    if (replay.limit != 0 && replay.calls >= replay.limit)
    {
        return;
    }

    double best = std::numeric_limits<double>::max();
    double worst = 0.0;

    for (int r = 0; r < replay.rounds; ++r)
    {
        const auto start = std::chrono::steady_clock::now();

        for (std::size_t i = 0; i < n; ++i)
        {
            qcx::integrals::BoysAllOrders(replay.order[i], replay.x[i], replay.generalOut.data());
            gSink += replay.generalOut[static_cast<std::size_t>(replay.order[i])];
        }

        const double span = Nanos(std::chrono::steady_clock::now() - start);
        best = std::min(best, span);
        worst = std::max(worst, span);
    }

    replay.generalMin += best;
    replay.generalMax += worst;

    // The general path's per-region split: the same call, for the calls of one
    // region only, so the sorted path's per-bucket entries have a counterpart.
    for (int region = 0; region < kRegionCount; ++region)
    {
        double regionBest = std::numeric_limits<double>::max();
        double regionWorst = 0.0;

        for (int r = 0; r < replay.rounds; ++r)
        {
            const auto start = std::chrono::steady_clock::now();

            for (std::size_t i = 0; i < n; ++i)
            {
                if (RegionOf(replay.x[i]) != region)
                {
                    continue;
                }

                qcx::integrals::BoysAllOrders(
                    replay.order[i], replay.x[i], replay.generalOut.data());
                gSink += replay.generalOut[static_cast<std::size_t>(replay.order[i])];
            }

            const double span = Nanos(std::chrono::steady_clock::now() - start);
            regionBest = std::min(regionBest, span);
            regionWorst = std::max(regionWorst, span);
        }

        replay.generalRegionMin[static_cast<std::size_t>(region)] += regionBest;
        replay.generalRegionMax[static_cast<std::size_t>(region)] += regionWorst;
    }

    best = std::numeric_limits<double>::max();
    worst = 0.0;

    for (int r = 0; r < replay.rounds; ++r)
    {
        const auto start = std::chrono::steady_clock::now();
        SortChunk(replay, n);
        const double span = Nanos(std::chrono::steady_clock::now() - start);
        best = std::min(best, span);
        worst = std::max(worst, span);
    }

    replay.sortMin += best;
    replay.sortMax += worst;

    std::vector<double> roundLanes(static_cast<std::size_t>(replay.rounds), 0.0);
    std::vector<double> bucketSpan(kBucketCount, 0.0);
    double bestBuckets = std::numeric_limits<double>::max();
    std::vector<double> bestBucketSpan(kBucketCount, 0.0);

    for (int r = 0; r < replay.rounds; ++r)
    {
        std::fill(bucketSpan.begin(), bucketSpan.end(), 0.0);
        const auto start = std::chrono::steady_clock::now();
        std::size_t offset = 0;

        for (std::size_t b = 0; b < kBucketCount; ++b)
        {
            const std::size_t count = replay.counts[b];

            if (count == 0)
            {
                continue;
            }

            const int order = static_cast<int>(b % static_cast<std::size_t>(kOrderCount));
            const std::size_t need = static_cast<std::size_t>(order + 1) * count;

            if (replay.scratch.size() < need)
            {
                replay.scratch.resize(need);
            }

            const auto bucketStart = std::chrono::steady_clock::now();
            RunBucketEntry(order, replay.sorted.data() + offset, replay.scratch.data(), count);
            bucketSpan[b] += Nanos(std::chrono::steady_clock::now() - bucketStart);
            gSink += replay.scratch[0];
            gSink += replay.scratch[need - 1];
            offset += count;
        }

        const double span = Nanos(std::chrono::steady_clock::now() - start);
        roundLanes[static_cast<std::size_t>(r)] = span;

        if (span < bestBuckets)
        {
            bestBuckets = span;
            bestBucketSpan = bucketSpan;
        }
    }

    double laneBest = std::numeric_limits<double>::max();
    double laneWorst = 0.0;

    for (const double span : roundLanes)
    {
        laneBest = std::min(laneBest, span);
        laneWorst = std::max(laneWorst, span);
    }

    replay.laneMin += laneBest;
    replay.laneMax += laneWorst;

    for (std::size_t b = 0; b < kBucketCount; ++b)
    {
        if (replay.counts[b] == 0)
        {
            continue;
        }

        replay.bucketLaneNanos[b] += bestBucketSpan[b];
        replay.bucketValues[b] +=
            static_cast<std::size_t>(b % static_cast<std::size_t>(kOrderCount) + 1) *
            replay.counts[b];
    }

    for (int region = 0; region < kRegionCount; ++region)
    {
        for (std::size_t o = 0; o < static_cast<std::size_t>(kOrderCount); ++o)
        {
            replay.regionCalls[static_cast<std::size_t>(region)] +=
                replay.counts[BucketOf(region, static_cast<int>(o))];
        }
    }

    replay.calls += n;
    replay.values += static_cast<std::size_t>(
        std::accumulate(replay.order.begin(), replay.order.end(), 0, [](int sum, int value) {
            return sum + value + 1;
        }));
    ++replay.chunks;
}

/// Checks the sorted path against the general one on one chunk, outside the
/// timed region: the same values must come out, and where they do not, the
/// difference is the two entries' documented bounds.
/// \param replay The chunk's state.
/// \returns The largest absolute difference, and the difference at the top
///          order of each call.
std::pair<double, double> VerifyChunk(Replay& replay) {
    const std::size_t n = replay.chunkCalls;
    std::vector<std::size_t> counts(kBucketCount, 0);

    for (std::size_t i = 0; i < n; ++i)
    {
        ++counts[BucketOf(RegionOf(replay.x[i]), replay.order[i])];
    }

    std::vector<std::size_t> cursor(kBucketCount, 0);
    std::vector<std::size_t> sortedIndex(n, 0);
    std::vector<double> sortedX(n, 0.0);
    std::size_t running = 0;

    for (std::size_t b = 0; b < kBucketCount; ++b)
    {
        cursor[b] = running;
        running += counts[b];
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        const std::size_t b = BucketOf(RegionOf(replay.x[i]), replay.order[i]);
        sortedX[cursor[b]] = replay.x[i];
        sortedIndex[cursor[b]] = i;
        ++cursor[b];
    }

    std::vector<double> lanes;
    std::vector<double> general(kOrderCount, 0.0);
    std::vector<double> back(n, 0.0);
    double worst = 0.0;
    double worstTop = 0.0;
    std::size_t offset = 0;

    for (std::size_t b = 0; b < kBucketCount; ++b)
    {
        const std::size_t count = counts[b];

        if (count == 0)
        {
            continue;
        }

        const int order = static_cast<int>(b % static_cast<std::size_t>(kOrderCount));
        lanes.assign(static_cast<std::size_t>(order + 1) * count, 0.0);
        RunBucketEntry(order, sortedX.data() + offset, lanes.data(), count);

        for (std::size_t p = 0; p < count; ++p)
        {
            const std::size_t call = sortedIndex[offset + p];
            qcx::integrals::BoysAllOrders(order, sortedX[offset + p], general.data());

            for (int k = 0; k <= order; ++k)
            {
                const double diff = std::abs(lanes[static_cast<std::size_t>(k) * count + p] -
                                             general[static_cast<std::size_t>(k)]);
                worst = std::max(worst, diff);
                back[call] = diff;
            }

            worstTop = std::max(worstTop, back[call]);
        }

        offset += count;
    }

    return {worst, worstTop};
}

qcx::Result<CpuTensor2> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basisSet) {
    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    const std::size_t n = kinetic->Shape()[0];

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*kinetic)(i, j) += (*nuclear)(i, j);
        }
    }

    kinetic->MarkHostDirty();

    return std::move(*kinetic);
}

qcx::Result<CpuTensor2> MakeDensity(std::size_t n) {
    auto density = CpuTensor2::Create({n, n});

    if (!density.has_value())
    {
        return std::unexpected(density.error());
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        (*density)(i, i) = 0.5;
    }

    density->MarkHostDirty();

    return std::move(*density);
}

void PrintUsage(const char* program) {
    std::printf("usage: %s [--basis <family>] [--carbons <n>] [--chunk <calls>] "
                "[--rounds <n>] [--replay-calls <n>] [--no-build]\n",
                program);
}

/// Adds one call to the census, counted only.
/// \param census The census.
/// \param x The Boys argument, >= 0.
/// \param mu The reduced exponent, > 0.
/// \param r2 The squared product-centre separation, > 0.
/// \param lClass The call's class L.
/// \param atomDistance The larger intra-pair shell-center separation of the two pairs.
void CountCall(Census& census, double x, double mu, double r2, int lClass, double atomDistance) {
    ++census.calls;
    census.values += static_cast<std::size_t>(lClass + 1);
    ++census.callsByClass[static_cast<std::size_t>(lClass)];
    ++census.histogram[HistogramBin(x)];
    ++census.regionCalls[static_cast<std::size_t>(RegionOf(x))];
    ++census.atomDistance[AtomBin(atomDistance)];

    if (x <= 0.0)
    {
        ++census.zeroCalls;

        return;
    }

    const double logMu = std::log10(mu);
    const double logR2 = std::log10(r2);
    const double logX = logMu + logR2;

    census.sumMu += logMu;
    census.sumMu2 += logMu * logMu;
    census.sumR2 += logR2;
    census.sumR22 += logR2 * logR2;
    census.sumCross += logMu * logR2;
    census.minMu = std::min(census.minMu, mu);
    census.maxMu = std::max(census.maxMu, mu);
    census.minR = std::min(census.minR, std::sqrt(r2));
    census.maxR = std::max(census.maxR, std::sqrt(r2));
    ++census.jointAll[static_cast<std::size_t>(MuBin(mu)) * (kR2Bins + 1) + R2Bin(r2)];
    (void)logX;

    for (std::size_t t = 0; t < kTailThresholds.size(); ++t)
    {
        if (x < kTailThresholds[t])
        {
            continue;
        }

        TailStats& tail = census.tails[t];
        ++tail.calls;
        tail.AddLogs(logMu, logR2);
        ++tail.atomDistance[AtomBin(atomDistance)];

        for (std::size_t r = 0; r < kNearFieldRadii.size(); ++r)
        {
            if (std::sqrt(r2) <= kNearFieldRadii[r])
            {
                ++tail.nearField[r];
            }
        }

        if (mu >= kTailThresholds[t])
        {
            ++tail.exponentAlone;
        } else if (r2 >= kTailThresholds[t])
        {
            ++tail.separationAlone;
        } else
        {
            ++tail.productOnly;
        }

        if (t + 1 == kTailThresholds.size())
        {
            ++census.jointTail[static_cast<std::size_t>(MuBin(mu)) * (kR2Bins + 1) + R2Bin(r2)];
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string family = "def2-svp";
    std::size_t carbons = 24;
    bool runBuild = true;
    Replay replay;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--basis" && i + 1 < argc)
        {
            family = argv[++i];
        } else if (arg == "--carbons" && i + 1 < argc)
        {
            carbons = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else if (arg == "--chunk" && i + 1 < argc)
        {
            replay.chunkCalls = static_cast<std::size_t>(std::atoll(argv[++i]));
        } else if (arg == "--rounds" && i + 1 < argc)
        {
            replay.rounds = std::atoi(argv[++i]);
        } else if (arg == "--replay-calls" && i + 1 < argc)
        {
            replay.limit = static_cast<std::size_t>(std::atoll(argv[++i]));
        } else if (arg == "--no-build")
        {
            runBuild = false;
        } else
        {
            PrintUsage(argv[0]);

            return 1;
        }
    }

    if (replay.chunkCalls == 0 || replay.rounds < 1)
    {
        PrintUsage(argv[0]);

        return 1;
    }

    const auto accuracy = qcx::integrals::AccuracyPreset::kNormal;
    auto molecule = qcx::testing::MakeAlkaneSto3g(carbons);

    if (!molecule.has_value())
    {
        std::printf("molecule: %s\n", molecule.error().message.c_str());

        return 1;
    }

    const std::string basisDirectory = std::string(QcxBasisDataDir) + "/" + family;
    auto basis = qcx::basisset::ParseNwchemDirectory(basisDirectory);

    if (!basis.has_value())
    {
        std::printf("basis %s: %s\n", basisDirectory.c_str(), basis.error().message.c_str());

        return 1;
    }

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        std::printf("pair list: %s\n", pairList.error().message.c_str());

        return 1;
    }

    std::printf("fixture: c%zuH%zu / %s, %zu basis functions, %zu canonical pairs\n",
                carbons,
                2 * carbons + 2,
                family.c_str(),
                pairList->functionCount,
                pairList->pairs.size());
    std::printf("AVX2 tier: %s\n",
                qcx::integrals::BoysAvx2Available() ? "available" : "NOT available");
    std::printf("region boundaries from the library's own table: kX0 %.17g  kX1 %.17g\n",
                kRegionAX0,
                kRegionBX1);

    std::vector<ShellPrims> shells(pairList->shells.size());
    const auto& atoms = molecule->Atoms();
    const auto& coordinates = molecule->CoordinatesBohr();

    for (std::size_t s = 0; s < pairList->shells.size(); ++s)
    {
        const qcx::integrals::ShellInfo& info = pairList->shells[s];
        const qcx::basisset::ElementBasis* element =
            basis->Find(atoms[info.atomIndex].atomicNumber);

        if (element == nullptr)
        {
            std::printf("shell %zu: the basis has no entry for %s\n",
                        s,
                        atoms[info.atomIndex].symbol.c_str());

            return 1;
        }

        shells[s].exponents = element->shells[info.elementShellIndex].exponents;
        shells[s].center = {coordinates(info.atomIndex, 0),
                            coordinates(info.atomIndex, 1),
                            coordinates(info.atomIndex, 2)};
    }

    const std::size_t nPairs = pairList->pairs.size();
    std::vector<PairRole> roles(nPairs);
    std::vector<std::vector<std::size_t>> byL(static_cast<std::size_t>(kMaxPairClass) + 1);

    for (std::size_t k = 0; k < nPairs; ++k)
    {
        const qcx::integrals::ShellPairIndex& pair = pairList->pairs[k];
        const ShellPrims& first = shells[pair.i];
        const ShellPrims& second = shells[pair.j];
        const int la = pairList->shells[pair.i].angularMomentum;
        const int lb = pairList->shells[pair.j].angularMomentum;
        roles[k].rowPairs =
            pairList->shells[pair.i].contractionCount * pairList->shells[pair.j].contractionCount;

        const double ax = first.center[0] - second.center[0];
        const double ay = first.center[1] - second.center[1];
        const double az = first.center[2] - second.center[2];
        roles[k].atomDistance = std::sqrt(ax * ax + ay * ay + az * az);

        for (const double a : first.exponents)
        {
            for (const double b : second.exponents)
            {
                const double p = a + b;
                roles[k].prim.push_back({p,
                                         (a * first.center[0] + b * second.center[0]) / p,
                                         (a * first.center[1] + b * second.center[1]) / p,
                                         (a * first.center[2] + b * second.center[2]) / p});
            }
        }

        if (la + lb <= kMaxPairClass)
        {
            byL[static_cast<std::size_t>(la + lb)].push_back(k);
        }
    }

    auto qSchwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);

    if (!qSchwarz.has_value())
    {
        std::printf("schwarz: %s\n", qSchwarz.error().message.c_str());

        return 1;
    }

    const double cutoff =
        qcx::integrals::SchwarzThreshold(accuracy) * qcx::integrals::internal::kNeighborListSlack;

    qcx::integrals::FockBuildStats stats;

    if (runBuild)
    {
        auto core = BuildCoreHamiltonian(*molecule, *basis);

        if (!core.has_value())
        {
            std::printf("core: %s\n", core.error().message.c_str());

            return 1;
        }

        auto density = MakeDensity(pairList->functionCount);

        if (!density.has_value())
        {
            std::printf("density: %s\n", density.error().message.c_str());

            return 1;
        }

        qcx::integrals::LeanFockBuildOptions options;
        options.accuracy = accuracy;
        auto builder =
            qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, options);

        if (!builder.has_value())
        {
            std::printf("builder: %s\n", builder.error().message.c_str());

            return 1;
        }

        auto fock = builder->BuildFock(*density, &stats);

        if (!fock.has_value())
        {
            std::printf("build: %s\n", fock.error().message.c_str());

            return 1;
        }

        std::printf("engine: %zu fp64 quartets, %zu VRR calls, %zu primitive passes\n",
                    stats.fp64QuartetCount,
                    stats.kernelVrrQuadruples,
                    stats.kernelPrimPasses);
    }

    // ---- The walk, counted and replayed ------------------------------------
    Census census;
    census.callsByClass.assign(kMaxClass + 1, 0);
    census.jointAll.assign(static_cast<std::size_t>(kMuBins + 1) * (kR2Bins + 1), 0);
    census.jointTail.assign(static_cast<std::size_t>(kMuBins + 1) * (kR2Bins + 1), 0);

    PrepareChunk(replay);

    std::size_t chunkFill = 0;
    bool verified = false;
    double worstDiff = 0.0;
    double worstTopDiff = 0.0;

    const auto walkStart = std::chrono::steady_clock::now();
    std::vector<std::size_t> candidates;

    for (int lBra = 0; lBra <= kMaxPairClass; ++lBra)
    {
        for (int lKet = lBra; lKet <= kMaxPairClass; ++lKet)
        {
            const auto& braList = byL[static_cast<std::size_t>(lBra)];
            const auto& ketList = byL[static_cast<std::size_t>(lKet)];

            for (const std::size_t kp : ketList)
            {
                const auto& ketPrim = roles[kp].prim;
                const double qk = (*qSchwarz)[kp];

                candidates.clear();

                for (const std::size_t bp : braList)
                {
                    // The canonical cell is (row, ket) with ket <= row; when
                    // both sides carry the same class each unordered pair is
                    // visited twice and the index order breaks the tie.
                    if (lBra == lKet && bp < kp)
                    {
                        continue;
                    }

                    if ((*qSchwarz)[bp] * qk < cutoff)
                    {
                        continue;
                    }

                    candidates.push_back(bp);
                }

                if (candidates.empty())
                {
                    continue;
                }

                // The documented canonical emission order inside a class:
                // (ketPair, braRowPairs, braPair) ascending.
                std::stable_sort(
                    candidates.begin(), candidates.end(), [&](std::size_t x, std::size_t y) {
                        if (roles[x].rowPairs != roles[y].rowPairs)
                        {
                            return roles[x].rowPairs < roles[y].rowPairs;
                        }

                        return x < y;
                    });

                const int lClass = lBra + lKet;
                const std::size_t nPrimKet = ketPrim.size();
                const double dAtomKet = roles[kp].atomDistance;

                for (std::size_t g = 0; g < nPrimKet; ++g)
                {
                    for (const std::size_t bp : candidates)
                    {
                        const PairRole& bra = roles[bp];
                        const double dAtom = std::max(bra.atomDistance, dAtomKet);
                        ++census.quartets;

                        for (const auto& prim : bra.prim)
                        {
                            const double p = prim[0];
                            const double q = ketPrim[g][0];
                            const double dx = prim[1] - ketPrim[g][1];
                            const double dy = prim[2] - ketPrim[g][2];
                            const double dz = prim[3] - ketPrim[g][3];
                            const double r2 = dx * dx + dy * dy + dz * dz;
                            const double mu = (p * q) / (p + q);
                            const double x = mu * r2;

                            CountCall(census, x, mu, r2, lClass, dAtom);

                            replay.x[chunkFill] = x;
                            replay.order[chunkFill] = lClass;
                            ++chunkFill;

                            if (chunkFill < replay.chunkCalls)
                            {
                                continue;
                            }

                            if (!verified)
                            {
                                const auto check = VerifyChunk(replay);
                                worstDiff = check.first;
                                worstTopDiff = check.second;
                                verified = true;
                            }

                            ReplayChunk(replay);
                            chunkFill = 0;
                        }
                    }
                }
            }
        }
    }

    if (chunkFill != 0 && chunkFill == replay.chunkCalls)
    {
        ReplayChunk(replay);
    }

    const double walkSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - walkStart).count();
    std::printf("walked: %zu screened-in quartets, %zu Boys calls, %zu values (sum of L + 1), "
                "%.1f s\n",
                census.quartets,
                census.calls,
                census.values,
                walkSeconds);

    if (runBuild)
    {
        std::printf("engine cross-check: quartets %s, calls %s\n",
                    census.quartets == stats.fp64QuartetCount ? "MATCH" : "MISMATCH",
                    census.calls == stats.kernelVrrQuadruples ? "MATCH" : "MISMATCH");
    }

    // ---- Report: the counted distribution ----------------------------------
    const double callCount = static_cast<double>(census.calls);
    const auto share = [&](std::size_t count) {
        return callCount == 0.0 ? 0.0 : 100.0 * static_cast<double>(count) / callCount;
    };

    double classSum = 0.0;

    for (int l = 0; l <= kMaxClass; ++l)
    {
        classSum += static_cast<double>(l * census.callsByClass[static_cast<std::size_t>(l)]);
    }

    std::printf("\ncounted - the argument distribution:\n");
    std::printf("  mean class L over the calls %.3f, mean ladder length %.3f\n",
                classSum / callCount,
                static_cast<double>(census.values) / callCount);
    std::printf("  regions: A %.4f%%  B %.4f%%  C %.4f%%\n",
                share(census.regionCalls[0]),
                share(census.regionCalls[1]),
                share(census.regionCalls[2]));
    std::printf("  reduced exponent mu %.3g..%.3g   product-centre separation R %.3g..%.3g bohr\n",
                census.minMu,
                census.maxMu,
                census.minR,
                census.maxR);

    std::printf("\n  argument histogram (calls; bin 0 is x == 0, then log10 edges):\n");

    for (int i = 0; i <= kHistBins; ++i)
    {
        const std::size_t count = census.histogram[static_cast<std::size_t>(i)];

        if (count == 0)
        {
            continue;
        }

        if (i == 0)
        {
            std::printf("    %14s  %14zu  %8.4f%%\n", "x == 0", count, share(count));

            continue;
        }

        const double lo = kHistT0 + (kHistT1 - kHistT0) * static_cast<double>(i - 1) /
                                        static_cast<double>(kHistBins - 1);
        const double hi = kHistT0 + (kHistT1 - kHistT0) * static_cast<double>(i) /
                                        static_cast<double>(kHistBins - 1);
        std::printf("    [1e%-6.3g,1e%-6.3g)  %14zu  %8.4f%%\n", lo, hi, count, share(count));
    }

    // ---- Report: the two factors -------------------------------------------
    const auto varianceShare =
        [](double sumA, double sumA2, double sumB, double sumB2, double sumCross, std::size_t n) {
            const double count = static_cast<double>(n);
            const double varAll =
                (sumA2 + sumB2 + 2.0 * sumCross) / count - std::pow((sumA + sumB) / count, 2.0);
            const double varA = sumA2 / count - std::pow(sumA / count, 2.0);
            const double varB = sumB2 / count - std::pow(sumB / count, 2.0);
            const double cov = sumCross / count - (sumA / count) * (sumB / count);

            return std::array<double, 3>{
                varAll == 0.0 ? 0.0 : 100.0 * varA / varAll,
                varAll == 0.0 ? 0.0 : 100.0 * varB / varAll,
                varAll == 0.0 ? 0.0 : 100.0 * 2.0 * cov / varAll,
            };
        };

    std::printf("\ncounted - x = mu * R^2, the two factors of the argument:\n");
    const auto overallParts = varianceShare(
        census.sumMu, census.sumMu2, census.sumR2, census.sumR22, census.sumCross, census.calls);
    std::printf("  Var(log10 x) decomposition, every call: mu %.2f%%  R^2 %.2f%%  2Cov %.2f%%\n",
                overallParts[0],
                overallParts[1],
                overallParts[2]);

    for (std::size_t t = 0; t < kTailThresholds.size(); ++t)
    {
        const TailStats& tail = census.tails[t];
        const auto parts = varianceShare(
            tail.sumMu, tail.sumMu2, tail.sumR2, tail.sumR22, tail.sumCross, tail.calls);

        std::printf("\n  tail x >= %.6g : %zu calls (%.4f%% of all)\n",
                    kTailThresholds[t],
                    tail.calls,
                    share(tail.calls));
        std::printf("    exponent alone (mu >= T)      %14zu  %8.4f%% of the tail\n",
                    tail.exponentAlone,
                    100.0 * static_cast<double>(tail.exponentAlone) /
                        static_cast<double>(tail.calls));
        std::printf("    separation alone (R^2 >= T)   %14zu  %8.4f%% of the tail\n",
                    tail.separationAlone,
                    100.0 * static_cast<double>(tail.separationAlone) /
                        static_cast<double>(tail.calls));
        std::printf("    product only                  %14zu  %8.4f%% of the tail\n",
                    tail.productOnly,
                    100.0 * static_cast<double>(tail.productOnly) /
                        static_cast<double>(tail.calls));
        std::printf("    Var(log10 x) decomposition: mu %.2f%%  R^2 %.2f%%  2Cov %.2f%%\n",
                    parts[0],
                    parts[1],
                    parts[2]);
        std::printf(
            "    R <= 5/10/20 bohr: %.4f%% / %.4f%% / %.4f%% of the tail\n",
            100.0 * static_cast<double>(tail.nearField[0]) / static_cast<double>(tail.calls),
            100.0 * static_cast<double>(tail.nearField[1]) / static_cast<double>(tail.calls),
            100.0 * static_cast<double>(tail.nearField[2]) / static_cast<double>(tail.calls));
        std::printf("    intra-pair atom distance of the tail:\n");
        PrintAtomDistance(tail.atomDistance, static_cast<double>(tail.calls));
    }

    std::printf("\n  intra-pair atom distance, every call:\n");
    PrintAtomDistance(census.atomDistance, callCount);

    const auto printJoint = [&](const char* title, const std::vector<std::size_t>& table) {
        std::printf("\n  %s (rows log10 mu, columns log10 R^2, decade bins):\n", title);
        std::printf("       mu \\ R2");

        for (int c = 0; c <= kR2Bins; ++c)
        {
            std::printf(" %10.4g", kR2T0 + static_cast<double>(c - 1));
        }

        std::printf("\n");

        for (int r = 0; r <= kMuBins; ++r)
        {
            std::size_t row = 0;

            for (int c = 0; c <= kR2Bins; ++c)
            {
                row += table[static_cast<std::size_t>(r) * (kR2Bins + 1) +
                             static_cast<std::size_t>(c)];
            }

            if (row == 0)
            {
                continue;
            }

            std::printf("  %10.4g", kMuT0 + static_cast<double>(r - 1));

            for (int c = 0; c <= kR2Bins; ++c)
            {
                std::printf(" %10zu",
                            table[static_cast<std::size_t>(r) * (kR2Bins + 1) +
                                  static_cast<std::size_t>(c)]);
            }

            std::printf("\n");
        }
    };

    printJoint("every call", census.jointAll);
    printJoint("the x >= 1000 tail", census.jointTail);

    // ---- Report: the replay ------------------------------------------------
    const double replayed = static_cast<double>(replay.calls);

    if (replay.calls == 0)
    {
        std::printf("\nno chunk was replayed (the walk produced fewer calls than --chunk)\n");

        return 0;
    }

    const double sortedMin = replay.sortMin + replay.laneMin;
    const double sortedMax = replay.sortMax + replay.laneMax;

    std::printf(
        "\ntimed - the replay (chunk %zu calls, %d rounds, least-interfered round per chunk;\n"
        "        the largest round is printed beside it as the load the machine carried):\n",
        replay.chunkCalls,
        replay.rounds);
    std::printf("  replayed %zu calls in %zu chunks, %zu values (mean ladder %.3f)\n",
                replay.calls,
                replay.chunks,
                replay.values,
                static_cast<double>(replay.values) / replayed);
    std::printf(
        "  general  BoysAllOrders ladder   %10.3f ns/call  %7.3f ns/value   [max round %+6.1f%%]\n",
        replay.generalMin / replayed,
        replay.generalMin / static_cast<double>(replay.values),
        100.0 * (replay.generalMax / replay.generalMin - 1.0));
    std::printf(
        "  sorted   counting sort        %10.3f ns/call  %7.3f ns/value   [max round %+6.1f%%]\n",
        replay.sortMin / replayed,
        replay.sortMin / static_cast<double>(replay.values),
        100.0 * (replay.sortMax / replay.sortMin - 1.0));
    std::printf(
        "  sorted   batch entry         %10.3f ns/call  %7.3f ns/value   [max round %+6.1f%%]\n",
        replay.laneMin / replayed,
        replay.laneMin / static_cast<double>(replay.values),
        100.0 * (replay.laneMax / replay.laneMin - 1.0));
    std::printf(
        "  sorted   total (sort + entry) %10.3f ns/call  %7.3f ns/value   [max round %+6.1f%%]\n",
        sortedMin / replayed,
        sortedMin / static_cast<double>(replay.values),
        100.0 * (sortedMax / sortedMin - 1.0));
    std::printf("  ratio: general / sorted = %.3fx   (sorted / general = %.3fx)\n"
                "         the earlier measurement of this ratio is 3.2 to 3.68x, on a uniform "
                "argument stream at one order\n",
                replay.generalMin / sortedMin,
                sortedMin / replay.generalMin);
    std::printf("  the sort alone is %.1f%% of the sorted total\n",
                100.0 * replay.sortMin / sortedMin);

    std::printf(
        "\n  per region (the general path over one region's calls; the entry over its buckets):\n");
    std::printf("    region      calls     share   general ns/call   entry ns/call   ratio\n");

    for (int region = 0; region < kRegionCount; ++region)
    {
        const std::size_t calls = replay.regionCalls[static_cast<std::size_t>(region)];
        double laneNanos = 0.0;
        std::size_t laneValues = 0;

        for (std::size_t o = 0; o < static_cast<std::size_t>(kOrderCount); ++o)
        {
            const std::size_t b = BucketOf(region, static_cast<int>(o));
            laneNanos += replay.bucketLaneNanos[b];
            laneValues += replay.bucketValues[b];
        }

        const double generalPerCall =
            calls == 0 ? 0.0
                       : replay.generalRegionMin[static_cast<std::size_t>(region)] /
                             static_cast<double>(calls);
        const double lanePerCall = calls == 0 ? 0.0 : laneNanos / static_cast<double>(calls);

        std::printf("    %-8s %11zu  %7.4f%%  %15.3f  %15.3f  %7.3fx\n",
                    region == 0 ? "A" : (region == 1 ? "B" : "C"),
                    calls,
                    share(calls),
                    generalPerCall,
                    lanePerCall,
                    lanePerCall == 0.0 ? 0.0 : generalPerCall / lanePerCall);
        std::printf("             values %zu, entry ns/value %.3f, general max round %+6.1f%%\n",
                    laneValues,
                    laneValues == 0 ? 0.0 : laneNanos / static_cast<double>(laneValues),
                    replay.generalRegionMin[static_cast<std::size_t>(region)] == 0.0
                        ? 0.0
                        : 100.0 * (replay.generalRegionMax[static_cast<std::size_t>(region)] /
                                       replay.generalRegionMin[static_cast<std::size_t>(region)] -
                                   1.0));
    }

    std::printf("\n  per bucket (region, L): the entry call the sorted path pays, and its yield\n");
    std::printf("    region  L      calls      values   entry ns/value   entry ns/call\n");

    for (int region = 0; region < kRegionCount; ++region)
    {
        for (std::size_t o = 0; o < static_cast<std::size_t>(kOrderCount); ++o)
        {
            const std::size_t b = BucketOf(region, static_cast<int>(o));
            const std::size_t values = replay.bucketValues[b];

            if (values == 0)
            {
                continue;
            }

            const std::size_t calls = values / (o + 1);

            std::printf("    %-6s %3zu %10zu %11zu %15.3f %14.3f\n",
                        region == 0 ? "A" : (region == 1 ? "B" : "C"),
                        o,
                        calls,
                        values,
                        replay.bucketLaneNanos[b] / static_cast<double>(values),
                        replay.bucketLaneNanos[b] / static_cast<double>(calls));
        }
    }

    if (verified)
    {
        std::printf("\n  sorted against general on the first chunk: max |diff| %.3e over every "
                    "order, %.3e at the top order (both entries hold the double batch lane's "
                    "5.5e-14 per value, so the difference is bounded by 1.1e-13)\n",
                    worstDiff,
                    worstTopDiff);
    }

    std::printf("\nsink %.6e\n", gSink);

    return 0;
}
