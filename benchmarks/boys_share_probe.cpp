// The Boys-share probe: what fraction of a real integral evaluation the Boys
// function is.
//
// Counts first, because a count does not care what else the machine is doing.
// The MD engine's VRR calls BoysBatch once per primitive quadruple of every
// screened-in shell quartet, so the Boys call count of one build is a
// combinatorial property of the screened quartet set - and that set is
// enumerated here a second time, from the public pair list and Schwarz
// bounds, and validated against the engine's own counters before any number
// is read off it. The nmax (= the quartet's total class LBra + LKet) and the
// argument x are properties of the same enumeration, so their distributions
// cost nothing beyond the walk.
//
// The cost half IS a timing and is labelled as one: the per-call cost of
// BoysBatch on the enumerated (nmax, x) distribution, and the engine's own
// phase spans of the kernel that contains the call. Both are taken on
// whatever machine runs this probe, so the share is reported as an interval -
// the cost read at each argument bin's center and at both its edges, which is
// the whole width of the one approximation the count half cannot remove.
//
// Usage: qcx-bench-boys-share [--basis <family>] [--carbons <n>]
//                             [--chunks <n>] [--no-build]
#include "alkane_sto3g.hpp"
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
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

/// The highest quartet class the engine can dispatch to: two shells of
/// kMaxShellL = 6 per side.
constexpr int kMaxClass = 24;

/// The argument bins of the cost table: bin 0 is x == 0 exactly (the
/// library's own short-circuit), bins 1.. are log-spaced over this range.
constexpr std::size_t kArgumentBins = 240;
constexpr double kArgumentLo = 1e-9;
constexpr double kArgumentHi = 1e6;

/// The Boys library's region boundaries: A is [0, kRegionAX0), B is
/// [kRegionAX0, kRegionBX1), C is [kRegionBX1, inf).
constexpr double kRegionAX0 = 1.18998481521084840e+01;
constexpr double kRegionBX1 = 2.89893377388207400e+01;

/// The four argument regions the cost table is read against.
enum class Region : int { kZero = 0, kA = 1, kB = 2, kC = 3 };

/// The region of \p x.
/// \param x The Boys argument, >= 0.
/// \returns The region.
Region RegionOf(double x) {
    if (x <= 0.0)
    {
        return Region::kZero;
    }

    if (x < kRegionAX0)
    {
        return Region::kA;
    }

    if (x < kRegionBX1)
    {
        return Region::kB;
    }

    return Region::kC;
}

/// The cost table's timing rounds and their floor: a round is repeated until
/// it spans at least this many nanoseconds or the repeat cap is hit, and the
/// MINIMUM round is the reading (the least interfered one).
constexpr double kRoundFloorNanos = 60000.0;
constexpr std::size_t kRoundRepeatCap = 200000;
constexpr int kRoundCount = 9;

double gSink = 0.0;

/// The argument bin of \p x.
/// \param x The Boys argument, >= 0.
/// \returns The bin index, 0 for x == 0.
std::size_t ArgumentBin(double x) {
    if (x <= 0.0)
    {
        return 0;
    }

    const double t = (std::log10(x) - std::log10(kArgumentLo)) /
                     (std::log10(kArgumentHi) - std::log10(kArgumentLo));
    const double clamped = std::clamp(t, 0.0, 1.0);

    return 1 + static_cast<std::size_t>(clamped * static_cast<double>(kArgumentBins - 1) + 0.5);
}

/// The representative argument of a bin: the geometric center of its edges.
/// \param bin The bin index.
/// \returns The representative x.
double ArgumentOf(std::size_t bin) {
    if (bin == 0)
    {
        return 0.0;
    }

    const double span = std::log10(kArgumentHi) - std::log10(kArgumentLo);
    const double t = static_cast<double>(bin - 1) / static_cast<double>(kArgumentBins - 1);

    return std::pow(10.0, std::log10(kArgumentLo) + t * span);
}

/// The slice-triangle steps of one class-L recurrence: the inner loop's
/// iteration count, summed over the T = 1..L slices and their tiers. A count,
/// so it says nothing about what a step costs - only how many the ladder takes
/// against the one Boys call that seeds it.
/// \param lClass The quartet's class L.
/// \returns The step count.
std::size_t RecurrenceSteps(int lClass) {
    std::size_t steps = 0;

    for (int t = 1; t <= lClass; ++t)
    {
        for (int n = 1; n <= t; ++n)
        {
            steps += static_cast<std::size_t>((n + 1) * (n + 2) / 2);
        }
    }

    return steps;
}

/// One shell's primitive exponents and its center, in pairList.shells order.
struct ShellPrims {
    std::vector<double> exponents;
    std::array<double, 3> center{0.0, 0.0, 0.0};
};

/// The primitive pairs of one canonical shell pair, flattened.
struct PairPrims {
    std::size_t offset = 0; ///< First index into the probe's flat arrays.
    std::size_t count = 0; ///< Primitive pairs of this pair.
    int lBra = 0; ///< la + lb.
};

/// The enumerated screened quartet set: the engine's cells, re-derived.
struct Census {
    std::size_t quartets = 0; ///< Screened-in canonical cells.
    std::size_t calls = 0; ///< Boys calls (primitive quadruples).
    std::vector<std::size_t> quartetsByClass; ///< Per class L, cells.
    std::vector<std::size_t> callsByClass; ///< Per class L, Boys calls.
    std::vector<std::size_t> binCounts; ///< Per (L, bin), Boys calls.
    std::vector<std::size_t> regionCalls; ///< Per Region, Boys calls.
    std::size_t recurrenceSteps = 0; ///< Slice-triangle steps over the cells.
    std::vector<double> argumentQuantiles; ///< Sampled x quantiles.
};

/// One timing cell's reading: the least interfered round and the middle one.
struct CallCost {
    double min = 0.0; ///< The smallest round: the least interfered reading.
    double median = 0.0; ///< The middle round: how much load moves the reading.
};

/// Times one BoysBatch call at (nmax, x) over \p kRoundCount rounds.
/// \param nmax The batch's highest order.
/// \param x The argument.
/// \returns Nanoseconds per call, at the minimum and the median round.
CallCost TimeBoysCall(int nmax, double x) {
    double out[1 + qcx::integrals::kMaxBoysOrder];
    std::size_t reps = 512;

    // Calibrate the repeat count so a round clears the floor: the cheap
    // arguments (x = 0's fill loop) would otherwise be timed by a clock
    // reading shorter than its own resolution.
    const auto calStart = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < reps; ++i)
    {
        qcx::integrals::BoysBatch(nmax, x, out);
    }

    const auto calStop = std::chrono::steady_clock::now();
    gSink += out[nmax];

    const double calNanos =
        std::chrono::duration<double, std::nano>(calStop - calStart).count() / reps;

    if (calNanos > 0.0)
    {
        const double wanted = kRoundFloorNanos / calNanos;
        reps = static_cast<std::size_t>(
            std::clamp(wanted, static_cast<double>(reps), static_cast<double>(kRoundRepeatCap)));
    }

    std::array<double, kRoundCount> rounds{};

    for (int round = 0; round < kRoundCount; ++round)
    {
        const auto start = std::chrono::steady_clock::now();

        for (std::size_t i = 0; i < reps; ++i)
        {
            qcx::integrals::BoysBatch(nmax, x, out);
        }

        const auto stop = std::chrono::steady_clock::now();
        gSink += out[nmax];
        rounds[static_cast<std::size_t>(round)] =
            std::chrono::duration<double, std::nano>(stop - start).count() /
            static_cast<double>(reps);
    }

    std::sort(rounds.begin(), rounds.end());

    return CallCost{rounds.front(), rounds[kRoundCount / 2]};
}

/// Times the census's own call stream: the (nmax, x) pairs drawn in the
/// proportions the walk measured, replayed back to back.
///
/// The per-bin table repeats one argument per timing cell and so inherits a
/// perfectly predicted region branch and a warm loop; the kernel's call site
/// presents the real mix instead, with the argument changing every call. This
/// loop is that mix, so its per-call cost is the one the kernel pays - the
/// per-bin table stays for the class and region decomposition.
/// \param census The enumerated call distribution.
/// \returns Nanoseconds per call, at the minimum and the median round.
CallCost TimeMixedStream(const Census& census) {
    // The cumulative distribution over (class, bin), so one draw is a binary
    // search and the stream needs no per-call allocation.
    std::vector<double> cumulative;
    cumulative.reserve(census.binCounts.size());
    double running = 0.0;

    for (const std::size_t count : census.binCounts)
    {
        running += static_cast<double>(count);
        cumulative.push_back(running);
    }

    constexpr std::size_t kStreamLength = 1u << 22;
    const double halfStep = 0.5 * (std::log10(kArgumentHi) - std::log10(kArgumentLo)) /
                            static_cast<double>(kArgumentBins - 1);
    std::vector<std::pair<int, double>> stream;
    stream.reserve(kStreamLength);
    std::mt19937_64 rng(20260921);
    std::uniform_real_distribution<double> draw(0.0, running);
    std::uniform_real_distribution<double> jitter(-halfStep, halfStep);

    for (std::size_t i = 0; i < kStreamLength; ++i)
    {
        const auto at = std::lower_bound(cumulative.begin(), cumulative.end(), draw(rng));
        const std::size_t index =
            static_cast<std::size_t>(at - cumulative.begin()) % cumulative.size();
        const std::size_t bin = index % (kArgumentBins + 1);
        const double x = bin == 0 ? 0.0 : ArgumentOf(bin) * std::pow(10.0, jitter(rng));

        stream.emplace_back(static_cast<int>(index / (kArgumentBins + 1)), x);
    }

    double out[1 + qcx::integrals::kMaxBoysOrder];
    std::array<double, kRoundCount> rounds{};

    for (int round = 0; round < kRoundCount; ++round)
    {
        const auto start = std::chrono::steady_clock::now();

        for (const auto& [lClass, x] : stream)
        {
            qcx::integrals::BoysBatch(lClass, x, out);
            gSink += out[lClass];
        }

        const auto stop = std::chrono::steady_clock::now();
        rounds[static_cast<std::size_t>(round)] =
            std::chrono::duration<double, std::nano>(stop - start).count() /
            static_cast<double>(stream.size());
    }

    std::sort(rounds.begin(), rounds.end());

    return CallCost{rounds.front(), rounds[kRoundCount / 2]};
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

/// Builds the probe's closed-shell density: the unit diagonal's half.
/// \param n The basis function count.
/// \returns The density, or an Error.
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

/// Enumerates the screened-in canonical pair-pair cells of the lean walk and
/// the Boys calls each generates.
///
/// The walk is the lean builder's own: every row (bra pair) over its full ket
/// prefix, each cell kept when the pair Schwarz bounds clear
/// SchwarzThreshold(accuracy) x kNeighborListSlack, and one Boys call per
/// (bra primitive pair, ket primitive pair). The cell's class is
/// LBra + LKet, the order the VRR seeds the recurrence at.
/// \param pairList The flattened shells and canonical pairs.
/// \param qSchwarz The Schwarz bound of every canonical pair.
/// \param shells The primitive exponents and centers of every shell.
/// \param pairs The primitive pairs of every canonical pair.
/// \param flat The flat primitive-pair arrays the pair offsets index.
/// \param accuracy The screening preset.
/// \returns The census.
Census Enumerate(const qcx::integrals::ShellPairList& pairList,
                 const std::vector<double>& qSchwarz,
                 const std::vector<ShellPrims>& shells,
                 const std::vector<PairPrims>& pairs,
                 const std::vector<std::array<double, 4>>& flat,
                 qcx::integrals::AccuracyPreset accuracy) {
    Census census;
    census.quartetsByClass.assign(kMaxClass + 1, 0);
    census.callsByClass.assign(kMaxClass + 1, 0);
    census.binCounts.assign(static_cast<std::size_t>(kMaxClass + 1) * (kArgumentBins + 1), 0);
    census.regionCalls.assign(4, 0);

    const double cutoff =
        qcx::integrals::SchwarzThreshold(accuracy) * qcx::integrals::internal::kNeighborListSlack;
    const std::size_t nPairs = pairList.pairs.size();
    std::vector<double> sampled;

    for (std::size_t row = 0; row < nPairs; ++row)
    {
        const std::size_t rowBegin = pairs[row].offset;
        const std::size_t rowEnd = rowBegin + pairs[row].count;
        const int lRow = pairs[row].lBra;

        for (std::size_t ket = 0; ket <= row; ++ket)
        {
            if (qSchwarz[row] * qSchwarz[ket] < cutoff)
            {
                continue;
            }

            const std::size_t ketBegin = pairs[ket].offset;
            const std::size_t ketEnd = ketBegin + pairs[ket].count;
            const int lClass = lRow + pairs[ket].lBra;
            std::size_t calls = 0;

            for (std::size_t a = rowBegin; a < rowEnd; ++a)
            {
                for (std::size_t b = ketBegin; b < ketEnd; ++b)
                {
                    const double p = flat[a][0];
                    const double q = flat[b][0];
                    const double dx = flat[a][1] - flat[b][1];
                    const double dy = flat[a][2] - flat[b][2];
                    const double dz = flat[a][3] - flat[b][3];
                    const double x = (p * q / (p + q)) * (dx * dx + dy * dy + dz * dz);
                    const std::size_t bin = ArgumentBin(x);

                    ++census
                          .binCounts[static_cast<std::size_t>(lClass) * (kArgumentBins + 1) + bin];
                    ++census.regionCalls[static_cast<std::size_t>(RegionOf(x))];
                    ++calls;

                    // A thinned sample carries the quantiles: the full set is
                    // hundreds of millions of values, the distribution is what
                    // the cost table needs, and every argument still lands in
                    // a bin.
                    if ((sampled.size() < 4000000) && (((a + b) & 1023u) == 0u))
                    {
                        sampled.push_back(x);
                    }
                }
            }

            ++census.quartets;
            census.quartetsByClass[static_cast<std::size_t>(lClass)] += 1;
            census.callsByClass[static_cast<std::size_t>(lClass)] += calls;
            census.recurrenceSteps += RecurrenceSteps(lClass);
            census.calls += calls;
        }
    }

    if (!sampled.empty())
    {
        std::sort(sampled.begin(), sampled.end());

        for (const double fraction : {0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0})
        {
            const std::size_t index =
                static_cast<std::size_t>(fraction * static_cast<double>(sampled.size() - 1));
            census.argumentQuantiles.push_back(sampled[index]);
        }
    }

    return census;
}

void PrintUsage(const char* program) {
    std::printf("usage: %s [--basis <family>] [--carbons <n>] [--chunks <n>] [--no-build]\n",
                program);
}

} // namespace

int main(int argc, char** argv) {
    std::string family = "def2-svp";
    std::size_t carbons = 24;
    std::size_t chunks = 1;
    bool runBuild = true;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--basis" && i + 1 < argc)
        {
            family = argv[++i];
        } else if (arg == "--carbons" && i + 1 < argc)
        {
            carbons = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else if (arg == "--chunks" && i + 1 < argc)
        {
            chunks = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else if (arg == "--no-build")
        {
            runBuild = false;
        } else
        {
            PrintUsage(argv[0]);

            return 1;
        }
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

    // The primitive pairs of every canonical pair, flattened as
    // (p, Px, Py, Pz) rows so the quadruple walk is two contiguous runs.
    std::vector<PairPrims> pairs(pairList->pairs.size());
    std::vector<std::array<double, 4>> flat;

    for (std::size_t k = 0; k < pairList->pairs.size(); ++k)
    {
        const qcx::integrals::ShellPairIndex& pair = pairList->pairs[k];
        const ShellPrims& first = shells[pair.i];
        const ShellPrims& second = shells[pair.j];
        pairs[k].offset = flat.size();
        pairs[k].lBra =
            pairList->shells[pair.i].angularMomentum + pairList->shells[pair.j].angularMomentum;

        for (const double a : first.exponents)
        {
            for (const double b : second.exponents)
            {
                const double p = a + b;
                flat.push_back({p,
                                (a * first.center[0] + b * second.center[0]) / p,
                                (a * first.center[1] + b * second.center[1]) / p,
                                (a * first.center[2] + b * second.center[2]) / p});
            }
        }

        pairs[k].count = flat.size() - pairs[k].offset;
    }

    auto qSchwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);

    if (!qSchwarz.has_value())
    {
        std::printf("schwarz: %s\n", qSchwarz.error().message.c_str());

        return 1;
    }

    const auto censusStart = std::chrono::steady_clock::now();
    const Census census = Enumerate(*pairList, *qSchwarz, shells, pairs, flat, accuracy);
    const double censusSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - censusStart).count();
    std::printf("enumerated: %zu screened-in quartets, %zu Boys calls, %.1f s\n",
                census.quartets,
                census.calls,
                censusSeconds);

    // The engine's own reading of the same build: the counters the census is
    // validated against, and the kernel phase spans that carry the cost.
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
        options.maxParallelChunks = chunks;
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

        const auto millis = [](std::chrono::nanoseconds span) {
            return std::chrono::duration<double, std::milli>(span).count();
        };
        std::printf("engine: %zu fp64 quartets, %zu VRR calls, %zu primitive passes\n",
                    stats.fp64QuartetCount,
                    stats.kernelVrrQuadruples,
                    stats.kernelPrimPasses);
        std::printf("spans ms: total %.1f  eri(%.1f) - prep(%.1f) = kernel %.1f "
                    "[vrr %.1f  ket %.1f  bra %.1f]  contract %.1f\n",
                    millis(stats.totalWallTime),
                    millis(stats.eriWallTime),
                    millis(stats.eriPrepWallTime),
                    millis(stats.eriWallTime - stats.eriPrepWallTime),
                    millis(stats.kernelVrrWallTime),
                    millis(stats.kernelKetWallTime),
                    millis(stats.kernelBraWallTime),
                    millis(stats.contractWallTime));
    }

    double boysNanos = 0.0;
    double boysNanosLow = 0.0;
    double boysNanosHigh = 0.0;
    double boysNanosMedian = 0.0;
    double boysNanosEdgeLow = 0.0;
    double boysNanosEdgeHigh = 0.0;
    std::vector<double> costLow(kMaxClass + 1, 0.0);
    std::vector<double> costHigh(kMaxClass + 1, 0.0);

    // The bin grid's half step, in decades: the edge multiplier that brackets
    // one bin. The cost is read at the bin's center and at both edges, so the
    // interval carries the argument mirror's own error - the step changes of
    // the library's tier thresholds included - and nothing wider.
    const double edgeScale = std::pow(10.0,
                                      0.5 * (std::log10(kArgumentHi) - std::log10(kArgumentLo)) /
                                          static_cast<double>(kArgumentBins - 1));

    for (int l = 0; l <= kMaxClass; ++l)
    {
        const std::size_t base = static_cast<std::size_t>(l) * (kArgumentBins + 1);
        double classLow = std::numeric_limits<double>::max();
        double classHigh = 0.0;

        for (std::size_t bin = 0; bin <= kArgumentBins; ++bin)
        {
            const std::size_t count = census.binCounts[base + bin];

            if (count == 0)
            {
                continue;
            }

            const double center = ArgumentOf(bin);
            const CallCost cost = TimeBoysCall(l, center);
            const CallCost low = TimeBoysCall(l, center / edgeScale);
            const CallCost high = TimeBoysCall(l, center * edgeScale);
            const double edgeLow = std::min({cost.min, low.min, high.min});
            const double edgeHigh = std::max({cost.min, low.min, high.min});
            boysNanos += cost.min * static_cast<double>(count);
            boysNanosEdgeLow += edgeLow * static_cast<double>(count);
            boysNanosEdgeHigh += edgeHigh * static_cast<double>(count);
            boysNanosMedian += cost.median * static_cast<double>(count);
            classLow = std::min(classLow, cost.min);
            classHigh = std::max(classHigh, cost.min);
        }

        // The loose outer bound: every call of a class charged that class's
        // cheapest or its dearest argument. Reported as a diagnostic only -
        // it spans region C against region A, which no real mix does.
        if (census.callsByClass[static_cast<std::size_t>(l)] != 0)
        {
            costLow[static_cast<std::size_t>(l)] = classLow;
            costHigh[static_cast<std::size_t>(l)] = classHigh;
            boysNanosLow += classLow * static_cast<double>(census.callsByClass[l]);
            boysNanosHigh += classHigh * static_cast<double>(census.callsByClass[l]);
        }
    }

    // The headline per-call cost: the same distribution replayed as the real
    // mix, so the region branch is as unpredictable as the kernel's is.
    const CallCost mixed = TimeMixedStream(census);
    const double mixedMs = mixed.min * static_cast<double>(census.calls) / 1e6;
    const double mixedMsLoaded = mixed.median * static_cast<double>(census.calls) / 1e6;

    std::printf("\nnmax distribution (L = LBra + LKet; quartets / Boys calls):\n");
    std::printf("   L   quartets    %-11s Boys calls    %-11s\n", "share", "share");

    for (int l = 0; l <= kMaxClass; ++l)
    {
        const std::size_t quartets = census.quartetsByClass[static_cast<std::size_t>(l)];

        if (quartets == 0 && census.callsByClass[static_cast<std::size_t>(l)] == 0)
        {
            continue;
        }

        std::printf("  %2d  %10zu  %9.4f%%  %11zu  %9.4f%%\n",
                    l,
                    quartets,
                    100.0 * static_cast<double>(quartets) / static_cast<double>(census.quartets),
                    census.callsByClass[static_cast<std::size_t>(l)],
                    100.0 * static_cast<double>(census.callsByClass[static_cast<std::size_t>(l)]) /
                        static_cast<double>(census.calls));
    }

    double classSumQuartets = 0.0;
    double classSumCalls = 0.0;

    for (int l = 0; l <= kMaxClass; ++l)
    {
        classSumQuartets +=
            static_cast<double>(l * census.quartetsByClass[static_cast<std::size_t>(l)]);
        classSumCalls += static_cast<double>(l * census.callsByClass[static_cast<std::size_t>(l)]);
    }

    std::printf("mean class L: %.3f over the quartets, %.3f over the Boys calls\n",
                classSumQuartets / static_cast<double>(census.quartets),
                classSumCalls / static_cast<double>(census.calls));
    std::printf("slice-triangle steps: %zu over the cells, %.1f per Boys call\n",
                census.recurrenceSteps,
                static_cast<double>(census.recurrenceSteps) / static_cast<double>(census.calls));

    if (!census.argumentQuantiles.empty())
    {
        std::printf("argument x: min %.3e  1%% %.3e  25%% %.3e  50%% %.3e  75%% %.3e  "
                    "99%% %.3e  max %.3e\n",
                    census.argumentQuantiles[0],
                    census.argumentQuantiles[1],
                    census.argumentQuantiles[2],
                    census.argumentQuantiles[3],
                    census.argumentQuantiles[4],
                    census.argumentQuantiles[5],
                    census.argumentQuantiles[6]);
    }

    const double callCount = static_cast<double>(census.calls);
    const auto regionShare = [&](Region region) {
        return callCount > 0.0
                   ? 100.0 *
                         static_cast<double>(census.regionCalls[static_cast<std::size_t>(region)]) /
                         callCount
                   : 0.0;
    };

    std::printf("argument regions: zero %.4f%%  A(0,11.9) %.4f%%  B[11.9,28.99) %.4f%%  "
                "C(28.99,inf) %.4f%%\n",
                regionShare(Region::kZero),
                regionShare(Region::kA),
                regionShare(Region::kB),
                regionShare(Region::kC));

    const double boysMs = boysNanos / 1e6;
    std::printf("\nBoys cost, this machine, under whatever load it carried:\n");
    std::printf("  per call ns, cheapest argument of each class: ");

    for (int l = 0; l <= kMaxClass; ++l)
    {
        if (census.callsByClass[static_cast<std::size_t>(l)] != 0)
        {
            std::printf("L%d %.1f  ", l, costLow[static_cast<std::size_t>(l)]);
        }
    }

    std::printf("\n");
    std::printf("  total Boys time %.1f ms  [load %.1f]  [argument mirror %.1f, %.1f]  "
                "[class worst case %.1f, %.1f]\n",
                boysMs,
                boysNanosMedian / 1e6,
                boysNanosEdgeLow / 1e6,
                boysNanosEdgeHigh / 1e6,
                boysNanosLow / 1e6,
                boysNanosHigh / 1e6);
    std::printf("  the same call set replayed as the real mix (the kernel's own branch pattern): "
                "%.1f ms, %.2f ns per call  [load %.1f]\n",
                mixedMs,
                mixed.min,
                mixedMsLoaded);

    if (runBuild && stats.totalWallTime.count() > 0)
    {
        const double vrrMs =
            std::chrono::duration<double, std::milli>(stats.kernelVrrWallTime).count();
        const double kernelMs =
            vrrMs + std::chrono::duration<double, std::milli>(stats.kernelKetWallTime).count() +
            std::chrono::duration<double, std::milli>(stats.kernelBraWallTime).count();
        const double eriMs = std::chrono::duration<double, std::milli>(stats.eriWallTime).count();
        const double totalMs =
            std::chrono::duration<double, std::milli>(stats.totalWallTime).count();
        const auto share = [&](double nanos, double span) {
            return span > 0.0 ? 100.0 * (nanos / 1e6) / span : 0.0;
        };
        const auto shareMs = [&](double ms, double span) {
            return span > 0.0 ? 100.0 * ms / span : 0.0;
        };

        std::printf("\nshare of the engine's own spans (both halves timed on this machine):\n");
        std::printf(
            "  kernel phase split: vrr %.1f%%  ket %.1f%%  bra %.1f%%\n",
            100.0 * vrrMs / kernelMs,
            100.0 * std::chrono::duration<double, std::milli>(stats.kernelKetWallTime).count() /
                kernelMs,
            100.0 * std::chrono::duration<double, std::milli>(stats.kernelBraWallTime).count() /
                kernelMs);
        std::printf("  Boys of the VRR span     %.2f%%  (mixed-stream replay; the per-bin bracket "
                    "is [%.2f, %.2f], the class worst case [%.2f, %.2f], load to %.2f)\n",
                    shareMs(mixedMs, vrrMs),
                    share(boysNanosEdgeLow, vrrMs),
                    share(boysNanosEdgeHigh, vrrMs),
                    share(boysNanosLow, vrrMs),
                    share(boysNanosHigh, vrrMs),
                    shareMs(mixedMsLoaded, vrrMs));
        std::printf("  Boys of the kernel span  %.2f%%  [%.2f, %.2f]\n",
                    share(boysNanos, kernelMs),
                    share(boysNanosEdgeLow, kernelMs),
                    share(boysNanosEdgeHigh, kernelMs));
        std::printf("  Boys of the ERI span     %.2f%%  (mixed-stream replay; the per-bin bracket "
                    "is [%.2f, %.2f])\n",
                    shareMs(mixedMs, eriMs),
                    share(boysNanosEdgeLow, eriMs),
                    share(boysNanosEdgeHigh, eriMs));
        std::printf("  Boys of the whole call   %.2f%%\n", shareMs(mixedMs, totalMs));
        std::printf("  the ERI span against the VRR span: the residue (the slice recurrence, the "
                    "block zeroing, the seeding and the two transforms) is %.1f%% of the ERI "
                    "span\n",
                    eriMs > 0.0 ? 100.0 * (eriMs - vrrMs) / eriMs : 0.0);
    }

    std::printf("\nsink %.6e\n", gSink);

    return 0;
}
