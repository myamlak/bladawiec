// The Boys argument probe: the (order, argument) joint distribution of one
// real Fock build, and the run structure the engine presents those calls in.
//
// The call-count harness is benchmarks/boys_share_probe.cpp; this probe reuses
// its enumeration walk, its validated quartet census and its per-(class, bin)
// cost table, and adds the three things that probe does not carry: the joint
// (order, region) and (order, table band) tabulation, an argument histogram
// with stated edges, and the run lengths - how many consecutive calls share a
// region and a table band when the engine hands them over.
//
// The walk is the engine's own order, not just its own set: class batches of
// (lBra, lKet), tasks inside a class sorted by (ketPair, braRowPairs, braPair)
// exactly as SortScreenedTasks' documented five-key comparator does, ket
// groups split where (ketPair, rowPairs, nPrimPairs) changes, and the
// innermost loop over the bra pair's primitive pairs - so the contiguous
// burst lengths measured here are the lengths the kernel's call site really
// presents. The region and band boundaries are read from the library's own
// generated tables, never restated.
//
// Usage: qcx-bench-boys-argument [--basis <family>] [--carbons <n>]
//                                [--no-build] [--no-cost]
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
#include <limits>
#include <string>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

/// The highest pair class the engine can dispatch to: two shells of
/// kMaxShellL = 6.
constexpr int kMaxPairClass = 12;

/// The highest quartet class.
constexpr int kMaxClass = 24;

/// The region-A table band edges, read from the library's own piece table:
/// every order's fits sit on the same two intervals, so order 0's pieces are
/// the whole of region A's banding.
constexpr double kBandA0 = boys::detail::kPieces[boys::detail::kPieceStart[0]].b;
constexpr double kBandX0 = boys::detail::kX0;
constexpr double kBandX1 = boys::detail::kX1;

/// The argument-region code of \p x, in the library's own boundaries.
/// \param x The Boys argument, >= 0.
/// \returns 0 for x == 0, else 1 (region A), 2 (region B) or 3 (region C).
inline int RegionOf(double x) noexcept {
    if (x <= 0.0)
    {
        return 0;
    }

    if (x < kBandX0)
    {
        return 1;
    }

    if (x < kBandX1)
    {
        return 2;
    }

    return 3;
}

/// The table-band code of \p x: the finest interval the library's own
/// dispatch separates.
/// \param x The Boys argument, >= 0.
/// \returns 0 zero, 1 A-low, 2 A-high, 3 B, 4 C.
inline int BandOf(double x) noexcept {
    if (x <= 0.0)
    {
        return 0;
    }

    if (x < kBandA0)
    {
        return 1;
    }

    if (x < kBandX0)
    {
        return 2;
    }

    if (x < kBandX1)
    {
        return 3;
    }

    return 4;
}

/// The code paths BoysAllOrders actually takes for one (nmax, x). Region A is
/// not one path: the batch serves the orders k with x >= kTierThresholds[k]
/// from the extended seed and the upward recursion, and the rest from the
/// per-order region-A fits, so a region-A call is pure-table, pure-recursion
/// or a mixture of both depending on where x sits against the order's own
/// threshold.
enum class Path : int {
    kZero = 0, ///< x == 0: the closed-form fill.
    kATable = 1, ///< x below every tier threshold: per-order fits only.
    kARecursion = 2, ///< x at or above the top order's threshold: extended seed only.
    kAMixed = 3, ///< the prefix recurses, the tail is fitted.
    kB = 4, ///< region B: one F0 fit plus upward recursion.
    kC = 5, ///< region C: the asymptotic form plus upward recursion.
};

/// The path of one call with \p order orders.
/// \param x The Boys argument, >= 0.
/// \param order The batch's highest order.
/// \returns The path code.
inline Path PathOf(double x, int order) noexcept {
    if (x <= 0.0)
    {
        return Path::kZero;
    }

    if (x >= kBandX0)
    {
        return x < kBandX1 ? Path::kB : Path::kC;
    }

    const double top = boys::detail::kTierThresholds[static_cast<std::size_t>(order)];

    if (x >= top)
    {
        return Path::kARecursion;
    }

    if (x < boys::detail::kTierThresholds[0])
    {
        return Path::kATable;
    }

    return Path::kAMixed;
}

/// The number of path codes.
constexpr std::size_t kPathCount = 6;

/// The path names, in code order.
constexpr const char* kPathNames[kPathCount] = {"zero", "A-table", "A-recur", "A-mixed", "B", "C"};

/// The argument histogram's edges: half-decade bins from 1e-4 to 1e6, with
/// bin 0 reserved for x == 0 exactly.
constexpr int kHistBins = 21;
constexpr double kHistT0 = -4.0;
constexpr double kHistT1 = 6.0;

/// The cost table's bins: the harness's own log grid.
constexpr std::size_t kCostBins = 240;
constexpr double kCostLo = 1e-9;
constexpr double kCostHi = 1e6;

/// The run-length histogram's cap: longer runs are counted as one overflow
/// bucket.
constexpr std::size_t kMaxRun = 64;

/// The run lengths of one contiguous unit, calls-weighted.
struct RunStats {
    std::size_t calls = 0;
    std::size_t runs = 0;
    std::array<std::size_t, kMaxRun + 1> hist{}; ///< hist[len], hist[0] = overflow.
    std::size_t maxRun = 0;

    /// Records one run.
    /// \param length The run's length, >= 1.
    void AddRun(std::size_t length) {
        ++runs;
        maxRun = std::max(maxRun, length);

        if (length > kMaxRun)
        {
            ++hist[0];
        } else
        {
            ++hist[length];
        }
    }

    /// The calls-weighted mean run length.
    /// \returns Calls per run, or 0 when no run was recorded.
    double Mean() const {
        return runs == 0 ? 0.0 : static_cast<double>(calls) / static_cast<double>(runs);
    }
};

/// A streaming run accumulator: it closes a run whenever the code changes,
/// and is drained by \ref Close.
struct RunTracker {
    RunStats stats;
    int last = -1;
    std::size_t length = 0;

    /// Pushes one element's code.
    /// \param code The region or band code of this element.
    void Push(int code) {
        ++stats.calls;

        if (code == last)
        {
            ++length;

            return;
        }

        if (length != 0)
        {
            stats.AddRun(length);
        }

        last = code;
        length = 1;
    }

    /// Closes the open run.
    void Close() {
        if (length != 0)
        {
            stats.AddRun(length);
        }

        last = -1;
        length = 0;
    }
};

/// Merges \p from into \p into and resets \p from.
/// \param into The destination, calls and runs accumulated.
/// \param from The source, cleared on return.
void Merge(RunStats& into, RunStats& from) {
    into.calls += from.calls;
    into.runs += from.runs;
    into.maxRun = std::max(into.maxRun, from.maxRun);

    for (std::size_t i = 0; i <= kMaxRun; ++i)
    {
        into.hist[i] += from.hist[i];
        from.hist[i] = 0;
    }

    from.calls = 0;
    from.runs = 0;
    from.maxRun = 0;
}

/// One shell's primitive exponents and its center.
struct ShellPrims {
    std::vector<double> exponents;
    std::array<double, 3> center{0.0, 0.0, 0.0};
};

/// One canonical shell pair's role data.
struct PairRole {
    int l = 0; ///< la + lb.
    std::size_t rowPairs = 0; ///< contractionCount(a) * contractionCount(b).
    std::vector<std::array<double, 4>> prim; ///< (p, Px, Py, Pz) rows.
};

/// Times one BoysAllOrders call at (nmax, x).
constexpr double kRoundFloorNanos = 40000.0;
constexpr std::size_t kRoundRepeatCap = 200000;
constexpr int kRoundCount = 5;

double gSink = 0.0;

/// The argument bin of \p x on the cost grid (0 is x == 0).
/// \param x The Boys argument, >= 0.
/// \returns The bin index.
std::size_t CostBin(double x) {
    if (x <= 0.0)
    {
        return 0;
    }

    const double t =
        (std::log10(x) - std::log10(kCostLo)) / (std::log10(kCostHi) - std::log10(kCostLo));
    const double clamped = std::clamp(t, 0.0, 1.0);

    return 1 + static_cast<std::size_t>(clamped * static_cast<double>(kCostBins - 1) + 0.5);
}

/// The representative argument of a cost bin.
/// \param bin The bin index.
/// \returns The bin's geometric center, or 0 for bin 0.
double CostArgumentOf(std::size_t bin) {
    if (bin == 0)
    {
        return 0.0;
    }

    const double span = std::log10(kCostHi) - std::log10(kCostLo);
    const double t = static_cast<double>(bin - 1) / static_cast<double>(kCostBins - 1);

    return std::pow(10.0, std::log10(kCostLo) + t * span);
}

/// Times one BoysAllOrders call at (nmax, x) over \p kRoundCount rounds.
/// \param nmax The batch's highest order.
/// \param x The argument.
/// \returns Nanoseconds per call at the minimum and the median round.
std::pair<double, double> TimeBoysCall(int nmax, double x) {
    double out[1 + qcx::integrals::kMaxBoysOrder];
    std::size_t reps = 256;

    const auto calStart = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < reps; ++i)
    {
        qcx::integrals::BoysAllOrders(nmax, x, out);
    }

    const auto calStop = std::chrono::steady_clock::now();
    gSink += out[nmax];

    const double calNanos = std::chrono::duration<double, std::nano>(calStop - calStart).count() /
                            static_cast<double>(reps);

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
            qcx::integrals::BoysAllOrders(nmax, x, out);
        }

        const auto stop = std::chrono::steady_clock::now();
        gSink += out[nmax];
        rounds[static_cast<std::size_t>(round)] =
            std::chrono::duration<double, std::nano>(stop - start).count() /
            static_cast<double>(reps);
    }

    std::sort(rounds.begin(), rounds.end());

    return {rounds.front(), rounds[kRoundCount / 2]};
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
    std::printf("usage: %s [--basis <family>] [--carbons <n>] [--no-build] [--no-cost]\n", program);
}

/// The whole measurement, accumulated over the engine-ordered walk.
struct Census {
    std::size_t quartets = 0;
    std::size_t calls = 0;
    std::vector<std::size_t> quartetsByClass;
    std::vector<std::size_t> callsByClass;

    std::array<std::size_t, 4> regionCalls{}; ///< Per region.
    std::array<std::size_t, 5> bandCalls{}; ///< Per table band.
    std::array<std::size_t, kPathCount> pathCalls{}; ///< Per dispatch path.
    std::vector<std::size_t> jointClassRegion; ///< [class][region].
    std::vector<std::size_t> jointClassBand; ///< [class][band].
    std::vector<std::size_t> jointClassPath; ///< [class][path].
    std::vector<std::size_t> histCounts; ///< The argument histogram.
    std::vector<std::size_t> costCounts; ///< [class][cost bin].

    std::array<std::size_t, 4> regionRuns{}; ///< Per region, runs over the whole concatenation.
    std::array<std::size_t, 5> bandRuns{};

    std::size_t burstCount = 0; ///< (task, ket primitive pair) units: the engine's innermost runs.
    std::array<std::size_t, kMaxRun + 1>
        burstLengths{}; ///< burstLengths[nPrimBra], burstLengths[0] overflow.
    std::size_t groupCount = 0; ///< Ket groups at one ket primitive pair.
    std::size_t groupMaxTasks = 0; ///< The widest ket group.
    std::size_t uniformBursts = 0; ///< Bursts whose calls all share one region.
    std::size_t uniformTasks = 0; ///< Quartet blocks whose calls all share one region.
    std::size_t uniformGroups = 0; ///< Group-at-a-ket-primitive runs that share one region.

    double maxArgument = 0.0;
    double minNonZeroArgument = std::numeric_limits<double>::max();
    std::size_t zeroCalls = 0;
    std::size_t above100 = 0;
    std::size_t above1000 = 0;

    // The run structures. kGlobal = the whole walk in the engine's own order,
    // kSub = one task at one ket primitive pair (the engine's innermost
    // contiguous burst), kTask = one task, kGroup = one ket group at one ket
    // primitive pair.
    RunStats globalRun;
    RunStats subBurst;
    RunStats taskBlock;
    RunStats groupRun;
    RunStats globalRunBand;
    RunStats subBurstBand;
    RunStats taskBlockBand;
    RunStats groupRunBand;
    RunStats globalRunPath;
    RunStats subBurstPath;
    RunStats taskBlockPath;
    RunStats groupRunPath;

    // The group runs' random-permutation baseline: the number of runs the
    // same multiset would show if the engine's order carried no information.
    double groupExpectedRuns = 0.0;
    std::size_t groupBaselineCalls = 0;
};

/// The baseline runs of a unit with \p counts per code under a uniformly
/// random permutation: 1 + (N-1)(1 - sum_r n_r (n_r - 1) / (N (N - 1))).
/// \param counts The per-code call counts.
/// \param n The unit's call count.
/// \returns The expected run count.
double ExpectedRuns(const std::array<std::size_t, 5>& counts, std::size_t n) {
    if (n <= 1)
    {
        return static_cast<double>(n);
    }

    double same = 0.0;

    for (const std::size_t c : counts)
    {
        same += static_cast<double>(c) * static_cast<double>(c - 1);
    }

    const double total = static_cast<double>(n) * static_cast<double>(n - 1);

    return 1.0 + static_cast<double>(n - 1) * (1.0 - same / total);
}

} // namespace

int main(int argc, char** argv) {
    std::string family = "def2-svp";
    std::size_t carbons = 24;
    bool runBuild = true;
    bool runCost = true;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--basis" && i + 1 < argc)
        {
            family = argv[++i];
        } else if (arg == "--carbons" && i + 1 < argc)
        {
            carbons = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else if (arg == "--no-build")
        {
            runBuild = false;
        } else if (arg == "--no-cost")
        {
            runCost = false;
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
    std::printf("region boundaries from the library's own table: kX0 %.17g  kX1 %.17g  "
                "region-A band edge %.17g\n",
                boys::detail::kX0,
                boys::detail::kX1,
                kBandA0);

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
    std::vector<int> pairL(nPairs, 0);
    std::vector<std::vector<std::size_t>> byL(static_cast<std::size_t>(kMaxPairClass) + 1);

    for (std::size_t k = 0; k < nPairs; ++k)
    {
        const qcx::integrals::ShellPairIndex& pair = pairList->pairs[k];
        const ShellPrims& first = shells[pair.i];
        const ShellPrims& second = shells[pair.j];
        const int la = pairList->shells[pair.i].angularMomentum;
        const int lb = pairList->shells[pair.j].angularMomentum;
        pairL[k] = la + lb;
        roles[k].l = la + lb;
        roles[k].rowPairs =
            pairList->shells[pair.i].contractionCount * pairList->shells[pair.j].contractionCount;

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

        if (pairL[k] <= kMaxPairClass)
        {
            byL[static_cast<std::size_t>(pairL[k])].push_back(k);
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
        options.maxParallelChunks = 1;
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

    // ---- The engine-ordered walk -------------------------------------------
    Census census;
    census.quartetsByClass.assign(kMaxClass + 1, 0);
    census.callsByClass.assign(kMaxClass + 1, 0);
    census.jointClassRegion.assign(static_cast<std::size_t>(kMaxClass + 1) * 4, 0);
    census.jointClassBand.assign(static_cast<std::size_t>(kMaxClass + 1) * 5, 0);
    census.jointClassPath.assign(static_cast<std::size_t>(kMaxClass + 1) * kPathCount, 0);
    census.histCounts.assign(kHistBins + 1, 0);
    census.costCounts.assign(static_cast<std::size_t>(kMaxClass + 1) * (kCostBins + 1), 0);

    const auto walkStart = std::chrono::steady_clock::now();
    std::vector<std::size_t> candidates;
    std::vector<int> groupRegion; // one group's calls, engine order (g-major, t, b-minor)
    std::vector<int> groupBand;
    std::vector<int> groupPath;
    std::vector<RunTracker> perGroupRun;
    std::vector<RunTracker> perGroupBand;
    std::vector<RunTracker> perGroupPath;
    std::vector<std::array<std::size_t, 5>> perGroupCounts;

    // The whole walk, one call at a time, in the engine's own order.
    RunTracker globalRunTracker;
    RunTracker globalBandTracker;
    RunTracker globalPathTracker;

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
                std::size_t at = 0;

                while (at < candidates.size())
                {
                    // One ket group: equal ketPair, equal rowPairs, equal
                    // primitive-pair count - the engine's group boundary.
                    std::size_t end = at + 1;

                    while (end < candidates.size() &&
                           roles[candidates[end]].rowPairs == roles[candidates[at]].rowPairs &&
                           roles[candidates[end]].prim.size() == roles[candidates[at]].prim.size())
                    {
                        ++end;
                    }

                    const std::size_t nGroup = end - at;
                    const std::size_t nPrimBra = roles[candidates[at]].prim.size();
                    const std::size_t nPrimKet = ketPrim.size();
                    census.burstCount += nGroup * nPrimKet;
                    census.groupCount += nPrimKet;
                    census.groupMaxTasks = std::max(census.groupMaxTasks, nGroup);

                    // One entry per (task, ket primitive pair) unit, so the
                    // histogram and burstCount share a denominator.
                    if (nPrimBra > kMaxRun)
                    {
                        census.burstLengths[0] += nGroup * nPrimKet;
                    } else
                    {
                        census.burstLengths[nPrimBra] += nGroup * nPrimKet;
                    }

                    perGroupRun.assign(nPrimKet, RunTracker{});
                    perGroupBand.assign(nPrimKet, RunTracker{});
                    perGroupPath.assign(nPrimKet, RunTracker{});
                    perGroupCounts.assign(nPrimKet, {});

                    for (std::size_t g = 0; g < nPrimKet; ++g)
                    {
                        for (std::size_t t = at; t < end; ++t)
                        {
                            const auto& braPrim = roles[candidates[t]].prim;

                            for (std::size_t b = 0; b < nPrimBra; ++b)
                            {
                                const double p = braPrim[b][0];
                                const double q = ketPrim[g][0];
                                const double dx = braPrim[b][1] - ketPrim[g][1];
                                const double dy = braPrim[b][2] - ketPrim[g][2];
                                const double dz = braPrim[b][3] - ketPrim[g][3];
                                const double x = (p * q / (p + q)) * (dx * dx + dy * dy + dz * dz);

                                ++perGroupCounts[g][static_cast<std::size_t>(BandOf(x))];
                            }
                        }
                    }

                    std::size_t groupCalls = 0;

                    for (std::size_t g = 0; g < nPrimKet; ++g)
                    {
                        census.groupExpectedRuns +=
                            ExpectedRuns(perGroupCounts[g], nGroup * nPrimBra);
                        groupCalls += nGroup * nPrimBra;
                    }

                    census.groupBaselineCalls += groupCalls;

                    // The group's calls, in the engine's own order: the ket
                    // primitive pair is the outer loop, so one g's whole
                    // (task, bra primitive) rectangle is contiguous and the
                    // concatenation of those rectangles is the real stream.
                    groupRegion.resize(nGroup * nPrimKet * nPrimBra);
                    groupBand.resize(nGroup * nPrimKet * nPrimBra);
                    groupPath.resize(nGroup * nPrimKet * nPrimBra);

                    for (std::size_t g = 0; g < nPrimKet; ++g)
                    {
                        RunTracker& gRun = perGroupRun[g];
                        RunTracker& gBand = perGroupBand[g];
                        RunTracker& gPath = perGroupPath[g];

                        for (std::size_t tl = 0; tl < nGroup; ++tl)
                        {
                            const auto& braPrim = roles[candidates[at + tl]].prim;

                            for (std::size_t b = 0; b < nPrimBra; ++b)
                            {
                                const double p = braPrim[b][0];
                                const double q = ketPrim[g][0];
                                const double dx = braPrim[b][1] - ketPrim[g][1];
                                const double dy = braPrim[b][2] - ketPrim[g][2];
                                const double dz = braPrim[b][3] - ketPrim[g][3];
                                const double x = (p * q / (p + q)) * (dx * dx + dy * dy + dz * dz);

                                const int region = RegionOf(x);
                                const int band = BandOf(x);
                                const int path = static_cast<int>(PathOf(x, lClass));
                                const std::size_t slot = (tl * nPrimKet + g) * nPrimBra + b;
                                groupRegion[slot] = region;
                                groupBand[slot] = band;
                                groupPath[slot] = path;
                                gRun.Push(region);
                                gBand.Push(band);
                                gPath.Push(path);
                                globalRunTracker.Push(region);
                                globalBandTracker.Push(band);
                                globalPathTracker.Push(path);

                                ++census.pathCalls[static_cast<std::size_t>(path)];
                                ++census.jointClassPath[static_cast<std::size_t>(lClass) *
                                                            kPathCount +
                                                        static_cast<std::size_t>(path)];
                                ++census.regionCalls[static_cast<std::size_t>(region)];
                                ++census.jointClassRegion[static_cast<std::size_t>(lClass) * 4 +
                                                          static_cast<std::size_t>(region)];
                                ++census.bandCalls[static_cast<std::size_t>(band)];
                                ++census.jointClassBand[static_cast<std::size_t>(lClass) * 5 +
                                                        static_cast<std::size_t>(band)];
                                ++census.costCounts[static_cast<std::size_t>(lClass) *
                                                        (kCostBins + 1) +
                                                    CostBin(x)];
                                census.maxArgument = std::max(census.maxArgument, x);

                                if (x > 0.0)
                                {
                                    census.minNonZeroArgument =
                                        std::min(census.minNonZeroArgument, x);
                                } else
                                {
                                    ++census.zeroCalls;
                                }

                                if (x > 100.0)
                                {
                                    ++census.above100;
                                }

                                if (x > 1000.0)
                                {
                                    ++census.above1000;
                                }

                                const std::size_t hist =
                                    x <= 0.0 ? 0
                                             : 1 + static_cast<std::size_t>(
                                                       std::clamp((std::log10(x) - kHistT0) /
                                                                      (kHistT1 - kHistT0),
                                                                  0.0,
                                                                  1.0) *
                                                           static_cast<double>(kHistBins - 1) +
                                                       0.5);
                                ++census.histCounts[hist];
                            }
                        }
                    }

                    for (std::size_t tl = 0; tl < nGroup; ++tl)
                    {
                        const std::size_t base = tl * nPrimKet * nPrimBra;

                        // One task: the engine's innermost burst is the run of
                        // nPrimBra calls inside one g, the task block is the
                        // whole (g, b) rectangle.
                        for (std::size_t g = 0; g < nPrimKet; ++g)
                        {
                            RunTracker sub;
                            RunTracker subBand;
                            RunTracker subPath;

                            for (std::size_t b = 0; b < nPrimBra; ++b)
                            {
                                const std::size_t slot = base + g * nPrimBra + b;
                                sub.Push(groupRegion[slot]);
                                subBand.Push(groupBand[slot]);
                                subPath.Push(groupPath[slot]);
                            }

                            sub.Close();
                            subBand.Close();
                            subPath.Close();

                            if (sub.stats.runs == 1)
                            {
                                ++census.uniformBursts;
                            }

                            Merge(census.subBurst, sub.stats);
                            Merge(census.subBurstBand, subBand.stats);
                            Merge(census.subBurstPath, subPath.stats);
                        }

                        {
                            RunTracker taskTracker;
                            RunTracker taskBandTracker;
                            RunTracker taskPathTracker;
                            const std::size_t taskCalls = nPrimKet * nPrimBra;

                            for (std::size_t slot = base; slot < base + taskCalls; ++slot)
                            {
                                taskTracker.Push(groupRegion[slot]);
                                taskBandTracker.Push(groupBand[slot]);
                                taskPathTracker.Push(groupPath[slot]);
                            }

                            taskTracker.Close();
                            taskBandTracker.Close();
                            taskPathTracker.Close();

                            if (taskTracker.stats.runs == 1)
                            {
                                ++census.uniformTasks;
                            }

                            Merge(census.taskBlock, taskTracker.stats);
                            Merge(census.taskBlockBand, taskBandTracker.stats);
                            Merge(census.taskBlockPath, taskPathTracker.stats);
                            census.quartets += 1;
                            census.calls += taskCalls;
                            census.callsByClass[static_cast<std::size_t>(lClass)] += taskCalls;
                            census.quartetsByClass[static_cast<std::size_t>(lClass)] += 1;
                        }
                    }

                    for (std::size_t g = 0; g < nPrimKet; ++g)
                    {
                        perGroupRun[g].Close();
                        perGroupBand[g].Close();
                        perGroupPath[g].Close();

                        if (perGroupRun[g].stats.runs == 1)
                        {
                            ++census.uniformGroups;
                        }

                        Merge(census.groupRun, perGroupRun[g].stats);
                        Merge(census.groupRunBand, perGroupBand[g].stats);
                        Merge(census.groupRunPath, perGroupPath[g].stats);
                    }

                    at = end;
                }
            }
        }
    }

    globalRunTracker.Close();
    globalBandTracker.Close();
    globalPathTracker.Close();
    census.globalRun = globalRunTracker.stats;
    census.globalRunBand = globalBandTracker.stats;
    census.globalRunPath = globalPathTracker.stats;

    const double walkSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - walkStart).count();
    std::printf("walked: %zu screened-in quartets, %zu Boys calls, %.1f s\n",
                census.quartets,
                census.calls,
                walkSeconds);

    if (runBuild)
    {
        std::printf("engine cross-check: quartets %s, calls %s\n",
                    census.quartets == stats.fp64QuartetCount ? "MATCH" : "MISMATCH",
                    census.calls == stats.kernelVrrQuadruples ? "MATCH" : "MISMATCH");
    }

    // ---- Report ------------------------------------------------------------
    std::printf("\nnmax distribution (L = LBra + LKet; quartets / Boys calls):\n");
    std::printf("   L   quartets    %-11s Boys calls    %-11s\n", "share", "share");

    for (int l = 0; l <= kMaxClass; ++l)
    {
        if (census.quartetsByClass[static_cast<std::size_t>(l)] == 0 &&
            census.callsByClass[static_cast<std::size_t>(l)] == 0)
        {
            continue;
        }

        std::printf("  %2d  %10zu  %9.4f%%  %11zu  %9.4f%%\n",
                    l,
                    census.quartetsByClass[static_cast<std::size_t>(l)],
                    100.0 *
                        static_cast<double>(census.quartetsByClass[static_cast<std::size_t>(l)]) /
                        static_cast<double>(census.quartets),
                    census.callsByClass[static_cast<std::size_t>(l)],
                    100.0 * static_cast<double>(census.callsByClass[static_cast<std::size_t>(l)]) /
                        static_cast<double>(census.calls));
    }

    double classSumCalls = 0.0;

    for (int l = 0; l <= kMaxClass; ++l)
    {
        classSumCalls += static_cast<double>(l * census.callsByClass[static_cast<std::size_t>(l)]);
    }

    double atOrBelow3 = 0.0;

    for (int l = 0; l <= 3; ++l)
    {
        atOrBelow3 += static_cast<double>(census.callsByClass[static_cast<std::size_t>(l)]);
    }

    std::printf("mean class L over the Boys calls: %.3f   calls at L <= 3: %.4f%%\n",
                classSumCalls / static_cast<double>(census.calls),
                100.0 * atOrBelow3 / static_cast<double>(census.calls));

    const double callCount = static_cast<double>(census.calls);
    const auto share = [&](std::size_t n) { return 100.0 * static_cast<double>(n) / callCount; };

    std::printf("\nargument regions (the library's own boundaries):\n");
    std::printf("  x == 0            %14zu  %8.4f%%\n",
                census.regionCalls[0],
                share(census.regionCalls[0]));
    std::printf("  A  (0, %.6g)      %14zu  %8.4f%%\n",
                kBandX0,
                census.regionCalls[1],
                share(census.regionCalls[1]));
    std::printf("  B  [%.6g, %.6g)  %14zu  %8.4f%%\n",
                kBandX0,
                kBandX1,
                census.regionCalls[2],
                share(census.regionCalls[2]));
    std::printf("  C  [%.6g, inf)   %14zu  %8.4f%%\n",
                kBandX1,
                census.regionCalls[3],
                share(census.regionCalls[3]));
    std::printf("  table bands: zero %.4f%%  A-low(0,%.6g) %.4f%%  A-high[%.6g,%.6g) %.4f%%  "
                "B[%.6g,%.6g) %.4f%%  C[%.6g,inf) %.4f%%\n",
                share(census.bandCalls[0]),
                kBandA0,
                share(census.bandCalls[1]),
                kBandA0,
                kBandX0,
                share(census.bandCalls[2]),
                kBandX0,
                kBandX1,
                share(census.bandCalls[3]),
                kBandX1,
                share(census.bandCalls[4]));
    std::printf("\nthe library's actual per-call dispatch (region A is three paths, not one):\n");

    for (std::size_t p = 0; p < kPathCount; ++p)
    {
        std::printf("  %-8s %14zu  %8.4f%%\n",
                    kPathNames[p],
                    census.pathCalls[p],
                    share(census.pathCalls[p]));
    }

    std::printf("  min non-zero %.6g   max %.6g   calls above 100: %.4f%%   above 1000: %.4f%%\n",
                census.minNonZeroArgument,
                census.maxArgument,
                share(census.above100),
                share(census.above1000));

    std::printf("\nargument histogram (calls; bin 0 is x == 0, then log10 edges):\n");

    for (int i = 0; i <= kHistBins; ++i)
    {
        const std::size_t count = census.histCounts[static_cast<std::size_t>(i)];

        if (count == 0)
        {
            continue;
        }

        if (i == 0)
        {
            std::printf("  %14s  %14zu  %8.4f%%\n", "x == 0", count, share(count));

            continue;
        }

        const double lo = kHistT0 + (kHistT1 - kHistT0) * static_cast<double>(i - 1) /
                                        static_cast<double>(kHistBins - 1);
        const double hi = kHistT0 + (kHistT1 - kHistT0) * static_cast<double>(i) /
                                        static_cast<double>(kHistBins - 1);
        std::printf("  [1e%-6.3g,1e%-6.3g)  %14zu  %8.4f%%\n", lo, hi, count, share(count));
    }

    std::printf("\njoint (order L, region) - Boys calls:\n");
    std::printf("   L        zero            A              B              C\n");

    for (int l = 0; l <= kMaxClass; ++l)
    {
        if (census.callsByClass[static_cast<std::size_t>(l)] == 0)
        {
            continue;
        }

        std::printf("  %2d ", l);

        for (std::size_t r = 0; r < 4; ++r)
        {
            std::printf(" %14zu", census.jointClassRegion[static_cast<std::size_t>(l) * 4 + r]);
        }

        std::printf("\n");
    }

    std::printf("\njoint (order L, dispatch path) - Boys calls:\n");
    std::printf("   L          zero       A-table       A-recur       A-mixed             B"
                "             C\n");

    for (int l = 0; l <= kMaxClass; ++l)
    {
        if (census.callsByClass[static_cast<std::size_t>(l)] == 0)
        {
            continue;
        }

        std::printf("  %2d ", l);

        for (std::size_t p = 0; p < kPathCount; ++p)
        {
            std::printf(" %13zu",
                        census.jointClassPath[static_cast<std::size_t>(l) * kPathCount + p]);
        }

        std::printf("\n");
    }

    std::printf("\norder composition inside each dispatch path (row %% of that path's calls):\n");
    std::printf("   L   A-table     A-recur     A-mixed           B           C\n");

    for (int l = 0; l <= kMaxClass; ++l)
    {
        if (census.callsByClass[static_cast<std::size_t>(l)] == 0)
        {
            continue;
        }

        std::printf("  %2d ", l);

        for (std::size_t p = 1; p < kPathCount; ++p)
        {
            const std::size_t n =
                census.jointClassPath[static_cast<std::size_t>(l) * kPathCount + p];
            const std::size_t base = census.pathCalls[p];

            std::printf("  %9.3f%%",
                        base == 0 ? 0.0
                                  : 100.0 * static_cast<double>(n) / static_cast<double>(base));
        }

        std::printf("\n");
    }

    std::printf("\njoint (order L, table band) - Boys calls:\n");
    std::printf("   L          zero        A-low       A-high            B            C\n");

    for (int l = 0; l <= kMaxClass; ++l)
    {
        if (census.callsByClass[static_cast<std::size_t>(l)] == 0)
        {
            continue;
        }

        std::printf("  %2d ", l);

        for (std::size_t b = 0; b < 5; ++b)
        {
            std::printf(" %13zu", census.jointClassBand[static_cast<std::size_t>(l) * 5 + b]);
        }

        std::printf("\n");
    }

    // Order composition inside each region, which is the batching question.
    std::printf("\norder composition inside each region (row %%, of that region's calls):\n");
    std::printf("   L   region A   region B   region C   all\n");

    for (int l = 0; l <= kMaxClass; ++l)
    {
        if (census.callsByClass[static_cast<std::size_t>(l)] == 0)
        {
            continue;
        }

        std::printf("  %2d ", l);

        for (std::size_t r = 1; r < 4; ++r)
        {
            const std::size_t n = census.jointClassRegion[static_cast<std::size_t>(l) * 4 + r];
            const std::size_t base = census.regionCalls[r];

            std::printf("  %8.4f%%",
                        base == 0 ? 0.0
                                  : 100.0 * static_cast<double>(n) / static_cast<double>(base));
        }

        std::printf("  %8.4f%%\n",
                    100.0 * static_cast<double>(census.callsByClass[static_cast<std::size_t>(l)]) /
                        callCount);
    }

    // ---- The run structure -------------------------------------------------
    // The unit sizes the runs are measured against: the engine's innermost
    // contiguous burst is one task at one ket primitive pair, and its length
    // is the bra pair's primitive-pair count.
    std::printf("\nthe engine's units: %zu quartets (%zu tasks x ket primitive pairs), "
                "%zu ket groups, widest group %zu tasks\n",
                census.quartets,
                census.burstCount,
                census.groupCount,
                census.groupMaxTasks);
    std::printf("  calls per quartet %.3f   calls per burst %.3f   burst-length histogram: ",
                static_cast<double>(census.calls) / static_cast<double>(census.quartets),
                static_cast<double>(census.calls) / static_cast<double>(census.burstCount));

    for (std::size_t i = 1; i <= 16; ++i)
    {
        if (census.burstLengths[i] != 0)
        {
            std::printf("%zu:%.2f%% ",
                        i,
                        100.0 * static_cast<double>(census.burstLengths[i]) /
                            static_cast<double>(census.burstCount));
        }
    }

    std::printf("(overflow %zu)\n", census.burstLengths[0]);

    std::printf(
        "  units whose calls ALL share one region: bursts %zu of %zu (%.2f%%), "
        "quartets %zu of %zu (%.2f%%), group x g %zu of %zu (%.2f%%)\n",
        census.uniformBursts,
        census.burstCount,
        100.0 * static_cast<double>(census.uniformBursts) / static_cast<double>(census.burstCount),
        census.uniformTasks,
        census.quartets,
        100.0 * static_cast<double>(census.uniformTasks) / static_cast<double>(census.quartets),
        census.uniformGroups,
        census.groupCount,
        100.0 * static_cast<double>(census.uniformGroups) / static_cast<double>(census.groupCount));

    std::printf("\nrun structure (a run = consecutive calls sharing a region, or a band):\n");
    std::printf(
        "  unit                 calls           runs    calls/run   runs of length 1   max run"
        "   calls in runs>=4\n");

    const auto printRuns = [&](const char* name, const RunStats& s) {
        const double single =
            s.runs == 0 ? 0.0
                        : 100.0 * static_cast<double>(s.hist[1]) / static_cast<double>(s.runs);
        std::size_t inLong = 0;

        for (std::size_t i = 4; i <= kMaxRun; ++i)
        {
            inLong += i * s.hist[i];
        }

        std::printf("  %-18s %13zu  %13zu  %10.3f  %15.2f%%  %7zu  %14.4f%%\n",
                    name,
                    s.calls,
                    s.runs,
                    s.Mean(),
                    single,
                    s.maxRun,
                    s.calls == 0
                        ? 0.0
                        : 100.0 * static_cast<double>(inLong) / static_cast<double>(s.calls));
    };

    printRuns("region global", census.globalRun);
    printRuns("region sub-burst", census.subBurst);
    printRuns("region task block", census.taskBlock);
    printRuns("region group x g", census.groupRun);
    printRuns("band global", census.globalRunBand);
    printRuns("band sub-burst", census.subBurstBand);
    printRuns("band task block", census.taskBlockBand);
    printRuns("band group x g", census.groupRunBand);
    printRuns("path global", census.globalRunPath);
    printRuns("path sub-burst", census.subBurstPath);
    printRuns("path task block", census.taskBlockPath);
    printRuns("path group x g", census.groupRunPath);

    std::printf("\nrun-length histogram (region), run-weighted:\n");
    std::printf("  unit        ");

    for (std::size_t i = 1; i <= 12; ++i)
    {
        std::printf(" len%-6zu", i);
    }

    std::printf("\n");

    const auto printHist = [&](const char* name, const RunStats& s) {
        std::printf("  %-10s ", name);

        for (std::size_t i = 1; i <= 12; ++i)
        {
            std::printf(" %7.2f%%",
                        s.runs == 0
                            ? 0.0
                            : 100.0 * static_cast<double>(s.hist[i]) / static_cast<double>(s.runs));
        }

        std::printf("\n");
    };

    printHist("global", census.globalRun);
    printHist("sub-burst", census.subBurst);
    printHist("task", census.taskBlock);
    printHist("group x g", census.groupRun);
    printHist("path-sub", census.subBurstPath);
    printHist("path-task", census.taskBlockPath);
    printHist("path-gxg", census.groupRunPath);

    const auto beyond = [](const RunStats& s, std::size_t len) {
        std::size_t acc = 0;

        for (std::size_t i = len + 1; i <= kMaxRun; ++i)
        {
            acc += s.hist[i];
        }

        return acc + s.hist[0];
    };

    std::printf("  group x g runs beyond len 12: %.2f%% (max %zu)\n",
                census.groupRun.runs == 0
                    ? 0.0
                    : 100.0 * static_cast<double>(beyond(census.groupRun, 12)) /
                          static_cast<double>(census.groupRun.runs),
                census.groupRun.maxRun);
    std::printf("  group x g band runs beyond len 12: %.2f%% (max %zu)\n",
                census.groupRunBand.runs == 0
                    ? 0.0
                    : 100.0 * static_cast<double>(beyond(census.groupRunBand, 12)) /
                          static_cast<double>(census.groupRunBand.runs),
                census.groupRunBand.maxRun);

    std::printf("  the same multiset under a random permutation, group x g: %.0f runs against "
                "%zu measured (calls %zu)\n",
                census.groupExpectedRuns,
                census.groupRun.runs,
                census.groupBaselineCalls);

    // ---- The cost table ----------------------------------------------------
    if (runCost)
    {
        const double edgeScale = std::pow(10.0,
                                          0.5 * (std::log10(kCostHi) - std::log10(kCostLo)) /
                                              static_cast<double>(kCostBins - 1));
        (void)edgeScale;
        std::array<double, 4> regionNanos{};
        std::array<double, 4> regionNanosLoaded{};
        std::array<double, 5> bandNanos{};
        std::array<double, kPathCount> pathNanos{};
        double totalNanos = 0.0;
        double totalNanosLoaded = 0.0;
        double worstLoad = 1.0;

        for (int l = 0; l <= kMaxClass; ++l)
        {
            for (std::size_t bin = 0; bin <= kCostBins; ++bin)
            {
                const std::size_t count =
                    census.costCounts[static_cast<std::size_t>(l) * (kCostBins + 1) + bin];

                if (count == 0)
                {
                    continue;
                }

                const double x = CostArgumentOf(bin);
                const auto [lo, mid] = TimeBoysCall(l, x);
                const int region = RegionOf(x);
                const int band = BandOf(x);
                regionNanos[static_cast<std::size_t>(region)] += lo * static_cast<double>(count);
                regionNanosLoaded[static_cast<std::size_t>(region)] +=
                    mid * static_cast<double>(count);
                bandNanos[static_cast<std::size_t>(band)] += lo * static_cast<double>(count);
                pathNanos[static_cast<std::size_t>(PathOf(x, l))] +=
                    lo * static_cast<double>(count);
                totalNanos += lo * static_cast<double>(count);
                totalNanosLoaded += mid * static_cast<double>(count);

                if (lo > 0.0)
                {
                    worstLoad = std::max(worstLoad, mid / lo);
                }
            }
        }

        std::printf("\ncost table (this machine, under whatever load it carried; ratios only):\n");
        std::printf(
            "  per-region share of the Boys time: zero %.2f%%  A %.2f%%  B %.2f%%  C %.2f%%\n",
            100.0 * regionNanos[0] / totalNanos,
            100.0 * regionNanos[1] / totalNanos,
            100.0 * regionNanos[2] / totalNanos,
            100.0 * regionNanos[3] / totalNanos);
        std::printf(
            "  per-band share: zero %.2f%%  A-low %.2f%%  A-high %.2f%%  B %.2f%%  C %.2f%%\n",
            100.0 * bandNanos[0] / totalNanos,
            100.0 * bandNanos[1] / totalNanos,
            100.0 * bandNanos[2] / totalNanos,
            100.0 * bandNanos[3] / totalNanos,
            100.0 * bandNanos[4] / totalNanos);
        std::printf("  per-path share: ");

        for (std::size_t p = 0; p < kPathCount; ++p)
        {
            std::printf("%s %.2f%%  ", kPathNames[p], 100.0 * pathNanos[p] / totalNanos);
        }

        std::printf("\n");
        std::printf("  worst min-to-median round movement (the load proxy): %.3fx; "
                    "loaded total would be %.1f%% of the min total\n",
                    worstLoad,
                    100.0 * totalNanosLoaded / totalNanos);

        if (runBuild && stats.kernelVrrWallTime.count() > 0)
        {
            const double vrrMs =
                std::chrono::duration<double, std::milli>(stats.kernelVrrWallTime).count();
            std::printf("  the same per-region split of the parent's VRR-phase attribution: "
                        "zero %.2f%%  A %.2f%%  B %.2f%%  C %.2f%% (of a %.1f%% Boys share)\n",
                        100.0 * (regionNanos[0] / 1e6) / vrrMs,
                        100.0 * (regionNanos[1] / 1e6) / vrrMs,
                        100.0 * (regionNanos[2] / 1e6) / vrrMs,
                        100.0 * (regionNanos[3] / 1e6) / vrrMs,
                        100.0 * (totalNanos / 1e6) / vrrMs);
        }
    }

    std::printf("\nsink %.6e\n", gSink);

    return 0;
}
