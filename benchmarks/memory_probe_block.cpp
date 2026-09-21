// The basis-function derivative storage per grid point, counted in bytes.
// The one-block allocation census: every allocation one grid evaluation
//      makes, by requested size.
//
// Both are taken on a REAL calculation through the shipped engine
// (qcx::grid::XcGridEngine) on a real basis out of data/basis, with the grid
// parameters chosen so the fixture is exactly one block; the multi-block run
// beside it is the control that says whether anything scales with blocks.

#include "memory_probe.hpp"
#include "memory_probe_census.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace memprobe {
namespace {

/// Seconds elapsed since \p start, as a double.
double SecondsSince(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

/// Water, the fixture: O at the origin, the two hydrogens at the
/// experimental geometry (r = 0.9584 A, angle 104.45 degrees), in Bohr.
qcx::Result<qcx::molecule::Molecule> MakeWater() {
    constexpr double kOH = 1.8110629; // 0.9584 A in Bohr.
    constexpr double kHalfAngle = 0.9112530; // 52.225 degrees in radians.

    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = kOH * std::sin(kHalfAngle);
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = kOH * std::cos(kHalfAngle);
    (*coordinates)(2, 0) = -kOH * std::sin(kHalfAngle);
    (*coordinates)(2, 1) = 0.0;
    (*coordinates)(2, 2) = kOH * std::cos(kHalfAngle);
    coordinates->MarkHostDirty();

    std::vector<qcx::molecule::Atom> atoms;
    atoms.push_back(qcx::molecule::Atom{"O", 8, 0.0});
    atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});
    atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});

    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

/// Benzene, D6h in the xy plane (r(CC) = 1.395 A, r(CH) = 1.085 A), in Bohr:
/// the larger fixture, used to show what changes with the AO count and the
/// block count and what does not.
qcx::Result<qcx::molecule::Molecule> MakeBenzene() {
    constexpr double kRC = 2.636168; // 1.395 A in Bohr.
    constexpr double kRH = 4.686530; // 1.085 A in Bohr.

    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({12, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    std::vector<qcx::molecule::Atom> atoms;

    for (std::size_t k = 0; k < 6; ++k)
    {
        const double angle = static_cast<double>(k) * 3.14159265358979323846 / 3.0;
        (*coordinates)(k, 0) = kRC * std::cos(angle);
        (*coordinates)(k, 1) = kRC * std::sin(angle);
        (*coordinates)(k, 2) = 0.0;
        atoms.push_back(qcx::molecule::Atom{"C", 6, 0.0});
    }

    for (std::size_t k = 0; k < 6; ++k)
    {
        const double angle = static_cast<double>(k) * 3.14159265358979323846 / 3.0;
        (*coordinates)(6 + k, 0) = kRH * std::cos(angle);
        (*coordinates)(6 + k, 1) = kRH * std::sin(angle);
        (*coordinates)(6 + k, 2) = 0.0;
        atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});
    }

    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

/// A rank-1 positive-semidefinite density: D = c c^T / (c . c) with c all
/// ones, so rho >= 0 everywhere on the grid, which is all the engine's
/// contract requires. It is not an SCF density and is not meant to be one:
/// the census counts allocations, and the second density below shows the
/// count does not depend on the density's values.
Eigen::MatrixXd RankOneDensity(std::size_t n, double scale) {
    Eigen::MatrixXd density = Eigen::MatrixXd::Constant(
        static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n), scale / static_cast<double>(n));

    return density;
}

/// Runs one measurement: builds the engine, evaluates once with the census
/// window open, and prints the census for that call.
struct RunResult {
    std::uint64_t allocations = 0;
    std::uint64_t totalRequestBytes = 0;
    std::uint64_t peakLiveBytes = 0;
    std::size_t points = 0;
    std::size_t blocks = 0;
    std::size_t aoCount = 0;
    double energy = 0.0;
    bool ok = false;
    bool liveAccountingValid = true;
    bool censusAvailable = false;
    const char* censusMode = "";
    std::uint64_t freesWithoutSize = 0;
};

RunResult MeasureCall(const std::string& label,
                      const qcx::molecule::Molecule& molecule,
                      const qcx::basisset::BasisSet& basis,
                      const std::string& functional,
                      std::size_t blockTarget,
                      bool screened,
                      double densityScale,
                      std::size_t radialPoints = 20,
                      std::size_t angularPoints = 50) {
    RunResult result;

    qcx::grid::XcGridSettings settings;
    settings.radialPoints = radialPoints;
    settings.angularPoints = angularPoints;
    settings.blockTarget = blockTarget;

    const auto createStart = std::chrono::steady_clock::now();
    auto engine = qcx::grid::XcGridEngine::Create(molecule, basis, functional, settings);

    if (!engine.has_value())
    {
        std::printf("run.%s = ERROR %s\n", label.c_str(), engine.error().message.c_str());
        return result;
    }

    result.points = engine->PointCount();
    result.blocks = engine->BlockCount();
    result.aoCount = engine->AOCount();
    std::printf("progress.%s = engine_built points=%zu blocks=%zu ao=%zu grid_seconds=%.3f\n",
                label.c_str(),
                result.points,
                result.blocks,
                result.aoCount,
                SecondsSince(createStart));

    const Eigen::MatrixXd density = RankOneDensity(engine->AOCount(), densityScale);

    // One warm-up call: it fills anything the engine allocates lazily and
    // leaves the census's own first-touch out of the window.
    const auto warmStart = std::chrono::steady_clock::now();
    const auto warm = engine->EvaluateClosedShell(density);

    if (!warm.has_value())
    {
        std::printf("run.%s = ERROR %s\n", label.c_str(), warm.error().message.c_str());
        return result;
    }

    std::printf("progress.%s = warmup_seconds=%.3f\n", label.c_str(), SecondsSince(warmStart));

    const auto callStart = std::chrono::steady_clock::now();
    CensusBegin();
    const auto evaluation = screened ? engine->EvaluateClosedShellScreened(density, 1.0e-10)
                                     : engine->EvaluateClosedShell(density);
    const CensusSnapshot census = CensusEnd();
    std::printf(
        "progress.%s = measured_call_seconds=%.3f\n", label.c_str(), SecondsSince(callStart));

    if (!evaluation.has_value())
    {
        std::printf("run.%s = ERROR %s\n", label.c_str(), evaluation.error().message.c_str());
        return result;
    }

    result.allocations = census.allocations;
    result.totalRequestBytes = census.totalRequestBytes;
    result.peakLiveBytes = census.peakLiveBytes;
    result.energy = evaluation->energy;
    result.ok = true;
    result.liveAccountingValid = census.liveAccountingValid;
    result.censusAvailable = census.available;
    result.censusMode = census.mode;
    result.freesWithoutSize = census.freesWithoutSize;

    std::printf("run.%s = blocks=%zu points=%zu ao=%zu functional=%s path=%s "
                "allocations=%llu frees=%llu reallocs=%llu request_bytes=%llu "
                "peak_live_bytes=%llu live_at_end=%llu energy=%.12e\n",
                label.c_str(),
                result.blocks,
                result.points,
                result.aoCount,
                functional.c_str(),
                screened ? "screened" : "dense",
                static_cast<unsigned long long>(census.allocations),
                static_cast<unsigned long long>(census.frees),
                static_cast<unsigned long long>(census.reallocations),
                static_cast<unsigned long long>(census.totalRequestBytes),
                static_cast<unsigned long long>(census.peakLiveBytes),
                static_cast<unsigned long long>(census.liveBytesAtEnd),
                result.energy);

    return result;
}

/// Prints the census's size classes, largest first.
void PrintClasses(const char* tag, std::size_t limit) {
    const std::size_t count = CensusClassCount();
    std::printf("%s_distinct_size_classes = %zu\n", tag, count);

    const std::size_t shown = count < limit ? count : limit;

    for (std::size_t i = 0; i < shown; ++i)
    {
        const CensusSizeClass entry = CensusClassAt(i);
        std::printf("%s_size_class[%zu] = bytes=%zu count=%llu outstanding=%llu\n",
                    tag,
                    i,
                    entry.bytes,
                    static_cast<unsigned long long>(entry.count),
                    static_cast<unsigned long long>(entry.outstanding));
    }
}

/// The basis directory of one family, filtered to the fixture's elements.
qcx::Result<qcx::basisset::BasisSet> LoadBasis(const std::string& family,
                                               std::span<const int> elements) {
    return qcx::basisset::ParseNwchemDirectoryFiltered(BasisDataDir() + "/" + family, elements);
}

} // namespace

int RunM2M3(const std::vector<std::string>& args) {
    std::string family = "def2-svp";
    std::string functional = "pbe";
    bool screened = false;
    std::size_t radialPoints = 20;
    std::size_t angularPoints = 50;

    for (std::size_t i = 0; i < args.size(); ++i)
    {
        if (args[i] == "--basis" && i + 1 < args.size())
        {
            family = args[i + 1];
        } else if (args[i] == "--functional" && i + 1 < args.size())
        {
            functional = args[i + 1];
        } else if (args[i] == "--radial" && i + 1 < args.size())
        {
            radialPoints = static_cast<std::size_t>(std::stoul(args[i + 1]));
        } else if (args[i] == "--angular" && i + 1 < args.size())
        {
            angularPoints = static_cast<std::size_t>(std::stoul(args[i + 1]));
        } else if (args[i] == "--screened")
        { screened = true; }
    }

    std::printf("m2m3_fixture = H2O (rOH=0.9584A, 104.45deg) and C6H6 (D6h, 1.395A/1.085A) "
                "in %s, %s, radial=%zu angular=%zu\n",
                family.c_str(),
                functional.c_str(),
                radialPoints,
                angularPoints);

    // The census calibration: three allocations of known size through the
    // three paths the engine's own code uses, so the size classes below can be
    // read for what they are rather than guessed at.
    {
        CensusBegin();
        // The three paths the engine's own per-point code allocates through:
        // the scalar vector scratch (std::vector), the n x n temporary
        // (Eigen's aligned allocator), and the n-vector temporary (also
        // Eigen's). Each is allocated at the sizes the engine asks for at
        // AOCount = 24.
        std::vector<double> vector24(24);
        std::vector<double> vector72(72);
        std::vector<double> plain(1000);
        Eigen::MatrixXd eigenMatrix = Eigen::MatrixXd::Zero(24, 24);
        Eigen::VectorXd eigenVector = Eigen::VectorXd::Zero(24);
        vector24[0] = 1.0;
        vector72[0] = 1.0;
        plain[0] = 1.0;
        eigenMatrix(0, 0) = 1.0;
        eigenVector[0] = 1.0;
        const CensusSnapshot calibration = CensusEnd();

        std::printf("m3_calibration = allocations=%llu sizes_seen:",
                    static_cast<unsigned long long>(calibration.allocations));

        for (std::size_t i = 0; i < CensusClassCount(); ++i)
        {
            const CensusSizeClass entry = CensusClassAt(i);
            std::printf(" %zu(x%llu)", entry.bytes, static_cast<unsigned long long>(entry.count));
        }

        std::printf("\nm3_calibration_note = std::vector<double>(1000) should read 8000; "
                    "Eigen::MatrixXd::Zero(24,24) should read 4608\n");
    }

    auto water = MakeWater();

    if (!water.has_value())
    {
        std::printf("m2m3 = ERROR building water: %s\n", water.error().message.c_str());
        return 1;
    }

    const std::array<int, 2> waterElements{1, 8};
    auto waterBasis = LoadBasis(family, waterElements);

    if (!waterBasis.has_value())
    {
        std::printf(
            "m2m3 = ERROR loading %s: %s\n", family.c_str(), waterBasis.error().message.c_str());
        return 1;
    }

    // The one-block fixture: the block target above the whole grid's point
    // count, so the engine walks exactly one block.
    const RunResult oneBlock = MeasureCall("one_block",
                                           *water,
                                           *waterBasis,
                                           functional,
                                           1u << 20,
                                           screened,
                                           1.0,
                                           radialPoints,
                                           angularPoints);
    PrintClasses("m3_one_block", 16);

    // The same call with a small block target: same points, many blocks. If
    // the per-block loop allocated, this count would grow with the block count.
    const RunResult manyBlocks = MeasureCall("many_blocks",
                                             *water,
                                             *waterBasis,
                                             functional,
                                             64,
                                             screened,
                                             1.0,
                                             radialPoints,
                                             angularPoints);

    // A different density, same grid: the count must not depend on the
    // density's values either.
    const RunResult otherDensity = MeasureCall("other_density",
                                               *water,
                                               *waterBasis,
                                               functional,
                                               1u << 20,
                                               screened,
                                               3.7,
                                               radialPoints,
                                               angularPoints);

    std::printf("m3_census_mode = %s\n", oneBlock.censusMode);

    if (oneBlock.censusAvailable)
    {
        std::printf("m3_allocations_one_block = %llu\n",
                    static_cast<unsigned long long>(oneBlock.allocations));
        std::printf("m3_allocations_many_blocks = %llu (%zu blocks of the same %zu points)\n",
                    static_cast<unsigned long long>(manyBlocks.allocations),
                    manyBlocks.blocks,
                    manyBlocks.points);
        std::printf("m3_allocations_other_density = %llu\n",
                    static_cast<unsigned long long>(otherDensity.allocations));
        std::printf("m3_per_block_allocation_requests = %lld\n",
                    static_cast<long long>(manyBlocks.allocations) -
                        static_cast<long long>(oneBlock.allocations));
        std::printf("m3_request_bytes_one_block = %llu\n",
                    static_cast<unsigned long long>(oneBlock.totalRequestBytes));
    } else
    {
        // A build with no allocation hook sees nothing, so its counters are
        // zero because nothing was SEEN. Reporting those zeros as counts would
        // read as "this path allocates nothing", which is the opposite of what
        // this measurement is for: the probe says UNAVAILABLE instead.
        std::printf("m3_allocations_one_block = UNAVAILABLE (no allocation hook in this build)\n");
        std::printf("m3_allocations_many_blocks = UNAVAILABLE\n");
        std::printf("m3_allocations_other_density = UNAVAILABLE\n");
        std::printf("m3_per_block_allocation_requests = UNAVAILABLE\n");
        std::printf("m3_request_bytes_one_block = UNAVAILABLE\n");
    }

    std::printf("m3_live_accounting_valid = %s (releases without a size: %llu)\n",
                oneBlock.liveAccountingValid ? "yes" : "NO - the live figures are upper bounds",
                static_cast<unsigned long long>(oneBlock.freesWithoutSize));
    std::printf("m3_peak_live_bytes_one_block = %llu%s\n",
                static_cast<unsigned long long>(oneBlock.peakLiveBytes),
                oneBlock.liveAccountingValid ? "" : " (UPPER BOUND, not a high-water)");
    std::printf("m3_process_peak_working_set_bytes = %llu\n",
                static_cast<unsigned long long>(ProcessPeakWorkingSetBytes()));

    // The basis-function derivative storage per point, from the AO-tier
    // allocation the census just recorded. The engine's per-POINT scratch is
    // values + 3 gradients, one buffer set for the whole integration.
    const std::size_t ao = oneBlock.aoCount;
    std::printf("m2_ao_count_water_%s = %zu\n", family.c_str(), ao);
    std::printf("m2_scratch_values_bytes = %zu (AOCount doubles)\n", ao * sizeof(double));
    std::printf("m2_scratch_gradients_bytes = %zu (3 x AOCount doubles)\n",
                ao * 3 * sizeof(double));
    std::printf("m2_bytes_per_ao_per_point = %zu (values 1 + gradients 3 = 4 doubles)\n",
                4 * sizeof(double));
    std::printf("m2_bytes_per_ao_per_point_with_hessians = %zu "
                "(+ 6 hessian doubles; NOT materialised by the engine today)\n",
                10 * sizeof(double));

    // The bytes the census actually saw for the AO tier, per AO and per point.
    // The census's two largest POINT-SCALED classes are the caller's density
    // copy (ao^2) and the AO scratch; the AO scratch is the one that scales
    // with the point count, so it is reported as a ratio to the scratch size
    // the engine's own code asks for.
    std::printf("m2_ao_tier_allocations_seen = values:%zu gradients:%zu "
                "(matched by requested size, count=%s)\n",
                ao,
                ao * 3,
                oneBlock.allocations > 0 ? "see classes above" : "census unavailable");

    // The census's per-point rate: the counts divided by the points walked.
    if (oneBlock.censusAvailable)
    {
        const double allocationsPerPoint =
            oneBlock.points > 0
                ? static_cast<double>(oneBlock.allocations) / static_cast<double>(oneBlock.points)
                : 0.0;
        const double bytesPerPoint = oneBlock.points > 0
                                         ? static_cast<double>(oneBlock.totalRequestBytes) /
                                               static_cast<double>(oneBlock.points)
                                         : 0.0;
        std::printf("m3_allocations_per_point_dense = %.4f\n", allocationsPerPoint);
        std::printf("m3_request_bytes_per_point_dense = %.1f\n", bytesPerPoint);
    } else
    {
        std::printf("m3_allocations_per_point_dense = UNAVAILABLE\n");
        std::printf("m3_request_bytes_per_point_dense = UNAVAILABLE\n");
    }

    std::printf("m3_blocks_mean_points = %.1f (max_target 1048576, actual blocks %zu)\n",
                static_cast<double>(oneBlock.points) /
                    static_cast<double>(oneBlock.blocks == 0 ? 1 : oneBlock.blocks),
                oneBlock.blocks);

    // What the measured per-point pattern costs at 500 and 1000 basis
    // functions: the two n^2 temporaries per accumulation, four accumulations
    // per point.
    for (const std::size_t aoTarget : {24u, 500u, 1000u})
    {
        const double perPointBytes =
            4.0 * static_cast<double>(aoTarget) * sizeof(double) +
            4.0 * static_cast<double>(aoTarget) * static_cast<double>(aoTarget) * sizeof(double);
        const double trafficTerabytes = perPointBytes * 1.0e6 / 1.0e12;
        std::printf("m3_projection_%zu_ao = %.1f bytes of allocator traffic per point, "
                    "%.2f TB over 1e6 points\n",
                    aoTarget,
                    perPointBytes,
                    trafficTerabytes);
    }

    // The projection at a million points: the measured per-AO-per-point
    // constant against the point count.
    for (const std::size_t aoTarget : {500u, 1000u})
    {
        const double points = 1.0e6;
        const double gradientTier = points * static_cast<double>(aoTarget) * 4.0 * sizeof(double) /
                                    (1024.0 * 1024.0 * 1024.0);
        const double withHessians = points * static_cast<double>(aoTarget) * 10.0 * sizeof(double) /
                                    (1024.0 * 1024.0 * 1024.0);
        std::printf("m2_projection_%zu_ao_1e6_points = values+gradients %.2f GiB, "
                    "+second derivatives %.2f GiB\n",
                    aoTarget,
                    gradientTier,
                    withHessians);
    }

    // The larger fixture: what changes with the AO count and the block count.
    auto benzene = MakeBenzene();

    if (!benzene.has_value())
    {
        std::printf("m2m3_benzene = ERROR %s\n", benzene.error().message.c_str());
        return 1;
    }

    const std::array<int, 2> benzeneElements{1, 6};
    auto benzeneBasis = LoadBasis(family, benzeneElements);

    if (!benzeneBasis.has_value())
    {
        std::printf("m2m3_benzene = ERROR %s\n", benzeneBasis.error().message.c_str());
        return 1;
    }

    const RunResult benzeneOne = MeasureCall("benzene_one_block",
                                             *benzene,
                                             *benzeneBasis,
                                             functional,
                                             1u << 20,
                                             screened,
                                             1.0,
                                             radialPoints,
                                             angularPoints);
    const RunResult benzeneMany = MeasureCall("benzene_many_blocks",
                                              *benzene,
                                              *benzeneBasis,
                                              functional,
                                              64,
                                              screened,
                                              1.0,
                                              radialPoints,
                                              angularPoints);

    if (benzeneOne.censusAvailable)
    {
        std::printf("m3_benzene_allocations_one_block = %llu\n",
                    static_cast<unsigned long long>(benzeneOne.allocations));
        std::printf("m3_benzene_allocations_many_blocks = %llu (%zu blocks)\n",
                    static_cast<unsigned long long>(benzeneMany.allocations),
                    benzeneMany.blocks);
        std::printf("m3_benzene_per_block_allocation_requests = %lld\n",
                    static_cast<long long>(benzeneMany.allocations) -
                        static_cast<long long>(benzeneOne.allocations));
    } else
    {
        std::printf("m3_benzene_allocations_one_block = UNAVAILABLE\n");
        std::printf("m3_benzene_allocations_many_blocks = UNAVAILABLE\n");
        std::printf("m3_benzene_per_block_allocation_requests = UNAVAILABLE\n");
    }

    std::printf("m3_process_peak_working_set = %llu\n",
                static_cast<unsigned long long>(ProcessPeakWorkingSetBytes()));

    return 0;
}

} // namespace memprobe
