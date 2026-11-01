// The AO evaluator's third-derivative tier and the fixed-density XC Hessian
// walk, measured on water in three basis sets.
//
// Three measurements, each printed as pipe-separated key=value rows:
//
//   1. The three selected fetch entry points - values and gradients, the same
//      plus Hessians, and the same plus third derivatives - on a fixed small
//      set of grid points, each with the same per-point shell selection. The
//      figures are per AO slot per point, because the work is linear in the
//      kept slots, and the third tier's marginal cost is the difference
//      between the last two entry points: one pass computes every tier, so
//      the marginal is the deeper arithmetic, not a second traversal.
//   2. The fixed-density Hessian walk for a density-only functional, a GGA and
//      a meta-GGA on the engine's own grid: wall time per visited point and
//      the walk's counts, including the third-tier slots it evaluated and the
//      second-derivative kernel contractions it issued. A functional the walk
//      refuses is reported on its row with the refusal's code and message,
//      and the remaining arms still run.
//   3. The walk's per-point second-derivative kernel call pattern, timed
//      directly at one polarized point: the active component count, the sides
//      per call and the calls per point from the contract's 16-double
//      capacity, then the time of that call pattern per point.
//
// Protocol, and it is not optional:
//
//   - every arm runs twice; the first traversal is warm-up and is discarded,
//     the second is reported. A single run is never reported.
//   - a fixed-work integer canary runs before and after every arm, and the
//     width between the two readings is printed on the arm's row, so a wide
//     band marks a figure taken on a noisy machine.
//   - the fetch and kernel arms repeat their traversal inside a run. That is
//     a timing loop, not extra samples: the per-traversal figure is the
//     measured run divided by the repetition count.
//   - figures from binaries of different build configurations are not
//     comparable; nothing here is a cross-binary comparison.
//
// The fixture is H2O at the experimental geometry, on a small grid (20 radial
// and 50 angular points per atom) so the whole run stays short. The density
// is a rank-one positive-semidefinite matrix with the beta spin at 0.8 of the
// alpha one, so the points are spin-polarized. It is a fixture, not an SCF
// density.
#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/grid/xc_gradient.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

/// The grid the walks run on: small, so a full run stays short.
constexpr std::size_t kRadialPoints = 20;
constexpr std::size_t kAngularPoints = 50;
constexpr std::size_t kBlockTarget = 256;

/// The screening threshold the walks run under.
constexpr double kScreeningTolerance = 1e-10;

/// The fixed subset of grid points the fetch tiers are measured on.
constexpr std::size_t kPointSubsetSize = 128;

/// A shell is kept at a fetch point when any of its AO values reaches this.
/// It is a benchmark choice that fixes the pattern, not the walk's screen.
constexpr double kSelectionFloor = 1e-6;

/// Fetch traversals inside one measured run (a timing loop, not samples).
constexpr std::size_t kFetchRepeats = 64;

/// Simulated points in one kernel arm; each point issues callsPerPoint calls.
constexpr std::size_t kKernelPointIterations = 20000;

/// Iterations of the canary's fixed integer work.
constexpr std::uint32_t kCanaryRounds = 1u << 21;

/// The observable use of every timed result. Written once per traversal, so
/// it cannot be dropped and no volatile read-modify-write sits in a loop.
volatile double gSink = 0.0;

double NanoSeconds(const Clock::time_point& from, const Clock::time_point& to) {
    return std::chrono::duration<double, std::nano>(to - from).count();
}

/// The canary: a fixed xorshift chain, no floating point and no memory beyond
/// its accumulator.
double CanaryNanoseconds() {
    constexpr std::uint64_t kSeed = 0x243F6A8885A308D3ull;
    volatile std::uint64_t sink = 0;
    const Clock::time_point from = Clock::now();

    std::uint64_t state = kSeed;

    for (std::uint32_t i = 0; i < kCanaryRounds; ++i)
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
    }

    sink = state;
    const Clock::time_point to = Clock::now();
    (void)sink;

    return NanoSeconds(from, to);
}

/// One arm's readings: the measured run and the two canary readings that
/// bracket it (and its warm-up).
struct Arm {
    double ns = 0.0;
    double canaryBeforeNs = 0.0;
    double canaryAfterNs = 0.0;
    double canaryWidthPercent = 0.0;
};

/// Runs \p body once as warm-up and once as the measured run.
template <typename Body> Arm TimeArm(const Body& body) {
    Arm arm;
    arm.canaryBeforeNs = CanaryNanoseconds();
    body();

    const Clock::time_point from = Clock::now();
    body();
    const Clock::time_point to = Clock::now();

    arm.ns = NanoSeconds(from, to);
    arm.canaryAfterNs = CanaryNanoseconds();
    const double low = std::min(arm.canaryBeforeNs, arm.canaryAfterNs);
    const double high = std::max(arm.canaryBeforeNs, arm.canaryAfterNs);
    arm.canaryWidthPercent = low > 0.0 ? 100.0 * (high - low) / low : 0.0;
    return arm;
}

/// Water at the experimental geometry, in Bohr: O at the origin, the H atoms
/// symmetric about the z axis.
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

/// A rank-one positive-semidefinite density, D = c c^T / n with c all ones:
/// rho is nonnegative at every point, which is the walk's contract. It is a
/// fixture, not an SCF density.
Eigen::MatrixXd RankOneDensity(std::size_t aoCount, double scale) {
    return Eigen::MatrixXd::Constant(static_cast<Eigen::Index>(aoCount),
                                     static_cast<Eigen::Index>(aoCount),
                                     scale / static_cast<double>(aoCount));
}

/// A deterministic, evenly spaced subset of the grid's points.
std::vector<std::array<double, 3>> FixedPointSubset(const excgrid::BlockGrid& grid,
                                                    std::size_t maxPoints) {
    std::vector<std::array<double, 3>> all;
    all.reserve(grid.TotalPointCount());

    for (const excgrid::Block& block : grid.Blocks())
    {
        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            all.push_back(block.points[point]);
        }
    }

    if (all.size() <= maxPoints)
    {
        return all;
    }

    std::vector<std::array<double, 3>> subset;
    subset.reserve(maxPoints);
    const double step = static_cast<double>(all.size()) / static_cast<double>(maxPoints);

    for (std::size_t index = 0; index < maxPoints; ++index)
    {
        subset.push_back(all[static_cast<std::size_t>(static_cast<double>(index) * step)]);
    }

    return subset;
}

/// The shell selection every fetch entry point sees at every point.
struct Selection {
    std::vector<std::vector<std::size_t>> shells; ///< Kept shell indices per point.
    std::size_t aoTotal = 0; ///< Selected AO slots summed over all points.
    std::size_t aoMin = 0; ///< Fewest AO slots selected at one point.
    std::size_t aoMax = 0; ///< Most AO slots selected at one point.
};

/// Builds the selection once, from the shell blocks' own AO values, so the
/// three entry points read the identical pattern.
Selection BuildSelection(const qcx::grid::AoEvaluator& evaluator,
                         const std::vector<std::array<double, 3>>& points) {
    Selection selection;
    selection.shells.resize(points.size());
    selection.aoMin = points.empty() ? 0 : evaluator.AOCount();
    std::vector<double> values(evaluator.AOCount());
    const std::span<const qcx::grid::ShellRange> ranges = evaluator.ShellRanges();

    for (std::size_t index = 0; index < points.size(); ++index)
    {
        evaluator.Evaluate(points[index], values);
        std::vector<std::size_t>& kept = selection.shells[index];
        std::size_t selected = 0;

        for (std::size_t shell = 0; shell < ranges.size(); ++shell)
        {
            bool significant = false;

            for (std::size_t slot = 0; slot < ranges[shell].functionCount; ++slot)
            {
                if (std::abs(values[ranges[shell].aoOffset + slot]) >= kSelectionFloor)
                {
                    significant = true;
                }
            }

            if (significant)
            {
                kept.push_back(shell);
                selected += ranges[shell].functionCount;
            }
        }

        // A point with no shell above the floor keeps its first shell, so the
        // pattern is never empty and the per-AO figures have a denominator.
        if (kept.empty() && !ranges.empty())
        {
            kept.push_back(0);
            selected = ranges[0].functionCount;
        }

        selection.aoTotal += selected;
        selection.aoMin = std::min(selection.aoMin, selected);
        selection.aoMax = std::max(selection.aoMax, selected);
    }

    return selection;
}

enum class FetchEntry {
    kGradients,
    kDerivatives,
    kThirdDerivatives,
};

const char* EntryName(FetchEntry entry) {
    switch (entry)
    {
    case FetchEntry::kGradients:
        return "gradients";
    case FetchEntry::kDerivatives:
        return "derivatives";
    case FetchEntry::kThirdDerivatives:
        return "derivatives_to_third";
    }

    return "unknown";
}

/// One basis set's scratch: the four tiers' output streams, sized by AOCount.
struct FetchBuffers {
    explicit FetchBuffers(std::size_t aoCount) :
        values(aoCount), gradients(3 * aoCount), hessians(6 * aoCount), thirds(10 * aoCount) {}

    std::vector<double> values;
    std::vector<double> gradients;
    std::vector<double> hessians;
    std::vector<double> thirds;
};

/// One traversal of the fixed point set through one selected entry point.
/// The reads of each tier the entry wrote are the observable, so the stores
/// cannot be dropped.
void FetchPass(const qcx::grid::AoEvaluator& evaluator,
               const std::vector<std::array<double, 3>>& points,
               const Selection& selection,
               FetchEntry entry,
               FetchBuffers& buffers) {
    double local = 0.0;

    for (std::size_t index = 0; index < points.size(); ++index)
    {
        const std::span<const std::size_t> shells(selection.shells[index]);

        if (shells.empty())
        {
            continue;
        }

        switch (entry)
        {
        case FetchEntry::kGradients:
            evaluator.EvaluateGradientsSelected(
                points[index], shells, buffers.values, buffers.gradients);
            break;
        case FetchEntry::kDerivatives:
            evaluator.EvaluateDerivativesSelected(
                points[index], shells, buffers.values, buffers.gradients, buffers.hessians);
            break;
        case FetchEntry::kThirdDerivatives:
            evaluator.EvaluateDerivativesToThirdSelected(points[index],
                                                         shells,
                                                         buffers.values,
                                                         buffers.gradients,
                                                         buffers.hessians,
                                                         buffers.thirds);
            break;
        }

        const qcx::grid::ShellRange& first = evaluator.ShellRanges()[shells.front()];
        local += buffers.values[first.aoOffset];
        local += buffers.gradients[3 * first.aoOffset];

        if (entry != FetchEntry::kGradients)
        {
            local += buffers.hessians[6 * first.aoOffset];
        }

        if (entry == FetchEntry::kThirdDerivatives)
        {
            local += buffers.thirds[10 * first.aoOffset];
        }
    }

    gSink = local;
}

/// One fixture: the basis, the engine that carries the grid and the evaluator,
/// and the spin-density pair the walks run at.
struct Fixture {
    std::string label;
    qcx::basisset::BasisSet basis;
    qcx::grid::XcGridEngine engine;
    Eigen::MatrixXd densityAlpha;
    Eigen::MatrixXd densityBeta;
};

void PrintError(const char* tag, const qcx::Error& error) {
    const std::string code(qcx::ToString(error.code));
    std::printf(
        "%s | status=error error_code=%s error=%s\n", tag, code.c_str(), error.message.c_str());
}

void PrintExcgridError(const char* tag, excgrid::ErrorCode error) {
    const char* name = "unknown";

    if (error == excgrid::ErrorCode::kInvalidArgument)
    {
        name = "kInvalidArgument";
    } else if (error == excgrid::ErrorCode::kUnsupported)
    { name = "kUnsupported"; }
    std::printf("%s | status=error error_code=%s\n", tag, name);
}

std::optional<Fixture> MakeFixture(const qcx::molecule::Molecule& water,
                                   const char* family,
                                   const char* label,
                                   const qcx::grid::XcGridSettings& settings) {
    const std::array<int, 2> waterElements{1, 8};
    auto basis = qcx::basisset::ParseNwchemDirectoryFiltered(
        std::string(QcxBasisDataDir) + "/" + family, waterElements);

    if (!basis.has_value())
    {
        PrintError((std::string("fixture | basis=") + label).c_str(), basis.error());
        return std::nullopt;
    }

    auto engine = qcx::grid::XcGridEngine::Create(water, *basis, "slater", settings);

    if (!engine.has_value())
    {
        PrintError((std::string("fixture | basis=") + label).c_str(), engine.error());
        return std::nullopt;
    }

    const std::size_t aoCount = engine->AOCount();
    Eigen::MatrixXd alpha = RankOneDensity(aoCount, 2.0);
    Eigen::MatrixXd beta = RankOneDensity(aoCount, 1.6);

    return Fixture{label, std::move(*basis), std::move(*engine), std::move(alpha), std::move(beta)};
}

/// Measurement 1: the three selected fetch entry points, per AO slot per
/// point, and the third tier's marginal cost.
void RunFetchPhase(const Fixture& fixture) {
    const qcx::grid::AoEvaluator& evaluator = fixture.engine.Evaluator();
    const std::vector<std::array<double, 3>> points =
        FixedPointSubset(fixture.engine.Grid(), kPointSubsetSize);
    const Selection selection = BuildSelection(evaluator, points);
    const double pointCount = static_cast<double>(points.size());
    const double selectedTotal = static_cast<double>(selection.aoTotal);

    std::printf("fetch_set | basis=%s ao_count=%zu shells=%zu points=%zu selected_ao_total=%zu "
                "selected_ao_mean=%.2f selected_ao_min=%zu selected_ao_max=%zu "
                "selection_floor=%.1e\n",
                fixture.label.c_str(),
                evaluator.AOCount(),
                evaluator.ShellRanges().size(),
                points.size(),
                selection.aoTotal,
                pointCount > 0.0 ? selectedTotal / pointCount : 0.0,
                selection.aoMin,
                selection.aoMax,
                kSelectionFloor);

    FetchBuffers buffers(evaluator.AOCount());
    constexpr FetchEntry kEntries[] = {
        FetchEntry::kGradients, FetchEntry::kDerivatives, FetchEntry::kThirdDerivatives};
    double perTraversal[3] = {0.0, 0.0, 0.0};

    for (std::size_t entry = 0; entry < 3; ++entry)
    {
        const Arm arm = TimeArm([&] {
            for (std::size_t repeat = 0; repeat < kFetchRepeats; ++repeat)
            {
                FetchPass(evaluator, points, selection, kEntries[entry], buffers);
            }
        });

        perTraversal[entry] = arm.ns / static_cast<double>(kFetchRepeats);
        std::printf("fetch | basis=%s entry=%s points=%zu selected_ao_total=%zu "
                    "ns_per_traversal=%.1f ns_per_point=%.1f ns_per_ao_point=%.2f "
                    "canary_before_ns=%.1f canary_after_ns=%.1f canary_width_pct=%.2f\n",
                    fixture.label.c_str(),
                    EntryName(kEntries[entry]),
                    points.size(),
                    selection.aoTotal,
                    perTraversal[entry],
                    pointCount > 0.0 ? perTraversal[entry] / pointCount : 0.0,
                    selectedTotal > 0.0 ? perTraversal[entry] / selectedTotal : 0.0,
                    arm.canaryBeforeNs,
                    arm.canaryAfterNs,
                    arm.canaryWidthPercent);
    }

    // The third tier is fetched by one pass that writes every tier, so its
    // marginal cost is the second entry point's figure subtracted from the
    // third's rather than a separate traversal.
    const double marginal = perTraversal[2] - perTraversal[1];
    std::printf("fetch | basis=%s entry=marginal_third ns_per_traversal=%.1f ns_per_point=%.1f "
                "ns_per_ao_point=%.2f source=derivatives_to_third_minus_derivatives\n",
                fixture.label.c_str(),
                marginal,
                pointCount > 0.0 ? marginal / pointCount : 0.0,
                selectedTotal > 0.0 ? marginal / selectedTotal : 0.0);
}

/// Measurement 2: the fixed-density Hessian walk, one row per functional.
void RunWalkPhase(const Fixture& fixture,
                  const excgrid::GridDerivativeProvider& provider,
                  const qcx::grid::XcGridSettings& settings) {
    const qcx::grid::XcIntegrationThresholds thresholds =
        qcx::grid::XcIntegrationThresholds::FromEnergyPath(settings, kScreeningTolerance);

    for (const char* name : {"slater", "pbe", "tau_x"})
    {
        const excgrid::XcFunctional* functional = excgrid::FindFunctional(name);

        if (functional == nullptr)
        {
            std::printf("walk | basis=%s functional=%s status=unknown_functional\n",
                        fixture.label.c_str(),
                        name);
            continue;
        }

        const auto run = [&] {
            return qcx::grid::EvaluateXcGridMotionHessian(fixture.engine.Grid().Blocks(),
                                                          provider,
                                                          fixture.engine.Evaluator(),
                                                          *functional,
                                                          fixture.engine.ShellEnvelopes(),
                                                          thresholds,
                                                          fixture.densityAlpha,
                                                          fixture.densityBeta);
        };

        Arm arm;
        arm.canaryBeforeNs = CanaryNanoseconds();

        {
            // Warm-up: the first traversal is discarded.
            const auto warm = run();
            (void)warm;
        }

        const Clock::time_point from = Clock::now();
        const auto measured = run();
        const Clock::time_point to = Clock::now();

        arm.ns = NanoSeconds(from, to);
        arm.canaryAfterNs = CanaryNanoseconds();
        const double low = std::min(arm.canaryBeforeNs, arm.canaryAfterNs);
        const double high = std::max(arm.canaryBeforeNs, arm.canaryAfterNs);
        arm.canaryWidthPercent = low > 0.0 ? 100.0 * (high - low) / low : 0.0;

        if (!measured.has_value())
        {
            const std::string code(qcx::ToString(measured.error().code));
            std::printf("walk | basis=%s functional=%s status=refused error_code=%s error=%s "
                        "canary_before_ns=%.1f canary_after_ns=%.1f canary_width_pct=%.2f\n",
                        fixture.label.c_str(),
                        name,
                        code.c_str(),
                        measured.error().message.c_str(),
                        arm.canaryBeforeNs,
                        arm.canaryAfterNs,
                        arm.canaryWidthPercent);
            continue;
        }

        gSink = measured->energy + measured->gradient.sum() + measured->hessian.sum();

        const qcx::grid::XcGradientCounts& counts = measured->counts;
        const double visited = static_cast<double>(counts.points);
        const double callsPerPoint =
            counts.points > 0 ? static_cast<double>(counts.kernelSecondDerivativeCalls) / visited
                              : 0.0;

        std::printf("walk | basis=%s functional=%s status=ok grid_points=%zu grid_blocks=%zu "
                    "points=%zu batches=%zu measured_ns=%.1f ns_per_point=%.1f "
                    "ao_third_slots_evaluated=%zu ao_third_slots_dense=%zu "
                    "electronic_third_terms=%zu kernel_second_derivative_calls=%zu "
                    "kernel_calls_per_point=%.3f "
                    "canary_before_ns=%.1f canary_after_ns=%.1f canary_width_pct=%.2f\n",
                    fixture.label.c_str(),
                    name,
                    fixture.engine.PointCount(),
                    fixture.engine.BlockCount(),
                    counts.points,
                    counts.batches,
                    arm.ns,
                    visited > 0.0 ? arm.ns / visited : 0.0,
                    counts.aoThirdSlotsEvaluated,
                    counts.aoThirdSlotsDense,
                    counts.electronicThirdTerms,
                    counts.kernelSecondDerivativeCalls,
                    callsPerPoint,
                    arm.canaryBeforeNs,
                    arm.canaryAfterNs,
                    arm.canaryWidthPercent);
    }
}

/// One point's field values, the quantities the kernel's components read.
struct PointFields {
    std::array<double, 2> rho{};
    std::array<std::array<double, 3>, 2> gradient{};
    std::array<double, 2> tau{};
    double sigmaAa = 0.0;
    double sigmaAb = 0.0;
    double sigmaBb = 0.0;
};

/// The seven collinear components at one point, from the same contractions
/// the energy path makes.
PointFields FieldsAt(const qcx::grid::AoEvaluator& evaluator,
                     const Eigen::MatrixXd& densityAlpha,
                     const Eigen::MatrixXd& densityBeta,
                     const std::array<double, 3>& point) {
    const std::size_t aoCount = evaluator.AOCount();
    std::vector<double> values(aoCount);
    std::vector<double> gradients(3 * aoCount);
    evaluator.EvaluateGradients(point, values, gradients);

    const Eigen::Map<const Eigen::VectorXd> phi(values.data(), static_cast<Eigen::Index>(aoCount));
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> gradient(
        gradients.data(), static_cast<Eigen::Index>(aoCount), 3);

    PointFields fields;

    for (std::size_t spin = 0; spin < 2; ++spin)
    {
        const Eigen::MatrixXd& density = spin == 0 ? densityAlpha : densityBeta;
        const Eigen::VectorXd weightedValues = density * phi;
        const Eigen::MatrixXd weightedGradients = density * gradient;

        fields.rho[spin] = phi.dot(weightedValues);

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            fields.gradient[spin][axis] = 2.0 * gradient.col(axis).dot(weightedValues);
        }

        fields.tau[spin] = 0.5 * (gradient.transpose() * weightedGradients).trace();
    }

    fields.sigmaAa = fields.gradient[0][0] * fields.gradient[0][0] +
                     fields.gradient[0][1] * fields.gradient[0][1] +
                     fields.gradient[0][2] * fields.gradient[0][2];
    fields.sigmaAb = fields.gradient[0][0] * fields.gradient[1][0] +
                     fields.gradient[0][1] * fields.gradient[1][1] +
                     fields.gradient[0][2] * fields.gradient[1][2];
    fields.sigmaBb = fields.gradient[1][0] * fields.gradient[1][0] +
                     fields.gradient[1][1] * fields.gradient[1][1] +
                     fields.gradient[1][2] * fields.gradient[1][2];

    return fields;
}

/// One component's value, by the identifier the kernel reads.
double ComponentValue(excgrid::Component id, const PointFields& fields) {
    switch (id)
    {
    case excgrid::Component::RhoA:
        return fields.rho[0];
    case excgrid::Component::RhoB:
        return fields.rho[1];
    case excgrid::Component::SigmaAa:
        return fields.sigmaAa;
    case excgrid::Component::SigmaAb:
        return fields.sigmaAb;
    case excgrid::Component::SigmaBb:
        return fields.sigmaBb;
    case excgrid::Component::TauA:
        return fields.tau[0];
    case excgrid::Component::TauB:
        return fields.tau[1];
    default:
        return 0.0;
    }
}

/// The input mask is exactly the functional's required set, in identifier
/// order, which is the layout the walk's curvature call uses.
excgrid::PointInputs BuildInputs(const excgrid::ComponentMask& required,
                                 const PointFields& fields) {
    excgrid::PointInputs inputs;

    for (std::size_t component = 0; component <= static_cast<std::size_t>(excgrid::Component::TauB);
         ++component)
    {
        const auto id = static_cast<excgrid::Component>(component);

        if (required.Test(id))
        {
            inputs.Set(id, ComponentValue(id, fields));
        }
    }

    return inputs;
}

/// Measurement 3: the walk's per-point second-derivative call pattern, timed
/// directly, at one polarized point.
void RunKernelPhase(const Fixture& fixture, const qcx::molecule::Molecule& water) {
    const excgrid::Block& block = fixture.engine.Grid().Blocks().front();
    const std::array<double, 3>& point = block.points.front();
    const PointFields fields =
        FieldsAt(fixture.engine.Evaluator(), fixture.densityAlpha, fixture.densityBeta, point);
    const std::size_t coordinates = 3 * water.AtomCount();
    const double pointIterations = static_cast<double>(kKernelPointIterations);

    for (const char* name : {"slater", "pbe", "tau_x"})
    {
        const excgrid::XcFunctional* functional = excgrid::FindFunctional(name);

        if (functional == nullptr)
        {
            std::printf("kernel | basis=%s functional=%s status=unknown_functional\n",
                        fixture.label.c_str(),
                        name);
            continue;
        }

        const excgrid::ComponentMask required = functional->RequiredMask();
        const std::size_t active = required.ActiveCount();

        if (active == 0)
        {
            std::printf("kernel | basis=%s functional=%s status=no_components\n",
                        fixture.label.c_str(),
                        name);
            continue;
        }

        const std::size_t sides =
            std::max<std::size_t>(1, excgrid::kSecondDerivativeCapacity / active);
        const std::size_t callsPerPoint = (coordinates + sides - 1) / sides;
        const excgrid::PointInputs inputs = BuildInputs(required, fields);
        std::array<double, excgrid::kSecondDerivativeCapacity> rightHandSide{};

        for (std::size_t entry = 0; entry < sides * active; ++entry)
        {
            rightHandSide[entry] = 1.0 + 0.25 * static_cast<double>(entry);
        }

        const std::span<const double> rightHandSides(rightHandSide.data(), sides * active);

        // One probe call settles the status; the timed runs below do not check
        // it again, so the loop measures the contraction and nothing else.
        excgrid::PointResult result;
        excgrid::PointSecondDerivative second;
        const excgrid::KernelStatus probe =
            functional->EvaluatePointWithSecondDerivatives(inputs, rightHandSides, result, second);

        if (probe != excgrid::KernelStatus::kOk)
        {
            const std::string status(excgrid::DescribeStatus(probe));
            std::printf("kernel | basis=%s functional=%s active_components=%zu "
                        "status=refused kernel_status=%s\n",
                        fixture.label.c_str(),
                        name,
                        active,
                        status.c_str());
            continue;
        }

        const Arm arm = TimeArm([&] {
            double local = 0.0;

            for (std::size_t iteration = 0; iteration < kKernelPointIterations; ++iteration)
            {
                for (std::size_t call = 0; call < callsPerPoint; ++call)
                {
                    (void)functional->EvaluatePointWithSecondDerivatives(
                        inputs, rightHandSides, result, second);
                    local += second.contracted[0];
                }
            }

            gSink = local + result.exc;
        });

        const double calls = pointIterations * static_cast<double>(callsPerPoint);
        const double nsPerCall = calls > 0.0 ? arm.ns / calls : 0.0;

        std::printf("kernel | basis=%s functional=%s active_components=%zu sides_per_call=%zu "
                    "calls_per_point=%zu iterations=%zu measured_ns=%.1f ns_per_call=%.3f "
                    "ns_per_point=%.1f rho_a=%.6e rho_b=%.6e status=ok "
                    "canary_before_ns=%.1f canary_after_ns=%.1f canary_width_pct=%.2f\n",
                    fixture.label.c_str(),
                    name,
                    active,
                    sides,
                    callsPerPoint,
                    kKernelPointIterations,
                    arm.ns,
                    nsPerCall,
                    nsPerCall * static_cast<double>(callsPerPoint),
                    fields.rho[0],
                    fields.rho[1],
                    arm.canaryBeforeNs,
                    arm.canaryAfterNs,
                    arm.canaryWidthPercent);
    }
}

/// The three basis sets, in the order the rows are printed.
struct Family {
    const char* directory;
    const char* label;
};

} // namespace

int main() {
    std::printf("qcx-bench-xc-hessian-tier | H2O, three basis sets\n");
    std::printf("protocol | every arm runs twice: the first traversal is warm-up and is "
                "discarded, the second is reported; a fixed-work integer canary runs before and "
                "after every arm and its width is printed on the arm's row; the fetch and kernel "
                "arms repeat their traversal inside a run, and the per-traversal figure is the "
                "measured run divided by the repetition count; figures are not comparable across "
                "binaries\n");
    std::printf("fixture | molecule=H2O rOH_bohr=1.8110629 angle_deg=104.45 radial=%zu "
                "angular=%zu trim_weight=1e-15 block_target=%zu\n",
                kRadialPoints,
                kAngularPoints,
                kBlockTarget);
    std::printf("phase_fetch | three selected fetch entry points on a fixed subset of %zu grid "
                "points; a shell is kept when any of its AO values at the point reaches %.1e; the "
                "same selection is reused by all three entry points\n",
                kPointSubsetSize,
                kSelectionFloor);
    std::printf(
        "phase_walk | fixed-density Hessian walk for slater, pbe and tau_x on the engine "
        "grid; screening_tolerance=%.1e; density is rank-one PSD with beta at 0.8 of alpha\n",
        kScreeningTolerance);
    std::printf("phase_kernel | the per-point second-derivative contraction at one polarized "
                "point; sides per call from the %zu-double capacity; the row's ns_per_point is "
                "calls_per_point calls\n",
                excgrid::kSecondDerivativeCapacity);

    auto water = MakeWater();

    if (!water.has_value())
    {
        PrintError("fixture | molecule=H2O", water.error());
        return 1;
    }

    qcx::grid::XcGridSettings settings;
    settings.radialPoints = kRadialPoints;
    settings.angularPoints = kAngularPoints;
    settings.blockTarget = kBlockTarget;

    auto geometry = qcx::grid::ToExcgridGeometry(*water);

    if (!geometry.has_value())
    {
        PrintError("provider", geometry.error());
        return 1;
    }

    auto provider = excgrid::CreateGridDerivativeProvider(
        *geometry, qcx::grid::ToExcgridParams(settings), excgrid::DerivativeOrder::kSecond);

    if (!provider.has_value())
    {
        PrintExcgridError("provider", provider.error());
        return 1;
    }

    constexpr Family kFamilies[] = {
        {"sto-3g", "STO-3G"}, {"def2-svp", "def2-SVP"}, {"def2-tzvp", "def2-TZVP"}};

    std::size_t fixturesBuilt = 0;
    bool kernelPhaseRan = false;

    for (const Family& family : kFamilies)
    {
        std::optional<Fixture> fixture =
            MakeFixture(*water, family.directory, family.label, settings);

        if (!fixture.has_value())
        {
            continue;
        }

        ++fixturesBuilt;
        RunFetchPhase(*fixture);
        RunWalkPhase(*fixture, **provider, settings);

        // The kernel call pattern depends on the component count and the
        // nuclear coordinate count, not on the basis; it runs once, on the
        // first fixture that builds.
        if (!kernelPhaseRan)
        {
            RunKernelPhase(*fixture, *water);
            kernelPhaseRan = true;
        }
    }

    std::printf("sink | %.17g\n", static_cast<double>(gSink));
    std::printf("complete | fixtures=%zu fetch_arms=%zu walk_arms=%zu kernel_arms=%d\n",
                fixturesBuilt,
                3 * fixturesBuilt,
                3 * fixturesBuilt,
                kernelPhaseRan ? 3 : 0);
    return 0;
}
