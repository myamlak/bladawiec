// The orbital-Hessian per-application instrument: the wall time of ONE
// response-operator application on each band - the dense MO tensor and the
// matrix-free direct Fock build - on the same fixture, trial vector and thread
// lane, with the fixed-work canary this tree's timing protocol uses.
//
// THE FIXTURE IS FIXED PER RUN: the tetrahedral C_nH_{2n+2} alkane fixture in
// def2-SVP, the basis whose dense admission the size series turns on. The dense
// band is admitted only while DenseEriTensorEstimateBytes sits under the 2 GiB
// cap (n <= 101 with the all-survive bound); past that the benchmark prints the
// build's own refusal and measures the matrix-free arm alone. That is not a
// fallback - it is the fact the matrix-free band exists for.
//
// THE SERIAL LANE IS THE DEFAULT (--parallel selects the production lane): the
// dense action is serial, and thread scheduling is a second variance term the
// canary would then attribute to the machine. Both arms run the lane the run
// states, so a reading difference is the representation and not the team.
//
// EVERY MEASURED APPLICATION IS WRAPPED IN THE FIXED-WORK INTEGER SPIN from
// benchmarks/boys_option_cost_windows.cpp: one dependent xorshift step per
// round, no memory beyond the accumulator, no floating point, so the reading's
// cost cannot move with the work being measured. The tool reports the raw spin
// times instead of a verdict - a single process cannot know the session's quiet
// floor; the quiet-window protocol takes the floor across the arms and reports
// the peak excess as the run's canary.
//
// The first --warmup applications of each arm are not measured: the first call
// of a process pays page faults and builder warm-up that later calls do not.
// Each reported figure is one application's steady_clock interval; the
// comparison reads the minimum, and the printed worst difference between the
// two arms' results is the checksum that proves both computed the same action
// to the printed precision.
//
// --exact puts both arms in the exact configuration - kTight with every screen
// off - which is the one the bands' agreement is stated under. Without it both
// arms run the production screening, and their difference carries the screening
// error as well as the arithmetic; the timing reading is taken in that default
// mode, because that is the cost a gradient pays.
//
// The two arms are the two public factories the gradient's BuildResponseState
// uses: the dense operator is created over the MO tensor the dense-band state
// built, and the matrix-free one over a MatrixFreeEriDerivative constructed with
// the same reference, core and accuracy settings.
//
// Usage:
//   qcx-bench-response-matvec [--carbons N] [--basis NAME] [--iterations N]
//                             [--warmup N] [--parallel] [--skip-dense] [--exact]
//                             [--json]
//
// --json writes the machine record to stdout and the human lines to stderr;
// without it the human summary goes to stdout.

#include "alkane_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/properties/gradient.hpp"
#include "qcx/response/matrix_free_eri_derivative.hpp"
#include "qcx/response/response_operator.hpp"
#include "qcx/response/response_state.hpp"
#include "qcx/scf/gradient_reference.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

/// Iterations of the canary spin. Fixed, because a reading is compared against
/// the best run of the same work and the work must not vary between them.
constexpr std::uint32_t kCanaryRounds = 1u << 23;

/// The fixed-work spin: a xorshift chain, one dependent step per iteration, no
/// memory beyond the accumulator and no floating point.
double CanaryMilliseconds() {
    volatile std::uint64_t sink = 0;
    const auto from = std::chrono::steady_clock::now();
    std::uint64_t state = 0x243F6A8885A308D3ull;

    for (std::uint32_t i = 0; i < kCanaryRounds; ++i)
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
    }

    sink = state;
    const auto to = std::chrono::steady_clock::now();
    (void)sink;

    return std::chrono::duration<double, std::milli>(to - from).count();
}

/// Row-major copy of an Eigen matrix, the layout the response module reads.
std::vector<double> RowMajor(const Eigen::MatrixXd& matrix) {
    std::vector<double> values(static_cast<std::size_t>(matrix.rows() * matrix.cols()));

    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
    {
        for (Eigen::Index column = 0; column < matrix.cols(); ++column)
        {
            values[static_cast<std::size_t>(row * matrix.cols() + column)] = matrix(row, column);
        }
    }

    return values;
}

/// The deterministic trial vector both arms are applied to: a fixed
/// non-symmetric pattern, so the action is a genuine contraction and the two
/// arms' results are comparable element by element.
std::vector<double> TrialVector(std::size_t dimension) {
    std::vector<double> values(dimension);

    for (std::size_t index = 0; index < dimension; ++index)
    {
        const double position = static_cast<double>(index) + 1.0;
        values[index] = std::sin(0.37 * position) + 0.1 * std::cos(0.11 * position);
    }

    return values;
}

void PrintDoubleArray(const char* key, const std::vector<double>& values) {
    std::printf("\"%s\": [", key);

    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
        {
            std::printf(", ");
        }

        std::printf("%.6f", values[i]);
    }

    std::printf("]");
}

/// The converged reference the two operators are built against.
struct Reference {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    Eigen::MatrixXd core;
    qcx::scf::ScfGradientReference state;
};

/// Runs the SCF on the alkane fixture: direct Fock builder, tight rung of the
/// convergence ladder (the response reads the converged orbitals, so the
/// reference's own residue must sit well below the agreement it is used to
/// check).
bool BuildReference(std::size_t carbons,
                    const std::string& basisName,
                    bool serial,
                    std::unique_ptr<Reference>& out) {
    auto molecule = qcx::testing::MakeAlkaneSto3g(carbons);

    if (!molecule.has_value())
    {
        std::fprintf(stderr, "molecule: %s\n", molecule.error().message.c_str());
        return false;
    }

    const std::string basisDirectory = std::string(QcxBasisDataDir) + "/" + basisName;
    auto basis = qcx::basisset::ParseNwchemDirectory(basisDirectory);

    if (!basis.has_value())
    {
        std::fprintf(
            stderr, "basis %s: %s\n", basisDirectory.c_str(), basis.error().message.c_str());
        return false;
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);

    if (!overlap.has_value() || !kinetic.has_value() || !nuclear.has_value())
    {
        std::fprintf(stderr, "one-electron integrals failed\n");
        return false;
    }

    const Eigen::MatrixXd core = ToMatrix(*kinetic) + ToMatrix(*nuclear);
    auto coreTensor = ToTensor(core);

    if (!coreTensor.has_value())
    {
        std::fprintf(stderr, "core tensor: %s\n", coreTensor.error().message.c_str());
        return false;
    }

    qcx::integrals::FockBuildOptions buildOptions;
    buildOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;

    if (serial)
    {
        buildOptions.maxParallelChunks = 1;
    }

    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *coreTensor, buildOptions);

    if (!builder.has_value())
    {
        std::fprintf(stderr, "Fock builder: %s\n", builder.error().message.c_str());
        return false;
    }

    qcx::scf::RhfOptions scfOptions;
    scfOptions.maxIterations = 300;
    scfOptions.energyTolerance = 1e-10;
    scfOptions.densityTolerance = 1e-8;

    auto result = qcx::scf::RunRhfScf(
        *molecule,
        ToMatrix(*overlap),
        core,
        scfOptions,
        [&builder](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
            // The builder contracts the spatial density rho = D / 2.
            auto spatial = ToTensor(0.5 * density);

            if (!spatial.has_value())
            {
                return std::unexpected(spatial.error());
            }

            auto fock = builder->BuildFock(*spatial);

            if (!fock.has_value())
            {
                return std::unexpected(fock.error());
            }

            return ToMatrix(*fock);
        },
        &*basis);

    if (!result.has_value())
    {
        std::fprintf(stderr, "SCF: %s\n", result.error().message.c_str());
        return false;
    }

    if (!result->converged)
    {
        std::fprintf(stderr, "SCF did not converge in %d iterations\n", result->iterations);
        return false;
    }

    const std::size_t electrons = static_cast<std::size_t>((*molecule).ElectronCount());

    if (electrons % 2 != 0)
    {
        std::fprintf(stderr, "the fixture's electron count is odd (%zu)\n", electrons);
        return false;
    }

    auto reference = qcx::scf::ExtractGradientReference(*result, electrons / 2);

    if (!reference.has_value())
    {
        std::fprintf(stderr, "reference: %s\n", reference.error().message.c_str());
        return false;
    }

    out = std::make_unique<Reference>(
        Reference{std::move(*molecule), std::move(*basis), core, std::move(*reference)});
    return true;
}

/// One arm's measured applications.
struct Arm {
    std::vector<double> perApplyMs;
    std::vector<double> canaryBeforeMs;
    std::vector<double> canaryAfterMs;
    double checksum = 0.0;
    double worstDifference = 0.0; ///< Against the other arm's result; 0 when alone.
};

} // namespace

int main(int argc, char** argv) {
    int carbons = 3;
    std::string basisName = "def2-svp";
    int iterations = 21;
    int warmup = 3;
    bool serial = true;
    bool skipDense = false;
    bool exact = false;
    bool json = false;

    for (int a = 1; a < argc; ++a)
    {
        const std::string arg(argv[a]);

        if (arg == "--parallel")
        {
            serial = false;
            continue;
        }

        if (arg == "--skip-dense")
        {
            skipDense = true;
            continue;
        }

        if (arg == "--exact")
        {
            exact = true;
            continue;
        }

        if (arg == "--json")
        {
            json = true;
            continue;
        }

        if (arg == "--carbons" && a + 1 < argc)
        {
            carbons = std::atoi(argv[++a]);
            continue;
        }

        if (arg == "--basis" && a + 1 < argc)
        {
            basisName = argv[++a];
            continue;
        }

        if (arg == "--iterations" && a + 1 < argc)
        {
            iterations = std::atoi(argv[++a]);
            continue;
        }

        if (arg == "--warmup" && a + 1 < argc)
        {
            warmup = std::atoi(argv[++a]);
            continue;
        }

        std::fprintf(stderr,
                     "usage: qcx-bench-response-matvec [--carbons N] [--basis NAME] "
                     "[--iterations N] [--warmup N] [--parallel] [--skip-dense] [--exact] "
                     "[--json]\n");
        return 2;
    }

    if (carbons < 1 || iterations < 1 || warmup < 0)
    {
        std::fprintf(stderr,
                     "qcx-bench-response-matvec: carbons must be >= 1, iterations >= 1 and "
                     "warmup >= 0\n");
        return 2;
    }

    std::unique_ptr<Reference> reference;

    if (!BuildReference(static_cast<std::size_t>(carbons), basisName, serial, reference))
    {
        return 1;
    }

    const std::size_t n = reference->state.numBasisFunctions;
    const std::size_t occupied = reference->state.numOccupied;
    const std::size_t virtualOrbitals = reference->state.numVirtual;
    const std::size_t dimension = occupied * virtualOrbitals;

    auto pairList = qcx::integrals::BuildShellPairs(reference->molecule, reference->basis);

    if (!pairList.has_value())
    {
        std::fprintf(stderr, "pair list: %s\n", pairList.error().message.c_str());
        return 1;
    }

    const std::size_t denseEstimate =
        qcx::integrals::DenseEriTensorEstimateBytes(n, pairList->pairs.size());
    const std::size_t denseCap = qcx::integrals::kDenseEriTensorByteCap;
    const bool denseWithinCap = denseEstimate <= denseCap;

    // The dense arm: the MO tensor the dense-band state builds, and the
    // operator over it. Past the cap the build refuses by name and the arm is
    // absent - the measured fact the matrix-free band exists for.
    std::optional<qcx::response::ResponseState> denseState;
    std::optional<qcx::response::OrbitalHessianOperator> denseOperator;
    std::string denseRefusal;

    if (!skipDense && denseWithinCap)
    {
        qcx::integrals::EriDenseOptions denseOptions;
        denseOptions.accuracy = exact ? qcx::integrals::AccuracyPreset::kTight
                                      : qcx::integrals::AccuracyPreset::kNormal;
        denseOptions.screen = !exact;

        auto built =
            qcx::properties::BuildResponseState(reference->molecule,
                                                reference->basis,
                                                reference->state,
                                                denseOptions,
                                                {},
                                                std::nullopt,
                                                qcx::response::ResponseOperatorBand::kDenseTensor);

        if (!built.has_value())
        {
            denseRefusal = built.error().message;
        } else
        {
            denseState.emplace(std::move(*built));
            auto op = qcx::response::OrbitalHessianOperator::Create(
                denseState->Layout(),
                qcx::response::OrbitalEnergies{denseState->OrbitalEnergies()},
                qcx::response::MoTwoElectronTensor{denseState->MoTwoElectron()});

            if (!op.has_value())
            {
                std::fprintf(stderr, "dense operator: %s\n", op.error().message.c_str());
                return 1;
            }

            denseOperator.emplace(std::move(*op));
        }
    } else if (!skipDense)
    {
        denseRefusal = "the a-priori estimate exceeds the admission cap (" +
                       std::to_string(denseEstimate) + " B against " + std::to_string(denseCap) +
                       " B)";
    }

    // The matrix-free arm: the derivative the gradient's matrix-free band
    // builds, with the run's thread lane.
    qcx::response::MatrixFreeEriDerivativeOptions derivativeOptions;
    derivativeOptions.accuracy =
        exact ? qcx::integrals::AccuracyPreset::kTight : qcx::integrals::AccuracyPreset::kNormal;
    derivativeOptions.useDensityScreening = !exact;
    derivativeOptions.usePerElementScreening = !exact;

    if (serial)
    {
        derivativeOptions.maxParallelChunks = 1;
    }

    const std::vector<double> coreRowMajor = RowMajor(reference->core);
    const std::vector<double> coefficients = RowMajor(reference->state.coefficients);
    const std::vector<double> energies(reference->state.orbitalEnergies.data(),
                                       reference->state.orbitalEnergies.data() +
                                           reference->state.orbitalEnergies.size());
    const qcx::response::ResponseLayout layout{occupied, virtualOrbitals};

    auto derivative = qcx::response::MatrixFreeEriDerivative::Create(reference->molecule,
                                                                     reference->basis,
                                                                     coreRowMajor,
                                                                     layout,
                                                                     n,
                                                                     coefficients,
                                                                     1.0,
                                                                     derivativeOptions);

    if (!derivative.has_value())
    {
        std::fprintf(stderr, "matrix-free derivative: %s\n", derivative.error().message.c_str());
        return 1;
    }

    auto derivativePointer =
        std::make_shared<qcx::response::MatrixFreeEriDerivative>(std::move(*derivative));

    auto freeOperator = qcx::response::OrbitalHessianOperator::CreateMatrixFree(
        layout,
        qcx::response::OrbitalEnergies{std::span<const double>(energies)},
        [derivativePointer](std::span<const double> x, std::span<double> y) -> qcx::Result<void> {
            return derivativePointer->Apply(x, y);
        },
        1.0);

    if (!freeOperator.has_value())
    {
        std::fprintf(stderr, "matrix-free operator: %s\n", freeOperator.error().message.c_str());
        return 1;
    }

    const std::vector<double> trial = TrialVector(dimension);
    std::vector<double> denseResult(dimension, 0.0);
    std::vector<double> freeResult(dimension, 0.0);

    auto applyDense = [&]() -> bool {
        std::fill(denseResult.begin(), denseResult.end(), 0.0);
        auto status = denseOperator->Apply(trial, denseResult);

        if (!status.has_value())
        {
            std::fprintf(stderr, "dense apply: %s\n", status.error().message.c_str());
            return false;
        }

        return true;
    };

    auto applyFree = [&]() -> bool {
        std::fill(freeResult.begin(), freeResult.end(), 0.0);
        auto status = freeOperator->Apply(trial, freeResult);

        if (!status.has_value())
        {
            std::fprintf(stderr, "matrix-free apply: %s\n", status.error().message.c_str());
            return false;
        }

        return true;
    };

    if (denseOperator.has_value() && !applyDense())
    {
        return 1;
    }

    if (!applyFree())
    {
        return 1;
    }

    double worstDifference = 0.0;

    if (denseOperator.has_value())
    {
        for (std::size_t index = 0; index < dimension; ++index)
        {
            worstDifference =
                std::max(worstDifference, std::fabs(denseResult[index] - freeResult[index]));
        }
    }

    for (int i = 0; i < warmup; ++i)
    {
        if (denseOperator.has_value() && !applyDense())
        {
            return 1;
        }

        if (!applyFree())
        {
            return 1;
        }
    }

    Arm denseArm;
    Arm freeArm;

    // Interleaved per iteration: the two arms see the same machine over the
    // same window, so a load that arrives mid-run lands on both.
    for (int i = 0; i < iterations; ++i)
    {
        if (denseOperator.has_value())
        {
            denseArm.canaryBeforeMs.push_back(CanaryMilliseconds());
            const auto start = std::chrono::steady_clock::now();

            if (!applyDense())
            {
                return 1;
            }

            const auto stop = std::chrono::steady_clock::now();
            denseArm.canaryAfterMs.push_back(CanaryMilliseconds());
            denseArm.perApplyMs.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count());
            double sum = 0.0;

            for (const double value : denseResult)
            {
                sum += value;
            }

            denseArm.checksum = sum;
        }

        freeArm.canaryBeforeMs.push_back(CanaryMilliseconds());
        const auto start = std::chrono::steady_clock::now();

        if (!applyFree())
        {
            return 1;
        }

        const auto stop = std::chrono::steady_clock::now();
        freeArm.canaryAfterMs.push_back(CanaryMilliseconds());
        freeArm.perApplyMs.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
        double sum = 0.0;

        for (const double value : freeResult)
        {
            sum += value;
        }

        freeArm.checksum = sum;
    }

    denseArm.worstDifference = worstDifference;
    freeArm.worstDifference = worstDifference;

    const auto summarize = [](const Arm& arm) {
        if (arm.perApplyMs.empty())
        {
            return std::pair<double, double>{0.0, 0.0};
        }

        double total = 0.0;
        double minimum = arm.perApplyMs.front();

        for (const double ms : arm.perApplyMs)
        {
            total += ms;

            if (ms < minimum)
            {
                minimum = ms;
            }
        }

        return std::pair<double, double>{total / static_cast<double>(arm.perApplyMs.size()),
                                         minimum};
    };

    const auto [denseMean, denseMin] = summarize(denseArm);
    const auto [freeMean, freeMin] = summarize(freeArm);

    // The refusal text is evidence, so it rides the record; it is written as a
    // JSON string, so a quote in the engine's message is replaced rather than
    // allowed to break the document.
    std::string refusalJson = denseRefusal;

    for (char& character : refusalJson)
    {
        if (character == '"' || character == '\\' || character == '\n')
        {
            character = '\'';
        }
    }

    if (json)
    {
        std::printf("{\"schema\": \"qcx-response-matvec/1\", ");
        std::printf("\"fixture\": \"C%zuH%zu/%s\", \"carbons\": %d, ",
                    static_cast<std::size_t>(carbons),
                    2 * static_cast<std::size_t>(carbons) + 2,
                    basisName.c_str(),
                    carbons);
        std::printf("\"basis_functions\": %zu, \"occupied\": %zu, \"virtual\": %zu, ",
                    n,
                    occupied,
                    virtualOrbitals);
        std::printf("\"dimension\": %zu, \"serial\": %s, \"exact\": %s, \"warmup\": %d, "
                    "\"iterations\": %d, ",
                    dimension,
                    serial ? "true" : "false",
                    exact ? "true" : "false",
                    warmup,
                    iterations);
        std::printf("\"dense_admitted\": %s, \"dense_estimate_bytes\": %zu, "
                    "\"dense_cap_bytes\": %zu, ",
                    denseOperator.has_value() ? "true" : "false",
                    denseEstimate,
                    denseCap);
        std::printf("\"dense_refusal\": \"%s\", ", refusalJson.c_str());
        std::printf("\"agreement_worst_difference\": %.17g, ", worstDifference);
        std::printf("\"dense_checksum\": %.17g, \"free_checksum\": %.17g, ",
                    denseArm.checksum,
                    freeArm.checksum);
        std::printf("\"canary_rounds\": %u, ", kCanaryRounds);
        PrintDoubleArray("dense_per_apply_ms", denseArm.perApplyMs);
        std::printf(", ");
        PrintDoubleArray("dense_canary_before_ms", denseArm.canaryBeforeMs);
        std::printf(", ");
        PrintDoubleArray("dense_canary_after_ms", denseArm.canaryAfterMs);
        std::printf(", ");
        PrintDoubleArray("free_per_apply_ms", freeArm.perApplyMs);
        std::printf(", ");
        PrintDoubleArray("free_canary_before_ms", freeArm.canaryBeforeMs);
        std::printf(", ");
        PrintDoubleArray("free_canary_after_ms", freeArm.canaryAfterMs);
        std::printf("}\n");
    } else
    {
        std::printf("fixture=C%zuH%zu/%s n=%zu occupied=%zu virtual=%zu dimension=%zu "
                    "serial=%s exact=%s warmup=%d iterations=%d\n",
                    static_cast<std::size_t>(carbons),
                    2 * static_cast<std::size_t>(carbons) + 2,
                    basisName.c_str(),
                    n,
                    occupied,
                    virtualOrbitals,
                    dimension,
                    serial ? "yes" : "no",
                    exact ? "yes" : "no",
                    warmup,
                    iterations);
        std::printf("dense_admitted=%s dense_estimate_bytes=%zu dense_cap_bytes=%zu\n",
                    denseOperator.has_value() ? "yes" : "no",
                    denseEstimate,
                    denseCap);

        if (!denseRefusal.empty())
        {
            std::printf("dense_refusal=%s\n", denseRefusal.c_str());
        }

        std::printf("agreement_worst_difference=%.17g\n", worstDifference);
    }

    if (denseOperator.has_value())
    {
        std::fprintf(stderr,
                     "qcx-bench-response-matvec: C%zuH%zu/%s n=%zu dim=%zu %s, "
                     "dense mean %.3f ms min %.3f ms, matrix-free mean %.3f ms min %.3f ms, "
                     "worst difference %.3g\n",
                     static_cast<std::size_t>(carbons),
                     2 * static_cast<std::size_t>(carbons) + 2,
                     basisName.c_str(),
                     n,
                     dimension,
                     serial ? "serial" : "parallel",
                     denseMean,
                     denseMin,
                     freeMean,
                     freeMin,
                     worstDifference);
    } else
    {
        std::fprintf(stderr,
                     "qcx-bench-response-matvec: C%zuH%zu/%s n=%zu dim=%zu %s, "
                     "dense refused, matrix-free mean %.3f ms min %.3f ms\n",
                     static_cast<std::size_t>(carbons),
                     2 * static_cast<std::size_t>(carbons) + 2,
                     basisName.c_str(),
                     n,
                     dimension,
                     serial ? "serial" : "parallel",
                     freeMean,
                     freeMin);
    }

    return 0;
}
