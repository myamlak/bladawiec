#pragma once

// The shared instrument of the auxiliary-pair density screen's cells: the SCF
// driver, the unscreened reference built from a builder's own retained tensor,
// the adapters and the per-preset budget row. Test-only helper - not part of
// the public API.
//
// It lives in its own header because the two cell files cannot be one: the
// basis fixtures define `qcx::testing::kSto3gCarbon` TWICE (benzene_sto3g.hpp
// and alkane_sto3g.hpp each carry the vendored STO-3G carbon text), so a
// translation unit including both fails to compile. That is a property of the
// fixtures, not of the cells; the split below is the consequence, and the
// carbon texts are not duplicated here to route around it.
//
// The SCF driver is the ri_occ_k_test.cpp / ri_jk_validation_test.cpp
// instrument, copied for the same reason those two copies are deliberate: the
// instrument belongs to landed measurements (the -74.96292827 anchor and the
// contraction suite's iteration counts), and extracting it into a shared header
// would move the instrument of measurements already taken. The copies are held
// together by re-anchoring the same in-tree pin, which is the drift detector.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_full_fock.hpp"
#include "qcx/integrals/ri_occ_k.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

namespace qcx::testing::riAuxScreen {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

/// The auxiliary basis of every cell: the J+K fit, pinned by name and recorded
/// beside every number (the ri_jk cells' rule - the error is the fixture's
/// AUXILIARY FIT, and a number without its aux has no class).
inline constexpr std::string_view kAuxName = "def2-universal-jkfit";

/// The density screen's acceptance bar: 0.01 kcal/mol = 1.59e-5 Eh,
/// screened against unscreened RI-K.
inline constexpr double kScreenTolerance = 1.59e-5;

/// kcal/mol per Hartree (the conversion behind the 0.01 kcal/mol bar).
inline constexpr double kCaloriesPerHartree = 627.5094740631;

/// The ceiling on the screen's own converged-energy move: 1e-6 Eh, an order
/// below the RECORDED auxiliary-fit error class on these fixtures
/// (8.515e-5 Eh on water/def2-SVP and 3.524e-4 Eh on H2O/STO-3G - the accuracy
/// cells of the validation suite, aux def2-universal-jkfit). A screen whose own
/// truncation reached that class would no longer be sub-dominant to the
/// approximation it screens, which is exactly the point (a non-variational
/// truncation does not shrink with the RI error). A CLASS bound with its
/// provenance, not a pin.
inline constexpr double kScreenTruncationClass = 1e-6;

/// The SCF fixture's convergence threshold: the operating setting (1e-8).
inline constexpr double kFixtureEnergyTolerance = 1e-8;

/// The per-preset J/K energy target the screen's truncation must stay inside
/// (accuracy.hpp's AccuracyPreset documentation: kLoose 1e-6, kNormal 1e-10,
/// kTight 1e-12 Eh). Carried here rather than read, because accuracy.hpp has
/// no mapper for the ENERGY target - it maps the Schwarz, density and
/// mixed-precision THRESHOLDS. If one is added, this is the site that switches
/// to it, and the switch is the whole change.
constexpr double PresetEnergyTarget(qcx::integrals::AccuracyPreset preset) {
    switch (preset)
    {
    case qcx::integrals::AccuracyPreset::kLoose:
        return 1e-6;
    case qcx::integrals::AccuracyPreset::kNormal:
        return 1e-10;
    case qcx::integrals::AccuracyPreset::kTight:
        return 1e-12;
    }

    return 1e-10;
}

constexpr std::string_view PresetWord(qcx::integrals::AccuracyPreset preset) {
    switch (preset)
    {
    case qcx::integrals::AccuracyPreset::kLoose:
        return "kLoose";
    case qcx::integrals::AccuracyPreset::kNormal:
        return "kNormal";
    case qcx::integrals::AccuracyPreset::kTight:
        return "kTight";
    }

    return "kNormal";
}

/// Parses a fixture's elements out of a vendored basis directory - the
/// filtered parse the engine's own molecule-scoped contract uses. The element
/// list is the caller's: the water fixture takes {H, O}, the carbon ones
/// {H, C}.
inline qcx::Result<qcx::basisset::BasisSet> ParseFixtureBasis(std::string_view directory,
                                                              const std::array<int, 2>& elements) {
    const std::filesystem::path root(QcxBasisDataDir);
    return qcx::basisset::ParseNwchemDirectoryFiltered((root / directory).string(), elements);
}

inline qcx::Result<Eigen::MatrixXd> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
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

    return ToMatrix(*kinetic) + ToMatrix(*nuclear);
}

/// The UNScreened half of the composition, built from a builder's own retained
/// tensor through the contraction entry points (ri_occ_k.hpp) - the shape the
/// composed builder ran before the screen landed. It is the control every
/// screened number is taken against: same B, same density, same occupied block,
/// one build without the screen.
inline qcx::Result<Eigen::MatrixXd> UnscreenedFockFromTransformed(
    const Eigen::MatrixXd& transformed,
    const Eigen::MatrixXd& core,
    const Eigen::MatrixXd& density,
    const Eigen::MatrixXd& occupiedOrbitals) {
    const Eigen::Index n = density.rows();
    const std::size_t occupiedCount = static_cast<std::size_t>(occupiedOrbitals.cols());
    Eigen::VectorXd densityVector(n * n);

    for (Eigen::Index u = 0; u < n; ++u)
    {
        for (Eigen::Index v = 0; v < n; ++v)
        {
            densityVector(u * n + v) = density(u, v);
        }
    }

    const Eigen::VectorXd weights = transformed.transpose() * densityVector;
    const Eigen::VectorXd coulombVector = transformed * weights;
    Eigen::MatrixXd coulomb(n, n);

    for (Eigen::Index u = 0; u < n; ++u)
    {
        for (Eigen::Index v = 0; v < n; ++v)
        {
            coulomb(u, v) = coulombVector(u * n + v);
        }
    }

    auto occTransformed =
        qcx::integrals::TransformToOccupiedOrbitals(transformed, occupiedOrbitals);

    if (!occTransformed.has_value())
    {
        return std::unexpected(occTransformed.error());
    }

    auto exchange = qcx::integrals::BuildRiExchangeMatrix(*occTransformed, occupiedCount);

    if (!exchange.has_value())
    {
        return std::unexpected(exchange.error());
    }

    return core + 2.0 * coulomb - *exchange;
}

/// The fixture RHF driver: the closed-shell fixed-point SCF in the spatial
/// density convention the builders use (rho = the spin-summed D over 2), from
/// the core-Hamiltonian guess, one Fock build per iteration.
struct RhfOutcome {
    double energy = 0.0;
    double deltaEnergy = 0.0;
    Eigen::MatrixXd density;
    Eigen::MatrixXd orbitals;
    std::size_t iterations = 0;
    bool converged = false;
};

template <typename FockFn>
qcx::Result<RhfOutcome> RunRhf(const Eigen::MatrixXd& core,
                               const Eigen::MatrixXd& overlap,
                               std::size_t occupiedCount,
                               const FockFn& fock,
                               double energyTolerance = kFixtureEnergyTolerance,
                               std::size_t maxIterations = 100) {
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> overlapSolver(overlap);

    if (overlapSolver.info() != Eigen::Success ||
        !(overlapSolver.eigenvalues().array() > 0.0).all())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the overlap matrix is not positive definite"});
    }

    const Eigen::VectorXd inverseRoots = overlapSolver.eigenvalues().cwiseSqrt().cwiseInverse();
    const Eigen::MatrixXd orthonormalizer = overlapSolver.eigenvectors() *
                                            inverseRoots.asDiagonal() *
                                            overlapSolver.eigenvectors().transpose();

    RhfOutcome outcome;
    Eigen::MatrixXd fockMatrix = core;
    double previousEnergy = 0.0;

    for (std::size_t iteration = 1; iteration <= maxIterations; ++iteration)
    {
        const Eigen::MatrixXd orthogonalized =
            orthonormalizer.transpose() * fockMatrix * orthonormalizer;
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> orbitalSolver(orthogonalized);

        if (orbitalSolver.info() != Eigen::Success)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "the Fock matrix failed its eigendecomposition"});
        }

        const Eigen::MatrixXd coefficients = orthonormalizer * orbitalSolver.eigenvectors();
        const Eigen::MatrixXd occupied =
            coefficients.leftCols(static_cast<Eigen::Index>(occupiedCount));
        const Eigen::MatrixXd density = occupied * occupied.transpose();
        auto fockResult = fock(density, occupied);

        if (!fockResult.has_value())
        {
            return std::unexpected(fockResult.error());
        }

        fockMatrix = std::move(*fockResult);
        const double energy = (density * (core + fockMatrix)).trace();
        outcome.density = density;
        outcome.orbitals = occupied;
        outcome.energy = energy;
        outcome.iterations = iteration;

        if (iteration > 1)
        {
            outcome.deltaEnergy = std::abs(energy - previousEnergy);

            if (outcome.deltaEnergy < energyTolerance)
            {
                outcome.converged = true;
                break;
            }
        }

        previousEnergy = energy;
    }

    return outcome;
}

inline void RecordMeasurement(const std::string& key, double value) {
    // ::testing, not testing: this header's own namespace is qcx::testing, so
    // the unqualified name resolves to it and not to googletest's.
    ::testing::Test::RecordProperty(key, value);
}

inline void RecordText(const std::string& key, std::string_view value) {
    ::testing::Test::RecordProperty(key, std::string(value));
}

/// The composed builder's adapter over the SCF instrument's (density,
/// orbitals) pair: RiFullFockBuilder needs both from the same iterate (its
/// coherence contract), which is why this is not the density-only shape the
/// direct builder uses. The sink is the caller's, so the LAST call's screen
/// record is the converged iterate's.
template <typename BuilderResult>
auto ScreenedFockFn(const BuilderResult& builder, qcx::integrals::RiAuxScreenStats& stats) {
    return [&builder, &stats](const Eigen::MatrixXd& density,
                              const Eigen::MatrixXd& occupied) -> qcx::Result<Eigen::MatrixXd> {
        auto densityTensor = ToTensor(density);

        if (!densityTensor.has_value())
        {
            return std::unexpected(densityTensor.error());
        }

        auto fock = builder->BuildFock(*densityTensor, occupied, nullptr, &stats);

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        return ToMatrix(*fock);
    };
}

/// The unscreened reference's adapter: the same builder's retained tensor
/// through the contraction entry points, one build with no screen.
inline auto UnscreenedFockFn(const Eigen::MatrixXd& transformed, const Eigen::MatrixXd& core) {
    return [&transformed, &core](const Eigen::MatrixXd& density,
                                 const Eigen::MatrixXd& occupied) -> qcx::Result<Eigen::MatrixXd> {
        return UnscreenedFockFromTransformed(transformed, core, density, occupied);
    };
}

/// One fixture-preset row of the budget cell.
struct BudgetOutcome {
    double energy = 0.0;
    double move = 0.0;
    double droppedFraction = 0.0;
    std::size_t iterations = 0;
    std::size_t droppedCells = 0;
    /// The two SCF runs' wall times (single run, contended - a DECISION INPUT
    /// for "does the screen pay", never an acceptance statistic).
    long long screenedNanoseconds = 0;
    long long unscreenedNanoseconds = 0;
    std::size_t unscreenedIterations = 0;
};

/// Runs one (fixture, preset) row: the screened SCF through the composed
/// builder and the unscreened SCF through the SAME builder's retained tensor,
/// the same instrument and the same threshold, and records both.
inline qcx::Result<BudgetOutcome> MeasureBudgetCell(const qcx::molecule::Molecule& molecule,
                                                    const qcx::basisset::BasisSet& basis,
                                                    const qcx::basisset::BasisSet& aux,
                                                    const CpuTensor2& coreTensor,
                                                    const Eigen::MatrixXd& core,
                                                    const Eigen::MatrixXd& overlap,
                                                    std::size_t occupiedCount,
                                                    qcx::integrals::AccuracyPreset preset,
                                                    std::string_view label) {
    qcx::integrals::RiEngineOptions engineOptions;
    engineOptions.accuracy = preset;
    qcx::integrals::RiAuxScreenOptions screenOptions;
    // The threshold this cell screens at: the preset's own density threshold -
    // the same three factors (density weight, orbital pair bound, aux shell
    // bound) the 3-center task grid is built at, with the density weight the
    // screen adds. No value is pinned here.
    screenOptions.threshold = qcx::integrals::DensityThreshold(preset);
    auto builder = qcx::integrals::RiFullFockBuilder::Create(
        molecule, basis, aux, coreTensor, engineOptions, screenOptions);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    qcx::integrals::RiAuxScreenStats stats;
    // The two SCF runs are timed, and the timing is a DECISION INPUT, not an
    // acceptance statistic: it is a single run on a machine with other work
    // and compilers live, so it is labelled as such wherever it is quoted, and
    // the paired-interleaved protocol at the pre-push block is what may turn it
    // into a claim. The two sides take the same inputs from the same builder -
    // one tensor, one density, one instrument, the same SCF threshold - so a
    // difference between them is the screen's, not a workload difference.
    const auto screenedStart = std::chrono::steady_clock::now();
    auto screened = RunRhf(core, overlap, occupiedCount, ScreenedFockFn(builder, stats));
    const auto screenedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - screenedStart)
                                .count();

    if (!screened.has_value())
    {
        return std::unexpected(screened.error());
    }

    if (!screened->converged)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "the screened SCF did not converge on this fixture"});
    }

    const auto unscreenedStart = std::chrono::steady_clock::now();
    auto unscreened = RunRhf(
        core, overlap, occupiedCount, UnscreenedFockFn(builder->MetricTransformedTensor(), core));
    const auto unscreenedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - unscreenedStart)
                                  .count();

    if (!unscreened.has_value())
    {
        return std::unexpected(unscreened.error());
    }

    if (!unscreened->converged)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "the unscreened SCF did not converge on this fixture"});
    }

    BudgetOutcome outcome;
    outcome.energy = screened->energy;
    outcome.move = std::abs(screened->energy - unscreened->energy);
    outcome.droppedFraction = stats.droppedFraction;
    outcome.droppedCells = stats.droppedCells;
    outcome.iterations = screened->iterations;
    outcome.screenedNanoseconds = screenedNs;
    outcome.unscreenedNanoseconds = unscreenedNs;
    outcome.unscreenedIterations = unscreened->iterations;

    const double target = PresetEnergyTarget(preset);
    const std::string row = std::string(label) + "." + std::string(PresetWord(preset));
    std::cout << "[ri_jk screen] " << label << " preset " << PresetWord(preset) << " threshold "
              << qcx::integrals::DensityThreshold(preset) << " (aux " << kAuxName << "): dropped "
              << outcome.droppedCells << " cells (" << outcome.droppedFraction
              << "), converged-energy move " << outcome.move << " Eh against the preset target "
              << target << " (ratio " << (outcome.move / target) << "), " << outcome.iterations
              << " iterations; single-run contended SCF wall: screened "
              << (outcome.screenedNanoseconds / 1000000) << " ms / " << outcome.iterations
              << " it, unscreened " << (outcome.unscreenedNanoseconds / 1000000) << " ms / "
              << outcome.unscreenedIterations << " it\n";

    RecordText(row + ".aux_basis", kAuxName);
    RecordMeasurement(row + ".screen_threshold", qcx::integrals::DensityThreshold(preset));
    RecordMeasurement(row + ".dropped_cells", static_cast<double>(outcome.droppedCells));
    RecordMeasurement(row + ".dropped_fraction", outcome.droppedFraction);
    RecordMeasurement(row + ".energy_move", outcome.move);
    RecordMeasurement(row + ".preset_energy_target", target);
    RecordMeasurement(row + ".energy_move_ratio", outcome.move / target);
    // Labelled single-run and contended in the key itself: these are the
    // decision inputs for "does the screen pay", not an acceptance statistic.
    RecordMeasurement(row + ".single_run_contended_screened_wall_ns",
                      static_cast<double>(outcome.screenedNanoseconds));
    RecordMeasurement(row + ".single_run_contended_unscreened_wall_ns",
                      static_cast<double>(outcome.unscreenedNanoseconds));
    return outcome;
}

} // namespace qcx::testing::riAuxScreen
