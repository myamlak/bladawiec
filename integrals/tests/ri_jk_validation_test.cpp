// The ri_jk validation and disclosure cells. The CONTRACTION was measured
// separately (ri_occ_k); the composed builder (RiFullFockBuilder) and its run
// path landed afterwards. This file measures the builder THAT ACTUALLY RUNS, so
// the number a reader of a `ri_jk` energy record is entitled to is taken on the
// object the run path constructs:
//   - AccuracyAgainstOurOwnExactScreenedExchange: K from the composed builder
//     against the exact screened exchange, elementwise
//     and as an exchange energy, per fixture and per atom. The comparator is
//     THIS REPO'S OWN exact screened exchange (DirectJkFockBuilder at the
//     same preset), not any external code; no pyscf, psi4
//     or other package participates in this cell. The auxiliary basis is
//     recorded beside every number (`aux_basis`), because the RI-K error is
//     the AUX FIT's and a K error read without its aux is a number without
//     its class. The cell also carries the 0.1 kcal/mol target
//     (1.59e-4 Eh) as a RECORDED comparison, never as the pass line: the
//     pass rule is the RI-K-versus-RI-J error class, and the contraction
//     already recorded that the target is missed on this fixture's STO-3G
//     shape.
//   - TheConvergedDensityMovesBetweenExactAndRiJkExchange: the
//     non-variationality of the approximated exchange. Robust RI fitting makes
//     the approximated exchange >= exact, so the composed SCF converges to a
//     slightly DIFFERENT density; the cell records that difference (elementwise,
//     Frobenius, and through a density-derived observable - the electronic
//     dipole trace) BESIDE the energy difference, because energy agreement
//     alone is explicitly refused as evidence.
//
// The SCF instrument below is the ri_occ_k_test.cpp one (its lines 234-300),
// copied - the third copy in this module, and the copy is deliberate for the
// same reason the second one was: the instrument belongs to a landed
// measurement (the -74.96292827 anchor and the contraction's iteration counts),
// and extracting it into a shared header would move the instrument of
// measurements already taken. The three copies are held together by
// re-anchoring the same in-tree pin, which is the drift detector.

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/ri_full_fock.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// The auxiliary basis of every cell below, pinned by name on the RI side and
// recorded beside every number. It is the J+K fit the ri_jk path requires, and
// it is NOT auto-selected here: a cell whose aux came from a default would be
// measuring a different approximation than the one it names (the
// fixture-construction rule, applied to the module-level cell as well).
constexpr std::string_view kAuxName = "def2-universal-jkfit";

// The target for the exchange error against exact K: 0.1 kcal/mol = 1.59e-4 Eh,
// carried into the accuracy cell below. A TARGET, recorded as a comparison -
// the error-class rule governs passing.
constexpr double kPlanExchangeTarget = 1.59e-4;

// kcal/mol per Hartree (CODATA-consistent, the conversion the target is
// stated in).
constexpr double kCaloriesPerHartree = 627.5094740631;

// The SCF fixture's convergence threshold: the operating setting (1e-8 energy).
constexpr double kFixtureEnergyTolerance = 1e-8;

// The exchange-energy ceilings of the accuracy cell: a class bound an order
// past the measured values, not a tight pin. MEASURED 2026-09-12 on the
// contraction (ri_occ_k_test.cpp) and re-measured here on the BUILDER: the
// aux K-fit exchange error is 3.524e-4 Eh on H2O/STO-3G and 8.515e-5 Eh on
// water/def2-SVP - the RI-J class on the same fixture and aux is 1.464e-5 and
// 2.6e-4 Eh respectively (the contraction's record), so the K fit, not the
// contraction, is the error source these ceilings leave room for.
constexpr double kExchangeEnergyCeilingSto3g = 1e-3;
constexpr double kExchangeEnergyCeilingDef2Svp = 3e-4;

// The elementwise class ceiling of the same cell. The RI-J elementwise error
// is recorded at 2.6e-4 Eh on the 7x7 STO-3G block
// (ri_engine_test.cpp:414-418), and the contraction measured the K/Coulomb
// elementwise ratio at 1.18x (STO-3G) and 29.8x (def2-SVP), so the expected
// elementwise K error sits at the 1e-4-1e-3 class. This bound is 5e-3: it
// refuses a sign, a factor
// of two, or a transposed term - each of which lands at O(1) - and it does not
// pretend to pin the fit. The measured values are recorded either way.
constexpr double kExchangeElementwiseClassCeiling = 5e-3;

// The density cell's sanity ceiling: the converged rho move a defect that
// returns the wrong object (or the wrong density convention) would trip. The
// expected move is the energy move's own order (~1e-4), so this is a class
// bound with a wide margin, not a measurement.
constexpr double kDensityMoveCeiling = 5e-2;

// The three presets the per-preset accuracy acceptance is measured at, in the
// preset ladder's own order (loose, normal, tight). The acceptance's bar for
// each is read from
// the preset mapping's table, never retyped here.
constexpr std::array<qcx::integrals::AccuracyPreset, 3> kPresets{
    qcx::integrals::AccuracyPreset::kLoose,
    qcx::integrals::AccuracyPreset::kNormal,
    qcx::integrals::AccuracyPreset::kTight};

// The preset's word, for the record and for the printed line. Not routed
// through io's ToString: this is a module-level cell and io is downstream of
// integrals in the DAG (a test that reached io from here would be the
// dependency inversion, not a convenience).
std::string_view PresetWord(qcx::integrals::AccuracyPreset preset) {
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

// Parses the fixture's elements (H, O) out of a vendored basis directory - the
// filtered parse the engine's own molecule-scoped contract uses.
qcx::Result<qcx::basisset::BasisSet> ParseFixtureBasis(std::string_view directory) {
    const std::filesystem::path root(QcxBasisDataDir);
    const std::array<int, 2> elements{1, 8};
    return qcx::basisset::ParseNwchemDirectoryFiltered((root / directory).string(), elements);
}

qcx::Result<Eigen::MatrixXd> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
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

// J = B (B^T d) with d in the contraction layout: the Coulomb half the
// composed builder reads from the SAME metric-transformed tensor as its
// exchange half. Used here only to EXTRACT K from the
// builder's F = H + 2 J_ri - K_ri - the extraction's own error is the
// J-from-B identity, which ri_occ_k_test.cpp pins at 8.7e-14 relative on this
// fixture shape.
// (transformed, density) names the AO-basis pair.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::MatrixXd CoulombFromTransformed(const Eigen::MatrixXd& transformed,
                                       const Eigen::MatrixXd& density) {
    const Eigen::Index n = density.rows();
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

    return coulomb;
}

// The fixture RHF driver: the plain closed-shell fixed-point SCF in the
// spatial density convention the builders use (rho = the spin-summed D over
// 2), from the core-Hamiltonian guess, one Fock build per iteration - the
// ri_occ_k_test.cpp instrument, unchanged, so the densities and iteration
// counts below are comparable with the contraction's.
struct RhfOutcome {
    double energy = 0.0; ///< The electronic energy tr(rho (H + F)) (no E_nuc).
    double deltaEnergy = 0.0; ///< The last |E_k - E_{k-1}|.
    Eigen::MatrixXd density; ///< The closed-shell spatial density rho at the last iteration.
    Eigen::MatrixXd orbitals; ///< The occupied MO block C_occ of the last iteration.
    std::size_t iterations = 0; ///< The Fock builds made.
    bool converged = false; ///< Whether the last move fell below the threshold.
};

template <typename FockFn>
qcx::Result<RhfOutcome> RunRhf(
    // (core, overlap) is the Hamiltonian-then-metric order of an RHF setup.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Eigen::MatrixXd& core,
    const Eigen::MatrixXd& overlap,
    std::size_t occupiedCount,
    const FockFn& fock,
    // (energyTolerance, maxIterations) is tolerance-then-iteration-cap, named at every call site.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
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

void RecordMeasurement(const std::string& key, double value) {
    testing::Test::RecordProperty(key, value);
}

void RecordText(const std::string& key, std::string_view value) {
    testing::Test::RecordProperty(key, std::string(value));
}

// The exact-K side's options: the exchange-only mode of the direct builder, at
// the same preset and with the same screening the composed path's removed
// nested call carried (ri_engine.cpp's composition, before the rebuilt path).
qcx::integrals::FockBuildOptions ExchangeOnlyOptions(qcx::integrals::AccuracyPreset preset) {
    qcx::integrals::FockBuildOptions options;
    options.accuracy = preset;
    options.useDensityScreening = true;
    options.useCertifiedMixedPrecision = false;
    options.buildExchangeOnly = true;
    return options;
}

// The composed builder's Fock adapter over the SCF instrument's (density,
// orbitals) pair. RiFullFockBuilder needs BOTH, from the same iterate - the
// coherence contract its header states - which is why this lambda is not the
// density-only shape the direct builder uses.
template <typename BuilderResult> auto RiFullFockFn(const BuilderResult& builder) {
    return [&builder](const Eigen::MatrixXd& density,
                      const Eigen::MatrixXd& occupied) -> qcx::Result<Eigen::MatrixXd> {
        auto densityTensor = ToTensor(density);

        if (!densityTensor.has_value())
        {
            return std::unexpected(densityTensor.error());
        }

        auto fock = builder->BuildFock(*densityTensor, occupied);

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        return ToMatrix(*fock);
    };
}

// The direct family's adapter: the density-only shape, because the direct
// builder needs no orbitals.
template <typename BuilderResult> auto DirectFockFn(const BuilderResult& builder) {
    return [&builder](const Eigen::MatrixXd& density,
                      const Eigen::MatrixXd&) -> qcx::Result<Eigen::MatrixXd> {
        auto densityTensor = ToTensor(density);

        if (!densityTensor.has_value())
        {
            return std::unexpected(densityTensor.error());
        }

        auto fock = builder->BuildFock(*densityTensor);

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        return ToMatrix(*fock);
    };
}

// ---------------------------------------------------------------------------
// The accuracy cell against our own exact screened exchange.
// ---------------------------------------------------------------------------

struct AccuracyFixture {
    std::string label;
    const qcx::molecule::Molecule& molecule;
    const qcx::basisset::BasisSet& basis;
    const qcx::basisset::BasisSet& aux;
    const CpuTensor2& coreHamiltonian;
    Eigen::MatrixXd core; ///< Held by value: a reference from a temporary would dangle.
    Eigen::MatrixXd overlap;
    std::size_t occupiedCount = 0;
    double exchangeEnergyCeiling = 0.0;
    /// The preset BOTH sides run at (the exact-exchange comparator: our own
    /// exact screened exchange, at the same preset). Carried per fixture so the
    /// per-preset acceptance below is a measurement and not a re-derivation.
    qcx::integrals::AccuracyPreset preset = qcx::integrals::AccuracyPreset::kTight;
};

void MeasureAccuracyCell(const AccuracyFixture& fixture) {
    const std::string& label = fixture.label;
    const qcx::integrals::AccuracyPreset preset = fixture.preset;
    qcx::integrals::RiEngineOptions engineOptions;
    engineOptions.accuracy = preset;

    // The density the comparison is taken at: the DIRECT SCF's converged rho
    // on this fixture - a physical density, so the cell measures the fit's
    // error and not a synthetic density's response to it.
    qcx::integrals::FockBuildOptions directOptions;
    directOptions.accuracy = preset;
    directOptions.useDensityScreening = true;
    directOptions.useCertifiedMixedPrecision = false;
    auto direct = qcx::integrals::DirectJkFockBuilder::Create(
        fixture.molecule, fixture.basis, fixture.coreHamiltonian, directOptions);
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    auto directOutcome =
        RunRhf(fixture.core, fixture.overlap, fixture.occupiedCount, DirectFockFn(direct));
    ASSERT_TRUE(directOutcome.has_value()) << directOutcome.error().message;
    ASSERT_TRUE(directOutcome->converged) << label << ": the direct SCF did not converge";

    // The ri_jk side: the composed builder, exactly as the run path constructs
    // it (one metric-transformed tensor, no direct-exchange call).
    auto builder = qcx::integrals::RiFullFockBuilder::Create(
        fixture.molecule, fixture.basis, fixture.aux, fixture.coreHamiltonian, engineOptions);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    auto densityTensor = ToTensor(directOutcome->density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    auto riFock = builder->BuildFock(*densityTensor, directOutcome->orbitals);
    ASSERT_TRUE(riFock.has_value()) << riFock.error().message;

    // K_ri is extracted from the composition: F = H + 2 J_B - K_ri, with J_B
    // the builder's own Coulomb half read from the SAME B it was created with.
    // The extraction's error is the J-from-B identity (pinned at 8.7e-14
    // relative in ri_occ_k_test.cpp), three orders below the fit error this
    // cell measures.
    const Eigen::MatrixXd coulombRi =
        CoulombFromTransformed(builder->MetricTransformedTensor(), directOutcome->density);
    const Eigen::MatrixXd exchangeRi = fixture.core + 2.0 * coulombRi - ToMatrix(*riFock);

    // The exact side: the same builder family and preset in exchange-only mode,
    // which returns F = H - K with the same density (the exact-K reference, and
    // the nested call the composed path no longer makes).
    auto exchangeOnly = qcx::integrals::DirectJkFockBuilder::Create(
        fixture.molecule, fixture.basis, fixture.coreHamiltonian, ExchangeOnlyOptions(preset));
    ASSERT_TRUE(exchangeOnly.has_value()) << exchangeOnly.error().message;
    auto exchangeFock = exchangeOnly->BuildFock(*densityTensor);
    ASSERT_TRUE(exchangeFock.has_value()) << exchangeFock.error().message;
    const Eigen::MatrixXd exchangeExact = fixture.core - ToMatrix(*exchangeFock);

    const Eigen::MatrixXd difference = exchangeRi - exchangeExact;
    const double elementwise = difference.cwiseAbs().maxCoeff();
    const double scale = exchangeExact.cwiseAbs().maxCoeff();
    // The exchange energy at this density: E_x = -tr(rho K) in the spatial
    // density convention (rho = D/2, so -tr(rho K) = -(1/2) tr(D K)). The
    // DIFFERENCE of the two paths' exchange energies is the move this cell
    // measures; the sign convention cancels in it.
    const double exchangeEnergyMove = std::abs((directOutcome->density * difference).trace());
    const double perAtom = exchangeEnergyMove / static_cast<double>(fixture.molecule.AtomCount());
    const double perAtomKcal = perAtom * kCaloriesPerHartree;
    const double targetRatio = exchangeEnergyMove / kPlanExchangeTarget;
    const bool targetMet = exchangeEnergyMove <= kPlanExchangeTarget;

    // The acceptance bar this cell carries: the PER-PRESET, PER-ATOM bar from
    // the preset mapping's own table (integrals accuracy.hpp
    // RiExchangeErrorBudgetPerAtom), so the bar and the measurement it is
    // compared against are one value read once. The bar is recorded as a
    // comparison, never as the pass line: the error class governs passing, not
    // the preset budget - the same convention this file already uses for the
    // 0.1 kcal/mol target. The measurement is what the acceptance is read from.
    const double perAtomBudget = qcx::integrals::RiExchangeErrorBudgetPerAtom(preset);
    const double perAtomBudgetRatio = perAtom / perAtomBudget;
    const bool perAtomBudgetMet = perAtom <= perAtomBudget;

    std::cout << "[ri_jk] " << label << " K accuracy vs our own exact screened exchange (aux "
              << kAuxName << ", n = " << fixture.core.rows() << ", preset " << PresetWord(preset)
              << "): elementwise " << elementwise << " (scale " << scale << "), exchange energy "
              << exchangeEnergyMove << " Eh = " << (exchangeEnergyMove * kCaloriesPerHartree)
              << " kcal/mol = " << perAtomKcal << " kcal/mol/atom; plan target "
              << kPlanExchangeTarget << " -> ratio " << targetRatio << " ("
              << (targetMet ? "met" : "MISSED") << "); per-preset per-atom budget " << perAtomBudget
              << " Eh/atom -> ratio " << perAtomBudgetRatio << " ("
              << (perAtomBudgetMet ? "met" : "MISSED") << ")\n";

    RecordText(label + ".aux_basis", kAuxName);
    RecordText(label + ".preset", PresetWord(preset));
    RecordMeasurement(label + ".exchange_error_elementwise", elementwise);
    RecordMeasurement(label + ".exchange_scale", scale);
    RecordMeasurement(label + ".exchange_energy_error", exchangeEnergyMove);
    RecordMeasurement(label + ".exchange_energy_error_kcal",
                      exchangeEnergyMove * kCaloriesPerHartree);
    RecordMeasurement(label + ".exchange_energy_error_per_atom_kcal", perAtomKcal);
    RecordMeasurement(label + ".plan_target", kPlanExchangeTarget);
    RecordMeasurement(label + ".plan_target_ratio", targetRatio);
    RecordMeasurement(label + ".plan_target_met", targetMet ? 1.0 : 0.0);
    RecordMeasurement(label + ".per_atom_budget", perAtomBudget);
    RecordMeasurement(label + ".per_atom_budget_ratio", perAtomBudgetRatio);
    RecordMeasurement(label + ".per_atom_budget_met", perAtomBudgetMet ? 1.0 : 0.0);

    EXPECT_LE(exchangeEnergyMove, fixture.exchangeEnergyCeiling)
        << label
        << ": the composed builder's exchange error is past its recorded class - "
           "a regression in the fit's use, not a re-measurement of it";
    EXPECT_LE(elementwise, kExchangeElementwiseClassCeiling)
        << label
        << ": the elementwise K deviation is past the class of every recorded fit error "
           "- a sign, factor or transposition defect, not an auxiliary fit";
}

TEST(RiJkValidationTest, AccuracyAgainstOurOwnExactScreenedExchange) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jkfit g shells";
    }

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto aux = ParseFixtureBasis(kAuxName);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    const std::size_t occupiedCount = static_cast<std::size_t>(molecule->ElectronCount()) / 2;

    // The exact-K accuracy fixture: the small one, where the module's own RI
    // cells already run (H2O/STO-3G + the J+K fit).
    auto sto3g = MakeH2oSto3gBasis();
    ASSERT_TRUE(sto3g.has_value()) << sto3g.error().message;
    auto sto3gCore = BuildCoreHamiltonian(*molecule, *sto3g);
    ASSERT_TRUE(sto3gCore.has_value()) << sto3gCore.error().message;
    auto sto3gCoreTensor = ToTensor(*sto3gCore);
    ASSERT_TRUE(sto3gCoreTensor.has_value()) << sto3gCoreTensor.error().message;
    auto sto3gOverlap = qcx::integrals::BuildOverlapMatrix(*molecule, *sto3g);
    ASSERT_TRUE(sto3gOverlap.has_value()) << sto3gOverlap.error().message;

    // The contraction's acceptance fixture, as the second shape: the one where
    // it measured the worst aux K-fit ratio (5.82x the RI-J class).
    auto svp = ParseFixtureBasis("def2-svp");
    ASSERT_TRUE(svp.has_value()) << svp.error().message;
    auto svpCore = BuildCoreHamiltonian(*molecule, *svp);
    ASSERT_TRUE(svpCore.has_value()) << svpCore.error().message;
    auto svpCoreTensor = ToTensor(*svpCore);
    ASSERT_TRUE(svpCoreTensor.has_value()) << svpCoreTensor.error().message;
    auto svpOverlap = qcx::integrals::BuildOverlapMatrix(*molecule, *svp);
    ASSERT_TRUE(svpOverlap.has_value()) << svpOverlap.error().message;

    // THE PRESET AXIS: each fixture is measured at ALL THREE
    // presets, both sides at the same one, because the acceptance that names
    // these three presets cannot be read from a cell that only ever ran at
    // kTight. The label carries the preset so the record's keys stay distinct
    // across the axis.
    for (const qcx::integrals::AccuracyPreset preset : kPresets)
    {
        const std::string presetSuffix = std::string{"_"} + std::string{PresetWord(preset)};
        MeasureAccuracyCell(AccuracyFixture{.label = "h2o_sto3g" + presetSuffix,
                                            .molecule = *molecule,
                                            .basis = *sto3g,
                                            .aux = *aux,
                                            .coreHamiltonian = *sto3gCoreTensor,
                                            .core = *sto3gCore,
                                            .overlap = ToMatrix(*sto3gOverlap),
                                            .occupiedCount = occupiedCount,
                                            .exchangeEnergyCeiling = kExchangeEnergyCeilingSto3g,
                                            .preset = preset});

        MeasureAccuracyCell(AccuracyFixture{.label = "water_def2svp" + presetSuffix,
                                            .molecule = *molecule,
                                            .basis = *svp,
                                            .aux = *aux,
                                            .coreHamiltonian = *svpCoreTensor,
                                            .core = *svpCore,
                                            .overlap = ToMatrix(*svpOverlap),
                                            .occupiedCount = occupiedCount,
                                            .exchangeEnergyCeiling = kExchangeEnergyCeilingDef2Svp,
                                            .preset = preset});
    }
}

// The preset mapping itself: the two facts a consumer of an
// approximated-exchange run reads the preset through, pinned so a later
// re-derivation of either has to move this cell deliberately rather than by
// drift. This asserts the MAPPING, not any physics: the measured per-atom
// errors live in the record of the accuracy cell above, where the fixtures are.
TEST(RiJkValidationTest, ThePresetMappingBarsTheScreenOnlyAndKeepsTheJkFit) {
    // The aux quality is NOT a preset knob: a JK-optimized fit is required at
    // every rung (the JK-fit predicate refuses a
    // J-only fit outright). If a slice ever needs this to vary by preset, that
    // slice is changing the path's approximation contract, not its screening.
    for (const qcx::integrals::AccuracyPreset preset : kPresets)
    {
        EXPECT_TRUE(qcx::integrals::RiExchangeRequiresJkFit(preset))
            << PresetWord(preset)
            << ": the auxiliary quality is preset-invariant on this path (the fit is the single "
               "approximation; the preset governs the screening co-term only)";
    }

    // The per-atom budget is the preset table carried in Eh: 0.5 / 0.02 /
    // 0.005 kcal/mol per atom. Strictly decreasing across the axis, because the
    // mapping is the acceptance's and the acceptance tightens with the preset.
    const double loose =
        qcx::integrals::RiExchangeErrorBudgetPerAtom(qcx::integrals::AccuracyPreset::kLoose);
    const double normal =
        qcx::integrals::RiExchangeErrorBudgetPerAtom(qcx::integrals::AccuracyPreset::kNormal);
    const double tight =
        qcx::integrals::RiExchangeErrorBudgetPerAtom(qcx::integrals::AccuracyPreset::kTight);

    EXPECT_GT(loose, normal) << "the per-atom budget must tighten with the preset";
    EXPECT_GT(normal, tight) << "the per-atom budget must tighten with the preset";

    // The Eh literals the mapper carries are the preset table's kcal/mol values
    // to
    // THREE SIGNIFICANT FIGURES, so the round trip back to kcal/mol lands
    // within ~0.1% of the round number and not within 1e-9: 7.97e-4 Eh/atom
    // reads 0.500125, 3.19e-5 reads 0.0200176, 7.97e-6 reads 0.00500125. The
    // tolerance below is that rounding and nothing looser - it pins that the
    // mapper carries the preset table's three bars and not three other numbers.
    constexpr double kBarRoundTripTolerance = 1e-3; // relative
    EXPECT_NEAR(loose * kCaloriesPerHartree / 0.5, 1.0, kBarRoundTripTolerance)
        << "the loose bar is the preset's 0.5 kcal/mol per atom";
    EXPECT_NEAR(normal * kCaloriesPerHartree / 0.02, 1.0, kBarRoundTripTolerance)
        << "the normal bar is the preset's 0.02 kcal/mol per atom";
    EXPECT_NEAR(tight * kCaloriesPerHartree / 0.005, 1.0, kBarRoundTripTolerance)
        << "the tight bar is the preset's 0.005 kcal/mol per atom";
}

// ---------------------------------------------------------------------------
// The density cell.
// ---------------------------------------------------------------------------

TEST(RiJkValidationTest, TheConvergedDensityMovesBetweenExactAndRiJkExchange) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jkfit g shells";
    }

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = ParseFixtureBasis(kAuxName);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto coreTensor = ToTensor(*core);
    ASSERT_TRUE(coreTensor.has_value()) << coreTensor.error().message;
    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    const Eigen::MatrixXd overlapMatrix = ToMatrix(*overlap);
    const std::size_t occupiedCount = static_cast<std::size_t>(molecule->ElectronCount()) / 2;
    qcx::integrals::RiEngineOptions engineOptions;
    engineOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;

    // Side A: exact screened exchange (the default family's own kernel).
    qcx::integrals::FockBuildOptions directOptions;
    directOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    directOptions.useDensityScreening = true;
    directOptions.useCertifiedMixedPrecision = false;
    auto direct =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *coreTensor, directOptions);
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    auto directOutcome = RunRhf(*core, overlapMatrix, occupiedCount, DirectFockFn(direct));
    ASSERT_TRUE(directOutcome.has_value()) << directOutcome.error().message;
    ASSERT_TRUE(directOutcome->converged)
        << "the exact path did not converge in " << directOutcome->iterations << " iterations";

    // Side B: ri_jk (the composed full-RI builder), the same instrument, the
    // same threshold, the same starting guess.
    auto builder = qcx::integrals::RiFullFockBuilder::Create(
        *molecule, *basis, *aux, *coreTensor, engineOptions);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;
    auto riOutcome = RunRhf(*core, overlapMatrix, occupiedCount, RiFullFockFn(builder));
    ASSERT_TRUE(riOutcome.has_value()) << riOutcome.error().message;
    ASSERT_TRUE(riOutcome->converged)
        << "the ri_jk path did not converge in " << riOutcome->iterations << " iterations";

    // The density move, in three readings: the largest element, the Frobenius
    // norm of the difference, and the move of a density-derived observable.
    const Eigen::MatrixXd densityMove = riOutcome->density - directOutcome->density;
    const double maxElement = densityMove.cwiseAbs().maxCoeff();
    const double frobenius = densityMove.norm();
    const double referenceNorm = directOutcome->density.norm();
    const double relativeFrobenius = frobenius / referenceNorm;

    // The density-derived observable: the ELECTRONIC dipole trace, from the
    // integrals module's own dipole matrices at the coordinate origin. The
    // nuclear contribution cancels exactly in a difference of two runs, so the
    // move below is the move of the physical dipole of this fixture (the
    // origin-dependent piece is common to both sides). Sign convention: the
    // electronic dipole is -sum D mu, so the recorded move is its absolute
    // value per component and its Euclidean norm.
    auto dipoleTensors = qcx::integrals::BuildDipoleMatrix(*molecule, *basis);
    ASSERT_TRUE(dipoleTensors.has_value()) << dipoleTensors.error().message;
    Eigen::Vector3d dipoleMove = Eigen::Vector3d::Zero();
    std::array<double, 3> dipoleExact{};

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const Eigen::MatrixXd dipoleMatrix = ToMatrix((*dipoleTensors)[axis]);
        const double electronic = 2.0 * (directOutcome->density * dipoleMatrix).trace();
        const double electronicRi = 2.0 * (riOutcome->density * dipoleMatrix).trace();
        dipoleExact[axis] = electronic;
        dipoleMove(static_cast<Eigen::Index>(axis)) = std::abs(electronicRi - electronic);
    }

    const double dipoleMoveNorm = dipoleMove.norm();
    const double energyMove = std::abs(riOutcome->energy - directOutcome->energy);

    std::cout << "[ri_jk] h2o_sto3g density cell (aux " << kAuxName << "): rho move max "
              << maxElement << ", Frobenius " << frobenius << " (relative " << relativeFrobenius
              << "), dipole move " << dipoleMoveNorm << " (x " << dipoleMove(0) << " y "
              << dipoleMove(1) << " z " << dipoleMove(2) << "), energy move " << energyMove
              << " Eh = " << (energyMove * kCaloriesPerHartree) << " kcal/mol\n";

    RecordMeasurement("h2o_sto3g.density_move_max_element", maxElement);
    RecordMeasurement("h2o_sto3g.density_move_frobenius", frobenius);
    RecordMeasurement("h2o_sto3g.density_move_relative_frobenius", relativeFrobenius);
    RecordMeasurement("h2o_sto3g.dipole_move_x", dipoleMove(0));
    RecordMeasurement("h2o_sto3g.dipole_move_y", dipoleMove(1));
    RecordMeasurement("h2o_sto3g.dipole_move_z", dipoleMove(2));
    RecordMeasurement("h2o_sto3g.dipole_move_norm", dipoleMoveNorm);
    RecordMeasurement("h2o_sto3g.exact_electronic_dipole_x", dipoleExact[0]);
    RecordMeasurement("h2o_sto3g.exact_electronic_dipole_y", dipoleExact[1]);
    RecordMeasurement("h2o_sto3g.exact_electronic_dipole_z", dipoleExact[2]);
    // The energy move BESIDE the density move - never instead of it.
    RecordMeasurement("h2o_sto3g.total_energy_move", energyMove);
    RecordMeasurement("h2o_sto3g.total_energy_move_kcal", energyMove * kCaloriesPerHartree);
    RecordMeasurement("h2o_sto3g.exact_iterations", static_cast<double>(directOutcome->iterations));
    RecordMeasurement("h2o_sto3g.ri_jk_iterations", static_cast<double>(riOutcome->iterations));

    EXPECT_LT(maxElement, kDensityMoveCeiling)
        << "the converged densities of the two exchange treatments differ beyond the class of "
           "the exchange approximation itself - the density cell exists because robustness "
           "makes the RI-K functional non-variational, which moves the density by the energy "
           "move's own order, not by an O(1) amount";
}
} // namespace
