// The RKS convergence probe.
//
// THE QUESTION. This probe asks for confirmation that DIIS/CDIIS behaves on DFT's
// numerical profile - the grid-quadrature noise HF's exact-integral Fock does
// not have. The first framing of this probe ("find a system whose stop is
// decided by the density leg with the energy leg never firing") was withdrawn
// on 2026-09-13 as mis-aimed, for a reason worth recording: near the fixed
// point E is QUADRATIC in D, so an energy tolerance of 1e-8 corresponds to a
// density tolerance of order 1e-4. The energy leg is therefore structurally
// the LOOSER of the two and is expected to bind almost never - a search over
// molecules would have burned the work without moving the answer.
//
// The axis that decides which leg binds is the tolerance ARRANGEMENT, not the
// molecule. And the question that actually follows is sharper than
// "which leg won": grid noise enters the ENERGY (Exc is quadrature-evaluated),
// so if the energy leg does not bind, grid noise cannot reach the stop through
// it at all - it can only reach the stop through the density leg, by moving
// Vxc and hence D. So:
//
//     does the grid resolution move the density-leg crossing, and by how
//     much, relative to the density tolerance?
//
// That is what this file measures, in three steps:
//   1. CONTROL - svwn at the defaults (1e-8 / 1e-6), both achieved legs
//      recorded, and the binding leg read off the pair.
//   2. SWEEP   - the (energy x density) tolerance pair, to establish which
//      arrangements make each leg bind (so a later reader is not guessing).
//   3. GRID    - the same run at two grid resolutions, to measure whether the
//      density-leg crossing moves, and by how much.
//
// THE COMPOSITION. The seam below replicates
// driver/src/internal/ks_composition.hpp:338-380 exactly. It reaches that
// shape through PUBLIC headers only, which is possible because
// XcGridEngine::EvaluateClosedShell(D) is literally `Evaluate(D/2, D/2)`
// (grid/src/xc_grid_engine.cpp:660) - the same call the driver makes as
// `xc(rho, rho)` with rho = D/2. Three traps the driver's own comments name,
// honoured here: c_HF is read from the engine rather than assumed (the
// registry's recipes already carry the (1 - c_HF) scaling, so applying it
// again would double-count); no exchange half is built at all when c_HF = 0;
// and the contribution carries the exchange term, not just Exc.
//
// This file is deliberately a TEST rather than a throwaway probe: the numbers
// it reports are the record, and the assertions are the invariant that
// survives a re-run on a different machine.

#include "h2o_sto3g.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace {

using qcx::testing::ToMatrix;

/// Which gate leg bound one run, read off the achieved pair rather than
/// inferred from the iteration count.
///
/// The gate requires BOTH legs, so the binding one is the one that came
/// closest to its own tolerance in RELATIVE terms - an absolute comparison
/// would always name whichever tolerance happened to be the larger number.
// Not a bulk-storage type; shrinking the base type is a deferred micro-optimization.
// NOLINTNEXTLINE(performance-enum-size)
enum class BindingLeg {
    kEnergy,
    kDensity,
    kNeither, ///< Not converged, so no leg bound.
};

/// One run's record: the gate's own operands plus what they were read against.
struct RunRecord {
    bool converged = false;
    int iterations = 0;
    double energy = 0.0;
    double achievedEnergyDelta = 0.0;
    double achievedRmsDensityChange = 0.0;
    double energyFraction = 0.0; ///< achieved / tolerance; >= 1 means it bound.
    double densityFraction = 0.0;
    BindingLeg leg = BindingLeg::kNeither;
};

RunRecord Measure(const qcx::scf::HfResult& result, const qcx::scf::RhfOptions& options) {
    RunRecord record;
    record.converged = result.converged;
    record.iterations = result.iterations;
    record.energy = result.totalEnergy;
    record.achievedEnergyDelta = result.achievedEnergyDelta;
    record.achievedRmsDensityChange = result.achievedRmsDensityChange;
    record.energyFraction = result.achievedEnergyDelta / options.energyTolerance;
    record.densityFraction = result.achievedRmsDensityChange / options.densityTolerance;

    if (result.converged)
    {
        record.leg = record.energyFraction > record.densityFraction ? BindingLeg::kEnergy
                                                                    : BindingLeg::kDensity;
    }

    return record;
}

const char* LegName(BindingLeg leg) {
    switch (leg)
    {
    case BindingLeg::kEnergy:
        return "ENERGY";
    case BindingLeg::kDensity:
        return "density";
    case BindingLeg::kNeither:
        return "neither";
    }

    return "?";
}

/// Prints one record as a fixed-width row, so a run's numbers are in the
/// output rather than only in an assertion that passed.
void PrintRow(const char* label, const RunRecord& record) {
    std::printf("%-34s conv=%d iters=%3d E=%.12f dE=%.3e (%.3f) rmsD=%.3e (%.3f) binding=%s\n",
                label,
                record.converged ? 1 : 0,
                record.iterations,
                record.energy,
                record.achievedEnergyDelta,
                record.energyFraction,
                record.achievedRmsDensityChange,
                record.densityFraction,
                LegName(record.leg));
    std::fflush(stdout);
}

/// H2O/STO-3G's one- and two-electron inputs, built once per test.
struct H2oInputs {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd core;
};

qcx::Result<H2oInputs> BuildH2oInputs() {
    auto molecule = qcx::testing::MakeH2oSto3g();

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto basis = qcx::testing::MakeH2oSto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    H2oInputs inputs{std::move(*molecule),
                     std::move(*basis),
                     ToMatrix(*overlap),
                     ToMatrix(*kinetic) + ToMatrix(*nuclear)};
    return inputs;
}

/// The three Kohn-Sham callbacks, composed exactly as the driver composes
/// them (see the file header for the line references and the three traps).
struct RksSeam {
    qcx::scf::FockBuilderFn fock;
    qcx::scf::CoulombFn coulomb;
    qcx::scf::EnergyContributionFn contribution;
    double exchangeFraction = 0.0;
    bool usesGradient = false;
};

/// Builds the seam for one functional and one grid resolution.
///
/// The halves are the integrals builder in its two half-modes, which is what
/// the driver uses and what the composition is written against: a
/// buildCoulombOnly build returns H + 2J(rho) and a buildExchangeOnly build
/// returns H - K(rho).
qcx::Result<RksSeam> MakeRksSeam(const H2oInputs& inputs,
                                 std::string_view functional,
                                 std::size_t radialPoints,
                                 std::size_t angularPoints) {
    auto engine =
        qcx::grid::XcGridEngine::Create(inputs.molecule,
                                        inputs.basis,
                                        functional,
                                        qcx::grid::XcGridSettings{radialPoints, angularPoints});

    if (!engine.has_value())
    {
        return std::unexpected(engine.error());
    }

    auto shared = std::make_shared<qcx::grid::XcGridEngine>(std::move(*engine));

    // c_HF comes from the functional itself. The registry's recipes already
    // carry the (1 - c_HF) exchange scaling, so this value is applied to the
    // EXACT-exchange half only - never to the DFT part.
    const double exchangeFraction = shared->ExchangeFraction();
    const bool hasExchange = exchangeFraction > 0.0;

    // NOT const: a const Tensor cannot be moved from (std::move on a const
    // lvalue yields const Tensor&&, which does not bind the move ctor, so it
    // falls back to the deleted copy).
    auto coreTensor = qcx::testing::ToTensor(inputs.core);

    if (!coreTensor.has_value())
    {
        return std::unexpected(coreTensor.error());
    }

    auto coreShared = std::make_shared<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>>(
        std::move(*coreTensor));

    qcx::integrals::FockBuildOptions coulombOptions;
    coulombOptions.buildCoulombOnly = true;

    qcx::integrals::FockBuildOptions exchangeOptions;
    exchangeOptions.buildExchangeOnly = true;

    auto coulombBuilder = qcx::integrals::DirectJkFockBuilder::Create(
        inputs.molecule, inputs.basis, *coreShared, coulombOptions);

    if (!coulombBuilder.has_value())
    {
        return std::unexpected(coulombBuilder.error());
    }

    std::shared_ptr<qcx::integrals::DirectJkFockBuilder> exchangeBuilder;

    if (hasExchange)
    {
        auto built = qcx::integrals::DirectJkFockBuilder::Create(
            inputs.molecule, inputs.basis, *coreShared, exchangeOptions);

        if (!built.has_value())
        {
            return std::unexpected(built.error());
        }

        exchangeBuilder = std::make_shared<qcx::integrals::DirectJkFockBuilder>(std::move(*built));
    }

    auto coulombShared =
        std::make_shared<qcx::integrals::DirectJkFockBuilder>(std::move(*coulombBuilder));

    // The per-density cache: every callback goes through it, so the three
    // halves are built once per density no matter which asks first. Keyed on
    // EXACT equality - these are copies of the loop's own iterate, so a hit
    // is the identical input to a deterministic build, never a near-miss.
    struct Cache {
        Eigen::MatrixXd density;
        Eigen::MatrixXd coulombHalf;
        Eigen::MatrixXd exchangeHalf;
        qcx::grid::XcEvaluation xc;
        bool valid = false;
    };

    auto cache = std::make_shared<Cache>();

    const auto ensure = [cache, coulombShared, exchangeBuilder, hasExchange, shared](
                            const Eigen::MatrixXd& density) -> qcx::Result<void> {
        if (cache->valid && cache->density.rows() == density.rows() &&
            cache->density.cols() == density.cols() && cache->density == density)
        {
            return {};
        }

        // The halves take the SPATIAL density rho = D/2 - the RHF seam's own
        // convention, so coulombHalf = H + 2J(D/2) = H + J[D]. BuildFock's
        // argument is that density, NOT the core Hamiltonian the builder was
        // constructed with.
        const auto rhoTensor = qcx::testing::ToTensor(0.5 * density);

        if (!rhoTensor.has_value())
        {
            return std::unexpected(rhoTensor.error());
        }

        const auto coulombTensor = coulombShared->BuildFock(*rhoTensor);

        if (!coulombTensor.has_value())
        {
            return std::unexpected(coulombTensor.error());
        }

        if (hasExchange)
        {
            const auto exchangeTensor = exchangeBuilder->BuildFock(*rhoTensor);

            if (!exchangeTensor.has_value())
            {
                return std::unexpected(exchangeTensor.error());
            }

            cache->exchangeHalf = ToMatrix(*exchangeTensor);
        }

        const auto evaluation = shared->EvaluateClosedShell(density);

        if (!evaluation.has_value())
        {
            return std::unexpected(evaluation.error());
        }

        cache->density = density;
        cache->coulombHalf = ToMatrix(*coulombTensor);
        cache->xc = *evaluation;
        cache->valid = true;
        return {};
    };

    RksSeam seam;
    seam.exchangeFraction = exchangeFraction;
    seam.usesGradient = shared->UsesGradient();

    seam.fock = [ensure, cache, hasExchange, exchangeFraction, core = inputs.core](
                    const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        const auto filled = ensure(density);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        // F_KS[D] = coulombHalf + c_HF (exchangeHalf - H) + (V_alpha + V_beta)/2.
        // The engine returns dE/dD_s, so the restricted potential is HALF the
        // sum - using the sum as it stands double-counts the spin split.
        Eigen::MatrixXd fock =
            cache->coulombHalf + 0.5 * (cache->xc.potentialAlpha + cache->xc.potentialBeta);

        if (hasExchange)
        {
            fock += exchangeFraction * (cache->exchangeHalf - core);
        }

        return fock;
    };

    seam.coulomb = [ensure, cache, core = inputs.core](
                       const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        const auto filled = ensure(density);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        // The half carries one full H copy; subtracting it back leaves
        // exactly the J[D] the seam's energy formula contracts.
        return cache->coulombHalf - core;
    };

    seam.contribution = [ensure, cache, hasExchange, exchangeFraction, core = inputs.core](
                            const Eigen::MatrixXd& density) -> qcx::Result<double> {
        const auto filled = ensure(density);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        double contribution = cache->xc.energy;

        if (hasExchange)
        {
            // K[D] = 2 (H - exchangeHalf), so -c_HF/4 Tr[D K[D]] is
            // -c_HF/2 Tr[D (H - exchangeHalf)].
            const Eigen::MatrixXd kHalf = core - cache->exchangeHalf;
            contribution -= 0.5 * exchangeFraction * (density.cwiseProduct(kHalf)).sum();
        }

        return contribution;
    };

    return seam;
}

/// Runs one RKS SCF at a given tolerance arrangement and grid.
qcx::Result<RunRecord> RunRks(const H2oInputs& inputs,
                              std::string_view functional,
                              // (energyTolerance, densityTolerance) name the two tolerances.
                              // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                              double energyTolerance,
                              double densityTolerance,
                              std::size_t radialPoints,
                              std::size_t angularPoints) {
    auto seam = MakeRksSeam(inputs, functional, radialPoints, angularPoints);

    if (!seam.has_value())
    {
        return std::unexpected(seam.error());
    }

    qcx::scf::RhfOptions options;
    options.energyTolerance = energyTolerance;
    options.densityTolerance = densityTolerance;

    const auto result = qcx::scf::RunRhfScf(inputs.molecule,
                                            inputs.overlap,
                                            inputs.core,
                                            options,
                                            seam->fock,
                                            seam->coulomb,
                                            seam->contribution,
                                            &inputs.basis);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    return Measure(*result, options);
}

// ---------------------------------------------------------------------------
// 1. The control.
// ---------------------------------------------------------------------------

TEST(RksConvergenceProbe, ControlSvwnAtDefaultTolerances) {
    const auto inputs = BuildH2oInputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const auto record = RunRks(*inputs, "svwn", 1e-8, 1e-6, 75, 302);
    ASSERT_TRUE(record.has_value()) << record.error().message;
    PrintRow("svwn 1e-8/1e-6 (75,302)", *record);

    EXPECT_TRUE(record->converged);

    // The invariants the gate guarantees by construction, asserted rather
    // than assumed: a converged result satisfies BOTH legs.
    EXPECT_LT(record->achievedEnergyDelta, 1e-8);
    EXPECT_LT(record->achievedRmsDensityChange, 1e-6);
}

// The peer's prediction, stated as a test: at the defaults the DENSITY leg is
// the binding one. If this ever flips, the arrangement has moved and the
// reading above (grid noise can only reach the stop through the density leg)
// has to be re-derived rather than re-quoted.
TEST(RksConvergenceProbe, DensityLegBindsAtDefaultTolerances) {
    const auto inputs = BuildH2oInputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const auto record = RunRks(*inputs, "svwn", 1e-8, 1e-6, 75, 302);
    ASSERT_TRUE(record.has_value()) << record.error().message;

    EXPECT_EQ(record->leg, BindingLeg::kDensity)
        << "energy fraction " << record->energyFraction << " vs density fraction "
        << record->densityFraction;
}

// The same control on a hybrid, which takes the exchange-half branch of the
// composition (c_HF = 0.20) - the branch the pure functional never enters.
TEST(RksConvergenceProbe, ControlB3lypAtDefaultTolerances) {
    const auto inputs = BuildH2oInputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const auto record = RunRks(*inputs, "b3lyp", 1e-8, 1e-6, 75, 302);
    ASSERT_TRUE(record.has_value()) << record.error().message;
    PrintRow("b3lyp 1e-8/1e-6 (75,302)", *record);

    EXPECT_TRUE(record->converged);
}

// ---------------------------------------------------------------------------
// 2. The tolerance sweep - which arrangement makes each leg bind.
// ---------------------------------------------------------------------------

TEST(RksConvergenceProbe, ToleranceSweepOnOneSystem) {
    const auto inputs = BuildH2oInputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    // A deliberately wide sweep, including arrangements on both sides of
    // realistic: the point is to SHOW which leg binds where, not to find the
    // one arrangement that flips it. The 1e-10 pair is an AXIS POINT of
    // this measurement, not a gate the probe assumes (owner ruling
    // 2026-09-15): the sweep's subject is the (energy x density) tolerance
    // arrangement itself, so tightening one axis point away would delete
    // the arrangement the sweep exists to compare.
    const double energyTolerances[] = {1e-6, 1e-8, 1e-10};
    const double densityTolerances[] = {1e-4, 1e-6, 1e-8};

    for (const double energyTolerance : energyTolerances)
    {
        for (const double densityTolerance : densityTolerances)
        {
            const auto record = RunRks(*inputs, "svwn", energyTolerance, densityTolerance, 75, 302);
            ASSERT_TRUE(record.has_value()) << record.error().message;

            char label[96] = {};
            std::snprintf(
                label, sizeof(label), "svwn %.0e/%.0e", energyTolerance, densityTolerance);
            PrintRow(label, *record);

            // A run that reports converged must satisfy both legs - the
            // cheap way for a probe to lie is to read a converged flag
            // without checking the operands it stood on.
            if (record->converged)
            {
                EXPECT_LT(record->achievedEnergyDelta, energyTolerance);
                EXPECT_LT(record->achievedRmsDensityChange, densityTolerance);
            } else
            {
                EXPECT_EQ(record->leg, BindingLeg::kNeither);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. The grid resolution - does it move the density-leg crossing?
// ---------------------------------------------------------------------------

TEST(RksConvergenceProbe, GridResolutionMovesTheDensityLeg) {
    const auto inputs = BuildH2oInputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    // The qcx defaults, and a refined partner. The refinement is the
    // measurement: what moves between these two runs is grid noise, since
    // nothing else differs. 434 is the LARGEST supported Lebedev size
    // (excgrid's table is 6..434), so it is the finest partner available -
    // 590 is not in the table and the engine refuses it rather than
    // silently degrading, which is the behaviour the first run of this test
    // demonstrated.
    const auto coarse = RunRks(*inputs, "svwn", 1e-8, 1e-6, 75, 302);
    ASSERT_TRUE(coarse.has_value()) << coarse.error().message;
    PrintRow("svwn (75,302)", *coarse);

    const auto fine = RunRks(*inputs, "svwn", 1e-8, 1e-6, 150, 434);
    ASSERT_TRUE(fine.has_value()) << fine.error().message;
    PrintRow("svwn (150,434)", *fine);

    const double energyShift = std::abs(fine->energy - coarse->energy);
    const double densityShift =
        std::abs(fine->achievedRmsDensityChange - coarse->achievedRmsDensityChange);

    std::printf("grid refinement moves:  dE = %.3e Ha   d(rmsD) = %.3e   "
                "d(rmsD)/densityTolerance = %.3f\n",
                energyShift,
                densityShift,
                densityShift / 1e-6);
    std::fflush(stdout);

    // Both resolutions must converge - a grid change that broke the run
    // would be a finding in its own right, not a data point.
    EXPECT_TRUE(coarse->converged);
    EXPECT_TRUE(fine->converged);

    // The energy may move by grid noise; it must not move by more than the
    // tolerance the run itself stops on, or the reported energy is not the
    // energy of the reported density. This bound is deliberately loose
    // (1e-5) - the MEASURED shift above is the record, and pinning it
    // tighter would encode one machine's quadrature as a contract.
    EXPECT_LT(energyShift, 1e-5);
}

// ---------------------------------------------------------------------------
// 4. Separating the 5.95 Ha b3lyp gap: gradient path or exchange path?
// ---------------------------------------------------------------------------

// b3lyp differs from svwn in TWO ways - it carries gradients AND an exact-
// exchange fraction - so a gap on b3lyp alone cannot say which of the two
// paths is responsible. `becke88` separates them: it is a pure GGA exchange
// with c_HF = 0 (gradient yes, exact exchange no), so a run of it exercises
// the gradient path on its own. `slater` is the gradient-free control.
//
// The pyscf references below were produced by tools/whole_scf_reference.py
// on the same fixture, geometry and grid (75,302):
//
//     lda_x   (qcx "slater")  = -74.059755873142
//     gga_x_b88 (qcx "becke88") = -74.938212013538
//
// READ THE TWO GAPS TOGETHER. If becke88 agrees and b3lyp does not, the
// gradient path is sound and the fault is in the composite - the Slater/B88/
// VWN5/LYP mixing or the exchange half. If becke88 ALSO disagrees, the
// gradient path is the suspect and b3lyp is merely where it shows worst.
//
// This test does not assert agreement; it MEASURES and prints. The gap is the
// finding, and pinning a tolerance before knowing its size would be assuming
// the answer.
TEST(RksConvergenceProbe, GgaPathSeparatesFromTheExchangePath) {
    const auto inputs = BuildH2oInputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    // The gradient-free control.
    const auto slater = RunRks(*inputs, "slater", 1e-8, 1e-6, 75, 302);
    ASSERT_TRUE(slater.has_value()) << slater.error().message;
    PrintRow("slater (LDA, c_HF=0)", *slater);

    // The gradient path with NO exact exchange: this is the separator.
    const auto becke88 = RunRks(*inputs, "becke88", 1e-8, 1e-6, 75, 302);
    ASSERT_TRUE(becke88.has_value()) << becke88.error().message;
    PrintRow("becke88 (GGA, c_HF=0)", *becke88);

    // The composite, for the side-by-side.
    const auto b3lyp = RunRks(*inputs, "b3lyp", 1e-8, 1e-6, 75, 302);
    ASSERT_TRUE(b3lyp.has_value()) << b3lyp.error().message;
    PrintRow("b3lyp (GGA + c_HF)", *b3lyp);

    constexpr double kPyscfSlater = -74.059755873142;
    constexpr double kPyscfBecke88 = -74.938212013538;
    constexpr double kPyscfB3lyp5 = -75.275066425621;

    std::printf("vs pyscf:  slater dE = %+.6e   becke88 dE = %+.6e   b3lyp dE = %+.6e\n",
                slater->energy - kPyscfSlater,
                becke88->energy - kPyscfBecke88,
                b3lyp->energy - kPyscfB3lyp5);
    std::fflush(stdout);

    // Only the claims that hold whatever the gaps turn out to be: each run
    // must converge, or the comparison is between non-solutions.
    EXPECT_TRUE(slater->converged);
    EXPECT_TRUE(becke88->converged);
    EXPECT_TRUE(b3lyp->converged);
}

// ---------------------------------------------------------------------------
// 5. Which TERM carries the b3lyp gap?
// ---------------------------------------------------------------------------

// The arithmetic bounds the suspect before anything is run. The exact-exchange
// term's own size is c_HF E_x, and for H2O/STO-3G E_x = -1/4 Tr[D K[D]] is of
// order 4 Ha, so |c_HF E_x| is under 1 Ha at EVERY shipped fraction (0.20,
// 0.25, 0.50). No mis-scaling of the exchange term, doubly applied or applied
// at the raw density, reaches 5.95 Ha on its own. That leaves the DFT part -
// and b3lyp was the only functional that mixed an LDA term (Slater, at the
// 0.80 the registry then shipped; 0.08 since the terms were folded) with GGA
// terms AND carries LYP (0.81), a kernel no comparison above exercises: the
// sound runs are single-component, and `svwn`'s composite is all-LDA at
// weight 1.0.
//
// The family separates those by construction:
//   b3pw91 - b3lyp's twin: same 0.20 fraction, the same pre-fix 0.80/0.72
//            weights (0.08/0.72 since the fold), PW91 correlation instead of
//            LYP. Off => the exchange term or the LDA/GGA mix. Agrees => LYP
//            is the suspect.
//   pbe0   - a 0.25 hybrid, no LYP, and no LDA term mixed with GGA.
//   vwn5   - LDA correlation alone (the 0.19 leg of b3lyp's recipe).
//   pbe_c  - a GGA correlation that is NOT LYP (the control for `lyp`).
//   lyp    - GGA correlation alone (the 0.81 leg of b3lyp's recipe).
//
// References: produced 2026-09-13 by a pyscf 2.14.0 run on the same fixture,
// geometry ('Bohr') and grid (75,302 => 43760 points), each converged. The
// codes are the matched ones: `b3pw91`->libxc 401 and `pbe0`->libxc 406 are
// pyscf's own aliases for those composites, `lyp`->gga_c_lyp (131) and
// `vwn5`->lda_c_vwn (7) are the single components the registry names.
//
// The second reference column is the load-bearing one, and it is HISTORICAL:
// it is pyscf run with the weights the registry shipped BEFORE the fold,
// written out explicitly (`0.2*HF + 0.8*LDA_X + 0.72*B88 + 0.19*VWN5 +
// 0.81*LYP`). The registry ships the folded recipe today (0.08 LDA_X + 0.72
// B88; the HybridFunctional note in excgrid/src/kernels_registry.cpp carries
// the arithmetic), so against the CURRENT registry this column is a NEGATIVE
// CONTROL and the `dE` it prints is the size of the defect the fold removed -
// 5.953 Ha. It is kept because it is what answered the question a bare gap
// cannot: does qcx's number equal a real functional evaluated by an
// independent implementation, or does it equal the recipe? Pre-fix the second
// column reproduced qcx and the first did not, which is what made the defect
// the RECIPE and not the machinery - the kernels and the SCF are certified by
// the column agreeing.
TEST(RksConvergenceProbe, HybridFamilyIsolatesTheOffendingTerm) {
    const auto inputs = BuildH2oInputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    struct Row {
        const char* name;
        const char* note;
        double reference; ///< pyscf, matched code (the functional as intended).

        /// pyscf with the registry's PRE-FIX (0.80) exchange weights; 0 = not measured.
        double registryRecipe;
    };

    const Row rows[] = {
        {"pbe0", "hybrid c_HF=0.25, GGA+GGA, no LYP", -75.245543202489, 0.0},
        {"b3lyp", "hybrid c_HF=0.20, slater+B88+vwn5+lyp", -75.275066425621, -81.228237100329},
        {"b3pw91", "hybrid c_HF=0.20, slater+B88+vwn5+pw91_c", -75.296402297811, -81.251356890791},
        {"b3p86", "hybrid c_HF=0.20, slater+B88+vwn5+p86", -75.339256176159, 0.0},
        {"bhandhlyp", "hybrid c_HF=0.50, slater+B88+lyp", -75.289575495135, -84.029130958034},
        // The unverified-by-construction functional. pyscf's name lookup
        // fails for MPW1PW91 in this libxc build, so the reference is the
        // EXPLICIT primitives - 0.25*HF + 0.75*GGA_X_MPW91 + GGA_C_PW91 - which
        // is the recipe the registry carries (kMPw1Pw91, mPW91 x 0.75 + PW91 c
        // 1.0). The dE below is therefore read against the recipe's own
        // definition rather than against a name. Measured 2026-09-18.
        {"mpw1pw91", "hybrid c_HF=0.25, mPW91 x + pw91_c, no LSDA base", -75.306794097420, 0.0},
        {"vwn5", "LDA correlation alone", -66.603506311888, 0.0},
        {"pbe_c", "GGA correlation alone (not LYP)", -66.273539042771, 0.0},
        {"lyp", "GGA correlation alone (LYP)", -66.271396323361, 0.0},
    };

    for (const Row& row : rows)
    {
        const auto record = RunRks(*inputs, row.name, 1e-8, 1e-6, 75, 302);
        ASSERT_TRUE(record.has_value()) << row.name << ": " << record.error().message;

        char label[128] = {};
        std::snprintf(label, sizeof(label), "%s (%s)", row.name, row.note);
        PrintRow(label, *record);

        std::printf("%-12s pyscf          %+.12f   dE = %+.6e\n",
                    row.name,
                    row.reference,
                    record->energy - row.reference);

        if (row.registryRecipe != 0.0)
        {
            // The PRE-FIX recipe (the note above the table), so this `dE` is
            // the size of the folding defect rather than an error of qcx's: it
            // collapses to ~0 if the unfolded weights are ever shipped again.
            std::printf("%-12s pyscf-as-prefix-qcx %+.12f   dE = %+.6e\n",
                        row.name,
                        row.registryRecipe,
                        record->energy - row.registryRecipe);
        }

        std::fflush(stdout);

        // Convergence only: pinning agreement before the gaps are read would
        // be assuming the answer this test exists to measure.
        EXPECT_TRUE(record->converged) << row.name;
    }
}

// ---------------------------------------------------------------------------
// 6. The hybrid agreement pin.
// ---------------------------------------------------------------------------

// The pin the family measurement above earns. It is deliberately on the two
// hybrids whose gaps are AT the floor, not on all four rows, because a pin has
// to be a claim the evidence supports:
//
//   b3lyp    +2.580e-06   (was -5.953167 before the 2026-09-13 registry fix)
//   bhandhlyp +1.782e-06  (was -8.739552)
//
// THE FLOOR, MEASURED NOT IMPORTED. Three independent measurements on this
// fixture bound how tight a qcx-vs-pyscf whole-SCF comparison can be:
//   (a) the probe's own grid shift, svwn (75,302) -> (150,434): 1.669e-06 Ha;
//   (b) the reference script's grid refinement, 1.15e-07 .. 3.5e-07 Ha;
//   (c) every functional measured here that both codes implement correctly:
//       slater +2.33e-06, becke88 +2.83e-06, svwn +2.6e-06, pbe0 +1.62e-06,
//       vwn5 +6.1e-07, pbe_c +7.6e-08, lyp +7.2e-07.
// The largest agreement gap observed anywhere is 2.83e-06, so the floor for
// this comparison is ~3e-06 and 1e-05 is the smallest round tolerance with
// real margin above it. The defect this pin guards against was 5.95 Ha -
// six orders larger - so nothing is being let through by the slack.
TEST(RksConvergenceProbe, HybridsAgreeWithPyscfAtTheGridFloor) {
    const auto inputs = BuildH2oInputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    struct Expectation {
        const char* name;
        double reference;
    };

    const Expectation expectations[] = {
        {"b3lyp", -75.275066425621},
        {"bhandhlyp", -75.289575495135},
    };

    constexpr double kGridFloorTolerance = 1e-5;

    for (const Expectation& expectation : expectations)
    {
        const auto record = RunRks(*inputs, expectation.name, 1e-8, 1e-6, 75, 302);
        ASSERT_TRUE(record.has_value()) << expectation.name << ": " << record.error().message;
        ASSERT_TRUE(record->converged) << expectation.name;

        const double gap = record->energy - expectation.reference;
        std::printf("%-12s E = %+.12f   pyscf %+.12f   dE = %+.6e  (tol %.0e)\n",
                    expectation.name,
                    record->energy,
                    expectation.reference,
                    gap,
                    kGridFloorTolerance);
        std::fflush(stdout);

        EXPECT_LT(std::abs(gap), kGridFloorTolerance)
            << expectation.name << " left the grid floor: " << gap;
    }
}

} // namespace
