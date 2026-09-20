// The preset-ladder recalibration sweep - the instrument of the theta/L
// re-derivation at the per-preset extents. For each (fixture,
// preset-tau, theta, L) cell the QFMM J-energy error vs the direct ground
// truth is measured together with the far-field pair count (a cell whose
// far field is empty meets its budget vacuously - the vacuous-ladder
// trap - so the count is printed per row) and the cost proxy nFarPairs x
// (L+1)^2 (the M2L blocks per interaction - the metric a ladder rung is
// picked by). The sweep is fp64-serial and deterministic - one run per
// cell, no timing: it is not a timed benchmark, so it needs no
// interference-free window (the optional wall-time confirmation of the
// chosen ladder is a separate mode).
//
// Grid: theta x {1.05, 0.85, 0.7, 0.55, 0.45, 0.35, 0.3} (the committed
// ladder's thetas IN the grid - they are its rungs and the fallback rungs
// if the design thetas turn out far-dead), L x {0..kQfmmMaxLMult} (0/1
// include the incumbent cells (1.05, 0) and (0.85, 1) so they reproduce
// in-run), presets {kLoose, kNormal, kTight} (the per-preset extent taus
// 1e-6 / 1e-8 / 1e-10, printed per row from QfmmExtentForPreset) = 63
// cells per preset per fixture once the translation tables reach L = 8
// (28 cells today - the current cap is 3; the grid self-adapts via
// kQfmmMaxLMult, and the fit-input CSV is RE-TAKEN once the cap lands).
// The designed "L x {2..10}" became {0..8}: the regenerated tables stop
// at L = 8, not 10.
//
// Measurement convention (the qfmm_fock_build_test.cpp
// PresetLadderDecreasesStrictlyAndStaysInBudget convention, so the gate
// below checks the incumbent pins IN-RUN): the QFMM side is the
// real QfmmJBuilder at fp64-only (useCertifiedMixedPrecision = false)
// with the serial pin (maxParallelChunks = 1) - options.accuracy drives
// the extent (the per-preset tau) and the near-field screening; theta and
// lMult are the cell's grid values. The direct ground truth is the
// fixture's BuildDirectFock (qfmm_fixture.hpp) computed ONCE per
// fixture at its default options (kNormal screening, serial pin, fp64,
// Coulomb-only), exactly as the test's reference. The J-energy error is
// the test's EnergyError (qfmm_fock_build_test.cpp:134): 0.5 * |sum_ij
// D_ij (F_qfmm - F_direct)_ij| in Eh - the metric the J-energy budgets
// are stated in - plus the relative Frobenius error. The density is the
// fixture's PhysicalDensity (seed 20260817, diagonal 0.5 + U(-0.25,
// 0.25), U(-0.25, 0.25) everywhere else, symmetrized), fixed per fixture.
//
// Fixtures: H2O/STO-3G first (the all-near degenerate control - every
// cell must give error ~0), then C12H26 / C24H50 / STO-3G (the
// budget-tooth and production-pin chains); C24H50/def2-SVP is the
// optional leg behind --def2-svp (the wide-exponent-spread probe). C60 is
// OUT of the pack entirely (the 16 GiB cap + compact-3D far-dead
// degeneracy - a compact cluster has no far field to truncate).
//
// Local-only (never CI) like the other benchmarks; every launch goes
// through tools/bench/memory_gate.py (16 GiB job cap, 12 GiB free
// floor). Usage:
//   qcx-bench-qfmm-ladder-sweep [--c12-only] [--def2-svp] [--out FILE]
// The CSV rows go to stdout (unitbuf, so the per-cell progress survives a
// crash) and, with --out, to FILE as well (the recalibration's committed
// evidence). One row per cell:
//   fixture,preset,tau,theta,L,jEnergyEh,relFrobenius,farPairs,costProxy,note
// e.g. C12H26,kLoose,1.000000e-06,1.05,0,4.524670e-01,3.4567e-03,1123,1123,
//
// The PASS gate is ENFORCED (not printed-only): after the C12 leg the
// binary compares the three incumbent cells against the re-measured
// values at full precision - (1.05, 0)@kLoose -> 0.45246731598,
// (0.85, 1)@kNormal -> 0.12946065326, (0.7, 2)@kTight -> 0.00014472812844
// Eh (the sweep is fp64-serial deterministic, so the full-precision
// values reproduce bit-identically and the +-1e-9 is the same-bits
// tolerance) - and every H2O cell < 1e-9. Prints PASS/FAIL and exits
// nonzero on a mismatch - stop and reconcile before any further run.

#include "alkane_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "internal/qfmm_tables_gen.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/qfmm_fock_build.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::MakeAlkaneSto3gBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// The theta rungs of the sweep grid: the committed ladder's thetas
// {1.05, 0.85, 0.7} IN the grid, then the design rungs
// {0.55, 0.45, 0.35, 0.3} - 7 rungs x (L 0..kQfmmMaxLMult) = 63 cells
// per preset once the translation tables reach L = 8 (28 today).
constexpr std::array<double, 7> kSweptThetas = {1.05, 0.85, 0.7, 0.55, 0.45, 0.35, 0.3};

// The presets whose extent taus {1e-6, 1e-8, 1e-10} are swept (the
// per-preset extent mapping). tau is FIXED per preset - the sweep never
// varies it.
constexpr std::array<qcx::integrals::AccuracyPreset, 3> kSweptPresets = {
    qcx::integrals::AccuracyPreset::kLoose,
    qcx::integrals::AccuracyPreset::kNormal,
    qcx::integrals::AccuracyPreset::kTight};

constexpr std::array<std::string_view, 3> kPresetNames = {"kLoose", "kNormal", "kTight"};

// The fixture's density shape (qfmm_fixture.hpp PhysicalDensity, copied
// benchmark-locally): 0.5 + U(-0.25, 0.25) on the diagonal, U(-0.25,
// 0.25) everywhere else, symmetrized - the physical-shaped probe the
// accuracy gates use (every shell pair's max block lands in [0.25,
// 0.5]; a diag-only density would zero entire density-screened classes).
Eigen::MatrixXd PhysicalDensity(std::size_t n) {
    std::mt19937_64 rng(20260817);
    std::uniform_real_distribution<double> dist(-0.25, 0.25);
    Eigen::MatrixXd d(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = (i == j) ? 0.5 + dist(rng) : dist(rng);
            d(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
            d(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = value;
        }
    }

    return d;
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

struct ChainFixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    CpuTensor2 coreTensor;
    Eigen::MatrixXd core;
    Eigen::MatrixXd density;
    std::string label;
};

std::unique_ptr<ChainFixture> MakeChainFixture(std::size_t carbonCount) {
    auto molecule = MakeAlkaneSto3g(carbonCount);

    if (!molecule.has_value())
    {
        std::cerr << "MakeChainFixture(" << carbonCount
                  << "): molecule failed: " << molecule.error().message << "\n";
        return nullptr;
    }

    auto basis = MakeAlkaneSto3gBasis();

    if (!basis.has_value())
    {
        std::cerr << "MakeChainFixture(" << carbonCount
                  << "): basis failed: " << basis.error().message << "\n";
        return nullptr;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        std::cerr << "MakeChainFixture(" << carbonCount
                  << "): core failed: " << core.error().message << "\n";
        return nullptr;
    }

    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd coreMatrix = ToMatrix(*core);
    // The core matrix is materialized BEFORE the moves: the aggregate's
    // clauses evaluate left-to-right, so an in-place ToMatrix(*core)
    // after std::move(*core) would read the moved-from tensor.
    return std::make_unique<ChainFixture>(ChainFixture{std::move(*molecule),
                                                       std::move(*basis),
                                                       std::move(*core),
                                                       coreMatrix,
                                                       PhysicalDensity(n),
                                                       "C" + std::to_string(carbonCount) + "H" +
                                                           std::to_string(2 * carbonCount + 2)});
}

std::unique_ptr<ChainFixture> MakeH2oControlFixture() {
    auto molecule = MakeH2oSto3g();

    if (!molecule.has_value())
    {
        std::cerr << "H2O fixture: molecule failed: " << molecule.error().message << "\n";
        return nullptr;
    }

    auto basis = MakeH2oSto3gBasis();

    if (!basis.has_value())
    {
        std::cerr << "H2O fixture: basis failed: " << basis.error().message << "\n";
        return nullptr;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        std::cerr << "H2O fixture: core failed: " << core.error().message << "\n";
        return nullptr;
    }

    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd coreMatrix = ToMatrix(*core);
    return std::make_unique<ChainFixture>(ChainFixture{std::move(*molecule),
                                                       std::move(*basis),
                                                       std::move(*core),
                                                       coreMatrix,
                                                       PhysicalDensity(n),
                                                       "H2O"});
}

// The optional def2-SVP leg: C24H50 with the def2-SVP basis (the wider
// exponent-spread regime - smaller falloff radii at equal tau, more far
// pairs). The same molecule as the STO-3G C24 leg, basis parsed from
// the data directory.
std::unique_ptr<ChainFixture> MakeDef2SvpFixture() {
    auto molecule = MakeAlkaneSto3g(24);

    if (!molecule.has_value())
    {
        std::cerr << "def2-SVP fixture: molecule failed: " << molecule.error().message << "\n";
        return nullptr;
    }

    const std::filesystem::path root(QcxBasisDataDir);
    auto basis = qcx::basisset::ParseNwchemDirectory((root / "def2-svp").string());

    if (!basis.has_value())
    {
        std::cerr << "def2-SVP fixture: basis failed: " << basis.error().message << "\n";
        return nullptr;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        std::cerr << "def2-SVP fixture: core failed: " << core.error().message << "\n";
        return nullptr;
    }

    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd coreMatrix = ToMatrix(*core);
    return std::make_unique<ChainFixture>(ChainFixture{std::move(*molecule),
                                                       std::move(*basis),
                                                       std::move(*core),
                                                       coreMatrix,
                                                       PhysicalDensity(n),
                                                       "C24H50-def2svp"});
}

// The J-energy error metric of the preset-ladder gate (qfmm_fock_build_
// test.cpp:134 EnergyError - the metric the J-energy budgets are stated
// in, Eh): 0.5 * |sum_ij
// D_ij (F_qfmm - F_direct)_ij|. Both Fock matrices are in the H + 2J(rho)
// convention, so the delta is 2 * (J_qfmm - J_direct) and the 1/2 maps
// it to the J energy.
double EnergyErrorEh(const Eigen::MatrixXd& qfmm,
                     const Eigen::MatrixXd& direct,
                     const Eigen::MatrixXd& density) {
    const Eigen::MatrixXd delta = qfmm - direct;
    return 0.5 * std::abs((density.cwiseProduct(delta)).sum());
}

// The direct ground truth: the fixture's BuildDirectFock convention
// (qfmm_fixture.hpp) - the unrestricted direct Coulomb builder at its
// DEFAULT options (kNormal screening, fp64-only, serial pin, Coulomb-
// only), computed once per fixture. The QFMM near-field halves cancel
// exactly against it only at kNormal; at kLoose/kTight the measured
// error additionally carries the preset's screening-truncation delta -
// the committed convention the incumbent pins were measured in (the
// multipole term dominates either way).
qcx::Result<Eigen::MatrixXd> BuildDirectReference(const ChainFixture& fixture) {
    qcx::integrals::FockBuildOptions options;
    options.useCertifiedMixedPrecision = false;
    options.buildCoulombOnly = true;
    options.maxParallelChunks = 1;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(
        fixture.molecule, fixture.basis, fixture.coreTensor, options);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    auto densityTensor = ToTensor(fixture.density);

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
}

// One QFMM grid cell: the real builder at (preset, theta, L) - the
// qfmm_fock_build_test.cpp:85-113 convention (the builder under test,
// not the fixture harness), fp64-only and serially pinned. Fills the
// record on success.
struct CellRecord {
    std::size_t farPairs = 0;
    double jEnergyEh = 0.0;
    double relFrobenius = 0.0;
    double costProxy = 0.0;
    bool ok = false;
};

bool RunCell(const ChainFixture& fixture,
             qcx::integrals::AccuracyPreset preset,
             double theta,
             int lMult,
             const Eigen::MatrixXd& direct,
             CellRecord& record) {
    qcx::integrals::QfmmOptions options;
    options.accuracy = preset;
    options.theta = theta;
    options.lMult = lMult;
    options.useCertifiedMixedPrecision = false;
    options.maxParallelChunks = 1;
    auto builder = qcx::integrals::QfmmJBuilder::Create(
        fixture.molecule, fixture.basis, fixture.coreTensor, options);

    if (!builder.has_value())
    {
        std::cerr << "QFMM Create failed: " << builder.error().message << "\n";
        return false;
    }

    auto densityTensor = ToTensor(fixture.density);

    if (!densityTensor.has_value())
    {
        std::cerr << "density tensor failed\n";
        return false;
    }

    auto fock = builder->BuildFock(*densityTensor);

    if (!fock.has_value())
    {
        std::cerr << "QFMM BuildFock failed: " << fock.error().message << "\n";
        return false;
    }

    const Eigen::MatrixXd qfmm = ToMatrix(*fock);
    record.farPairs = builder->FarFieldPairCount();
    record.jEnergyEh = EnergyErrorEh(qfmm, direct, fixture.density);
    record.relFrobenius = (qfmm - direct).norm() / direct.norm();
    // The cost proxy of the rung pick: the M2L blocks per interaction.
    record.costProxy = static_cast<double>(record.farPairs) * (lMult + 1) * (lMult + 1);
    record.ok = true;
    return true;
}

// The gate's captured values: the three incumbent cells of C12 and
// the largest H2O error (the far-dead control must stay ~0 everywhere).
struct S1Pins {
    std::optional<double> kLoose1050; // (1.05, 0) @ kLoose.
    std::optional<double> kNormal0851; // (0.85, 1) @ kNormal.
    std::optional<double> kTight0702; // (0.7, 2) @ kTight.
    double maxH2oError = 0.0; // The H2O leg's largest |jEnergyEh|.
};

// Runs one fixture's whole grid. Failure rows go to BOTH stdout and the
// CSV (the fit must distinguish a partial table from a complete one),
// and a single failed cell aborts the fixture with false - the caller
// turns that into exit 1.
bool RunFixture(const ChainFixture& fixture, std::ostream& csvOut, S1Pins& pins) {
    std::cout << "fixture: " << fixture.label << "  n = " << fixture.core.rows()
              << "  L cap kQfmmMaxLMult = " << qcx::integrals::internal::kQfmmMaxLMult << "\n";

    const auto direct = BuildDirectReference(fixture);

    if (!direct.has_value())
    {
        std::cerr << fixture.label << ": direct reference failed: " << direct.error().message
                  << "\n";
        return false;
    }

    for (std::size_t p = 0; p < kSweptPresets.size(); ++p)
    {
        const double tau = qcx::integrals::QfmmExtentForPreset(kSweptPresets[p]);

        for (const double theta : kSweptThetas)
        {
            for (int lMult = 0; lMult <= qcx::integrals::internal::kQfmmMaxLMult; ++lMult)
            {
                CellRecord record;
                const bool ran = RunCell(fixture, kSweptPresets[p], theta, lMult, *direct, record);

                if (!ran)
                {
                    std::ostringstream row;
                    row << std::scientific << std::setprecision(10) << fixture.label << ","
                        << kPresetNames[p] << "," << tau << "," << theta << "," << lMult
                        << ",--,--,--,--,CELL_FAILED\n";
                    std::cout << row.str();
                    csvOut << row.str();
                    csvOut.flush();
                    return false;
                }

                // Scientific with 10 significant digits: the gate
                // reproduces the incumbent pins to +-1e-9, and
                // std::to_string would round tau (1e-8) to "0.000000".
                const bool farDead = record.farPairs == 0;
                std::ostringstream row;
                row << std::scientific << std::setprecision(10) << fixture.label << ","
                    << kPresetNames[p] << "," << tau << "," << theta << "," << lMult << ","
                    << record.jEnergyEh << "," << record.relFrobenius << "," << record.farPairs
                    << "," << record.costProxy << "," << (farDead ? "FAR-DEAD" : "") << "\n";
                std::cout << row.str();
                csvOut << row.str();
                csvOut.flush();

                // The gate captures: the incumbent cells of C12 and the
                // H2O control's largest error.
                if (fixture.label == "C12H26")
                {
                    const qcx::integrals::AccuracyPreset preset = kSweptPresets[p];

                    if (preset == qcx::integrals::AccuracyPreset::kLoose && theta == 1.05 &&
                        lMult == 0)
                    {
                        pins.kLoose1050 = record.jEnergyEh;
                    } else if (preset == qcx::integrals::AccuracyPreset::kNormal && theta == 0.85 &&
                               lMult == 1)
                    {
                        pins.kNormal0851 = record.jEnergyEh;
                    } else if (preset == qcx::integrals::AccuracyPreset::kTight && theta == 0.7 &&
                               lMult == 2)
                    { pins.kTight0702 = record.jEnergyEh; }
                } else if (fixture.label == "H2O")
                { pins.maxH2oError = std::max(pins.maxH2oError, std::abs(record.jEnergyEh)); }
            }
        }
    }

    return true;
}

// The PASS gate, ENFORCED: the three incumbent cells must reproduce the
// re-measured values at full precision and every H2O cell must stay
// below 1e-9. Returns 0 on pass; prints the failing rows and returns 1
// on mismatch - stop and reconcile before any further run.
int CheckS1Pins(const S1Pins& pins) {
    bool ok = true;
    const auto check = [&ok](const char* cell, std::optional<double> measured, double pin) {
        if (!measured.has_value())
        {
            std::cerr << "ladder gate FAIL: " << cell << " was never measured\n";
            ok = false;
        } else if (std::abs(*measured - pin) > 1e-9)
        {
            std::cerr << "ladder gate FAIL: " << cell << " " << std::scientific
                      << std::setprecision(10) << *measured << " vs pin " << pin
                      << " (diff > 1e-9)\n";
            ok = false;
        }
    };

    check("(1.05, 0)@kLoose", pins.kLoose1050, 4.5246731598e-01);
    check("(0.85, 1)@kNormal", pins.kNormal0851, 1.2946065326e-01);
    check("(0.7, 2)@kTight", pins.kTight0702, 1.4472812844e-04);

    if (pins.maxH2oError >= 1e-9)
    {
        std::cerr << "ladder gate FAIL: the H2O control's largest error " << std::scientific
                  << std::setprecision(10) << pins.maxH2oError << " (must be < 1e-9)\n";
        ok = false;
    }

    if (ok)
    {
        std::cout << "ladder gate PASS: the incumbent C12 pins and the H2O control reproduce\n";
    }

    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    // Unbuffered stdout: the C24 legs run tens of minutes and the
    // per-cell progress must survive a crash.
    std::cout << std::unitbuf;

    bool c12Only = false;
    bool includeDef2Svp = false;
    std::ofstream csvFile;

    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg(argv[i]);

        if (arg == "--c12-only")
        {
            c12Only = true;
        } else if (arg == "--def2-svp")
        {
            includeDef2Svp = true;
        } else if (arg == "--out")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--out needs a path\n";
                return 1;
            }

            csvFile.open(argv[++i]);

            if (!csvFile)
            {
                std::cerr << "cannot open " << argv[i] << " for writing\n";
                return 1;
            }
        } else
        {
            std::cerr << "unknown argument " << arg << "\n";
            return 1;
        }
    }

    std::ostream& csvOut = csvFile.is_open() ? static_cast<std::ostream&>(csvFile)
                                             : static_cast<std::ostream&>(std::cout);

    std::cout << "QFMM preset-ladder recalibration sweep\n"
              << "thetas {1.05, 0.85, 0.7, 0.55, 0.45, 0.35, 0.3}, L {0..kQfmmMaxLMult}, "
                 "presets {kLoose, kNormal, kTight}\n";
    csvOut << "fixture,preset,tau,theta,L,jEnergyEh,relFrobenius,farPairs,costProxy,note\n";

    // RunFixture returns false on the first failed cell of a fixture
    // (the failure rows are already in the CSV) - a broken cell makes the
    // whole sweep untrustworthy, so stop and reconcile rather than carry
    // a partial table into the fit.
    auto h2o = MakeH2oControlFixture();

    if (h2o == nullptr)
    {
        std::cerr << "H2O control fixture failed\n";
        return 1;
    }

    S1Pins pins;

    if (!RunFixture(*h2o, csvOut, pins))
    {
        return 1;
    }

    auto c12 = MakeChainFixture(12);

    if (c12 == nullptr)
    {
        std::cerr << "C12 fixture failed\n";
        return 1;
    }

    if (!RunFixture(*c12, csvOut, pins))
    {
        return 1;
    }

    // The gate, after the C12 leg: the three incumbent cells must
    // reproduce their re-measured values and the H2O control must stay
    // ~0. FAIL -> stop; reconcile before any further run.
    const int gate = CheckS1Pins(pins);

    if (gate != 0)
    {
        return 1;
    }

    if (c12Only)
    {
        return 0;
    }

    auto c24 = MakeChainFixture(24);

    if (c24 == nullptr)
    {
        std::cerr << "C24 fixture failed\n";
        return 1;
    }

    if (!RunFixture(*c24, csvOut, pins))
    {
        return 1;
    }

    if (includeDef2Svp)
    {
        auto def2 = MakeDef2SvpFixture();

        if (def2 == nullptr)
        {
            std::cerr << "C24H50/def2-SVP fixture failed\n";
            return 1;
        }

        if (!RunFixture(*def2, csvOut, pins))
        {
            return 1;
        }
    }

    return 0;
}
