// The qcx-internal crossover benchmark: per-iteration BuildFock wall time of the
// DirectJkFockBuilder / RiJkFockBuilder / QfmmJBuilder over the
// molecule family spanning ~7 -> ~840 basis functions, at every accuracy
// preset. The two outputs this exists for:
//
//   1. the direct-vs-RI-J crossover table,
//   2. the QFMM-vs-RI-J crossover table (the first real
//      measurement of the RecommendQfmmOverRiJ threshold).
//
// Rows are recorded, not asserted (the preset_sweep pattern). The one
// thing that IS asserted is the physics guard: H2O/STO-3G at kTight through
// the FockBuilderFn seam must reproduce the pinned energy -74.96292827
// (+-1e-5) or the benchmark refuses to produce tables (a timing
// benchmark whose fixture construction silently regressed the physics would
// produce beautiful meaningless numbers).
//
// Fairness notes: the RI-J and QFMM rows carry their real
// production work - the RI-J row includes the direct-exchange build (K is
// not part of the RI approximation), the QFMM row is Coulomb-only (the
// multipole far field has no exchange analog) and its density-weighted
// screening runs at the fixture's diagonal-0.5 rho. OMP_NUM_THREADS is read
// from the environment and recorded, exactly like the SCF driver.
//
// The far-field-liveness flag (the vacuous-ladder
// trap): the QFMM rows must report whether the octree's far field was live
// (far-pair count). The count comes from QfmmJBuilder::FarFieldPairCount()
// (integrals/qfmm_fock_build.{hpp,cpp}); the
// liveness column prints the real count on every row, and a zero count on
// a chain fixture is the vacuous-ladder alarm (MarkQfmmLiveness below).
//
// Local-only like every benchmark (QCX_BUILD_BENCHMARKS); plain main(),
// NOT a google-benchmark run.
//
// Usage:
//   qcx-bench-crossover [--no-timing] [--iterations N] [--fixtures a,b,...]
//                          [--presets a,b,...] [--memory-cap-gib N]
//   --no-timing: physics guard + fixture statistics only (basis counts,
//     aux counts) - no timing rows. Harness verification runs in
//     this mode so no timing claims are produced; the timed
//     runs belong to a gated timed pass.
//   --fixtures a,b: comma list of fixture names (default: all).
//   --presets a,b: comma list of the three preset names (kLoose/kNormal/
//     kTight; default: all three). A single-preset invocation gives each
//     builder a FRESH --memory-cap-gib budget: with
//     the CWA sequence, kNormal/kTight big-fixture Creates are refused on
//     the budget the earlier presets already consumed - a per-preset
//     invocation isolates the cell's own admission; additive, absent =
//     today's 3-preset sequence).
//   --memory-cap-gib N: wire ONE WorkspaceBudget of N GiB into every builder
//     Create (the adaptive-memory budget wiring): the fixed
//     per-fixture 3-preset x 3-builder sequence then charges it cumulatively
//     (CWA - Release never restores), so later Creates see cap minus
//     committedSoFar. That deterministic ordering is the regression surface
//     the acceptance runs. Absent = legacy: no budget, every Create
//     admits exactly as before.

#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/qfmm_fock_build.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/memory/workspace_budget.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// The test fixtures (tests/fixtures/*.hpp).
#include "alkane_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "large_molecules.hpp"

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::molecule::testing::MakeBuckminsterfullerene;
using qcx::molecule::testing::MakeZigzagNanotube;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::MakeAlkaneSto3gBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;

// The physics-guard pin (the harness self-test pins the same
// value, tools/bench/tests/test_harness.py).
constexpr double kH2oSto3gTightEnergy = -74.96292827;
constexpr double kH2oSto3gTightTolerance = 1e-5;

std::optional<std::size_t> gQfmmFarFieldPairs;

void MarkQfmmLiveness(const qcx::integrals::QfmmJBuilder& builder) {
    gQfmmFarFieldPairs = builder.FarFieldPairCount();
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

struct Fixture {
    std::string name;
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    qcx::basisset::BasisSet aux; // def2-universal-jfit for every fixture.
    CpuTensor2 core; // H = T + V (the builders' Create input).
    CpuTensor2 density; // The spatial rho (diagonal 0.5: the spin-summed convention).
    std::size_t nBasis = 0;
    std::size_t nAux = 0;
};

qcx::Result<std::unique_ptr<Fixture>> MakeFixture(std::string name,
                                                  qcx::Result<qcx::molecule::Molecule> molecule,
                                                  qcx::Result<qcx::basisset::BasisSet> basis,
                                                  const std::filesystem::path& basisRoot) {
    if (!molecule.has_value() || !basis.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, name + ": fixture construction failed"});
    }

    // The aux parse is filtered to the molecule's elements: the parsed set
    // is then exactly the molecule-scoped count, so the auxShells accounting
    // below sums the molecule's shells automatically - a directory-wide
    // parse would over-count elements the molecule does not contain. The
    // grid fixtures are C/H/O/N-only, so this never drops a needed element.
    std::vector<int> auxElements;
    auxElements.reserve(molecule->AtomCount());

    for (const qcx::molecule::Atom& atom : molecule->Atoms())
    {
        auxElements.push_back(atom.atomicNumber);
    }

    std::sort(auxElements.begin(), auxElements.end());
    auxElements.erase(std::unique(auxElements.begin(), auxElements.end()), auxElements.end());
    auto aux = qcx::basisset::ParseNwchemDirectoryFiltered(
        (basisRoot / "def2-universal-jfit").string(), auxElements);

    if (!aux.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, name + ": aux basis parse failed"});
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        return std::unexpected(core.error());
    }

    const std::size_t n = core->Shape()[0];
    Eigen::MatrixXd densityMat =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        densityMat(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = 0.5;
    }

    auto density = qcx::testing::ToTensor(densityMat);

    if (!density.has_value())
    {
        return std::unexpected(density.error());
    }

    std::size_t auxShells = 0;

    for (const auto& element : aux->Elements())
    {
        auxShells += element.shells.size();
    }

    return std::make_unique<Fixture>(Fixture{std::move(name),
                                             std::move(*molecule),
                                             std::move(*basis),
                                             std::move(*aux),
                                             std::move(*core),
                                             std::move(*density),
                                             n,
                                             auxShells});
}

std::vector<std::unique_ptr<Fixture>> gFixtures;
bool gUseTiming = true;
std::size_t gIterations = 5;

// The --presets filter (empty = all three, today's sequence): a non-empty
// list restricts RunFixture to those preset rows. Additive - the default
// run is unchanged.
std::vector<std::string> gPresetFilter;

// The budget wiring: when
// --memory-cap-gib is given, ONE WorkspaceBudget at the cap rides the whole
// process - every builder Create below receives it through the options seam
// (FockBuildOptions::workspaceBudget and the RI-J/QFMM twins). The fixed
// per-fixture 3-preset x 3-builder sequence charges it cumulatively (CWA -
// Release never restores Remaining), so a later Create sees cap minus
// committedSoFar: that deterministic ordering is the retention regression
// surface the acceptance runs. Unset = legacy: no budget, every Create
// admits exactly as today.
std::optional<qcx::memory::WorkspaceBudget> gWorkspaceBudget;
double gMemoryCapGib = 0.0;

qcx::memory::WorkspaceBudget* ActiveWorkspaceBudget() noexcept {
    return gWorkspaceBudget.has_value() ? &*gWorkspaceBudget : nullptr;
}

// The physics guard: H2O/STO-3G at kTight through the
// FockBuilderFn seam must reproduce the pinned energy. Returns false when
// the run failed or missed the pin - the caller then refuses to time.
bool RunPhysicsGuard() {
    const Fixture& fixture = *gFixtures[0]; // H2O/STO-3G is always first.

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kTight;
    // The pin rides the budget seam unchanged: H2O/STO-3G fits the fast path
    // at any real cap - LightPath is unreachable at the small end - so the
    // pinned energy must hold with a budget wired exactly as without one.
    options.workspaceBudget = ActiveWorkspaceBudget();
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(
        fixture.molecule, fixture.basis, fixture.core, options);

    if (!builder.has_value())
    {
        std::cerr << "physics guard: builder creation failed: " << builder.error().message << "\n";
        return false;
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(fixture.molecule, fixture.basis);

    if (!overlap.has_value())
    {
        std::cerr << "physics guard: overlap failed\n";
        return false;
    }

    const auto result = qcx::scf::RunRhfScf(
        fixture.molecule,
        qcx::testing::ToMatrix(*overlap),
        qcx::testing::ToMatrix(fixture.core),
        qcx::scf::RhfOptions{},
        [builder = *builder](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
            auto densityTensor = qcx::testing::ToTensor(0.5 * density);

            if (!densityTensor.has_value())
            {
                return std::unexpected(densityTensor.error());
            }

            auto fock = builder.BuildFock(*densityTensor);

            if (!fock.has_value())
            {
                return std::unexpected(fock.error());
            }

            return qcx::testing::ToMatrix(*fock);
        });

    if (!result.has_value())
    {
        std::cerr << "physics guard: scf failed: " << result.error().message << "\n";
        return false;
    }

    const double energy = result->totalEnergy;
    const bool passes = std::abs(energy - kH2oSto3gTightEnergy) <= kH2oSto3gTightTolerance;
    std::cout << "physics guard: H2O/STO-3G kTight E = " << std::setprecision(15) << energy
              << "  pin " << kH2oSto3gTightEnergy << " (+-" << kH2oSto3gTightTolerance << ") -> "
              << (passes ? "PASS" : "FAIL") << "\n";

    if (!passes)
    {
        std::cerr << "physics guard FAILED: no timing row may be produced "
                     "from a regressed fixture construction\n";
    }

    return passes;
}

template <typename Builder>
std::optional<double> TimeBuildFock(const Builder& builder,
                                    const Fixture& fixture,
                                    std::string_view label) {
    // One warm-up build (page-in, OpenMP team spin-up), then gIterations
    // timed builds; reports wall ms per iteration. Records only - the
    // gated timed runs are the numbers.
    auto warm = builder.BuildFock(fixture.density);

    if (!warm.has_value())
    {
        std::cerr << label << ": warm-up build failed: " << warm.error().message << "\n";
        return std::nullopt;
    }

    const auto start = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < gIterations; ++i)
    {
        auto fock = builder.BuildFock(fixture.density);

        if (!fock.has_value())
        {
            std::cerr << label << ": build " << i << " failed: " << fock.error().message << "\n";
            return std::nullopt;
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const double msPerIteration = std::chrono::duration<double, std::milli>(end - start).count() /
                                  static_cast<double>(gIterations);
    return msPerIteration;
}

void RunFixture(const Fixture& fixture) {
    const std::string_view presets[] = {"kLoose", "kNormal", "kTight"};
    const qcx::integrals::AccuracyPreset presetValues[] = {qcx::integrals::AccuracyPreset::kLoose,
                                                           qcx::integrals::AccuracyPreset::kNormal,
                                                           qcx::integrals::AccuracyPreset::kTight};

    std::cout << "\nfixture " << fixture.name << ": n = " << fixture.nBasis << " basis functions, "
              << fixture.nAux << " aux shells (jfit)\n";

    for (std::size_t p = 0; p < 3; ++p)
    {
        const qcx::integrals::AccuracyPreset preset = presetValues[p];

        if (!gPresetFilter.empty() &&
            std::find(gPresetFilter.begin(), gPresetFilter.end(), presets[p]) ==
                gPresetFilter.end())
        {
            continue;
        }

        std::optional<double> directMs, rijMs, qfmmMs;

        // One live builder at a time (the no-paging memory rule):
        // each builder is created, measured, and
        // destroyed inside its own block, so the peak footprint is the
        // largest single builder, never the sum. The RI-J builder owns the
        // n^2 x nAux Coulomb tensor (the dominating allocation), the QFMM
        // builder owns its octree plus an embedded near-field direct
        // builder, and the direct builder is recompute-only.

        // Direct (recompute-only: shell-pair tables and Schwarz bounds).
        {
            qcx::integrals::FockBuildOptions directOptions;
            directOptions.accuracy = preset;
            directOptions.workspaceBudget = ActiveWorkspaceBudget();
            auto direct = qcx::integrals::DirectJkFockBuilder::Create(
                fixture.molecule, fixture.basis, fixture.core, directOptions);

            if (!direct.has_value())
            {
                std::cerr << fixture.name << ": direct builder failed: " << direct.error().message
                          << "\n";
                continue;
            }

            if (gUseTiming)
            {
                directMs = TimeBuildFock(*direct, fixture, "direct");
            }
        }

        // RI-J (includes the direct-exchange build; the n^2 x nAux Coulomb
        // tensor lives only inside this block).
        {
            qcx::integrals::RiEngineOptions riOptions;
            riOptions.accuracy = preset;
            riOptions.workspaceBudget = ActiveWorkspaceBudget();
            auto rij = qcx::integrals::RiJkFockBuilder::Create(
                fixture.molecule, fixture.basis, fixture.aux, fixture.core, riOptions);

            if (!rij.has_value())
            {
                std::cerr << fixture.name << ": ri_j builder failed: " << rij.error().message
                          << "\n";
                continue;
            }

            if (gUseTiming)
            {
                rijMs = TimeBuildFock(*rij, fixture, "ri_j");
            }
        }

        // QFMM (Coulomb-only; embeds a near-field DirectJkFockBuilder).
        {
            qcx::integrals::QfmmOptions qfmmOptions;
            qfmmOptions.accuracy = preset;
            qfmmOptions.workspaceBudget = ActiveWorkspaceBudget();
            auto qfmm = qcx::integrals::QfmmJBuilder::Create(
                fixture.molecule, fixture.basis, fixture.core, qfmmOptions);

            if (!qfmm.has_value())
            {
                std::cerr << fixture.name << ": qfmm builder failed: " << qfmm.error().message
                          << "\n";
                continue;
            }

            MarkQfmmLiveness(*qfmm);

            if (gUseTiming)
            {
                qfmmMs = TimeBuildFock(*qfmm, fixture, "qfmm");
            }
        }

        if (!gUseTiming)
        {
            std::cout << "  " << presets[p] << ": builders created OK (no timing mode); "
                      << "qfmm far-field-pairs: "
                      << (gQfmmFarFieldPairs.has_value() ? std::to_string(*gQfmmFarFieldPairs)
                                                         : "n/a")
                      << "\n";
            continue;
        }

        std::cout << "  " << presets[p] << ":";
        std::cout << " direct " << std::setprecision(6)
                  << (directMs.has_value() ? *directMs : std::numeric_limits<double>::quiet_NaN())
                  << " ms/iter;";
        std::cout << " ri_j "
                  << (rijMs.has_value() ? *rijMs : std::numeric_limits<double>::quiet_NaN())
                  << " ms/iter;";
        std::cout << " qfmm "
                  << (qfmmMs.has_value() ? *qfmmMs : std::numeric_limits<double>::quiet_NaN())
                  << " ms/iter;";
        std::cout << " qfmm far-field-pairs: "
                  << (gQfmmFarFieldPairs.has_value() ? std::to_string(*gQfmmFarFieldPairs) : "n/a")
                  << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    bool runPhysicsGuardOnly = false;
    std::vector<std::string> fixtureFilter;

    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg(argv[i]);

        if (arg == "--no-timing")
        {
            runPhysicsGuardOnly = true;
        } else if (arg == "--iterations" && i + 1 < argc)
        {
            gIterations = std::max<std::size_t>(1, std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--fixtures" && i + 1 < argc)
        {
            const std::string_view list(argv[++i]);
            std::size_t start = 0;

            while (start < list.size())
            {
                const std::size_t comma = list.find(',', start);
                const std::size_t end = comma == std::string_view::npos ? list.size() : comma;

                if (end > start)
                {
                    fixtureFilter.emplace_back(list.substr(start, end - start));
                }

                start = comma == std::string_view::npos ? list.size() : comma + 1;
            }
        } else if (arg == "--presets" && i + 1 < argc)
        {
            const std::string_view list(argv[++i]);
            std::size_t start = 0;

            while (start < list.size())
            {
                const std::size_t comma = list.find(',', start);
                const std::size_t end = comma == std::string_view::npos ? list.size() : comma;

                if (end > start)
                {
                    const std::string name(list.substr(start, end - start));

                    if (name != "kLoose" && name != "kNormal" && name != "kTight")
                    {
                        std::cerr << "unknown preset " << name
                                  << " (expected kLoose/kNormal/kTight)\n";
                        return 1;
                    }

                    gPresetFilter.push_back(name);
                }

                start = comma == std::string_view::npos ? list.size() : comma + 1;
            }
        } else if (arg == "--memory-cap-gib" && i + 1 < argc)
        {
            const double gib = std::strtod(argv[++i], nullptr);

            if (!(gib > 0.0) || !std::isfinite(gib))
            {
                std::cerr << "--memory-cap-gib must be a finite positive number of GiB\n";
                return 1;
            }

            const std::size_t capacityBytes =
                static_cast<std::size_t>(std::round(gib * 1024.0 * 1024.0 * 1024.0));
            auto budget = qcx::memory::WorkspaceBudget::Create(capacityBytes);

            if (!budget.has_value())
            {
                std::cerr << "--memory-cap-gib: " << budget.error().message << "\n";
                return 1;
            }

            gWorkspaceBudget.emplace(std::move(*budget));
            gMemoryCapGib = gib;
        } else
        {
            std::cerr << "usage: qcx-bench-crossover [--no-timing] "
                         "[--iterations N] [--fixtures a,b,...] "
                         "[--presets kLoose,kNormal,kTight] "
                         "[--memory-cap-gib N]\n";
            return 1;
        }
    }

    // Line-flush stdout (the buffered-death lesson):
    // redirected stdout is fully buffered, so a cap death (fail-fast exit
    // 3221226505) discards every row since the last flush - unitbuf keeps
    // the per-row output observable up to the dying Create. Rows print once
    // per fixture/preset, so the flush cost is unmeasurable.
    std::cout << std::unitbuf;

    gUseTiming = !runPhysicsGuardOnly;

    const std::filesystem::path basisRoot(QcxBasisDataDir);

    // The crossover family plus the C12H26 chain,
    // smallest to largest. H2O/STO-3G is ALWAYS first - the physics guard
    // indexes gFixtures[0]. push_back, not list-init: the Result<unique_ptr>
    // values are move-only (initializer_list requires copyability).
    std::vector<qcx::Result<std::unique_ptr<Fixture>>> fixtures;
    fixtures.reserve(8);
    fixtures.push_back(MakeFixture("h2o_sto3g", MakeH2oSto3g(), MakeH2oSto3gBasis(), basisRoot));
    fixtures.push_back(
        MakeFixture("h2o_def2svp",
                    MakeH2oSto3g(),
                    qcx::basisset::ParseNwchemDirectory((basisRoot / "def2-svp").string()),
                    basisRoot));
    fixtures.push_back(
        MakeFixture("h2o_ccpvdz",
                    MakeH2oSto3g(),
                    qcx::basisset::ParseNwchemDirectory((basisRoot / "cc-pvdz").string()),
                    basisRoot));
    fixtures.push_back(
        MakeFixture("c12h26_sto3g", MakeAlkaneSto3g(12), MakeAlkaneSto3gBasis(), basisRoot));
    fixtures.push_back(
        MakeFixture("c60_sto3g", MakeBuckminsterfullerene(), MakeAlkaneSto3gBasis(), basisRoot));
    fixtures.push_back(
        MakeFixture("nt84_sto3g", MakeZigzagNanotube(8, 4), MakeAlkaneSto3gBasis(), basisRoot));
    fixtures.push_back(
        MakeFixture("c24h50_def2svp",
                    MakeAlkaneSto3g(24),
                    qcx::basisset::ParseNwchemDirectory((basisRoot / "def2-svp").string()),
                    basisRoot));
    fixtures.push_back(
        MakeFixture("c60_def2svp",
                    MakeBuckminsterfullerene(),
                    qcx::basisset::ParseNwchemDirectory((basisRoot / "def2-svp").string()),
                    basisRoot));

    for (auto& result : fixtures)
    {
        if (!result.has_value())
        {
            std::cerr << result.error().message << "\n";
            return 1;
        }

        gFixtures.push_back(std::move(*result));
    }

    const char* ompThreads = std::getenv("OMP_NUM_THREADS");
    std::cout << "Direct/RI-J/QFMM crossover benchmark: OMP_NUM_THREADS="
              << (ompThreads != nullptr ? ompThreads : "(unset)") << ", "
              << (gUseTiming ? "timing mode (" + std::to_string(gIterations) + " iterations)"
                             : "no-timing mode");

    if (gWorkspaceBudget.has_value())
    {
        std::cout << ", memory-cap-gib " << gMemoryCapGib << " ("
                  << gWorkspaceBudget->CapacityBytes() << " bytes, CWA across every Create)";
    }

    std::cout << "\n";

    if (!RunPhysicsGuard())
    {
        return 1;
    }

    if (fixtureFilter.empty())
    {
        for (const auto& fixture : gFixtures)
        {
            RunFixture(*fixture);
        }
    } else
    {
        for (const auto& wanted : fixtureFilter)
        {
            const auto found =
                std::find_if(gFixtures.begin(), gFixtures.end(), [&wanted](const auto& fixture) {
                    return fixture->name == wanted;
                });

            if (found == gFixtures.end())
            {
                std::cerr << "unknown fixture " << wanted << "\n";
                return 1;
            }

            RunFixture(**found);
        }
    }

    return 0;
}
