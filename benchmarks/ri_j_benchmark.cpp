// The RI-J benchmark: RiJkFockBuilder::BuildFock against the direct J/K
// builder on H2O/def2-SVP with def2-universal-jfit - the RI path must
// reach at least 3x the direct one at this size.

#include "h2o_sto3g.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <filesystem>
#include <memory>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

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
    // The Create()-time inputs kept alive so the benchmark can re-run the
    // metric-factorization path: the one-shot cost of the RI-J metric solve,
    // reported separately from the per-iteration BuildFock wall time.
    // Molecule is move-only, hence the heap copies.
    std::unique_ptr<qcx::molecule::Molecule> molecule;
    std::unique_ptr<qcx::basisset::BasisSet> basis;
    std::unique_ptr<qcx::basisset::BasisSet> aux;
    CpuTensor2 core;
    CpuTensor2 densityTensor;
    qcx::integrals::DirectJkFockBuilder direct;
    qcx::integrals::RiJkFockBuilder ri;
};

std::unique_ptr<Fixture> gFixture;

void PrepareFixture(benchmark::State& state) {
    if (gFixture != nullptr)
    {
        return;
    }

    if (!qcx::integrals::SupportsL(4))
    {
        state.SkipWithError("this build's kMaxEngineL is below the jfit g shells");
        return;
    }

    auto molecule = MakeH2oSto3g();
    const std::filesystem::path root(QcxBasisDataDir);
    auto basis = qcx::basisset::ParseNwchemDirectory((root / "def2-svp").string());
    auto aux = qcx::basisset::ParseNwchemDirectory((root / "def2-universal-jfit").string());

    if (!molecule.has_value() || !basis.has_value() || !aux.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        state.SkipWithError(core.error().message.c_str());
        return;
    }

    const std::size_t n = core->Shape()[0];
    Eigen::MatrixXd density =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        density(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = 0.5;
    }

    auto densityTensor = ToTensor(density);

    if (!densityTensor.has_value())
    {
        state.SkipWithError("density tensor preparation failed");
        return;
    }

    // The fixture keeps the Create() inputs alive (what the re-Create loop
    // runs on) alongside the ready-built builders; the builders are
    // constructed from the fixture's own copies.
    auto fixtureMolecule = std::make_unique<qcx::molecule::Molecule>(std::move(*molecule));
    auto fixtureBasis = std::make_unique<qcx::basisset::BasisSet>(std::move(*basis));
    auto fixtureAux = std::make_unique<qcx::basisset::BasisSet>(std::move(*aux));

    qcx::integrals::FockBuildOptions directOptions;
    directOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    auto direct = qcx::integrals::DirectJkFockBuilder::Create(
        *fixtureMolecule, *fixtureBasis, *core, directOptions);

    qcx::integrals::RiEngineOptions riOptions;
    riOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    auto ri = qcx::integrals::RiJkFockBuilder::Create(
        *fixtureMolecule, *fixtureBasis, *fixtureAux, *core, riOptions);

    if (!direct.has_value() || !ri.has_value())
    {
        state.SkipWithError("builder preparation failed");
        return;
    }

    gFixture = std::make_unique<Fixture>(Fixture{std::move(fixtureMolecule),
                                                 std::move(fixtureBasis),
                                                 std::move(fixtureAux),
                                                 std::move(*core),
                                                 std::move(*densityTensor),
                                                 std::move(*direct),
                                                 std::move(*ri)});
}

void BenchFockDirect(benchmark::State& state) {
    PrepareFixture(state);

    if (gFixture == nullptr)
    {
        return;
    }

    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        auto fock = gFixture->direct.BuildFock(gFixture->densityTensor);

        if (!fock.has_value())
        {
            state.SkipWithError(fock.error().message.c_str());
            return;
        }

        sink += (*fock)(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

void BenchFockRiJ(benchmark::State& state) {
    PrepareFixture(state);

    if (gFixture == nullptr)
    {
        return;
    }

    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        auto fock = gFixture->ri.BuildFock(gFixture->densityTensor);

        if (!fock.has_value())
        {
            state.SkipWithError(fock.error().message.c_str());
            return;
        }

        sink += (*fock)(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

// The one-shot Create() cost (aux metric build + the factorization whose
// epsilon floor is a per-engine runtime setting), isolated from the
// per-iteration BuildFock cost above. This is the number the
// "is the eigendecomposition worth replacing" decision rests on.
void BenchRiJCreate(benchmark::State& state) {
    PrepareFixture(state);

    if (gFixture == nullptr)
    {
        return;
    }

    qcx::integrals::RiEngineOptions riOptions;
    riOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        auto ri = qcx::integrals::RiJkFockBuilder::Create(
            *gFixture->molecule, *gFixture->basis, *gFixture->aux, gFixture->core, riOptions);

        if (!ri.has_value())
        {
            state.SkipWithError(ri.error().message.c_str());
            return;
        }

        sink += ri->RiMatrix()(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

BENCHMARK(BenchFockDirect)->Unit(benchmark::kMillisecond)->MinTime(2.0);
BENCHMARK(BenchFockRiJ)->Unit(benchmark::kMillisecond)->MinTime(2.0);
BENCHMARK(BenchRiJCreate)->Unit(benchmark::kMillisecond)->MinTime(1.0);

} // namespace

BENCHMARK_MAIN();
