// Wall time of the plain DiagonalizeFock
// vs the symmetry-blocked path at a mid-size molecule with real, non-C1
// symmetry (benzene: D6h detected, D2h computational group). The
// one-time BuildSymmetryBlocks + BuildBlockedDiagonalizeData cost is
// measured separately so the per-iteration saving can be weighed against
// the amortization.

#include "corpus.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/symmetry/detection.hpp"
#include "scf_common.hpp"
#include "symmetry_blocks.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

using qcx::testing::ToMatrix;

// STO-3G merged from the vendored corpus files (same helper the
// symmetry_block_test uses, so the benchmark runs on the identical
// benzene D6h -> D2h fixture).
qcx::Result<qcx::basisset::BasisSet> MakeSto3g(const std::vector<std::string_view>& symbols) {
    qcx::basisset::BasisSet merged;

    for (const std::string_view symbol : symbols)
    {
        const std::string path =
            std::string(QcxBasisDataDir) + "/sto-3g/" + std::string(symbol) + ".nwchem";
        auto parsed = qcx::basisset::ParseNwchemFile(path);

        if (!parsed.has_value())
        {
            return std::unexpected(parsed.error());
        }

        auto mergeResult = merged.Merge(*parsed);

        if (!mergeResult.has_value())
        {
            return std::unexpected(mergeResult.error());
        }
    }

    return merged;
}

struct Fixture {
    // Molecule is move-only, hence the heap copies.
    std::unique_ptr<qcx::molecule::Molecule> molecule;
    std::unique_ptr<qcx::basisset::BasisSet> basis;
    qcx::symmetry::SymmetryAnalysis analysis;
    Eigen::MatrixXd fock;
    Eigen::MatrixXd x;
    qcx::scf::internal::SymmetryBlocks blocks;
    qcx::scf::internal::BlockedDiagonalizeData data;
};

std::unique_ptr<Fixture> gFixture;

void PrepareFixture(benchmark::State& state) {
    if (gFixture != nullptr)
    {
        return;
    }

    auto molecule = qcx::symmetry::testing::MakeBenzene();
    auto basis = MakeSto3g({"C", "H"});

    if (!molecule.has_value() || !basis.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    const qcx::symmetry::SymmetryAnalysis analysis = qcx::symmetry::DetectPointGroup(*molecule);
    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);

    if (!overlap.has_value() || !kinetic.has_value() || !nuclear.has_value())
    {
        state.SkipWithError("one-electron matrix construction failed");
        return;
    }

    const Eigen::MatrixXd fock = ToMatrix(*kinetic) + ToMatrix(*nuclear);
    auto xResult = qcx::scf::internal::OrthogonalizeOverlap(ToMatrix(*overlap));

    if (!xResult.has_value())
    {
        state.SkipWithError(xResult.error().message.c_str());
        return;
    }

    const Eigen::MatrixXd x = *xResult;
    auto blocks = qcx::scf::internal::BuildSymmetryBlocks(*molecule, *basis, analysis);

    if (!blocks.has_value() || blocks->isTrivial)
    {
        state.SkipWithError("benzene did not yield a non-trivial blocking");
        return;
    }

    auto data = qcx::scf::internal::BuildBlockedDiagonalizeData(x, *blocks);

    if (!data.has_value())
    {
        state.SkipWithError(data.error().message.c_str());
        return;
    }

    auto fixtureMolecule = std::make_unique<qcx::molecule::Molecule>(std::move(*molecule));
    auto fixtureBasis = std::make_unique<qcx::basisset::BasisSet>(std::move(*basis));

    gFixture = std::make_unique<Fixture>(Fixture{std::move(fixtureMolecule),
                                                 std::move(fixtureBasis),
                                                 analysis,
                                                 fock,
                                                 x,
                                                 std::move(*blocks),
                                                 std::move(*data)});
}

// The today path: one n x n SelfAdjointEigenSolver per iteration.
void BenchDiagonalizePlain(benchmark::State& state) {
    PrepareFixture(state);

    if (gFixture == nullptr)
    {
        return;
    }

    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        const Eigen::MatrixXd coefficients =
            qcx::scf::internal::DiagonalizeFock(gFixture->fock, gFixture->x);
        sink += coefficients(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

// The blocked path with prebuilt blocks: per-iteration cost only.
void BenchDiagonalizeBlocked(benchmark::State& state) {
    PrepareFixture(state);

    if (gFixture == nullptr)
    {
        return;
    }

    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        auto coefficients = qcx::scf::internal::DiagonalizeFockBlocked(
            gFixture->fock, gFixture->x, gFixture->blocks, gFixture->data);

        if (!coefficients.has_value())
        {
            state.SkipWithError(coefficients.error().message.c_str());
            return;
        }

        sink += (*coefficients)(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

// The one-time build (DetectPointGroup + BuildSymmetryBlocks +
// BuildBlockedDiagonalizeData) that the per-iteration saving has to
// amortize against.
void BenchSymmetryBlocksBuild(benchmark::State& state) {
    PrepareFixture(state);

    if (gFixture == nullptr)
    {
        return;
    }

    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        const qcx::symmetry::SymmetryAnalysis analysis =
            qcx::symmetry::DetectPointGroup(*gFixture->molecule);
        auto blocks = qcx::scf::internal::BuildSymmetryBlocks(
            *gFixture->molecule, *gFixture->basis, analysis);

        if (!blocks.has_value())
        {
            state.SkipWithError(blocks.error().message.c_str());
            return;
        }

        auto data = qcx::scf::internal::BuildBlockedDiagonalizeData(gFixture->x, *blocks);

        if (!data.has_value())
        {
            state.SkipWithError(data.error().message.c_str());
            return;
        }

        sink += blocks->blockSizes[0] + data->sizes.size();
    }

    benchmark::DoNotOptimize(sink);
}

BENCHMARK(BenchDiagonalizePlain)->Unit(benchmark::kMillisecond)->MinTime(1.0);
BENCHMARK(BenchDiagonalizeBlocked)->Unit(benchmark::kMillisecond)->MinTime(1.0);
BENCHMARK(BenchSymmetryBlocksBuild)->Unit(benchmark::kMillisecond)->MinTime(1.0);

} // namespace

BENCHMARK_MAIN();
