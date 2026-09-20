// The acceptance benchmark for the merged fp64/fp32 Fock fork and the
// 64-byte chunk alignment: wall-time of a direct Fock build on the certified
// mixed-precision lane (kLoose carries the most permissive fp32 gate), plus
// an exact (%.17g) Fock-matrix dump for the bit-level before/after checks
// across the two-pass -> merged -> aligned chunking states. Plain main()
// program, not a google-benchmark run: run it with identical flags in each
// build state and diff the dumps. The min iteration time is the robust
// number to compare - the mean includes the first-call warmup.
//
// Usage: qcx-bench-fock-parallel [--iterations N] [--serial] [--dump FILE]

#include "h2o_sto3g.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

struct Fixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    Eigen::MatrixXd core;
    CpuTensor2 density;
    qcx::integrals::DirectJkFockBuilder builder;
};

// H2O/def2-SVP with a fixed diagonal density: the same deterministic input
// the md_fock_benchmark uses, so every build state sees identical work. The
// fixture is s/p-dominated (plus one O d shell), the many-small-class
// regime where boundary false sharing between adjacent chunks shows up
// at all.
bool BuildFixture(std::unique_ptr<Fixture>& out, bool serial) {
    auto molecule = MakeH2oSto3g();
    const std::filesystem::path root(QcxBasisDataDir);
    auto basis = qcx::basisset::ParseNwchemDirectory((root / "def2-svp").string());

    if (!molecule.has_value() || !basis.has_value())
    {
        return false;
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);

    if (!kinetic.has_value())
    {
        return false;
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);

    if (!nuclear.has_value())
    {
        return false;
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

    Eigen::MatrixXd densityMatrix =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        densityMatrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = 0.5;
    }

    auto density = ToTensor(densityMatrix);

    if (!density.has_value())
    {
        return false;
    }

    qcx::integrals::FockBuildOptions options;
    // kLoose: the permissive MixedPrecisionThreshold routes a meaningful
    // share of quartets through the certified fp32 lane - the regime the
    // merged fp64/fp32 fork exists for.
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;

    if (serial)
    {
        // The serial reference lane: force the deterministic single-chunk
        // fallback (0/absent = auto, the parallel path).
        options.maxParallelChunks = 1;
    }

    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *kinetic, options);

    if (!builder.has_value())
    {
        return false;
    }

    out = std::make_unique<Fixture>(Fixture{std::move(*molecule),
                                            std::move(*basis),
                                            ToMatrix(*kinetic),
                                            std::move(*density),
                                            std::move(*builder)});
    return true;
}

} // namespace

int main(int argc, char** argv) {
    int iterations = 200;
    bool serial = false;
    std::string dumpPath;

    for (int a = 1; a < argc; ++a)
    {
        const std::string arg(argv[a]);

        if (arg == "--serial")
        {
            serial = true;
            continue;
        }

        if (arg == "--iterations" && a + 1 < argc)
        {
            iterations = std::atoi(argv[++a]);
            continue;
        }

        if (arg == "--dump" && a + 1 < argc)
        {
            dumpPath = argv[++a];
        }
    }

    std::unique_ptr<Fixture> fixture;

    if (!BuildFixture(fixture, serial))
    {
        std::fprintf(stderr, "fixture construction failed\n");
        return 1;
    }

    const Eigen::Index n = static_cast<Eigen::Index>(fixture->core.rows());
    std::vector<double> msPerIteration;
    msPerIteration.reserve(static_cast<std::size_t>(iterations));
    Eigen::MatrixXd lastFock;

    for (int i = 0; i < iterations; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        auto fock = fixture->builder.BuildFock(fixture->density);

        if (!fock.has_value())
        {
            std::fprintf(stderr, "BuildFock failed: %s\n", fock.error().message.c_str());
            return 1;
        }

        const auto stop = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
        msPerIteration.push_back(ms);
        lastFock = ToMatrix(*fock);
    }

    double totalMs = 0.0;
    double minMs = msPerIteration.front();
    double maxMs = msPerIteration.front();

    for (double ms : msPerIteration)
    {
        totalMs += ms;

        if (ms < minMs)
        {
            minMs = ms;
        }

        if (ms > maxMs)
        {
            maxMs = ms;
        }
    }

    std::printf("preset=kLoose fixture=H2O/def2-SVP serial=%s iterations=%d\n",
                serial ? "yes" : "no",
                iterations);
    std::printf("total_ms=%.3f mean_ms=%.3f min_ms=%.3f max_ms=%.3f\n",
                totalMs,
                totalMs / static_cast<double>(iterations),
                minMs,
                maxMs);

    if (!dumpPath.empty())
    {
        std::ofstream dump(dumpPath);

        if (!dump.is_open())
        {
            std::fprintf(stderr, "cannot open dump file %s\n", dumpPath.c_str());
            return 1;
        }

        dump.precision(17);

        for (Eigen::Index i = 0; i < n; ++i)
        {
            for (Eigen::Index j = 0; j < n; ++j)
            {
                if (j > 0)
                {
                    dump << " ";
                }

                dump << lastFock(i, j);
            }

            dump << "\n";
        }

        dump << lastFock.sum() << "\n";
    }

    return 0;
}
