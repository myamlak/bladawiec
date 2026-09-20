// The overlap conditioning probe (linear-dependence lane, 2026-09-14): builds
// the AO overlap of a bench case file and reports its spectrum conditioning -
// s_min, s_max, the ratio s_min/s_max, and the count of eigenvalues strictly
// below kOverlapEigenvalueFloorTolerance * s_max.
//
// It exists because the SCF's linear-dependence removal (scf_common.hpp
// OrthogonalizeOverlap) is gated on that ratio, and the threshold is only as
// good as the measurement behind it. The 2026-09-14 sweep: def2-TZVPD sits at
// 9.50e-08 on C12H26 (678 BF) and 5.88e-08 on C24H50 (1338 BF) - both above
// the 1e-8 floor and well below a proposed 1e-7, which is what REFUTED that
// proposal - while aug-cc-pVTZ on C24H50 (2254 BF) sits at 4.92e-09 with one
// direction below the floor, the case the removal exists for. Re-run it before
// touching the threshold.
//
// No chemistry beyond the one-electron overlap; no SCF. Plain main() (the
// d31/r19 probe precedent).
//
//   qcx-bench-overlap-condition <case.toml> <basis-name> [<basis-name> ...]
//
// <case.toml> is a tools/bench/cases file; only its atoms= array is read.
// Basis names resolve under QcxBasisDataDir.
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/molecule/elements.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Dense>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kFloorTolerance = 1e-8;

struct ParsedCase {
    std::vector<qcx::molecule::Atom> atoms;
    std::vector<double> coordinatesBohr; // Row-major, 3 per atom.
    std::vector<int> atomicNumbers;
};

// Scans the atoms= array: repeated ["Sym", x, y, z] groups in Angstrom.
ParsedCase ParseCase(const std::string& path) {
    std::ifstream stream(path);
    std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

    const std::size_t atomsAt = text.find("atoms");
    const std::size_t open = text.find('[', atomsAt);
    std::size_t pos = open + 1;

    ParsedCase parsed;
    std::vector<double> flat;

    while (pos < text.size())
    {
        const char c = text[pos];

        if (c == ']' && (pos + 1 >= text.size() || text[pos + 1] != ','))
        {
            break; // The outer array's closer.
        }

        if (c == '"')
        {
            const std::size_t end = text.find('"', pos + 1);
            const std::string symbol = text.substr(pos + 1, end - pos - 1);
            const qcx::molecule::ElementData* element = qcx::molecule::FindElement(symbol);

            if (element == nullptr)
            {
                std::fprintf(stderr, "unknown element %s\n", symbol.c_str());
                std::exit(2);
            }

            parsed.atoms.push_back(qcx::molecule::Atom{symbol, element->atomicNumber, 0.0});
            parsed.atomicNumbers.push_back(element->atomicNumber);

            // Three reals follow: "Sym", x, y, z.
            pos = text.find(',', end) + 1;

            for (int i = 0; i < 3; ++i)
            {
                const std::size_t stop = text.find_first_of(",]", pos);
                flat.push_back(std::strtod(text.substr(pos, stop - pos).c_str(), nullptr));
                pos = stop + 1;
            }

            continue;
        }

        ++pos;
    }

    const std::size_t count = parsed.atoms.size();
    parsed.coordinatesBohr.resize(count * 3);

    for (std::size_t i = 0; i < count * 3; ++i)
    {
        parsed.coordinatesBohr[i] = flat[i] * qcx::molecule::kAngstromToBohr;
    }

    return parsed;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s <case.toml> <basis-name>...\n", argv[0]);
        return 2;
    }

    const ParsedCase parsed = ParseCase(argv[1]);
    const std::size_t atomCount = parsed.atoms.size();

    auto coordinates =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({atomCount, std::size_t{3}});

    if (!coordinates.has_value())
    {
        std::fprintf(stderr, "coordinates: %s\n", coordinates.error().message.c_str());
        return 2;
    }

    for (std::size_t i = 0; i < atomCount; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            (*coordinates)(i, j) = parsed.coordinatesBohr[i * 3 + j];
        }
    }

    coordinates->MarkHostDirty();

    auto molecule = qcx::molecule::Molecule::Create(parsed.atoms, std::move(*coordinates), 0, 1);

    if (!molecule.has_value())
    {
        std::fprintf(stderr, "molecule: %s\n", molecule.error().message.c_str());
        return 2;
    }

    for (int i = 2; i < argc; ++i)
    {
        const std::string basisName = argv[i];
        const std::string directory = std::string(QcxBasisDataDir) + "/" + basisName;
        auto basis = qcx::basisset::ParseNwchemDirectoryFiltered(directory, parsed.atomicNumbers);

        if (!basis.has_value())
        {
            std::fprintf(
                stderr, "%s: basis: %s\n", basisName.c_str(), basis.error().message.c_str());
            continue;
        }

        auto overlapTensor = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);

        if (!overlapTensor.has_value())
        {
            std::fprintf(stderr,
                         "%s: overlap: %s\n",
                         basisName.c_str(),
                         overlapTensor.error().message.c_str());
            continue;
        }

        const Eigen::Index n = static_cast<Eigen::Index>(overlapTensor->Shape()[0]);
        Eigen::MatrixXd overlap(n, n);

        for (Eigen::Index row = 0; row < n; ++row)
        {
            for (Eigen::Index col = 0; col < n; ++col)
            {
                overlap(row, col) =
                    (*overlapTensor)(static_cast<std::size_t>(row), static_cast<std::size_t>(col));
            }
        }

        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(overlap);
        const Eigen::VectorXd& eigenvalues = solver.eigenvalues();
        const double smallest = eigenvalues.minCoeff();
        const double largest = eigenvalues.maxCoeff();
        const double floor = kFloorTolerance * largest;

        std::size_t below = 0;
        std::size_t belowAbsolute = 0;

        for (Eigen::Index i = 0; i < eigenvalues.size(); ++i)
        {
            if (eigenvalues(i) < floor)
            {
                ++below;
            }

            if (eigenvalues(i) < 1e-8)
            {
                ++belowAbsolute;
            }
        }

        std::printf("%-14s atoms=%zu n=%lld smin=%.6e smax=%.6e ratio=%.6e below=%.1e->%zu "
                    "below_abs=%.1e->%zu\n",
                    basisName.c_str(),
                    parsed.atoms.size(),
                    static_cast<long long>(n),
                    smallest,
                    largest,
                    smallest / largest,
                    kFloorTolerance,
                    below,
                    1e-8,
                    belowAbsolute);
        std::fflush(stdout);
    }

    return 0;
}
