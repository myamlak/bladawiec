// The fixed-density Coulomb-half error of the composed QFMM path at a
// converged SCF density - the C24 bound lane's M2 instrument.
//
// It measures
//
//   dJ = 0.5 * sum_ij D_ij (F_Q - F_D)_ij
//
// where D is the AO density the SCF converged to (trace = the electron
// count), F_Q and F_D are the QFMM J builder's and the restricted direct
// Coulomb builder's Fock matrices at the SPATIAL closed-shell density
// rho = D/2 (both builders' documented input), and the difference is taken
// at ONE density: the exchange halves and the core Hamiltonian cancel, so
// dJ is the Coulomb-half error the composed run's energy difference carries.
// The metric and both builders are the committed ones of
// qfmm_ladder_sweep.cpp (its EnergyErrorEh, RunCell and BuildDirectReference).
//
// The density is read from the driver's own SCF side channel
// ([scf] density_dump: "QXCDFDMP", a u64 dimension, then per-iteration
// records of u8 tag + u64 iteration + u64 n + n*n row-major f64; tag 0 is
// the overlap, tag 2 the density). The last tag-2 record is the density in
// hand at the converged iteration, one SCF pass behind the returned
// density; dJ is quadratic in the density, so that one-pass lag contributes
// a relative factor of the rms density change (below the 1e-6 SCF gate) -
// orders under the measurement. The AO basis is NOT orthonormal, so the
// normalization invariant is tr(D S) = the electron count, not tr(D) (for
// C24H50/STO-3G tr D = 148.37 while tr(D S) = 194): the dump's own overlap
// record supplies S. A dump whose tr(dumped S) is half the electron count
// stores rho, and D is re-formed.
//
// Two policies, same density:
//   sem    - fp64-only and serially pinned on both builders: the contract
//            quantity the preset's far-field budget describes (the sweep's
//            convention).
//   as-run - the driver's option defaults: the QFMM lane permission ON
//            (QfmmOptions::useCertifiedMixedPrecision, which the driver does
//            not seed) with auto chunks, and the direct builder's certified
//            lane request left unset (the device profile's default) with auto
//            chunks.
//
// --attribution adds the two builds that separate the as-run delta's causes:
// the lane at serial chunks (lane ON, chunks 1) and the chunk count at the
// lane OFF (lane OFF, chunks 0), both against the fp64/serial direct
// reference. It costs two more QFMM builds (about 90 s on this fixture).
//
// Usage (local-only, plain main(), like the sweep):
//   qcx-bench-qfmm-fixed-density-error --dump FILE [--carbons 24]
//                                      [--theta 0.3] [--lmult 5] [--attribution]
// Exit 0 on a completed measurement, 1 on any failed leg (a measurement that
// did not run must not read as one).

#include "alkane_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/qfmm_fock_build.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::MakeAlkaneSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// H = T + V from the one-electron engines (the fixtures' convention).
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

// The overlap and the last density record of an SCF dump, row-major as
// written.
struct DensityDump {
    std::uint64_t dimension = 0;
    std::uint64_t iteration = 0;
    std::vector<double> density;
    std::vector<double> overlap;
};

std::optional<DensityDump> ReadDensityDump(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);

    if (!stream.is_open())
    {
        std::cerr << "cannot open the dump " << path << "\n";
        return std::nullopt;
    }

    char magic[8] = {};

    if (!stream.read(magic, 8) || std::memcmp(magic, "QXCDFDMP", 8) != 0)
    {
        std::cerr << "not an SCF density dump (magic mismatch): " << path << "\n";
        return std::nullopt;
    }

    DensityDump dump;

    if (!stream.read(reinterpret_cast<char*>(&dump.dimension), sizeof(dump.dimension)) ||
        dump.dimension == 0 || dump.dimension > 10000)
    {
        std::cerr << "bad dump dimension\n";
        return std::nullopt;
    }

    bool hasDensity = false;

    for (;;)
    {
        std::uint8_t tag = 0;
        std::uint64_t iteration = 0;
        std::uint64_t n = 0;

        if (!stream.read(reinterpret_cast<char*>(&tag), sizeof(tag)))
        {
            break; // Clean end of stream.
        }

        if (!stream.read(reinterpret_cast<char*>(&iteration), sizeof(iteration)) ||
            !stream.read(reinterpret_cast<char*>(&n), sizeof(n)) || n != dump.dimension)
        {
            std::cerr << "malformed dump record\n";
            return std::nullopt;
        }

        std::vector<double> matrix(static_cast<std::size_t>(n * n));

        if (!stream.read(reinterpret_cast<char*>(matrix.data()),
                         static_cast<std::streamsize>(matrix.size() * sizeof(double))))
        {
            std::cerr << "truncated dump record\n";
            return std::nullopt;
        }

        if (tag == 0)
        {
            dump.overlap = std::move(matrix);
        } else if (tag == 2)
        {
            dump.iteration = iteration;
            dump.density = std::move(matrix);
            hasDensity = true;
        }
    }

    if (!hasDensity)
    {
        std::cerr << "the dump holds no density record\n";
        return std::nullopt;
    }

    if (dump.overlap.empty())
    {
        std::cerr << "the dump holds no overlap record\n";
        return std::nullopt;
    }

    return dump;
}

// Row-major doubles -> the Eigen matrix the builders take.
Eigen::MatrixXd ToEigen(std::uint64_t n, const std::vector<double>& rowMajor) {
    Eigen::MatrixXd matrix(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::uint64_t row = 0; row < n; ++row)
    {
        for (std::uint64_t col = 0; col < n; ++col)
        {
            matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
                rowMajor[static_cast<std::size_t>(row * n + col)];
        }
    }

    return matrix;
}

// The J-energy error at a fixed density (qfmm_ladder_sweep.cpp's
// EnergyErrorEh): both builders return H + 2J(rho), so the delta is
// 2(J_Q - J_D) and the 1/2 maps it to the J energy, with D the AO density.
double SignedJEnergyErrorEh(const Eigen::MatrixXd& qfmm,
                            const Eigen::MatrixXd& direct,
                            const Eigen::MatrixXd& density) {
    return 0.5 * (density.cwiseProduct(qfmm - direct)).sum();
}

struct JBuildResult {
    Eigen::MatrixXd fock;
    std::size_t farPairs = 0;
};

std::optional<JBuildResult> BuildQfmmJ(const qcx::molecule::Molecule& molecule,
                                       const qcx::basisset::BasisSet& basisSet,
                                       const CpuTensor2& core,
                                       const Eigen::MatrixXd& rho,
                                       const qcx::integrals::QfmmOptions& options) {
    auto builder = qcx::integrals::QfmmJBuilder::Create(molecule, basisSet, core, options);

    if (!builder.has_value())
    {
        std::cerr << "QFMM Create failed: " << builder.error().message << "\n";
        return std::nullopt;
    }

    auto densityTensor = ToTensor(rho);

    if (!densityTensor.has_value())
    {
        std::cerr << "QFMM density tensor failed: " << densityTensor.error().message << "\n";
        return std::nullopt;
    }

    auto fock = builder->BuildFock(*densityTensor);

    if (!fock.has_value())
    {
        std::cerr << "QFMM BuildFock failed: " << fock.error().message << "\n";
        return std::nullopt;
    }

    return JBuildResult{ToMatrix(*fock), builder->FarFieldPairCount()};
}

std::optional<Eigen::MatrixXd> BuildDirectJ(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet,
                                            const CpuTensor2& core,
                                            const Eigen::MatrixXd& rho,
                                            const qcx::integrals::AccuracyPreset preset,
                                            const std::optional<bool> certifiedLane,
                                            const std::size_t maxParallelChunks) {
    qcx::integrals::FockBuildOptions options;
    options.accuracy = preset;
    options.buildCoulombOnly = true;
    options.useCertifiedMixedPrecision = certifiedLane;
    options.maxParallelChunks = maxParallelChunks;
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, core, options);

    if (!builder.has_value())
    {
        std::cerr << "direct Coulomb Create failed: " << builder.error().message << "\n";
        return std::nullopt;
    }

    auto densityTensor = ToTensor(rho);

    if (!densityTensor.has_value())
    {
        std::cerr << "direct Coulomb density tensor failed: " << densityTensor.error().message
                  << "\n";
        return std::nullopt;
    }

    auto fock = builder->BuildFock(*densityTensor);

    if (!fock.has_value())
    {
        std::cerr << "direct Coulomb BuildFock failed: " << fock.error().message << "\n";
        return std::nullopt;
    }

    return ToMatrix(*fock);
}

} // namespace

int main(int argc, char** argv) {
    std::string dumpPath;
    std::size_t carbonCount = 24;
    double theta = 0.3;
    int lMult = 5;
    bool attribution = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg(argv[i]);

        if (arg == "--dump" && i + 1 < argc)
        {
            dumpPath = argv[++i];
        } else if (arg == "--carbons" && i + 1 < argc)
        {
            carbonCount = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--theta" && i + 1 < argc)
        {
            theta = std::stod(argv[++i]);
        } else if (arg == "--lmult" && i + 1 < argc)
        {
            lMult = std::stoi(argv[++i]);
        } else if (arg == "--attribution")
        {
            attribution = true;
        } else
        {
            std::cerr << "usage: qcx-bench-qfmm-fixed-density-error --dump FILE [--carbons N] "
                         "[--theta T] [--lmult L] [--attribution]\n";
            return 1;
        }
    }

    if (dumpPath.empty())
    {
        std::cerr << "--dump is required\n";
        return 1;
    }

    auto molecule = MakeAlkaneSto3g(carbonCount);

    if (!molecule.has_value())
    {
        std::cerr << "molecule failed: " << molecule.error().message << "\n";
        return 1;
    }

    auto basis = MakeAlkaneSto3gBasis();

    if (!basis.has_value())
    {
        std::cerr << "basis failed: " << basis.error().message << "\n";
        return 1;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        std::cerr << "core Hamiltonian failed: " << core.error().message << "\n";
        return 1;
    }

    const std::size_t nBasis = core->Shape()[0];
    auto dump = ReadDensityDump(dumpPath);

    if (!dump.has_value())
    {
        return 1;
    }

    if (dump->dimension != nBasis)
    {
        std::cerr << "dump dimension " << dump->dimension << " != the fixture's " << nBasis
                  << " basis functions\n";
        return 1;
    }

    const Eigen::MatrixXd dumped = ToEigen(dump->dimension, dump->density);
    const Eigen::MatrixXd overlap = ToEigen(dump->dimension, dump->overlap);
    const double electrons = static_cast<double>(molecule->ElectronCount());

    // The normalization invariant of a non-orthogonal AO basis: tr(D S), not
    // tr(D). The dump stores D most often (tr(D S) = the electron count); a
    // tr(D S) of half that stores the spatial density rho = D/2.
    const double dumpedNormalization = (dumped.cwiseProduct(overlap.transpose())).sum();
    Eigen::MatrixXd density = dumped;

    if (std::abs(dumpedNormalization - electrons) <= 1e-6 * electrons)
    {
        density = dumped;
    } else if (std::abs(dumpedNormalization - 0.5 * electrons) <= 1e-6 * electrons)
    {
        density = 2.0 * dumped;
    } else
    {
        std::cerr << "density normalization tr(dumped S) " << dumpedNormalization
                  << " is neither the electron count " << electrons << " nor half of it\n";
        return 1;
    }

    const Eigen::MatrixXd rho = 0.5 * density;
    const double normalization = (density.cwiseProduct(overlap.transpose())).sum();
    const double asymmetry = (density - density.transpose()).cwiseAbs().maxCoeff();

    std::cout << std::setprecision(17);
    std::cout << "QFMM fixed-density Coulomb-half error (M2)\n"
              << "dump: " << dumpPath << "\n"
              << "density records: last iteration " << dump->iteration << "\n"
              << "n_basis " << nBasis << ", electrons " << electrons << "\n"
              << "trace(D) " << density.trace() << ", trace(D S) " << normalization << "\n"
              << "trace(rho = D/2) " << rho.trace() << ", trace(rho S) " << 0.5 * normalization
              << "\n"
              << "max |D - D^T| " << asymmetry << "\n";

    const auto preset = qcx::integrals::AccuracyPreset::kNormal;

    // The direct reference at both policies.
    auto directSem = BuildDirectJ(*molecule, *basis, *core, rho, preset, false, 1);

    if (!directSem.has_value())
    {
        return 1;
    }

    auto directAsRun = BuildDirectJ(*molecule, *basis, *core, rho, preset, std::nullopt, 0);

    if (!directAsRun.has_value())
    {
        return 1;
    }

    // The QFMM builder at both policies: accuracy + theta + L are the
    // driver's resolved kNormal options (the preset's own 0.3 and 5).
    qcx::integrals::QfmmOptions qfmmSem;
    qfmmSem.accuracy = preset;
    qfmmSem.theta = theta;
    qfmmSem.lMult = lMult;
    qfmmSem.useCertifiedMixedPrecision = false;
    qfmmSem.maxParallelChunks = 1;
    auto qfmmSemResult = BuildQfmmJ(*molecule, *basis, *core, rho, qfmmSem);

    if (!qfmmSemResult.has_value())
    {
        return 1;
    }

    qcx::integrals::QfmmOptions qfmmAsRun;
    qfmmAsRun.accuracy = preset;
    qfmmAsRun.theta = theta;
    qfmmAsRun.lMult = lMult;
    auto qfmmAsRunResult = BuildQfmmJ(*molecule, *basis, *core, rho, qfmmAsRun);

    if (!qfmmAsRunResult.has_value())
    {
        return 1;
    }

    const double dJsem = SignedJEnergyErrorEh(qfmmSemResult->fock, *directSem, density);
    const double dJasRun = SignedJEnergyErrorEh(qfmmAsRunResult->fock, *directAsRun, density);

    std::cout << "theta " << theta << ", lMult " << lMult << ", accuracy kNormal\n"
              << "far pairs: sem " << qfmmSemResult->farPairs << ", as-run "
              << qfmmAsRunResult->farPairs << "\n"
              << "M2-sem    dJ = " << dJsem << " Eh (|dJ| = " << std::abs(dJsem) << ")\n"
              << "M2-as-run dJ = " << dJasRun << " Eh (|dJ| = " << std::abs(dJasRun) << ")\n"
              << "lane delta (as-run - sem) = " << dJasRun - dJsem << " Eh\n";

    if (attribution)
    {
        // The as-run delta's two candidate causes, separated: the lane at
        // serial chunks, and the chunk count with the lane off. Both are
        // measured against the fp64/serial direct reference, so the only
        // difference from M2-sem is the field named.
        qcx::integrals::QfmmOptions laneOnSerialOptions = qfmmSem;
        laneOnSerialOptions.useCertifiedMixedPrecision = true;
        laneOnSerialOptions.maxParallelChunks = 1;
        auto laneOnSerial = BuildQfmmJ(*molecule, *basis, *core, rho, laneOnSerialOptions);

        if (!laneOnSerial.has_value())
        {
            return 1;
        }

        qcx::integrals::QfmmOptions laneOffAutoOptions = qfmmSem;
        laneOffAutoOptions.maxParallelChunks = 0;
        auto laneOffAuto = BuildQfmmJ(*molecule, *basis, *core, rho, laneOffAutoOptions);

        if (!laneOffAuto.has_value())
        {
            return 1;
        }

        const double dJLaneOnSerial = SignedJEnergyErrorEh(laneOnSerial->fock, *directSem, density);
        const double dJLaneOffAuto = SignedJEnergyErrorEh(laneOffAuto->fock, *directSem, density);
        const double dJDirectPolicy =
            SignedJEnergyErrorEh(qfmmSemResult->fock, *directAsRun, density) - dJsem;

        std::cout << "attribution vs the fp64/serial direct reference:\n"
                  << "  lane ON,  chunks 1 dJ = " << dJLaneOnSerial << " (far pairs "
                  << laneOnSerial->farPairs << ")\n"
                  << "  lane OFF, chunks 0 dJ = " << dJLaneOffAuto << " (far pairs "
                  << laneOffAuto->farPairs << ")\n"
                  << "  lane effect (ON,1 - OFF,1)  = " << dJLaneOnSerial - dJsem << "\n"
                  << "  chunk effect (OFF,0 - OFF,1) = " << dJLaneOffAuto - dJsem << "\n"
                  << "  direct as-run vs sem policy = " << dJDirectPolicy << "\n";
    }

    return 0;
}
