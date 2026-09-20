// The SCF robustness harness: the O2/STO-3G seed x accelerator x gate matrix
// that records the two-phase DIIS, virtual-space level shift, robust-gate,
// and joint-system DIIS evidence - per row: converged flag, iterations, total
// energy, <S^2>, and the last per-spin DIIS error norms read from the
// restart histories. Rows are RECORDED, not asserted (the gtest pins live in
// scf/tests/robustness_uhf_test.cpp). Plain main() program rather than a
// google-benchmark run; run standalone on an idle machine - the full
// matrix is a measurement, not a correctness gate. The shift-value sweep
// {0.5, 1.0, 2.0} runs on the P=0 + CDIIS + shift row only (the
// kLevelShiftDefault evidence); the direct-path leg (the UhfFockBuilderFn
// seam, direct coulomb/exchange builders, the direct-path assembly) runs
// under {plain, CDIIS, CDIIS + robust gate} - the case where a plain or
// DIIS-only run settles prematurely on the wrong solution of the
// two-solution O2 landscape lives there.
#include "o2_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/scf/uhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;
using qcx::testing::MakeO2Sto3gBasis;
using qcx::testing::MakeO2Sto3gTriplet;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

struct Fixtures {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd coreHamiltonian;
    CpuTensor4 eri;
    Eigen::MatrixXd unpolarizedSadAlpha;
    Eigen::MatrixXd unpolarizedSadBeta;
    Eigen::MatrixXd polarizedSadAlpha;
    Eigen::MatrixXd polarizedSadBeta;
    Eigen::MatrixXd gwhAlpha;
    Eigen::MatrixXd gwhBeta;
};

qcx::Result<Fixtures> BuildFixtures() {
    auto basis = MakeO2Sto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto molecule = MakeO2Sto3gTriplet();

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    auto eri = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    const Eigen::MatrixXd core = ToMatrix(*kinetic) + ToMatrix(*nuclear);

    // SAD: one atomic O run at the origin (the uhf_test.cpp helper).
    auto coordinates = CpuTensor2::Create({1, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    coordinates->MarkHostDirty();
    auto atom = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}}, std::move(*coordinates), 0, 3);

    if (!atom.has_value())
    {
        return std::unexpected(atom.error());
    }

    auto atomOverlap = qcx::integrals::BuildOverlapMatrix(*atom, *basis);

    if (!atomOverlap.has_value())
    {
        return std::unexpected(atomOverlap.error());
    }

    auto atomKinetic = qcx::integrals::BuildKineticMatrix(*atom, *basis);

    if (!atomKinetic.has_value())
    {
        return std::unexpected(atomKinetic.error());
    }

    auto atomNuclear = qcx::integrals::BuildNuclearAttractionMatrix(*atom, *basis);

    if (!atomNuclear.has_value())
    {
        return std::unexpected(atomNuclear.error());
    }

    auto atomEri = qcx::integrals::BuildEriTensorGeneral(*atom, *basis);

    if (!atomEri.has_value())
    {
        return std::unexpected(atomEri.error());
    }

    std::map<int, qcx::scf::AtomicUhfInputs> atomicMap;
    atomicMap.emplace(8,
                      qcx::scf::AtomicUhfInputs{ToMatrix(*atomOverlap),
                                                ToMatrix(*atomKinetic) + ToMatrix(*atomNuclear),
                                                std::move(*atomEri)});
    auto sad = qcx::scf::BuildSadGuess(*molecule, *basis, atomicMap);

    if (!sad.has_value())
    {
        return std::unexpected(sad.error());
    }

    // GWH: the generalized Wolfsberg-Helmholtz guess, both spins seeded
    // with the unpolarized densities.
    auto gwh = qcx::scf::BuildGwhGuess(ToMatrix(*overlap), core, /*nAlpha=*/8, /*nBeta=*/6);

    if (!gwh.has_value())
    {
        return std::unexpected(gwh.error());
    }

    return Fixtures{std::move(*molecule),
                    std::move(*basis),
                    ToMatrix(*overlap),
                    core,
                    std::move(*eri),
                    0.5 * (sad->densityAlpha + sad->densityBeta),
                    0.5 * (sad->densityAlpha + sad->densityBeta),
                    sad->densityAlpha,
                    sad->densityBeta,
                    gwh->first,
                    gwh->second};
}

// The direct-path core Hamiltonian as a tensor (the DirectJkFockBuilder
// input), the direct_uhf_test.cpp construction.
qcx::Result<CpuTensor2> BuildCoreHamiltonianTensor(const qcx::molecule::Molecule& molecule,
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

void PrintRow(const std::string& label, const qcx::scf::UhfResult& result, bool shifted) {
    // The final per-spin DIIS error norms come from the restart histories
    // (the last appended pair) - the two-way gate never sees them, the
    // robust gate's leg does.
    double diisErrorAlpha = 0.0;
    double diisErrorBeta = 0.0;

    if (!result.restart.diisAlpha.errorHistory.empty())
    {
        diisErrorAlpha = result.restart.diisAlpha.errorHistory.back().norm();
    }

    if (!result.restart.diisBeta.errorHistory.empty())
    {
        diisErrorBeta = result.restart.diisBeta.errorHistory.back().norm();
    }

    std::cout << label << "  converged=" << (result.converged ? "yes" : "no")
              << "  iterations=" << result.iterations << "  totalEnergy=" << result.totalEnergy
              << "  spinSquared=" << result.spinSquared << "  diisErrorAlpha=" << diisErrorAlpha
              << "  diisErrorBeta=" << diisErrorBeta;

    if (shifted)
    {
        // The energy of the shifted runs: printed with its own marker so the
        // shift's effect on the landing can be read directly.
        std::cout << "  (level-shift run)";
    }

    std::cout << "\n";
}

} // namespace

int main() {
    auto fixtures = BuildFixtures();

    if (!fixtures.has_value())
    {
        std::cerr << "fixture construction failed: " << fixtures.error().message << "\n";
        return 1;
    }

    // The seed names: P=0 (no initial densities), SAD-u (unpolarized
    // SAD: both spins the same density), SAD-p (polarized SAD), GWH.
    const std::vector<std::pair<std::string, qcx::scf::UhfOptions>> seeds = {
        {"P0", [] { return qcx::scf::UhfOptions{}; }()},
        {"SAD-u",
         [&] {
             qcx::scf::UhfOptions options;
             options.initialDensityAlpha = fixtures->unpolarizedSadAlpha;
             options.initialDensityBeta = fixtures->unpolarizedSadBeta;
             return options;
         }()},
        {"SAD-p",
         [&] {
             qcx::scf::UhfOptions options;
             options.initialDensityAlpha = fixtures->polarizedSadAlpha;
             options.initialDensityBeta = fixtures->polarizedSadBeta;
             return options;
         }()},
        {"GWH",
         [&] {
             qcx::scf::UhfOptions options;
             options.initialDensityAlpha = fixtures->gwhAlpha;
             options.initialDensityBeta = fixtures->gwhBeta;
             return options;
         }()},
    };

    // The accelerators: plain Roothaan, CDIIS-8 (the default), two-phase
    // DIIS, CDIIS + level shift, two-phase + level shift, plain + level
    // shift - the shift rows at kLevelShiftDefault (1.0; the sweep row set
    // follows). Each accelerator is a partial UhfOptions the seed/gate loop
    // layers over the seed's own options.
    struct Accelerator {
        std::string name;
        qcx::scf::UhfOptions options;
        bool hasShift;
    };

    std::vector<Accelerator> accelerators;
    auto plain = qcx::scf::UhfOptions{};
    plain.useDiis = false;
    accelerators.push_back({"plain", plain, false});
    auto diis8 = qcx::scf::UhfOptions{};
    diis8.useDiis = true;
    accelerators.push_back({"cdiis8", diis8, false});
    auto twoPhase = diis8;
    twoPhase.useTwoPhaseDiis = true;
    accelerators.push_back({"twoPhase", twoPhase, false});

    // The joint-system DIIS leg: one subspace over the combined
    // alpha+beta error vectors (the joint form), the per-spin-vs-joint
    // comparison on every seed. The two-phase cap does not combine with the
    // joint form, so no twoPhase+joint row exists.
    auto joint = diis8;
    joint.useJointDiis = true;
    accelerators.push_back({"joint", joint, false});

    const auto withShift = [&](qcx::scf::UhfOptions options, double shift) {
        options.useLevelShift = true;
        options.levelShiftAlpha = shift;
        options.levelShiftBeta = shift;
        return options;
    };
    accelerators.push_back({"cdiis8+shift", withShift(diis8, qcx::scf::kLevelShiftDefault), true});
    accelerators.push_back(
        {"twoPhase+shift", withShift(twoPhase, qcx::scf::kLevelShiftDefault), true});
    accelerators.push_back({"plain+shift", withShift(plain, qcx::scf::kLevelShiftDefault), true});
    accelerators.push_back({"joint+shift", withShift(joint, qcx::scf::kLevelShiftDefault), true});

    std::cout << "The robustness harness: O2/STO-3G triplet, seed x accelerator x gate\n";
    std::cout << "pin: energy " << -147.63394678545018 << ", <S^2> " << 2.0034108576810308
              << " (1e-7 / 1e-6)\n";

    for (const auto& [seedName, seedOptions] : seeds)
    {
        for (const auto& accelerator : accelerators)
        {
            for (const bool robustGate : {false, true})
            {
                qcx::scf::UhfOptions options = seedOptions;
                options.useDiis = accelerator.options.useDiis;
                options.useTwoPhaseDiis = accelerator.options.useTwoPhaseDiis;
                options.useJointDiis = accelerator.options.useJointDiis;
                options.useLevelShift = accelerator.options.useLevelShift;
                options.levelShiftAlpha = accelerator.options.levelShiftAlpha;
                options.levelShiftBeta = accelerator.options.levelShiftBeta;
                options.useRobustGate = robustGate;
                const auto result = qcx::scf::RunUhfScf(fixtures->molecule,
                                                        fixtures->overlap,
                                                        fixtures->coreHamiltonian,
                                                        fixtures->eri,
                                                        options);

                if (!result.has_value())
                {
                    std::cerr << seedName << "/" << accelerator.name << "/"
                              << (robustGate ? "robust" : "twoWay")
                              << ": scf failed: " << result.error().message << "\n";
                    continue;
                }

                PrintRow(seedName + "/" + accelerator.name + "/" +
                             (robustGate ? "robust" : "twoWay"),
                         *result,
                         accelerator.hasShift);
            }
        }
    }

    // The shift-value sweep on the P=0 + CDIIS + shift row (the
    // kLevelShiftDefault evidence).
    for (const double shift : {0.5, 1.0, 2.0})
    {
        qcx::scf::UhfOptions options = withShift(diis8, shift);
        const auto result = qcx::scf::RunUhfScf(fixtures->molecule,
                                                fixtures->overlap,
                                                fixtures->coreHamiltonian,
                                                fixtures->eri,
                                                options);

        if (!result.has_value())
        {
            std::cerr << "P0/cdiis8+shift" << shift << ": scf failed: " << result.error().message
                      << "\n";
            continue;
        }

        PrintRow("P0/cdiis8+shift" + std::to_string(shift), *result, true);
    }

    // The direct-path leg: the split coulomb/exchange builders wired into
    // the UhfFockBuilderFn seam with the direct-path assembly
    // fock_sigma = coulomb + exchangeSigma - H (the double-H trap: the split
    // builders already carry H; kTight for the fp64 pin, the
    // direct_uhf_test.cpp convention).
    auto core = BuildCoreHamiltonianTensor(fixtures->molecule, fixtures->basis);

    if (!core.has_value())
    {
        std::cerr << "direct leg: core Hamiltonian failed: " << core.error().message << "\n";
        return 1;
    }

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    qcx::integrals::FockBuildOptions coulombOptions = fockOptions;
    coulombOptions.buildCoulombOnly = true;
    auto coulombBuilder = qcx::integrals::DirectJkFockBuilder::Create(
        fixtures->molecule, fixtures->basis, *core, coulombOptions);

    if (!coulombBuilder.has_value())
    {
        std::cerr << "direct leg: coulomb builder failed: " << coulombBuilder.error().message
                  << "\n";
        return 1;
    }

    qcx::integrals::FockBuildOptions exchangeOptions = fockOptions;
    exchangeOptions.buildExchangeOnly = true;
    auto exchangeBuilder = qcx::integrals::DirectJkFockBuilder::Create(
        fixtures->molecule, fixtures->basis, *core, exchangeOptions);

    if (!exchangeBuilder.has_value())
    {
        std::cerr << "direct leg: exchange builder failed: " << exchangeBuilder.error().message
                  << "\n";
        return 1;
    }

    const Eigen::MatrixXd coreMatrix = ToMatrix(*core);
    const qcx::scf::UhfFockBuilderFn directBuilder =
        [coulombBuilder, exchangeBuilder, coreMatrix](const Eigen::MatrixXd& dAlpha,
                                                      const Eigen::MatrixXd& dBeta)
        -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        auto dTotalHalf = ToTensor(0.5 * (dAlpha + dBeta));

        if (!dTotalHalf.has_value())
        {
            return std::unexpected(dTotalHalf.error());
        }

        auto coulomb = coulombBuilder->BuildFock(*dTotalHalf);

        if (!coulomb.has_value())
        {
            return std::unexpected(coulomb.error());
        }

        auto dAlphaTensor = ToTensor(dAlpha);

        if (!dAlphaTensor.has_value())
        {
            return std::unexpected(dAlphaTensor.error());
        }

        auto exchangeAlpha = exchangeBuilder->BuildFock(*dAlphaTensor);

        if (!exchangeAlpha.has_value())
        {
            return std::unexpected(exchangeAlpha.error());
        }

        auto dBetaTensor = ToTensor(dBeta);

        if (!dBetaTensor.has_value())
        {
            return std::unexpected(dBetaTensor.error());
        }

        auto exchangeBeta = exchangeBuilder->BuildFock(*dBetaTensor);

        if (!exchangeBeta.has_value())
        {
            return std::unexpected(exchangeBeta.error());
        }

        const Eigen::MatrixXd coulombMatrix = ToMatrix(*coulomb);
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{
            coulombMatrix + ToMatrix(*exchangeAlpha) - coreMatrix,
            coulombMatrix + ToMatrix(*exchangeBeta) - coreMatrix};
    };

    // The SAD-u seed under {plain, CDIIS, CDIIS + robust gate}: the
    // premature-convergence case (2.2e-7 off the pin in 45 iterations
    // through direct + CDIIS + the two-way gate) and the robust gate's
    // demonstration.
    for (const bool robustGate : {false, true})
    {
        qcx::scf::UhfOptions options;
        options.useDiis = true;
        options.useRobustGate = robustGate;
        options.initialDensityAlpha = fixtures->unpolarizedSadAlpha;
        options.initialDensityBeta = fixtures->unpolarizedSadBeta;
        const auto result = qcx::scf::RunUhfScf(fixtures->molecule,
                                                fixtures->overlap,
                                                fixtures->coreHamiltonian,
                                                options,
                                                directBuilder);

        if (!result.has_value())
        {
            std::cerr << "direct/SAD-u/cdiis8/" << (robustGate ? "robust" : "twoWay")
                      << ": scf failed: " << result.error().message << "\n";
            continue;
        }

        PrintRow(
            "direct/SAD-u/cdiis8/" + std::string(robustGate ? "robust" : "twoWay"), *result, false);
    }

    qcx::scf::UhfOptions plainOptions;
    plainOptions.useDiis = false;
    plainOptions.initialDensityAlpha = fixtures->unpolarizedSadAlpha;
    plainOptions.initialDensityBeta = fixtures->unpolarizedSadBeta;
    const auto plainResult = qcx::scf::RunUhfScf(fixtures->molecule,
                                                 fixtures->overlap,
                                                 fixtures->coreHamiltonian,
                                                 plainOptions,
                                                 directBuilder);

    if (!plainResult.has_value())
    {
        std::cerr << "direct/SAD-u/plain: scf failed: " << plainResult.error().message << "\n";
        return 1;
    }

    PrintRow("direct/SAD-u/plain", *plainResult, false);
    return 0;
}
