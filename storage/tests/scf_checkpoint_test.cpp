// The SCF checkpoint/restart acceptance: a budget-capped
// run restarts to the same converged energy with a bit-identical
// trajectory, the storage side round-trips the restart state
// bit-identically (RHF and UHF), and the append-only + wrong-system
// rejections hold.
#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "o2_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/uhf.hpp"
#include "qcx/storage/scf_checkpoint.hpp"
#include "temp_store.hpp"
#include "tensor_conversions.hpp"

#include <cstring>
#include <gtest/gtest.h>
#include <hdf5.h>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>
#include <map>
#include <utility>
#include <vector>

namespace {

using qcx::scf::HfResult;
using qcx::scf::RhfOptions;
using qcx::scf::RunRhfScf;
using qcx::scf::RunUhfScf;
using qcx::scf::ScfRestartState;
using qcx::scf::UhfOptions;
using qcx::scf::UhfResult;
using qcx::storage::LoadScfCheckpoint;
using qcx::storage::SaveScfCheckpoint;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeO2Sto3gBasis;
using qcx::testing::MakeO2Sto3gTriplet;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ScopedTempStoreFile;
using qcx::testing::ToMatrix;

constexpr double kEnergyTolerance = 1e-12;

// Matrix bit-identity (both sides column-major, same dims).
void ExpectBitIdentical(const Eigen::MatrixXd& expected, const Eigen::MatrixXd& actual) {
    ASSERT_EQ(actual.rows(), expected.rows());
    ASSERT_EQ(actual.cols(), expected.cols());
    EXPECT_EQ(std::memcmp(actual.data(),
                          expected.data(),
                          static_cast<std::size_t>(expected.rows()) *
                              static_cast<std::size_t>(expected.cols()) * sizeof(double)),
              0);
}

void ExpectDiisBitIdentical(const qcx::scf::DiisState& expected,
                            const qcx::scf::DiisState& actual) {
    ASSERT_EQ(actual.fockHistory.size(), expected.fockHistory.size());
    ASSERT_EQ(actual.errorHistory.size(), expected.errorHistory.size());

    for (std::size_t i = 0; i < expected.fockHistory.size(); ++i)
    {
        ExpectBitIdentical(expected.fockHistory[i], actual.fockHistory[i]);
    }

    for (std::size_t i = 0; i < expected.errorHistory.size(); ++i)
    {
        ExpectBitIdentical(expected.errorHistory[i], actual.errorHistory[i]);
    }
}

// The one-shot reference inputs of every SCF test below (the supermatrix
// overload takes the 4-D ERI tensor, not a matrix).
struct RhfInputs {
    qcx::molecule::Molecule molecule;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd core;
    qcx::memory::Tensor<double, 4, qcx::backend::CpuTag> eri;
};

qcx::Result<RhfInputs> MakeRhfInputs(qcx::molecule::Molecule& molecule,
                                     const qcx::basisset::BasisSet& basis) {
    const auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basis);
    // The general-l dense engine: the s-only BuildEriTensor rejects the
    // p shells of H2O/O2 (the dense_rhf_test precedent).
    auto eri = qcx::integrals::BuildEriTensorGeneral(molecule, basis);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    auto cloned = molecule.Clone();

    if (!cloned.has_value())
    {
        return std::unexpected(cloned.error());
    }

    return RhfInputs{std::move(*cloned),
                     ToMatrix(*overlap),
                     ToMatrix(*kinetic) + ToMatrix(*nuclear),
                     std::move(*eri)};
}

TEST(ScfCheckpointTest, RestartContinuesToTheSameConvergedEnergy) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto inputs = MakeRhfInputs(*molecule, *basis);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    // The one-shot reference run.
    qcx::scf::RhfOptions referenceOptions;
    referenceOptions.useDiis = true;
    const auto reference =
        RunRhfScf(inputs->molecule, inputs->overlap, inputs->core, inputs->eri, referenceOptions);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    ASSERT_TRUE(reference->converged);

    // A budget-capped run, then a restart from its saved state. The
    // fixture is H2O/sto-3g, not H2/sto-3g: H2 converges
    // within 2 iterations in this worktree, and the iteration > 0 leg of
    // the convergence gate (scf/src/rhf.cpp) forbids first-pass
    // convergence on a resume, so no H2 budget leaves a non-converged
    // partial with a trajectory claim. H2O needs well over the 3-iteration
    // budget, and the trajectory claim 3 + (k - 3) == k holds because the
    // resumed run re-executes the one-shot trajectory bit for bit from
    // the capped run's last density, DIIS history, and gate energies.
    qcx::scf::RhfOptions partialOptions;
    partialOptions.maxIterations = 3;
    const auto partial =
        RunRhfScf(inputs->molecule, inputs->overlap, inputs->core, inputs->eri, partialOptions);
    ASSERT_TRUE(partial.has_value()) << partial.error().message;
    ASSERT_FALSE(partial->converged);
    ASSERT_FALSE(partial->restart.density.size() == 0);
    ASSERT_GE(partial->restart.diis.fockHistory.size(), std::size_t{3});

    qcx::scf::RhfOptions resumedOptions;
    resumedOptions.initialScfState = partial->restart;
    const auto resumed =
        RunRhfScf(inputs->molecule, inputs->overlap, inputs->core, inputs->eri, resumedOptions);
    ASSERT_TRUE(resumed.has_value()) << resumed.error().message;
    EXPECT_TRUE(resumed->converged);
    EXPECT_NEAR(resumed->totalEnergy, reference->totalEnergy, kEnergyTolerance);
    // Bit-identical trajectory: restart total iterations == the one-shot run.
    EXPECT_EQ(partial->iterations + resumed->iterations, reference->iterations);
}

TEST(ScfCheckpointTest, CheckpointSaveLoadRoundTrip) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto inputs = MakeRhfInputs(*molecule, *basis);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    qcx::scf::RhfOptions partialOptions;
    partialOptions.maxIterations = 3;
    const auto partial =
        RunRhfScf(inputs->molecule, inputs->overlap, inputs->core, inputs->eri, partialOptions);
    ASSERT_TRUE(partial.has_value()) << partial.error().message;
    ASSERT_FALSE(partial->converged);

    const auto oneShot =
        RunRhfScf(inputs->molecule, inputs->overlap, inputs->core, inputs->eri, RhfOptions{});
    ASSERT_TRUE(oneShot.has_value()) << oneShot.error().message;
    ASSERT_TRUE(oneShot->converged);
    ScopedTempStoreFile tempFile("qcx_scf_checkpoint");
    auto saved =
        SaveScfCheckpoint(tempFile.Path(), inputs->molecule, "sto-3g", "", partial->restart);
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    // Append-only: a second save refuses.
    const auto second =
        SaveScfCheckpoint(tempFile.Path(), inputs->molecule, "sto-3g", "", partial->restart);
    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error().code, qcx::ErrorCode::kInvalidArgument);

    auto loaded = LoadScfCheckpoint(tempFile.Path(), inputs->molecule, "sto-3g", "");
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;

    // Every field bit-identical.
    ExpectBitIdentical(partial->restart.density, loaded->density);
    ExpectDiisBitIdentical(partial->restart.diis, loaded->diis);
    EXPECT_DOUBLE_EQ(loaded->previousTotalEnergy, partial->restart.previousTotalEnergy);

    // The inactive channels stay empty: an RHF checkpoint never seeds the
    // UHF alpha/beta DIIS histories.
    EXPECT_TRUE(loaded->diisAlpha.fockHistory.empty());
    EXPECT_TRUE(loaded->diisAlpha.errorHistory.empty());
    EXPECT_TRUE(loaded->diisBeta.fockHistory.empty());
    EXPECT_TRUE(loaded->diisBeta.errorHistory.empty());
    EXPECT_DOUBLE_EQ(loaded->previousElectronicEnergy, partial->restart.previousElectronicEnergy);

    // The loaded state continues the run to the same converged energy.
    qcx::scf::RhfOptions resumedOptions;
    resumedOptions.initialScfState = *loaded;
    const auto resumed =
        RunRhfScf(inputs->molecule, inputs->overlap, inputs->core, inputs->eri, resumedOptions);
    ASSERT_TRUE(resumed.has_value()) << resumed.error().message;
    EXPECT_TRUE(resumed->converged);
    EXPECT_NEAR(resumed->totalEnergy, oneShot->totalEnergy, kEnergyTolerance);
}

// The dense per-element atomic integrals of the SAD guess: one O atom at
// the origin (the fragment run; the geometry never affects its own
// integrals) - the uhf_test.cpp helper, replicated here so the checkpoint
// test seeds the same unpolarized SAD start as the pin test (the
// direct_uhf_test.cpp precedent).
qcx::Result<qcx::scf::AtomicUhfInputs> BuildAtomicOxygenInputs() {
    using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
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

    auto basis = MakeO2Sto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*atom, *basis);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(*atom, *basis);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*atom, *basis);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    auto eri = qcx::integrals::BuildEriTensorGeneral(*atom, *basis);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    return qcx::scf::AtomicUhfInputs{
        ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), std::move(*eri)};
}

// The O2/STO-3G SAD guess (the fragment runs + embedding; the step-B core
// test seeds the UNPOLARIZED variant - the alpha/beta average - which
// reproduces the minao-like unpolarized start the pyscf reference run
// itself used, and the literal P=0 start cannot converge this
// system).
qcx::Result<qcx::scf::SadGuess> BuildSadGuessForO2(const qcx::molecule::Molecule& molecule,
                                                   const qcx::basisset::BasisSet& basisSet) {
    auto atomicInputs = BuildAtomicOxygenInputs();

    if (!atomicInputs.has_value())
    {
        return std::unexpected(atomicInputs.error());
    }

    std::map<int, qcx::scf::AtomicUhfInputs> atomicMap;
    atomicMap.emplace(8, std::move(*atomicInputs));
    return qcx::scf::BuildSadGuess(molecule, basisSet, atomicMap);
}

TEST(ScfCheckpointTest, CheckpointUhfRoundTrip) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "the O2 UHF run is the heaviest SCF case";
    }

    const auto basis = MakeO2Sto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeO2Sto3gTriplet();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto inputs = MakeRhfInputs(*molecule, *basis);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    // The unpolarized SAD seed of the step-B core test: the P=0
    // start locks the higher-lying saddle solution at -147.37855918, so
    // the partial run must be seeded like the pin test.
    const auto sad = BuildSadGuessForO2(*molecule, *basis);
    ASSERT_TRUE(sad.has_value()) << sad.error().message;
    const Eigen::MatrixXd unpolarizedStart = 0.5 * (sad->densityAlpha + sad->densityBeta);

    qcx::scf::UhfOptions partialOptions;
    partialOptions.maxIterations = 3;
    partialOptions.initialDensityAlpha = unpolarizedStart;
    partialOptions.initialDensityBeta = unpolarizedStart;
    const auto partial =
        RunUhfScf(inputs->molecule, inputs->overlap, inputs->core, inputs->eri, partialOptions);
    ASSERT_TRUE(partial.has_value()) << partial.error().message;
    ASSERT_FALSE(partial->converged);
    ASSERT_FALSE(partial->restart.densityAlpha.size() == 0);
    ASSERT_FALSE(partial->restart.densityBeta.size() == 0);

    ScopedTempStoreFile tempFile("qcx_scf_checkpoint_uhf");
    auto saved =
        SaveScfCheckpoint(tempFile.Path(), inputs->molecule, "sto-3g", "", partial->restart);
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    auto loaded = LoadScfCheckpoint(tempFile.Path(), inputs->molecule, "sto-3g", "");
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;

    ExpectBitIdentical(partial->restart.densityAlpha, loaded->densityAlpha);
    ExpectBitIdentical(partial->restart.densityBeta, loaded->densityBeta);
    ExpectDiisBitIdentical(partial->restart.diisAlpha, loaded->diisAlpha);
    ExpectDiisBitIdentical(partial->restart.diisBeta, loaded->diisBeta);
    EXPECT_DOUBLE_EQ(loaded->previousTotalEnergy, partial->restart.previousTotalEnergy);

    // The inactive RHF channel stays empty: a UHF checkpoint never seeds
    // the RHF DIIS history.
    EXPECT_TRUE(loaded->diis.fockHistory.empty());
    EXPECT_TRUE(loaded->diis.errorHistory.empty());

    // The loaded per-spin state continues the UHF run to the pinned
    // triplet energy (the fixture's documented tolerance). The O2/STO-3G
    // triplet runs land at different points of the near-degenerate DIIS
    // fixed-point cluster (documented inter-route spread
    // 8.6e-7..3.3e-6); the per-batch parallel Fock layout
    // flips the resumed path's landing (observed
    // 1.86e-6), so the pin widens to the cluster scale.
    qcx::scf::UhfOptions resumedOptions;
    resumedOptions.initialScfState = *loaded;
    const auto resumed =
        RunUhfScf(inputs->molecule, inputs->overlap, inputs->core, inputs->eri, resumedOptions);
    ASSERT_TRUE(resumed.has_value()) << resumed.error().message;
    EXPECT_TRUE(resumed->converged);
    EXPECT_NEAR(resumed->totalEnergy, qcx::testing::kO2PinnedTotalEnergy, 5e-6);
}

TEST(ScfCheckpointTest, CheckpointWrongSystemIsRejected) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto hydrogen = MakeH2Sto3g();
    ASSERT_TRUE(hydrogen.has_value()) << hydrogen.error().message;
    const auto h2Inputs = MakeRhfInputs(*hydrogen, *basis);
    ASSERT_TRUE(h2Inputs.has_value()) << h2Inputs.error().message;

    qcx::scf::RhfOptions partialOptions;
    partialOptions.maxIterations = 3;
    const auto partial = RunRhfScf(
        h2Inputs->molecule, h2Inputs->overlap, h2Inputs->core, h2Inputs->eri, partialOptions);
    ASSERT_TRUE(partial.has_value()) << partial.error().message;

    ScopedTempStoreFile tempFile("qcx_scf_wrong_system");
    auto saved =
        SaveScfCheckpoint(tempFile.Path(), h2Inputs->molecule, "sto-3g", "", partial->restart);
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    const auto water = MakeH2oSto3g();
    ASSERT_TRUE(water.has_value()) << water.error().message;
    const auto loaded = LoadScfCheckpoint(tempFile.Path(), *water, "sto-3g", "");
    ASSERT_FALSE(loaded.has_value());
    EXPECT_EQ(loaded.error().code, qcx::ErrorCode::kInvalidArgument);
}

// A minimal valid RHF restart state (n = 2) for the corruption tests.
qcx::scf::ScfRestartState MakeMinimalRestartState() {
    qcx::scf::ScfRestartState state;
    state.density = Eigen::MatrixXd::Identity(2, 2);
    state.previousTotalEnergy = -1.0;
    state.previousElectronicEnergy = -1.5;
    return state;
}

// A DIIS history matrix that disagrees with the density's n must be
// refused before anything is written - the old code wrote the history's
// own size against the declared {k,n,n} dataspace.
TEST(ScfCheckpointTest, SaveRejectsMismatchedDiisHistory) {
    qcx::scf::ScfRestartState state = MakeMinimalRestartState();
    state.diis.fockHistory.push_back(Eigen::MatrixXd::Identity(3, 3));
    auto hydrogen = MakeH2Sto3g();
    ASSERT_TRUE(hydrogen.has_value()) << hydrogen.error().message;

    ScopedTempStoreFile tempFile("qcx_scf_bad_history");
    auto saved = SaveScfCheckpoint(tempFile.Path(), *hydrogen, "sto-3g", "", state);
    ASSERT_FALSE(saved.has_value());
    EXPECT_EQ(saved.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(saved.error().message.find("histories"), std::string::npos);
}

// A non-square density dataset must be refused with a clean
// kIOError - the old code took dims.at(0) as n and read past the buffer.
TEST(ScfCheckpointTest, LoadRejectsMalformedDensityShape) {
    auto hydrogen = MakeH2Sto3g();
    ASSERT_TRUE(hydrogen.has_value()) << hydrogen.error().message;
    ScopedTempStoreFile tempFile("qcx_scf_bad_shape");
    auto saved =
        SaveScfCheckpoint(tempFile.Path(), *hydrogen, "sto-3g", "", MakeMinimalRestartState());
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    // Replace the density dataset with a rank-1 {4} one.
    {
        HighFive::File file(tempFile.Path().string(), HighFive::File::ReadWrite);
        file.unlink("scf/state/density");
        HighFive::Group group = file.getGroup("scf/state");
        HighFive::DataSet corrupt = group.createDataSet(
            "density", HighFive::DataSpace({4}), HighFive::AtomicType<double>());
        corrupt.write(std::vector<double>(4, 0.25));
    }

    const auto loaded = LoadScfCheckpoint(tempFile.Path(), *hydrogen, "sto-3g", "");
    ASSERT_FALSE(loaded.has_value());
    EXPECT_EQ(loaded.error().code, qcx::ErrorCode::kIOError);
    EXPECT_NE(loaded.error().message.find("square"), std::string::npos);
}

// A flipped density element must be refused by the state checksum -
// the load recomputes FNV-1a-64 over the read-back data and compares.
TEST(ScfCheckpointTest, LoadRejectsTamperedDensityContent) {
    auto hydrogen = MakeH2Sto3g();
    ASSERT_TRUE(hydrogen.has_value()) << hydrogen.error().message;
    ScopedTempStoreFile tempFile("qcx_scf_tampered");
    auto saved =
        SaveScfCheckpoint(tempFile.Path(), *hydrogen, "sto-3g", "", MakeMinimalRestartState());
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    // Flip one density element in place (the dataset shape stays {2,2}).
    {
        HighFive::File file(tempFile.Path().string(), HighFive::File::ReadWrite);
        HighFive::DataSet density = file.getDataSet("scf/state/density");
        std::vector<double> flat(4);
        density.read_raw(flat.data(), density.getDataType());
        flat[0] += 1.0;
        density.write_raw(flat.data(), density.getDataType());
    }

    const auto loaded = LoadScfCheckpoint(tempFile.Path(), *hydrogen, "sto-3g", "");
    ASSERT_FALSE(loaded.has_value());
    EXPECT_EQ(loaded.error().code, qcx::ErrorCode::kIOError);
    EXPECT_NE(loaded.error().message.find("checksum"), std::string::npos);
}

// A checkpoint without the state_checksum attribute (written by an
// older build) must be refused, not silently trusted.
TEST(ScfCheckpointTest, LoadRejectsMissingStateChecksum) {
    auto hydrogen = MakeH2Sto3g();
    ASSERT_TRUE(hydrogen.has_value()) << hydrogen.error().message;
    ScopedTempStoreFile tempFile("qcx_scf_no_checksum");
    auto saved =
        SaveScfCheckpoint(tempFile.Path(), *hydrogen, "sto-3g", "", MakeMinimalRestartState());
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    {
        HighFive::File file(tempFile.Path().string(), HighFive::File::ReadWrite);
        HighFive::Group group = file.getGroup("scf/state");
        ASSERT_GE(H5Adelete(group.getId(), "state_checksum"), 0);
    }

    const auto loaded = LoadScfCheckpoint(tempFile.Path(), *hydrogen, "sto-3g", "");
    ASSERT_FALSE(loaded.has_value());
    EXPECT_EQ(loaded.error().code, qcx::ErrorCode::kIOError);
}

} // namespace
