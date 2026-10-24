// The two-electron ERI gradient's tests (two_electron.hpp). The checks, in
// the order the surface's acceptance rests on them:
//
//  - the reported gradient against the CENTRAL FINITE DIFFERENCE of the
//    gradient's OWN energy - the density-contracted Coulomb energy the same
//    call returns - on a low-symmetry displaced H2O/STO-3G geometry, so a
//    gradient that differentiates a different contraction than the one it
//    reports fails here;
//  - the translational sum of the gradient, which is exactly zero because
//    moving every atom together changes no integral;
//  - the reported energy against the ordered shell-quadruple contraction of
//    the same density, evaluated one quartet at a time - the pair
//    multiplicities and the density bookkeeping reached two ways.
//
// The internal derivative tier's own acceptance lives in
// two_electron_derivative_test.cpp; this file tests the public surface built
// on it.
#include "h2o_sto3g.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_eri_derivative.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

/// The central-difference step: the truncation is O(h^2) and the
/// cancellation error O(eps E / h), both far inside the 1e-7 threshold at
/// this fixture's O(1) magnitudes.
constexpr double kFiniteDifferenceStep = 1e-5;

/// The analytic-versus-finite-difference threshold of the public gradient.
constexpr double kFiniteDifferenceTolerance = 1e-7;

/// The energy pin's tolerance: the two contractions sum the same products in
/// a different order, so they agree to roundoff and not to the bit.
constexpr double kEnergyTolerance = 1e-12;

/// The translational sum's threshold. The identity is exact in arithmetic;
/// in floating point the sum is the roundoff of O(1) contributions, which
/// this fixture measures orders of magnitude below the gate.
constexpr double kTranslationTolerance = 1e-12;

/// The symmetry-breaking displacement of the fixture's three atoms, one
/// triple per atom in (x, y, z) order: the base geometry is put off the
/// water's symmetry planes so no gradient entry is zero by symmetry alone.
constexpr std::array<double, 9> kBaseShifts = {
    0.013, -0.007, 0.019, -0.011, 0.017, -0.023, 0.029, -0.013, 0.011};

/// A double as a short scientific string, for the recorded properties.
std::string ShortDouble(double value) {
    char text[32] = {};
    std::snprintf(text, sizeof(text), "%.3e", value);
    return text;
}

/// The geometry with the listed flat coordinates (3 * atom + axis) displaced
/// by the listed deltas; a repeated coordinate accumulates. Errors when the
/// canonical renumbering would move an atom - the base geometry's shell
/// indices must name the same shells at every displaced geometry.
/// \param base The base molecule.
/// \param flatCoordinates The displaced coordinates.
/// \param deltas The displacements.
/// \returns The displaced molecule, or an Error.
qcx::Result<qcx::molecule::Molecule> Moved(const qcx::molecule::Molecule& base,
                                           std::span<const std::size_t> flatCoordinates,
                                           std::span<const double> deltas) {
    if (flatCoordinates.size() != deltas.size())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "displacement list shape mismatch"});
    }

    auto coordinates = CpuTensor2::Create({base.AtomCount(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    const auto& source = base.CoordinatesBohr();
    std::vector<std::array<double, 3>> expected;
    expected.reserve(base.AtomCount());

    for (std::size_t atom = 0; atom < base.AtomCount(); ++atom)
    {
        std::array<double, 3> row{};

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            row[axis] = source(atom, axis);
            (*coordinates)(atom, axis) = source(atom, axis);
        }

        expected.push_back(row);
    }

    for (std::size_t k = 0; k < flatCoordinates.size(); ++k)
    {
        const std::size_t flat = flatCoordinates[k];

        if (flat >= 3 * base.AtomCount())
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "displaced coordinate out of range"});
        }

        (*coordinates)(flat / 3, flat % 3) += deltas[k];
        expected[flat / 3][flat % 3] += deltas[k];
    }

    coordinates->MarkHostDirty();
    auto displaced = qcx::molecule::Molecule::Create(
        base.Atoms(), std::move(*coordinates), base.Charge(), base.Multiplicity());

    if (!displaced.has_value())
    {
        return std::unexpected(displaced.error());
    }

    for (std::size_t atom = 0; atom < base.AtomCount(); ++atom)
    {
        if (displaced->Atoms()[atom].atomicNumber != base.Atoms()[atom].atomicNumber ||
            displaced->CoordinatesBohr()(atom, 0) != expected[atom][0] ||
            displaced->CoordinatesBohr()(atom, 1) != expected[atom][1] ||
            displaced->CoordinatesBohr()(atom, 2) != expected[atom][2])
        {
            return std::unexpected(qcx::Error{
                qcx::ErrorCode::kInvalidArgument,
                "the displaced geometry is not the base geometry with one row moved - the shell "
                "indices of the finite-difference loop would name other shells"});
        }
    }

    return displaced;
}

/// The fixture's water with the symmetry-breaking displacement applied.
/// \param base The fixture molecule.
/// \returns The displaced molecule, or an Error.
qcx::Result<qcx::molecule::Molecule> DisplacedBase(const qcx::molecule::Molecule& base) {
    if (base.AtomCount() != kBaseShifts.size() / 3)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "the fixture is not a three-atom water"});
    }

    std::array<std::size_t, 9> coordinates{};

    for (std::size_t flat = 0; flat < coordinates.size(); ++flat)
    {
        coordinates[flat] = flat;
    }

    return Moved(base, coordinates, kBaseShifts);
}

/// A fixed density for the walk: the core-Hamiltonian eigenvector density, not
/// a converged SCF density. The gradient is at fixed density by contract, so
/// any symmetric density exercises the same code - and this one needs no SCF
/// module, which the integrals tests cannot depend on.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \returns The density, or an Error.
qcx::Result<Eigen::MatrixXd> CoreDensity(const qcx::molecule::Molecule& molecule,
                                         const qcx::basisset::BasisSet& basisSet) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto attraction = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!attraction.has_value())
    {
        return std::unexpected(attraction.error());
    }

    const Eigen::MatrixXd core =
        qcx::testing::ToMatrix(*kinetic) + qcx::testing::ToMatrix(*attraction);
    const Eigen::MatrixXd metric = qcx::testing::ToMatrix(*overlap);
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(core, metric);

    if (solver.info() != Eigen::Success)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "the core-Hamiltonian density solve failed"});
    }

    const int electrons = molecule.ElectronCount();
    const Eigen::Index occupied = static_cast<Eigen::Index>(std::max(electrons, 0) / 2);
    const Eigen::MatrixXd coefficients = solver.eigenvectors().leftCols(occupied);
    return 2.0 * coefficients * coefficients.transpose();
}

/// The fixture's low-symmetry water, its basis set and its fixed density.
struct Fixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basisSet;
    Eigen::MatrixXd density;
    CpuTensor2 densityTensor;
};

/// Builds the fixture: the displaced H2O/STO-3G geometry and the
/// core-Hamiltonian density evaluated there.
/// \returns The fixture, or an Error.
qcx::Result<Fixture> MakeFixture() {
    auto base = qcx::testing::MakeH2oSto3g();

    if (!base.has_value())
    {
        return std::unexpected(base.error());
    }

    auto displaced = DisplacedBase(*base);

    if (!displaced.has_value())
    {
        return std::unexpected(displaced.error());
    }

    auto basisSet = qcx::testing::MakeH2oSto3gBasis();

    if (!basisSet.has_value())
    {
        return std::unexpected(basisSet.error());
    }

    auto density = CoreDensity(*displaced, *basisSet);

    if (!density.has_value())
    {
        return std::unexpected(density.error());
    }

    auto tensor = qcx::testing::ToTensor(*density);

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    return Fixture{
        std::move(*displaced), std::move(*basisSet), std::move(*density), std::move(*tensor)};
}

/// Records one gradient's sum over each Cartesian axis: moving every atom
/// together leaves every integral's value alone, so each of the three sums is
/// zero to the arithmetic - a walk that put a block's derivatives on the
/// wrong coordinates cannot produce one.
/// \param gradient The 3N gradient.
/// \param atomCount The molecule's atom count.
void RecordTranslationSums(const Eigen::VectorXd& gradient, std::size_t atomCount) {
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        double sum = 0.0;

        for (std::size_t atom = 0; atom < atomCount; ++atom)
        {
            sum += gradient(static_cast<Eigen::Index>(3 * atom + axis));
        }

        ::testing::Test::RecordProperty("translation_axis_" + std::to_string(axis),
                                        ShortDouble(sum));
    }
}

/// The density-contracted Coulomb energy, evaluated as the ORDERED shell
/// quadruple sum 1/2 sum_uvls D_uv D_ls (uv|ls), one quartet built at a
/// time. This is the functional the public entry's gradient differentiates,
/// reached without the entry's canonical-pair bookkeeping: the same value
/// builder, the same weights, no multiplicity factors to get right.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \param density The density matrix, n x n.
/// \returns The energy (Hartree), or an Error.
qcx::Result<double> ExactDensityEnergy(const qcx::molecule::Molecule& molecule,
                                       const qcx::basisset::BasisSet& basisSet,
                                       const Eigen::MatrixXd& density) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto shells = qcx::integrals::internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    const std::size_t shellCount = pairList->shells.size();
    qcx::integrals::internal::MdEriDerivativeScratch scratch;
    std::vector<double> block;
    double energy = 0.0;

    for (std::size_t i = 0; i < shellCount; ++i)
    {
        for (std::size_t j = 0; j < shellCount; ++j)
        {
            for (std::size_t k = 0; k < shellCount; ++k)
            {
                for (std::size_t l = 0; l < shellCount; ++l)
                {
                    qcx::integrals::internal::MdEriDerivativeQuartet quartet;
                    const std::array<std::size_t, 4> indices = {i, j, k, l};

                    for (int slot = 0; slot < 4; ++slot)
                    {
                        const std::size_t index = indices[static_cast<std::size_t>(slot)];
                        quartet.shells[static_cast<std::size_t>(slot)] = &(*shells)[index];
                        quartet.atoms[static_cast<std::size_t>(slot)] =
                            pairList->shells[index].atomIndex;
                    }

                    const std::size_t elements =
                        qcx::integrals::internal::EriQuartetBlockElements(quartet);
                    block.assign(elements, 0.0);
                    qcx::integrals::internal::BuildEriQuartetValue(quartet, 0, block, scratch);

                    const std::size_t nI =
                        qcx::integrals::internal::EriShellFunctions(*quartet.shells[0]);
                    const std::size_t nJ =
                        qcx::integrals::internal::EriShellFunctions(*quartet.shells[1]);
                    const std::size_t nK =
                        qcx::integrals::internal::EriShellFunctions(*quartet.shells[2]);
                    const std::size_t nL =
                        qcx::integrals::internal::EriShellFunctions(*quartet.shells[3]);
                    const std::size_t offI = pairList->shells[i].functionOffset;
                    const std::size_t offJ = pairList->shells[j].functionOffset;
                    const std::size_t offK = pairList->shells[k].functionOffset;
                    const std::size_t offL = pairList->shells[l].functionOffset;

                    for (std::size_t fa = 0; fa < nI; ++fa)
                    {
                        for (std::size_t fb = 0; fb < nJ; ++fb)
                        {
                            for (std::size_t fc = 0; fc < nK; ++fc)
                            {
                                for (std::size_t fd = 0; fd < nL; ++fd)
                                {
                                    energy += density(static_cast<Eigen::Index>(offI + fa),
                                                      static_cast<Eigen::Index>(offJ + fb)) *
                                              density(static_cast<Eigen::Index>(offK + fc),
                                                      static_cast<Eigen::Index>(offL + fd)) *
                                              block[qcx::integrals::EriBlockIndex(
                                                  fa, fb, fc, fd, nI, nJ, nK, nL)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return 0.5 * energy;
}

/// The acceptance test: the public gradient is the derivative of the energy
/// the same call reports. The base geometry is the fixture's water with a
/// fixed symmetry-breaking displacement, the density is held fixed, and every
/// atom and axis is central-differenced.
TEST(EriGradientTest, TheGradientIsTheDerivativeOfItsOwnEnergy) {
    auto fixture = MakeFixture();

    ASSERT_TRUE(fixture.has_value()) << fixture.error().message;

    const qcx::integrals::EriGradientOptions options;
    auto analytic = qcx::integrals::ComputeEriGradient(
        fixture->molecule, fixture->basisSet, fixture->densityTensor, options);

    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    const std::size_t coordinateCount = 3 * fixture->molecule.AtomCount();
    ASSERT_EQ(analytic->gradient.size(), static_cast<Eigen::Index>(coordinateCount));
    ASSERT_GT(analytic->counts.quartets, 0u);
    // A gradient that came back all zero would pass a finite difference of a
    // vanished energy, so the fixture's own scale is checked first.
    ASSERT_GT(analytic->gradient.cwiseAbs().maxCoeff(), 0.0);

    double worst = 0.0;
    std::size_t worstCoordinate = 0;

    for (std::size_t flat = 0; flat < coordinateCount; ++flat)
    {
        const std::array<std::size_t, 1> one = {flat};
        const std::array<double, 1> forward = {kFiniteDifferenceStep};
        const std::array<double, 1> backward = {-kFiniteDifferenceStep};
        auto plus = Moved(fixture->molecule, one, forward);
        auto minus = Moved(fixture->molecule, one, backward);

        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        auto plusResult = qcx::integrals::ComputeEriGradient(
            *plus, fixture->basisSet, fixture->densityTensor, options);
        auto minusResult = qcx::integrals::ComputeEriGradient(
            *minus, fixture->basisSet, fixture->densityTensor, options);

        ASSERT_TRUE(plusResult.has_value()) << plusResult.error().message;
        ASSERT_TRUE(minusResult.has_value()) << minusResult.error().message;

        const double numeric =
            (plusResult->energy - minusResult->energy) / (2.0 * kFiniteDifferenceStep);
        const double deviation =
            std::abs(numeric - analytic->gradient(static_cast<Eigen::Index>(flat)));

        if (deviation > worst)
        {
            worst = deviation;
            worstCoordinate = flat;
        }
    }

    RecordProperty("worst_deviation", ShortDouble(worst));
    RecordProperty("energy", ShortDouble(analytic->energy));
    RecordProperty("gradient_scale", ShortDouble(analytic->gradient.cwiseAbs().maxCoeff()));
    RecordProperty("density_trace", ShortDouble(fixture->density.trace()));
    RecordProperty("bra_pairs", std::to_string(analytic->counts.braPairs));
    RecordProperty("ket_pairs", std::to_string(analytic->counts.ketPairs));
    RecordProperty("quartets", std::to_string(analytic->counts.quartets));
    RecordProperty("screened_quartets", std::to_string(analytic->counts.screenedQuartets));
    RecordTranslationSums(analytic->gradient, fixture->molecule.AtomCount());
    EXPECT_LT(worst, kFiniteDifferenceTolerance)
        << "coordinate " << worstCoordinate << " of " << coordinateCount;
}

/// Moving every atom together changes no integral, so the gradient's sum over
/// each Cartesian axis is zero - exactly so in arithmetic, and to the
/// accumulation's roundoff here. A walk that attached a block's derivatives to
/// the wrong coordinates cannot pass this.
TEST(EriGradientTest, TheTranslationalSumIsZero) {
    auto fixture = MakeFixture();

    ASSERT_TRUE(fixture.has_value()) << fixture.error().message;

    auto analytic = qcx::integrals::ComputeEriGradient(
        fixture->molecule, fixture->basisSet, fixture->densityTensor);

    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        double sum = 0.0;

        for (std::size_t atom = 0; atom < fixture->molecule.AtomCount(); ++atom)
        {
            sum += analytic->gradient(static_cast<Eigen::Index>(3 * atom + axis));
        }

        EXPECT_LE(std::abs(sum), kTranslationTolerance)
            << "axis " << axis << ": translational sum " << sum;
    }
}

/// The reported energy is the density-contracted Coulomb energy and not some
/// other contraction: the ordered shell-quadruple sum of the same density
/// gives the same number. The walk is run unscreened so the reference is
/// compared against the full contraction.
TEST(EriGradientTest, TheReportedEnergyIsTheDensityContractedCoulombEnergy) {
    auto fixture = MakeFixture();

    ASSERT_TRUE(fixture.has_value()) << fixture.error().message;

    auto exact = ExactDensityEnergy(fixture->molecule, fixture->basisSet, fixture->density);

    ASSERT_TRUE(exact.has_value()) << exact.error().message;
    ASSERT_GT(std::abs(*exact), 0.0) << "the fixture's contraction must not vanish";

    qcx::integrals::EriGradientOptions options;
    options.useDensityScreening = false;
    auto assembled = qcx::integrals::ComputeEriGradient(
        fixture->molecule, fixture->basisSet, fixture->densityTensor, options);

    ASSERT_TRUE(assembled.has_value()) << assembled.error().message;
    ASSERT_EQ(assembled->counts.screenedQuartets, 0u);

    const double scale = std::max(1.0, std::abs(*exact));
    RecordProperty("exact_energy", ShortDouble(*exact));
    RecordProperty("assembled_energy", ShortDouble(assembled->energy));
    RecordProperty("energy_deviation", ShortDouble(std::abs(assembled->energy - *exact)));
    EXPECT_LE(std::abs(assembled->energy - *exact), kEnergyTolerance * scale);
}

} // namespace
