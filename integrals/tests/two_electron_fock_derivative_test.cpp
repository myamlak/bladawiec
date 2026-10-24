// The two-electron derivative Fock blocks' tests (two_electron.hpp).
// ComputeTwoElectronFockDerivative returns the occupied-virtual blocks of
// C^T J' C and C^T K' C at fixed density and fixed MO coefficients, the two
// skeleton gradients, and the two energies. The checks, in the order the
// surface's acceptance rests on them:
//
//  - the energies, the AO-contracted MO blocks and the skeletons against an
//    ORDERED shell-quadruple walk - every quadruple visited in its own order,
//    one block built at a time, no canonicalization and no multiplicity
//    bookkeeping - so the canonical walk's counting is checked where it
//    cannot borrow the reference's conventions;
//  - the Coulomb skeleton and energy against ComputeEriGradient under the
//    SAME screening preset: the entry's contract is that the Coulomb half
//    keeps exactly the quartets that entry keeps;
//  - J' and K' against the CENTRAL FINITE DIFFERENCE of the ordered walk's
//    AO matrices at the same fixed coefficients;
//  - the translation identity of both blocks and both skeletons;
//  - the carried screening structure: exact at zero motion, and losing
//    nothing numerically significant when lifted over a displacement it
//    covers;
//  - the documented refusals.
//
// The fixture is the displaced H2O/STO-3G geometry (symmetry broken so no
// entry vanishes by symmetry) with the core-Hamiltonian density and
// eigenvectors.
#include "h2o_sto3g.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_eri_derivative.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/screening.hpp"
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
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::integrals::ShellPairList;
using qcx::integrals::internal::MdEriDerivativeQuartet;
using qcx::integrals::internal::MdEriDerivativeScratch;
using qcx::integrals::internal::MdShellInput;

/// The central-difference step: the blocks are O(1) and smooth, so the
/// truncation and the cancellation error both sit inside the 1e-7 threshold.
constexpr double kFiniteDifferenceStep = 1e-5;

/// The analytic-versus-finite-difference threshold of the Fock blocks.
constexpr double kFiniteDifferenceTolerance = 1e-7;

/// The ordered-walk contract's threshold: the two walks sum the same products
/// in a different order, so they agree to roundoff and not to the bit.
constexpr double kReferenceTolerance = 1e-12;

/// The translation sum's threshold.
constexpr double kTranslationTolerance = 1e-12;

/// The lifted warm start's threshold: both screens are conservative, and a
/// quartet kept by one alone contributes at the screen threshold's scale.
constexpr double kLiftedTolerance = 1e-7;

/// The largest motion a warm start is lifted over in these tests. The
/// displacement it covers is the finite-difference step, orders of magnitude
/// smaller, so the first-order lift has room.
constexpr double kWarmStartMotion = 0.01;

/// The symmetry-breaking displacement of the fixture's three atoms, one
/// triple per atom in (x, y, z) order.
constexpr std::array<double, 9> kBaseShifts = {
    0.013, -0.007, 0.019, -0.011, 0.017, -0.023, 0.029, -0.013, 0.011};

/// A double as a short scientific string, for the recorded properties.
std::string ShortDouble(double value) {
    char text[32] = {};
    std::snprintf(text, sizeof(text), "%.3e", value);
    return text;
}

/// The geometry with the listed flat coordinates (3 * atom + axis) displaced
/// by the listed deltas.
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
/// \param base The base molecule.
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

/// One flat coordinate's displacement: which coordinate and by how much.
struct Displacement {
    std::size_t flatCoordinate = 0; ///< The coordinate (3 * atom + axis).
    double delta = 0.0; ///< The displacement, Bohr.
};

/// The geometry with one flat coordinate displaced.
/// \param base The base molecule.
/// \param displacement The coordinate (3 * atom + axis) and the displacement (Bohr).
/// \returns The displaced molecule, or an Error.
qcx::Result<qcx::molecule::Molecule> MakeDisplaced(const qcx::molecule::Molecule& base,
                                                   Displacement displacement) {
    const std::array<std::size_t, 1> coordinates = {displacement.flatCoordinate};
    const std::array<double, 1> deltas = {displacement.delta};
    return Moved(base, coordinates, deltas);
}

/// The distinct atoms a quartet's four shells sit on, three flat coordinates
/// (3 * atom + axis) each, in ascending atom order.
/// \param atoms The quartet's four atoms.
/// \returns The flat coordinates.
std::vector<std::size_t> QuartetCoordinates(const std::array<std::size_t, 4>& atoms) {
    std::array<std::size_t, 4> distinct = atoms;
    std::sort(distinct.begin(), distinct.end());
    const auto last = std::unique(distinct.begin(), distinct.end());
    std::vector<std::size_t> coordinates;

    for (auto atom = distinct.begin(); atom != last; ++atom)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            coordinates.push_back(3 * (*atom) + static_cast<std::size_t>(axis));
        }
    }

    return coordinates;
}

/// The derivative tier's view of one ordered quartet.
/// \param pairList The shell pair list.
/// \param shells The flattened shell inputs.
/// \param indices The quartet's four shell indices, in order.
/// \returns The quartet the tiers take.
MdEriDerivativeQuartet DerivativeQuartet(const ShellPairList& pairList,
                                         const std::vector<MdShellInput>& shells,
                                         const std::array<std::size_t, 4>& indices) {
    MdEriDerivativeQuartet out;

    for (int slot = 0; slot < 4; ++slot)
    {
        const std::size_t index = indices[static_cast<std::size_t>(slot)];
        out.shells[static_cast<std::size_t>(slot)] = &shells[index];
        out.atoms[static_cast<std::size_t>(slot)] = pairList.shells[index].atomIndex;
    }

    return out;
}

/// The ordered shell-quadruple reference: every ordered quadruple visited
/// once, one block built at a time, the Coulomb contraction accumulated on
/// (mu, nu) and the exchange contraction on (mu, nu) with its two inner
/// indices carrying the density. No canonicalization and no multiplicities -
/// the pair bookkeeping the entry under test does is exactly what this
/// reference cannot share.
///
/// E_J = 1/2 sum_uvls D_uv D_ls (uv|ls); J_uv = sum_ls D_ls (uv|ls);
/// E_K = 1/2 sum_uvls D_uv D_ls (ul|vs); K_uv = sum_ls D_ls (ul|vs).
struct OrderedFock {
    Eigen::MatrixXd coulomb; ///< J in the AO basis.
    Eigen::MatrixXd exchange; ///< K in the AO basis.
    double coulombEnergy = 0.0; ///< 1/2 Tr(D J).
    double exchangeEnergy = 0.0; ///< 1/2 Tr(D K).
    std::vector<Eigen::MatrixXd> coulombDerivative; ///< J' per flat coordinate.
    std::vector<Eigen::MatrixXd> exchangeDerivative; ///< K' per flat coordinate.
};

/// Walks the ordered quartets.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \param density The density matrix, n x n.
/// \param withDerivatives Build J' and K' as well.
/// \returns The reference, or an Error.
qcx::Result<OrderedFock> OrderedWalk(const qcx::molecule::Molecule& molecule,
                                     const qcx::basisset::BasisSet& basisSet,
                                     const Eigen::MatrixXd& density,
                                     bool withDerivatives) {
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
    const std::size_t n = pairList->functionCount;
    const std::size_t coordinateCount = 3 * molecule.AtomCount();
    OrderedFock reference;
    reference.coulomb =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    reference.exchange =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    if (withDerivatives)
    {
        reference.coulombDerivative.assign(
            coordinateCount,
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)));
        reference.exchangeDerivative.assign(
            coordinateCount,
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)));
    }

    MdEriDerivativeScratch scratch;
    std::vector<double> value;
    std::vector<double> derivatives;

    for (std::size_t i = 0; i < shellCount; ++i)
    {
        for (std::size_t j = 0; j < shellCount; ++j)
        {
            for (std::size_t k = 0; k < shellCount; ++k)
            {
                for (std::size_t l = 0; l < shellCount; ++l)
                {
                    const MdEriDerivativeQuartet quartet =
                        DerivativeQuartet(*pairList, *shells, {i, j, k, l});
                    const std::size_t elements =
                        qcx::integrals::internal::EriQuartetBlockElements(quartet);
                    value.assign(elements, 0.0);
                    qcx::integrals::internal::BuildEriQuartetValue(quartet, 0, value, scratch);

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
                    std::vector<std::size_t> coordinates;

                    if (withDerivatives)
                    {
                        coordinates = QuartetCoordinates(quartet.atoms);
                        derivatives.assign(coordinates.size() * elements, 0.0);
                        auto built = qcx::integrals::internal::BuildEriQuartetDerivative(
                            quartet, 1, coordinates, derivatives, scratch, molecule.AtomCount());

                        if (!built.has_value())
                        {
                            return std::unexpected(built.error());
                        }
                    }

                    for (std::size_t fa = 0; fa < nI; ++fa)
                    {
                        for (std::size_t fb = 0; fb < nJ; ++fb)
                        {
                            for (std::size_t fc = 0; fc < nK; ++fc)
                            {
                                for (std::size_t fd = 0; fd < nL; ++fd)
                                {
                                    const Eigen::Index mu = static_cast<Eigen::Index>(offI + fa);
                                    const Eigen::Index nu = static_cast<Eigen::Index>(offK + fc);
                                    const Eigen::Index lambda =
                                        static_cast<Eigen::Index>(offJ + fb);
                                    const Eigen::Index sigma = static_cast<Eigen::Index>(offL + fd);
                                    const std::size_t element = qcx::integrals::EriBlockIndex(
                                        fa, fb, fc, fd, nI, nJ, nK, nL);
                                    const double block = value[element];
                                    // Coulomb contracts the density block of
                                    // the two ket slots, exchange that of the
                                    // two inner slots.
                                    const double coulombDensity = density(nu, sigma);
                                    const double densityInner = density(lambda, sigma);

                                    reference.coulomb(mu, static_cast<Eigen::Index>(offJ + fb)) +=
                                        coulombDensity * block;
                                    reference.exchange(mu, nu) += densityInner * block;
                                    reference.coulombEnergy +=
                                        0.5 * density(mu, static_cast<Eigen::Index>(offJ + fb)) *
                                        coulombDensity * block;
                                    reference.exchangeEnergy +=
                                        0.5 * density(mu, nu) * densityInner * block;

                                    if (!withDerivatives)
                                    {
                                        continue;
                                    }

                                    for (std::size_t c = 0; c < coordinates.size(); ++c)
                                    {
                                        const double derivative =
                                            derivatives[c * elements + element];
                                        Eigen::MatrixXd& coulombDerivative =
                                            reference.coulombDerivative[coordinates[c]];
                                        Eigen::MatrixXd& exchangeDerivative =
                                            reference.exchangeDerivative[coordinates[c]];
                                        coulombDerivative(mu,
                                                          static_cast<Eigen::Index>(offJ + fb)) +=
                                            coulombDensity * derivative;
                                        exchangeDerivative(mu, nu) += densityInner * derivative;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return reference;
}

/// The MO-basis matrix of an AO matrix at the fixture's fixed coefficients:
/// C^T X C.
/// \param ao The AO matrix.
/// \param coefficients The MO coefficients, nBasis x nMo.
/// \returns The MO-basis matrix.
Eigen::MatrixXd Transform(const Eigen::MatrixXd& ao, const Eigen::MatrixXd& coefficients) {
    return coefficients.transpose() * ao * coefficients;
}

/// The fixture's low-symmetry water, its basis set, its fixed density and its
/// fixed MO coefficients.
struct Fixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basisSet;
    Eigen::MatrixXd density;
    CpuTensor2 densityTensor;
    Eigen::MatrixXd coefficients; ///< nBasis x nMo, the core eigenvectors.
    std::vector<double> rowMajor; ///< The same coefficients in the entry's layout.
    std::size_t numOccupied = 0;
    std::size_t numVirtual = 0;
};

/// Builds the fixture.
/// \returns The fixture, or an Error.
qcx::Result<Fixture> MakeFixture() {
    auto base = qcx::testing::MakeH2oSto3g();

    if (!base.has_value())
    {
        return std::unexpected(base.error());
    }

    auto molecule = DisplacedBase(*base);

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto basisSet = qcx::testing::MakeH2oSto3gBasis();

    if (!basisSet.has_value())
    {
        return std::unexpected(basisSet.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto attraction = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basisSet);

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
                                          "the core-Hamiltonian eigenvector solve failed"});
    }

    const std::size_t numOccupied =
        static_cast<std::size_t>(std::max(molecule->ElectronCount(), 0) / 2);
    const std::size_t n = static_cast<std::size_t>(metric.rows());
    const std::size_t numVirtual = n - numOccupied;
    const Eigen::MatrixXd coefficients =
        solver.eigenvectors().leftCols(static_cast<Eigen::Index>(numOccupied + numVirtual));

    // The core-Hamiltonian density: 2 C_occ C_occ^T.
    const Eigen::MatrixXd occupiedColumns =
        solver.eigenvectors().leftCols(static_cast<Eigen::Index>(numOccupied));
    const Eigen::MatrixXd density = 2.0 * occupiedColumns * occupiedColumns.transpose();

    auto tensor = qcx::testing::ToTensor(density);

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    CpuTensor2 densityTensor = std::move(*tensor);
    std::vector<double> rowMajor(
        static_cast<std::size_t>(coefficients.rows() * coefficients.cols()));

    for (Eigen::Index p = 0; p < coefficients.rows(); ++p)
    {
        for (Eigen::Index q = 0; q < coefficients.cols(); ++q)
        {
            rowMajor[static_cast<std::size_t>(p * coefficients.cols() + q)] = coefficients(p, q);
        }
    }

    return Fixture{std::move(*molecule),
                   std::move(*basisSet),
                   density,
                   std::move(densityTensor),
                   coefficients,
                   std::move(rowMajor),
                   numOccupied,
                   numVirtual};
}

/// The largest entry of |a - b|.
/// \param a The first matrix.
/// \param b The second matrix.
/// \returns The largest absolute difference.
double MaxDifference(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

/// The largest entry of |a - b|.
/// \param a The first vector.
/// \param b The second vector.
/// \returns The largest absolute difference.
double MaxDifference(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

/// The occupied-virtual block of one MO-basis matrix.
/// \param block The MO matrix, nMo x nMo.
/// \param numOccupied The occupied count.
/// \param numVirtual The virtual count.
/// \returns The virtual-by-occupied block.
Eigen::MatrixXd Block(const Eigen::MatrixXd& block,
                      std::size_t numOccupied,
                      std::size_t numVirtual) {
    Eigen::MatrixXd out(static_cast<Eigen::Index>(numVirtual),
                        static_cast<Eigen::Index>(numOccupied));

    for (std::size_t a = 0; a < numVirtual; ++a)
    {
        for (std::size_t i = 0; i < numOccupied; ++i)
        {
            out(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(i)) =
                block(static_cast<Eigen::Index>(numOccupied + a), static_cast<Eigen::Index>(i));
        }
    }

    return out;
}

/// The entry's stored rows of one block at one coordinate.
/// \param stored The entry's displacement-major matrix.
/// \param numOccupied The occupied count.
/// \param numVirtual The virtual count.
/// \param coordinate The flat coordinate.
/// \returns The stored sub-block.
Eigen::MatrixXd StoredBlock(const Eigen::MatrixXd& stored,
                            std::size_t numOccupied,
                            std::size_t numVirtual,
                            std::size_t coordinate) {
    Eigen::MatrixXd out(static_cast<Eigen::Index>(numVirtual),
                        static_cast<Eigen::Index>(numOccupied));

    for (std::size_t a = 0; a < numVirtual; ++a)
    {
        for (std::size_t i = 0; i < numOccupied; ++i)
        {
            out(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(i)) =
                stored(static_cast<Eigen::Index>(coordinate * numVirtual + a),
                       static_cast<Eigen::Index>(i));
        }
    }

    return out;
}

} // namespace

/// The energies, the AO-contracted MO blocks and the two skeletons against
/// the ordered walk. The skeletons here are the ordered reference's own
/// derivative contraction, 1/2 Tr(D J') and 1/2 Tr(D K'), so the entry's
/// exchange contraction - four targets, canonical multiplicities and a
/// doubled write per target - is checked against a walk that carries none of
/// those conventions.
TEST(TwoElectronFockDerivativeTest, TheResultsMatchTheOrderedWalk) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto reference = OrderedWalk(fixture->molecule, fixture->basisSet, fixture->density, true);

    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    qcx::integrals::TwoElectronFockDerivativeOptions options;
    options.useDensityScreening = false;
    auto derivative = qcx::integrals::ComputeTwoElectronFockDerivative(fixture->molecule,
                                                                       fixture->basisSet,
                                                                       fixture->densityTensor,
                                                                       fixture->rowMajor,
                                                                       fixture->numOccupied,
                                                                       fixture->numVirtual,
                                                                       options);

    ASSERT_TRUE(derivative.has_value()) << derivative.error().message;

    const std::size_t coordinateCount = 3 * fixture->molecule.AtomCount();
    const double energyGap =
        std::max(std::abs(derivative->coulombEnergy - reference->coulombEnergy),
                 std::abs(derivative->exchangeEnergy - reference->exchangeEnergy));
    RecordProperty("energy_gap", ShortDouble(energyGap));
    RecordProperty("coulomb_energy", ShortDouble(derivative->coulombEnergy));
    RecordProperty("exchange_energy", ShortDouble(derivative->exchangeEnergy));
    EXPECT_LT(energyGap, kReferenceTolerance);

    double worstBlock = 0.0;

    for (std::size_t coordinate = 0; coordinate < coordinateCount; ++coordinate)
    {
        const Eigen::MatrixXd coulomb =
            Transform(reference->coulombDerivative[coordinate], fixture->coefficients);
        const Eigen::MatrixXd exchange =
            Transform(reference->exchangeDerivative[coordinate], fixture->coefficients);
        worstBlock =
            std::max(worstBlock,
                     MaxDifference(StoredBlock(derivative->coulombOccupiedVirtual,
                                               fixture->numOccupied,
                                               fixture->numVirtual,
                                               coordinate),
                                   Block(coulomb, fixture->numOccupied, fixture->numVirtual)));
        worstBlock =
            std::max(worstBlock,
                     MaxDifference(StoredBlock(derivative->exchangeOccupiedVirtual,
                                               fixture->numOccupied,
                                               fixture->numVirtual,
                                               coordinate),
                                   Block(exchange, fixture->numOccupied, fixture->numVirtual)));
    }

    RecordProperty("worst_block", ShortDouble(worstBlock));
    EXPECT_LT(worstBlock, kReferenceTolerance);

    // The skeleton gradients from the ordered walks' own J' and K': the
    // density contraction 1/2 Tr(D X') at every coordinate.
    Eigen::VectorXd coulombSkeleton =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(coordinateCount));
    Eigen::VectorXd exchangeSkeleton =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(coordinateCount));

    for (std::size_t coordinate = 0; coordinate < coordinateCount; ++coordinate)
    {
        const Eigen::MatrixXd coulombDensity =
            (0.5 * fixture->density).cwiseProduct(reference->coulombDerivative[coordinate]);
        const Eigen::MatrixXd exchangeDensity =
            (0.5 * fixture->density).cwiseProduct(reference->exchangeDerivative[coordinate]);
        coulombSkeleton(static_cast<Eigen::Index>(coordinate)) = coulombDensity.sum();
        exchangeSkeleton(static_cast<Eigen::Index>(coordinate)) = exchangeDensity.sum();
    }

    const double skeletonGap =
        std::max(MaxDifference(derivative->coulombSkeleton, coulombSkeleton),
                 MaxDifference(derivative->exchangeSkeleton, exchangeSkeleton));
    RecordProperty("skeleton_gap", ShortDouble(skeletonGap));
    RecordProperty("coulomb_skeleton_scale",
                   ShortDouble(derivative->coulombSkeleton.cwiseAbs().maxCoeff()));
    RecordProperty("exchange_skeleton_scale",
                   ShortDouble(derivative->exchangeSkeleton.cwiseAbs().maxCoeff()));
    EXPECT_LT(skeletonGap, kReferenceTolerance);
}

/// The Coulomb half keeps exactly the quartets ComputeEriGradient keeps:
/// under the same preset the skeleton and the energy are that entry's,
/// accumulated from the same weights in the same order.
TEST(TwoElectronFockDerivativeTest, TheCoulombHalfIsTheEriGradient) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto gradient = qcx::integrals::ComputeEriGradient(
        fixture->molecule, fixture->basisSet, fixture->densityTensor);
    auto derivative = qcx::integrals::ComputeTwoElectronFockDerivative(fixture->molecule,
                                                                       fixture->basisSet,
                                                                       fixture->densityTensor,
                                                                       fixture->rowMajor,
                                                                       fixture->numOccupied,
                                                                       fixture->numVirtual);

    ASSERT_TRUE(gradient.has_value()) << gradient.error().message;
    ASSERT_TRUE(derivative.has_value()) << derivative.error().message;

    const double skeletonGap = MaxDifference(derivative->coulombSkeleton, gradient->gradient);
    const double energyGap = std::abs(derivative->coulombEnergy - gradient->energy);
    RecordProperty("skeleton_gap", ShortDouble(skeletonGap));
    RecordProperty("energy_gap", ShortDouble(energyGap));
    RecordProperty("quartets", std::to_string(derivative->counts.quartets));
    RecordProperty("screened_quartets", std::to_string(derivative->counts.screenedQuartets));
    EXPECT_LT(skeletonGap, kReferenceTolerance);
    EXPECT_LT(energyGap, kReferenceTolerance);
}

/// J' and K' against the central finite difference of the ordered walk's AO
/// matrices at the fixed coefficients: the entry's derivative is the
/// derivative of the block it reports, at every coordinate.
TEST(TwoElectronFockDerivativeTest, TheBlocksAreFiniteDifferences) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    qcx::integrals::TwoElectronFockDerivativeOptions options;
    options.useDensityScreening = false;
    auto derivative = qcx::integrals::ComputeTwoElectronFockDerivative(fixture->molecule,
                                                                       fixture->basisSet,
                                                                       fixture->densityTensor,
                                                                       fixture->rowMajor,
                                                                       fixture->numOccupied,
                                                                       fixture->numVirtual,
                                                                       options);

    ASSERT_TRUE(derivative.has_value()) << derivative.error().message;

    double worstCoulomb = 0.0;
    double worstExchange = 0.0;
    std::size_t worstCoordinate = 0;

    for (std::size_t flat = 0; flat < 3 * fixture->molecule.AtomCount(); ++flat)
    {
        auto plus = MakeDisplaced(fixture->molecule, {flat, kFiniteDifferenceStep});
        auto minus = MakeDisplaced(fixture->molecule, {flat, -kFiniteDifferenceStep});

        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        auto plusReference = OrderedWalk(*plus, fixture->basisSet, fixture->density, false);
        auto minusReference = OrderedWalk(*minus, fixture->basisSet, fixture->density, false);

        ASSERT_TRUE(plusReference.has_value()) << plusReference.error().message;
        ASSERT_TRUE(minusReference.has_value()) << minusReference.error().message;

        const Eigen::MatrixXd coulombNumeric = Transform(
            (plusReference->coulomb - minusReference->coulomb) / (2.0 * kFiniteDifferenceStep),
            fixture->coefficients);
        const Eigen::MatrixXd exchangeNumeric = Transform(
            (plusReference->exchange - minusReference->exchange) / (2.0 * kFiniteDifferenceStep),
            fixture->coefficients);

        const double coulombDeviation =
            MaxDifference(StoredBlock(derivative->coulombOccupiedVirtual,
                                      fixture->numOccupied,
                                      fixture->numVirtual,
                                      flat),
                          Block(coulombNumeric, fixture->numOccupied, fixture->numVirtual));
        const double exchangeDeviation =
            MaxDifference(StoredBlock(derivative->exchangeOccupiedVirtual,
                                      fixture->numOccupied,
                                      fixture->numVirtual,
                                      flat),
                          Block(exchangeNumeric, fixture->numOccupied, fixture->numVirtual));

        if (std::max(coulombDeviation, exchangeDeviation) > std::max(worstCoulomb, worstExchange))
        {
            worstCoordinate = flat;
        }

        worstCoulomb = std::max(worstCoulomb, coulombDeviation);
        worstExchange = std::max(worstExchange, exchangeDeviation);
    }

    RecordProperty("worst_coulomb", ShortDouble(worstCoulomb));
    RecordProperty("worst_exchange", ShortDouble(worstExchange));
    RecordProperty("worst_coordinate", std::to_string(worstCoordinate));
    EXPECT_LT(worstCoulomb, kFiniteDifferenceTolerance);
    EXPECT_LT(worstExchange, kFiniteDifferenceTolerance);
}

/// Translating every atom together changes no integral, so each MO block and
/// each skeleton sums to zero over the atoms of every axis.
TEST(TwoElectronFockDerivativeTest, TheBlocksAreTranslationInvariant) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    qcx::integrals::TwoElectronFockDerivativeOptions options;
    options.useDensityScreening = false;
    auto derivative = qcx::integrals::ComputeTwoElectronFockDerivative(fixture->molecule,
                                                                       fixture->basisSet,
                                                                       fixture->densityTensor,
                                                                       fixture->rowMajor,
                                                                       fixture->numOccupied,
                                                                       fixture->numVirtual,
                                                                       options);

    ASSERT_TRUE(derivative.has_value()) << derivative.error().message;

    const std::size_t atomCount = fixture->molecule.AtomCount();
    double worst = 0.0;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        for (std::size_t a = 0; a < fixture->numVirtual; ++a)
        {
            for (std::size_t i = 0; i < fixture->numOccupied; ++i)
            {
                double coulombSum = 0.0;
                double exchangeSum = 0.0;

                for (std::size_t atom = 0; atom < atomCount; ++atom)
                {
                    const Eigen::Index row =
                        static_cast<Eigen::Index>((3 * atom + axis) * fixture->numVirtual + a);
                    coulombSum +=
                        derivative->coulombOccupiedVirtual(row, static_cast<Eigen::Index>(i));
                    exchangeSum +=
                        derivative->exchangeOccupiedVirtual(row, static_cast<Eigen::Index>(i));
                }

                worst = std::max({worst, std::abs(coulombSum), std::abs(exchangeSum)});
            }
        }

        double coulombSkeletonSum = 0.0;
        double exchangeSkeletonSum = 0.0;

        for (std::size_t atom = 0; atom < atomCount; ++atom)
        {
            const Eigen::Index flat = static_cast<Eigen::Index>(3 * atom + axis);
            coulombSkeletonSum += derivative->coulombSkeleton(flat);
            exchangeSkeletonSum += derivative->exchangeSkeleton(flat);
        }

        worst = std::max({worst, std::abs(coulombSkeletonSum), std::abs(exchangeSkeletonSum)});
    }

    RecordProperty("worst_sum", ShortDouble(worst));
    EXPECT_LT(worst, kTranslationTolerance);
}

/// A warm start lifted over zero motion carries the very bounds the fresh
/// build would: value + 6 * 0 * |derivative| is the value itself. The entry
/// therefore returns the same bits as the run without one.
TEST(TwoElectronFockDerivativeTest, TheWarmStartIsExactAtZeroMotion) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto warmStart =
        qcx::integrals::BuildFockDerivativeWarmStart(fixture->molecule, fixture->basisSet);
    ASSERT_TRUE(warmStart.has_value()) << warmStart.error().message;
    warmStart->maxAtomMotionBohr = 0.0;

    qcx::integrals::TwoElectronFockDerivativeOptions options;
    options.warmStart = *warmStart;
    auto lifted = qcx::integrals::ComputeTwoElectronFockDerivative(fixture->molecule,
                                                                   fixture->basisSet,
                                                                   fixture->densityTensor,
                                                                   fixture->rowMajor,
                                                                   fixture->numOccupied,
                                                                   fixture->numVirtual,
                                                                   options);
    auto plain = qcx::integrals::ComputeTwoElectronFockDerivative(fixture->molecule,
                                                                  fixture->basisSet,
                                                                  fixture->densityTensor,
                                                                  fixture->rowMajor,
                                                                  fixture->numOccupied,
                                                                  fixture->numVirtual);

    ASSERT_TRUE(lifted.has_value()) << lifted.error().message;
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    EXPECT_EQ(MaxDifference(lifted->coulombOccupiedVirtual, plain->coulombOccupiedVirtual), 0.0);
    EXPECT_EQ(MaxDifference(lifted->exchangeOccupiedVirtual, plain->exchangeOccupiedVirtual), 0.0);
    EXPECT_EQ(MaxDifference(lifted->coulombSkeleton, plain->coulombSkeleton), 0.0);
    EXPECT_EQ(MaxDifference(lifted->exchangeSkeleton, plain->exchangeSkeleton), 0.0);
    EXPECT_EQ(lifted->coulombEnergy, plain->coulombEnergy);
    EXPECT_EQ(lifted->exchangeEnergy, plain->exchangeEnergy);
    EXPECT_EQ(lifted->counts.quartets, plain->counts.quartets);
    EXPECT_EQ(lifted->counts.screenedQuartets, plain->counts.screenedQuartets);
}

/// A warm start built at the base geometry and lifted over a motion that
/// covers the displacement loses nothing numerically significant: the blocks,
/// skeletons and energies stay inside the screen's own truncation of the
/// exact contraction. The screen is a truncation, not a different functional,
/// so this is the conservative direction the lift promises.
TEST(TwoElectronFockDerivativeTest, TheLiftedWarmStartLosesNothing) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    const std::size_t flat = 0;
    auto displaced = MakeDisplaced(fixture->molecule, {flat, kFiniteDifferenceStep});

    ASSERT_TRUE(displaced.has_value()) << displaced.error().message;

    auto warmStart =
        qcx::integrals::BuildFockDerivativeWarmStart(fixture->molecule, fixture->basisSet);
    ASSERT_TRUE(warmStart.has_value()) << warmStart.error().message;
    warmStart->maxAtomMotionBohr = kWarmStartMotion;

    qcx::integrals::TwoElectronFockDerivativeOptions liftedOptions;
    liftedOptions.warmStart = *warmStart;
    auto lifted = qcx::integrals::ComputeTwoElectronFockDerivative(*displaced,
                                                                   fixture->basisSet,
                                                                   fixture->densityTensor,
                                                                   fixture->rowMajor,
                                                                   fixture->numOccupied,
                                                                   fixture->numVirtual,
                                                                   liftedOptions);
    qcx::integrals::TwoElectronFockDerivativeOptions exactOptions;
    exactOptions.useDensityScreening = false;
    auto exact = qcx::integrals::ComputeTwoElectronFockDerivative(*displaced,
                                                                  fixture->basisSet,
                                                                  fixture->densityTensor,
                                                                  fixture->rowMajor,
                                                                  fixture->numOccupied,
                                                                  fixture->numVirtual,
                                                                  exactOptions);

    ASSERT_TRUE(lifted.has_value()) << lifted.error().message;
    ASSERT_TRUE(exact.has_value()) << exact.error().message;

    const double worst =
        std::max({MaxDifference(lifted->coulombOccupiedVirtual, exact->coulombOccupiedVirtual),
                  MaxDifference(lifted->exchangeOccupiedVirtual, exact->exchangeOccupiedVirtual),
                  MaxDifference(lifted->coulombSkeleton, exact->coulombSkeleton),
                  MaxDifference(lifted->exchangeSkeleton, exact->exchangeSkeleton),
                  std::abs(lifted->coulombEnergy - exact->coulombEnergy),
                  std::abs(lifted->exchangeEnergy - exact->exchangeEnergy)});
    RecordProperty("worst_difference", ShortDouble(worst));
    RecordProperty("lifted_quartets", std::to_string(lifted->counts.quartets));
    RecordProperty("exact_quartets", std::to_string(exact->counts.quartets));
    EXPECT_LT(worst, kLiftedTolerance);
}

/// The documented refusals: the coefficient and occupied/virtual shapes, a
/// carried screening structure of the wrong size, and a motion that is not a
/// finite non-negative displacement.
TEST(TwoElectronFockDerivativeTest, TheEntryNamesItsRefusals) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    const std::size_t n = static_cast<std::size_t>(fixture->coefficients.rows());

    {
        const std::size_t occupied = n - 1;
        const std::size_t virtualOrbitals = 2;
        std::vector<double> coefficients(n * (occupied + virtualOrbitals), 0.0);
        auto refused = qcx::integrals::ComputeTwoElectronFockDerivative(fixture->molecule,
                                                                        fixture->basisSet,
                                                                        fixture->densityTensor,
                                                                        coefficients,
                                                                        occupied,
                                                                        virtualOrbitals);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(refused.error().message.find("exceed the"), std::string::npos)
            << refused.error().message;
    }

    {
        std::vector<double> coefficients(fixture->rowMajor.begin(), fixture->rowMajor.end() - 1);
        auto refused = qcx::integrals::ComputeTwoElectronFockDerivative(fixture->molecule,
                                                                        fixture->basisSet,
                                                                        fixture->densityTensor,
                                                                        coefficients,
                                                                        fixture->numOccupied,
                                                                        fixture->numVirtual);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(refused.error().message.find("MO coefficient span holds"), std::string::npos)
            << refused.error().message;
    }

    {
        auto pairs = qcx::integrals::BuildShellPairs(fixture->molecule, fixture->basisSet);
        ASSERT_TRUE(pairs.has_value()) << pairs.error().message;
        qcx::integrals::FockDerivativeWarmStart warmStart;
        warmStart.pairValues.assign(pairs->pairs.size() - 1, 1.0);
        warmStart.pairDerivatives.assign(pairs->pairs.size() - 1, 1.0);
        warmStart.maxAtomMotionBohr = 1.0;
        qcx::integrals::TwoElectronFockDerivativeOptions options;
        options.warmStart = warmStart;
        auto refused = qcx::integrals::ComputeTwoElectronFockDerivative(fixture->molecule,
                                                                        fixture->basisSet,
                                                                        fixture->densityTensor,
                                                                        fixture->rowMajor,
                                                                        fixture->numOccupied,
                                                                        fixture->numVirtual,
                                                                        options);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(refused.error().message.find("canonical pairs of this basis set"),
                  std::string::npos)
            << refused.error().message;
    }

    {
        auto pairs = qcx::integrals::BuildShellPairs(fixture->molecule, fixture->basisSet);
        ASSERT_TRUE(pairs.has_value()) << pairs.error().message;
        qcx::integrals::FockDerivativeWarmStart warmStart;
        warmStart.pairValues.assign(pairs->pairs.size(), 1.0);
        warmStart.pairDerivatives.assign(pairs->pairs.size(), 1.0);
        warmStart.maxAtomMotionBohr = -0.5;
        qcx::integrals::TwoElectronFockDerivativeOptions options;
        options.warmStart = warmStart;
        auto refused = qcx::integrals::ComputeTwoElectronFockDerivative(fixture->molecule,
                                                                        fixture->basisSet,
                                                                        fixture->densityTensor,
                                                                        fixture->rowMajor,
                                                                        fixture->numOccupied,
                                                                        fixture->numVirtual,
                                                                        options);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(refused.error().message.find("finite non-negative displacement"),
                  std::string::npos)
            << refused.error().message;
    }
}
