// The RI Coulomb derivative tests (ri_gradient.hpp). The checks, in the order
// the path's acceptance rests on them:
//
//  - the order-1 derivatives against the CENTRAL FINITE DIFFERENCE of the
//    derivative call's OWN energy - the 1/2 d^T M^-1 d the call returns, so
//    the same quantity is reached two ways, and a gradient that
//    differentiates a different contraction than the one it reports fails
//    here. Run across the auxiliary ladder on a low-symmetry displaced H2O
//    geometry, so no entry is zero by symmetry alone;
//  - the order-2 derivatives against the central finite difference of the
//    order-1 gradient, every coordinate pair, plus the Hessian's own
//    symmetry and translation identities and its diagonal against the
//    energy's second difference;
//  - the fitted gradient against the analytic four-index Coulomb gradient
//    over an auxiliary ladder of increasing quality. What is under test is
//    that the DIFFERENCE is the fitting error and falls towards zero as the
//    fit improves - not that the two numbers agree, which they do not and
//    must not;
//  - the four-index reference's own contraction against the finite difference
//    of the four-index energy, so a mistake in the reference cannot be read
//    as a mistake in the fitted path;
//  - the edges: the wrapper against the general entry (bit-identical, by
//    contract), a retained set that does not move over the finite-difference
//    step, an exactly zero result at zero density, a single atom, a duplicate
//    auxiliary shell that puts the metric's inverse on its floor, and every
//    refusal the entry documents.
#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_eri_derivative.hpp"
#include "internal/shells_flat.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_gradient.hpp"
#include "qcx/integrals/shell_pairs.hpp"
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
#include <filesystem>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::integrals::ShellInfo;
using qcx::integrals::ShellPairList;
using qcx::integrals::internal::MdEriDerivativeQuartet;
using qcx::integrals::internal::MdEriDerivativeScratch;
using qcx::integrals::internal::MdShellInput;

/// The central-difference step: the truncation is O(h^2) and the cancellation
/// error O(eps E / h), both near 1e-11 at this fixture's O(1) magnitudes -
/// comfortably inside the 1e-7 threshold.
constexpr double kFiniteDifferenceStep = 1e-5;

/// The analytic-versus-finite-difference threshold of the derivatives.
constexpr double kFiniteDifferenceTolerance = 1e-7;

/// The exact-fit fixture's threshold: the fit is exact by construction, so
/// the two paths differ by the arithmetic alone - the metric solve's own
/// roundoff, scaled by the O(1) energies involved.
constexpr double kExactFitTolerance = 1e-10;

/// The translation sum's threshold. The identity is exact in arithmetic; in
/// floating point the sum is the roundoff of O(1) contributions, which this
/// fixture measures orders of magnitude below the gate.
constexpr double kTranslationTolerance = 1e-12;

/// The Hessian symmetry threshold: a mixed second derivative is one number
/// reached in two tuple orders, so the two entries differ by roundoff alone.
constexpr double kHessianSymmetryTolerance = 1e-12;

/// The diagonal second-difference step and threshold. The second difference
/// divides the energy's roundoff by h^2, which is why the step is larger than
/// the first-difference one and the threshold looser. Fitted from the measured
/// deviations at h = 1e-4 and 2e-4, this fixture's roundoff term is 1.6e-5 at
/// 1e-4 against an 8e-7 truncation term; the two balance near 2e-4, where the
/// total measures 7e-6 inside the threshold. The analytic diagonal is the
/// accurate side: the gradient finite difference above pins it to 1e-7.
constexpr double kSecondDifferenceStep = 2e-4;
constexpr double kSecondDifferenceTolerance = 1e-5;

/// The auxiliary ladder, coarse to fine: the orbital basis used as its own
/// auxiliary set (its highest angular momentum on oxygen is p, so an oxygen
/// p x p product has no d function to be fitted by), cc-pVDZ (d on oxygen, p
/// on hydrogen - the products are spanned), and the vendored Coulomb-fitting
/// set. The ladder's claim is about the DIRECTION of the residual across it,
/// not about any one rung's value.
struct AuxRung {
    const char* name = ""; ///< The name the recorded properties carry.
    const char* directory = ""; ///< The vendored directory under QcxBasisDataDir.
};

constexpr std::array<AuxRung, 3> kAuxLadder = {{
    {"sto3g", "sto-3g"},
    {"ccpvdz", "cc-pvdz"},
    {"jfit", "def2-universal-jfit"},
}};

/// The symmetry-breaking displacement of the fixture's three atoms, one
/// triple per atom in (x, y, z) order: the base geometry is put off the
/// water's symmetry planes so no derivative entry is zero by symmetry alone.
constexpr std::array<double, 9> kBaseShifts = {
    0.013, -0.007, 0.019, -0.011, 0.017, -0.023, 0.029, -0.013, 0.011};

/// The orbital basis of the exact-fit fixture: one primitive s and one
/// primitive p per hydrogen, both of exponent 1, so a product of two of them
/// on one centre is a single primitive of exponent 2 of the matching angular
/// momentum - the shape the auxiliary basis below holds exactly.
inline constexpr std::string_view kExactFitOrbital = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0     1.0
H    P
      1.0     1.0
END
)";

/// The auxiliary basis of the exact-fit fixture: at each hydrogen, the s and
/// the p of twice the orbital exponent, so the auxiliary span holds every
/// product the fixture's density carries - and the p makes the shells carry
/// unequal function counts, which is where a block read at the wrong offsets
/// stops being a rearrangement of the right one.
inline constexpr std::string_view kExactFitAuxiliary = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      2.0     1.0
H    P
      2.0     1.0
END
)";

/// Two identical s shells at each hydrogen plus the exact fit's p shell: the
/// auxiliary metric then carries one exact null direction per centre, so the
/// floored inverse is the only thing between the solve and a division by
/// zero.
inline constexpr std::string_view kDuplicateAuxiliary = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      2.0     1.0
H    S
      2.0     1.0
H    P
      2.0     1.0
END
)";

/// A double as a short scientific string, for the recorded properties.
std::string ShortDouble(double value) {
    char text[32] = {};
    std::snprintf(text, sizeof(text), "%.3e", value);
    return text;
}

/// Records one gradient's sum over each Cartesian axis. Moving every atom
/// together leaves every integral's value alone, so each of the three sums is
/// an exact zero - a walk that put a block's derivatives on the wrong
/// coordinates cannot produce one.
/// \param prefix The property-name prefix.
/// \param gradient The 3N gradient.
/// \param atomCount The molecule's atom count.
void RecordTranslationSums(const std::string& prefix,
                           const Eigen::VectorXd& gradient,
                           std::size_t atomCount) {
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        double sum = 0.0;

        for (std::size_t atom = 0; atom < atomCount; ++atom)
        {
            sum += gradient(static_cast<Eigen::Index>(3 * atom + axis));
        }

        ::testing::Test::RecordProperty(prefix + "_axis_" + std::to_string(axis), ShortDouble(sum));
    }
}

/// The largest of one gradient's three axis sums.
/// \param gradient The 3N gradient.
/// \param atomCount The molecule's atom count.
/// \returns The largest |sum| over the three axes.
double WorstTranslationSum(const Eigen::VectorXd& gradient, std::size_t atomCount) {
    double worst = 0.0;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        double sum = 0.0;

        for (std::size_t atom = 0; atom < atomCount; ++atom)
        {
            sum += gradient(static_cast<Eigen::Index>(3 * atom + axis));
        }

        worst = std::max(worst, std::abs(sum));
    }

    return worst;
}

/// The distinct atoms a quartet's four shells sit on, three flat coordinates
/// (3 * atom + axis) each, in ascending atom order. A coordinate whose atom
/// carries none of the quartet's shells contributes an exactly zero block, so
/// requesting the atoms the quartet actually carries is the whole of it.
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

/// The derivative tier's view of one quartet, at the fixture's flattened
/// shells and the pair list's atom indices.
/// \param pairList The shell pair list.
/// \param shells The flattened shell inputs (FlattenShells).
/// \param indices The quartet's four shell indices.
/// \returns The quartet the derivative builder takes.
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

/// The geometry with one flat coordinate displaced; errors when the canonical
/// renumbering would move an atom, which the finite-difference loop's
/// coordinate indexing assumes it does not.
/// \param base The base molecule.
/// \param flatCoordinate The coordinate (3 * atom + axis) to displace.
/// \param delta The displacement (Bohr).
/// \returns The displaced molecule, or an Error.
qcx::Result<qcx::molecule::Molecule> MakeDisplaced(const qcx::molecule::Molecule& base,
                                                   std::size_t flatCoordinate,
                                                   double delta) {
    auto coordinates = CpuTensor2::Create({base.AtomCount(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    for (std::size_t atom = 0; atom < base.AtomCount(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            (*coordinates)(atom, axis) = base.CoordinatesBohr()(atom, axis);
        }
    }

    (*coordinates)(flatCoordinate / 3, flatCoordinate % 3) += delta;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        base.Atoms(), std::move(*coordinates), base.Charge(), base.Multiplicity());
}

/// The four-index Coulomb gradient of a fixed density, differentiated
/// directly: 1/2 sum over the ordered shell quadruples of D_uv D_ls
/// d(uv|ls)/dR, through the same derivative tier. Summing the quadruples
/// without canonicalization keeps the reference free of the multiplicity
/// bookkeeping the fitted walk does with its pair factors, so the two walks
/// agree only where the fitted one's counting is right.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \param density The density matrix, n x n.
/// \returns The 3N gradient, or an Error.
qcx::Result<Eigen::VectorXd> ExactCoulombGradient(const qcx::molecule::Molecule& molecule,
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
    Eigen::VectorXd gradient =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(3 * molecule.AtomCount()));
    MdEriDerivativeScratch scratch;
    std::vector<double> blocks;

    for (std::size_t i = 0; i < shellCount; ++i)
    {
        for (std::size_t j = 0; j < shellCount; ++j)
        {
            for (std::size_t k = 0; k < shellCount; ++k)
            {
                for (std::size_t l = 0; l < shellCount; ++l)
                {
                    const std::array<std::size_t, 4> indices = {i, j, k, l};
                    const MdEriDerivativeQuartet quartet =
                        DerivativeQuartet(*pairList, *shells, indices);
                    const std::vector<std::size_t> coordinates = QuartetCoordinates(quartet.atoms);
                    const std::size_t elements =
                        qcx::integrals::internal::EriQuartetBlockElements(quartet);
                    blocks.assign(coordinates.size() * elements, 0.0);
                    auto built = qcx::integrals::internal::BuildEriQuartetDerivative(
                        quartet, 1, coordinates, blocks, scratch, molecule.AtomCount());

                    if (!built.has_value())
                    {
                        return std::unexpected(built.error());
                    }

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
                                    const double weight =
                                        density(static_cast<Eigen::Index>(offI + fa),
                                                static_cast<Eigen::Index>(offJ + fb)) *
                                        density(static_cast<Eigen::Index>(offK + fc),
                                                static_cast<Eigen::Index>(offL + fd));
                                    const std::size_t element = qcx::integrals::EriBlockIndex(
                                        fa, fb, fc, fd, nI, nJ, nK, nL);

                                    for (std::size_t c = 0; c < coordinates.size(); ++c)
                                    {
                                        gradient(static_cast<Eigen::Index>(coordinates[c])) +=
                                            0.5 * weight * blocks[c * elements + element];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return gradient;
}

/// The four-index Coulomb energy of a fixed density: 1/2 sum over the ordered
/// shell quadruples of D_uv D_ls (uv|ls), one quartet built at a time. The
/// reference gradient above is the derivative of THIS functional, so the two
/// are one contraction reached two ways: the same enumeration, the same
/// weights, one reading values and one reading derivatives.
///
/// The batch form would put the engine's quartet canonicalization between the
/// requested form and the blocks it returns, which is a second convention to
/// get right in a reference that exists to be independent of the path under
/// test. Building each quartet in the order it is contracted in has none.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \param density The density matrix, n x n.
/// \returns The energy (Hartree), or an Error.
qcx::Result<double> ExactCoulombEnergy(const qcx::molecule::Molecule& molecule,
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
    MdEriDerivativeScratch scratch;
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
                    const MdEriDerivativeQuartet quartet =
                        DerivativeQuartet(*pairList, *shells, {i, j, k, l});
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

    auto density = CoreDensity(*molecule, *basisSet);

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
        std::move(*molecule), std::move(*basisSet), std::move(*density), std::move(*tensor)};
}

/// The vendored auxiliary basis of one ladder rung, for the fixture's own
/// elements: a directory-wide parse would carry every element the file set
/// holds, and the fixture's molecule has two.
/// \param rung The rung.
/// \returns The parsed basis set, or an Error.
qcx::Result<qcx::basisset::BasisSet> AuxOfRung(const AuxRung& rung) {
    const std::filesystem::path root(QcxBasisDataDir);
    return qcx::basisset::ParseNwchemDirectoryFiltered((root / rung.directory).string(),
                                                       std::array<int, 2>{1, 8});
}

/// The function count of one basis set, through its own shell pair list.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \returns The function count.
std::size_t FunctionCount(const qcx::molecule::Molecule& molecule,
                          const qcx::basisset::BasisSet& basisSet) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);
    EXPECT_TRUE(pairList.has_value());
    return pairList.has_value() ? pairList->functionCount : 0;
}

/// The order-2 enumeration as a dense matrix, row = first coordinate, column
/// = second, in atom-major flat-coordinate order.
/// \param derivative The order-2 result.
/// \param count The flat-coordinate count, 3 * atomCount.
/// \returns The Hessian.
Eigen::MatrixXd ToHessian(const qcx::integrals::RiCoulombDerivative& derivative,
                          std::size_t count) {
    Eigen::MatrixXd hessian(static_cast<Eigen::Index>(count), static_cast<Eigen::Index>(count));

    for (std::size_t first = 0; first < count; ++first)
    {
        for (std::size_t second = 0; second < count; ++second)
        {
            hessian(static_cast<Eigen::Index>(first), static_cast<Eigen::Index>(second)) =
                derivative.values(static_cast<Eigen::Index>(first * count + second));
        }
    }

    return hessian;
}

} // namespace

/// The order-1 derivatives are the derivative of the energy the same call
/// returns. The energy travels through the metric inverse and the three-index
/// tensor at every displaced geometry, so this finite difference
/// differentiates the fitting coefficients as well - and a walk that dropped
/// the coefficients' own geometry dependence fails it by that term's size.
///
/// The ladder runs coarse to fine on the displaced water: the top rung's f
/// and g auxiliary shells exceed a capped build's engine cap, so that rung
/// runs only where the full class matrix is instantiated. The coarse rung
/// runs in every configuration.
TEST(RiCoulombGradientTest, TheGradientIsTheDerivativeOfItsOwnEnergy) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    std::vector<const AuxRung*> rungs = {&kAuxLadder[0]};

    if (!qcx::testing::IsFastOnlyMode())
    {
        rungs.push_back(&kAuxLadder[1]);
    }

    if (qcx::integrals::SupportsL(4))
    {
        rungs.push_back(&kAuxLadder[2]);
    }

    for (const AuxRung* rung : rungs)
    {
        auto aux = AuxOfRung(*rung);

        ASSERT_TRUE(aux.has_value()) << rung->name << ": " << aux.error().message;

        auto fitted = qcx::integrals::ComputeRiCoulombGradient(
            fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor);

        ASSERT_TRUE(fitted.has_value()) << rung->name << ": " << fitted.error().message;
        ASSERT_EQ(fitted->gradient.size(),
                  static_cast<Eigen::Index>(3 * fixture->molecule.AtomCount()));

        double worst = 0.0;
        std::size_t worstCoordinate = 0;

        for (std::size_t flat = 0; flat < 3 * fixture->molecule.AtomCount(); ++flat)
        {
            auto plus = MakeDisplaced(fixture->molecule, flat, kFiniteDifferenceStep);
            auto minus = MakeDisplaced(fixture->molecule, flat, -kFiniteDifferenceStep);

            ASSERT_TRUE(plus.has_value()) << plus.error().message;
            ASSERT_TRUE(minus.has_value()) << minus.error().message;

            auto plusEnergy = qcx::integrals::ComputeRiCoulombGradient(
                *plus, fixture->basisSet, *aux, fixture->densityTensor);
            auto minusEnergy = qcx::integrals::ComputeRiCoulombGradient(
                *minus, fixture->basisSet, *aux, fixture->densityTensor);

            ASSERT_TRUE(plusEnergy.has_value()) << plusEnergy.error().message;
            ASSERT_TRUE(minusEnergy.has_value()) << minusEnergy.error().message;

            const double numeric =
                (plusEnergy->energy - minusEnergy->energy) / (2.0 * kFiniteDifferenceStep);
            const double deviation =
                std::abs(numeric - fitted->gradient(static_cast<Eigen::Index>(flat)));

            if (deviation > worst)
            {
                worst = deviation;
                worstCoordinate = flat;
            }
        }

        RecordProperty(std::string("worst_deviation_") + rung->name, ShortDouble(worst));
        RecordProperty(std::string("energy_") + rung->name, ShortDouble(fitted->energy));
        RecordProperty(std::string("density_trace") + rung->name,
                       ShortDouble(fixture->density.trace()));
        RecordProperty(std::string("tensor_quartets_") + rung->name,
                       std::to_string(fitted->counts.tensorQuartets));
        RecordProperty(std::string("metric_quartets_") + rung->name,
                       std::to_string(fitted->counts.metricQuartets));
        RecordTranslationSums(std::string("fitted_translation_") + rung->name,
                              fitted->gradient,
                              fixture->molecule.AtomCount());
        EXPECT_LT(worst, kFiniteDifferenceTolerance)
            << rung->name << ", coordinate " << worstCoordinate << " of "
            << 3 * fixture->molecule.AtomCount();
        EXPECT_LT(WorstTranslationSum(fitted->gradient, fixture->molecule.AtomCount()),
                  kTranslationTolerance)
            << rung->name;
    }
}

/// The order-2 derivatives are the derivative of the order-1 gradient: every
/// one of the 9N^2 entries against the central difference of a gradient
/// assembled at the displaced geometries, with the same screened retained set
/// at every geometry.
TEST(RiCoulombGradientTest, TheHessianIsTheDerivativeOfTheGradient) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto aux = AuxOfRung(kAuxLadder[0]);

    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    qcx::integrals::RiCoulombDerivativeOptions options;
    options.order = 2;
    auto hessian = qcx::integrals::ComputeRiCoulombDerivative(
        fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor, options);

    ASSERT_TRUE(hessian.has_value()) << hessian.error().message;

    const std::size_t count = 3 * fixture->molecule.AtomCount();
    ASSERT_EQ(hessian->values.size(), static_cast<Eigen::Index>(count * count));
    ASSERT_EQ(hessian->counts.order, 2);
    ASSERT_EQ(hessian->counts.tuples, count * count);

    const Eigen::MatrixXd matrix = ToHessian(*hessian, count);
    double worst = 0.0;
    std::size_t worstFirst = 0;
    std::size_t worstSecond = 0;

    for (std::size_t second = 0; second < count; ++second)
    {
        auto plus = MakeDisplaced(fixture->molecule, second, kFiniteDifferenceStep);
        auto minus = MakeDisplaced(fixture->molecule, second, -kFiniteDifferenceStep);

        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        qcx::integrals::RiCoulombDerivativeOptions gradientOptions;
        auto plusGradient = qcx::integrals::ComputeRiCoulombDerivative(
            *plus, fixture->basisSet, *aux, fixture->densityTensor, gradientOptions);
        auto minusGradient = qcx::integrals::ComputeRiCoulombDerivative(
            *minus, fixture->basisSet, *aux, fixture->densityTensor, gradientOptions);

        ASSERT_TRUE(plusGradient.has_value()) << plusGradient.error().message;
        ASSERT_TRUE(minusGradient.has_value()) << minusGradient.error().message;
        ASSERT_EQ(plusGradient->values.size(), static_cast<Eigen::Index>(count));
        ASSERT_EQ(minusGradient->values.size(), static_cast<Eigen::Index>(count));

        for (std::size_t first = 0; first < count; ++first)
        {
            const double numeric = (plusGradient->values(static_cast<Eigen::Index>(first)) -
                                    minusGradient->values(static_cast<Eigen::Index>(first))) /
                                   (2.0 * kFiniteDifferenceStep);
            const double deviation = std::abs(numeric - matrix(static_cast<Eigen::Index>(first),
                                                               static_cast<Eigen::Index>(second)));

            if (deviation > worst)
            {
                worst = deviation;
                worstFirst = first;
                worstSecond = second;
            }
        }
    }

    RecordProperty("worst_deviation", ShortDouble(worst));
    RecordProperty("hessian_scale", ShortDouble(matrix.cwiseAbs().maxCoeff()));
    RecordProperty("tensor_quartets", std::to_string(hessian->counts.tensorQuartets));
    EXPECT_LT(worst, kFiniteDifferenceTolerance)
        << "entry (" << worstFirst << ", " << worstSecond << ")";
}

/// A mixed second derivative is one number reached in two tuple orders, so
/// the Hessian is symmetric to roundoff - a walk that dropped the cross terms
/// of one orientation would show it here.
TEST(RiCoulombGradientTest, TheHessianIsSymmetric) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto aux = AuxOfRung(kAuxLadder[0]);

    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    qcx::integrals::RiCoulombDerivativeOptions options;
    options.order = 2;
    auto hessian = qcx::integrals::ComputeRiCoulombDerivative(
        fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor, options);

    ASSERT_TRUE(hessian.has_value()) << hessian.error().message;

    const std::size_t count = 3 * fixture->molecule.AtomCount();
    const Eigen::MatrixXd matrix = ToHessian(*hessian, count);
    const double asymmetry = (matrix - matrix.transpose()).cwiseAbs().maxCoeff();

    RecordProperty("asymmetry", ShortDouble(asymmetry));
    RecordProperty("hessian_scale", ShortDouble(matrix.cwiseAbs().maxCoeff()));
    EXPECT_LT(asymmetry, kHessianSymmetryTolerance);
}

/// Translating every atom together leaves the energy alone, so for each
/// Cartesian axis the derivative of the gradient's axis sum vanishes:
/// sum_atoms H(3 * atom + axis, any coordinate) is zero.
TEST(RiCoulombGradientTest, TheHessianIsTranslationInvariant) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto aux = AuxOfRung(kAuxLadder[0]);

    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    qcx::integrals::RiCoulombDerivativeOptions options;
    options.order = 2;
    auto hessian = qcx::integrals::ComputeRiCoulombDerivative(
        fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor, options);

    ASSERT_TRUE(hessian.has_value()) << hessian.error().message;

    const std::size_t atomCount = fixture->molecule.AtomCount();
    const std::size_t count = 3 * atomCount;
    const Eigen::MatrixXd matrix = ToHessian(*hessian, count);
    double worst = 0.0;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        for (std::size_t second = 0; second < count; ++second)
        {
            double sum = 0.0;

            for (std::size_t atom = 0; atom < atomCount; ++atom)
            {
                sum += matrix(static_cast<Eigen::Index>(3 * atom + axis),
                              static_cast<Eigen::Index>(second));
            }

            worst = std::max(worst, std::abs(sum));
        }
    }

    RecordProperty("worst_sum", ShortDouble(worst));
    RecordProperty("hessian_scale", ShortDouble(matrix.cwiseAbs().maxCoeff()));
    EXPECT_LT(worst, kTranslationTolerance);
}

/// A diagonal second derivative is also the energy's second difference along
/// that coordinate, at a step large enough to keep the division's roundoff
/// inside the threshold.
TEST(RiCoulombGradientTest, TheDiagonalMatchesTheEnergySecondDifference) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto aux = AuxOfRung(kAuxLadder[0]);

    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto energy = qcx::integrals::ComputeRiCoulombDerivative(
        fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor);

    ASSERT_TRUE(energy.has_value()) << energy.error().message;

    double worst = 0.0;

    for (std::size_t flat = 0; flat < 3 * fixture->molecule.AtomCount(); ++flat)
    {
        const std::array<std::size_t, 2> tuple = {flat, flat};
        qcx::integrals::RiCoulombDerivativeOptions options;
        options.order = 2;
        auto diagonal = qcx::integrals::ComputeRiCoulombDerivative(
            fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor, tuple, options);

        ASSERT_TRUE(diagonal.has_value()) << diagonal.error().message;
        ASSERT_EQ(diagonal->values.size(), 1);

        auto plus = MakeDisplaced(fixture->molecule, flat, kSecondDifferenceStep);
        auto minus = MakeDisplaced(fixture->molecule, flat, -kSecondDifferenceStep);

        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        auto plusEnergy = qcx::integrals::ComputeRiCoulombDerivative(
            *plus, fixture->basisSet, *aux, fixture->densityTensor);
        auto minusEnergy = qcx::integrals::ComputeRiCoulombDerivative(
            *minus, fixture->basisSet, *aux, fixture->densityTensor);

        ASSERT_TRUE(plusEnergy.has_value()) << plusEnergy.error().message;
        ASSERT_TRUE(minusEnergy.has_value()) << minusEnergy.error().message;

        const double numeric = (plusEnergy->energy - 2.0 * energy->energy + minusEnergy->energy) /
                               (kSecondDifferenceStep * kSecondDifferenceStep);
        worst = std::max(worst, std::abs(numeric - diagonal->values(0)));
    }

    RecordProperty("worst_deviation", ShortDouble(worst));
    EXPECT_LT(worst, kSecondDifferenceTolerance);
}

/// The wrapper is the general entry at order 1 with every coordinate
/// enumerated, by contract: same entries, same bits.
TEST(RiCoulombGradientTest, TheWrapperIsTheGeneralEntryBitForBit) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto aux = AuxOfRung(kAuxLadder[0]);

    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto wrapper = qcx::integrals::ComputeRiCoulombGradient(
        fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor);
    auto general = qcx::integrals::ComputeRiCoulombDerivative(
        fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor);

    ASSERT_TRUE(wrapper.has_value()) << wrapper.error().message;
    ASSERT_TRUE(general.has_value()) << general.error().message;
    ASSERT_EQ(wrapper->gradient.size(), general->values.size());

    for (Eigen::Index i = 0; i < wrapper->gradient.size(); ++i)
    {
        EXPECT_EQ(wrapper->gradient(i), general->values(i)) << "entry " << i;
    }

    EXPECT_EQ(wrapper->energy, general->energy);
    EXPECT_EQ(wrapper->counts.orbitalPairs, general->counts.orbitalPairs);
    EXPECT_EQ(wrapper->counts.auxShells, general->counts.auxShells);
    EXPECT_EQ(wrapper->counts.tensorQuartets, general->counts.tensorQuartets);
    EXPECT_EQ(wrapper->counts.metricQuartets, general->counts.metricQuartets);
}

/// The retained task set is the union of the energy predicate and the
/// derivative-aware one; over the finite-difference step it must not move.
/// The count is the visible part of that: an equal count at the base and both
/// displacements is what makes the finite-difference gates read one
/// functional.
TEST(RiCoulombGradientTest, TheRetainedSetDoesNotMoveOverTheDifferenceStep) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto aux = AuxOfRung(kAuxLadder[0]);

    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto base = qcx::integrals::ComputeRiCoulombDerivative(
        fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor);

    ASSERT_TRUE(base.has_value()) << base.error().message;

    for (std::size_t flat = 0; flat < 3 * fixture->molecule.AtomCount(); ++flat)
    {
        for (const double delta : {kFiniteDifferenceStep, -kFiniteDifferenceStep})
        {
            auto moved = MakeDisplaced(fixture->molecule, flat, delta);

            ASSERT_TRUE(moved.has_value()) << moved.error().message;

            auto displaced = qcx::integrals::ComputeRiCoulombDerivative(
                *moved, fixture->basisSet, *aux, fixture->densityTensor);

            ASSERT_TRUE(displaced.has_value()) << displaced.error().message;
            ASSERT_EQ(displaced->counts.order, 1);
            EXPECT_EQ(displaced->counts.tensorTasksOffered, base->counts.tensorTasksOffered)
                << "coordinate " << flat << ", displacement " << delta;
            EXPECT_EQ(displaced->counts.tensorQuartets, base->counts.tensorQuartets)
                << "coordinate " << flat << ", displacement " << delta;
            EXPECT_EQ(displaced->counts.metricQuartets, base->counts.metricQuartets)
                << "coordinate " << flat << ", displacement " << delta;
        }
    }
}

/// A zero density is an exactly zero contraction: no product is formed, so
/// every entry and the energy are the floating-point zero itself, at both
/// orders.
TEST(RiCoulombGradientTest, AZeroDensityIsExactlyZero) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto aux = AuxOfRung(kAuxLadder[0]);

    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    const std::size_t n = static_cast<std::size_t>(fixture->density.rows());
    auto tensor = qcx::testing::ToTensor(
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)));

    ASSERT_TRUE(tensor.has_value()) << tensor.error().message;

    for (const int order : {1, 2})
    {
        qcx::integrals::RiCoulombDerivativeOptions options;
        options.order = order;
        auto derivative = qcx::integrals::ComputeRiCoulombDerivative(
            fixture->molecule, fixture->basisSet, *aux, *tensor, options);

        ASSERT_TRUE(derivative.has_value()) << order << ": " << derivative.error().message;
        EXPECT_EQ(derivative->energy, 0.0) << "order " << order;

        for (Eigen::Index i = 0; i < derivative->values.size(); ++i)
        {
            EXPECT_EQ(derivative->values(i), 0.0) << "order " << order << ", entry " << i;
        }
    }
}

/// A single atom's integrals depend on no distance, so every derivative is
/// exactly zero - at both orders. The fixture carries s and p functions on
/// both sides, so the zero is the walk's, not the basis's.
TEST(RiCoulombGradientTest, ASingleAtomHasZeroDerivatives) {
    auto h2 = qcx::testing::MakeH2Sto3g();

    ASSERT_TRUE(h2.has_value()) << h2.error().message;

    auto coordinates = CpuTensor2::Create({1, 3});

    ASSERT_TRUE(coordinates.has_value()) << coordinates.error().message;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        (*coordinates)(0, axis) = 0.0;
    }

    coordinates->MarkHostDirty();
    auto molecule =
        qcx::molecule::Molecule::Create({h2->Atoms()[0]}, std::move(*coordinates), 0, 2);

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::basisset::ParseNwchemText(kExactFitOrbital);

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kExactFitAuxiliary);

    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const std::size_t n = pairList->functionCount;
    Eigen::MatrixXd density =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t f = 0; f < n; ++f)
    {
        density(static_cast<Eigen::Index>(f), static_cast<Eigen::Index>(f)) = 1.0;
    }

    auto tensor = qcx::testing::ToTensor(density);

    ASSERT_TRUE(tensor.has_value()) << tensor.error().message;

    for (const int order : {1, 2})
    {
        qcx::integrals::RiCoulombDerivativeOptions options;
        options.order = order;
        auto derivative =
            qcx::integrals::ComputeRiCoulombDerivative(*molecule, *basis, *aux, *tensor, options);

        ASSERT_TRUE(derivative.has_value()) << order << ": " << derivative.error().message;
        EXPECT_EQ(derivative->values.size(), static_cast<Eigen::Index>(order == 1 ? 3 : 9));
        EXPECT_LT(derivative->values.cwiseAbs().maxCoeff(), kTranslationTolerance)
            << "order " << order;
        EXPECT_TRUE(derivative->values.allFinite());
    }
}

/// An auxiliary basis carrying a duplicated shell makes the metric singular.
/// The floored inverse zeroes the null direction; the energy stays finite and
/// the derivative stays the finite difference of that energy, because the
/// null space does not turn with the geometry. A pseudo-inverse applied
/// without a floor, or a derivative that differentiated a solve it did not
/// report, fails here - the first as a non-finite value, the second as a
/// finite-difference gap.
TEST(RiCoulombGradientTest, ADuplicateAuxiliaryShellRidesTheMetricFloor) {
    auto molecule = qcx::testing::MakeH2Sto3g();

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::basisset::ParseNwchemText(kExactFitOrbital);

    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto aux = qcx::basisset::ParseNwchemText(kDuplicateAuxiliary);

    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    constexpr std::size_t kFixtureFunctions = 8;
    Eigen::MatrixXd density = Eigen::MatrixXd::Zero(kFixtureFunctions, kFixtureFunctions);

    for (std::size_t centre = 0; centre < 2; ++centre)
    {
        const Eigen::Index s = static_cast<Eigen::Index>(4 * centre);
        density(s, s) = 1.0;
        density(s, s + 1) = 1.0;
        density(s + 1, s) = 1.0;
    }

    auto tensor = qcx::testing::ToTensor(density);

    ASSERT_TRUE(tensor.has_value()) << tensor.error().message;

    auto fitted = qcx::integrals::ComputeRiCoulombDerivative(*molecule, *basis, *aux, *tensor);

    ASSERT_TRUE(fitted.has_value()) << fitted.error().message;
    EXPECT_TRUE(std::isfinite(fitted->energy));
    EXPECT_TRUE(fitted->values.allFinite());
    EXPECT_FALSE(fitted->values.hasNaN());

    double worst = 0.0;

    for (std::size_t flat = 0; flat < 3 * molecule->AtomCount(); ++flat)
    {
        auto plus = MakeDisplaced(*molecule, flat, kFiniteDifferenceStep);
        auto minus = MakeDisplaced(*molecule, flat, -kFiniteDifferenceStep);

        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        auto plusEnergy = qcx::integrals::ComputeRiCoulombDerivative(*plus, *basis, *aux, *tensor);
        auto minusEnergy =
            qcx::integrals::ComputeRiCoulombDerivative(*minus, *basis, *aux, *tensor);

        ASSERT_TRUE(plusEnergy.has_value()) << plusEnergy.error().message;
        ASSERT_TRUE(minusEnergy.has_value()) << minusEnergy.error().message;
        ASSERT_TRUE(std::isfinite(plusEnergy->energy));
        ASSERT_TRUE(std::isfinite(minusEnergy->energy));

        const double numeric =
            (plusEnergy->energy - minusEnergy->energy) / (2.0 * kFiniteDifferenceStep);
        worst =
            std::max(worst, std::abs(numeric - fitted->values(static_cast<Eigen::Index>(flat))));
    }

    RecordProperty("energy", ShortDouble(fitted->energy));
    RecordProperty("worst_deviation", ShortDouble(worst));
    EXPECT_LT(worst, kFiniteDifferenceTolerance);
}

/// The four-index reference's own contraction: its analytic gradient against
/// the finite difference of the four-index energy it claims to differentiate.
/// The reference is what the acceptance test below measures the fitted walk
/// against, so it is checked before it is used.
TEST(RiCoulombGradientTest, TheFourIndexReferenceMatchesItsOwnEnergy) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto reference = ExactCoulombGradient(fixture->molecule, fixture->basisSet, fixture->density);

    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    double worst = 0.0;

    for (std::size_t flat = 0; flat < 3 * fixture->molecule.AtomCount(); ++flat)
    {
        auto plus = MakeDisplaced(fixture->molecule, flat, kFiniteDifferenceStep);
        auto minus = MakeDisplaced(fixture->molecule, flat, -kFiniteDifferenceStep);

        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        auto plusEnergy = ExactCoulombEnergy(*plus, fixture->basisSet, fixture->density);
        auto minusEnergy = ExactCoulombEnergy(*minus, fixture->basisSet, fixture->density);

        ASSERT_TRUE(plusEnergy.has_value()) << plusEnergy.error().message;
        ASSERT_TRUE(minusEnergy.has_value()) << minusEnergy.error().message;

        const double numeric = (*plusEnergy - *minusEnergy) / (2.0 * kFiniteDifferenceStep);
        worst = std::max(worst, std::abs(numeric - (*reference)(static_cast<Eigen::Index>(flat))));
    }

    auto referenceEnergy =
        ExactCoulombEnergy(fixture->molecule, fixture->basisSet, fixture->density);

    ASSERT_TRUE(referenceEnergy.has_value()) << referenceEnergy.error().message;

    RecordProperty("worst_deviation", ShortDouble(worst));
    RecordProperty("reference_energy", ShortDouble(*referenceEnergy));
    RecordProperty("reference_scale", ShortDouble(reference->cwiseAbs().maxCoeff()));
    RecordTranslationSums("reference_translation", *reference, fixture->molecule.AtomCount());
    EXPECT_LT(worst, kFiniteDifferenceTolerance);
    EXPECT_LT(WorstTranslationSum(*reference, fixture->molecule.AtomCount()),
              kTranslationTolerance);
}

/// The one comparison in this file with no fitting error to absorb a missing
/// term: an auxiliary basis that spans the orbital products makes the fit
/// exact, so the fitted energy and gradient must BE the four-index ones, not
/// merely close to them. The screen is off here, so the fitted walk is the
/// exact contraction as well.
///
/// The density carries only same-centre s-s and s-p products (the p-p element
/// is zero), each of which is one auxiliary function on that centre up to a
/// scalar the fit absorbs. A gradient that dropped the fitting coefficients'
/// own geometry dependence, or a metric term carrying the wrong factor, cannot
/// hide behind a fit error here: it shows up at the size of the term it
/// dropped.
TEST(RiCoulombGradientTest, AnExactFitReproducesTheFourIndexWalk) {
    auto molecule = qcx::testing::MakeH2Sto3g();

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::basisset::ParseNwchemText(kExactFitOrbital);

    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kExactFitAuxiliary);

    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    // A fixed symmetric density whose charge distribution the auxiliary span
    // holds exactly: on each hydrogen the s-s element and the s-p element, and
    // no p-p element - the product of two p functions is the one shape a
    // single-exponent auxiliary basis does not carry.
    constexpr std::size_t kFixtureFunctions = 8;
    Eigen::MatrixXd density = Eigen::MatrixXd::Zero(kFixtureFunctions, kFixtureFunctions);

    for (std::size_t centre = 0; centre < 2; ++centre)
    {
        const Eigen::Index s = static_cast<Eigen::Index>(4 * centre);
        density(s, s) = 1.0;
        density(s, s + 1) = 1.0;
        density(s + 1, s) = 1.0;
    }

    auto tensor = qcx::testing::ToTensor(density);

    ASSERT_TRUE(tensor.has_value()) << tensor.error().message;

    auto reference = ExactCoulombGradient(*molecule, *basis, density);
    auto referenceEnergy = ExactCoulombEnergy(*molecule, *basis, density);

    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    ASSERT_TRUE(referenceEnergy.has_value()) << referenceEnergy.error().message;

    qcx::integrals::RiCoulombDerivativeOptions options;
    options.useDensityScreening = false;
    auto fitted =
        qcx::integrals::ComputeRiCoulombDerivative(*molecule, *basis, *aux, *tensor, options);

    ASSERT_TRUE(fitted.has_value()) << fitted.error().message;

    RecordProperty("fitted_energy", ShortDouble(fitted->energy));
    RecordProperty("reference_energy", ShortDouble(*referenceEnergy));
    RecordProperty("energy_gap", ShortDouble(std::abs(fitted->energy - *referenceEnergy)));
    EXPECT_LT(std::abs(fitted->energy - *referenceEnergy), kExactFitTolerance)
        << "the auxiliary span holds every product the density carries, so the fit is exact";

    const double againstReference = (fitted->values - *reference).cwiseAbs().maxCoeff();
    RecordProperty("gradient_against_reference", ShortDouble(againstReference));
    EXPECT_LT(againstReference, kExactFitTolerance);

    double worst = 0.0;

    for (std::size_t flat = 0; flat < 3 * molecule->AtomCount(); ++flat)
    {
        auto plus = MakeDisplaced(*molecule, flat, kFiniteDifferenceStep);
        auto minus = MakeDisplaced(*molecule, flat, -kFiniteDifferenceStep);

        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        auto plusEnergy =
            qcx::integrals::ComputeRiCoulombDerivative(*plus, *basis, *aux, *tensor, options);
        auto minusEnergy =
            qcx::integrals::ComputeRiCoulombDerivative(*minus, *basis, *aux, *tensor, options);

        ASSERT_TRUE(plusEnergy.has_value()) << plusEnergy.error().message;
        ASSERT_TRUE(minusEnergy.has_value()) << minusEnergy.error().message;

        const double numeric =
            (plusEnergy->energy - minusEnergy->energy) / (2.0 * kFiniteDifferenceStep);
        worst =
            std::max(worst, std::abs(numeric - fitted->values(static_cast<Eigen::Index>(flat))));
    }

    RecordProperty("worst_deviation", ShortDouble(worst));
    EXPECT_LT(worst, kFiniteDifferenceTolerance);
}

/// The acceptance test: the fitted gradient approaches the exact four-index
/// gradient as the auxiliary basis is improved, and the residual tracks the
/// auxiliary fitting error rather than sitting at a floor of its own.
///
/// The ladder's two numbers move together: the energy residual is the fitting
/// error measured where no derivative is involved, and the gradient residual
/// is the quantity this path exists to produce. A residual that stayed put
/// while the fit improved would be a contraction error, not a fitting error.
TEST(RiCoulombGradientTest, TheFittedGradientTracksTheAuxiliaryFittingError) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "the auxiliary ladder runs in Release configurations only";
    }

    // The ladder's top rung is the vendored Coulomb-fitting set, whose oxygen
    // carries f and g shells: a capped build refuses that basis at the engine's
    // entry point rather than fitting anything with it, so the residuals the
    // two rungs above are compared across never form.
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "the ladder's f and g aux shells exceed this build's kMaxEngineL "
                        "(a reduced-size build; CI builds 6)";
    }

    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto reference = ExactCoulombGradient(fixture->molecule, fixture->basisSet, fixture->density);
    auto referenceEnergy =
        ExactCoulombEnergy(fixture->molecule, fixture->basisSet, fixture->density);

    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    ASSERT_TRUE(referenceEnergy.has_value()) << referenceEnergy.error().message;

    const double gradientScale = reference->cwiseAbs().maxCoeff();
    auto orbitalPairs = qcx::integrals::BuildShellPairs(fixture->molecule, fixture->basisSet);

    ASSERT_TRUE(orbitalPairs.has_value()) << orbitalPairs.error().message;

    std::vector<double> gradientResiduals;
    std::vector<double> energyResiduals;

    for (const AuxRung& rung : kAuxLadder)
    {
        auto aux = AuxOfRung(rung);

        ASSERT_TRUE(aux.has_value()) << rung.name << ": " << aux.error().message;

        auto fitted = qcx::integrals::ComputeRiCoulombGradient(
            fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor);

        ASSERT_TRUE(fitted.has_value()) << rung.name << ": " << fitted.error().message;

        const double gradientResidual = (fitted->gradient - *reference).cwiseAbs().maxCoeff();
        const double energyResidual = std::abs(fitted->energy - *referenceEnergy);
        gradientResiduals.push_back(gradientResidual);
        energyResiduals.push_back(energyResidual);
        RecordProperty(std::string("gradient_residual_") + rung.name,
                       ShortDouble(gradientResidual));
        RecordProperty(std::string("energy_residual_") + rung.name, ShortDouble(energyResidual));
        RecordProperty(std::string("aux_functions_") + rung.name,
                       std::to_string(FunctionCount(fixture->molecule, *aux)));
        RecordProperty(std::string("gradient_relative_") + rung.name,
                       ShortDouble(gradientResidual / gradientScale));
        // The tensor term walks one canonical pair per auxiliary shell; the
        // screen drops the tasks whose bound contribution is under the
        // threshold at both orders.
        const std::size_t candidates = fitted->counts.orbitalPairs * fitted->counts.auxShells;
        RecordProperty(std::string("tensor_candidates_") + rung.name, std::to_string(candidates));
        RecordProperty(std::string("tensor_differentiated_") + rung.name,
                       std::to_string(fitted->counts.tensorQuartets));
        RecordProperty(std::string("tensor_discarded_") + rung.name,
                       std::to_string(candidates - fitted->counts.tensorQuartets));
        RecordProperty(std::string("metric_differentiated_") + rung.name,
                       std::to_string(fitted->counts.metricQuartets));
    }

    RecordProperty("gradient_scale", ShortDouble(gradientScale));
    RecordProperty("reference_energy", ShortDouble(*referenceEnergy));
    RecordProperty("orbital_pairs", std::to_string(orbitalPairs->pairs.size()));

    for (std::size_t rung = 1; rung < kAuxLadder.size(); ++rung)
    {
        EXPECT_LT(gradientResiduals[rung], gradientResiduals[rung - 1])
            << kAuxLadder[rung].name << " did not improve on " << kAuxLadder[rung - 1].name;
        EXPECT_LT(energyResiduals[rung], energyResiduals[rung - 1])
            << kAuxLadder[rung].name << " did not improve on " << kAuxLadder[rung - 1].name;
    }
}

/// Every refusal the entry documents: the code AND the bound it names. The
/// jfit refusal is the capped build's own - the full class matrix fits that
/// basis, so the gate skips there.
TEST(RiCoulombGradientTest, TheDerivativeEntryNamesItsRefusals) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto aux = AuxOfRung(kAuxLadder[0]);

    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    const std::size_t count = 3 * fixture->molecule.AtomCount();

    {
        qcx::integrals::RiCoulombDerivativeOptions options;
        options.order = 0;
        auto refused = qcx::integrals::ComputeRiCoulombDerivative(
            fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor, options);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(refused.error().message.find("outside 1..2"), std::string::npos)
            << refused.error().message;
    }

    {
        qcx::integrals::RiCoulombDerivativeOptions options;
        options.order = 3;
        auto refused = qcx::integrals::ComputeRiCoulombDerivative(
            fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor, options);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(refused.error().message.find("outside 1..2"), std::string::npos)
            << refused.error().message;
    }

    {
        const std::array<std::size_t, 1> coordinates = {0};
        qcx::integrals::RiCoulombDerivativeOptions options;
        options.order = 2;
        auto refused = qcx::integrals::ComputeRiCoulombDerivative(fixture->molecule,
                                                                  fixture->basisSet,
                                                                  *aux,
                                                                  fixture->densityTensor,
                                                                  coordinates,
                                                                  options);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(refused.error().message.find("do not divide into tuples"), std::string::npos)
            << refused.error().message;
    }

    {
        const std::array<std::size_t, 1> coordinates = {count};
        auto refused = qcx::integrals::ComputeRiCoulombDerivative(
            fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor, coordinates);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(refused.error().message.find("outside the"), std::string::npos)
            << refused.error().message;
    }

    {
        qcx::integrals::RiCoulombDerivativeOptions options;
        options.maxBatchBytes = 0;
        auto refused = qcx::integrals::ComputeRiCoulombDerivative(
            fixture->molecule, fixture->basisSet, *aux, fixture->densityTensor, options);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(refused.error().message.find("batch byte cap"), std::string::npos)
            << refused.error().message;
    }

    {
        auto wrongShape = qcx::testing::ToTensor(Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(count), static_cast<Eigen::Index>(count)));
        ASSERT_TRUE(wrongShape.has_value()) << wrongShape.error().message;
        auto refused = qcx::integrals::ComputeRiCoulombDerivative(
            fixture->molecule, fixture->basisSet, *aux, *wrongShape);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(refused.error().message.find("density matrix"), std::string::npos)
            << refused.error().message;
    }

    if (!qcx::integrals::SupportsL(4))
    {
        auto jfit = AuxOfRung(kAuxLadder[2]);

        ASSERT_TRUE(jfit.has_value()) << jfit.error().message;
        auto refused = qcx::integrals::ComputeRiCoulombDerivative(
            fixture->molecule, fixture->basisSet, *jfit, fixture->densityTensor);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kUnimplemented);
        EXPECT_NE(refused.error().message.find("kMaxEngineL"), std::string::npos)
            << refused.error().message;
    }
}
