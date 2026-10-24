// The displacement-sweep state's acceptance: the warm-start evidence the state
// exists for, the seeded-SCF leg, and the refusals that make a stale or
// malformed reference unable to reach a solve.
//
// The warm-start fixture is water/STO-3G - seven orbitals, five occupied, two
// virtual, so each displacement is a ten-dimensional response - solved at R0
// and then at R0 + 0.05 Bohr on one atom. The second sweep must report every
// one of the 3N solves warm-started, cost strictly fewer iterations than the
// first, and reproduce a cold state's answers.
//
// The derivative right-hand side is NOT this module's: the tree has no public
// derivative-integral API, so it is built here by central differences of the
// engine's own value builders, exactly as response_gradient_test.cpp does. The
// invariants below therefore check the state - its ownership, its per-ordinal
// solver set and its reuse - not the differentiation.

#include "h2o_sto3g.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/response/response_state.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using qcx::response::FockDerivativeContributionFn;
using qcx::response::FockDerivativeRequest;
using qcx::response::NuclearDisplacement;
using qcx::response::NuclearDisplacementRightHandSide;
using qcx::response::ResponseLayout;
using qcx::response::ResponseScreening;
using qcx::response::ResponseState;
using qcx::response::ResponseStateReference;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;

constexpr std::size_t kNumAtoms = 3;

// The experimental water geometry the fixture carries (Bohr), written
// atom-major: O at the origin, two H in the xy plane.
constexpr std::array<double, 3 * kNumAtoms> kReferenceCoordinates{0.0,
                                                                  0.0,
                                                                  0.0,
                                                                  1.430428808474167,
                                                                  1.107157044080814,
                                                                  0.0,
                                                                  -1.430428808474167,
                                                                  1.107157044080814,
                                                                  0.0};

/// A geometry, as the flat atom-major array a displacement moves in.
struct Geometry {
    std::array<double, 3 * kNumAtoms> coordinates = kReferenceCoordinates;

    /// \param atom The atom index to displace.
    /// \param direction The Cartesian axis to displace along.
    /// \param delta The displacement, in Bohr.
    /// \returns The displaced geometry.
    Geometry Displaced(std::size_t atom, std::size_t direction, double delta) const {
        Geometry moved = *this;
        moved.coordinates[3 * atom + direction] += delta;
        return moved;
    }
};

/// Builds the fixture molecule at one geometry: H2O in STO-3G, singlet.
/// \param geometry The geometry.
/// \returns The molecule, or an Error.
qcx::Result<qcx::molecule::Molecule> MakeMolecule(const Geometry& geometry) {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({kNumAtoms, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            (*coordinates)(atom, direction) = geometry.coordinates[3 * atom + direction];
        }
    }

    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

/// Every AO quantity one displacement's derivative needs, at fixed density.
struct DerivativeData {
    Eigen::MatrixXd core; ///< d(T + V)/dR.
    Eigen::MatrixXd eri; ///< d(mu nu | l s)/dR as the SCF's own supermatrix.
};

/// Builds one displacement's derivative quantities by central differences of
/// the engine's value builders.
/// \param geometry The reference geometry.
/// \param basis The basis set, which does not move with the nuclei.
/// \param atom The atom index to displace.
/// \param direction The Cartesian axis to displace along.
/// \param step The central-difference step, in Bohr.
/// \returns The derivative data, or an Error.
qcx::Result<DerivativeData> BuildDerivativeData(const Geometry& geometry,
                                                const qcx::basisset::BasisSet& basis,
                                                std::size_t atom,
                                                std::size_t direction,
                                                double step) {
    const auto plusMolecule = MakeMolecule(geometry.Displaced(atom, direction, step));
    const auto minusMolecule = MakeMolecule(geometry.Displaced(atom, direction, -step));

    if (!plusMolecule.has_value() || !minusMolecule.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "a displaced fixture failed"});
    }

    const auto plusEri = qcx::integrals::BuildEriTensorGeneral(*plusMolecule, basis);
    const auto minusEri = qcx::integrals::BuildEriTensorGeneral(*minusMolecule, basis);

    if (!plusEri.has_value() || !minusEri.has_value())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError, "an ERI build failed"});
    }

    const auto plusKinetic = qcx::integrals::BuildKineticMatrix(*plusMolecule, basis);
    const auto minusKinetic = qcx::integrals::BuildKineticMatrix(*minusMolecule, basis);
    const auto plusNuclear = qcx::integrals::BuildNuclearAttractionMatrix(*plusMolecule, basis);
    const auto minusNuclear = qcx::integrals::BuildNuclearAttractionMatrix(*minusMolecule, basis);

    if (!plusKinetic.has_value() || !minusKinetic.has_value() || !plusNuclear.has_value() ||
        !minusNuclear.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "a one-electron build failed"});
    }

    const std::size_t numBasis = static_cast<std::size_t>(ToMatrix(*plusKinetic).rows());
    const double inverseStep = 1.0 / (2.0 * step);

    DerivativeData data;
    data.core = (ToMatrix(*plusKinetic) + ToMatrix(*plusNuclear) - ToMatrix(*minusKinetic) -
                 ToMatrix(*minusNuclear)) *
                inverseStep;
    const Eigen::Index eriSide = static_cast<Eigen::Index>(numBasis * numBasis);
    data.eri = Eigen::MatrixXd::Zero(eriSide, eriSide);

    for (std::size_t mu = 0; mu < numBasis; ++mu)
    {
        for (std::size_t nu = 0; nu < numBasis; ++nu)
        {
            for (std::size_t lambda = 0; lambda < numBasis; ++lambda)
            {
                for (std::size_t sigma = 0; sigma < numBasis; ++sigma)
                {
                    data.eri(static_cast<int>(mu * numBasis + nu),
                             static_cast<int>(lambda * numBasis + sigma)) =
                        ((*plusEri)(mu, nu, lambda, sigma) - (*minusEri)(mu, nu, lambda, sigma)) *
                        inverseStep;
                }
            }
        }
    }

    return data;
}

/// The reference wavefunction of one geometry: the SCF result and the
/// quantities the state's input and the right-hand side read.
struct ReferenceState {
    Geometry geometry;
    Eigen::MatrixXd density; ///< The spin-summed density, 2 C_occ C_occ^T.
    Eigen::MatrixXd fock; ///< The AO Fock at that density.
    Eigen::MatrixXd coefficients; ///< The converged MO coefficients.
    std::vector<double> densityRowMajor;
    std::vector<double> fockRowMajor;
    std::vector<double> moCoefficients;
    std::vector<double> moEri;
    std::vector<double> orbitalEnergies;
    double totalEnergy = 0.0;
    int scfIterations = 0;
    double achievedEnergyDelta = 0.0;
    double achievedRmsDensityChange = 0.0;
    std::size_t numBasis = 0;
    ResponseLayout layout;
};

/// Builds the reference: converged RHF over the fixture at the given geometry,
/// with the MO two-electron tensor, the AO Fock and the row-major copies the
/// state's input takes.
/// \param geometry The geometry.
/// \param basis The basis set.
/// \returns The reference, or an Error.
qcx::Result<ReferenceState> BuildReference(const Geometry& geometry,
                                           const qcx::basisset::BasisSet& basis) {
    ReferenceState state;
    state.geometry = geometry;

    const auto molecule = MakeMolecule(geometry);

    if (!molecule.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "the fixture did not build"});
    }

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, basis);
    const auto eri = qcx::integrals::BuildEriTensorGeneral(*molecule, basis);

    if (!overlap.has_value() || !kinetic.has_value() || !nuclear.has_value() || !eri.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "an integral build failed"});
    }

    const Eigen::MatrixXd overlapMatrix = ToMatrix(*overlap);
    const Eigen::MatrixXd coreHamiltonian = ToMatrix(*kinetic) + ToMatrix(*nuclear);
    state.numBasis = static_cast<std::size_t>(overlapMatrix.rows());

    const auto reference = qcx::scf::RunRhfScf(*molecule, overlapMatrix, coreHamiltonian, *eri);

    if (!reference.has_value())
    {
        return std::unexpected(reference.error());
    }

    if (!reference->converged)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "the reference did not converge"});
    }

    state.totalEnergy = reference->totalEnergy;
    state.scfIterations = reference->iterations;
    state.achievedEnergyDelta = reference->achievedEnergyDelta;
    state.achievedRmsDensityChange = reference->achievedRmsDensityChange;
    state.coefficients = reference->coefficients;
    state.density = reference->density;
    state.orbitalEnergies.assign(reference->orbitalEnergies.begin(),
                                 reference->orbitalEnergies.end());

    // Water/STO-3G carries ten electrons: five occupied spatial orbitals and
    // two virtual ones.
    constexpr std::size_t kNumOccupied = 5;
    state.layout = ResponseLayout{kNumOccupied, state.numBasis - kNumOccupied};

    state.moCoefficients.assign(static_cast<std::size_t>(state.coefficients.size()), 0.0);
    state.densityRowMajor.assign(static_cast<std::size_t>(state.density.size()), 0.0);

    for (std::size_t row = 0; row < state.numBasis; ++row)
    {
        for (std::size_t column = 0; column < state.numBasis; ++column)
        {
            state.moCoefficients[row * state.numBasis + column] =
                state.coefficients(static_cast<int>(row), static_cast<int>(column));
            state.densityRowMajor[row * state.numBasis + column] =
                state.density(static_cast<int>(row), static_cast<int>(column));
        }
    }

    // The Fock at the reference density: F = H + J[D] - K[D]/2, the same
    // supermatrix contraction the SCF loop runs.
    Eigen::MatrixXd coulomb(state.numBasis * state.numBasis, state.numBasis * state.numBasis);
    Eigen::MatrixXd exchange(state.numBasis * state.numBasis, state.numBasis * state.numBasis);

    for (std::size_t mu = 0; mu < state.numBasis; ++mu)
    {
        for (std::size_t nu = 0; nu < state.numBasis; ++nu)
        {
            for (std::size_t lambda = 0; lambda < state.numBasis; ++lambda)
            {
                for (std::size_t sigma = 0; sigma < state.numBasis; ++sigma)
                {
                    coulomb(static_cast<int>(mu * state.numBasis + nu),
                            static_cast<int>(lambda * state.numBasis + sigma)) =
                        (*eri)(mu, nu, lambda, sigma);
                    exchange(static_cast<int>(mu * state.numBasis + nu),
                             static_cast<int>(lambda * state.numBasis + sigma)) =
                        (*eri)(mu, lambda, sigma, nu);
                }
            }
        }
    }

    const Eigen::VectorXd flatDensity =
        Eigen::Map<const Eigen::VectorXd>(state.densityRowMajor.data(), state.density.size());
    const Eigen::VectorXd j = coulomb * flatDensity;
    const Eigen::VectorXd k = exchange * flatDensity;
    state.fock = coreHamiltonian;

    for (std::size_t mu = 0; mu < state.numBasis; ++mu)
    {
        for (std::size_t nu = 0; nu < state.numBasis; ++nu)
        {
            const int index = static_cast<int>(mu * state.numBasis + nu);
            state.fock(static_cast<int>(mu), static_cast<int>(nu)) += j(index) - 0.5 * k(index);
        }
    }

    state.fockRowMajor.assign(static_cast<std::size_t>(state.fock.size()), 0.0);

    for (std::size_t i = 0; i < state.numBasis; ++i)
    {
        for (std::size_t jIndex = 0; jIndex < state.numBasis; ++jIndex)
        {
            state.fockRowMajor[i * state.numBasis + jIndex] =
                state.fock(static_cast<int>(i), static_cast<int>(jIndex));
        }
    }

    // The MO two-electron tensor, chemists' notation (pq|rs) and full n^4.
    state.moEri.assign(state.numBasis * state.numBasis * state.numBasis * state.numBasis, 0.0);

    for (std::size_t p = 0; p < state.numBasis; ++p)
    {
        for (std::size_t q = 0; q < state.numBasis; ++q)
        {
            for (std::size_t r = 0; r < state.numBasis; ++r)
            {
                for (std::size_t s = 0; s < state.numBasis; ++s)
                {
                    double value = 0.0;

                    for (std::size_t mu = 0; mu < state.numBasis; ++mu)
                    {
                        for (std::size_t nu = 0; nu < state.numBasis; ++nu)
                        {
                            for (std::size_t lambda = 0; lambda < state.numBasis; ++lambda)
                            {
                                for (std::size_t sigma = 0; sigma < state.numBasis; ++sigma)
                                {
                                    value += state.coefficients(static_cast<int>(mu),
                                                                static_cast<int>(p)) *
                                             state.coefficients(static_cast<int>(nu),
                                                                static_cast<int>(q)) *
                                             state.coefficients(static_cast<int>(lambda),
                                                                static_cast<int>(r)) *
                                             state.coefficients(static_cast<int>(sigma),
                                                                static_cast<int>(s)) *
                                             (*eri)(mu, nu, lambda, sigma);
                                }
                            }
                        }
                    }

                    state.moEri[((p * state.numBasis + q) * state.numBasis + r) * state.numBasis +
                                s] = value;
                }
            }
        }
    }

    return state;
}

/// The MO-basis overlap derivative C^T S^R C for one displacement, by central
/// differences of the engine's own overlap builder.
/// \param state The reference.
/// \param basis The basis set.
/// \param atom The atom index.
/// \param direction The Cartesian axis.
/// \param step The central-difference step, in Bohr.
/// \returns C^T S^R C, or an Error.
qcx::Result<Eigen::MatrixXd> BuildMoOverlapDerivative(const ReferenceState& state,
                                                      const qcx::basisset::BasisSet& basis,
                                                      std::size_t atom,
                                                      std::size_t direction,
                                                      double step) {
    const auto plusMolecule = MakeMolecule(state.geometry.Displaced(atom, direction, step));
    const auto minusMolecule = MakeMolecule(state.geometry.Displaced(atom, direction, -step));

    if (!plusMolecule.has_value() || !minusMolecule.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "a displaced fixture failed"});
    }

    const auto plusOverlap = qcx::integrals::BuildOverlapMatrix(*plusMolecule, basis);
    const auto minusOverlap = qcx::integrals::BuildOverlapMatrix(*minusMolecule, basis);

    if (!plusOverlap.has_value() || !minusOverlap.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "an overlap build failed"});
    }

    const Eigen::MatrixXd overlapDerivative =
        (ToMatrix(*plusOverlap) - ToMatrix(*minusOverlap)) / (2.0 * step);
    return state.coefficients.transpose() * overlapDerivative * state.coefficients;
}

/// The one contribution the state test wires in: the whole occupied-virtual
/// block of the nuclear-derivative Fock matrix, one-electron and two-electron
/// terms together. The multi-contribution seam is response_gradient_test's.
/// \param state The reference.
/// \param basis The basis set.
/// \param step The central-difference step, in Bohr.
/// \returns The right-hand side, ready to assemble, or an Error.
qcx::Result<NuclearDisplacementRightHandSide> BuildRightHandSide(
    const ReferenceState& state, const qcx::basisset::BasisSet& basis, double step) {
    auto rightHandSide = NuclearDisplacementRightHandSide::Create(state.layout, state.numBasis);

    if (!rightHandSide.has_value())
    {
        return std::unexpected(rightHandSide.error());
    }

    FockDerivativeContributionFn contribution =
        [&state, &basis, step](const FockDerivativeRequest& request,
                               std::span<double> block) -> qcx::Result<void> {
        auto data = BuildDerivativeData(
            state.geometry, basis, request.displacement.atom, request.displacement.direction, step);

        if (!data.has_value())
        {
            return std::unexpected(data.error());
        }

        Eigen::MatrixXd exchangeDerivative(data->eri.rows(), data->eri.cols());

        for (std::size_t mu = 0; mu < state.numBasis; ++mu)
        {
            for (std::size_t nu = 0; nu < state.numBasis; ++nu)
            {
                for (std::size_t lambda = 0; lambda < state.numBasis; ++lambda)
                {
                    for (std::size_t sigma = 0; sigma < state.numBasis; ++sigma)
                    {
                        exchangeDerivative(static_cast<int>(mu * state.numBasis + nu),
                                           static_cast<int>(lambda * state.numBasis + sigma)) =
                            data->eri(static_cast<int>(mu * state.numBasis + lambda),
                                      static_cast<int>(sigma * state.numBasis + nu));
                    }
                }
            }
        }

        const Eigen::VectorXd flatDensity =
            Eigen::Map<const Eigen::VectorXd>(state.densityRowMajor.data(), state.density.size());
        const Eigen::VectorXd j = data->eri * flatDensity;
        const Eigen::VectorXd k = exchangeDerivative * flatDensity;
        Eigen::MatrixXd fockDerivative(state.numBasis, state.numBasis);

        for (std::size_t mu = 0; mu < state.numBasis; ++mu)
        {
            for (std::size_t nu = 0; nu < state.numBasis; ++nu)
            {
                const int index = static_cast<int>(mu * state.numBasis + nu);
                fockDerivative(static_cast<int>(mu), static_cast<int>(nu)) =
                    data->core(static_cast<int>(mu), static_cast<int>(nu)) + j(index) -
                    0.5 * k(index);
            }
        }

        const Eigen::MatrixXd moDerivative =
            state.coefficients.transpose() * fockDerivative * state.coefficients;

        for (std::size_t i = 0; i < request.layout.numOccupied; ++i)
        {
            for (std::size_t a = 0; a < request.layout.numVirtual; ++a)
            {
                block[i * request.layout.numVirtual + a] += moDerivative(
                    static_cast<int>(request.layout.numOccupied + a), static_cast<int>(i));
            }
        }

        return {};
    };

    auto added = rightHandSide->Add("nuclear-derivative-fock", std::move(contribution));

    if (!added.has_value())
    {
        return std::unexpected(added.error());
    }

    return rightHandSide;
}

/// The state's input for one reference: every array is a span into the
/// reference's own storage, which outlives the Create or Rebase call.
/// \param state The reference.
/// \returns The input.
ResponseStateReference MakeStateReference(const ReferenceState& state) {
    ResponseStateReference input;
    input.converged = true;
    input.numBasisFunctions = state.numBasis;
    input.numOccupied = state.layout.numOccupied;
    input.numVirtual = state.layout.numVirtual;
    input.coordinatesBohr = state.geometry.coordinates;
    input.moCoefficients = state.moCoefficients;
    input.orbitalEnergies = state.orbitalEnergies;
    input.density = state.densityRowMajor;
    input.fock = state.fockRowMajor;
    input.moTwoElectron = state.moEri;
    input.history.scfIterations = static_cast<std::size_t>(state.scfIterations);
    input.history.achievedEnergyDelta = state.achievedEnergyDelta;
    input.history.achievedRmsDensityChange = state.achievedRmsDensityChange;
    return input;
}

/// The screening structure for one reference: the digest of its geometry and
/// one entry per AO pair. The bounds' values are the builder's contract; the
/// state carries and guards them but does not interpret them.
/// \param state The reference.
/// \returns The screening structure, or an Error.
qcx::Result<ResponseScreening> MakeScreening(const ReferenceState& state) {
    auto identity = qcx::response::MakeResponseReferenceIdentity(state.geometry.coordinates,
                                                                 state.densityRowMajor);

    if (!identity.has_value())
    {
        return std::unexpected(identity.error());
    }

    ResponseScreening screening;
    screening.geometry = identity->geometry;
    const std::size_t pairCount = state.numBasis * (state.numBasis + 1) / 2;
    screening.pairValues.assign(pairCount, 1.0);
    screening.pairDerivativeBounds.assign(pairCount, 0.5);
    return screening;
}

/// The largest absolute difference between two equal-length arrays.
/// \param left The first array.
/// \param right The second array.
/// \returns The largest |left[k] - right[k]|, or 0 for empty arrays.
double MaxDifference(std::span<const double> left, std::span<const double> right) {
    double worst = 0.0;

    for (std::size_t k = 0; k < left.size(); ++k)
    {
        worst = std::max(worst, std::abs(left[k] - right[k]));
    }

    return worst;
}

// The central-difference step the derivative inputs use, in Bohr. It is the
// response_gradient_test value: far below the geometry change between the two
// references, and far above the integral builders' own rounding.
constexpr double kDerivativeStep = 1e-4;

// --- the warm-start evidence ------------------------------------------------

TEST(ResponseStateTest, TheSecondSweepStartsFromTheStatesPreviousAnswers) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const Geometry firstGeometry;
    const Geometry secondGeometry = firstGeometry.Displaced(0, 0, 0.05);

    const auto firstReference = BuildReference(firstGeometry, *basis);
    ASSERT_TRUE(firstReference.has_value()) << firstReference.error().message;
    const auto secondReference = BuildReference(secondGeometry, *basis);
    ASSERT_TRUE(secondReference.has_value()) << secondReference.error().message;

    const std::size_t displacements = 3 * kNumAtoms;

    auto firstScreening = MakeScreening(*firstReference);
    ASSERT_TRUE(firstScreening.has_value()) << firstScreening.error().message;
    auto state = ResponseState::Create(MakeStateReference(*firstReference), *firstScreening);
    ASSERT_TRUE(state.has_value()) << state.error().message;

    // First sweep, at the geometry the state was built for: every solve is
    // cold, because no answer has been stored yet.
    std::size_t firstIterations = 0;
    std::size_t firstWarm = 0;

    {
        auto rightHandSide = BuildRightHandSide(*firstReference, *basis, kDerivativeStep);
        ASSERT_TRUE(rightHandSide.has_value()) << rightHandSide.error().message;

        for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
        {
            for (std::size_t direction = 0; direction < 3; ++direction)
            {
                const auto overlap = BuildMoOverlapDerivative(
                    *firstReference, *basis, atom, direction, kDerivativeStep);
                ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

                auto solution = state->SolveDisplacement(
                    NuclearDisplacement{atom, direction},
                    *rightHandSide,
                    std::span<const double>(overlap->data(), overlap->size()));
                ASSERT_TRUE(solution.has_value()) << solution.error().message;

                EXPECT_FALSE(solution->warmStarted);
                firstIterations += solution->iterations;
                firstWarm += solution->warmStarted ? 1 : 0;
            }
        }
    }

    EXPECT_EQ(firstWarm, 0u);
    EXPECT_EQ(state->Report().solves, displacements);
    EXPECT_EQ(state->Report().warmStarted, 0u);

    // Rebase to the moved geometry. The solvers keep their stored answers.
    auto secondScreening = MakeScreening(*secondReference);
    ASSERT_TRUE(secondScreening.has_value()) << secondScreening.error().message;
    auto rebased = state->Rebase(MakeStateReference(*secondReference), *secondScreening);
    ASSERT_TRUE(rebased.has_value()) << rebased.error().message;

    // The rebase must have taken: the state now carries the new reference.
    EXPECT_TRUE(state->Identity() ==
                *qcx::response::MakeResponseReferenceIdentity(secondReference->geometry.coordinates,
                                                              secondReference->densityRowMajor));
    EXPECT_EQ(state->Screening().geometry, secondScreening->geometry);

    // Second sweep, at the moved geometry and with every right-hand side
    // rebuilt for it: every solve must report that it started from the stored
    // answer for its own ordinal.
    std::size_t secondIterations = 0;
    std::size_t secondWarm = 0;
    std::vector<std::vector<double>> warmDensities(displacements);
    std::vector<std::vector<double>> warmAmplitudes(displacements);

    {
        auto rightHandSide = BuildRightHandSide(*secondReference, *basis, kDerivativeStep);
        ASSERT_TRUE(rightHandSide.has_value()) << rightHandSide.error().message;

        for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
        {
            for (std::size_t direction = 0; direction < 3; ++direction)
            {
                const auto overlap = BuildMoOverlapDerivative(
                    *secondReference, *basis, atom, direction, kDerivativeStep);
                ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

                auto solution = state->SolveDisplacement(
                    NuclearDisplacement{atom, direction},
                    *rightHandSide,
                    std::span<const double>(overlap->data(), overlap->size()));
                ASSERT_TRUE(solution.has_value()) << solution.error().message;

                EXPECT_TRUE(solution->warmStarted) << "atom " << atom << " direction " << direction;
                secondWarm += solution->warmStarted ? 1 : 0;
                secondIterations += solution->iterations;
                warmDensities[3 * atom + direction] = solution->response.perturbedDensity;
                warmAmplitudes[3 * atom + direction] = solution->response.amplitudes;
            }
        }
    }

    EXPECT_EQ(secondWarm, displacements);
    EXPECT_EQ(state->Report().warmStarted, displacements);
    EXPECT_EQ(state->Report().solves, 2 * displacements);
    EXPECT_EQ(state->Report().iterations, firstIterations + secondIterations);

    // The first and second sweeps solve different geometries, so their totals
    // are not comparable: the second geometry is a different problem and its
    // cold cost is measured below. The accepted bar is the paired one, warm
    // against cold at the same geometry and right-hand sides.
    std::cout << "[warm start] first sweep " << firstIterations << " iterations, second sweep "
              << secondIterations << " iterations over " << displacements << " displacements\n";

    // The agreement claim: a state built cold at the same geometry must reach
    // the same answers.
    auto coldState = ResponseState::Create(MakeStateReference(*secondReference), *secondScreening);
    ASSERT_TRUE(coldState.has_value()) << coldState.error().message;

    std::size_t coldIterations = 0;
    double worstDensity = 0.0;
    double worstAmplitude = 0.0;

    {
        auto rightHandSide = BuildRightHandSide(*secondReference, *basis, kDerivativeStep);
        ASSERT_TRUE(rightHandSide.has_value()) << rightHandSide.error().message;

        for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
        {
            for (std::size_t direction = 0; direction < 3; ++direction)
            {
                const auto overlap = BuildMoOverlapDerivative(
                    *secondReference, *basis, atom, direction, kDerivativeStep);
                ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

                auto solution = coldState->SolveDisplacement(
                    NuclearDisplacement{atom, direction},
                    *rightHandSide,
                    std::span<const double>(overlap->data(), overlap->size()));
                ASSERT_TRUE(solution.has_value()) << solution.error().message;

                EXPECT_FALSE(solution->warmStarted);
                coldIterations += solution->iterations;

                const std::size_t ordinal = 3 * atom + direction;
                worstDensity = std::max(
                    worstDensity,
                    MaxDifference(solution->response.perturbedDensity, warmDensities[ordinal]));
                worstAmplitude =
                    std::max(worstAmplitude,
                             MaxDifference(solution->response.amplitudes, warmAmplitudes[ordinal]));
            }
        }
    }

    std::cout << "[warm start] cold state at the second geometry " << coldIterations
              << " iterations; worst |density difference| " << worstDensity
              << ", worst |amplitude difference| " << worstAmplitude << "\n";

    // The accepted bar: at one geometry and one set of right-hand sides, the
    // warm sweep costs strictly fewer iterations than the cold one, and both
    // reach the same answer.
    EXPECT_LT(secondIterations, coldIterations);
    EXPECT_LT(worstDensity, 1e-5);
    EXPECT_LT(worstAmplitude, 1e-5);
}

// The plan's other half: the density the state owns is what a sweep seeds the
// next geometry's SCF with, and seeding with it converges to the same energy
// without costing more iterations than a cold start.
TEST(ResponseStateTest, TheStatesDensitySeedsTheNextGeometryScf) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const Geometry firstGeometry;
    const Geometry secondGeometry = firstGeometry.Displaced(0, 0, 0.05);

    const auto firstReference = BuildReference(firstGeometry, *basis);
    ASSERT_TRUE(firstReference.has_value()) << firstReference.error().message;
    const auto secondReference = BuildReference(secondGeometry, *basis);
    ASSERT_TRUE(secondReference.has_value()) << secondReference.error().message;

    auto screening = MakeScreening(*firstReference);
    ASSERT_TRUE(screening.has_value()) << screening.error().message;
    auto state = ResponseState::Create(MakeStateReference(*firstReference), *screening);
    ASSERT_TRUE(state.has_value()) << state.error().message;

    // The state's density is the reference's, element for element.
    EXPECT_DOUBLE_EQ(MaxDifference(state->Density(), firstReference->densityRowMajor), 0.0);

    const auto molecule = MakeMolecule(secondGeometry);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value());
    ASSERT_TRUE(kinetic.has_value());
    ASSERT_TRUE(nuclear.has_value());
    ASSERT_TRUE(eri.has_value());

    const Eigen::MatrixXd overlapMatrix = ToMatrix(*overlap);
    const Eigen::MatrixXd coreHamiltonian = ToMatrix(*kinetic) + ToMatrix(*nuclear);

    qcx::scf::RhfOptions options;
    options.initialScfState.density =
        Eigen::Map<const Eigen::MatrixXd>(state->Density().data(),
                                          static_cast<Eigen::Index>(state->NumBasisFunctions()),
                                          static_cast<Eigen::Index>(state->NumBasisFunctions()));

    auto seeded = qcx::scf::RunRhfScf(*molecule, overlapMatrix, coreHamiltonian, *eri, options);
    ASSERT_TRUE(seeded.has_value()) << seeded.error().message;
    ASSERT_TRUE(seeded->converged);

    auto fresh = qcx::scf::RunRhfScf(*molecule, overlapMatrix, coreHamiltonian, *eri);
    ASSERT_TRUE(fresh.has_value()) << fresh.error().message;
    ASSERT_TRUE(fresh->converged);

    std::cout << "[seeded scf] seeded " << seeded->iterations << " iterations, cold "
              << fresh->iterations << " iterations; energy " << seeded->totalEnergy
              << " against reference " << secondReference->totalEnergy << "\n";

    EXPECT_NEAR(seeded->totalEnergy, secondReference->totalEnergy, 1e-10);
    EXPECT_NEAR(fresh->totalEnergy, secondReference->totalEnergy, 1e-10);
    EXPECT_LE(seeded->iterations, fresh->iterations);
}

// --- the refusals and the structure -----------------------------------------

namespace {

// A small synthetic reference that passes every check: three basis functions,
// one occupied and two virtual orbitals, identity coefficients, a diagonal
// Fock and a zero two-electron tensor. The operator is never applied in these
// tests - each stops at a refusal.
struct SyntheticReference {
    std::array<double, 3> coordinates{0.0, 0.0, 0.0};
    std::vector<double> coefficients{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<double> energies{-0.5, 0.25, 0.75};
    std::vector<double> density{2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> fock{-0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.75};
    std::vector<double> moTwoElectron = std::vector<double>(81, 0.0);

    ResponseStateReference Reference() const {
        ResponseStateReference input;
        input.converged = true;
        input.numBasisFunctions = 3;
        input.numOccupied = 1;
        input.numVirtual = 2;
        input.coordinatesBohr = coordinates;
        input.moCoefficients = coefficients;
        input.orbitalEnergies = energies;
        input.density = density;
        input.fock = fock;
        input.moTwoElectron = moTwoElectron;
        return input;
    }

    qcx::Result<ResponseScreening> Screening() const {
        auto identity = qcx::response::MakeResponseReferenceIdentity(coordinates, density);

        if (!identity.has_value())
        {
            return std::unexpected(identity.error());
        }

        ResponseScreening screening;
        screening.geometry = identity->geometry;
        screening.pairValues.assign(6, 1.0);
        screening.pairDerivativeBounds.assign(6, 0.5);
        return screening;
    }
};

} // namespace

TEST(ResponseStateTest, ANonConvergedReferenceIsRefusedByName) {
    SyntheticReference synthetic;
    auto input = synthetic.Reference();
    input.converged = false;
    auto screening = synthetic.Screening();
    ASSERT_TRUE(screening.has_value());

    auto state = ResponseState::Create(input, *screening);
    ASSERT_FALSE(state.has_value());
    EXPECT_EQ(state.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(state.error().message.find("converged = false"), std::string::npos);
}

TEST(ResponseStateTest, ADensityThatIsNotTheOccupiedProjectorIsRefused) {
    SyntheticReference synthetic;
    synthetic.density[4] = 1.0;
    const auto input = synthetic.Reference();
    auto screening = synthetic.Screening();
    ASSERT_TRUE(screening.has_value());

    auto state = ResponseState::Create(input, *screening);
    ASSERT_FALSE(state.has_value());
    EXPECT_EQ(state.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(state.error().message.find("2 C_occ C_occ^T"), std::string::npos);
}

TEST(ResponseStateTest, AnAsymmetricFockIsRefused) {
    SyntheticReference synthetic;
    synthetic.fock[1] = 0.25;
    const auto input = synthetic.Reference();
    auto screening = synthetic.Screening();
    ASSERT_TRUE(screening.has_value());

    auto state = ResponseState::Create(input, *screening);
    ASSERT_FALSE(state.has_value());
    EXPECT_EQ(state.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(state.error().message.find("not symmetric"), std::string::npos);
}

TEST(ResponseStateTest, TheScreeningStructureMustCarryOneValuePerPair) {
    SyntheticReference synthetic;
    auto screening = synthetic.Screening();
    ASSERT_TRUE(screening.has_value());
    screening->pairValues.resize(5);

    auto state = ResponseState::Create(synthetic.Reference(), *screening);
    ASSERT_FALSE(state.has_value());
    EXPECT_EQ(state.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(state.error().message.find("same pair set"), std::string::npos);

    auto truncated = synthetic.Screening();
    ASSERT_TRUE(truncated.has_value());
    truncated->pairValues.resize(5);
    truncated->pairDerivativeBounds.resize(5);

    auto pairRefusal = ResponseState::Create(synthetic.Reference(), *truncated);
    ASSERT_FALSE(pairRefusal.has_value());
    EXPECT_EQ(pairRefusal.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(pairRefusal.error().message.find("one value per AO pair"), std::string::npos);
}

TEST(ResponseStateTest, AStaleScreeningDigestIsRefused) {
    SyntheticReference synthetic;
    auto screening = synthetic.Screening();
    ASSERT_TRUE(screening.has_value());
    screening->geometry += 1;

    auto state = ResponseState::Create(synthetic.Reference(), *screening);
    ASSERT_FALSE(state.has_value());
    EXPECT_EQ(state.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(state.error().message.find("belong to another geometry"), std::string::npos);

    // The same refusal on rebase, and a refused rebase leaves the state as it
    // was.
    auto good = synthetic.Screening();
    ASSERT_TRUE(good.has_value());
    auto created = ResponseState::Create(synthetic.Reference(), *good);
    ASSERT_TRUE(created.has_value()) << created.error().message;

    const auto before = created->Identity();
    const auto densityBefore = created->Density();

    SyntheticReference moved;
    moved.coordinates = {0.05, 0.0, 0.0};
    auto movedScreening = moved.Screening();
    ASSERT_TRUE(movedScreening.has_value());
    movedScreening->geometry += 1;

    auto rebased = created->Rebase(moved.Reference(), *movedScreening);
    ASSERT_FALSE(rebased.has_value());
    EXPECT_EQ(rebased.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_TRUE(created->Identity() == before);
    EXPECT_EQ(MaxDifference(created->Density(), densityBefore), 0.0);
}

TEST(ResponseStateTest, ARebaseCannotChangeTheLayout) {
    SyntheticReference synthetic;
    auto screening = synthetic.Screening();
    ASSERT_TRUE(screening.has_value());
    auto created = ResponseState::Create(synthetic.Reference(), *screening);
    ASSERT_TRUE(created.has_value()) << created.error().message;

    // A consistent reference with two occupied orbitals: legal on its own, so
    // only the layout comparison can refuse it.
    SyntheticReference wider;
    wider.density[4] = 2.0;
    wider.energies = {-0.5, -0.4, 0.75};
    wider.fock = {-0.5, 0.0, 0.0, 0.0, -0.4, 0.0, 0.0, 0.0, 0.75};
    auto input = wider.Reference();
    input.numOccupied = 2;
    input.numVirtual = 1;
    auto widerScreening = wider.Screening();
    ASSERT_TRUE(widerScreening.has_value());

    auto rebased = created->Rebase(input, *widerScreening);
    ASSERT_FALSE(rebased.has_value());
    EXPECT_EQ(rebased.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(rebased.error().message.find("occupied/virtual layout"), std::string::npos);

    // A changed atom count is the same class of refusal: the per-coordinate
    // solver set is keyed on it. Two atoms in the same basis is otherwise a
    // legal reference, so only the atom-count comparison can refuse it.
    const std::array<double, 6> twoAtomCoordinates{0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    ResponseStateReference atomInput = synthetic.Reference();
    atomInput.coordinatesBohr = twoAtomCoordinates;
    auto atomScreening = synthetic.Screening();
    ASSERT_TRUE(atomScreening.has_value());
    auto atomIdentity =
        qcx::response::MakeResponseReferenceIdentity(twoAtomCoordinates, synthetic.density);
    ASSERT_TRUE(atomIdentity.has_value());
    atomScreening->geometry = atomIdentity->geometry;

    auto atomRebase = created->Rebase(atomInput, *atomScreening);
    ASSERT_FALSE(atomRebase.has_value());
    EXPECT_EQ(atomRebase.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(atomRebase.error().message.find("atom count"), std::string::npos);
}

TEST(ResponseStateTest, ADisplacementOutsideTheMoleculeIsRefused) {
    SyntheticReference synthetic;
    auto screening = synthetic.Screening();
    ASSERT_TRUE(screening.has_value());
    auto created = ResponseState::Create(synthetic.Reference(), *screening);
    ASSERT_TRUE(created.has_value()) << created.error().message;

    auto rightHandSide =
        NuclearDisplacementRightHandSide::Create(created->Layout(), created->NumBasisFunctions());
    ASSERT_TRUE(rightHandSide.has_value());
    std::vector<double> overlapDerivative(9, 0.0);

    auto outside =
        created->SolveDisplacement(NuclearDisplacement{1, 0}, *rightHandSide, overlapDerivative);
    ASSERT_FALSE(outside.has_value());
    EXPECT_EQ(outside.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(outside.error().message.find("outside the 1 atoms"), std::string::npos);

    auto axis =
        created->SolveDisplacement(NuclearDisplacement{0, 3}, *rightHandSide, overlapDerivative);
    ASSERT_FALSE(axis.has_value());
    EXPECT_EQ(axis.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(axis.error().message.find("three Cartesian axes"), std::string::npos);
}

TEST(ResponseStateTest, ARightHandSideOfAnotherLayoutIsRefused) {
    SyntheticReference synthetic;
    auto screening = synthetic.Screening();
    ASSERT_TRUE(screening.has_value());
    auto created = ResponseState::Create(synthetic.Reference(), *screening);
    ASSERT_TRUE(created.has_value()) << created.error().message;

    auto otherLayout = NuclearDisplacementRightHandSide::Create(ResponseLayout{2, 1}, 3);
    ASSERT_TRUE(otherLayout.has_value());
    std::vector<double> overlapDerivative(9, 0.0);

    auto solved =
        created->SolveDisplacement(NuclearDisplacement{0, 0}, *otherLayout, overlapDerivative);
    ASSERT_FALSE(solved.has_value());
    EXPECT_EQ(solved.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(solved.error().message.find("right-hand side's occupied/virtual layout"),
              std::string::npos);
}

} // namespace
