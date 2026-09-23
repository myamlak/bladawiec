#include "h2_sto3g.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/mass_properties.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/response/response_gradient.hpp"
#include "qcx/response/response_operator.hpp"
#include "qcx/response/response_solver.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <span>
#include <string>
#include <vector>

// The nuclear-displacement right-hand side, checked on a real H3+/STO-3G
// reference at a scalene, off-axis geometry so that neither the derivative Fock
// block nor any gradient component is zero by symmetry.
//
// The derivative integrals are NOT this module's - they belong to the integral
// engine, and this tree has no public API for them. They are therefore built
// here by central differences of the engine's own value builders, which are
// public. That makes the invariants below checks of THIS block - the placement
// of each term, the solve, and the contraction - and not of the
// differentiation, which is checked where it lives. The last check in the file
// is a finite difference of the energy, and it is last because it is the
// weakest of the five.

namespace {

using qcx::response::EnergyWeightedTerm;
using qcx::response::FockDerivativeContributionFn;
using qcx::response::FockDerivativeRequest;
using qcx::response::HessianExtraTermFn;
using qcx::response::MoTwoElectronTensor;
using qcx::response::NuclearDisplacement;
using qcx::response::NuclearDisplacementRightHandSide;
using qcx::response::OrbitalEnergies;
using qcx::response::OrbitalGradient;
using qcx::response::OrbitalHessianOperator;
using qcx::response::OrbitalResponse;
using qcx::response::OrbitalResponseCache;
using qcx::response::ResponseCacheHandle;
using qcx::response::ResponseLayout;
using qcx::response::ResponseReferenceIdentity;
using qcx::response::ResponseSolveOptions;
using qcx::response::ResponseSolver;
using qcx::response::SolveNuclearDisplacementResponse;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

constexpr std::size_t kNumAtoms = 3;

/// The reference geometry, in Bohr: a scalene, off-axis H3+ triangle, written
/// atom-major.
///
/// The asymmetry is deliberate. For two atoms of one element in a basis
/// symmetric under their exchange, every derivative matrix has the block form
/// [[M, N], [N, M]], and the occupied-virtual block of C^T [[M, N], [N, M]] C is
/// identically zero for a g/u-split coefficient matrix: the derivative Fock
/// block, and with it the response, vanishes, and every check on the response
/// passes without testing anything.
constexpr std::array<double, 3 * kNumAtoms> kReferenceCoordinates{
    0.00, 0.00, 0.00, 1.50, 0.35, -0.20, -0.40, 1.10, 0.55};

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

    /// \param atom The atom index.
    /// \returns That atom's position, in Bohr.
    std::array<double, 3> Position(std::size_t atom) const {
        return {coordinates[3 * atom], coordinates[3 * atom + 1], coordinates[3 * atom + 2]};
    }
};

/// Builds the fixture molecule at one geometry: H3+ in the hydrogen STO-3G
/// basis, so two electrons over three orbitals - one occupied, two virtual.
/// \param geometry The geometry.
/// \returns The molecule, or an Error.
qcx::Result<qcx::molecule::Molecule> MakeMolecule(const Geometry& geometry) {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({kNumAtoms, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    std::vector<qcx::molecule::Atom> atoms(kNumAtoms, qcx::molecule::Atom{"H", 1, 0.0});

    for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            (*coordinates)(atom, direction) = geometry.coordinates[3 * atom + direction];
        }
    }

    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 1, 1);
}

/// The geometry's coordinates in Bohr, as the reference identity takes them.
/// \param geometry The geometry.
/// \returns The flat atom-major array, borrowed for the caller's lifetime.
std::span<const double> GeometryCoordinates(const Geometry& geometry) {
    return geometry.coordinates;
}

/// Every AO quantity one displacement's derivative needs, at fixed density.
struct DerivativeData {
    /// d(T + V)/dR, the core-Hamiltonian derivative.
    Eigen::MatrixXd core;
    /// d(mu nu | l s)/dR as the (mu n + nu, l n + s) supermatrix the SCF's own
    /// J/K contraction uses.
    Eigen::MatrixXd eri;
    /// dE_nuc/dR for this displacement, Hartree/Bohr.
    double nuclearRepulsion = 0.0;
};

/// The nuclear-repulsion energy of the fixture: every pair of unit charges.
/// \param geometry The geometry.
/// \returns The repulsion, in Hartree.
double NuclearRepulsionEnergy(const Geometry& geometry) {
    double energy = 0.0;

    for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
    {
        for (std::size_t other = atom + 1; other < kNumAtoms; ++other)
        {
            const std::array<double, 3> left = geometry.Position(atom);
            const std::array<double, 3> right = geometry.Position(other);
            const double dx = left[0] - right[0];
            const double dy = left[1] - right[1];
            const double dz = left[2] - right[2];

            energy += 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    return energy;
}

/// The derivative of the nuclear repulsion under one atom's displacement.
///
/// Every atom carries unit charge, so E_nuc is the sum of 1/r over the pairs,
/// and the derivative with respect to moving one atom runs over the pairs that
/// atom is in.
/// \param geometry The geometry.
/// \param atom The atom index to displace.
/// \param direction The Cartesian axis to displace along.
/// \returns dE_nuc/dR, in Hartree/Bohr.
double NuclearRepulsionDerivative(const Geometry& geometry,
                                  std::size_t atom,
                                  std::size_t direction) {
    const std::array<double, 3> moved = geometry.Position(atom);
    double derivative = 0.0;

    for (std::size_t other = 0; other < kNumAtoms; ++other)
    {
        if (other == atom)
        {
            continue;
        }

        const std::array<double, 3> fixed = geometry.Position(other);
        const std::array<double, 3> separation{
            moved[0] - fixed[0], moved[1] - fixed[1], moved[2] - fixed[2]};
        const double squared = separation[0] * separation[0] + separation[1] * separation[1] +
                               separation[2] * separation[2];

        derivative -= separation[direction] / (squared * std::sqrt(squared));
    }

    return derivative;
}

/// Builds one displacement's derivative quantities by central differences of
/// the engine's value builders.
///
/// SCOPE: this makes the differentiation a finite difference. It is the input
/// to the checks below, not one of them - what the checks then test is the
/// right-hand side assembled from it, which is this module's own code.
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
    const Geometry plus = geometry.Displaced(atom, direction, step);
    const Geometry minus = geometry.Displaced(atom, direction, -step);

    const auto plusMolecule = MakeMolecule(plus);
    const auto minusMolecule = MakeMolecule(minus);

    if (!plusMolecule.has_value() || !minusMolecule.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "the displaced fixtures did not build"});
    }

    const auto plusEri = qcx::integrals::BuildEriTensor(*plusMolecule, basis);
    const auto minusEri = qcx::integrals::BuildEriTensor(*minusMolecule, basis);

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
    data.eri = Eigen::MatrixXd::Zero(numBasis * numBasis, numBasis * numBasis);

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

    data.nuclearRepulsion = NuclearRepulsionDerivative(geometry, atom, direction);

    return data;
}

/// The reference wavefunction of the off-axis fixture: the SCF result and the
/// MO-basis quantities the response consumes.
struct ReferenceState {
    Geometry geometry;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd coreHamiltonian;
    Eigen::MatrixXd density;
    /// The Fock matrix at the reference density, in the AO basis.
    Eigen::MatrixXd fock;
    /// The same matrix in the MO basis, C^T F C - which is what the orbital
    /// gradient L = 2 F_occ,virt is read from, and where the AO one is only
    /// diagonal in the MO basis at convergence.
    Eigen::MatrixXd moFock;
    Eigen::MatrixXd coefficients;
    /// The MO coefficients as a row-major buffer. Eigen stores column-major, and
    /// the request is specified row-major, so the copy is made here rather than
    /// left to a reinterpretation that happens to work at n = 2.
    std::vector<double> moCoefficients;
    std::vector<double> moEri;
    std::vector<double> orbitalEnergies;
    double totalEnergy = 0.0;
    std::size_t numBasis = 0;
    ResponseLayout layout;

    /// \param atom The atom index.
    /// \param direction The Cartesian axis.
    /// \returns The displacement naming that coordinate.
    static NuclearDisplacement DisplacementOf(std::size_t atom, std::size_t direction) {
        NuclearDisplacement displacement;
        displacement.atom = atom;
        displacement.direction = direction;
        return displacement;
    }
};

/// Builds the reference: converged RHF over the fixture at the reference
/// geometry, with the MO two-electron tensor and the MO Fock matrix.
/// \param basis The basis set.
/// \returns The reference, or an Error.
qcx::Result<ReferenceState> BuildReference(const qcx::basisset::BasisSet& basis) {
    ReferenceState state;
    state.geometry = Geometry{};

    const auto molecule = MakeMolecule(state.geometry);

    if (!molecule.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "the fixture did not build"});
    }

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, basis);

    if (!overlap.has_value() || !kinetic.has_value() || !nuclear.has_value() || !eri.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "an integral build failed"});
    }

    state.overlap = ToMatrix(*overlap);
    state.coreHamiltonian = ToMatrix(*kinetic) + ToMatrix(*nuclear);
    state.numBasis = static_cast<std::size_t>(state.overlap.rows());

    const auto reference =
        qcx::scf::RunRhfScf(*molecule, state.overlap, state.coreHamiltonian, *eri);

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
    state.coefficients = reference->coefficients;
    state.orbitalEnergies.assign(reference->orbitalEnergies.begin(),
                                 reference->orbitalEnergies.end());
    state.moCoefficients.assign(static_cast<std::size_t>(state.coefficients.size()), 0.0);

    for (std::size_t mu = 0; mu < static_cast<std::size_t>(state.coefficients.rows()); ++mu)
    {
        for (std::size_t p = 0; p < static_cast<std::size_t>(state.coefficients.cols()); ++p)
        {
            state.moCoefficients[mu * static_cast<std::size_t>(state.coefficients.cols()) + p] =
                state.coefficients(static_cast<int>(mu), static_cast<int>(p));
        }
    }

    constexpr std::size_t kNumOccupied = 1;
    state.layout = ResponseLayout{kNumOccupied, state.numBasis - kNumOccupied};

    // The spin-summed closed-shell density, the convention RunRhfScf documents.
    state.density = 2.0 * state.coefficients.leftCols(static_cast<int>(kNumOccupied)) *
                    state.coefficients.leftCols(static_cast<int>(kNumOccupied)).transpose();

    // The Fock matrix at the reference density, from the same contraction the
    // SCF loop uses: F = H + J[D] - K[D]/2 with J and K the supermatrix passes.
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

    Eigen::VectorXd flatDensity(state.numBasis * state.numBasis);

    for (std::size_t mu = 0; mu < state.numBasis; ++mu)
    {
        for (std::size_t nu = 0; nu < state.numBasis; ++nu)
        {
            flatDensity(static_cast<int>(mu * state.numBasis + nu)) =
                state.density(static_cast<int>(mu), static_cast<int>(nu));
        }
    }

    const Eigen::VectorXd j = coulomb * flatDensity;
    const Eigen::VectorXd k = exchange * flatDensity;
    Eigen::MatrixXd jMatrix(state.numBasis, state.numBasis);
    Eigen::MatrixXd kMatrix(state.numBasis, state.numBasis);

    for (std::size_t mu = 0; mu < state.numBasis; ++mu)
    {
        for (std::size_t nu = 0; nu < state.numBasis; ++nu)
        {
            const int index = static_cast<int>(mu * state.numBasis + nu);
            jMatrix(static_cast<int>(mu), static_cast<int>(nu)) = j(index);
            kMatrix(static_cast<int>(mu), static_cast<int>(nu)) = k(index);
        }
    }

    state.fock = state.coreHamiltonian + jMatrix - 0.5 * kMatrix;
    state.moFock = state.coefficients.transpose() * state.fock * state.coefficients;

    // The MO two-electron tensor, flattened the way OrbitalHessianOperator
    // consumes it: (pq|rs) in chemists' notation, occupied-major.
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
    const Geometry plus = state.geometry.Displaced(atom, direction, step);
    const Geometry minus = state.geometry.Displaced(atom, direction, -step);
    const auto plusMolecule = MakeMolecule(plus);
    const auto minusMolecule = MakeMolecule(minus);

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

/// The right-hand side this module is about: B, the occupied-virtual block of
/// the nuclear-derivative Fock matrix, with the one-electron and two-electron
/// terms wired in as separate named contributions.
///
/// The two-electron contribution is the one the fitting-coefficient discussion
/// is about on an RI path: a fitted Coulomb term's derivative is the derivative
/// of the three-index tensor AND of the metric inverse, and wiring only the
/// first is the classic omission. Both are carried here by using the
/// derivative of the assembled Coulomb matrix, so the fitted case's omission
/// would show up as a disagreement with the finite difference below.
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

    // Term 1: the one-electron derivative. It does not multiply the response
    // amplitudes, and it depends on the reference orbitals only through the
    // transformation into the MO basis - so it is a right-hand-side term.
    FockDerivativeContributionFn oneElectron =
        [&state, &basis, step](const FockDerivativeRequest& request,
                               std::span<double> block) -> qcx::Result<void> {
        auto data = BuildDerivativeData(
            state.geometry, basis, request.displacement.atom, request.displacement.direction, step);

        if (!data.has_value())
        {
            return std::unexpected(data.error());
        }

        const Eigen::MatrixXd moDerivative =
            state.coefficients.transpose() * data->core * state.coefficients;

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

    // Term 2: the two-electron derivative, at the reference density. Same
    // reasoning as term 1 - it is the skeleton part of the derivative Fock
    // matrix, carrying no factor of the amplitudes.
    FockDerivativeContributionFn twoElectron =
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

        Eigen::VectorXd flatDensity(state.numBasis * state.numBasis);

        for (std::size_t mu = 0; mu < state.numBasis; ++mu)
        {
            for (std::size_t nu = 0; nu < state.numBasis; ++nu)
            {
                flatDensity(static_cast<int>(mu * state.numBasis + nu)) =
                    state.density(static_cast<int>(mu), static_cast<int>(nu));
            }
        }

        const Eigen::VectorXd j = data->eri * flatDensity;
        const Eigen::VectorXd k = exchangeDerivative * flatDensity;
        Eigen::MatrixXd fockDerivative(state.numBasis, state.numBasis);

        for (std::size_t mu = 0; mu < state.numBasis; ++mu)
        {
            for (std::size_t nu = 0; nu < state.numBasis; ++nu)
            {
                const int index = static_cast<int>(mu * state.numBasis + nu);
                fockDerivative(static_cast<int>(mu), static_cast<int>(nu)) =
                    j(index) - 0.5 * k(index);
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

    auto first = rightHandSide->Add("one-electron-derivative", std::move(oneElectron));

    if (!first.has_value())
    {
        return std::unexpected(first.error());
    }

    auto second = rightHandSide->Add("two-electron-derivative", std::move(twoElectron));

    if (!second.has_value())
    {
        return std::unexpected(second.error());
    }

    return rightHandSide;
}

/// The `FockDerivativeRequest` one displacement is assembled with.
/// \param state The reference.
/// \param atom The atom index.
/// \param direction The Cartesian axis.
/// \returns The request.
FockDerivativeRequest MakeRequest(const ReferenceState& state,
                                  std::size_t atom,
                                  std::size_t direction) {
    FockDerivativeRequest request;
    request.layout = state.layout;
    request.displacement = ReferenceState::DisplacementOf(atom, direction);
    request.numBasisFunctions = state.numBasis;
    request.moCoefficients = std::span<const double>(state.moCoefficients);
    request.density = std::span<const double>(state.density.data(), state.density.size());

    return request;
}

/// The SCF's total energy at a displaced geometry.
/// \param state The reference.
/// \param basis The basis set.
/// \param atom The atom index.
/// \param direction The Cartesian axis.
/// \param delta The displacement, in Bohr.
/// \returns The total energy, or an Error.
qcx::Result<double> EnergyAt(const ReferenceState& state,
                             const qcx::basisset::BasisSet& basis,
                             std::size_t atom,
                             std::size_t direction,
                             double delta) {
    const Geometry moved = state.geometry.Displaced(atom, direction, delta);
    const auto molecule = MakeMolecule(moved);

    if (!molecule.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "a displaced fixture failed"});
    }

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, basis);

    if (!overlap.has_value() || !kinetic.has_value() || !nuclear.has_value() || !eri.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "an integral build failed"});
    }

    const auto reference = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri);

    if (!reference.has_value())
    {
        return std::unexpected(reference.error());
    }

    return reference->totalEnergy;
}

/// The total derivative, as the decomposition writes it:
/// dE/dR = (the derivative at fixed reference orbitals) + L^T U.
///
/// This is the one place the three buckets are put back together, so that the
/// checks below read the same expression a consumer would.
/// \param state The reference.
/// \param basis The basis set.
/// \param step The central-difference step for the derivative inputs, in Bohr.
/// \param gradient Output, 3 per atom, Hartree/Bohr.
/// \returns An Error from any stage, or nothing.
qcx::Result<void> AssembleTotalGradient(const ReferenceState& state,
                                        const qcx::basisset::BasisSet& basis,
                                        double step,
                                        std::span<double> gradient) {
    auto rightHandSide = BuildRightHandSide(state, basis, step);

    if (!rightHandSide.has_value())
    {
        return std::unexpected(rightHandSide.error());
    }

    const auto hessian = OrbitalHessianOperator::Create(
        state.layout,
        OrbitalEnergies{std::span<const double>(state.orbitalEnergies)},
        MoTwoElectronTensor{std::span<const double>(state.moEri)});

    if (!hessian.has_value())
    {
        return std::unexpected(hessian.error());
    }

    const auto identity = qcx::response::MakeResponseReferenceIdentity(
        GeometryCoordinates(state.geometry),
        std::span<const double>(state.density.data(), state.density.size()));

    if (!identity.has_value())
    {
        return std::unexpected(identity.error());
    }

    ResponseSolver<OrbitalHessianOperator> solver;

    for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            auto data = BuildDerivativeData(state.geometry, basis, atom, direction, step);

            if (!data.has_value())
            {
                return std::unexpected(data.error());
            }

            Eigen::VectorXd flatDensity(state.numBasis * state.numBasis);

            for (std::size_t mu = 0; mu < state.numBasis; ++mu)
            {
                for (std::size_t nu = 0; nu < state.numBasis; ++nu)
                {
                    flatDensity(static_cast<int>(mu * state.numBasis + nu)) =
                        state.density(static_cast<int>(mu), static_cast<int>(nu));
                }
            }

            const Eigen::VectorXd j = data->eri * flatDensity;
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

            const Eigen::VectorXd k = exchangeDerivative * flatDensity;
            Eigen::MatrixXd fockDerivative(state.numBasis, state.numBasis);

            for (std::size_t mu = 0; mu < state.numBasis; ++mu)
            {
                for (std::size_t nu = 0; nu < state.numBasis; ++nu)
                {
                    const int index = static_cast<int>(mu * state.numBasis + nu);
                    fockDerivative(static_cast<int>(mu), static_cast<int>(nu)) =
                        j(index) - 0.5 * k(index);
                }
            }

            // E = Tr(D H) + 1/2 Tr(D G) + E_nuc, so at fixed density the
            // derivative is Tr(D H^R) + 1/2 Tr(D G^R) + dE_nuc/dR. The core
            // derivative carries no 1/2: it appears once in the density's own
            // term and once more through the Fock matrix, and the derivative
            // Fock matrix built above holds only the two-electron part.
            const double fixed = (state.density.cwiseProduct(data->core)).sum() +
                                 0.5 * (state.density.cwiseProduct(fockDerivative)).sum() +
                                 data->nuclearRepulsion;

            const auto overlapDerivative =
                BuildMoOverlapDerivative(state, basis, atom, direction, step);

            if (!overlapDerivative.has_value())
            {
                return std::unexpected(overlapDerivative.error());
            }

            const auto response = SolveNuclearDisplacementResponse(
                *hessian,
                *rightHandSide,
                MakeRequest(state, atom, direction),
                std::span<const double>(overlapDerivative->data(), overlapDerivative->size()),
                *identity,
                solver);

            if (!response.has_value())
            {
                return std::unexpected(response.error());
            }

            const auto weighted = EnergyWeightedTerm(
                std::span<const double>(state.fock.data(), state.fock.size()), *response);

            if (!weighted.has_value())
            {
                return std::unexpected(weighted.error());
            }

            gradient[3 * atom + direction] = fixed + *weighted;
        }
    }

    return {};
}

// --- the seam ---------------------------------------------------------------

TEST(ResponseGradientTest, ContributionsAccumulateAndTheirNamesAreKept) {
    const ResponseLayout layout{2, 3};
    auto rightHandSide = NuclearDisplacementRightHandSide::Create(layout, 5);
    ASSERT_TRUE(rightHandSide.has_value()) << rightHandSide.error().message;

    const auto constant = [](double value) {
        return [value](const FockDerivativeRequest&, std::span<double> block) -> qcx::Result<void> {
            for (double& entry : block)
            {
                entry += value;
            }

            return {};
        };
    };

    ASSERT_TRUE(rightHandSide->Add("a", constant(1.0)).has_value());
    ASSERT_TRUE(rightHandSide->Add("b", constant(2.0)).has_value());
    EXPECT_EQ(rightHandSide->Names().size(), 2u);
    EXPECT_EQ(rightHandSide->Names()[0], "a");
    EXPECT_EQ(rightHandSide->Names()[1], "b");

    // A duplicate name is refused rather than silently replacing the first term:
    // the failure this seam can hide is a term that never ran.
    EXPECT_FALSE(rightHandSide->Add("a", constant(4.0)).has_value());
    EXPECT_FALSE(rightHandSide->Add("", constant(4.0)).has_value());
    EXPECT_FALSE(rightHandSide->Add("c", FockDerivativeContributionFn{}).has_value());

    FockDerivativeRequest request;
    request.layout = layout;
    request.numBasisFunctions = 5;
    std::vector<double> block(layout.Dimension(), 99.0);
    ASSERT_TRUE(rightHandSide->Assemble(request, block).has_value());

    for (const double entry : block)
    {
        EXPECT_DOUBLE_EQ(entry, 3.0);
    }
}

TEST(ResponseGradientTest, TheReferenceIdentitySeparatesGeometriesAndDensities) {
    const std::vector<double> coordinates{0.0, 0.0, 0.0, 1.4, 0.18, -0.12};
    const std::vector<double> density{1.0, 0.5, 0.5, 1.0};

    const auto base = qcx::response::MakeResponseReferenceIdentity(coordinates, density);
    ASSERT_TRUE(base.has_value()) << base.error().message;

    // A digest is only useful if it is finer than the thing it guards: a
    // displacement far below any tolerance must still change it.
    std::vector<double> movedCoordinates = coordinates;
    movedCoordinates[3] += 1e-12;
    const auto moved = qcx::response::MakeResponseReferenceIdentity(movedCoordinates, density);
    ASSERT_TRUE(moved.has_value());
    EXPECT_NE(base->geometry, moved->geometry);

    std::vector<double> movedDensity = density;
    movedDensity[1] += 1e-12;
    const auto densityMoved =
        qcx::response::MakeResponseReferenceIdentity(coordinates, movedDensity);
    ASSERT_TRUE(densityMoved.has_value());
    EXPECT_NE(base->density, densityMoved->density);
    EXPECT_EQ(base->geometry, densityMoved->geometry);

    EXPECT_FALSE(qcx::response::MakeResponseReferenceIdentity(std::span<const double>{}, density)
                     .has_value());
    EXPECT_FALSE(
        qcx::response::MakeResponseReferenceIdentity(coordinates, std::span<const double>{})
            .has_value());
    EXPECT_FALSE(
        qcx::response::MakeResponseReferenceIdentity(std::array<double, 2>{0.0, 0.0}, density)
            .has_value());
}

TEST(ResponseGradientTest, AStaleHandleMissesRatherThanReturningTheOldAnswer) {
    OrbitalResponseCache cache;
    ResponseReferenceIdentity first;
    first.geometry = 11;
    first.density = 22;

    OrbitalResponse response;
    response.numBasisFunctions = 2;
    response.numOrbitals = 2;
    response.displacement = ReferenceState::DisplacementOf(1, 0);
    response.reference = first;
    response.amplitudes.assign(4, 0.0);
    response.perturbedDensity.assign(4, 0.0);

    ASSERT_TRUE(cache.Store(response).has_value());
    ASSERT_EQ(cache.Size(), 1u);

    ResponseCacheHandle handle;
    handle.displacement = response.displacement;
    handle.reference = first;
    ASSERT_NE(cache.Lookup(handle), nullptr);

    // Same displacement, different reference: the answer belongs to another
    // geometry and must not be handed back.
    ResponseCacheHandle stale = handle;
    stale.reference.geometry = 12;
    EXPECT_EQ(cache.Lookup(stale), nullptr);

    stale.reference = first;
    stale.reference.density = 23;
    EXPECT_EQ(cache.Lookup(stale), nullptr);

    // Same reference, different displacement.
    ResponseCacheHandle otherDisplacement = handle;
    otherDisplacement.displacement = ReferenceState::DisplacementOf(1, 1);
    EXPECT_EQ(cache.Lookup(otherDisplacement), nullptr);

    cache.Invalidate();
    EXPECT_EQ(cache.Size(), 0u);
    EXPECT_EQ(cache.Lookup(handle), nullptr);
}

// --- the invariants, in the order the decomposition puts them in ------------

// Orthonormality is the check that tests the solver rather than the energy:
// the perturbed orbitals must stay orthonormal, which is a constraint on the
// amplitudes alone - (C(1+U))^T S(R) (C(1+U)) = 1 to first order gives
// U + U^T = -C^T S^R C - and it holds whatever the energy does.
TEST(ResponseGradientTest, ThePerturbedOrbitalsStayOrthonormal) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto state = BuildReference(*basis);
    ASSERT_TRUE(state.has_value()) << state.error().message;

    auto rightHandSide = BuildRightHandSide(*state, *basis, 1e-4);
    ASSERT_TRUE(rightHandSide.has_value()) << rightHandSide.error().message;

    const auto hessian = OrbitalHessianOperator::Create(
        state->layout,
        OrbitalEnergies{std::span<const double>(state->orbitalEnergies)},
        MoTwoElectronTensor{std::span<const double>(state->moEri)});
    ASSERT_TRUE(hessian.has_value()) << hessian.error().message;

    ResponseSolver<OrbitalHessianOperator> solver;
    double worst = 0.0;

    for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            const auto overlapDerivative =
                BuildMoOverlapDerivative(*state, *basis, atom, direction, 1e-4);
            ASSERT_TRUE(overlapDerivative.has_value()) << overlapDerivative.error().message;

            const auto identity = qcx::response::MakeResponseReferenceIdentity(
                GeometryCoordinates(state->geometry),
                std::span<const double>(state->density.data(), state->density.size()));
            ASSERT_TRUE(identity.has_value());

            const auto response = SolveNuclearDisplacementResponse(
                *hessian,
                *rightHandSide,
                MakeRequest(*state, atom, direction),
                std::span<const double>(overlapDerivative->data(), overlapDerivative->size()),
                *identity,
                solver);
            ASSERT_TRUE(response.has_value()) << response.error().message;

            const std::size_t n = response->numOrbitals;

            for (std::size_t p = 0; p < n; ++p)
            {
                for (std::size_t q = 0; q < n; ++q)
                {
                    const double residual =
                        response->Amplitude(p, q) + response->Amplitude(q, p) +
                        (*overlapDerivative)(static_cast<int>(p), static_cast<int>(q));
                    worst = std::max(worst, std::abs(residual));
                }
            }
        }
    }

    std::cout << "[orthonormality] worst |U + U^T + C^T S^R C| over 6 displacements " << worst
              << "\n";
    EXPECT_LT(worst, 1e-10);
}

// The response equation's own residual: A U + B, which must vanish. It is the
// check that the right-hand side is the companion of this operator and not of
// another one, and it is independent of any energy.
TEST(ResponseGradientTest, TheResponseSatisfiesItsOwnEquation) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto state = BuildReference(*basis);
    ASSERT_TRUE(state.has_value()) << state.error().message;

    auto rightHandSide = BuildRightHandSide(*state, *basis, 1e-4);
    ASSERT_TRUE(rightHandSide.has_value()) << rightHandSide.error().message;

    const auto hessian = OrbitalHessianOperator::Create(
        state->layout,
        OrbitalEnergies{std::span<const double>(state->orbitalEnergies)},
        MoTwoElectronTensor{std::span<const double>(state->moEri)});
    ASSERT_TRUE(hessian.has_value()) << hessian.error().message;

    ResponseSolver<OrbitalHessianOperator> solver;
    const auto overlapDerivative = BuildMoOverlapDerivative(*state, *basis, 1, 0, 1e-4);
    ASSERT_TRUE(overlapDerivative.has_value());

    const auto identity = qcx::response::MakeResponseReferenceIdentity(
        GeometryCoordinates(state->geometry),
        std::span<const double>(state->density.data(), state->density.size()));
    ASSERT_TRUE(identity.has_value());

    const auto request = MakeRequest(*state, 1, 0);
    const auto response = SolveNuclearDisplacementResponse(
        *hessian,
        *rightHandSide,
        request,
        std::span<const double>(overlapDerivative->data(), overlapDerivative->size()),
        *identity,
        solver);
    ASSERT_TRUE(response.has_value()) << response.error().message;

    std::vector<double> assembled(state->layout.Dimension(), 0.0);
    ASSERT_TRUE(rightHandSide->Assemble(request, assembled).has_value());

    // The amplitude the operator acts on is the antisymmetric half of the
    // occupied-virtual block that Solve returned.
    std::vector<double> solved(state->layout.Dimension(), 0.0);

    for (std::size_t i = 0; i < state->layout.numOccupied; ++i)
    {
        for (std::size_t a = 0; a < state->layout.numVirtual; ++a)
        {
            const std::size_t virtualOrbital = state->layout.numOccupied + a;
            solved[i * state->layout.numVirtual + a] =
                0.5 *
                (response->Amplitude(virtualOrbital, i) - response->Amplitude(i, virtualOrbital));
        }
    }

    std::vector<double> action(state->layout.Dimension(), 0.0);
    ASSERT_TRUE(hessian->Apply(solved, action).has_value());

    double residual = 0.0;
    double scale = 0.0;

    for (std::size_t k = 0; k < action.size(); ++k)
    {
        residual += (action[k] + assembled[k]) * (action[k] + assembled[k]);
        scale += assembled[k] * assembled[k];
    }

    const double relative = std::sqrt(residual) / std::sqrt(scale);
    std::cout << "[response equation] ||A U + B|| / ||B|| " << relative << ", ||B|| "
              << std::sqrt(scale) << "\n";

    EXPECT_LT(relative, 1e-6);
    EXPECT_GT(std::sqrt(scale), 1e-6) << "a zero right-hand side would make the check vacuous";
}

// The decomposition's own evidence, measured. At a converged reference the
// orbital gradient L vanishes, so an energy-weighted term written as an
// amplitude contraction - L^T U - is identically zero here. That is the trap:
// the perturbed density is not zero, and neither is its contraction with the
// Fock matrix.
//
// With F diagonal in the MO basis at convergence the contraction collapses to
// the occupied diagonal of the perturbed density, and the orthonormality
// constraint U + U^T = -S^R turns that diagonal into the overlap derivative's:
// Tr(F D(1)) = -2 sum_i eps_i S^R_ii over the occupied orbitals. Both are
// checked, because the second is exactly what an amplitude-only contraction
// drops.
TEST(ResponseGradientTest, TheEnergyWeightedTermSurvivesTheConvergedReference) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto state = BuildReference(*basis);
    ASSERT_TRUE(state.has_value()) << state.error().message;

    std::vector<double> orbitalGradient(state->layout.Dimension(), 0.0);
    const auto gradient =
        OrbitalGradient(std::span<const double>(state->moFock.data(), state->moFock.size()),
                        state->layout,
                        orbitalGradient);
    ASSERT_TRUE(gradient.has_value()) << gradient.error().message;

    double largest = 0.0;

    for (const double entry : orbitalGradient)
    {
        largest = std::max(largest, std::abs(entry));
    }

    auto rightHandSide = BuildRightHandSide(*state, *basis, 1e-4);
    ASSERT_TRUE(rightHandSide.has_value());

    const auto hessian = OrbitalHessianOperator::Create(
        state->layout,
        OrbitalEnergies{std::span<const double>(state->orbitalEnergies)},
        MoTwoElectronTensor{std::span<const double>(state->moEri)});
    ASSERT_TRUE(hessian.has_value());

    ResponseSolver<OrbitalHessianOperator> solver;
    const auto overlapDerivative = BuildMoOverlapDerivative(*state, *basis, 1, 0, 1e-4);
    ASSERT_TRUE(overlapDerivative.has_value());

    const auto identity = qcx::response::MakeResponseReferenceIdentity(
        GeometryCoordinates(state->geometry),
        std::span<const double>(state->density.data(), state->density.size()));
    ASSERT_TRUE(identity.has_value());

    const auto response = SolveNuclearDisplacementResponse(
        *hessian,
        *rightHandSide,
        MakeRequest(*state, 1, 0),
        std::span<const double>(overlapDerivative->data(), overlapDerivative->size()),
        *identity,
        solver);
    ASSERT_TRUE(response.has_value()) << response.error().message;

    double amplitudeNorm = 0.0;

    for (std::size_t i = 0; i < state->layout.numOccupied; ++i)
    {
        for (std::size_t a = 0; a < state->layout.numVirtual; ++a)
        {
            const double entry = response->Amplitude(state->layout.numOccupied + a, i);
            amplitudeNorm += entry * entry;
        }
    }

    const auto term = EnergyWeightedTerm(
        std::span<const double>(state->fock.data(), state->fock.size()), *response);
    ASSERT_TRUE(term.has_value()) << term.error().message;

    // The overlap derivative's own form of the same number, at a converged
    // reference.
    double fromOverlap = 0.0;

    for (std::size_t i = 0; i < state->layout.numOccupied; ++i)
    {
        fromOverlap -= 2.0 * state->orbitalEnergies[i] *
                       (*overlapDerivative)(static_cast<int>(i), static_cast<int>(i));
    }

    std::cout << "[decomposition] max |L| " << largest << ", ||U_ov|| " << std::sqrt(amplitudeNorm)
              << ", Tr(F D(1)) " << *term << ", -2 sum_i eps_i S^R_ii " << fromOverlap << "\n";

    EXPECT_LT(largest, 1e-8) << "the reference's orbital gradient should vanish";
    EXPECT_GT(std::sqrt(amplitudeNorm), 1e-6) << "the response itself must not be zero";
    EXPECT_GT(std::abs(*term), 1e-6) << "an amplitude-only contraction would leave this zero";
    EXPECT_NEAR(*term, fromOverlap, 1e-9);
}

// Translational invariance: the same displacement applied to every atom leaves
// the energy unchanged, so the gradient summed over the atoms is zero. It is a
// necessary condition and NOT a sufficient one - a gradient that is exactly
// half the truth also sums to zero - which is why it is not the only check.
TEST(ResponseGradientTest, TheGradientIsTranslationallyInvariant) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto state = BuildReference(*basis);
    ASSERT_TRUE(state.has_value()) << state.error().message;

    std::vector<double> total(3 * kNumAtoms, 0.0);
    const auto assembled = AssembleTotalGradient(*state, *basis, 1e-4, total);
    ASSERT_TRUE(assembled.has_value()) << assembled.error().message;

    // The sum over atoms of the gradient, per Cartesian direction.
    std::array<double, 3> net{0.0, 0.0, 0.0};

    for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            net[direction] += total[3 * atom + direction];
        }
    }

    std::cout << "[translation] net force (" << net[0] << ", " << net[1] << ", " << net[2] << ")\n";

    for (const double entry : net)
    {
        EXPECT_LT(std::abs(entry), 1e-6);
    }
}

// Rotational invariance, the stronger of the two: it couples the positions to
// the gradient components. A rigid rotation of the whole molecule leaves the
// energy unchanged, so sum_k r_k x g_k is zero.
TEST(ResponseGradientTest, TheGradientCarriesNoTorque) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto state = BuildReference(*basis);
    ASSERT_TRUE(state.has_value()) << state.error().message;

    // The torque is read about the origin, which is where the first atom sits.
    const std::array<std::array<double, 3>, kNumAtoms> positions{
        state->geometry.Position(0), state->geometry.Position(1), state->geometry.Position(2)};

    std::vector<double> gradient(3 * kNumAtoms, 0.0);
    ASSERT_TRUE(AssembleTotalGradient(*state, *basis, 1e-4, gradient).has_value());

    std::array<double, 3> torque{0.0, 0.0, 0.0};

    for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
    {
        const double gx = gradient[3 * atom + 0];
        const double gy = gradient[3 * atom + 1];
        const double gz = gradient[3 * atom + 2];
        torque[0] += positions[atom][1] * gz - positions[atom][2] * gy;
        torque[1] += positions[atom][2] * gx - positions[atom][0] * gz;
        torque[2] += positions[atom][0] * gy - positions[atom][1] * gx;
    }

    std::cout << "[rotation] torque (" << torque[0] << ", " << torque[1] << ", " << torque[2]
              << ")\n";

    for (const double entry : torque)
    {
        EXPECT_LT(std::abs(entry), 1e-6);
    }
}

// The energy-derivative consistency, and last the finite difference. The
// analytic total is the same one the checks above used; the finite difference
// is a central difference of the SCF's own total energy, taken at two steps so
// that its truncation can be seen rather than assumed.
//
// SCOPE, stated because it matters: the derivative integrals in this file are
// finite differences of the engine's value builders, so this pair of checks
// measures the placement of every term and the contraction, and it also inherits
// the difference's own error. It does not re-validate the integral engine's
// differentiation, which is checked where it lives.
TEST(ResponseGradientTest, TheGradientAgreesWithTheEnergyItDifferentiates) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto state = BuildReference(*basis);
    ASSERT_TRUE(state.has_value()) << state.error().message;

    std::vector<double> gradient(3 * kNumAtoms, 0.0);
    const auto assembled = AssembleTotalGradient(*state, *basis, 1e-4, gradient);
    ASSERT_TRUE(assembled.has_value()) << assembled.error().message;

    // The y displacement of the second atom: nonzero, and not the bond axis.
    const std::size_t index = 3 * 1 + 1;
    double worst = 0.0;

    for (const double step : {1e-3, 3e-4})
    {
        const auto plus = EnergyAt(*state, *basis, 1, 1, step);
        const auto minus = EnergyAt(*state, *basis, 1, 1, -step);
        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        const double difference = (*plus - *minus) / (2.0 * step);
        const double deviation = std::abs(difference - gradient[index]);
        worst = std::max(worst, deviation);

        std::cout << "[energy derivative] dE/dR_y analytic " << gradient[index] << ", finite "
                  << difference << " at h = " << step << " (deviation " << deviation << ")\n";
    }

    EXPECT_LT(worst, 1e-6);
}

// The premise every term below is contracted against, checked on the
// reference's own objects: E = 1/2 Tr(D H) + 1/2 Tr(D F) + E_nuc. A convention
// slip here - a density that is not the one the Fock matrix was built from, or
// a repulsion that is not the one the energy carries - would leave every check
// in this file measuring a different functional than the one it differentiates.
TEST(ResponseGradientTest, TheReferenceEnergyMatchesItsOwnDecomposition) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto state = BuildReference(*basis);
    ASSERT_TRUE(state.has_value()) << state.error().message;

    const double energy = 0.5 * (state->density.cwiseProduct(state->coreHamiltonian)).sum() +
                          0.5 * (state->density.cwiseProduct(state->fock)).sum() +
                          NuclearRepulsionEnergy(state->geometry);

    std::cout << "[reference energy] SCF " << state->totalEnergy << ", decomposition " << energy
              << " (difference " << (energy - state->totalEnergy) << ")\n";
    EXPECT_NEAR(energy, state->totalEnergy, 1e-12);
}

} // namespace
