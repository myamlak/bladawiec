// The total analytic nuclear gradient: the response state's build (dense AO
// tensor, four quarter-transforms, the reference Fock, the screening
// structure), the two derivative walks, the two refusal gates, and the sum of
// the terms into the emitted gradient.

#include "qcx/properties/gradient.hpp"

#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/molecule/mass_properties.hpp"
#include "qcx/response/response_gradient.hpp"
#include "qcx/response/response_operator.hpp"
#include "qcx/scf/density_energy.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qcx::properties {
namespace {

qcx::Error Invalid(std::string message) {
    return qcx::Error{qcx::ErrorCode::kInvalidArgument, std::move(message)};
}

// The row-major n x n flattening every state input and every response entry
// reads; Eigen stores column-major, so the copy is explicit rather than a
// reinterpretation of the buffer.
std::vector<double> RowMajor(const Eigen::MatrixXd& matrix) {
    const std::size_t rows = static_cast<std::size_t>(matrix.rows());
    const std::size_t columns = static_cast<std::size_t>(matrix.cols());
    std::vector<double> flat(rows * columns, 0.0);

    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t column = 0; column < columns; ++column)
        {
            flat[row * columns + column] =
                matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
        }
    }

    return flat;
}

// The inverse of RowMajor for the square matrices the MO-basis reads need.
Eigen::MatrixXd FromRowMajor(std::span<const double> flat, std::size_t size) {
    Eigen::MatrixXd matrix(static_cast<Eigen::Index>(size), static_cast<Eigen::Index>(size));

    for (std::size_t row = 0; row < size; ++row)
    {
        for (std::size_t column = 0; column < size; ++column)
        {
            matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) =
                flat[row * size + column];
        }
    }

    return matrix;
}

// A rank-2 CPU tensor from a row-major span - the shape the derivative walks
// take their density in.
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> DensityTensor(
    std::span<const double> density, std::size_t numBasis) {
    if (density.size() != numBasis * numBasis)
    {
        return std::unexpected(Invalid(
            "the reference density is not numBasis^2 = " + std::to_string(numBasis * numBasis) +
            " values; it carries " + std::to_string(density.size())));
    }

    auto tensor =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({numBasis, numBasis});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    for (std::size_t row = 0; row < numBasis; ++row)
    {
        for (std::size_t column = 0; column < numBasis; ++column)
        {
            (*tensor)(row, column) = density[row * numBasis + column];
        }
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

// The molecule's nuclear coordinates, flat and atom-major.
std::vector<double> CoordinatesRowMajor(const qcx::molecule::Molecule& molecule) {
    const auto& coordinates = molecule.CoordinatesBohr();
    std::vector<double> flat(3 * molecule.AtomCount(), 0.0);

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            flat[3 * atom + direction] = coordinates(atom, direction);
        }
    }

    return flat;
}

// The axis a quarter-transform contracts and the basis dimension it runs over.
struct AxisContraction {
    std::size_t axis = 0;
    std::size_t numBasis = 0;
};

// One quarter-transform of the AO two-electron tensor: contracts one of its
// four axes with the MO coefficients. The tensor is row-major, so axis k has
// stride numBasis^(3-k); the three free axes are decoded from a counter and
// the summed axis is the innermost loop.
std::vector<double> ContractAxis(const std::vector<double>& tensor,
                                 std::span<const double> coefficients,
                                 AxisContraction contraction) {
    const std::size_t numBasis = contraction.numBasis;
    const std::size_t axis = contraction.axis;
    const std::array<std::size_t, 4> strides{
        numBasis * numBasis * numBasis, numBasis * numBasis, numBasis, 1};
    const std::size_t axisStride = strides[axis];
    std::vector<double> transformed(tensor.size(), 0.0);
    std::array<std::size_t, 4> index{};
    const std::size_t freeCount = numBasis * numBasis * numBasis;

    for (std::size_t free = 0; free < freeCount; ++free)
    {
        // Decode the counter into the three axes this transform does not sum
        // over, in storage order.
        std::size_t remaining = free;
        std::size_t offset = 0;

        for (std::size_t dimension = 4; dimension-- > 0;)
        {
            if (dimension == axis)
            {
                continue;
            }

            index[dimension] = remaining % numBasis;
            remaining /= numBasis;
            offset += index[dimension] * strides[dimension];
        }

        for (std::size_t target = 0; target < numBasis; ++target)
        {
            double sum = 0.0;

            for (std::size_t source = 0; source < numBasis; ++source)
            {
                sum +=
                    coefficients[source * numBasis + target] * tensor[offset + source * axisStride];
            }

            transformed[offset + target * axisStride] = sum;
        }
    }

    return transformed;
}

// Transforms the AO tensor (uv|ls) into the MO tensor (pq|rs), chemists'
// notation, by the four quarter-transforms the state's input declares. A
// direct four-index transform would be numBasis^8 operations; this is four
// numBasis^5 passes.
std::vector<double> TransformEriToMo(const std::vector<double>& aoTensor,
                                     std::span<const double> coefficients,
                                     std::size_t numBasis) {
    const std::vector<double> lastAxis = ContractAxis(aoTensor, coefficients, {3, numBasis});
    const std::vector<double> thirdAxis = ContractAxis(lastAxis, coefficients, {2, numBasis});
    const std::vector<double> secondAxis = ContractAxis(thirdAxis, coefficients, {1, numBasis});
    return ContractAxis(secondAxis, coefficients, {0, numBasis});
}

// The screening structure the response state carries: its geometry digest,
// and one bound per AO pair, mu <= nu, each AO pair taking its containing
// shell pair's bound (the shell pair's own maximum over that block).
qcx::Result<response::ResponseScreening> BuildScreening(const qcx::molecule::Molecule& molecule,
                                                        const qcx::basisset::BasisSet& basisSet,
                                                        std::span<const double> coordinatesBohr,
                                                        std::span<const double> density,
                                                        std::size_t numBasis) {
    auto shellPairs = integrals::BuildShellPairs(molecule, basisSet);

    if (!shellPairs.has_value())
    {
        return std::unexpected(shellPairs.error());
    }

    auto bounds = integrals::ComputeTwoElectronPairBounds(molecule, basisSet);

    if (!bounds.has_value())
    {
        return std::unexpected(bounds.error());
    }

    if (bounds->size() != shellPairs->pairs.size())
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInternalError,
            "total gradient: the pair-bound build returned " + std::to_string(bounds->size()) +
                " bounds for " + std::to_string(shellPairs->pairs.size()) + " shell pairs"});
    }

    auto identity = response::MakeResponseReferenceIdentity(coordinatesBohr, density);

    if (!identity.has_value())
    {
        return std::unexpected(identity.error());
    }

    response::ResponseScreening screening;
    screening.geometry = identity->geometry;
    const std::size_t pairCount = numBasis * (numBasis + 1) / 2;
    screening.pairValues.assign(pairCount, 0.0);
    screening.pairDerivativeBounds.assign(pairCount, 0.0);

    for (std::size_t pair = 0; pair < shellPairs->pairs.size(); ++pair)
    {
        const integrals::ShellInfo& left = shellPairs->shells[shellPairs->pairs[pair].i];
        const integrals::ShellInfo& right = shellPairs->shells[shellPairs->pairs[pair].j];
        const std::size_t leftCount = integrals::ShellFunctionCount(left);
        const std::size_t rightCount = integrals::ShellFunctionCount(right);

        for (std::size_t rowOffset = 0; rowOffset < leftCount; ++rowOffset)
        {
            for (std::size_t columnOffset = 0; columnOffset < rightCount; ++columnOffset)
            {
                const std::size_t row = left.functionOffset + rowOffset;
                const std::size_t column = right.functionOffset + columnOffset;

                // A shell pair with i == j also carries its block's lower
                // triangle; the pair set is the upper one.
                if (row > column)
                {
                    continue;
                }

                const std::size_t index = row * (2 * numBasis - row + 1) / 2 + (column - row);
                screening.pairValues[index] = (*bounds)[pair].value;
                screening.pairDerivativeBounds[index] = (*bounds)[pair].derivative;
            }
        }
    }

    return screening;
}

// dE_nuc/dR for E_nuc = sum_{A<B} Z_A Z_B / |R_A - R_B|: the analytic
// derivative of the pair sum. The molecule module exposes the energy and no
// derivative of it, so the closed form is written here rather than reached
// through a second integral path.
Eigen::VectorXd NuclearRepulsionGradient(const qcx::molecule::Molecule& molecule) {
    const std::size_t atoms = molecule.AtomCount();
    const auto& coordinates = molecule.CoordinatesBohr();
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(3 * atoms));

    for (std::size_t atom = 0; atom < atoms; ++atom)
    {
        for (std::size_t other = 0; other < atoms; ++other)
        {
            if (atom == other)
            {
                continue;
            }

            std::array<double, 3> delta{};
            double distanceSquared = 0.0;

            for (std::size_t direction = 0; direction < 3; ++direction)
            {
                delta[direction] = coordinates(atom, direction) - coordinates(other, direction);
                distanceSquared += delta[direction] * delta[direction];
            }

            const double distance = std::sqrt(distanceSquared);
            const double charges = static_cast<double>(molecule.Atoms()[atom].atomicNumber) *
                                   static_cast<double>(molecule.Atoms()[other].atomicNumber);
            const double factor = charges / (distanceSquared * distance);

            for (std::size_t direction = 0; direction < 3; ++direction)
            {
                gradient(static_cast<Eigen::Index>(3 * atom + direction)) -=
                    factor * delta[direction];
            }
        }
    }

    return gradient;
}

// The largest absolute entry of a block.
double LargestMagnitude(std::span<const double> values) {
    double largest = 0.0;

    for (const double value : values)
    {
        largest = std::max(largest, std::abs(value));
    }

    return largest;
}

} // namespace

qcx::Result<response::ResponseState> BuildResponseState(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::scf::ScfGradientReference& reference,
    const integrals::EriDenseOptions& eriOptions,
    const response::ResponseSolveOptions& solverOptions) {
    const std::size_t numBasis = reference.numBasisFunctions;
    const Eigen::Index basis = static_cast<Eigen::Index>(numBasis);

    if (reference.coefficients.rows() != basis || reference.coefficients.cols() != basis)
    {
        return std::unexpected(
            Invalid("total gradient: the reference MO coefficients must be numBasis x numBasis = " +
                    std::to_string(numBasis) + " x " + std::to_string(numBasis) +
                    "; the reference carries " + std::to_string(reference.coefficients.rows()) +
                    " x " + std::to_string(reference.coefficients.cols())));
    }

    if (reference.density.rows() != basis || reference.density.cols() != basis)
    {
        return std::unexpected(
            Invalid("total gradient: the reference density must be numBasis x numBasis = " +
                    std::to_string(numBasis) + " x " + std::to_string(numBasis) +
                    "; the reference carries " + std::to_string(reference.density.rows()) + " x " +
                    std::to_string(reference.density.cols())));
    }

    if (reference.orbitalEnergies.size() != basis)
    {
        return std::unexpected(
            Invalid("total gradient: the reference orbital energies must have numBasis = " +
                    std::to_string(numBasis) + " entries; the reference carries " +
                    std::to_string(reference.orbitalEnergies.size())));
    }

    if (reference.numOccupied == 0 || reference.numOccupied + reference.numVirtual != numBasis)
    {
        return std::unexpected(Invalid(
            "total gradient: the occupied and virtual counts must partition the orbital count, "
            "with at least one occupied orbital; the reference states " +
            std::to_string(reference.numOccupied) + " occupied and " +
            std::to_string(reference.numVirtual) + " virtual of " + std::to_string(numBasis)));
    }

    const std::size_t electrons = static_cast<std::size_t>(molecule.ElectronCount());

    if (electrons % 2 != 0)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kUnimplemented,
            "total gradient: the molecule has " + std::to_string(electrons) +
                " electrons, so it is not a closed shell; the total gradient is closed-shell "
                "only (an open-shell response is not implemented)"});
    }

    if (electrons != 2 * reference.numOccupied)
    {
        return std::unexpected(Invalid(
            "total gradient: the molecule's " + std::to_string(electrons) + " electrons need " +
            std::to_string(electrons / 2) +
            " occupied spatial orbitals, but the reference states " +
            std::to_string(reference.numOccupied) + "; the reference belongs to another system"));
    }

    const auto kinetic = integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    const auto nuclearAttraction = integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclearAttraction.has_value())
    {
        return std::unexpected(nuclearAttraction.error());
    }

    Eigen::MatrixXd core(static_cast<Eigen::Index>(numBasis), static_cast<Eigen::Index>(numBasis));

    for (std::size_t row = 0; row < numBasis; ++row)
    {
        for (std::size_t column = 0; column < numBasis; ++column)
        {
            core(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) =
                (*kinetic)(row, column) + (*nuclearAttraction)(row, column);
        }
    }

    auto eriTensor = integrals::BuildEriTensorGeneral(molecule, basisSet, eriOptions);

    if (!eriTensor.has_value())
    {
        return std::unexpected(eriTensor.error());
    }

    const auto& shape = eriTensor->Shape();

    if (shape[0] != numBasis || shape[1] != numBasis || shape[2] != numBasis ||
        shape[3] != numBasis)
    {
        return std::unexpected(
            Invalid("total gradient: the reference states " + std::to_string(numBasis) +
                    " basis functions but the basis set spans " + std::to_string(shape[0]) +
                    "; the reference belongs to another basis"));
    }

    const std::vector<double> density = RowMajor(reference.density);
    const std::vector<double> coefficients = RowMajor(reference.coefficients);
    const std::vector<double> orbitalEnergies(reference.orbitalEnergies.data(),
                                              reference.orbitalEnergies.data() +
                                                  reference.orbitalEnergies.size());

    // The Fock at the reference density, through the SCF module's own
    // evaluator: the same Roothaan contraction the SCF loop runs, so the
    // response operator and the energy-weighted term read the matrix the
    // reference was converged against.
    auto evaluated = scf::EvaluateRhfDensityEnergy(molecule, core, *eriTensor, reference.density);

    if (!evaluated.has_value())
    {
        return std::unexpected(evaluated.error());
    }

    const std::vector<double> fock = RowMajor(evaluated->fock);
    const std::vector<double>& aoTensor = eriTensor->HostView();
    const std::vector<double> moTensor = TransformEriToMo(aoTensor, coefficients, numBasis);
    const std::vector<double> coordinates = CoordinatesRowMajor(molecule);

    auto screening = BuildScreening(molecule, basisSet, coordinates, density, numBasis);

    if (!screening.has_value())
    {
        return std::unexpected(screening.error());
    }

    response::ResponseStateReference input;
    input.converged = true;
    input.numBasisFunctions = numBasis;
    input.numOccupied = reference.numOccupied;
    input.numVirtual = reference.numVirtual;
    input.coordinatesBohr = coordinates;
    input.moCoefficients = coefficients;
    input.orbitalEnergies = orbitalEnergies;
    input.density = density;
    input.fock = fock;
    input.moTwoElectron = moTensor;
    input.history.scfIterations = static_cast<std::size_t>(reference.iterations);
    input.history.achievedEnergyDelta = reference.achievedEnergyDelta;
    input.history.achievedRmsDensityChange = reference.achievedRmsDensityChange;

    // The state copies every array at construction, so the locals above only
    // have to outlive this call.
    return response::ResponseState::Create(input, std::move(*screening), solverOptions);
}

qcx::Result<TotalGradient> ComputeTotalGradient(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const qcx::scf::ScfGradientReference& reference,
                                                const TotalGradientOptions& options) {
    if (!(options.stationarityTolerance > 0.0) || !(options.energyTolerance > 0.0))
    {
        return std::unexpected(
            Invalid("total gradient: both gate tolerances must be positive; the request states " +
                    std::to_string(options.stationarityTolerance) + " (stationarity) and " +
                    std::to_string(options.energyTolerance) + " (energy)"));
    }

    // The dispersion correction first: it is cheap, and a D4 request refuses
    // here by name before any heavy build runs.
    std::optional<dispersion::DispersionGradient> dispersion;

    if (options.dispersion.has_value())
    {
        auto evaluated = dispersion::EvaluateDispersionGradient(molecule, *options.dispersion);

        if (!evaluated.has_value())
        {
            return std::unexpected(evaluated.error());
        }

        dispersion = *evaluated;
    }

    auto state = BuildResponseState(molecule, basisSet, reference, options.eri, options.solver);

    if (!state.has_value())
    {
        return std::unexpected(state.error());
    }

    const std::size_t numBasis = state->NumBasisFunctions();
    const std::size_t atoms = state->NumAtoms();
    const std::size_t displacements = 3 * atoms;
    const response::ResponseLayout layout = state->Layout();
    const std::size_t numOccupied = layout.numOccupied;
    const std::size_t numVirtual = layout.numVirtual;
    const std::size_t orbitals = numOccupied + numVirtual;

    // The stationarity gate: a reference whose occupied-virtual Fock block is
    // not zero is not the stationary point this gradient belongs to, and its
    // gradient would look plausible while differentiating a different energy.
    const Eigen::MatrixXd moCoefficients = FromRowMajor(state->MoCoefficients(), numBasis);
    const Eigen::MatrixXd aoFock = FromRowMajor(state->Fock(), numBasis);
    const Eigen::MatrixXd moFock = moCoefficients.transpose() * aoFock * moCoefficients;
    std::vector<double> orbitalGradient(layout.Dimension(), 0.0);
    auto extracted = response::OrbitalGradient(RowMajor(moFock), layout, orbitalGradient);

    if (!extracted.has_value())
    {
        return std::unexpected(extracted.error());
    }

    const double stationarityResidual = LargestMagnitude(orbitalGradient);

    if (stationarityResidual > options.stationarityTolerance)
    {
        return std::unexpected(Invalid(
            "total gradient: the reference is not stationary - max|2 F_ai| = " +
            std::to_string(stationarityResidual) + " exceeds the tolerance " +
            std::to_string(options.stationarityTolerance) +
            " - so the gradient would differentiate an energy this density is not the minimum "
            "of; converge the reference tighter, or pass the tolerance the run needs"));
    }

    auto densityTensor = DensityTensor(state->Density(), numBasis);

    if (!densityTensor.has_value())
    {
        return std::unexpected(densityTensor.error());
    }

    auto oneElectron = integrals::ComputeOneElectronGradient(molecule, basisSet, *densityTensor);

    if (!oneElectron.has_value())
    {
        return std::unexpected(oneElectron.error());
    }

    // The MO-basis one-electron derivatives: the core-Hamiltonian
    // occupied-virtual block (the response right-hand side's own core term)
    // and C^T S' C, the overlap derivative whose symmetric part fixes the
    // response's redundant rotations.
    integrals::OneElectronFockDerivativeOptions fockDerivativeOptions;
    auto oneElectronFock = integrals::ComputeOneElectronFockDerivative(molecule,
                                                                       basisSet,
                                                                       state->MoCoefficients(),
                                                                       numOccupied,
                                                                       numVirtual,
                                                                       fockDerivativeOptions);

    if (!oneElectronFock.has_value())
    {
        return std::unexpected(oneElectronFock.error());
    }

    integrals::TwoElectronFockDerivativeOptions twoElectronOptions;
    twoElectronOptions.accuracy = options.eri.accuracy;
    twoElectronOptions.useDensityScreening = options.useDensityScreening;
    auto twoElectron = integrals::ComputeTwoElectronFockDerivative(molecule,
                                                                   basisSet,
                                                                   *densityTensor,
                                                                   state->MoCoefficients(),
                                                                   numOccupied,
                                                                   numVirtual,
                                                                   twoElectronOptions);

    if (!twoElectron.has_value())
    {
        return std::unexpected(twoElectron.error());
    }

    // The energy-reconstruction gate: the derivative walks' own energies must
    // reproduce the reference's electronic energy. One density, one set of
    // terms - a mismatch is a reference (or a term) that belongs elsewhere.
    const double reconstructedElectronic =
        oneElectron->kineticContraction + oneElectron->nuclearContraction +
        twoElectron->coulombEnergy - 0.5 * twoElectron->exchangeEnergy;
    const double energyResidual = std::abs(reconstructedElectronic - reference.electronicEnergy);

    if (energyResidual > options.energyTolerance)
    {
        return std::unexpected(
            Invalid("total gradient: the derivative terms rebuild an electronic energy of " +
                    std::to_string(reconstructedElectronic) + " Hartree against the reference's " +
                    std::to_string(reference.electronicEnergy) + " (residual " +
                    std::to_string(energyResidual) + " exceeds the tolerance " +
                    std::to_string(options.energyTolerance) +
                    "); the reference and the molecule or basis do not describe one state"));
    }

    auto rightHandSide = response::NuclearDisplacementRightHandSide::Create(layout, numBasis);

    if (!rightHandSide.has_value())
    {
        return std::unexpected(rightHandSide.error());
    }

    const Eigen::MatrixXd& oneElectronBlock = oneElectronFock->occupiedVirtual;
    const Eigen::MatrixXd& coulombBlock = twoElectron->coulombOccupiedVirtual;
    const Eigen::MatrixXd& exchangeBlock = twoElectron->exchangeOccupiedVirtual;

    auto added = rightHandSide->Add(
        "one-electron",
        [&oneElectronBlock, numOccupied, numVirtual](const response::FockDerivativeRequest& request,
                                                     std::span<double> block) -> qcx::Result<void> {
            const std::size_t row =
                (3 * request.displacement.atom + request.displacement.direction) * numVirtual;

            for (std::size_t occupied = 0; occupied < numOccupied; ++occupied)
            {
                for (std::size_t virtualOrbital = 0; virtualOrbital < numVirtual; ++virtualOrbital)
                {
                    block[occupied * numVirtual + virtualOrbital] +=
                        oneElectronBlock(static_cast<Eigen::Index>(row + virtualOrbital),
                                         static_cast<Eigen::Index>(occupied));
                }
            }

            return {};
        });

    if (!added.has_value())
    {
        return std::unexpected(added.error());
    }

    added = rightHandSide->Add(
        "coulomb",
        [&coulombBlock, numOccupied, numVirtual](const response::FockDerivativeRequest& request,
                                                 std::span<double> block) -> qcx::Result<void> {
            const std::size_t row =
                (3 * request.displacement.atom + request.displacement.direction) * numVirtual;

            for (std::size_t occupied = 0; occupied < numOccupied; ++occupied)
            {
                for (std::size_t virtualOrbital = 0; virtualOrbital < numVirtual; ++virtualOrbital)
                {
                    block[occupied * numVirtual + virtualOrbital] +=
                        coulombBlock(static_cast<Eigen::Index>(row + virtualOrbital),
                                     static_cast<Eigen::Index>(occupied));
                }
            }

            return {};
        });

    if (!added.has_value())
    {
        return std::unexpected(added.error());
    }

    // The exchange block carries the exchange Fock matrix itself; the
    // closed-shell energy subtracts half of it.
    added = rightHandSide->Add(
        "exchange",
        [&exchangeBlock, numOccupied, numVirtual](const response::FockDerivativeRequest& request,
                                                  std::span<double> block) -> qcx::Result<void> {
            const std::size_t row =
                (3 * request.displacement.atom + request.displacement.direction) * numVirtual;

            for (std::size_t occupied = 0; occupied < numOccupied; ++occupied)
            {
                for (std::size_t virtualOrbital = 0; virtualOrbital < numVirtual; ++virtualOrbital)
                {
                    block[occupied * numVirtual + virtualOrbital] +=
                        -0.5 * exchangeBlock(static_cast<Eigen::Index>(row + virtualOrbital),
                                             static_cast<Eigen::Index>(occupied));
                }
            }

            return {};
        });

    if (!added.has_value())
    {
        return std::unexpected(added.error());
    }

    TotalGradient result;
    result.terms.oneElectron = oneElectron->kineticGradient + oneElectron->nuclearGradient;
    result.terms.nuclearRepulsion = NuclearRepulsionGradient(molecule);
    result.terms.coulomb = twoElectron->coulombSkeleton;
    result.terms.exchange = -0.5 * twoElectron->exchangeSkeleton;
    result.terms.response = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(displacements));

    const std::size_t overlapBlock = orbitals * orbitals;

    if (oneElectronFock->moOverlapDerivative.size() != displacements * overlapBlock)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError,
                       "total gradient: the overlap-derivative walk returned " +
                           std::to_string(oneElectronFock->moOverlapDerivative.size()) +
                           " values for " + std::to_string(displacements) + " displacements of " +
                           std::to_string(orbitals) + " orbitals"});
    }

    for (std::size_t atom = 0; atom < atoms; ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            const std::size_t ordinal = 3 * atom + direction;
            const std::span<const double> overlapDerivative(oneElectronFock->moOverlapDerivative);
            auto solution = state->SolveDisplacement(
                response::NuclearDisplacement{atom, direction},
                *rightHandSide,
                overlapDerivative.subspan(ordinal * overlapBlock, overlapBlock));

            if (!solution.has_value())
            {
                return std::unexpected(solution.error());
            }

            auto energyWeighted = response::EnergyWeightedTerm(state->Fock(), solution->response);

            if (!energyWeighted.has_value())
            {
                return std::unexpected(energyWeighted.error());
            }

            result.terms.response(static_cast<Eigen::Index>(ordinal)) = *energyWeighted;
        }
    }

    // The total is the terms summed in the documented order - never an
    // independently assembled number, so the decomposition and the total
    // cannot disagree.
    Eigen::VectorXd total = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(displacements));
    total += result.terms.oneElectron;
    total += result.terms.nuclearRepulsion;
    total += result.terms.coulomb;
    total += result.terms.exchange;
    total += result.terms.response;

    if (dispersion.has_value())
    {
        result.terms.dispersion = dispersion->gradient;
        total += result.terms.dispersion;
    }

    result.energies.electronic = reference.electronicEnergy;
    result.energies.reconstructedElectronic = reconstructedElectronic;
    result.energies.nuclearRepulsion = qcx::molecule::NuclearRepulsionEnergy(molecule);
    result.energies.dispersion = dispersion.has_value() ? dispersion->energy : 0.0;
    result.energies.total =
        result.energies.electronic + result.energies.nuclearRepulsion + result.energies.dispersion;
    result.counts.oneElectron = oneElectron->counts;
    result.counts.twoElectron = twoElectron->counts;
    result.atomCount = atoms;
    result.numBasisFunctions = numBasis;
    result.numOccupied = numOccupied;
    result.numVirtual = numVirtual;
    result.response = state->Report();
    result.stationarityResidual = stationarityResidual;
    result.energyResidual = energyResidual;
    result.rightHandSideTerms = rightHandSide->Names();
    result.gradient = std::move(total);
    return result;
}

} // namespace qcx::properties
