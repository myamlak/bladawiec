#include "qcx/response/response_operator.hpp"

#include <utility>

namespace qcx::response {

namespace {

/// Validates the occupied/virtual block structure both operators share.
/// \param layout The layout to check.
/// \returns An Error (kInvalidArgument) when a block is empty.
qcx::Result<void> ValidateLayout(const ResponseLayout& layout) {
    if (layout.numOccupied == 0 || layout.numVirtual == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "a response operator needs at least one occupied and one virtual orbital"});
    }

    return {};
}

/// Validates a matrix against the layout whose squared dimension it must hold.
/// \param layout The occupied/virtual block structure.
/// \param matrix Row-major matrix, required to be Dimension() squared long.
/// \returns An Error (kInvalidArgument) on a length mismatch.
qcx::Result<void> ValidateMatrix(const ResponseLayout& layout, std::span<const double> matrix) {
    if (matrix.size() != layout.Dimension() * layout.Dimension())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the matrix must be Dimension() squared"});
    }

    return {};
}

} // namespace

OrbitalHessianOperator::OrbitalHessianOperator(ResponseLayout layout,
                                               OrbitalEnergies orbitalEnergies,
                                               MoTwoElectronTensor moTwoElectron) :
    _layout(layout), _numOrbitals(layout.numOccupied + layout.numVirtual),
    _orbitalEnergies(orbitalEnergies.values), _moTwoElectron(moTwoElectron.values) {}

qcx::Result<OrbitalHessianOperator> OrbitalHessianOperator::Create(
    ResponseLayout layout, OrbitalEnergies orbitalEnergies, MoTwoElectronTensor moTwoElectron) {
    auto layoutStatus = ValidateLayout(layout);

    if (!layoutStatus.has_value())
    {
        return std::unexpected(layoutStatus.error());
    }

    const std::size_t numOrbitals = layout.numOccupied + layout.numVirtual;

    if (orbitalEnergies.values.size() != numOrbitals)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the orbital energies must number numOccupied + numVirtual"});
    }

    if (moTwoElectron.values.size() != numOrbitals * numOrbitals * numOrbitals * numOrbitals)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "the two-electron tensor must be the full fourth power of the orbital count"});
    }

    return OrbitalHessianOperator(layout, orbitalEnergies, moTwoElectron);
}

qcx::Result<void> OrbitalHessianOperator::Apply(std::span<const double> x,
                                                std::span<double> y) const {
    const std::size_t dimension = Dimension();

    if (x.size() != dimension || y.size() != dimension)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the trial and result vectors must be sized "
                                          "numOccupied * numVirtual"});
    }

    const std::size_t numOccupied = _layout.numOccupied;
    const std::size_t numVirtual = _layout.numVirtual;
    const std::size_t numOrbitals = _numOrbitals;
    const double* integrals = _moTwoElectron.data();

    for (std::size_t i = 0; i < numOccupied; ++i)
    {
        for (std::size_t a = 0; a < numVirtual; ++a)
        {
            const std::size_t virtualOrbital = numOccupied + a;
            double value =
                (_orbitalEnergies[virtualOrbital] - _orbitalEnergies[i]) * x[i * numVirtual + a];

            for (std::size_t j = 0; j < numOccupied; ++j)
            {
                for (std::size_t b = 0; b < numVirtual; ++b)
                {
                    const std::size_t virtualPartner = numOccupied + b;

                    // (ai|bj) = (a, i, b, j); (ab|ij) = (a, b, i, j);
                    // (aj|ib) = (a, j, i, b) - chemists' notation throughout.
                    // In these subscripts a and b are the virtual orbitals
                    // numOccupied + a and numOccupied + b; the loop variables
                    // index the virtual block, the tensor indexes orbitals.
                    const double coulomb =
                        4.0 * integrals[((virtualOrbital * numOrbitals + i) * numOrbitals +
                                         virtualPartner) *
                                            numOrbitals +
                                        j];
                    const double directExchange =
                        integrals[((virtualOrbital * numOrbitals + virtualPartner) * numOrbitals +
                                   i) *
                                      numOrbitals +
                                  j];
                    const double exchange =
                        integrals[((virtualOrbital * numOrbitals + j) * numOrbitals + i) *
                                      numOrbitals +
                                  virtualPartner];

                    value += (coulomb - directExchange - exchange) * x[j * numVirtual + b];
                }
            }

            y[i * numVirtual + a] = value;
        }
    }

    return {};
}

qcx::Result<void> OrbitalHessianOperator::Preconditioner(std::span<double> diagonal) const {
    if (diagonal.size() != Dimension())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the preconditioner must be sized numOccupied * "
                                          "numVirtual"});
    }

    for (std::size_t i = 0; i < _layout.numOccupied; ++i)
    {
        for (std::size_t a = 0; a < _layout.numVirtual; ++a)
        {
            diagonal[i * _layout.numVirtual + a] =
                _orbitalEnergies[_layout.numOccupied + a] - _orbitalEnergies[i];
        }
    }

    return {};
}

DenseResponseOperator::DenseResponseOperator(ResponseLayout layout,
                                             std::vector<double> matrix,
                                             std::vector<double> preconditioner) :
    _layout(layout), _matrix(std::move(matrix)), _preconditioner(std::move(preconditioner)) {}

qcx::Result<DenseResponseOperator> DenseResponseOperator::Create(ResponseLayout layout,
                                                                 std::span<const double> matrix) {
    auto layoutStatus = ValidateLayout(layout);

    if (!layoutStatus.has_value())
    {
        return std::unexpected(layoutStatus.error());
    }

    auto matrixStatus = ValidateMatrix(layout, matrix);

    if (!matrixStatus.has_value())
    {
        return std::unexpected(matrixStatus.error());
    }

    const std::size_t dimension = layout.Dimension();
    std::vector<double> diagonal(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        diagonal[i] = matrix[i * dimension + i];
    }

    return Create(layout, matrix, diagonal);
}

qcx::Result<DenseResponseOperator> DenseResponseOperator::Create(
    ResponseLayout layout, std::span<const double> matrix, std::span<const double> preconditioner) {
    auto layoutStatus = ValidateLayout(layout);

    if (!layoutStatus.has_value())
    {
        return std::unexpected(layoutStatus.error());
    }

    auto matrixStatus = ValidateMatrix(layout, matrix);

    if (!matrixStatus.has_value())
    {
        return std::unexpected(matrixStatus.error());
    }

    const std::size_t dimension = layout.Dimension();

    if (preconditioner.size() != dimension)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the preconditioner must be Dimension()"});
    }

    return DenseResponseOperator(layout,
                                 std::vector<double>(matrix.begin(), matrix.end()),
                                 std::vector<double>(preconditioner.begin(), preconditioner.end()));
}

qcx::Result<void> DenseResponseOperator::Apply(std::span<const double> x,
                                               std::span<double> y) const {
    const std::size_t dimension = Dimension();

    if (x.size() != dimension || y.size() != dimension)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the trial and result vectors must be Dimension()"});
    }

    for (std::size_t row = 0; row < dimension; ++row)
    {
        const double* matrixRow = _matrix.data() + row * dimension;
        double value = 0.0;

        for (std::size_t column = 0; column < dimension; ++column)
        {
            value += matrixRow[column] * x[column];
        }

        y[row] = value;
    }

    return {};
}

qcx::Result<void> DenseResponseOperator::Preconditioner(std::span<double> diagonal) const {
    if (diagonal.size() != Dimension())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the preconditioner must be Dimension()"});
    }

    for (std::size_t i = 0; i < Dimension(); ++i)
    {
        diagonal[i] = _preconditioner[i];
    }

    return {};
}

} // namespace qcx::response
