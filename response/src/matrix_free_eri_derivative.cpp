// The matrix-free two-electron orbital-Hessian action: the trial vector's AO
// density, one closed-shell Fock build through the direct screened engine, and
// the occupied-virtual block of the transform back. No n^4 array is formed on
// either side.

#include "qcx/response/matrix_free_eri_derivative.hpp"

#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace qcx::response {

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

qcx::Error MakeError(std::string message) {
    return qcx::Error{qcx::ErrorCode::kInvalidArgument, std::move(message)};
}

// The Fock a builder produces at one density: the builder's core is zero, so
// the result is the two-electron term alone (2 J for the Coulomb-only mode,
// -K for the exchange-only one, 2 J - K for the full one).
qcx::Result<std::vector<double>> BuildDenseFock(const integrals::DirectJkFockBuilder& builder,
                                                std::span<const double> density,
                                                std::size_t numBasis) {
    auto tensor = CpuTensor2::Create({numBasis, numBasis});

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

    auto fock = builder.BuildFock(*tensor);

    if (!fock.has_value())
    {
        return std::unexpected(fock.error());
    }

    const std::vector<double>& values = fock->HostView();

    if (values.size() != numBasis * numBasis)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "matrix-free ERI derivative: the Fock builder returned " +
                                              std::to_string(values.size()) + " values for " +
                                              std::to_string(numBasis) + " basis functions"});
    }

    return std::vector<double>(values.begin(), values.end());
}

} // namespace

struct MatrixFreeEriDerivative::State {
    ResponseLayout layout;
    std::size_t numBasis = 0;
    std::size_t numOrbitals = 0;
    double exchangeFraction = 1.0;
    MatrixFreeEriDerivativeOptions options;

    std::vector<double> core; ///< H = T + V, row-major n x n.
    std::vector<double> coefficients; ///< Row-major n x orbitals, occupied first.

    /// Present for every fraction: the Coulomb-only Fock is what the
    /// reference's Kohn-Sham reconstruction reads.
    std::optional<integrals::DirectJkFockBuilder> coulombBuilder;
    /// Present when the fraction needs a full Fock: the unscaled action
    /// reads it in one build, and the Roothaan reference Fock uses it.
    std::optional<integrals::DirectJkFockBuilder> fullBuilder;
    /// Present when the fraction is strictly between zero and one: the
    /// exchange-only Fock combines with the Coulomb one for the scaled action
    /// and with both for the Roothaan reference Fock.
    std::optional<integrals::DirectJkFockBuilder> exchangeBuilder;

    // The application's scratch; reused between applications.
    mutable std::vector<double> trialDensity; ///< M, n x n.
    mutable std::vector<double> symmetricDensity; ///< S = (M + M^T) / 2, n x n.
    mutable std::vector<double> fock; ///< The Fock the action contracts, n x n.
    mutable std::vector<double> half; ///< n x orbitals scratch.

    /// The action: y += 2 C^T F(S) C on the occupied-virtual block.
    qcx::Result<void> Apply(std::span<const double> x, std::span<double> y) const;
};

qcx::Result<MatrixFreeEriDerivative> MatrixFreeEriDerivative::Create(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    std::span<const double> coreHamiltonian,
    ResponseLayout layout,
    std::size_t numBasisFunctions,
    std::span<const double> moCoefficients,
    double exchangeFraction,
    MatrixFreeEriDerivativeOptions options) {
    if (layout.numOccupied == 0 || layout.numVirtual == 0)
    {
        return std::unexpected(
            MakeError("matrix-free ERI derivative: the layout needs at least one occupied and one "
                      "virtual orbital"));
    }

    const std::size_t numOrbitals = layout.numOccupied + layout.numVirtual;

    if (numBasisFunctions < numOrbitals)
    {
        return std::unexpected(
            MakeError("matrix-free ERI derivative: the basis-function count cannot be below the "
                      "orbital count; the request states " +
                      std::to_string(numBasisFunctions) + " basis functions and " +
                      std::to_string(numOrbitals) + " orbitals"));
    }

    if (moCoefficients.size() != numBasisFunctions * numOrbitals)
    {
        return std::unexpected(MakeError("matrix-free ERI derivative: the MO coefficients must be "
                                         "numBasisFunctions x numOrbitals = " +
                                         std::to_string(numBasisFunctions) + " x " +
                                         std::to_string(numOrbitals) + "; the request carries " +
                                         std::to_string(moCoefficients.size()) + " values"));
    }

    if (coreHamiltonian.size() != numBasisFunctions * numBasisFunctions)
    {
        return std::unexpected(MakeError("matrix-free ERI derivative: the core Hamiltonian must be "
                                         "numBasisFunctions^2 = " +
                                         std::to_string(numBasisFunctions * numBasisFunctions) +
                                         " values; the request carries " +
                                         std::to_string(coreHamiltonian.size())));
    }

    // Written as a positive test so that a Not-a-Number fraction fails too.
    if (!(exchangeFraction >= 0.0 && exchangeFraction <= 1.0))
    {
        return std::unexpected(MakeError(
            "matrix-free ERI derivative: the exact-exchange fraction must lie in [0, 1]"));
    }

    if (options.maxBatchBytes == 0)
    {
        return std::unexpected(
            MakeError("matrix-free ERI derivative: maxBatchBytes must be positive"));
    }

    auto state = std::make_unique<State>();
    state->layout = layout;
    state->numBasis = numBasisFunctions;
    state->numOrbitals = numOrbitals;
    state->exchangeFraction = exchangeFraction;
    state->options = options;
    state->core.assign(coreHamiltonian.begin(), coreHamiltonian.end());
    state->coefficients.assign(moCoefficients.begin(), moCoefficients.end());

    // The builders carry a zero core: the two-electron term is what the
    // action and the reference Fock algebra combine, and the core is added
    // by the Fock methods where it belongs.
    auto zeroCore = CpuTensor2::Create({numBasisFunctions, numBasisFunctions});

    if (!zeroCore.has_value())
    {
        return std::unexpected(zeroCore.error());
    }

    for (std::size_t row = 0; row < numBasisFunctions; ++row)
    {
        for (std::size_t column = 0; column < numBasisFunctions; ++column)
        {
            (*zeroCore)(row, column) = 0.0;
        }
    }

    zeroCore->MarkHostDirty();

    integrals::FockBuildOptions buildOptions;
    buildOptions.accuracy = options.accuracy;
    buildOptions.useDensityScreening = options.useDensityScreening;
    buildOptions.usePerElementScreening = options.usePerElementScreening;
    // The action must not change its arithmetic between applications: the
    // certified fp32 lane's routing depends on the trial density, so it is
    // pinned off rather than left to the device's default.
    buildOptions.useCertifiedMixedPrecision = false;
    buildOptions.maxBatchBytes = options.maxBatchBytes;
    buildOptions.maxParallelChunks = options.maxParallelChunks;

    auto coulombOptions = buildOptions;
    coulombOptions.buildCoulombOnly = true;
    auto coulomb =
        integrals::DirectJkFockBuilder::Create(molecule, basisSet, *zeroCore, coulombOptions);

    if (!coulomb.has_value())
    {
        return std::unexpected(coulomb.error());
    }

    state->coulombBuilder.emplace(std::move(*coulomb));

    if (exchangeFraction == 1.0)
    {
        auto full =
            integrals::DirectJkFockBuilder::Create(molecule, basisSet, *zeroCore, buildOptions);

        if (!full.has_value())
        {
            return std::unexpected(full.error());
        }

        state->fullBuilder.emplace(std::move(*full));
    } else if (exchangeFraction > 0.0)
    {
        auto exchangeOptions = buildOptions;
        exchangeOptions.buildExchangeOnly = true;
        auto exchange =
            integrals::DirectJkFockBuilder::Create(molecule, basisSet, *zeroCore, exchangeOptions);

        if (!exchange.has_value())
        {
            return std::unexpected(exchange.error());
        }

        state->exchangeBuilder.emplace(std::move(*exchange));
    }

    const std::size_t numBasis = numBasisFunctions;
    state->trialDensity.assign(numBasis * numBasis, 0.0);
    state->symmetricDensity.assign(numBasis * numBasis, 0.0);
    state->fock.assign(numBasis * numBasis, 0.0);
    state->half.assign(numBasis * numOrbitals, 0.0);

    return MatrixFreeEriDerivative(std::move(state));
}

qcx::Result<void> MatrixFreeEriDerivative::State::Apply(std::span<const double> x,
                                                        std::span<double> y) const {
    const std::size_t numBasis = this->numBasis;
    const std::size_t numOrbitals = this->numOrbitals;
    const std::size_t numOccupied = layout.numOccupied;
    const std::size_t numVirtual = layout.numVirtual;
    const double* coefficients = this->coefficients.data();
    const double exchangeFraction = this->exchangeFraction;

    if (x.size() != layout.Dimension() || y.size() != layout.Dimension())
    {
        return std::unexpected(
            MakeError("matrix-free ERI derivative: the trial and result vectors must be sized "
                      "numOccupied * numVirtual"));
    }

    // T = C_v X^T, the n x numOccupied intermediate: T(mu, j) sums over the
    // virtual index of the trial vector.
    for (std::size_t mu = 0; mu < numBasis; ++mu)
    {
        for (std::size_t j = 0; j < numOccupied; ++j)
        {
            double sum = 0.0;

            for (std::size_t b = 0; b < numVirtual; ++b)
            {
                sum += coefficients[mu * numOrbitals + numOccupied + b] * x[j * numVirtual + b];
            }

            half[mu * numOccupied + j] = sum;
        }
    }

    // M = T C_o^T.
    for (std::size_t mu = 0; mu < numBasis; ++mu)
    {
        for (std::size_t nu = 0; nu < numBasis; ++nu)
        {
            double sum = 0.0;

            for (std::size_t j = 0; j < numOccupied; ++j)
            {
                sum += half[mu * numOccupied + j] * coefficients[nu * numOrbitals + j];
            }

            trialDensity[mu * numBasis + nu] = sum;
        }
    }

    // S = (M + M^T) / 2, written symmetrically: the builder's canonical-pair
    // contractions read both orientations of every pair block.
    for (std::size_t mu = 0; mu < numBasis; ++mu)
    {
        symmetricDensity[mu * numBasis + mu] = trialDensity[mu * numBasis + mu];

        for (std::size_t nu = mu + 1; nu < numBasis; ++nu)
        {
            const double value =
                0.5 * (trialDensity[mu * numBasis + nu] + trialDensity[nu * numBasis + mu]);
            symmetricDensity[mu * numBasis + nu] = value;
            symmetricDensity[nu * numBasis + mu] = value;
        }
    }

    // The scaled closed-shell Fock at S: 2 J(S) - c_HF K(S), assembled from
    // one full build or from the Coulomb and exchange builds the fraction
    // needs.
    if (exchangeFraction == 1.0)
    {
        auto built = BuildDenseFock(*fullBuilder, symmetricDensity, numBasis);

        if (!built.has_value())
        {
            return std::unexpected(built.error());
        }

        fock = std::move(*built);
    } else if (exchangeFraction == 0.0)
    {
        auto built = BuildDenseFock(*coulombBuilder, symmetricDensity, numBasis);

        if (!built.has_value())
        {
            return std::unexpected(built.error());
        }

        fock = std::move(*built);
    } else
    {
        auto coulomb = BuildDenseFock(*coulombBuilder, symmetricDensity, numBasis);

        if (!coulomb.has_value())
        {
            return std::unexpected(coulomb.error());
        }

        auto exchange = BuildDenseFock(*exchangeBuilder, symmetricDensity, numBasis);

        if (!exchange.has_value())
        {
            return std::unexpected(exchange.error());
        }

        for (std::size_t index = 0; index < fock.size(); ++index)
        {
            fock[index] = (*coulomb)[index] + exchangeFraction * (*exchange)[index];
        }
    }

    // G = F C_o.
    for (std::size_t mu = 0; mu < numBasis; ++mu)
    {
        for (std::size_t j = 0; j < numOccupied; ++j)
        {
            double sum = 0.0;

            for (std::size_t nu = 0; nu < numBasis; ++nu)
            {
                sum += fock[mu * numBasis + nu] * coefficients[nu * numOrbitals + j];
            }

            half[mu * numOccupied + j] = sum;
        }
    }

    // y += 2 C_v^T G: the occupied-virtual block of 2 C^T F C.
    for (std::size_t j = 0; j < numOccupied; ++j)
    {
        for (std::size_t b = 0; b < numVirtual; ++b)
        {
            double sum = 0.0;

            for (std::size_t mu = 0; mu < numBasis; ++mu)
            {
                sum +=
                    coefficients[mu * numOrbitals + numOccupied + b] * half[mu * numOccupied + j];
            }

            y[j * numVirtual + b] += 2.0 * sum;
        }
    }

    return {};
}

qcx::Result<void> MatrixFreeEriDerivative::Apply(std::span<const double> x,
                                                 std::span<double> y) const {
    return _state->Apply(x, y);
}

qcx::Result<std::vector<double>> MatrixFreeEriDerivative::ReferenceFock(
    std::span<const double> spatialDensity) const {
    const std::size_t numBasis = _state->numBasis;

    if (spatialDensity.size() != numBasis * numBasis)
    {
        return std::unexpected(
            MakeError("matrix-free ERI derivative: the reference density must be "
                      "numBasisFunctions^2 = " +
                      std::to_string(numBasis * numBasis) + " values; the request carries " +
                      std::to_string(spatialDensity.size())));
    }

    std::vector<double> fock;

    if (_state->fullBuilder.has_value())
    {
        auto built = BuildDenseFock(*_state->fullBuilder, spatialDensity, numBasis);

        if (!built.has_value())
        {
            return std::unexpected(built.error());
        }

        fock = std::move(*built);
    } else if (_state->exchangeBuilder.has_value())
    {
        auto coulomb = BuildDenseFock(*_state->coulombBuilder, spatialDensity, numBasis);

        if (!coulomb.has_value())
        {
            return std::unexpected(coulomb.error());
        }

        auto exchange = BuildDenseFock(*_state->exchangeBuilder, spatialDensity, numBasis);

        if (!exchange.has_value())
        {
            return std::unexpected(exchange.error());
        }

        fock.assign(numBasis * numBasis, 0.0);

        for (std::size_t index = 0; index < fock.size(); ++index)
        {
            fock[index] = (*coulomb)[index] + (*exchange)[index];
        }
    } else
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       "matrix-free ERI derivative: the Roothaan Fock includes exchange, and this "
                       "derivative was built with c_HF = 0 (no exchange builder); the Coulomb-only "
                       "route uses CoulombFock"});
    }

    for (std::size_t index = 0; index < fock.size(); ++index)
    {
        fock[index] += _state->core[index];
    }

    return fock;
}

qcx::Result<std::vector<double>> MatrixFreeEriDerivative::CoulombFock(
    std::span<const double> spatialDensity) const {
    const std::size_t numBasis = _state->numBasis;

    if (spatialDensity.size() != numBasis * numBasis)
    {
        return std::unexpected(
            MakeError("matrix-free ERI derivative: the reference density must be "
                      "numBasisFunctions^2 = " +
                      std::to_string(numBasis * numBasis) + " values; the request carries " +
                      std::to_string(spatialDensity.size())));
    }

    auto built = BuildDenseFock(*_state->coulombBuilder, spatialDensity, numBasis);

    if (!built.has_value())
    {
        return std::unexpected(built.error());
    }

    std::vector<double> fock = std::move(*built);

    for (std::size_t index = 0; index < fock.size(); ++index)
    {
        fock[index] += _state->core[index];
    }

    return fock;
}

ResponseLayout MatrixFreeEriDerivative::Layout() const {
    return _state->layout;
}

std::size_t MatrixFreeEriDerivative::NumBasisFunctions() const {
    return _state->numBasis;
}

double MatrixFreeEriDerivative::ExchangeFraction() const {
    return _state->exchangeFraction;
}

std::span<const double> MatrixFreeEriDerivative::MoCoefficients() const {
    return _state->coefficients;
}

const MatrixFreeEriDerivativeOptions& MatrixFreeEriDerivative::Options() const {
    return _state->options;
}

MatrixFreeEriDerivative::MatrixFreeEriDerivative(std::unique_ptr<State> state) :
    _state(std::move(state)) {}

MatrixFreeEriDerivative::MatrixFreeEriDerivative(MatrixFreeEriDerivative&&) noexcept = default;

MatrixFreeEriDerivative& MatrixFreeEriDerivative::operator=(MatrixFreeEriDerivative&&) noexcept =
    default;

MatrixFreeEriDerivative::~MatrixFreeEriDerivative() = default;

} // namespace qcx::response
